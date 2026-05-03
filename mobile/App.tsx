import {
  Barometer,
  DeviceMotion,
  type BarometerMeasurement,
  type DeviceMotionMeasurement,
} from 'expo-sensors';
import { StatusBar } from 'expo-status-bar';
import {
  isFallDetectorLoaded,
  loadFallDetectorModel,
  predictFall,
} from 'sentio-coreml';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { BASE_NUMERIC_COLUMNS } from './src/baseNumericColumns';
import { FallFeatureRingBuffer, toModelFeatureDict } from './src/fallFeaturePipeline';
import { DEFAULT_WINDOW_SAMPLES, fallManifest } from './src/inferenceManifest';
import { fusedRowFromDeviceMotion, updateBaroState, type BaroState } from './src/rowFusion';

const BG = '#F4F0EB';
const CARD = '#FFFFFF';
const TEXT = '#3D3832';
const MUTED = '#7A7268';
function fmt(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) {
    return '—';
  }
  return n.toFixed(digits);
}

function vec3(
  v:
    | { x: number; y: number; z: number }
    | { alpha: number; beta: number; gamma: number }
    | null
    | undefined,
): string {
  if (!v) return '—';
  if ('alpha' in v) {
    return `${fmt(v.alpha)} / ${fmt(v.beta)} / ${fmt(v.gamma)}`;
  }
  return `${fmt(v.x)} / ${fmt(v.y)} / ${fmt(v.z)}`;
}

export default function App() {
  const [perm, setPerm] = useState<string>('');
  const [dm, setDm] = useState<DeviceMotionMeasurement | null>(null);
  const [baro, setBaro] = useState<BarometerMeasurement | null>(null);
  const [bufLen, setBufLen] = useState(0);
  const [coremlLoading, setCoremlLoading] = useState(false);
  const [coremlReady, setCoremlReady] = useState(false);
  const [coremlError, setCoremlError] = useState<string | null>(null);
  const [coremlProbe, setCoremlProbe] = useState<string | null>(null);
  const [livePred, setLivePred] = useState<string | null>(null);
  const [liveErr, setLiveErr] = useState<string | null>(null);

  const baroRef = useRef<BaroState>({
    relativeAltitudeM: null,
    pressureHpa: null,
  });
  const ringRef = useRef(new FallFeatureRingBuffer());
  const t0Ref = useRef<number | null>(null);

  useEffect(() => {
    const b = fallManifest.base_numeric_columns;
    if (b && b.join(',') !== [...BASE_NUMERIC_COLUMNS].join(',')) {
      console.warn(
        '[Sentio] fallManifest.base_numeric_columns does not match BASE_NUMERIC_COLUMNS (Python row_session / row_windows order).',
      );
    }
  }, []);

  useEffect(() => {
    let dmSub: { remove: () => void } | undefined;
    let brSub: { remove: () => void } | undefined;

    (async () => {
      const dmOk = await DeviceMotion.isAvailableAsync();
      if (!dmOk) {
        setPerm('设备运动传感器不可用');
        return;
      }
      const p = await DeviceMotion.requestPermissionsAsync();
      if (p.status !== 'granted') {
        setPerm('需要运动权限以读取加速度、重力与陀螺仪');
        return;
      }
      setPerm('');

      DeviceMotion.setUpdateInterval(20);
      dmSub = DeviceMotion.addListener((e) => {
        setDm(e);
        if (t0Ref.current === null) {
          t0Ref.current = Date.now() / 1000;
        }
        const t = Date.now() / 1000 - t0Ref.current;
        const row = fusedRowFromDeviceMotion(e, t, baroRef.current);
        ringRef.current.push(row);
        if (ringRef.current.length % 5 === 0) {
          setBufLen(ringRef.current.length);
        }
      });

      const bOk = await Barometer.isAvailableAsync();
      if (bOk) {
        Barometer.setUpdateInterval(500);
        brSub = Barometer.addListener((e) => {
          baroRef.current = updateBaroState(baroRef.current, e);
          setBaro(e);
        });
      }
    })();

    return () => {
      dmSub?.remove();
      brSub?.remove();
    };
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'ios') {
      return;
    }
    let cancelled = false;
    (async () => {
      setCoremlLoading(true);
      setCoremlError(null);
      try {
        await loadFallDetectorModel();
        if (!cancelled) {
          setCoremlReady(await isFallDetectorLoaded());
        }
      } catch (e) {
        if (!cancelled) {
          setCoremlError(e instanceof Error ? e.message : String(e));
          setCoremlReady(false);
        }
      } finally {
        if (!cancelled) {
          setCoremlLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const inc = dm?.accelerationIncludingGravity;
  const user = dm?.acceleration;
  const grav =
    inc && user
      ? {
          x: inc.x - user.x,
          y: inc.y - user.y,
          z: inc.z - user.z,
        }
      : null;
  const gyro = dm?.rotationRate;
  const rot = dm?.rotation;

  const nFeat = fallManifest.n_features ?? 0;

  const runCoreMLSmokeTest = useCallback(async () => {
    if (Platform.OS !== 'ios') {
      return;
    }
    setCoremlProbe(null);
    setCoremlError(null);
    try {
      const feats: Record<string, number> = {};
      for (const name of fallManifest.feature_names_in_order) {
        feats[name] = 0;
      }
      const r = await predictFall(feats);
      setCoremlProbe(`label=${r.label}  P(fall)=${r.probability ?? '—'}`);
    } catch (e) {
      setCoremlProbe(null);
      setCoremlError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const runLivePredict = useCallback(async () => {
    if (Platform.OS !== 'ios') {
      return;
    }
    setLiveErr(null);
    setLivePred(null);
    try {
      const wf = ringRef.current.getLatestWindowFeatures(DEFAULT_WINDOW_SAMPLES);
      if (!wf) {
        setLiveErr(`缓冲未满 ${DEFAULT_WINDOW_SAMPLES} 帧（当前 ${ringRef.current.length}）`);
        return;
      }
      const dict = toModelFeatureDict(wf);
      const r = await predictFall(dict);
      setLivePred(`label=${r.label}  P(fall)=${r.probability ?? '—'}`);
    } catch (e) {
      setLiveErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  return (
    <View style={styles.root}>
      <StatusBar style="dark" />
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.title}>Sentio</Text>
        <Text style={styles.subtitle}>实时传感器 · 与 data_sensor 命名对齐</Text>
        {perm ? <Text style={styles.warn}>{perm}</Text> : null}

        <View style={styles.card}>
          <Text style={styles.cardTitle}>线性加速度</Text>
          <Text style={styles.cardHint}>Accelerometer.csv（合加速度，m/s²）</Text>
          <Text style={styles.value}>{vec3(user)}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>重力向量</Text>
          <Text style={styles.cardHint}>Gravity.csv（由含重力 − 线性加速度 估计，m/s²）</Text>
          <Text style={styles.value}>{vec3(grav)}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>旋转速率</Text>
          <Text style={styles.cardHint}>Gyroscope.csv（本机为 °/s；采集 CSV 可能为 rad/s）</Text>
          <Text style={styles.value}>{vec3(gyro)}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>设备方向</Text>
          <Text style={styles.cardHint}>Orientation.csv（欧拉角 α/β/γ，弧度）</Text>
          <Text style={styles.value}>
            {rot
              ? `${fmt(rot.alpha)} / ${fmt(rot.beta)} / ${fmt(rot.gamma)}`
              : '—'}
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>海拔与气压</Text>
          <Text style={styles.cardHint}>Barometer.csv（relativeAltitude m · pressure hPa）</Text>
          <Text style={styles.label}>相对海拔 (m)</Text>
          <Text style={styles.value}>{fmt(baro?.relativeAltitude, 4)}</Text>
          <Text style={styles.label}>气压 (hPa)</Text>
          <Text style={styles.value}>{fmt(baro?.pressure, 2)}</Text>
        </View>

        <View style={[styles.card, styles.coremlCard]}>
          <Text style={styles.cardTitle}>Core ML（Config Plugin + 原生模块）</Text>
          <Text style={styles.coremlText}>
            配置插件在 prebuild 时复制 mlmodel；sentio-coreml 模块在 iOS 上加载 Core ML。
            滑窗统计与 Python `row_windows.py` 对齐（nan/inf→0、mean、总体 std、max；窗长{' '}
            {DEFAULT_WINDOW_SAMPLES}）。需 Dev Client / prebuild 工程。融合字段顺序见 manifest
            base_numeric_columns（与 `row_session` 一致）。缓冲帧数：{bufLen} /{' '}
            {DEFAULT_WINDOW_SAMPLES}。
          </Text>
          {Platform.OS === 'ios' ? (
            <>
              {coremlLoading ? (
                <ActivityIndicator style={styles.coremlSpinner} color={MUTED} />
              ) : (
                <Text style={styles.coremlStatus}>
                  模型：{coremlReady ? '已加载' : '未加载'}
                  {coremlError ? ` · ${coremlError}` : ''}
                </Text>
              )}
              <View style={styles.coremlBtnRow}>
                <Pressable
                  style={({ pressed }) => [
                    styles.coremlBtn,
                    pressed && styles.coremlBtnPressed,
                    !coremlReady && styles.coremlBtnDisabled,
                  ]}
                  onPress={runCoreMLSmokeTest}
                  disabled={!coremlReady || coremlLoading}
                >
                  <Text style={styles.coremlBtnText}>全零自检</Text>
                </Pressable>
                <Pressable
                  style={({ pressed }) => [
                    styles.coremlBtn,
                    styles.coremlBtnSecondary,
                    pressed && styles.coremlBtnPressed,
                    (!coremlReady || coremlLoading) && styles.coremlBtnDisabled,
                  ]}
                  onPress={runLivePredict}
                  disabled={!coremlReady || coremlLoading}
                >
                  <Text style={styles.coremlBtnText}>实时滑窗推理</Text>
                </Pressable>
              </View>
              {coremlProbe ? (
                <Text style={styles.coremlProbe}>自检：{coremlProbe}</Text>
              ) : null}
              {livePred ? <Text style={styles.coremlProbe}>实时：{livePred}</Text> : null}
              {liveErr ? <Text style={styles.coremlErr}>{liveErr}</Text> : null}
            </>
          ) : (
            <Text style={styles.coremlText}>Core ML 推理仅在 iOS 设备 / 模拟器可用。</Text>
          )}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: BG,
  },
  scroll: {
    paddingHorizontal: 20,
    paddingTop: 56,
    paddingBottom: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: '300',
    color: TEXT,
    letterSpacing: 1,
  },
  subtitle: {
    marginTop: 6,
    fontSize: 14,
    color: MUTED,
    marginBottom: 20,
  },
  warn: {
    color: '#A65D57',
    marginBottom: 12,
    fontSize: 14,
  },
  card: {
    backgroundColor: CARD,
    borderRadius: 18,
    padding: 18,
    marginBottom: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
    elevation: 2,
  },
  coremlCard: {
    borderWidth: 1,
    borderColor: '#C5D4C8',
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: TEXT,
    marginBottom: 4,
  },
  cardHint: {
    fontSize: 12,
    color: MUTED,
    marginBottom: 10,
  },
  label: {
    fontSize: 12,
    color: MUTED,
    marginTop: 6,
  },
  value: {
    fontSize: 15,
    fontVariant: ['tabular-nums'],
    color: TEXT,
  },
  coremlText: {
    fontSize: 13,
    lineHeight: 20,
    color: MUTED,
    marginBottom: 12,
  },
  coremlSpinner: {
    marginVertical: 8,
  },
  coremlStatus: {
    fontSize: 13,
    color: TEXT,
    marginBottom: 10,
  },
  coremlBtnRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 4,
  },
  coremlBtn: {
    alignSelf: 'flex-start',
    backgroundColor: '#3D3832',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 12,
    marginRight: 10,
    marginBottom: 8,
  },
  coremlBtnSecondary: {
    backgroundColor: '#5C6B63',
  },
  coremlBtnPressed: {
    opacity: 0.85,
  },
  coremlBtnDisabled: {
    opacity: 0.4,
  },
  coremlBtnText: {
    color: '#FAF7F4',
    fontSize: 14,
    fontWeight: '600',
  },
  coremlProbe: {
    marginTop: 10,
    fontSize: 13,
    color: TEXT,
    fontVariant: ['tabular-nums'],
  },
  coremlErr: {
    marginTop: 8,
    fontSize: 13,
    color: '#A65D57',
  },
});
