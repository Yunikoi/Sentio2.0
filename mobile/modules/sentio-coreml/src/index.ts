import { requireNativeModule } from 'expo-modules-core';
import { Platform } from 'react-native';

type NativeSentioCoreML = {
  loadModelAsync(): Promise<void>;
  isLoaded(): Promise<boolean>;
  predict(features: Record<string, number>): Promise<{
    label?: number;
    probability?: number | null;
  }>;
};

const native = requireNativeModule<NativeSentioCoreML>('SentioCoreML');

export async function loadFallDetectorModel(): Promise<void> {
  if (Platform.OS !== 'ios') {
    return;
  }
  await native.loadModelAsync();
}

export async function isFallDetectorLoaded(): Promise<boolean> {
  if (Platform.OS !== 'ios') {
    return false;
  }
  return native.isLoaded();
}

export type FallPredictResult = {
  label: number;
  probability?: number | null;
};

export async function predictFall(
  features: Record<string, number>,
): Promise<FallPredictResult> {
  if (Platform.OS !== 'ios') {
    throw new Error('SentioCoreML predict is only supported on iOS');
  }
  const out = await native.predict(features);
  return {
    label: Number(out.label ?? 0),
    probability: out.probability ?? null,
  };
}
