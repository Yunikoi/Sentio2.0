/**
 * Live fusion aligned with `src/sentio_v2/row_session.py` naming and axis order.
 * Note: Android CSV may expose orientation quaternion independent of Euler rows; here
 * quaternions are derived from Expo `rotation` (α,β,γ) with the same scipy `zyx` rule as
 * `eulerQuat.ts`, so raw streams can differ from a given CSV row while window stats still
 * follow the identical mean/std/max pipeline as `row_windows.py`.
 *
 * Axis mapping highlights:
 * - Accelerometer CSV z,y,x → acc_z, acc_y, acc_x (Expo user acceleration m/s²)
 * - Gravity CSV z,y,x → grav_z, grav_y, grav_x (estimated: includingGravity − user)
 * - Gyroscope CSV z,y,x → gyro_z, gyro_y, gyro_x (**rad/s**; Expo rotationRate is °/s)
 * - Orientation columns yaw,qx,qz,roll,qw,qy,pitch with prefixed `orient_*`
 * - Barometer relativeAltitude (m), pressure (hPa) — forward-filled from last sample
 */

import type { DeviceMotionMeasurement } from 'expo-sensors';
import type { BarometerMeasurement } from 'expo-sensors';

import type { FusedNumericRow } from './baseNumericColumns';
import { eulerZyxRadiansToQuatXyzw } from './eulerQuat';
import { nanToNum } from './rowWindows';

const DEG_TO_RAD = Math.PI / 180;

export type BaroState = {
  relativeAltitudeM: number | null;
  pressureHpa: number | null;
};

export function updateBaroState(
  prev: BaroState,
  m: BarometerMeasurement | null | undefined,
): BaroState {
  if (!m) {
    return prev;
  }
  return {
    relativeAltitudeM:
      m.relativeAltitude !== undefined && m.relativeAltitude !== null
        ? m.relativeAltitude
        : prev.relativeAltitudeM,
    pressureHpa: Number.isFinite(m.pressure) ? m.pressure : prev.pressureHpa,
  };
}

/**
 * Map one DeviceMotion sample (+ last baro) to a fused numeric row.
 * `t` should use the same seconds timeline as training `seconds_elapsed` (monotonic).
 */
export function fusedRowFromDeviceMotion(
  dm: DeviceMotionMeasurement,
  t: number,
  baro: BaroState,
): FusedNumericRow {
  const user = dm.acceleration;
  const inc = dm.accelerationIncludingGravity;
  const rr = dm.rotationRate;
  const rot = dm.rotation;

  const accX = nanToNum(user?.x ?? NaN);
  const accY = nanToNum(user?.y ?? NaN);
  const accZ = nanToNum(user?.z ?? NaN);

  const gravX =
    inc !== undefined
      ? nanToNum(inc.x - (user?.x ?? 0))
      : nanToNum(NaN);
  const gravY =
    inc !== undefined ? nanToNum(inc.y - (user?.y ?? 0)) : nanToNum(NaN);
  const gravZ =
    inc !== undefined ? nanToNum(inc.z - (user?.z ?? 0)) : nanToNum(NaN);

  // Expo: alpha→X, beta→Y, gamma→Z rate (°/s). CSV Gyroscope z,y,x → gyro_z, gyro_y, gyro_x.
  const gyroX = rr ? nanToNum(rr.alpha * DEG_TO_RAD) : 0;
  const gyroY = rr ? nanToNum(rr.beta * DEG_TO_RAD) : 0;
  const gyroZ = rr ? nanToNum(rr.gamma * DEG_TO_RAD) : 0;

  const yaw = rot ? nanToNum(rot.alpha) : 0;
  const roll = rot ? nanToNum(rot.beta) : 0;
  const pitch = rot ? nanToNum(rot.gamma) : 0;
  const q = eulerZyxRadiansToQuatXyzw(yaw, roll, pitch);

  const baroRel =
    baro.relativeAltitudeM !== null && baro.relativeAltitudeM !== undefined
      ? nanToNum(baro.relativeAltitudeM)
      : Number.NaN;
  const baroPr =
    baro.pressureHpa !== null && baro.pressureHpa !== undefined
      ? nanToNum(baro.pressureHpa)
      : Number.NaN;

  return {
    acc_z: accZ,
    acc_y: accY,
    acc_x: accX,
    grav_z: gravZ,
    grav_y: gravY,
    grav_x: gravX,
    gyro_z: gyroZ,
    gyro_y: gyroY,
    gyro_x: gyroX,
    orient_yaw: yaw,
    orient_qx: q.qx,
    orient_qz: q.qz,
    orient_roll: roll,
    orient_qw: q.qw,
    orient_qy: q.qy,
    orient_pitch: pitch,
    baro_relative_altitude_m: baroRel,
    baro_pressure_hpa: baroPr,
  };
}
