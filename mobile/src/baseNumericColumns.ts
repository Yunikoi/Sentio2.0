/**
 * Numeric column order after `row_session.load_fused_session` merge, matching
 * `pandas.DataFrame.columns` iteration in `row_windows._numeric_feature_matrix`
 * (excluding time, t, session_group, label).
 *
 * Must stay in lockstep with `src/sentio_v2/row_session.py`.
 */
export const BASE_NUMERIC_COLUMNS = [
  'acc_z',
  'acc_y',
  'acc_x',
  'grav_z',
  'grav_y',
  'grav_x',
  'gyro_z',
  'gyro_y',
  'gyro_x',
  'orient_yaw',
  'orient_qx',
  'orient_qz',
  'orient_roll',
  'orient_qw',
  'orient_qy',
  'orient_pitch',
  'baro_relative_altitude_m',
  'baro_pressure_hpa',
] as const;

export type BaseNumericColumn = (typeof BASE_NUMERIC_COLUMNS)[number];

/** One IMU-aligned fused row (same fields as `row_session` numeric columns). */
export type FusedNumericRow = Record<BaseNumericColumn, number>;
