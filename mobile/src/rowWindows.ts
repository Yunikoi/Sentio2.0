/**
 * Sliding-window statistics aligned with `src/sentio_v2/row_windows.py`:
 * - `np.nan_to_num` (nan, ±inf → 0) on the numeric matrix before stats
 * - `np.mean`, `np.std` with default `ddof=0` (population std), `np.max`
 * - feature names `{baseCol}_{mean|std|max}` in the same column-major order as Python
 */

import {
  BASE_NUMERIC_COLUMNS,
  type BaseNumericColumn,
  type FusedNumericRow,
} from './baseNumericColumns';

export function nanToNum(v: number): number {
  if (!Number.isFinite(v) || Number.isNaN(v)) {
    return 0;
  }
  return v;
}

export function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  let s = 0;
  for (const v of values) {
    s += v;
  }
  return s / values.length;
}

/** Population standard deviation (numpy `std` default, ddof=0). */
export function stdPop(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const m = mean(values);
  let acc = 0;
  for (const v of values) {
    const d = v - m;
    acc += d * d;
  }
  return Math.sqrt(acc / values.length);
}

export function max1(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return Math.max(...values);
}

/**
 * Build one `{name_mean, name_std, name_max}` record from the last `windowSamples` rows.
 * Rows must be oldest-first; uses the same base column order as training.
 */
export function buildWindowFeatureRecord(
  rows: FusedNumericRow[],
  windowSamples: number,
): Record<string, number> | null {
  if (rows.length < windowSamples) {
    return null;
  }
  const slice = rows.slice(-windowSamples);
  const nCols = BASE_NUMERIC_COLUMNS.length;
  const mat: number[][] = [];
  for (let i = 0; i < slice.length; i++) {
    const row = slice[i];
    const vec: number[] = new Array(nCols);
    for (let j = 0; j < nCols; j++) {
      const key = BASE_NUMERIC_COLUMNS[j];
      vec[j] = nanToNum(row[key]);
    }
    mat.push(vec);
  }

  const out: Record<string, number> = {};
  for (let j = 0; j < nCols; j++) {
    const name = BASE_NUMERIC_COLUMNS[j];
    const col = mat.map((r) => r[j]);
    out[`${name}_mean`] = mean(col);
    out[`${name}_std`] = stdPop(col);
    out[`${name}_max`] = max1(col);
  }
  return out;
}

/** Mean of `t` over the window (for parity with Python `window_t_mid`; not a model input). */
export function windowTMid(timestamps: number[], windowSamples: number): number | null {
  if (timestamps.length < windowSamples) {
    return null;
  }
  const slice = timestamps.slice(-windowSamples);
  return mean(slice);
}
