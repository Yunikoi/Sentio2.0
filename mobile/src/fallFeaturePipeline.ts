import type { FusedNumericRow } from './baseNumericColumns';
import { DEFAULT_WINDOW_SAMPLES, fallManifest } from './inferenceManifest';
import { buildWindowFeatureRecord } from './rowWindows';

/** Ring buffer of fused numeric rows (oldest first), matching offline session order. */
export class FallFeatureRingBuffer {
  private readonly rows: FusedNumericRow[] = [];
  private readonly capacity: number;

  constructor(capacity = 512) {
    this.capacity = capacity;
  }

  push(row: FusedNumericRow): void {
    this.rows.push(row);
    if (this.rows.length > this.capacity) {
      this.rows.shift();
    }
  }

  clear(): void {
    this.rows.length = 0;
  }

  get length(): number {
    return this.rows.length;
  }

  /**
   * Latest-window mean/std/max feature dict aligned with training (`row_windows.py`).
   */
  getLatestWindowFeatures(
    windowSamples: number = DEFAULT_WINDOW_SAMPLES,
  ): Record<string, number> | null {
    return buildWindowFeatureRecord(this.rows, windowSamples);
  }
}

/** Restrict to model keys and verify manifest order / length. */
export function toModelFeatureDict(
  windowFeatures: Record<string, number>,
): Record<string, number> {
  const keys = fallManifest.feature_names_in_order;
  const out: Record<string, number> = {};
  for (const k of keys) {
    if (!(k in windowFeatures)) {
      throw new Error(`Missing window feature key: ${k}`);
    }
    out[k] = windowFeatures[k];
  }
  return out;
}
