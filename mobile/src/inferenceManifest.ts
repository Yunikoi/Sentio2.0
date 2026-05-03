import raw from '../assets/coreml/FallDetectorRowRF_manifest.json';

export type FallManifest = {
  feature_names_in_order: string[];
  n_features: number;
  window_samples?: number;
  hop_samples?: number;
  base_numeric_columns?: string[];
  target_output?: string;
  probability_output?: string;
};

export const fallManifest = raw as FallManifest;

export const DEFAULT_WINDOW_SAMPLES = fallManifest.window_samples ?? 101;
export const DEFAULT_HOP_SAMPLES = fallManifest.hop_samples ?? 25;
