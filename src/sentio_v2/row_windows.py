"""Sliding-window feature table for row-format fused sessions."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# Fused-frame numeric column order (after `row_session.load_fused_session`); must match
# `mobile/src/baseNumericColumns.ts` for on-device feature parity.
ROW_BASE_NUMERIC_COLUMN_NAMES: Tuple[str, ...] = (
    "acc_z",
    "acc_y",
    "acc_x",
    "grav_z",
    "grav_y",
    "grav_x",
    "gyro_z",
    "gyro_y",
    "gyro_x",
    "orient_yaw",
    "orient_qx",
    "orient_qz",
    "orient_roll",
    "orient_qw",
    "orient_qy",
    "orient_pitch",
    "baro_relative_altitude_m",
    "baro_pressure_hpa",
)


def _numeric_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    drop = {"time", "t", "session_group", "label", "multiclass"}
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric columns for window features")
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, cols


def windows_from_session(
    df: pd.DataFrame,
    window_samples: int,
    hop_samples: int,
) -> pd.DataFrame:
    """Turn one fused session into one row per window (mean/std/max per channel)."""
    if len(df) < window_samples:
        return pd.DataFrame()

    X, col_names = _numeric_feature_matrix(df)
    groups = df["session_group"].to_numpy()
    labels = df["label"].to_numpy()
    mc = df["multiclass"].to_numpy() if "multiclass" in df.columns else np.zeros(len(df), dtype=int)
    n = X.shape[0]

    t_series = df["t"].to_numpy(dtype=float)

    rows = []
    for start in range(0, n - window_samples + 1, hop_samples):
        end = start + window_samples
        chunk = X[start:end]
        feats = {}
        for j, name in enumerate(col_names):
            v = chunk[:, j]
            feats[f"{name}_mean"] = float(np.mean(v))
            feats[f"{name}_std"] = float(np.std(v))
            feats[f"{name}_max"] = float(np.max(v))
        feats["window_t_mid"] = float(np.mean(t_series[start:end]))
        feats["session_group"] = str(groups[start])
        feats["label"] = int(labels[start])
        feats["multiclass"] = int(mc[start])
        rows.append(feats)

    return pd.DataFrame(rows)


def build_window_dataset(
    fused_frames: Iterable[pd.DataFrame],
    window_samples: int,
    hop_samples: int,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for df in fused_frames:
        w = windows_from_session(df, window_samples=window_samples, hop_samples=hop_samples)
        if not w.empty:
            parts.append(w)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
