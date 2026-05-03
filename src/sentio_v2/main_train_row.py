"""Train a binary fall classifier on data_sensor/row (AdultManFall = 1, all else = 0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .row_loader import FALL_ACTIVITY, list_row_sessions
from .row_session import load_fused_session
from .row_windows import build_window_dataset


def _positive_class_proba(clf: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Probability of label 1; zeros if the model was fit on a single class."""
    if 1 not in clf.classes_:
        return np.zeros(len(X), dtype=np.float64)
    col = list(clf.classes_).index(1)
    return clf.predict_proba(X)[:, col].astype(np.float64)


def _train_test_indices(
    win_df: pd.DataFrame,
    test_size: float,
    seed: int,
    fall_time_train_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Negatives: group shuffle by session. Falls: earlier windows -> train, later -> test (single clip)."""
    meta: dict = {
        "split_policy": "neg_session_group_shuffle_and_fall_chronological",
        "fall_time_train_ratio": float(fall_time_train_ratio),
    }
    fall_mask = win_df["label"].to_numpy(dtype=int) == 1
    if not fall_mask.any():
        raise SystemExit("No AdultManFall windows after fusion.")

    fall_ordered = win_df.loc[fall_mask].sort_values("window_t_mid")
    fall_idx = fall_ordered.index.to_numpy()
    n_fall = len(fall_idx)
    if n_fall < 2:
        fall_train_idx = fall_idx
        fall_test_idx = np.array([], dtype=int)
        meta["fall_split_note"] = "all fall windows in train (need >=2 windows for time split)"
    else:
        split_at = max(1, min(n_fall - 1, int(round(n_fall * fall_time_train_ratio))))
        fall_train_idx = fall_idx[:split_at]
        fall_test_idx = fall_idx[split_at:]

    neg_df = win_df.loc[~fall_mask]
    if neg_df.empty:
        meta["neg_split_note"] = "no negative windows"
        return fall_train_idx, fall_test_idx, meta

    exclude = {"label", "session_group", "window_t_mid"}
    feat_cols = [c for c in neg_df.columns if c not in exclude]
    Xn = neg_df[feat_cols]
    yn = neg_df["label"].to_numpy(dtype=int)
    gn = neg_df["session_group"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    neg_tr_rel, neg_te_rel = next(gss.split(Xn, yn, gn))
    neg_train_idx = neg_df.index[neg_tr_rel].to_numpy()
    neg_test_idx = neg_df.index[neg_te_rel].to_numpy()

    train_idx = np.unique(np.concatenate([fall_train_idx, neg_train_idx]))
    test_idx = np.unique(np.concatenate([fall_test_idx, neg_test_idx]))
    meta["n_fall_windows_train"] = int(len(fall_train_idx))
    meta["n_fall_windows_test"] = int(len(fall_test_idx))
    return train_idx, test_idx, meta


def _estimate_imu_hz(df: pd.DataFrame) -> float:
    t = df["t"].to_numpy(dtype=float)
    if len(t) < 2:
        return 50.0
    dt = np.diff(t)
    dt = dt[dt > 1e-6]
    if dt.size == 0:
        return 50.0
    return float(1.0 / np.median(dt))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sklearn model on data_sensor/row sessions.")
    parser.add_argument(
        "--row-root",
        type=Path,
        default=Path("data_sensor/row"),
        help="Root folder containing activity subfolders",
    )
    parser.add_argument("--window-sec", type=float, default=2.0, help="Window length in seconds")
    parser.add_argument("--hop-sec", type=float, default=0.5, help="Hop between windows in seconds")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of sessions in the test split (group shuffle by session)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fall-train-ratio",
        type=float,
        default=0.75,
        help="Fraction of fall (AdultManFall) windows by time assigned to train; rest go to test",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ml_row"))
    parser.add_argument(
        "--export-coreml",
        action="store_true",
        help="After training, export Core ML to --coreml-out-dir (needs coremltools + joblib)",
    )
    parser.add_argument(
        "--coreml-out-dir",
        type=Path,
        default=Path("mobile/assets/coreml"),
        help="Destination for FallDetectorRowRF.mlmodel and manifest JSON",
    )
    parser.add_argument(
        "--coreml-target-name",
        type=str,
        default="fall",
        help="Core ML predicted output feature name",
    )
    args = parser.parse_args()

    sessions = list_row_sessions(args.row_root)
    fused: List[pd.DataFrame] = []
    skipped = 0
    n_fall_sessions = 0
    n_neg_sessions = 0
    for s in sessions:
        df = load_fused_session(s)
        if df is None:
            skipped += 1
            continue
        fused.append(df)
        if s.label == 1:
            n_fall_sessions += 1
        else:
            n_neg_sessions += 1

    if not fused:
        raise SystemExit("No fused sessions; check CSV paths under --row-root")

    hz = np.median([_estimate_imu_hz(d) for d in fused])
    window_samples = max(8, int(round(args.window_sec * hz)))
    hop_samples = max(1, int(round(args.hop_sec * hz)))

    win_df = build_window_dataset(fused, window_samples=window_samples, hop_samples=hop_samples)
    if win_df.empty:
        raise SystemExit("Window dataset empty; shorten --window-sec or add longer clips")

    feature_cols = [c for c in win_df.columns if c not in ("label", "session_group", "window_t_mid")]

    train_idx, test_idx, split_meta = _train_test_indices(
        win_df,
        test_size=args.test_size,
        seed=args.seed,
        fall_time_train_ratio=args.fall_train_ratio,
    )

    X_tr = win_df.loc[train_idx, feature_cols]
    X_te = win_df.loc[test_idx, feature_cols]
    y_tr = win_df.loc[train_idx, "label"].to_numpy(dtype=int)
    y_te = win_df.loc[test_idx, "label"].to_numpy(dtype=int)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    proba = _positive_class_proba(clf, X_te)
    pred = (proba >= 0.5).astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "fall_activity_folder": FALL_ACTIVITY,
        "label_rule": f"label=1 iff activity folder name == {FALL_ACTIVITY!r}; all other folders label=0",
        "sessions_total": len(sessions),
        "sessions_skipped_missing_imu": skipped,
        "sessions_fall_used": int(n_fall_sessions),
        "sessions_nonfall_used": int(n_neg_sessions),
        "estimated_imu_hz_median": float(hz),
        "window_samples": int(window_samples),
        "hop_samples": int(hop_samples),
        "n_windows": int(len(win_df)),
        "n_windows_train": int(len(train_idx)),
        "n_windows_test": int(len(test_idx)),
        "warning_single_fall_session": bool(n_fall_sessions <= 1),
        **split_meta,
    }
    with open(args.out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    report = classification_report(y_te, pred, output_dict=True, zero_division=0, labels=[0, 1])
    cm = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
    out_metrics: dict = {"confusion_matrix": cm, "classification_report": report}
    try:
        auc = float(roc_auc_score(y_te, proba))
        out_metrics["roc_auc"] = auc if auc == auc else None  # NaN -> None for JSON
    except ValueError:
        out_metrics["roc_auc"] = None

    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)

    try:
        import joblib
    except ImportError:  # pragma: no cover
        import pickle

        model_path = args.out_dir / "fall_rf_row.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": clf, "feature_cols": feature_cols}, f)
        saved_msg = f"Saved model to {model_path} (pickle; install joblib for .joblib)"
    else:
        model_path = args.out_dir / "fall_rf_row.joblib"
        joblib.dump({"model": clf, "feature_cols": feature_cols}, model_path)
        saved_msg = f"Saved model to {model_path}"

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(out_metrics, indent=2, ensure_ascii=False))
    print(saved_msg)

    if args.export_coreml:
        try:
            from .coreml_export import export_rf_joblib_to_coreml
        except ImportError as e:  # pragma: no cover
            print(f"Core ML export skipped (import): {e}", file=sys.stderr)
        else:
            try:
                if not model_path.is_file():
                    raise FileNotFoundError(model_path)
                out_m, out_manifest = export_rf_joblib_to_coreml(
                    model_path,
                    args.coreml_out_dir,
                    target_name=args.coreml_target_name,
                    window_samples=int(window_samples),
                    hop_samples=int(hop_samples),
                )
                print(f"Core ML: {out_m}")
                print(f"Core ML manifest: {out_manifest}")
            except Exception as e:  # pragma: no cover
                print(f"Core ML export failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
