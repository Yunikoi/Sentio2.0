"""Train a binary fall classifier on data_sensor/row (AdultManFall = 1, all else = 0)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .row_loader import FALL_ACTIVITY, list_row_sessions
from .row_session import load_fused_session
from .row_windows import build_window_dataset


def _peak_acc_vector_max_norm(X: pd.DataFrame) -> Optional[np.ndarray]:
    """Per-window score: L2 norm of per-axis max acceleration inside the window (same split as RF)."""
    cols = ("acc_x_max", "acc_y_max", "acc_z_max")
    if not all(c in X.columns for c in cols):
        return None
    a = X[list(cols)].to_numpy(dtype=np.float64)
    return np.sqrt(np.sum(a * a, axis=1))


def _fit_threshold_max_f1_positive(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold for score >= T maximizing F1 for label 1; fit on train only."""
    if scores.size == 0:
        return 0.0
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0
    if hi <= lo:
        return lo
    qs = np.linspace(0, 100, 201)
    cand = np.unique(np.concatenate([np.percentile(scores, qs), scores]))
    best_f1 = -1.0
    best_thr = float(cand[0])
    for thr in cand:
        pred = (scores >= float(thr)).astype(int)
        f1 = f1_score(y, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _timed_predict_ms(pred_fn, n: int) -> Tuple[float, float]:
    """Return (total_ms, per_sample_ms) for pred_fn() which must run one batch."""
    t0 = time.perf_counter()
    pred_fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per = elapsed_ms / max(1, n)
    return elapsed_ms, per


def _fall_class_proba(clf: Any, X: Any) -> np.ndarray:
    """Probability of label 1; zeros if the model was fit on a single class."""
    classes = getattr(clf, "classes_", None)
    if classes is None or len(classes) == 0:
        return np.zeros(len(X), dtype=np.float64)
    if 1 not in clf.classes_:
        return np.zeros(len(X), dtype=np.float64)
    col = list(clf.classes_).index(1)
    return clf.predict_proba(X)[:, col].astype(np.float64)


def _report_scalar_scores(report: dict) -> Dict[str, float]:
    return {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "fall_f1": float(report["1"]["f1-score"]) if "1" in report else 0.0,
    }


def _print_run_line(tag: str, scores: Dict[str, float], roc_auc_val, latency_ms_per: float) -> None:
    auc_s = (
        f"{float(roc_auc_val):.4f}"
        if roc_auc_val is not None and roc_auc_val == roc_auc_val
        else "n/a"
    )
    print(
        f"[{tag}] macro_f1={scores['macro_f1']:.4f} fall_f1={scores['fall_f1']:.4f} "
        f"roc_auc={auc_s} latency_ms_per_window_avg={latency_ms_per:.4f}"
    )


def _print_aggregate(agg: Dict[str, Any]) -> None:
    print(f"[aggregate over {agg['n_runs']} runs] mean ± std (macro_f1 / fall_f1 / roc_auc where applicable)")
    for name in ("baseline_peak_acc", "logistic_regression", "random_forest"):
        block = agg.get(name)
        if not block or block.get("skipped"):
            continue
        mf_m, mf_s = block["macro_f1_mean"], block["macro_f1_std"]
        ff_m, ff_s = block["fall_f1_mean"], block["fall_f1_std"]
        line = f"  {name}: macro_f1 {mf_m:.4f}±{mf_s:.4f}  fall_f1 {ff_m:.4f}±{ff_s:.4f}"
        if "roc_auc_mean" in block:
            am, astd = block["roc_auc_mean"], block["roc_auc_std"]
            auc_part = f"  roc_auc {am:.4f}±{astd:.4f}" if am == am else "  roc_auc n/a"
            line += auc_part
        if "latency_ms_per_window_avg_mean" in block:
            lm, ls = block["latency_ms_per_window_avg_mean"], block["latency_ms_per_window_avg_std"]
            line += f"  latency_ms/w {lm:.4f}±{ls:.4f}"
        print(line)


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


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    try:
        v = float(roc_auc_score(y_true, proba))
        return v if v == v else None
    except ValueError:
        return None


def evaluate_one_seed(
    win_df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float,
    fall_time_train_ratio: float,
    seed: int,
) -> Tuple[dict, RandomForestClassifier, Optional[LogisticRegression], Optional[StandardScaler]]:
    """One split + threshold baseline + LR + RF. Returns (json-able run dict, clf_rf, clf_lr|None, scaler|None)."""
    train_idx, test_idx, split_meta = _train_test_indices(
        win_df,
        test_size=test_size,
        seed=seed,
        fall_time_train_ratio=fall_time_train_ratio,
    )

    X_tr = win_df.loc[train_idx, feature_cols]
    X_te = win_df.loc[test_idx, feature_cols]
    y_tr = win_df.loc[train_idx, "label"].to_numpy(dtype=int)
    y_te = win_df.loc[test_idx, "label"].to_numpy(dtype=int)
    n_te = int(len(X_te))

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    t_rf0 = time.perf_counter()
    proba = _fall_class_proba(clf, X_te)
    rf_total_ms = (time.perf_counter() - t_rf0) * 1000.0
    rf_per_ms = rf_total_ms / max(1, n_te)
    pred = (proba >= 0.5).astype(int)
    report_rf = classification_report(y_te, pred, output_dict=True, zero_division=0, labels=[0, 1])
    cm_rf = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
    roc_rf = _safe_roc_auc(y_te, proba)

    baseline_block: dict = {"skipped": True, "reason": "missing acc_x_max/acc_y_max/acc_z_max in features"}
    scores_tr = _peak_acc_vector_max_norm(X_tr)
    scores_te = _peak_acc_vector_max_norm(X_te)
    bl_total_ms = 0.0
    bl_per_ms = 0.0
    pred_bl: Optional[np.ndarray] = None
    if scores_tr is not None and scores_te is not None:
        thr = _fit_threshold_max_f1_positive(scores_tr, y_tr)

        def _run_bl() -> None:
            nonlocal pred_bl
            pred_bl = (scores_te >= thr).astype(int)

        bl_total_ms, bl_per_ms = _timed_predict_ms(_run_bl, n_te)
        assert pred_bl is not None
        report_bl = classification_report(y_te, pred_bl, output_dict=True, zero_division=0, labels=[0, 1])
        cm_bl = confusion_matrix(y_te, pred_bl, labels=[0, 1]).tolist()
        baseline_block = {
            "skipped": False,
            "name": "peak_acc_vector_max_norm_threshold",
            "description": (
                "Binary pred: sqrt(acc_x_max^2+acc_y_max^2+acc_z_max^2) >= T. "
                "T fit on train windows only to maximize F1 (positive class = fall)."
            ),
            "threshold_fit_on_train": thr,
            "confusion_matrix": cm_bl,
            **_report_scalar_scores(report_bl),
        }

    clf_lr: Optional[LogisticRegression] = None
    scaler: Optional[StandardScaler] = None
    lr_block: dict = {"skipped": False, "name": "logistic_regression_scaled"}
    lr_total_ms = 0.0
    lr_per_ms = 0.0
    roc_lr: Optional[float] = None
    try:
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr.to_numpy(dtype=np.float64))
        X_tes = scaler.transform(X_te.to_numpy(dtype=np.float64))
        clf_lr = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
        clf_lr.fit(X_trs, y_tr)
        t_lr0 = time.perf_counter()
        proba_lr = _fall_class_proba(clf_lr, X_tes)
        lr_total_ms = (time.perf_counter() - t_lr0) * 1000.0
        lr_per_ms = lr_total_ms / max(1, n_te)
        pred_lr = (proba_lr >= 0.5).astype(int)
        report_lr = classification_report(y_te, pred_lr, output_dict=True, zero_division=0, labels=[0, 1])
        cm_lr = confusion_matrix(y_te, pred_lr, labels=[0, 1]).tolist()
        roc_lr = _safe_roc_auc(y_te, proba_lr)
        lr_block = {
            "skipped": False,
            "name": "logistic_regression_scaled",
            "description": "StandardScaler on window features + sklearn LogisticRegression(class_weight=balanced).",
            "confusion_matrix": cm_lr,
            **_report_scalar_scores(report_lr),
            "roc_auc": roc_lr,
            "latency_ms_total": lr_total_ms,
            "latency_ms_per_window_avg": lr_per_ms,
        }
    except ValueError as e:
        lr_block = {"skipped": True, "reason": f"logistic_regression: {e}"}
        clf_lr = None
        scaler = None

    run_out = {
        "seed": int(seed),
        "split_meta": split_meta,
        "n_windows_train": int(len(train_idx)),
        "n_windows_test": n_te,
        "random_forest": {
            "confusion_matrix": cm_rf,
            **_report_scalar_scores(report_rf),
            "roc_auc": roc_rf,
            "latency_ms_total": rf_total_ms,
            "latency_ms_per_window_avg": rf_per_ms,
        },
        "logistic_regression": lr_block,
        "baseline_peak_acc": baseline_block,
        "latency_ms_on_test_windows": {
            "random_forest_total": rf_total_ms,
            "random_forest_per_window_avg": rf_per_ms,
            "baseline_total": bl_total_ms,
            "baseline_per_window_avg": bl_per_ms,
            "logistic_regression_total": lr_total_ms,
            "logistic_regression_per_window_avg": lr_per_ms,
        },
    }
    return run_out, clf, clf_lr, scaler


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    s = float(np.std(a, ddof=0)) if a.size > 1 else 0.0
    return m, s


def _aggregate_from_runs(runs: List[dict]) -> dict:
    def collect(path: List[str]) -> List[float]:
        out: List[float] = []
        for r in runs:
            cur: Any = r
            ok = True
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if not ok:
                continue
            if isinstance(cur, (int, float)) and cur == cur:
                out.append(float(cur))
        return out

    lat_key_for = {
        "baseline_peak_acc": "baseline_per_window_avg",
        "logistic_regression": "logistic_regression_per_window_avg",
        "random_forest": "random_forest_per_window_avg",
    }

    def block_for(prefix: str) -> dict:
        mf = collect([prefix, "macro_f1"])
        ff = collect([prefix, "fall_f1"])
        if not mf:
            return {"skipped": True, "reason": "no numeric summaries (method skipped or missing)"}
        roc = collect([prefix, "roc_auc"]) if prefix != "baseline_peak_acc" else []
        lat_key = lat_key_for[prefix]
        lat_list: List[float] = []
        for r in runs:
            v = r.get("latency_ms_on_test_windows", {}).get(lat_key)
            if isinstance(v, (int, float)) and v == v:
                lat_list.append(float(v))

        out: Dict[str, Any] = {
            "skipped": False,
            "macro_f1_mean": _mean_std(mf)[0],
            "macro_f1_std": _mean_std(mf)[1],
            "fall_f1_mean": _mean_std(ff)[0],
            "fall_f1_std": _mean_std(ff)[1],
        }
        if roc:
            rm, rs = _mean_std(roc)
            out["roc_auc_mean"] = rm
            out["roc_auc_std"] = rs
        if lat_list:
            lm, ls = _mean_std(lat_list)
            out["latency_ms_per_window_avg_mean"] = lm
            out["latency_ms_per_window_avg_std"] = ls
        return out

    n = len(runs)
    return {
        "n_runs": n,
        "random_forest": block_for("random_forest"),
        "logistic_regression": block_for("logistic_regression"),
        "baseline_peak_acc": block_for("baseline_peak_acc"),
    }


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
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for the first run")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of repeated evaluations with seeds seed, seed+1, ... (report mean±std)",
    )
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

    n_runs = max(1, int(args.n_runs))
    runs: List[dict] = []
    best_macro = -1.0
    best_clf: Optional[RandomForestClassifier] = None
    best_seed = args.seed
    best_run_idx = 0

    for i in range(n_runs):
        seed = int(args.seed + i)
        run_out, clf_rf, _, _ = evaluate_one_seed(
            win_df,
            feature_cols,
            test_size=args.test_size,
            fall_time_train_ratio=args.fall_train_ratio,
            seed=seed,
        )
        runs.append(run_out)
        mf = float(run_out["random_forest"]["macro_f1"])
        if mf > best_macro:
            best_macro = mf
            best_clf = clf_rf
            best_seed = seed
            best_run_idx = i

    assert best_clf is not None
    aggregate = _aggregate_from_runs(runs)
    best_run = runs[best_run_idx]

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
        "n_eval_runs": n_runs,
        "eval_seeds": [int(args.seed + i) for i in range(n_runs)],
        "artifact_policy": "random_forest_joblib_from_run_with_highest_test_macro_f1",
        "artifact_run_index": int(best_run_idx),
        "artifact_seed": int(best_seed),
        "warning_single_fall_session": bool(n_fall_sessions <= 1),
    }
    with open(args.out_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Full metrics for best RF run (backward-compatible top-level keys)
    br_rf = best_run["random_forest"]
    out_metrics: dict = {
        "confusion_matrix": br_rf["confusion_matrix"],
        "classification_report": {},  # filled below from reconstructed preds unavailable; keep compact
        "roc_auc": br_rf.get("roc_auc"),
        "n_runs": n_runs,
        "runs": runs,
        "aggregate_mean_std": aggregate,
        "best_run_by_random_forest_macro_f1": best_run,
        "baseline_vs_rf_same_test_split": {
            "note": "Per-run fields use *_peak_acc / random_forest / logistic_regression; see runs[] and aggregate_mean_std.",
            "random_forest": {k: br_rf[k] for k in br_rf if k != "confusion_matrix"},
            "latency_ms_on_test_windows": best_run["latency_ms_on_test_windows"],
            "baseline": best_run["baseline_peak_acc"],
            "logistic_regression": best_run["logistic_regression"],
        },
    }
    # Rebuild classification_report dict for best run RF (for tools expecting full report)
    train_idx, test_idx, _ = _train_test_indices(
        win_df,
        test_size=args.test_size,
        seed=best_seed,
        fall_time_train_ratio=args.fall_train_ratio,
    )
    X_te = win_df.loc[test_idx, feature_cols]
    y_te = win_df.loc[test_idx, "label"].to_numpy(dtype=int)
    pred_best = (_fall_class_proba(best_clf, X_te) >= 0.5).astype(int)
    out_metrics["classification_report"] = classification_report(
        y_te, pred_best, output_dict=True, zero_division=0, labels=[0, 1]
    )

    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)

    try:
        import joblib
    except ImportError:  # pragma: no cover
        import pickle

        model_path = args.out_dir / "fall_rf_row.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": best_clf, "feature_cols": feature_cols}, f)
        saved_msg = f"Saved model to {model_path} (pickle; install joblib for .joblib)"
    else:
        model_path = args.out_dir / "fall_rf_row.joblib"
        joblib.dump({"model": best_clf, "feature_cols": feature_cols}, model_path)
        saved_msg = f"Saved model to {model_path}"

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[best run index={best_run_idx} seed={best_seed}] detail:")
    br = best_run["random_forest"]
    _print_run_line("RandomForest (best run)", {k: br[k] for k in ("macro_f1", "fall_f1")}, br.get("roc_auc"), br["latency_ms_per_window_avg"])
    lr = best_run["logistic_regression"]
    if not lr.get("skipped"):
        _print_run_line(
            "LogisticRegression (best run)",
            {k: lr[k] for k in ("macro_f1", "fall_f1")},
            lr.get("roc_auc"),
            float(lr.get("latency_ms_per_window_avg", 0.0)),
        )
    bl = best_run["baseline_peak_acc"]
    if not bl.get("skipped"):
        thr = bl.get("threshold_fit_on_train")
        thr_s = f"{float(thr):.6g}" if isinstance(thr, (int, float)) and thr == thr else "n/a"
        print(
            f"[baseline_peak_acc best run] macro_f1={bl['macro_f1']:.4f} fall_f1={bl['fall_f1']:.4f} "
            f"threshold_fit_on_train={thr_s} latency_ms_per_window_avg={best_run['latency_ms_on_test_windows']['baseline_per_window_avg']:.4f}"
        )
    else:
        print(f"[baseline skipped] {bl.get('reason', '')}")

    _print_aggregate(aggregate)
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
