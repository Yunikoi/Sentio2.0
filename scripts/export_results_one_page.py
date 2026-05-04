"""
Build a one-page results summary from outputs/ml_row/metrics.json (+ train_meta.json).

Usage:
  python scripts/export_results_one_page.py
  python scripts/export_results_one_page.py --metrics outputs/ml_row/metrics.json --pdf
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pm(m: float, s: float, nd: int = 2) -> str:
    if m != m:
        return "n/a"
    ss = 0.0 if (s != s) else s
    return f"{m:.{nd}f} ± {ss:.{nd}f}"


def _agg_row(
    agg: Dict[str, Any],
    key: str,
    label: str,
    tier: str,
) -> Optional[Tuple[str, str, str, str, str, str]]:
    b = agg.get(key) or {}
    if b.get("skipped"):
        return None
    mf = _pm(float(b["macro_f1_mean"]), float(b["macro_f1_std"]))
    ff = _pm(float(b["fall_f1_mean"]), float(b["fall_f1_std"]))
    roc = "—"
    if "roc_auc_mean" in b:
        am, ast = float(b["roc_auc_mean"]), float(b["roc_auc_std"])
        roc = _pm(am, ast) if am == am else "n/a"
    lat = "—"
    if "latency_ms_per_window_avg_mean" in b:
        lm, ls = float(b["latency_ms_per_window_avg_mean"]), float(b["latency_ms_per_window_avg_std"])
        lat = _pm(lm, ls, 4)
    return (label, tier, mf, ff, roc, lat)


def build_markdown(metrics: dict, meta: Optional[dict], metrics_path: Path) -> str:
    agg = metrics.get("aggregate_mean_std") or {}
    n_runs = int(agg.get("n_runs", metrics.get("n_runs", 1)))
    seeds = (meta or {}).get("eval_seeds", [])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = [
        "# Sentio 2.0 — ML evaluation (one page)",
        "",
        f"**Generated:** {now}  ",
        f"**Source:** `{metrics_path.as_posix()}`",
        "",
        "## Setup",
        "",
    ]
    if meta:
        lines.extend(
            [
                f"- **Sessions (used):** fall={meta.get('sessions_fall_used', '?')}, "
                f"non-fall={meta.get('sessions_nonfall_used', '?')}, "
                f"skipped_missing_imu={meta.get('sessions_skipped_missing_imu', '?')}",
                f"- **Windows:** total={meta.get('n_windows', '?')}, "
                f"window_samples={meta.get('window_samples', '?')}, hop_samples={meta.get('hop_samples', '?')}",
                f"- **IMU rate (median est.):** {meta.get('estimated_imu_hz_median', '?')} Hz",
                f"- **Multi-run:** N={meta.get('n_eval_runs', n_runs)}, seeds={seeds}",
                f"- **Saved RF checkpoint:** seed={meta.get('artifact_seed', '?')} "
                f"(run index {meta.get('artifact_run_index', '?')}, best test macro-F1 among runs)",
                f"- **Label rule:** {meta.get('label_rule', '(see train_meta.json)')}",
            ]
        )
        if meta.get("warning_single_fall_session"):
            lines.append(
                "- **Caveat:** `warning_single_fall_session=true` — fall windows may all sit in train; "
                "treat metrics as exploratory unless you add more fall sessions."
            )
    else:
        lines.append("- *(No train_meta.json — add `--train-meta` for full setup block.)*")

    lines.extend(["", "## Method hierarchy — multi-run mean ± std", ""])

    rows: List[List[str]] = []
    spec = [
        ("baseline_peak_acc", "Threshold", "heuristic"),
        ("logistic_regression", "Logistic Regression", "classical ML"),
        ("random_forest", "Random Forest", "advanced ML"),
    ]
    for key, lab, tier in spec:
        r = _agg_row(agg, key, lab, tier)
        if r is None:
            continue
        label, t, mf, ff, roc, lat = r
        rows.append([label, t, mf, ff, roc, lat])

    if rows:
        header = "| Method | Type | macro-F1 | fall-F1 | ROC-AUC | latency ms/window |"
        sep = "|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("*No aggregate block found — run `main_train_row` with `--n-runs` first.*")

    lines.extend(["", "## One-line takeaway (copy)", ""])
    one: List[str] = []
    for key, short in (
        ("baseline_peak_acc", "Baseline"),
        ("logistic_regression", "LogReg"),
        ("random_forest", "RF"),
    ):
        b = agg.get(key) or {}
        if b.get("skipped"):
            continue
        one.append(
            f"{short} macro-F1: {_pm(float(b['macro_f1_mean']), float(b['macro_f1_std']))}"
        )
    if one:
        lines.append("  " + "  |  ".join(one))
    else:
        lines.append("  *(No aggregate metrics.)*")

    br = metrics.get("best_run_by_random_forest_macro_f1") or {}
    br_rf = br.get("random_forest") or {}
    cm = br_rf.get("confusion_matrix")
    if cm:
        lines.extend(
            [
                "",
                "## Best RF run — confusion matrix (test)",
                "",
                f"seed={br.get('seed', '?')}, ROC-AUC={br_rf.get('roc_auc', 'n/a')}",
                "",
                "| | pred 0 | pred 1 |",
                "|:---:|:---:|:---:|",
                f"| **true 0** | {cm[0][0]} | {cm[0][1]} |",
                f"| **true 1** | {cm[1][0]} | {cm[1][1]} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Cold-email snippet (English, factual)",
            "",
            "Use 2–3 sentences; attach this PDF or link the repo.",
            "",
            "> I am applying to [program] and am interested in your work on [topic]. "
            "For my current project (Sentio 2.0), I collected on-device IMU/barometer sessions and "
            f"evaluated a three-tier pipeline (threshold → logistic regression → random forest) "
            f"under {n_runs} random session splits; headline numbers are in the attached one-page summary "
            f"(mean ± std macro-F1). I would appreciate the chance to discuss whether this direction fits your lab.",
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def write_pdf(_md_text: str, out_pdf: Path, metrics: dict, meta: Optional[dict]) -> None:
    import matplotlib.pyplot as plt

    agg = metrics.get("aggregate_mean_std") or {}
    n_runs = int(agg.get("n_runs", metrics.get("n_runs", 1)))

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Sentio 2.0 — ML evaluation (one page)",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )

    ax = fig.add_axes([0.08, 0.58, 0.84, 0.32])
    ax.axis("off")
    ax.set_title("Method hierarchy (mean ± std over runs)", fontsize=11, loc="left", pad=12)

    col_labels = ["Method", "Type", "macro-F1", "fall-F1", "ROC-AUC", "ms/window"]
    cell_text: List[List[str]] = []
    spec = [
        ("baseline_peak_acc", "Threshold", "heuristic"),
        ("logistic_regression", "Logistic Reg.", "classical ML"),
        ("random_forest", "Random Forest", "advanced ML"),
    ]
    for key, lab, tier in spec:
        r = _agg_row(agg, key, lab, tier)
        if r is None:
            continue
        cell_text.append(list(r))

    if not cell_text:
        cell_text = [["(no aggregate)", "—", "—", "—", "—", "—"]]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")

    y = 0.52
    fig.text(0.08, y, f"Multi-run: N = {n_runs}", fontsize=10, fontweight="bold")
    y -= 0.03
    if meta:
        fig.text(
            0.08,
            y,
            f"Seeds: {meta.get('eval_seeds', [])}  |  sessions fall/non-fall: "
            f"{meta.get('sessions_fall_used', '?')}/{meta.get('sessions_nonfall_used', '?')}",
            fontsize=9,
        )
        y -= 0.025
        if meta.get("warning_single_fall_session"):
            fig.text(
                0.08,
                y,
                "Caveat: single fall session — metrics may be optimistic for fall class.",
                fontsize=8,
                style="italic",
            )
            y -= 0.025

    br = metrics.get("best_run_by_random_forest_macro_f1") or {}
    br_rf = br.get("random_forest") or {}
    cm = br_rf.get("confusion_matrix")
    if cm:
        y -= 0.02
        fig.text(0.08, y, "Best RF run — test confusion matrix", fontsize=10, fontweight="bold")
        y -= 0.03
        fig.text(0.08, y, f"seed={br.get('seed')}  ROC-AUC={br_rf.get('roc_auc')}", fontsize=9)
        y -= 0.035
        fig.text(0.08, y, f"[[TN, FP], [FN, TP]] = {cm}", fontsize=9, family="monospace")

    y = 0.22
    fig.text(0.08, y, "One-line takeaway", fontsize=10, fontweight="bold")
    y -= 0.028
    take = []
    for key, short in (
        ("baseline_peak_acc", "Baseline"),
        ("logistic_regression", "LogReg"),
        ("random_forest", "RF"),
    ):
        b = agg.get(key) or {}
        if b.get("skipped"):
            continue
        take.append(
            f"{short}: {_pm(float(b['macro_f1_mean']), float(b['macro_f1_std']))}"
        )
    fig.text(0.08, y, "  |  ".join(take) if take else "(no aggregate)", fontsize=9)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one-page ML results (Markdown + optional PDF).")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/ml_row/metrics.json"),
        help="Path to metrics.json from main_train_row",
    )
    parser.add_argument(
        "--train-meta",
        type=Path,
        default=None,
        help="Path to train_meta.json (default: metrics directory / train_meta.json)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output Markdown path (default: same dir as metrics / RESULTS_ONE_PAGE.md)",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Output PDF path (default: same dir as metrics / RESULTS_ONE_PAGE.pdf if --pdf)",
    )
    parser.add_argument("--pdf", action="store_true", help="Also write a one-page PDF (matplotlib)")
    args = parser.parse_args()

    metrics_path = args.metrics.resolve()
    if not metrics_path.is_file():
        raise SystemExit(f"Missing metrics file: {metrics_path}")

    meta_path = args.train_meta
    if meta_path is None:
        meta_path = metrics_path.parent / "train_meta.json"
    meta: Optional[dict] = None
    if meta_path.is_file():
        meta = _load_json(meta_path)

    metrics = _load_json(metrics_path)
    md = build_markdown(metrics, meta, metrics_path)

    out_md = args.out_md or (metrics_path.parent / "RESULTS_ONE_PAGE.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md}")

    if args.pdf:
        out_pdf = args.out_pdf or (metrics_path.parent / "RESULTS_ONE_PAGE.pdf")
        write_pdf(md, out_pdf, metrics, meta)
        print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
