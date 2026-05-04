# Sentio 2.0 — ML evaluation (one page)

**Generated:** 2026-05-04 04:39 UTC  
**Source:** `D:/Japan/Sentio2.0/outputs/ml_row/metrics.json`

## Setup

- **Sessions (used):** fall=1, non-fall=8, skipped_missing_imu=1
- **Windows:** total=195, window_samples=101, hop_samples=25
- **IMU rate (median est.):** 50.37510761284006 Hz
- **Multi-run:** N=5, seeds=[42, 43, 44, 45, 46]
- **Saved RF checkpoint:** seed=42 (run index 0, best test macro-F1 among runs)
- **Label rule:** label=1 iff activity folder name == 'AdultManFall'; all other folders label=0
- **Caveat:** `warning_single_fall_session=true` — fall windows may all sit in train; treat metrics as exploratory unless you add more fall sessions.

## Method hierarchy — multi-run mean ± std

| Method | Type | macro-F1 | fall-F1 | ROC-AUC | latency ms/window |
|---|---|---|---|---|---|
| Threshold | heuristic | 0.76 ± 0.23 | 0.64 ± 0.29 | — | 0.0001 ± 0.0002 |
| Logistic Regression | classical ML | 0.88 ± 0.15 | 0.76 ± 0.29 | 1.00 ± 0.00 | 0.0065 ± 0.0098 |
| Random Forest | advanced ML | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.3607 ± 2.0206 |

## One-line takeaway (copy)

  Baseline macro-F1: 0.76 ± 0.23  |  LogReg macro-F1: 0.88 ± 0.15  |  RF macro-F1: 1.00 ± 0.00

## Best RF run — confusion matrix (test)

seed=42, ROC-AUC=1.0

| | pred 0 | pred 1 |
|:---:|:---:|:---:|
| **true 0** | 124 | 0 |
| **true 1** | 0 | 1 |

## Cold-email snippet (English, factual)

Use 2–3 sentences; attach this PDF or link the repo.

> I am applying to [program] and am interested in your work on [topic]. For my current project (Sentio 2.0), I collected on-device IMU/barometer sessions and evaluated a three-tier pipeline (threshold → logistic regression → random forest) under 5 random session splits; headline numbers are in the attached one-page summary (mean ± std macro-F1). I would appreciate the chance to discuss whether this direction fits your lab.

