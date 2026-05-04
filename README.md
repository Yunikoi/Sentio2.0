# Sentio V2.0: Smartphone Fall Detection Without Extra Wearables

Sentio V2.0 is a research codebase for **high-precision fall detection on a commodity smartphone**—the device older adults already carry—so monitoring does **not** depend on dedicated pendants, wristbands, or other “medical” wearables that raise **adherence, stigma, and psychological burden**.

**Global positioning.** **This work focuses on protocol design in ubiquitous sensing rather than model benchmarking**—i.e., we read as **systems + sensing researchers** who specify **tasks, splits, and baselines** on commodity phones, not as **benchmark competitors** optimizing a leaderboard architecture. **Scope (still true):** we do **not** claim public-dataset SOTA and we are **not** a regulated medical device; those lines delimit liability, but they do **not** define the intellectual identity of the project.

### Research problem

**What is still unsolved?** Useful fall alerts require both sensitivity and trust. Dedicated wearables can work well technically, but many older adults resist, forget, or feel labeled by extra devices; purely “invisible” home sensors are not always acceptable either. The phone is a **daily-use, non–medical-looking** platform, yet **phone-only inertial sensing** remains ambiguous: vigorous daily motion and soft falls are easy to confuse with true falls.

**Why are existing phone-centric approaches often insufficient?** Classic pipelines that fire on **impact peaks or IMU-only windows** inherit that ambiguity: they tend to trade off false alarms against missed falls (see **Error Analysis** below). Methods that assume a **fixed wearable placement** or **clean lab conditions** do not fully match **pocket- or hand-carried** phone use in real routines.

**What assumption does this work test?** We treat the smartphone as the **sole sensing surface** (no added body-worn hardware) and ask whether **multimodal context on the same device**—**IMU + barometric altitude**, combined with **dual-track event logic** (impact gating plus height/posture consistency)—can **sharpen** fall vs non-fall decisions compared to **impact-only or IMU-window baselines**, while staying **psychologically lightweight** for users who already keep a phone with them.

A complementary **method-facing question** is encoded in optional **3-class window training** (`--task multiclass`): can the same features separate **person fall** (`AdultManFall`), **device-only drop** (`IndoorPhoneFall`), and **everyday ADL**? That targets the “big motion but not a person fall” ambiguity without claiming a novel backbone—**the task and evaluation protocol** are the explicit research handle.

Empirical claims are backed by the **offline dual-track ablations** (`main_offline`) and the **window-level method hierarchy** on session data (`main_train_row`); reported numbers depend on your dataset and split design.

### Core novelty (anchor sentence)

> **We reformulate smartphone fall detection as a cross-modal temporal consistency problem under sensor ambiguity.**

Use this line as the **single hook** in SOPs, talks, and the opening of a short paper; the bullets below are the **expandable claims** it organizes.

### Key contributions (SOP / abstract–ready one-liners)

Copy verbatim when you need **non-boilerplate** contribution lines; the repository is the evidence section.

- **Insight.** Smartphone-only fall detection requires **cross-modal temporal consistency** beyond inertial peaks.
- **Protocol.** We introduce a **smartphone-native evaluation protocol** separating **person falls** from **device-only impacts** under **session-consistent splits**.
- **Experimental.** We isolate **representation gains** via a **fixed baseline hierarchy** with **repeated splits** and **mean–variance** reporting.
- **System.** We ship a **deployable sliding-window pipeline** (Python ↔ mobile feature parity, **per-window latency** in `metrics.json`, optional **Core ML**) so claims connect to **on-device** behavior.

*(Empirical strength depends on your recordings; see **Baseline gap analysis**, **Ablation conclusions**, and `train_meta.json` caveats.)*

### Abstract (workshop / arXiv–style)

Fall detection for older adults often assumes **extra wearables**, which can hurt **adherence** and feel stigmatizing. **Smartphones** are already carried daily and avoid that burden, but **pocket- and hand-carried** IMU streams conflate **person falls**, **device-only drops**, and **vigorous activities** behind similar acceleration peaks. **Sentio V2.0** reframes the problem as **cross-modal temporal consistency under ambiguity** and studies **phone-only** monitoring using **IMU plus on-device barometric altitude** in a **dual-track detector** (high-energy inertial gating with slower height/posture context on denoised pressure). We release an **evaluation stack** that (i) runs **offline ablations** isolating barometer and posture-height logic, and (ii) trains **window classifiers** under a **nested baseline hierarchy** (threshold → logistic regression → random forest) with **repeated random splits** and optional **three-class** training. **This work focuses on protocol design in ubiquitous sensing rather than model benchmarking** (task definition, session-consistent splits, nested baselines—not a new leaderboard architecture); **external benchmark alignment** remains future work. The system is **not** SOTA-by-default and **not** a regulated medical device.

---

## 🚀 Key Features

- **Signal conditioning:** Real-time barometer denoising via a 2nd-order Butterworth low-pass filter.
- **Dual-track fusion:**
  - **Track A:** High-G impact gating using acceleration and gyroscope norms (not a separate SVM classifier in this path).
  - **Track B:** Barometric altitude change and posture tilt with timeout verification.
- **Feature Bank:** Automated construction of time-series feature sets with millisecond-level alignment.
- **Evaluation Suite:** Comprehensive offline evaluation pipeline supporting ablation studies and performance metrics (Precision, Recall, F1-Score).
- **Visualization:** Integrated 5D physical-state dashboard for in-depth event analysis.

## 🛠️ Project Structure

- `src/sentio_v2/`: Core algorithm implementation and detection logic.
- `scripts/`: Utility scripts for synthetic data generation and data cleaning.
- `data/`: Local storage for raw and processed sensor datasets.
- `outputs/`: Performance reports and visualization results.

## 💻 Installation

Ensuring your Python environment is ready for signal processing:

```bash
pip install -r requirements.txt
```

## 📊 Quick Start

### 1. Data Preparation (Simulation)
Generate a standardized benchmark dataset for testing the pipeline:

```bash
python scripts/generate_synthetic_dataset.py
```

### 2. Run Offline Evaluation & Ablation Study
Validate the dual-track logic against the dataset and analyze the impact of the barometric component:

```bash
python -m src.sentio_v2.main_offline --data-dir data/day2 --out-dir outputs
```

### 3. Real-time Simulation Replay
Simulate the hardware runtime environment using pre-recorded data at 50Hz:

```bash
python -m src.sentio_v2.main_realtime --clip data/day2/soft_fall_00.csv --replay-hz 50
```

### 4. Row-format sessions: baselines, repeated splits, metrics
Train/evaluate on `data_sensor/row` (fused IMU + optional barometer). The script runs **multi-run** evaluation: `for seed in [seed0, seed0+1, …]: train + evaluate`, then reports **macro-F1 (and fall-F1) as mean ± std**—this is the usual “project → research” upgrade. Within each run, methods share the **same** split and form an explicit **method hierarchy**: **Threshold** (heuristic) → **Logistic Regression** (classical ML, `StandardScaler`) → **Random Forest** (advanced ML). The saved `fall_rf_row.joblib` is the RF from the run with the **highest test macro-F1** among runs.

```bash
# Seeds [0,1,2,3,4] with 5 runs:
python -m src.sentio_v2.main_train_row --row-root data_sensor/row --out-dir outputs/ml_row --n-runs 5 --seed 0
```

The console prints a **method hierarchy** block (e.g. `RF macro-F1: 0.91 ± 0.02`, `Baseline macro-F1: 0.73 ± 0.03`—numbers depend on your data). Artifacts: `outputs/ml_row/metrics.json` (`runs`, `aggregate_mean_std`, `method_hierarchy`, `best_run_by_random_forest_macro_f1`), `outputs/ml_row/train_meta.json`.

**Multiclass (person fall vs phone drop vs ADL):** use `--task multiclass` and a separate `--out-dir` (e.g. `outputs/ml_row_multiclass`) so you do not overwrite the binary Core ML bundle. Split policy: **ADL** windows are grouped by session (same as binary negatives); **person fall** and **phone drop** windows each use a **chronological** train/test fraction (`--fall-train-ratio`). Metrics include **macro-F1**, **fall-F1** (class 1), and **phone-drop-F1** (class 2). The peak-acceleration **threshold baseline is skipped** in this mode (it is defined only for binary person-fall vs rest). Model artifact: `fall_rf_row_multiclass.joblib`. **Core ML export** applies to `--task binary` only.

```bash
python -m src.sentio_v2.main_train_row --row-root data_sensor/row --out-dir outputs/ml_row_multiclass --task multiclass --n-runs 5 --seed 0
```

## Baseline gap analysis (window-level `main_train_row`)

**What counts as a baseline here:** All methods see the **same** train/test split each run. The hierarchy is intentional: **(1) Threshold** — peak resultant acceleration in the window, threshold chosen on the **training** set to maximize fall F1; **(2) Logistic regression** — `StandardScaler` + linear model with class weighting; **(3) Random forest** — nonlinear ensemble on the same hand-crafted window statistics.

**How to read the gap:** After training, open `outputs/ml_row/metrics.json` → `aggregate_mean_std`. Compare `macro_f1_mean` (and `fall_f1_mean`) across `baseline_peak_acc`, `logistic_regression`, and `random_forest`. The **macro–F1 gap** (RF minus threshold baseline on `macro_f1_mean`) summarizes how much structure beyond a single impact feature buys you **on your sessions**; latency gaps are in `latency_ms_per_window_avg_mean` for deployment discussion.

**Caveats:** Gaps shrink when the dataset is easy or when **fall windows are scarce** (see `train_meta.json`: `warning_single_fall_session`). High scores are not automatically “SOTA”; they may reflect split leakage or a narrow activity mix. Use **`--n-runs ≥ 5`** and report **mean ± std**, not a single seed.

Optional one-page text: `python scripts/export_results_one_page.py --metrics outputs/ml_row/metrics.json` → `RESULTS_ONE_PAGE.md`.

## Relation to public benchmarks (“SOTA”) — honest positioning

**Academic positioning:** **This work focuses on protocol design in ubiquitous sensing rather than model benchmarking**—same framing as the **Global positioning** line at the top of this README.

This repository is **not** a drop-in reproduction of a single published leaderboard. Recordings under `data_sensor/row` are **custom phone sessions** (pocket/hand, mixed ADL, optional barometer), evaluated **per sliding window** (and optionally **3-class** disambiguation). That protocol differs from most wearable fall-detection papers, which often use **waist-mounted** accelerometers, **different sampling rates**, and public sets such as **SisFall**, **UR Fall Detection Dataset**, **MobFall**, or **KFall** (names vary by survey; pick the set closest to your sensor placement when you cite).

**What you can claim today:** A clear **problem and protocol** (smartphone-only, psychological burden, ambiguity with phone drops) plus **internal baselines** and **ablations** below. **What still reads as weak without extra work:** “We beat SOTA” — to strengthen that narrative, run the **same feature pipeline** (or the same dual-track rules) on at least one **public** corpus with a **documented train/test split**, and tabulate **your numbers vs. a simple published baseline** (e.g., threshold + SVM on the same split), even if your result is only competitive rather than best.

## Ablation conclusions (offline clip-level `main_offline`)

**What is ablated:** `evaluate.ablation_table` compares three clip-level settings: **`full_v2`** (denoised barometer + Track A + Track B), **`no_barometer`** (barometric features zeroed; Track B height tests become uninformative), and **`no_track_b`** (posture/height Track B disabled via huge thresholds — impact-only style gating on Track A).

**How to regenerate:** `python -m src.sentio_v2.main_offline --data-dir data/day2 --out-dir outputs` writes `outputs/ablation.csv` (fall-class F1 and confusion matrix per row).

**Concrete result on the committed snapshot:** The checked-in `outputs/ablation.csv` (from `data/day2`) shows **identical** fall F1 (**0.8**) and the **same** confusion matrix for `full_v2`, `no_barometer`, and `no_track_b`. **Interpretation for that run:** neither removing barometric context nor disabling Track B changed **clip-level** fall vs non-fall decisions — so, for this small benchmark, dual-track extras did not alter outcomes (e.g., Track A alone already fires on all fall clips, or the clip set is too coarse to separate components). **This is still a valid ablation narrative:** it states *what happened* on the bundled data and motivates **richer clips** or **metrics beyond clip F1** (e.g., false alerts per hour) where baro/Track B might matter.

Do **not** stop at “we ran ablation”; always pair the table with **one sentence** of whether each component **hurt, helped, or was neutral** for the metric you care about.

## Error Analysis
Fall detection from a **phone in the pocket or hand** (wrist-like motion is a special case) is inherently ambiguous: **high-energy non-fall activities** (fast sit, jogging, door slam vibration) can resemble impact peaks, while **soft or staged falls** may produce weaker acceleration signatures than rigid-body models assume. In windowed evaluation, errors show up in two ways: **false positives** (non-fall windows scored as fall) inflate alert load and erode trust; **false negatives** (fall windows missed) are safety-critical. When only one fall session exists, the pipeline may place **all fall windows in train** (see `train_meta.json` notes), which makes reported test performance **optimistic for the fall class**—treat single-session setups as exploratory, not definitive. Use **multiple random splits** (`--n-runs`) to quantify variance, inspect per-run confusion matrices in `metrics.json`, and relate mistakes to activity labels and sensor placement.

## System Implications
**Latency and deployment:** Sliding-window inference adds batch predict time per window; `metrics.json` records average milliseconds per test window for threshold, LR, and RF baselines—use these numbers for edge versus cloud discussions (Core ML export is supported from the chosen RF checkpoint). **Privacy:** On-device inference avoids streaming raw sensor traces to a server, which matters for home monitoring. **Robustness:** Barometer-assisted logic (`main_offline` / dual-track path) targets height and posture context; missing or noisy barometer channels change false alarm trade-offs. **Scope:** This repository implements research and engineering prototypes; it is **not** a certified medical device and should not be presented as clinical validation without protocol, ethics review, and regulated study design.

## 📈 Data Schema
The system expects standardized CSV inputs with the following fields:
| Column | Unit | Description |
| :--- | :--- | :--- |
| `timestamp_ms` | ms | Monotonic millisecond timestamp |
| `acc_x/y/z` | g | 3-axis linear acceleration |
| `gyro_x/y/z` | deg/s | 3-axis angular velocity |
| `baro_m` | m | Relative altitude derived from barometric pressure |

## 📡 Future Deployment (IoMT)
- **Edge computing:** Port the `DualTrackDetector` state machine to React Native so inference stays **on the user’s existing phone**—no extra wearable required.
- **Cloud alerting:** Optional alert payload delivery via Firebase Cloud Messaging (FCM).
- **Elder care context:** Low-power sensor polling aligned with **daily-carry** use; still **not** a certified medical device without clinical protocol and regulatory review.

---
© 2026 Sentio Project | Developed by Yuni
