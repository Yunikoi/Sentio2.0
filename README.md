# Sentio V2.0: Dual-Track Biometric Fall Detection System

Sentio V2.0 is a research-oriented IoT framework designed for robust human fall detection. By integrating multi-modal sensor data (3-axis Accelerometer, 3-axis Gyroscope, and Barometric Altimeter), the system implements a **Dual-Track Decision Logic** to minimize false alarms while maintaining high sensitivity for complex fall scenarios.

##  Key Features

- **Signal Conditioning:** Real-time barometer denoising via a 2nd-order Butterworth low-pass filter....
- **Dual-Track Fusion:** - `Track A`: High-G impact detection and SVM (Signal Vector Magnitude) analysis.
    - `Track B`: Barometric altitude change monitoring and posture timeout verification.
- **Feature Bank:** Automated construction of time-series feature sets with millisecond-level alignment.
- **Evaluation Suite:** Comprehensive offline evaluation pipeline supporting ablation studies and performance metrics (Precision, Recall, F1-Score).
- **Visualization:** Integrated 5D physical-state dashboard for in-depth event analysis.

##  Project Structure

- `src/sentio_v2/`: Core algorithm implementation and detection logic.
- `scripts/`: Utility scripts for synthetic data generation and data cleaning.
- `data/`: Local storage for raw and processed sensor datasets.
- `outputs/`: Performance reports and visualization results.

##  Installation

Ensuring your Python environment is ready for signal processing:

```bash
pip install -r requirements.txt
```

##  Quick Start

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



## Error Analysis
Fall detection from wrist- or pocket-mounted IMU is inherently ambiguous: **high-energy non-fall activities** (fast sit, jogging, door slam vibration) can resemble impact peaks, while **soft or staged falls** may produce weaker acceleration signatures than rigid-body models assume. In windowed evaluation, errors show up in two ways: **false positives** (non-fall windows scored as fall) inflate alert load and erode trust; **false negatives** (fall windows missed) are safety-critical. When only one fall session exists, the pipeline may place **all fall windows in train** (see `train_meta.json` notes), which makes reported test performance **optimistic for the fall class**—treat single-session setups as exploratory, not definitive. Use **multiple random splits** (`--n-runs`) to quantify variance, inspect per-run confusion matrices in `metrics.json`, and relate mistakes to activity labels and sensor placement.

## System Implications
**Latency and deployment:** Sliding-window inference adds batch predict time per window; `metrics.json` records average milliseconds per test window for threshold, LR, and RF baselines—use these numbers for edge versus cloud discussions (Core ML export is supported from the chosen RF checkpoint). **Privacy:** On-device inference avoids streaming raw sensor traces to a server, which matters for home monitoring. **Robustness:** Barometer-assisted logic (`main_offline` / dual-track path) targets height and posture context; missing or noisy barometer channels change false alarm trade-offs. **Scope:** This repository implements research and engineering prototypes; it is **not** a certified medical device and should not be presented as clinical validation without protocol, ethics review, and regulated study design.

##  Data Schema
The system expects standardized CSV inputs with the following fields:
| Column | Unit | Description |
| :--- | :--- | :--- |
| `timestamp_ms` | ms | Monotonic millisecond timestamp |
| `acc_x/y/z` | g | 3-axis linear acceleration |
| `gyro_x/y/z` | deg/s | 3-axis angular velocity |
| `baro_m` | m | Relative altitude derived from barometric pressure |

## 📡 Future Deployment (IoMT)
- **Edge Computing:** Porting the `DualTrackDetector` state machine to React Native for mobile-side inference.
- **Cloud Alerting:** Integrated alert payload delivery via Firebase Cloud Messaging (FCM).
- **Medical IoT:** Optimized for elderly care monitoring with low-power sensor polling.

---
© 2026 Sentio Project | Developed by Yuni
