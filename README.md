# Sentio V2.0: Dual-Track Biometric Fall Detection System

Sentio V2.0 is a research-oriented IoT framework designed for robust human fall detection. By integrating multi-modal sensor data (3-axis Accelerometer, 3-axis Gyroscope, and Barometric Altimeter), the system implements a **Dual-Track Decision Logic** to minimize false alarms while maintaining high sensitivity for complex fall scenarios.

## 🚀 Key Features

- **Signal Conditioning:** Real-time barometer denoising via a 2nd-order Butterworth low-pass filter.
- **Dual-Track Fusion:** - `Track A`: High-G impact detection and SVM (Signal Vector Magnitude) analysis.
    - `Track B`: Barometric altitude change monitoring and posture timeout verification.
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

## 📈 Data Schema
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