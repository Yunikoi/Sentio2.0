# Sentio V2.0 (Dual-Track Fall Detection)

This repository provides an executable V2.0 prototype for:

- Day 3: barometer denoising with Butterworth low-pass filter
- Day 4: millisecond alignment and feature-bank construction
- Day 5: dual-track decision logic (`Track A` impact + `Track B` height/posture timeout)
- Day 6: offline evaluation and ablation study
- Day 7: 5D physical-state dashboard plotting

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Generate Day-2 style dataset (demo)

```bash
python scripts/generate_synthetic_dataset.py
```

This creates 5 classes x 15 clips under `data/day2`.

## 3) Run offline evaluation + ablation

```bash
python -m src.sentio_v2.main_offline --data-dir data/day2 --one-clip data/day2/hard_fall_00.csv --out-dir outputs
```

Outputs:

- `outputs/ablation.csv`
- `outputs/five_dim_dashboard.png`
- `outputs/run_meta.json`

## 4) Realtime replay simulation

```bash
python -m src.sentio_v2.main_realtime --clip data/day2/soft_fall_00.csv --replay-hz 50
```

## CSV schema

Each clip CSV should include:

- `timestamp_ms`
- `acc_x`, `acc_y`, `acc_z` (g)
- `gyro_x`, `gyro_y`, `gyro_z` (deg/s)
- `baro_m` (relative meter scale)

## Notes for Week-2 App deployment

- Keep this Python implementation as the algorithm baseline.
- Port `DualTrackDetector` thresholds and state machine to React Native / Flutter runtime.
- Send alert payload to Firebase when event reason is `track_a_impact` or `track_b_height_posture_timeout`.

