import argparse
import time
from pathlib import Path

import pandas as pd

from .align import align_to_millisecond
from .config import V2Config
from .detector import DualTrackDetector
from .features import build_feature_bank
from .filtering import denoise_barometer
from .io_utils import load_sensor_csv


def stream_rows(df: pd.DataFrame, rate_hz: float):
    sleep_s = 1.0 / rate_hz
    for _, row in df.iterrows():
        yield row
        time.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--replay-hz", type=float, default=50.0)
    args = parser.parse_args()

    cfg = V2Config()
    detector = DualTrackDetector(cfg)

    df = load_sensor_csv(args.clip)
    df = align_to_millisecond(df)
    df = denoise_barometer(
        df,
        sample_rate_hz=cfg.sensor.sample_rate_hz,
        cutoff_hz=cfg.sensor.lowpass_cutoff_hz,
        order=cfg.sensor.lowpass_order,
    )
    feat = build_feature_bank(df)

    print("Start realtime replay...")
    rolling = []
    for row in stream_rows(feat, args.replay_hz):
        rolling.append(row._asdict() if hasattr(row, "_asdict") else dict(row))
        if len(rolling) < 10:
            continue
        frame = pd.DataFrame(rolling[-200:])
        events = detector.run(frame)
        if events:
            latest = events[-1]
            print(f"[ALERT] {latest.timestamp_ms}: {latest.reason}")
            break


if __name__ == "__main__":
    main()

