import math
from pathlib import Path

import numpy as np
import pandas as pd


def make_clip(kind: str, seed: int, duration_s: float = 8.0, fs: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    ts = (t * 1000).astype(int)

    acc = np.ones((n, 3)) * np.array([0.0, 0.0, 1.0])
    gyro = rng.normal(0.0, 12.0, size=(n, 3))
    baro = np.ones(n) * 100.0 + rng.normal(0.0, 0.02, size=n)

    if kind == "walking":
        acc[:, 0] += 0.2 * np.sin(2 * math.pi * 1.8 * t)
        acc[:, 2] += 0.15 * np.sin(2 * math.pi * 1.8 * t + 0.5)
    elif kind == "fast_sit":
        acc[:, 2] -= 0.25 * (t > 2.0)
        baro -= 0.15 * (t > 2.1)
    elif kind == "drop":
        idx = np.argmin(np.abs(t - 3.5))
        acc[idx : idx + 4, :] += np.array([1.5, 1.5, 2.8])
        gyro[idx : idx + 4, :] += 180
    elif kind == "hard_fall":
        idx = np.argmin(np.abs(t - 3.0))
        acc[idx : idx + 5, :] += np.array([2.0, 2.0, 3.3])
        gyro[idx : idx + 8, :] += 340
        baro -= 0.55 * (t > 3.0)
        acc[t > 3.1, 2] = 0.2
    elif kind == "soft_fall":
        baro -= 0.45 * (t > 2.5)
        acc[t > 2.6, 2] = 0.35
        gyro[t > 2.6, :] += 80
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    return pd.DataFrame(
        {
            "timestamp_ms": ts,
            "acc_x": acc[:, 0] + rng.normal(0, 0.05, n),
            "acc_y": acc[:, 1] + rng.normal(0, 0.05, n),
            "acc_z": acc[:, 2] + rng.normal(0, 0.05, n),
            "gyro_x": gyro[:, 0],
            "gyro_y": gyro[:, 1],
            "gyro_z": gyro[:, 2],
            "baro_m": baro,
        }
    )


def main() -> None:
    out_dir = Path("data/day2")
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = ["walking", "fast_sit", "drop", "hard_fall", "soft_fall"]
    for cls in classes:
        for i in range(15):
            df = make_clip(cls, seed=1000 + i)
            df.to_csv(out_dir / f"{cls}_{i:02d}.csv", index=False)
    print(f"Synthetic dataset generated: {out_dir}")


if __name__ == "__main__":
    main()

