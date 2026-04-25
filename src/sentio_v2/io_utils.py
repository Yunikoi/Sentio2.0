from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp_ms",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "baro_m",
}


def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.sort_values("timestamp_ms").reset_index(drop=True)


def load_labeled_folder(root: Path) -> Dict[str, pd.DataFrame]:
    data = {}
    for csv_path in sorted(root.glob("*.csv")):
        data[csv_path.stem] = load_sensor_csv(csv_path)
    if not data:
        raise FileNotFoundError(f"No CSV found in {root}")
    return data

