"""Load and fuse per-session CSV exports (Accelerometer, Gravity, Gyroscope, Orientation, Barometer)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .row_loader import RowSession

BARO_FILE = "Barometer.csv"


def _require_file(session_path: Path, name: str) -> Path:
    p = session_path / name
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def load_fused_session(session: RowSession) -> Optional[pd.DataFrame]:
    """Return a single table on IMU timestamps, or None if core IMU files are missing."""
    path = session.path
    try:
        acc = pd.read_csv(_require_file(path, "Accelerometer.csv"))
        grav = pd.read_csv(_require_file(path, "Gravity.csv"))
        gyro = pd.read_csv(_require_file(path, "Gyroscope.csv"))
        orient = pd.read_csv(_require_file(path, "Orientation.csv"))
    except FileNotFoundError:
        return None

    key = "seconds_elapsed"
    for name, df in ("acc", acc), ("grav", grav), ("gyro", gyro), ("orient", orient):
        if key not in df.columns:
            raise ValueError(f"{name}: missing {key}")

    base = acc[["time", key]].copy()
    base = base.rename(columns={key: "t"})
    base["acc_z"] = acc["z"]
    base["acc_y"] = acc["y"]
    base["acc_x"] = acc["x"]

    grav_slim = grav[[key, "z", "y", "x"]].rename(
        columns={key: "t", "z": "grav_z", "y": "grav_y", "x": "grav_x"}
    )
    gyro_slim = gyro[[key, "z", "y", "x"]].rename(
        columns={key: "t", "z": "gyro_z", "y": "gyro_y", "x": "gyro_x"}
    )
    base = base.merge(grav_slim, on="t", how="inner")
    base = base.merge(gyro_slim, on="t", how="inner")

    orient_cols = [c for c in orient.columns if c not in ("time", key)]
    orient_slim = orient[[key] + orient_cols].rename(columns={key: "t"})
    ren = {c: f"orient_{c}" for c in orient_cols}
    orient_slim = orient_slim.rename(columns=ren)
    base = base.merge(orient_slim, on="t", how="inner")

    baro_path = path / BARO_FILE
    if baro_path.is_file():
        baro = pd.read_csv(baro_path).sort_values(key)
        t_imu = base["t"].to_numpy(dtype=float)
        rel = np.interp(
            t_imu,
            baro[key].to_numpy(dtype=float),
            baro["relativeAltitude"].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
        pr = np.interp(
            t_imu,
            baro[key].to_numpy(dtype=float),
            baro["pressure"].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
        base["baro_relative_altitude_m"] = rel
        base["baro_pressure_hpa"] = pr
    else:
        base["baro_relative_altitude_m"] = np.nan
        base["baro_pressure_hpa"] = np.nan

    base["session_group"] = session.group_key
    base["label"] = int(session.label)
    base["multiclass"] = int(session.multiclass)
    return base.reset_index(drop=True)
