import numpy as np
import pandas as pd


def build_feature_bank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["acc_norm_g"] = np.sqrt(out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2)
    out["gyro_norm_dps"] = np.sqrt(
        out["gyro_x"] ** 2 + out["gyro_y"] ** 2 + out["gyro_z"] ** 2
    )
    out["tilt_deg"] = np.degrees(np.arccos(np.clip(out["acc_z"] / out["acc_norm_g"], -1.0, 1.0)))
    out["baro_drop_from_start_m"] = out["baro_m"].iloc[0] - out["baro_m"]
    out["jerk_gps"] = out["acc_norm_g"].diff().fillna(0.0) * 50.0
    return out

