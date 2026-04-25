import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(
    signal: np.ndarray, sample_rate_hz: float, cutoff_hz: float, order: int
) -> np.ndarray:
    nyquist = 0.5 * sample_rate_hz
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def denoise_barometer(
    df: pd.DataFrame, sample_rate_hz: float, cutoff_hz: float, order: int
) -> pd.DataFrame:
    out = df.copy()
    out["baro_m_raw"] = out["baro_m"]
    out["baro_m"] = butter_lowpass_filter(
        out["baro_m"].to_numpy(),
        sample_rate_hz=sample_rate_hz,
        cutoff_hz=cutoff_hz,
        order=order,
    )
    return out

