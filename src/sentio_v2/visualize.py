from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_five_dim_dashboard(feature_df: pd.DataFrame, out_png: Path) -> None:
    t = (feature_df["timestamp_ms"] - feature_df["timestamp_ms"].iloc[0]) / 1000.0
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, feature_df["acc_norm_g"], label="acc_norm_g")
    axes[0].set_ylabel("g")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, feature_df["gyro_norm_dps"], label="gyro_norm_dps", color="tab:orange")
    axes[1].set_ylabel("dps")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, feature_df["baro_m"], label="baro_m", color="tab:green")
    axes[2].set_ylabel("m")
    axes[2].legend(loc="upper right")

    axes[3].plot(
        t,
        feature_df["baro_drop_from_start_m"],
        label="baro_drop_from_start_m",
        color="tab:red",
    )
    axes[3].set_ylabel("drop(m)")
    axes[3].legend(loc="upper right")

    axes[4].plot(t, feature_df["tilt_deg"], label="tilt_deg", color="tab:purple")
    axes[4].set_ylabel("deg")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

