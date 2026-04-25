import argparse
import json
from pathlib import Path

from .align import align_to_millisecond
from .config import V2Config
from .evaluate import ablation_table
from .features import build_feature_bank
from .filtering import denoise_barometer
from .io_utils import load_sensor_csv
from .visualize import plot_five_dim_dashboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with labeled CSV clips")
    parser.add_argument(
        "--one-clip",
        type=Path,
        default=None,
        help="Optional clip CSV for dashboard rendering",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    cfg = V2Config()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = ablation_table(args.data_dir, cfg)
    table_path = args.out_dir / "ablation.csv"
    table.to_csv(table_path, index=False)

    if args.one_clip:
        df = load_sensor_csv(args.one_clip)
        df = align_to_millisecond(df)
        df = denoise_barometer(
            df,
            sample_rate_hz=cfg.sensor.sample_rate_hz,
            cutoff_hz=cfg.sensor.lowpass_cutoff_hz,
            order=cfg.sensor.lowpass_order,
        )
        feat = build_feature_bank(df)
        plot_five_dim_dashboard(feat, args.out_dir / "five_dim_dashboard.png")

    with open(args.out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({"data_dir": str(args.data_dir), "ablation_rows": len(table)}, f, indent=2)

    print(f"Saved ablation to {table_path}")


if __name__ == "__main__":
    main()

