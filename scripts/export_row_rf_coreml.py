#!/usr/bin/env python3
"""CLI wrapper for :func:`src.sentio_v2.coreml_export.export_rf_joblib_to_coreml`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sentio_v2.coreml_export import export_rf_joblib_to_coreml


def _as_int(v: Any) -> Optional[int]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    return None


def _read_train_meta(joblib_path: Path) -> Tuple[Optional[int], Optional[int]]:
    meta_path = joblib_path.parent / "train_meta.json"
    if not meta_path.is_file():
        return None, None
    data: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return _as_int(data.get("window_samples")), _as_int(data.get("hop_samples"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joblib-path",
        type=Path,
        default=Path("outputs/ml_row/fall_rf_row.joblib"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mobile/assets/coreml"),
    )
    parser.add_argument("--target-name", type=str, default="fall")
    args = parser.parse_args()

    ws, hs = _read_train_meta(args.joblib_path)
    out_model, manifest_path = export_rf_joblib_to_coreml(
        args.joblib_path,
        args.out_dir,
        target_name=args.target_name,
        window_samples=ws,
        hop_samples=hs,
    )
    print(f"Wrote {out_model}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
