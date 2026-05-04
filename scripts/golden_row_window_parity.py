#!/usr/bin/env python3
"""Golden test: same fused IMU window → Python ``row_windows`` vs TypeScript ``rowWindows.ts``.

Writes ``outputs/golden/`` and runs ``mobile/node_modules/.bin/tsx`` to compute TS features,
then compares float keys with a tight tolerance.

Example::

    python scripts/golden_row_window_parity.py \\
        --session-path data_sensor/row/IndoorAdultManWalking/01_0427_ByLiu \\
        --start 0
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sentio_v2.row_loader import FALL_ACTIVITY, RowSession, multiclass_for_activity
from src.sentio_v2.row_session import load_fused_session
from src.sentio_v2.row_windows import ROW_BASE_NUMERIC_COLUMN_NAMES, windows_from_session


def _tsx_bin(mobile: Path) -> Path:
    if sys.platform == "win32":
        p = mobile / "node_modules" / ".bin" / "tsx.cmd"
        if p.is_file():
            return p
    p = mobile / "node_modules" / ".bin" / "tsx"
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"tsx not found under {mobile / 'node_modules' / '.bin'}; run: cd mobile && npm install"
    )


def _session_from_path(session_path: Path) -> RowSession:
    p = session_path.resolve()
    activity = p.parent.name
    return RowSession(
        activity=activity,
        session_id=p.name,
        path=p,
        label=1 if activity == FALL_ACTIVITY else 0,
        multiclass=multiclass_for_activity(activity),
    )


def _read_train_window_hop() -> Tuple[int, int]:
    meta = ROOT / "outputs" / "ml_row" / "train_meta.json"
    if meta.is_file():
        data = json.loads(meta.read_text(encoding="utf-8"))
        ws = data.get("window_samples")
        hs = data.get("hop_samples")
        if isinstance(ws, int) and isinstance(hs, int):
            return ws, hs
    return 101, 25


def _row_to_jsonable(row: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in ROW_BASE_NUMERIC_COLUMN_NAMES:
        v = row[c]
        if pd.isna(v):
            out[c] = None
        else:
            out[c] = float(v)
    return out


def _compare_dicts(py: Dict[str, float], ts: Dict[str, Any], rtol: float, atol: float) -> List[str]:
    errs: List[str] = []
    keys = sorted(set(py) | set(ts))
    for k in keys:
        if k not in py:
            errs.append(f"missing in Python: {k}")
            continue
        if k not in ts:
            errs.append(f"missing in TS: {k}")
            continue
        a, b = float(py[k]), float(ts[k])
        if not math.isfinite(a) or not math.isfinite(b):
            if (math.isnan(a) and math.isnan(b)) or a == b:
                continue
            errs.append(f"{k}: py={a} ts={b}")
            continue
        tol = atol + rtol * max(abs(a), abs(b))
        if abs(a - b) > tol:
            errs.append(f"{k}: py={a!r} ts={b!r} diff={abs(a - b):.3e} tol={tol:.3e}")
    return errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-path",
        type=Path,
        default=Path("data_sensor/row/IndoorAdultManWalking/01_0427_ByLiu"),
        help="Folder containing Accelerometer.csv, Gravity.csv, …",
    )
    parser.add_argument("--start", type=int, default=0, help="Start row index in fused frame")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()

    window_samples, _hop = _read_train_window_hop()
    out_dir = ROOT / "outputs" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = _session_from_path(args.session_path)
    df = load_fused_session(session)
    if df is None or df.empty:
        raise SystemExit(f"No fused data for {args.session_path}")

    n = len(df)
    if args.start + window_samples > n:
        raise SystemExit(f"start={args.start} + window={window_samples} exceeds n={n}")

    sub = df.iloc[args.start : args.start + window_samples].reset_index(drop=True)
    rows_json = [_row_to_jsonable(sub.iloc[i]) for i in range(len(sub))]
    fused_path = out_dir / "fused_window_rows.json"
    fused_path.write_text(json.dumps(rows_json, indent=2), encoding="utf-8")

    win_df = windows_from_session(sub, window_samples=len(sub), hop_samples=1)
    if win_df.empty:
        raise SystemExit("Python produced no window (unexpected)")
    py_row = win_df.iloc[0].to_dict()
    for drop in ("session_group", "label", "window_t_mid"):
        py_row.pop(drop, None)
    py_feats = {k: float(v) for k, v in py_row.items()}
    py_path = out_dir / "features_python.json"
    py_path.write_text(json.dumps(py_feats, indent=2, sort_keys=True), encoding="utf-8")

    mobile = ROOT / "mobile"
    ts_path = out_dir / "features_ts.json"
    tsx = _tsx_bin(mobile)
    cmd = [
        str(tsx),
        "scripts/goldenCompute.ts",
        str(fused_path.resolve()),
        str(ts_path.resolve()),
    ]
    print("Running:", " ".join(cmd), f"(cwd={mobile})")
    subprocess.run(cmd, cwd=mobile, check=True)

    ts_feats = json.loads(ts_path.read_text(encoding="utf-8"))
    errs = _compare_dicts(py_feats, ts_feats, args.rtol, args.atol)
    if errs:
        print("MISMATCH:\n" + "\n".join(errs[:50]))
        if len(errs) > 50:
            print(f"... and {len(errs) - 50} more")
        raise SystemExit(1)

    print(f"OK: {len(py_feats)} keys match within rtol={args.rtol} atol={args.atol}")
    print(f"Wrote {fused_path}")
    print(f"Wrote {py_path}")
    print(f"Wrote {ts_path}")


if __name__ == "__main__":
    main()
