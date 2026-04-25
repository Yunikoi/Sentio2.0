from pathlib import Path
from typing import Dict, List
from copy import deepcopy

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .align import align_to_millisecond
from .config import V2Config
from .detector import DualTrackDetector, classify_clip
from .features import build_feature_bank
from .filtering import denoise_barometer
from .io_utils import load_labeled_folder


def infer_label_from_name(name: str) -> int:
    key = name.lower()
    return int("fall" in key or "drop" in key)


def run_offline_eval(data_dir: Path, cfg: V2Config, use_baro: bool = True, use_track_b: bool = True) -> Dict:
    clips = load_labeled_folder(data_dir)
    y_true: List[int] = []
    y_pred: List[int] = []
    records = []

    local_cfg = deepcopy(cfg)
    if not use_track_b:
        local_cfg.track_b.timeout_ms = 10**9
        local_cfg.track_b.height_drop_m = 10**9
        local_cfg.track_b.posture_tilt_deg = 10**9

    for name, raw_df in clips.items():
        df = align_to_millisecond(raw_df)
        if use_baro:
            df = denoise_barometer(
                df,
                sample_rate_hz=cfg.sensor.sample_rate_hz,
                cutoff_hz=cfg.sensor.lowpass_cutoff_hz,
                order=cfg.sensor.lowpass_order,
            )
        feat = build_feature_bank(df)
        if not use_baro:
            feat["baro_drop_from_start_m"] = 0.0
        detector = DualTrackDetector(local_cfg)
        events = detector.run(feat)
        pred = classify_clip(events)["pred_fall"]
        truth = infer_label_from_name(name)
        y_true.append(truth)
        y_pred.append(pred)
        records.append({"clip": name, "truth": truth, "pred": pred, "events": len(events)})

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {"confusion_matrix": cm, "report": report, "per_clip": records}


def ablation_table(data_dir: Path, cfg: V2Config) -> pd.DataFrame:
    full = run_offline_eval(data_dir, cfg, use_baro=True, use_track_b=True)
    no_baro = run_offline_eval(data_dir, cfg, use_baro=False, use_track_b=True)
    no_track_b = run_offline_eval(data_dir, cfg, use_baro=True, use_track_b=False)

    rows = []
    for label, result in [
        ("full_v2", full),
        ("no_barometer", no_baro),
        ("no_track_b", no_track_b),
    ]:
        f1 = result["report"]["1"]["f1-score"] if "1" in result["report"] else 0.0
        rows.append({"setting": label, "fall_f1": f1, "cm": result["confusion_matrix"]})
    return pd.DataFrame(rows)

