from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .config import V2Config


@dataclass
class Event:
    timestamp_ms: int
    event_type: str
    reason: str


class DualTrackDetector:
    def __init__(self, config: V2Config):
        self.cfg = config
        self._last_trigger_ms = -10**12

    def run(self, feature_df: pd.DataFrame) -> List[Event]:
        events: List[Event] = []
        state_b = {"active": False, "start_ms": None}

        for row in feature_df.itertuples(index=False):
            ts = int(row.timestamp_ms)
            in_cooldown = (ts - self._last_trigger_ms) < self.cfg.decision.cooldown_ms
            if in_cooldown:
                continue

            track_a_hit = (
                row.acc_norm_g >= self.cfg.track_a.impact_acc_g
                and row.gyro_norm_dps >= self.cfg.track_a.impact_gyro_dps
            )

            posture_bad = row.tilt_deg >= self.cfg.track_b.posture_tilt_deg
            height_bad = row.baro_drop_from_start_m >= self.cfg.track_b.height_drop_m

            if posture_bad and height_bad and not state_b["active"]:
                state_b["active"] = True
                state_b["start_ms"] = ts

            if state_b["active"] and (not posture_bad or not height_bad):
                state_b["active"] = False
                state_b["start_ms"] = None

            track_b_hit = False
            if state_b["active"] and state_b["start_ms"] is not None:
                track_b_hit = (ts - state_b["start_ms"]) >= self.cfg.track_b.timeout_ms

            if track_a_hit:
                events.append(Event(ts, "fall_alert", "track_a_impact"))
                self._last_trigger_ms = ts
                state_b = {"active": False, "start_ms": None}
                continue

            if track_b_hit:
                events.append(Event(ts, "fall_alert", "track_b_height_posture_timeout"))
                self._last_trigger_ms = ts
                state_b = {"active": False, "start_ms": None}

        return events


def classify_clip(events: List[Event]) -> Dict[str, int]:
    return {"pred_fall": int(len(events) > 0), "n_events": len(events)}

