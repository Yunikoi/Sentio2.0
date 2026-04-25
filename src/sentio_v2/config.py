from dataclasses import dataclass, field


@dataclass
class SensorConfig:
    sample_rate_hz: float = 50.0
    lowpass_cutoff_hz: float = 2.5
    lowpass_order: int = 3


@dataclass
class TrackAConfig:
    impact_acc_g: float = 2.8
    impact_gyro_dps: float = 280.0
    impact_window_ms: int = 800


@dataclass
class TrackBConfig:
    height_drop_m: float = 0.35
    posture_tilt_deg: float = 55.0
    timeout_ms: int = 2200


@dataclass
class DecisionConfig:
    cooldown_ms: int = 5000
    confirm_window_ms: int = 30000


@dataclass
class V2Config:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    track_a: TrackAConfig = field(default_factory=TrackAConfig)
    track_b: TrackBConfig = field(default_factory=TrackBConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

