from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    grid_size: float = 1000.0
    episode_len: int = 240
    num_uavs: int = 8
    num_rsus: int = 6
    num_vehicles: int = 300
    num_events: int = 3
    num_false_events: int = 2
    uav_speed: float = 30.0
    uav_sensing_radius: float = 160.0
    uav_service_radius: float = 180.0
    uav_comm_radius: float = 300.0
    uav_service_capacity: float = 1.0
    initial_energy: float = 1.0
    move_energy_cost: float = 0.003
    verify_energy_cost: float = 0.005
    serve_energy_cost: float = 0.007
    vehicle_speed: float = 8.0
    rsu_base_capacity: float = 1.0
    packet_loss: float = 0.0
    detection_noise: float = 0.05
    malicious_uav_ratio: float = 0.0
    detector_true_positive_rate: float = 0.92
    detector_false_positive_rate: float = 0.08
    malicious_report_flip_prob: float = 0.9
    initial_trust: float = 0.75
    trust_update_rate: float = 0.08
    min_operational_energy: float = 0.02
    base_vehicle_throughput_mbps: float = 12.0
    uav_throughput_gain_mbps: float = 8.0
    qos_outage_threshold_mbps: float = 6.0
    base_service_latency_ms: float = 25.0
    max_service_latency_ms: float = 250.0
    pdr_sensitivity: float = 0.08
    false_event_severity: float = 0.25
    event_rsu_load_increase: float = 0.55
    recovery_service_requirement: float = 2.0
    recovery_load_relief_rate: float = 0.04
    trace_hotspot_enabled: bool = False
    trace_hotspot_strength: float = 0.35
    mobility_mode: str = "manhattan"
    mobility_trace_path: str = ""
    mobility_trace_loop: bool = True


@dataclass
class ValueConfig:
    mode: str = "submodular"
    normalize_components: bool = True
    info_weight: float = 1.0
    relief_weight: float = 0.8
    qos_weight: float = 0.8
    enable_energy_penalty: bool = True
    enable_comm_penalty: bool = True
    energy_penalty: float = 0.2
    comm_penalty: float = 0.05
    duplicate_coverage_threshold: float = 2.0
    navigation_shaping_weight: float = 0.5
    true_confirmation_bonus: float = 3.0
    true_rejection_bonus: float = 1.0
    false_confirmation_penalty: float = 3.0
    false_rejection_penalty: float = 2.0
    missed_event_penalty: float = 3.0
    recovery_bonus: float = 2.0
    missed_recovery_penalty: float = 1.0


@dataclass
class ConsensusConfig:
    enabled: bool = True
    protocol: str = "spbft"
    value_trigger_enabled: bool = True
    greedy_coalition_enabled: bool = True
    coalition_selection: str = "greedy"
    adaptive_coalition_enabled: bool = False
    periodic_interval: int = 0
    detection_trigger_threshold: float = 0.05
    trigger_threshold: float = 0.50
    min_candidates: int = 2
    max_coalition_size: int = 4
    adaptive_max_coalition_size: int = 8
    trust_threshold: float = 0.2
    risk_expansion_threshold: float = 0.35
    base_latency_ms: float = 40.0
    per_node_latency_ms: float = 12.0
    message_size_kb: float = 8.0
    round_trip_time_ms: float = 12.0
    max_phase_retries: int = 1


@dataclass
class EvaluationConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_from_dict(data: dict[str, Any]) -> EvaluationConfig:
    merged = _merge_dict(EvaluationConfig().to_dict(), data)
    unknown = set(merged) - {"env", "value", "consensus"}
    if unknown:
        raise ValueError(f"Unsupported evaluation config sections: {sorted(unknown)}")
    return EvaluationConfig(
        env=EnvConfig(**merged["env"]),
        value=ValueConfig(**merged["value"]),
        consensus=ConsensusConfig(**merged["consensus"]),
    )


def load_config(path: str | Path | None = None) -> EvaluationConfig:
    if path is None:
        return EvaluationConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "train" in data or "training" in data:
        raise ValueError("This model-only release does not accept training settings")
    return config_from_dict(data)
