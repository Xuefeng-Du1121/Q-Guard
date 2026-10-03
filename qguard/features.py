from __future__ import annotations

import numpy as np

from sivc_marl.config import EnvConfig


def agent_feature_dim() -> int:
    return 14


def global_feature_dim() -> int:
    return 3 * agent_feature_dim() + 8


def extract_agent_features(observation: dict, env_cfg: EnvConfig) -> np.ndarray:
    """Convert environment observation into per-UAV fixed-size features."""

    uav_pos = observation["uav_positions"]
    uav_energy = observation["uav_energy"]
    uav_trust = observation.get("uav_trust", np.ones_like(uav_energy))
    event_pos = observation.get("event_positions", np.zeros((0, 2), dtype=float))
    event_uncertainty = observation.get("event_uncertainty", np.zeros(0, dtype=float))
    event_confirmed = observation.get("event_confirmed", np.zeros(0, dtype=bool))
    event_resolved = observation.get("event_resolved", event_confirmed)
    rsu_pos = observation.get("rsu_positions", np.zeros((0, 2), dtype=float))
    rsu_loads = observation.get("rsu_loads", np.zeros(0, dtype=float))

    features = np.zeros((uav_pos.shape[0], agent_feature_dim()), dtype=np.float32)
    scale = max(env_cfg.grid_size, 1.0)
    max_dist = np.sqrt(2.0) * scale

    for idx, position in enumerate(uav_pos):
        row = features[idx]
        row[0:2] = position / scale
        row[2] = uav_energy[idx]
        row[3] = uav_trust[idx]

        if event_pos.size:
            active = ~event_resolved
            if np.any(active):
                event_scores = np.where(active, event_uncertainty, -np.inf)
                event_idx = int(np.argmax(event_scores))
                delta = event_pos[event_idx] - position
                dist = np.linalg.norm(delta)
                row[4:6] = delta / scale
                row[6] = dist / max_dist
                row[7] = event_uncertainty[event_idx]
                row[8] = 1.0

        if rsu_pos.size:
            overload = np.maximum(0.0, rsu_loads - env_cfg.rsu_base_capacity)
            if np.max(overload) > 0.0:
                rsu_idx = int(np.argmax(overload))
            else:
                distances = np.linalg.norm(rsu_pos - position, axis=1)
                rsu_idx = int(np.argmin(distances))
            delta = rsu_pos[rsu_idx] - position
            dist = np.linalg.norm(delta)
            row[9:11] = delta / scale
            row[11] = dist / max_dist
            row[12] = rsu_loads[rsu_idx]

        row[13] = float(observation.get("step", 0)) / max(env_cfg.episode_len, 1)

    return features


def extract_global_features(observation: dict, env_cfg: EnvConfig) -> np.ndarray:
    """Permutation-invariant centralized state used only during training."""

    local = extract_agent_features(observation, env_cfg)
    if local.size:
        pooled = np.concatenate(
            [np.mean(local, axis=0), np.max(local, axis=0), np.min(local, axis=0)]
        )
    else:
        pooled = np.zeros(3 * agent_feature_dim(), dtype=np.float32)

    rsu_loads = observation.get("rsu_loads", np.zeros(0, dtype=float))
    if rsu_loads.size:
        rsu_stats = np.array(
            [
                np.mean(rsu_loads),
                np.max(rsu_loads),
                np.mean(rsu_loads > env_cfg.rsu_base_capacity),
            ],
            dtype=np.float32,
        )
    else:
        rsu_stats = np.zeros(3, dtype=np.float32)

    uncertainty = observation.get("event_uncertainty", np.zeros(0, dtype=float))
    confirmed = observation.get("event_confirmed", np.zeros(0, dtype=bool))
    resolved = observation.get("event_resolved", confirmed)
    if uncertainty.size:
        event_stats = np.array(
            [
                np.mean(uncertainty),
                np.max(uncertainty),
                np.mean(confirmed),
                np.mean(~resolved),
            ],
            dtype=np.float32,
        )
    else:
        event_stats = np.zeros(4, dtype=np.float32)
    step = np.array(
        [float(observation.get("step", 0)) / max(env_cfg.episode_len, 1)],
        dtype=np.float32,
    )
    return np.concatenate([pooled, rsu_stats, event_stats, step]).astype(np.float32)


def one_hot_action(actions: np.ndarray, action_dim: int) -> np.ndarray:
    encoded = np.zeros((actions.shape[0], action_dim), dtype=np.float32)
    encoded[np.arange(actions.shape[0]), actions.astype(int)] = 1.0
    return encoded
