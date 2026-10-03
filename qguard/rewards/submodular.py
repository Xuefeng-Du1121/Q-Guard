from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sivc_marl.config import ValueConfig


@dataclass
class Contribution:
    uav_id: int
    event_probs: np.ndarray
    rsu_capacity: np.ndarray
    vehicle_quality: np.ndarray


@dataclass
class ValueParts:
    info: float
    relief: float
    qos: float

    @property
    def total(self) -> float:
        return self.info + self.relief + self.qos


class SubmodularValue:
    """Information-service value model.

    In the paper, SIVC uses the submodular mode. The additive mode is kept
    for legacy and ablation baselines where duplicate coverage is not
    explicitly discounted.
    """

    def __init__(self, cfg: ValueConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        contributions: list[Contribution],
        event_weights: np.ndarray,
        rsu_weights: np.ndarray,
        vehicle_weights: np.ndarray,
    ) -> ValueParts:
        if not contributions:
            return ValueParts(info=0.0, relief=0.0, qos=0.0)

        event_probs = np.vstack([c.event_probs for c in contributions])
        rsu_capacity = np.vstack([c.rsu_capacity for c in contributions])
        vehicle_quality = np.vstack([c.vehicle_quality for c in contributions])

        if self.cfg.mode == "submodular":
            info_raw = np.sum(event_weights * (1.0 - np.prod(1.0 - event_probs, axis=0)))
            relief_raw = np.sum(rsu_weights * np.log1p(np.sum(rsu_capacity, axis=0)))
            qos_raw = np.sum(vehicle_weights * (1.0 - np.exp(-np.sum(vehicle_quality, axis=0))))
        elif self.cfg.mode == "additive":
            info_raw = np.sum(event_weights * np.sum(event_probs, axis=0))
            relief_raw = np.sum(rsu_weights * np.sum(rsu_capacity, axis=0))
            qos_raw = np.sum(vehicle_weights * np.sum(vehicle_quality, axis=0))
        else:
            raise ValueError(f"Unknown value mode: {self.cfg.mode}")

        if self.cfg.normalize_components:
            info_raw /= max(float(np.sum(np.abs(event_weights))), 1e-8)
            relief_raw /= max(float(np.sum(np.abs(rsu_weights))), 1e-8)
            qos_raw /= max(float(np.sum(np.abs(vehicle_weights))), 1e-8)

        return ValueParts(
            info=float(self.cfg.info_weight * info_raw),
            relief=float(self.cfg.relief_weight * relief_raw),
            qos=float(self.cfg.qos_weight * qos_raw),
        )

    def marginal_gains(
        self,
        contributions: list[Contribution],
        event_weights: np.ndarray,
        rsu_weights: np.ndarray,
        vehicle_weights: np.ndarray,
    ) -> np.ndarray:
        if not contributions:
            return np.zeros(0, dtype=float)
        total = self.evaluate(
            contributions,
            event_weights=event_weights,
            rsu_weights=rsu_weights,
            vehicle_weights=vehicle_weights,
        ).total
        gains = np.zeros(len(contributions), dtype=float)
        for idx in range(len(contributions)):
            without = contributions[:idx] + contributions[idx + 1 :]
            reduced = self.evaluate(
                without,
                event_weights=event_weights,
                rsu_weights=rsu_weights,
                vehicle_weights=vehicle_weights,
            ).total
            gains[idx] = max(0.0, total - reduced)
        return gains

    def greedy_select(
        self,
        contributions: list[Contribution],
        budget: int,
        event_weights: np.ndarray,
        rsu_weights: np.ndarray,
        vehicle_weights: np.ndarray,
    ) -> list[int]:
        """Select a cardinality-constrained set with recomputed marginal gains."""

        selected: list[int] = []
        remaining = set(range(len(contributions)))
        current_value = 0.0
        while remaining and len(selected) < max(0, budget):
            best_idx = None
            best_gain = -np.inf
            for idx in sorted(remaining):
                candidate_ids = selected + [idx]
                value = self.evaluate(
                    [contributions[item] for item in candidate_ids],
                    event_weights=event_weights,
                    rsu_weights=rsu_weights,
                    vehicle_weights=vehicle_weights,
                ).total
                gain = value - current_value
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx is None or best_gain <= 0.0:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            current_value += best_gain
        return selected
