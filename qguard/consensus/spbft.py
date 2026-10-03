from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from sivc_marl.config import ConsensusConfig, EnvConfig
from sivc_marl.envs.entities import EventState, UAVState
from sivc_marl.rewards.submodular import Contribution, SubmodularValue


@dataclass
class ConsensusResult:
    attempted: bool
    success: bool
    coalition: list[int]
    messages: int
    bytes_kb: float
    latency_ms: float
    packet_success: bool
    malicious_count: int
    decision: bool | None = None
    positive_votes: int = 0
    negative_votes: int = 0
    quorum: int = 0
    delivered_messages: int = 0
    byzantine_safe: bool = False
    reports: dict[int, bool] = field(default_factory=dict)
    trigger_suppressed: bool = False
    trigger_score: float = 0.0


class SPBFTSimulator:
    """Phase-level packet simulator for event-triggered BFT confirmation.

    The simulator models committee selection, noisy local reports, Byzantine
    report flipping, per-phase packet loss, quorum decisions, and trust updates.
    It is not presented as a production implementation of a blockchain stack.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        consensus_cfg: ConsensusConfig,
        rng: np.random.Generator,
    ) -> None:
        self.env_cfg = env_cfg
        self.cfg = consensus_cfg
        self.rng = rng

    def run(
        self,
        event: EventState,
        uavs: list[UAVState],
        marginal_gains: np.ndarray,
        contributions: list[Contribution] | None = None,
        value_model: SubmodularValue | None = None,
        event_weights: np.ndarray | None = None,
        rsu_weights: np.ndarray | None = None,
        vehicle_weights: np.ndarray | None = None,
        event_evidence: float = 1.0,
    ) -> ConsensusResult:
        candidates = self._candidate_ids(event, uavs)
        if len(candidates) < self.cfg.min_candidates:
            return self._not_attempted()

        coalition = self._select_coalition(
            candidates,
            marginal_gains,
            uavs,
            contributions,
            value_model,
            event_weights,
            rsu_weights,
            vehicle_weights,
        )
        n = len(coalition)
        if n < self.cfg.min_candidates:
            return self._not_attempted()

        value = self._coalition_value(
            coalition,
            uavs,
            marginal_gains,
            contributions,
            value_model,
            event_weights,
            rsu_weights,
            vehicle_weights,
        )
        comm_cost = self._communication_cost(n)
        trigger_score = float(np.clip(event_evidence, 0.0, 1.0) * value)
        if (
            self.cfg.value_trigger_enabled
            and trigger_score <= self.cfg.trigger_threshold + comm_cost
        ):
            return self._not_attempted(
                trigger_suppressed=True,
                trigger_score=trigger_score,
            )

        tolerated = max(0, (n - 1) // 3)
        quorum = min(n, max(self.cfg.min_candidates, 2 * tolerated + 1))
        malicious = sum(1 for idx in coalition if uavs[idx].malicious)
        reports = {idx: self._sample_report(event, uavs[idx]) for idx in coalition}
        delivered_reports = {
            idx: report
            for idx, report in reports.items()
            if self.rng.random() >= self.env_cfg.packet_loss
        }

        messages, delivered_messages, phase_attempts, protocol_complete = self._run_protocol_phases(
            n, quorum
        )
        positive_votes = sum(delivered_reports.values())
        negative_votes = len(delivered_reports) - positive_votes
        decision: bool | None = None
        if positive_votes >= quorum:
            decision = True
        elif negative_votes >= quorum:
            decision = False

        success = protocol_complete and decision is not None
        if success:
            self._update_trust(uavs, delivered_reports, bool(decision))

        bytes_kb = float(messages * self.cfg.message_size_kb)
        latency = phase_attempts * self.cfg.round_trip_time_ms
        latency += self.cfg.base_latency_ms + self.cfg.per_node_latency_ms * n
        latency += 0.015 * bytes_kb

        return ConsensusResult(
            attempted=True,
            success=success,
            coalition=coalition,
            messages=messages,
            bytes_kb=bytes_kb,
            latency_ms=float(latency),
            packet_success=protocol_complete,
            malicious_count=malicious,
            decision=decision,
            positive_votes=positive_votes,
            negative_votes=negative_votes,
            quorum=quorum,
            delivered_messages=delivered_messages,
            byzantine_safe=malicious <= tolerated,
            reports=delivered_reports,
            trigger_score=trigger_score,
        )

    def _not_attempted(
        self,
        trigger_suppressed: bool = False,
        trigger_score: float = 0.0,
    ) -> ConsensusResult:
        return ConsensusResult(
            attempted=False,
            success=False,
            coalition=[],
            messages=0,
            bytes_kb=0.0,
            latency_ms=0.0,
            packet_success=False,
            malicious_count=0,
            trigger_suppressed=trigger_suppressed,
            trigger_score=trigger_score,
        )

    def _candidate_ids(self, event: EventState, uavs: list[UAVState]) -> list[int]:
        operational: list[int] = []
        trusted: list[int] = []
        for uav in uavs:
            dist = np.linalg.norm(uav.position - event.position)
            if dist > self.env_cfg.uav_comm_radius:
                continue
            if uav.energy <= self.env_cfg.min_operational_energy:
                continue
            operational.append(uav.uid)
            if uav.trust_score >= self.cfg.trust_threshold:
                trusted.append(uav.uid)
        if len(trusted) >= self.cfg.min_candidates:
            return trusted
        return sorted(operational, key=lambda idx: -uavs[idx].trust_score)

    def _select_coalition(
        self,
        candidates: list[int],
        marginal_gains: np.ndarray,
        uavs: list[UAVState],
        contributions: list[Contribution] | None,
        value_model: SubmodularValue | None,
        event_weights: np.ndarray | None,
        rsu_weights: np.ndarray | None,
        vehicle_weights: np.ndarray | None,
    ) -> list[int]:
        limit = self.cfg.max_coalition_size
        if self.cfg.adaptive_coalition_enabled:
            mean_trust = float(np.mean([uavs[idx].trust_score for idx in candidates]))
            risk = self.env_cfg.packet_loss + (1.0 - mean_trust)
            if risk >= self.cfg.risk_expansion_threshold:
                limit = self.cfg.adaptive_max_coalition_size
        limit = min(max(self.cfg.min_candidates, limit), len(candidates))

        if not self.cfg.greedy_coalition_enabled:
            return sorted(candidates)[:limit]
        if self._has_value_context(
            contributions,
            value_model,
            event_weights,
            rsu_weights,
            vehicle_weights,
        ):
            weighted = [
                self._weighted_contribution(contributions[idx], uavs[idx]) for idx in candidates
            ]
            if self.cfg.coalition_selection.lower() == "exact":
                selected_local = self._exact_select(
                    weighted,
                    limit,
                    value_model,
                    event_weights,
                    rsu_weights,
                    vehicle_weights,
                )
            else:
                selected_local = value_model.greedy_select(
                    weighted,
                    limit,
                    event_weights=event_weights,
                    rsu_weights=rsu_weights,
                    vehicle_weights=vehicle_weights,
                )
            selected = [candidates[idx] for idx in selected_local]
            if len(selected) < self.cfg.min_candidates:
                selected_set = set(selected)
                fallback = self._rank_candidates(candidates, marginal_gains, uavs)
                selected.extend(
                    idx
                    for idx in fallback
                    if idx not in selected_set and len(selected) < self.cfg.min_candidates
                )
            return selected
        return self._rank_candidates(candidates, marginal_gains, uavs)[:limit]

    def _rank_candidates(
        self,
        candidates: list[int],
        marginal_gains: np.ndarray,
        uavs: list[UAVState],
    ) -> list[int]:
        ranked = sorted(
            candidates,
            key=lambda idx: (
                -(
                    max(0.0, float(marginal_gains[idx]))
                    * (0.5 + uavs[idx].trust_score)
                    * max(uavs[idx].energy, 0.0)
                )
            ),
        )
        return ranked

    def _weighted_contribution(self, contribution: Contribution, uav: UAVState) -> Contribution:
        energy_ratio = uav.energy / max(self.env_cfg.initial_energy, 1e-8)
        reliability = float(np.clip((0.5 + 0.5 * uav.trust_score) * energy_ratio, 0.0, 1.0))
        return Contribution(
            uav_id=contribution.uav_id,
            event_probs=np.clip(contribution.event_probs * reliability, 0.0, 1.0),
            rsu_capacity=contribution.rsu_capacity * reliability,
            vehicle_quality=contribution.vehicle_quality * reliability,
        )

    def _coalition_value(
        self,
        coalition: list[int],
        uavs: list[UAVState],
        marginal_gains: np.ndarray,
        contributions: list[Contribution] | None,
        value_model: SubmodularValue | None,
        event_weights: np.ndarray | None,
        rsu_weights: np.ndarray | None,
        vehicle_weights: np.ndarray | None,
    ) -> float:
        if not self._has_value_context(
            contributions,
            value_model,
            event_weights,
            rsu_weights,
            vehicle_weights,
        ):
            return float(np.sum(marginal_gains[coalition]))
        selected = [self._weighted_contribution(contributions[idx], uavs[idx]) for idx in coalition]
        return float(
            value_model.evaluate(
                selected,
                event_weights=event_weights,
                rsu_weights=rsu_weights,
                vehicle_weights=vehicle_weights,
            ).total
        )

    def _exact_select(
        self,
        contributions: list[Contribution],
        budget: int,
        value_model: SubmodularValue,
        event_weights: np.ndarray,
        rsu_weights: np.ndarray,
        vehicle_weights: np.ndarray,
    ) -> list[int]:
        best: list[int] = []
        best_value = -np.inf
        for size in range(1, min(budget, len(contributions)) + 1):
            for candidate in combinations(range(len(contributions)), size):
                value = value_model.evaluate(
                    [contributions[idx] for idx in candidate],
                    event_weights=event_weights,
                    rsu_weights=rsu_weights,
                    vehicle_weights=vehicle_weights,
                ).total
                if value > best_value:
                    best = list(candidate)
                    best_value = value
        return best

    @staticmethod
    def _has_value_context(
        contributions: list[Contribution] | None,
        value_model: SubmodularValue | None,
        event_weights: np.ndarray | None,
        rsu_weights: np.ndarray | None,
        vehicle_weights: np.ndarray | None,
    ) -> bool:
        return all(
            item is not None
            for item in (
                contributions,
                value_model,
                event_weights,
                rsu_weights,
                vehicle_weights,
            )
        )

    def _sample_report(self, event: EventState, uav: UAVState) -> bool:
        distance = float(np.linalg.norm(uav.position - event.position))
        quality = float(np.exp(-distance / max(self.env_cfg.uav_sensing_radius, 1.0)))
        if event.is_real:
            probability = 0.5 + (self.env_cfg.detector_true_positive_rate - 0.5) * quality
        else:
            probability = 0.5 + (self.env_cfg.detector_false_positive_rate - 0.5) * quality
        benign_report = self.rng.random() < probability
        if uav.malicious and self.rng.random() < self.env_cfg.malicious_report_flip_prob:
            return not benign_report
        return bool(benign_report)

    def _update_trust(
        self,
        uavs: list[UAVState],
        reports: dict[int, bool],
        decision: bool,
    ) -> None:
        rate = self.env_cfg.trust_update_rate
        for idx, report in reports.items():
            trust = uavs[idx].trust_score
            if report == decision:
                trust += rate * (1.0 - trust)
            else:
                trust -= rate * trust
            uavs[idx].trust_score = float(np.clip(trust, 0.0, 1.0))

    def _run_protocol_phases(self, n: int, quorum: int) -> tuple[int, int, int, bool]:
        phase_sizes = self._phase_message_counts(n)
        messages = 0
        delivered = 0
        attempts = 0
        for phase_messages in phase_sizes:
            phase_complete = False
            for _ in range(max(0, self.cfg.max_phase_retries) + 1):
                attempts += 1
                messages += phase_messages
                delivered += int(self.rng.binomial(phase_messages, 1.0 - self.env_cfg.packet_loss))
                active_nodes = int(self.rng.binomial(n, 1.0 - self.env_cfg.packet_loss))
                if active_nodes >= quorum:
                    phase_complete = True
                    break
            if not phase_complete:
                return messages, delivered, attempts, False
        return messages, delivered, attempts, True

    def _phase_message_counts(self, n: int) -> list[int]:
        protocol = self.cfg.protocol.lower()
        if n <= 1:
            return [0]
        if protocol in {"hotstuff", "chained_bft", "linear_bft"}:
            # Proposal and vote for prepare, pre-commit, and commit.
            return [n - 1, n - 1, n - 1, n - 1, n - 1, n - 1]
        # PBFT and local SPBFT use the same phases; SPBFT differs by committee
        # selection and event triggering rather than altered PBFT semantics.
        return [n - 1, n * (n - 1), n * (n - 1), n]

    def _communication_cost(self, n: int) -> float:
        return 0.001 * sum(self._phase_message_counts(n))

    def _message_count(self, n: int) -> int:
        return sum(self._phase_message_counts(n))
