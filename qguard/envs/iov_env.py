from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from sivc_marl.config import ConsensusConfig, EnvConfig, ValueConfig
from sivc_marl.consensus.spbft import SPBFTSimulator
from sivc_marl.envs.entities import EventState, RSUState, UAVState, VehicleState, WorldState
from sivc_marl.envs.mobility import manhattan_grid_positions, random_walk_positions
from sivc_marl.rewards.submodular import Contribution, SubmodularValue
from sivc_marl.utils.metrics import StepMetrics


class Action(IntEnum):
    STAY = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4
    VERIFY = 5
    SERVE = 6


MOVE_DELTAS = {
    Action.STAY: np.array([0.0, 0.0]),
    Action.NORTH: np.array([0.0, 1.0]),
    Action.SOUTH: np.array([0.0, -1.0]),
    Action.EAST: np.array([1.0, 0.0]),
    Action.WEST: np.array([-1.0, 0.0]),
}


class IoVEnv:
    """Minimal IoV-UAV simulator for validating the experiment pipeline."""

    action_space_n = len(Action)

    def __init__(
        self,
        env_cfg: EnvConfig,
        value_cfg: ValueConfig,
        consensus_cfg: ConsensusConfig,
        seed: int = 7,
    ) -> None:
        self.env_cfg = env_cfg
        self.value_cfg = value_cfg
        self.consensus_cfg = consensus_cfg
        seed_streams = np.random.SeedSequence(seed).spawn(4)
        self.rng = np.random.default_rng(seed_streams[0])
        self.mobility_rng = np.random.default_rng(seed_streams[1])
        self.sensing_rng = np.random.default_rng(seed_streams[2])
        consensus_rng = np.random.default_rng(seed_streams[3])
        self.value_model = SubmodularValue(value_cfg)
        self.consensus = SPBFTSimulator(env_cfg, consensus_cfg, consensus_rng)
        self.world: WorldState | None = None
        self._traffic_hotspots = np.zeros((0, 2), dtype=float)
        self._rsu_positions = np.zeros((0, 2), dtype=float)
        self._rsu_loads = np.zeros(0, dtype=float)
        self._vehicle_positions = np.zeros((0, 2), dtype=float)
        self._vehicle_demands = np.zeros(0, dtype=float)
        self._mobility_trace_positions: np.ndarray | None = None
        self._mobility_trace_demands: np.ndarray | None = None
        self._mobility_trace_step = 0

    def reset(self) -> dict[str, Any]:
        cfg = self.env_cfg
        malicious_count = int(round(cfg.num_uavs * cfg.malicious_uav_ratio))
        malicious_ids = set(self.rng.choice(cfg.num_uavs, size=malicious_count, replace=False))

        uavs = [
            UAVState(
                uid=i,
                position=self.rng.uniform(0.0, cfg.grid_size, size=2),
                energy=cfg.initial_energy,
                malicious=i in malicious_ids,
                trust_score=cfg.initial_trust,
            )
            for i in range(cfg.num_uavs)
        ]

        self._traffic_hotspots = (
            self._sample_hotspots() if cfg.trace_hotspot_enabled else np.zeros((0, 2), dtype=float)
        )

        rsu_positions = manhattan_grid_positions(cfg.num_rsus, cfg.grid_size, self.rng)
        rsu_loads = self.rng.uniform(0.4, 1.4, size=cfg.num_rsus)
        rsu_loads *= self._hotspot_demand_multipliers(rsu_positions)
        rsus = [
            RSUState(
                rid=i,
                position=rsu_positions[i],
                load=float(rsu_loads[i]),
                capacity=cfg.rsu_base_capacity,
            )
            for i in range(cfg.num_rsus)
        ]

        mobility_mode = cfg.mobility_mode.lower()
        if mobility_mode in {"trace", "sumo"}:
            vehicle_positions = self._reset_mobility_trace()
        elif mobility_mode == "uniform":
            vehicle_positions = self.rng.uniform(0.0, cfg.grid_size, size=(cfg.num_vehicles, 2))
        else:
            vehicle_positions = manhattan_grid_positions(cfg.num_vehicles, cfg.grid_size, self.rng)
        if cfg.trace_hotspot_enabled:
            vehicle_positions = self._apply_hotspot_vehicle_mix(vehicle_positions)
        if self._mobility_trace_demands is not None:
            vehicle_demands = self._mobility_trace_demands[
                self._mobility_trace_step, : cfg.num_vehicles
            ].copy()
        else:
            vehicle_demands = self.rng.uniform(0.2, 1.0, size=cfg.num_vehicles)
        vehicle_demands *= self._hotspot_demand_multipliers(vehicle_positions)
        vehicles = [
            VehicleState(
                vid=i,
                position=vehicle_positions[i],
                demand=float(vehicle_demands[i]),
            )
            for i in range(cfg.num_vehicles)
        ]

        events: list[EventState] = []
        primary_rsus = self.rng.permutation(cfg.num_rsus)
        for i in range(cfg.num_events):
            primary = int(primary_rsus[i % len(primary_rsus)])
            severity = float(self.rng.uniform(0.5, 1.0))
            event_position = np.clip(
                rsu_positions[primary] + self.rng.normal(0.0, 45.0, size=2),
                0.0,
                cfg.grid_size,
            )
            rsu_loads[primary] = max(
                rsu_loads[primary],
                cfg.rsu_base_capacity + 0.15 + cfg.event_rsu_load_increase * severity,
            )
            events.append(
                EventState(
                    eid=i,
                    position=event_position,
                    severity=severity,
                    uncertainty=float(self.rng.uniform(0.45, 0.85)),
                    is_real=True,
                    affected_rsu_ids=[primary],
                )
            )
        for j in range(cfg.num_false_events):
            events.append(
                EventState(
                    eid=cfg.num_events + j,
                    position=self.rng.uniform(0.0, cfg.grid_size, size=2),
                    severity=cfg.false_event_severity,
                    uncertainty=float(self.rng.uniform(0.35, 0.65)),
                    is_real=False,
                )
            )

        for rsu, load in zip(rsus, rsu_loads):
            rsu.load = float(load)

        self._rsu_positions = np.asarray(rsu_positions, dtype=float)
        self._rsu_loads = np.asarray(rsu_loads, dtype=float)
        self._vehicle_positions = np.asarray(vehicle_positions, dtype=float)
        self._vehicle_demands = np.asarray(vehicle_demands, dtype=float)
        self.world = WorldState(step=0, uavs=uavs, rsus=rsus, vehicles=vehicles, events=events)
        return self.observe()

    def observe(self) -> dict[str, Any]:
        world = self._world()
        return {
            "step": world.step,
            "uav_positions": np.array([u.position for u in world.uavs], dtype=float),
            "uav_energy": np.array([u.energy for u in world.uavs], dtype=float),
            "uav_trust": np.array([u.trust_score for u in world.uavs], dtype=float),
            "rsu_positions": self._rsu_positions.copy(),
            "rsu_loads": self._rsu_loads.copy(),
            "event_positions": np.array([e.position for e in world.events], dtype=float),
            "event_uncertainty": np.array([e.uncertainty for e in world.events], dtype=float),
            "event_confirmed": np.array([e.confirmed for e in world.events], dtype=bool),
            "event_resolved": np.array([e.resolved for e in world.events], dtype=bool),
        }

    def step(self, actions: list[int]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        world = self._world()
        if len(actions) != len(world.uavs):
            raise ValueError(f"Expected {len(world.uavs)} actions, got {len(actions)}")

        actions_enum = [Action(int(a)) for a in actions]
        navigation_targets = self._coordination_targets()
        positions_before = np.array([uav.position for uav in world.uavs], dtype=float)
        actions_enum, energy_costs = self._apply_actions(actions_enum)
        positions_after = np.array([uav.position for uav in world.uavs], dtype=float)
        navigation_rewards = self._navigation_rewards(
            positions_before, positions_after, navigation_targets
        )
        energy_cost = float(np.sum(energy_costs))
        self._update_background_dynamics()

        contributions = self._build_contributions(actions_enum)
        event_weights = np.array([e.severity * e.uncertainty for e in world.events], dtype=float)
        rsu_weights = np.maximum(0.0, self._rsu_loads - self.env_cfg.rsu_base_capacity)
        vehicle_weights = self._vehicle_demands
        value_parts = self.value_model.evaluate(
            contributions=contributions,
            event_weights=event_weights,
            rsu_weights=rsu_weights,
            vehicle_weights=vehicle_weights,
        )
        marginal_gains = self.value_model.marginal_gains(
            contributions=contributions,
            event_weights=event_weights,
            rsu_weights=rsu_weights,
            vehicle_weights=vehicle_weights,
        )

        event_signal = self._event_signal(contributions)
        detection_metrics = self._update_detection_state(event_signal)
        consensus_contributions = self._build_potential_contributions()
        consensus_metrics = self._run_consensus(consensus_contributions, event_signal)
        recovery_metrics = self._update_recovery(contributions)
        consensus_utility = self._reliability_utility(consensus_metrics)
        recovery_utility = self.value_cfg.recovery_bonus * recovery_metrics.recovered_events
        reliability_utility = consensus_utility + recovery_utility
        service_metrics = self._service_metrics(contributions)
        comm_cost_share = (
            self.value_cfg.comm_penalty
            * (consensus_metrics.consensus_messages / 1000.0)
            / max(len(world.uavs), 1)
            if self.value_cfg.enable_comm_penalty
            else 0.0
        )
        per_agent_rewards = (
            marginal_gains
            - (
                self.value_cfg.energy_penalty * energy_costs
                if self.value_cfg.enable_energy_penalty
                else 0.0
            )
            - comm_cost_share
            + navigation_rewards
        )
        verify_mask = np.array([action == Action.VERIFY for action in actions_enum], dtype=bool)
        serve_mask = np.array([action == Action.SERVE for action in actions_enum], dtype=bool)
        if consensus_utility != 0.0:
            recipients = (
                verify_mask if np.any(verify_mask) else np.ones(len(world.uavs), dtype=bool)
            )
            per_agent_rewards[recipients] += consensus_utility / np.sum(recipients)
        if recovery_utility != 0.0:
            recipients = serve_mask if np.any(serve_mask) else np.ones(len(world.uavs), dtype=bool)
            per_agent_rewards[recipients] += recovery_utility / np.sum(recipients)
        reward = (
            value_parts.total
            - (
                self.value_cfg.energy_penalty * energy_cost
                if self.value_cfg.enable_energy_penalty
                else 0.0
            )
            - (
                self.value_cfg.comm_penalty * (consensus_metrics.consensus_messages / 1000.0)
                if self.value_cfg.enable_comm_penalty
                else 0.0
            )
            + float(np.mean(navigation_rewards))
            + reliability_utility
        )
        redundant_coverage = self._redundant_coverage(contributions)

        step_metrics = StepMetrics(
            reward=float(reward),
            value_total=float(value_parts.total),
            value_info=float(value_parts.info),
            value_relief=float(value_parts.relief),
            value_qos=float(value_parts.qos),
            energy_cost=float(energy_cost),
            consensus_messages=consensus_metrics.consensus_messages,
            consensus_bytes_kb=consensus_metrics.consensus_bytes_kb,
            consensus_latency_ms=consensus_metrics.consensus_latency_ms,
            consensus_successes=consensus_metrics.consensus_successes,
            consensus_attempts=consensus_metrics.consensus_attempts,
            consensus_packet_successes=consensus_metrics.consensus_packet_successes,
            consensus_delivered_messages=consensus_metrics.consensus_delivered_messages,
            consensus_safe_decisions=consensus_metrics.consensus_safe_decisions,
            consensus_decisions=consensus_metrics.consensus_decisions,
            consensus_trigger_suppressions=consensus_metrics.consensus_trigger_suppressions,
            consensus_trigger_score_sum=consensus_metrics.consensus_trigger_score_sum,
            consensus_trigger_score_count=consensus_metrics.consensus_trigger_score_count,
            redundant_coverage=float(redundant_coverage),
            confirmed_events=consensus_metrics.confirmed_events,
            true_confirmations=consensus_metrics.true_confirmations,
            false_confirmations=consensus_metrics.false_confirmations,
            true_rejections=consensus_metrics.true_rejections,
            false_rejections=consensus_metrics.false_rejections,
            reliability_utility=float(reliability_utility),
            navigation_utility=float(np.mean(navigation_rewards)),
            detected_events=detection_metrics.detected_events,
            detected_real_events=detection_metrics.detected_real_events,
            detected_false_events=detection_metrics.detected_false_events,
            detection_latency_sum=detection_metrics.detection_latency_sum,
            detection_latency_count=detection_metrics.detection_latency_count,
            confirmation_latency_sum=consensus_metrics.confirmation_latency_sum,
            confirmation_latency_count=consensus_metrics.confirmation_latency_count,
            recovery_time_sum=recovery_metrics.recovery_time_sum,
            recovery_time_count=recovery_metrics.recovery_time_count,
            recovered_events=recovery_metrics.recovered_events,
            avg_vehicle_throughput_mbps=service_metrics.avg_vehicle_throughput_mbps,
            p5_vehicle_throughput_mbps=service_metrics.p5_vehicle_throughput_mbps,
            qos_outage_ratio=service_metrics.qos_outage_ratio,
            avg_service_latency_ms=service_metrics.avg_service_latency_ms,
            p95_service_latency_ms=service_metrics.p95_service_latency_ms,
            packet_delivery_ratio=service_metrics.packet_delivery_ratio,
            rsu_peak_load=service_metrics.rsu_peak_load,
            rsu_overload_ratio=service_metrics.rsu_overload_ratio,
        )

        world.step += 1
        done = world.step >= self.env_cfg.episode_len
        if done:
            final_event_metrics = self._final_event_metrics()
            step_metrics.event_true_positives = final_event_metrics.event_true_positives
            step_metrics.event_false_positives = final_event_metrics.event_false_positives
            step_metrics.event_false_negatives = final_event_metrics.event_false_negatives
            step_metrics.event_true_negatives = final_event_metrics.event_true_negatives
            unresolved_real_events = sum(
                int(event.is_real and not event.resolved) for event in world.events
            )
            unrecovered_confirmed = sum(
                int(event.is_real and event.confirmed and not event.recovered)
                for event in world.events
            )
            step_metrics.unrecovered_confirmed_events = unrecovered_confirmed
            terminal_penalty = (
                self.value_cfg.missed_event_penalty * unresolved_real_events
                + self.value_cfg.missed_recovery_penalty * unrecovered_confirmed
            )
            reward -= terminal_penalty
            step_metrics.reward -= terminal_penalty
            step_metrics.reliability_utility -= terminal_penalty
            per_agent_rewards -= terminal_penalty / max(len(world.uavs), 1)
        return (
            self.observe(),
            float(reward),
            done,
            {
                "metrics": step_metrics,
                "marginal_gains": marginal_gains,
                "energy_costs": energy_costs,
                "per_agent_rewards": per_agent_rewards,
                "event_signal": event_signal,
                "effective_actions": np.asarray(actions_enum, dtype=np.int64),
            },
        )

    def _apply_actions(self, actions: list[Action]) -> tuple[list[Action], np.ndarray]:
        world = self._world()
        cfg = self.env_cfg
        costs = np.zeros(len(world.uavs), dtype=float)
        effective: list[Action] = []
        for idx, (uav, action) in enumerate(zip(world.uavs, actions)):
            if uav.energy <= cfg.min_operational_energy:
                action = Action.STAY
            effective.append(action)
            if action in MOVE_DELTAS:
                uav.position = np.clip(
                    uav.position + MOVE_DELTAS[action] * cfg.uav_speed,
                    0.0,
                    cfg.grid_size,
                )
                cost = cfg.move_energy_cost if action != Action.STAY else 0.0
            elif action == Action.VERIFY:
                cost = cfg.verify_energy_cost
            elif action == Action.SERVE:
                cost = cfg.serve_energy_cost
            else:
                cost = 0.0
            uav.energy = max(0.0, uav.energy - cost)
            costs[idx] = cost
        return effective, costs

    def _update_background_dynamics(self) -> None:
        world = self._world()
        cfg = self.env_cfg
        if cfg.mobility_mode.lower() in {"trace", "sumo"}:
            self._advance_mobility_trace()
        else:
            self._vehicle_positions = random_walk_positions(
                self._vehicle_positions,
                cfg.vehicle_speed,
                cfg.grid_size,
                self.mobility_rng,
            )
        vehicle_hotspot_drift = 0.015 * (
            self._hotspot_demand_multipliers(self._vehicle_positions) - 1.0
        )
        self._vehicle_demands = np.clip(
            self._vehicle_demands
            + vehicle_hotspot_drift
            + self.mobility_rng.normal(0.0, 0.03, size=len(self._vehicle_demands)),
            0.1,
            1.4,
        )
        if self._mobility_trace_demands is not None:
            self._vehicle_demands = self._mobility_trace_demands[
                self._mobility_trace_step, : cfg.num_vehicles
            ].copy()
        rsu_hotspot_drift = 0.02 * (self._hotspot_demand_multipliers(self._rsu_positions) - 1.0)
        self._rsu_loads = np.clip(
            self._rsu_loads
            + rsu_hotspot_drift
            + self.mobility_rng.normal(0.0, 0.04, size=len(self._rsu_loads)),
            0.2,
            2.2,
        )
        for rsu, load in zip(world.rsus, self._rsu_loads):
            rsu.load = float(load)

    def _build_contributions(self, actions: list[Action]) -> list[Contribution]:
        world = self._world()
        cfg = self.env_cfg
        event_positions = np.array([event.position for event in world.events], dtype=float)
        contributions: list[Contribution] = []
        for uav, action in zip(world.uavs, actions):
            event_probs = np.zeros(len(world.events), dtype=float)
            rsu_capacity = np.zeros(len(world.rsus), dtype=float)
            vehicle_quality = np.zeros(len(world.vehicles), dtype=float)

            if action == Action.VERIFY and len(event_positions):
                distances = np.linalg.norm(event_positions - uav.position, axis=1)
                mask = distances <= cfg.uav_sensing_radius
                noise = self.sensing_rng.normal(0.0, cfg.detection_noise, size=len(event_positions))
                sensor_rates = np.asarray(
                    [
                        (
                            cfg.detector_true_positive_rate
                            if event.is_real
                            else cfg.detector_false_positive_rate
                        )
                        for event in world.events
                    ],
                    dtype=float,
                )
                event_probs[mask] = np.clip(
                    sensor_rates[mask] * np.exp(-distances[mask] / max(cfg.uav_sensing_radius, 1.0))
                    + noise[mask],
                    0.0,
                    1.0,
                )
                if uav.malicious:
                    event_probs *= 0.25

            if action == Action.SERVE:
                rsu_distances = np.linalg.norm(self._rsu_positions - uav.position, axis=1)
                rsu_mask = rsu_distances <= cfg.uav_service_radius
                rsu_capacity[rsu_mask] = cfg.uav_service_capacity * np.exp(
                    -rsu_distances[rsu_mask] / max(cfg.uav_service_radius, 1.0)
                )
                vehicle_distances = np.linalg.norm(self._vehicle_positions - uav.position, axis=1)
                vehicle_mask = vehicle_distances <= cfg.uav_service_radius
                vehicle_quality[vehicle_mask] = np.exp(
                    -vehicle_distances[vehicle_mask] / max(cfg.uav_service_radius, 1.0)
                )

            contributions.append(
                Contribution(
                    uav_id=uav.uid,
                    event_probs=event_probs,
                    rsu_capacity=rsu_capacity,
                    vehicle_quality=vehicle_quality,
                )
            )
        return contributions

    def _build_potential_contributions(self) -> list[Contribution]:
        """Deterministic local utility used for event-triggered committee selection."""

        world = self._world()
        cfg = self.env_cfg
        event_positions = np.array([event.position for event in world.events], dtype=float)
        contributions: list[Contribution] = []
        for uav in world.uavs:
            event_probs = np.zeros(len(world.events), dtype=float)
            rsu_capacity = np.zeros(len(world.rsus), dtype=float)
            vehicle_quality = np.zeros(len(world.vehicles), dtype=float)
            if len(event_positions):
                event_distances = np.linalg.norm(event_positions - uav.position, axis=1)
                event_mask = event_distances <= cfg.uav_comm_radius
                event_probs[event_mask] = np.exp(
                    -event_distances[event_mask] / max(cfg.uav_sensing_radius, 1.0)
                )
            rsu_distances = np.linalg.norm(self._rsu_positions - uav.position, axis=1)
            rsu_mask = rsu_distances <= cfg.uav_service_radius * 1.5
            rsu_capacity[rsu_mask] = cfg.uav_service_capacity * np.exp(
                -rsu_distances[rsu_mask] / max(cfg.uav_service_radius, 1.0)
            )
            vehicle_distances = np.linalg.norm(self._vehicle_positions - uav.position, axis=1)
            vehicle_mask = vehicle_distances <= cfg.uav_service_radius
            vehicle_quality[vehicle_mask] = np.exp(
                -vehicle_distances[vehicle_mask] / max(cfg.uav_service_radius, 1.0)
            )
            contributions.append(
                Contribution(
                    uav_id=uav.uid,
                    event_probs=event_probs,
                    rsu_capacity=rsu_capacity,
                    vehicle_quality=vehicle_quality,
                )
            )
        return contributions

    def _event_signal(self, contributions: list[Contribution]) -> np.ndarray:
        world = self._world()
        if not contributions:
            return np.zeros(len(world.events), dtype=float)
        event_probs = np.vstack([c.event_probs for c in contributions])
        return np.max(event_probs, axis=0)

    def _update_detection_state(self, event_signal: np.ndarray) -> StepMetrics:
        metrics = StepMetrics()
        world = self._world()
        for event in world.events:
            if event.detected_step is not None:
                continue
            if event_signal[event.eid] >= self.consensus_cfg.detection_trigger_threshold:
                event.detected_step = world.step
                metrics.detected_events += 1
                if event.is_real:
                    metrics.detected_real_events += 1
                    metrics.detection_latency_count += 1
                    metrics.detection_latency_sum += world.step
                else:
                    metrics.detected_false_events += 1
        return metrics

    def _run_consensus(
        self,
        contributions: list[Contribution],
        event_signal: np.ndarray,
    ) -> StepMetrics:
        metrics = StepMetrics()
        world = self._world()
        if not self.consensus_cfg.enabled:
            return metrics

        periodic_due = (
            self.consensus_cfg.periodic_interval > 0
            and world.step % self.consensus_cfg.periodic_interval == 0
        )

        for event in world.events:
            if event.confirmed or event.resolved:
                continue
            detected = event_signal[event.eid] >= self.consensus_cfg.detection_trigger_threshold
            if not detected and not periodic_due:
                continue
            event_weights, rsu_weights, vehicle_weights = self._event_local_weights(event)
            marginal_gains = self.value_model.marginal_gains(
                contributions,
                event_weights=event_weights,
                rsu_weights=rsu_weights,
                vehicle_weights=vehicle_weights,
            )
            result = self.consensus.run(
                event,
                world.uavs,
                marginal_gains,
                contributions=contributions,
                value_model=self.value_model,
                event_weights=event_weights,
                rsu_weights=rsu_weights,
                vehicle_weights=vehicle_weights,
                event_evidence=float(event_signal[event.eid]),
            )
            metrics.consensus_trigger_score_sum += result.trigger_score
            metrics.consensus_trigger_score_count += 1
            if result.trigger_suppressed:
                metrics.consensus_trigger_suppressions += 1
                continue
            if not result.attempted:
                continue
            metrics.consensus_attempts += 1
            metrics.consensus_messages += result.messages
            metrics.consensus_bytes_kb += result.bytes_kb
            metrics.consensus_latency_ms += result.latency_ms
            metrics.consensus_packet_successes += int(result.packet_success)
            metrics.consensus_delivered_messages += result.delivered_messages
            if result.success:
                metrics.consensus_successes += 1
                metrics.consensus_decisions += 1
                metrics.consensus_safe_decisions += int(result.byzantine_safe)
                event.decision_step = world.step
                event.resolved = True
                event.resolved_step = world.step
                if result.decision:
                    metrics.confirmed_events += 1
                    event.confirmed = True
                    event.first_confirmed_step = world.step
                    if event.is_real:
                        metrics.true_confirmations += 1
                        metrics.confirmation_latency_count += 1
                        metrics.confirmation_latency_sum += world.step
                    else:
                        metrics.false_confirmations += 1
                else:
                    event.rejected = True
                    if event.is_real:
                        metrics.false_rejections += 1
                    else:
                        metrics.true_rejections += 1
        return metrics

    def _update_recovery(self, contributions: list[Contribution]) -> StepMetrics:
        metrics = StepMetrics()
        world = self._world()
        if contributions:
            service_capacity = np.sum(
                [contribution.rsu_capacity for contribution in contributions], axis=0
            )
        else:
            service_capacity = np.zeros(len(world.rsus), dtype=float)
        for event in world.events:
            if not event.is_real or not event.confirmed or event.recovered:
                continue
            affected = np.asarray(event.affected_rsu_ids, dtype=int)
            if affected.size == 0:
                continue
            delivered_service = float(np.sum(service_capacity[affected]))
            event.recovery_service += delivered_service
            if delivered_service > 0.0:
                self._rsu_loads[affected] = np.maximum(
                    0.2,
                    self._rsu_loads[affected]
                    - self.env_cfg.recovery_load_relief_rate * service_capacity[affected],
                )
            load_recovered = bool(
                np.all(self._rsu_loads[affected] <= self.env_cfg.rsu_base_capacity * 1.05)
            )
            service_recovered = event.recovery_service >= self.env_cfg.recovery_service_requirement
            if load_recovered and service_recovered:
                event.recovered = True
                event.recovery_step = world.step
                metrics.recovered_events += 1
                metrics.recovery_time_count += 1
                confirmation_step = (
                    event.first_confirmed_step
                    if event.first_confirmed_step is not None
                    else world.step
                )
                metrics.recovery_time_sum += max(0, world.step - confirmation_step)
        for rsu, load in zip(world.rsus, self._rsu_loads):
            rsu.load = float(load)
        return metrics

    def _reliability_utility(self, metrics: StepMetrics) -> float:
        cfg = self.value_cfg
        return float(
            cfg.true_confirmation_bonus * metrics.true_confirmations
            + cfg.true_rejection_bonus * metrics.true_rejections
            - cfg.false_confirmation_penalty * metrics.false_confirmations
            - cfg.false_rejection_penalty * metrics.false_rejections
        )

    def _coordination_targets(self) -> np.ndarray:
        world = self._world()
        positions = np.array([uav.position for uav in world.uavs], dtype=float)
        targets = positions.copy()
        event_candidates = [event for event in world.events if not event.resolved]
        assigned_to_event: set[int] = set()
        if event_candidates:
            event = max(event_candidates, key=lambda item: item.uncertainty)
            distances = np.linalg.norm(positions - event.position, axis=1)
            event_budget = min(len(world.uavs), self.consensus_cfg.max_coalition_size)
            assigned_to_event = set(int(idx) for idx in np.argsort(distances)[:event_budget])
            for idx in assigned_to_event:
                targets[idx] = event.position

        if len(self._rsu_positions):
            overload = np.maximum(0.0, self._rsu_loads - self.env_cfg.rsu_base_capacity)
            for idx in range(len(world.uavs)):
                if idx in assigned_to_event:
                    continue
                distances = np.linalg.norm(self._rsu_positions - positions[idx], axis=1)
                scores = overload / (1.0 + distances / max(self.env_cfg.uav_service_radius, 1.0))
                target_idx = (
                    int(np.argmax(scores)) if np.max(scores) > 0.0 else int(np.argmin(distances))
                )
                targets[idx] = self._rsu_positions[target_idx]
        return targets

    def _navigation_rewards(
        self,
        before: np.ndarray,
        after: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        before_distance = np.linalg.norm(before - targets, axis=1)
        after_distance = np.linalg.norm(after - targets, axis=1)
        progress = (before_distance - after_distance) / max(self.env_cfg.uav_speed, 1.0)
        return self.value_cfg.navigation_shaping_weight * np.clip(progress, -1.0, 1.0)

    def _event_local_weights(self, event: EventState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        world = self._world()
        cfg = self.env_cfg
        event_weights = np.zeros(len(world.events), dtype=float)
        event_weights[event.eid] = event.severity * event.uncertainty
        locality_scale = max(cfg.uav_service_radius * 2.0, 1.0)
        rsu_distances = np.linalg.norm(self._rsu_positions - event.position, axis=1)
        rsu_weights = np.maximum(0.0, self._rsu_loads - cfg.rsu_base_capacity) * np.exp(
            -rsu_distances / locality_scale
        )
        vehicle_distances = np.linalg.norm(self._vehicle_positions - event.position, axis=1)
        vehicle_weights = self._vehicle_demands * np.exp(-vehicle_distances / locality_scale)
        return event_weights, rsu_weights, vehicle_weights

    def _final_event_metrics(self) -> StepMetrics:
        metrics = StepMetrics()
        for event in self._world().events:
            if event.is_real and event.confirmed:
                metrics.event_true_positives += 1
            elif event.is_real:
                metrics.event_false_negatives += 1
            elif event.confirmed:
                metrics.event_false_positives += 1
            else:
                metrics.event_true_negatives += 1
        return metrics

    def _redundant_coverage(self, contributions: list[Contribution]) -> float:
        if not contributions:
            return 0.0
        coverage = np.sum([c.vehicle_quality > 0.0 for c in contributions], axis=0)
        duplicate = np.maximum(0.0, coverage - self.value_cfg.duplicate_coverage_threshold)
        return float(np.mean(duplicate > 0.0))

    def _service_metrics(self, contributions: list[Contribution]) -> StepMetrics:
        metrics = StepMetrics()
        world = self._world()
        cfg = self.env_cfg
        if not world.vehicles:
            return metrics

        if contributions:
            vehicle_quality = np.sum([c.vehicle_quality for c in contributions], axis=0)
        else:
            vehicle_quality = np.zeros(len(world.vehicles), dtype=float)

        distances = np.linalg.norm(
            self._vehicle_positions[:, None, :] - self._rsu_positions[None, :, :],
            axis=2,
        )
        nearest = np.argmin(distances, axis=1)
        loads = np.maximum(self._rsu_loads[nearest], 0.1)
        rsu_component = cfg.base_vehicle_throughput_mbps / np.maximum(1.0, loads)
        uav_component = cfg.uav_throughput_gain_mbps * (1.0 - np.exp(-vehicle_quality))
        throughputs = np.maximum(0.0, rsu_component + uav_component)
        overload_penalty = np.maximum(0.0, loads - 1.0)
        latencies = cfg.base_service_latency_ms * (1.0 + overload_penalty)
        latencies /= np.maximum(throughputs / cfg.base_vehicle_throughput_mbps, 0.25)
        latencies = np.minimum(cfg.max_service_latency_ms, latencies)
        pdrs = np.clip(
            1.0 - cfg.packet_loss - cfg.pdr_sensitivity * overload_penalty + 0.03 * uav_component,
            0.0,
            1.0,
        )

        metrics.avg_vehicle_throughput_mbps = float(np.mean(throughputs))
        metrics.p5_vehicle_throughput_mbps = float(np.percentile(throughputs, 5))
        metrics.qos_outage_ratio = float(np.mean(throughputs < cfg.qos_outage_threshold_mbps))
        metrics.avg_service_latency_ms = float(np.mean(latencies))
        metrics.p95_service_latency_ms = float(np.percentile(latencies, 95))
        metrics.packet_delivery_ratio = float(np.mean(pdrs))
        metrics.rsu_peak_load = float(np.max(self._rsu_loads)) if self._rsu_loads.size else 0.0
        metrics.rsu_overload_ratio = (
            float(np.mean(self._rsu_loads > cfg.rsu_base_capacity)) if self._rsu_loads.size else 0.0
        )
        return metrics

    def _world(self) -> WorldState:
        if self.world is None:
            raise RuntimeError("Environment is not reset.")
        return self.world

    def _sample_hotspots(self) -> np.ndarray:
        cfg = self.env_cfg
        roads = np.linspace(0.0, cfg.grid_size, 8)
        hotspots = []
        for _ in range(3):
            hotspots.append([float(self.rng.choice(roads)), float(self.rng.choice(roads))])
        return np.array(hotspots, dtype=float)

    def _apply_hotspot_vehicle_mix(self, positions: np.ndarray) -> np.ndarray:
        cfg = self.env_cfg
        if self._traffic_hotspots.size == 0 or positions.size == 0:
            return positions
        hotspot_fraction = float(np.clip(0.25 + cfg.trace_hotspot_strength, 0.0, 0.85))
        count = int(round(len(positions) * hotspot_fraction))
        if count <= 0:
            return positions
        selected = self.rng.choice(len(positions), size=count, replace=False)
        sigma = max(cfg.grid_size * (0.08 + 0.08 * (1.0 - cfg.trace_hotspot_strength)), 20.0)
        hotspot_idx = self.rng.integers(0, len(self._traffic_hotspots), size=count)
        positions[selected] = self._traffic_hotspots[hotspot_idx] + self.rng.normal(
            0.0, sigma, size=(count, 2)
        )
        return np.clip(positions, 0.0, cfg.grid_size)

    def _hotspot_demand_multiplier(self, position: np.ndarray) -> float:
        return float(self._hotspot_demand_multipliers(position[None, :])[0])

    def _hotspot_demand_multipliers(self, positions: np.ndarray) -> np.ndarray:
        cfg = self.env_cfg
        positions = np.asarray(positions, dtype=float)
        if len(positions) == 0:
            return np.zeros(0, dtype=float)
        if not cfg.trace_hotspot_enabled or self._traffic_hotspots.size == 0:
            return np.ones(len(positions), dtype=float)
        distances = np.linalg.norm(
            positions[:, None, :] - self._traffic_hotspots[None, :, :], axis=2
        )
        sigma = max(cfg.grid_size * 0.18, 1.0)
        proximity = np.max(np.exp(-(distances**2) / (2.0 * sigma**2)), axis=1)
        return 1.0 + cfg.trace_hotspot_strength * proximity

    def _load_mobility_trace(self) -> None:
        path = self.env_cfg.mobility_trace_path
        if not path:
            raise ValueError("mobility_trace_path is required for trace mobility")
        loaded = np.load(path)
        positions = np.asarray(loaded["positions"], dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("Trace positions must have shape [time, vehicle, 2]")
        if positions.shape[1] < self.env_cfg.num_vehicles:
            raise ValueError(
                "Trace contains fewer vehicles than env.num_vehicles: "
                f"{positions.shape[1]} < {self.env_cfg.num_vehicles}"
            )
        self._mobility_trace_positions = np.clip(positions, 0.0, self.env_cfg.grid_size)
        demands = loaded["demands"] if "demands" in loaded.files else None
        if demands is not None:
            demands = np.asarray(demands, dtype=float)
            if demands.shape[:2] != positions.shape[:2]:
                raise ValueError("Trace demands must have shape [time, vehicle]")
            self._mobility_trace_demands = demands

    def _reset_mobility_trace(self) -> np.ndarray:
        if self._mobility_trace_positions is None:
            self._load_mobility_trace()
        assert self._mobility_trace_positions is not None
        frames = len(self._mobility_trace_positions)
        required = self.env_cfg.episode_len + 1
        if frames < required and not self.env_cfg.mobility_trace_loop:
            raise ValueError(f"Trace has {frames} frames but episode requires {required}")
        max_start = max(0, frames - required)
        self._mobility_trace_step = int(self.rng.integers(0, max_start + 1)) if max_start else 0
        return self._mobility_trace_positions[
            self._mobility_trace_step, : self.env_cfg.num_vehicles
        ].copy()

    def _advance_mobility_trace(self) -> None:
        if self._mobility_trace_positions is None:
            raise RuntimeError("Mobility trace was not initialized")
        next_step = self._mobility_trace_step + 1
        if next_step >= len(self._mobility_trace_positions):
            if not self.env_cfg.mobility_trace_loop:
                next_step = len(self._mobility_trace_positions) - 1
            else:
                next_step = 0
        self._mobility_trace_step = next_step
        self._vehicle_positions = self._mobility_trace_positions[
            next_step, : self.env_cfg.num_vehicles
        ].copy()
