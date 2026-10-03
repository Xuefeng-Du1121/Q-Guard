from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class StepMetrics:
    reward: float = 0.0
    value_total: float = 0.0
    value_info: float = 0.0
    value_relief: float = 0.0
    value_qos: float = 0.0
    energy_cost: float = 0.0
    consensus_messages: int = 0
    consensus_bytes_kb: float = 0.0
    consensus_latency_ms: float = 0.0
    consensus_successes: int = 0
    consensus_attempts: int = 0
    consensus_packet_successes: int = 0
    consensus_delivered_messages: int = 0
    consensus_safe_decisions: int = 0
    consensus_decisions: int = 0
    consensus_trigger_suppressions: int = 0
    consensus_trigger_score_sum: float = 0.0
    consensus_trigger_score_count: int = 0
    redundant_coverage: float = 0.0
    confirmed_events: int = 0
    true_confirmations: int = 0
    false_confirmations: int = 0
    true_rejections: int = 0
    false_rejections: int = 0
    reliability_utility: float = 0.0
    navigation_utility: float = 0.0
    recovered_events: int = 0
    unrecovered_confirmed_events: int = 0
    event_true_positives: int = 0
    event_false_positives: int = 0
    event_false_negatives: int = 0
    event_true_negatives: int = 0
    detected_events: int = 0
    detected_real_events: int = 0
    detected_false_events: int = 0
    detection_latency_sum: float = 0.0
    detection_latency_count: int = 0
    confirmation_latency_sum: float = 0.0
    confirmation_latency_count: int = 0
    recovery_time_sum: float = 0.0
    recovery_time_count: int = 0
    avg_vehicle_throughput_mbps: float = 0.0
    p5_vehicle_throughput_mbps: float = 0.0
    qos_outage_ratio: float = 0.0
    avg_service_latency_ms: float = 0.0
    p95_service_latency_ms: float = 0.0
    packet_delivery_ratio: float = 0.0
    rsu_peak_load: float = 0.0
    rsu_overload_ratio: float = 0.0


@dataclass
class EpisodeMetrics:
    steps: int = 0
    total_reward: float = 0.0
    total_value: float = 0.0
    info_value: float = 0.0
    relief_value: float = 0.0
    qos_value: float = 0.0
    energy_cost: float = 0.0
    consensus_messages: int = 0
    consensus_bytes_kb: float = 0.0
    consensus_latency_ms: float = 0.0
    consensus_attempts: int = 0
    consensus_successes: int = 0
    consensus_packet_successes: int = 0
    consensus_delivered_messages: int = 0
    consensus_safe_decisions: int = 0
    consensus_decisions: int = 0
    consensus_trigger_suppressions: int = 0
    consensus_trigger_score_sum: float = 0.0
    consensus_trigger_score_count: int = 0
    confirmed_events: int = 0
    true_confirmations: int = 0
    false_confirmations: int = 0
    true_rejections: int = 0
    false_rejections: int = 0
    reliability_utility: float = 0.0
    navigation_utility: float = 0.0
    recovered_events: int = 0
    unrecovered_confirmed_events: int = 0
    event_true_positives: int = 0
    event_false_positives: int = 0
    event_false_negatives: int = 0
    event_true_negatives: int = 0
    detected_events: int = 0
    detected_real_events: int = 0
    detected_false_events: int = 0
    detection_latency_sum: float = 0.0
    detection_latency_count: int = 0
    confirmation_latency_sum: float = 0.0
    confirmation_latency_count: int = 0
    recovery_time_sum: float = 0.0
    recovery_time_count: int = 0
    redundant_coverage: float = 0.0
    avg_vehicle_throughput_mbps: float = 0.0
    p5_vehicle_throughput_mbps: float = 0.0
    qos_outage_ratio: float = 0.0
    avg_service_latency_ms: float = 0.0
    p95_service_latency_ms: float = 0.0
    packet_delivery_ratio: float = 0.0
    rsu_peak_load: float = 0.0
    rsu_overload_ratio: float = 0.0

    def update(self, step: StepMetrics) -> None:
        self.steps += 1
        self.total_reward += step.reward
        self.total_value += step.value_total
        self.info_value += step.value_info
        self.relief_value += step.value_relief
        self.qos_value += step.value_qos
        self.energy_cost += step.energy_cost
        self.consensus_messages += step.consensus_messages
        self.consensus_bytes_kb += step.consensus_bytes_kb
        self.consensus_latency_ms += step.consensus_latency_ms
        self.consensus_attempts += step.consensus_attempts
        self.consensus_successes += step.consensus_successes
        self.consensus_packet_successes += step.consensus_packet_successes
        self.consensus_delivered_messages += step.consensus_delivered_messages
        self.consensus_safe_decisions += step.consensus_safe_decisions
        self.consensus_decisions += step.consensus_decisions
        self.consensus_trigger_suppressions += step.consensus_trigger_suppressions
        self.consensus_trigger_score_sum += step.consensus_trigger_score_sum
        self.consensus_trigger_score_count += step.consensus_trigger_score_count
        self.confirmed_events += step.confirmed_events
        self.true_confirmations += step.true_confirmations
        self.false_confirmations += step.false_confirmations
        self.true_rejections += step.true_rejections
        self.false_rejections += step.false_rejections
        self.reliability_utility += step.reliability_utility
        self.navigation_utility += step.navigation_utility
        self.recovered_events += step.recovered_events
        self.unrecovered_confirmed_events += step.unrecovered_confirmed_events
        self.event_true_positives += step.event_true_positives
        self.event_false_positives += step.event_false_positives
        self.event_false_negatives += step.event_false_negatives
        self.event_true_negatives += step.event_true_negatives
        self.detected_events += step.detected_events
        self.detected_real_events += step.detected_real_events
        self.detected_false_events += step.detected_false_events
        self.detection_latency_sum += step.detection_latency_sum
        self.detection_latency_count += step.detection_latency_count
        self.confirmation_latency_sum += step.confirmation_latency_sum
        self.confirmation_latency_count += step.confirmation_latency_count
        self.recovery_time_sum += step.recovery_time_sum
        self.recovery_time_count += step.recovery_time_count
        self.redundant_coverage += step.redundant_coverage
        self.avg_vehicle_throughput_mbps += step.avg_vehicle_throughput_mbps
        self.p5_vehicle_throughput_mbps += step.p5_vehicle_throughput_mbps
        self.qos_outage_ratio += step.qos_outage_ratio
        self.avg_service_latency_ms += step.avg_service_latency_ms
        self.p95_service_latency_ms += step.p95_service_latency_ms
        self.packet_delivery_ratio += step.packet_delivery_ratio
        self.rsu_peak_load += step.rsu_peak_load
        self.rsu_overload_ratio += step.rsu_overload_ratio

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        if self.steps > 0:
            data["avg_reward"] = self.total_reward / self.steps
            data["avg_value"] = self.total_value / self.steps
            data["avg_redundant_coverage"] = self.redundant_coverage / self.steps
            data["avg_vehicle_throughput_mbps"] = self.avg_vehicle_throughput_mbps / self.steps
            data["avg_p5_vehicle_throughput_mbps"] = self.p5_vehicle_throughput_mbps / self.steps
            data["avg_qos_outage_ratio"] = self.qos_outage_ratio / self.steps
            data["avg_service_latency_ms"] = self.avg_service_latency_ms / self.steps
            data["avg_p95_service_latency_ms"] = self.p95_service_latency_ms / self.steps
            data["avg_packet_delivery_ratio"] = self.packet_delivery_ratio / self.steps
            data["avg_rsu_peak_load"] = self.rsu_peak_load / self.steps
            data["avg_rsu_overload_ratio"] = self.rsu_overload_ratio / self.steps
        else:
            data["avg_reward"] = 0.0
            data["avg_value"] = 0.0
            data["avg_redundant_coverage"] = 0.0
            data["avg_vehicle_throughput_mbps"] = 0.0
            data["avg_p5_vehicle_throughput_mbps"] = 0.0
            data["avg_qos_outage_ratio"] = 0.0
            data["avg_service_latency_ms"] = 0.0
            data["avg_p95_service_latency_ms"] = 0.0
            data["avg_packet_delivery_ratio"] = 0.0
            data["avg_rsu_peak_load"] = 0.0
            data["avg_rsu_overload_ratio"] = 0.0

        data["consensus_success_rate"] = safe_div_or_nan(
            self.consensus_successes, self.consensus_attempts
        )
        data["avg_consensus_latency_ms"] = safe_div_or_nan(
            self.consensus_latency_ms, self.consensus_attempts
        )
        data["consensus_packet_success_rate"] = safe_div_or_nan(
            self.consensus_packet_successes, self.consensus_attempts
        )
        data["consensus_message_delivery_rate"] = safe_div_or_nan(
            self.consensus_delivered_messages, self.consensus_messages
        )
        data["consensus_safety_rate"] = safe_div_or_nan(
            self.consensus_safe_decisions, self.consensus_decisions
        )
        data["consensus_trigger_suppression_rate"] = safe_div_or_nan(
            self.consensus_trigger_suppressions,
            self.consensus_attempts + self.consensus_trigger_suppressions,
        )
        data["avg_consensus_trigger_score"] = safe_div_or_nan(
            self.consensus_trigger_score_sum,
            self.consensus_trigger_score_count,
        )
        data["false_confirmation_rate"] = safe_div_or_nan(
            self.false_confirmations, self.true_confirmations + self.false_confirmations
        )
        tp = self.event_true_positives
        fp = self.event_false_positives
        fn = self.event_false_negatives
        tn = self.event_true_negatives
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        data["confirmation_precision"] = precision
        data["confirmation_recall"] = recall
        data["confirmation_tpr"] = recall
        data["confirmation_fpr"] = safe_div(fp, fp + tn)
        data["confirmation_miss_rate"] = safe_div(fn, tp + fn)
        data["confirmation_f1"] = safe_div(2.0 * precision * recall, precision + recall)
        data["detection_precision"] = safe_div(
            self.detected_real_events,
            self.detected_real_events + self.detected_false_events,
        )
        data["detection_recall"] = safe_div(self.detected_real_events, tp + fn)
        data["detection_false_alarm_rate"] = safe_div(self.detected_false_events, fp + tn)
        data["avg_detection_latency"] = safe_div(
            self.detection_latency_sum, self.detection_latency_count
        )
        data["avg_confirmation_latency"] = safe_div(
            self.confirmation_latency_sum, self.confirmation_latency_count
        )
        data["avg_recovery_time"] = safe_div(self.recovery_time_sum, self.recovery_time_count)
        data["recovery_success_rate"] = safe_div_or_nan(
            self.recovered_events, self.true_confirmations
        )
        data["communication_kb_per_confirmed_event"] = safe_div_or_nan(
            self.consensus_bytes_kb, self.confirmed_events
        )
        data["communication_kb_per_true_confirmation"] = safe_div_or_nan(
            self.consensus_bytes_kb, tp
        )
        data["messages_per_true_confirmation"] = safe_div_or_nan(self.consensus_messages, tp)
        data["consensus_messages_per_attempt"] = safe_div_or_nan(
            self.consensus_messages, self.consensus_attempts
        )
        data["consensus_kb_per_attempt"] = safe_div_or_nan(
            self.consensus_bytes_kb, self.consensus_attempts
        )
        return data


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def safe_div_or_nan(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))
