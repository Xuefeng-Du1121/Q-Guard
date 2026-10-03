from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UAVState:
    uid: int
    position: np.ndarray
    energy: float
    malicious: bool = False
    trust_score: float = 0.75


@dataclass
class RSUState:
    rid: int
    position: np.ndarray
    load: float
    capacity: float


@dataclass
class VehicleState:
    vid: int
    position: np.ndarray
    demand: float
    throughput: float = 0.0


@dataclass
class EventState:
    eid: int
    position: np.ndarray
    severity: float
    uncertainty: float
    is_real: bool = True
    confirmed: bool = False
    resolved: bool = False
    rejected: bool = False
    detected_step: int | None = None
    first_confirmed_step: int | None = None
    resolved_step: int | None = None
    decision_step: int | None = None
    recovered: bool = False
    recovery_step: int | None = None
    recovery_service: float = 0.0
    affected_rsu_ids: list[int] = field(default_factory=list)


@dataclass
class WorldState:
    step: int
    uavs: list[UAVState]
    rsus: list[RSUState]
    vehicles: list[VehicleState]
    events: list[EventState]
