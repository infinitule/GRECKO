"""Allocator interface and shared input/output containers.

All three solvers (GreedyMyopic, EconomicMDP, OracleLP) implement Allocator.
Callers depend only on this file — never on solver internals.
"""
from __future__ import annotations

import abc
import dataclasses
import math
from typing import Dict, List, Optional

import numpy as np

from sim.alloc.types import Assignment, MagazineState
from sim.classify.classifier import ThreatAssessment
from sim.effectors.catalogue import EffectorType
from sim.fusion.tracker import InternalTrack


@dataclasses.dataclass
class InterceptorState:
    interceptor_id: str
    pos: np.ndarray
    effector_type: str     # key into EffectorType catalogue
    endurance_s: float
    speed_mps: float


@dataclasses.dataclass
class AllocInput:
    t: float
    interceptors: List[InterceptorState]
    assessments: List[ThreatAssessment]   # sorted by priority descending
    magazine: MagazineState
    effector_catalogue: Dict[str, EffectorType]
    adjacency: Dict[str, List[str]]       # comms graph from PC layer
    asset_pos: np.ndarray
    asset_value: float = 1_000_000.0     # dollar value of the protected asset
    estimated_waves_remaining: int = 1   # hint for EconomicMDP horizon
    lambda_cost: float = 0.05            # cost-exchange knob in [0,1]; higher = more rationing
                                         # semantics: benefit = v_j*Pk - λ*(cost/asset_value)


class Allocator(abc.ABC):
    @abc.abstractmethod
    def allocate(self, inp: AllocInput) -> List[Assignment]:
        """Map the current air picture + magazine to a set of Assignments.

        Every interceptor must appear in the output exactly once, with action
        ASSIGN, HOLD_FIRE, or RTB. Every HOLD_FIRE must have a non-empty
        hold_reason in its provenance.
        """
