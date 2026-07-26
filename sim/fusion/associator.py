"""Data association behind an Associator interface.

GNN (Global Nearest Neighbor) first: Hungarian assignment on gated Mahalanobis
distance. JPDA can be added later as a second implementation of the same
interface — callers depend only on `associate()`.
"""
from __future__ import annotations

import abc
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from sim.fusion.tracker import InternalTrack
    from sim.sensing.sensors import SensorReport

# chi-square gate, 2 dof (cartesian) and 1 dof (bearing), ~99% containment
GATE_CART = 9.21
GATE_BEARING = 6.63
NO_MATCH_COST = 1e6


class Associator(abc.ABC):
    @abc.abstractmethod
    def associate(
        self,
        tracks: List["InternalTrack"],
        reports: List["SensorReport"],
    ) -> Tuple[Dict[int, int], List[int]]:
        """Return ({track_index: report_index}, [unassociated report indices])."""


class GNNAssociator(Associator):
    def associate(
        self,
        tracks: List["InternalTrack"],
        reports: List["SensorReport"],
    ) -> Tuple[Dict[int, int], List[int]]:
        nT, nR = len(tracks), len(reports)
        if nT == 0 or nR == 0:
            return {}, list(range(nR))

        cost = np.full((nT, nR), NO_MATCH_COST)
        for i, trk in enumerate(tracks):
            for j, rep in enumerate(reports):
                if rep.kind == "cartesian":
                    d2 = trk.kf.mahalanobis2_cart(rep.position, rep.covariance)
                    if d2 <= GATE_CART:
                        cost[i, j] = d2
                else:
                    d2 = trk.kf.mahalanobis2_bearing(rep.bearing, rep.covariance, rep.sensor_pos)
                    if d2 <= GATE_BEARING:
                        cost[i, j] = d2

        rows, cols = linear_sum_assignment(cost)
        assignments: Dict[int, int] = {}
        used_reports = set()
        for i, j in zip(rows, cols):
            if cost[i, j] < NO_MATCH_COST:
                assignments[i] = j
                used_reports.add(j)

        unassociated = [j for j in range(nR) if j not in used_reports]
        return assignments, unassociated
