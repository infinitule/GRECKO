"""Minimal greedy allocator for P0 — one interceptor per hostile, nearest first.

The real allocator (Pillar A / PA phase) replaces this behind the same interface.
This is only used by the P0 run harness so the engagement kernel can demonstrate
end-to-end operation before the allocation layer exists.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from sim.core.entities import HostileUAS, Interceptor
from sim.core.vec import norm


def greedy_assign(
    interceptors: Dict[str, Interceptor],
    hostiles: Dict[str, HostileUAS],
) -> Dict[str, Optional[str]]:
    """Return {interceptor_id: hostile_id | None}."""
    alive_i = [iv for iv in interceptors.values() if iv.alive]
    alive_h = [h for h in hostiles.values() if h.alive]
    assignments: Dict[str, Optional[str]] = {iv.id: None for iv in alive_i}
    assigned_targets = set()

    # Sort interceptors by their distance to nearest unassigned hostile
    for iv in alive_i:
        best_h = None
        best_dist = float("inf")
        for h in alive_h:
            if h.id in assigned_targets:
                continue
            d = norm(iv.pos - h.pos)
            if d < best_dist:
                best_dist = d
                best_h = h
        if best_h:
            assignments[iv.id] = best_h.id
            assigned_targets.add(best_h.id)

    return assignments
