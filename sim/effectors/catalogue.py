"""Effector type catalogue — parameter sets only.

SCOPE BOUNDARY: this module contains costs, kill probabilities, kinematic
envelopes, and engagement geometry constraints. It contains NO fire-control
code, NO RF waveform design, NO hardware integration, NO weapon-system API
calls. Kill probabilities are abstract sim parameters used by the allocator's
cost-exchange optimisation.

Four effector types cover the relevant intercept modes:
  kinetic_interceptor — fast, expensive, high Pk vs hardened threats.
  net_capture_drone   — cheap, slow, high Pk vs quadrotors; leaves asset intact.
  ew_soft_kill        — low cost per round, RF-dependent threats only; soft kill.
  collision_drone     — kamikaze; single-use, low cost, moderate Pk.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional


@dataclasses.dataclass(frozen=True)
class EffectorType:
    """All parameters that the allocator and kinematics engine need.

    Nothing here touches hardware. `p_k_table` keys match the
    `threat_type` field classification will eventually produce.
    """
    effector_id: str
    description: str
    cost_usd: float
    p_k_table: Dict[str, float]
    reload_time_s: float
    max_range_m: float
    min_range_m: float
    min_engagement_angle_rad: float
    endurance_s: float
    max_speed_mps: float
    max_turn_rate_radps: float
    soft_kill_only: bool = False

    def p_k(self, threat_type: str) -> float:
        """Return kill probability for this threat type; default to lowest
        table value if type not explicitly listed (conservative)."""
        if threat_type in self.p_k_table:
            return self.p_k_table[threat_type]
        if self.p_k_table:
            return min(self.p_k_table.values())
        return 0.0

    def geometry_valid(self, range_m: float, aspect_angle_rad: float) -> bool:
        """True when the engagement geometry is within this effector's envelope."""
        if range_m < self.min_range_m or range_m > self.max_range_m:
            return False
        if aspect_angle_rad < self.min_engagement_angle_rad:
            return False
        return True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Catalogue — four canonical effector types
# ---------------------------------------------------------------------------

KINETIC_INTERCEPTOR = EffectorType(
    effector_id="kinetic_interceptor",
    description="High-speed kinetic hit-to-kill missile. Highest Pk, highest cost.",
    cost_usd=90_000,
    p_k_table={
        "quadrotor":   0.85,
        "fixed_wing":  0.90,
        "rf_dependent": 0.80,
        "hardened":    0.75,
    },
    reload_time_s=30.0,
    max_range_m=5000.0,
    min_range_m=100.0,
    min_engagement_angle_rad=0.0,
    endurance_s=120.0,
    max_speed_mps=120.0,
    max_turn_rate_radps=2.0,
    soft_kill_only=False,
)

NET_CAPTURE_DRONE = EffectorType(
    effector_id="net_capture_drone",
    description="Slow pursuit drone deploying a capture net. Cheap, nonlethal, limited to quadrotors.",
    cost_usd=3_000,
    p_k_table={
        "quadrotor":   0.78,
        "fixed_wing":  0.15,   # too fast; net rarely connects
        "rf_dependent": 0.60,
        "hardened":    0.05,
    },
    reload_time_s=60.0,
    max_range_m=800.0,
    min_range_m=10.0,
    min_engagement_angle_rad=0.0,
    endurance_s=180.0,
    max_speed_mps=22.0,
    max_turn_rate_radps=1.2,
    soft_kill_only=False,
)

EW_SOFT_KILL = EffectorType(
    effector_id="ew_soft_kill",
    description=(
        "Abstract EW-effect payload. Modelled purely as a soft-kill probability "
        "against RF-dependent threats. NOT a jammer or waveform design — this is "
        "a kill-probability parameter for the sim allocator only."
    ),
    cost_usd=500,
    p_k_table={
        "quadrotor":   0.0,    # non-RF-dependent: unaffected
        "fixed_wing":  0.0,
        "rf_dependent": 0.90,  # high Pk on RF-guided/controlled threats
        "hardened":    0.0,
    },
    reload_time_s=5.0,
    max_range_m=1200.0,
    min_range_m=0.0,
    min_engagement_angle_rad=0.0,
    endurance_s=300.0,
    max_speed_mps=60.0,
    max_turn_rate_radps=1.5,
    soft_kill_only=True,
)

COLLISION_DRONE = EffectorType(
    effector_id="collision_drone",
    description="Single-use collision interceptor. Low cost, moderate Pk, one-shot.",
    cost_usd=800,
    p_k_table={
        "quadrotor":   0.70,
        "fixed_wing":  0.50,
        "rf_dependent": 0.65,
        "hardened":    0.20,
    },
    reload_time_s=45.0,
    max_range_m=600.0,
    min_range_m=5.0,
    min_engagement_angle_rad=0.0,
    endurance_s=90.0,
    max_speed_mps=35.0,
    max_turn_rate_radps=1.8,
    soft_kill_only=False,
)

CATALOGUE: Dict[str, EffectorType] = {
    e.effector_id: e
    for e in [KINETIC_INTERCEPTOR, NET_CAPTURE_DRONE, EW_SOFT_KILL, COLLISION_DRONE]
}


def get(effector_id: str) -> Optional[EffectorType]:
    return CATALOGUE.get(effector_id)


def best_effector_for(threat_type: str) -> EffectorType:
    """Return the effector with highest cost-adjusted Pk for this threat type.

    cost-adjusted Pk = Pk / cost_usd (kill value per dollar spent).
    This is what the EconomicMDP allocator maximises when all effectors are
    available and feasible.
    """
    return max(
        CATALOGUE.values(),
        key=lambda e: e.p_k(threat_type) / e.cost_usd,
    )
