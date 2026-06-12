"""ValidationGate registry — the "strategy" half of P-S2R.

Every dimension in the RealityGap envelope maps to exactly one real-world
measurement that would pin the sim parameter to reality. A simulation
conclusion may only be carried across the reality gap once the gates for
every parameter it is sensitive to (see sensitivity.tornado) have passed.

Tiers:
  bench        lab measurement of a component in isolation
  hil          hardware-in-the-loop: real component driving the live sim
  field-analog data collection from representative trials (no effectors,
               no weapon hardware — instrumented observation only)

SCOPE: gates describe measurement and data collection ONLY. None of them
involve hardware control, RF transmission design, or effector integration.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

from s2r.gap import GAP_DIMS

VALID_TIERS = ("bench", "hil", "field-analog")


@dataclasses.dataclass(frozen=True)
class ValidationGate:
    gap_dim: str          # which RealityGap dimension this gate pins down
    parameter: str        # the modelled sim parameter being validated
    tier: str             # "bench" | "hil" | "field-analog"
    measurement: str      # what is measured, how
    accept_criterion: str # machine-checkable pass condition
    at_risk: str          # which sim conclusion is falsified if the gate fails

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


VALIDATION_GATES: List[ValidationGate] = [
    ValidationGate(
        gap_dim="pd_scale",
        parameter="SensorNode.pd_max / pd_knee_frac (Pd-vs-range curve)",
        tier="field-analog",
        measurement=(
            "Calibrated target presentations (instrumented commercial UAS with "
            "GPS truth logger) flown at 5 range bins per sensor; >=200 scan "
            "opportunities per bin; empirical detection rate per bin."
        ),
        accept_criterion=(
            "Empirical Pd within +/-0.05 of the modelled curve at every bin, "
            "else re-fit pd_max/pd_knee_frac and re-run the P1 acceptance suite."
        ),
        at_risk="Track formation time and therefore every downstream lead-time claim.",
    ),
    ValidationGate(
        gap_dim="noise_scale",
        parameter="SensorNode.noise_std (measurement scatter)",
        tier="bench",
        measurement=(
            "Static surveyed target at known position; >=1000 measurements per "
            "sensor; per-axis sample standard deviation vs modelled noise_std."
        ),
        accept_criterion=(
            "Sample sigma within 25% of modelled sigma per axis; else update "
            "noise_std and re-run P2 tracker purity tests."
        ),
        at_risk="Kalman gating and GNN association purity (P2 saturation knee).",
    ),
    ValidationGate(
        gap_dim="clutter_scale",
        parameter="SensorNode.false_alarm_rate (Poisson clutter)",
        tier="field-analog",
        measurement=(
            "24h clutter census at the deployment-representative site with no "
            "targets present; count of confirmed-track-grade false reports per "
            "sensor per hour."
        ),
        accept_criterion=(
            "Observed clutter rate within 2x of modelled rate; else update "
            "false_alarm_rate and re-verify P3 'birds non-hostile >=95%'."
        ),
        at_risk="Classifier false-engagement rate and magazine waste estimates.",
    ),
    ValidationGate(
        gap_dim="latency_add_s",
        parameter="SensorNode.latency (report timestamp offset)",
        tier="bench",
        measurement=(
            "Timestamped loopback: GPS-disciplined clock at sensor and fusion "
            "host; distribution of report-arrival minus truth-event time over "
            ">=10k reports."
        ),
        accept_criterion=(
            "p95 latency within modelled latency + 50 ms; else update latency "
            "and re-check allocator intercept-geometry feasibility (PA)."
        ),
        at_risk="Intercept geometry margins computed by the allocator.",
    ),
    ValidationGate(
        gap_dim="update_rate_scale",
        parameter="SensorNode.update_rate (sustained scan rate)",
        tier="bench",
        measurement=(
            "8h soak test at full target load (replayed P2 saturation scenario "
            "via HIL injection); achieved scan interval distribution."
        ),
        accept_criterion=(
            "p99 scan interval <= 1.2x modelled period under sustained load; "
            "else derate update_rate and re-run P2 saturation tests."
        ),
        at_risk="Track continuity under saturation (the 80-100 target knee).",
    ),
    ValidationGate(
        gap_dim="comms_drop_add",
        parameter="CommsConfig.base_drop_rate (abstract link loss)",
        tier="hil",
        measurement=(
            "Representative datalink radios on a bench RF channel emulator "
            "(commercial, receive-side statistics only); packet delivery ratio "
            "across the modelled geometry envelope."
        ),
        accept_criterion=(
            "Measured loss within +0.05 of modelled base_drop_rate across the "
            "envelope; else update CommsConfig and re-run PC degradation tests."
        ),
        at_risk="Mesh partition behaviour and allocator staleness assumptions (PC/PA).",
    ),
    ValidationGate(
        gap_dim="comms_radius_scale",
        parameter="CommsConfig.comm_radius (usable link range)",
        tier="field-analog",
        measurement=(
            "Connectivity-vs-distance walk test with the deployment radio set "
            "at the representative site; range at which delivery ratio crosses "
            "90%."
        ),
        accept_criterion=(
            "Measured 90%-delivery range >= 0.8x modelled comm_radius; else "
            "update comm_radius and re-run PC partition scenarios."
        ),
        at_risk="Interceptor coordination beyond line-of-sight of the C2 node.",
    ),
    ValidationGate(
        gap_dim="target_speed_scale",
        parameter="HostileUAS.speed envelope (threat library kinematics)",
        tier="field-analog",
        measurement=(
            "Track-speed distribution extracted from instrumented trials and "
            "open threat-library survey; p95 closing speed vs modelled envelope."
        ),
        accept_criterion=(
            "p95 observed speed <= modelled max speed; else extend PARAM_BOUNDS "
            "and re-run PA/PL acceptance (allocator + league still converge)."
        ),
        at_risk="Reaction-time budget: every intercept-feasibility conclusion.",
    ),
]


def gates_by_dim() -> Dict[str, ValidationGate]:
    return {g.gap_dim: g for g in VALIDATION_GATES}


def coverage_check() -> List[str]:
    """Return the list of RealityGap dimensions with NO validation gate.

    Empty list == full coverage (acceptance criterion). Also flags gates
    pointing at unknown dimensions, which would silently void coverage.
    """
    covered = {g.gap_dim for g in VALIDATION_GATES}
    missing = [dim for dim in GAP_DIMS if dim not in covered]
    unknown = [d for d in covered if d not in GAP_DIMS]
    return missing + [f"unknown:{d}" for d in unknown]
