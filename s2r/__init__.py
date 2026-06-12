"""P-S2R — sim-to-real validation strategy.

Quantifies how much the AEGISNET headline conclusions depend on simulation
fidelity, and documents the real-world measurement that would validate each
modelled parameter before any conclusion is carried across the reality gap.

Modules:
  gap          RealityGap — bounded perturbation envelope over modelled params
  gates        ValidationGate registry — one real-world measurement per param
  episodes     Fixed probe engagement run under a perturbed world
  sensitivity  One-at-a-time sweeps + tornado ranking
  robustness   Domain-randomized conclusion-stability study

SCOPE: simulation and analysis only. The gates describe measurements
(bench / hardware-in-loop / field-analog data collection), never hardware
control, RF design, or weapon integration.
"""
