# Changelog

All notable changes to GRECKO are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-06-13

First production-grade release. The full counter-swarm decision engine,
validated in simulation end to end.

### Added
- **Pipeline (P0–PE):** deterministic 50 Hz kernel, imperfect sensor mesh,
  Kalman + GNN multi-target tracker, transparent classifier, abstract comms
  degradation model, effector parameter catalogue.
- **Pillar A (PA):** economic, magazine-rationing allocator.
- **Pillar B (PB):** swarm-intent prediction with value multipliers.
- **Pillar C (PL):** adversarial co-evolution league ((μ+λ)-ES).
- **PM:** mutual co-evolution — Blue adapts effector loadout + rationing back to
  discovered Red tactics. Cuts cost-per-intercept 97.7% (≈43×) at identical
  intercept rate.
- **C2 (PV):** human-on-the-loop console (WebSocket bridge + TypeScript/Vite),
  authorization interlock, audit trail, replay.
- **Validation (PS, PX):** sim-to-real reality-gap strategy with measurement
  gates; Monte Carlo cost-exchange evaluation.
- **Hardening (PK):** `grecko verify` invariant gate (POSG, interlock, scope,
  determinism) with negative-control tests.
- **Operator surface:** unified `grecko` CLI (`version`, `verify`, `demo`,
  `eval`, `serve`, `figures`).
- **Deployment:** Dockerfile (non-root, healthcheck), console image,
  `docker-compose.yml`, `Makefile`, `DEPLOYMENT.md`.
- **Project assets:** investor landing page, swarm-for-swarm demo animation,
  result figures, ADR-000 … ADR-012.

### Results
- Economic allocator vs greedy baseline: ~22% lower cost-per-intercept.
- Swarm-for-swarm demo: 8/11 stopped for $6.4k vs 7/11 for $630k (≈98× cheaper).
- 305 automated tests passing; 4/4 architectural invariants gated in CI.

### Notes
- Scope: simulation, research, and C2-software only. No fire-control, RF design,
  or weapon/hardware integration. Enforced by the invariant gate.
- Headline claims (cost-exchange ratio, intercept rate) are integer-grounded and
  platform-stable; continuous geometry metrics (margins) are platform-dependent
  and are not used as acceptance thresholds.
