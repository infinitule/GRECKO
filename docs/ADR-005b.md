# ADR-005b: Swarm intent prediction — collective inference under observation dropout

**Status:** Accepted  
**Phase:** PB — Pillar B, swarm intent prediction

## Context

Tracking treats N drones as N independent objects. Real swarms coordinate:
feint, mass, split, screen. The value estimate v_j that Pillar A optimises is
only as good as the intent behind it — a feint that *looks* committed (direct
heading, confirmed hostile) will absorb expensive rounds unless something
infers the collective behaviour. The contribution claimed by the plan is
robustness to PARTIAL observation: inferring swarm intent when a large
fraction of tracks is occluded.

## Decisions

### Scripted doctrines as the bootstrap dataset

Five generators (`frontal_saturation`, `feint_main_axis`, `pincer`,
`isr_loiter`, `leader_follower`) produce labelled trajectories in a stable
`Scenario` container. The league (PL) will export its discovered doctrines in
this same format for retraining — the container is the contract.

The feint signature is deliberately learnable-but-not-trivially: feints are
smaller, slower, looser, and *overcommit their heading* (perfectly direct)
until they break off at t≈20–30 s. A heading-only heuristic therefore reads
the feint as maximally committed until the turn; the model reads size,
speed, and formation tightness from the first observation.

### Lightweight model with the full interface, GNN deferred

Clusters from single-linkage on (proximity AND velocity-similarity) — the
graph structure of the plan — followed by a 10-feature collective summary per
cluster and a two-hidden-layer numpy MLP (32/16, softmax over 5 intents).
The Trajectron++-style graph-attention encoder + trajectory decoder is the
documented v2 upgrade; consumers depend on the predictor interface
(`tracks → SwarmIntent messages`) which is already final, so the upgrade is a
checkpoint swap. Rationale: this repo must train and evaluate deterministically
in CI in seconds; the architecture novelty is not load-bearing for the
acceptance criteria, the observation-dropout training discipline is.

### Observation dropout as the robustness mechanism

Every training sample is extracted from a view of the swarm with 40% of agents
randomly occluded. The model never sees a complete swarm in training and is
evaluated at 40% dropout on held-out scenarios.

### Forecast: linear extrapolation with growing sigma

Cluster-centroid forecast at 1 s steps over a 10 s horizon, 1-sigma uncertainty
growing 4 m/s. Honest placeholder; the decoder upgrade replaces it. Sigma
growth is test-asserted so no consumer can assume certainty.

### Value multiplier wiring into PA (the staged wiring protocol)

`m = 1 + 0.8·P(main_axis) − 0.9·P(feint) − 0.5·P(isr) − 0.5·P(reserve)`,
clipped to [0.1, 2.0], applied multiplicatively to member tracks' v_j.
This makes a confident feint nearly free to decline and a confident main axis
nearly twice as valuable.

## Rejected alternatives

- **Graph-attention network now:** jax/torch dependency, minutes-long training,
  GPU variance in CI — for no change in the acceptance evidence. Deferred, not
  rejected; the interface is already shaped for it.
- **Per-track (non-collective) intent classification:** misses exactly the
  coordination signal (relative size, formation) that distinguishes feint from
  thrust. Rejected.
- **Wiring intent directly into classification (P3):** intent is a *swarm-level*
  posterior; classification is per-track and human-auditable. Mixing them muddies
  the audit trail. The multiplier wires into allocation, where the economic
  decision lives. Rejected.

## Acceptance evidence (held-out scenarios, 40% occlusion)

- **Lead time:** model stably identifies the true main axis **24.2 s earlier**
  (mean over 12 held-out feint scenarios) than the kinematics heading heuristic;
  model called 12/12, heuristic 12/12.
- **Calibration:** ECE = 0.011 (n=286).
- **Accuracy:** 99.6% held-out cluster-intent accuracy (n=279).
- **PA wiring delta (staged wiring protocol):** on a feint+main scenario where
  the stub (uniform-prior) allocator spends 1 round on the feint, the PB-wired
  allocator spends 0 — main-axis coverage unchanged. The test asserts the stub
  baseline is non-vacuous (must engage the feint) before asserting improvement.
- Forecast sigma strictly grows; 5-s centroid forecast error < 100 m on
  straight ingress.
