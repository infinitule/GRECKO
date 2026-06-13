# AEGISNET

A counter-swarm air-defense **simulation and research testbed**. AEGISNET
models the full sense → fuse → classify → predict → allocate → engage decision
loop against adversarial UAS swarms, with a human-on-the-loop command-and-control
console and an adversarial co-evolution league that discovers attack tactics the
defense was never scripted against.

The headline result: a magazine-conscious **economic allocator** intercepts
swarm raids at materially lower cost-per-kill than a greedy baseline, and the
advantage is largest against the adversarially-discovered attacks.

---

## Scope boundary (read this first)

AEGISNET is **simulation, research, and C2-software only.** It deliberately
contains none of the following, and a CI gate
(`python -m tools.verify_invariants`) enforces their absence:

- **No fire-control.** Nothing computes a firing solution or commands a launch.
- **No RF / jamming / waveform design.** Communications denial is an *abstract
  link-probability model*; the "EW" effector is a *kill-probability parameter*
  for the allocator, explicitly not a jammer or waveform.
- **No effector or weapon hardware control.** Effectors are **parameter sets**
  (cost, kill probability, kinematic envelope) consumed by the cost-exchange
  optimizer. There is no hardware API, no integration layer.

The simulation studies the *decision problem* — which interceptor, against
which threat, at what cost, under what authorization — not the kinetics of any
real effector.

---

## Headline result

Monte Carlo over adversarial attack formations (see `python demo.py`):

| Metric                    | EconomicMDP | GreedyMyopic |
|---------------------------|-------------|--------------|
| Cost-exchange ratio (CER) | **24.8**    | 32.0         |
| Intercept rate            | 33%         | 38%          |
| Mean defense spend / ep   | **$74,000** | $96,000      |

**~22% lower cost per intercepted threat**, trading off ~5 points of raw
intercept rate — the economic allocator rationally holds fire on low-value
feint tracks to preserve magazine for the main axis. The advantage is largest
(up to 34% CER reduction) on the attack patterns discovered by the
co-evolution league. Full reasoning in [docs/ADR-010.md](docs/ADR-010.md).

---

## Architecture

The world state is plain data; systems are pure functions over it. A fixed
50 Hz timestep (`DT = 0.02`) plus a seeded RNG make every run deterministic:
the SHA-256 of the JSONL event log is the acceptance criterion for replay.

The cardinal invariant is **POSG discipline** — nothing downstream of sensing
may read ground truth. Sensors consume truth and emit noisy, identity-free
`SensorReport`s; everything after that sees only estimates.

| Phase | Module | What it does |
|-------|--------|--------------|
| P0 | `sim/core` | World kernel, entities, kinematics, deterministic event log |
| P1 | `sim/sensing` | Heterogeneous imperfect sensor mesh (radar / EO-IR / acoustic) |
| P2 | `sim/fusion` | Kalman multi-target tracker, two-pass GNN association |
| P3 | `sim/classify` | Transparent, swappable threat classifier with provenance |
| PC | `sim/comms` | Degradable comms link-probability model (abstract) |
| PE | `sim/effectors` | Effector catalogue — **parameter sets only** |
| **PA** | `sim/alloc` | **Pillar A:** economic magazine-rationing allocator |
| **PB** | `learn/intent` | **Pillar B:** swarm-intent prediction, value multipliers |
| PV | `sim/bridge`, `viz` | Human-on-the-loop C2 console (WebSocket + TypeScript) |
| **PL** | `league` | **Pillar C:** adversarial co-evolution league (μ+λ ES) |
| PS | `s2r` | Sim-to-real validation strategy (reality-gap + gates) |
| PX | `eval` | Monte Carlo cost-exchange evaluation, headline figure |

The three **Pillars** (PA / PB / PL) are the research contributions; the other
phases are the substrate they need to be measured honestly.

Each phase has an architecture decision record in [`docs/`](docs/)
(ADR-000 … ADR-010).

---

## Quick start

```bash
pip install -e .            # numpy, scipy, pyyaml

python demo.py --fast       # headline study + invariant gate (~45 s)
python demo.py              # full study (~3 min)

python -m tools.verify_invariants   # architectural invariant gate
python -m pytest sim/tests -q       # full acceptance suite (259 tests)
```

### The C2 console (human-on-the-loop)

```bash
python -m sim.bridge.server          # asyncio WebSocket sim server
cd viz && npm install && npm run dev # TypeScript C2 console
```

The console renders the live air picture with uncertainty ellipses, intent
forecasts, and the comms mesh; the operator authorizes, holds, or marks-friendly
each track. The HOTL interlock is an *architectural* property: `world.assign()`
is reachable on the production path only through `C2State.can_engage()`, and
the invariant verifier proves there is exactly one such guarded call site.

---

## Hardening: the invariant gate

`python -m tools.verify_invariants` mechanically checks the four properties the
whole system rests on:

1. **POSG** — no production fusion module reads the truth sidecar.
2. **INTERLOCK** — exactly one `world.assign()` in the bridge, guarded by
   `can_engage()`.
3. **SCOPE** — no fire-control / RF-waveform / weapon vocabulary outside
   disclaimers and boundary tests.
4. **DETERMINISM** — a fixed scenario replays to a byte-identical log hash.

It exits non-zero on any violation, so it doubles as a CI gate.

---

## Repository layout

```
sim/        simulation kernel and the sense->...->engage pipeline
  core/     world, entities, kinematics, event log (P0)
  sensing/  imperfect sensor mesh (P1)
  fusion/   Kalman tracker + GNN association (P2)
  classify/ threat classifier (P3)
  comms/    abstract link-degradation model (PC)
  effectors/ parameter-set catalogue (PE)
  alloc/    economic allocator + greedy / oracle baselines (PA)
  bridge/   full-stack scenario + C2 WebSocket server (PV)
  tests/    acceptance suite, one file per phase
learn/      intent model + training (PB)
league/     adversarial co-evolution (PL)
s2r/        sim-to-real validation strategy (PS)
eval/       Monte Carlo cost-exchange evaluation (PX)
viz/        TypeScript + Vite C2 console (PV)
tools/      architectural invariant verifier
docs/       ADR-000 … ADR-010, figures
demo.py     headline demo entry point
```

---

## Determinism & reproducibility

Every result in this repository is reproducible from a seed. The simulation is
single-threaded, fixed-timestep, and seeded; the league, the sensitivity sweep,
and the Monte Carlo evaluation all thread their seeds explicitly. If a change
breaks replay determinism, `tools/verify_invariants.py` fails.
