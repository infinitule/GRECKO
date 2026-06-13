# Contributing to GRECKO

Thanks for your interest. GRECKO is a counter-swarm **simulation and C2** code
base; contributions must stay within that scope (see below).

## Ground rules

1. **Scope boundary is non-negotiable.** No fire-control, RF/waveform/jammer
   design, or weapon/effector hardware control. Effectors are parameter sets.
   The invariant gate enforces this — a PR that trips it will not merge.
2. **Determinism.** The simulation is single-threaded, fixed-timestep, and
   seeded. New code must keep replays reproducible (same seed → same event-log
   hash). Assert on integer-grounded, platform-stable metrics — not on
   continuous geometry extrema that vary across BLAS backends.
3. **Every phase/feature ships with tests and rationale.** Add or update an ADR
   in `docs/` for design decisions.

## Development setup

```bash
make dev        # editable install with dev + viz extras
make verify     # architectural invariant gate
make test       # full acceptance suite (305 tests)
make demo       # headline cost-exchange study
```

## Before you open a PR

- [ ] `make verify` passes (POSG, interlock, scope, determinism)
- [ ] `make test` passes
- [ ] New behavior has tests; design changes have an ADR
- [ ] No out-of-scope vocabulary outside disclaimers/tests
- [ ] Commit messages explain the *why*, not just the *what*

CI runs the invariant gate then the full suite on every push and PR.

## Architecture orientation

The pipeline is `sense → fuse → classify → predict → allocate → engage` with a
human-on-the-loop interlock. Start at [`README.md`](README.md) and the ADR index
in [`docs/`](docs/) (ADR-000 … ADR-012). The operator CLI is `grecko`
(`grecko --help`).
