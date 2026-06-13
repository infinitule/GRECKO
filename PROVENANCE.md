# Build Provenance

GRECKO was built **AI-assisted**, and this document records how — for
transparency with reviewers, investors, and contributors.

## How it was built

- **Tooling:** [Claude Code](https://claude.ai/code), Anthropic's agentic
  coding environment, driving Anthropic's **Claude** models.
- **Direction:** human-in-the-loop. The project owner set the objectives and
  scope, and advanced the build phase by phase at explicit checkpoints; the
  agent implemented, tested, and documented each phase.
- **Method:** a checkpoint-gated, phased build (P0 → PM). Every phase shipped
  with an Architecture Decision Record (`docs/ADR-*.md`), an acceptance test
  suite, and measured results before the next phase began.

## Why you can trust the result

The provenance does not ask for trust — the artifacts earn it:

- **305 automated tests** pass (`make test`).
- **4/4 architectural invariants** are enforced mechanically on every commit
  (`grecko verify`): POSG discipline, the C2 authorization interlock, the scope
  boundary, and replay determinism. The gate is proven to *detect* violations
  via negative-control tests, not merely to return green.
- **Determinism:** every result is reproducible from a seed; the SHA-256 of the
  event log is the replay acceptance criterion.
- **Honest metrics:** headline claims (cost-exchange ratio, intercept rate) are
  integer-grounded and platform-stable. Continuous geometry metrics that vary
  across math-library backends are *not* used as acceptance thresholds — a
  discipline adopted after CI surfaced the difference across platforms.

## Scope

GRECKO is **simulation, research, and C2-software only**: no fire-control, no
RF/waveform design, no weapon or effector hardware control. Effectors are
parameter sets. This boundary is part of the AI build's guardrails and is
enforced by the invariant gate on every commit.

## Reproducing the build's outputs

```bash
pip install -e .
grecko verify         # architectural invariant gate
make test             # full acceptance suite
grecko demo --fast    # headline cost-exchange study
make gif              # the swarm-for-swarm demo animation
```
