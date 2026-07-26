# Security Policy

## Scope boundary is a security property

GRECKO is **simulation, research, and C2-software only**. It contains no
fire-control, no RF/jamming/waveform design, and no effector or weapon hardware
control. Effectors are parameter sets (cost, kill probability). This boundary is
enforced mechanically on every commit by the architectural invariant gate
(`grecko verify` / `python -m tools.verify_invariants`), which fails CI if:

- any production fusion module reads ground truth (POSG violation),
- the C2 authorization interlock is bypassed (more than one, or an unguarded,
  engagement path),
- out-of-scope vocabulary appears outside disclaimers, or
- a fixed scenario stops replaying to a byte-identical event-log hash.

A change that weakens any of these is treated as a security regression.

## Reporting a vulnerability

Please report suspected vulnerabilities privately rather than via a public
issue. Use GitHub's **"Report a vulnerability"** (Security → Advisories) on the
repository, or open a minimal private channel with the maintainers.

Include: affected version/commit, a description, and reproduction steps. We aim
to acknowledge within a few business days and to agree on a disclosure timeline
with the reporter.

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |
| < 1.0   | ❌        |

## Hardening notes for deployers

- Run the C2 bridge as the provided non-root container user.
- Terminate the WebSocket bridge behind a TLS-terminating reverse proxy and
  authenticated network boundary; the bridge itself speaks plaintext WS for the
  simulation/console and is not intended to be exposed to untrusted networks.
- Treat the event log as the audit record of record; ship it to durable storage.
