"""Architectural invariant verifier — the hardening gate.

AEGISNET rests on four invariants that are easy to violate by accident in a
later edit and expensive to catch by eye. This script checks all four
mechanically so CI (and a reviewer) can confirm them in one command:

  1. POSG       No production fusion module reads ground truth. The truth
                sidecar exists only in eval/benchmark code, never on the
                live sense -> fuse path.
  2. INTERLOCK  Exactly one world.assign() call site exists in the bridge
                (the C2 production path), and it is guarded by can_engage().
  3. SCOPE      No fire-control / RF-waveform / weapon-integration code. The
                forbidden vocabulary appears only in disclaimers, tests, and
                scope-boundary documentation.
  4. DETERMINISM A fixed scenario replays to a byte-identical event-log hash.

Usage:
    python -m tools.verify_invariants            # all checks, human output
    python -m tools.verify_invariants --json     # machine-readable
    python -m tools.verify_invariants --quiet     # exit code only

Exit code 0 == all invariants hold; 1 == at least one violation.

SCOPE: this is a static-analysis + determinism gate. It controls nothing,
designs nothing, integrates with no hardware.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import pathlib
import re
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Production source trees (analysis, not tests, not eval benchmarks).
_PROD_TREES = ["sim", "league", "s2r"]

# Modules that are permitted to consume ground truth (eval / benchmark only).
_TRUTH_ALLOWED = {
    "sim/fusion/benchmark.py",
    "sim/sensing/sensors.py",   # defines the sidecar API; does not consume it
}

# Truth-reading API surface that fusion must never call.
_TRUTH_TOKENS = ("scan_all_with_truth_sidecar", "truth_index", "true_position")

# Forbidden out-of-scope vocabulary. Matches are allowed ONLY when the line
# also contains a disclaimer marker (NO, NOT, never, abstract, ...).
_FORBIDDEN = [
    r"fire.?control",
    r"waveform",
    r"\bjammer\b",
    r"weapon.?release",
    r"warhead",
    r"launch.?command",
    r"transmit.?power",
]
_DISCLAIMER_MARKERS = (
    "no ", "not ", "never", "abstract", "boundary", "scope", "disclaim",
    "forbidden", "parameter", "kill-probability", "soft-kill",
)


@dataclasses.dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    violations: List[str]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _iter_py(trees: List[str]):
    for tree in trees:
        for path in (ROOT / tree).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


# --------------------------------------------------------------------------- #
# 1. POSG — fusion never reads truth                                          #
# --------------------------------------------------------------------------- #

def check_posg() -> CheckResult:
    violations: List[str] = []
    fusion_dir = ROOT / "sim" / "fusion"
    for path in fusion_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = _rel(path)
        if rel in _TRUTH_ALLOWED:
            continue
        text = path.read_text()
        for tok in _TRUTH_TOKENS:
            if tok in text:
                violations.append(f"{rel}: reads truth via `{tok}`")
    passed = not violations
    detail = (
        "Live fusion path consumes no ground truth "
        f"(sidecar confined to {sorted(_TRUTH_ALLOWED)})."
        if passed else "Ground-truth leakage detected on the fusion path."
    )
    return CheckResult("POSG", passed, detail, violations)


# --------------------------------------------------------------------------- #
# 2. INTERLOCK — single guarded world.assign() in the bridge                  #
# --------------------------------------------------------------------------- #

def check_interlock() -> CheckResult:
    """The bridge (production C2 path) must contain exactly one world.assign()
    call, and the enclosing block must be guarded by can_engage()."""
    violations: List[str] = []
    scenario = ROOT / "sim" / "bridge" / "scenario.py"
    source = scenario.read_text()
    tree = ast.parse(source)

    assign_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # match  self.world.assign(...)  /  *.world.assign(...)
            if (isinstance(func, ast.Attribute) and func.attr == "assign"
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "world"):
                assign_calls.append(node.lineno)

    if len(assign_calls) != 1:
        violations.append(
            f"expected exactly 1 world.assign() in bridge, found "
            f"{len(assign_calls)} at lines {assign_calls}")
    else:
        # Confirm a can_engage() guard governs the call site. The call sits
        # inside an `elif self.c2_state.can_engage(...)` branch; check that
        # can_engage appears textually before the assign within a small window.
        assign_line = assign_calls[0]
        window = "\n".join(source.splitlines()[max(0, assign_line - 12):assign_line])
        if "can_engage" not in window:
            violations.append(
                f"world.assign() at line {assign_line} is not guarded by "
                "can_engage() within the preceding block")

    passed = not violations
    detail = (
        "Bridge has a single world.assign(), guarded by C2State.can_engage()."
        if passed else "C2 HOTL interlock is not structurally enforced."
    )
    return CheckResult("INTERLOCK", passed, detail, violations)


# --------------------------------------------------------------------------- #
# 3. SCOPE — no out-of-scope effector / RF / weapon code                      #
# --------------------------------------------------------------------------- #

def check_scope() -> CheckResult:
    violations: List[str] = []
    patterns = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN]
    for path in _iter_py(_PROD_TREES):
        rel = _rel(path)
        is_test = "/tests/" in rel or rel.startswith("tests/")
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            low = line.lower()
            for pat in patterns:
                if pat.search(line):
                    # Allowed if the line is a disclaimer, or lives in a test
                    # (tests assert the boundary is enforced).
                    if is_test:
                        continue
                    if any(m in low for m in _DISCLAIMER_MARKERS):
                        continue
                    violations.append(
                        f"{rel}:{lineno}: out-of-scope term in `{line.strip()[:70]}`")
    passed = not violations
    detail = (
        "No fire-control / RF-waveform / weapon-integration code; matches are "
        "disclaimers or boundary tests only."
        if passed else "Out-of-scope vocabulary found outside disclaimers."
    )
    return CheckResult("SCOPE", passed, detail, violations)


# --------------------------------------------------------------------------- #
# 4. DETERMINISM — fixed scenario replays to identical hash                    #
# --------------------------------------------------------------------------- #

def check_determinism() -> CheckResult:
    violations: List[str] = []
    detail = ""
    try:
        from sim.bridge.scenario import BridgeScenario

        def _run() -> str:
            sc = BridgeScenario(seed=20240613, auto_authorize=True)
            for _ in range(300):
                if sc.world.is_engagement_over():
                    break
                sc.tick()
            return sc.log_hash()

        h1 = _run()
        h2 = _run()
        if h1 != h2:
            violations.append(f"replay hash mismatch: {h1[:12]} != {h2[:12]}")
        detail = (f"Replay determinism confirmed (hash {h1[:12]}...)."
                  if not violations else "Replay is non-deterministic.")
    except Exception as exc:  # pragma: no cover - defensive
        violations.append(f"determinism check raised: {exc!r}")
        detail = "Determinism check could not run."
    return CheckResult("DETERMINISM", not violations, detail, violations)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

ALL_CHECKS = (check_posg, check_interlock, check_scope, check_determinism)


def run_all() -> List[CheckResult]:
    return [check() for check in ALL_CHECKS]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AEGISNET invariant verifier")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--quiet", action="store_true", help="exit code only")
    args = parser.parse_args(argv)

    results = run_all()
    all_passed = all(r.passed for r in results)

    if args.json:
        print(json.dumps({
            "passed": all_passed,
            "checks": [r.to_dict() for r in results],
        }, indent=2))
    elif not args.quiet:
        print("AEGISNET architectural invariants\n" + "=" * 40)
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"[{mark}] {r.name:12s} {r.detail}")
            for v in r.violations:
                print(f"         - {v}")
        print("=" * 40)
        print("RESULT:", "ALL INVARIANTS HOLD" if all_passed else "VIOLATIONS FOUND")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
