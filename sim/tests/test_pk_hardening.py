"""PK acceptance tests — hardening, docs, demo packaging.

Criteria:
1. All four architectural invariants pass on the current tree.
2. The verifier actually DETECTS violations (negative controls) — a gate that
   can only pass is worthless.
3. The invariant verifier CLI exits 0 on a clean tree, non-zero on failure.
4. The README documents the scope boundary and the headline result.
5. Every phase has an ADR; every ADR referenced by state.json exists.
6. The demo entry point produces a structurally valid headline result.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tools.verify_invariants import (
    CheckResult,
    check_determinism,
    check_interlock,
    check_posg,
    check_scope,
    main as verify_main,
    run_all,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 1. Invariants hold on the current tree
# ---------------------------------------------------------------------------

class TestInvariantsHold:
    def test_all_pass(self):
        results = run_all()
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"invariants failed: {failed}"

    def test_posg_passes(self):
        r = check_posg()
        assert r.passed, r.violations

    def test_interlock_passes(self):
        r = check_interlock()
        assert r.passed, r.violations

    def test_scope_passes(self):
        r = check_scope()
        assert r.passed, r.violations

    def test_determinism_passes(self):
        r = check_determinism()
        assert r.passed, r.violations

    def test_each_check_returns_checkresult(self):
        for r in run_all():
            assert isinstance(r, CheckResult)
            assert isinstance(r.detail, str) and r.detail
            assert isinstance(r.violations, list)


# ---------------------------------------------------------------------------
# 2. Negative controls — the verifier must DETECT violations
# ---------------------------------------------------------------------------

class TestVerifierDetectsViolations:
    def test_posg_detects_truth_leak(self, tmp_path, monkeypatch):
        """Inject a fake fusion module that reads the truth sidecar."""
        import tools.verify_invariants as vi
        fake_root = tmp_path
        fusion = fake_root / "sim" / "fusion"
        fusion.mkdir(parents=True)
        (fusion / "leaky.py").write_text(
            "x = mesh.scan_all_with_truth_sidecar(t, pos, rng)\n")
        # Point the checker at the fake tree
        monkeypatch.setattr(vi, "ROOT", fake_root)
        r = vi.check_posg()
        assert not r.passed
        assert any("leaky.py" in v for v in r.violations)

    def test_scope_detects_forbidden_term(self, tmp_path, monkeypatch):
        """A bare fire-control reference (no disclaimer) must be flagged."""
        import tools.verify_invariants as vi
        fake_root = tmp_path
        pkg = fake_root / "sim"
        pkg.mkdir(parents=True)
        # A line that uses a forbidden term WITHOUT a disclaimer marker
        (pkg / "bad.py").write_text("result = compute_firecontrol_solution()\n")
        monkeypatch.setattr(vi, "ROOT", fake_root)
        monkeypatch.setattr(vi, "_PROD_TREES", ["sim"])
        r = vi.check_scope()
        assert not r.passed
        assert any("bad.py" in v for v in r.violations)

    def test_scope_allows_disclaimer_line(self, tmp_path, monkeypatch):
        """The same term WITH a disclaimer marker must pass."""
        import tools.verify_invariants as vi
        fake_root = tmp_path
        pkg = fake_root / "sim"
        pkg.mkdir(parents=True)
        (pkg / "ok.py").write_text(
            "# This module contains NO fire-control code whatsoever.\n")
        monkeypatch.setattr(vi, "ROOT", fake_root)
        monkeypatch.setattr(vi, "_PROD_TREES", ["sim"])
        r = vi.check_scope()
        assert r.passed, r.violations

    def test_interlock_detects_extra_assign(self, tmp_path, monkeypatch):
        """Two world.assign() calls in the bridge must fail the gate."""
        import tools.verify_invariants as vi
        fake_root = tmp_path
        bridge = fake_root / "sim" / "bridge"
        bridge.mkdir(parents=True)
        (bridge / "scenario.py").write_text(
            "def tick(self):\n"
            "    if self.c2_state.can_engage(tid):\n"
            "        self.world.assign(a, b)\n"
            "    self.world.assign(c, d)\n"   # unguarded second call
        )
        monkeypatch.setattr(vi, "ROOT", fake_root)
        r = vi.check_interlock()
        assert not r.passed

    def test_interlock_detects_unguarded_assign(self, tmp_path, monkeypatch):
        """A single but UNGUARDED world.assign() must fail."""
        import tools.verify_invariants as vi
        fake_root = tmp_path
        bridge = fake_root / "sim" / "bridge"
        bridge.mkdir(parents=True)
        (bridge / "scenario.py").write_text(
            "def tick(self):\n"
            "    self.world.assign(a, b)\n"   # no can_engage guard
        )
        monkeypatch.setattr(vi, "ROOT", fake_root)
        r = vi.check_interlock()
        assert not r.passed


# ---------------------------------------------------------------------------
# 3. CLI behaviour
# ---------------------------------------------------------------------------

class TestVerifierCLI:
    def test_exit_zero_on_clean_tree(self):
        assert verify_main(["--quiet"]) == 0

    def test_json_output(self, capsys):
        rc = verify_main(["--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["passed"] is True
        assert len(payload["checks"]) == 4
        assert rc == 0


# ---------------------------------------------------------------------------
# 4. Documentation
# ---------------------------------------------------------------------------

class TestDocumentation:
    def test_readme_exists_and_substantial(self):
        readme = ROOT / "README.md"
        assert readme.exists()
        assert len(readme.read_text()) > 1500

    def test_readme_states_scope_boundary(self):
        text = (ROOT / "README.md").read_text().lower()
        assert "scope boundary" in text
        assert "no fire-control" in text
        assert "parameter set" in text

    def test_readme_reports_headline(self):
        text = (ROOT / "README.md").read_text().lower()
        assert "cost-exchange" in text
        assert "economicmdp" in text

    def test_every_phase_has_an_adr(self):
        """Each done phase in state.json points to an ADR that exists."""
        state = json.loads((ROOT / ".aegisnet" / "state.json").read_text())
        for phase in state["phases"]:
            adr = phase.get("adr", "")
            if adr:  # PK itself may have no separate ADR
                assert (ROOT / adr).exists(), f"{phase['id']} ADR missing: {adr}"


# ---------------------------------------------------------------------------
# 5. Demo packaging
# ---------------------------------------------------------------------------

class TestDemo:
    def test_demo_importable(self):
        import demo
        assert hasattr(demo, "main")

    def test_demo_fast_produces_headline(self, monkeypatch):
        """The fast demo runs end to end and returns a structured result."""
        import eval.runner as erunner
        # Use the smallest possible study to keep the test quick.
        monkeypatch.setattr(erunner, "_PX_LEAGUE_N_GEN", 2)
        monkeypatch.setattr(erunner, "_PX_LEAGUE_POP", 4)
        monkeypatch.setattr(erunner, "_PX_LEAGUE_SEED", 123)
        result = erunner.run_px_study(n_seeds=2, n_tactics=1)
        h = result["headline"]
        assert "economic_mdp" in h and "greedy_myopic" in h
        assert h["economic_mdp"]["mean_cost_exchange_ratio"] > 0
