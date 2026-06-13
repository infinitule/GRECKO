"""Tests for the GRECKO operator CLI (grecko ...)."""
from __future__ import annotations

import json

import pytest

import grecko
from grecko.cli import build_parser, main


class TestCli:
    def test_version_constant(self):
        assert grecko.__version__ == "1.0.0"

    def test_parser_has_all_subcommands(self):
        parser = build_parser()
        # argparse stores subparser choices on the _SubParsersAction
        subs = [a for a in parser._actions
                if a.__class__.__name__ == "_SubParsersAction"]
        assert subs, "no subparsers registered"
        choices = set(subs[0].choices)
        assert {"version", "verify", "demo", "eval", "serve", "figures"} <= choices

    def test_version_command_exits_zero(self, capsys):
        rc = main(["version"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "GRECKO 1.0.0" in out
        assert "simulation" in out.lower()

    def test_global_version_flag(self):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0

    def test_verify_command_runs_gate(self, capsys):
        """`grecko verify --json` runs the invariant gate and returns 0."""
        rc = main(["verify", "--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["passed"] is True
        assert len(payload["checks"]) == 4
        assert rc == 0

    def test_missing_subcommand_errors(self):
        with pytest.raises(SystemExit):
            main([])
