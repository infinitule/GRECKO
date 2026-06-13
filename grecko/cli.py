"""GRECKO unified command-line entrypoint.

Installed as the `grecko` console script (see pyproject [project.scripts]):

    grecko version                 print version + scope banner
    grecko verify                  run the architectural invariant gate
    grecko demo [--fast]           run the headline cost-exchange study
    grecko eval [--seeds N]        run the Monte Carlo evaluation, print JSON
    grecko serve [--host --port]   start the C2 WebSocket bridge server
    grecko figures                 (re)generate the docs/figures assets

Every subcommand is a thin, documented wrapper over an existing module so the
package has a single, discoverable operator surface.
"""
from __future__ import annotations

import argparse
import json
import sys

from grecko import __version__

_SCOPE = ("GRECKO — simulation, research, and C2-software only. "
          "Effectors are parameter sets; no fire-control, RF design, or weapon "
          "integration.")


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"GRECKO {__version__}")
    print(_SCOPE)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from tools.verify_invariants import main as verify_main
    return verify_main([] if not args.json else ["--json"])


def _cmd_demo(args: argparse.Namespace) -> int:
    import demo
    argv = ["--fast"] if args.fast else []
    if args.json:
        argv += ["--json", args.json]
    return demo.main(argv)


def _cmd_eval(args: argparse.Namespace) -> int:
    from eval.runner import run_px_study
    result = run_px_study(n_seeds=args.seeds, n_tactics=args.tactics)
    print(json.dumps(result["headline"], indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import asyncio
    from sim.bridge.server import serve
    print(f"GRECKO {__version__} — C2 bridge starting on "
          f"ws://{args.host}:{args.port} (seed={args.seed})")
    try:
        asyncio.run(serve(args.host, args.port, args.seed))
    except KeyboardInterrupt:
        print("\nGRECKO C2 bridge stopped.")
    return 0


def _cmd_figures(args: argparse.Namespace) -> int:
    from tools.make_figures import main as figures_main
    figures_main([] if not args.pm else ["--pm", args.pm])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grecko",
        description="GRECKO counter-swarm decision engine (simulation & C2 only).",
    )
    p.add_argument("--version", action="version", version=f"GRECKO {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("version", help="print version and scope banner")
    sp.set_defaults(func=_cmd_version)

    sp = sub.add_parser("verify", help="run the architectural invariant gate")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=_cmd_verify)

    sp = sub.add_parser("demo", help="run the headline cost-exchange study")
    sp.add_argument("--fast", action="store_true", help="reduced study (~45s)")
    sp.add_argument("--json", metavar="PATH", default=None, help="write raw result")
    sp.set_defaults(func=_cmd_demo)

    sp = sub.add_parser("eval", help="Monte Carlo evaluation; print headline JSON")
    sp.add_argument("--seeds", type=int, default=6, help="seeds per scenario")
    sp.add_argument("--tactics", type=int, default=3, help="PL tactics to include")
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("serve", help="start the C2 WebSocket bridge server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=_cmd_serve)

    sp = sub.add_parser("figures", help="(re)generate docs/figures assets")
    sp.add_argument("--pm", default=None, help="path to a PM study JSON")
    sp.set_defaults(func=_cmd_figures)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
