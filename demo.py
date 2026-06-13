"""GRECKO headline demo — one command, the whole story.

Reproduces the project's headline result: the EconomicMDP allocator's
cost-exchange advantage over a greedy baseline, measured by Monte Carlo
across adversarial attack formations discovered by the co-evolution league.

Usage:
    python demo.py                  # full headline study (~3 min)
    python demo.py --fast           # reduced seeds/tactics (~45 s)
    python demo.py --json out.json  # also write the raw result

This is a SIMULATION. Effectors are parameter sets (cost, kill probability).
Nothing here controls hardware, designs RF, or integrates a weapon.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import eval.runner as erunner
from tools.verify_invariants import run_all


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _print_invariants() -> bool:
    _banner("ARCHITECTURAL INVARIANTS")
    results = run_all()
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.name:12s} {r.detail}")
    return all(r.passed for r in results)


def _print_headline(result: dict) -> None:
    h = result["headline"]
    eco = h["economic_mdp"]
    grd = h["greedy_myopic"]

    _banner("HEADLINE: COST-EXCHANGE (EconomicMDP vs GreedyMyopic)")
    print(f"  {'metric':28s}{'EconomicMDP':>14s}{'GreedyMyopic':>14s}")
    print("  " + "-" * 54)
    print(f"  {'cost-exchange ratio':28s}"
          f"{eco['mean_cost_exchange_ratio']:>14.2f}"
          f"{grd['mean_cost_exchange_ratio']:>14.2f}")
    print(f"  {'intercept rate':28s}"
          f"{eco['mean_intercept_rate']:>14.2%}"
          f"{grd['mean_intercept_rate']:>14.2%}")
    print(f"  {'mean defense spend / ep':28s}"
          f"{eco['mean_defense_spend_usd']:>14,.0f}"
          f"{grd['mean_defense_spend_usd']:>14,.0f}")
    print(f"  {'mean leakers':28s}"
          f"{eco['mean_leakers']:>14.2f}"
          f"{grd['mean_leakers']:>14.2f}")
    print("  " + "-" * 54)
    print(f"  Cost-efficiency gain (lower CER): "
          f"{h['cer_improvement_frac']:+.1%}")
    print(f"  Intercept-rate trade-off:         "
          f"{h['intercept_rate_delta']:+.1%}")

    _banner("PER-SCENARIO BREAKDOWN")
    print(f"  {'scenario':12s}{'eco CER':>10s}{'grd CER':>10s}"
          f"{'eco IR':>9s}{'grd IR':>9s}")
    print("  " + "-" * 48)
    eco_by = result["by_scenario"]["EconomicMDP"]
    grd_by = result["by_scenario"]["GreedyMyopic"]
    for label in eco_by:
        e, g = eco_by[label], grd_by[label]
        print(f"  {label:12s}"
              f"{e['mean_cost_exchange_ratio']:>10.1f}"
              f"{g['mean_cost_exchange_ratio']:>10.1f}"
              f"{e['mean_intercept_rate']:>9.0%}"
              f"{g['mean_intercept_rate']:>9.0%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRECKO headline demo")
    parser.add_argument("--fast", action="store_true",
                        help="reduced study for a quick look (~45 s)")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="write the raw study result to PATH")
    parser.add_argument("--skip-invariants", action="store_true",
                        help="skip the architectural invariant gate")
    args = parser.parse_args(argv)

    print("GRECKO — counter-swarm defense simulation")
    print("Simulation / research / C2-software only. No weapon integration.")

    invariants_ok = True
    if not args.skip_invariants:
        invariants_ok = _print_invariants()
        if not invariants_ok:
            print("\nWARNING: architectural invariants FAILED — results suspect.")

    _banner("MONTE CARLO STUDY")
    if args.fast:
        erunner._PX_LEAGUE_N_GEN = 2
        erunner._PX_LEAGUE_POP = 4
        erunner._PX_LEAGUE_SEED = 77
        n_seeds, n_tactics = 3, 1
        print("  mode: FAST (2x4 league, 3 seeds, 1 tactic)")
    else:
        n_seeds, n_tactics = 6, 3
        print("  mode: FULL (4x6 league, 6 seeds, 3 tactics)")

    t0 = time.time()
    result = erunner.run_px_study(n_seeds=n_seeds, n_tactics=n_tactics)
    print(f"  completed in {time.time() - t0:.0f}s")

    _print_headline(result)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n  raw result written to {args.json}")

    print()
    return 0 if invariants_ok else 1


if __name__ == "__main__":
    sys.exit(main())
