"""CLI entry point: python -m sim.run --scenario <path> [--headless] [--seed <N>]

Prints an end-of-run summary to stdout; optionally writes the JSONL event log.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from sim.core.events import EventLog
from sim.core.scenario import load_scenario
from sim.core.simple_alloc import greedy_assign


MAX_SIM_TIME = 600.0  # seconds; safety ceiling to prevent infinite loops


def run(
    scenario_path: str,
    seed: int = 42,
    headless: bool = True,
    log_path: Optional[str] = None,
) -> dict:
    rng = np.random.default_rng(seed)
    log = EventLog()

    world = load_scenario(scenario_path, rng, log)

    log.append(0.0, "SIM_START", data={"seed": seed, "scenario": scenario_path})

    alloc_interval = 10  # re-run allocator every N ticks
    tick = 0

    while not world.is_engagement_over() and world.t < MAX_SIM_TIME:
        if tick % alloc_interval == 0:
            assignments = greedy_assign(world.interceptors, world.hostiles)
            for iid, hid in assignments.items():
                world.assign(iid, hid)
        world.step()
        tick += 1

    summary = world.summary()
    log.append(world.t, "SIM_END", data=summary)

    if log_path:
        log.write_to(log_path)

    return {**summary, "log_hash": world.log_hash(), "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGISNET headless engagement runner")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--log", help="Write JSONL event log to this path")
    args = parser.parse_args()

    result = run(args.scenario, seed=args.seed, headless=args.headless, log_path=args.log)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
