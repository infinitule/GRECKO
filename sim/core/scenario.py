"""Load and validate a scenario YAML into World-ready data structures."""
from __future__ import annotations

import numpy as np
import yaml

from sim.core.entities import Asset, HostileUAS, Interceptor
from sim.core.events import EventLog
from sim.core.world import World


def load_scenario(path: str, rng: np.random.Generator, log: EventLog) -> World:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    dt = float(cfg.get("dt", 0.02))  # 50 Hz default

    asset_cfg = cfg["asset"]
    asset = Asset(
        id=asset_cfg.get("id", "asset_0"),
        pos=np.array(asset_cfg.get("pos", [0.0, 0.0])),
        hp=float(asset_cfg.get("hp", 10.0)),
        value=float(asset_cfg.get("value", 1.0)),
    )

    world = World(dt=dt, rng=rng, asset=asset, log=log)

    for hcfg in cfg.get("hostiles", []):
        pos = np.array(hcfg["pos"], dtype=float)
        waypoints = [np.array(wp, dtype=float) for wp in hcfg.get("waypoints", [[0.0, 0.0]])]
        direction = waypoints[0] - pos
        mag = float(np.linalg.norm(direction))
        if mag < 1e-9:
            heading = 0.0
        else:
            heading = float(np.arctan2(direction[1], direction[0]))
        speed = float(hcfg.get("speed", 20.0))
        vel = np.array([np.cos(heading), np.sin(heading)]) * speed

        h = HostileUAS(
            id=hcfg["id"],
            pos=pos,
            vel=vel,
            heading=heading,
            speed=speed,
            max_turn_rate=float(hcfg.get("max_turn_rate", 0.5)),
            weave_amplitude=float(hcfg.get("weave_amplitude", 0.0)),
            weave_period=float(hcfg.get("weave_period", 5.0)),
            waypoints=waypoints,
            t_spawned=0.0,
        )
        world.spawn_hostile(h)

    for icfg in cfg.get("interceptors", []):
        pos = np.array(icfg["pos"], dtype=float)
        heading = float(icfg.get("heading", 0.0))
        speed = float(icfg.get("speed", 50.0))
        vel = np.array([np.cos(heading), np.sin(heading)]) * speed

        iv = Interceptor(
            id=icfg["id"],
            pos=pos,
            vel=vel,
            heading=heading,
            speed=speed,
            max_turn_rate=float(icfg.get("max_turn_rate", 1.0)),
            endurance=float(icfg.get("endurance", 120.0)),
            effector_type=icfg.get("effector_type", "kinetic"),
            t_spawned=0.0,
        )
        world.spawn_interceptor(iv)

    return world
