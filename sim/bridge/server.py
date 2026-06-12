"""Async WebSocket C2 bridge server.

Usage:
    python -m sim.bridge.server [--host 0.0.0.0] [--port 8765] [--seed 42]

Clients connect to ws://host:port and receive BroadcastState JSON messages
at BROADCAST_HZ (default 5 Hz).  Clients may send C2 command JSON at any
time.

Protocol (client → server):
  { "type": "AUTHORIZE",    "track_id": "T0042" }
  { "type": "HOLD",         "track_id": "T0042" }
  { "type": "MARK_FRIENDLY","track_id": "T0042" }
  { "type": "LIFT_HOLD",    "track_id": "T0042" }
  { "type": "WEAPONS_HOLD", "active": true|false }
  { "type": "SET_LAMBDA",   "value": 0.10 }
  { "type": "PING" }        → server replies { "type": "PONG", "t": ... }

Protocol (server → client):
  BroadcastState JSON (see sim/bridge/scenario.py _serialize_state)
  plus { "type": "PONG", ... } and { "type": "ERROR", "msg": "..." }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Set

import websockets
import websockets.exceptions

from sim.bridge.scenario import BridgeScenario, DT

BROADCAST_HZ = 5          # UI refresh rate
TICK_HZ = 50              # physics rate
_TICKS_PER_BROADCAST = TICK_HZ // BROADCAST_HZ   # = 10


async def _scenario_loop(
    scenario: BridgeScenario,
    clients: Set,
    paused: asyncio.Event,
    stop: asyncio.Event,
    speed_mult: list,   # [float] — mutable box
) -> None:
    """Run physics at TICK_HZ and broadcast every _TICKS_PER_BROADCAST ticks."""
    tick = 0
    last_state: dict = {}
    while not stop.is_set():
        if paused.is_set():
            await asyncio.sleep(DT)
            continue

        t0 = asyncio.get_event_loop().time()
        state = scenario.tick()
        tick += 1

        if tick % _TICKS_PER_BROADCAST == 0:
            last_state = state
            msg = json.dumps(state)
            dead = set()
            for ws in list(clients):
                try:
                    await ws.send(msg)
                except websockets.exceptions.ConnectionClosed:
                    dead.add(ws)
            clients -= dead

        elapsed = asyncio.get_event_loop().time() - t0
        target = DT / max(speed_mult[0], 0.1)
        sleep_s = max(0.0, target - elapsed)
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

        if scenario.world.is_engagement_over():
            end_msg = json.dumps({"type": "SIM_END", "t": scenario.world.t,
                                  "summary": scenario.world.summary(),
                                  "log_hash": scenario.log_hash()})
            for ws in list(clients):
                try:
                    await ws.send(end_msg)
                except websockets.exceptions.ConnectionClosed:
                    pass
            stop.set()


async def _handle_client(
    ws,
    scenario: BridgeScenario,
    paused: asyncio.Event,
    speed_mult: list,
) -> None:
    """Receive C2 commands from one connected client."""
    try:
        async for raw in ws:
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "ERROR", "msg": "invalid JSON"}))
                continue

            t = cmd.get("type", "")
            if t == "PING":
                await ws.send(json.dumps({"type": "PONG",
                                          "t": scenario.world.t,
                                          "wall": time.time()}))
            elif t == "PAUSE":
                paused.set()
            elif t == "PLAY":
                paused.clear()
            elif t == "SET_SPEED":
                speed_mult[0] = float(cmd.get("value", 1.0))
            else:
                if not scenario.apply_command(cmd):
                    await ws.send(json.dumps({"type": "ERROR",
                                              "msg": f"unknown command: {t}"}))
    except websockets.exceptions.ConnectionClosed:
        pass


async def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    seed: int = 42,
) -> None:
    scenario = BridgeScenario(seed=seed)
    clients: Set = set()
    paused = asyncio.Event()
    stop = asyncio.Event()
    speed_mult = [1.0]

    async def on_connect(ws):
        clients.add(ws)
        try:
            await _handle_client(ws, scenario, paused, speed_mult)
        finally:
            clients.discard(ws)

    loop_task = asyncio.create_task(
        _scenario_loop(scenario, clients, paused, stop, speed_mult)
    )

    async with websockets.serve(on_connect, host, port):
        print(f"AEGISNET C2 bridge listening on ws://{host}:{port}")
        await stop.wait()

    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="AEGISNET C2 WebSocket bridge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    asyncio.run(serve(args.host, args.port, args.seed))


if __name__ == "__main__":
    main()
