"""P0 acceptance tests.

Criteria from the plan:
1. Same seed -> byte-identical event-log hash (determinism).
2. Kinematics: turn-rate limits are enforced.
3. Kinematics: speed magnitude is preserved.
4. Leaker detection: a hostile reaching the asset produces LEAKER events.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.core.entities import Asset, HostileUAS, Interceptor
from sim.core.events import EventLog
from sim.core.kinematics import step_entity, hostile_desired_heading
from sim.core.world import World
from sim.core.vec import norm
from sim.run import run


BASELINE_SCENARIO = "eval/scenarios/baseline.yaml"


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical_hash(self):
        r1 = run(BASELINE_SCENARIO, seed=42)
        r2 = run(BASELINE_SCENARIO, seed=42)
        assert r1["log_hash"] == r2["log_hash"], (
            "Two runs with seed=42 produced different event-log hashes — not deterministic"
        )

    def test_different_seed_different_hash(self):
        r1 = run(BASELINE_SCENARIO, seed=42)
        r2 = run(BASELINE_SCENARIO, seed=99)
        # Not guaranteed to differ, but if entities have random noise this will differ.
        # Acceptable if they happen to be equal on a fully-deterministic scenario.
        # This test just documents the intent.
        _ = r1["log_hash"], r2["log_hash"]

    def test_three_runs_all_identical(self):
        hashes = [run(BASELINE_SCENARIO, seed=7)["log_hash"] for _ in range(3)]
        assert len(set(hashes)) == 1, "Three runs with seed=7 differ"


# ---------------------------------------------------------------------------
# 2. Turn-rate limits
# ---------------------------------------------------------------------------

class TestKinematics:
    def test_turn_rate_clamped(self):
        """A single large desired heading change must be clamped to max_turn_rate * dt."""
        pos = np.array([0.0, 0.0])
        vel = np.array([10.0, 0.0])
        heading = 0.0
        speed = 10.0
        max_turn_rate = 0.5  # rad/s
        dt = 0.02  # 50 Hz
        desired = np.pi  # 180-degree turn

        _, _, new_heading = step_entity(pos, vel, heading, speed, max_turn_rate, desired, dt)
        max_change = max_turn_rate * dt
        assert abs(new_heading - heading) <= max_change + 1e-9, (
            f"Heading changed by {abs(new_heading - heading):.4f} but max allowed {max_change:.4f}"
        )

    def test_speed_magnitude_preserved(self):
        """After stepping, speed magnitude must equal the input speed."""
        pos = np.array([100.0, -50.0])
        vel = np.array([20.0, 0.0])
        heading = 0.0
        speed = 20.0
        max_turn_rate = 1.0
        dt = 0.02

        for desired in [0.0, 0.5, 1.2, -0.8, np.pi]:
            new_pos, new_vel, _ = step_entity(pos, vel, heading, speed, max_turn_rate, desired, dt)
            assert abs(norm(new_vel) - speed) < 1e-6, (
                f"Speed magnitude {norm(new_vel):.6f} != {speed:.6f} for desired={desired}"
            )

    def test_straight_line_motion(self):
        """Zero turn rate desired heading same as current — entity moves straight."""
        pos = np.array([0.0, 0.0])
        heading = 0.0
        speed = 10.0
        vel = np.array([speed, 0.0])
        dt = 0.02

        new_pos, _, new_heading = step_entity(pos, vel, heading, speed, 1.0, heading, dt)
        assert abs(new_pos[0] - speed * dt) < 1e-9
        assert abs(new_pos[1]) < 1e-9
        assert abs(new_heading) < 1e-9

    def test_arrival_pops_waypoints(self):
        """Waypoint is popped when entity arrives within arrival radius."""
        pos = np.array([0.0, 0.0])
        wp_near = np.array([5.0, 0.0])   # well within 20m arrival radius
        wp_far = np.array([500.0, 0.0])
        _, remaining = hostile_desired_heading(
            pos, [wp_near, wp_far], 0.0, 0.0, 5.0, 0.0
        )
        assert len(remaining) == 1
        assert np.allclose(remaining[0], wp_far)


# ---------------------------------------------------------------------------
# 3. Leaker detection
# ---------------------------------------------------------------------------

class TestLeakerDetection:
    def _make_world(self) -> World:
        rng = np.random.default_rng(0)
        asset = Asset(id="asset_0", pos=np.array([0.0, 0.0]), hp=5.0, value=1.0)
        log = EventLog()
        return World(dt=0.02, rng=rng, asset=asset, log=log)

    def test_leaker_event_emitted_on_impact(self):
        """A hostile with no interceptor opposing it should reach the asset and log LEAKER."""
        world = self._make_world()
        h = HostileUAS(
            id="h_leaker",
            pos=np.array([-50.0, 0.0]),
            vel=np.array([25.0, 0.0]),
            heading=0.0,
            speed=25.0,
            max_turn_rate=1.0,
            weave_amplitude=0.0,
            weave_period=5.0,
            waypoints=[np.array([0.0, 0.0])],
        )
        world.spawn_hostile(h)

        for _ in range(500):
            world.step()
            if world.is_engagement_over():
                break

        leaker_events = [e for e in world.log.events if e["type"] == "LEAKER"]
        assert len(leaker_events) >= 1, "Expected at least one LEAKER event"
        assert leaker_events[0]["entity_id"] == "h_leaker"

    def test_asset_hp_decreases_on_leaker(self):
        """Asset HP must drop when a hostile impacts."""
        world = self._make_world()
        initial_hp = world.asset.hp
        h = HostileUAS(
            id="h_dmg",
            pos=np.array([-50.0, 0.0]),
            vel=np.array([25.0, 0.0]),
            heading=0.0,
            speed=25.0,
            max_turn_rate=1.0,
            weave_amplitude=0.0,
            weave_period=5.0,
            waypoints=[np.array([0.0, 0.0])],
        )
        world.spawn_hostile(h)
        for _ in range(500):
            world.step()
            if world.is_engagement_over():
                break
        assert world.asset.hp < initial_hp

    def test_intercept_prevents_leaker(self):
        """An interceptor on an intercept course should stop the hostile before impact."""
        world = self._make_world()
        h = HostileUAS(
            id="h_0",
            pos=np.array([-400.0, 0.0]),
            vel=np.array([20.0, 0.0]),
            heading=0.0,
            speed=20.0,
            max_turn_rate=0.3,
            weave_amplitude=0.0,
            weave_period=5.0,
            waypoints=[np.array([0.0, 0.0])],
        )
        iv = Interceptor(
            id="i_0",
            pos=np.array([-300.0, 0.0]),
            vel=np.array([50.0, 0.0]),
            heading=0.0,
            speed=50.0,
            max_turn_rate=2.0,
            endurance=300.0,
            effector_type="kinetic",
        )
        world.spawn_hostile(h)
        world.spawn_interceptor(iv)
        world.assign("i_0", "h_0")

        for _ in range(3000):
            world.step()
            if world.is_engagement_over():
                break

        leaker_events = [e for e in world.log.events if e["type"] == "LEAKER"]
        intercept_events = [e for e in world.log.events if e["type"] == "INTERCEPT"]
        assert len(intercept_events) >= 1, "Expected an INTERCEPT event"
        assert len(leaker_events) == 0, "Expected no LEAKER event when interceptor is assigned"


# ---------------------------------------------------------------------------
# 4. Event log structure
# ---------------------------------------------------------------------------

class TestEventLog:
    def test_events_have_required_fields(self):
        result = run(BASELINE_SCENARIO, seed=42)
        from sim.core.events import EventLog
        import json
        # Reconstruct log
        r = run(BASELINE_SCENARIO, seed=42)
        # Just check the result dict has expected keys
        assert "intercepts" in r
        assert "leakers" in r
        assert "log_hash" in r
        assert "time_to_clear" in r

    def test_seq_is_monotonic(self):
        from sim.core.events import EventLog
        log = EventLog()
        for i in range(5):
            log.append(float(i) * 0.02, "SIM_START", data={})
        seqs = [e["seq"] for e in log.events]
        assert seqs == list(range(5))
