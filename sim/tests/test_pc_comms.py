"""PC acceptance tests.

Criteria from the plan:
1. Messages never cross a denial zone.
2. Partition count matches a hand-computed scenario.
3. Allocation quality degrades monotonically as comm radius shrinks.
4. Degradation scenarios load from config, not code.
5. Latency, drop rate, intermittency, full partition behave as configured.
6. Isolation: /sim/comms imports no other sim layer (grep invariant).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

from sim.comms.links import CommsConfig, DenialZone, LinkModel
from sim.comms.network import CommsNetwork
from sim.comms.sweep import run_sweep


def _net(config: CommsConfig, seed: int = 0) -> CommsNetwork:
    return CommsNetwork(config, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# 1. Denial zones
# ---------------------------------------------------------------------------

class TestDenialZones:
    def test_no_delivery_across_zone(self):
        """Two nodes in range of each other but with a denial zone between
        them: no message is ever delivered."""
        zone = DenialZone(center=(0.0, 0.0), radius=200.0)
        net = _net(CommsConfig(comm_radius=5000.0, denial_zones=[zone], latency=0.0))
        net.set_position("a", np.array([-500.0, 0.0]))
        net.set_position("b", np.array([500.0, 0.0]))   # segment crosses the zone
        delivered = []
        for i in range(200):
            t = i * 0.02
            net.send(t, "a", "b", "bid", {"v": i})
            delivered.extend(net.deliver(t))
        assert delivered == [], f"{len(delivered)} messages crossed a denial zone"

    def test_no_delivery_from_inside_zone(self):
        zone = DenialZone(center=(0.0, 0.0), radius=300.0)
        net = _net(CommsConfig(comm_radius=5000.0, denial_zones=[zone], latency=0.0))
        net.set_position("a", np.array([100.0, 0.0]))   # inside the zone
        net.set_position("b", np.array([900.0, 0.0]))   # outside
        assert net.send(0.0, "a", "b", "bid", {}) is None
        assert net.send(0.0, "b", "a", "bid", {}) is None

    def test_delivery_around_zone(self):
        """A pair whose segment misses the zone communicates normally."""
        zone = DenialZone(center=(0.0, 0.0), radius=200.0)
        net = _net(CommsConfig(comm_radius=5000.0, denial_zones=[zone], latency=0.0))
        net.set_position("a", np.array([-500.0, 600.0]))
        net.set_position("b", np.array([500.0, 600.0]))  # segment passes north of zone
        assert net.send(0.0, "a", "b", "bid", {}) is not None


# ---------------------------------------------------------------------------
# 2. Partition counting — hand-computed scenario
# ---------------------------------------------------------------------------

class TestPartitions:
    def test_hand_computed_partition_count(self):
        """Three clusters: A {a0,a1} at x≈0, B {b0,b1} at x≈2000, C {c0} at
        x≈10000, radius 600. Hand count: within-cluster links only -> 3
        partitions: [a0,a1], [b0,b1], [c0]."""
        net = _net(CommsConfig(comm_radius=600.0))
        net.set_position("a0", np.array([0.0, 0.0]))
        net.set_position("a1", np.array([300.0, 0.0]))
        net.set_position("b0", np.array([2000.0, 0.0]))
        net.set_position("b1", np.array([2300.0, 0.0]))
        net.set_position("c0", np.array([10000.0, 0.0]))
        topo = net.topology(0.0)
        assert topo.partition_count == 3
        assert sorted(map(tuple, topo.partitions)) == [
            ("a0", "a1"), ("b0", "b1"), ("c0",)
        ]

    def test_bridge_node_merges_partitions(self):
        """Adding a node midway between clusters A and B merges them: 2 -> 1
        partitions among {A, B, bridge} (hand-computed)."""
        net = _net(CommsConfig(comm_radius=600.0))
        net.set_position("a0", np.array([0.0, 0.0]))
        net.set_position("b0", np.array([1000.0, 0.0]))
        assert net.topology(0.0).partition_count == 2
        net.set_position("bridge", np.array([500.0, 0.0]))
        assert net.topology(0.0).partition_count == 1

    def test_full_partition_isolates_every_node(self):
        net = _net(CommsConfig(comm_radius=4000.0, full_partition=True))
        for k in range(5):
            net.set_position(f"n{k}", np.array([k * 10.0, 0.0]))
        topo = net.topology(0.0)
        assert topo.partition_count == 5
        assert topo.edges == []

    def test_adjacency_matches_topology_edges(self):
        net = _net(CommsConfig(comm_radius=600.0))
        net.set_position("a", np.array([0.0, 0.0]))
        net.set_position("b", np.array([400.0, 0.0]))
        net.set_position("c", np.array([5000.0, 0.0]))
        adj = net.adjacency(0.0)
        assert adj["a"] == ["b"] and adj["b"] == ["a"] and adj["c"] == []


# ---------------------------------------------------------------------------
# 3. Monotonic degradation sweep
# ---------------------------------------------------------------------------

class TestDegradationSweep:
    def test_quality_monotone_nonincreasing_as_radius_shrinks(self):
        radii = [4000.0, 2000.0, 1000.0, 500.0, 100.0]
        results = run_sweep(radii, n_seeds=30)
        ordered = [results[r] for r in radii]  # descending radius
        for i in range(1, len(ordered)):
            assert ordered[i] <= ordered[i - 1] + 0.02, (
                f"coverage rose as radius shrank: {radii[i-1]}m={ordered[i-1]:.3f} "
                f"-> {radii[i]}m={ordered[i]:.3f}"
            )
        # endpoints must show real degradation, not a flat line
        assert ordered[0] - ordered[-1] > 0.1, (
            f"sweep shows no meaningful degradation: {ordered}"
        )


# ---------------------------------------------------------------------------
# 4. Scenarios are config, not code
# ---------------------------------------------------------------------------

class TestConfigScenarios:
    SCENARIO_DIR = pathlib.Path(__file__).parent.parent.parent / "eval" / "scenarios" / "comms"

    def _load(self, name: str) -> CommsConfig:
        with open(self.SCENARIO_DIR / name) as f:
            return CommsConfig.from_dict(yaml.safe_load(f))

    def test_all_five_degradation_scenarios_load(self):
        for name in ["clear.yaml", "denial_zone.yaml", "range_collapse.yaml",
                     "intermittent.yaml", "full_partition.yaml"]:
            cfg = self._load(name)
            assert isinstance(cfg, CommsConfig)

    def test_loaded_configs_express_their_mode(self):
        assert self._load("denial_zone.yaml").denial_zones
        assert self._load("range_collapse.yaml").comm_radius <= 500.0
        assert self._load("intermittent.yaml").intermittency_period > 0
        assert self._load("full_partition.yaml").full_partition is True


# ---------------------------------------------------------------------------
# 5. Channel behaviour
# ---------------------------------------------------------------------------

class TestChannel:
    def test_latency_delays_delivery(self):
        net = _net(CommsConfig(comm_radius=1000.0, latency=0.1))
        net.set_position("a", np.array([0.0, 0.0]))
        net.set_position("b", np.array([100.0, 0.0]))
        net.send(0.0, "a", "b", "bid", {"x": 1})
        assert net.deliver(0.05) == []                     # before latency
        out = net.deliver(0.10)                            # at latency
        assert len(out) == 1 and out[0].t_delivered == 0.10

    def test_drop_rate_loses_expected_fraction(self):
        net = _net(CommsConfig(comm_radius=1000.0, base_drop_rate=0.3, latency=0.0), seed=5)
        net.set_position("a", np.array([0.0, 0.0]))
        net.set_position("b", np.array([100.0, 0.0]))
        sent = sum(1 for i in range(10_000)
                   if net.send(0.0, "a", "b", "bid", {}) is not None)
        assert abs(sent / 10_000 - 0.7) < 0.02

    def test_intermittency_flaps_links(self):
        cfg = CommsConfig(comm_radius=1000.0, intermittency_period=2.0,
                          intermittency_duty=0.4)
        lm = LinkModel(cfg)
        a, b = np.array([0.0, 0.0]), np.array([100.0, 0.0])
        up = sum(1 for i in range(2000)
                 if lm.link_up(i * 0.01, "a", a, "b", b))
        assert abs(up / 2000 - 0.4) < 0.05, f"duty cycle {up/2000:.2f} != 0.4"

    def test_out_of_range_send_fails(self):
        net = _net(CommsConfig(comm_radius=500.0))
        net.set_position("a", np.array([0.0, 0.0]))
        net.set_position("b", np.array([600.0, 0.0]))
        assert net.send(0.0, "a", "b", "bid", {}) is None

    def test_broadcast_reaches_only_neighbours(self):
        net = _net(CommsConfig(comm_radius=500.0, latency=0.0))
        net.set_position("src", np.array([0.0, 0.0]))
        net.set_position("near", np.array([300.0, 0.0]))
        net.set_position("far", np.array([2000.0, 0.0]))
        assert net.broadcast(0.0, "src", "bid", {}) == 1
        out = net.deliver(0.0)
        assert [e.dst for e in out] == ["near"]


# ---------------------------------------------------------------------------
# 6. Isolation — grep invariant
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_comms_imports_no_other_sim_layer(self):
        """The comms layer is a substrate: it must not import sensing, fusion,
        classify, or core entity state. (Everything routes through it; it
        depends on nothing above it.)"""
        comms_dir = pathlib.Path(__file__).parent.parent / "comms"
        forbidden = ["sim.sensing", "sim.fusion", "sim.classify", "sim.core"]
        for py in comms_dir.glob("*.py"):
            text = py.read_text()
            for mod in forbidden:
                assert mod not in text, f"{py.name} imports {mod} — isolation violation"
