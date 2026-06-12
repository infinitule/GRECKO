"""BridgeScenario — full-stack simulation with C2 authorization interlock.

Wires all completed pillars:
  P0  World kernel (entities, kinematics, event log)
  P1  SensorMesh (imperfect, POSG-correct)
  P2  Tracker (Kalman, two-pass GNN, lifecycle)
  P3  RuleClassifier + FeatureExtractor
  PC  CommsNetwork (degradable link model)
  PE  Effector catalogue (parameter sets only)
  PA  EconomicMDP allocator (magazine-constrained, λ-rationing)
  PB  IntentPredictor (swarm intent, value multiplier wiring)
  PV  C2State interlock — no world.assign() without can_engage() == True
  PL  auto_authorize + policy spawn-hook for league fitness evaluation

The interlock is the sole path through which terminal engagements enter the
World. grep-verifiable: only one call site of world.assign() below, guarded
by c2_state.can_engage().
"""
from __future__ import annotations

import math
import pathlib
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from league.policy import SwarmPolicy

import numpy as np

from sim.alloc.economic_mdp import EconomicMDP
from sim.alloc.interface import AllocInput, InterceptorState
from sim.alloc.types import Assignment, MagazineState
from sim.bridge.state import C2State
from sim.classify.classifier import RuleClassifier, ThreatAssessment
from sim.classify.features import FeatureVector, extract
from sim.comms.links import CommsConfig
from sim.comms.network import CommsNetwork
from sim.core.entities import Asset, HostileUAS, Interceptor
from sim.core.events import EventLog
from sim.core.world import World
from sim.effectors.catalogue import CATALOGUE
from sim.fusion.tracker import Tracker, TrackMessage
from sim.sensing.sensors import EOIRSensor, RadarSensor, SensorMesh

# Optional intent predictor (PB) — load checkpoint if available
try:
    from learn.intent.model import IntentMLP
    from learn.intent.predictor import IntentPredictor
    _INTENT_AVAILABLE = True
except ImportError:
    _INTENT_AVAILABLE = False

_CHECKPOINT = pathlib.Path(__file__).parent.parent.parent / "learn/checkpoints/intent_mlp.npz"

ASSET_POS = np.array([0.0, 0.0])
DT = 0.02          # 50 Hz physics
ALLOC_INTERVAL = 25  # re-run allocator every 25 ticks (0.5 s)
BROADCAST_INTERVAL = 5  # ticks between full state broadcasts (0.1 s)


def _default_magazine() -> MagazineState:
    return MagazineState({
        "kinetic_interceptor": 8,
        "net_capture_drone": 12,
        "ew_soft_kill": 20,
        "collision_drone": 10,
    })


def default_sensor_mesh() -> SensorMesh:
    """The bridge demo's nominal sensor layout. Exposed so the S2R harness
    can build a perturbed copy without duplicating the configuration."""
    return SensorMesh([
        RadarSensor(
            "R0", np.array([-400.0, 0.0]),
            max_range=3000.0, pd_max=0.92, pd_knee_frac=0.6,
            noise_std=np.array([20.0, 20.0]),
            update_rate=2.0, false_alarm_rate=0.3,
        ),
        RadarSensor(
            "R1", np.array([400.0, 0.0]),
            max_range=3000.0, pd_max=0.90, pd_knee_frac=0.6,
            noise_std=np.array([22.0, 22.0]),
            update_rate=2.0, false_alarm_rate=0.3,
        ),
        EOIRSensor(
            "E0", np.array([0.0, 300.0]),
            bearing_only=False,
            fov_width=2 * math.pi,  # omnidirectional for demo
            max_range=1200.0, pd_max=0.95, pd_knee_frac=0.7,
            noise_std=np.array([6.0, 6.0]),
            update_rate=5.0, false_alarm_rate=0.1,
        ),
    ])


class BridgeScenario:
    """Full-pipeline scenario for C2 bridge consumption.

    Call `tick()` repeatedly.  Call `apply_command(cmd)` to inject C2 events.
    Call `get_state()` for the serialisable broadcast payload.

    The HOTL interlock: before any world.assign() call for an ASSIGN decision,
    `c2_state.can_engage(track_id)` is checked. If False, the assignment is
    logged in the audit trail but NOT forwarded to the world.
    """

    def __init__(
        self,
        seed: int = 42,
        auto_authorize: bool = False,
        policy: "Optional[SwarmPolicy]" = None,
        sensor_mesh: Optional[SensorMesh] = None,
        comms_cfg: Optional[CommsConfig] = None,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.c2_state = C2State()
        self._auto_authorize = auto_authorize
        self._policy = policy

        # Event log + world
        log = EventLog()
        asset = Asset(id="ASSET0", pos=ASSET_POS.copy(), hp=10.0, value=1_000_000.0)
        self.world = World(dt=DT, rng=self.rng, asset=asset, log=log)

        # Spawn entities
        self._spawn_entities()

        # Sensing mesh (injectable for S2R perturbation studies)
        self.sensor_mesh = sensor_mesh if sensor_mesh is not None else default_sensor_mesh()

        # Tracker
        self.tracker = Tracker()

        # Classifier
        self.classifier = RuleClassifier()

        # Comms network (injectable for S2R perturbation studies)
        if comms_cfg is None:
            comms_cfg = CommsConfig(comm_radius=2000.0, base_drop_rate=0.0)
        self.comms = CommsNetwork(comms_cfg, self.rng)
        self._register_comms_nodes()

        # Effectors
        self.effector_catalogue = dict(CATALOGUE)

        # Allocator
        self.allocator = EconomicMDP()
        self.magazine = _default_magazine()

        # Intent predictor (PB)
        self.intent_predictor: Optional[IntentPredictor] = None
        if _INTENT_AVAILABLE and _CHECKPOINT.exists():
            try:
                model = IntentMLP()
                model.load(str(_CHECKPOINT))
                self.intent_predictor = IntentPredictor(model, ASSET_POS)
            except Exception:
                pass  # graceful degradation: run without intent model

        # Pipeline state
        self._track_history: Dict[str, List[TrackMessage]] = defaultdict(list)
        self._track_entity_map: Dict[str, str] = {}  # track_id → entity_id (bridge layer)
        self._last_assessments: List[ThreatAssessment] = []
        self._last_assignments: List[Assignment] = []
        self._last_intents: list = []
        self._tick_count: int = 0

        # Seed the event log
        self.world.log.append(0.0, "SIM_START",
                              data={"seed": seed, "scenario": "bridge_demo"})

    # ------------------------------------------------------------------ #
    # Entity setup                                                         #
    # ------------------------------------------------------------------ #

    def _spawn_entities(self) -> None:
        """Spawn entities. If a policy is provided, use it; else use the default demo formation."""
        if self._policy is not None:
            self._spawn_from_policy(self._policy)
            return
        self._spawn_default_entities()

    def _spawn_from_policy(self, policy: "SwarmPolicy") -> None:
        """Spawn HostileUAS from a SwarmPolicy's initial conditions + fixed interceptors."""
        rng = self.rng
        agent_idx = 0

        def _spawn_group(n: int, angle: float, r0: float, speed: float,
                         spread: float, weave_amp: float, label: str) -> None:
            nonlocal agent_idx
            base = np.array([math.cos(angle), math.sin(angle)]) * r0
            for _ in range(n):
                offset = rng.normal(0.0, spread, 2)
                pos = base + offset
                aim = -pos / max(float(np.linalg.norm(pos)), 1.0)
                vel = aim * speed + rng.normal(0.0, 0.5, 2)
                h = HostileUAS(
                    id=f"H{agent_idx:02d}",
                    pos=pos.copy(),
                    vel=vel.copy(),
                    heading=float(math.atan2(vel[1], vel[0])),
                    speed=speed,
                    max_turn_rate=0.3,
                    weave_amplitude=weave_amp,
                    weave_period=8.0,
                    waypoints=[ASSET_POS.copy()],
                )
                self.world.spawn_hostile(h)
                agent_idx += 1

        if policy.n_main > 0:
            _spawn_group(policy.n_main, policy.main_angle, policy.main_range,
                         policy.main_speed, policy.main_spread, policy.weave_amp,
                         "main_axis")
        if policy.n_feint > 0:
            _spawn_group(policy.n_feint, policy.feint_angle, policy.feint_range,
                         policy.feint_speed, policy.feint_spread, policy.weave_amp * 2,
                         "feint")
        if policy.n_screen > 0:
            _spawn_group(policy.n_screen, policy.main_angle,
                         policy.main_range * 0.7, policy.main_speed * 0.9,
                         policy.main_spread * 1.5, policy.weave_amp * 3, "screen")

        self._spawn_interceptors()

    def _spawn_interceptors(self) -> None:
        """Spawn the fixed blue-team interceptors."""
        iv_configs = [
            ("IV0", np.array([250.0, 80.0]),  60.0, "kinetic_interceptor"),
            ("IV1", np.array([-250.0, 80.0]), 45.0, "net_capture_drone"),
            ("IV2", np.array([0.0, 150.0]),   45.0, "net_capture_drone"),
        ]
        for iv_id, pos, speed, eff_type in iv_configs:
            iv = Interceptor(
                id=iv_id,
                pos=pos.copy(),
                vel=np.zeros(2),
                heading=math.pi / 2,
                speed=speed,
                max_turn_rate=1.5,
                endurance=240.0,
                effector_type=eff_type,
            )
            self.world.spawn_interceptor(iv)

    def _spawn_default_entities(self) -> None:
        """8 hostile UAS + 3 interceptors in a mixed feint+main formation."""
        rng = self.rng

        # Main axis (5 agents from north-north-east, 1200–1700m away)
        main_angles = [-0.4, -0.2, 0.0, 0.2, 0.4]
        for i, ang in enumerate(main_angles):
            r = rng.uniform(1200.0, 1600.0)
            px = r * math.sin(ang)
            py = r * math.cos(ang)
            speed = rng.uniform(22.0, 30.0)
            # velocity pointing roughly toward origin
            vx = -speed * math.sin(ang) + rng.normal(0, 1.0)
            vy = -speed * math.cos(ang) + rng.normal(0, 1.0)
            h = HostileUAS(
                id=f"H{i:02d}",
                pos=np.array([px, py]),
                vel=np.array([vx, vy]),
                heading=math.atan2(vy, vx),
                speed=speed,
                max_turn_rate=0.3,
                weave_amplitude=0.05,
                weave_period=8.0,
                waypoints=[ASSET_POS.copy()],
            )
            self.world.spawn_hostile(h)

        # Feint group (3 agents from north-west, slightly off-axis)
        feint_angles = [1.0, 1.2, 1.4]
        for i, ang in enumerate(feint_angles):
            r = rng.uniform(1100.0, 1400.0)
            px = r * math.sin(ang)
            py = r * math.cos(ang)
            speed = rng.uniform(16.0, 20.0)  # slower: feint signature
            vx = -speed * math.sin(ang) + rng.normal(0, 0.5)
            vy = -speed * math.cos(ang) + rng.normal(0, 0.5)
            h = HostileUAS(
                id=f"H{5 + i:02d}",
                pos=np.array([px, py]),
                vel=np.array([vx, vy]),
                heading=math.atan2(vy, vx),
                speed=speed,
                max_turn_rate=0.4,
                weave_amplitude=0.12,
                weave_period=6.0,
                waypoints=[ASSET_POS.copy()],
            )
            self.world.spawn_hostile(h)

        self._spawn_interceptors()

    def _register_comms_nodes(self) -> None:
        for iv in self.world.interceptors.values():
            self.comms.set_position(iv.id, iv.pos)
        self.comms.set_position("C2", ASSET_POS.copy())

    # ------------------------------------------------------------------ #
    # Main tick                                                            #
    # ------------------------------------------------------------------ #

    def tick(self) -> dict:
        """Run one physics step (dt=0.02 s) with the full pipeline.

        Returns a serialisable broadcast payload regardless of whether this is
        a broadcast tick — callers may throttle how often they send it.
        """
        t = self.world.t

        # Update comms node positions (interceptors move)
        for iv in self.world.interceptors.values():
            if iv.alive:
                self.comms.set_position(iv.id, iv.pos)

        # ---- Sensing ----
        alive_positions = [h.pos for h in self.world.hostiles.values() if h.alive]
        reports = self.sensor_mesh.scan_all(t, alive_positions, self.rng)

        # ---- Fusion ----
        tracks = self.tracker.update(t, reports)
        for trk in tracks:
            hist = self._track_history[trk.track_id]
            hist.append(trk)
            if len(hist) > 30:
                hist.pop(0)

        # ---- Track-entity mapping (bridge layer only) ----
        self._update_track_entity_map(tracks)

        # ---- Classification ----
        confirmed = [trk for trk in tracks if trk.status in ("confirmed", "coasted")]
        assessments: List[ThreatAssessment] = []
        for trk in confirmed:
            hist = self._track_history[trk.track_id]
            fv = extract(trk, hist, ASSET_POS)
            is_friendly = trk.track_id in self.c2_state.friendly_marked
            assess = self.classifier.classify(fv, is_friendly, ASSET_POS, trk.state)
            assessments.append(assess)
        assessments.sort(key=lambda a: a.priority_score, reverse=True)

        # ---- League auto-authorisation (PL mode only) ----
        # Auto-authorize all non-friendly confirmed tracks so Blue fights optimally
        # for fitness evaluation. This is NOT part of the C2 interlock — it is an
        # evaluation convenience used by the league episode runner only.
        if self._auto_authorize:
            for assess in assessments:
                if assess.label != "friendly":
                    if assess.track_id not in self.c2_state.authorized_tracks:
                        self.c2_state.authorized_tracks.add(assess.track_id)

        # ---- Intent prediction (PB) ----
        intents = []
        if confirmed and self.intent_predictor is not None:
            positions = np.array([trk.state[:2] for trk in confirmed])
            velocities = np.array([trk.state[2:4] for trk in confirmed])
            track_ids = [trk.track_id for trk in confirmed]
            try:
                intents = self.intent_predictor.predict(t, track_ids, positions, velocities)
                self._apply_intent_multipliers(assessments, intents)
            except Exception:
                pass  # intent model failure is non-fatal

        # ---- Allocation (every ALLOC_INTERVAL ticks) ----
        if self._tick_count % ALLOC_INTERVAL == 0 and assessments:
            adjacency = self.comms.adjacency(t)
            iv_states = [
                InterceptorState(
                    interceptor_id=iv.id,
                    pos=iv.pos.copy(),
                    effector_type=iv.effector_type,
                    endurance_s=iv.endurance,
                    speed_mps=iv.speed,
                )
                for iv in self.world.interceptors.values()
                if iv.alive
            ]
            alloc_in = AllocInput(
                t=t,
                interceptors=iv_states,
                assessments=assessments,
                magazine=self.magazine.copy(),
                effector_catalogue=self.effector_catalogue,
                adjacency=adjacency,
                asset_pos=ASSET_POS,
                asset_value=self.world.asset.value,
                lambda_cost=self.c2_state.lambda_cost,
            )
            raw_assignments = self.allocator.allocate(alloc_in)

            # ---- C2 INTERLOCK (the architectural HOTL invariant) ----
            for a in raw_assignments:
                if a.action == "ASSIGN" and a.track_id:
                    entity_id = self._track_entity_map.get(a.track_id)
                    if entity_id is None:
                        # track not yet resolved to an entity — defer
                        self.c2_state.log_hold_pending(t, a.interceptor_id, a.track_id)
                    elif self.c2_state.weapons_hold:
                        self.c2_state.log_weapons_hold_block(t, a.interceptor_id, a.track_id)
                    elif a.track_id in self.c2_state.held_tracks:
                        self.c2_state.log_operator_hold_block(t, a.interceptor_id, a.track_id)
                    elif self.c2_state.can_engage(a.track_id):
                        # Only path to world.assign() for ASSIGN decisions.
                        # Skip re-assignment to the same entity (allocator re-runs
                        # every ALLOC_INTERVAL but the assignment hasn't changed).
                        already = self.world._interceptor_assignments.get(a.interceptor_id)
                        if already != entity_id:
                            self.world.assign(a.interceptor_id, entity_id)
                            self.c2_state.log_engage(t, a.interceptor_id, a.track_id)
                            self.magazine.expend(a.effector_id or "")
                    else:
                        self.c2_state.log_hold_pending(t, a.interceptor_id, a.track_id)

            self._last_assignments = raw_assignments
        else:
            # Keep prior assignments visible in the UI between alloc rounds
            pass

        self._last_assessments = assessments
        self._last_intents = intents
        self._tick_count += 1

        # ---- Physics step ----
        self.world.step()

        return self._serialize_state(tracks, assessments, self._last_assignments, intents)

    # ------------------------------------------------------------------ #
    # C2 command injection                                                 #
    # ------------------------------------------------------------------ #

    def apply_command(self, cmd: dict) -> bool:
        return self.c2_state.apply_command(cmd, sim_t=self.world.t)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _update_track_entity_map(self, tracks: List[TrackMessage]) -> None:
        """Bridge layer: map track_ids to entity_ids by nearest position.

        This is a greedy nearest-neighbour match — acceptable for the bridge
        demo layer. The POSG invariant (fusion never reads truth) is not
        violated: this mapping runs in the bridge layer, which is allowed to
        see both estimates and truth for operational resolution.
        """
        alive = {h.id: h.pos.copy() for h in self.world.hostiles.values() if h.alive}

        # Remove mappings to dead entities
        self._track_entity_map = {
            tid: eid for tid, eid in self._track_entity_map.items()
            if eid in alive
        }

        # Remove mappings for dropped tracks
        live_track_ids = {trk.track_id for trk in tracks}
        self._track_entity_map = {
            tid: eid for tid, eid in self._track_entity_map.items()
            if tid in live_track_ids
        }

        established = [trk for trk in tracks if trk.status in ("confirmed", "coasted")]
        unmapped = [trk for trk in established if trk.track_id not in self._track_entity_map]
        used = set(self._track_entity_map.values())
        free = {eid: pos for eid, pos in alive.items() if eid not in used}

        for trk in unmapped:
            if not free:
                break
            est = trk.state[:2]
            best = min(free, key=lambda eid: float(np.linalg.norm(est - free[eid])))
            self._track_entity_map[trk.track_id] = best
            del free[best]

    def _apply_intent_multipliers(
        self,
        assessments: List[ThreatAssessment],
        intents: list,
    ) -> None:
        """Multiply each track's priority_score by its cluster's value_multiplier."""
        track_mult: Dict[str, float] = {}
        for intent in intents:
            for tid in intent.member_track_ids:
                track_mult[tid] = intent.value_multiplier
        for a in assessments:
            m = track_mult.get(a.track_id, 1.0)
            a.priority_score = float(np.clip(a.priority_score * m, 0.0, 1e6))

    # ------------------------------------------------------------------ #
    # State serialisation                                                  #
    # ------------------------------------------------------------------ #

    def _serialize_state(
        self,
        tracks: List[TrackMessage],
        assessments: List[ThreatAssessment],
        assignments: List[Assignment],
        intents: list,
    ) -> dict:
        t = self.world.t

        # Tracks
        track_list = []
        for trk in tracks:
            cov_diag = [float(trk.covariance[0, 0]), float(trk.covariance[1, 1])]
            track_list.append({
                "track_id": trk.track_id,
                "status": trk.status,
                "pos": trk.state[:2].tolist(),
                "vel": trk.state[2:4].tolist(),
                "cov_diag": cov_diag,
                "quality": round(trk.quality, 3),
                "age": round(trk.age, 2),
                "n_updates": trk.n_updates,
            })

        # Assessments keyed by track_id
        assess_map = {}
        for a in assessments:
            pos_list = a.features.pos.tolist() if a.features.pos is not None else [0, 0]
            assess_map[a.track_id] = {
                "label": a.label,
                "confidence": round(a.confidence, 3),
                "priority_score": round(a.priority_score, 5),
                "why": a.why,
            }

        # Assignments
        assign_list = []
        for a in assignments:
            assign_list.append({
                "interceptor_id": a.interceptor_id,
                "action": a.action,
                "track_id": a.track_id,
                "effector_id": a.effector_id,
                "bid_value": round(a.provenance.bid_value, 5),
                "hold_reason": a.provenance.hold_reason or "",
            })

        # Intents
        intent_list = []
        for intent in intents:
            intent_list.append({
                "cluster_id": intent.cluster_id,
                "member_track_ids": intent.member_track_ids,
                "intent_distribution": {
                    k: round(v, 3) for k, v in intent.intent_distribution.items()
                },
                "dominant_intent": intent.dominant_intent(),
                "value_multiplier": round(intent.value_multiplier, 3),
                "forecast_centroids": intent.forecast_centroids.tolist(),
            })

        # Interceptors
        interceptor_list = []
        for iv in self.world.interceptors.values():
            interceptor_list.append({
                "id": iv.id,
                "pos": iv.pos.tolist(),
                "vel": iv.vel.tolist(),
                "heading": round(iv.heading, 4),
                "alive": iv.alive,
                "endurance": round(iv.endurance, 1),
                "effector_type": iv.effector_type,
                "assigned_track": self._entity_to_track(iv.id),
            })

        # Truth entities (debug; gated in production UI)
        truth_list = []
        for h in self.world.hostiles.values():
            if h.alive:
                truth_list.append({
                    "id": h.id,
                    "pos": h.pos.tolist(),
                    "vel": h.vel.tolist(),
                    "alive": h.alive,
                })

        # Comms topology
        topo = self.comms.topology(t)

        return {
            "t": round(t, 4),
            "tracks": track_list,
            "assessments": assess_map,
            "assignments": assign_list,
            "intents": intent_list,
            "interceptors": interceptor_list,
            "truth_entities": truth_list,
            "asset": {
                "id": self.world.asset.id,
                "pos": self.world.asset.pos.tolist(),
                "hp": self.world.asset.hp,
                "alive": self.world.asset.alive,
            },
            "mesh_topology": {
                "edges": [list(e) for e in topo.edges],
                "partitions": topo.partitions,
                "partition_count": topo.partition_count,
            },
            "magazine": dict(self.magazine.rounds),
            "weapons_hold": self.c2_state.weapons_hold,
            "authorized_tracks": sorted(self.c2_state.authorized_tracks),
            "held_tracks": sorted(self.c2_state.held_tracks),
            "friendly_tracks": sorted(self.c2_state.friendly_marked),
            "lambda_cost": round(self.c2_state.lambda_cost, 3),
            "audit_trail": self.c2_state.recent_audit(n=80),
        }

    def _entity_to_track(self, interceptor_id: str) -> Optional[str]:
        """Reverse-look up which track_id an interceptor is currently assigned to."""
        entity_target = self.world._interceptor_assignments.get(interceptor_id)
        if entity_target is None:
            return None
        # find the track that maps to this entity
        for tid, eid in self._track_entity_map.items():
            if eid == entity_target:
                return tid
        return None

    # ------------------------------------------------------------------ #
    # Determinism                                                          #
    # ------------------------------------------------------------------ #

    def log_hash(self) -> str:
        return self.world.log_hash()
