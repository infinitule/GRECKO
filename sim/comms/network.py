"""CommsNetwork: the single routing layer for all inter-node traffic.

Every inter-interceptor and interceptor-to-C2 message goes through
send()/broadcast() here. Modules never exchange state directly — this is the
grep-able isolation guarantee (CI test asserts no cross-layer imports).

Also hosts the PartitionTracker: a connected-components service that publishes
a MeshTopology message (per /proto/comms.schema.json) each tick; the C2
console visualises it and the allocator receives it as its adjacency input.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

from sim.comms.links import CommsConfig, LinkModel


@dataclasses.dataclass
class Envelope:
    """Mirrors CommsEnvelope in /proto/comms.schema.json."""
    msg_id: int
    t_sent: float
    t_delivered: Optional[float]
    src: str
    dst: str
    kind: str
    payload: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MeshTopology:
    """Mirrors MeshTopology in /proto/comms.schema.json."""
    t: float
    edges: List[Tuple[str, str]]
    partitions: List[List[str]]
    partition_count: int

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "edges": [list(e) for e in self.edges],
            "partitions": self.partitions,
            "partition_count": self.partition_count,
        }


class CommsNetwork:
    def __init__(self, config: CommsConfig, rng: np.random.Generator):
        self.link_model = LinkModel(config)
        self.rng = rng
        self._positions: Dict[str, np.ndarray] = {}
        self._in_flight: List[Envelope] = []
        self._next_msg_id = 0

    # -- node registry -------------------------------------------------------

    def set_position(self, node_id: str, pos: np.ndarray) -> None:
        self._positions[node_id] = np.asarray(pos, dtype=float)

    def remove_node(self, node_id: str) -> None:
        self._positions.pop(node_id, None)

    @property
    def nodes(self) -> List[str]:
        return sorted(self._positions)

    # -- messaging -----------------------------------------------------------

    def send(self, t: float, src: str, dst: str, kind: str, payload: dict) -> Optional[int]:
        """Queue a message. Returns msg_id if it entered the channel, None if
        the link is down or the message dropped. The SENDER cannot distinguish
        a drop from a delivery — no delivery receipt exists at this layer."""
        if src not in self._positions or dst not in self._positions:
            return None
        if not self.link_model.link_up(t, src, self._positions[src], dst, self._positions[dst]):
            return None
        if self.link_model.config.base_drop_rate > 0 and \
                self.rng.random() < self.link_model.config.base_drop_rate:
            return None
        env = Envelope(
            msg_id=self._next_msg_id, t_sent=t, t_delivered=None,
            src=src, dst=dst, kind=kind, payload=payload,
        )
        self._next_msg_id += 1
        self._in_flight.append(env)
        return env.msg_id

    def broadcast(self, t: float, src: str, kind: str, payload: dict) -> int:
        """Send to every current neighbour. Returns number queued."""
        n = 0
        for dst in self.nodes:
            if dst != src and self.send(t, src, dst, kind, payload) is not None:
                n += 1
        return n

    def deliver(self, t: float) -> List[Envelope]:
        """Pop and return all messages whose latency has elapsed."""
        due, pending = [], []
        for env in self._in_flight:
            if t >= env.t_sent + self.link_model.config.latency:
                env.t_delivered = t
                due.append(env)
            else:
                pending.append(env)
        self._in_flight = pending
        return due

    # -- topology / partitions ------------------------------------------------

    def topology(self, t: float) -> MeshTopology:
        ids = self.nodes
        edges: List[Tuple[str, str]] = []
        adj: Dict[str, List[str]] = {i: [] for i in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if self.link_model.link_up(t, a, self._positions[a], b, self._positions[b]):
                    edges.append((a, b))
                    adj[a].append(b)
                    adj[b].append(a)
        partitions = self._components(ids, adj)
        return MeshTopology(t=t, edges=edges, partitions=partitions,
                            partition_count=len(partitions))

    @staticmethod
    def _components(ids: List[str], adj: Dict[str, List[str]]) -> List[List[str]]:
        seen, comps = set(), []
        for start in ids:
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(sorted(comp))
        return comps

    def adjacency(self, t: float) -> Dict[str, List[str]]:
        """Adjacency structure handed to allocators each round (PA input)."""
        topo = self.topology(t)
        adj: Dict[str, List[str]] = {i: [] for i in self.nodes}
        for a, b in topo.edges:
            adj[a].append(b)
            adj[b].append(a)
        return adj
