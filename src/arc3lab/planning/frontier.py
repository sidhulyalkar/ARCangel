from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from arc3lab.planning.counterfactual import DecisionCandidate
from arc3lab.types import ActionSpec, Scene


@dataclass(slots=True)
class FrontierEdge:
    destinations: Counter[str] = field(default_factory=Counter)
    effects: Counter[str] = field(default_factory=Counter)
    observations: int = 0
    game_over: int = 0
    level: int = 0

    @property
    def dominant_destination(self) -> str | None:
        return self.destinations.most_common(1)[0][0] if self.destinations else None

    @property
    def destination_confidence(self) -> float:
        if not self.destinations or self.observations <= 0:
            return 0.0
        return self.destinations.most_common(1)[0][1] / self.observations

    @property
    def game_over_probability(self) -> float:
        return self.game_over / max(self.observations, 1)


@dataclass(slots=True)
class FrontierNode:
    visits: int = 0
    available_keys: set[str] = field(default_factory=set)
    tested_keys: set[str] = field(default_factory=set)
    edges: dict[str, FrontierEdge] = field(default_factory=dict)

    @property
    def untested_keys(self) -> set[str]:
        return self.available_keys - self.tested_keys


class ExplorationFrontier:
    """Directed state-action graph for systematic, private-safe exploration.

    The graph never assigns game semantics. It remembers which exact visual states have
    been visited, which grounded real actions were available/tested there, and where
    deterministic tested edges led. It can therefore route toward genuinely untested
    state-action frontiers without equating novelty with the goal.
    """

    def __init__(self, *, max_nodes: int = 4096) -> None:
        self.max_nodes = max(128, int(max_nodes))
        self.nodes: dict[str, FrontierNode] = {}
        self.current_signature: str | None = None
        self.last_route: list[str] = []

    @staticmethod
    def state_key(scene: Scene) -> str:
        return f"L{int(scene.level)}:{scene.signature}"

    @staticmethod
    def action_key(scene: Scene, spec: ActionSpec) -> str:
        aid = int(spec.action_id)
        if aid == 6:
            return f"A6:{int(spec.x) if spec.x is not None else -1}:{int(spec.y) if spec.y is not None else -1}"
        return f"A{aid}"

    @classmethod
    def candidate_key(cls, scene: Scene, candidate: DecisionCandidate) -> str | None:
        if candidate.spec is not None:
            return cls.action_key(scene, candidate.spec)
        return None

    def _node(self, signature: str) -> FrontierNode:
        if signature not in self.nodes:
            if len(self.nodes) >= self.max_nodes:
                victims = [
                    (n.visits, sig)
                    for sig, n in self.nodes.items()
                    if sig != self.current_signature and not n.untested_keys
                ]
                if victims:
                    _, victim = min(victims)
                    self.nodes.pop(victim, None)
            self.nodes.setdefault(signature, FrontierNode())
        return self.nodes[signature]

    def observe_state(self, scene: Scene, candidates: list[DecisionCandidate] | None = None) -> None:
        signature = self.state_key(scene)
        node = self._node(signature)
        if self.current_signature != signature or node.visits == 0:
            node.visits += 1
        self.current_signature = signature
        if candidates:
            for c in candidates:
                key = self.candidate_key(scene, c)
                if key is not None:
                    node.available_keys.add(key)

    def observe_transition(
        self,
        before: Scene,
        action: ActionSpec,
        after: Scene,
        *,
        effect: str,
        game_over: bool = False,
        level_completed: bool = False,
    ) -> None:
        before_signature = self.state_key(before)
        after_signature = self.state_key(after)
        before_node = self._node(before_signature)
        after_node = self._node(after_signature)
        after_node.visits += 1
        key = self.action_key(before, action)
        before_node.available_keys.add(key)
        before_node.tested_keys.add(key)
        edge = before_node.edges.setdefault(key, FrontierEdge())
        edge.observations += 1
        edge.destinations[after_signature] += 1
        edge.effects[str(effect)] += 1
        edge.game_over += int(bool(game_over))
        edge.level += int(bool(level_completed))
        self.current_signature = after_signature

    def safe_edge_destination(
        self,
        signature: str,
        key: str,
        *,
        min_support: int = 1,
        min_confidence: float = 0.90,
    ) -> str | None:
        node = self.nodes.get(signature)
        if node is None:
            return None
        edge = node.edges.get(key)
        if edge is None or edge.observations < min_support:
            return None
        if edge.game_over > 0 or edge.destination_confidence < min_confidence:
            return None
        return edge.dominant_destination

    def route_to_frontier(
        self,
        start: str,
        *,
        max_depth: int = 64,
        min_support: int = 1,
        min_confidence: float = 0.90,
    ) -> list[str]:
        if start not in self.nodes:
            return []
        if self.nodes[start].untested_keys:
            return []
        q = deque([(start, [])])
        seen = {start}
        while q:
            sig, path = q.popleft()
            if len(path) >= max_depth:
                continue
            node = self.nodes.get(sig)
            if node is None:
                continue
            for key in sorted(node.edges):
                dest = self.safe_edge_destination(sig, key, min_support=min_support, min_confidence=min_confidence)
                if dest is None or dest in seen:
                    continue
                next_path = path + [key]
                dest_node = self.nodes.get(dest)
                if dest_node is not None and dest_node.untested_keys:
                    self.last_route = next_path
                    return next_path
                seen.add(dest)
                q.append((dest, next_path))
        self.last_route = []
        return []

    def current_candidate_for_key(self, scene: Scene, candidates: list[DecisionCandidate], key: str) -> DecisionCandidate | None:
        for candidate in candidates:
            if self.candidate_key(scene, candidate) == key:
                return candidate
        return None

    @staticmethod
    def _fallback_priority(candidate: DecisionCandidate) -> tuple[float, float, float, str]:
        payload = candidate.payload if isinstance(candidate.payload, dict) else {}
        posterior = payload.get("posterior") if isinstance(payload.get("posterior"), dict) else payload
        risk = float(posterior.get("game_over_probability", payload.get("game_over_probability", 0.0)) or 0.0)
        dead = float(posterior.get("dead_probability", payload.get("dead_probability", 0.0)) or 0.0)
        info = float(posterior.get("information_value", payload.get("information_value", 1.0)) or 0.0)
        return (risk, dead, -info, candidate.candidate_id)

    def fallback_candidate(self, scene: Scene, candidates: list[DecisionCandidate]) -> tuple[DecisionCandidate | None, str]:
        self.observe_state(scene, candidates)
        signature = self.state_key(scene)
        node = self.nodes[signature]
        untested = [
            c for c in candidates
            if c.spec is not None
            and (key := self.candidate_key(scene, c)) is not None
            and key in node.untested_keys
        ]
        if untested:
            return min(untested, key=self._fallback_priority), "local_untested_frontier"

        route = self.route_to_frontier(signature)
        if route:
            candidate = self.current_candidate_for_key(scene, candidates, route[0])
            if candidate is not None and candidate.spec is not None:
                return candidate, "known_safe_route_to_frontier"
        return None, "no_safe_frontier"

    def summary(self, scene: Scene, candidates: list[DecisionCandidate]) -> dict[str, Any]:
        self.observe_state(scene, candidates)
        signature = self.state_key(scene)
        node = self.nodes[signature]
        route = self.route_to_frontier(signature)
        route_candidate = None
        if route:
            c = self.current_candidate_for_key(scene, candidates, route[0])
            route_candidate = c.candidate_id if c is not None else None
        local_untested = []
        for c in candidates:
            key = self.candidate_key(scene, c)
            if key is not None and key in node.untested_keys:
                local_untested.append(c.candidate_id)
        return {
            "known_states": len(self.nodes),
            "known_tested_edges": sum(len(n.edges) for n in self.nodes.values()),
            "frontier_states": sum(bool(n.untested_keys) for n in self.nodes.values()),
            "current_state_visits": node.visits,
            "local_untested_candidates": local_untested[:24],
            "nearest_frontier_distance": len(route) if route else (0 if node.untested_keys else None),
            "safe_route_first_candidate": route_candidate,
            "safe_route_action_keys": route[:12],
        }
