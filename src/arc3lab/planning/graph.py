from __future__ import annotations

from collections import Counter, defaultdict, deque

from arc3lab.types import ActionSpec, Transition


class TransitionGraph:
    """Observed state graph used for novelty search and deterministic backtracking."""

    def __init__(self) -> None:
        self.edges: dict[str, dict[tuple, str]] = defaultdict(dict)
        self.edge_counts: Counter[tuple[str, tuple]] = Counter()
        self.level_states: dict[int, set[str]] = defaultdict(set)

    @staticmethod
    def key(action: ActionSpec) -> tuple:
        return (action.action_id, action.x, action.y)

    def add(self, t: Transition) -> None:
        k = self.key(t.action)
        self.edges[t.before_signature][k] = t.after_signature
        self.edge_counts[(t.before_signature, k)] += 1
        self.level_states[t.level].update([t.before_signature, t.after_signature])

    def known_destination(self, state: str, action: ActionSpec) -> str | None:
        return self.edges.get(state, {}).get(self.key(action))

    def action_trials(self, state: str, action: ActionSpec) -> int:
        return self.edge_counts[(state, self.key(action))]

    def shortest_path_to_novel(self, start: str, level: int) -> list[tuple] | None:
        """Find a path through known edges toward the least-visited frontier state."""
        q = deque([(start, [])])
        seen = {start}
        while q:
            state, path = q.popleft()
            outgoing = self.edges.get(state, {})
            if state != start and len(outgoing) < 2:
                return path
            for akey, nxt in outgoing.items():
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + [akey]))
        return None
