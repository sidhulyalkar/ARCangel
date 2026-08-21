from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from arc3lab.types import Transition


class EpisodeMemory:
    """Lossless transition ledger plus compact statistics.

    The full in-memory history is never summarized away. Compact summaries are derived
    views, so a model/planner can retrieve details without trusting a lossy narrative.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.transitions: list[Transition] = []
        self.path = Path(path) if path else None
        self.action_effects: dict[int, Counter[str]] = defaultdict(Counter)
        self.state_visits: Counter[str] = Counter()
        self.target_effects: dict[str, Counter[str]] = defaultdict(Counter)
        self.level_action_counts: Counter[int] = Counter()

    def append(self, t: Transition, target_shape: str | None = None) -> None:
        self.transitions.append(t)
        self.state_visits[t.after_signature] += 1
        self.level_action_counts[t.level] += 1
        effect = "level" if t.level_completed else ("change" if t.meaningful_changed_cells else "dead")
        self.action_effects[t.action.action_id][effect] += 1
        if target_shape:
            self.target_effects[target_shape][effect] += 1
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(t), separators=(",", ":")) + "\n")

    def recent(self, n: int = 12) -> list[Transition]:
        return self.transitions[-n:]

    def dead_action_rate(self, action_id: int) -> float:
        c = self.action_effects[action_id]
        total = sum(c.values())
        return c["dead"] / total if total else 0.0

    def dead_target_rate(self, shape_hash: str) -> float:
        c = self.target_effects[shape_hash]
        total = sum(c.values())
        return c["dead"] / total if total else 0.0

    def compact(self) -> dict:
        return {
            "steps": len(self.transitions),
            "level_actions": dict(self.level_action_counts),
            "action_effects": {str(k): dict(v) for k, v in self.action_effects.items()},
            "recent": [
                {
                    "a": t.action.action_id,
                    "from": t.before_signature,
                    "to": t.after_signature,
                    "changed": t.meaningful_changed_cells,
                    "level_up": t.level_completed,
                }
                for t in self.recent(8)
            ],
        }
