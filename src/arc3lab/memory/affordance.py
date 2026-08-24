from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from arc3lab.perception.visual import VisualSignature, VisualTracker, visual_signature
from arc3lab.types import ActionSpec, Scene


@dataclass(frozen=True, slots=True)
class EffectPosterior:
    counts: dict[str, int]
    support: int
    dominant: str | None
    confidence: float
    dead_probability: float
    game_over_probability: float
    level_probability: float
    information_value: float


class AffordanceMemory:
    """Causal visual affordance memory for action channels and click targets.

    The memory is intentionally distributional. A visually similar target may be useful
    in one context and inert in another, so evidence is a soft prior, never an absolute
    suppression rule.
    """

    def __init__(self) -> None:
        self.action_effects: dict[int, Counter[str]] = defaultdict(Counter)
        self.click_effects: dict[VisualSignature, Counter[str]] = defaultdict(Counter)
        self.click_attempts: Counter[VisualSignature] = Counter()
        self.observations = 0

    @staticmethod
    def coarse_effect(
        before: Scene,
        after: Scene,
        *,
        level_completed: bool = False,
        game_over: bool = False,
    ) -> str:
        if level_completed:
            return "level"
        if game_over:
            return "game_over"
        if before.grid.shape != after.grid.shape:
            return "global_change"
        changed = int(np.count_nonzero(np.asarray(before.grid) != np.asarray(after.grid)))
        if changed == 0:
            return "dead"
        before_count = len(before.components)
        after_count = len(after.components)
        if after_count < before_count:
            return "remove"
        if after_count > before_count:
            return "spawn"
        return "change"

    @staticmethod
    def clicked_component(scene: Scene, action: ActionSpec) -> int | None:
        if int(action.action_id) != 6 or action.x is None or action.y is None:
            return None
        cell = (int(action.y), int(action.x))
        for i, comp in enumerate(scene.components):
            if cell in comp.cells:
                return i
        return None

    def observe(
        self,
        before: Scene,
        after: Scene,
        action: ActionSpec,
        *,
        level_completed: bool = False,
        game_over: bool = False,
    ) -> str:
        effect = self.coarse_effect(
            before,
            after,
            level_completed=level_completed,
            game_over=game_over,
        )
        aid = int(action.action_id)
        self.action_effects[aid][effect] += 1
        self.observations += 1
        if aid == 6:
            index = self.clicked_component(before, action)
            if index is not None:
                sig = visual_signature(before.components[index])
                self.click_effects[sig][effect] += 1
                self.click_attempts[sig] += 1
        return effect

    @staticmethod
    def _posterior(counts: Counter[str]) -> EffectPosterior:
        support = int(sum(counts.values()))
        if support:
            dominant, n = counts.most_common(1)[0]
            confidence = n / support
        else:
            dominant = None
            confidence = 0.0
        # Smoothed probabilities keep rare actions alive while still exposing risk.
        denom = support + 2.0
        dead = (counts.get("dead", 0) + 0.5) / denom
        game_over = (counts.get("game_over", 0) + 0.25) / (support + 1.0)
        level = (counts.get("level", 0) + 0.1) / (support + 1.0)
        info = 1.0 / sqrt(support + 1.0)
        return EffectPosterior(
            counts=dict(counts),
            support=support,
            dominant=dominant,
            confidence=confidence,
            dead_probability=round(dead, 4),
            game_over_probability=round(game_over, 4),
            level_probability=round(level, 4),
            information_value=round(info, 4),
        )

    def action_posterior(self, action_id: int) -> EffectPosterior:
        return self._posterior(self.action_effects[int(action_id)])

    def click_posterior(self, signature: VisualSignature) -> EffectPosterior:
        return self._posterior(self.click_effects[signature])

    def action_summary(self, valid_actions: tuple[int, ...] | list[int]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for aid in valid_actions:
            p = self.action_posterior(int(aid))
            out[str(int(aid))] = {
                "counts": p.counts,
                "support": p.support,
                "dominant": p.dominant,
                "confidence": round(p.confidence, 4),
                "dead_probability": p.dead_probability,
                "game_over_probability": p.game_over_probability,
                "level_probability": p.level_probability,
                "information_value": p.information_value,
            }
        return out

    @staticmethod
    def representative_cells(component, *, max_cells: int = 5) -> list[tuple[int, int]]:
        cells = list(component.cells)
        if not cells:
            return []
        chosen = [component.center_cell]
        if len(cells) <= 1:
            return chosen
        extremes = [
            min(cells, key=lambda p: (p[0], p[1])),
            max(cells, key=lambda p: (p[0], -p[1])),
            min(cells, key=lambda p: (p[1], p[0])),
            max(cells, key=lambda p: (p[1], -p[0])),
        ]
        for cell in extremes:
            if cell not in chosen:
                chosen.append(cell)
            if len(chosen) >= max_cells:
                break
        return chosen[:max_cells]

    def click_candidates(
        self,
        scene: Scene,
        tracker: VisualTracker | None = None,
        *,
        max_candidates: int = 24,
    ) -> list[dict[str, Any]]:
        if 6 not in scene.available_actions:
            return []
        color_counts = Counter(int(c.color) for c in scene.components)
        candidates: list[tuple[float, dict[str, Any]]] = []
        seen_cells: set[tuple[int, int]] = set()
        for i, comp in enumerate(scene.components):
            sig = visual_signature(comp)
            posterior = self.click_posterior(sig)
            rarity = 1.0 / max(1, color_counts[int(comp.color)])
            size_term = 1.0 / sqrt(max(1, int(comp.pixels)))
            # Positive causal evidence dominates. Novelty is useful when evidence is absent.
            impact = 1.0 - posterior.dead_probability
            base_score = (
                2.4 * posterior.level_probability
                + 0.7 * impact
                + 0.45 * posterior.information_value
                + 0.18 * rarity
                + 0.12 * size_term
                - 1.3 * posterior.game_over_probability
            )
            tid = tracker.track_for_component(i) if tracker is not None else None
            for rank, (r, c) in enumerate(self.representative_cells(comp)):
                if (r, c) in seen_cells:
                    continue
                seen_cells.add((r, c))
                # Center is preferred slightly; extra samples preserve large/hollow objects.
                score = base_score - 0.035 * rank
                record = {
                    "object": i,
                    "track": tid,
                    "x": int(c),
                    "y": int(r),
                    "color": int(comp.color),
                    "pixels": int(comp.pixels),
                    "shape": str(comp.shape_hash),
                    "support": posterior.support,
                    "effect_counts": posterior.counts,
                    "dead_probability": posterior.dead_probability,
                    "game_over_probability": posterior.game_over_probability,
                    "level_probability": posterior.level_probability,
                    "information_value": posterior.information_value,
                    "score": round(score, 4),
                }
                candidates.append((score, record))
        return [record for _, record in sorted(candidates, key=lambda x: -x[0])[: max(1, int(max_candidates))]]
