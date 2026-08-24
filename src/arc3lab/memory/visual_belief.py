from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arc3lab.perception.spatial import component_relation
from arc3lab.perception.visual import VisualSignature, VisualTracker, visual_signature
from arc3lab.types import Scene


SUPPORTED_RELATIONS = {
    "touch",
    "adjacent8",
    "overlap",
    "center",
    "inside",
    "click",
    "remove",
    "transform",
    "unknown",
}


@dataclass(slots=True)
class VisualGoalBelief:
    relation: str
    target_signature: VisualSignature
    confidence: float
    evidence: str
    first_level: int
    last_level: int
    support: int = 1
    validations: int = 0
    contradictions: int = 0
    last_object: int | None = None
    last_track: int | None = None

    @property
    def calibrated_confidence(self) -> float:
        bonus = 0.08 * min(self.validations, 3)
        penalty = 0.12 * min(self.contradictions, 3)
        return max(0.0, min(1.0, self.confidence + bonus - penalty))


class VisualBeliefState:
    """Persistent, executable visual goal hypotheses.

    Beliefs are keyed by a translation-invariant visual target signature and a generic
    relation. They are carried across levels softly, not treated as immutable rules.
    """

    def __init__(self, *, max_beliefs: int = 24) -> None:
        self.max_beliefs = max(4, int(max_beliefs))
        self.beliefs: list[VisualGoalBelief] = []
        self.last_selected: tuple[str, VisualSignature] | None = None

    @staticmethod
    def _normalize_relation(value: Any) -> str:
        relation = str(value or "unknown").strip().lower()
        aliases = {
            "adjacent": "adjacent8",
            "enter": "overlap",
            "contain": "inside",
            "center_on": "center",
            "collect": "remove",
            "destroy": "remove",
        }
        relation = aliases.get(relation, relation)
        return relation if relation in SUPPORTED_RELATIONS else "unknown"

    def _find(self, relation: str, signature: VisualSignature) -> VisualGoalBelief | None:
        for belief in self.beliefs:
            if belief.relation == relation and belief.target_signature == signature:
                return belief
        return None

    def update_from_model(
        self,
        raw_goals: Any,
        scene: Scene,
        tracker: VisualTracker,
        *,
        level: int,
    ) -> None:
        if not isinstance(raw_goals, list):
            return
        for raw in raw_goals[:4]:
            if not isinstance(raw, dict):
                continue
            try:
                object_index = int(raw.get("target_object"))
            except Exception:
                continue
            if not (0 <= object_index < len(scene.components)):
                continue
            relation = self._normalize_relation(raw.get("relation"))
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
            except Exception:
                confidence = 0.5
            if confidence < 0.25:
                continue
            comp = scene.components[object_index]
            signature = visual_signature(comp)
            evidence = " ".join(str(raw.get("evidence", "")).split())[:300]
            track = tracker.track_for_component(object_index)
            existing = self._find(relation, signature)
            if existing is None:
                existing = VisualGoalBelief(
                    relation=relation,
                    target_signature=signature,
                    confidence=confidence,
                    evidence=evidence,
                    first_level=int(level),
                    last_level=int(level),
                    last_object=object_index,
                    last_track=track,
                )
                self.beliefs.append(existing)
            else:
                # Repeated independent model support increases confidence softly.
                existing.support += 1
                existing.confidence = max(existing.confidence, confidence)
                existing.last_level = int(level)
                existing.last_object = object_index
                existing.last_track = track
                if evidence:
                    existing.evidence = evidence
            self.last_selected = (relation, signature)
        self.beliefs.sort(
            key=lambda b: (b.calibrated_confidence, b.validations, b.support),
            reverse=True,
        )
        if len(self.beliefs) > self.max_beliefs:
            self.beliefs = self.beliefs[: self.max_beliefs]

    @staticmethod
    def matching_objects(scene: Scene, signature: VisualSignature) -> list[int]:
        return [i for i, comp in enumerate(scene.components) if visual_signature(comp) == signature]

    def current_matches(self, scene: Scene) -> list[tuple[VisualGoalBelief, int]]:
        out: list[tuple[VisualGoalBelief, int]] = []
        for belief in self.beliefs:
            for i in self.matching_objects(scene, belief.target_signature):
                out.append((belief, i))
        out.sort(key=lambda x: x[0].calibrated_confidence, reverse=True)
        return out

    def top_current(self, scene: Scene, *, limit: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        used: set[tuple[str, VisualSignature, int]] = set()
        for belief, i in self.current_matches(scene):
            key = (belief.relation, belief.target_signature, i)
            if key in used:
                continue
            used.add(key)
            out.append(
                {
                    "relation": belief.relation,
                    "target_object": i,
                    "target_signature": list(belief.target_signature),
                    "confidence": round(belief.calibrated_confidence, 4),
                    "support": belief.support,
                    "validations": belief.validations,
                    "contradictions": belief.contradictions,
                    "evidence": belief.evidence,
                }
            )
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def relation_holds(scene: Scene, actor_index: int, target_index: int, relation: str) -> bool:
        if not (0 <= actor_index < len(scene.components) and 0 <= target_index < len(scene.components)):
            return False
        actor = scene.components[actor_index]
        target = scene.components[target_index]
        rel = component_relation(actor, target)
        actor_cells = set(actor.cells)
        target_cells = set(target.cells)
        if relation == "touch":
            return bool(rel["touching"])
        if relation == "adjacent8":
            if actor_cells & target_cells:
                return True
            return min(
                max(abs(ar - tr), abs(ac - tc))
                for ar, ac in actor_cells
                for tr, tc in target_cells
            ) <= 1
        if relation in {"overlap", "center"}:
            if relation == "center":
                return actor.center_cell in target_cells or target.center_cell in actor_cells
            return bool(actor_cells & target_cells)
        if relation == "inside":
            return actor_cells <= target_cells or (
                target.bbox[0] <= actor.bbox[0] <= actor.bbox[2] <= target.bbox[2]
                and target.bbox[1] <= actor.bbox[1] <= actor.bbox[3] <= target.bbox[3]
            )
        return False

    def validate_completion(self, scene_before: Scene, actor_index: int | None) -> None:
        """Credit goal hypotheses that were visually satisfied when the level advanced."""
        if actor_index is None or not (0 <= actor_index < len(scene_before.components)):
            return
        for belief in self.beliefs:
            matches = self.matching_objects(scene_before, belief.target_signature)
            if not matches:
                continue
            if belief.relation in {"touch", "adjacent8", "overlap", "center", "inside"}:
                if any(self.relation_holds(scene_before, actor_index, j, belief.relation) for j in matches):
                    belief.validations += 1
                    belief.confidence = min(1.0, belief.confidence + 0.08)
            elif self.last_selected == (belief.relation, belief.target_signature):
                # For click/remove/transform predicates the transition itself is evidence.
                belief.validations += 1
                belief.confidence = min(1.0, belief.confidence + 0.05)

    def summary(self, scene: Scene) -> dict[str, Any]:
        return {
            "current_goal_candidates": self.top_current(scene, limit=6),
            "persistent_belief_count": len(self.beliefs),
            "validated_beliefs": sum(int(b.validations > 0) for b in self.beliefs),
        }
