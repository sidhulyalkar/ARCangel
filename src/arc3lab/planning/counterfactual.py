from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from arc3lab.memory.affordance import AffordanceMemory
from arc3lab.memory.visual_belief import VisualBeliefState, VisualGoalBelief
from arc3lab.perception.spatial import SpatialControlModel, actor_offsets, anchor_valid_mask
from arc3lab.perception.visual import VisualTracker, visual_signature
from arc3lab.planning.spatial import shortest_spatial_plan
from arc3lab.types import ActionSpec, Scene


@dataclass(slots=True)
class DecisionCandidate:
    candidate_id: str
    kind: str
    score: float
    spec: ActionSpec | None
    payload: dict[str, Any]

    def record(self) -> dict[str, Any]:
        out = {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "score": round(float(self.score), 4),
            **self.payload,
        }
        if self.spec is not None:
            out["action"] = {
                "id": int(self.spec.action_id),
                "x": self.spec.x,
                "y": self.spec.y,
            }
        return out


def _cells_at_anchor(scene: Scene, actor_index: int, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    offsets = actor_offsets(scene.components[actor_index])
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets}


def _goal_distance(
    scene: Scene,
    actor_index: int,
    actor_cells: set[tuple[int, int]],
    target_index: int,
    relation: str,
) -> float:
    target = scene.components[target_index]
    target_cells = set(target.cells)
    if not actor_cells or not target_cells:
        return 1e6
    if relation == "touch":
        return float(min(abs(ar - tr) + abs(ac - tc) for ar, ac in actor_cells for tr, tc in target_cells))
    if relation == "adjacent8":
        return float(min(max(abs(ar - tr), abs(ac - tc)) for ar, ac in actor_cells for tr, tc in target_cells))
    if relation in {"overlap", "inside"}:
        if actor_cells & target_cells:
            return 0.0
        return float(min(abs(ar - tr) + abs(ac - tc) for ar, ac in actor_cells for tr, tc in target_cells))
    if relation == "center":
        ar = sum(r for r, _ in actor_cells) / len(actor_cells)
        ac = sum(c for _, c in actor_cells) / len(actor_cells)
        tr, tc = target.center_cell
        return abs(ar - tr) + abs(ac - tc)
    return 0.0


def _best_goal_progress(
    scene: Scene,
    actor_index: int,
    current_anchor: tuple[int, int],
    predicted_anchor: tuple[int, int] | None,
    beliefs: VisualBeliefState,
) -> tuple[float, dict[str, Any] | None]:
    if predicted_anchor is None:
        return 0.0, None
    current_cells = _cells_at_anchor(scene, actor_index, current_anchor)
    predicted_cells = _cells_at_anchor(scene, actor_index, predicted_anchor)
    best = 0.0
    best_meta: dict[str, Any] | None = None
    for belief, target_index in beliefs.current_matches(scene)[:8]:
        if belief.relation not in {"touch", "adjacent8", "overlap", "center", "inside"}:
            continue
        before = _goal_distance(scene, actor_index, current_cells, target_index, belief.relation)
        after = _goal_distance(scene, actor_index, predicted_cells, target_index, belief.relation)
        raw = before - after
        weighted = raw * belief.calibrated_confidence
        if weighted > best:
            best = weighted
            best_meta = {
                "target_object": target_index,
                "relation": belief.relation,
                "goal_confidence": round(belief.calibrated_confidence, 4),
                "distance_before": round(before, 3),
                "distance_after": round(after, 3),
            }
    return best, best_meta


def enumerate_decision_candidates(
    scene: Scene,
    control: SpatialControlModel,
    tracker: VisualTracker,
    affordances: AffordanceMemory,
    beliefs: VisualBeliefState,
    *,
    max_click_candidates: int = 16,
    max_spatial_plans: int = 4,
) -> list[DecisionCandidate]:
    """Enumerate exact candidate actions/plans with causal priors and visual progress.

    Scores are deliberately advisory. They make tradeoffs explicit for the model and
    provide a safe candidate registry, but do not hard-code hidden-game semantics.
    """
    out: list[DecisionCandidate] = []
    actor = control.actor_hypothesis(scene)
    actor_index = int(actor["index"]) if actor is not None else None
    current_anchor = tuple(actor["center"]) if actor is not None else None
    vectors = control.action_vectors(scene, min_support=1)
    valid_mask = anchor_valid_mask(scene, actor_index) if actor_index is not None else None

    simple_n = 0
    for aid in scene.available_actions:
        aid = int(aid)
        if aid in {0, 6}:
            continue
        posterior = affordances.action_posterior(aid)
        predicted_anchor = None
        collision_risk = 0.0
        vector_meta = None
        if actor_index is not None and current_anchor is not None and aid in vectors:
            ev = vectors[aid]
            predicted_anchor = (current_anchor[0] + ev.delta[0], current_anchor[1] + ev.delta[1])
            vector_meta = {
                "delta": list(ev.delta),
                "support": ev.support,
                "confidence": round(ev.confidence, 4),
                "purity": round(ev.purity, 4),
            }
            if valid_mask is not None:
                r, c = predicted_anchor
                if not (0 <= r < valid_mask.shape[0] and 0 <= c < valid_mask.shape[1] and bool(valid_mask[r, c])):
                    collision_risk = 1.0
                    predicted_anchor = None
        progress, goal_meta = (0.0, None)
        if actor_index is not None and current_anchor is not None:
            progress, goal_meta = _best_goal_progress(
                scene, actor_index, current_anchor, predicted_anchor, beliefs
            )
        score = (
            1.15 * progress
            + 0.55 * posterior.information_value
            + 3.0 * posterior.level_probability
            - 0.8 * posterior.dead_probability
            - 2.2 * posterior.game_over_probability
            - 1.5 * collision_risk
        )
        spec = ActionSpec(
            aid,
            reason="counterfactual visual candidate",
            confidence=0.5,
        )
        out.append(
            DecisionCandidate(
                candidate_id=f"a{simple_n}",
                kind="primitive",
                score=score,
                spec=spec,
                payload={
                    "posterior": {
                        "support": posterior.support,
                        "counts": posterior.counts,
                        "dead_probability": posterior.dead_probability,
                        "game_over_probability": posterior.game_over_probability,
                        "level_probability": posterior.level_probability,
                        "information_value": posterior.information_value,
                    },
                    "movement": vector_meta,
                    "predicted_actor_center": list(predicted_anchor) if predicted_anchor is not None else None,
                    "goal_progress": round(progress, 4),
                    "goal_progress_detail": goal_meta,
                    "collision_risk": collision_risk,
                },
            )
        )
        simple_n += 1

    click_records = affordances.click_candidates(
        scene, tracker, max_candidates=max_click_candidates
    )
    goal_click_matches = {
        (belief.target_signature, belief.relation): belief.calibrated_confidence
        for belief in beliefs.beliefs
        if belief.relation in {"click", "remove", "transform"}
    }
    for k, record in enumerate(click_records):
        comp = scene.components[int(record["object"])]
        sig = visual_signature(comp)
        goal_bonus = max(
            [
                conf
                for (target_sig, _relation), conf in goal_click_matches.items()
                if target_sig == sig
            ]
            or [0.0]
        )
        score = float(record["score"]) + 1.4 * goal_bonus
        spec = ActionSpec(
            6,
            x=int(record["x"]),
            y=int(record["y"]),
            reason="visual affordance click candidate",
            confidence=0.5,
        )
        out.append(
            DecisionCandidate(
                candidate_id=f"c{k}",
                kind="click",
                score=score,
                spec=spec,
                payload={**record, "goal_match_confidence": round(goal_bonus, 4)},
            )
        )

    if actor_index is not None and control.planner_ready(scene):
        plans: list[tuple[float, VisualGoalBelief, int, Any]] = []
        for belief, target_index in beliefs.current_matches(scene)[:10]:
            if belief.relation not in {"touch", "adjacent8", "overlap", "center", "inside"}:
                continue
            plan = shortest_spatial_plan(
                scene,
                actor_index,
                target_index,
                vectors,
                relation=belief.relation,
                max_steps=128,
            )
            if plan is None or plan.steps <= 0:
                continue
            utility = 2.4 * belief.calibrated_confidence - 0.035 * plan.steps
            plans.append((utility, belief, target_index, plan))
        for k, (utility, belief, target_index, plan) in enumerate(
            sorted(plans, key=lambda x: -x[0])[: max_spatial_plans]
        ):
            out.append(
                DecisionCandidate(
                    candidate_id=f"p{k}",
                    kind="spatial_plan",
                    score=utility,
                    spec=None,
                    payload={
                        "target_object": target_index,
                        "relation": belief.relation,
                        "goal_confidence": round(belief.calibrated_confidence, 4),
                        "steps": plan.steps,
                        "actions_preview": list(plan.actions[:12]),
                    },
                )
            )

    # Stable ordering by score, while preserving exact candidate IDs for selection.
    return sorted(out, key=lambda x: (-x.score, x.candidate_id))


def candidate_records(candidates: list[DecisionCandidate], *, limit: int = 28) -> list[dict[str, Any]]:
    return [c.record() for c in candidates[: max(1, int(limit))]]
