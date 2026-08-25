from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import log2, sqrt
from typing import Any

import numpy as np

from arc3lab.memory.affordance import AffordanceMemory
from arc3lab.memory.visual_belief import VisualBeliefState
from arc3lab.perception.spatial import SpatialControlModel, anchor_valid_mask
from arc3lab.perception.visual import VisualTracker, visual_signature
from arc3lab.types import ActionSpec, Scene


def _entropy(counts: dict[str, int] | Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 1.0
    probs = [v / total for v in counts.values() if v > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * log2(p) for p in probs)
    return min(1.0, h / log2(len(probs)))


def _symmetry_scores(grid: np.ndarray) -> dict[str, float]:
    g = np.asarray(grid)
    out = {
        "horizontal": float(np.mean(g == np.flipud(g))),
        "vertical": float(np.mean(g == np.fliplr(g))),
        "rot180": float(np.mean(g == np.rot90(g, 2))),
    }
    if g.shape[0] == g.shape[1]:
        out["rot90"] = float(np.mean(g == np.rot90(g, 1)))
        out["diag_main"] = float(np.mean(g == g.T))
        out["diag_anti"] = float(np.mean(g == np.fliplr(np.flipud(g)).T))
    return {k: round(v, 4) for k, v in out.items()}


def _mask_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _reachable(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    if not (0 <= start[0] < mask.shape[0] and 0 <= start[1] < mask.shape[1] and bool(mask[start])):
        return set()
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in _mask_neighbors(mask, *u):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _articulation_points(mask: np.ndarray, allowed: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Exact articulation points of the reachable 4-neighbor graph, iteratively."""
    disc: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    child_count: Counter[tuple[int, int]] = Counter()
    arts: set[tuple[int, int]] = set()
    tick = 0

    for root in allowed:
        if root in disc:
            continue
        tick += 1
        disc[root] = low[root] = tick
        parent[root] = None
        stack: list[tuple[tuple[int, int], Any]] = [
            (root, iter(v for v in _mask_neighbors(mask, *root) if v in allowed))
        ]
        while stack:
            u, it = stack[-1]
            try:
                v = next(it)
            except StopIteration:
                stack.pop()
                p = parent.get(u)
                if p is None:
                    if child_count[u] > 1:
                        arts.add(u)
                else:
                    low[p] = min(low[p], low[u])
                    if parent.get(p) is not None and low[u] >= disc[p]:
                        arts.add(p)
                continue

            if v not in disc:
                parent[v] = u
                child_count[u] += 1
                tick += 1
                disc[v] = low[v] = tick
                stack.append((v, iter(w for w in _mask_neighbors(mask, *v) if w in allowed)))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])
    return arts


@dataclass(slots=True)
class PerceptualSnapshot:
    summary: dict[str, Any]


class PerceptualStateEstimator:
    """Compact high-level state estimator for visual interactive reasoning.

    It deliberately extracts facts a strong human would stabilize before planning:
    agency, scene phase, topology, symmetry, novelty, hazards, irreversibility,
    goal uncertainty, action-model uncertainty, and likely semantic object roles.
    No game-specific role/color/action priors are used.
    """

    def __init__(self, *, history: int = 24) -> None:
        self.signature_history: deque[str] = deque(maxlen=max(8, int(history)))
        self.color_history: deque[tuple[int, ...]] = deque(maxlen=max(8, int(history)))
        self.topology_history: deque[tuple[int, int, int]] = deque(maxlen=max(8, int(history)))
        self.topology_events: deque[dict[str, Any]] = deque(maxlen=max(8, int(history)))
        self.last_summary: dict[str, Any] = {}

    def reset_temporal(self) -> None:
        self.signature_history.clear()
        self.color_history.clear()
        self.topology_history.clear()
        self.topology_events.clear()
        self.last_summary = {}

    @staticmethod
    def _cycle_period(signatures: list[str], max_period: int = 8) -> int | None:
        n = len(signatures)
        for period in range(1, min(max_period, n // 2) + 1):
            if signatures[-period:] == signatures[-2 * period : -period]:
                return period
        return None

    @staticmethod
    def _goal_uncertainty(beliefs: VisualBeliefState, scene: Scene) -> tuple[float, list[dict[str, Any]]]:
        goals = beliefs.top_current(scene, limit=6)
        if not goals:
            return 1.0, []
        confs = np.asarray([max(1e-6, float(g["confidence"])) for g in goals], dtype=np.float64)
        probs = confs / confs.sum()
        if len(probs) == 1:
            entropy = 1.0 - float(confs[0])
        else:
            entropy = float(-(probs * np.log2(probs)).sum() / np.log2(len(probs)))
            entropy = 0.55 * entropy + 0.45 * (1.0 - float(confs.max()))
        return round(max(0.0, min(1.0, entropy)), 4), goals

    @staticmethod
    def _action_uncertainty(affordances: AffordanceMemory, valid_actions: tuple[int, ...]) -> tuple[float, dict[str, Any]]:
        rows: dict[str, Any] = {}
        values = []
        for aid in valid_actions:
            if int(aid) == 0:
                continue
            p = affordances.action_posterior(int(aid))
            ent = _entropy(p.counts)
            if p.support == 0:
                ent = 1.0
            values.append(ent)
            rows[str(int(aid))] = {
                "support": p.support,
                "effect_entropy": round(ent, 4),
                "dominant": p.dominant,
                "confidence": round(p.confidence, 4),
                "information_value": p.information_value,
                "game_over_probability": p.game_over_probability,
                "level_probability": p.level_probability,
            }
        return round(float(np.mean(values)) if values else 1.0, 4), rows

    @staticmethod
    def _role_hypotheses(
        scene: Scene,
        tracker: VisualTracker,
        control: SpatialControlModel,
        affordances: AffordanceMemory,
        beliefs: VisualBeliefState,
        *,
        limit: int = 14,
    ) -> list[dict[str, Any]]:
        actor = control.actor_hypothesis(scene)
        actor_index = int(actor["index"]) if actor is not None else None
        actor_conf = float(actor["confidence"]) if actor is not None else 0.0
        goal_by_object: dict[int, float] = {}
        for goal in beliefs.top_current(scene, limit=8):
            goal_by_object[int(goal["target_object"])] = max(
                goal_by_object.get(int(goal["target_object"]), 0.0),
                float(goal["confidence"]),
            )
        out = []
        for obj in tracker.summary(scene, limit=max(limit * 2, 24)).get("tracked_objects", []):
            i = int(obj["object"])
            comp = scene.components[i]
            click = affordances.click_posterior(visual_signature(comp))
            motion = min(1.0, 0.25 * float(obj["motion_events"]))
            transform = min(1.0, 0.30 * float(obj["transform_events"]))
            agency = actor_conf if i == actor_index else 0.0
            target = goal_by_object.get(i, 0.0)
            trigger = min(1.0, 0.7 * (1.0 - click.dead_probability) + 0.3 * click.level_probability) if click.support else 0.0
            hazard = min(1.0, click.game_over_probability + 0.2 * transform)
            dynamic = min(1.0, motion + 0.5 * transform)
            decoration = max(0.0, min(1.0, 0.7 * float(obj["seen"] > 2) - 0.6 * dynamic - 0.5 * target - 0.5 * trigger))
            role_scores = {
                "controlled": round(agency, 4),
                "goal_target": round(target, 4),
                "trigger_or_button": round(trigger, 4),
                "hazard": round(hazard, 4),
                "dynamic_entity": round(dynamic, 4),
                "decoration": round(decoration, 4),
            }
            best_role = max(role_scores, key=role_scores.get)
            out.append({
                "object": i,
                "track": obj.get("track"),
                "center": obj["center"],
                "signature": list(visual_signature(comp)),
                "salience": obj["salience"],
                "best_role": best_role,
                "role_scores": role_scores,
            })
        out.sort(key=lambda x: (max(x["role_scores"].values()), x["salience"]), reverse=True)
        return out[:limit]

    def summarize(
        self,
        scene: Scene,
        tracker: VisualTracker,
        control: SpatialControlModel,
        affordances: AffordanceMemory,
        beliefs: VisualBeliefState,
        *,
        recent_grids: list[np.ndarray] | None = None,
        last_action: ActionSpec | None = None,
        prediction_mismatch: bool = False,
    ) -> dict[str, Any]:
        grid = np.asarray(scene.grid)
        if not self.signature_history or self.signature_history[-1] != scene.signature:
            self.signature_history.append(scene.signature)
            colors = tuple(int(x) for x in np.bincount(grid.ravel(), minlength=16))
            self.color_history.append(colors)

        actor = control.actor_hypothesis(scene)
        actor_conf = float(actor["confidence"]) if actor is not None else 0.0
        candidate_count = int(actor.get("candidate_count", 0)) if actor is not None else 0
        vectors = control.action_vectors(scene, min_support=1)
        vector_conf = max((float(v.confidence) for v in vectors.values()), default=0.0)
        control_uncertainty = max(1.0 - actor_conf, 0.35 if candidate_count > 1 else 0.0)
        if len(vectors) < 2:
            control_uncertainty = max(control_uncertainty, 0.75)

        goal_uncertainty, goals = self._goal_uncertainty(beliefs, scene)
        action_uncertainty, action_models = self._action_uncertainty(affordances, scene.available_actions)

        topology: dict[str, Any] = {"available": False}
        topology_uncertainty = 1.0
        if actor is not None:
            try:
                actor_i = int(actor["index"])
                mask = anchor_valid_mask(scene, actor_i)
                anchor = tuple(int(x) for x in actor["center"])
                reachable = _reachable(mask, anchor)
                arts = _articulation_points(mask, reachable) if reachable else set()
                free = int(np.count_nonzero(mask))
                reachable_ratio = len(reachable) / max(free, 1)
                nearest_choke = None
                if arts:
                    nearest_choke = min(abs(anchor[0]-r) + abs(anchor[1]-c) for r, c in arts)
                current_topology = (free, len(reachable), len(arts))
                previous_topology = self.topology_history[-1] if self.topology_history else None
                topology_delta = None
                if previous_topology is not None:
                    topology_delta = {
                        "valid_anchors": free - previous_topology[0],
                        "reachable_anchors": len(reachable) - previous_topology[1],
                        "articulation_points": len(arts) - previous_topology[2],
                    }
                    if last_action is not None and any(int(v) != 0 for v in topology_delta.values()):
                        self.topology_events.append({
                            "action": int(last_action.action_id),
                            "delta": dict(topology_delta),
                            "reachable_ratio": round(reachable_ratio, 4),
                        })
                topology = {
                    "available": True,
                    "valid_anchors": free,
                    "reachable_anchors": len(reachable),
                    "reachable_ratio": round(reachable_ratio, 4),
                    "articulation_points": len(arts),
                    "nearest_bottleneck_distance": nearest_choke,
                    "actor_anchor": list(anchor),
                    "delta_from_previous": topology_delta,
                    "recent_topology_events": list(self.topology_events)[-6:],
                }
                self.topology_history.append(current_topology)
                topology_uncertainty = 1.0 - min(1.0, actor_conf * max(vector_conf, 0.4))
            except Exception:
                pass

        symmetry = _symmetry_scores(grid)
        strongest_sym = max(symmetry.items(), key=lambda kv: kv[1]) if symmetry else ("none", 0.0)
        symmetry_anomaly: dict[str, Any] | None = None
        if strongest_sym[1] >= 0.72:
            transformed = None
            if strongest_sym[0] == "horizontal": transformed = np.flipud(grid)
            elif strongest_sym[0] == "vertical": transformed = np.fliplr(grid)
            elif strongest_sym[0] == "rot180": transformed = np.rot90(grid, 2)
            elif strongest_sym[0] == "rot90" and grid.shape[0] == grid.shape[1]: transformed = np.rot90(grid, 1)
            elif strongest_sym[0] == "diag_main" and grid.shape[0] == grid.shape[1]: transformed = grid.T
            elif strongest_sym[0] == "diag_anti" and grid.shape[0] == grid.shape[1]: transformed = np.fliplr(np.flipud(grid)).T
            if transformed is not None and transformed.shape == grid.shape:
                mismatch = np.argwhere(grid != transformed)
                if len(mismatch):
                    mismatch_cells = {(int(r), int(c)) for r, c in mismatch}
                    anomaly_objects = [i for i, comp in enumerate(scene.components) if any(cell in mismatch_cells for cell in comp.cells)]
                    symmetry_anomaly = {
                        "transform": strongest_sym[0],
                        "mismatch_cells": int(len(mismatch)),
                        "objects_intersecting_mismatch": anomaly_objects[:12],
                        "bbox": [int(mismatch[:,0].min()), int(mismatch[:,1].min()), int(mismatch[:,0].max()), int(mismatch[:,1].max())],
                    }

        recent = list(recent_grids or [])[-3:]
        change: dict[str, Any] = {"changed_fraction": 0.0, "changed_cells": 0}
        if len(recent) >= 2 and recent[-1].shape == recent[-2].shape:
            delta = np.asarray(recent[-1]) != np.asarray(recent[-2])
            changed = np.argwhere(delta)
            change = {
                "changed_fraction": round(float(delta.mean()), 5),
                "changed_cells": int(delta.sum()),
                "changed_bbox": None if len(changed) == 0 else [int(changed[:,0].min()), int(changed[:,1].min()), int(changed[:,0].max()), int(changed[:,1].max())],
            }

        current_objects = tracker.summary(scene, limit=64).get("tracked_objects", [])
        new_tracks = sum(1 for obj in current_objects if obj.get("track") in tracker.tracks and tracker.tracks[int(obj["track"])].first_step == tracker.last_step)
        moving_tracks = sum(int(obj["motion_events"] > 0) for obj in current_objects)
        transforming_tracks = sum(int(obj["transform_events"] > 0) for obj in current_objects)
        recent_events = list(tracker.recent_events)[-12:]
        irreversible_events = [e for e in recent_events if e.get("kind") in {"appear", "disappear", "transform"}]

        signatures = list(self.signature_history)
        period = self._cycle_period(signatures)
        phase = {"cycle_detected": period is not None, "period": period, "distinct_recent_states": len(set(signatures[-8:])), "history": len(signatures)}

        causal_edges: dict[str, dict[str, Any]] = {}
        for event in list(tracker.recent_events)[-32:]:
            aid = event.get("action")
            if aid is None:
                continue
            row = causal_edges.setdefault(str(int(aid)), {"tracks": set(), "event_counts": Counter()})
            if event.get("track") is not None:
                row["tracks"].add(int(event["track"]))
            row["event_counts"][str(event.get("kind", "unknown"))] += 1
        causal_graph = {aid: {"tracks": sorted(row["tracks"]), "event_counts": dict(row["event_counts"])} for aid, row in causal_edges.items()}

        hazard_actions = sorted([
            {"action": int(aid), "game_over_probability": row["game_over_probability"], "support": row["support"]}
            for aid, row in action_models.items() if row["support"] > 0
        ], key=lambda x: x["game_over_probability"], reverse=True)[:4]

        uncertainty = {"control": round(control_uncertainty, 4), "goal": goal_uncertainty, "action_model": action_uncertainty, "topology": round(topology_uncertainty, 4)}
        if prediction_mismatch:
            uncertainty["model_consistency"] = 1.0
        dominant = max(uncertainty, key=uncertainty.get)
        orientation_entropy = float(np.mean(list(uncertainty.values())))

        if prediction_mismatch: mode = "REPAIR_MODEL"
        elif control_uncertainty >= 0.62: mode = "IDENTIFY_AGENCY"
        elif goal_uncertainty >= 0.62: mode = "IDENTIFY_GOAL"
        elif action_uncertainty >= 0.58: mode = "DISCRIMINATE_DYNAMICS"
        elif period is not None and period > 1: mode = "REASON_ABOUT_PHASE"
        elif actor is not None and control.planner_ready(scene) and goal_uncertainty <= 0.38: mode = "PLAN_AND_EXECUTE"
        else: mode = "MODEL_AND_TEST"

        summary = {
            "recommended_mode": mode,
            "orientation_entropy": round(orientation_entropy, 4),
            "dominant_uncertainty": dominant,
            "uncertainty": uncertainty,
            "agency": {"actor": None if actor is None else {"object": int(actor["index"]), "center": list(actor["center"]), "confidence": round(actor_conf, 4), "candidate_count": candidate_count}, "learned_action_vectors": {str(a): {"delta": list(v.delta), "support": v.support, "confidence": round(v.confidence, 4), "purity": round(v.purity, 4)} for a, v in vectors.items()}},
            "goals": {"entropy": goal_uncertainty, "top": goals[:4]},
            "action_models": action_models,
            "topology": topology,
            "symmetry": {**symmetry, "strongest": strongest_sym[0], "strongest_score": strongest_sym[1], "anomaly": symmetry_anomaly},
            "causal_interaction_graph": causal_graph,
            "temporal_phase": phase,
            "change": change,
            "novelty": {"new_tracks_now": new_tracks, "moving_tracks": moving_tracks, "transforming_tracks": transforming_tracks, "recent_irreversible_events": irreversible_events[-6:]},
            "risk": {"highest_game_over_actions": hazard_actions},
            "role_hypotheses": self._role_hypotheses(scene, tracker, control, affordances, beliefs),
            "last_action": None if last_action is None else int(last_action.action_id),
        }
        self.last_summary = summary
        return summary
