from __future__ import annotations

from typing import Any

from arc3lab.perception.state_estimator import PerceptualStateEstimator
from arc3lab.planning.counterfactual import DecisionCandidate, enumerate_decision_candidates
from arc3lab.planning.frontier import ExplorationFrontier
from arc3lab.policy.visual_coding import VisualDecisionPolicy
from arc3lab.types import ActionSpec, Scene


PERCEPTUAL_INSTRUCTIONS = """
PERCEPTUAL STATE ESTIMATION
The `perceptual_state` block is a compact, exact evidence layer intended to prevent premature
planning. It summarizes agency, goal uncertainty, action-model entropy, free-space topology,
bottlenecks, symmetry, temporal cycles, novelty/irreversibility, risk, and soft object-role
hypotheses. These are evidence summaries, not hidden game rules.

Use the following discipline:
- If recommended_mode is IDENTIFY_AGENCY, prefer a low-risk action that most cleanly tests which
  object is controlled or whether a primitive action is locomotion.
- If IDENTIFY_GOAL, compare salient persistent objects, visual progress cues, and prior level
  completions before committing to a target.
- If DISCRIMINATE_DYNAMICS, prefer a candidate with high information value and low irreversible
  risk that separates competing mechanic hypotheses.
- If REASON_ABOUT_PHASE, account for repeated/periodic visual states before treating a change as
  causal.
- If PLAN_AND_EXECUTE, use the exact grounded planner when its assumptions match the visual world.
- If REPAIR_MODEL, retrodict the failed prediction against prior observations before acting again.

High symmetry can imply correspondence, matching, or a symmetry-breaking interaction, but never
assume the goal from symmetry alone. Articulation points are geometric bottlenecks, not automatic
goals. Appearance/disappearance/transform events are potential irreversible state changes and
should receive extra caution until their meaning is understood.

The model receives TWO visual views on every reasoning turn:
1. CURRENT HIGH-RESOLUTION board for precise spatial interpretation.
2. TEMPORAL CONTEXT packet containing t-2, t-1, current, and the diagnostic delta.
Use both. The current view answers "what is where?"; the temporal view answers "what changed and
what was caused?". When a rule matters to the next plan but its support is unclear, use the sandbox
`retrodict_action(action_id)` helper to check whether the rule actually explains prior transitions.

`exploration_graph` is a generic directed state-action graph. It reports local untested actions
and the shortest known safe deterministic route to a state with something still untested. Use it
to avoid loops and repeated guessing when the semantic goal remains uncertain; do not confuse
frontier novelty with goal progress.
"""


class PerceptualDecisionPolicy(VisualDecisionPolicy):
    """V009 perception-first policy with multi-view vision and latent state estimation."""

    def __init__(
        self,
        *args: Any,
        current_view_side: int = 512,
        temporal_view_side: int = 384,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("candidate_limit", 40)
        kwargs.setdefault("max_click_candidates", 24)
        super().__init__(*args, **kwargs)
        self.perceptual_estimator = PerceptualStateEstimator()
        self.current_view_side = max(256, min(int(current_view_side), 768))
        self.temporal_view_side = max(256, min(int(temporal_view_side), 768))
        self.multiview_calls = 0
        self.last_perceptual_state: dict[str, Any] = {}
        self.exploration_frontier = ExplorationFrontier()
        self.frontier_fallback_actions = 0

    def on_level_reset(self) -> None:
        super().on_level_reset()
        self.perceptual_estimator.reset_temporal()
        self.last_perceptual_state = {}

    def observe(self, frame: Any) -> Scene:
        before_scene = self.scenes[-1] if self.scenes else None
        action = self.last_action
        scene = super().observe(frame)
        if before_scene is not None and action is not None and self.memory.transitions:
            latest = self.memory.transitions[-1]
            effect = self.affordances.coarse_effect(
                before_scene, scene,
                level_completed=bool(latest.level_completed),
                game_over=bool(latest.game_over),
            )
            self.exploration_frontier.observe_transition(
                before_scene, action, scene, effect=effect,
                game_over=bool(latest.game_over),
                level_completed=bool(latest.level_completed),
            )
        self.last_perceptual_state = self.perceptual_estimator.summarize(
            scene,
            self.visual_tracker,
            self.spatial_control,
            self.affordances,
            self.visual_beliefs,
            recent_grids=self.grids,
            last_action=self.last_action,
            prediction_mismatch=bool(self.prediction_mismatch),
        )
        return scene

    def _orientation_state(self, scene: Scene) -> dict[str, Any]:
        if not self.last_perceptual_state:
            self.last_perceptual_state = self.perceptual_estimator.summarize(
                scene,
                self.visual_tracker,
                self.spatial_control,
                self.affordances,
                self.visual_beliefs,
                recent_grids=self.grids,
                last_action=self.last_action,
                prediction_mismatch=bool(self.prediction_mismatch),
            )
        return {
            "phase": self.last_perceptual_state.get("recommended_mode", "MODEL_AND_TEST"),
            "orientation_entropy": self.last_perceptual_state.get("orientation_entropy", 1.0),
            "dominant_uncertainty": self.last_perceptual_state.get("dominant_uncertainty", "unknown"),
            "uncertainty": self.last_perceptual_state.get("uncertainty", {}),
            "planner_ready": self.spatial_control.planner_ready(scene),
        }

    def _visual_context(self, scene: Scene, candidates: list[DecisionCandidate]) -> dict[str, Any]:
        # Normal execution reaches this method only after observe(), which populates self.grids.
        # Keep the helper robust for direct diagnostic use by seeding the current scene if a
        # caller asks for visual context before the first observed frame.
        if not self.grids:
            self.grids.append(scene.grid.copy())
        context = super()._visual_context(scene, candidates)
        records = []
        for candidate in sorted(candidates, key=lambda c: c.candidate_id)[: self.candidate_limit]:
            row = candidate.record()
            row.pop("score", None)
            records.append(row)
        context["decision_candidates"] = records
        context["exploration_graph"] = self.exploration_frontier.summary(scene, candidates)
        context["perceptual_state"] = self.last_perceptual_state or self.perceptual_estimator.summarize(
            scene,
            self.visual_tracker,
            self.spatial_control,
            self.affordances,
            self.visual_beliefs,
            recent_grids=self.grids,
            last_action=self.last_action,
            prediction_mismatch=bool(self.prediction_mismatch),
        )
        return context

    def _should_reason(self, scene: Scene) -> bool:
        if self.model is None or not self._model_budget_available():
            return False
        if self.prediction_mismatch:
            return True
        if self.last_perceptual_state:
            entropy = float(self.last_perceptual_state.get("orientation_entropy", 1.0))
            mode = str(self.last_perceptual_state.get("recommended_mode", ""))
            if entropy >= 0.52 or mode in {"IDENTIFY_AGENCY", "IDENTIFY_GOAL", "DISCRIMINATE_DYNAMICS", "REASON_ABOUT_PHASE", "REPAIR_MODEL"}:
                return True
        return super()._should_reason(scene)

    def choose(self, scene: Scene) -> ActionSpec:
        if self.action_queue:
            queued = self.action_queue[0]
            if id(queued) in self._planned_steps and not self._queued_spatial_step_safe(scene, queued):
                self.spatial_plan_mismatches += 1
                self.action_queue.clear()
                self._planned_steps.clear()
                self.last_reason_step = -10_000
                self._remember(
                    "Spatial route became unsafe under current geometry; re-plan before acting.",
                    0.98,
                    source="spatial_preaction_guard",
                )

        while self.action_queue:
            spec = self.action_queue.pop(0)
            if spec.action_id in scene.available_actions:
                if spec.action_id != 6 or (
                    spec.x is not None and spec.y is not None
                    and 0 <= spec.x < scene.grid.shape[1]
                    and 0 <= spec.y < scene.grid.shape[0]
                ):
                    self.last_action, self.last_target_shape = spec, None
                    self.queued_actions_used += 1
                    self._arm_prediction(scene, spec)
                    return spec

        model_actions = self._reason(scene)
        if model_actions and model_actions[0].confidence >= 0.35:
            first, rest = model_actions[0], model_actions[1:]
            self.action_queue.extend(rest)
            self.last_action, self.last_target_shape = first, None
            self._arm_prediction(scene, first)
            return first

        candidates = enumerate_decision_candidates(
            scene, self.spatial_control, self.visual_tracker, self.affordances, self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        candidate, why = self.exploration_frontier.fallback_candidate(scene, candidates)
        if candidate is not None and candidate.spec is not None:
            delegated = self._candidate_action(
                {
                    "candidate_id": candidate.candidate_id,
                    "confidence": 0.42,
                    "hypothesis": f"systematic exploration frontier: {why}",
                },
                scene,
                candidates,
            )
            if delegated:
                spec = delegated[0]
                self.frontier_fallback_actions += 1
                self.fallback_actions += 1
                self.last_action, self.last_target_shape = spec, None
                self._arm_prediction(scene, spec)
                return spec

        from arc3lab.policy.effect_posterior import EffectPosteriorPolicy
        self.fallback_actions += 1
        spec = EffectPosteriorPolicy.choose(self, scene)
        self._arm_prediction(scene, spec)
        return spec

    def _call_model_with_packet(self, user: str, scene: Scene, packet) -> dict[str, Any] | None:
        if self.model is None or not self._model_budget_available():
            return None
        try:
            from arc3lab.model.adapter import extract_json
            self.model_calls += 1
            self.visual_packet_calls += 1
            self.multiview_calls += 1
            views = {
                "views": [
                    {"label": "CURRENT HIGH-RESOLUTION BOARD", "grid": scene.grid, "side": self.current_view_side},
                    {"label": "TEMPORAL CONTEXT: t-2 | t-1 | current | delta", "grid": packet, "side": self.temporal_view_side},
                ]
            }
            text = self.model.complete(
                self._system_prompt(),
                user + "\n\n" + PERCEPTUAL_INSTRUCTIONS,
                grid=views,
            )
            return extract_json(text)
        except Exception:
            self.model_failures += 1
            return None

    @staticmethod
    def _system_prompt() -> str:
        from arc3lab.policy.coding_prompt import CODING_SYSTEM_PROMPT
        return CODING_SYSTEM_PROMPT

    def _sandbox_context(self, scene: Scene) -> dict[str, Any]:
        context = super()._sandbox_context(scene)
        perceptual = self.last_perceptual_state or self.perceptual_estimator.summarize(
            scene,
            self.visual_tracker,
            self.spatial_control,
            self.affordances,
            self.visual_beliefs,
            recent_grids=self.grids,
            last_action=self.last_action,
            prediction_mismatch=bool(self.prediction_mismatch),
        )

        def retrodict_action(action_id: int) -> dict[str, Any]:
            aid = int(action_id)
            transitions = []
            for t in self.memory.transitions:
                if int(t.action.action_id) != aid:
                    continue
                transitions.append({
                    "step": int(t.step),
                    "level": int(t.level),
                    "before": t.before_signature,
                    "after": t.after_signature,
                    "changed": int(t.meaningful_changed_cells),
                    "level_completed": bool(t.level_completed),
                    "game_over": bool(t.game_over),
                })
            events = [e for e in self.visual_tracker.recent_events if e.get("action") == aid]
            return {
                "action": aid,
                "effect_posterior": self.affordances.action_summary([aid]).get(str(aid), {}),
                "transitions": transitions[-24:],
                "visual_events": events[-24:],
            }

        candidates = enumerate_decision_candidates(
            scene, self.spatial_control, self.visual_tracker, self.affordances, self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )

        def exploration_graph() -> dict[str, Any]:
            return self.exploration_frontier.summary(scene, candidates)

        context.update({
            "perceptual_state": perceptual,
            "retrodict_action": retrodict_action,
            "exploration_graph": exploration_graph,
        })
        return context
