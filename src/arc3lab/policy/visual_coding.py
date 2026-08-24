from __future__ import annotations

import json
from typing import Any

from arc3lab.memory.affordance import AffordanceMemory
from arc3lab.memory.visual_belief import VisualBeliefState
from arc3lab.model.adapter import extract_json
from arc3lab.perception.scene import compact_scene, grid_ascii
from arc3lab.perception.spatial import spatial_summary
from arc3lab.perception.visual import VisualTracker, temporal_visual_packet
from arc3lab.planning.counterfactual import (
    DecisionCandidate,
    candidate_records,
    enumerate_decision_candidates,
)
from arc3lab.policy.coding_prompt import CODING_SYSTEM_PROMPT, CODING_USER_TEMPLATE, TOOL_RESULT_TEMPLATE
from arc3lab.policy.sandbox import SandboxError, run_analysis_code
from arc3lab.policy.spatial_coding import SPATIAL_INSTRUCTIONS, SpatialCodingPolicy
from arc3lab.types import ActionSpec, Scene


VISUAL_INSTRUCTIONS = """
TEMPORAL VISUAL COGNITION
The image attached to this turn is not only the current board. It is a 2x2 temporal packet:
- top-left: t-2
- top-right: t-1
- bottom-left: CURRENT board
- bottom-right: diagnostic delta from t-1 to current. Unchanged cells use the current
  background; changed cells retain their new game color; cells that vanished are shown with
  a diagnostic marker color. The delta quadrant is evidence, not part of the game world.

Use the packet to establish object permanence, motion, causality and visual progress before
choosing an action. Do not infer mechanics from one static screenshot when temporal evidence
exists.

`visual.tracked_objects` gives persistent track IDs across frames. Object indices remain local
to the current frame. `visual.recent_events` records appearances, disappearances, translations
and mild transformations. `affordances` gives empirical action-effect posteriors. `visual_goals`
contains persistent executable goal hypotheses proposed on prior turns/levels.

`decision_candidates` is an exact registry of legal primitive actions, object-grounded ACTION6
clicks, and (when sufficiently supported) exact spatial plans. Its numeric score is only an
advisory causal prior combining observed progress, information value and risk. It is NOT a
hard-coded game rule. Prefer choosing a `candidate_id` when one matches your visual reasoning;
you may reject the ranking when the image provides stronger evidence.

First orient, then act. Separate these questions:
1. What is visually persistent and what just changed?
2. Which object, if any, is under causal control?
3. What future visual relation appears to represent progress/winning?
4. Which uncertainty blocks a confident plan?
5. Which legal candidate best trades off goal progress, information gain and irreversible risk?

Before choosing, populate `orientation` with the most important perceptual facts you
actually infer from the temporal packet: the likely controlled object/track (or uncertain),
what changed since t-1, which objects look causally important, the strongest progress cue,
and the single uncertainty that most blocks a plan. Do not fill fields with guesses merely
to satisfy the schema.

Return optional `visual_goals` with up to three falsifiable hypotheses:
  {"target_object": <current object index>, "relation": "touch|adjacent8|overlap|center|inside|click|remove|transform|unknown", "confidence": 0..1, "evidence": "brief visual evidence"}

Return optional `candidate_id` from the supplied candidate registry. If you choose a spatial
plan candidate, ARCangel still verifies causal control and geometry before compiling it.
When uncertain, take ONE discriminating action rather than a speculative sequence.
"""


class VisualDecisionPolicy(SpatialCodingPolicy):
    """V008 vision-first policy with temporal tracking and counterfactual selection.

    Qwen is used for semantic orientation and goal hypotheses. Code maintains temporal
    object identity, empirical affordances, exact legal candidates and spatial plans.
    """

    def __init__(
        self,
        *args: Any,
        max_click_candidates: int = 14,
        candidate_limit: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.visual_tracker = VisualTracker()
        self.affordances = AffordanceMemory()
        self.visual_beliefs = VisualBeliefState()
        self.max_click_candidates = max(4, min(int(max_click_candidates), 32))
        self.candidate_limit = max(8, min(int(candidate_limit), 40))
        self.visual_packet_calls = 0
        self.visual_candidate_selections = 0
        self.visual_goal_updates = 0
        self.visual_affordance_observations = 0
        self.last_orientation: dict[str, Any] = {}
        self.last_decision_mode = "unknown"
        self.visual_expectation_mismatches = 0
        self._candidate_expectations: dict[int, dict[str, Any]] = {}

    def on_level_reset(self) -> None:
        super().on_level_reset()
        # Preserve learned affordances and goal signatures, but sever frame association.
        self.visual_tracker.reset_link()
        self.last_orientation = {}
        self.last_decision_mode = "unknown"
        self._candidate_expectations.clear()

    def observe(self, frame: Any) -> Scene:
        before_scene = self.scenes[-1] if self.scenes else None
        previous_level = self.level
        action = self.last_action
        visual_pending = dict(self.pending_transition) if self.pending_transition is not None else None
        actor_before = (
            self.spatial_control.actor_hypothesis(before_scene)
            if before_scene is not None
            else None
        )

        scene = super().observe(frame)
        level_changed = scene.level > previous_level

        if before_scene is not None and action is not None and self.memory.transitions:
            latest = self.memory.transitions[-1]
            observed_effect = self.affordances.observe(
                before_scene,
                scene,
                action,
                level_completed=bool(latest.level_completed),
                game_over=bool(latest.game_over),
            )
            self.visual_affordance_observations += 1
            if visual_pending is not None:
                expected_effect = visual_pending.get("visual_expected_effect")
                expected_center = visual_pending.get("visual_expected_actor_center")
                mismatch = False
                if expected_effect is not None and str(expected_effect) != str(observed_effect):
                    mismatch = True
                if expected_center is not None:
                    actor_now = self.spatial_control.actor_hypothesis(scene)
                    if actor_now is not None and tuple(actor_now["center"]) != tuple(expected_center):
                        mismatch = True
                if mismatch:
                    self.visual_expectation_mismatches += 1
                    self.action_queue.clear()
                    self._planned_steps.clear()
                    self.last_reason_step = -10_000
                    self._remember(
                        "Visual counterfactual contradiction: the selected candidate's observed outcome did not match its supported expectation; re-orient before continuing.",
                        0.97,
                        source="visual_counterfactual_error",
                    )
            if level_changed:
                actor_index = int(actor_before["index"]) if actor_before is not None else None
                self.visual_beliefs.validate_completion(before_scene, actor_index)

        self.visual_tracker.observe(
            scene,
            step=self.step,
            action=action,
            level_changed=level_changed,
        )
        return scene

    def _arm_prediction(self, scene: Scene, spec: ActionSpec) -> None:
        super()._arm_prediction(scene, spec)
        if self.pending_transition is not None:
            expectation = self._candidate_expectations.pop(id(spec), None)
            if expectation:
                self.pending_transition.update(expectation)

    def _orientation_state(self, scene: Scene) -> dict[str, Any]:
        actor = self.spatial_control.actor_hypothesis(scene)
        actor_conf = float(actor["confidence"]) if actor is not None else 0.0
        current_goals = self.visual_beliefs.top_current(scene, limit=4)
        goal_conf = max((float(g["confidence"]) for g in current_goals), default=0.0)
        vectors = self.spatial_control.action_vectors(scene, min_support=1)
        if self.prediction_mismatch:
            phase = "REPAIR"
        elif actor_conf < 0.55:
            phase = "ORIENT_CONTROL"
        elif goal_conf < 0.45:
            phase = "ORIENT_GOAL"
        elif self.spatial_control.planner_ready(scene) and goal_conf >= 0.68:
            phase = "PLAN_OR_EXECUTE"
        else:
            phase = "MODEL_AND_TEST"
        return {
            "phase": phase,
            "actor_confidence": round(actor_conf, 4),
            "goal_confidence": round(goal_conf, 4),
            "learned_motion_actions": len(vectors),
            "planner_ready": self.spatial_control.planner_ready(scene),
            "prediction_mismatch": bool(self.prediction_mismatch),
        }

    def _visual_context(
        self,
        scene: Scene,
        candidates: list[DecisionCandidate],
    ) -> dict[str, Any]:
        packet, packet_meta = temporal_visual_packet(
            self.grids[-3:],
            background=int(scene.background),
        )
        # packet itself is not serialized into text; metadata explains the attached image.
        return {
            "packet": packet,
            "packet_meta": packet_meta,
            "orientation": self._orientation_state(scene),
            "tracking": self.visual_tracker.summary(scene, limit=24),
            "affordances": self.affordances.action_summary(scene.available_actions),
            "visual_goals": self.visual_beliefs.summary(scene),
            "decision_candidates": candidate_records(candidates, limit=self.candidate_limit),
        }

    def _call_model_with_packet(
        self,
        user: str,
        scene: Scene,
        packet,
    ) -> dict[str, Any] | None:
        if self.model is None or not self._model_budget_available():
            return None
        try:
            self.model_calls += 1
            self.visual_packet_calls += 1
            text = self.model.complete(CODING_SYSTEM_PROMPT, user, grid=packet)
            return extract_json(text)
        except Exception:
            self.model_failures += 1
            return None

    def _sandbox_context(self, scene: Scene) -> dict[str, Any]:
        context = super()._sandbox_context(scene)
        candidates = enumerate_decision_candidates(
            scene,
            self.spatial_control,
            self.visual_tracker,
            self.affordances,
            self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )

        def visual_track(object_index: int) -> dict[str, Any] | None:
            i = int(object_index)
            tid = self.visual_tracker.track_for_component(i)
            if tid is None:
                return None
            track = self.visual_tracker.tracks.get(tid)
            if track is None:
                return None
            return {
                "track": tid,
                "signature": list(track.signature),
                "age": track.age,
                "seen": track.seen,
                "motion_events": track.motion_events,
                "transform_events": track.transform_events,
                "last_center": list(track.last_center),
                "last_delta": list(track.last_delta),
            }

        def visual_candidates() -> list[dict[str, Any]]:
            return candidate_records(candidates, limit=self.candidate_limit)

        def click_candidates() -> list[dict[str, Any]]:
            return self.affordances.click_candidates(
                scene,
                self.visual_tracker,
                max_candidates=self.max_click_candidates,
            )

        context.update(
            {
                "visual": self.visual_tracker.summary(scene, limit=24),
                "visual_goals": self.visual_beliefs.summary(scene),
                "affordances": self.affordances.action_summary(scene.available_actions),
                "visual_track": visual_track,
                "visual_candidates": visual_candidates,
                "click_candidates": click_candidates,
            }
        )
        return context

    def _remember_visual_parse(self, parsed: dict[str, Any], scene: Scene) -> None:
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except Exception:
            confidence = 0.5
        self.last_hypothesis = str(parsed.get("hypothesis", self.last_hypothesis or "visual orientation"))[:500]
        self.last_expected_change = str(parsed.get("expected_change", self.last_expected_change))[:500]
        self._remember(str(parsed.get("memory_note", "")), confidence)
        self._remember_goal(str(parsed.get("goal", "")), confidence)
        orientation = parsed.get("orientation")
        if isinstance(orientation, dict):
            self.last_orientation = {
                str(k): v for k, v in list(orientation.items())[:12]
            }
        self.last_decision_mode = str(parsed.get("decision_mode", "unknown"))[:64]
        before = len(self.visual_beliefs.beliefs)
        self.visual_beliefs.update_from_model(
            parsed.get("visual_goals"),
            scene,
            self.visual_tracker,
            level=scene.level,
        )
        if len(self.visual_beliefs.beliefs) != before or isinstance(parsed.get("visual_goals"), list):
            self.visual_goal_updates += 1

    @staticmethod
    def _candidate_map(candidates: list[DecisionCandidate], limit: int) -> dict[str, DecisionCandidate]:
        return {c.candidate_id: c for c in candidates[:limit]}

    def _candidate_action(
        self,
        parsed: dict[str, Any],
        scene: Scene,
        candidates: list[DecisionCandidate],
    ) -> list[ActionSpec]:
        cid = str(parsed.get("candidate_id", "")).strip()
        if not cid:
            return []
        candidate = self._candidate_map(candidates, self.candidate_limit).get(cid)
        if candidate is None:
            return []
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except Exception:
            confidence = 0.5
        hypothesis = str(parsed.get("hypothesis", "visual candidate selection"))[:500]

        if candidate.kind == "spatial_plan":
            request = {
                "target_object": candidate.payload.get("target_object"),
                "relation": candidate.payload.get("relation", "touch"),
                "execute": True,
            }
            delegated = dict(parsed)
            delegated["spatial_plan"] = request
            compiled = self._compile_requested_plan(delegated, scene)
            if compiled:
                self.visual_candidate_selections += 1
                return compiled
            return []

        spec = candidate.spec
        if spec is None:
            return []
        if int(spec.action_id) not in scene.available_actions:
            return []
        selected = ActionSpec(
            int(spec.action_id),
            x=spec.x,
            y=spec.y,
            reason=f"{hypothesis} [candidate {cid}]",
            confidence=confidence,
        )
        expectation: dict[str, Any] = {}
        posterior = candidate.payload.get("posterior") if isinstance(candidate.payload, dict) else None
        if isinstance(posterior, dict):
            support = int(posterior.get("support", 0) or 0)
            counts = posterior.get("counts") or {}
            if support >= 2 and isinstance(counts, dict) and counts:
                dominant, n = max(counts.items(), key=lambda kv: kv[1])
                if int(n) / max(support, 1) >= 0.80:
                    expectation["visual_expected_effect"] = str(dominant)
        predicted_center = candidate.payload.get("predicted_actor_center") if isinstance(candidate.payload, dict) else None
        movement = candidate.payload.get("movement") if isinstance(candidate.payload, dict) else None
        if predicted_center is not None and isinstance(movement, dict):
            if float(movement.get("confidence", 0.0)) >= 0.80 and float(movement.get("purity", 0.0)) >= 0.66:
                expectation["visual_expected_actor_center"] = tuple(predicted_center)
        if expectation:
            self._candidate_expectations[id(selected)] = expectation
        self.visual_candidate_selections += 1
        return [selected]

    def _reason(self, scene: Scene) -> list[ActionSpec]:
        if not self._should_reason(scene):
            return []
        self.reasoning_cycles += 1
        self.last_reason_step = self.step
        self.last_reason_level = scene.level

        candidates = enumerate_decision_candidates(
            scene,
            self.spatial_control,
            self.visual_tracker,
            self.affordances,
            self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        visual_context = self._visual_context(scene, candidates)
        packet = visual_context.pop("packet")
        spatial = spatial_summary(scene, self.spatial_control)

        user = CODING_USER_TEMPLATE.format(
            scene=json.dumps(compact_scene(scene), separators=(",", ":")),
            memory=json.dumps(self.memory.compact(), separators=(",", ":")),
            beliefs=json.dumps(self.beliefs[-12:], separators=(",", ":")),
            goals=json.dumps(self.goals[-8:], separators=(",", ":")),
            predictive=json.dumps(self.predictive.summary(), separators=(",", ":")),
            ascii_grid=grid_ascii(scene.grid),
            valid_actions=list(scene.available_actions),
        )
        user += "\n\n" + SPATIAL_INSTRUCTIONS + "\n" + json.dumps(spatial, separators=(",", ":"))
        user += "\n\n" + VISUAL_INSTRUCTIONS + "\nVISUAL_STATE\n" + json.dumps(visual_context, separators=(",", ":"))
        user += "\n\nOUTPUT EXTENSIONS\nInclude optional keys `orientation`, `visual_goals`, `decision_mode`, and `candidate_id` as described above, while retaining the normal ARCangel JSON schema."

        parsed = self._call_model_with_packet(user, scene, packet)
        if not parsed:
            return []
        self._remember_visual_parse(parsed, scene)

        program = parsed.get("python")
        delegated = bool(parsed.get("delegate_world_model", False))
        if (
            isinstance(program, str)
            and program.strip()
            and self._tool_budget_available()
            and self._model_budget_available()
        ):
            if delegated:
                self.world_model_delegations += 1
            self.tool_calls += 1
            try:
                tool_result = run_analysis_code(program, self._sandbox_context(scene))
            except (SandboxError, SyntaxError, ValueError, TypeError) as exc:
                self.tool_failures += 1
                tool_result = f"TOOL_ERROR {type(exc).__name__}: {exc}"
            follow = user + "\n\n" + TOOL_RESULT_TEMPLATE.format(tool_result=tool_result)
            second = self._call_model_with_packet(follow, scene, packet)
            if second:
                parsed = second
                self._remember_visual_parse(parsed, scene)

        # Re-enumerate after visual goal updates so newly grounded goals can expose plans.
        candidates = enumerate_decision_candidates(
            scene,
            self.spatial_control,
            self.visual_tracker,
            self.affordances,
            self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        selected = self._candidate_action(parsed, scene, candidates)
        if selected:
            return selected

        compiled = self._compile_requested_plan(parsed, scene)
        if compiled:
            return compiled
        return self._parse_actions(parsed, scene)
