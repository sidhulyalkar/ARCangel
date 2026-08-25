from __future__ import annotations

import json
from collections import Counter
from typing import Any

from arc3lab.memory.hypothesis import HypothesisRegistry
from arc3lab.perception.scene import compact_scene, grid_ascii
from arc3lab.perception.spatial import spatial_summary
from arc3lab.planning.counterfactual import DecisionCandidate, enumerate_decision_candidates
from arc3lab.policy.perceptual_coding import PERCEPTUAL_INSTRUCTIONS, PerceptualDecisionPolicy
from arc3lab.policy.sandbox import SandboxError, run_analysis_code
from arc3lab.policy.spatial_coding import SPATIAL_INSTRUCTIONS
from arc3lab.policy.visual_coding import VISUAL_INSTRUCTIONS
from arc3lab.types import ActionSpec, Scene


ACTIVE_SCIENTIST_SYSTEM_PROMPT = """You are ARCangel V010, an active scientist in a novel ARC-AGI-3 world.
Infer objects, agency, mechanics, hidden state and the goal from pixels and consequences. Never
assume familiar game identity, action meanings, semantic colors or public solutions.

Preserve uncertainty. Separate confidence in your theory from confidence that an action is the
best next experiment. Spend real actions only on progress, pivotal information, or verified
execution. Use Python/history when it can answer a question without a real action. Build a model
only when it explains relevant prior transitions.

Return exactly one JSON object:
{
 "orientation":{"controlled_entity":"...","important_changes":[],"salient_objects":[],
                "dominant_uncertainty":"..."},
 "hypotheses":{"agency":[],"mechanics":[],"goals":[],"abstraction":[]},
 "visual_goals":[{"target_object":0,"relation":"touch|adjacent8|overlap|center|inside|click|remove|transform|unknown",
                  "confidence":0.0,"evidence":"..."}],
 "decision_mode":"ACT_DIRECTLY|TEST_HYPOTHESIS|EXPLORE_FRONTIER|QUERY_HISTORY|BUILD_MODEL|EXECUTE_VERIFIED_PLAN|REPAIR_MODEL",
 "candidate_id":"",
 "actions":[{"id":1,"x":null,"y":null}],
 "hypothesis_confidence":0.0,
 "action_confidence":0.0,
 "experiment":{"question":"...","distinguishes":[],"expected_outcomes":[]},
 "python":"","analysis_question":"","delegate_world_model":false,"plan_reliable":false,
 "expected_change":"...","memory_note":"","goal":"...","hypothesis":"..."
}

Each hypothesis row is {"statement":"...","confidence":0..1,"evidence":"..."}.
`hypothesis_confidence` may be low while `action_confidence` is high if one experiment is clearly
best. It can be high even when hypothesis_confidence is low. Prefer exact candidate_id when possible. If EXPLORE_FRONTIER is best, choose it deliberately.
Do not leave candidate_id/actions empty unless Python is non-empty.
"""

ACTIVE_USER_TEMPLATE = """CURRENT SCENE
{scene}
MEMORY
{memory}
TRANSFERABLE BELIEFS
{beliefs}
GOAL NOTES
{goals}
TEMPORAL PREDICTIVE MEMORY
{predictive}
LOSSLESS GRID
{ascii_grid}
VALID ACTION IDS: {valid_actions}

Python can inspect the complete ledger, recent frames, components, visual tracks, affordances,
typed hypotheses, retrodict_action(action_id), exploration_graph(), spatial helpers, and history
queries. It cannot take a real environment action.
Use the exact V010 DECISION SCHEMA from the system prompt.
"""

MODES = {
    "ACT_DIRECTLY", "TEST_HYPOTHESIS", "EXPLORE_FRONTIER", "QUERY_HISTORY",
    "BUILD_MODEL", "EXECUTE_VERIFIED_PLAN", "REPAIR_MODEL",
}


class ActiveScientistPolicy(PerceptualDecisionPolicy):
    """V010: semantic scientist control with uncertainty-specific experimental fallback."""

    def __init__(self, *args: Any, min_action_confidence: float = 0.30, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.min_action_confidence = max(0.0, min(1.0, float(min_action_confidence)))
        self.hypothesis_registry = HypothesisRegistry()
        self.model_parse_successes = self.model_parse_contract_errors = 0
        self.semantic_actions = self.semantic_candidate_actions = self.semantic_direct_actions = 0
        self.model_directed_frontier_actions = self.emergency_fallback_actions = 0
        self.low_action_confidence_rejections = self.goal_proposals = 0
        self.typed_hypothesis_updates = self.typed_hypothesis_contradictions = 0
        self.model_decision_modes: Counter[str] = Counter()
        self.fallback_modes: Counter[str] = Counter()
        self.last_decision_trace: dict[str, Any] = {}
        self._seen_visual_mismatch = self._seen_predictive_mismatch = 0

    @staticmethod
    def _system_prompt() -> str:
        return ACTIVE_SCIENTIST_SYSTEM_PROMPT

    @staticmethod
    def _conf(v: Any, default: float = 0.5) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return default

    def observe(self, frame: Any) -> Scene:
        scene = super().observe(frame)
        visual = int(self.visual_expectation_mismatches)
        predictive = int(self.prediction_mismatch_count)
        if visual > self._seen_visual_mismatch or predictive > self._seen_predictive_mismatch:
            self.typed_hypothesis_contradictions += self.hypothesis_registry.contradict_recent(
                level=scene.level, step=self.step
            )
        self._seen_visual_mismatch, self._seen_predictive_mismatch = visual, predictive
        return scene

    def _normalize(self, parsed: dict[str, Any]) -> dict[str, Any]:
        out = dict(parsed)
        h = self._conf(out.get("hypothesis_confidence", out.get("confidence", 0.5)))
        a = self._conf(out.get("action_confidence", out.get("confidence", 0.5)))
        mode = str(out.get("decision_mode", "")).upper().strip()
        aliases = {
            "IDENTIFY_AGENCY": "TEST_HYPOTHESIS", "IDENTIFY_GOAL": "TEST_HYPOTHESIS",
            "DISCRIMINATE_DYNAMICS": "TEST_HYPOTHESIS", "REASON_ABOUT_PHASE": "TEST_HYPOTHESIS",
            "MODEL_AND_TEST": "TEST_HYPOTHESIS", "PLAN_AND_EXECUTE": "EXECUTE_VERIFIED_PLAN",
        }
        contract_error = any(k not in out for k in ("decision_mode", "hypothesis_confidence", "action_confidence"))
        if mode not in MODES:
            mode = aliases.get(mode, "TEST_HYPOTHESIS")
            contract_error = True
        has_intent = bool(str(out.get("candidate_id", "")).strip()) or bool(out.get("actions")) or bool(str(out.get("python", "")).strip())
        contract_error = contract_error or not has_intent
        self.model_parse_contract_errors += int(contract_error)
        out.update(hypothesis_confidence=h, action_confidence=a, confidence=a, decision_mode=mode, contract_valid=not contract_error)
        return out

    def _remember_active(self, parsed: dict[str, Any], scene: Scene) -> None:
        memory_view = dict(parsed)
        memory_view["confidence"] = parsed["hypothesis_confidence"]
        self._remember_visual_parse(memory_view, scene)
        self.typed_hypothesis_updates += self.hypothesis_registry.update_from_model(
            parsed.get("hypotheses"), level=scene.level, step=self.step
        )
        if isinstance(parsed.get("visual_goals"), list):
            self.goal_proposals += sum(isinstance(x, dict) for x in parsed["visual_goals"])
        self.model_decision_modes[parsed["decision_mode"]] += 1

    def _visual_context(self, scene: Scene, candidates: list[DecisionCandidate]) -> dict[str, Any]:
        ctx = super()._visual_context(scene, candidates)
        ctx["typed_hypotheses"] = self.hypothesis_registry.summary()
        return ctx

    def _sandbox_context(self, scene: Scene) -> dict[str, Any]:
        ctx = super()._sandbox_context(scene)
        ctx["typed_hypotheses"] = self.hypothesis_registry.summary()
        return ctx

    def _trace(self, **kw: Any) -> None:
        self.last_decision_trace = {"step": self.step, "level": self.level, **kw}

    def _model_action(
        self, parsed: dict[str, Any], scene: Scene, candidates: list[DecisionCandidate]
    ) -> list[ActionSpec]:
        if parsed["action_confidence"] < self.min_action_confidence:
            self.low_action_confidence_rejections += 1
            self._trace(accepted=False, model_action_accepted=False, reason="low_action_confidence",
                        mode=parsed["decision_mode"], model_decision_mode=parsed["decision_mode"], action_confidence=parsed["action_confidence"])
            return []
        selected = self._candidate_action(parsed, scene, candidates)
        source = "candidate"
        if not selected:
            selected = self._compile_requested_plan(parsed, scene)
            source = "plan"
        if not selected:
            selected = self._parse_actions(parsed, scene)
            source = "direct"
        if not selected:
            self._trace(accepted=False, model_action_accepted=False, reason="no_executable_action",
                        mode=parsed["decision_mode"], model_decision_mode=parsed["decision_mode"], action_confidence=parsed["action_confidence"])
            return []
        self.semantic_actions += 1
        self.semantic_candidate_actions += int(source == "candidate")
        self.semantic_direct_actions += int(source == "direct")
        self.model_directed_frontier_actions += int(parsed["decision_mode"] == "EXPLORE_FRONTIER")
        self._trace(accepted=True, model_action_accepted=True, source=source, mode=parsed["decision_mode"], model_decision_mode=parsed["decision_mode"],
                    hypothesis_confidence=parsed["hypothesis_confidence"],
                    action_confidence=parsed["action_confidence"],
                    candidate_id=str(parsed.get("candidate_id", "")),
                    action_id=int(selected[0].action_id),
                    experiment=parsed.get("experiment", {}))
        return selected

    def _model_action_from_parse(
        self, parsed: dict[str, Any], scene: Scene, candidates: list[DecisionCandidate]
    ) -> list[ActionSpec]:
        """Compatibility/test alias for the explicit V010 semantic-action gate."""
        return self._model_action(parsed, scene, candidates)

    def _reason(self, scene: Scene) -> list[ActionSpec]:
        if not self._should_reason(scene):
            return []
        self.reasoning_cycles += 1
        self.last_reason_step, self.last_reason_level = self.step, scene.level
        candidates = enumerate_decision_candidates(
            scene, self.spatial_control, self.visual_tracker, self.affordances, self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        visual = self._visual_context(scene, candidates)
        packet = visual.pop("packet")
        user = ACTIVE_USER_TEMPLATE.format(
            scene=json.dumps(compact_scene(scene), separators=(",", ":")),
            memory=json.dumps(self.memory.compact(), separators=(",", ":")),
            beliefs=json.dumps(self.beliefs[-12:], separators=(",", ":")),
            goals=json.dumps(self.goals[-8:], separators=(",", ":")),
            predictive=json.dumps(self.predictive.summary(), separators=(",", ":")),
            ascii_grid=grid_ascii(scene.grid), valid_actions=list(scene.available_actions),
        )
        user += "\nSPATIAL EVIDENCE\n" + SPATIAL_INSTRUCTIONS + "\n" + json.dumps(
            spatial_summary(scene, self.spatial_control), separators=(",", ":")
        )
        user += "\nVISUAL/TEMPORAL EVIDENCE\n" + VISUAL_INSTRUCTIONS + "\n" + json.dumps(
            visual, separators=(",", ":")
        )
        user += "\n" + PERCEPTUAL_INSTRUCTIONS
        parsed = self._call_model_with_packet(user, scene, packet)
        if not isinstance(parsed, dict):
            self._trace(accepted=False, reason="parse_failure")
            return []
        self.model_parse_successes += 1
        parsed = self._normalize(parsed)
        self._remember_active(parsed, scene)

        program = parsed.get("python")
        if isinstance(program, str) and program.strip() and self._tool_budget_available() and self._model_budget_available():
            self.world_model_delegations += int(bool(parsed.get("delegate_world_model", False)))
            self.tool_calls += 1
            try:
                result = run_analysis_code(program, self._sandbox_context(scene))
            except (SandboxError, SyntaxError, ValueError, TypeError) as exc:
                self.tool_failures += 1
                result = f"TOOL_ERROR {type(exc).__name__}: {exc}"
            follow = user + "\nPYTHON RESULT\n" + str(result) + \
                "\nEnvironment unchanged. Return the final V010 JSON decision."
            second = self._call_model_with_packet(follow, scene, packet)
            if isinstance(second, dict):
                self.model_parse_successes += 1
                parsed = self._normalize(second)
                self._remember_active(parsed, scene)

        candidates = enumerate_decision_candidates(
            scene, self.spatial_control, self.visual_tracker, self.affordances, self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        return self._model_action(parsed, scene, candidates)

    @staticmethod
    def _risk(c: DecisionCandidate) -> tuple[float, float, float]:
        p = c.payload if isinstance(c.payload, dict) else {}
        q = p.get("posterior") if isinstance(p.get("posterior"), dict) else p
        return (
            float(q.get("game_over_probability", p.get("game_over_probability", 0.0)) or 0.0),
            float(q.get("dead_probability", p.get("dead_probability", 0.0)) or 0.0),
            float(q.get("information_value", p.get("information_value", 0.0)) or 0.0),
        )

    def _mode_fallback(
        self, scene: Scene, candidates: list[DecisionCandidate]
    ) -> tuple[DecisionCandidate | None, str]:
        mode = str((self.last_perceptual_state or {}).get("recommended_mode", "MODEL_AND_TEST"))
        primitives = [c for c in candidates if c.spec is not None and int(c.spec.action_id) != 6]

        if mode == "IDENTIFY_AGENCY" and primitives:
            repeat = []
            for c in primitives:
                m = c.payload.get("movement") if isinstance(c.payload, dict) else None
                if isinstance(m, dict) and 1 <= int(m.get("support", 0) or 0) <= 2 and self._risk(c)[0] <= .25:
                    repeat.append((-float(m.get("confidence", 0.0)), c.candidate_id, c))
            if repeat:
                return min(repeat)[-1], "agency_replicate_motion"
            return min(primitives, key=lambda c: (self._risk(c)[0],
                                                   int(c.payload.get("posterior", {}).get("support", 0)),
                                                   -self._risk(c)[2], c.candidate_id)), "agency_low_risk_probe"

        if mode == "REASON_ABOUT_PHASE" and primitives:
            dead = []
            for c in primitives:
                q = c.payload.get("posterior", {})
                if int(q.get("support", 0)) >= 1 and q.get("counts", {}).get("dead", 0):
                    dead.append((self._risk(c)[0], -self._risk(c)[1], c.candidate_id, c))
            if dead:
                return min(dead)[-1], "phase_quasi_noop"

        if mode == "REPAIR_MODEL" and self.last_action is not None:
            for c in candidates:
                if c.spec is None or int(c.spec.action_id) != int(self.last_action.action_id) or self._risk(c)[0] > .25:
                    continue
                if int(c.spec.action_id) == 6 and (c.spec.x, c.spec.y) != (self.last_action.x, self.last_action.y):
                    continue
                return c, "repair_retest"

        if mode in {"IDENTIFY_GOAL", "DISCRIMINATE_DYNAMICS", "MODEL_AND_TEST"}:
            viable = [c for c in candidates if c.spec is not None]
            if viable:
                return min(viable, key=lambda c: (self._risk(c)[0], self._risk(c)[1],
                                                  -self._risk(c)[2], c.candidate_id)), \
                       f"{mode.lower()}_information_probe"

        return self.exploration_frontier.fallback_candidate(scene, candidates)

    def choose(self, scene: Scene) -> ActionSpec:
        if self.action_queue and id(self.action_queue[0]) in self._planned_steps and \
                not self._queued_spatial_step_safe(scene, self.action_queue[0]):
            self.spatial_plan_mismatches += 1
            self.action_queue.clear()
            self._planned_steps.clear()
            self.last_reason_step = -10_000

        while self.action_queue:
            spec = self.action_queue.pop(0)
            if spec.action_id in scene.available_actions and (
                spec.action_id != 6 or (
                    spec.x is not None and spec.y is not None
                    and 0 <= spec.x < scene.grid.shape[1] and 0 <= spec.y < scene.grid.shape[0]
                )
            ):
                self.last_action, self.last_target_shape = spec, None
                self.queued_actions_used += 1
                self._arm_prediction(scene, spec)
                return spec

        actions = self._reason(scene)
        if actions:
            first, rest = actions[0], actions[1:]
            self.action_queue.extend(rest)
            self.last_action, self.last_target_shape = first, None
            self._arm_prediction(scene, first)
            return first

        candidates = enumerate_decision_candidates(
            scene, self.spatial_control, self.visual_tracker, self.affordances, self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        candidate, why = self._mode_fallback(scene, candidates)
        if candidate is not None and candidate.spec is not None:
            spec = ActionSpec(int(candidate.spec.action_id), x=candidate.spec.x, y=candidate.spec.y,
                              reason=f"mode fallback: {why}", confidence=.40)
            mode = str((self.last_perceptual_state or {}).get("recommended_mode", "MODEL_AND_TEST"))
            self.fallback_modes[mode] += 1
            self.emergency_fallback_actions += 1
            self.fallback_actions += 1
            self.frontier_fallback_actions += int(why in {"local_untested_frontier", "known_safe_route_to_frontier"})
            self._trace(accepted=False, fallback=True, mode=mode, reason=why, action_id=spec.action_id)
            self.last_action, self.last_target_shape = spec, None
            self._arm_prediction(scene, spec)
            return spec

        from arc3lab.policy.effect_posterior import EffectPosteriorPolicy
        self.emergency_fallback_actions += 1
        self.fallback_actions += 1
        self.fallback_modes["EFFECT_POSTERIOR"] += 1
        spec = EffectPosteriorPolicy.choose(self, scene)
        self._arm_prediction(scene, spec)
        return spec

    def semantic_telemetry(self) -> dict[str, Any]:
        return {
            "model_parse_successes": self.model_parse_successes,
            "model_parse_contract_errors": self.model_parse_contract_errors,
            "semantic_actions": self.semantic_actions,
            "semantic_candidate_actions": self.semantic_candidate_actions,
            "semantic_direct_actions": self.semantic_direct_actions,
            "model_directed_frontier_actions": self.model_directed_frontier_actions,
            "emergency_fallback_actions": self.emergency_fallback_actions,
            "low_action_confidence_rejections": self.low_action_confidence_rejections,
            "goal_proposals": self.goal_proposals,
            "typed_hypothesis_updates": self.typed_hypothesis_updates,
            "typed_hypothesis_contradictions": self.typed_hypothesis_contradictions,
            "typed_hypothesis_count": len(self.hypothesis_registry),
            "model_decision_modes": dict(self.model_decision_modes),
            "fallback_modes": dict(self.fallback_modes),
            "last_decision_trace": dict(self.last_decision_trace),
        }
