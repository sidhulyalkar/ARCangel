from __future__ import annotations

import json
from typing import Any

from arc3lab.model.adapter import extract_json
from arc3lab.perception.scene import compact_scene, grid_ascii
from arc3lab.planning.counterfactual import DecisionCandidate, enumerate_decision_candidates
from arc3lab.policy.active_scientist import ActiveScientistPolicy
from arc3lab.policy.sandbox import SandboxError, run_analysis_code
from arc3lab.types import ActionSpec, Scene


LEAN_SCIENTIST_SYSTEM_PROMPT = """You are ARCangel V011, a model-led scientist acting in a novel ARC-AGI-3 world.
There are no instructions and no assumed action meanings. Infer agency, mechanics and the goal from
pixels plus consequences, then act efficiently. Treat every real environment action as costly.

You receive two visual views, a lossless ASCII grid, a compact recent transition ledger, persistent
reflection memory, empirical action outcomes and an exact registry of legal candidates. The code
layer is a guardrail and calculator, not the semantic decision maker. Do not defer to a heuristic
just because it exists. Prefer your own visual/temporal interpretation when it is grounded.

Reason in this order:
1. What changed and what object, if any, is causally controlled?
2. What state appears desirable or completion-like, and what evidence supports that?
3. Which mechanics are already supported by repeated transitions?
4. Is the best next move progress, one discriminating experiment, or verified execution?
5. If a short plan is reliable, emit it. Otherwise emit one action.

Use QUERY_HISTORY/BUILD_MODEL only when computation can answer a concrete question without spending
a real action. If the semantic answer is clear, act directly. If you are uncertain, choose a
low-risk action that most sharply distinguishes plausible hypotheses.

Return exactly one JSON object with this contract:
{
  "orientation":{"controlled_entity":"","important_changes":[],"dominant_uncertainty":""},
  "hypotheses":{"agency":[],"mechanics":[],"goals":[],"abstraction":[]},
  "visual_goals":[],
  "decision_mode":"ACT_DIRECTLY|TEST_HYPOTHESIS|EXPLORE_FRONTIER|QUERY_HISTORY|BUILD_MODEL|EXECUTE_VERIFIED_PLAN|REPAIR_MODEL",
  "candidate_id":"",
  "actions":[{"id":1,"x":null,"y":null}],
  "hypothesis_confidence":0.0,
  "action_confidence":0.0,
  "experiment":{"question":"","distinguishes":[],"expected_outcomes":[]},
  "reflection":{"goal":"","rules":[],"avoid":[],"next_test":"","confidence":0.0},
  "python":"",
  "analysis_question":"",
  "delegate_world_model":false,
  "plan_reliable":false,
  "expected_change":"",
  "memory_note":"",
  "goal":"",
  "hypothesis":""
}

Hypothesis rows, when used, are {"statement":"...","confidence":0..1,"evidence":"..."}.
`action_confidence` is confidence that this is the best next action, not confidence that the whole
world model is correct. A high-value experiment can therefore have high action confidence while
hypothesis confidence is low. Prefer candidate_id when it precisely matches your intended action.
"""


class LeanReflectiveScientistPolicy(ActiveScientistPolicy):
    """V011: model-led control with compact persistent reflection and bounded reasoning cadence.

    V009/V010 retain rich exact perception and verification internally. V011 deliberately exposes
    a much smaller semantic packet to the model so model capability, not hand-authored arbitration,
    drives the next action. Expensive model calls are concentrated at decision boundaries rather
    than every frame once the interaction begins to stabilize.
    """

    def __init__(
        self,
        *args: Any,
        reasoning_interval: int = 3,
        bootstrap_reasoning_steps: int = 5,
        reflection_rule_limit: int = 10,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("min_action_confidence", 0.20)
        kwargs.setdefault("candidate_limit", 24)
        kwargs.setdefault("max_click_candidates", 18)
        super().__init__(*args, **kwargs)
        self.reasoning_interval = max(2, int(reasoning_interval))
        self.bootstrap_reasoning_steps = max(2, int(bootstrap_reasoning_steps))
        self.reflection_rule_limit = max(4, int(reflection_rule_limit))
        self.reflection: dict[str, Any] = {
            "goal": "",
            "rules": [],
            "avoid": [],
            "next_test": "",
            "confidence": 0.0,
        }
        self.reflection_updates = 0
        self.reasoning_gate_skips = 0

    @staticmethod
    def _system_prompt() -> str:
        return LEAN_SCIENTIST_SYSTEM_PROMPT

    @staticmethod
    def _clean_text(value: Any, limit: int = 420) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _update_reflection(self, parsed: dict[str, Any]) -> None:
        raw = parsed.get("reflection")
        if not isinstance(raw, dict):
            return
        changed = False
        goal = self._clean_text(raw.get("goal"))
        if goal:
            self.reflection["goal"] = goal
            changed = True
            self._remember_goal(goal, self._conf(raw.get("confidence"), parsed["hypothesis_confidence"]))

        for key, limit in (("rules", self.reflection_rule_limit), ("avoid", 8)):
            values = raw.get(key)
            if not isinstance(values, list):
                continue
            cleaned = [self._clean_text(v, 260) for v in values]
            cleaned = [v for v in cleaned if v]
            if cleaned:
                merged: list[str] = list(self.reflection.get(key, []))
                for value in cleaned:
                    if value.lower() not in {x.lower() for x in merged}:
                        merged.append(value)
                self.reflection[key] = merged[-limit:]
                changed = True

        next_test = self._clean_text(raw.get("next_test"), 320)
        if next_test:
            self.reflection["next_test"] = next_test
            changed = True
        self.reflection["confidence"] = round(
            self._conf(raw.get("confidence"), parsed["hypothesis_confidence"]), 3
        )
        if changed:
            self.reflection_updates += 1

    def _recent_outcomes(self, limit: int = 10) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for transition in self.memory.transitions[-max(1, int(limit)):]:
            action = transition.action
            rows.append({
                "step": int(getattr(transition, "step", 0)),
                "level": int(getattr(transition, "level", 0)),
                "action": int(action.action_id),
                "x": action.x,
                "y": action.y,
                "changed": int(getattr(transition, "meaningful_changed_cells", 0)),
                "level_completed": bool(getattr(transition, "level_completed", False)),
                "game_over": bool(getattr(transition, "game_over", False)),
            })
        return rows

    def _compact_actor(self, scene: Scene) -> dict[str, Any] | None:
        actor = self.spatial_control.actor_hypothesis(scene)
        if actor is None:
            return None
        return {
            "index": int(actor["index"]),
            "center": list(actor["center"]),
            "confidence": round(float(actor["confidence"]), 4),
        }

    def _should_reason(self, scene: Scene) -> bool:
        if self.model is None or not self._model_budget_available():
            return False
        if self.prediction_mismatch or self.stuck > 0 or scene.level != self.last_reason_level:
            return True

        since = self.step - self.last_reason_step
        if scene.level == 0 and self.step < self.bootstrap_reasoning_steps:
            return since >= 1

        actor = self._compact_actor(scene)
        actor_conf = float(actor["confidence"]) if actor is not None else 0.0
        goals = self.visual_beliefs.top_current(scene, limit=3)
        goal_conf = max((float(g.get("confidence", 0.0)) for g in goals), default=0.0)

        # Once bootstrap actions have been observed, high uncertainty no longer means
        # "ask the 27B model every frame". A short cadence gives the environment time to
        # provide causal evidence and protects the nine-hour Kaggle runtime budget.
        interval = 2 if actor_conf < 0.55 or goal_conf < 0.40 else self.reasoning_interval
        if self.semantic_actions == 0:
            interval = min(interval, 2)
        should = since >= interval
        self.reasoning_gate_skips += int(not should)
        return should

    def _call_model_with_packet(self, user: str, scene: Scene, packet: Any) -> dict[str, Any] | None:
        if self.model is None or not self._model_budget_available():
            return None
        try:
            self.model_calls += 1
            self.visual_packet_calls += 1
            self.multiview_calls += 1
            views = {
                "views": [
                    {"label": "CURRENT HIGH-RESOLUTION BOARD", "grid": scene.grid, "side": self.current_view_side},
                    {"label": "TEMPORAL CONTEXT: t-2 | t-1 | current | delta", "grid": packet, "side": self.temporal_view_side},
                ]
            }
            text = self.model.complete(self._system_prompt(), user, grid=views)
            return extract_json(text)
        except Exception:
            self.model_failures += 1
            return None

    def _lean_user_packet(
        self,
        scene: Scene,
        candidates: list[DecisionCandidate],
    ) -> tuple[str, Any]:
        visual = self._visual_context(scene, candidates)
        packet = visual.pop("packet")
        candidate_rows = list(visual.get("decision_candidates", []))[: self.candidate_limit]
        frontier = visual.get("exploration_graph", {})
        goals = self.visual_beliefs.top_current(scene, limit=4)
        payload = {
            "scene": compact_scene(scene),
            "actor": self._compact_actor(scene),
            "recent_outcomes": self._recent_outcomes(10),
            "reflection": self.reflection,
            "typed_hypotheses": self.hypothesis_registry.summary(),
            "empirical_action_outcomes": self.affordances.action_summary(scene.available_actions),
            "visual_goals": goals,
            "frontier": frontier,
            "legal_candidates": candidate_rows,
            "prediction": self.predictive.summary(),
        }
        user = (
            "LOSSLESS GRID\n"
            + grid_ascii(scene.grid)
            + "\nVALID ACTION IDS: "
            + json.dumps(list(scene.available_actions))
            + "\nCOMPACT EVIDENCE\n"
            + json.dumps(payload, separators=(",", ":"))
            + "\nChoose the next semantic action using the V011 JSON contract."
        )
        return user, packet

    def _reason(self, scene: Scene) -> list[ActionSpec]:
        if not self._should_reason(scene):
            return []
        self.reasoning_cycles += 1
        self.last_reason_step, self.last_reason_level = self.step, scene.level
        candidates = enumerate_decision_candidates(
            scene,
            self.spatial_control,
            self.visual_tracker,
            self.affordances,
            self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        user, packet = self._lean_user_packet(scene, candidates)
        parsed = self._call_model_with_packet(user, scene, packet)
        if not isinstance(parsed, dict):
            self._trace(accepted=False, reason="parse_failure")
            return []

        self.model_parse_successes += 1
        parsed = self._normalize(parsed)
        self._remember_active(parsed, scene)
        self._update_reflection(parsed)

        program = parsed.get("python")
        if (
            isinstance(program, str)
            and program.strip()
            and self._tool_budget_available()
            and self._model_budget_available()
        ):
            self.world_model_delegations += int(bool(parsed.get("delegate_world_model", False)))
            self.tool_calls += 1
            try:
                result = run_analysis_code(program, self._sandbox_context(scene))
            except (SandboxError, SyntaxError, ValueError, TypeError) as exc:
                self.tool_failures += 1
                result = f"TOOL_ERROR {type(exc).__name__}: {exc}"
            follow = (
                user
                + "\nPYTHON RESULT\n"
                + str(result)
                + "\nThe environment has not changed. Return one final V011 JSON decision."
            )
            second = self._call_model_with_packet(follow, scene, packet)
            if isinstance(second, dict):
                self.model_parse_successes += 1
                parsed = self._normalize(second)
                self._remember_active(parsed, scene)
                self._update_reflection(parsed)

        candidates = enumerate_decision_candidates(
            scene,
            self.spatial_control,
            self.visual_tracker,
            self.affordances,
            self.visual_beliefs,
            max_click_candidates=self.max_click_candidates,
        )
        return self._model_action(parsed, scene, candidates)

    def semantic_telemetry(self) -> dict[str, Any]:
        out = super().semantic_telemetry()
        out.update({
            "reflection_updates": int(self.reflection_updates),
            "reasoning_gate_skips": int(self.reasoning_gate_skips),
            "reflection": dict(self.reflection),
        })
        return out
