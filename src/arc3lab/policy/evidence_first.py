from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from arc3lab.model.adapter import extract_json
from arc3lab.perception.diffs import diff_summary
from arc3lab.perception.scene import grid_ascii
from arc3lab.policy.coding import CodingPolicy
from arc3lab.policy.evidence_prompt import (
    EVIDENCE_FIRST_SYSTEM_PROMPT,
    EVIDENCE_FIRST_USER_TEMPLATE,
    TOOL_FOLLOWUP_TEMPLATE,
)
from arc3lab.policy.evidence_workspace import EvidenceWorkspace
from arc3lab.policy.sandbox import SandboxError, run_analysis_code
from arc3lab.types import ActionSpec, Scene, Transition


@dataclass(slots=True)
class PlannedAction:
    spec: ActionSpec
    expect: dict[str, Any]
    supports: tuple[str, ...] = ()


class EvidenceFirstCodingPolicy(CodingPolicy):
    """V012: evidence-first model authority with falsification-tested plans.

    This policy deliberately bypasses the inherited heuristic/candidate fallback path.
    ARCangel's deterministic perception and spatial code remains available inside the
    Python evidence API, but it no longer decides what semantic abstraction matters.
    """

    def __init__(
        self,
        *args: Any,
        max_reasoning_rounds: int = 4,
        max_model_calls: int | None = 220,
        max_tool_calls: int | None = 96,
        max_plan_actions: int = 16,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("reasoning_interval", 1)
        super().__init__(
            *args,
            max_model_calls=max_model_calls,
            max_tool_calls=max_tool_calls,
            **kwargs,
        )
        self.max_reasoning_rounds = max(1, int(max_reasoning_rounds))
        self.max_plan_actions = max(1, int(max_plan_actions))
        self.workspace = EvidenceWorkspace()
        self.plan_queue: list[PlannedAction] = []
        self.pending_expectation: dict[str, Any] | None = None
        self.pending_support_ids: tuple[str, ...] = ()
        self.expectation_checks = 0
        self.expectation_mismatches = 0
        self.hypothesis_tests = 0
        self.hypothesis_test_failures = 0
        self.world_model_validations = 0
        self.world_model_validation_failures = 0
        self.analysis_rounds = 0
        self.model_authored_actions = 0
        self.model_authored_probes = 0
        self.model_authored_plan_actions = 0
        self.emergency_transport_fallbacks = 0
        self.no_plan_rounds = 0
        self.last_mode = ""
        self.last_reason = ""

    def on_level_reset(self) -> None:
        super().on_level_reset()
        self.plan_queue.clear()
        self.pending_expectation = None
        self.pending_support_ids = ()

    @staticmethod
    def _decode_result(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    @staticmethod
    def _truth(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"true", "yes", "1"}:
                return True
            if value in {"false", "no", "0"}:
                return False
        return bool(value)

    def _expectation_matches(
        self,
        expect: dict[str, Any],
        transition: Transition,
        scene: Scene,
        *,
        previous_level: int,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        board_change = str(expect.get("board_change", "any")).lower()
        changed = bool(transition.meaningful_changed_cells or transition.level_completed)
        if board_change == "yes" and not changed:
            reasons.append("expected board change")
        elif board_change == "no" and changed:
            reasons.append("expected no meaningful board change")

        if expect.get("level_delta") is not None:
            try:
                wanted = int(expect["level_delta"])
                actual = int(scene.level - previous_level)
                if actual != wanted:
                    reasons.append(f"level_delta {actual} != {wanted}")
            except Exception:
                reasons.append("malformed level_delta expectation")

        game_over = self._truth(expect.get("game_over"))
        if game_over is not None and bool(transition.game_over) != game_over:
            reasons.append("game_over expectation mismatch")

        win = self._truth(expect.get("win"))
        if win is not None and bool(transition.win) != win:
            reasons.append("win expectation mismatch")

        if expect.get("min_meaningful_changed_cells") is not None:
            try:
                minimum = max(0, int(expect["min_meaningful_changed_cells"]))
                if int(transition.meaningful_changed_cells) < minimum:
                    reasons.append("too few meaningful changed cells")
            except Exception:
                reasons.append("malformed changed-cell expectation")

        after_signature = expect.get("after_signature")
        if after_signature and str(after_signature) != str(scene.signature):
            reasons.append("after_signature mismatch")
        return not reasons, reasons

    def observe(self, frame: Any) -> Scene:
        previous_level = self.level
        pending_expectation = self.pending_expectation
        pending_support_ids = self.pending_support_ids
        scene = super().observe(frame)

        expectation_matched = True
        if pending_expectation is not None and self.memory.transitions:
            latest = self.memory.transitions[-1]
            self.expectation_checks += 1
            expectation_matched, reasons = self._expectation_matches(
                pending_expectation,
                latest,
                scene,
                previous_level=previous_level,
            )
            if not expectation_matched:
                self.expectation_mismatches += 1
                self.plan_queue.clear()
                self.action_queue.clear()
                self.prediction_mismatch = True
                self.last_reason_step = -10_000
                self.workspace.apply_patch(
                    {
                        "questions": [
                            "Repair the action model after expectation mismatch: "
                            + "; ".join(reasons)
                        ]
                    },
                    step=self.step,
                )
            self.pending_expectation = None
            self.pending_support_ids = ()

        if scene.level > previous_level:
            self.plan_queue.clear()
            self.action_queue.clear()
            self.workspace.validate_from_progress(
                level=previous_level,
                step=self.step,
                supporting_ids=list(pending_support_ids) if expectation_matched else [],
                note=self.last_reason or "level completed under current evidence model",
            )
        return scene

    def _full_evidence_context(self, scene: Scene) -> dict[str, Any]:
        base = super()._sandbox_context(scene)
        compact_transitions = self._transition_records()

        def frame(i: int = -1) -> list[list[int]]:
            if not self.grids:
                return []
            idx = int(i)
            if idx < 0:
                idx += len(self.grids)
            if not 0 <= idx < len(self.grids):
                return []
            return self.grids[idx].tolist()

        def transition(i: int = -1) -> dict[str, Any]:
            if not compact_transitions:
                return {}
            idx = int(i)
            if idx < 0:
                idx += len(compact_transitions)
            if not 0 <= idx < len(compact_transitions):
                return {}
            record = dict(compact_transitions[idx])
            if idx < len(self.grids):
                record["before_grid"] = self.grids[idx].tolist()
            if idx + 1 < len(self.grids):
                record["after_grid"] = self.grids[idx + 1].tolist()
            return record

        def components_at(i: int = -1) -> list[dict[str, Any]]:
            if not self.scenes:
                return []
            idx = int(i)
            if idx < 0:
                idx += len(self.scenes)
            if not 0 <= idx < len(self.scenes):
                return []
            return self._component_records(self.scenes[idx])

        base.update(
            {
                "transition_count": len(compact_transitions),
                "frame_count": len(self.grids),
                "transitions": compact_transitions,
                "frame": frame,
                "transition": transition,
                "components_at": components_at,
                "current_components": self._component_records(scene),
                "workspace": self.workspace.summary(),
            }
        )
        return base

    def _latest_change(self, scene: Scene) -> dict[str, Any]:
        if len(self.grids) < 2:
            return {"changed_cells": 0, "meaningful_changed_cells": 0, "bbox": None}
        return diff_summary(self.grids[-2], self.grids[-1], scene.hud_mask)

    def _user_prompt(self, scene: Scene) -> str:
        return EVIDENCE_FIRST_USER_TEMPLATE.format(
            level=int(scene.level),
            step=int(self.step),
            valid_actions=json.dumps(list(scene.available_actions)),
            signature=scene.signature,
            ascii_grid=grid_ascii(scene.grid),
            latest_change=json.dumps(self._latest_change(scene), separators=(",", ":")),
            workspace=json.dumps(self.workspace.summary(), separators=(",", ":")),
        )

    def _call_evidence_model(self, user: str, scene: Scene) -> dict[str, Any] | None:
        if self.model is None or not self._model_budget_available():
            return None
        try:
            self.model_calls += 1
            text = self.model.complete(EVIDENCE_FIRST_SYSTEM_PROMPT, user, grid=scene.grid)
            parsed = extract_json(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            self.model_failures += 1
            return None

    def _run_tool(self, program: str, scene: Scene) -> tuple[bool, Any, str]:
        if not isinstance(program, str) or not program.strip() or not self._tool_budget_available():
            return False, None, ""
        self.tool_calls += 1
        try:
            raw = run_analysis_code(program, self._full_evidence_context(scene))
            return True, self._decode_result(raw), raw
        except (SandboxError, SyntaxError, ValueError, TypeError) as exc:
            self.tool_failures += 1
            return False, None, f"TOOL_ERROR {type(exc).__name__}: {exc}"

    def _apply_workspace_and_tests(self, parsed: dict[str, Any], scene: Scene) -> list[str]:
        feedback: list[str] = []
        hypothesis = parsed.get("hypothesis")
        record = self.workspace.upsert_hypothesis(hypothesis, step=self.step)
        if record is not None and record.test_python.strip() and self._tool_budget_available():
            self.hypothesis_tests += 1
            ok, result, raw = self._run_tool(record.test_python, scene)
            if ok and isinstance(result, dict):
                self.workspace.record_falsification(record.hypothesis_id, result, step=self.step)
                feedback.append(f"HYPOTHESIS TEST {record.hypothesis_id}: {raw}")
            else:
                self.hypothesis_test_failures += 1
                feedback.append(f"HYPOTHESIS TEST FAILED {record.hypothesis_id}: {raw}")

        patch = parsed.get("workspace_patch")
        previous_world_model = self.workspace.world_model_code
        self.workspace.apply_patch(patch, step=self.step)
        if self.workspace.world_model_code and self.workspace.world_model_code != previous_world_model:
            self.world_model_validations += 1
            ok, result, raw = self._run_tool(self.workspace.world_model_code, scene)
            if ok and isinstance(result, dict):
                checked = int(result.get("checked", 0) or 0)
                mismatches = int(result.get("mismatches", 0) or 0)
                status = "validated" if checked > 0 and mismatches == 0 else "rejected"
                result = {**result, "status": status}
                self.workspace.set_world_model_validation(result, step=self.step)
                feedback.append(f"WORLD MODEL CHECK: {raw}")
            else:
                self.world_model_validation_failures += 1
                self.workspace.set_world_model_validation(
                    {"status": "invalid", "detail": raw},
                    step=self.step,
                )
                feedback.append(f"WORLD MODEL CHECK FAILED: {raw}")
        return feedback

    @staticmethod
    def _support_ids(parsed: dict[str, Any]) -> tuple[str, ...]:
        raw = parsed.get("supports")
        if not isinstance(raw, list):
            return ()
        return tuple(dict.fromkeys(str(value).strip() for value in raw if str(value).strip()))

    def _parse_plan(self, parsed: dict[str, Any], scene: Scene) -> list[PlannedAction]:
        raw_plan = parsed.get("plan")
        if not isinstance(raw_plan, list):
            return []
        confidence = 0.82 if bool(parsed.get("plan_reliable", False)) else 0.58
        reason = str(
            parsed.get("reason") or parsed.get("goal") or "V012 model-authored action"
        )[:500]
        supports = self._support_ids(parsed)
        plan: list[PlannedAction] = []
        for raw in raw_plan[: self.max_plan_actions]:
            if not isinstance(raw, dict):
                continue
            spec = self._parse_one(
                raw,
                scene.available_actions,
                scene.grid.shape,
                confidence,
                reason,
            )
            if spec is None:
                continue
            expect = raw.get("expect") if isinstance(raw.get("expect"), dict) else {}
            plan.append(PlannedAction(spec=spec, expect=dict(expect), supports=supports))

        if len(plan) <= 1:
            return plan

        validation = self.workspace.world_model_validation
        model_validated = (
            validation.get("status") == "validated"
            and int(validation.get("checked", 0) or 0) > 0
            and int(validation.get("mismatches", 0) or 0) == 0
        )
        supports_grounded = self.workspace.grounded(list(supports))
        all_expected = all(bool(step.expect) for step in plan)
        if (
            not bool(parsed.get("plan_reliable", False))
            or not all_expected
            or not (model_validated or supports_grounded)
        ):
            return plan[:1]
        return plan

    def _reason(self, scene: Scene) -> list[PlannedAction]:
        if self.model is None or not self._model_budget_available():
            return []
        user = self._user_prompt(scene)
        for _ in range(self.max_reasoning_rounds):
            self.analysis_rounds += 1
            parsed = self._call_evidence_model(user, scene)
            if not isinstance(parsed, dict):
                user += (
                    "\nPrevious response was not valid JSON. "
                    "Return the compact V012 JSON contract."
                )
                continue
            self.last_mode = str(parsed.get("mode", "")).upper().strip()
            self.last_reason = str(parsed.get("reason", ""))[:500]
            feedback = self._apply_workspace_and_tests(parsed, scene)

            program = parsed.get("analysis_python")
            if isinstance(program, str) and program.strip() and self._tool_budget_available():
                ok, _result, raw = self._run_tool(program, scene)
                feedback.append(("ANALYSIS: " if ok else "ANALYSIS FAILED: ") + raw)
                user += "\n\n" + TOOL_FOLLOWUP_TEMPLATE.format(
                    tool_result="\n".join(feedback)
                )
                continue

            plan = self._parse_plan(parsed, scene)
            if plan:
                if self.last_mode == "PROBE":
                    plan = plan[:1]
                    self.model_authored_probes += 1
                else:
                    self.model_authored_plan_actions += len(plan)
                self.model_authored_actions += len(plan)
                return plan

            self.no_plan_rounds += 1
            user += (
                "\nNo executable action was produced. If another historical query is "
                "necessary, ANALYZE it now. Otherwise choose exactly one legal "
                "discriminating PROBE."
            )
        return []

    def _emergency_action(self, scene: Scene) -> ActionSpec:
        """Last-resort transport/parse failure path, never a normal cognitive fallback."""
        self.emergency_transport_fallbacks += 1
        valid = list(scene.available_actions)
        simple = [a for a in valid if int(a) in {1, 2, 3, 4, 5, 7}]
        if simple:
            counts = {
                int(a): sum(
                    int(transition.action.action_id) == int(a)
                    for transition in self.memory.transitions
                )
                for a in simple
            }
            action_id = min(simple, key=lambda a: (counts[int(a)], int(a)))
            return ActionSpec(
                int(action_id),
                reason="V012 emergency transport fallback",
                confidence=0.05,
            )
        if 6 in valid:
            row = int(scene.grid.shape[0] // 2)
            col = int(scene.grid.shape[1] // 2)
            return ActionSpec(
                6,
                x=col,
                y=row,
                reason="V012 emergency transport fallback",
                confidence=0.02,
            )
        action_id = int(valid[0]) if valid else 0
        return ActionSpec(
            action_id,
            reason="V012 emergency transport fallback",
            confidence=0.01,
        )

    def _arm_planned_action(
        self,
        scene: Scene,
        planned: PlannedAction,
        *,
        queued: bool,
    ) -> ActionSpec:
        spec = planned.spec
        self.pending_expectation = dict(planned.expect)
        self.pending_support_ids = tuple(planned.supports)
        self.last_action, self.last_target_shape = spec, None
        self.queued_actions_used += int(queued)
        self._arm_prediction(scene, spec)
        return spec

    def choose(self, scene: Scene) -> ActionSpec:
        while self.plan_queue:
            planned = self.plan_queue.pop(0)
            spec = planned.spec
            if spec.action_id not in scene.available_actions:
                self.plan_queue.clear()
                break
            if spec.action_id == 6 and (
                spec.x is None
                or spec.y is None
                or not 0 <= int(spec.x) < scene.grid.shape[1]
                or not 0 <= int(spec.y) < scene.grid.shape[0]
            ):
                self.plan_queue.clear()
                break
            return self._arm_planned_action(scene, planned, queued=True)

        plan = self._reason(scene)
        if plan:
            first, rest = plan[0], plan[1:]
            self.plan_queue.extend(rest)
            return self._arm_planned_action(scene, first, queued=False)

        spec = self._emergency_action(scene)
        self.pending_expectation = {"board_change": "any"}
        self.pending_support_ids = ()
        self.last_action, self.last_target_shape = spec, None
        self._arm_prediction(scene, spec)
        return spec

    def evidence_telemetry(self) -> dict[str, Any]:
        return {
            "analysis_rounds": self.analysis_rounds,
            "model_calls": self.model_calls,
            "model_failures": self.model_failures,
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "model_authored_actions": self.model_authored_actions,
            "model_authored_probes": self.model_authored_probes,
            "model_authored_plan_actions": self.model_authored_plan_actions,
            "queued_actions_used": self.queued_actions_used,
            "expectation_checks": self.expectation_checks,
            "expectation_mismatches": self.expectation_mismatches,
            "hypothesis_tests": self.hypothesis_tests,
            "hypothesis_test_failures": self.hypothesis_test_failures,
            "world_model_validations": self.world_model_validations,
            "world_model_validation_failures": self.world_model_validation_failures,
            "emergency_transport_fallbacks": self.emergency_transport_fallbacks,
            "no_plan_rounds": self.no_plan_rounds,
            "last_mode": self.last_mode,
            "workspace": self.workspace.summary(),
        }
