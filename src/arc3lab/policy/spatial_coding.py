from __future__ import annotations

import json
from typing import Any

from arc3lab.perception.scene import compact_scene, grid_ascii
from arc3lab.perception.spatial import (
    SpatialControlModel,
    actor_key,
    anchor_valid_mask,
    component_relation,
    spatial_summary,
)
from arc3lab.planning.spatial import shortest_spatial_plan
from arc3lab.policy.coding import CodingPolicy
from arc3lab.policy.coding_prompt import CODING_SYSTEM_PROMPT, CODING_USER_TEMPLATE, TOOL_RESULT_TEMPLATE
from arc3lab.policy.sandbox import SandboxError, run_analysis_code
from arc3lab.types import ActionSpec, Scene


SPATIAL_INSTRUCTIONS = """
SPATIAL WORLD MODEL
The JSON below is computed exactly from the current board. Object indices are local to
this frame. `control.actor` is a causal hypothesis inferred from which object moved under
simple actions; it is not guessed from color. `action_vectors` are learned displacements
(row_delta,col_delta), never hard-coded button meanings. `rays8` gives 360-degree
N/NE/E/SE/S/SW/W/NW visibility from the inferred actor. `topology` is calculated for the
actor's full footprint rather than a point robot.

When navigation is the likely solution and a target object is identifiable, you may request
an exact path by returning:
  "spatial_plan": {"target_object": <index>, "relation": "touch", "execute": true}
Supported relations: touch, adjacent8, overlap, center, inside.
The planner executes only when causal control evidence is sufficiently supported. Otherwise
the request is advisory and normal reasoning continues. In the Python sandbox you can inspect
`spatial`, call `spatial_relations(object_index)`, and call
`spatial_plan(target_index, relation="touch", max_steps=128)` before committing. Never infer
public-game semantics from object indices; indices merely reference this observation.
"""


class SpatialCodingPolicy(CodingPolicy):
    """V006 coding agent with exact geometric world state and guarded path execution."""

    def __init__(
        self,
        *args: Any,
        spatial_plan_horizon: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_control = SpatialControlModel()
        self.spatial_plan_horizon = max(1, min(int(spatial_plan_horizon), 32))
        self.spatial_plans_requested = 0
        self.spatial_plans_compiled = 0
        self.spatial_plan_actions = 0
        self.spatial_plan_mismatches = 0
        self._planned_steps: dict[int, dict[str, Any]] = {}

    def on_level_reset(self) -> None:
        super().on_level_reset()
        self._planned_steps.clear()

    def observe(self, frame: Any) -> Scene:
        before_scene = self.scenes[-1] if self.scenes else None
        action = self.last_action
        expected_center = None
        if self.pending_transition is not None:
            expected_center = self.pending_transition.get("spatial_expected_center")

        scene = super().observe(frame)

        if before_scene is not None and action is not None:
            self.spatial_control.observe(before_scene, scene, action)

        if expected_center is not None:
            actor = self.spatial_control.actor_hypothesis(scene)
            if actor is not None and tuple(actor["center"]) != tuple(expected_center):
                self.spatial_plan_mismatches += 1
                self.action_queue.clear()
                self._planned_steps.clear()
                self.last_reason_step = -10_000
                self._remember(
                    "Spatial-plan contradiction: controlled-object position did not match "
                    "the compiled route; discard the remaining route and repair geometry.",
                    0.98,
                    source="spatial_prediction_error",
                )
        return scene

    def _arm_prediction(self, scene: Scene, spec: ActionSpec) -> None:
        super()._arm_prediction(scene, spec)
        if self.pending_transition is not None:
            step = self._planned_steps.pop(id(spec), None)
            if step is not None:
                self.pending_transition["spatial_expected_center"] = tuple(step["center"])

    def _spatial_plan_payload(
        self,
        scene: Scene,
        target_index: int,
        relation: str = "touch",
        max_steps: int = 128,
    ) -> dict:
        actor = self.spatial_control.actor_hypothesis(scene)
        if actor is None:
            return {"ok": False, "reason": "controlled object not identified"}
        if not (0 <= int(target_index) < len(scene.components)):
            return {"ok": False, "reason": "target index out of range"}
        vectors = self.spatial_control.action_vectors(scene, min_support=1)
        if not vectors:
            return {"ok": False, "reason": "no learned translational action semantics"}
        plan = shortest_spatial_plan(
            scene,
            int(actor["index"]),
            int(target_index),
            vectors,
            relation=str(relation),
            max_steps=max(1, min(int(max_steps), 256)),
        )
        if plan is None:
            return {"ok": False, "reason": "no route under current geometry/action model"}
        return {
            "ok": True,
            "actor": int(plan.actor_index),
            "target": int(plan.target_index),
            "relation": plan.relation,
            "steps": plan.steps,
            "actions": list(plan.actions),
            "anchors": [list(x) for x in plan.anchors],
            "control_confidence": actor["confidence"],
            "planner_ready": self.spatial_control.planner_ready(scene),
        }

    def _sandbox_context(self, scene: Scene) -> dict[str, Any]:
        context = super()._sandbox_context(scene)
        summary = spatial_summary(scene, self.spatial_control)

        def spatial_plan(target_index: int, relation: str = "touch", max_steps: int = 128) -> dict:
            return self._spatial_plan_payload(scene, int(target_index), str(relation), int(max_steps))

        def spatial_relations(object_index: int) -> list[dict]:
            i = int(object_index)
            if not (0 <= i < len(scene.components)):
                return []
            out = []
            for j, comp in enumerate(scene.components):
                if i == j:
                    continue
                out.append({"object": j, **component_relation(scene.components[i], comp)})
            return out

        context.update(
            {
                "spatial": summary,
                "spatial_plan": spatial_plan,
                "spatial_relations": spatial_relations,
            }
        )
        return context

    def _compile_requested_plan(self, parsed: dict[str, Any], scene: Scene) -> list[ActionSpec]:
        request = parsed.get("spatial_plan")
        if not isinstance(request, dict) or not bool(request.get("execute", False)):
            return []
        self.spatial_plans_requested += 1
        if not self.spatial_control.planner_ready(scene):
            return []
        try:
            target = int(request.get("target_object"))
        except Exception:
            return []
        relation = str(request.get("relation", "touch"))
        payload = self._spatial_plan_payload(scene, target, relation)
        if not payload.get("ok") or not payload.get("planner_ready"):
            return []
        actions = payload["actions"][: self.spatial_plan_horizon]
        anchors = payload["anchors"]
        if not actions:
            return []
        try:
            model_conf = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
        except Exception:
            model_conf = 0.5
        confidence = min(model_conf, float(payload["control_confidence"]))
        if confidence < 0.62:
            return []
        specs: list[ActionSpec] = []
        for k, aid in enumerate(actions):
            if int(aid) not in scene.available_actions:
                return []
            spec = ActionSpec(
                int(aid),
                reason=f"compiled spatial route to object {target} ({relation})",
                confidence=confidence,
            )
            if k + 1 < len(anchors):
                target_component = scene.components[target]
                self._planned_steps[id(spec)] = {
                    "center": tuple(anchors[k + 1]),
                    "target_key": actor_key(target_component),
                    "target_center": tuple(target_component.center_cell),
                    "relation": relation.lower(),
                }
            specs.append(spec)
        self.spatial_plans_compiled += 1
        self.spatial_plan_actions += len(specs)
        return specs

    def _queued_spatial_step_safe(self, scene: Scene, spec: ActionSpec) -> bool:
        step = self._planned_steps.get(id(spec))
        if step is None:
            return True
        expected = tuple(step["center"])
        actor = self.spatial_control.actor_hypothesis(scene)
        if actor is None:
            return False
        vectors = self.spatial_control.action_vectors(scene, min_support=1)
        evidence = vectors.get(int(spec.action_id))
        if evidence is None or evidence.confidence < 0.80 or evidence.purity < 0.66:
            return False
        current = tuple(actor["center"])
        predicted = (current[0] + evidence.delta[0], current[1] + evidence.delta[1])
        if predicted != expected:
            return False

        passable_component = None
        relation = str(step.get("relation", "touch"))
        if relation in {"overlap", "enter", "center", "center_on", "inside", "contain"}:
            key = tuple(step.get("target_key", ()))
            center = tuple(step.get("target_center", ()))
            matches = [
                i for i, comp in enumerate(scene.components)
                if actor_key(comp) == key
            ]
            if not matches:
                return False
            passable_component = min(
                matches,
                key=lambda i: abs(scene.components[i].center_cell[0] - center[0])
                + abs(scene.components[i].center_cell[1] - center[1]),
            )
        valid = anchor_valid_mask(scene, int(actor["index"]), passable_component=passable_component)
        return (
            0 <= expected[0] < valid.shape[0]
            and 0 <= expected[1] < valid.shape[1]
            and bool(valid[expected])
        )

    def choose(self, scene: Scene) -> ActionSpec:
        if self.action_queue:
            spec = self.action_queue[0]
            if id(spec) in self._planned_steps and not self._queued_spatial_step_safe(scene, spec):
                self.spatial_plan_mismatches += 1
                self.action_queue.clear()
                self._planned_steps.clear()
                self.last_reason_step = -10_000
                self._remember(
                    "Spatial route became unsafe under current geometry; re-plan before acting.",
                    0.98,
                    source="spatial_preaction_guard",
                )
        return super().choose(scene)

    def _reason(self, scene: Scene) -> list[ActionSpec]:
        if not self._should_reason(scene):
            return []
        self.reasoning_cycles += 1
        self.last_reason_step = self.step
        self.last_reason_level = scene.level
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
        parsed = self._call_model(user, scene)
        if not parsed:
            return []

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
            second = self._call_model(follow, scene)
            if second:
                parsed = second

        compiled = self._compile_requested_plan(parsed, scene)
        if compiled:
            return compiled
        return self._parse_actions(parsed, scene)
