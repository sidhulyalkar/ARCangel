from __future__ import annotations

from typing import Any

from arc3lab.policy.coding import CodingPolicy
from arc3lab.types import ActionSpec, Scene


_VALID_EFFECTS = {"dead", "change", "level", "unknown"}


def _coarse_effect_from_latest(policy: CodingPolicy) -> str | None:
    if not policy.memory.transitions:
        return None
    latest = policy.memory.transitions[-1]
    if latest.level_completed:
        return "level"
    return "change" if latest.meaningful_changed_cells else "dead"


class VerifiedCodingPolicy(CodingPolicy):
    """Falsifiable execution contracts layered over canonical V005 CodingPolicy.

    The base V005 policy stays unchanged. This layer remembers the model's optional
    coarse expected effect for each action, checks it after the real transition, and
    invalidates stale queued plans on contradiction. S130 can additionally require
    every multi-action queued step to be explicitly falsifiable.
    """

    def __init__(
        self,
        *args: Any,
        require_falsifiable_queue: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.require_falsifiable_queue = bool(require_falsifiable_queue)
        self.expectation_mismatch = False
        self.expectation_mismatch_count = 0
        self.queue_gate_truncations = 0
        self._expected_by_spec_id: dict[int, str] = {}

    def on_level_reset(self) -> None:
        super().on_level_reset()
        self.expectation_mismatch = False
        self._expected_by_spec_id.clear()

    @staticmethod
    def _expected_effect(raw: dict[str, Any]) -> str:
        value = str(raw.get("expected_effect", "unknown")).strip().lower()
        return value if value in _VALID_EFFECTS else "unknown"

    @staticmethod
    def _raw_matches_spec(raw: dict[str, Any], spec: ActionSpec) -> bool:
        try:
            if int(raw.get("id")) != int(spec.action_id):
                return False
        except Exception:
            return False
        if int(spec.action_id) != 6:
            return True
        try:
            return int(raw.get("x")) == int(spec.x) and int(raw.get("y")) == int(spec.y)
        except Exception:
            return False

    def _parse_actions(self, parsed: dict[str, Any], scene: Scene) -> list[ActionSpec]:
        specs = super()._parse_actions(parsed, scene)
        raw_actions = parsed.get("actions")
        raw_actions = raw_actions if isinstance(raw_actions, list) else []

        unused = [raw for raw in raw_actions if isinstance(raw, dict)]
        for spec in specs:
            match_i = next(
                (i for i, raw in enumerate(unused) if self._raw_matches_spec(raw, spec)),
                None,
            )
            expected = "unknown"
            if match_i is not None:
                raw = unused.pop(match_i)
                expected = self._expected_effect(raw)
            self._expected_by_spec_id[id(spec)] = expected

        if self.require_falsifiable_queue and len(specs) > 1:
            expectations = [
                self._expected_by_spec_id.get(id(spec), "unknown")
                for spec in specs
            ]
            if any(value == "unknown" for value in expectations):
                self.queue_gate_truncations += len(specs) - 1
                for spec in specs[1:]:
                    self._expected_by_spec_id.pop(id(spec), None)
                specs = specs[:1]
        return specs

    def _arm_prediction(self, scene: Scene, spec: ActionSpec) -> None:
        super()._arm_prediction(scene, spec)
        if self.pending_transition is not None:
            self.pending_transition["expected_effect"] = self._expected_by_spec_id.pop(
                id(spec), "unknown"
            )

    def observe(self, frame: Any) -> Scene:
        pending = dict(self.pending_transition) if self.pending_transition else None
        scene = super().observe(frame)

        if pending is not None:
            expected = str(pending.get("expected_effect", "unknown")).strip().lower()
            actual = _coarse_effect_from_latest(self)
            if expected in {"dead", "change", "level"} and actual is not None:
                if expected != actual:
                    self.expectation_mismatch = True
                    self.expectation_mismatch_count += 1
                    self.action_queue.clear()
                    self.last_reason_step = -10_000
                    self._remember(
                        "Execution-contract contradiction: expected coarse effect "
                        f"{expected!r} but observed {actual!r}; repair before "
                        "trusting the remaining plan.",
                        0.95,
                        source="expected_effect_error",
                    )
                else:
                    self.expectation_mismatch = False
        return scene

    def _should_reason(self, scene: Scene) -> bool:
        if self.expectation_mismatch:
            return self.model is not None and self._model_budget_available()
        return super()._should_reason(scene)
