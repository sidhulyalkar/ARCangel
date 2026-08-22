from __future__ import annotations

import json
from typing import Any

from arc3lab.model.adapter import ModelAdapter, extract_json
from arc3lab.perception.scene import compact_scene, grid_ascii
from arc3lab.policy.effect_posterior import EffectPosteriorPolicy
from arc3lab.policy.prompt import SYSTEM_PROMPT, USER_TEMPLATE
from arc3lab.types import ActionSpec, Scene


class HybridPolicy(EffectPosteriorPolicy):
    """Effect-posterior fallback + uncertainty-gated local model policy."""

    def __init__(
        self,
        model: ModelAdapter | None = None,
        *args: Any,
        model_every: int = 1,
        max_model_calls: int | None = 96,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_every = max(1, model_every)
        self.max_model_calls = None if max_model_calls is None else max(0, max_model_calls)
        self.model_calls = 0
        self.model_failures = 0
        self.action_queue: list[ActionSpec] = []

    def on_level_reset(self) -> None:
        super().on_level_reset()
        self.action_queue.clear()

    def observe(self, frame: Any) -> Scene:
        before_level = self.level
        before_stuck = self.stuck
        scene = super().observe(frame)
        if scene.level > before_level or self.stuck > before_stuck:
            self.action_queue.clear()
        return scene

    @staticmethod
    def _parse_one(
        a: dict[str, Any],
        valid: tuple[int, ...],
        shape: tuple[int, int],
        confidence: float,
        reason: str,
    ) -> ActionSpec | None:
        try:
            action_id = int(a.get("id"))
        except Exception:
            return None
        if action_id not in valid:
            return None
        if action_id == 6:
            try:
                x, y = int(a.get("x")), int(a.get("y"))
            except Exception:
                return None
            if not (0 <= x < shape[1] and 0 <= y < shape[0]):
                return None
            return ActionSpec(6, x=x, y=y, reason=reason, confidence=confidence)
        return ActionSpec(action_id, reason=reason, confidence=confidence)

    def _model_budget_available(self) -> bool:
        return self.max_model_calls is None or self.model_calls < self.max_model_calls

    def _model_actions(self, scene: Scene) -> list[ActionSpec]:
        if self.model is None or not self._model_budget_available():
            return []
        should_ask = self.step <= 6 or self.stuck > 0 or self.step % self.model_every == 0
        if not should_ask:
            return []
        prompt = USER_TEMPLATE.format(
            scene=json.dumps(compact_scene(scene), separators=(",", ":")),
            memory=json.dumps(self.memory.compact(), separators=(",", ":")),
            ascii_grid=grid_ascii(scene.grid),
        )
        try:
            self.model_calls += 1
            parsed = extract_json(self.model.complete(SYSTEM_PROMPT, prompt, grid=scene.grid))
            if not parsed:
                return []
            confidence = float(parsed.get("confidence", 0.5))
            reason = str(parsed.get("hypothesis", "model policy"))[:240]
            raw_actions = parsed.get("actions")
            if not isinstance(raw_actions, list):
                legacy = parsed.get("action")
                raw_actions = [legacy] if isinstance(legacy, dict) else []
            out: list[ActionSpec] = []
            for raw in raw_actions[:4]:
                if not isinstance(raw, dict):
                    continue
                spec = self._parse_one(
                    raw,
                    scene.available_actions,
                    scene.grid.shape,
                    confidence,
                    reason,
                )
                if spec is not None:
                    out.append(spec)
            return out
        except Exception:
            self.model_failures += 1
            return []

    def choose(self, scene: Scene) -> ActionSpec:
        while self.action_queue:
            spec = self.action_queue.pop(0)
            if spec.action_id in scene.available_actions:
                self.last_action, self.last_target_shape = spec, None
                return spec
        model_actions = self._model_actions(scene)
        if model_actions and model_actions[0].confidence >= 0.35:
            first, rest = model_actions[0], model_actions[1:]
            self.action_queue.extend(rest)
            self.last_action, self.last_target_shape = first, None
            return first
        return super().choose(scene)
