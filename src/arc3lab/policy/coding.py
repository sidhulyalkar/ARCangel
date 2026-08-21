from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any

from arc3lab.model.adapter import ModelAdapter, extract_json
from arc3lab.perception.diffs import diff_summary
from arc3lab.perception.scene import compact_scene, grid_ascii
from arc3lab.policy.coding_prompt import (
    CODING_SYSTEM_PROMPT,
    CODING_USER_TEMPLATE,
    TOOL_RESULT_TEMPLATE,
)
from arc3lab.policy.hybrid import HybridPolicy
from arc3lab.policy.sandbox import SandboxError, run_analysis_code
from arc3lab.types import ActionSpec, Scene


class CodingPolicy(HybridPolicy):
    """Code-assisted ARC3 policy with programmatic full-history retrieval.

    A local model can either act directly or write a bounded Python analysis program
    over the complete transition ledger. The Python tool is deliberately read-only
    with respect to the environment; real actions still pass through normal legality
    guards and the competition runner.
    """

    def __init__(
        self,
        model: ModelAdapter | None = None,
        *args: Any,
        reasoning_interval: int = 2,
        max_model_calls: int | None = 96,
        max_tool_calls: int | None = 36,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, *args, max_model_calls=max_model_calls, **kwargs)
        self.reasoning_interval = max(1, reasoning_interval)
        self.max_tool_calls = None if max_tool_calls is None else max(0, max_tool_calls)
        self.tool_calls = 0
        self.tool_failures = 0
        self.reasoning_cycles = 0
        self.fallback_actions = 0
        self.queued_actions_used = 0
        self.beliefs: list[dict[str, Any]] = []
        self.last_reason_step = -10_000
        self.last_reason_level = -1
        self.last_hypothesis = ""
        self.last_expected_change = ""

    def on_level_reset(self) -> None:
        super().on_level_reset()
        # Preserve verified history/beliefs but force a fresh decision after a failed attempt.
        self.last_reason_step = -10_000

    def observe(self, frame: Any) -> Scene:
        previous_level = self.level
        scene = super().observe(frame)
        if scene.level > previous_level:
            self.action_queue.clear()
            self.last_reason_step = -10_000
            if self.last_hypothesis:
                self._remember(
                    f"Level {previous_level} completed under hypothesis: {self.last_hypothesis}",
                    0.85,
                    source="level_completion",
                )
        if self.stuck > 0:
            self.action_queue.clear()
        return scene

    def _remember(self, note: str, confidence: float, *, source: str = "model") -> None:
        note = " ".join(str(note).split())[:500]
        if not note or confidence < 0.62:
            return
        # Deduplicate near-identical notes without deleting the underlying lossless ledger.
        if any(note.lower() == str(x.get("note", "")).lower() for x in self.beliefs[-16:]):
            return
        self.beliefs.append(
            {
                "level": self.level,
                "step": self.step,
                "confidence": round(float(confidence), 3),
                "source": source,
                "note": note,
            }
        )
        if len(self.beliefs) > 48:
            self.beliefs = self.beliefs[-48:]

    def _transition_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in self.memory.transitions:
            d = asdict(t)
            # Keep a flat, coding-agent-friendly action representation.
            d["action"] = asdict(t.action)
            out.append(d)
        return out

    def _component_records(self, scene: Scene) -> list[dict[str, Any]]:
        records = []
        for c in scene.components:
            records.append(
                {
                    "color": c.color,
                    "pixels": c.pixels,
                    "bbox": c.bbox,
                    "center": c.center_cell,
                    "shape": c.shape_hash,
                    "edge": c.edge_touch,
                    "cells_sample": list(c.cells[:96]),
                }
            )
        return records

    def _sandbox_context(self, scene: Scene) -> dict[str, Any]:
        transitions = self._transition_records()
        frames = [g.tolist() for g in self.grids[-24:]]

        def recent(n: int = 12) -> list[dict[str, Any]]:
            n = max(0, min(int(n), 200))
            return transitions[-n:]

        def by_action(action_id: int) -> list[dict[str, Any]]:
            aid = int(action_id)
            return [t for t in transitions if int(t["action"]["action_id"]) == aid]

        def action_stats() -> dict[int, dict[str, int]]:
            stats: dict[int, Counter[str]] = {}
            for t in transitions:
                aid = int(t["action"]["action_id"])
                bucket = stats.setdefault(aid, Counter())
                effect = "level" if t["level_completed"] else (
                    "change" if int(t["meaningful_changed_cells"]) else "dead"
                )
                bucket[effect] += 1
            return {k: dict(v) for k, v in stats.items()}

        def level_wins() -> list[dict[str, Any]]:
            return [t for t in transitions if t["level_completed"] or t["win"]]

        def frame(i: int = -1) -> list[list[int]]:
            if not frames:
                return []
            return frames[int(i)]

        def diff_frames(i: int = -2, j: int = -1) -> dict[str, Any]:
            if len(self.grids) < 2:
                return {"changed_cells": 0, "meaningful_changed_cells": 0, "bbox": None}
            gi = self.grids[int(i)]
            gj = self.grids[int(j)]
            return diff_summary(gi, gj, scene.hud_mask)

        return {
            "grid": scene.grid.tolist(),
            "rows": int(scene.grid.shape[0]),
            "cols": int(scene.grid.shape[1]),
            "level": int(scene.level),
            "step": int(self.step),
            "current_signature": scene.signature,
            "valid_actions": list(scene.available_actions),
            "components": self._component_records(scene),
            "transitions": transitions,
            "recent_frames": frames,
            "beliefs": list(self.beliefs),
            "recent": recent,
            "by_action": by_action,
            "action_stats": action_stats,
            "level_wins": level_wins,
            "frame": frame,
            "diff_frames": diff_frames,
        }

    def _tool_budget_available(self) -> bool:
        return self.max_tool_calls is None or self.tool_calls < self.max_tool_calls

    def _should_reason(self, scene: Scene) -> bool:
        if self.model is None or not self._model_budget_available():
            return False
        if scene.level != self.last_reason_level:
            return True
        if self.stuck > 0:
            return True
        # RHAE charges every environment action, including failed attempts and post-reset
        # retries, to the level where it happened. Level 0 is therefore the cheapest place
        # to identify mechanics. Once the game advances, prefer transferred beliefs and
        # reliable queued plans, re-reasoning mainly when the state is novel or the plan fails.
        interval = 1 if scene.level == 0 else max(2, self.reasoning_interval)
        return self.step - self.last_reason_step >= interval

    def _parse_actions(self, parsed: dict[str, Any], scene: Scene) -> list[ActionSpec]:
        try:
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        hypothesis = str(parsed.get("hypothesis", "coding agent"))[:500]
        self.last_hypothesis = hypothesis
        self.last_expected_change = str(parsed.get("expected_change", ""))[:500]
        self._remember(str(parsed.get("memory_note", "")), confidence)
        raw_actions = parsed.get("actions")
        if not isinstance(raw_actions, list):
            return []
        out: list[ActionSpec] = []
        for raw in raw_actions[:6]:
            if not isinstance(raw, dict):
                continue
            spec = self._parse_one(
                raw,
                scene.available_actions,
                scene.grid.shape,
                confidence,
                hypothesis,
            )
            if spec is not None:
                out.append(spec)
        if not bool(parsed.get("plan_reliable", False)) and len(out) > 1:
            out = out[:1]
        return out

    def _call_model(self, user: str, scene: Scene) -> dict[str, Any] | None:
        if self.model is None or not self._model_budget_available():
            return None
        try:
            self.model_calls += 1
            text = self.model.complete(CODING_SYSTEM_PROMPT, user, grid=scene.grid)
            return extract_json(text)
        except Exception:
            self.model_failures += 1
            return None

    def _reason(self, scene: Scene) -> list[ActionSpec]:
        if not self._should_reason(scene):
            return []
        self.reasoning_cycles += 1
        self.last_reason_step = self.step
        self.last_reason_level = scene.level
        user = CODING_USER_TEMPLATE.format(
            scene=json.dumps(compact_scene(scene), separators=(",", ":")),
            memory=json.dumps(self.memory.compact(), separators=(",", ":")),
            beliefs=json.dumps(self.beliefs[-12:], separators=(",", ":")),
            ascii_grid=grid_ascii(scene.grid),
            valid_actions=list(scene.available_actions),
        )
        parsed = self._call_model(user, scene)
        if not parsed:
            return []

        program = parsed.get("python")
        if (
            isinstance(program, str)
            and program.strip()
            and self._tool_budget_available()
            and self._model_budget_available()
        ):
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

        return self._parse_actions(parsed, scene)

    def choose(self, scene: Scene) -> ActionSpec:
        while self.action_queue:
            spec = self.action_queue.pop(0)
            if spec.action_id in scene.available_actions:
                if spec.action_id != 6 or (
                    spec.x is not None
                    and spec.y is not None
                    and 0 <= spec.x < scene.grid.shape[1]
                    and 0 <= spec.y < scene.grid.shape[0]
                ):
                    self.last_action, self.last_target_shape = spec, None
                    self.queued_actions_used += 1
                    return spec

        model_actions = self._reason(scene)
        if model_actions and model_actions[0].confidence >= 0.35:
            first, rest = model_actions[0], model_actions[1:]
            self.action_queue.extend(rest)
            self.last_action, self.last_target_shape = first, None
            return first

        self.fallback_actions += 1
        return super(HybridPolicy, self).choose(scene)
