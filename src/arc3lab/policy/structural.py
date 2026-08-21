from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from arc3lab.memory import EpisodeMemory
from arc3lab.perception.diffs import diff_summary, infer_hud_mask
from arc3lab.perception.scene import build_scene, frame_grid
from arc3lab.planning import TransitionGraph
from arc3lab.types import ActionSpec, Component, Scene, Transition


class StructuralPolicy:
    """Model-free exploration baseline with object memory and graph novelty.

    It deliberately knows no public game IDs or mechanics. This makes it useful as a
    leakage-resistant calibration submission and as a fallback beneath an LLM policy.
    """

    def __init__(self, seed: int = 0, memory_path: str | None = None) -> None:
        self.seed = seed
        self.memory = EpisodeMemory(memory_path)
        self.graph = TransitionGraph()
        self.grids: list[np.ndarray] = []
        self.scenes: list[Scene] = []
        self.last_action: ActionSpec | None = None
        self.last_target_shape: str | None = None
        self.probed_simple: dict[int, set[int]] = defaultdict(set)
        self.probed_simple_global: set[int] = set()
        self.target_trials: Counter[tuple[int, str]] = Counter()
        self.coord_trials: Counter[tuple[str, int, int]] = Counter()
        self.stuck = 0
        self.level = 0
        self.step = 0

    def on_level_reset(self) -> None:
        """Preserve learned game mechanics while clearing transient action state."""
        self.last_action = None
        self.last_target_shape = None
        self.stuck = 0

    def observe(self, frame: Any) -> Scene:
        grid = frame_grid(frame)
        future_grids = self.grids + [grid]
        hud = infer_hud_mask(future_grids)
        scene = build_scene(frame, hud_mask=hud, step=self.step)
        if self.scenes and self.last_action is not None:
            before = self.scenes[-1]
            d = diff_summary(before.grid, scene.grid, hud)
            level_up = scene.level > before.level
            t = Transition(
                step=self.step,
                level=before.level,
                before_signature=before.signature,
                action=self.last_action,
                after_signature=scene.signature,
                changed_cells=d["changed_cells"],
                meaningful_changed_cells=d["meaningful_changed_cells"],
                level_completed=level_up,
                game_over=str(getattr(frame, "state", "")).endswith("GAME_OVER"),
                win=str(getattr(frame, "state", "")).endswith("WIN"),
                metadata={"bbox": d["bbox"]},
            )
            self.memory.append(t, self.last_target_shape)
            self.graph.add(t)
            self.stuck = 0 if d["meaningful_changed_cells"] or level_up else self.stuck + 1
            if level_up:
                self.stuck = 0
        self.grids.append(grid)
        self.scenes.append(scene)
        self.level = scene.level
        self.step += 1
        return scene

    def _click_score(self, c: Component, scene: Scene) -> float:
        same_color = sum(1 for x in scene.components if x.color == c.color)
        same_shape = sum(1 for x in scene.components if x.shape_hash == c.shape_hash)
        area_term = 3.0 / math.sqrt(max(c.pixels, 1))
        compact = c.pixels / max(c.height * c.width, 1)
        edge_penalty = 2.5 if c.edge_touch else 0.0
        tried = self.target_trials[(scene.level, c.shape_hash)]
        dead = self.memory.dead_target_rate(c.shape_hash)
        coord = c.center_cell
        coord_tried = self.coord_trials[(scene.signature, coord[0], coord[1])]
        return (
            3.0 / same_color
            + 2.0 / same_shape
            + area_term
            + 0.8 * compact
            - edge_penalty
            - 0.25 * tried
            - 1.25 * dead
            - 5.0 * coord_tried
        )

    def _best_click(self, scene: Scene) -> tuple[ActionSpec, str] | None:
        if 6 not in scene.available_actions:
            return None
        candidates = [c for c in scene.components if c.pixels <= 512]
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda c: self._click_score(c, scene), reverse=True)
        c = ranked[0]
        row, col = c.center_cell
        spec = ActionSpec(6, x=col, y=row, reason=f"novel object {c.shape_hash}", confidence=0.55)
        return spec, c.shape_hash

    def choose(self, scene: Scene) -> ActionSpec:
        valid = set(scene.available_actions)
        # Learn primitive action semantics aggressively in the first level. Later
        # weighted levels inherit that evidence instead of blindly re-probing every action.
        simple = [a for a in (1, 2, 3, 4, 5, 7) if a in valid]
        unprobed = [a for a in simple if a not in self.probed_simple_global] if scene.level == 0 else []
        if unprobed:
            order = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
            a = min(unprobed, key=lambda x: order[x])
            self.probed_simple[scene.level].add(a)
            self.probed_simple_global.add(a)
            spec = ActionSpec(a, reason="first-level action-effect probe", confidence=0.45)
            self.last_action, self.last_target_shape = spec, None
            return spec

        click = self._best_click(scene)
        if click is not None:
            spec, shape = click
            if (
                spec.y is not None
                and spec.x is not None
                and self.coord_trials[(scene.signature, spec.y, spec.x)] == 0
            ) or not simple:
                self.target_trials[(scene.level, shape)] += 1
                if spec.y is not None and spec.x is not None:
                    self.coord_trials[(scene.signature, spec.y, spec.x)] += 1
                self.last_action, self.last_target_shape = spec, shape
                return spec

        if simple:
            scored = []
            for a in simple:
                spec = ActionSpec(a)
                trials = self.graph.action_trials(scene.signature, spec)
                dest = self.graph.known_destination(scene.signature, spec)
                novelty = 2.0 if dest is None else 1.0 / (1 + self.memory.state_visits[dest])
                dead = self.memory.dead_action_rate(a)
                repeat_penalty = trials * 0.8 + dead * 2.0
                scored.append((novelty - repeat_penalty, -a, a))
            a = max(scored)[2]
            spec = ActionSpec(a, reason="state-graph novelty", confidence=0.5)
            self.last_action, self.last_target_shape = spec, None
            return spec

        if click is not None:
            spec, shape = click
            self.target_trials[(scene.level, shape)] += 1
            if spec.y is not None and spec.x is not None:
                self.coord_trials[(scene.signature, spec.y, spec.x)] += 1
            self.last_action, self.last_target_shape = spec, shape
            return spec

        spec = ActionSpec(0, reason="no legal action discovered", confidence=0.1)
        self.last_action, self.last_target_shape = spec, None
        return spec
