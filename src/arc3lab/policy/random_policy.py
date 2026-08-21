from __future__ import annotations

import random
from typing import Any

from arc3lab.perception.scene import build_scene
from arc3lab.types import ActionSpec, Scene


class RandomPolicy:
    """Reproducible random control, intentionally game-agnostic."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.step = 0
        self.model_failures = 0

    def observe(self, frame: Any) -> Scene:
        s = build_scene(frame, step=self.step)
        self.step += 1
        return s

    def choose(self, scene: Scene) -> ActionSpec:
        legal = [a for a in scene.available_actions if a != 0]
        if not legal:
            return ActionSpec(0, reason="random-control reset")
        aid = self.rng.choice(legal)
        if aid == 6:
            return ActionSpec(
                6,
                x=self.rng.randrange(scene.grid.shape[1]),
                y=self.rng.randrange(scene.grid.shape[0]),
                reason="random-control click",
            )
        return ActionSpec(aid, reason="random-control action")
