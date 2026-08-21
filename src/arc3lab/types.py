from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ActionSpec:
    action_id: int
    x: int | None = None
    y: int | None = None
    reason: str = ""
    confidence: float = 0.5

    @property
    def data(self) -> dict[str, int]:
        if self.action_id == 6:
            if self.x is None or self.y is None:
                raise ValueError("ACTION6 requires x and y")
            return {"x": int(self.x), "y": int(self.y)}
        return {}


@dataclass(frozen=True, slots=True)
class Component:
    color: int
    pixels: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    edge_touch: bool
    shape_hash: str
    cells: tuple[tuple[int, int], ...] = field(repr=False)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def center_cell(self) -> tuple[int, int]:
        cr, cc = self.centroid
        return min(self.cells, key=lambda p: (p[0] - cr) ** 2 + (p[1] - cc) ** 2)


@dataclass(slots=True)
class Scene:
    grid: np.ndarray
    background: int
    components: list[Component]
    signature: str
    level: int
    step: int
    available_actions: tuple[int, ...]
    hud_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass(slots=True)
class Transition:
    step: int
    level: int
    before_signature: str
    action: ActionSpec
    after_signature: str
    changed_cells: int
    meaningful_changed_cells: int
    level_completed: bool = False
    game_over: bool = False
    win: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
