from __future__ import annotations

import hashlib
from collections import Counter, deque
from typing import Any, Iterable

import numpy as np

from arc3lab.types import Component, Scene


def frame_grid(frame: Any) -> np.ndarray:
    """Return the last rendered grid from FrameData/FrameDataRaw without mutating it."""
    raw = getattr(frame, "frame", None)
    if raw is None or len(raw) == 0:
        return np.zeros((64, 64), dtype=np.int8)
    grid = np.asarray(raw[-1], dtype=np.int8)
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D grid, got {grid.shape}")
    return grid


def _background(grid: np.ndarray, mask: np.ndarray | None = None) -> int:
    vals = grid.ravel() if mask is None else grid[~mask]
    if vals.size == 0:
        vals = grid.ravel()
    counts = np.bincount(vals.astype(np.int64), minlength=16)
    return int(counts.argmax())


def _shape_hash(color: int, cells: Iterable[tuple[int, int]]) -> str:
    cells = tuple(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    payload = f"{color}:" + ";".join(f"{r},{c}" for r, c in norm)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def components(grid: np.ndarray, background: int, hud_mask: np.ndarray | None = None) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    if hud_mask is not None:
        seen |= hud_mask
    out: list[Component] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == background:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            out.append(
                Component(
                    color=color,
                    pixels=len(cells),
                    bbox=bbox,
                    centroid=(sum(rs) / len(rs), sum(cs) / len(cs)),
                    edge_touch=bbox[0] == 0 or bbox[1] == 0 or bbox[2] == h - 1 or bbox[3] == w - 1,
                    shape_hash=_shape_hash(color, cells),
                    cells=tuple(cells),
                )
            )
    out.sort(key=lambda x: (x.bbox[0], x.bbox[1], x.color, x.pixels))
    return out


def structural_signature(grid: np.ndarray, background: int, hud_mask: np.ndarray | None = None) -> str:
    work = grid.copy()
    if hud_mask is not None:
        work[hud_mask] = background
    return hashlib.blake2b(work.tobytes(), digest_size=12).hexdigest()


def build_scene(
    frame: Any,
    *,
    history_grids: list[np.ndarray] | None = None,
    hud_mask: np.ndarray | None = None,
    step: int = 0,
) -> Scene:
    grid = frame_grid(frame)
    bg = _background(grid, hud_mask)
    comps = components(grid, bg, hud_mask)
    valid = tuple(int(x.value if hasattr(x, "value") else x) for x in (getattr(frame, "available_actions", None) or []))
    level = int(getattr(frame, "levels_completed", 0))
    return Scene(
        grid=grid,
        background=bg,
        components=comps,
        signature=structural_signature(grid, bg, hud_mask),
        level=level,
        step=step,
        available_actions=valid,
        hud_mask=hud_mask,
    )


def compact_scene(scene: Scene, limit: int = 24) -> dict[str, Any]:
    color_counts = Counter(c.color for c in scene.components)
    objs = []
    for c in sorted(scene.components, key=lambda x: (x.pixels, x.edge_touch))[:limit]:
        objs.append(
            {
                "color": c.color,
                "pixels": c.pixels,
                "bbox": c.bbox,
                "center": tuple(round(v, 1) for v in c.centroid),
                "shape": c.shape_hash,
                "edge": c.edge_touch,
                "same_color_count": color_counts[c.color],
            }
        )
    return {
        "level_completed_count": scene.level,
        "signature": scene.signature,
        "background": scene.background,
        "available_actions": scene.available_actions,
        "objects": objs,
        "object_count": len(scene.components),
    }


def grid_ascii(grid: np.ndarray) -> str:
    """Lossless 16-symbol text view, useful when the local model is text-only."""
    alphabet = "0123456789ABCDEF"
    return "\n".join("".join(alphabet[int(v)] for v in row) for row in grid)
