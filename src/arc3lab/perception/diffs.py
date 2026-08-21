from __future__ import annotations

from typing import Any

import numpy as np


def infer_hud_mask(grids: list[np.ndarray], *, edge_width: int = 7, min_frames: int = 4) -> np.ndarray | None:
    """Mask volatile border cells, a common timer/HUD failure mode.

    A cell is masked only if it is near an edge and has changed on at least half of
    observed transitions. Interior volatility is preserved as gameplay signal.
    """
    if len(grids) < min_frames:
        return None
    arr = np.stack(grids[-min(len(grids), 10) :])
    changes = np.count_nonzero(arr[1:] != arr[:-1], axis=0)
    threshold = max(2, (arr.shape[0] - 1 + 1) // 2)
    volatile = changes >= threshold
    h, w = volatile.shape
    edge = np.zeros((h, w), dtype=bool)
    edge[:edge_width, :] = True
    edge[-edge_width:, :] = True
    edge[:, :edge_width] = True
    edge[:, -edge_width:] = True
    candidate = volatile & edge
    # Avoid erasing a whole active border. HUD masks should be sparse/strip-like.
    if candidate.mean() > 0.30:
        return None
    return candidate


def diff_summary(before: np.ndarray, after: np.ndarray, hud_mask: np.ndarray | None = None) -> dict[str, Any]:
    if before.shape != after.shape:
        return {"changed_cells": int(after.size), "meaningful_changed_cells": int(after.size), "bbox": None}
    changed = before != after
    meaningful = changed.copy()
    if hud_mask is not None:
        meaningful &= ~hud_mask
    pts = np.argwhere(meaningful)
    bbox = None
    if len(pts):
        bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
    return {
        "changed_cells": int(changed.sum()),
        "meaningful_changed_cells": int(meaningful.sum()),
        "bbox": bbox,
    }
