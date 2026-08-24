from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import hypot
from typing import Iterable

import numpy as np

from arc3lab.types import ActionSpec, Component, Scene


DIRECTIONS_8: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "NE": (-1, 1),
    "E": (0, 1),
    "SE": (1, 1),
    "S": (1, 0),
    "SW": (1, -1),
    "W": (0, -1),
    "NW": (-1, -1),
}


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def direction8(a: tuple[float, float], b: tuple[float, float]) -> str:
    dr = _sign(b[0] - a[0])
    dc = _sign(b[1] - a[1])
    if (dr, dc) == (0, 0):
        return "SAME"
    reverse = {v: k for k, v in DIRECTIONS_8.items()}
    return reverse[(dr, dc)]


def bbox_gap(a: Component, b: Component) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return row_gap, col_gap


def component_relation(a: Component, b: Component) -> dict:
    dr = float(b.centroid[0] - a.centroid[0])
    dc = float(b.centroid[1] - a.centroid[1])
    rg, cg = bbox_gap(a, b)
    a_cells = set(a.cells)
    b_cells = set(b.cells)
    touching4 = bool(a_cells & b_cells) or any(
        (r + dr, c + dc) in b_cells
        for r, c in a_cells
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    overlaps_row = not (a.bbox[2] < b.bbox[0] or b.bbox[2] < a.bbox[0])
    overlaps_col = not (a.bbox[3] < b.bbox[1] or b.bbox[3] < a.bbox[1])
    return {
        "direction": direction8(a.centroid, b.centroid),
        "delta": [round(dr, 3), round(dc, 3)],
        "manhattan": round(abs(dr) + abs(dc), 3),
        "chebyshev": round(max(abs(dr), abs(dc)), 3),
        "euclidean": round(hypot(dr, dc), 3),
        "bbox_gap": [rg, cg],
        "same_row_band": overlaps_row,
        "same_col_band": overlaps_col,
        "touching": touching4,
    }


def _cell_to_component(scene: Scene) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for i, comp in enumerate(scene.components):
        for cell in comp.cells:
            out[cell] = i
    return out


def raycast8(scene: Scene, origin: tuple[int, int], *, ignore_component: int | None = None) -> dict[str, dict]:
    h, w = scene.grid.shape
    cell_index = _cell_to_component(scene)
    ignore_cells = set(scene.components[ignore_component].cells) if ignore_component is not None else set()
    out: dict[str, dict] = {}
    for name, (dr, dc) in DIRECTIONS_8.items():
        r, c = origin
        distance = 0
        hit = None
        while True:
            r += dr
            c += dc
            distance += 1
            if not (0 <= r < h and 0 <= c < w):
                hit = {"kind": "boundary", "distance": distance - 1}
                break
            if (r, c) in ignore_cells:
                continue
            if int(scene.grid[r, c]) != int(scene.background):
                idx = cell_index.get((r, c))
                hit = {
                    "kind": "object",
                    "distance": distance,
                    "cell": [r, c],
                    "color": int(scene.grid[r, c]),
                    "object": idx,
                }
                break
        out[name] = hit
    return out


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dc = abs(c1 - c0)
    dr = -abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc + dr
    out: list[tuple[int, int]] = []
    while True:
        out.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 >= dr:
            err += dr
            c0 += sc
        if e2 <= dc:
            err += dc
            r0 += sr
    return out


def line_of_sight(scene: Scene, first_index: int, second_index: int) -> bool:
    first = scene.components[first_index]
    second = scene.components[second_index]
    ignore = set(first.cells) | set(second.cells)
    for r, c in _bresenham(first.center_cell, second.center_cell)[1:-1]:
        if (r, c) in ignore:
            continue
        if int(scene.grid[r, c]) != int(scene.background):
            return False
        if scene.hud_mask is not None and bool(scene.hud_mask[r, c]):
            return False
    return True


def actor_offsets(component: Component) -> tuple[tuple[int, int], ...]:
    ar, ac = component.center_cell
    return tuple(sorted((r - ar, c - ac) for r, c in component.cells))


def anchor_valid_mask(
    scene: Scene,
    actor_index: int,
    *,
    passable_component: int | None = None,
) -> np.ndarray:
    """Valid anchor cells for translating an actor footprint without collision.

    The calculation is footprint-exact but vectorized over anchor positions, so even
    64x64 boards remain cheap enough to recompute before every queued route step.
    """
    h, w = scene.grid.shape
    actor = scene.components[actor_index]
    offsets = actor_offsets(actor)
    occupied = np.asarray(scene.grid != scene.background, dtype=bool).copy()
    for r, c in actor.cells:
        occupied[r, c] = False
    if passable_component is not None:
        for r, c in scene.components[passable_component].cells:
            occupied[r, c] = False
    if scene.hud_mask is not None:
        occupied |= np.asarray(scene.hud_mask, dtype=bool)

    valid = np.ones((h, w), dtype=bool)
    for dr, dc in offsets:
        r0 = max(0, -dr)
        r1 = min(h, h - dr)
        c0 = max(0, -dc)
        c1 = min(w, w - dc)

        if r0 > 0:
            valid[:r0, :] = False
        if r1 < h:
            valid[r1:, :] = False
        if c0 > 0:
            valid[:, :c0] = False
        if c1 < w:
            valid[:, c1:] = False
        if r0 < r1 and c0 < c1:
            valid[r0:r1, c0:c1] &= ~occupied[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
    return valid


def connected_regions(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sorted(sizes, reverse=True)


def reachable_anchors(
    valid: np.ndarray,
    start: tuple[int, int],
    deltas: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = valid.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and valid[start]):
        return set()
    steps = tuple(dict.fromkeys(tuple(map(int, d)) for d in deltas if tuple(d) != (0, 0)))
    if not steps:
        return {start}
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in steps:
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and valid[nxt] and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def narrow_anchor_count(valid: np.ndarray) -> int:
    h, w = valid.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            degree = sum(
                0 <= r + dr < h and 0 <= c + dc < w and bool(valid[r + dr, c + dc])
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if degree <= 2:
                count += 1
    return count


def dihedral_delta(delta: tuple[int, int]) -> dict[str, tuple[int, int]]:
    r, c = delta
    return {
        "identity": (r, c),
        "rot90": (c, -r),
        "rot180": (-r, -c),
        "rot270": (-c, r),
        "reflect_h": (r, -c),
        "reflect_v": (-r, c),
        "reflect_diag": (c, r),
        "reflect_anti": (-c, -r),
    }


ActorKey = tuple[int, str, int]


def actor_key(component: Component) -> ActorKey:
    return (int(component.color), str(component.shape_hash), int(component.pixels))


def _greedy_matches(before: Scene, after: Scene) -> list[tuple[int, int, tuple[int, int]]]:
    """Match shape/color-identical components by minimum center displacement."""
    by_key_before: dict[ActorKey, list[int]] = defaultdict(list)
    by_key_after: dict[ActorKey, list[int]] = defaultdict(list)
    for i, c in enumerate(before.components):
        by_key_before[actor_key(c)].append(i)
    for i, c in enumerate(after.components):
        by_key_after[actor_key(c)].append(i)

    pairs: list[tuple[int, int, tuple[int, int]]] = []
    for key in set(by_key_before) & set(by_key_after):
        left = set(by_key_before[key])
        right = set(by_key_after[key])
        candidates: list[tuple[float, int, int, tuple[int, int]]] = []
        for i in left:
            br, bc = before.components[i].center_cell
            for j in right:
                ar, ac = after.components[j].center_cell
                d = (ar - br, ac - bc)
                candidates.append((abs(d[0]) + abs(d[1]), i, j, d))
        for _, i, j, d in sorted(candidates):
            if i in left and j in right:
                left.remove(i)
                right.remove(j)
                pairs.append((i, j, d))
    return pairs


@dataclass(frozen=True)
class VectorEvidence:
    delta: tuple[int, int]
    support: int
    total: int
    confidence: float
    pure_support: int = 0
    purity: float = 1.0


class SpatialControlModel:
    """Learn controlled-object identity and action translations from causal motion.

    No action meaning or player color is assumed. Camera-like global translation is
    rejected, and planner readiness is confidence-gated.
    """

    def __init__(self) -> None:
        self.motion_counts: Counter[ActorKey] = Counter()
        self.action_sets: dict[ActorKey, set[int]] = defaultdict(set)
        self.vector_counts: dict[tuple[ActorKey, int], Counter[tuple[int, int]]] = defaultdict(Counter)
        self.pure_vector_counts: dict[tuple[ActorKey, int], Counter[tuple[int, int]]] = defaultdict(Counter)
        self.last_center: dict[ActorKey, tuple[int, int]] = {}
        self.observed_transitions = 0
        self.rejected_global_motion = 0

    def observe(self, before: Scene, after: Scene, action: ActionSpec) -> None:
        aid = int(action.action_id)
        if aid not in {1, 2, 3, 4, 5, 7}:
            return
        pairs = _greedy_matches(before, after)
        moved = [(i, j, d) for i, j, d in pairs if d != (0, 0)]
        self.observed_transitions += 1
        if not moved:
            return

        deltas = Counter(d for _, _, d in moved)
        _, dominant_n = deltas.most_common(1)[0]
        if len(pairs) >= 3 and dominant_n >= 3 and dominant_n / max(len(pairs), 1) >= 0.65:
            self.rejected_global_motion += 1
            return

        for i, j, delta in moved:
            key = actor_key(before.components[i])
            self.motion_counts[key] += 1
            self.action_sets[key].add(aid)
            self.vector_counts[(key, aid)][delta] += 1
            if len(moved) == 1 and len(before.components) == len(after.components):
                self.pure_vector_counts[(key, aid)][delta] += 1
            self.last_center[key] = after.components[j].center_cell

    def _ranked_actor_keys(self) -> list[tuple[float, ActorKey]]:
        ranked = []
        for key, n in self.motion_counts.items():
            distinct = len(self.action_sets[key])
            consistent = 0.0
            for aid in self.action_sets[key]:
                counts = self.vector_counts[(key, aid)]
                total = sum(counts.values())
                if total:
                    consistent += counts.most_common(1)[0][1] / total
            mean_consistency = consistent / max(distinct, 1)
            score = 2.5 * distinct + n + 2.0 * mean_consistency
            ranked.append((score, key))
        return sorted(ranked, reverse=True)

    def actor_confidence(self) -> float:
        ranked = self._ranked_actor_keys()
        if not ranked:
            return 0.0
        top = ranked[0][0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        evidence = self.motion_counts[ranked[0][1]]
        distinct = len(self.action_sets[ranked[0][1]])
        evidence_term = min(1.0, (evidence + distinct) / 6.0)
        margin = top / max(top + second, 1e-9)
        return round(0.55 * evidence_term + 0.45 * margin, 4)

    def actor_hypothesis(self, scene: Scene) -> dict | None:
        ranked = self._ranked_actor_keys()
        if not ranked:
            return None
        key = ranked[0][1]
        matches = [i for i, c in enumerate(scene.components) if actor_key(c) == key]
        if not matches:
            return None
        if len(matches) == 1:
            idx = matches[0]
        else:
            last = self.last_center.get(key)
            if last is None:
                return None
            idx = min(matches, key=lambda i: abs(scene.components[i].center_cell[0] - last[0]) + abs(scene.components[i].center_cell[1] - last[1]))
        base_confidence = self.actor_confidence()
        effective_confidence = base_confidence if len(matches) == 1 else base_confidence * 0.65
        return {
            "index": idx,
            "key": key,
            "center": list(scene.components[idx].center_cell),
            "confidence": round(effective_confidence, 4),
            "base_confidence": base_confidence,
            "candidate_count": len(matches),
            "motion_evidence": int(self.motion_counts[key]),
            "distinct_actions": len(self.action_sets[key]),
        }

    def action_vectors(self, scene: Scene, *, min_support: int = 1) -> dict[int, VectorEvidence]:
        actor = self.actor_hypothesis(scene)
        if actor is None:
            return {}
        key = actor["key"]
        out: dict[int, VectorEvidence] = {}
        for aid in sorted(self.action_sets[key]):
            counts = self.vector_counts[(key, aid)]
            if not counts:
                continue
            delta, support = counts.most_common(1)[0]
            total = sum(counts.values())
            if support < min_support or delta == (0, 0):
                continue
            pure_support = int(self.pure_vector_counts[(key, aid)][delta])
            purity = pure_support / support if support else 0.0
            out[aid] = VectorEvidence(delta, support, total, support / total, pure_support, purity)
        return out

    def planner_ready(self, scene: Scene) -> bool:
        actor = self.actor_hypothesis(scene)
        vectors = self.action_vectors(scene, min_support=1)
        if actor is None or actor["confidence"] < 0.72 or actor.get("candidate_count", 2) != 1:
            return False
        if actor["motion_evidence"] < 3 or actor["distinct_actions"] < 2:
            return False
        if len(vectors) < 2:
            return False
        return all(v.confidence >= 0.80 and v.purity >= 0.66 for v in vectors.values())

    def summary(self, scene: Scene) -> dict:
        actor = self.actor_hypothesis(scene)
        vectors = self.action_vectors(scene, min_support=1)
        return {
            "actor": actor,
            "planner_ready": self.planner_ready(scene),
            "action_vectors": {
                str(aid): {
                    "delta": list(ev.delta),
                    "support": ev.support,
                    "total": ev.total,
                    "confidence": round(ev.confidence, 4),
                    "pure_support": ev.pure_support,
                    "purity": round(ev.purity, 4),
                    "symmetries": {k: list(v) for k, v in dihedral_delta(ev.delta).items()},
                }
                for aid, ev in vectors.items()
            },
            "observed_transitions": self.observed_transitions,
            "rejected_global_motion": self.rejected_global_motion,
        }


def spatial_summary(scene: Scene, control: SpatialControlModel, *, object_limit: int = 20) -> dict:
    actor = control.actor_hypothesis(scene)
    actor_idx = int(actor["index"]) if actor is not None else None

    if actor_idx is not None:
        ar, ac = scene.components[actor_idx].center_cell
        ranked = sorted(
            (i for i in range(len(scene.components)) if i != actor_idx),
            key=lambda i: (
                abs(scene.components[i].center_cell[0] - ar) + abs(scene.components[i].center_cell[1] - ac),
                scene.components[i].pixels,
                i,
            ),
        )
        selected = [actor_idx, *ranked[: max(0, object_limit - 1)]]
    else:
        selected = list(range(min(len(scene.components), object_limit)))

    objects = []
    for i in selected:
        comp = scene.components[i]
        item = {
            "index": i,
            "color": int(comp.color),
            "pixels": int(comp.pixels),
            "bbox": list(comp.bbox),
            "center": list(comp.center_cell),
            "shape": comp.shape_hash,
            "edge": bool(comp.edge_touch),
        }
        if actor_idx is not None and i != actor_idx:
            item["from_actor"] = {
                **component_relation(scene.components[actor_idx], comp),
                "line_of_sight": line_of_sight(scene, actor_idx, i),
            }
        objects.append(item)

    topology = None
    rays = None
    if actor_idx is not None:
        actor_comp = scene.components[actor_idx]
        valid = anchor_valid_mask(scene, actor_idx)
        regions = connected_regions(valid)
        vectors = control.action_vectors(scene, min_support=1)
        reachable = reachable_anchors(valid, actor_comp.center_cell, [v.delta for v in vectors.values()])
        topology = {
            "valid_actor_anchors": int(valid.sum()),
            "region_count": len(regions),
            "largest_regions": regions[:8],
            "narrow_anchor_count": narrow_anchor_count(valid),
            "reachable_under_learned_actions": len(reachable),
        }
        rays = raycast8(scene, actor_comp.center_cell, ignore_component=actor_idx)

    return {
        "control": control.summary(scene),
        "objects": objects,
        "rays8": rays,
        "topology": topology,
    }
