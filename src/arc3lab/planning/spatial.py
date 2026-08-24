from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from arc3lab.perception.spatial import VectorEvidence, actor_offsets, anchor_valid_mask
from arc3lab.types import Scene


@dataclass(frozen=True)
class SpatialPlan:
    actions: tuple[int, ...]
    anchors: tuple[tuple[int, int], ...]
    relation: str
    target_index: int
    actor_index: int

    @property
    def steps(self) -> int:
        return len(self.actions)


def _actor_cells(anchor: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets}


def _goal_satisfied(
    anchor: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
    target_cells: set[tuple[int, int]],
    target_center: tuple[int, int],
    relation: str,
) -> bool:
    cells = _actor_cells(anchor, offsets)
    relation = relation.lower()
    if relation in {"overlap", "enter"}:
        return bool(cells & target_cells)
    if relation in {"center", "center_on"}:
        return anchor == target_center
    if relation in {"inside", "contain"}:
        return anchor in target_cells
    if relation in {"adjacent8", "near"}:
        for r, c in cells:
            for tr, tc in target_cells:
                if max(abs(r - tr), abs(c - tc)) <= 1:
                    return True
        return False
    for r, c in cells:
        if (r, c) in target_cells:
            return True
        if any((r + dr, c + dc) in target_cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            return True
    return False


def shortest_spatial_plan(
    scene: Scene,
    actor_index: int,
    target_index: int,
    vectors: dict[int, VectorEvidence],
    *,
    relation: str = "touch",
    max_steps: int = 128,
) -> SpatialPlan | None:
    if actor_index == target_index or not vectors:
        return None
    actor = scene.components[actor_index]
    target = scene.components[target_index]
    offsets = actor_offsets(actor)
    start = actor.center_cell
    passable = target_index if relation.lower() in {"overlap", "enter", "center", "center_on", "inside", "contain"} else None
    valid = anchor_valid_mask(scene, actor_index, passable_component=passable)
    if not valid[start]:
        return None

    target_cells = set(target.cells)
    target_center = target.center_cell
    if _goal_satisfied(start, offsets, target_cells, target_center, relation):
        return SpatialPlan((), (start,), relation, target_index, actor_index)

    q = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
    depth = {start: 0}
    h, w = valid.shape

    ordered = sorted(vectors.items(), key=lambda kv: (-kv[1].confidence, -kv[1].support, kv[0]))
    goal_anchor: tuple[int, int] | None = None
    while q:
        cur = q.popleft()
        d = depth[cur]
        if d >= max_steps:
            continue
        for aid, evidence in ordered:
            dr, dc = evidence.delta
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w) or not valid[nxt] or nxt in parent:
                continue
            parent[nxt] = (cur, aid)
            depth[nxt] = d + 1
            if _goal_satisfied(nxt, offsets, target_cells, target_center, relation):
                goal_anchor = nxt
                q.clear()
                break
            q.append(nxt)

    if goal_anchor is None:
        return None

    actions_rev: list[int] = []
    anchors_rev: list[tuple[int, int]] = [goal_anchor]
    cur = goal_anchor
    while parent[cur] is not None:
        prev, aid = parent[cur]
        actions_rev.append(aid)
        anchors_rev.append(prev)
        cur = prev
    actions = tuple(reversed(actions_rev))
    anchors = tuple(reversed(anchors_rev))
    return SpatialPlan(actions, anchors, relation, target_index, actor_index)
