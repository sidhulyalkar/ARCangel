from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arc3lab.perception.scene import build_scene
from arc3lab.perception.spatial import (
    SpatialControlModel,
    anchor_valid_mask,
    component_relation,
    direction8,
    line_of_sight,
    raycast8,
)
from arc3lab.planning.spatial import shortest_spatial_plan
from arc3lab.types import ActionSpec


@dataclass
class Frame:
    grid: np.ndarray
    available_actions: tuple[int, ...] = (1, 2, 3, 4)
    levels_completed: int = 0

    @property
    def frame(self):
        return [self.grid]


def scene(grid, actions=(1, 2, 3, 4)):
    return build_scene(Frame(np.asarray(grid, dtype=np.int8), actions), step=0)


def move_cell(grid, src, dst):
    out = np.asarray(grid, dtype=np.int8).copy()
    color = int(out[src])
    out[src] = 0
    out[dst] = color
    return out


def test_direction8_all_octants():
    o = (5.0, 5.0)
    expected = {
        (4, 5): "N", (4, 6): "NE", (5, 6): "E", (6, 6): "SE",
        (6, 5): "S", (6, 4): "SW", (5, 4): "W", (4, 4): "NW",
    }
    for point, label in expected.items():
        assert direction8(o, point) == label


def test_relation_raycast_and_line_of_sight():
    g = np.zeros((9, 9), dtype=np.int8)
    g[4, 4] = 2
    g[1, 4] = 3
    g[4, 7] = 4
    s = scene(g)
    actor = next(i for i, c in enumerate(s.components) if c.color == 2)
    north = next(i for i, c in enumerate(s.components) if c.color == 3)
    east = next(i for i, c in enumerate(s.components) if c.color == 4)
    assert component_relation(s.components[actor], s.components[north])["direction"] == "N"
    assert component_relation(s.components[actor], s.components[east])["direction"] == "E"
    rays = raycast8(s, s.components[actor].center_cell, ignore_component=actor)
    assert rays["N"]["object"] == north and rays["N"]["distance"] == 3
    assert rays["E"]["object"] == east and rays["E"]["distance"] == 3
    assert line_of_sight(s, actor, east)
    g[4, 6] = 8
    s2 = scene(g)
    actor2 = next(i for i, c in enumerate(s2.components) if c.color == 2)
    east2 = next(i for i, c in enumerate(s2.components) if c.color == 4)
    assert not line_of_sight(s2, actor2, east2)


def test_control_model_learns_actor_without_color_prior():
    g0 = np.zeros((8, 8), dtype=np.int8)
    g0[4, 3] = 7
    g0[1, 1] = 2
    g0[6, 6] = 5
    states = [scene(g0)]
    cur = g0
    for src, dst in (((4, 3), (3, 3)), ((3, 3), (3, 4)), ((3, 4), (4, 4))):
        cur = move_cell(cur, src, dst)
        states.append(scene(cur))

    model = SpatialControlModel()
    for aid in range(1, 4):
        model.observe(states[aid - 1], states[aid], ActionSpec(aid))
    actor = model.actor_hypothesis(states[-1])
    assert actor is not None
    assert states[-1].components[actor["index"]].color == 7
    vectors = model.action_vectors(states[-1])
    assert vectors[1].delta == (-1, 0)
    assert vectors[2].delta == (0, 1)
    assert vectors[3].delta == (1, 0)
    assert model.planner_ready(states[-1])


def test_global_camera_translation_is_rejected():
    g0 = np.zeros((10, 10), dtype=np.int8)
    for color, (r, c) in zip((2, 3, 4, 5), ((2, 2), (2, 5), (5, 2), (5, 5))):
        g0[r, c] = color
    g1 = np.zeros_like(g0)
    for r, c in zip(*np.nonzero(g0)):
        g1[r, c + 1] = g0[r, c]
    model = SpatialControlModel()
    model.observe(scene(g0), scene(g1), ActionSpec(1))
    assert model.rejected_global_motion == 1
    assert model.actor_hypothesis(scene(g1)) is None


def test_duplicate_actor_and_collateral_motion_block_auto_planner():
    g0 = np.zeros((8, 8), dtype=np.int8)
    g0[4, 2] = 7
    g0[4, 5] = 7
    model = SpatialControlModel()
    cur = g0
    for aid, src, dst in ((1, (4, 2), (3, 2)), (2, (3, 2), (3, 3)), (3, (3, 3), (4, 3))):
        nxt = move_cell(cur, src, dst)
        model.observe(scene(cur), scene(nxt), ActionSpec(aid))
        cur = nxt
    hyp = model.actor_hypothesis(scene(cur))
    assert hyp is not None and hyp["candidate_count"] == 2
    assert not model.planner_ready(scene(cur))

    g0 = np.zeros((8, 8), dtype=np.int8)
    g0[4, 2] = 7
    g0[1, 1] = 3
    model = SpatialControlModel()
    cur = g0
    transitions = [
        (1, (4, 2), (3, 2), (1, 1), (1, 2)),
        (2, (3, 2), (3, 3), (1, 2), (2, 2)),
        (3, (3, 3), (4, 3), (2, 2), (2, 3)),
    ]
    for aid, a0, a1, d0, d1 in transitions:
        nxt = cur.copy()
        ac, dc = int(nxt[a0]), int(nxt[d0])
        nxt[a0] = 0; nxt[a1] = ac
        nxt[d0] = 0; nxt[d1] = dc
        model.observe(scene(cur), scene(nxt), ActionSpec(aid))
        cur = nxt
    vectors = model.action_vectors(scene(cur))
    assert vectors and all(v.purity == 0.0 for v in vectors.values())
    assert not model.planner_ready(scene(cur))


def test_footprint_aware_collision_and_shortest_detour():
    g = np.zeros((8, 8), dtype=np.int8)
    g[4:6, 1:3] = 2
    g[:, 4] = 8
    g[4, 4] = 0
    s = scene(g)
    actor = next(i for i, c in enumerate(s.components) if c.color == 2)
    assert not anchor_valid_mask(s, actor)[4, 4]

    g = np.zeros((9, 10), dtype=np.int8)
    g[4, 1] = 2
    g[4, 8] = 3
    g[1:8, 5] = 8
    g[7, 5] = 0
    s = scene(g)
    actor = next(i for i, c in enumerate(s.components) if c.color == 2)
    target = next(i for i, c in enumerate(s.components) if c.color == 3)
    from arc3lab.perception.spatial import VectorEvidence
    vectors = {
        1: VectorEvidence((-1, 0), 2, 2, 1.0),
        2: VectorEvidence((0, 1), 2, 2, 1.0),
        3: VectorEvidence((1, 0), 2, 2, 1.0),
        4: VectorEvidence((0, -1), 2, 2, 1.0),
    }
    plan = shortest_spatial_plan(s, actor, target, vectors, relation="touch")
    assert plan is not None and plan.steps > 6
    assert len(plan.anchors) == plan.steps + 1
    assert len(set(plan.anchors)) == len(plan.anchors)
