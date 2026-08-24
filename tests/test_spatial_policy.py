from __future__ import annotations

from collections import Counter

import numpy as np

from arc3lab.perception.scene import build_scene
from arc3lab.perception.spatial import actor_key
from arc3lab.policy.spatial_coding import SpatialCodingPolicy
from arc3lab.types import ActionSpec


class Frame:
    def __init__(self, grid, actions=(1, 2, 3, 4)):
        self.frame = [np.asarray(grid, dtype=np.int8)]
        self.available_actions = actions
        self.levels_completed = 0
        self.state = "NOT_FINISHED"


def scene(grid):
    return build_scene(Frame(grid))


def seeded_policy(grid: np.ndarray) -> tuple[SpatialCodingPolicy, object]:
    policy = SpatialCodingPolicy(model=None, predictive_history_depth=2, spatial_plan_horizon=12)
    s = scene(grid)
    actor_i = next(i for i, c in enumerate(s.components) if c.color == 7)
    key = actor_key(s.components[actor_i])
    policy.spatial_control.motion_counts[key] = 4
    policy.spatial_control.action_sets[key].update({1, 2, 3, 4})
    vectors = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
    for aid, delta in vectors.items():
        policy.spatial_control.vector_counts[(key, aid)] = Counter({delta: 2})
        policy.spatial_control.pure_vector_counts[(key, aid)] = Counter({delta: 2})
    policy.spatial_control.last_center[key] = s.components[actor_i].center_cell
    return policy, s


def test_spatial_sandbox_exposes_exact_geometry_and_planner():
    g = np.zeros((8, 9), dtype=np.int8)
    g[4, 1] = 7
    g[4, 7] = 3
    policy, s = seeded_policy(g)
    ctx = policy._sandbox_context(s)
    assert "spatial" in ctx
    assert "spatial_plan" in ctx
    assert "spatial_relations" in ctx
    target = next(i for i, c in enumerate(s.components) if c.color == 3)
    plan = ctx["spatial_plan"](target, "touch")
    assert plan["ok"]
    assert plan["steps"] == 5
    assert plan["actions"] == [2, 2, 2, 2, 2]


def test_compile_requested_plan_is_gated_and_tracks_expected_centers():
    g = np.zeros((8, 9), dtype=np.int8)
    g[4, 1] = 7
    g[4, 7] = 3
    policy, s = seeded_policy(g)
    target = next(i for i, c in enumerate(s.components) if c.color == 3)
    parsed = {
        "spatial_plan": {"target_object": target, "relation": "touch", "execute": True},
        "confidence": 0.95,
    }
    specs = policy._compile_requested_plan(parsed, s)
    assert len(specs) == 5
    assert all(spec.action_id == 2 for spec in specs)
    assert policy.spatial_plans_requested == 1
    assert policy.spatial_plans_compiled == 1
    assert policy.spatial_plan_actions == 5
    assert all(id(spec) in policy._planned_steps for spec in specs)


def test_unknown_control_does_not_auto_compile():
    g = np.zeros((8, 9), dtype=np.int8)
    g[4, 1] = 7
    g[4, 7] = 3
    s = scene(g)
    policy = SpatialCodingPolicy(model=None)
    target = next(i for i, c in enumerate(s.components) if c.color == 3)
    parsed = {
        "spatial_plan": {"target_object": target, "relation": "touch", "execute": True},
        "confidence": 0.99,
    }
    assert policy._compile_requested_plan(parsed, s) == []
    assert policy.spatial_plans_requested == 1
    assert policy.spatial_plans_compiled == 0


def test_dynamic_obstacle_invalidates_queued_spatial_step_before_action():
    g = np.zeros((8, 9), dtype=np.int8)
    g[4, 1] = 7
    g[4, 7] = 3
    policy, s = seeded_policy(g)
    target = next(i for i, c in enumerate(s.components) if c.color == 3)
    specs = policy._compile_requested_plan(
        {
            "spatial_plan": {"target_object": target, "relation": "touch", "execute": True},
            "confidence": 0.95,
        },
        s,
    )
    assert specs
    first = specs[0]
    assert policy._queued_spatial_step_safe(s, first)

    blocked = g.copy()
    expected = tuple(policy._planned_steps[id(first)]["center"])
    blocked[expected] = 8
    blocked_scene = scene(blocked)
    assert not policy._queued_spatial_step_safe(blocked_scene, first)


def test_arm_prediction_attaches_compiled_spatial_expectation():
    g = np.zeros((8, 9), dtype=np.int8)
    g[4, 1] = 7
    g[4, 7] = 3
    policy, s = seeded_policy(g)
    target = next(i for i, c in enumerate(s.components) if c.color == 3)
    spec = policy._compile_requested_plan(
        {
            "spatial_plan": {"target_object": target, "relation": "touch", "execute": True},
            "confidence": 0.95,
        },
        s,
    )[0]
    expected = tuple(policy._planned_steps[id(spec)]["center"])
    policy._arm_prediction(s, spec)
    assert policy.pending_transition is not None
    assert tuple(policy.pending_transition["spatial_expected_center"]) == expected
    assert id(spec) not in policy._planned_steps
