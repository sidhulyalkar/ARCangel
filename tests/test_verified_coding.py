from __future__ import annotations

import numpy as np

from arc3lab.policy.verified_coding import VerifiedCodingPolicy, _coarse_effect_from_latest
from arc3lab.types import ActionSpec, Scene, Transition


def scene() -> Scene:
    grid = np.zeros((4, 4), dtype=np.uint8)
    return Scene(
        grid=grid,
        background=0,
        components=[],
        signature="s0",
        level=0,
        step=0,
        available_actions=(1, 2, 6),
    )


def test_expected_effect_normalization() -> None:
    assert VerifiedCodingPolicy._expected_effect({"expected_effect": "CHANGE"}) == "change"
    assert VerifiedCodingPolicy._expected_effect({"expected_effect": "nonsense"}) == "unknown"


def test_s125_preserves_multiaction_queue() -> None:
    policy = VerifiedCodingPolicy(model=None, require_falsifiable_queue=False)
    parsed = {
        "confidence": 0.9,
        "hypothesis": "test",
        "plan_reliable": True,
        "actions": [
            {"id": 1, "expected_effect": "change"},
            {"id": 2, "expected_effect": "unknown"},
        ],
    }
    specs = policy._parse_actions(parsed, scene())
    assert [x.action_id for x in specs] == [1, 2]
    assert policy.queue_gate_truncations == 0


def test_s130_truncates_unfalsifiable_multiaction_queue() -> None:
    policy = VerifiedCodingPolicy(model=None, require_falsifiable_queue=True)
    parsed = {
        "confidence": 0.9,
        "hypothesis": "test",
        "plan_reliable": True,
        "actions": [
            {"id": 1, "expected_effect": "change"},
            {"id": 2, "expected_effect": "unknown"},
        ],
    }
    specs = policy._parse_actions(parsed, scene())
    assert [x.action_id for x in specs] == [1]
    assert policy.queue_gate_truncations == 1


def test_coarse_effect_uses_level_change_over_visual_change() -> None:
    policy = VerifiedCodingPolicy(model=None)
    policy.memory.transitions.append(
        Transition(
            step=1,
            level=0,
            before_signature="a",
            action=ActionSpec(1),
            after_signature="b",
            changed_cells=3,
            meaningful_changed_cells=3,
            level_completed=True,
        )
    )
    assert _coarse_effect_from_latest(policy) == "level"
