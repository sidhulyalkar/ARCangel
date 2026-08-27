import json

import numpy as np

from arc3lab.perception.scene import build_scene
from arc3lab.policy.lean_scientist import (
    LEAN_SCIENTIST_SYSTEM_PROMPT,
    LeanReflectiveScientistPolicy,
)


class Frame:
    def __init__(self, grid, actions=(1, 2, 3, 4, 6), level=0, state="NOT_FINISHED"):
        self.frame = [np.asarray(grid, dtype=np.int8)]
        self.available_actions = actions
        self.levels_completed = level
        self.state = state


class FakeModel:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def complete(self, system, user, *, grid=None):
        self.calls.append((system, user, grid))
        return json.dumps(self.payloads.pop(0))


def grid():
    g = np.zeros((10, 10), dtype=np.int8)
    g[7, 2] = 7
    g[2, 8] = 3
    return g


def payload(**kwargs):
    out = {
        "orientation": {
            "controlled_entity": "uncertain isolated object",
            "important_changes": [],
            "dominant_uncertainty": "agency",
        },
        "hypotheses": {"agency": [], "mechanics": [], "goals": [], "abstraction": []},
        "visual_goals": [],
        "decision_mode": "TEST_HYPOTHESIS",
        "candidate_id": "a0",
        "actions": [],
        "hypothesis_confidence": 0.15,
        "action_confidence": 0.85,
        "experiment": {
            "question": "does ACTION1 move the isolated object?",
            "distinguishes": ["controlled", "not controlled"],
            "expected_outcomes": ["motion", "no motion"],
        },
        "reflection": {
            "goal": "",
            "rules": [],
            "avoid": [],
            "next_test": "test ACTION1 once",
            "confidence": 0.2,
        },
        "python": "",
        "analysis_question": "",
        "delegate_world_model": False,
        "plan_reliable": False,
        "expected_change": "one object moves",
        "memory_note": "",
        "goal": "",
        "hypothesis": "ACTION1 may control the actor",
    }
    out.update(kwargs)
    return out


def test_v011_prompt_is_model_led_and_has_structured_reflection():
    assert "code layer is a guardrail" in LEAN_SCIENTIST_SYSTEM_PROMPT
    assert '"reflection"' in LEAN_SCIENTIST_SYSTEM_PROMPT
    assert "Return exactly one JSON object" in LEAN_SCIENTIST_SYSTEM_PROMPT


def test_lean_model_call_does_not_duplicate_v009_perceptual_instruction_block():
    model = FakeModel([payload()])
    policy = LeanReflectiveScientistPolicy(model=model, max_model_calls=4, max_tool_calls=0)
    action = policy.choose(policy.observe(Frame(grid(), actions=(1, 2, 3, 4))))
    assert action.action_id == 1
    assert model.calls
    _, user, views = model.calls[0]
    assert "PERCEPTUAL STATE ESTIMATION" not in user
    assert "COMPACT EVIDENCE" in user
    assert isinstance(views, dict) and len(views["views"]) == 2


def test_lower_action_gate_accepts_a_clear_experiment_without_high_theory_confidence():
    model = FakeModel([payload(hypothesis_confidence=0.08, action_confidence=0.24)])
    policy = LeanReflectiveScientistPolicy(model=model, max_model_calls=4, max_tool_calls=0)
    action = policy.choose(policy.observe(Frame(grid(), actions=(1, 2, 3, 4))))
    assert action.action_id == 1
    assert policy.semantic_actions == 1
    assert policy.emergency_fallback_actions == 0


def test_reflection_is_bounded_persistent_state_and_feeds_goal_memory():
    refl = {
        "goal": "touch the isolated upper-right object",
        "rules": ["ACTION1 changes the controlled object", "ACTION2 appears inert"],
        "avoid": ["do not repeat ACTION2 without new context"],
        "next_test": "test whether ACTION3 reverses the last displacement",
        "confidence": 0.71,
    }
    model = FakeModel([payload(reflection=refl)])
    policy = LeanReflectiveScientistPolicy(model=model, max_model_calls=4, max_tool_calls=0)
    policy.choose(policy.observe(Frame(grid(), actions=(1, 2, 3, 4))))
    assert policy.reflection_updates == 1
    assert policy.reflection["goal"] == refl["goal"]
    assert policy.reflection["rules"] == refl["rules"]
    assert policy.goals and policy.goals[-1]["goal"] == refl["goal"]


def test_post_bootstrap_uncertainty_does_not_force_a_model_call_every_action():
    policy = LeanReflectiveScientistPolicy(model=FakeModel([]), max_model_calls=20, max_tool_calls=0)
    scene = build_scene(Frame(grid(), actions=(1, 2, 3, 4)), step=7)
    policy.step = 7
    policy.level = 0
    policy.last_reason_level = 0
    policy.last_reason_step = 6
    policy.semantic_actions = 2
    policy.last_perceptual_state = {"orientation_entropy": 0.99, "recommended_mode": "IDENTIFY_GOAL"}
    assert policy._should_reason(scene) is False
    policy.step = 8
    assert policy._should_reason(scene) is True


def test_reflection_deduplicates_rules_and_respects_limit():
    policy = LeanReflectiveScientistPolicy(model=None, reflection_rule_limit=4)
    parsed = policy._normalize(payload())
    for i in range(8):
        parsed["reflection"] = {
            "goal": "",
            "rules": [f"rule {i}", f"rule {i}"],
            "avoid": [],
            "next_test": "",
            "confidence": 0.5,
        }
        policy._update_reflection(parsed)
    assert len(policy.reflection["rules"]) == 4
    assert policy.reflection["rules"] == ["rule 4", "rule 5", "rule 6", "rule 7"]


def test_semantic_telemetry_reports_reflection_and_reasoning_gate():
    policy = LeanReflectiveScientistPolicy(model=None)
    policy.reflection_updates = 3
    policy.reasoning_gate_skips = 7
    tel = policy.semantic_telemetry()
    assert tel["reflection_updates"] == 3
    assert tel["reasoning_gate_skips"] == 7
    assert "reflection" in tel
