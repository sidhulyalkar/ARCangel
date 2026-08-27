import json

import numpy as np

from arc3lab.perception.scene import build_scene
from arc3lab.policy.evidence_first import EvidenceFirstCodingPolicy
from arc3lab.policy.evidence_workspace import EvidenceWorkspace


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


def grid(marker=(7, 2)):
    g = np.zeros((10, 10), dtype=np.int8)
    g[marker] = 7
    g[2, 8] = 3
    return g


def probe(action_id=1, **extra):
    payload = {
        "mode": "PROBE",
        "analysis_python": "",
        "hypothesis": {},
        "workspace_patch": {},
        "supports": [],
        "plan": [
            {
                "id": action_id,
                "x": None,
                "y": None,
                "expect": {"board_change": "any", "game_over": False},
            }
        ],
        "plan_reliable": False,
        "goal": "discover the action semantics",
        "reason": "one discriminating action",
    }
    payload.update(extra)
    return payload


def grounded_hypothesis(workspace, hypothesis_id="grounded"):
    record = workspace.upsert_hypothesis(
        {
            "id": hypothesis_id,
            "kind": "mechanic",
            "claim": f"{hypothesis_id} controls are stable",
            "confidence": 0.8,
        },
        step=0,
    )
    assert record is not None
    workspace.record_falsification(
        hypothesis_id,
        {"consistent": True, "support": 3, "contradictions": 0},
        step=0,
    )
    return record


def test_workspace_falsification_separates_theory_from_ground_truth():
    workspace = EvidenceWorkspace()
    record = workspace.upsert_hypothesis(
        {
            "id": "move-left",
            "kind": "mechanic",
            "claim": "ACTION1 always moves the controlled entity left",
            "confidence": 0.8,
            "test_python": (
                "result={'consistent': False, 'support': 2, 'contradictions': 1}"
            ),
        },
        step=3,
    )
    assert record is not None
    workspace.record_falsification(
        "move-left",
        {"consistent": False, "support": 2, "contradictions": 1},
        step=4,
    )
    assert workspace.hypotheses["move-left"].status == "falsified"
    assert workspace.hypotheses["move-left"].confidence <= 0.2
    assert "ACTION1 always moves the controlled entity left" in workspace.falsified_rules


def test_model_workspace_patch_cannot_self_certify_rules():
    workspace = EvidenceWorkspace()
    workspace.apply_patch(
        {
            "validated_add": ["I declare this true"],
            "falsified_add": ["I declare this false"],
            "questions": ["what evidence distinguishes the theories?"],
        },
        step=4,
    )
    assert workspace.validated_rules == []
    assert workspace.falsified_rules == []
    assert workspace.open_questions == ["what evidence distinguishes the theories?"]


def test_progress_promotes_only_explicitly_supporting_hypotheses():
    workspace = EvidenceWorkspace()
    grounded_hypothesis(workspace, "used")
    grounded_hypothesis(workspace, "unrelated")
    workspace.validate_from_progress(level=0, step=8, supporting_ids=["used"])
    assert workspace.hypotheses["used"].status == "validated"
    assert workspace.hypotheses["unrelated"].status == "history_consistent"


def test_v012_model_probe_has_no_normal_heuristic_fallback():
    model = FakeModel([probe(3)])
    policy = EvidenceFirstCodingPolicy(model=model, max_model_calls=4, max_tool_calls=0)
    action = policy.choose(policy.observe(Frame(grid(), actions=(1, 2, 3, 4))))
    assert action.action_id == 3
    assert policy.model_authored_actions == 1
    assert policy.model_authored_probes == 1
    assert policy.emergency_transport_fallbacks == 0


def test_v012_can_analyze_history_then_act_without_spending_environment_action():
    model = FakeModel(
        [
            {
                "mode": "ANALYZE",
                "analysis_python": (
                    "result={'frames': frame_count, 'transitions': transition_count}"
                ),
                "hypothesis": {},
                "workspace_patch": {
                    "questions": ["which primitive changes gameplay state?"]
                },
                "supports": [],
                "plan": [],
                "plan_reliable": False,
                "goal": "",
                "reason": "inspect evidence first",
            },
            probe(2),
        ]
    )
    policy = EvidenceFirstCodingPolicy(model=model, max_model_calls=6, max_tool_calls=4)
    action = policy.choose(policy.observe(Frame(grid(), actions=(1, 2, 3, 4))))
    assert action.action_id == 2
    assert policy.tool_calls == 1
    assert policy.model_calls == 2
    assert policy.emergency_transport_fallbacks == 0
    assert len(model.calls) == 2


def test_hypothesis_test_runs_against_evidence_before_probe():
    model = FakeModel(
        [
            probe(
                1,
                hypothesis={
                    "id": "h1",
                    "kind": "mechanic",
                    "claim": "ACTION1 has not contradicted a movement hypothesis",
                    "confidence": 0.7,
                    "test_python": (
                        "result={'consistent': True, 'support': 2, "
                        "'contradictions': 0}"
                    ),
                },
            )
        ]
    )
    policy = EvidenceFirstCodingPolicy(model=model, max_model_calls=4, max_tool_calls=4)
    policy.choose(policy.observe(Frame(grid(), actions=(1, 2, 3, 4))))
    record = policy.workspace.hypotheses["h1"]
    assert policy.hypothesis_tests == 1
    assert record.status == "history_consistent"
    assert record.support == 2


def test_long_plan_requires_explicit_grounded_support_and_expectations():
    policy = EvidenceFirstCodingPolicy(model=None)
    scene = build_scene(Frame(grid(), actions=(1, 2, 3, 4)), step=0)
    parsed = {
        "mode": "EXECUTE",
        "plan_reliable": True,
        "reason": "known route",
        "supports": ["controls"],
        "plan": [
            {"id": 1, "expect": {"board_change": "any"}},
            {"id": 2, "expect": {"board_change": "any"}},
        ],
    }
    assert len(policy._parse_plan(parsed, scene)) == 1

    grounded_hypothesis(policy.workspace, "unrelated")
    assert len(policy._parse_plan(parsed, scene)) == 1

    grounded_hypothesis(policy.workspace, "controls")
    accepted = policy._parse_plan(parsed, scene)
    assert len(accepted) == 2
    assert accepted[0].supports == ("controls",)

    parsed_without_expectation = {
        **parsed,
        "plan": [
            {"id": 1, "expect": {"board_change": "any"}},
            {"id": 2},
        ],
    }
    assert len(policy._parse_plan(parsed_without_expectation, scene)) == 1


def test_expectation_mismatch_clears_queued_plan_and_forces_repair():
    policy = EvidenceFirstCodingPolicy(model=None)
    first = policy.observe(Frame(grid(), actions=(1, 2, 3, 4)))
    policy.last_action = policy._parse_one(
        {"id": 1},
        first.available_actions,
        first.grid.shape,
        0.8,
        "test",
    )
    assert policy.last_action is not None
    policy.pending_expectation = {"board_change": "yes", "game_over": False}
    policy.plan_queue.append(
        policy._parse_plan(
            {
                "mode": "EXECUTE",
                "plan_reliable": False,
                "supports": [],
                "plan": [{"id": 2, "expect": {"board_change": "any"}}],
            },
            first,
        )[0]
    )
    policy.observe(Frame(grid(), actions=(1, 2, 3, 4)))
    assert policy.expectation_checks == 1
    assert policy.expectation_mismatches == 1
    assert policy.plan_queue == []
    assert policy.workspace.open_questions


def test_expected_no_change_does_not_cancel_remaining_plan():
    policy = EvidenceFirstCodingPolicy(model=None)
    first = policy.observe(Frame(grid(), actions=(1, 2, 3, 4)))
    spec = policy._parse_one(
        {"id": 1},
        first.available_actions,
        first.grid.shape,
        0.8,
        "test no-op",
    )
    assert spec is not None
    policy.last_action = spec
    policy.pending_expectation = {"board_change": "no", "game_over": False}
    policy.plan_queue.append(
        policy._parse_plan(
            {
                "mode": "EXECUTE",
                "plan_reliable": False,
                "supports": [],
                "plan": [{"id": 2, "expect": {"board_change": "any"}}],
            },
            first,
        )[0]
    )
    policy.observe(Frame(grid(), actions=(1, 2, 3, 4)))
    assert policy.expectation_mismatches == 0
    assert len(policy.plan_queue) == 1


def test_level_delta_expectation_uses_pre_action_level():
    policy = EvidenceFirstCodingPolicy(model=None)
    first = policy.observe(Frame(grid(), actions=(1, 2), level=0))
    spec = policy._parse_one(
        {"id": 1},
        first.available_actions,
        first.grid.shape,
        0.8,
        "finish level",
    )
    assert spec is not None
    policy.last_action = spec
    policy.pending_expectation = {
        "board_change": "any",
        "level_delta": 1,
        "game_over": False,
    }
    policy.observe(Frame(grid((6, 2)), actions=(1, 2), level=1))
    assert policy.expectation_checks == 1
    assert policy.expectation_mismatches == 0


def test_successful_level_promotes_only_plan_supports():
    policy = EvidenceFirstCodingPolicy(model=None)
    grounded_hypothesis(policy.workspace, "route")
    grounded_hypothesis(policy.workspace, "spectator")
    first = policy.observe(Frame(grid(), actions=(1, 2), level=0))
    spec = policy._parse_one(
        {"id": 1},
        first.available_actions,
        first.grid.shape,
        0.8,
        "finish using route",
    )
    assert spec is not None
    policy.last_action = spec
    policy.pending_expectation = {"level_delta": 1, "game_over": False}
    policy.pending_support_ids = ("route",)
    policy.observe(Frame(grid((6, 2)), actions=(1, 2), level=1))
    assert policy.workspace.hypotheses["route"].status == "validated"
    assert policy.workspace.hypotheses["spectator"].status == "history_consistent"


def test_world_model_code_must_report_checked_history_and_zero_mismatches():
    model = FakeModel(
        [
            {
                **probe(1),
                "workspace_patch": {
                    "world_model_code": (
                        "result={'checked': 3, 'mismatches': 0, "
                        "'claim': 'toy verified model'}"
                    )
                },
            }
        ]
    )
    policy = EvidenceFirstCodingPolicy(model=model, max_model_calls=4, max_tool_calls=4)
    policy.choose(policy.observe(Frame(grid(), actions=(1, 2, 3, 4))))
    validation = policy.workspace.world_model_validation
    assert validation["status"] == "validated"
    assert validation["checked"] == 3
    assert validation["mismatches"] == 0


def test_probe_telemetry_counts_only_the_executed_probe():
    model = FakeModel(
        [
            {
                **probe(1),
                "plan": [
                    {"id": 1, "expect": {"board_change": "any"}},
                    {"id": 2, "expect": {"board_change": "any"}},
                ],
            }
        ]
    )
    policy = EvidenceFirstCodingPolicy(model=model, max_model_calls=4, max_tool_calls=0)
    policy.choose(policy.observe(Frame(grid(), actions=(1, 2))))
    assert policy.model_authored_actions == 1
    assert policy.model_authored_probes == 1
