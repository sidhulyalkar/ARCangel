import numpy as np

from arc3lab.policy.coding import CodingPolicy
from arc3lab.policy.sandbox import SandboxError, run_analysis_code
from arc3lab.types import Scene


class ToolThenActModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, user, grid=None):
        self.calls += 1
        if self.calls == 1:
            return '{"hypothesis":"need exact count","python":"result = sum(row.count(2) for row in grid)","analysis_question":"count 2s","actions":[],"confidence":0.8}'
        return '{"hypothesis":"counted target","memory_note":"ACTION1 is the best current probe","python":"","actions":[{"id":1}],"confidence":0.9,"plan_reliable":false}'


def scene():
    g = np.zeros((8, 8), dtype=np.int8)
    g[2, 3] = 2
    return Scene(
        grid=g,
        background=0,
        components=[],
        signature="s",
        level=0,
        step=0,
        available_actions=(1, 6),
    )


def test_sandbox_can_analyze_but_not_import():
    ctx = {"grid": [[0, 1], [1, 1]]}
    assert run_analysis_code("result = sum(sum(r) for r in grid)", ctx) == "3"
    try:
        run_analysis_code("import os\nresult=1", ctx)
    except SandboxError:
        pass
    else:
        raise AssertionError("import should be blocked")


def test_coding_policy_tool_round_then_action():
    model = ToolThenActModel()
    policy = CodingPolicy(model=model, max_model_calls=3, max_tool_calls=1)
    action = policy.choose(scene())
    assert action.action_id == 1
    assert policy.model_calls == 2
    assert policy.tool_calls == 1
    assert model.calls == 2
    assert policy.beliefs


def test_coding_policy_supports_uncapped_budgets():
    policy = CodingPolicy(model=ToolThenActModel(), max_model_calls=None, max_tool_calls=None)
    assert policy._model_budget_available()
    assert policy._tool_budget_available()


def test_later_level_does_not_force_simple_action_reprobe():
    policy = CodingPolicy(model=None)
    s = scene()
    s.level = 1
    action = policy.choose(s)
    assert action.reason != "first-level action-effect probe"
