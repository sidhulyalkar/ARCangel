import numpy as np

from arc3lab.policy.coding import CodingPolicy
from arc3lab.policy.coding_prompt import CODING_USER_TEMPLATE
from arc3lab.types import Scene


class GoalToolModel:
    def __init__(self) -> None:
        self.calls = 0
        self.first_prompt = ""

    def complete(self, system, user, grid=None):
        self.calls += 1
        if self.calls == 1:
            self.first_prompt = user
            return (
                '{"hypothesis":"movement is deterministic","goal":"reach the unique target",'
                '"memory_note":"ACTION1 moved the controlled object","python":"result = predict(1)",'
                '"analysis_question":"is ACTION1 covered by temporal memory?","actions":[],'
                '"confidence":0.9,"plan_reliable":false,"expected_change":"move",'
                '"delegate_world_model":true}'
            )
        return (
            '{"hypothesis":"use the discriminating probe","goal":"reach the unique target",'
            '"memory_note":"","python":"","analysis_question":"","actions":[{"id":1}],'
            '"confidence":0.9,"plan_reliable":false,"expected_change":"move",'
            '"delegate_world_model":false}'
        )


def scene() -> Scene:
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[2, 3] = 2
    return Scene(
        grid=grid,
        background=0,
        components=[],
        signature="s",
        level=0,
        step=0,
        available_actions=(1, 6),
    )


def test_v005_prompt_exposes_goal_and_predictive_memory():
    assert "PERSISTENT GOAL HYPOTHESES" in CODING_USER_TEMPLATE
    assert "TEMPORAL PREDICTIVE MEMORY" in CODING_USER_TEMPLATE
    assert "predict(action_id" in CODING_USER_TEMPLATE


def test_world_model_delegation_is_counted_only_when_tool_runs():
    model = GoalToolModel()
    policy = CodingPolicy(model=model, max_model_calls=3, max_tool_calls=1)
    action = policy.choose(scene())
    assert action.action_id == 1
    assert policy.tool_calls == 1
    assert policy.world_model_delegations == 1
    assert policy.goals
    assert policy.goals[-1]["goal"] == "reach the unique target"
    assert "PERSISTENT GOAL HYPOTHESES" in model.first_prompt
    assert "TEMPORAL PREDICTIVE MEMORY" in model.first_prompt
