import numpy as np

from arc3lab.model.adapter import extract_json
from arc3lab.policy.hybrid import HybridPolicy
from arc3lab.types import Scene


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, user, grid=None):
        self.calls += 1
        return 'prefix {"confidence": 0.9, "hypothesis": "probe", "actions": [{"id": 6, "x": 2, "y": 3}]} suffix'


def test_extract_json_with_wrapper_text():
    assert extract_json('x {"a": 1} y') == {"a": 1}


def test_hybrid_validates_action6_and_honors_model_budget():
    model = FakeModel()
    policy = HybridPolicy(model=model, max_model_calls=1)
    scene = Scene(
        grid=np.zeros((8, 8), dtype=np.int8),
        background=0,
        components=[],
        signature="state",
        level=0,
        step=0,
        available_actions=(1, 6),
    )
    first = policy._model_actions(scene)
    assert len(first) == 1
    assert first[0].action_id == 6
    assert first[0].data == {"x": 2, "y": 3}
    assert policy.model_calls == 1
    assert policy._model_actions(scene) == []
    assert model.calls == 1
