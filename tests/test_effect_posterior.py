import numpy as np

from arc3lab.policy.effect_posterior import EffectPosteriorPolicy
from arc3lab.types import ActionSpec, Scene, Transition


def scene() -> Scene:
    return Scene(
        grid=np.zeros((6, 6), dtype=np.int8),
        background=0,
        components=[],
        signature="current",
        level=1,
        step=10,
        available_actions=(1, 2),
    )


def append(policy: EffectPosteriorPolicy, action_id: int, changed: bool, step: int) -> None:
    policy.memory.append(
        Transition(
            step=step,
            level=0,
            before_signature=f"b{step}",
            action=ActionSpec(action_id),
            after_signature=f"a{step}",
            changed_cells=int(changed),
            meaningful_changed_cells=int(changed),
        )
    )


def test_effect_posterior_softly_prefers_demonstrated_change():
    policy = EffectPosteriorPolicy()
    for step in range(4):
        append(policy, 1, True, step)
        append(policy, 2, False, step + 10)

    action = policy.choose(scene())
    assert action.action_id == 1
    assert action.reason == "D110R2 causal effect posterior"


def test_later_level_does_not_force_primitive_reprobe():
    policy = EffectPosteriorPolicy()
    action = policy.choose(scene())
    assert action.reason != "first-level action-effect probe"
