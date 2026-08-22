from arc3lab.memory.predictive import PredictiveTransitionMemory, predictive_state_key
from arc3lab.types import ActionSpec, Transition


def transition(before: str, after: str, action_id: int, step: int) -> Transition:
    return Transition(
        step=step,
        level=0,
        before_signature=before,
        action=ActionSpec(action_id),
        after_signature=after,
        changed_cells=1,
        meaningful_changed_cells=1,
    )


def test_h2_key_uses_last_two_state_action_pairs():
    history = [
        transition("s0", "s1", 1, 1),
        transition("s1", "s2", 2, 2),
        transition("s2", "s3", 3, 3),
    ]
    key = predictive_state_key("s3", history, history_depth=2)
    same_tail = predictive_state_key(
        "s3",
        [transition("other", "s1", 7, 0), *history[-2:]],
        history_depth=2,
    )
    different_tail = predictive_state_key(
        "s3",
        [history[0], history[1], transition("s2", "s3", 4, 3)],
        history_depth=2,
    )
    assert key == same_tail
    assert key != different_tail


def test_prediction_is_distributional_and_reports_evidence():
    memory = PredictiveTransitionMemory(history_depth=2)
    action = ActionSpec(1)
    memory.observe("state", action, "next", "change")
    memory.observe("state", action, "next", "change")
    prediction = memory.prediction("state", action)
    assert prediction is not None
    assert prediction["next_signature"] == "next"
    assert prediction["confidence"] == 1.0
    assert prediction["evidence"] == 2
    assert prediction["effect"] == "change"


def test_context_coverage_counts_only_observed_actions():
    memory = PredictiveTransitionMemory()
    first = ActionSpec(1)
    second = ActionSpec(2)
    memory.observe("state", first, "next", "change")
    assert memory.context_coverage("state", [first, second]) == 0.5
