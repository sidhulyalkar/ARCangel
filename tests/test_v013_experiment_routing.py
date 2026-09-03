from __future__ import annotations

from arc3lab.arena.experiment_routing import classify_experiment, route_battle_plan


def _row(
    proposal_id: str,
    *,
    target: str,
    control: str,
    experiment: str,
    hypothesis: str = "test hypothesis",
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "role_id": "scientist",
        "provider_id": "test-provider",
        "hypothesis": hypothesis,
        "experiment": experiment,
        "target_metric": "robust_delta",
        "selection_split": "dev",
        "falsifier": "non-positive paired delta",
        "implementation": "measure the declared comparison",
        "failure_mode": "extra machinery may hurt",
        "target_profile": target,
        "control_profile": control,
    }


def test_existing_profile_comparison_is_measured_before_coding() -> None:
    row = _row(
        "planner",
        target="v012",
        control="coding-minimal",
        experiment="Run V012 and coding-minimal on identical DEV/VALIDATION splits and seeds.",
    )
    assert classify_experiment(row) == "existing_profile_comparison"


def test_reverse_profile_comparisons_are_deduplicated() -> None:
    planner = _row(
        "planner",
        target="v012",
        control="coding-minimal",
        experiment="Compare V012 against coding-minimal on identical splits.",
        hypothesis="ledger helps",
    )
    runtime = _row(
        "runtime",
        target="coding-minimal",
        control="v012",
        experiment="Run coding-minimal versus V012 on identical DEV/VALIDATION seeds.",
        hypothesis="minimalism helps",
    )
    memory = _row(
        "memory",
        target="v012",
        control="v012-lite",
        experiment="Compare V012 and V012-lite on identical splits.",
    )
    battle = {"generation": 1, "selected_count": 3, "selected": [planner, runtime, memory]}

    routed, evaluation, mutation = route_battle_plan(battle)

    assert routed["measurement_count"] == 2
    assert routed["mutation_count"] == 0
    assert evaluation["selected_count"] == 2
    assert mutation["selected_count"] == 0
    first = evaluation["selected"][0]
    assert first["proposal_id"] == "planner"
    assert first["source_proposal_ids"] == ["planner", "runtime"]
    assert first["deduplicated_hypotheses"] == ["minimalism helps"]


def test_explicit_new_mechanism_stays_in_coding_lane() -> None:
    row = _row(
        "probe",
        target="v012-lite",
        control="coding-minimal",
        experiment="Patch v012-lite to add disagreement-guided probe selection, then compare it.",
    )
    assert classify_experiment(row) == "cognition_patch"


def test_named_variant_stays_in_coding_lane() -> None:
    row = _row(
        "retrodict",
        target="v012-lite",
        control="v012",
        experiment="Create a new variant that enforces retrodiction before action selection.",
    )
    assert classify_experiment(row) == "cognition_patch"


def test_unknown_profile_cannot_bypass_mutation_guard() -> None:
    row = _row(
        "unknown",
        target="future-agent",
        control="coding-minimal",
        experiment="Compare future-agent against coding-minimal.",
    )
    assert classify_experiment(row) == "cognition_patch"
