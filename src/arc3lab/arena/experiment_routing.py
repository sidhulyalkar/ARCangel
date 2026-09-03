from __future__ import annotations

from copy import deepcopy
from typing import Any


RUNNABLE_PROFILES = frozenset({"coding-minimal", "v011", "v012", "v012-lite"})


def classify_experiment(row: dict[str, Any]) -> str:
    """Route a council experiment without trusting proposal rhetoric as execution authority.

    Existing-profile comparisons are measurement questions and should be answered before any
    model rewrites cognition. A proposal is treated as a mutation when its experiment explicitly
    requires a new variant/mechanism or when the declared profiles are not already runnable.
    """

    target = str(row.get("target_profile", "")).strip()
    control = str(row.get("control_profile", "")).strip()
    experiment = " ".join(str(row.get("experiment", "")).lower().split())

    if target not in RUNNABLE_PROFILES or control not in RUNNABLE_PROFILES:
        return "cognition_patch"
    if target == control:
        return "cognition_patch"

    explicit_mutation = (
        experiment.startswith("patch ")
        or "+" in experiment
        or " patched " in f" {experiment} "
        or " new variant" in experiment
        or " variant that" in experiment
        or " add " in f" {experiment} "
        or " modify " in f" {experiment} "
        or " replace " in f" {experiment} "
    )
    if explicit_mutation:
        return "cognition_patch"

    measurement_cue = (
        experiment.startswith("compare ")
        or experiment.startswith("run ")
        or " compare " in f" {experiment} "
        or " identical dev/validation " in f" {experiment} "
        or " identical splits " in f" {experiment} "
    )
    if measurement_cue:
        return "existing_profile_comparison"

    return "cognition_patch"


def _plan(template: dict[str, Any], rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    payload = {
        "phase": phase,
        "generation": template.get("generation"),
        "selected": rows,
        "selected_count": len(rows),
        "authority": (
            "routing chooses execution lane only; paired DEV/VALIDATION evidence remains authoritative"
        ),
    }
    return payload


def route_battle_plan(battle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return enriched battle, deduplicated measurement plan, and mutation plan."""

    enriched = deepcopy(battle)
    measurement: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    comparison_index: dict[tuple[str, str], int] = {}

    for raw in enriched.get("selected", []):
        row = dict(raw)
        kind = classify_experiment(row)
        row["experiment_kind"] = kind
        raw["experiment_kind"] = kind

        if kind == "cognition_patch":
            mutations.append(row)
            continue

        target = str(row["target_profile"])
        control = str(row["control_profile"])
        pair = tuple(sorted((target, control)))
        if pair in comparison_index:
            kept = measurement[comparison_index[pair]]
            aliases = list(kept.get("source_proposal_ids", [kept["proposal_id"]]))
            aliases.append(str(row["proposal_id"]))
            kept["source_proposal_ids"] = aliases
            kept.setdefault("deduplicated_hypotheses", []).append(str(row.get("hypothesis", "")))
            continue

        row["source_proposal_ids"] = [str(row["proposal_id"])]
        comparison_index[pair] = len(measurement)
        measurement.append(row)

    evaluation_plan = _plan(enriched, measurement, "swarm-existing-profile-measurements")
    mutation_plan = _plan(enriched, mutations, "swarm-cognition-mutations")
    enriched["measurement_count"] = len(measurement)
    enriched["mutation_count"] = len(mutations)
    enriched["routing_authority"] = (
        "measure existing profiles first; only genuinely new mechanisms may enter the coding lane"
    )
    return enriched, evaluation_plan, mutation_plan
