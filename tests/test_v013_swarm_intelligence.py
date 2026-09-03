from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc3lab.arena.research_agents import ResearchCall, ResearchProposal
from arc3lab.arena.swarm_intelligence import (
    ResearchReview,
    ReviewAssignment,
    SwarmCouncil,
    SwarmMemory,
    SwarmOutcome,
)


def proposal(provider: str, role: str, hypothesis: str) -> ResearchProposal:
    return ResearchProposal(
        provider_id=provider,
        role_id=role,
        hypothesis=hypothesis,
        experiment="run controlled ablation",
        target_metric="solve_rate",
        split="validation",
        falsifier="no repeatable improvement",
        implementation="small isolated patch",
        failure_mode="overfit",
        valid=True,
    )


def review(
    proposal_key: str,
    provider: str,
    role: str,
    *,
    falsifiability: float,
    generalization: float,
    information_gain: float,
    feasibility: float,
    redundancy: float = 0.0,
    persuasion_risk: float = 0.0,
    confidence: float = 0.8,
) -> ResearchReview:
    return ResearchReview(
        proposal_key=proposal_key,
        reviewer_provider_id=provider,
        reviewer_role_id=role,
        falsifiability=falsifiability,
        generalization=generalization,
        information_gain=information_gain,
        feasibility=feasibility,
        redundancy=redundancy,
        persuasion_risk=persuasion_risk,
        confidence=confidence,
        verdict="advance",
        strongest_objection="could fail on a permuted environment",
        decisive_test="repeat held-out validation against control",
        valid=True,
    )


def test_review_assignments_are_blinded_and_avoid_exact_self_review() -> None:
    p = proposal("model-a", "scientist", "test causal retrieval")
    reviewers = [
        ResearchCall("model-a", "scientist", "a"),
        ResearchCall("model-a", "red_team", "a"),
        ResearchCall("model-b", "scientist", "b"),
        ResearchCall("model-b", "planner", "b"),
        ResearchCall("model-c", "vision", "c"),
    ]
    assignments = SwarmCouncil([p]).assign_reviews(reviewers, reviews_per_proposal=3)
    assert len(assignments) == 3
    assert all(
        not (
            row.reviewer_provider_id == "model-a"
            and row.reviewer_role_id == "scientist"
        )
        for row in assignments
    )
    assert len({row.reviewer_provider_id for row in assignments}) >= 2
    assert len({row.reviewer_role_id for row in assignments}) >= 2
    payload = SwarmCouncil.blind_payload(p)
    assert "provider_id" not in payload
    assert "role_id" not in payload
    assert "raw_text" not in payload


def test_review_parser_requires_structured_decisive_critique() -> None:
    assignment = ReviewAssignment("a__scientist", "b", "red_team")
    parsed = ResearchReview.from_text(
        assignment,
        json.dumps(
            {
                "falsifiability": 0.9,
                "generalization": 0.8,
                "information_gain": 0.8,
                "feasibility": 0.7,
                "redundancy": 0.1,
                "persuasion_risk": 0.2,
                "confidence": 0.8,
                "verdict": "advance",
                "strongest_objection": "may overfit representation",
                "decisive_test": "permute representation and rerun validation",
            }
        ),
    )
    assert parsed.valid
    bad = ResearchReview.from_text(assignment, '{"verdict":"advance"}')
    assert not bad.valid


def test_persuasive_but_weak_proposal_does_not_win_council_priority() -> None:
    glossy = proposal("a", "scientist", "large persuasive mechanism")
    rigorous = proposal("b", "planner", "small falsifiable mechanism")
    reviews = [
        review(
            "a__scientist",
            "r1",
            "red_team",
            falsifiability=0.4,
            generalization=0.4,
            information_gain=0.4,
            feasibility=0.8,
            redundancy=0.8,
            persuasion_risk=1.0,
            confidence=1.0,
        ),
        review(
            "a__scientist",
            "r2",
            "generalization",
            falsifiability=0.4,
            generalization=0.4,
            information_gain=0.4,
            feasibility=0.8,
            redundancy=0.8,
            persuasion_risk=1.0,
            confidence=1.0,
        ),
        review(
            "b__planner",
            "r1",
            "scientist",
            falsifiability=0.8,
            generalization=0.8,
            information_gain=0.8,
            feasibility=0.7,
            redundancy=0.1,
            persuasion_risk=0.1,
            confidence=0.7,
        ),
        review(
            "b__planner",
            "r2",
            "red_team",
            falsifiability=0.8,
            generalization=0.8,
            information_gain=0.8,
            feasibility=0.7,
            redundancy=0.1,
            persuasion_risk=0.1,
            confidence=0.7,
        ),
    ]
    priorities = SwarmCouncil([glossy, rigorous], reviews).priorities()
    assert priorities[0].proposal_key == "b__planner"
    assert priorities[0].robust_priority > priorities[1].robust_priority


def test_high_information_disagreement_becomes_experiment_not_majority_collapse() -> None:
    p = proposal("a", "explorer", "ambiguous intervention strategy")
    reviews = [
        review(
            "a__explorer",
            "r1",
            "scientist",
            falsifiability=1.0,
            generalization=1.0,
            information_gain=0.9,
            feasibility=1.0,
            confidence=1.0,
        ),
        review(
            "a__explorer",
            "r2",
            "red_team",
            falsifiability=0.1,
            generalization=0.1,
            information_gain=0.9,
            feasibility=0.1,
            confidence=0.1,
        ),
    ]
    priority = SwarmCouncil([p], reviews).priorities()[0]
    assert priority.disagreement >= 0.12
    assert priority.dissent_experiment
    battle = SwarmCouncil([p], reviews).battle_plan(max_proposals=1, min_reviews=2)
    assert battle["selected"][0]["disagreement_experiment"] is True
    assert "arena outcomes decide promotion" in battle["authority"]


def test_selection_preserves_role_diversity_before_extra_same_role_slots() -> None:
    p1 = proposal("a", "scientist", "science one")
    p2 = proposal("b", "scientist", "science two")
    p3 = proposal("c", "vision", "vision minority")
    reviews = []
    for key, quality in (("a__scientist", 0.95), ("b__scientist", 0.90), ("c__vision", 0.70)):
        reviews.extend(
            [
                review(
                    key,
                    "r1",
                    "red_team",
                    falsifiability=quality,
                    generalization=quality,
                    information_gain=quality,
                    feasibility=quality,
                ),
                review(
                    key,
                    "r2",
                    "planner",
                    falsifiability=quality,
                    generalization=quality,
                    information_gain=quality,
                    feasibility=quality,
                ),
            ]
        )
    selected = SwarmCouncil([p1, p2, p3], reviews).select(max_proposals=2, min_reviews=2)
    assert {row[0].role_id for row in selected} == {"scientist", "vision"}


def test_swarm_memory_uses_measured_attractors_but_rejects_blind(tmp_path: Path) -> None:
    memory = SwarmMemory(tmp_path / "memory.jsonl")
    memory.append(
        SwarmOutcome(
            proposal_id="p-global",
            provider_id="a",
            role_id="planner",
            split="validation",
            utility=0.20,
            source="arena",
            note="small plan compiler won",
        )
    )
    memory.append(
        SwarmOutcome(
            proposal_id="p-personal",
            provider_id="b",
            role_id="vision",
            split="dev",
            utility=0.12,
            source="arena",
            note="delta view helped",
        )
    )
    guidance = memory.guidance("b", "vision")
    assert "p-global" in guidance
    assert "p-personal" in guidance
    assert "mutate" in guidance.lower()
    with pytest.raises(ValueError, match="DEV/VALIDATION"):
        memory.append(
            SwarmOutcome(
                proposal_id="forbidden",
                provider_id="b",
                role_id="vision",
                split="blind",
                utility=1.0,
                source="private",
            )
        )
