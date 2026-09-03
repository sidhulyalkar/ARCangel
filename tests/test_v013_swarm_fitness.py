from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from arc3lab.arena.research_agents import ResearchProposal
from arc3lab.arena.schema import ArenaManifest, ArenaResult
from arc3lab.arena.swarm_fitness import evaluate_swarm_fitness
from arc3lab.arena.swarm_intelligence import SwarmCouncil, SwarmMemory, SwarmOutcome


def manifest(tmp_path: Path) -> ArenaManifest:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "swarm-fitness-test",
                "seeds": [1, 2, 3],
                "weights": {"solve_rate": 1.0},
                "promotion": {
                    "max_emergency_fraction": 0.02,
                    "max_failure_rate": 0.05,
                    "min_validation_runs": 2,
                    "min_blind_runs": 2,
                },
                "contestants": [
                    {
                        "id": "control",
                        "family": "coding",
                        "role": "control",
                        "enabled": False,
                    }
                ],
            }
        )
    )
    return ArenaManifest.load(path)


def result(cid: str, seed: int, value: float, *, failure: float = 0.0) -> ArenaResult:
    return ArenaResult(
        contestant_id=cid,
        split="validation",
        seed=seed,
        metrics={
            "solve_rate": value,
            "failure_rate": failure,
            "emergency_fraction": 0.0,
        },
    )


def battle() -> dict[str, object]:
    return {
        "proposal_id": "SWARM-01-scientist",
        "provider_id": "model-a",
        "role_id": "scientist",
        "selection_split": "validation",
        "target_profile": "v012",
        "control_profile": "coding-minimal",
    }


def test_swarm_proposal_requires_executable_contract_for_council() -> None:
    legacy = ResearchProposal.from_text(
        "model-a",
        "scientist",
        json.dumps(
            {
                "hypothesis": "test memory",
                "experiment": "ablate retrieval",
                "target_metric": "solve_rate",
                "split": "validation",
                "falsifier": "no gain",
                "implementation": "small patch",
                "failure_mode": "overfit",
            }
        ),
    )
    assert legacy.valid
    assert not legacy.executable_contract_valid
    assert SwarmCouncil([legacy]).eligible_proposals() == []

    executable = ResearchProposal.from_text(
        "model-a",
        "scientist",
        json.dumps(
            {
                "hypothesis": "test memory",
                "experiment": "ablate retrieval",
                "target_metric": "solve_rate",
                "split": "validation",
                "falsifier": "no gain",
                "implementation": "small patch",
                "failure_mode": "overfit",
                "target_profile": "v012",
                "control_profile": "coding-minimal",
            }
        ),
    )
    assert executable.executable_contract_valid
    assert SwarmCouncil([executable]).eligible_proposals() == [executable]


def test_v011_cannot_be_reintroduced_as_swarm_target() -> None:
    proposal = ResearchProposal(
        provider_id="a",
        role_id="scientist",
        hypothesis="revive old controller",
        experiment="retry",
        target_metric="solve_rate",
        split="validation",
        falsifier="no gain",
        implementation="patch old code",
        failure_mode="known failure",
        target_profile="v011",
        control_profile="coding-minimal",
    )
    assert not proposal.executable_contract_valid
    assert not SwarmCouncil([proposal]).eligible_proposals()


def test_paired_fitness_uses_lower_bound_and_repeatability(tmp_path: Path) -> None:
    evidence = evaluate_swarm_fitness(
        battle(),
        [result("candidate", 1, 0.60), result("candidate", 2, 0.50)],
        [result("control", 1, 0.40), result("control", 2, 0.45)],
        manifest(tmp_path),
    )
    assert evidence.runs == 2
    assert evidence.mean_delta == pytest.approx(0.125)
    assert evidence.robust_delta < evidence.mean_delta
    assert evidence.robust_delta > 0
    assert evidence.memory_status == "measured"
    outcome = evidence.to_outcome("paired-receipts")
    assert outcome.utility == evidence.robust_delta
    assert outcome.status == "measured"


def test_one_run_can_be_audited_but_not_become_swarm_attractor(tmp_path: Path) -> None:
    evidence = evaluate_swarm_fitness(
        battle(),
        [result("candidate", 1, 0.90)],
        [result("control", 1, 0.20)],
        manifest(tmp_path),
    )
    assert evidence.robust_delta == pytest.approx(0.70)
    assert evidence.memory_status == "preliminary"
    memory = SwarmMemory(tmp_path / "memory.jsonl")
    memory.append(evidence.to_outcome("lucky-single-run"))
    guidance = memory.guidance("model-a", "scientist")
    assert "global_best: none measured yet" in guidance


def test_unhealthy_candidate_cannot_become_measured_attractor(tmp_path: Path) -> None:
    evidence = evaluate_swarm_fitness(
        battle(),
        [
            result("candidate", 1, 0.90, failure=0.20),
            result("candidate", 2, 0.90, failure=0.20),
        ],
        [result("control", 1, 0.20), result("control", 2, 0.20)],
        manifest(tmp_path),
    )
    assert evidence.robust_delta > 0
    assert evidence.memory_status == "preliminary"


def test_fitness_requires_paired_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no paired seeds"):
        evaluate_swarm_fitness(
            battle(),
            [result("candidate", 1, 0.60)],
            [result("control", 2, 0.40)],
            manifest(tmp_path),
        )


def test_duplicate_seed_receipts_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate validation result"):
        evaluate_swarm_fitness(
            battle(),
            [result("candidate", 1, 0.60), result("candidate", 1, 0.61)],
            [result("control", 1, 0.40)],
            manifest(tmp_path),
        )


def test_swarm_memory_rejects_nonfinite_utility_and_private_evidence(tmp_path: Path) -> None:
    memory = SwarmMemory(tmp_path / "memory.jsonl")
    with pytest.raises(ValueError, match="finite"):
        memory.append(
            SwarmOutcome(
                proposal_id="nan",
                provider_id="a",
                role_id="scientist",
                split="validation",
                utility=math.nan,
                source="bad",
            )
        )
    with pytest.raises(ValueError, match="DEV/VALIDATION"):
        memory.append(
            SwarmOutcome(
                proposal_id="blind",
                provider_id="a",
                role_id="scientist",
                split="blind",
                utility=1.0,
                source="private",
            )
        )


def test_zero_swarm_budget_selects_nothing() -> None:
    proposal = ResearchProposal(
        provider_id="a",
        role_id="scientist",
        hypothesis="test",
        experiment="test",
        target_metric="solve_rate",
        split="validation",
        falsifier="no gain",
        implementation="patch",
        failure_mode="overfit",
        target_profile="v012",
        control_profile="coding-minimal",
    )
    assert SwarmCouncil([proposal]).select(max_proposals=0) == []
