from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc3lab.arena.leaderboard import artifact_evidence
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest, ArenaResult
from arc3lab.arena.scoring import score_result


def _manifest(tmp_path: Path) -> ArenaManifest:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "noise-test",
                "seeds": [1, 2],
                "leaderboard_control_id": "duck",
                "min_leaderboard_delta": 0.0,
                "min_leaderboard_candidate_runs": 2,
                "min_leaderboard_control_runs": 2,
                "leaderboard_confidence_se": 1.0,
                "require_leaderboard_artifact_hash": True,
                "weights": {"solve_rate": 1.0},
                "promotion": {
                    "min_validation_delta": 0.0,
                    "min_dev_delta": -1.0,
                    "min_blind_delta": -1.0,
                    "min_validation_runs": 1,
                    "min_blind_runs": 1,
                    "max_emergency_fraction": 1.0,
                    "max_failure_rate": 1.0,
                    "require_control": True,
                },
                "contestants": [
                    {"id": "duck", "family": "duck", "role": "external", "enabled": False},
                    {"id": "control", "family": "coding", "role": "control", "enabled": False},
                    {
                        "id": "challenger",
                        "family": "v012",
                        "role": "scientist",
                        "control_id": "control",
                        "enabled": False,
                    },
                ],
            }
        )
    )
    return ArenaManifest.load(path)


def _behavior(cid: str, split: str, score: float) -> ArenaResult:
    return ArenaResult(
        contestant_id=cid,
        split=split,
        seed=1,
        metrics={"solve_rate": score, "failure_rate": 0.0, "emergency_fraction": 0.0},
    )


def _make_ready(lab: ArenaOrchestrator) -> None:
    lab.ledger.extend(
        [
            _behavior("control", "dev", 0.2),
            _behavior("control", "validation", 0.2),
            _behavior("challenger", "dev", 0.4),
            _behavior("challenger", "validation", 0.4),
            _behavior("control", "blind", 0.2),
            _behavior("challenger", "blind", 0.4),
        ]
    )
    assert lab.kaggle_ready_queue()


def _kaggle(cid: str, seed: int, score: float, artifact: str) -> ArenaResult:
    return ArenaResult(
        contestant_id=cid,
        split="kaggle",
        seed=seed,
        metrics={"official_score": score},
        metadata={"artifact_sha256": artifact},
    )


def test_degraded_suite_retains_gradient_instead_of_flat_failure() -> None:
    degraded = ArenaResult(
        contestant_id="x",
        split="validation",
        seed=1,
        status="degraded",
        metrics={"solve_rate": 0.8, "failure_rate": 0.1},
    )
    failed = ArenaResult(
        contestant_id="x",
        split="validation",
        seed=2,
        status="failed",
        metrics={"solve_rate": 0.8, "failure_rate": 0.1},
    )
    assert score_result(degraded, {"solve_rate": 1.0}) > -1.0
    assert score_result(failed, {"solve_rate": 1.0}) == -1.0


def test_normal_plan_cannot_masquerade_as_kaggle_run(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    with pytest.raises(ValueError, match="external"):
        lab.plan(splits=["kaggle"])


def test_strict_kaggle_receipt_requires_exact_artifact_hash(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    with pytest.raises(ValueError, match="SHA-256"):
        lab.record_kaggle_score(
            contestant_id="duck",
            score=2.0,
            seed=1,
            source="copy-edit",
        )


def test_artifact_evidence_never_pools_different_notebook_hashes() -> None:
    rows = [
        _kaggle("x", 1, 2.0, "aaa"),
        _kaggle("x", 2, 2.2, "aaa"),
        _kaggle("x", 3, 9.9, "bbb"),
    ]
    evidence = artifact_evidence(rows, "x", require_hash=True)
    assert len(evidence) == 2
    aaa = next(row for row in evidence if row.artifact_sha256 == "aaa")
    bbb = next(row for row in evidence if row.artifact_sha256 == "bbb")
    assert aaa.runs == 2
    assert abs(aaa.mean_score - 2.1) < 1e-12
    assert bbb.runs == 1


def test_leaderboard_requires_repeated_exact_artifact_evidence(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    _make_ready(lab)
    lab.ledger.extend(
        [
            _kaggle("duck", 10, 2.0, "duckhash"),
            _kaggle("challenger", 10, 2.4, "candidatehash"),
        ]
    )
    assert lab.leaderboard_queue() == []

    lab.ledger.extend(
        [
            _kaggle("duck", 11, 2.0, "duckhash"),
            _kaggle("challenger", 11, 2.4, "candidatehash"),
        ]
    )
    queue = lab.leaderboard_queue()
    assert len(queue) == 1
    assert queue[0]["candidate_artifact_sha256"] == "candidatehash"
    assert queue[0]["control_artifact_sha256"] == "duckhash"
    assert queue[0]["candidate_runs"] == 2
    assert queue[0]["control_runs"] == 2


def test_noisy_candidate_must_win_after_uncertainty_penalty(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    _make_ready(lab)
    # The candidate has a higher mean, but its 2.11/0.89 style variance is too large
    # to call it a robust win from only two exact-artifact repeats.
    lab.ledger.extend(
        [
            _kaggle("duck", 10, 1.80, "duckhash"),
            _kaggle("duck", 11, 1.82, "duckhash"),
            _kaggle("challenger", 10, 2.11, "candidatehash"),
            _kaggle("challenger", 11, 0.89, "candidatehash"),
        ]
    )
    assert lab.leaderboard_queue() == []
