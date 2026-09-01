from __future__ import annotations

import json
from pathlib import Path

from arc3lab.arena.campaign import CampaignDirector
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest, ArenaResult


IDS = (
    "B-coding-minimal",
    "C-v011-reflective",
    "D-v012-evidence-first",
    "E-v012-lite",
)


def _manifest(tmp_path: Path) -> ArenaManifest:
    path = tmp_path / "manifest.json"
    contestants = [
        {
            "id": "A-duck-qwen38-public",
            "family": "duck",
            "role": "external",
            "enabled": False,
        }
    ]
    for contestant_id in IDS:
        family = {
            "B-coding-minimal": "coding",
            "C-v011-reflective": "v011",
            "D-v012-evidence-first": "v012",
            "E-v012-lite": "v012-lite",
        }[contestant_id]
        row = {
            "id": contestant_id,
            "family": family,
            "role": "test",
            "enabled": True,
            "command": ["python", "fake.py", "{contestant}", "{split}", "{seed}"],
            "judge_command": ["python", "fake.py", "{contestant}", "blind", "{seed}"],
        }
        if contestant_id != "B-coding-minimal":
            row["control_id"] = "B-coding-minimal"
        contestants.append(row)
    path.write_text(
        json.dumps(
            {
                "experiment_id": "campaign-test",
                "seeds": [1, 2],
                "leaderboard_control_id": "A-duck-qwen38-public",
                "min_leaderboard_delta": 0.0,
                "min_leaderboard_candidate_runs": 2,
                "min_leaderboard_control_runs": 2,
                "leaderboard_confidence_se": 1.0,
                "require_leaderboard_artifact_hash": True,
                "weights": {"solve_rate": 1.0},
                "promotion": {
                    "min_validation_delta": 0.02,
                    "min_dev_delta": -0.01,
                    "min_blind_delta": -0.01,
                    "max_emergency_fraction": 0.05,
                    "max_failure_rate": 0.05,
                    "min_validation_runs": 2,
                    "min_blind_runs": 2,
                    "require_control": True,
                },
                "contestants": contestants,
            }
        )
    )
    return ArenaManifest.load(path)


def _result(cid: str, split: str, seed: int, score: float) -> ArenaResult:
    return ArenaResult(
        contestant_id=cid,
        split=split,
        seed=seed,
        metrics={
            "solve_rate": score,
            "failure_rate": 0.0,
            "emergency_fraction": 0.0,
        },
    )


def _kaggle(cid: str, seed: int, score: float, sha: str) -> ArenaResult:
    return ArenaResult(
        contestant_id=cid,
        split="kaggle",
        seed=seed,
        metrics={"official_score": score},
        metadata={"artifact_sha256": sha},
    )


def _director(tmp_path: Path, lab: ArenaOrchestrator) -> CampaignDirector:
    root = tmp_path / "arena"
    return CampaignDirector(
        lab,
        root=root,
        public_registry=root / "splits.public.json",
        private_registry=root / "splits.private.json",
        package_dir=root / "packages",
    )


def _write_splits(tmp_path: Path) -> None:
    root = tmp_path / "arena"
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits.public.json").write_text('{"dev":["a"],"validation":["b"]}')
    (root / "splits.private.json").write_text(
        '{"dev":["a"],"validation":["b"],"blind":["c"],"salt":"secret"}'
    )


def _complete_internal_tournament(lab: ArenaOrchestrator) -> None:
    rows: list[ArenaResult] = []
    for seed in (1, 2):
        rows.extend(
            [
                _result("B-coding-minimal", "dev", seed, 0.40),
                _result("B-coding-minimal", "validation", seed, 0.40),
                _result("D-v012-evidence-first", "dev", seed, 0.50),
                _result("D-v012-evidence-first", "validation", seed, 0.50),
            ]
        )
    # C/E complete the primary DEV screen but are eliminated cheaply.
    rows.extend(
        [
            _result("C-v011-reflective", "dev", 1, 0.10),
            _result("E-v012-lite", "dev", 1, 0.10),
        ]
    )
    lab.ledger.extend(rows)


def _complete_blind(lab: ArenaOrchestrator) -> None:
    lab.ledger.extend(
        [
            _result("B-coding-minimal", "blind", 1, 0.40),
            _result("B-coding-minimal", "blind", 2, 0.40),
            _result("D-v012-evidence-first", "blind", 1, 0.50),
            _result("D-v012-evidence-first", "blind", 2, 0.50),
        ]
    )


def _write_package(tmp_path: Path, sha: str = "candidatehash") -> None:
    package_dir = tmp_path / "arena" / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "D-v012-evidence-first.json").write_text(
        json.dumps(
            {
                "contestant_id": "D-v012-evidence-first",
                "family": "v012",
                "profile": "v012",
                "build_id": "test",
                "notebook": "kaggle/candidate.ipynb",
                "notebook_sha256": sha,
                "status": "PACKAGED_AND_VERIFIED",
            }
        )
    )


def test_campaign_requires_splits_before_any_experiment(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    assert _director(tmp_path, lab).decide().state == "NEED_SPLITS"


def test_campaign_moves_from_tournament_to_blind_to_package(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    _write_splits(tmp_path)
    assert _director(tmp_path, lab).decide().state == "NEED_TOURNAMENT"

    _complete_internal_tournament(lab)
    decision = _director(tmp_path, lab).decide()
    assert decision.state == "NEED_BLIND_JUDGE"
    assert "D-v012-evidence-first" in decision.details["promoted"]

    _complete_blind(lab)
    assert _director(tmp_path, lab).decide().state == "NEED_PACKAGE"


def test_campaign_external_evidence_sequence_is_exact_artifact(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    _write_splits(tmp_path)
    _complete_internal_tournament(lab)
    _complete_blind(lab)
    _write_package(tmp_path)

    decision = _director(tmp_path, lab).decide()
    assert decision.state == "NEED_DUCK_CONTROL"

    lab.ledger.extend(
        [
            _kaggle("A-duck-qwen38-public", 10, 1.80, "duckhash"),
            _kaggle("A-duck-qwen38-public", 11, 1.82, "duckhash"),
        ]
    )
    decision = _director(tmp_path, lab).decide()
    assert decision.state == "NEED_KAGGLE_CANDIDATE_RUN"
    assert decision.details["artifact_sha256"] == "candidatehash"

    # Two volatile runs meet the count requirement but do not establish superiority.
    lab.ledger.extend(
        [
            _kaggle("D-v012-evidence-first", 10, 2.11, "candidatehash"),
            _kaggle("D-v012-evidence-first", 11, 0.89, "candidatehash"),
        ]
    )
    assert _director(tmp_path, lab).decide().state == "NEED_MORE_KAGGLE_EVIDENCE"


def test_campaign_declares_nominee_only_after_robust_exact_artifact_win(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    _write_splits(tmp_path)
    _complete_internal_tournament(lab)
    _complete_blind(lab)
    _write_package(tmp_path)
    lab.ledger.extend(
        [
            _kaggle("A-duck-qwen38-public", 10, 1.80, "duckhash"),
            _kaggle("A-duck-qwen38-public", 11, 1.82, "duckhash"),
            _kaggle("D-v012-evidence-first", 10, 2.10, "candidatehash"),
            _kaggle("D-v012-evidence-first", 11, 2.12, "candidatehash"),
        ]
    )
    decision = _director(tmp_path, lab).decide()
    assert decision.state == "LEADERBOARD_NOMINEE"
    assert decision.details["nominee"]["candidate_artifact_sha256"] == "candidatehash"
