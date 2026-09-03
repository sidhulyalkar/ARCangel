from __future__ import annotations

import json
import tarfile
from pathlib import Path

from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.research_context import (
    assert_research_payload_safe,
    build_research_scorecard,
    sanitize_research_payload,
)
from arc3lab.arena.research_packet import ResearchPacketBuilder
from arc3lab.arena.schema import ArenaManifest, ArenaResult


def manifest(tmp_path: Path) -> ArenaManifest:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "research-context-test",
                "seeds": [1, 2],
                "weights": {"solve_rate": 1.0},
                "promotion": {
                    "min_validation_delta": 0.0,
                    "min_dev_delta": -1.0,
                    "min_blind_delta": -1.0,
                    "max_emergency_fraction": 1.0,
                    "max_failure_rate": 1.0,
                    "min_validation_runs": 1,
                    "min_blind_runs": 1,
                    "require_control": True,
                },
                "contestants": [
                    {
                        "id": "control",
                        "family": "coding",
                        "role": "control",
                        "enabled": False,
                    },
                    {
                        "id": "candidate",
                        "family": "v012",
                        "role": "challenger",
                        "control_id": "control",
                        "enabled": False,
                    },
                ],
            }
        )
    )
    return ArenaManifest.load(path)


def result(cid: str, split: str, seed: int, score: float) -> ArenaResult:
    metrics = {"solve_rate": score}
    if split == "kaggle":
        metrics = {"official_score": score}
    return ArenaResult(
        contestant_id=cid,
        split=split,
        seed=seed,
        metrics=metrics,
        metadata={"artifact_sha256": "secret-hash"} if split == "kaggle" else {},
    )


def test_research_scorecard_reads_dev_validation_only(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(manifest(tmp_path), tmp_path / "arena")
    lab.ledger.extend(
        [
            result("control", "dev", 1, 0.30),
            result("candidate", "dev", 1, 0.40),
            result("control", "validation", 1, 0.30),
            result("candidate", "validation", 1, 0.45),
            result("control", "blind", 1, 0.99),
            result("candidate", "blind", 1, 0.01),
            result("control", "kaggle", 9, 2.23),
        ]
    )
    card = build_research_scorecard(lab)
    assert card["evidence_scope"] == ["dev", "validation"]
    assert card["result_count"] == 4
    assert set(card["rankings"]) == {"dev", "validation"}
    serialized = json.dumps(card).lower()
    assert "blind" not in serialized
    assert "kaggle" not in serialized
    assert "leaderboard" not in serialized
    assert "secret-hash" not in serialized
    assert_research_payload_safe(card)


def test_recursive_sanitizer_removes_nested_dynamic_evidence() -> None:
    unsafe = {
        "rankings": {"dev": [1], "blind": [2]},
        "leaderboard_evidence": {
            "candidate_artifacts": [{"kaggle_score": 2.0}],
        },
        "safe": {
            "validation": [{"score": 0.4}],
            "blind_delta": 0.9,
        },
    }
    clean = sanitize_research_payload(unsafe)
    assert clean == {
        "rankings": {"dev": [1]},
        "safe": {"validation": [{"score": 0.4}]},
    }
    assert_research_payload_safe(clean)


def test_research_packet_removes_ambiguous_result_count_from_judge_scorecard(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "public.md").write_text("public context")
    packet = tmp_path / "packet.tar.gz"
    ResearchPacketBuilder(repo).build(
        packet,
        experiment_id="x",
        scorecard={
            "result_count": 99,
            "rankings": {
                "dev": [{"contestant_id": "a"}],
                "blind": [{"contestant_id": "private"}],
            },
            "leaderboard_evidence": {
                "control_artifacts": [{"artifact_sha256": "should-not-leak"}]
            },
            "leaderboard_queue": [{"kaggle_delta": 1.0}],
        },
        include_paths=["public.md"],
    )
    with tarfile.open(packet, "r:gz") as archive:
        handle = archive.extractfile("arena/scorecard.json")
        assert handle is not None
        scorecard = json.loads(handle.read())
    assert "result_count" not in scorecard
    serialized = json.dumps(scorecard).lower()
    assert "blind" not in serialized
    assert "kaggle" not in serialized
    assert "leaderboard" not in serialized
    assert "should-not-leak" not in serialized


def test_research_packet_preserves_safe_dev_validation_count(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    packet = tmp_path / "safe.tar.gz"
    ResearchPacketBuilder(repo).build(
        packet,
        experiment_id="x",
        scorecard={
            "result_count": 4,
            "evidence_scope": ["dev", "validation"],
            "rankings": {"dev": [], "validation": []},
        },
        include_paths=[],
    )
    with tarfile.open(packet, "r:gz") as archive:
        handle = archive.extractfile("arena/scorecard.json")
        assert handle is not None
        scorecard = json.loads(handle.read())
    assert scorecard["result_count"] == 4
    assert scorecard["evidence_scope"] == ["dev", "validation"]
