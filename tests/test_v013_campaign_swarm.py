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


def manifest(tmp_path: Path) -> ArenaManifest:
    path = tmp_path / "manifest.json"
    contestants = []
    for cid in IDS:
        row = {
            "id": cid,
            "family": "coding" if cid == "B-coding-minimal" else "v012",
            "role": "test",
            "enabled": True,
            "command": ["python", "fake.py", "{contestant}", "{split}", "{seed}"],
            "judge_command": ["python", "fake.py", "{contestant}", "blind", "{seed}"],
        }
        if cid != "B-coding-minimal":
            row["control_id"] = "B-coding-minimal"
        contestants.append(row)
    path.write_text(
        json.dumps(
            {
                "experiment_id": "campaign-swarm-test",
                "seeds": [1, 2],
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


def result(cid: str, split: str, seed: int, score: float) -> ArenaResult:
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


def director(tmp_path: Path, lab: ArenaOrchestrator) -> CampaignDirector:
    root = tmp_path / "arena"
    root.mkdir(parents=True, exist_ok=True)
    (root / "splits.public.json").write_text('{"dev":["a"],"validation":["b"]}')
    (root / "splits.private.json").write_text('{"blind":["c"],"salt":"secret"}')
    return CampaignDirector(
        lab,
        root=root,
        public_registry=root / "splits.public.json",
        private_registry=root / "splits.private.json",
        package_dir=root / "packages",
    )


def test_failed_dev_screen_routes_to_heterogeneous_swarm(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(manifest(tmp_path), tmp_path / "arena")
    lab.ledger.extend(
        [
            result("B-coding-minimal", "dev", 1, 0.40),
            result("C-v011-reflective", "dev", 1, 0.10),
            result("D-v012-evidence-first", "dev", 1, 0.10),
            result("E-v012-lite", "dev", 1, 0.10),
        ]
    )
    decision = director(tmp_path, lab).decide()
    assert decision.state == "NEED_SWARM_RESEARCH"
    assert decision.details["generation"] == 1
    assert "nvidia-swarm" in decision.details["providers"]
    assert "paired arena evidence" in decision.details["authority"]


def test_completed_blind_rejection_routes_back_to_swarm_without_private_scores(
    tmp_path: Path,
) -> None:
    lab = ArenaOrchestrator(manifest(tmp_path), tmp_path / "arena")
    rows = []
    for seed in (1, 2):
        rows.extend(
            [
                result("B-coding-minimal", "dev", seed, 0.40),
                result("B-coding-minimal", "validation", seed, 0.40),
                result("D-v012-evidence-first", "dev", seed, 0.50),
                result("D-v012-evidence-first", "validation", seed, 0.50),
                result("B-coding-minimal", "blind", seed, 0.40),
                result("D-v012-evidence-first", "blind", seed, 0.10),
            ]
        )
    rows.extend(
        [
            result("C-v011-reflective", "dev", 1, 0.10),
            result("E-v012-lite", "dev", 1, 0.10),
        ]
    )
    lab.ledger.extend(rows)
    decision = director(tmp_path, lab).decide()
    assert decision.state == "NEED_SWARM_RESEARCH"
    assert "BLIND" in decision.reason
    serialized = json.dumps(decision.to_dict())
    assert '"0.1"' not in serialized
    assert "private BLIND judge rejected" in decision.details["falsifier"]


def test_partial_blind_evidence_still_requests_judge_not_swarm(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(manifest(tmp_path), tmp_path / "arena")
    rows = []
    for seed in (1, 2):
        rows.extend(
            [
                result("B-coding-minimal", "dev", seed, 0.40),
                result("B-coding-minimal", "validation", seed, 0.40),
                result("D-v012-evidence-first", "dev", seed, 0.50),
                result("D-v012-evidence-first", "validation", seed, 0.50),
            ]
        )
    rows.extend(
        [
            result("C-v011-reflective", "dev", 1, 0.10),
            result("E-v012-lite", "dev", 1, 0.10),
            result("B-coding-minimal", "blind", 1, 0.40),
            result("D-v012-evidence-first", "blind", 1, 0.50),
        ]
    )
    lab.ledger.extend(rows)
    decision = director(tmp_path, lab).decide()
    assert decision.state == "NEED_BLIND_JUDGE"
    assert decision.details["missing_blind"][0]["required_runs"] == 2
