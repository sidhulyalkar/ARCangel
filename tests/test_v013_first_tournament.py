from __future__ import annotations

import json
from pathlib import Path

from arc3lab.arena.first_tournament import FirstTournamentDirector
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest, ArenaResult


def _manifest(tmp_path: Path) -> ArenaManifest:
    path = tmp_path / "manifest.json"
    contestants = []
    rows = [
        ("B-coding-minimal", None),
        ("C-v011-reflective", "B-coding-minimal"),
        ("D-v012-evidence-first", "B-coding-minimal"),
        ("E-v012-lite", "B-coding-minimal"),
    ]
    for contestant_id, control_id in rows:
        row = {
            "id": contestant_id,
            "family": contestant_id,
            "role": "test",
            "enabled": True,
            "command": [
                "python",
                "fake.py",
                "{contestant}",
                "{split}",
                "{seed}",
                "{result}",
            ],
            "judge_command": ["python", "fake.py", "{contestant}", "blind", "{seed}"],
        }
        if control_id:
            row["control_id"] = control_id
        contestants.append(row)
    path.write_text(
        json.dumps(
            {
                "experiment_id": "first-tournament-test",
                "seeds": [1, 2, 3],
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


def test_first_tournament_successive_halving(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    director = FirstTournamentDirector(lab)

    stage = director.next_stage()
    assert stage is not None
    assert stage.name == "dev-screen"
    assert len(stage.runs) == 4
    assert {run.seed for run in stage.runs} == {1}

    lab.ledger.extend(
        [
            _result("B-coding-minimal", "dev", 1, 0.40),
            _result("C-v011-reflective", "dev", 1, 0.20),
            _result("D-v012-evidence-first", "dev", 1, 0.42),
            _result("E-v012-lite", "dev", 1, 0.39),
        ]
    )
    assert director.screen_survivors() == (
        "D-v012-evidence-first",
        "E-v012-lite",
    )
    stage = director.next_stage()
    assert stage is not None
    assert stage.name == "validation-screen"
    assert set(stage.contestant_ids) == {
        "B-coding-minimal",
        "D-v012-evidence-first",
        "E-v012-lite",
    }
    assert len(stage.runs) == 3

    lab.ledger.extend(
        [
            _result("B-coding-minimal", "validation", 1, 0.40),
            _result("D-v012-evidence-first", "validation", 1, 0.45),
            _result("E-v012-lite", "validation", 1, 0.39),
        ]
    )
    assert director.repeat_finalists() == (
        "D-v012-evidence-first",
        "E-v012-lite",
    )
    stage = director.next_stage()
    assert stage is not None
    assert stage.name == "validation-repeat"
    assert len(stage.runs) == 6
    assert {run.seed for run in stage.runs} == {2, 3}

    lab.ledger.extend(
        [
            _result("B-coding-minimal", "validation", 2, 0.40),
            _result("B-coding-minimal", "validation", 3, 0.40),
            _result("D-v012-evidence-first", "validation", 2, 0.46),
            _result("D-v012-evidence-first", "validation", 3, 0.44),
            _result("E-v012-lite", "validation", 2, 0.35),
            _result("E-v012-lite", "validation", 3, 0.36),
        ]
    )
    assert director.dev_repeat_targets() == ("E-v012-lite",)
    stage = director.next_stage()
    assert stage is not None
    assert stage.name == "dev-confirmation"
    assert set(stage.contestant_ids) == {"B-coding-minimal", "E-v012-lite"}
    assert len(stage.runs) == 4

    lab.ledger.extend(
        [
            _result("B-coding-minimal", "dev", 2, 0.40),
            _result("B-coding-minimal", "dev", 3, 0.40),
            _result("E-v012-lite", "dev", 2, 0.39),
            _result("E-v012-lite", "dev", 3, 0.39),
        ]
    )
    assert director.next_stage() is None
    assert "D-v012-evidence-first" in lab.promotion_queue()
    assert "C-v011-reflective" not in lab.promotion_queue()


def test_hard_failed_screen_candidate_receives_no_validation_compute(tmp_path: Path) -> None:
    lab = ArenaOrchestrator(_manifest(tmp_path), tmp_path / "arena")
    director = FirstTournamentDirector(lab)
    lab.ledger.extend(
        [
            _result("B-coding-minimal", "dev", 1, 0.40),
            ArenaResult(
                contestant_id="C-v011-reflective",
                split="dev",
                seed=1,
                status="failed",
                metrics={"solve_rate": 0.9, "failure_rate": 1.0},
            ),
            _result("D-v012-evidence-first", "dev", 1, 0.41),
            _result("E-v012-lite", "dev", 1, 0.39),
        ]
    )
    stage = director.next_stage()
    assert stage is not None
    assert stage.name == "validation-screen"
    assert "C-v011-reflective" not in stage.contestant_ids
