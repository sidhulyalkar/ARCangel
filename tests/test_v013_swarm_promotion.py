from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from arc3lab.arena.experiment_guard import audit_experiment_scope
from arc3lab.arena.ledger import ResultLedger
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest, ArenaResult
from arc3lab.arena.swarm_promotion import (
    SwarmPromotionRegistry,
    augment_manifest,
    build_promotion,
    import_promotion_validation,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest(tmp_path: Path) -> ArenaManifest:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "promotion-test",
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
                "contestants": [
                    {
                        "id": "B-coding-minimal",
                        "family": "coding",
                        "role": "control",
                        "enabled": True,
                    },
                    {
                        "id": "D-v012-evidence-first",
                        "family": "v012",
                        "role": "scientist",
                        "control_id": "B-coding-minimal",
                        "enabled": True,
                    },
                    {
                        "id": "E-v012-lite",
                        "family": "v012-lite",
                        "role": "minimalist",
                        "control_id": "B-coding-minimal",
                        "enabled": True,
                    },
                ],
            }
        )
    )
    return ArenaManifest.load(path)


def _repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "src/arc3lab/policy").mkdir(parents=True)
    (root / "src/arc3lab/arena").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src/arc3lab/policy/a.py").write_text("VALUE = 1\n")
    (root / "src/arc3lab/arena/judge.py").write_text("VALUE = 1\n")
    (root / "scripts/run_arena_contestant.py").write_text("print('trusted')\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "base")
    base = _git(root, "rev-parse", "HEAD")
    candidate = tmp_path / "candidate"
    _git(root, "worktree", "add", "-b", "candidate", str(candidate), base)
    _git(candidate, "config", "user.name", "test")
    _git(candidate, "config", "user.email", "test@example.invalid")
    (candidate / "src/arc3lab/policy/a.py").write_text("VALUE = 2\n")
    _git(candidate, "add", "-A")
    _git(candidate, "commit", "-m", "candidate")
    head = _git(candidate, "rev-parse", "HEAD")
    return root, candidate, base, head


def _result(path: Path, contestant: str, seed: int, score: float) -> None:
    path.write_text(
        json.dumps(
            ArenaResult(
                contestant_id=contestant,
                split="validation",
                seed=seed,
                metrics={
                    "solve_rate": score,
                    "failure_rate": 0.0,
                    "emergency_fraction": 0.0,
                },
            ).to_dict()
        )
    )


def _fitness_receipt(
    tmp_path: Path,
    repo: Path,
    candidate: Path,
    base: str,
    head: str,
    *,
    split: str = "validation",
) -> Path:
    exp = tmp_path / "fitness"
    exp.mkdir()
    candidate_paths = []
    control_paths = []
    for seed, cscore, bscore in ((1, 0.55, 0.40), (2, 0.53, 0.40)):
        cpath = exp / f"candidate-{seed}.json"
        bpath = exp / f"control-{seed}.json"
        _result(cpath, "proposal", seed, cscore)
        _result(bpath, "control", seed, bscore)
        candidate_paths.append(str(cpath))
        control_paths.append(str(bpath))
    scope = audit_experiment_scope(
        repo,
        candidate,
        base_sha=base,
        candidate_sha=head,
    )
    (exp / "judge-boundary.json").write_text(
        json.dumps(
            {
                "candidate_head": head,
                "trusted_base_sha": base,
                "scope": scope.to_dict(),
            }
        )
    )
    receipt = exp / "fitness-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "proposal": {
                    "proposal_id": "G1-SWARM-01-scientist-model",
                    "generation": 1,
                    "provider_id": "model",
                    "role_id": "scientist",
                    "selection_split": split,
                    "target_profile": "v012",
                    "control_profile": "coding-minimal",
                },
                "candidate_git_sha": head,
                "control_git_sha": base,
                "candidate_results": candidate_paths,
                "control_results": control_paths,
                "fitness": {
                    "robust_delta": 0.12,
                    "mean_delta": 0.14,
                    "runs": 2,
                    "candidate_failure_rate": 0.0,
                    "candidate_emergency_fraction": 0.0,
                    "memory_status": "measured",
                },
                "memory_outcome": {"status": "measured"},
            }
        )
    )
    return receipt


def test_validation_winner_becomes_first_class_arena_contestant(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    repo, candidate, base, head = _repo(tmp_path)
    receipt = _fitness_receipt(tmp_path, repo, candidate, base, head)
    promotion = build_promotion(receipt, candidate, manifest, repo_root=repo)
    registry_path = tmp_path / "promotions.jsonl"
    SwarmPromotionRegistry(registry_path).append(promotion)
    augmented = augment_manifest(manifest, registry_path)
    ids = {row.contestant_id for row in augmented.contestants}
    assert promotion.contestant_id in ids
    assert promotion.control_id in ids

    arena_root = tmp_path / "arena"
    ledger = ResultLedger(arena_root / "ledger.jsonl")
    imported = import_promotion_validation(promotion, ledger)
    assert len(imported) == 4
    lab = ArenaOrchestrator(augmented, arena_root)
    assert promotion.contestant_id in lab.promotion_queue()


def test_dev_only_fitness_cannot_enter_population(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    repo, candidate, base, head = _repo(tmp_path)
    receipt = _fitness_receipt(tmp_path, repo, candidate, base, head, split="dev")
    with pytest.raises(ValueError, match="VALIDATION"):
        build_promotion(receipt, candidate, manifest, repo_root=repo)


def test_judge_tampering_cannot_be_promoted_even_with_good_score(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    repo, candidate, base, _ = _repo(tmp_path)
    (candidate / "src/arc3lab/arena/judge.py").write_text("VALUE = 999\n")
    _git(candidate, "add", "-A")
    _git(candidate, "commit", "-m", "tamper judge")
    head = _git(candidate, "rev-parse", "HEAD")
    receipt = _fitness_receipt(tmp_path, repo, candidate, base, head)
    with pytest.raises(ValueError, match="judge-owned"):
        build_promotion(receipt, candidate, manifest, repo_root=repo)
