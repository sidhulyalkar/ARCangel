from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from arc3lab.arena.workers import ExperimentWorker, WorkerPool, WorkerSpec


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def initialized_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    (repo / "README.md").write_text("base\n")
    run_git(repo, "add", "README.md")
    run_git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    return repo


def test_worker_spec_rejects_shell_string() -> None:
    with pytest.raises(TypeError):
        WorkerSpec.from_dict(
            {
                "id": "unsafe",
                "command": "python agent.py",
                "qualification_commands": [],
            }
        )


def test_experiment_worker_commits_only_after_qualification(tmp_path: Path) -> None:
    repo = initialized_repo(tmp_path)
    spec = WorkerSpec(
        worker_id="test-worker",
        command=(
            sys.executable,
            "-c",
            "from pathlib import Path; Path('experiment.txt').write_text('candidate\\n')",
        ),
        qualification_commands=(
            (
                sys.executable,
                "-c",
                "from pathlib import Path; assert Path('experiment.txt').read_text() == 'candidate\\n'",
            ),
        ),
    )
    worker = ExperimentWorker(
        spec,
        repo_root=repo,
        worktree_root=tmp_path / "worktrees",
        receipt_root=tmp_path / "receipts",
    )
    proposal = {
        "proposal_id": "R1-01-test",
        "role_id": "scientist",
        "hypothesis": "a small qualified change should become an isolated commit",
        "suggested_branch": "experiment/r1-01-test",
    }
    receipt = worker.run(proposal, base_ref="HEAD")
    assert receipt.status == "qualified_commit"
    assert receipt.qualification_passed
    assert receipt.commit_sha
    assert "experiment.txt" in receipt.changed_files
    shown = run_git(repo, "show", "experiment/r1-01-test:experiment.txt")
    assert shown.stdout == "candidate\n"
    saved = json.loads(
        (tmp_path / "receipts" / "R1-01-test__test-worker.json").read_text()
    )
    assert saved["commit_sha"] == receipt.commit_sha


def test_failed_qualification_cleans_branch_for_safe_retry(tmp_path: Path) -> None:
    repo = initialized_repo(tmp_path)
    spec = WorkerSpec(
        worker_id="test-worker",
        command=(
            sys.executable,
            "-c",
            "from pathlib import Path; Path('bad.txt').write_text('bad')",
        ),
        qualification_commands=((sys.executable, "-c", "raise SystemExit(3)"),),
    )
    worker = ExperimentWorker(
        spec,
        repo_root=repo,
        worktree_root=tmp_path / "worktrees",
        receipt_root=tmp_path / "receipts",
    )
    receipt = worker.run(
        {
            "proposal_id": "R1-02-bad",
            "role_id": "red_team",
            "hypothesis": "this must fail qualification",
            "suggested_branch": "experiment/r1-02-bad",
        },
        base_ref="HEAD",
    )
    assert receipt.status == "qualification_failed"
    assert not receipt.qualification_passed
    # Failed branches are intentionally deleted after their receipt is written so the
    # exact proposal can be retried without a stale ref blocking worktree creation.
    assert run_git(repo, "branch", "--list", "experiment/r1-02-bad").stdout.strip() == ""
    assert not (tmp_path / "worktrees" / "R1-02-bad__test-worker").exists()
    saved = json.loads(
        (tmp_path / "receipts" / "R1-02-bad__test-worker.json").read_text()
    )
    assert saved["status"] == "qualification_failed"


def test_worker_pool_assigns_roles_without_executing(tmp_path: Path) -> None:
    config = tmp_path / "workers.json"
    config.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "id": "science-worker",
                        "command": ["agent", "{proposal}"],
                        "qualification_commands": [["python", "-m", "pytest", "-q"]],
                        "roles": ["scientist"],
                    }
                ]
            }
        )
    )
    pool = WorkerPool.load(
        config,
        repo_root=tmp_path,
        worktree_root=tmp_path / "worktrees",
        receipt_root=tmp_path / "receipts",
    )
    plan = pool.plan(
        {
            "selected": [
                {
                    "proposal_id": "one",
                    "role_id": "scientist",
                    "suggested_branch": "experiment/one",
                },
                {
                    "proposal_id": "two",
                    "role_id": "vision",
                    "suggested_branch": "experiment/two",
                },
            ]
        }
    )
    assert plan == [
        {
            "proposal_id": "one",
            "role_id": "scientist",
            "worker_id": "science-worker",
            "branch": "experiment/one",
        }
    ]
