from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arc3lab.arena.experiment_guard import require_valid_experiment_scope


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "src/arc3lab/policy").mkdir(parents=True)
    (root / "src/arc3lab/arena").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/arc3lab/policy/a.py").write_text("VALUE = 1\n")
    (root / "src/arc3lab/arena/judge.py").write_text("VALUE = 1\n")
    (root / "tests/test_a.py").write_text("def test_a(): assert True\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "base")
    return root


def _candidate(root: Path, tmp_path: Path) -> tuple[Path, str]:
    base = _git(root, "rev-parse", "HEAD")
    candidate = tmp_path / "candidate"
    _git(root, "worktree", "add", "-b", "candidate", str(candidate), base)
    _git(candidate, "config", "user.name", "test")
    _git(candidate, "config", "user.email", "test@example.invalid")
    return candidate, base


def test_cognition_plus_tests_is_allowed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate, base = _candidate(root, tmp_path)
    (candidate / "src/arc3lab/policy/a.py").write_text("VALUE = 2\n")
    (candidate / "tests/test_a.py").write_text("def test_a(): assert 2 == 2\n")
    _git(candidate, "add", "-A")
    _git(candidate, "commit", "-m", "cognition")
    head = _git(candidate, "rev-parse", "HEAD")
    audit = require_valid_experiment_scope(
        root,
        candidate,
        base_sha=base,
        candidate_sha=head,
    )
    assert audit.valid
    assert audit.cognition_paths == ("src/arc3lab/policy/a.py",)
    assert audit.support_paths == ("tests/test_a.py",)
    assert not audit.forbidden_paths


def test_candidate_cannot_modify_judge_owned_arena(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate, base = _candidate(root, tmp_path)
    (candidate / "src/arc3lab/policy/a.py").write_text("VALUE = 2\n")
    (candidate / "src/arc3lab/arena/judge.py").write_text("VALUE = 999\n")
    _git(candidate, "add", "-A")
    _git(candidate, "commit", "-m", "tamper")
    head = _git(candidate, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="judge-owned"):
        require_valid_experiment_scope(
            root,
            candidate,
            base_sha=base,
            candidate_sha=head,
        )


def test_candidate_must_be_clean_and_change_cognition(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate, base = _candidate(root, tmp_path)
    (candidate / "tests/test_a.py").write_text("def test_a(): assert 3 == 3\n")
    _git(candidate, "add", "-A")
    _git(candidate, "commit", "-m", "tests only")
    head = _git(candidate, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="no cognition-owned"):
        require_valid_experiment_scope(
            root,
            candidate,
            base_sha=base,
            candidate_sha=head,
        )

    (candidate / "src/arc3lab/policy/a.py").write_text("VALUE = 4\n")
    with pytest.raises(ValueError, match="dirty"):
        require_valid_experiment_scope(
            root,
            candidate,
            base_sha=base,
            candidate_sha=head,
        )
