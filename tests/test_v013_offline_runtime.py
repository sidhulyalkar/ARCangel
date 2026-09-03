from __future__ import annotations

import os
from pathlib import Path

import pytest

from arc3lab.arena.offline_runtime import (
    configure_offline_environment,
    discover_environment_dir,
)


def _fake_environment_tree(root: Path) -> Path:
    environment_dir = root / "environment_files"
    version = environment_dir / "sp00" / "deadbeef"
    version.mkdir(parents=True)
    (version / "metadata.json").write_text("{}\n", encoding="utf-8")
    return environment_dir


def test_discover_environment_dir_accepts_explicit_local_tree(tmp_path: Path) -> None:
    environment_dir = _fake_environment_tree(tmp_path)
    resolved = discover_environment_dir(environment_dir)
    assert resolved == environment_dir.resolve()


def test_configure_offline_environment_overrides_competition_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_dir = _fake_environment_tree(tmp_path)
    recordings = tmp_path / "recordings"
    monkeypatch.setenv("OPERATION_MODE", "competition")
    monkeypatch.delenv("ENVIRONMENTS_DIR", raising=False)
    monkeypatch.delenv("RECORDINGS_DIR", raising=False)

    resolved = configure_offline_environment(environment_dir, recordings_dir=recordings)

    assert resolved == environment_dir.resolve()
    assert os.environ["OPERATION_MODE"] == "OFFLINE"
    assert os.environ["ENVIRONMENTS_DIR"] == str(environment_dir.resolve())
    assert os.environ["RECORDINGS_DIR"] == str(recordings.resolve())


def test_discover_environment_dir_rejects_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "environment_files"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        discover_environment_dir(empty)
