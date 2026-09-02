from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


COGNITION_ROOTS: tuple[str, ...] = (
    "src/arc3lab/policy/",
    "src/arc3lab/memory/",
    "src/arc3lab/perception/",
    "src/arc3lab/planning/",
)

SUPPORT_ROOTS: tuple[str, ...] = (
    "tests/",
    "docs/",
)


@dataclass(frozen=True, slots=True)
class ExperimentScopeAudit:
    base_sha: str
    candidate_sha: str
    changed_paths: tuple[str, ...]
    cognition_paths: tuple[str, ...]
    support_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    clean_worktree: bool

    @property
    def valid(self) -> bool:
        return (
            self.clean_worktree
            and bool(self.cognition_paths)
            and not self.forbidden_paths
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["valid"] = self.valid
        return payload


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def resolve_ref(root: str | Path, ref: str) -> str:
    return _git(Path(root).resolve(), "rev-parse", ref)


def worktree_is_clean(root: str | Path) -> bool:
    return not _git(Path(root).resolve(), "status", "--porcelain")


def changed_paths(root: str | Path, base_sha: str, candidate_sha: str) -> tuple[str, ...]:
    text = _git(
        Path(root).resolve(),
        "diff",
        "--name-only",
        "--diff-filter=ACDMRTUXB",
        f"{base_sha}..{candidate_sha}",
    )
    return tuple(sorted(line.strip() for line in text.splitlines() if line.strip()))


def _under(path: str, roots: Iterable[str]) -> bool:
    return any(path.startswith(root) for root in roots)


def audit_experiment_scope(
    repo_root: str | Path,
    candidate_root: str | Path,
    *,
    base_sha: str,
    candidate_sha: str,
    cognition_roots: Iterable[str] = COGNITION_ROOTS,
    support_roots: Iterable[str] = SUPPORT_ROOTS,
) -> ExperimentScopeAudit:
    repo = Path(repo_root).resolve()
    candidate = Path(candidate_root).resolve()
    base_resolved = resolve_ref(repo, base_sha)
    candidate_resolved = resolve_ref(candidate, candidate_sha)
    paths = changed_paths(repo, base_resolved, candidate_resolved)
    cognition_roots = tuple(cognition_roots)
    support_roots = tuple(support_roots)
    cognition = tuple(path for path in paths if _under(path, cognition_roots))
    support = tuple(path for path in paths if _under(path, support_roots))
    allowed = set(cognition) | set(support)
    forbidden = tuple(path for path in paths if path not in allowed)
    return ExperimentScopeAudit(
        base_sha=base_resolved,
        candidate_sha=candidate_resolved,
        changed_paths=paths,
        cognition_paths=cognition,
        support_paths=support,
        forbidden_paths=forbidden,
        clean_worktree=worktree_is_clean(candidate),
    )


def require_valid_experiment_scope(
    repo_root: str | Path,
    candidate_root: str | Path,
    *,
    base_sha: str,
    candidate_sha: str,
) -> ExperimentScopeAudit:
    audit = audit_experiment_scope(
        repo_root,
        candidate_root,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
    )
    if not audit.clean_worktree:
        raise ValueError("candidate worktree is dirty; experiment identity is not reproducible")
    if not audit.cognition_paths:
        raise ValueError("candidate changes no cognition-owned runtime path")
    if audit.forbidden_paths:
        joined = ", ".join(audit.forbidden_paths)
        raise ValueError(f"candidate modifies judge-owned or unsupported paths: {joined}")
    return audit
