from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,180}$")


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    worker_id: str
    command: tuple[str, ...]
    qualification_commands: tuple[tuple[str, ...], ...]
    roles: tuple[str, ...] = ()
    timeout_seconds: float = 7200.0
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerSpec":
        command = data.get("command", ())
        quals = data.get("qualification_commands", ())
        if isinstance(command, str) or any(isinstance(row, str) for row in quals):
            raise TypeError("worker commands must be argv lists, never shell strings")
        return cls(
            worker_id=str(data["id"]),
            command=tuple(str(token) for token in command),
            qualification_commands=tuple(
                tuple(str(token) for token in row) for row in quals
            ),
            roles=tuple(str(role) for role in data.get("roles", ())),
            timeout_seconds=max(30.0, float(data.get("timeout_seconds", 7200.0))),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(slots=True)
class WorkerReceipt:
    proposal_id: str
    worker_id: str
    branch: str
    status: str
    commit_sha: str = ""
    changed_files: tuple[str, ...] = ()
    qualification_passed: bool = False
    elapsed_seconds: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "worker_id": self.worker_id,
            "branch": self.branch,
            "status": self.status,
            "commit_sha": self.commit_sha,
            "changed_files": list(self.changed_files),
            "qualification_passed": self.qualification_passed,
            "elapsed_seconds": self.elapsed_seconds,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
        }


class ExperimentWorker:
    """Run one coding-agent experiment inside an isolated git worktree.

    The worker never pushes or merges. A successful experiment becomes a local branch/commit
    that must still enter the arena and beat its declared control. Failed worktrees are removed
    after their receipt is written.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        repo_root: str | Path,
        worktree_root: str | Path,
        receipt_root: str | Path,
    ) -> None:
        self.spec = spec
        self.repo_root = Path(repo_root).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.receipt_root = Path(receipt_root).resolve()
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.receipt_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(
        argv: Iterable[str],
        *,
        cwd: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _render(argv: tuple[str, ...], mapping: dict[str, str]) -> list[str]:
        return [token.format(**mapping) for token in argv]

    def _write_receipt(self, receipt: WorkerReceipt) -> None:
        path = self.receipt_root / f"{receipt.proposal_id}__{receipt.worker_id}.json"
        path.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n")

    def run(self, proposal: dict[str, Any], *, base_ref: str) -> WorkerReceipt:
        started = time.monotonic()
        proposal_id = str(proposal["proposal_id"])
        branch = str(proposal.get("suggested_branch") or f"experiment/{proposal_id}")
        if not _BRANCH_RE.fullmatch(branch) or ".." in branch:
            raise ValueError(f"unsafe experiment branch: {branch}")
        worktree = self.worktree_root / re.sub(r"[^A-Za-z0-9._-]+", "_", proposal_id)
        proposal_path = self.receipt_root / f"{proposal_id}.proposal.json"
        proposal_path.write_text(json.dumps(proposal, indent=2) + "\n")
        mapping = {
            "proposal": str(proposal_path),
            "proposal_id": proposal_id,
            "branch": branch,
            "base_ref": base_ref,
            "repo": str(self.repo_root),
            "worktree": str(worktree),
        }
        receipt = WorkerReceipt(
            proposal_id=proposal_id,
            worker_id=self.spec.worker_id,
            branch=branch,
            status="started",
        )
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        created = False
        keep_worktree = False
        try:
            if worktree.exists():
                shutil.rmtree(worktree)
            add = self._run(
                ["git", "worktree", "add", "-b", branch, str(worktree), base_ref],
                cwd=self.repo_root,
                timeout=120.0,
            )
            stdout_parts.append(add.stdout)
            stderr_parts.append(add.stderr)
            if add.returncode != 0:
                raise RuntimeError(f"git worktree add failed: {add.stderr[-1000:]}")
            created = True

            worker = self._run(
                self._render(self.spec.command, mapping),
                cwd=worktree,
                timeout=self.spec.timeout_seconds,
            )
            stdout_parts.append(worker.stdout)
            stderr_parts.append(worker.stderr)
            if worker.returncode != 0:
                receipt.status = "worker_failed"
                receipt.error = f"worker return code {worker.returncode}"
                return receipt

            changed = self._run(
                ["git", "status", "--porcelain"],
                cwd=worktree,
                timeout=60.0,
            )
            files = []
            for line in changed.stdout.splitlines():
                if len(line) >= 4:
                    files.append(line[3:])
            receipt.changed_files = tuple(files)
            if not files:
                receipt.status = "no_change"
                return receipt

            for command in self.spec.qualification_commands:
                qualified = self._run(
                    self._render(command, mapping),
                    cwd=worktree,
                    timeout=self.spec.timeout_seconds,
                )
                stdout_parts.append(qualified.stdout)
                stderr_parts.append(qualified.stderr)
                if qualified.returncode != 0:
                    receipt.status = "qualification_failed"
                    receipt.error = f"qualification failed: {' '.join(command)}"
                    return receipt
            receipt.qualification_passed = True

            add_all = self._run(["git", "add", "-A"], cwd=worktree, timeout=60.0)
            if add_all.returncode != 0:
                raise RuntimeError(add_all.stderr[-1000:])
            commit = self._run(
                [
                    "git",
                    "-c",
                    "user.name=ARCangel Research Lab",
                    "-c",
                    "user.email=arcangel-lab@localhost",
                    "commit",
                    "-m",
                    f"Experiment {proposal_id}: {str(proposal.get('hypothesis', ''))[:100]}",
                ],
                cwd=worktree,
                timeout=120.0,
            )
            stdout_parts.append(commit.stdout)
            stderr_parts.append(commit.stderr)
            if commit.returncode != 0:
                raise RuntimeError(f"experiment commit failed: {commit.stderr[-1000:]}")
            head = self._run(["git", "rev-parse", "HEAD"], cwd=worktree, timeout=30.0)
            receipt.commit_sha = head.stdout.strip()
            receipt.status = "qualified_commit"
            keep_worktree = True
            return receipt
        except subprocess.TimeoutExpired as exc:
            receipt.status = "timeout"
            receipt.error = f"timeout after {exc.timeout}s"
            return receipt
        except Exception as exc:
            receipt.status = "error"
            receipt.error = f"{type(exc).__name__}: {exc}"
            return receipt
        finally:
            receipt.elapsed_seconds = round(time.monotonic() - started, 3)
            receipt.stdout_tail = "\n".join(stdout_parts)[-6000:]
            receipt.stderr_tail = "\n".join(stderr_parts)[-6000:]
            self._write_receipt(receipt)
            if created and not keep_worktree:
                self._run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=self.repo_root,
                    timeout=120.0,
                )


class WorkerPool:
    def __init__(
        self,
        workers: Iterable[WorkerSpec],
        *,
        repo_root: str | Path,
        worktree_root: str | Path,
        receipt_root: str | Path,
    ) -> None:
        self.workers = tuple(worker for worker in workers if worker.enabled)
        self.repo_root = Path(repo_root)
        self.worktree_root = Path(worktree_root)
        self.receipt_root = Path(receipt_root)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repo_root: str | Path,
        worktree_root: str | Path,
        receipt_root: str | Path,
    ) -> "WorkerPool":
        raw = json.loads(Path(path).read_text())
        workers = [WorkerSpec.from_dict(row) for row in raw.get("workers", [])]
        return cls(
            workers,
            repo_root=repo_root,
            worktree_root=worktree_root,
            receipt_root=receipt_root,
        )

    def worker_for(self, role_id: str) -> WorkerSpec | None:
        for worker in self.workers:
            if not worker.roles or role_id in worker.roles:
                return worker
        return None

    def plan(self, battle_plan: dict[str, Any]) -> list[dict[str, str]]:
        rows = []
        for proposal in battle_plan.get("selected", []):
            worker = self.worker_for(str(proposal.get("role_id", "")))
            if worker is None:
                continue
            rows.append(
                {
                    "proposal_id": str(proposal["proposal_id"]),
                    "role_id": str(proposal.get("role_id", "")),
                    "worker_id": worker.worker_id,
                    "branch": str(proposal.get("suggested_branch", "")),
                }
            )
        return rows

    def run(self, battle_plan: dict[str, Any], *, base_ref: str) -> list[WorkerReceipt]:
        receipts: list[WorkerReceipt] = []
        for proposal in battle_plan.get("selected", []):
            worker_spec = self.worker_for(str(proposal.get("role_id", "")))
            if worker_spec is None:
                continue
            worker = ExperimentWorker(
                worker_spec,
                repo_root=self.repo_root,
                worktree_root=self.worktree_root,
                receipt_root=self.receipt_root,
            )
            receipts.append(worker.run(dict(proposal), base_ref=base_ref))
        return receipts
