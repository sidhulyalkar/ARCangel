#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from arc3lab.arena.experiment_guard import require_valid_experiment_scope, resolve_ref


def _remove(repo: Path, worktree: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize one portable swarm patch as a local candidate commit")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--receipt", default="")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    metadata_path = Path(args.metadata).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PORTABLE_PATCH_EXPORTED":
        raise ValueError("metadata is not a qualified portable patch receipt")
    base_sha = resolve_ref(repo, str(metadata["base_sha"]))
    patch_path = Path(str(metadata["patch"]))
    if not patch_path.is_absolute():
        patch_path = metadata_path.parent / patch_path.name
    patch = patch_path.read_bytes()
    digest = hashlib.sha256(patch).hexdigest()
    if digest != str(metadata.get("patch_sha256", "")):
        raise ValueError("portable patch SHA-256 mismatch")

    worktree = Path(args.worktree).resolve()
    if worktree.exists():
        _remove(repo, worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), base_sha],
        cwd=repo,
        check=True,
    )
    try:
        checked = subprocess.run(
            ["git", "apply", "--check", "--whitespace=error", str(patch_path)],
            cwd=worktree,
            check=False,
            capture_output=True,
            text=True,
        )
        if checked.returncode != 0:
            raise ValueError(f"portable patch no longer applies to frozen base: {checked.stderr[-3000:]}")
        subprocess.run(
            ["git", "apply", "--whitespace=error", str(patch_path)],
            cwd=worktree,
            check=True,
        )
        subprocess.run(["git", "add", "-A"], cwd=worktree, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=ARCangel Portable Swarm",
                "-c",
                "user.email=arcangel-portable@localhost",
                "commit",
                "-m",
                f"Materialize {metadata['proposal_id']}",
            ],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        )
        candidate_sha = resolve_ref(worktree, "HEAD")
        audit = require_valid_experiment_scope(
            repo,
            worktree,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
        )
        receipt = {
            "status": "PORTABLE_PATCH_MATERIALIZED",
            "proposal_id": metadata["proposal_id"],
            "source_candidate_sha": metadata.get("candidate_sha"),
            "base_sha": base_sha,
            "candidate_sha": candidate_sha,
            "patch_sha256": digest,
            "worktree": str(worktree),
            "scope": audit.to_dict(),
        }
        receipt_path = Path(args.receipt) if args.receipt else metadata_path.with_suffix(".materialized.json")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0
    except Exception:
        _remove(repo, worktree)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
