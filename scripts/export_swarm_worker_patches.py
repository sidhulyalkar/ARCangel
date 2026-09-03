#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from arc3lab.arena.experiment_guard import require_valid_experiment_scope, resolve_ref


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:120] or "proposal"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _qualified_receipt(receipt_root: Path, proposal_id: str) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for path in sorted(receipt_root.glob(f"{proposal_id}__*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if row.get("status") == "qualified_commit" and row.get("commit_sha"):
            row["receipt_path"] = str(path)
            rows.append(row)
    if len(rows) > 1:
        raise ValueError(f"multiple qualified worker receipts for {proposal_id}")
    return rows[0] if rows else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Export software-qualified swarm worktrees as portable patches")
    ap.add_argument("--battle-plan", required=True)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--worktree-root", default="../arcangel-experiment-worktrees")
    ap.add_argument("--receipt-root", default="artifacts/arena/v013/worker-receipts")
    ap.add_argument("--output-dir", default="artifacts/arena/v013/portable-patches")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    base_sha = resolve_ref(repo, args.base_sha)
    battle = json.loads(Path(args.battle_plan).read_text(encoding="utf-8"))
    worktree_root = Path(args.worktree_root).resolve()
    receipt_root = Path(args.receipt_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []

    for proposal in battle.get("selected", []):
        proposal_id = str(proposal.get("proposal_id", ""))
        if not proposal_id:
            continue
        receipt = _qualified_receipt(receipt_root, proposal_id)
        if receipt is None:
            continue
        worktree = worktree_root / _slug(proposal_id)
        if not worktree.exists():
            raise FileNotFoundError(f"qualified worktree disappeared: {worktree}")
        candidate_sha = resolve_ref(worktree, "HEAD")
        if candidate_sha != resolve_ref(worktree, str(receipt["commit_sha"])):
            raise ValueError(f"worker receipt SHA does not match worktree HEAD for {proposal_id}")
        audit = require_valid_experiment_scope(
            repo,
            worktree,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
        )
        patch = subprocess.run(
            ["git", "diff", "--binary", "--full-index", base_sha, candidate_sha],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        if not patch:
            raise ValueError(f"qualified candidate {proposal_id} produced an empty portable patch")
        patch_path = output / f"{_slug(proposal_id)}.patch"
        patch_path.write_bytes(patch)
        patch_sha = hashlib.sha256(patch).hexdigest()
        metadata = {
            "format": "arcangel-swarm-portable-patch-v1",
            "proposal_id": proposal_id,
            "generation": proposal.get("generation"),
            "provider_id": proposal.get("provider_id"),
            "role_id": proposal.get("role_id"),
            "target_profile": proposal.get("target_profile"),
            "control_profile": proposal.get("control_profile"),
            "base_sha": base_sha,
            "candidate_sha": candidate_sha,
            "patch": str(patch_path),
            "patch_sha256": patch_sha,
            "scope": audit.to_dict(),
            "worker_receipt": receipt.get("receipt_path"),
            "status": "PORTABLE_PATCH_EXPORTED",
        }
        metadata_path = output / f"{_slug(proposal_id)}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        exported.append(metadata)

    manifest = {
        "format": "arcangel-swarm-portable-patch-set-v1",
        "base_sha": base_sha,
        "battle_plan": str(Path(args.battle_plan)),
        "exported_count": len(exported),
        "candidates": exported,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if exported else 2


if __name__ == "__main__":
    raise SystemExit(main())
