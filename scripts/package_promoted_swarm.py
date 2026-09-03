#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from arc3lab.arena.experiment_guard import require_valid_experiment_scope
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest
from arc3lab.arena.swarm_promotion import SwarmPromotionRegistry, augment_manifest


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:100] or "swarm"


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
    ap = argparse.ArgumentParser(description="Package one BLIND-qualified promoted swarm contestant")
    ap.add_argument("--contestant", required=True)
    ap.add_argument("--promotion-registry", default="artifacts/arena/v013/swarm-promotions.jsonl")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output-dir", default="kaggle/v013_candidates")
    ap.add_argument("--receipt-dir", default="artifacts/arena/v013/packages")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    registry = SwarmPromotionRegistry(args.promotion_registry)
    promotion = registry.by_contestant(args.contestant)
    manifest = augment_manifest(ArenaManifest.load(args.manifest), args.promotion_registry)
    lab = ArenaOrchestrator(manifest, args.arena_root)
    ready = {str(row["contestant_id"]): row for row in lab.kaggle_ready_queue()}
    if promotion.contestant_id not in ready:
        raise ValueError(f"{promotion.contestant_id} has not passed private BLIND qualification")

    source = Path(args.arena_root).resolve() / "_package_sources" / promotion.candidate_git_sha[:16]
    if source.exists():
        _remove(repo, source)
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(source), promotion.candidate_git_sha],
        cwd=repo,
        check=True,
    )
    try:
        scope = require_valid_experiment_scope(
            repo,
            source,
            base_sha=promotion.trusted_base_sha,
            candidate_sha=promotion.candidate_git_sha,
        )
        build_id = f"V013-SWARM-{_slug(promotion.contestant_id).upper()}-{promotion.candidate_git_sha[:12]}"
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        notebook = out_dir / f"ARCangel_{_slug(promotion.contestant_id)}_Qwen38.ipynb"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_v013_candidate_notebook.py",
                "--profile",
                promotion.target_profile,
                "--contestant-id",
                promotion.contestant_id,
                "--build-id",
                build_id,
                "--output",
                str(notebook),
            ],
            cwd=source,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/verify_v013_candidate_notebook.py",
                str(notebook),
                "--profile",
                promotion.target_profile,
                "--contestant-id",
                promotion.contestant_id,
                "--build-id",
                build_id,
            ],
            cwd=source,
            check=True,
        )
        notebook_sha = hashlib.sha256(notebook.read_bytes()).hexdigest()
        receipt = {
            "contestant_id": promotion.contestant_id,
            "family": f"swarm-{promotion.target_profile}",
            "profile": promotion.target_profile,
            "build_id": build_id,
            "git_head": promotion.candidate_git_sha,
            "trusted_base_sha": promotion.trusted_base_sha,
            "proposal_id": promotion.proposal_id,
            "generation": promotion.generation,
            "notebook": str(notebook),
            "notebook_sha256": notebook_sha,
            "blind_gate": ready[promotion.contestant_id],
            "scope": scope.to_dict(),
            "status": "PACKAGED_AND_VERIFIED",
        }
        receipt_dir = Path(args.receipt_dir).resolve()
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"{_slug(promotion.contestant_id)}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 0
    finally:
        _remove(repo, source)


if __name__ == "__main__":
    raise SystemExit(main())
