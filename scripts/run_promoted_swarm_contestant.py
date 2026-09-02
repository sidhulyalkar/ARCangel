#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from arc3lab.arena.experiment_guard import require_valid_experiment_scope
from arc3lab.arena.swarm_promotion import SwarmPromotionRegistry


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


def _materialize(repo: Path, sha: str, path: Path) -> None:
    if path.exists():
        _remove(repo, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), sha],
        cwd=repo,
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute one pinned promoted swarm candidate/control")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--contestant", required=True, help="promoted candidate id used for registry lookup")
    ap.add_argument("--result-contestant", default="")
    ap.add_argument("--mode", choices=["candidate", "control"], required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--split-registry", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--base-url", default=os.getenv("ARC3_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.getenv("ARC3_MODEL_NAME", "arc3"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-actions", type=int, default=900)
    ap.add_argument("--max-resets", type=int, default=2)
    ap.add_argument("--max-model-calls", type=int, default=180)
    ap.add_argument("--max-tool-calls", type=int, default=96)
    ap.add_argument("--time-budget-seconds", type=float, default=3600.0)
    ap.add_argument("--game-time-budget-seconds", type=float, default=900.0)
    ap.add_argument("--coverage-reserve-fraction", type=float, default=0.05)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    promotion = SwarmPromotionRegistry(args.registry).by_contestant(args.contestant)
    result_contestant = args.result_contestant or promotion.contestant_id
    result_path = Path(args.result).resolve()
    source_root = result_path.parent / (
        f"_swarm_source_{promotion.contestant_id}_{args.mode}_{args.seed}"
    )
    sha = promotion.candidate_git_sha if args.mode == "candidate" else promotion.trusted_base_sha
    profile = promotion.target_profile if args.mode == "candidate" else promotion.control_profile

    _materialize(repo, sha, source_root)
    try:
        if args.mode == "candidate":
            require_valid_experiment_scope(
                repo,
                source_root,
                base_sha=promotion.trusted_base_sha,
                candidate_sha=promotion.candidate_git_sha,
            )
        runner = source_root / "scripts" / "run_arena_contestant.py"
        if not runner.exists():
            raise FileNotFoundError(runner)
        command = [
            sys.executable,
            str(runner),
            "--profile",
            profile,
            "--contestant",
            result_contestant,
            "--split",
            args.split,
            "--seed",
            str(args.seed),
            "--result",
            str(result_path),
            "--split-registry",
            str(Path(args.split_registry).resolve()),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--workers",
            str(args.workers),
            "--max-actions",
            str(args.max_actions),
            "--max-resets",
            str(args.max_resets),
            "--max-model-calls",
            str(args.max_model_calls),
            "--max-tool-calls",
            str(args.max_tool_calls),
            "--time-budget-seconds",
            str(args.time_budget_seconds),
            "--game-time-budget-seconds",
            str(args.game_time_budget_seconds),
            "--coverage-reserve-fraction",
            str(args.coverage_reserve_fraction),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(source_root / "src")
        print("PROMOTED SWARM EXEC:", " ".join(command), flush=True)
        subprocess.run(command, cwd=source_root, env=env, check=True)
    finally:
        _remove(repo, source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
