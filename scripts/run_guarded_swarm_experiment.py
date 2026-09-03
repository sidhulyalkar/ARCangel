#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from arc3lab.arena.experiment_guard import require_valid_experiment_scope, resolve_ref


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _remove_worktree(repo: Path, path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one swarm experiment with a frozen trusted judge/control snapshot"
    )
    ap.add_argument("--battle-plan", required=True)
    ap.add_argument("--proposal-id", required=True)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--split-registry", default="artifacts/arena/v013/splits.public.json")
    ap.add_argument("--output-root", default="artifacts/arena/v013/swarm-experiments")
    ap.add_argument("--memory", default="artifacts/arena/v013/swarm-memory.jsonl")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--max-seeds", type=int, default=2)
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
    candidate = Path(args.worktree).resolve()
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    current_head = resolve_ref(repo, "HEAD")
    candidate_head = resolve_ref(candidate, "HEAD")
    base_sha = _git(repo, "merge-base", current_head, candidate_head)
    scope = require_valid_experiment_scope(
        repo,
        candidate,
        base_sha=base_sha,
        candidate_sha=candidate_head,
    )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trusted = output_root / "_trusted_bases" / base_sha[:16]
    if trusted.exists():
        _remove_worktree(repo, trusted)
    trusted.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(trusted), base_sha],
        cwd=repo,
        check=True,
    )

    guard_receipt = output_root / args.proposal_id / "judge-boundary.json"
    guard_receipt.parent.mkdir(parents=True, exist_ok=True)
    guard_receipt.write_text(
        json.dumps(
            {
                "proposal_id": args.proposal_id,
                "current_research_head": current_head,
                "candidate_head": candidate_head,
                "trusted_base_sha": base_sha,
                "scope": scope.to_dict(),
                "authority": (
                    "candidate may mutate cognition-owned paths only; harness/control execute "
                    "from the frozen merge-base snapshot"
                ),
            },
            indent=2,
        )
        + "\n"
    )

    harness = trusted / "scripts" / "run_swarm_experiment.py"
    if not harness.exists():
        raise RuntimeError(
            "candidate predates the trusted swarm experiment harness; recreate it from a qualified V013 base"
        )
    manifest = trusted / args.manifest
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    command = [
        sys.executable,
        str(harness),
        "--battle-plan",
        str(Path(args.battle_plan).resolve()),
        "--proposal-id",
        args.proposal_id,
        "--worktree",
        str(candidate),
        "--repo-root",
        str(trusted),
        "--manifest",
        args.manifest,
        "--split-registry",
        str(Path(args.split_registry).resolve()),
        "--output-root",
        str(output_root),
        "--memory",
        str(Path(args.memory).resolve()),
        "--base-url",
        args.base_url,
        "--server-mode",
        args.server_mode,
        "--max-seeds",
        str(args.max_seeds),
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
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    if args.seeds:
        command.extend(["--seeds", args.seeds])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(trusted / "src")
    print("GUARDED SWARM EXEC:", " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=trusted, env=env, check=True)
    finally:
        _remove_worktree(repo, trusted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
