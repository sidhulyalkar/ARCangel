from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("SWARM CYCLE EXEC:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stamp_generation(path: Path, generation: int) -> dict[str, object]:
    payload = json.loads(path.read_text())
    payload["generation"] = generation
    for row in payload.get("selected", []):
        old_id = str(row.get("proposal_id", "")).strip()
        if not old_id:
            raise ValueError("selected swarm proposal is missing proposal_id")
        prefix = f"G{generation}-"
        proposal_id = old_id if old_id.startswith(prefix) else prefix + old_id
        row["proposal_id"] = proposal_id
        row["generation"] = generation
        branch_slug = re.sub(r"[^a-z0-9]+", "-", proposal_id.lower()).strip("-")
        row["suggested_branch"] = f"experiment/{branch_slug[:160]}"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one ARCangel hypothesis-space swarm generation end to end"
    )
    ap.add_argument(
        "--providers",
        default="configs/research-providers.nvidia-swarm.json",
    )
    ap.add_argument("--workers", default="")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--generation", type=int, default=1)
    ap.add_argument("--max-proposals", type=int, default=8)
    ap.add_argument("--max-research-requests", type=int, default=20)
    ap.add_argument("--max-review-requests", type=int, default=60)
    ap.add_argument("--reviews-per-proposal", type=int, default=3)
    ap.add_argument("--min-reviews", type=int, default=2)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--execute-workers", action="store_true")
    args = ap.parse_args()

    generation = max(1, args.generation)
    root = Path(args.arena_root)
    proposals = root / "proposals" / f"generation-{generation}"
    reviews = root / "reviews" / f"generation-{generation}"
    battle = root / f"swarm-battle-generation-{generation}.json"
    receipt = root / f"swarm-cycle-generation-{generation}.json"

    run(
        [
            sys.executable,
            "scripts/run_research_swarm.py",
            "--providers",
            args.providers,
            "--manifest",
            args.manifest,
            "--arena-root",
            args.arena_root,
            "--repo-root",
            args.repo_root,
            "--output-dir",
            str(proposals),
            "--memory",
            str(root / "swarm-memory.jsonl"),
            "--max-requests",
            str(args.max_research_requests),
            "--max-workers",
            str(args.max_workers),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/run_swarm_council.py",
            "--providers",
            args.providers,
            "--proposals",
            str(proposals),
            "--reviews",
            str(reviews),
            "--manifest",
            args.manifest,
            "--arena-root",
            args.arena_root,
            "--repo-root",
            args.repo_root,
            "--reviews-per-proposal",
            str(args.reviews_per_proposal),
            "--min-reviews",
            str(args.min_reviews),
            "--max-proposals",
            str(args.max_proposals),
            "--max-requests",
            str(args.max_review_requests),
            "--max-workers",
            str(args.max_workers),
            "--battle-plan",
            str(battle),
        ]
    )
    battle_payload = stamp_generation(battle, generation)

    worker_mode = "not_configured"
    if args.workers:
        command = [
            sys.executable,
            "scripts/run_experiment_workers.py",
            "--workers",
            args.workers,
            "--battle-plan",
            str(battle),
            "--repo-root",
            args.repo_root,
            "--base-ref",
            "agent/v013-autonomous-research-swarm",
        ]
        if not args.execute_workers:
            command.append("--plan-only")
            worker_mode = "planned"
        else:
            worker_mode = "executed"
        run(command)

    payload = {
        "generation": generation,
        "providers": args.providers,
        "proposal_dir": str(proposals),
        "review_dir": str(reviews),
        "battle_plan": str(battle),
        "selected": battle_payload.get("selected_count", 0),
        "worker_mode": worker_mode,
        "next_step": (
            "For each software-qualified selected worktree, run scripts/run_guarded_swarm_experiment.py "
            "against its declared paired control. The guarded runner writes repeat-aware DEV/VALIDATION "
            "fitness into swarm memory without exposing BLIND or Kaggle evidence."
        ),
        "authority": "swarm searches; trusted paired arena evidence promotes",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if int(payload["selected"]) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
