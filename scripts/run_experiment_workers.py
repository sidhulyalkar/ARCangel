from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.workers import WorkerPool


def main() -> int:
    ap = argparse.ArgumentParser(description="Run selected ARCangel proposals in isolated git worktrees")
    ap.add_argument("--workers", required=True)
    ap.add_argument("--battle-plan", default="artifacts/arena/v013/battle-plan.json")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--worktree-root", default="../arcangel-experiment-worktrees")
    ap.add_argument("--receipt-root", default="artifacts/arena/v013/worker-receipts")
    ap.add_argument("--base-ref", default="agent/v013-autonomous-research-swarm")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    battle = json.loads(Path(args.battle_plan).read_text())
    pool = WorkerPool.load(
        args.workers,
        repo_root=args.repo_root,
        worktree_root=args.worktree_root,
        receipt_root=args.receipt_root,
    )
    plan = pool.plan(battle)
    if args.plan_only:
        print(json.dumps({"planned": plan, "count": len(plan)}, indent=2))
        return 0

    receipts = pool.run(battle, base_ref=args.base_ref)
    payload = {
        "attempted": len(receipts),
        "qualified_commits": sum(receipt.status == "qualified_commit" for receipt in receipts),
        "receipts": [receipt.to_dict() for receipt in receipts],
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["qualified_commits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
