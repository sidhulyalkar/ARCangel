#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.ledger import ResultLedger
from arc3lab.arena.schema import ArenaManifest
from arc3lab.arena.swarm_promotion import (
    SwarmPromotionRegistry,
    build_promotion,
    import_promotion_validation,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Promote one guarded repeatable VALIDATION swarm winner into the arena population"
    )
    ap.add_argument("--fitness-receipt", required=True)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument(
        "--registry",
        default="artifacts/arena/v013/swarm-promotions.jsonl",
    )
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    manifest = ArenaManifest.load(args.manifest)
    promotion = build_promotion(
        args.fitness_receipt,
        args.worktree,
        manifest,
        repo_root=args.repo_root,
    )
    registry = SwarmPromotionRegistry(args.registry)
    existing = registry.read()
    if any(row.contestant_id == promotion.contestant_id for row in existing):
        raise ValueError(f"contestant {promotion.contestant_id} is already registered")
    if any(row.proposal_id == promotion.proposal_id for row in existing):
        raise ValueError(f"proposal {promotion.proposal_id} is already registered")

    ledger = ResultLedger(Path(args.arena_root) / "ledger.jsonl")
    imported = import_promotion_validation(promotion, ledger)
    registry.append(promotion)
    payload = {
        "status": "ARENA_PROMOTED",
        "promotion": promotion.to_dict(),
        "imported_validation_results": [row.to_dict() for row in imported],
        "next_gate": "private BLIND against the frozen shadow control",
        "authority": (
            "paired VALIDATION evidence is preserved; promotion creates identity, not new evidence"
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
