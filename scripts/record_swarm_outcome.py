from __future__ import annotations

import argparse
import json

from arc3lab.arena.swarm_intelligence import SwarmMemory, SwarmOutcome


def main() -> int:
    ap = argparse.ArgumentParser(description="Record one measured DEV/VALIDATION swarm experiment outcome")
    ap.add_argument("--memory", default="artifacts/arena/v013/swarm-memory.jsonl")
    ap.add_argument("--proposal-id", required=True)
    ap.add_argument("--provider-id", required=True)
    ap.add_argument("--role-id", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "validation"])
    ap.add_argument("--utility", required=True, type=float)
    ap.add_argument("--source", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    outcome = SwarmOutcome(
        proposal_id=args.proposal_id,
        provider_id=args.provider_id,
        role_id=args.role_id,
        split=args.split,
        utility=args.utility,
        source=args.source,
        note=args.note,
    )
    SwarmMemory(args.memory).append(outcome)
    print(json.dumps(outcome.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
