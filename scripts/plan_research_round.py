from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.evolution import ProposalTournament
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Select independent proposals and prepare round-2 exchange")
    ap.add_argument("--proposals", default="artifacts/arena/v013/proposals/round1")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--max-proposals", type=int, default=10)
    ap.add_argument("--battle-plan", default="artifacts/arena/v013/battle-plan.json")
    ap.add_argument("--exchange-brief", default="artifacts/arena/v013/round2-exchange.md")
    args = ap.parse_args()

    tournament = ProposalTournament.load(args.proposals)
    battle = tournament.battle_plan(max_proposals=args.max_proposals)
    battle_path = Path(args.battle_plan)
    battle_path.parent.mkdir(parents=True, exist_ok=True)
    battle_path.write_text(json.dumps(battle, indent=2) + "\n")

    manifest = ArenaManifest.load(args.manifest)
    lab = ArenaOrchestrator(manifest, args.arena_root)
    exchange = tournament.exchange_brief(lab.scorecard(include_blind=False))
    exchange_path = Path(args.exchange_brief)
    exchange_path.parent.mkdir(parents=True, exist_ok=True)
    exchange_path.write_text(exchange)

    print(
        json.dumps(
            {
                "battle_plan": str(battle_path),
                "selected": battle["selected_count"],
                "eligible": battle["eligible_count"],
                "exchange_brief": str(exchange_path),
            },
            indent=2,
        )
    )
    return 0 if battle["selected_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
