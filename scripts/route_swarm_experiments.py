#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.experiment_routing import route_battle_plan


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Route selected swarm experiments into measurement and cognition-mutation lanes"
    )
    ap.add_argument("--battle-plan", required=True)
    ap.add_argument("--routed-output", required=True)
    ap.add_argument("--evaluation-output", required=True)
    ap.add_argument("--mutation-output", required=True)
    args = ap.parse_args()

    battle = json.loads(Path(args.battle_plan).read_text(encoding="utf-8"))
    routed, evaluation, mutation = route_battle_plan(battle)
    _write(Path(args.routed_output), routed)
    _write(Path(args.evaluation_output), evaluation)
    _write(Path(args.mutation_output), mutation)

    payload = {
        "selected_count": int(routed.get("selected_count", 0)),
        "measurement_count": int(routed.get("measurement_count", 0)),
        "mutation_count": int(routed.get("mutation_count", 0)),
        "deduplicated_measurement_sources": sum(
            len(row.get("source_proposal_ids", []))
            for row in evaluation.get("selected", [])
        ),
        "routed_output": args.routed_output,
        "evaluation_output": args.evaluation_output,
        "mutation_output": args.mutation_output,
        "authority": "routing allocates execution lane only; measured arena evidence decides fitness",
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["selected_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
