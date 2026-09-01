from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.schema import ArenaManifest, ArenaResult
from arc3lab.arena.swarm_fitness import evaluate_swarm_fitness
from arc3lab.arena.swarm_intelligence import SwarmMemory


def load_results(paths: list[str]) -> list[ArenaResult]:
    rows: list[ArenaResult] = []
    for raw in paths:
        path = Path(raw)
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            rows.extend(ArenaResult.from_dict(row) for row in payload)
        else:
            rows.append(ArenaResult.from_dict(payload))
    return rows


def battle_row(path: str, proposal_id: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    matches = [
        row
        for row in payload.get("selected", [])
        if str(row.get("proposal_id", "")) == proposal_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one battle row for {proposal_id!r}, found {len(matches)}")
    return dict(matches[0])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert paired candidate/control arena receipts into swarm fitness"
    )
    ap.add_argument("--battle-plan", required=True)
    ap.add_argument("--proposal-id", required=True)
    ap.add_argument("--candidate-result", action="append", required=True)
    ap.add_argument("--control-result", action="append", required=True)
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--memory", default="artifacts/arena/v013/swarm-memory.jsonl")
    ap.add_argument("--confidence-se", type=float, default=1.0)
    ap.add_argument("--min-measured-runs", type=int, default=2)
    ap.add_argument("--receipt", default="")
    args = ap.parse_args()

    manifest = ArenaManifest.load(args.manifest)
    row = battle_row(args.battle_plan, args.proposal_id)
    evidence = evaluate_swarm_fitness(
        row,
        load_results(args.candidate_result),
        load_results(args.control_result),
        manifest,
        confidence_se=args.confidence_se,
        min_measured_runs=args.min_measured_runs,
    )
    source = ";".join(args.candidate_result + args.control_result)
    outcome = evidence.to_outcome(source)
    SwarmMemory(args.memory).append(outcome)
    payload = {
        "fitness": evidence.to_dict(),
        "memory_outcome": outcome.to_dict(),
        "memory": args.memory,
        "authority": "development swarm guidance only; this does not satisfy promotion or BLIND gates",
    }
    if args.receipt:
        target = Path(args.receipt)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
