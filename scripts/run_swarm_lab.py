from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.metrics import suite_payload_to_result
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.research_packet import ResearchPacketBuilder
from arc3lab.arena.schema import ArenaManifest
from arc3lab.arena.splits import SplitRegistry


def load_lab(args: argparse.Namespace) -> tuple[ArenaManifest, ArenaOrchestrator]:
    manifest = ArenaManifest.load(args.manifest)
    return manifest, ArenaOrchestrator(manifest, args.root)


def validate_manifest(manifest: ArenaManifest) -> list[str]:
    ids = [contestant.contestant_id for contestant in manifest.contestants]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("contestant ids must be unique")
    known = set(ids)
    for contestant in manifest.contestants:
        if contestant.control_id and contestant.control_id not in known:
            errors.append(
                f"{contestant.contestant_id}: unknown control {contestant.control_id}"
            )
        if contestant.parent and contestant.parent not in known:
            errors.append(f"{contestant.contestant_id}: unknown parent {contestant.parent}")
    return errors


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = ArenaManifest.load(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print("ERROR", error)
        return 1
    print(
        json.dumps(
            {
                "experiment_id": manifest.experiment_id,
                "contestants": len(manifest.contestants),
                "enabled": sum(contestant.enabled for contestant in manifest.contestants),
                "seeds": list(manifest.seeds),
                "status": "VALID",
            },
            indent=2,
        )
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    _, lab = load_lab(args)
    splits = tuple(args.splits.split(","))
    runs = lab.plan(splits=splits, include_completed=args.include_completed)
    print(lab.describe_plan(runs))
    print(f"planned_runs={len(runs)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _, lab = load_lab(args)
    results = lab.run_all(splits=tuple(args.splits.split(",")))
    print(json.dumps([result.to_dict() for result in results], indent=2))
    return 0


def cmd_ingest_suite(args: argparse.Namespace) -> int:
    _, lab = load_lab(args)
    payload = json.loads(Path(args.suite).read_text())
    result = suite_payload_to_result(
        payload,
        contestant_id=args.contestant,
        split=args.split,
        seed=args.seed,
        source=str(args.suite),
    )
    lab.ledger.append(result)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_ingest_result(args: argparse.Namespace) -> int:
    _, lab = load_lab(args)
    result = lab.import_result(args.result)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    _, lab = load_lab(args)
    card = lab.scorecard(include_blind=args.include_blind)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    _, lab = load_lab(args)
    promoted = lab.promotion_queue()
    payload = {"experiment_id": lab.manifest.experiment_id, "promoted": promoted}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if promoted else 2


def cmd_split(args: argparse.Namespace) -> int:
    game_ids = [line.strip() for line in Path(args.games).read_text().splitlines() if line.strip()]
    registry = SplitRegistry.build(
        game_ids,
        salt=args.salt,
        dev_fraction=args.dev_fraction,
        validation_fraction=args.validation_fraction,
    )
    registry.write(args.public, args.private)
    print(json.dumps(registry.public_dict(), indent=2))
    return 0


def cmd_packet(args: argparse.Namespace) -> int:
    manifest, lab = load_lab(args)
    scorecard = lab.scorecard(include_blind=False)
    include_paths = [
        line.strip()
        for line in Path(args.include_list).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    digest = ResearchPacketBuilder(args.repo_root).build(
        args.output,
        experiment_id=manifest.experiment_id,
        scorecard=scorecard,
        include_paths=include_paths,
    )
    print(json.dumps({"packet": args.output, "sha256": digest}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ARCangel autonomous research swarm")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--root", default="artifacts/arena/v013")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("validate")

    plan = sub.add_parser("plan")
    plan.add_argument("--splits", default="dev,validation")
    plan.add_argument("--include-completed", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--splits", default="dev,validation")

    ingest_suite = sub.add_parser("ingest-suite")
    ingest_suite.add_argument("--suite", required=True)
    ingest_suite.add_argument("--contestant", required=True)
    ingest_suite.add_argument("--split", required=True, choices=["dev", "validation", "blind", "kaggle"])
    ingest_suite.add_argument("--seed", required=True, type=int)

    ingest_result = sub.add_parser("ingest-result")
    ingest_result.add_argument("--result", required=True)

    score = sub.add_parser("score")
    score.add_argument("--output", default="artifacts/arena/v013/scorecard.json")
    score.add_argument("--include-blind", action="store_true")

    promote = sub.add_parser("promote")
    promote.add_argument("--output", default="artifacts/arena/v013/promotion.json")

    split = sub.add_parser("split")
    split.add_argument("--games", required=True)
    split.add_argument("--salt", required=True)
    split.add_argument("--dev-fraction", type=float, default=0.60)
    split.add_argument("--validation-fraction", type=float, default=0.20)
    split.add_argument("--public", default="artifacts/arena/v013/splits.public.json")
    split.add_argument("--private", default="artifacts/arena/v013/splits.private.json")

    packet = sub.add_parser("packet")
    packet.add_argument("--repo-root", default=".")
    packet.add_argument("--include-list", default="configs/swarm-packet-files.txt")
    packet.add_argument("--output", default="artifacts/arena/v013/arcangel_research_packet.tar.gz")
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "ingest-suite":
        return cmd_ingest_suite(args)
    if args.command == "ingest-result":
        return cmd_ingest_result(args)
    if args.command == "score":
        return cmd_score(args)
    if args.command == "promote":
        return cmd_promote(args)
    if args.command == "split":
        return cmd_split(args)
    if args.command == "packet":
        return cmd_packet(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
