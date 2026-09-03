#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from arc3lab.arena.campaign import CampaignDecision, CampaignDirector
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest
from arc3lab.arena.swarm_promotion import SwarmPromotionRegistry, augment_manifest


BASE_AUTO_STATES = {"NEED_SPLITS", "NEED_TOURNAMENT", "NEED_BLIND_JUDGE", "NEED_PACKAGE"}


def _load_manifest(args: argparse.Namespace) -> ArenaManifest:
    manifest = ArenaManifest.load(args.manifest)
    if Path(args.promotion_registry).exists():
        manifest = augment_manifest(manifest, args.promotion_registry)
    return manifest


def _director(args: argparse.Namespace) -> CampaignDirector:
    lab = ArenaOrchestrator(_load_manifest(args), args.arena_root)
    return CampaignDirector(
        lab,
        root=args.arena_root,
        public_registry=Path(args.arena_root) / "splits.public.json",
        private_registry=Path(args.arena_root) / "splits.private.json",
        package_dir=Path(args.arena_root) / "packages",
    )


def _run(command: list[str]) -> None:
    print("CAMPAIGN EXEC:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _tournament_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_first_tournament.py",
        "--manifest",
        args.manifest,
        "--root",
        args.arena_root,
        "--server-mode",
        args.server_mode,
        "--judge",
        "--private-registry",
        str(Path(args.arena_root) / "splits.private.json"),
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    return command


def _blind_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_augmented_blind.py",
        "--manifest",
        args.manifest,
        "--promotion-registry",
        args.promotion_registry,
        "--arena-root",
        args.arena_root,
        "--private-registry",
        str(Path(args.arena_root) / "splits.private.json"),
        "--server-mode",
        args.server_mode,
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    return command


def _package_command(args: argparse.Namespace, decision: CampaignDecision) -> list[str]:
    ready = list(decision.details.get("kaggle_ready") or [])
    if not ready:
        ready = _director(args).lab.kaggle_ready_queue()
    if not ready:
        raise RuntimeError("NEED_PACKAGE has no Kaggle-ready contestant")
    contestant_id = str(ready[0]["contestant_id"])
    if Path(args.promotion_registry).exists():
        registry = SwarmPromotionRegistry(args.promotion_registry)
        try:
            registry.by_contestant(contestant_id)
        except KeyError:
            pass
        else:
            return [
                sys.executable,
                "scripts/package_promoted_swarm.py",
                "--contestant",
                contestant_id,
                "--promotion-registry",
                args.promotion_registry,
                "--manifest",
                args.manifest,
                "--arena-root",
                args.arena_root,
                "--receipt-dir",
                str(Path(args.arena_root) / "packages"),
            ]
    return [
        sys.executable,
        "scripts/package_kaggle_ready.py",
        "--manifest",
        args.manifest,
        "--arena-root",
        args.arena_root,
        "--receipt-dir",
        str(Path(args.arena_root) / "packages"),
    ]


def _swarm_command(args: argparse.Namespace, decision: CampaignDecision) -> list[str]:
    if not args.auto_swarm:
        raise RuntimeError("NEED_SWARM_RESEARCH requires --auto-swarm for automatic advancement")
    if not args.workers_config:
        raise RuntimeError("--auto-swarm requires --workers-config")
    generation = int(decision.details["generation"])
    command = [
        sys.executable,
        "scripts/run_autonomous_swarm_generation.py",
        "--generation",
        str(generation),
        "--providers",
        args.providers,
        "--workers-config",
        args.workers_config,
        "--manifest",
        args.manifest,
        "--arena-root",
        args.arena_root,
        "--promotion-registry",
        args.promotion_registry,
        "--server-mode",
        args.server_mode,
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    return command


def _advance_once(args: argparse.Namespace, decision: CampaignDecision) -> None:
    state = decision.state
    if state == "NEED_SPLITS":
        salt = args.split_salt or os.getenv("ARCANGEL_SPLIT_SALT", "")
        if not salt:
            raise RuntimeError(
                "campaign needs ARCANGEL_SPLIT_SALT (or --split-salt) to create the private holdout"
            )
        _run(
            [
                sys.executable,
                "scripts/init_swarm_splits.py",
                "--root",
                args.arena_root,
                "--salt",
                salt,
            ]
        )
        return
    if state == "NEED_TOURNAMENT":
        _run(_tournament_command(args))
        return
    if state == "NEED_BLIND_JUDGE":
        _run(_blind_command(args))
        return
    if state == "NEED_PACKAGE":
        _run(_package_command(args, decision))
        return
    if state == "NEED_SWARM_RESEARCH":
        _run(_swarm_command(args, decision))
        return
    raise ValueError(f"state {state!r} is not automatically advanceable")


def _auto_state(args: argparse.Namespace, state: str) -> bool:
    return state in BASE_AUTO_STATES or (state == "NEED_SWARM_RESEARCH" and args.auto_swarm)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect or advance the complete V013 research campaign")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--promotion-registry", default="artifacts/arena/v013/swarm-promotions.jsonl")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--split-salt", default="")
    ap.add_argument("--providers", default="configs/research-providers.nvidia-swarm.json")
    ap.add_argument("--workers-config", default="")
    ap.add_argument("--auto-swarm", action="store_true")
    ap.add_argument("--advance", action="store_true")
    ap.add_argument("--max-auto-transitions", type=int, default=12)
    ap.add_argument(
        "--status-output",
        default="artifacts/arena/v013/campaign-status.json",
    )
    args = ap.parse_args()

    for transition in range(max(1, args.max_auto_transitions)):
        director = _director(args)
        decision = director.decide()
        payload = decision.to_dict()
        payload["transition"] = transition
        output = Path(args.status_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("V013 CAMPAIGN STATE:", json.dumps(payload, indent=2), flush=True)

        if not args.advance or not _auto_state(args, decision.state):
            return 0
        _advance_once(args, decision)

    final = _director(args).decide()
    print(
        "V013 CAMPAIGN STOPPED AT AUTO-TRANSITION LIMIT:",
        json.dumps(final.to_dict(), indent=2),
    )
    return 0 if not _auto_state(args, final.state) else 3


if __name__ == "__main__":
    raise SystemExit(main())
