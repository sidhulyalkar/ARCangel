#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from arc3lab.arena.campaign import CampaignDirector
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest


AUTO_STATES = {"NEED_SPLITS", "NEED_TOURNAMENT", "NEED_BLIND_JUDGE", "NEED_PACKAGE"}


def _director(args: argparse.Namespace) -> CampaignDirector:
    manifest = ArenaManifest.load(args.manifest)
    lab = ArenaOrchestrator(manifest, args.arena_root)
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


def _advance_once(args: argparse.Namespace, state: str) -> None:
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
        command = [
            sys.executable,
            "scripts/run_first_tournament.py",
            "--manifest",
            args.manifest,
            "--root",
            args.arena_root,
            "--server-mode",
            args.server_mode,
        ]
        if args.model_path:
            command.extend(["--model-path", args.model_path])
        _run(command)
        return

    if state == "NEED_BLIND_JUDGE":
        command = [
            sys.executable,
            "scripts/run_first_tournament.py",
            "--manifest",
            args.manifest,
            "--root",
            args.arena_root,
            "--private-registry",
            str(Path(args.arena_root) / "splits.private.json"),
            "--server-mode",
            args.server_mode,
            "--judge",
        ]
        if args.model_path:
            command.extend(["--model-path", args.model_path])
        _run(command)
        return

    if state == "NEED_PACKAGE":
        _run(
            [
                sys.executable,
                "scripts/package_kaggle_ready.py",
                "--manifest",
                args.manifest,
                "--arena-root",
                args.arena_root,
                "--receipt-dir",
                str(Path(args.arena_root) / "packages"),
            ]
        )
        return

    raise ValueError(f"state {state!r} is not automatically advanceable")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect or advance the complete V013 research campaign")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--split-salt", default="")
    ap.add_argument("--advance", action="store_true")
    ap.add_argument("--max-auto-transitions", type=int, default=8)
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

        if not args.advance or decision.state not in AUTO_STATES:
            return 0
        _advance_once(args, decision.state)

    final = _director(args).decide()
    print(
        "V013 CAMPAIGN STOPPED AT AUTO-TRANSITION LIMIT:",
        json.dumps(final.to_dict(), indent=2),
    )
    return 0 if final.state not in AUTO_STATES else 3


if __name__ == "__main__":
    raise SystemExit(main())
