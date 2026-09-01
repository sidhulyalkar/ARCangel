#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

from arc3lab.arena.first_tournament import FirstTournamentDirector
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest


def _server_ready(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=3.0)
        return bool(response.ok)
    except requests.RequestException:
        return False


def _launch_server_if_needed(args: argparse.Namespace) -> Any | None:
    if _server_ready(args.base_url):
        print(f"REUSING MODEL SERVER: {args.base_url}", flush=True)
        return None
    if args.server_mode == "reuse":
        raise RuntimeError(f"No OpenAI-compatible model server is ready at {args.base_url}")
    if args.base_url.rstrip("/") != "http://127.0.0.1:8000/v1":
        raise ValueError("automatic launch currently requires the default localhost:8000 endpoint")

    from arc3lab.model import discover_model_path, launch_vllm

    model_path = args.model_path or discover_model_path()
    if not model_path:
        raise FileNotFoundError(
            "No local model was discovered. Supply --model-path or start the shared server first."
        )
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
    print(f"LAUNCHING SHARED QWEN SERVER: {model_path}", flush=True)
    return launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(8, int(args.server_max_sequences)),
        log_path=str(Path(args.root) / "first-tournament-vllm.log"),
        timeout=420.0,
    )


def _terminate(server: Any | None) -> None:
    if server is None:
        return
    server.terminate()
    try:
        server.wait(timeout=20)
    except Exception:
        server.kill()
    handle = getattr(server, "_arcangel_log_handle", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ARCangel's first adaptive B/C/D/E tournament")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--root", default="artifacts/arena/v013")
    ap.add_argument(
        "--public-registry",
        default="artifacts/arena/v013/splits.public.json",
    )
    ap.add_argument(
        "--private-registry",
        default="artifacts/arena/v013/splits.private.json",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--server-mode", choices=["auto", "reuse"], default="auto")
    ap.add_argument("--server-max-sequences", type=int, default=16)
    ap.add_argument("--screen-min-delta", type=float, default=-0.08)
    ap.add_argument("--validation-min-delta", type=float, default=-0.05)
    ap.add_argument("--max-repeat-challengers", type=int, default=2)
    ap.add_argument("--dev-repeat-margin", type=float, default=0.03)
    ap.add_argument("--max-stages", type=int, default=8)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    public_registry = Path(args.public_registry)
    if not public_registry.exists():
        raise FileNotFoundError(
            f"Public split registry is missing: {public_registry}. Run scripts/init_swarm_splits.py first."
        )
    if args.judge and not Path(args.private_registry).exists():
        raise FileNotFoundError("--judge requires the private split registry")

    manifest = ArenaManifest.load(args.manifest)
    lab = ArenaOrchestrator(manifest, args.root)
    director = FirstTournamentDirector(
        lab,
        screen_min_delta=args.screen_min_delta,
        validation_min_delta=args.validation_min_delta,
        max_repeat_challengers=args.max_repeat_challengers,
        dev_repeat_margin=args.dev_repeat_margin,
    )

    status_path = Path(args.root) / "first-tournament-status.json"
    scorecard_path = Path(args.root) / "first-tournament-scorecard.json"

    if args.plan_only:
        stage = director.next_stage()
        payload = {
            "status": director.status(),
            "next_stage": stage.to_dict() if stage else None,
        }
        print(json.dumps(payload, indent=2))
        return 0

    os.environ["ARC3_MODEL_BASE_URL"] = args.base_url
    os.environ.setdefault("ARC3_MODEL_NAME", "arc3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    server = _launch_server_if_needed(args)
    try:
        if not _server_ready(args.base_url):
            raise RuntimeError("Shared model server failed readiness after launch")

        for stage_index in range(max(1, args.max_stages)):
            stage = director.next_stage()
            if stage is None:
                print("RESEARCH TOURNAMENT REACHED A DECISION BOUNDARY", flush=True)
                break
            print(
                f"TOURNAMENT STAGE {stage_index + 1}: {stage.name} "
                f"split={stage.split} runs={len(stage.runs)}",
                flush=True,
            )
            print(stage.rationale, flush=True)
            for run in stage.runs:
                print(f"RUN {run.run_key}", flush=True)
                result = lab.execute(run)
                print(
                    json.dumps(
                        {
                            "run_key": run.run_key,
                            "status": result.status,
                            "metrics": result.metrics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            director.write_status(status_path)
            lab.write_scorecard(scorecard_path)
        else:
            raise RuntimeError("first tournament exceeded --max-stages without reaching a boundary")

        promoted = lab.promotion_queue()
        print("INTERNAL PROMOTION QUEUE:", json.dumps(promoted), flush=True)
        if args.judge and promoted:
            print("ENTERING PRIVATE BLIND JUDGE PHASE", flush=True)
            blind_runs = lab.plan_blind(private_registry=args.private_registry)
            for run in blind_runs:
                print(f"BLIND RUN {run.run_key}", flush=True)
                lab.execute(run)
            lab.write_scorecard(scorecard_path, include_blind=True)
            director.write_status(status_path)
            print("KAGGLE READY:", json.dumps(lab.kaggle_ready_queue(), indent=2), flush=True)

        print("FIRST TOURNAMENT STATUS:", json.dumps(director.status(), indent=2), flush=True)
        return 0
    finally:
        _terminate(server)


if __name__ == "__main__":
    raise SystemExit(main())
