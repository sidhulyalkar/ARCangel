#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest
from arc3lab.arena.swarm_promotion import augment_manifest


def _server_ready(base_url: str) -> bool:
    try:
        return bool(requests.get(f"{base_url.rstrip('/')}/models", timeout=3.0).ok)
    except requests.RequestException:
        return False


def _model_identity(path: str) -> str:
    model_path = Path(path)
    pieces = [str(model_path)]
    config = model_path / "config.json"
    if config.exists():
        try:
            pieces.append(config.read_text(encoding="utf-8")[:12000])
        except Exception:
            pass
    return "".join(ch for ch in " ".join(pieces).lower() if ch.isalnum())


def _launch(args: argparse.Namespace) -> Any | None:
    if args.server_mode == "reuse":
        if not _server_ready(args.base_url):
            raise RuntimeError(f"no model server is ready at {args.base_url}")
        return None
    if _server_ready(args.base_url):
        raise RuntimeError("model endpoint already occupied; use --server-mode reuse explicitly")
    if args.base_url.rstrip("/") != "http://127.0.0.1:8000/v1":
        raise ValueError("managed Qwen launch requires localhost:8000")
    from arc3lab.model import discover_model_path, launch_vllm

    model_path = args.model_path or discover_model_path()
    if not model_path:
        raise FileNotFoundError("no local Qwen3.8 27B FP8 model was discovered")
    identity = _model_identity(model_path)
    missing = [token for token in ("qwen", "38", "27b", "fp8") if token not in identity]
    if missing:
        raise RuntimeError(f"BLIND judge requires Qwen3.8 27B FP8; missing {missing}: {model_path}")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
    return launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(8, int(args.server_max_sequences)),
        log_path=str(Path(args.arena_root) / "augmented-blind-vllm.log"),
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
    ap = argparse.ArgumentParser(description="Run private BLIND for static and swarm-promoted contestants")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--promotion-registry", default="artifacts/arena/v013/swarm-promotions.jsonl")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--private-registry", default="artifacts/arena/v013/splits.private.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--server-max-sequences", type=int, default=16)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    private_registry = Path(args.private_registry)
    if not private_registry.exists():
        raise FileNotFoundError(private_registry)
    manifest = ArenaManifest.load(args.manifest)
    if Path(args.promotion_registry).exists():
        manifest = augment_manifest(manifest, args.promotion_registry)
    lab = ArenaOrchestrator(manifest, args.arena_root)
    runs = lab.plan_blind(private_registry=private_registry)
    if args.plan_only:
        print(json.dumps({"runs": [run.run_key for run in runs], "count": len(runs)}, indent=2))
        return 0
    if not runs:
        print(json.dumps({"status": "NO_MISSING_BLIND_RUNS"}, indent=2))
        return 0

    os.environ["ARC3_MODEL_BASE_URL"] = args.base_url
    os.environ.setdefault("ARC3_MODEL_NAME", "arc3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    server = _launch(args)
    try:
        if not _server_ready(args.base_url):
            raise RuntimeError("BLIND judge model server failed readiness")
        for run in runs:
            print(f"AUGMENTED BLIND RUN {run.run_key}", flush=True)
            lab.execute(run)
        payload = {
            "status": "BLIND_COMPLETE",
            "executed": [run.run_key for run in runs],
            "kaggle_ready": lab.kaggle_ready_queue(),
            "authority": "private judge outcomes are retained locally and excluded from research context",
        }
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        _terminate(server)


if __name__ == "__main__":
    raise SystemExit(main())
