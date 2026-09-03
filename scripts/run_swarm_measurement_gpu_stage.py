#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

from arc3lab.arena.schema import ArenaManifest


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:100] or "measurement"


def _server_ready(base_url: str) -> bool:
    try:
        return bool(requests.get(f"{base_url.rstrip('/')}/models", timeout=3.0).ok)
    except requests.RequestException:
        return False


def _model_identity(path: str) -> str:
    model = Path(path)
    pieces = [str(model)]
    config = model / "config.json"
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
        raise RuntimeError(
            f"measurement stage requires Qwen3.8 27B FP8; missing {missing}: {model_path}"
        )
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
    return launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(8, int(args.server_max_sequences)),
        log_path=str(Path(args.arena_root) / "swarm-measurement-vllm.log"),
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


def _variant(row: dict[str, Any], split: str, output: Path) -> Path:
    payload = {
        "phase": f"existing-profile-{split}",
        "generation": row.get("generation"),
        "selected_count": 1,
        "eligible_count": 1,
        "selected": [dict(row, selection_split=split)],
        "authority": "existing profiles are measured without a cognition rewrite",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _run_experiment(
    args: argparse.Namespace,
    *,
    variant: Path,
    proposal_id: str,
    max_seeds: int,
) -> None:
    repo = Path(args.repo_root).resolve()
    command = [
        sys.executable,
        "scripts/run_swarm_experiment.py",
        "--battle-plan",
        str(variant.resolve()),
        "--proposal-id",
        proposal_id,
        "--worktree",
        str(repo),
        "--repo-root",
        str(repo),
        "--manifest",
        args.manifest,
        "--split-registry",
        str(Path(args.split_registry).resolve()),
        "--output-root",
        str(Path(args.arena_root).resolve() / "swarm-measurements"),
        "--memory",
        str(Path(args.arena_root).resolve() / "swarm-memory.jsonl"),
        "--base-url",
        args.base_url,
        "--server-mode",
        "reuse",
        "--max-seeds",
        str(max_seeds),
        "--workers",
        str(args.workers),
        "--max-actions",
        str(args.max_actions),
        "--max-resets",
        str(args.max_resets),
        "--max-model-calls",
        str(args.max_model_calls),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--time-budget-seconds",
        str(args.time_budget_seconds),
        "--game-time-budget-seconds",
        str(args.game_time_budget_seconds),
        "--coverage-reserve-fraction",
        str(args.coverage_reserve_fraction),
    ]
    print("SWARM MEASUREMENT EXEC:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def _fitness(arena_root: str, proposal_id: str) -> dict[str, Any]:
    path = (
        Path(arena_root).resolve()
        / "swarm-measurements"
        / _slug(proposal_id)
        / "fitness-receipt.json"
    )
    return dict(json.loads(path.read_text(encoding="utf-8")).get("fitness") or {})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure council-selected comparisons that require no cognition patch"
    )
    ap.add_argument("--evaluation-plan", required=True)
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--split-registry", default="artifacts/arena/v013/splits.public.json")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--server-max-sequences", type=int, default=16)
    ap.add_argument("--dev-screen-seeds", type=int, default=1)
    ap.add_argument("--validation-seeds", type=int, default=2)
    ap.add_argument("--dev-screen-delta", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-actions", type=int, default=900)
    ap.add_argument("--max-resets", type=int, default=2)
    ap.add_argument("--max-model-calls", type=int, default=180)
    ap.add_argument("--max-tool-calls", type=int, default=96)
    ap.add_argument("--time-budget-seconds", type=float, default=3600.0)
    ap.add_argument("--game-time-budget-seconds", type=float, default=900.0)
    ap.add_argument("--coverage-reserve-fraction", type=float, default=0.05)
    args = ap.parse_args()

    plan = json.loads(Path(args.evaluation_plan).read_text(encoding="utf-8"))
    rows = [dict(row) for row in plan.get("selected", [])]
    if not rows:
        print(json.dumps({"status": "NO_MEASUREMENT_EXPERIMENTS"}, indent=2))
        return 0

    manifest = ArenaManifest.load(args.manifest)
    rules = manifest.promotion
    dev_delta = rules.min_dev_delta if args.dev_screen_delta is None else float(args.dev_screen_delta)
    validation_seeds = max(2, rules.min_validation_runs, int(args.validation_seeds))
    arena = Path(args.arena_root).resolve()
    variant_root = arena / "swarm-measurement-variants"

    server = _launch(args)
    summaries: list[dict[str, Any]] = []
    try:
        if not _server_ready(args.base_url):
            raise RuntimeError("Qwen server failed readiness")
        for row in rows:
            proposal_id = str(row["proposal_id"])
            dev_variant = _variant(row, "dev", variant_root / f"{proposal_id}__dev.json")
            _run_experiment(
                args,
                variant=dev_variant,
                proposal_id=proposal_id,
                max_seeds=max(1, int(args.dev_screen_seeds)),
            )
            dev = _fitness(args.arena_root, proposal_id)
            healthy = (
                float(dev.get("candidate_failure_rate", 1.0)) <= rules.max_failure_rate
                and float(dev.get("candidate_emergency_fraction", 1.0))
                <= rules.max_emergency_fraction
            )
            passes_dev = healthy and float(dev.get("robust_delta", -1e9)) >= dev_delta
            summary: dict[str, Any] = {
                "proposal_id": proposal_id,
                "source_proposal_ids": list(row.get("source_proposal_ids", [proposal_id])),
                "target_profile": row.get("target_profile"),
                "control_profile": row.get("control_profile"),
                "dev_fitness": dev,
                "dev_passed": passes_dev,
                "validation_run": False,
                "validation_winner": False,
            }
            if passes_dev:
                validation_variant = _variant(
                    row,
                    "validation",
                    variant_root / f"{proposal_id}__validation.json",
                )
                _run_experiment(
                    args,
                    variant=validation_variant,
                    proposal_id=proposal_id,
                    max_seeds=validation_seeds,
                )
                validation = _fitness(args.arena_root, proposal_id)
                summary["validation_run"] = True
                summary["validation_fitness"] = validation
                summary["validation_winner"] = (
                    validation.get("memory_status") == "measured"
                    and float(validation.get("robust_delta", -1e9))
                    >= rules.min_validation_delta
                )
            summaries.append(summary)
    finally:
        _terminate(server)

    payload = {
        "status": "SWARM_MEASUREMENT_GPU_STAGE_COMPLETE",
        "measurement_count": len(summaries),
        "measurements": summaries,
        "next_gate": (
            "architecture decision from paired evidence; mutation lane remains separate"
        ),
        "authority": (
            "existing profiles were measured without rewriting cognition; DEV screens allocate "
            "VALIDATION compute and repeated VALIDATION decides whether complexity earned its cost"
        ),
    }
    output = arena / "swarm-measurement-gpu-stage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
