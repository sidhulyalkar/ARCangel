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


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("AUTONOMOUS SWARM EXEC:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _server_ready(base_url: str) -> bool:
    try:
        return bool(requests.get(f"{base_url.rstrip('/')}/models", timeout=3.0).ok)
    except requests.RequestException:
        return False


def _identity(path: str) -> str:
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
        raise ValueError("managed launch requires localhost:8000")
    from arc3lab.model import discover_model_path, launch_vllm

    model_path = args.model_path or discover_model_path()
    if not model_path:
        raise FileNotFoundError("no local Qwen3.8 27B FP8 model was discovered")
    identity = _identity(model_path)
    missing = [token for token in ("qwen", "38", "27b", "fp8") if token not in identity]
    if missing:
        raise RuntimeError(f"swarm generation requires Qwen3.8 27B FP8; missing {missing}: {model_path}")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
    return launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(8, int(args.server_max_sequences)),
        log_path=str(Path(args.arena_root) / f"swarm-generation-{args.generation}-vllm.log"),
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


def _receipt_for(receipt_root: Path, proposal_id: str) -> dict[str, Any] | None:
    matches = sorted(receipt_root.glob(f"{proposal_id}__*.json"))
    qualified = []
    for path in matches:
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if row.get("status") == "qualified_commit" and row.get("commit_sha"):
            row["receipt_path"] = str(path)
            qualified.append(row)
    if len(qualified) > 1:
        raise ValueError(f"multiple qualified worker receipts for {proposal_id}")
    return qualified[0] if qualified else None


def _worktree_path(worktree_root: Path, proposal_id: str) -> Path:
    return worktree_root / re.sub(r"[^A-Za-z0-9._-]+", "_", proposal_id)


def _variant(row: dict[str, Any], split: str, path: Path) -> Path:
    payload = {
        "phase": f"autonomous-{split}-screen",
        "generation": row.get("generation"),
        "selected_count": 1,
        "eligible_count": 1,
        "selected": [dict(row, selection_split=split)],
        "authority": "autonomous director chooses DEV screen then held-out VALIDATION",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _fitness(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()).get("fitness", {})


def _guarded_command(
    args: argparse.Namespace,
    *,
    battle: Path,
    proposal_id: str,
    worktree: Path,
    max_seeds: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_guarded_swarm_experiment.py",
        "--battle-plan",
        str(battle),
        "--proposal-id",
        proposal_id,
        "--worktree",
        str(worktree),
        "--repo-root",
        args.repo_root,
        "--manifest",
        args.manifest,
        "--split-registry",
        str(Path(args.arena_root) / "splits.public.json"),
        "--output-root",
        str(Path(args.arena_root) / "swarm-experiments"),
        "--memory",
        str(Path(args.arena_root) / "swarm-memory.jsonl"),
        "--base-url",
        args.base_url,
        "--server-mode",
        "reuse",
        "--max-seeds",
        str(max_seeds),
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    return command


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one swarm generation from ideas to measured promotion")
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--providers", default="configs/research-providers.nvidia-swarm.json")
    ap.add_argument("--workers-config", required=True)
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--promotion-registry", default="artifacts/arena/v013/swarm-promotions.jsonl")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--worktree-root", default="../arcangel-experiment-worktrees")
    ap.add_argument("--receipt-root", default="artifacts/arena/v013/worker-receipts")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--server-max-sequences", type=int, default=16)
    ap.add_argument("--max-experiments", type=int, default=4)
    ap.add_argument("--dev-screen-seeds", type=int, default=1)
    ap.add_argument("--validation-seeds", type=int, default=2)
    ap.add_argument("--dev-screen-delta", type=float, default=-0.01)
    args = ap.parse_args()

    generation = max(1, int(args.generation))
    arena = Path(args.arena_root)
    battle = arena / f"swarm-battle-generation-{generation}.json"
    receipt_root = Path(args.receipt_root)
    worktree_root = Path(args.worktree_root).resolve()

    _run(
        [
            sys.executable,
            "scripts/run_swarm_research_cycle.py",
            "--providers",
            args.providers,
            "--manifest",
            args.manifest,
            "--arena-root",
            args.arena_root,
            "--repo-root",
            args.repo_root,
            "--generation",
            str(generation),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/run_experiment_workers.py",
            "--workers",
            args.workers_config,
            "--battle-plan",
            str(battle),
            "--repo-root",
            args.repo_root,
            "--worktree-root",
            str(worktree_root),
            "--receipt-root",
            str(receipt_root),
            "--base-ref",
            "agent/v013-autonomous-research-swarm",
        ]
    )

    payload = json.loads(battle.read_text())
    selected = list(payload.get("selected", []))[: max(0, int(args.max_experiments))]
    qualified = []
    for row in selected:
        proposal_id = str(row["proposal_id"])
        worker = _receipt_for(receipt_root, proposal_id)
        if worker is None:
            continue
        qualified.append((row, _worktree_path(worktree_root, proposal_id), worker))

    if not qualified:
        summary = {
            "generation": generation,
            "status": "NO_SOFTWARE_QUALIFIED_EXPERIMENTS",
            "selected": len(selected),
        }
        print(json.dumps(summary, indent=2))
        return 2

    server = _launch(args)
    promotions: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    try:
        if not _server_ready(args.base_url):
            raise RuntimeError("shared Qwen server failed readiness")
        for row, worktree, worker in qualified:
            proposal_id = str(row["proposal_id"])
            proposal_root = arena / "swarm-experiments" / re.sub(
                r"[^A-Za-z0-9]+", "_", proposal_id
            ).strip("_")[:100]
            dev_battle = _variant(
                row,
                "dev",
                arena / "swarm-variants" / f"{proposal_id}__dev.json",
            )
            _run(
                _guarded_command(
                    args,
                    battle=dev_battle,
                    proposal_id=proposal_id,
                    worktree=worktree,
                    max_seeds=max(1, int(args.dev_screen_seeds)),
                )
            )
            dev_receipt = proposal_root / "fitness-receipt.json"
            dev_fitness = _fitness(dev_receipt)
            row_summary: dict[str, Any] = {
                "proposal_id": proposal_id,
                "worker": worker,
                "dev_fitness": dev_fitness,
                "validation_run": False,
                "promoted": False,
            }
            healthy = (
                float(dev_fitness.get("candidate_failure_rate", 1.0)) <= 0.05
                and float(dev_fitness.get("candidate_emergency_fraction", 1.0)) <= 0.02
            )
            if float(dev_fitness.get("robust_delta", -1e9)) < args.dev_screen_delta or not healthy:
                experiments.append(row_summary)
                continue

            validation_battle = _variant(
                row,
                "validation",
                arena / "swarm-variants" / f"{proposal_id}__validation.json",
            )
            _run(
                _guarded_command(
                    args,
                    battle=validation_battle,
                    proposal_id=proposal_id,
                    worktree=worktree,
                    max_seeds=max(2, int(args.validation_seeds)),
                )
            )
            validation_receipt = proposal_root / "fitness-receipt.json"
            validation_fitness = _fitness(validation_receipt)
            row_summary["validation_run"] = True
            row_summary["validation_fitness"] = validation_fitness
            if (
                validation_fitness.get("memory_status") == "measured"
                and float(validation_fitness.get("robust_delta", -1e9)) >= 0.02
            ):
                promote = [
                    sys.executable,
                    "scripts/promote_swarm_experiment.py",
                    "--fitness-receipt",
                    str(validation_receipt),
                    "--worktree",
                    str(worktree),
                    "--manifest",
                    args.manifest,
                    "--arena-root",
                    args.arena_root,
                    "--registry",
                    args.promotion_registry,
                    "--repo-root",
                    args.repo_root,
                ]
                _run(promote)
                row_summary["promoted"] = True
                promotions.append(
                    {
                        "proposal_id": proposal_id,
                        "fitness_receipt": str(validation_receipt),
                        "worktree": str(worktree),
                    }
                )
            experiments.append(row_summary)
    finally:
        _terminate(server)

    summary = {
        "generation": generation,
        "status": "COMPLETE",
        "selected": len(selected),
        "software_qualified": len(qualified),
        "experiments": experiments,
        "promotions": promotions,
        "promotion_count": len(promotions),
        "next_gate": (
            "private BLIND for promoted population"
            if promotions
            else "next heterogeneous swarm generation with measured memory"
        ),
    }
    path = arena / f"autonomous-swarm-generation-{generation}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
