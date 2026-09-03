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

from arc3lab.arena.schema import ArenaManifest, ArenaResult
from arc3lab.arena.swarm_fitness import evaluate_swarm_fitness
from arc3lab.arena.swarm_intelligence import SwarmMemory


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:100] or "experiment"


def _server_ready(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=3.0)
        return bool(response.ok)
    except requests.RequestException:
        return False


def _model_identity(path: str) -> str:
    p = Path(path)
    pieces = [str(p)]
    cfg = p / "config.json"
    if cfg.exists():
        try:
            pieces.append(cfg.read_text(encoding="utf-8")[:12000])
        except Exception:
            pass
    return "".join(ch for ch in " ".join(pieces).lower() if ch.isalnum())


def _launch(args: argparse.Namespace) -> tuple[Any | None, str]:
    if args.server_mode == "reuse":
        if not _server_ready(args.base_url):
            raise RuntimeError(f"no model server is ready at {args.base_url}")
        return None, "explicit-reuse"
    if args.base_url.rstrip("/") != "http://127.0.0.1:8000/v1":
        raise ValueError("automatic Qwen launch requires the default localhost:8000 endpoint")
    from arc3lab.model import discover_model_path, launch_vllm

    model_path = args.model_path or discover_model_path()
    if not model_path:
        raise FileNotFoundError("no local Qwen3.8 27B FP8 model was discovered")
    identity = _model_identity(model_path)
    missing = [token for token in ("qwen", "38", "27b", "fp8") if token not in identity]
    if missing:
        raise RuntimeError(f"swarm experiment requires Qwen3.8 27B FP8; missing {missing}: {model_path}")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
    server = launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(8, int(args.workers)),
        log_path=str(Path(args.output_root) / "swarm-experiment-vllm.log"),
        timeout=420.0,
    )
    if not _server_ready(args.base_url):
        raise RuntimeError("launched Qwen server did not become ready")
    return server, model_path


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


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _battle_row(path: Path, proposal_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = [
        dict(row)
        for row in payload.get("selected", [])
        if str(row.get("proposal_id", "")) == proposal_id
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one battle row for {proposal_id!r}; found {len(rows)}")
    row = rows[0]
    if str(row.get("selection_split", "")).lower() not in {"dev", "validation"}:
        raise ValueError("swarm experiment may run DEV/VALIDATION only")
    if not row.get("target_profile") or not row.get("control_profile"):
        raise ValueError("battle row lacks executable target/control profile")
    return row


def _run_contestant(
    *,
    code_root: Path,
    profile: str,
    contestant: str,
    split: str,
    seed: int,
    result_path: Path,
    registry: Path,
    args: argparse.Namespace,
) -> None:
    script = code_root / "scripts" / "run_arena_contestant.py"
    if not script.exists():
        raise FileNotFoundError(script)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(code_root / "src") + (os.pathsep + existing if existing else "")
    command = [
        sys.executable,
        str(script),
        "--profile",
        profile,
        "--contestant",
        contestant,
        "--split",
        split,
        "--seed",
        str(seed),
        "--result",
        str(result_path),
        "--split-registry",
        str(registry),
        "--base-url",
        args.base_url,
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
    result_path.parent.mkdir(parents=True, exist_ok=True)
    print("SWARM ARENA EXEC:", " ".join(command), flush=True)
    subprocess.run(command, cwd=code_root, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate one qualified swarm branch against its declared paired control"
    )
    ap.add_argument("--battle-plan", required=True)
    ap.add_argument("--proposal-id", required=True)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--split-registry", default="artifacts/arena/v013/splits.public.json")
    ap.add_argument("--output-root", default="artifacts/arena/v013/swarm-experiments")
    ap.add_argument("--memory", default="artifacts/arena/v013/swarm-memory.jsonl")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--max-seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-actions", type=int, default=900)
    ap.add_argument("--max-resets", type=int, default=2)
    ap.add_argument("--max-model-calls", type=int, default=180)
    ap.add_argument("--max-tool-calls", type=int, default=96)
    ap.add_argument("--time-budget-seconds", type=float, default=3600.0)
    ap.add_argument("--game-time-budget-seconds", type=float, default=900.0)
    ap.add_argument("--coverage-reserve-fraction", type=float, default=0.05)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    worktree = Path(args.worktree).resolve()
    registry = Path(args.split_registry).resolve()
    if not worktree.exists():
        raise FileNotFoundError(worktree)
    if not registry.exists():
        raise FileNotFoundError(registry)
    manifest = ArenaManifest.load(repo_root / args.manifest)
    row = _battle_row(Path(args.battle_plan), args.proposal_id)
    configured = list(manifest.seeds)
    if args.seeds:
        requested = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if any(seed not in configured for seed in requested):
            raise ValueError("requested seed is not part of the frozen V013 manifest")
        seeds = requested
    else:
        seeds = configured[: max(1, int(args.max_seeds))]
    if not seeds:
        raise ValueError("no experiment seeds selected")

    split = str(row["selection_split"]).lower()
    target_profile = str(row["target_profile"])
    control_profile = str(row["control_profile"])
    slug = _slug(args.proposal_id)
    output_root = Path(args.output_root).resolve() / slug
    candidate_dir = output_root / "candidate"
    control_dir = output_root / "control"
    base_sha = _git_head(repo_root)
    candidate_sha = _git_head(worktree)
    server, model_source = _launch(args)
    try:
        candidate_paths: list[Path] = []
        control_paths: list[Path] = []
        for index, seed in enumerate(seeds):
            candidate_path = candidate_dir / f"{split}__{seed}.json"
            control_path = control_dir / f"{split}__{seed}.json"
            pair = [
                (worktree, target_profile, args.proposal_id, candidate_path),
                (repo_root, control_profile, f"CONTROL-{control_profile}", control_path),
            ]
            # Alternate execution order across seeds to reduce temporal/server-order bias.
            if index % 2:
                pair.reverse()
            for code_root, profile, contestant, result_path in pair:
                _run_contestant(
                    code_root=code_root,
                    profile=profile,
                    contestant=contestant,
                    split=split,
                    seed=seed,
                    result_path=result_path,
                    registry=registry,
                    args=args,
                )
            candidate_paths.append(candidate_path)
            control_paths.append(control_path)

        candidate_results = [ArenaResult.from_dict(json.loads(path.read_text())) for path in candidate_paths]
        control_results = [ArenaResult.from_dict(json.loads(path.read_text())) for path in control_paths]
        evidence = evaluate_swarm_fitness(row, candidate_results, control_results, manifest)
        source = (
            f"candidate_sha={candidate_sha};base_sha={base_sha};model={model_source};"
            f"candidate_results={','.join(str(path) for path in candidate_paths)};"
            f"control_results={','.join(str(path) for path in control_paths)}"
        )
        outcome = evidence.to_outcome(source)
        SwarmMemory(args.memory).append(outcome)
        receipt = {
            "proposal": row,
            "candidate_git_sha": candidate_sha,
            "control_git_sha": base_sha,
            "model_source": model_source,
            "server_mode": args.server_mode,
            "seeds": seeds,
            "candidate_results": [str(path) for path in candidate_paths],
            "control_results": [str(path) for path in control_paths],
            "fitness": evidence.to_dict(),
            "memory_outcome": outcome.to_dict(),
            "authority": "paired development evidence only; BLIND and promotion remain separate gates",
        }
        receipt_path = output_root / "fitness-receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 0
    finally:
        _terminate(server)


if __name__ == "__main__":
    raise SystemExit(main())
