#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

from arc3lab.arena.schema import ArenaManifest


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:100] or "proposal"


def _run(command: list[str]) -> None:
    print("PORTABLE SWARM GPU EXEC:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _server_ready(base_url: str) -> bool:
    try:
        return bool(requests.get(f"{base_url.rstrip('/')}/models", timeout=3.0).ok)
    except requests.RequestException:
        return False


def _model_identity(path: str) -> str:
    model = Path(path)
    pieces = [str(model)]
    cfg = model / "config.json"
    if cfg.exists():
        try:
            pieces.append(cfg.read_text(encoding="utf-8")[:12000])
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
        raise RuntimeError(f"GPU swarm stage requires Qwen3.8 27B FP8; missing {missing}: {model_path}")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")
    return launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(8, int(args.server_max_sequences)),
        log_path=str(Path(args.arena_root) / "portable-swarm-gpu-vllm.log"),
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


def _remove_worktree(repo: Path, path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _variant(row: dict[str, Any], split: str, output: Path) -> Path:
    payload = {
        "phase": f"portable-{split}",
        "generation": row.get("generation"),
        "selected_count": 1,
        "eligible_count": 1,
        "selected": [dict(row, selection_split=split)],
        "authority": "portable candidate is screened on DEV then selected on held-out VALIDATION",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _guarded_command(
    args: argparse.Namespace,
    *,
    battle: Path,
    proposal_id: str,
    worktree: Path,
    max_seeds: int,
) -> list[str]:
    return [
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


def _fitness_receipt(arena: Path, proposal_id: str) -> Path:
    return arena / "swarm-experiments" / _slug(proposal_id) / "fitness-receipt.json"


def _fitness(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")).get("fitness") or {})


def _portable_metadata_path(manifest_path: Path, row: dict[str, Any]) -> Path:
    raw_patch = Path(str(row.get("patch", "")))
    name = raw_patch.with_suffix(".json").name
    candidate = manifest_path.parent / name
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def main() -> int:
    ap = argparse.ArgumentParser(description="Run GPU evidence stage from a portable remote swarm handoff")
    ap.add_argument("--patch-manifest", required=True)
    ap.add_argument("--battle-plan", required=True)
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--promotion-registry", default="artifacts/arena/v013/swarm-promotions.jsonl")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--worktree-root", default="../arcangel-portable-worktrees")
    ap.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--server-mode", choices=["launch", "reuse"], default="launch")
    ap.add_argument("--server-max-sequences", type=int, default=16)
    ap.add_argument("--dev-screen-seeds", type=int, default=1)
    ap.add_argument("--validation-seeds", type=int, default=2)
    ap.add_argument("--dev-screen-delta", type=float, default=None)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    arena = Path(args.arena_root)
    manifest = ArenaManifest.load(args.manifest)
    rules = manifest.promotion
    dev_delta = rules.min_dev_delta if args.dev_screen_delta is None else float(args.dev_screen_delta)
    validation_seeds = max(2, rules.min_validation_runs, int(args.validation_seeds))
    patch_manifest_path = Path(args.patch_manifest).resolve()
    patch_manifest = json.loads(patch_manifest_path.read_text(encoding="utf-8"))
    battle = json.loads(Path(args.battle_plan).read_text(encoding="utf-8"))
    battle_rows = {str(row["proposal_id"]): dict(row) for row in battle.get("selected", [])}
    worktree_root = Path(args.worktree_root).resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)

    materialized: list[tuple[dict[str, Any], Path]] = []
    for metadata in patch_manifest.get("candidates", []):
        proposal_id = str(metadata.get("proposal_id", ""))
        if proposal_id not in battle_rows:
            raise ValueError(f"portable patch {proposal_id} is absent from supplied battle plan")
        metadata_path = _portable_metadata_path(patch_manifest_path, dict(metadata))
        worktree = worktree_root / _slug(proposal_id)
        _run(
            [
                sys.executable,
                "scripts/materialize_swarm_patch.py",
                "--metadata",
                str(metadata_path),
                "--repo-root",
                str(repo),
                "--worktree",
                str(worktree),
            ]
        )
        materialized.append((battle_rows[proposal_id], worktree))

    if not materialized:
        print(json.dumps({"status": "NO_PORTABLE_CANDIDATES"}, indent=2))
        return 2

    server = _launch(args)
    experiments: list[dict[str, Any]] = []
    promoted_ids: set[str] = set()
    try:
        if not _server_ready(args.base_url):
            raise RuntimeError("Qwen server failed readiness")
        for row, worktree in materialized:
            proposal_id = str(row["proposal_id"])
            dev_variant = _variant(
                row,
                "dev",
                arena / "swarm-variants" / f"{proposal_id}__portable-dev.json",
            )
            _run(
                _guarded_command(
                    args,
                    battle=dev_variant,
                    proposal_id=proposal_id,
                    worktree=worktree,
                    max_seeds=max(1, int(args.dev_screen_seeds)),
                )
            )
            receipt = _fitness_receipt(arena, proposal_id)
            dev = _fitness(receipt)
            summary: dict[str, Any] = {
                "proposal_id": proposal_id,
                "dev_fitness": dev,
                "validation_run": False,
                "promoted": False,
            }
            healthy = (
                float(dev.get("candidate_failure_rate", 1.0)) <= rules.max_failure_rate
                and float(dev.get("candidate_emergency_fraction", 1.0))
                <= rules.max_emergency_fraction
            )
            if not healthy or float(dev.get("robust_delta", -1e9)) < dev_delta:
                experiments.append(summary)
                continue

            validation_variant = _variant(
                row,
                "validation",
                arena / "swarm-variants" / f"{proposal_id}__portable-validation.json",
            )
            _run(
                _guarded_command(
                    args,
                    battle=validation_variant,
                    proposal_id=proposal_id,
                    worktree=worktree,
                    max_seeds=validation_seeds,
                )
            )
            validation = _fitness(receipt)
            summary["validation_run"] = True
            summary["validation_fitness"] = validation
            if (
                validation.get("memory_status") == "measured"
                and float(validation.get("robust_delta", -1e9)) >= rules.min_validation_delta
            ):
                _run(
                    [
                        sys.executable,
                        "scripts/promote_swarm_experiment.py",
                        "--fitness-receipt",
                        str(receipt),
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
                )
                summary["promoted"] = True
                promoted_ids.add(proposal_id)
            experiments.append(summary)
    finally:
        _terminate(server)
        for row, worktree in materialized:
            if str(row["proposal_id"]) not in promoted_ids:
                _remove_worktree(repo, worktree)

    payload = {
        "status": "PORTABLE_GPU_STAGE_COMPLETE",
        "candidate_count": len(materialized),
        "experiments": experiments,
        "promoted_proposal_ids": sorted(promoted_ids),
        "next_gate": "private BLIND" if promoted_ids else "next swarm generation",
        "authority": "remote agents supplied patches; local paired Qwen evidence decided promotion",
    }
    output = arena / "portable-swarm-gpu-stage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
