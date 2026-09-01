#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pathlib
import threading
import time
from typing import Any


def _model_identity(path: str) -> str:
    p = pathlib.Path(path)
    pieces = [str(p)]
    cfg = p / "config.json"
    if cfg.exists():
        try:
            pieces.append(cfg.read_text(encoding="utf-8")[:12000])
        except Exception:
            pass
    return "".join(ch for ch in " ".join(pieces).lower() if ch.isalnum())


def _game_seed(seed: int, game_id: str) -> int:
    return seed ^ sum((index + 1) * ord(ch) for index, ch in enumerate(game_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="ARCangel V013 policy-neutral Kaggle runner")
    parser.add_argument(
        "--profile",
        required=True,
        choices=["coding-minimal", "v011", "v012", "v012-lite"],
    )
    parser.add_argument("--contestant-id", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-actions", type=int, default=900)
    parser.add_argument("--max-resets", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=200)
    parser.add_argument("--max-tool-calls", type=int, default=96)
    parser.add_argument("--time-budget-seconds", type=float, default=25200.0)
    parser.add_argument("--game-time-budget-seconds", type=float, default=7800.0)
    parser.add_argument("--coverage-reserve-fraction", type=float, default=0.05)
    parser.add_argument("--notebook-limit-seconds", type=float, default=32400.0)
    parser.add_argument("--setup-reserve-seconds", type=float, default=3600.0)
    parser.add_argument("--expected-scored-games", type=int, default=110)
    parser.add_argument("--output", default="/kaggle/working/arcangel_v013_receipt.json")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    os.environ["OPERATION_MODE"] = "competition"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")

    from arc3lab.arena.runtime_budget import audit_runtime_budget
    from arc3lab.evaluation.runner import run_suite
    from arc3lab.model import OpenAICompatLocalAdapter, discover_model_path, launch_vllm
    from arc3lab.policy.coding import CodingPolicy
    from arc3lab.policy.evidence_first import EvidenceFirstCodingPolicy
    from arc3lab.policy.lean_scientist import LeanReflectiveScientistPolicy

    print(f"ARCANGEL SUBMISSION BUILD: {args.build_id}", flush=True)
    print(f"ARCANGEL CONTESTANT: {args.contestant_id} profile={args.profile}", flush=True)

    envelope = audit_runtime_budget(
        total_games=args.expected_scored_games,
        workers=args.workers,
        notebook_limit_seconds=args.notebook_limit_seconds,
        setup_reserve_seconds=args.setup_reserve_seconds,
        global_budget_seconds=args.time_budget_seconds,
        requested_game_budget_seconds=args.game_time_budget_seconds,
        coverage_reserve_fraction=args.coverage_reserve_fraction,
    )
    print("COMPETITION RUNTIME ENVELOPE:", json.dumps(envelope.to_dict(), sort_keys=True), flush=True)

    model_path = args.model_path or discover_model_path()
    if not model_path:
        raise FileNotFoundError("No local Qwen 27B FP8 model was discovered under /kaggle/input")
    identity = _model_identity(model_path)
    required = ("qwen", "38", "27b", "fp8")
    missing = [token for token in required if token not in identity]
    if missing:
        raise RuntimeError(f"V013 candidate requires Qwen3.8 27B FP8; missing {missing}: {model_path}")
    print(f"MODEL INPUT PASS: {model_path}", flush=True)

    server = launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(1, args.workers),
        log_path="/kaggle/working/arcangel_v013_vllm.log",
        timeout=420.0,
    )
    print("VLLM SERVER PASS", flush=True)
    model = OpenAICompatLocalAdapter(
        model="arc3",
        max_tokens=768,
        timeout=180.0,
        temperature=0.0,
    )

    import numpy as np

    smoke = np.zeros((8, 8), dtype=np.int8)
    smoke[3, 3] = 7
    text = model.complete(
        "Return one JSON object only.",
        '{"mode":"PROBE","plan":[{"id":1}],"reason":"smoke"}',
        grid=smoke,
    )
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Qwen3.8 model smoke returned no text")
    print("FULL INFRASTRUCTURE PREFLIGHT PASS", flush=True)

    policies: dict[str, Any] = {}
    lock = threading.Lock()

    def factory(game_id: str):
        seed = _game_seed(args.seed, game_id)
        if args.profile == "coding-minimal":
            policy = CodingPolicy(
                model=model,
                seed=seed,
                reasoning_interval=1,
                max_model_calls=args.max_model_calls,
                max_tool_calls=args.max_tool_calls,
                predictive_history_depth=2,
            )
        elif args.profile == "v011":
            policy = LeanReflectiveScientistPolicy(
                model=model,
                seed=seed,
                max_model_calls=args.max_model_calls,
                max_tool_calls=args.max_tool_calls,
                reasoning_interval=3,
                bootstrap_reasoning_steps=5,
            )
        else:
            lite = args.profile == "v012-lite"
            policy = EvidenceFirstCodingPolicy(
                model=model,
                seed=seed,
                max_model_calls=args.max_model_calls,
                max_tool_calls=min(args.max_tool_calls, 48) if lite else args.max_tool_calls,
                max_reasoning_rounds=2 if lite else 4,
                max_plan_actions=8 if lite else 16,
                predictive_history_depth=2,
            )
        with lock:
            policies[game_id] = policy
        return policy

    stop = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not stop.wait(60.0):
            with lock:
                snapshot = list(policies.values())
            model_calls = sum(int(getattr(policy, "model_calls", 0)) for policy in snapshot)
            fallback = sum(int(getattr(policy, "fallback_actions", 0)) for policy in snapshot)
            authored = sum(
                int(getattr(policy, "model_authored_actions", 0)) for policy in snapshot
            )
            emergency = sum(
                int(getattr(policy, "emergency_transport_fallbacks", 0))
                for policy in snapshot
            )
            print(
                "ARCANGEL V013 HEARTBEAT"
                f" elapsed_s={int(time.monotonic() - started)}"
                f" games={len(snapshot)} model_calls={model_calls}"
                f" authored_actions={authored} fallback={fallback} emergency={emergency}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, name="arcangel-v013-heartbeat", daemon=True)
    thread.start()
    try:
        out = run_suite(
            policy_factory=factory,
            max_actions=args.max_actions,
            max_resets=args.max_resets,
            workers=args.workers,
            time_budget_seconds=args.time_budget_seconds,
            game_time_budget_seconds=args.game_time_budget_seconds,
            coverage_reserve_fraction=args.coverage_reserve_fraction,
            tags=["arcangel-v013", args.contestant_id, args.profile, "qwen38"],
            output_path=args.output,
        )
        out["build_id"] = args.build_id
        out["contestant_id"] = args.contestant_id
        out["profile"] = args.profile
        out["model_path"] = model_path
        out["model_family"] = "qwen3.8-27b-fp8"
        out["competition_runtime_envelope"] = envelope.to_dict()
        pathlib.Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        diagnostics = dict(out.get("diagnostics") or {})
        runtime_budget = dict(out.get("runtime_budget") or {})
        print("campaign runtime budget:", json.dumps(runtime_budget, sort_keys=True), flush=True)
        print("campaign diagnostics:", json.dumps(diagnostics, sort_keys=True), flush=True)
        if int(diagnostics.get("model_failures", 0)) != 0:
            raise RuntimeError("candidate had model transport failures; inspect receipt and vLLM log")
        if int(diagnostics.get("errors", 0)) != 0:
            raise RuntimeError("candidate had environment/runner errors; inspect receipt")
        skipped = int(diagnostics.get("skipped_deadline_games", 0))
        global_exhausted = int(diagnostics.get("global_deadline_exhausted_games", 0))
        if skipped or global_exhausted:
            raise RuntimeError(
                "coverage gate failed: shared deadline starved one or more games; "
                f"skipped={skipped} global_exhausted={global_exhausted}"
            )
        capped = int(diagnostics.get("game_budget_exhausted_games", 0))
        if capped:
            print(
                f"COVERAGE-SAFE GAME CAPS USED: {capped} games yielded to the suite budget",
                flush=True,
            )
        print(f"V013 CANDIDATE CAMPAIGN PASS: {args.build_id}", flush=True)
    finally:
        stop.set()
        thread.join(timeout=2.0)
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


if __name__ == "__main__":
    main()
