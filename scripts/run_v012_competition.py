#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pathlib
import threading
import time

BUILD_ID = "S210A-V012-EVIDENCE-FIRST-QWEN38-20260827"


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


def _aggregate(policies: dict[str, object]) -> dict:
    fields = [
        "analysis_rounds",
        "model_calls",
        "model_failures",
        "tool_calls",
        "tool_failures",
        "model_authored_actions",
        "model_authored_probes",
        "model_authored_plan_actions",
        "queued_actions_used",
        "expectation_checks",
        "expectation_mismatches",
        "hypothesis_tests",
        "hypothesis_test_failures",
        "world_model_validations",
        "world_model_validation_failures",
        "emergency_transport_fallbacks",
        "no_plan_rounds",
    ]
    totals = {field: 0 for field in fields}
    per_game = {}
    for game_id, policy in sorted(policies.items()):
        fn = getattr(policy, "evidence_telemetry", None)
        tel = fn() if callable(fn) else {}
        row = {field: int(tel.get(field, 0)) for field in fields}
        row["last_mode"] = tel.get("last_mode", "")
        row["workspace"] = tel.get("workspace", {})
        per_game[game_id] = row
        for field in fields:
            totals[field] += row[field]
    totals["games_instantiated"] = len(policies)
    totals["per_game"] = per_game
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="ARCangel V012 evidence-first coding agent")
    parser.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-actions", type=int, default=900)
    parser.add_argument("--max-resets", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=200)
    parser.add_argument("--max-tool-calls", type=int, default=96)
    parser.add_argument("--max-reasoning-rounds", type=int, default=4)
    parser.add_argument("--time-budget-seconds", type=float, default=25200.0)
    parser.add_argument("--game-time-budget-seconds", type=float, default=7800.0)
    parser.add_argument("--output", default="/kaggle/working/arcangel_s210a_receipt.json")
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    os.environ["OPERATION_MODE"] = "competition"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")

    from arc3lab.evaluation.runner import run_suite
    from arc3lab.model import OpenAICompatLocalAdapter, discover_model_path, launch_vllm
    from arc3lab.policy.evidence_first import EvidenceFirstCodingPolicy

    print(f"ARCANGEL SUBMISSION BUILD: {BUILD_ID}", flush=True)
    model_path = args.model_path or discover_model_path()
    if not model_path:
        raise FileNotFoundError("No local Qwen 27B FP8 model was discovered under /kaggle/input")
    identity = _model_identity(model_path)
    required = ("qwen", "38", "27b", "fp8")
    missing = [token for token in required if token not in identity]
    if missing:
        raise RuntimeError(f"S210A requires Qwen3.8 27B FP8; missing {missing}: {model_path}")
    print(f"MODEL INPUT PASS: {model_path}", flush=True)

    server = launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(1, args.workers),
        log_path="/kaggle/working/arcangel_v012_vllm.log",
        timeout=420.0,
    )
    print("VLLM SERVER PASS", flush=True)
    model = OpenAICompatLocalAdapter(
        model="arc3",
        max_tokens=768,
        timeout=180.0,
        temperature=0.0,
    )

    # Fail fast on the exact model/template/vision path before the scorecard is opened.
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

    policies: dict[str, object] = {}
    lock = threading.Lock()

    def factory(game_id: str):
        policy = EvidenceFirstCodingPolicy(
            model=model,
            seed=args.seed ^ sum(ord(c) for c in game_id),
            max_model_calls=args.max_model_calls,
            max_tool_calls=args.max_tool_calls,
            max_reasoning_rounds=args.max_reasoning_rounds,
            max_plan_actions=16,
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
            model_calls = sum(int(getattr(p, "model_calls", 0)) for p in snapshot)
            authored = sum(int(getattr(p, "model_authored_actions", 0)) for p in snapshot)
            emergency = sum(int(getattr(p, "emergency_transport_fallbacks", 0)) for p in snapshot)
            mismatches = sum(int(getattr(p, "expectation_mismatches", 0)) for p in snapshot)
            print(
                "ARCANGEL V012 HEARTBEAT"
                f" elapsed_s={int(time.monotonic() - started)}"
                f" games={len(snapshot)} model_calls={model_calls}"
                f" authored_actions={authored} emergency={emergency}"
                f" expectation_mismatches={mismatches}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, name="arcangel-v012-heartbeat", daemon=True)
    thread.start()

    try:
        out = run_suite(
            policy_factory=factory,
            max_actions=args.max_actions,
            max_resets=args.max_resets,
            workers=args.workers,
            time_budget_seconds=args.time_budget_seconds,
            game_time_budget_seconds=args.game_time_budget_seconds,
            tags=["arcangel-v012", "evidence-first", "qwen38"],
            output_path=args.output,
        )
        with lock:
            evidence = _aggregate(dict(policies))
        out["build_id"] = BUILD_ID
        out["model_path"] = model_path
        out["model_family"] = "qwen3.8-27b-fp8"
        out["evidence_diagnostics"] = evidence
        pathlib.Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

        diag = out.get("diagnostics", {})
        print("campaign diagnostics:", json.dumps(diag, sort_keys=True), flush=True)
        print(
            "evidence diagnostics:",
            json.dumps({k: v for k, v in evidence.items() if k != "per_game"}, sort_keys=True),
            flush=True,
        )
        if int(diag.get("model_failures", 0)) != 0:
            raise RuntimeError("V012 campaign had model transport failures; inspect receipt and vLLM log")
        if evidence["model_authored_actions"] <= evidence["emergency_transport_fallbacks"]:
            raise RuntimeError("V012 semantic authority gate failed: emergency path owns too many actions")
        print(f"V012 CAMPAIGN PASS: {BUILD_ID}", flush=True)
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
