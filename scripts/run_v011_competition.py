#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pathlib
import threading
import time

BUILD_ID = "S190A-V011-QWEN38-20260826"


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


def _aggregate_semantic(policies: dict[str, object]) -> dict:
    fields = [
        "model_parse_successes",
        "model_parse_contract_errors",
        "semantic_actions",
        "semantic_candidate_actions",
        "semantic_direct_actions",
        "model_directed_frontier_actions",
        "emergency_fallback_actions",
        "low_action_confidence_rejections",
        "goal_proposals",
        "typed_hypothesis_updates",
        "typed_hypothesis_contradictions",
        "reflection_updates",
        "reasoning_gate_skips",
    ]
    totals = {field: 0 for field in fields}
    per_game = {}
    for game_id, policy in sorted(policies.items()):
        fn = getattr(policy, "semantic_telemetry", None)
        tel = fn() if callable(fn) else {}
        row = {field: int(tel.get(field, 0)) for field in fields}
        row["model_calls"] = int(getattr(policy, "model_calls", 0))
        row["fallback_actions"] = int(getattr(policy, "fallback_actions", 0))
        row["reflection"] = tel.get("reflection", {})
        per_game[game_id] = row
        for field in fields:
            totals[field] += row[field]
    totals["games_instantiated"] = len(policies)
    totals["per_game"] = per_game
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="ARCangel V011 lean reflective contender")
    parser.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    parser.add_argument("--allow-qwen36-fallback", action="store_true")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--max-resets", type=int, default=2)
    parser.add_argument("--max-model-calls", type=int, default=160)
    parser.add_argument("--max-tool-calls", type=int, default=24)
    parser.add_argument("--time-budget-seconds", type=float, default=25200.0)
    parser.add_argument("--game-time-budget-seconds", type=float, default=7800.0)
    parser.add_argument("--output", default="/kaggle/working/arcangel_s190a_receipt.json")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    # Must be set before arc_agi is imported or Arcade is constructed.
    os.environ["OPERATION_MODE"] = "competition"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from arc3lab.evaluation.runner import run_suite
    from arc3lab.model import OpenAICompatLocalAdapter, discover_model_path, launch_vllm
    from arc3lab.policy.lean_scientist import LeanReflectiveScientistPolicy

    print(f"ARCANGEL SUBMISSION BUILD: {BUILD_ID}", flush=True)
    model_path = args.model_path or discover_model_path()
    if not model_path:
        raise FileNotFoundError("No local Qwen 27B FP8 model was discovered under /kaggle/input")
    identity = _model_identity(model_path)
    required = ("qwen", "27b", "fp8")
    missing = [token for token in required if token not in identity]
    if missing:
        raise RuntimeError(f"Mounted model failed identity check; missing {missing}: {model_path}")
    is_qwen38 = "qwen38" in identity
    is_qwen36 = "qwen36" in identity
    if not is_qwen38 and not (args.allow_qwen36_fallback and is_qwen36):
        raise RuntimeError(
            "S190A requires Qwen3.8 27B FP8. Pass --allow-qwen36-fallback only for the explicit S190B control."
        )
    print(f"MODEL INPUT PASS: {BUILD_ID}: {model_path}", flush=True)

    server = launch_vllm(
        model_path,
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 2, "video": 0},
        max_num_seqs=max(1, args.workers),
        log_path="/kaggle/working/arcangel_vllm.log",
        timeout=420.0,
    )
    print("VLLM SERVER PASS", flush=True)
    model = OpenAICompatLocalAdapter(
        model="arc3",
        max_tokens=320,
        timeout=180.0,
        temperature=0.0,
    )

    # Force a real two-image path before the scorecard is opened. This catches model/template
    # incompatibility while failure is still cheap and diagnosable.
    import numpy as np
    smoke = np.zeros((8, 8), dtype=np.int8)
    smoke[3, 3] = 7
    smoke_views = {
        "views": [
            {"label": "CURRENT HIGH-RESOLUTION BOARD", "grid": smoke, "side": 256},
            {"label": "TEMPORAL CONTEXT", "grid": smoke, "side": 256},
        ]
    }
    text = model.complete(
        "Return JSON only.",
        '{"probe":"multiview","answer":"ok"}',
        grid=smoke_views,
    )
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Two-image model smoke returned no text")
    print("FULL INFRASTRUCTURE PREFLIGHT PASS", flush=True)

    policies: dict[str, object] = {}
    policies_lock = threading.Lock()

    def factory(game_id: str):
        policy = LeanReflectiveScientistPolicy(
            model=model,
            seed=args.seed ^ sum(ord(c) for c in game_id),
            reasoning_interval=3,
            bootstrap_reasoning_steps=5,
            reflection_rule_limit=10,
            min_action_confidence=0.20,
            max_model_calls=args.max_model_calls,
            max_tool_calls=args.max_tool_calls,
            predictive_history_depth=2,
            current_view_side=512,
            temporal_view_side=384,
            candidate_limit=24,
            max_click_candidates=18,
        )
        with policies_lock:
            policies[game_id] = policy
        return policy

    stop_heartbeat = threading.Event()
    campaign_started = time.monotonic()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(60.0):
            with policies_lock:
                snapshot = list(policies.values())
            model_calls = sum(int(getattr(p, "model_calls", 0)) for p in snapshot)
            semantic = sum(int(getattr(p, "semantic_actions", 0)) for p in snapshot)
            fallback = sum(int(getattr(p, "emergency_fallback_actions", 0)) for p in snapshot)
            print(
                "ARCANGEL HEARTBEAT"
                f" elapsed_s={int(time.monotonic() - campaign_started)}"
                f" games={len(snapshot)} model_calls={model_calls}"
                f" semantic_actions={semantic} emergency_fallbacks={fallback}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, name="arcangel-heartbeat", daemon=True)
    thread.start()

    try:
        out = run_suite(
            policy_factory=factory,
            max_actions=args.max_actions,
            max_resets=args.max_resets,
            workers=args.workers,
            time_budget_seconds=args.time_budget_seconds,
            game_time_budget_seconds=args.game_time_budget_seconds,
            tags=["arcangel-v011", "lean-reflective", "qwen38" if is_qwen38 else "qwen36-control"],
            output_path=args.output,
        )
        with policies_lock:
            semantic = _aggregate_semantic(dict(policies))
        out["build_id"] = BUILD_ID
        out["model_path"] = model_path
        out["model_family"] = "qwen3.8-27b-fp8" if is_qwen38 else "qwen3.6-27b-fp8-control"
        out["semantic_diagnostics"] = semantic
        pathlib.Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

        diag = out.get("diagnostics", {})
        print("campaign diagnostics:", json.dumps(diag, sort_keys=True), flush=True)
        print("semantic diagnostics:", json.dumps({k: v for k, v in semantic.items() if k != "per_game"}, sort_keys=True), flush=True)
        if int(diag.get("model_failures", 0)) != 0:
            raise RuntimeError("V011 campaign had model transport failures; inspect receipt and vLLM log")
        print(f"V011 CAMPAIGN PASS: {BUILD_ID}", flush=True)
    finally:
        stop_heartbeat.set()
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
