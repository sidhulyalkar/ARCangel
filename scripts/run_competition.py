#!/usr/bin/env python
from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["structural", "hybrid"], default="hybrid")
    p.add_argument("--model-path", default=os.getenv("ARC3_MODEL_PATH", ""))
    p.add_argument("--max-actions", type=int, default=int(os.getenv("ARC3_MAX_ACTIONS", "1200")))
    p.add_argument("--workers", type=int, default=int(os.getenv("ARC3_WORKERS", "6")))
    p.add_argument(
        "--time-budget-seconds",
        type=float,
        default=float(os.getenv("ARC3_TIME_BUDGET_SECONDS", "29400")),
        help="Shared play budget. 29,400s = 8h10m, leaving Kaggle runtime headroom.",
    )
    p.add_argument(
        "--max-model-calls",
        type=int,
        default=int(os.getenv("ARC3_MAX_MODEL_CALLS", "96")),
    )
    p.add_argument("--output", default="/kaggle/working/arc3_scorecard.json")
    p.add_argument("--seed", type=int, default=int(os.getenv("ARC3_SEED", "20260820")))
    p.add_argument("--launch-vllm", action="store_true")
    args = p.parse_args()

    # Must be set before Arcade is constructed. Kaggle also forces this server-side.
    os.environ["OPERATION_MODE"] = "competition"

    from arc3lab.evaluation.runner import run_suite
    from arc3lab.model import OpenAICompatLocalAdapter, discover_model_path, launch_vllm
    from arc3lab.policy.hybrid import HybridPolicy
    from arc3lab.policy.structural import StructuralPolicy

    model = None
    server = None
    if args.policy == "hybrid":
        model_path = args.model_path or discover_model_path()
        if args.launch_vllm:
            if not model_path:
                raise RuntimeError(
                    "No local model found. Attach a public model dataset or pass --model-path."
                )
            server = launch_vllm(model_path)
        model = OpenAICompatLocalAdapter(max_tokens=448, timeout=180)

    def factory(game_id: str):
        seed = args.seed ^ sum(ord(c) for c in game_id)
        if args.policy == "hybrid":
            return HybridPolicy(
                model=model,
                seed=seed,
                model_every=1,
                max_model_calls=args.max_model_calls,
            )
        return StructuralPolicy(seed=seed)

    try:
        out = run_suite(
            policy_factory=factory,
            max_actions=args.max_actions,
            max_resets=2,
            workers=args.workers,
            time_budget_seconds=args.time_budget_seconds,
            tags=[
                args.policy,
                "frontier-v002" if args.policy == "hybrid" else "frontier-v001",
            ],
            output_path=args.output,
        )
        sc = out.get("scorecard") or {}
        print("score:", sc.get("score"))
        print("levels:", sc.get("total_levels_completed"), "/", sc.get("total_levels"))
        print("diagnostics:", out.get("diagnostics"))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except Exception:
                server.kill()


if __name__ == "__main__":
    main()
