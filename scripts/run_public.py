#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from arc3lab.evaluation.runner import run_suite
from arc3lab.model import TransformersAdapter
from arc3lab.policy.hybrid import HybridPolicy
from arc3lab.policy.structural import StructuralPolicy
from arc3lab.policy.random_policy import RandomPolicy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-dir", required=True)
    p.add_argument("--policy", choices=["random", "structural", "hybrid"], default="structural")
    p.add_argument("--model-path", default=None)
    p.add_argument("--max-actions", type=int, default=600)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--games", default="")
    p.add_argument("--output", default="artifacts/public_run.json")
    p.add_argument("--seed", type=int, default=20260820)
    args = p.parse_args()
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = str(Path(args.env_dir).resolve())
    games = [x.strip() for x in args.games.split(",") if x.strip()] or None
    model = TransformersAdapter(args.model_path) if args.policy == "hybrid" and args.model_path else None

    def factory(game_id: str):
        seed = args.seed ^ sum(ord(c) for c in game_id)
        if args.policy == "random":
            return RandomPolicy(seed=seed)
        if args.policy == "hybrid":
            return HybridPolicy(model=model, seed=seed)
        return StructuralPolicy(seed=seed)

    out = run_suite(
        policy_factory=factory,
        games=games,
        max_actions=args.max_actions,
        workers=args.workers,
        tags=[args.policy, "v001"],
        output_path=args.output,
    )
    sc = out.get("scorecard") or {}
    print("score:", sc.get("score"))
    print("levels:", sc.get("total_levels_completed"), "/", sc.get("total_levels"))
    print("actions:", sc.get("total_actions"))
    print("output:", args.output)


if __name__ == "__main__":
    main()
