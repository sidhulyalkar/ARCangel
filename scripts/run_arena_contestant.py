from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from arc3lab.arena.metrics import suite_payload_to_result


def load_games(split: str, registry_path: str | None) -> list[str] | None:
    if not registry_path:
        return None
    registry = json.loads(Path(registry_path).read_text())
    games = registry.get(split)
    if games is None:
        raise ValueError(f"split {split!r} is unavailable in {registry_path}")
    return [str(game) for game in games]


def build_policy_factory(args: argparse.Namespace) -> Callable[[str], Any]:
    from arc3lab.model import OpenAICompatLocalAdapter
    from arc3lab.policy.coding import CodingPolicy
    from arc3lab.policy.evidence_first import EvidenceFirstCodingPolicy
    from arc3lab.policy.lean_scientist import LeanReflectiveScientistPolicy

    model = OpenAICompatLocalAdapter(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.model_timeout,
        temperature=0.0,
    )

    def common_seed(game_id: str) -> int:
        return int(args.seed) ^ sum((i + 1) * ord(ch) for i, ch in enumerate(game_id))

    if args.profile == "coding-minimal":

        def factory(game_id: str) -> CodingPolicy:
            return CodingPolicy(
                model=model,
                seed=common_seed(game_id),
                reasoning_interval=1,
                max_model_calls=args.max_model_calls,
                max_tool_calls=args.max_tool_calls,
                predictive_history_depth=2,
            )

        return factory

    if args.profile == "v011":

        def factory(game_id: str) -> LeanReflectiveScientistPolicy:
            return LeanReflectiveScientistPolicy(
                model=model,
                seed=common_seed(game_id),
                max_model_calls=args.max_model_calls,
                max_tool_calls=args.max_tool_calls,
                reasoning_interval=3,
                bootstrap_reasoning_steps=5,
            )

        return factory

    if args.profile in {"v012", "v012-lite"}:
        lite = args.profile == "v012-lite"

        def factory(game_id: str) -> EvidenceFirstCodingPolicy:
            return EvidenceFirstCodingPolicy(
                model=model,
                seed=common_seed(game_id),
                max_model_calls=args.max_model_calls,
                max_tool_calls=min(args.max_tool_calls, 48) if lite else args.max_tool_calls,
                max_reasoning_rounds=2 if lite else 4,
                max_plan_actions=8 if lite else 16,
                predictive_history_depth=2,
            )

        return factory

    raise ValueError(f"unsupported profile: {args.profile}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one ARCangel arena contestant")
    ap.add_argument(
        "--profile",
        required=True,
        choices=["coding-minimal", "v011", "v012", "v012-lite"],
    )
    ap.add_argument("--contestant", default=os.getenv("ARCANGEL_CONTESTANT_ID", ""))
    ap.add_argument("--split", default=os.getenv("ARCANGEL_SPLIT", "dev"))
    ap.add_argument("--seed", type=int, default=int(os.getenv("ARCANGEL_SEED", "20260831")))
    ap.add_argument("--result", default=os.getenv("ARCANGEL_RESULT_PATH", "arena_result.json"))
    ap.add_argument("--suite-output", default="")
    ap.add_argument("--split-registry", default=os.getenv("ARCANGEL_SPLIT_REGISTRY", ""))
    ap.add_argument("--environments-dir", default=os.getenv("ENVIRONMENTS_DIR", ""))
    ap.add_argument(
        "--base-url",
        default=os.getenv("ARC3_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    ap.add_argument("--model", default=os.getenv("ARC3_MODEL_NAME", "arc3"))
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--model-timeout", type=float, default=180.0)
    ap.add_argument("--max-model-calls", type=int, default=180)
    ap.add_argument("--max-tool-calls", type=int, default=96)
    ap.add_argument("--max-actions", type=int, default=900)
    ap.add_argument("--max-resets", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-budget-seconds", type=float, default=7200.0)
    ap.add_argument("--game-time-budget-seconds", type=float, default=1200.0)
    ap.add_argument("--coverage-reserve-fraction", type=float, default=0.05)
    args = ap.parse_args()

    if not args.contestant:
        raise ValueError("contestant id is required")
    if args.split == "blind" and not args.split_registry:
        raise ValueError("blind evaluation requires an explicit private split registry")

    from arc3lab.arena.offline_runtime import configure_offline_environment

    environment_dir = configure_offline_environment(
        args.environments_dir or None,
        recordings_dir=Path(args.result).parent / "recordings",
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print(f"ARCANGEL ARENA OFFLINE ENVIRONMENTS: {environment_dir}", flush=True)

    from arc3lab.evaluation.runner import run_suite

    suite_output = args.suite_output or str(Path(args.result).with_suffix(".suite.json"))
    payload = run_suite(
        policy_factory=build_policy_factory(args),
        games=load_games(args.split, args.split_registry),
        max_actions=args.max_actions,
        max_resets=args.max_resets,
        workers=args.workers,
        tags=["arcangel-arena", args.profile, args.contestant, args.split],
        output_path=suite_output,
        time_budget_seconds=args.time_budget_seconds,
        game_time_budget_seconds=args.game_time_budget_seconds,
        coverage_reserve_fraction=args.coverage_reserve_fraction,
    )
    result = suite_payload_to_result(
        payload,
        contestant_id=args.contestant,
        split=args.split,
        seed=args.seed,
        source=suite_output,
    )
    result.metadata["runtime_budget"] = payload.get("runtime_budget", {})
    result.metadata["operation_mode"] = "OFFLINE"
    result.metadata["environments_dir"] = str(environment_dir)
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
