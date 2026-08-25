from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from arc3lab.policy.structural import StructuralPolicy
from arc3lab.types import ActionSpec


def _state_name(frame: Any) -> str:
    s = getattr(frame, "state", "")
    return getattr(s, "name", str(s)).split(".")[-1]


def _error_result(game_id: str, exc: BaseException) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "state": "ERROR",
        "levels_completed": 0,
        "win_levels": 0,
        "actions": 0,
        "resets": 0,
        "level_events": [],
        "model_calls": 0,
        "model_failures": 0,
        "reasoning_cycles": 0,
        "tool_calls": 0,
        "tool_failures": 0,
        "queued_actions_used": 0,
        "fallback_actions": 0,
        "prediction_mismatches": 0,
        "spatial_plans_requested": 0,
        "spatial_plans_compiled": 0,
        "spatial_plan_actions": 0,
        "spatial_plan_mismatches": 0,
        "visual_packet_calls": 0,
        "multiview_calls": 0,
        "frontier_fallback_actions": 0,
        "frontier_known_states": 0,
        "visual_candidate_selections": 0,
        "visual_goal_updates": 0,
        "visual_affordance_observations": 0,
        "visual_expectation_mismatches": 0,
        "world_model_delegations": 0,
        "goal_hypotheses": 0,
        "deadline_exhausted": False,
        "error": f"{type(exc).__name__}: {exc}"[:1000],
    }


def run_game(
    arc: Any,
    scorecard_id: str,
    game_id: str,
    policy_factory: Callable[[str], StructuralPolicy],
    *,
    max_actions: int = 1200,
    max_resets: int = 0,
    stop_at_monotonic: float | None = None,
    game_time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one environment without mutating shared GameAction enum state.

    max_actions is a *safety ceiling*, not the primary budget. On Kaggle the shared
    wall-clock deadline should be the primary limiter because valid ARC3 levels can
    naturally require hundreds of actions.
    """
    from arcengine import GameAction

    env = arc.make(game_id, scorecard_id=scorecard_id)
    policy = policy_factory(game_id)
    frame = env.step(GameAction.RESET, data={}, reasoning={"reason": "start"})
    resets = 0
    actions = 0
    level_start_actions = 0
    events: list[dict[str, Any]] = []
    deadline_exhausted = False
    game_started_mono = time.monotonic()
    game_stop_at = (
        game_started_mono + game_time_budget_seconds
        if game_time_budget_seconds is not None
        else None
    )

    while actions < max_actions:
        now = time.monotonic()
        if stop_at_monotonic is not None and now >= stop_at_monotonic:
            deadline_exhausted = True
            break
        if game_stop_at is not None and now >= game_stop_at:
            deadline_exhausted = True
            break

        state = _state_name(frame)
        if state == "WIN":
            break
        if state == "GAME_OVER":
            if resets >= max_resets:
                break
            # Record the losing transition before retrying. In competition mode RESET
            # restarts the current level, so retain the game's learned mechanics/memory.
            try:
                policy.observe(frame)
            except Exception:
                pass
            frame = env.step(GameAction.RESET, data={}, reasoning={"reason": "recover"})
            resets += 1
            reset_hook = getattr(policy, "on_level_reset", None)
            if callable(reset_hook):
                reset_hook()
            continue

        scene = policy.observe(frame)
        before_level = int(getattr(frame, "levels_completed", 0))
        spec = policy.choose(scene)
        if spec.action_id not in scene.available_actions:
            # Never burn a scored action on an illegal model output.
            legal = next((a for a in scene.available_actions if a != 0), 0)
            spec = ActionSpec(legal, reason="illegal-output guard", confidence=0.0)

        action = GameAction.from_id(spec.action_id)
        # GameAction enum members carry mutable action_data. Calling set_data() can race
        # across concurrent games for ACTION6. Pass thread-local data directly instead.
        frame = env.step(
            action,
            data=spec.data,
            reasoning={"reason": spec.reason, "confidence": spec.confidence},
        )
        actions += 1
        after_level = int(getattr(frame, "levels_completed", 0))
        if after_level > before_level:
            events.append(
                {"level": after_level, "actions_for_level": actions - level_start_actions}
            )
            level_start_actions = actions

    # Observe terminal/latest frame once so the final transition is not lost.
    try:
        policy.observe(frame)
    except Exception:
        pass

    return {
        "game_id": game_id,
        "state": _state_name(frame),
        "levels_completed": int(getattr(frame, "levels_completed", 0)),
        "win_levels": int(getattr(frame, "win_levels", 0) or 0),
        "actions": actions,
        "resets": resets,
        "level_events": events,
        "model_calls": int(getattr(policy, "model_calls", 0)),
        "model_failures": int(getattr(policy, "model_failures", 0)),
        "reasoning_cycles": int(getattr(policy, "reasoning_cycles", 0)),
        "tool_calls": int(getattr(policy, "tool_calls", 0)),
        "tool_failures": int(getattr(policy, "tool_failures", 0)),
        "queued_actions_used": int(getattr(policy, "queued_actions_used", 0)),
        "fallback_actions": int(getattr(policy, "fallback_actions", 0)),
        "prediction_mismatches": int(getattr(policy, "prediction_mismatch_count", 0)),
        "spatial_plans_requested": int(getattr(policy, "spatial_plans_requested", 0)),
        "spatial_plans_compiled": int(getattr(policy, "spatial_plans_compiled", 0)),
        "spatial_plan_actions": int(getattr(policy, "spatial_plan_actions", 0)),
        "spatial_plan_mismatches": int(getattr(policy, "spatial_plan_mismatches", 0)),
        "visual_packet_calls": int(getattr(policy, "visual_packet_calls", 0)),
        "multiview_calls": int(getattr(policy, "multiview_calls", 0)),
        "frontier_fallback_actions": int(getattr(policy, "frontier_fallback_actions", 0)),
        "frontier_known_states": len(getattr(getattr(policy, "exploration_frontier", None), "nodes", {})),
        "visual_candidate_selections": int(getattr(policy, "visual_candidate_selections", 0)),
        "visual_goal_updates": int(getattr(policy, "visual_goal_updates", 0)),
        "visual_affordance_observations": int(getattr(policy, "visual_affordance_observations", 0)),
        "visual_expectation_mismatches": int(getattr(policy, "visual_expectation_mismatches", 0)),
        "final_orientation_entropy": float(
            getattr(policy, "last_perceptual_state", {}).get("orientation_entropy", 1.0)
        ) if isinstance(getattr(policy, "last_perceptual_state", {}), dict) else 1.0,
        "final_perceptual_mode": str(
            getattr(policy, "last_perceptual_state", {}).get("recommended_mode", "")
        ) if isinstance(getattr(policy, "last_perceptual_state", {}), dict) else "",
        "world_model_delegations": int(getattr(policy, "world_model_delegations", 0)),
        "goal_hypotheses": len(getattr(policy, "goals", [])),
        "predictive_summary": (
            getattr(policy, "predictive").summary()
            if getattr(policy, "predictive", None) is not None else None
        ),
        "belief_count": len(getattr(policy, "beliefs", [])),
        "elapsed_seconds": round(time.monotonic() - game_started_mono, 3),
        "deadline_exhausted": deadline_exhausted,
        "error": None,
    }


def run_suite(
    *,
    policy_factory: Callable[[str], StructuralPolicy],
    games: list[str] | None = None,
    max_actions: int = 1200,
    max_resets: int = 0,
    workers: int = 1,
    tags: list[str] | None = None,
    output_path: str | Path | None = None,
    time_budget_seconds: float | None = None,
    game_time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Run a scorecard with fail-soft per-game execution and a shared deadline."""
    from arc_agi import Arcade

    arc = Arcade()
    available = [x.game_id for x in arc.get_environments()]
    if games:
        selected = []
        for requested in games:
            if requested in available:
                selected.append(requested)
                continue
            matches = [gid for gid in available if gid.startswith(requested)]
            if len(matches) == 1:
                selected.append(matches[0])
            elif not matches:
                raise ValueError(f"Unknown/unavailable game prefix: {requested}")
            else:
                raise ValueError(f"Ambiguous game prefix {requested}: {matches}")
    else:
        selected = available

    card_id = arc.open_scorecard(tags=tags or ["arc3-frontier"])
    started_wall = time.time()
    started_mono = time.monotonic()
    stop_at = started_mono + time_budget_seconds if time_budget_seconds else None
    results: list[dict[str, Any]] = []
    scorecard = None

    def safe_one(game_id: str) -> dict[str, Any]:
        if stop_at is not None and time.monotonic() >= stop_at:
            return {
                **_error_result(game_id, TimeoutError("shared deadline reached before start")),
                "state": "SKIPPED_DEADLINE",
                "deadline_exhausted": True,
            }
        try:
            return run_game(
                arc,
                card_id,
                game_id,
                policy_factory,
                max_actions=max_actions,
                max_resets=max_resets,
                stop_at_monotonic=stop_at,
                game_time_budget_seconds=game_time_budget_seconds,
            )
        except BaseException as exc:  # submission robustness: preserve other games
            return _error_result(game_id, exc)

    try:
        if workers <= 1:
            for game_id in selected:
                if stop_at is not None and time.monotonic() >= stop_at:
                    results.append(
                        {
                            **_error_result(game_id, TimeoutError("shared deadline reached")),
                            "state": "SKIPPED_DEADLINE",
                            "deadline_exhausted": True,
                        }
                    )
                    continue
                results.append(safe_one(game_id))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(safe_one, gid): gid for gid in selected}
                for fut in as_completed(futs):
                    gid = futs[fut]
                    try:
                        results.append(fut.result())
                    except BaseException as exc:
                        results.append(_error_result(gid, exc))
    finally:
        # Closing the single scorecard is more important than propagating one game error.
        try:
            scorecard = arc.close_scorecard(card_id)
        except Exception as exc:
            results.append(_error_result("__scorecard_close__", exc))

    payload = {
        "elapsed_seconds": round(time.time() - started_wall, 3),
        "policy": tags or ["arc3-frontier"],
        "games": sorted(results, key=lambda x: x["game_id"]),
        "scorecard": scorecard.model_dump(mode="json") if scorecard else None,
        "diagnostics": {
            "errors": sum(bool(x.get("error")) for x in results),
            "deadline_exhausted_games": sum(bool(x.get("deadline_exhausted")) for x in results),
            "model_calls": sum(int(x.get("model_calls", 0)) for x in results),
            "model_failures": sum(int(x.get("model_failures", 0)) for x in results),
            "reasoning_cycles": sum(int(x.get("reasoning_cycles", 0)) for x in results),
            "tool_calls": sum(int(x.get("tool_calls", 0)) for x in results),
            "tool_failures": sum(int(x.get("tool_failures", 0)) for x in results),
            "queued_actions_used": sum(int(x.get("queued_actions_used", 0)) for x in results),
            "fallback_actions": sum(int(x.get("fallback_actions", 0)) for x in results),
            "prediction_mismatches": sum(int(x.get("prediction_mismatches", 0)) for x in results),
            "spatial_plans_requested": sum(int(x.get("spatial_plans_requested", 0)) for x in results),
            "spatial_plans_compiled": sum(int(x.get("spatial_plans_compiled", 0)) for x in results),
            "spatial_plan_actions": sum(int(x.get("spatial_plan_actions", 0)) for x in results),
            "spatial_plan_mismatches": sum(int(x.get("spatial_plan_mismatches", 0)) for x in results),
            "visual_packet_calls": sum(int(x.get("visual_packet_calls", 0)) for x in results),
            "multiview_calls": sum(int(x.get("multiview_calls", 0)) for x in results),
            "frontier_fallback_actions": sum(int(x.get("frontier_fallback_actions", 0)) for x in results),
            "frontier_known_states": sum(int(x.get("frontier_known_states", 0)) for x in results),
            "visual_candidate_selections": sum(int(x.get("visual_candidate_selections", 0)) for x in results),
            "visual_goal_updates": sum(int(x.get("visual_goal_updates", 0)) for x in results),
            "visual_affordance_observations": sum(int(x.get("visual_affordance_observations", 0)) for x in results),
            "visual_expectation_mismatches": sum(int(x.get("visual_expectation_mismatches", 0)) for x in results),
            "mean_final_orientation_entropy": round(
                sum(float(x.get("final_orientation_entropy", 1.0)) for x in results) / max(len(results), 1),
                4,
            ),
            "perceptual_modes": {
                mode: sum(int(x.get("final_perceptual_mode", "") == mode) for x in results)
                for mode in sorted({str(x.get("final_perceptual_mode", "")) for x in results if x.get("final_perceptual_mode")})
            },
            "world_model_delegations": sum(int(x.get("world_model_delegations", 0)) for x in results),
            "goal_hypotheses": sum(int(x.get("goal_hypotheses", 0)) for x in results),
        },
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
