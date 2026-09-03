from __future__ import annotations

from typing import Any

from arc3lab.arena.schema import ArenaResult


def _ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator) / float(denominator) if denominator else float(default)


def suite_payload_to_result(
    payload: dict[str, Any],
    *,
    contestant_id: str,
    split: str,
    seed: int,
    source: str = "",
) -> ArenaResult:
    games = list(payload.get("games") or [])
    diagnostics = dict(payload.get("diagnostics") or {})
    real_games = [row for row in games if row.get("game_id") != "__scorecard_close__"]
    game_count = max(1, len(real_games))
    wins = sum(str(row.get("state", "")) == "WIN" for row in real_games)
    levels = sum(max(0, int(row.get("levels_completed", 0))) for row in real_games)
    actions = sum(max(0, int(row.get("actions", 0))) for row in real_games)
    model_calls = sum(max(0, int(row.get("model_calls", 0))) for row in real_games)
    failures = sum(bool(row.get("error")) for row in real_games)
    deadlines = sum(bool(row.get("deadline_exhausted")) for row in real_games)
    emergency = int(diagnostics.get("emergency_transport_fallbacks", 0))
    if not emergency:
        emergency = int(diagnostics.get("fallback_actions", 0))

    expectation_checks = int(diagnostics.get("expectation_checks", 0))
    expectation_mismatches = int(diagnostics.get("expectation_mismatches", 0))
    hypothesis_tests = int(diagnostics.get("hypothesis_tests", 0))
    hypothesis_failures = int(diagnostics.get("hypothesis_test_failures", 0))

    # These transforms keep each objective in [0, 1] while preserving useful gradients.
    solve_rate = _ratio(wins, game_count)
    level_rate = 1.0 - 1.0 / (1.0 + _ratio(levels, game_count))
    actions_per_level = _ratio(actions, max(1, levels), default=float(actions))
    action_efficiency = 1.0 / (1.0 + actions_per_level / 50.0)
    calls_per_action = _ratio(model_calls, max(1, actions))
    model_efficiency = 1.0 / (1.0 + calls_per_action)
    prediction_accuracy = (
        1.0 - _ratio(expectation_mismatches, expectation_checks)
        if expectation_checks
        else 0.5
    )
    falsification_health = (
        1.0 - _ratio(hypothesis_failures, hypothesis_tests)
        if hypothesis_tests
        else 0.5
    )
    failure_rate = _ratio(failures, game_count)
    timeout_fraction = _ratio(deadlines, game_count)
    emergency_fraction = _ratio(emergency, max(1, actions))
    stability = max(0.0, 1.0 - failure_rate - 0.5 * timeout_fraction)

    scorecard = payload.get("scorecard")
    official_score = 0.0
    if isinstance(scorecard, dict):
        for key in ("score", "total_score", "reward"):
            value = scorecard.get(key)
            if isinstance(value, (int, float)):
                official_score = float(value)
                break

    metrics = {
        "solve_rate": solve_rate,
        "level_rate": level_rate,
        "action_efficiency": action_efficiency,
        "model_efficiency": model_efficiency,
        "prediction_accuracy": prediction_accuracy,
        "falsification_health": falsification_health,
        "stability": stability,
        "official_score": official_score,
        "failure_rate": failure_rate,
        "timeout_fraction": timeout_fraction,
        "emergency_fraction": emergency_fraction,
    }
    # A structurally valid suite remains scoreable when one game fails. The explicit
    # failure/timeout metrics carry the penalty and preserve information about the rest
    # of the run. Reserve non-ok status for runner-level failures that produced no
    # trustworthy suite receipt.
    return ArenaResult(
        contestant_id=contestant_id,
        split=split,
        seed=seed,
        metrics=metrics,
        status="ok",
        source=source,
        metadata={
            "games": len(real_games),
            "wins": wins,
            "levels": levels,
            "actions": actions,
            "model_calls": model_calls,
            "expectation_checks": expectation_checks,
            "hypothesis_tests": hypothesis_tests,
            "failed_games": failures,
            "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
        },
    )
