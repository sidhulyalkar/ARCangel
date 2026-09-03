from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class RuntimeBudgetAudit:
    total_games: int
    workers: int
    notebook_limit_seconds: float
    setup_reserve_seconds: float
    global_budget_seconds: float
    requested_game_budget_seconds: float
    waves: int
    coverage_safe_game_budget_seconds: float
    effective_game_budget_seconds: float
    games_per_hour_required: float
    mean_worker_service_seconds_available: float
    notebook_headroom_seconds: float
    coverage_limited: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_runtime_budget(
    *,
    total_games: int,
    workers: int,
    notebook_limit_seconds: float,
    setup_reserve_seconds: float,
    global_budget_seconds: float,
    requested_game_budget_seconds: float,
    coverage_reserve_fraction: float = 0.05,
) -> RuntimeBudgetAudit:
    total_games = int(total_games)
    workers = int(workers)
    if total_games <= 0:
        raise ValueError("total_games must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if notebook_limit_seconds <= 0 or global_budget_seconds <= 0:
        raise ValueError("runtime budgets must be positive")
    if setup_reserve_seconds < 0:
        raise ValueError("setup reserve cannot be negative")
    if not 0 <= coverage_reserve_fraction < 1:
        raise ValueError("coverage reserve fraction must be in [0, 1)")
    if global_budget_seconds + setup_reserve_seconds > notebook_limit_seconds:
        raise ValueError(
            "global campaign budget plus setup reserve exceeds notebook runtime limit"
        )

    waves = ceil(total_games / workers)
    coverage_budget = (
        global_budget_seconds * (1.0 - coverage_reserve_fraction) / max(1, waves)
    )
    effective = min(float(requested_game_budget_seconds), coverage_budget)
    if effective <= 0:
        raise ValueError("effective per-game budget must be positive")

    campaign_hours = global_budget_seconds / 3600.0
    games_per_hour = total_games / campaign_hours
    mean_service = workers * global_budget_seconds / total_games
    headroom = notebook_limit_seconds - setup_reserve_seconds - global_budget_seconds

    return RuntimeBudgetAudit(
        total_games=total_games,
        workers=workers,
        notebook_limit_seconds=float(notebook_limit_seconds),
        setup_reserve_seconds=float(setup_reserve_seconds),
        global_budget_seconds=float(global_budget_seconds),
        requested_game_budget_seconds=float(requested_game_budget_seconds),
        waves=waves,
        coverage_safe_game_budget_seconds=coverage_budget,
        effective_game_budget_seconds=effective,
        games_per_hour_required=games_per_hour,
        mean_worker_service_seconds_available=mean_service,
        notebook_headroom_seconds=headroom,
        coverage_limited=effective + 1e-9 < float(requested_game_budget_seconds),
    )
