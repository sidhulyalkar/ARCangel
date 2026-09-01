from __future__ import annotations

import pytest

from arc3lab.arena.runtime_budget import audit_runtime_budget


def test_hidden_110_game_envelope_tightens_four_wave_cap() -> None:
    audit = audit_runtime_budget(
        total_games=110,
        workers=28,
        notebook_limit_seconds=32400,
        setup_reserve_seconds=3600,
        global_budget_seconds=25200,
        requested_game_budget_seconds=7800,
        coverage_reserve_fraction=0.05,
    )
    assert audit.waves == 4
    assert audit.coverage_limited
    assert abs(audit.coverage_safe_game_budget_seconds - 5985.0) < 1e-9
    assert audit.effective_game_budget_seconds == audit.coverage_safe_game_budget_seconds
    assert audit.notebook_headroom_seconds == 3600.0


def test_public_25_game_envelope_keeps_requested_game_cap() -> None:
    audit = audit_runtime_budget(
        total_games=25,
        workers=28,
        notebook_limit_seconds=32400,
        setup_reserve_seconds=3600,
        global_budget_seconds=25200,
        requested_game_budget_seconds=7800,
        coverage_reserve_fraction=0.05,
    )
    assert audit.waves == 1
    assert not audit.coverage_limited
    assert audit.effective_game_budget_seconds == 7800.0
    assert audit.coverage_safe_game_budget_seconds == 23940.0


def test_runtime_audit_rejects_impossible_notebook_envelope() -> None:
    with pytest.raises(ValueError, match="exceeds notebook runtime"):
        audit_runtime_budget(
            total_games=110,
            workers=28,
            notebook_limit_seconds=32400,
            setup_reserve_seconds=8000,
            global_budget_seconds=25200,
            requested_game_budget_seconds=7800,
        )


def test_more_workers_reduce_waves_and_raise_safe_cap() -> None:
    fewer = audit_runtime_budget(
        total_games=110,
        workers=20,
        notebook_limit_seconds=32400,
        setup_reserve_seconds=3600,
        global_budget_seconds=25200,
        requested_game_budget_seconds=7800,
        coverage_reserve_fraction=0.05,
    )
    more = audit_runtime_budget(
        total_games=110,
        workers=28,
        notebook_limit_seconds=32400,
        setup_reserve_seconds=3600,
        global_budget_seconds=25200,
        requested_game_budget_seconds=7800,
        coverage_reserve_fraction=0.05,
    )
    assert fewer.waves == 6
    assert more.waves == 4
    assert more.coverage_safe_game_budget_seconds > fewer.coverage_safe_game_budget_seconds
