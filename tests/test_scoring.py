import pytest

from arc3lab.evaluation.scoring import game_score, level_score


def test_level_score_squared_ratio_and_bonus_cap():
    assert level_score(10, 20) == 25.0
    assert level_score(20, 10) == 115.0
    assert level_score(10, 0) == 0.0


def test_weighted_game_score_and_completion_cap():
    assert game_score([10, 10], [10, 20], 2) == 50.0
    assert game_score([20, 10], [1, 0], 1) == pytest.approx(100 / 3)
