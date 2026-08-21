from __future__ import annotations


def level_score(human_actions: int, agent_actions: int) -> float:
    """arc-agi 0.9.8 level score in points (0..115 for a completed level)."""
    if agent_actions <= 0:
        return 0.0
    return min(((human_actions / agent_actions) ** 2) * 100.0, 115.0)


def game_score(baselines: list[int], actions: list[int], completed: int) -> float:
    """Mirror arc-agi 0.9.8 weighted scoring and completion-dependent game cap."""
    if not baselines:
        return 0.0
    weights = list(range(1, len(baselines) + 1))
    scores = [level_score(baselines[i], actions[i]) if i < completed else 0.0 for i in range(len(baselines))]
    total_w = sum(weights)
    raw = sum(w * s for w, s in zip(weights, scores)) / total_w
    completed_w = sum(weights[:completed])
    max_score = completed_w / total_w * 100.0
    return min(raw, max_score)
