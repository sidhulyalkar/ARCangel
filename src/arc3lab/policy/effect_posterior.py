from __future__ import annotations

import math

from arc3lab.policy.structural import StructuralPolicy
from arc3lab.types import ActionSpec, Scene


class EffectPosteriorPolicy(StructuralPolicy):
    """D110R2-promoted generic fallback.

    Preserve first-level primitive identification, then softly favor action channels
    with demonstrated state-changing effects. This deliberately avoids hard global
    dead-action bans because D110R2 showed anti-dead suppression was harmful.
    """

    def _posterior(self, action_id: int) -> tuple[float, int]:
        rows = [t for t in self.memory.transitions if t.action.action_id == action_id]
        successes = sum(bool(t.meaningful_changed_cells or t.level_completed) for t in rows)
        probability = (successes + 1.5) / (len(rows) + 3.0)  # Beta(1.5, 1.5)
        return probability, len(rows)

    def choose(self, scene: Scene) -> ActionSpec:
        valid = set(scene.available_actions)
        simple = [a for a in (1, 2, 3, 4, 5, 7) if a in valid]

        # Preserve the V004 discipline: establish primitive semantics in level 0,
        # then carry that evidence rather than blindly re-probing every later level.
        if scene.level == 0:
            unprobed = [a for a in simple if a not in self.probed_simple_global]
            if unprobed:
                return super().choose(scene)

        scored: list[tuple[float, int, int]] = []
        for action_id in simple:
            probability, evidence = self._posterior(action_id)
            dead_rate = self.memory.dead_action_rate(action_id)
            local_trials = self.graph.action_trials(scene.signature, ActionSpec(action_id))
            score = (
                2.2 * probability
                - 1.4 * dead_rate
                - 0.22 * local_trials
                + 0.12 / math.sqrt(1 + evidence)
            )
            scored.append((score, -action_id, action_id))

        if scored and max(scored)[0] > 0.25:
            action_id = max(scored)[2]
            spec = ActionSpec(
                action_id,
                reason="D110R2 causal effect posterior",
                confidence=0.60,
            )
            self.last_action, self.last_target_shape = spec, None
            return spec

        return super().choose(scene)
