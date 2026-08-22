from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from arc3lab.types import ActionSpec, Transition


def action_key(action: ActionSpec) -> tuple[int, int | None, int | None]:
    """Stable identity for a primitive/click action."""
    return (int(action.action_id), action.x, action.y)


def predictive_state_key(
    current_signature: str,
    transitions: list[Transition],
    *,
    history_depth: int = 2,
) -> str:
    """Return the compact temporal state key promoted by D210R2.

    D210R2 showed that an exact current-grid signature alone aliases hidden/temporal
    state on the public diagnostic set. The smallest global representation exceeding
    the 0.99 repeated-key consistency target was h2: the current scene plus the two
    most recent before-state/action pairs.
    """
    tail = transitions[-max(0, int(history_depth)) :]
    payload = {
        "current": current_signature,
        "history": [
            {
                "before": t.before_signature,
                "action": action_key(t.action),
            }
            for t in tail
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


class PredictiveTransitionMemory:
    """Distributional temporal state-action transition memory.

    This is a verification/cache layer, not a claim that every ARC environment is
    globally Markov under h2. Uncovered or contradicted contexts remain model-led.
    """

    def __init__(self, history_depth: int = 2) -> None:
        self.history_depth = max(0, int(history_depth))
        self.next_counts: dict[tuple[str, tuple], Counter[str]] = defaultdict(Counter)
        self.effect_counts: dict[tuple[str, tuple], Counter[str]] = defaultdict(Counter)
        self.observations = 0
        self.correct_high_confidence = 0
        self.incorrect_high_confidence = 0

    def state_key(self, current_signature: str, transitions: list[Transition]) -> str:
        return predictive_state_key(
            current_signature,
            transitions,
            history_depth=self.history_depth,
        )

    def observe(
        self,
        state_key: str,
        action: ActionSpec,
        next_signature: str,
        effect: str,
    ) -> None:
        key = (state_key, action_key(action))
        self.next_counts[key][next_signature] += 1
        self.effect_counts[key][effect] += 1
        self.observations += 1

    def prediction(self, state_key: str, action: ActionSpec) -> dict | None:
        key = (state_key, action_key(action))
        counts = self.next_counts.get(key)
        if not counts:
            return None
        next_signature, support = counts.most_common(1)[0]
        total = sum(counts.values())
        effects = self.effect_counts.get(key, Counter())
        effect = effects.most_common(1)[0][0] if effects else None
        return {
            "next_signature": next_signature,
            "confidence": support / total,
            "evidence": total,
            "effect": effect,
        }

    def context_coverage(self, state_key: str, actions: list[ActionSpec]) -> float:
        """Fraction of candidate actions observed at least once in this temporal state."""
        if not actions:
            return 0.0
        covered = sum(
            bool(self.next_counts.get((state_key, action_key(action))))
            for action in actions
        )
        return covered / len(actions)

    def record_verification(self, matched: bool, confidence: float) -> None:
        if confidence < 0.80:
            return
        if matched:
            self.correct_high_confidence += 1
        else:
            self.incorrect_high_confidence += 1

    def summary(self) -> dict:
        deterministic = 0
        for counts in self.next_counts.values():
            total = sum(counts.values())
            if total and counts.most_common(1)[0][1] / total >= 0.95:
                deterministic += 1
        checked = self.correct_high_confidence + self.incorrect_high_confidence
        return {
            "history_depth": self.history_depth,
            "observations": self.observations,
            "covered_state_actions": len(self.next_counts),
            "high_confidence_state_actions": deterministic,
            "high_confidence_verification_accuracy": (
                self.correct_high_confidence / checked if checked else None
            ),
            "prediction_mismatches": self.incorrect_high_confidence,
        }
