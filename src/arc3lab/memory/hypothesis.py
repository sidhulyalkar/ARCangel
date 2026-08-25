from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CATEGORIES = ("agency", "mechanics", "goals", "abstraction")


@dataclass(slots=True)
class TypedHypothesis:
    category: str
    statement: str
    confidence: float
    evidence: str
    first_level: int
    first_step: int
    last_level: int
    last_step: int
    support: int = 1
    contradictions: int = 0

    @property
    def calibrated_confidence(self) -> float:
        bonus = min(0.12, 0.025 * max(0, self.support - 1))
        penalty = min(0.45, 0.15 * self.contradictions)
        return max(0.0, min(1.0, self.confidence + bonus - penalty))


class HypothesisRegistry:
    """Small typed theory ledger; continuity memory, never a hidden action rule."""

    def __init__(self, *, max_per_category: int = 16) -> None:
        self.max_per_category = max(4, int(max_per_category))
        self.items: list[TypedHypothesis] = []

    @staticmethod
    def _norm(statement: Any) -> str:
        return " ".join(str(statement or "").split())[:500]

    def _find(self, category: str, statement: str) -> TypedHypothesis | None:
        key = statement.casefold()
        return next((x for x in self.items if x.category == category and x.statement.casefold() == key), None)

    def update_from_model(self, raw: Any, *, level: int, step: int) -> int:
        if not isinstance(raw, dict):
            return 0
        updated = 0
        for category in CATEGORIES:
            rows = raw.get(category)
            if not isinstance(rows, list):
                continue
            for row in rows[:4]:
                if not isinstance(row, dict):
                    continue
                statement = self._norm(row.get("statement"))
                if not statement:
                    continue
                try:
                    confidence = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
                except Exception:
                    confidence = 0.5
                evidence = self._norm(row.get("evidence"))
                existing = self._find(category, statement)
                if existing is None:
                    self.items.append(TypedHypothesis(
                        category, statement, confidence, evidence,
                        int(level), int(step), int(level), int(step),
                    ))
                else:
                    existing.support += 1
                    existing.confidence = max(existing.confidence, 0.72 * existing.confidence + 0.28 * confidence)
                    if evidence:
                        existing.evidence = evidence
                    existing.last_level, existing.last_step = int(level), int(step)
                updated += 1

        pruned: list[TypedHypothesis] = []
        for category in CATEGORIES:
            rows = [x for x in self.items if x.category == category]
            rows.sort(key=lambda x: (x.calibrated_confidence, x.support, x.last_level, x.last_step), reverse=True)
            pruned.extend(rows[: self.max_per_category])
        self.items = pruned
        return updated

    def contradict_recent(
        self, *, categories: tuple[str, ...] = ("agency", "mechanics"),
        level: int, step: int, window: int = 8,
    ) -> int:
        candidates = [
            x for x in self.items
            if x.category in categories and x.last_level == int(level) and int(step) - x.last_step <= int(window)
        ]
        if not candidates:
            return 0
        latest = max(x.last_step for x in candidates)
        touched = 0
        for item in candidates:
            if item.last_step == latest:
                item.contradictions += 1
                touched += 1
        return touched

    def summary(self, *, limit_per_category: int = 6) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for category in CATEGORIES:
            rows = [x for x in self.items if x.category == category]
            rows.sort(key=lambda x: (x.calibrated_confidence, x.support, x.last_level, x.last_step), reverse=True)
            out[category] = [{
                "statement": x.statement,
                "confidence": round(x.calibrated_confidence, 4),
                "raw_confidence": round(x.confidence, 4),
                "support": x.support,
                "contradictions": x.contradictions,
                "evidence": x.evidence,
                "first_level": x.first_level,
                "last_level": x.last_level,
            } for x in rows[: max(1, int(limit_per_category))]]
        return out

    def __len__(self) -> int:
        return len(self.items)
