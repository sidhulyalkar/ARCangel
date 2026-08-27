from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class HypothesisRecord:
    hypothesis_id: str
    kind: str
    claim: str
    confidence: float
    status: str = "provisional"
    support: int = 0
    contradictions: int = 0
    first_step: int = 0
    last_step: int = 0
    test_python: str = ""
    evidence_note: str = ""

    def record(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceWorkspace:
    """Persistent per-game scientific state, kept separate from immutable evidence.

    The transition ledger remains the source of truth. This workspace stores only
    provisional interpretations of that evidence: hypotheses, a compact playbook,
    open questions, and optional executable world-model source. Nothing in this
    object is allowed to overwrite or delete recorded observations.
    """

    def __init__(self, *, max_hypotheses: int = 48, max_notes: int = 32) -> None:
        self.max_hypotheses = max(8, int(max_hypotheses))
        self.max_notes = max(8, int(max_notes))
        self.hypotheses: dict[str, HypothesisRecord] = {}
        self.validated_rules: list[str] = []
        self.falsified_rules: list[str] = []
        self.open_questions: list[str] = []
        self.level_notes: list[str] = []
        self.world_model_code = ""
        self.world_model_validation: dict[str, Any] = {}
        self.revision = 0

    @staticmethod
    def _text(value: Any, limit: int = 1200) -> str:
        return " ".join(str(value or "").split())[:limit]

    @staticmethod
    def _confidence(value: Any, default: float = 0.5) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except Exception:
            return default

    def upsert_hypothesis(self, raw: Any, *, step: int) -> HypothesisRecord | None:
        if not isinstance(raw, dict):
            return None
        claim = self._text(raw.get("claim"), 900)
        if not claim:
            return None
        hypothesis_id = self._text(raw.get("id"), 80) or f"h{len(self.hypotheses) + 1}"
        kind = self._text(raw.get("kind"), 60) or "mechanic"
        record = self.hypotheses.get(hypothesis_id)
        if record is None:
            record = HypothesisRecord(
                hypothesis_id=hypothesis_id,
                kind=kind,
                claim=claim,
                confidence=self._confidence(raw.get("confidence")),
                first_step=int(step),
                last_step=int(step),
                test_python=str(raw.get("test_python") or "")[:7000],
                evidence_note=self._text(raw.get("evidence"), 800),
            )
            self.hypotheses[hypothesis_id] = record
        else:
            record.kind = kind
            record.claim = claim
            record.confidence = self._confidence(raw.get("confidence"), record.confidence)
            record.last_step = int(step)
            if raw.get("test_python"):
                record.test_python = str(raw.get("test_python"))[:7000]
            if raw.get("evidence"):
                record.evidence_note = self._text(raw.get("evidence"), 800)
        self._trim_hypotheses()
        self.revision += 1
        return record

    def record_falsification(self, hypothesis_id: str, result: Any, *, step: int) -> None:
        record = self.hypotheses.get(str(hypothesis_id))
        if record is None or not isinstance(result, dict):
            return
        try:
            support = max(0, int(result.get("support", 0)))
            contradictions = max(0, int(result.get("contradictions", 0)))
        except Exception:
            support, contradictions = 0, 0
        consistent = bool(result.get("consistent", contradictions == 0))
        record.support = support
        record.contradictions = contradictions
        record.last_step = int(step)
        if contradictions > 0 or not consistent:
            record.status = "falsified"
            record.confidence = min(record.confidence, 0.20)
            self._append_unique(self.falsified_rules, record.claim)
        elif support > 0:
            record.status = "history_consistent"
            if support >= 2:
                record.confidence = max(record.confidence, 0.70)
        self.revision += 1

    def validate_from_progress(self, *, level: int, step: int, note: str = "") -> None:
        """Promote history-consistent high-confidence rules after real progress.

        Level completion does not prove every active theory, but it is strong evidence
        that the currently retained high-confidence mechanics deserve cross-level reuse.
        """
        for record in self.hypotheses.values():
            if record.status == "history_consistent" and record.confidence >= 0.65:
                record.status = "validated"
                record.last_step = int(step)
                self._append_unique(self.validated_rules, record.claim)
        if note:
            self._append_unique(self.level_notes, f"level {level}: {self._text(note, 500)}")
        self.revision += 1

    def apply_patch(self, patch: Any, *, step: int) -> None:
        if not isinstance(patch, dict):
            return
        for key, target in (
            ("validated_add", self.validated_rules),
            ("falsified_add", self.falsified_rules),
            ("questions", self.open_questions),
            ("level_notes", self.level_notes),
        ):
            values = patch.get(key)
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                for value in values:
                    text = self._text(value, 700)
                    if text:
                        self._append_unique(target, text)
        world_model_code = patch.get("world_model_code")
        if isinstance(world_model_code, str) and world_model_code.strip():
            self.world_model_code = world_model_code[:7000]
            self.world_model_validation = {
                "status": "unvalidated",
                "step": int(step),
            }
        self.revision += 1

    def set_world_model_validation(self, result: Any, *, step: int) -> None:
        if isinstance(result, dict):
            self.world_model_validation = {**result, "step": int(step)}
        else:
            self.world_model_validation = {"status": "invalid", "result": str(result)[:500], "step": int(step)}
        self.revision += 1

    def _append_unique(self, target: list[str], value: str) -> None:
        lowered = value.lower()
        if lowered not in {x.lower() for x in target}:
            target.append(value)
        if len(target) > self.max_notes:
            del target[:-self.max_notes]

    def _trim_hypotheses(self) -> None:
        if len(self.hypotheses) <= self.max_hypotheses:
            return
        ranked = sorted(
            self.hypotheses.values(),
            key=lambda h: (
                h.status in {"validated", "history_consistent"},
                h.confidence,
                h.last_step,
            ),
            reverse=True,
        )[: self.max_hypotheses]
        self.hypotheses = {h.hypothesis_id: h for h in ranked}

    def summary(self) -> dict[str, Any]:
        active = [
            record.record()
            for record in sorted(
                self.hypotheses.values(),
                key=lambda h: (h.status == "validated", h.status == "history_consistent", h.confidence, h.last_step),
                reverse=True,
            )
            if record.status != "falsified"
        ][:16]
        falsified = [
            record.record()
            for record in self.hypotheses.values()
            if record.status == "falsified"
        ][-8:]
        return {
            "revision": self.revision,
            "validated_rules": self.validated_rules[-12:],
            "active_hypotheses": active,
            "falsified_hypotheses": falsified,
            "open_questions": self.open_questions[-10:],
            "level_notes": self.level_notes[-8:],
            "world_model": {
                "present": bool(self.world_model_code.strip()),
                "validation": dict(self.world_model_validation),
            },
        }
