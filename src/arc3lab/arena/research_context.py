from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from arc3lab.arena.scoring import aggregate_results, promotion_decision, rank_split


FORBIDDEN_DYNAMIC_EVIDENCE = ("blind", "kaggle", "leaderboard")


def sanitize_research_payload(payload: Any) -> Any:
    """Recursively remove dynamic private/leaderboard evidence from research payloads.

    Public competition documentation may still discuss Kaggle or historical public baselines.
    This sanitizer protects dynamic arena state, where current BLIND/Kaggle observations must not
    become architecture-invention features.
    """
    if isinstance(payload, dict):
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_DYNAMIC_EVIDENCE):
                continue
            clean[str(key)] = sanitize_research_payload(value)
        return clean
    if isinstance(payload, list):
        return [sanitize_research_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return [sanitize_research_payload(value) for value in payload]
    return payload


def assert_research_payload_safe(payload: Any) -> None:
    """Fail closed if forbidden dynamic-evidence keys survive projection."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_DYNAMIC_EVIDENCE):
                raise ValueError(f"research payload contains forbidden dynamic evidence key: {key}")
            assert_research_payload_safe(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_research_payload_safe(value)


def build_research_scorecard(lab: Any) -> dict[str, Any]:
    """Build a scorecard from DEV/VALIDATION rows only.

    This is intentionally separate from the ordinary arena scorecard. The ordinary scorecard is
    allowed to contain private-judge and exact-artifact leaderboard state for the campaign judge;
    the research scorecard is not.
    """
    results = [
        row
        for row in lab.ledger.read()
        if str(row.split).lower() in {"dev", "validation"}
    ]
    aggregates = aggregate_results(results, lab.manifest)
    rankings = {
        split: [asdict(item) for item in rank_split(aggregates, split)]
        for split in ("dev", "validation")
        if any(row.split == split for row in results)
    }
    decisions = [
        asdict(promotion_decision(contestant, aggregates, lab.manifest))
        for contestant in lab.manifest.contestants
        if contestant.control_id
    ]
    payload = {
        "experiment_id": lab.manifest.experiment_id,
        "result_count": len(results),
        "rankings": rankings,
        "promotion_decisions": decisions,
        "evidence_scope": ["dev", "validation"],
        "authority": "development research context only",
    }
    clean = sanitize_research_payload(json.loads(json.dumps(payload)))
    assert_research_payload_safe(clean)
    return clean
