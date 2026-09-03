from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable

from arc3lab.arena.schema import ArenaManifest, ArenaResult, ContestantSpec


@dataclass(frozen=True, slots=True)
class AggregateScore:
    contestant_id: str
    split: str
    runs: int
    mean_score: float
    robust_score: float
    score_std: float
    metrics: dict[str, float]
    failure_rate: float
    emergency_fraction: float


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    contestant_id: str
    control_id: str | None
    promoted: bool
    reasons: tuple[str, ...]
    validation_delta: float | None
    dev_delta: float | None


def score_result(result: ArenaResult, weights: dict[str, float]) -> float:
    # Hard execution failures carry no trustworthy behavioral signal. A degraded suite,
    # however, can contain many valid games and should retain its metric gradient while
    # paying explicit failure/timeout penalties below.
    if result.status not in {"ok", "degraded"}:
        return -1.0
    score = 0.0
    norm = sum(abs(weight) for weight in weights.values()) or 1.0
    for metric, weight in weights.items():
        score += weight * float(result.metrics.get(metric, 0.0))
    failure = max(0.0, min(1.0, float(result.metrics.get("failure_rate", 0.0))))
    emergency = max(0.0, min(1.0, float(result.metrics.get("emergency_fraction", 0.0))))
    timeout = max(0.0, min(1.0, float(result.metrics.get("timeout_fraction", 0.0))))
    return score / norm - 0.40 * failure - 0.30 * emergency - 0.20 * timeout


def aggregate_results(
    results: Iterable[ArenaResult],
    manifest: ArenaManifest,
) -> dict[tuple[str, str], AggregateScore]:
    grouped: dict[tuple[str, str], list[ArenaResult]] = defaultdict(list)
    for result in results:
        grouped[(result.contestant_id, result.split)].append(result)

    aggregates: dict[tuple[str, str], AggregateScore] = {}
    for key, rows in grouped.items():
        raw_scores = [score_result(row, manifest.weights) for row in rows]
        metric_keys = sorted({metric for row in rows for metric in row.metrics})
        metric_means = {
            metric: mean(float(row.metrics.get(metric, 0.0)) for row in rows)
            for metric in metric_keys
        }
        sigma = pstdev(raw_scores) if len(raw_scores) > 1 else 0.0
        # A one-standard-error lower bound rewards repeatability instead of lucky runs.
        robust = mean(raw_scores) - sigma / sqrt(max(1, len(raw_scores)))
        failures = mean(
            1.0
            if row.status not in {"ok", "degraded"}
            else float(row.metrics.get("failure_rate", 0.0))
            for row in rows
        )
        emergency = mean(float(row.metrics.get("emergency_fraction", 0.0)) for row in rows)
        aggregates[key] = AggregateScore(
            contestant_id=key[0],
            split=key[1],
            runs=len(rows),
            mean_score=mean(raw_scores),
            robust_score=robust,
            score_std=sigma,
            metrics=metric_means,
            failure_rate=float(failures),
            emergency_fraction=float(emergency),
        )
    return aggregates


def rank_split(
    aggregates: dict[tuple[str, str], AggregateScore],
    split: str,
) -> list[AggregateScore]:
    return sorted(
        (aggregate for (_, row_split), aggregate in aggregates.items() if row_split == split),
        key=lambda item: (item.robust_score, item.mean_score),
        reverse=True,
    )


def promotion_decision(
    contestant: ContestantSpec,
    aggregates: dict[tuple[str, str], AggregateScore],
    manifest: ArenaManifest,
) -> PromotionDecision:
    rules = manifest.promotion
    control_id = contestant.control_id
    reasons: list[str] = []
    validation = aggregates.get((contestant.contestant_id, "validation"))
    dev = aggregates.get((contestant.contestant_id, "dev"))

    if validation is None:
        reasons.append("missing validation result")
    elif validation.runs < rules.min_validation_runs:
        reasons.append(
            f"validation runs {validation.runs} < required {rules.min_validation_runs}"
        )
    if validation and validation.emergency_fraction > rules.max_emergency_fraction:
        reasons.append("emergency ownership exceeds promotion ceiling")
    if validation and validation.failure_rate > rules.max_failure_rate:
        reasons.append("failure rate exceeds promotion ceiling")

    validation_delta: float | None = None
    dev_delta: float | None = None
    if control_id:
        control_validation = aggregates.get((control_id, "validation"))
        control_dev = aggregates.get((control_id, "dev"))
        if control_validation is None:
            reasons.append("control is missing validation evidence")
        elif validation is not None:
            validation_delta = validation.robust_score - control_validation.robust_score
            if validation_delta < rules.min_validation_delta:
                reasons.append(
                    f"validation delta {validation_delta:.4f} < {rules.min_validation_delta:.4f}"
                )
        if dev is not None and control_dev is not None:
            dev_delta = dev.robust_score - control_dev.robust_score
            if dev_delta < rules.min_dev_delta:
                reasons.append(f"dev delta {dev_delta:.4f} < {rules.min_dev_delta:.4f}")
    elif rules.require_control:
        reasons.append("contestant has no explicit control")

    return PromotionDecision(
        contestant_id=contestant.contestant_id,
        control_id=control_id,
        promoted=not reasons,
        reasons=tuple(reasons),
        validation_delta=validation_delta,
        dev_delta=dev_delta,
    )
