from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable

from arc3lab.arena.schema import ArenaResult


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    contestant_id: str
    artifact_sha256: str
    runs: int
    mean_score: float
    score_std: float
    score_se: float
    lower_bound: float
    upper_bound: float
    scores: tuple[float, ...]
    mean_runtime_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LeaderboardComparison:
    contestant_id: str
    candidate_artifact_sha256: str
    control_id: str
    control_artifact_sha256: str
    candidate_runs: int
    control_runs: int
    candidate_mean: float
    control_mean: float
    mean_delta: float
    delta_lower_bound: float
    ready: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def artifact_evidence(
    results: Iterable[ArenaResult],
    contestant_id: str,
    *,
    confidence_se: float = 1.0,
    require_hash: bool = True,
) -> list[ArtifactEvidence]:
    grouped: dict[str, list[ArenaResult]] = defaultdict(list)
    for result in results:
        if result.contestant_id != contestant_id or result.split != "kaggle":
            continue
        if result.status != "ok" or "official_score" not in result.metrics:
            continue
        artifact = str(result.metadata.get("artifact_sha256", "")).strip().lower()
        if require_hash and not artifact:
            continue
        grouped[artifact or "<unhashed>"].append(result)

    evidence: list[ArtifactEvidence] = []
    for artifact, rows in grouped.items():
        scores = tuple(float(row.metrics["official_score"]) for row in rows)
        center = mean(scores)
        sigma = pstdev(scores) if len(scores) > 1 else 0.0
        se = sigma / sqrt(len(scores)) if len(scores) > 1 else 0.0
        runtimes = [
            float(row.metadata["runtime_seconds"])
            for row in rows
            if isinstance(row.metadata.get("runtime_seconds"), (int, float))
        ]
        evidence.append(
            ArtifactEvidence(
                contestant_id=contestant_id,
                artifact_sha256=artifact,
                runs=len(rows),
                mean_score=center,
                score_std=sigma,
                score_se=se,
                lower_bound=center - confidence_se * se,
                upper_bound=center + confidence_se * se,
                scores=scores,
                mean_runtime_seconds=mean(runtimes) if runtimes else None,
            )
        )
    return sorted(
        evidence,
        key=lambda row: (row.runs, row.lower_bound, row.mean_score),
        reverse=True,
    )


def compare_artifacts(
    candidate: ArtifactEvidence,
    control: ArtifactEvidence,
    *,
    min_candidate_runs: int,
    min_control_runs: int,
    min_delta: float,
    confidence_se: float = 1.0,
) -> LeaderboardComparison:
    reasons: list[str] = []
    if candidate.runs < min_candidate_runs:
        reasons.append(
            f"candidate repeats {candidate.runs} < required {min_candidate_runs}"
        )
    if control.runs < min_control_runs:
        reasons.append(f"control repeats {control.runs} < required {min_control_runs}")

    mean_delta = candidate.mean_score - control.mean_score
    combined_se = sqrt(candidate.score_se**2 + control.score_se**2)
    delta_lower = mean_delta - confidence_se * combined_se
    if not reasons and delta_lower < min_delta:
        reasons.append(
            f"uncertainty-adjusted Kaggle delta {delta_lower:.4f} < {min_delta:.4f}"
        )

    return LeaderboardComparison(
        contestant_id=candidate.contestant_id,
        candidate_artifact_sha256=candidate.artifact_sha256,
        control_id=control.contestant_id,
        control_artifact_sha256=control.artifact_sha256,
        candidate_runs=candidate.runs,
        control_runs=control.runs,
        candidate_mean=candidate.mean_score,
        control_mean=control.mean_score,
        mean_delta=mean_delta,
        delta_lower_bound=delta_lower,
        ready=not reasons,
        reasons=tuple(reasons),
    )
