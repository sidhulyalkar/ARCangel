from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Any, Iterable

from arc3lab.arena.schema import ArenaManifest, ArenaResult
from arc3lab.arena.scoring import score_result
from arc3lab.arena.swarm_intelligence import SwarmOutcome


@dataclass(frozen=True, slots=True)
class SwarmFitnessEvidence:
    proposal_id: str
    provider_id: str
    role_id: str
    split: str
    target_profile: str
    control_profile: str
    paired_seeds: tuple[int, ...]
    runs: int
    mean_delta: float
    delta_std: float
    delta_se: float
    robust_delta: float
    candidate_failure_rate: float
    candidate_emergency_fraction: float
    memory_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_outcome(self, source: str) -> SwarmOutcome:
        return SwarmOutcome(
            proposal_id=self.proposal_id,
            provider_id=self.provider_id,
            role_id=self.role_id,
            split=self.split,
            utility=self.robust_delta,
            source=source,
            status=self.memory_status,
            note=(
                f"paired {self.target_profile} vs {self.control_profile}; runs={self.runs}; "
                f"mean_delta={self.mean_delta:.4f}; robust_delta={self.robust_delta:.4f}; "
                f"failure={self.candidate_failure_rate:.4f}; "
                f"emergency={self.candidate_emergency_fraction:.4f}"
            ),
        )


def _unique_by_seed(rows: Iterable[ArenaResult], split: str) -> dict[int, ArenaResult]:
    selected: dict[int, ArenaResult] = {}
    for row in rows:
        if row.split != split:
            continue
        if row.seed in selected:
            raise ValueError(f"duplicate {split} result for seed {row.seed}")
        selected[row.seed] = row
    return selected


def evaluate_swarm_fitness(
    battle_row: dict[str, Any],
    candidate_results: Iterable[ArenaResult],
    control_results: Iterable[ArenaResult],
    manifest: ArenaManifest,
    *,
    confidence_se: float = 1.0,
    min_measured_runs: int = 2,
) -> SwarmFitnessEvidence:
    split = str(battle_row.get("selection_split", "")).lower()
    if split not in {"dev", "validation"}:
        raise ValueError("swarm fitness accepts DEV/VALIDATION experiments only")
    proposal_id = str(battle_row.get("proposal_id", "")).strip()
    provider_id = str(battle_row.get("provider_id", "")).strip()
    role_id = str(battle_row.get("role_id", "")).strip()
    target_profile = str(battle_row.get("target_profile", "")).strip()
    control_profile = str(battle_row.get("control_profile", "")).strip()
    if not all((proposal_id, provider_id, role_id, target_profile, control_profile)):
        raise ValueError("battle row lacks swarm identity or executable profile contract")

    candidate = _unique_by_seed(candidate_results, split)
    control = _unique_by_seed(control_results, split)
    paired = tuple(sorted(set(candidate) & set(control)))
    if not paired:
        raise ValueError("candidate/control have no paired seeds")

    deltas = [
        score_result(candidate[seed], manifest.weights)
        - score_result(control[seed], manifest.weights)
        for seed in paired
    ]
    center = mean(deltas)
    sigma = pstdev(deltas) if len(deltas) > 1 else 0.0
    se = sigma / sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    robust = center - max(0.0, float(confidence_se)) * se
    failures = mean(
        1.0
        if candidate[seed].status not in {"ok", "degraded"}
        else float(candidate[seed].metrics.get("failure_rate", 0.0))
        for seed in paired
    )
    emergency = mean(
        float(candidate[seed].metrics.get("emergency_fraction", 0.0))
        for seed in paired
    )
    healthy = (
        failures <= manifest.promotion.max_failure_rate
        and emergency <= manifest.promotion.max_emergency_fraction
    )
    memory_status = (
        "measured"
        if len(paired) >= max(1, int(min_measured_runs)) and healthy
        else "preliminary"
    )
    return SwarmFitnessEvidence(
        proposal_id=proposal_id,
        provider_id=provider_id,
        role_id=role_id,
        split=split,
        target_profile=target_profile,
        control_profile=control_profile,
        paired_seeds=paired,
        runs=len(paired),
        mean_delta=center,
        delta_std=sigma,
        delta_se=se,
        robust_delta=robust,
        candidate_failure_rate=failures,
        candidate_emergency_fraction=emergency,
        memory_status=memory_status,
    )
