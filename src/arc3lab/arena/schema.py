from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ALLOWED_SPLITS = {"dev", "validation", "blind", "kaggle"}


@dataclass(frozen=True, slots=True)
class ContestantSpec:
    contestant_id: str
    family: str
    role: str
    description: str = ""
    command: tuple[str, ...] = ()
    judge_command: tuple[str, ...] = ()
    parent: str | None = None
    control_id: str | None = None
    tags: tuple[str, ...] = ()
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContestantSpec":
        contestant_id = str(data["id"]).strip()
        if not contestant_id:
            raise ValueError("contestant id cannot be empty")
        command = data.get("command", ())
        judge_command = data.get("judge_command", ())
        if isinstance(command, str) or isinstance(judge_command, str):
            raise TypeError("contestant commands must be argv lists, not shell strings")
        return cls(
            contestant_id=contestant_id,
            family=str(data.get("family", contestant_id)),
            role=str(data.get("role", "researcher")),
            description=str(data.get("description", "")),
            command=tuple(str(token) for token in command),
            judge_command=tuple(str(token) for token in judge_command),
            parent=data.get("parent"),
            control_id=data.get("control_id"),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class PromotionRules:
    min_validation_delta: float = 0.02
    min_dev_delta: float = -0.01
    min_blind_delta: float = -0.01
    max_emergency_fraction: float = 0.02
    max_failure_rate: float = 0.05
    min_validation_runs: int = 2
    min_blind_runs: int = 2
    require_control: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PromotionRules":
        return cls(**(data or {}))


@dataclass(frozen=True, slots=True)
class ArenaManifest:
    experiment_id: str
    seeds: tuple[int, ...]
    contestants: tuple[ContestantSpec, ...]
    weights: dict[str, float]
    promotion: PromotionRules
    timeout_seconds: int = 7200
    max_parallel: int = 1
    split_salt: str = "arcangel-v013"
    leaderboard_control_id: str | None = None
    min_leaderboard_delta: float = 0.0
    min_leaderboard_candidate_runs: int = 2
    min_leaderboard_control_runs: int = 2
    leaderboard_confidence_se: float = 1.0
    require_leaderboard_artifact_hash: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "ArenaManifest":
        raw = json.loads(Path(path).read_text())
        contestants = tuple(ContestantSpec.from_dict(item) for item in raw["contestants"])
        seeds = tuple(int(seed) for seed in raw.get("seeds", [20260831]))
        if not seeds:
            raise ValueError("manifest requires at least one seed")
        weights = {str(key): float(value) for key, value in raw.get("weights", {}).items()}
        if not weights:
            raise ValueError("manifest requires scoring weights")
        if sum(abs(value) for value in weights.values()) <= 0:
            raise ValueError("scoring weights cannot all be zero")
        candidate_runs = max(1, int(raw.get("min_leaderboard_candidate_runs", 2)))
        control_runs = max(1, int(raw.get("min_leaderboard_control_runs", 2)))
        confidence_se = max(0.0, float(raw.get("leaderboard_confidence_se", 1.0)))
        return cls(
            experiment_id=str(raw["experiment_id"]),
            seeds=seeds,
            contestants=contestants,
            weights=weights,
            promotion=PromotionRules.from_dict(raw.get("promotion")),
            timeout_seconds=max(1, int(raw.get("timeout_seconds", 7200))),
            max_parallel=max(1, int(raw.get("max_parallel", 1))),
            split_salt=str(raw.get("split_salt", "arcangel-v013")),
            leaderboard_control_id=(
                str(raw["leaderboard_control_id"])
                if raw.get("leaderboard_control_id")
                else None
            ),
            min_leaderboard_delta=float(raw.get("min_leaderboard_delta", 0.0)),
            min_leaderboard_candidate_runs=candidate_runs,
            min_leaderboard_control_runs=control_runs,
            leaderboard_confidence_se=confidence_se,
            require_leaderboard_artifact_hash=bool(
                raw.get("require_leaderboard_artifact_hash", True)
            ),
        )


@dataclass(slots=True)
class ArenaResult:
    contestant_id: str
    split: str
    seed: int
    metrics: dict[str, float]
    status: str = "ok"
    run_id: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        self.metrics = {str(key): float(value) for key, value in self.metrics.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArenaResult":
        return cls(
            contestant_id=str(data["contestant_id"]),
            split=str(data["split"]),
            seed=int(data["seed"]),
            metrics=dict(data.get("metrics", {})),
            status=str(data.get("status", "ok")),
            run_id=str(data.get("run_id", "")),
            source=str(data.get("source", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contestant_id": self.contestant_id,
            "split": self.split,
            "seed": self.seed,
            "metrics": self.metrics,
            "status": self.status,
            "run_id": self.run_id,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PlannedRun:
    contestant_id: str
    split: str
    seed: int
    command: tuple[str, ...]
    result_path: str

    @property
    def run_key(self) -> str:
        return f"{self.contestant_id}:{self.split}:{self.seed}"
