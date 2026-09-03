from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from arc3lab.arena.leaderboard import artifact_evidence, compare_artifacts
from arc3lab.arena.ledger import ResultLedger
from arc3lab.arena.schema import ArenaManifest, ArenaResult, ContestantSpec, PlannedRun
from arc3lab.arena.scoring import aggregate_results, promotion_decision, rank_split


class ArenaOrchestrator:
    """Plan, execute, ingest, score, and promote ARCangel research contestants."""

    def __init__(self, manifest: ArenaManifest, root: str | Path) -> None:
        self.manifest = manifest
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ResultLedger(self.root / "ledger.jsonl")

    def _contestant(self, contestant_id: str) -> ContestantSpec:
        for contestant in self.manifest.contestants:
            if contestant.contestant_id == contestant_id:
                return contestant
        raise KeyError(contestant_id)

    def _planned_run(
        self,
        contestant: ContestantSpec,
        *,
        command: tuple[str, ...],
        split: str,
        seed: int,
        private_registry: str = "",
    ) -> PlannedRun:
        result_path = self.results_dir / f"{contestant.contestant_id}__{split}__{seed}.json"
        mapping = {
            "contestant": contestant.contestant_id,
            "split": split,
            "seed": str(seed),
            "result": str(result_path),
            "arena_root": str(self.root),
            "private_registry": private_registry,
        }
        argv = tuple(token.format(**mapping) for token in command)
        return PlannedRun(
            contestant_id=contestant.contestant_id,
            split=split,
            seed=seed,
            command=argv,
            result_path=str(result_path),
        )

    def plan(
        self,
        *,
        splits: Iterable[str] = ("dev", "validation"),
        include_completed: bool = False,
    ) -> list[PlannedRun]:
        completed = self.ledger.run_keys()
        runs: list[PlannedRun] = []
        for split in splits:
            if split == "blind":
                raise ValueError("BLIND is judge-owned; use plan_blind() with the private registry")
            if split == "kaggle":
                raise ValueError("Kaggle evidence is external; use record_kaggle_score()")
            if split not in {"dev", "validation"}:
                raise ValueError(f"unsupported split: {split}")
        for contestant in self.manifest.contestants:
            if not contestant.enabled or not contestant.command:
                continue
            for split in splits:
                for seed in self.manifest.seeds:
                    key = (contestant.contestant_id, split, seed)
                    if not include_completed and key in completed:
                        continue
                    runs.append(
                        self._planned_run(
                            contestant,
                            command=contestant.command,
                            split=split,
                            seed=seed,
                        )
                    )
        return runs

    def plan_blind(
        self,
        *,
        private_registry: str | Path,
        include_completed: bool = False,
    ) -> list[PlannedRun]:
        private_path = Path(private_registry)
        if not private_path.exists():
            raise FileNotFoundError(private_path)
        aggregates = aggregate_results(self.ledger.read(), self.manifest)
        promoted = self._internal_promoted(aggregates)
        target_ids = set(promoted)
        for contestant_id in promoted:
            control_id = self._contestant(contestant_id).control_id
            if control_id:
                target_ids.add(control_id)
        completed = self.ledger.run_keys()
        runs: list[PlannedRun] = []
        for contestant_id in sorted(target_ids):
            contestant = self._contestant(contestant_id)
            if not contestant.judge_command:
                continue
            for seed in self.manifest.seeds:
                key = (contestant_id, "blind", seed)
                if not include_completed and key in completed:
                    continue
                runs.append(
                    self._planned_run(
                        contestant,
                        command=contestant.judge_command,
                        split="blind",
                        seed=seed,
                        private_registry=str(private_path),
                    )
                )
        return runs

    def execute(self, run: PlannedRun) -> ArenaResult:
        result_path = Path(run.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists():
            result_path.unlink()
        env = os.environ.copy()
        env.update(
            {
                "ARCANGEL_CONTESTANT_ID": run.contestant_id,
                "ARCANGEL_SPLIT": run.split,
                "ARCANGEL_SEED": str(run.seed),
                "ARCANGEL_RESULT_PATH": str(result_path),
                "ARCANGEL_ARENA_ROOT": str(self.root),
            }
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(run.command),
                check=False,
                timeout=self.manifest.timeout_seconds,
                env=env,
                capture_output=True,
                text=True,
            )
            elapsed = time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            result = ArenaResult(
                contestant_id=run.contestant_id,
                split=run.split,
                seed=run.seed,
                status="timeout",
                metrics={"failure_rate": 1.0, "timeout_fraction": 1.0},
                metadata={"command": list(run.command), "timeout_seconds": exc.timeout},
            )
            self.ledger.append(result)
            return result

        if result_path.exists():
            payload = json.loads(result_path.read_text())
            result = ArenaResult.from_dict(payload)
            if result.contestant_id != run.contestant_id:
                raise ValueError("result contestant_id does not match planned run")
            if result.split != run.split or result.seed != run.seed:
                raise ValueError("result split/seed does not match planned run")
        else:
            status = "ok" if completed.returncode == 0 else "failed"
            result = ArenaResult(
                contestant_id=run.contestant_id,
                split=run.split,
                seed=run.seed,
                status=status,
                metrics={"failure_rate": 0.0 if status == "ok" else 1.0},
            )
        result.metadata.setdefault("wall_time_s", elapsed)
        result.metadata.setdefault("returncode", completed.returncode)
        result.metadata.setdefault("command", list(run.command))
        result.metadata.setdefault("stdout_tail", completed.stdout[-4000:])
        result.metadata.setdefault("stderr_tail", completed.stderr[-4000:])
        self.ledger.append(result)
        return result

    def run_all(self, *, splits: Iterable[str] = ("dev", "validation")) -> list[ArenaResult]:
        results: list[ArenaResult] = []
        for run in self.plan(splits=splits):
            results.append(self.execute(run))
        return results

    def run_blind(self, *, private_registry: str | Path) -> list[ArenaResult]:
        results: list[ArenaResult] = []
        for run in self.plan_blind(private_registry=private_registry):
            results.append(self.execute(run))
        return results

    def import_result(self, path: str | Path) -> ArenaResult:
        result = ArenaResult.from_dict(json.loads(Path(path).read_text()))
        self._contestant(result.contestant_id)
        self.ledger.append(result)
        return result

    def record_kaggle_score(
        self,
        *,
        contestant_id: str,
        score: float,
        seed: int,
        source: str,
        artifact_sha256: str = "",
        runtime_seconds: float | None = None,
    ) -> ArenaResult:
        self._contestant(contestant_id)
        artifact_sha256 = artifact_sha256.strip().lower()
        if self.manifest.require_leaderboard_artifact_hash and not artifact_sha256:
            raise ValueError("Kaggle score receipts require the exact notebook artifact SHA-256")
        if not source.strip():
            raise ValueError("Kaggle score receipts require an external source/provenance label")
        if (contestant_id, "kaggle", seed) in self.ledger.run_keys():
            raise ValueError(f"duplicate Kaggle run key for {contestant_id} seed={seed}")
        if contestant_id != self.manifest.leaderboard_control_id:
            ready = {row["contestant_id"] for row in self.kaggle_ready_queue()}
            if contestant_id not in ready:
                raise ValueError(
                    f"{contestant_id} has not passed internal promotion plus private BLIND qualification"
                )
        metadata: dict[str, object] = {
            "artifact_sha256": artifact_sha256,
            "external_source": source,
        }
        if runtime_seconds is not None:
            metadata["runtime_seconds"] = float(runtime_seconds)
        result = ArenaResult(
            contestant_id=contestant_id,
            split="kaggle",
            seed=seed,
            metrics={"official_score": float(score)},
            status="ok",
            source=source,
            metadata=metadata,
        )
        self.ledger.append(result)
        return result

    def _internal_promoted(self, aggregates: dict) -> set[str]:
        promoted: set[str] = set()
        for contestant in self.manifest.contestants:
            if not contestant.control_id:
                continue
            decision = promotion_decision(contestant, aggregates, self.manifest)
            if decision.promoted:
                promoted.add(contestant.contestant_id)
        return promoted

    def kaggle_ready_queue(self) -> list[dict[str, object]]:
        """Require private BLIND evidence before an internal winner may consume a Kaggle slot."""
        aggregates = aggregate_results(self.ledger.read(), self.manifest)
        rules = self.manifest.promotion
        queue: list[dict[str, object]] = []
        for contestant_id in sorted(self._internal_promoted(aggregates)):
            contestant = self._contestant(contestant_id)
            if not contestant.control_id:
                continue
            candidate = aggregates.get((contestant_id, "blind"))
            control = aggregates.get((contestant.control_id, "blind"))
            if candidate is None or control is None:
                continue
            if candidate.runs < rules.min_blind_runs or control.runs < rules.min_blind_runs:
                continue
            if candidate.emergency_fraction > rules.max_emergency_fraction:
                continue
            if candidate.failure_rate > rules.max_failure_rate:
                continue
            delta = candidate.robust_score - control.robust_score
            if delta < rules.min_blind_delta:
                continue
            queue.append(
                {
                    "contestant_id": contestant_id,
                    "control_id": contestant.control_id,
                    "blind_delta": delta,
                    "candidate_runs": candidate.runs,
                    "control_runs": control.runs,
                }
            )
        return sorted(queue, key=lambda item: float(item["blind_delta"]), reverse=True)

    def leaderboard_evidence(self) -> dict[str, object]:
        results = self.ledger.read()
        control_id = self.manifest.leaderboard_control_id
        control_rows = (
            artifact_evidence(
                results,
                control_id,
                confidence_se=self.manifest.leaderboard_confidence_se,
                require_hash=self.manifest.require_leaderboard_artifact_hash,
            )
            if control_id
            else []
        )
        ready = {row["contestant_id"] for row in self.kaggle_ready_queue()}
        candidate_rows = {
            contestant_id: [row.to_dict() for row in artifact_evidence(
                results,
                contestant_id,
                confidence_se=self.manifest.leaderboard_confidence_se,
                require_hash=self.manifest.require_leaderboard_artifact_hash,
            )]
            for contestant_id in sorted(ready)
        }
        return {
            "control_id": control_id,
            "required_control_runs": self.manifest.min_leaderboard_control_runs,
            "required_candidate_runs": self.manifest.min_leaderboard_candidate_runs,
            "confidence_se": self.manifest.leaderboard_confidence_se,
            "control_artifacts": [row.to_dict() for row in control_rows],
            "candidate_artifacts": candidate_rows,
        }

    def leaderboard_queue(self) -> list[dict[str, object]]:
        """Use repeated exact-artifact Kaggle evidence, not a single noisy public score."""
        results = self.ledger.read()
        control_id = self.manifest.leaderboard_control_id
        if not control_id:
            return []
        control_groups = artifact_evidence(
            results,
            control_id,
            confidence_se=self.manifest.leaderboard_confidence_se,
            require_hash=self.manifest.require_leaderboard_artifact_hash,
        )
        qualified_controls = [
            row for row in control_groups if row.runs >= self.manifest.min_leaderboard_control_runs
        ]
        # Multiple qualified control hashes make the baseline ambiguous. Give each exact
        # control notebook its own contestant ID rather than cherry-picking a control run.
        if len(qualified_controls) != 1:
            return []
        control = qualified_controls[0]
        ready = {row["contestant_id"] for row in self.kaggle_ready_queue()}
        queue: list[dict[str, object]] = []
        for contestant_id in sorted(ready):
            for candidate in artifact_evidence(
                results,
                contestant_id,
                confidence_se=self.manifest.leaderboard_confidence_se,
                require_hash=self.manifest.require_leaderboard_artifact_hash,
            ):
                comparison = compare_artifacts(
                    candidate,
                    control,
                    min_candidate_runs=self.manifest.min_leaderboard_candidate_runs,
                    min_control_runs=self.manifest.min_leaderboard_control_runs,
                    min_delta=self.manifest.min_leaderboard_delta,
                    confidence_se=self.manifest.leaderboard_confidence_se,
                )
                if comparison.ready:
                    queue.append(comparison.to_dict())
        return sorted(
            queue,
            key=lambda item: (float(item["delta_lower_bound"]), float(item["mean_delta"])),
            reverse=True,
        )

    def scorecard(self, *, include_blind: bool = False) -> dict[str, object]:
        results = self.ledger.read()
        if not include_blind:
            results = [row for row in results if row.split != "blind"]
        aggregates = aggregate_results(results, self.manifest)
        split_rankings = {
            split: [asdict(item) for item in rank_split(aggregates, split)]
            for split in ("dev", "validation")
            if any(row.split == split for row in results)
        }
        if include_blind and any(row.split == "blind" for row in results):
            split_rankings["blind"] = [asdict(item) for item in rank_split(aggregates, "blind")]
        decisions = []
        for contestant in self.manifest.contestants:
            if contestant.control_id:
                decisions.append(
                    asdict(promotion_decision(contestant, aggregates, self.manifest))
                )
        return {
            "experiment_id": self.manifest.experiment_id,
            "result_count": len(results),
            "rankings": split_rankings,
            "promotion_decisions": decisions,
            "leaderboard_control_id": self.manifest.leaderboard_control_id,
            "leaderboard_evidence": self.leaderboard_evidence(),
            "leaderboard_queue": self.leaderboard_queue(),
        }

    def write_scorecard(self, path: str | Path, *, include_blind: bool = False) -> None:
        Path(path).write_text(
            json.dumps(self.scorecard(include_blind=include_blind), indent=2) + "\n"
        )

    def promotion_queue(self) -> list[str]:
        aggregates = aggregate_results(self.ledger.read(), self.manifest)
        return sorted(self._internal_promoted(aggregates))

    def describe_plan(self, runs: Iterable[PlannedRun]) -> str:
        lines = []
        for run in runs:
            lines.append(f"{run.run_key}: {shlex.join(run.command)} -> {run.result_path}")
        return "\n".join(lines)
