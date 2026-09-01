from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

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

    def plan(
        self,
        *,
        splits: Iterable[str] = ("dev", "validation"),
        include_completed: bool = False,
    ) -> list[PlannedRun]:
        completed = self.ledger.run_keys()
        runs: list[PlannedRun] = []
        for contestant in self.manifest.contestants:
            if not contestant.enabled or not contestant.command:
                continue
            for split in splits:
                if split not in {"dev", "validation", "blind", "kaggle"}:
                    raise ValueError(f"unsupported split: {split}")
                for seed in self.manifest.seeds:
                    key = (contestant.contestant_id, split, seed)
                    if not include_completed and key in completed:
                        continue
                    result_path = self.results_dir / (
                        f"{contestant.contestant_id}__{split}__{seed}.json"
                    )
                    mapping = {
                        "contestant": contestant.contestant_id,
                        "split": split,
                        "seed": str(seed),
                        "result": str(result_path),
                        "arena_root": str(self.root),
                    }
                    argv = tuple(token.format(**mapping) for token in contestant.command)
                    runs.append(
                        PlannedRun(
                            contestant_id=contestant.contestant_id,
                            split=split,
                            seed=seed,
                            command=argv,
                            result_path=str(result_path),
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
        # Serial execution is the default because local GPU contestants often contend for one device.
        results: list[ArenaResult] = []
        for run in self.plan(splits=splits):
            results.append(self.execute(run))
        return results

    def import_result(self, path: str | Path) -> ArenaResult:
        result = ArenaResult.from_dict(json.loads(Path(path).read_text()))
        self._contestant(result.contestant_id)
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

    def leaderboard_queue(self) -> list[dict[str, object]]:
        """Nominate only internally promoted contestants that beat the public Kaggle control."""
        aggregates = aggregate_results(self.ledger.read(), self.manifest)
        control_id = self.manifest.leaderboard_control_id
        if not control_id:
            return []
        control = aggregates.get((control_id, "kaggle"))
        if control is None:
            return []
        internally_promoted = self._internal_promoted(aggregates)
        queue: list[dict[str, object]] = []
        for contestant_id in sorted(internally_promoted):
            row = aggregates.get((contestant_id, "kaggle"))
            if row is None:
                continue
            delta = row.robust_score - control.robust_score
            if delta < self.manifest.min_leaderboard_delta:
                continue
            queue.append(
                {
                    "contestant_id": contestant_id,
                    "control_id": control_id,
                    "kaggle_delta": delta,
                    "contestant_score": row.robust_score,
                    "control_score": control.robust_score,
                }
            )
        return sorted(queue, key=lambda item: float(item["kaggle_delta"]), reverse=True)

    def scorecard(self, *, include_blind: bool = False) -> dict[str, object]:
        results = self.ledger.read()
        if not include_blind:
            results = [row for row in results if row.split != "blind"]
        aggregates = aggregate_results(results, self.manifest)
        split_rankings = {
            split: [asdict(item) for item in rank_split(aggregates, split)]
            for split in ("dev", "validation", "kaggle")
            if any(row.split == split for row in results)
        }
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
