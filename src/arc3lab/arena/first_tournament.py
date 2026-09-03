from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import PlannedRun
from arc3lab.arena.scoring import AggregateScore, aggregate_results


CONTROL_ID = "B-coding-minimal"
INITIAL_CONTESTANTS = (
    CONTROL_ID,
    "C-v011-reflective",
    "D-v012-evidence-first",
    "E-v012-lite",
)


@dataclass(frozen=True, slots=True)
class TournamentStage:
    name: str
    split: str
    seeds: tuple[int, ...]
    contestant_ids: tuple[str, ...]
    runs: tuple[PlannedRun, ...]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["runs"] = [
            {
                "run_key": run.run_key,
                "command": list(run.command),
                "result_path": run.result_path,
            }
            for run in self.runs
        ]
        return payload


class FirstTournamentDirector:
    """Adaptive B/C/D/E tournament that spends repeat compute only on survivors.

    Stage 1 screens every architecture on the same DEV seed. Stage 2 gives surviving
    challengers one VALIDATION seed. Stage 3 repeats VALIDATION only for the strongest
    credible challengers and the common control. DEV repeats are requested only when a
    finalist's one-seed DEV delta is close enough to the promotion boundary that the
    result could plausibly be noise.
    """

    def __init__(
        self,
        lab: ArenaOrchestrator,
        *,
        control_id: str = CONTROL_ID,
        initial_contestants: Iterable[str] = INITIAL_CONTESTANTS,
        screen_min_delta: float = -0.08,
        validation_min_delta: float = -0.05,
        max_repeat_challengers: int = 2,
        dev_repeat_margin: float = 0.03,
    ) -> None:
        self.lab = lab
        self.control_id = control_id
        self.initial_contestants = tuple(initial_contestants)
        self.screen_min_delta = float(screen_min_delta)
        self.validation_min_delta = float(validation_min_delta)
        self.max_repeat_challengers = max(1, int(max_repeat_challengers))
        self.dev_repeat_margin = max(0.0, float(dev_repeat_margin))
        known = {row.contestant_id for row in lab.manifest.contestants}
        missing = [cid for cid in self.initial_contestants if cid not in known]
        if missing:
            raise ValueError(f"unknown first-tournament contestants: {missing}")
        if self.control_id not in self.initial_contestants:
            raise ValueError("control must be part of the initial tournament")
        if len(lab.manifest.seeds) < 2:
            raise ValueError("first tournament requires at least two configured seeds")

    @property
    def primary_seed(self) -> int:
        return int(self.lab.manifest.seeds[0])

    @property
    def repeat_seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.lab.manifest.seeds[1:])

    def _aggregates(self) -> dict[tuple[str, str], AggregateScore]:
        return aggregate_results(self.lab.ledger.read(), self.lab.manifest)

    def _run(
        self,
        contestant_id: str,
        *,
        split: str,
        seed: int,
    ) -> PlannedRun:
        contestant = next(
            row for row in self.lab.manifest.contestants if row.contestant_id == contestant_id
        )
        result_path = self.lab.results_dir / f"{contestant_id}__{split}__{seed}.json"
        mapping = {
            "contestant": contestant_id,
            "split": split,
            "seed": str(seed),
            "result": str(result_path),
            "arena_root": str(self.lab.root),
            "private_registry": "",
        }
        return PlannedRun(
            contestant_id=contestant_id,
            split=split,
            seed=seed,
            command=tuple(token.format(**mapping) for token in contestant.command),
            result_path=str(result_path),
        )

    def _missing_runs(
        self,
        contestant_ids: Iterable[str],
        *,
        split: str,
        seeds: Iterable[int],
    ) -> tuple[PlannedRun, ...]:
        completed = self.lab.ledger.run_keys()
        runs: list[PlannedRun] = []
        for contestant_id in contestant_ids:
            for seed in seeds:
                if (contestant_id, split, int(seed)) in completed:
                    continue
                runs.append(self._run(contestant_id, split=split, seed=int(seed)))
        return tuple(runs)

    @staticmethod
    def _credible(row: AggregateScore | None, max_failure: float, max_emergency: float) -> bool:
        if row is None:
            return False
        return row.failure_rate <= max_failure and row.emergency_fraction <= max_emergency

    def screen_survivors(self) -> tuple[str, ...]:
        aggregates = self._aggregates()
        control = aggregates.get((self.control_id, "dev"))
        if control is None:
            return tuple(cid for cid in self.initial_contestants if cid != self.control_id)
        rules = self.lab.manifest.promotion
        survivors: list[tuple[float, str]] = []
        for contestant_id in self.initial_contestants:
            if contestant_id == self.control_id:
                continue
            row = aggregates.get((contestant_id, "dev"))
            if not self._credible(row, rules.max_failure_rate, rules.max_emergency_fraction):
                continue
            assert row is not None
            delta = row.robust_score - control.robust_score
            if delta >= self.screen_min_delta:
                survivors.append((delta, contestant_id))
        survivors.sort(reverse=True)
        return tuple(contestant_id for _, contestant_id in survivors)

    def repeat_finalists(self) -> tuple[str, ...]:
        aggregates = self._aggregates()
        control = aggregates.get((self.control_id, "validation"))
        if control is None:
            return ()
        rules = self.lab.manifest.promotion
        finalists: list[tuple[float, str]] = []
        for contestant_id in self.screen_survivors():
            row = aggregates.get((contestant_id, "validation"))
            if not self._credible(row, rules.max_failure_rate, rules.max_emergency_fraction):
                continue
            assert row is not None
            delta = row.robust_score - control.robust_score
            if delta >= self.validation_min_delta:
                finalists.append((delta, contestant_id))
        finalists.sort(reverse=True)
        return tuple(
            contestant_id for _, contestant_id in finalists[: self.max_repeat_challengers]
        )

    def dev_repeat_targets(self) -> tuple[str, ...]:
        """Repeat DEV only when one-seed evidence sits near the promotion boundary."""
        aggregates = self._aggregates()
        control = aggregates.get((self.control_id, "dev"))
        if control is None:
            return ()
        threshold = self.lab.manifest.promotion.min_dev_delta
        targets: list[str] = []
        for contestant_id in self.repeat_finalists():
            row = aggregates.get((contestant_id, "dev"))
            if row is None:
                continue
            delta = row.robust_score - control.robust_score
            boundary = threshold + self.dev_repeat_margin
            if delta + 1e-12 < boundary:
                targets.append(contestant_id)
        return tuple(targets)

    def next_stage(self) -> TournamentStage | None:
        primary = self.primary_seed
        screen_runs = self._missing_runs(
            self.initial_contestants,
            split="dev",
            seeds=(primary,),
        )
        if screen_runs:
            return TournamentStage(
                name="dev-screen",
                split="dev",
                seeds=(primary,),
                contestant_ids=self.initial_contestants,
                runs=screen_runs,
                rationale="One common-seed DEV screen for B/C/D/E before repeat compute is granted.",
            )

        survivors = self.screen_survivors()
        if not survivors:
            return None
        validation_ids = (self.control_id, *survivors)
        first_validation = self._missing_runs(
            validation_ids,
            split="validation",
            seeds=(primary,),
        )
        if first_validation:
            return TournamentStage(
                name="validation-screen",
                split="validation",
                seeds=(primary,),
                contestant_ids=validation_ids,
                runs=first_validation,
                rationale="Only DEV survivors receive the first held-out validation run.",
            )

        finalists = self.repeat_finalists()
        if not finalists:
            return None
        repeat_ids = (self.control_id, *finalists)
        validation_repeats = self._missing_runs(
            repeat_ids,
            split="validation",
            seeds=self.repeat_seeds,
        )
        if validation_repeats:
            return TournamentStage(
                name="validation-repeat",
                split="validation",
                seeds=self.repeat_seeds,
                contestant_ids=repeat_ids,
                runs=validation_repeats,
                rationale=(
                    "Repeat only the common control and strongest credible challengers to estimate "
                    "validation uncertainty."
                ),
            )

        dev_targets = self.dev_repeat_targets()
        if dev_targets:
            dev_ids = (self.control_id, *dev_targets)
            dev_repeats = self._missing_runs(
                dev_ids,
                split="dev",
                seeds=self.repeat_seeds,
            )
            if dev_repeats:
                return TournamentStage(
                    name="dev-confirmation",
                    split="dev",
                    seeds=self.repeat_seeds,
                    contestant_ids=dev_ids,
                    runs=dev_repeats,
                    rationale=(
                        "A finalist is close to the DEV regression boundary, so repeat DEV only "
                        "where extra evidence can change promotion."
                    ),
                )
        return None

    def status(self) -> dict[str, object]:
        stage = self.next_stage()
        aggregates = self._aggregates()
        return {
            "control_id": self.control_id,
            "primary_seed": self.primary_seed,
            "repeat_seeds": list(self.repeat_seeds),
            "screen_survivors": list(self.screen_survivors()),
            "repeat_finalists": list(self.repeat_finalists()),
            "dev_repeat_targets": list(self.dev_repeat_targets()),
            "next_stage": stage.to_dict() if stage else None,
            "promotion_queue": self.lab.promotion_queue(),
            "kaggle_ready_queue": self.lab.kaggle_ready_queue(),
            "observed_cells": sorted(f"{cid}:{split}" for cid, split in aggregates),
        }

    def write_status(self, path: str | Path) -> None:
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.status(), indent=2) + "\n")
