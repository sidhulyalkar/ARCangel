from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from arc3lab.arena.schema import ArenaResult


class ResultLedger:
    """Append-only JSONL ledger for experiment receipts.

    The ledger is intentionally simple so results produced on Kaggle, CI, local machines,
    or external agents can all be imported without trusting a mutable database.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: ArenaResult) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")

    def extend(self, results: Iterable[ArenaResult]) -> None:
        for result in results:
            self.append(result)

    def read(self) -> list[ArenaResult]:
        if not self.path.exists():
            return []
        rows: list[ArenaResult] = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(ArenaResult.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid arena ledger line {line_number}: {exc}") from exc
        return rows

    def run_keys(self) -> set[tuple[str, str, int]]:
        return {(row.contestant_id, row.split, row.seed) for row in self.read()}
