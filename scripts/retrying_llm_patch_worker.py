#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


TRANSIENT_MARKERS = (
    "remotedisconnected",
    "connection aborted",
    "connectionerror",
    "read timed out",
    "connecttimeout",
    "timeout",
    "rate limited",
    "rate_limited",
    "status code 429",
    "status code 500",
    "status code 502",
    "status code 503",
    "status code 504",
    "temporarily unavailable",
    "server error",
)


def _argument_value(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
    except ValueError as exc:
        raise ValueError(f"missing required forwarded argument {name}") from exc
    if index + 1 >= len(argv):
        raise ValueError(f"missing value for forwarded argument {name}")
    return argv[index + 1]


def _clean(worktree: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _transient(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TRANSIENT_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Retry llm_patch_worker only when transport failed before any worktree mutation"
    )
    ap.add_argument("--transient-attempts", type=int, default=3)
    ap.add_argument("--retry-backoff-seconds", type=float, default=3.0)
    known, forwarded = ap.parse_known_args()

    worktree = Path(_argument_value(forwarded, "--worktree")).resolve()
    if not worktree.exists():
        raise FileNotFoundError(worktree)
    attempts = max(1, int(known.transient_attempts))
    history: list[dict[str, object]] = []

    for attempt in range(1, attempts + 1):
        command = [sys.executable, "scripts/llm_patch_worker.py", *forwarded]
        completed = subprocess.run(
            command,
            cwd=worktree,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        combined = f"{completed.stdout}\n{completed.stderr}"
        clean = _clean(worktree)
        transient = completed.returncode != 0 and clean and _transient(combined)
        history.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "clean_worktree": clean,
                "transient_transport_failure": transient,
            }
        )
        if completed.returncode == 0:
            print(json.dumps({"transport_retry_history": history}, indent=2))
            return 0
        if not transient or attempt >= attempts:
            print(json.dumps({"transport_retry_history": history}, indent=2), file=sys.stderr)
            return completed.returncode
        time.sleep(max(0.0, float(known.retry_backoff_seconds)) * attempt)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
