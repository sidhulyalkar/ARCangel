#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = (
    "workflow_dispatch",
    "[run-swarm-research]",
    ".github/swarm-research-trigger.json",
    "NVIDIA_API_KEY",
    "check_research_providers.py",
    "run_swarm_research_cycle.py",
    "export_swarm_worker_patches.py",
    "actions/upload-artifact@v4",
)

FORBIDDEN_EXECUTION = (
    "run_first_tournament.py",
    "run_augmented_blind.py",
    "run_arena_contestant.py",
    "run_swarm_experiment.py",
    "run_guarded_swarm_experiment.py",
    "run_portable_swarm_gpu_stage.py",
    "package_kaggle_ready.py",
    "package_promoted_swarm.py",
    "record-kaggle",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify GitHub swarm workflow stays research-only")
    ap.add_argument("path", nargs="?", default=".github/workflows/swarm-research.yml")
    args = ap.parse_args()
    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED if token not in text]
    forbidden = [token for token in FORBIDDEN_EXECUTION if token in text]
    if missing or forbidden:
        lines = []
        if missing:
            lines.append(f"missing required research workflow contracts: {missing}")
        if forbidden:
            lines.append(f"research workflow contains forbidden evaluator commands: {forbidden}")
        raise SystemExit("\n".join(lines))
    print(
        "SWARM RESEARCH WORKFLOW VERIFIED: explicit trigger only; NVIDIA agents may "
        "generate/review/code patches; Qwen arena, BLIND, packaging, and Kaggle remain outside."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
