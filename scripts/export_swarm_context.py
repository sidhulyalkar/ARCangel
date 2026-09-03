from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.research_context import build_research_scorecard
from arc3lab.arena.research_packet import ResearchPacketBuilder
from arc3lab.arena.schema import ArenaManifest


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def include_paths(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def file_hashes(repo_root: Path, paths: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    root = repo_root.resolve()
    for rel in paths:
        path = (repo_root / rel).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"context path escapes repository: {rel}")
        if not path.exists() or not path.is_file():
            continue
        if "blind" in path.name.lower() or "/blind/" in path.as_posix().lower():
            continue
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export one deterministic DEV/VALIDATION-only context capsule for remote swarm research"
    )
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--include-list", default="configs/swarm-packet-files.txt")
    ap.add_argument(
        "--output",
        default="artifacts/arena/v013/swarm-context-capsule.tar.gz",
    )
    ap.add_argument(
        "--receipt",
        default="artifacts/arena/v013/swarm-context-capsule.json",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest = ArenaManifest.load(repo_root / args.manifest)
    lab = ArenaOrchestrator(manifest, args.arena_root)
    scorecard = build_research_scorecard(lab)
    selected_paths = include_paths(repo_root / args.include_list)
    output = Path(args.output)
    digest = ResearchPacketBuilder(repo_root).build(
        output,
        experiment_id=manifest.experiment_id,
        scorecard=scorecard,
        include_paths=selected_paths,
    )
    receipt = {
        "format": "arcangel-swarm-context-v1",
        "experiment_id": manifest.experiment_id,
        "git_head": git_head(repo_root),
        "capsule": str(output),
        "capsule_sha256": digest,
        "research_evidence_scope": scorecard["evidence_scope"],
        "development_result_count": scorecard["result_count"],
        "included_file_sha256": file_hashes(repo_root, selected_paths),
        "forbidden_dynamic_evidence": ["blind", "kaggle", "leaderboard"],
        "status": "SAFE_RESEARCH_CONTEXT_EXPORTED",
    }
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
