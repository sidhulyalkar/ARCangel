from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.research_agents import ResearchSwarm
from arc3lab.arena.schema import ArenaManifest


def build_context(
    repo_root: Path,
    include_list: Path,
    scorecard: dict[str, object],
    *,
    max_chars: int,
) -> str:
    sections = ["# CURRENT ARENA SCORECARD\n" + json.dumps(scorecard, indent=2)]
    used = len(sections[0])
    for line in include_list.read_text().splitlines():
        rel = line.strip()
        if not rel or rel.startswith("#") or "blind" in rel.lower():
            continue
        path = repo_root / rel
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        section = f"\n\n# FILE: {rel}\n{text[:remaining]}"
        sections.append(section)
        used += len(section)
    return "".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run independent frontier-model ARC research roles")
    ap.add_argument("--providers", required=True)
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--include-list", default="configs/swarm-packet-files.txt")
    ap.add_argument("--output-dir", default="artifacts/arena/v013/proposals/round1")
    ap.add_argument("--max-context-chars", type=int, default=120000)
    ap.add_argument("--max-requests", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    manifest = ArenaManifest.load(args.manifest)
    swarm = ResearchSwarm.load(args.providers, max_workers=args.max_workers)
    calls = swarm.plan()
    if args.plan_only:
        print(
            json.dumps(
                [
                    {"provider_id": call.provider_id, "role_id": call.role_id, "model": call.model}
                    for call in calls[: args.max_requests]
                ],
                indent=2,
            )
        )
        return 0

    lab = ArenaOrchestrator(manifest, args.arena_root)
    scorecard = lab.scorecard(include_blind=False)
    context = build_context(
        Path(args.repo_root),
        Path(args.include_list),
        scorecard,
        max_chars=args.max_context_chars,
    )
    proposals = swarm.run_independent_round(
        experiment_id=manifest.experiment_id,
        context=context,
        output_dir=args.output_dir,
        max_requests=args.max_requests,
    )
    summary = {
        "experiment_id": manifest.experiment_id,
        "requested": min(len(calls), args.max_requests),
        "valid": sum(proposal.valid for proposal in proposals),
        "invalid": sum(not proposal.valid for proposal in proposals),
        "proposal_files": [f"{proposal.provider_id}__{proposal.role_id}.json" for proposal in proposals],
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
