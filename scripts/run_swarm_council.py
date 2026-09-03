from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc3lab.arena.evolution import ProposalTournament
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.research_agents import ResearchSwarm
from arc3lab.arena.research_context import build_research_scorecard
from arc3lab.arena.schema import ArenaManifest
from arc3lab.arena.swarm_intelligence import ResearchReview, SwarmCouncil


def build_context(
    repo_root: Path,
    include_list: Path,
    scorecard: dict[str, object],
    *,
    max_chars: int,
) -> str:
    sections = ["# DEV/VALIDATION RESEARCH SCORECARD\n" + json.dumps(scorecard, indent=2)]
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


def load_reviews(directory: Path) -> list[ResearchReview]:
    rows: list[ResearchReview] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json")):
        rows.append(ResearchReview(**json.loads(path.read_text())))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run blinded cross-review and evidence-weighted ARCangel swarm selection"
    )
    ap.add_argument("--providers", required=True)
    ap.add_argument("--proposals", default="artifacts/arena/v013/proposals/round1")
    ap.add_argument("--reviews", default="artifacts/arena/v013/reviews/round1")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--include-list", default="configs/swarm-packet-files.txt")
    ap.add_argument("--max-context-chars", type=int, default=90000)
    ap.add_argument("--reviews-per-proposal", type=int, default=3)
    ap.add_argument("--min-reviews", type=int, default=2)
    ap.add_argument("--max-proposals", type=int, default=8)
    ap.add_argument("--max-requests", type=int, default=60)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--battle-plan", default="artifacts/arena/v013/swarm-battle-plan.json")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    manifest = ArenaManifest.load(args.manifest)
    tournament = ProposalTournament.load(args.proposals)
    proposals = list(tournament.proposals)
    swarm = ResearchSwarm.load(args.providers, max_workers=args.max_workers)
    council = SwarmCouncil(proposals)
    assignments = council.assign_reviews(
        swarm.plan(),
        reviews_per_proposal=args.reviews_per_proposal,
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "assignments": [
                        {
                            "proposal_key": row.proposal_key,
                            "reviewer_provider_id": row.reviewer_provider_id,
                            "reviewer_role_id": row.reviewer_role_id,
                        }
                        for row in assignments[: args.max_requests]
                    ],
                    "proposal_count": len(council.eligible_proposals()),
                },
                indent=2,
            )
        )
        return 0

    lab = ArenaOrchestrator(manifest, args.arena_root)
    scorecard = build_research_scorecard(lab)
    context = build_context(
        Path(args.repo_root),
        Path(args.include_list),
        scorecard,
        max_chars=args.max_context_chars,
    )
    review_dir = Path(args.reviews)
    swarm.run_review_round(
        assignments=assignments,
        proposals=proposals,
        experiment_id=manifest.experiment_id,
        context=context,
        output_dir=review_dir,
        max_requests=args.max_requests,
    )
    reviews = load_reviews(review_dir)
    judged = SwarmCouncil(proposals, reviews)
    battle = judged.battle_plan(
        max_proposals=args.max_proposals,
        min_reviews=args.min_reviews,
    )
    battle_path = Path(args.battle_plan)
    battle_path.parent.mkdir(parents=True, exist_ok=True)
    battle_path.write_text(json.dumps(battle, indent=2) + "\n")
    print(
        json.dumps(
            {
                "experiment_id": manifest.experiment_id,
                "reviews": len(reviews),
                "valid_reviews": sum(review.valid for review in reviews),
                "selected": battle["selected_count"],
                "research_evidence_scope": scorecard["evidence_scope"],
                "battle_plan": str(battle_path),
                "authority": battle["authority"],
            },
            indent=2,
        )
    )
    return 0 if battle["selected_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
