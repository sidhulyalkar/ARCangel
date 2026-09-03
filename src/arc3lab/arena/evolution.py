from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from arc3lab.arena.research_agents import ResearchProposal


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:64] or "proposal"


class ProposalTournament:
    """Turn independent research proposals into a diversity-preserving battle queue."""

    def __init__(self, proposals: Iterable[ResearchProposal]) -> None:
        self.proposals = tuple(proposals)

    @classmethod
    def load(cls, directory: str | Path) -> "ProposalTournament":
        proposals: list[ResearchProposal] = []
        for path in sorted(Path(directory).glob("*.json")):
            data = json.loads(path.read_text())
            proposals.append(
                ResearchProposal(
                    provider_id=str(data.get("provider_id", "")),
                    role_id=str(data.get("role_id", "")),
                    hypothesis=str(data.get("hypothesis", "")),
                    experiment=str(data.get("experiment", "")),
                    target_metric=str(data.get("target_metric", "")),
                    split=str(data.get("split", "")),
                    falsifier=str(data.get("falsifier", "")),
                    implementation=str(data.get("implementation", "")),
                    failure_mode=str(data.get("failure_mode", "")),
                    target_profile=str(data.get("target_profile", "")),
                    control_profile=str(data.get("control_profile", "")),
                    raw_text=str(data.get("raw_text", "")),
                    valid=bool(data.get("valid", False)),
                )
            )
        return cls(proposals)

    def eligible(self) -> list[ResearchProposal]:
        # Research agents never get authority to request BLIND or Kaggle as invention splits.
        return [
            proposal
            for proposal in self.proposals
            if proposal.valid and proposal.split.lower() in {"dev", "validation"}
        ]

    def select(self, max_proposals: int = 10) -> list[ResearchProposal]:
        """Round-robin by role so one provider cannot collapse architectural diversity."""
        by_role: dict[str, list[ResearchProposal]] = defaultdict(list)
        seen_hypotheses: set[str] = set()
        for proposal in self.eligible():
            normalized = " ".join(proposal.hypothesis.lower().split())
            if normalized in seen_hypotheses:
                continue
            seen_hypotheses.add(normalized)
            by_role[proposal.role_id].append(proposal)
        for proposals in by_role.values():
            proposals.sort(key=lambda item: (item.provider_id, item.hypothesis))

        selected: list[ResearchProposal] = []
        role_ids = sorted(by_role)
        cursor = 0
        while role_ids and len(selected) < max(0, int(max_proposals)):
            role_id = role_ids[cursor % len(role_ids)]
            rows = by_role[role_id]
            if rows:
                selected.append(rows.pop(0))
            role_ids = [role for role in role_ids if by_role[role]]
            cursor += 1
        return selected

    def battle_plan(self, max_proposals: int = 10) -> dict[str, Any]:
        rows = []
        for index, proposal in enumerate(self.select(max_proposals), start=1):
            key = f"R1-{index:02d}-{_slug(proposal.role_id)}-{_slug(proposal.provider_id)}"
            rows.append(
                {
                    "proposal_id": key,
                    "provider_id": proposal.provider_id,
                    "role_id": proposal.role_id,
                    "hypothesis": proposal.hypothesis,
                    "experiment": proposal.experiment,
                    "target_metric": proposal.target_metric,
                    "selection_split": proposal.split,
                    "falsifier": proposal.falsifier,
                    "implementation": proposal.implementation,
                    "failure_mode": proposal.failure_mode,
                    "target_profile": proposal.target_profile,
                    "control_profile": proposal.control_profile,
                    "suggested_branch": f"experiment/{_slug(key)}",
                }
            )
        return {
            "phase": "independent-battle",
            "selected": rows,
            "selected_count": len(rows),
            "eligible_count": len(self.eligible()),
        }

    def exchange_brief(self, scorecard: dict[str, Any]) -> str:
        """Round-2 context is generated only after independent proposals exist."""
        validation = scorecard.get("rankings", {}).get("validation", [])
        decisions = scorecard.get("promotion_decisions", [])
        leader_lines = []
        for row in validation[:3]:
            leader_lines.append(
                f"- {row.get('contestant_id')}: robust={row.get('robust_score')} "
                f"mean={row.get('mean_score')}"
            )
        failure_lines = []
        for decision in decisions:
            if decision.get("promoted"):
                continue
            reasons = "; ".join(str(reason) for reason in decision.get("reasons", []))
            failure_lines.append(f"- {decision.get('contestant_id')}: {reasons}")
        peer_lines = [
            f"- [{proposal.role_id}/{proposal.provider_id}] {proposal.hypothesis}"
            for proposal in self.select(max_proposals=10)
        ]
        return "\n".join(
            [
                "# ROUND 2 EXCHANGE BRIEF",
                "Independent work is complete. You may now use peer discoveries, but measured results remain authoritative.",
                "",
                "## Validation leaders",
                *(leader_lines or ["- no validation results yet"]),
                "",
                "## Failed promotion gates",
                *(failure_lines or ["- no failed promotion records yet"]),
                "",
                "## Independent peer hypotheses",
                *(peer_lines or ["- no structurally valid proposals"]),
            ]
        ) + "\n"
