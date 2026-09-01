from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return text[:72] or "particle"


def proposal_key(proposal: Any) -> str:
    return f"{proposal.provider_id}__{proposal.role_id}"


@dataclass(frozen=True, slots=True)
class ReviewAssignment:
    proposal_key: str
    reviewer_provider_id: str
    reviewer_role_id: str

    @property
    def key(self) -> str:
        return (
            f"{self.proposal_key}__reviewed-by__"
            f"{self.reviewer_provider_id}__{self.reviewer_role_id}"
        )


@dataclass(frozen=True, slots=True)
class ResearchReview:
    proposal_key: str
    reviewer_provider_id: str
    reviewer_role_id: str
    falsifiability: float
    generalization: float
    information_gain: float
    feasibility: float
    redundancy: float
    persuasion_risk: float
    confidence: float
    verdict: str
    strongest_objection: str
    decisive_test: str
    raw_text: str = ""
    valid: bool = True

    @classmethod
    def from_text(
        cls,
        assignment: ReviewAssignment,
        text: str,
    ) -> "ResearchReview":
        start, end = text.find("{"), text.rfind("}")
        payload: dict[str, Any] = {}
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = {}
        verdict = str(payload.get("verdict", "")).strip().lower()
        objection = str(payload.get("strongest_objection", "")).strip()
        decisive_test = str(payload.get("decisive_test", "")).strip()
        score_keys = (
            "falsifiability",
            "generalization",
            "information_gain",
            "feasibility",
            "redundancy",
            "persuasion_risk",
            "confidence",
        )
        valid = (
            verdict in {"advance", "test_disagreement", "reject"}
            and bool(objection)
            and bool(decisive_test)
            and all(key in payload for key in score_keys)
        )
        return cls(
            proposal_key=assignment.proposal_key,
            reviewer_provider_id=assignment.reviewer_provider_id,
            reviewer_role_id=assignment.reviewer_role_id,
            falsifiability=_clamp01(payload.get("falsifiability")),
            generalization=_clamp01(payload.get("generalization")),
            information_gain=_clamp01(payload.get("information_gain")),
            feasibility=_clamp01(payload.get("feasibility")),
            redundancy=_clamp01(payload.get("redundancy")),
            persuasion_risk=_clamp01(payload.get("persuasion_risk")),
            confidence=_clamp01(payload.get("confidence")),
            verdict=verdict,
            strongest_objection=objection,
            decisive_test=decisive_test,
            raw_text=text,
            valid=valid,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SwarmPriority:
    proposal_key: str
    provider_id: str
    role_id: str
    review_count: int
    mean_quality: float
    disagreement: float
    persuasion_risk: float
    redundancy: float
    robust_priority: float
    dissent_experiment: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SwarmOutcome:
    proposal_id: str
    provider_id: str
    role_id: str
    split: str
    utility: float
    source: str
    status: str = "measured"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SwarmMemory:
    """Append-only measured utility memory for hypothesis-space swarm search.

    The memory may contain DEV and VALIDATION outcomes only. It is development guidance,
    never a promotion authority and never a carrier for BLIND/Kaggle information.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> list[SwarmOutcome]:
        if not self.path.exists():
            return []
        rows: list[SwarmOutcome] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(SwarmOutcome(**payload))
        return rows

    def append(self, outcome: SwarmOutcome) -> None:
        if outcome.split.lower() not in {"dev", "validation"}:
            raise ValueError("swarm memory accepts DEV/VALIDATION evidence only")
        if not math.isfinite(float(outcome.utility)):
            raise ValueError("swarm utility must be finite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(outcome.to_dict(), sort_keys=True) + "\n")

    @staticmethod
    def _best(rows: Iterable[SwarmOutcome]) -> SwarmOutcome | None:
        measured = [row for row in rows if row.status == "measured"]
        return max(measured, key=lambda row: row.utility, default=None)

    def guidance(self, provider_id: str, role_id: str) -> str:
        rows = self.read()
        global_best = self._best(rows)
        personal_best = self._best(
            row
            for row in rows
            if row.provider_id == provider_id and row.role_id == role_id
        )
        role_best = self._best(row for row in rows if row.role_id == role_id)
        lines = [
            "# MEASURED SWARM SEARCH GUIDANCE",
            (
                "These are attractors for mutation, not truths. Preserve independent search "
                "and do not copy them mechanically."
            ),
        ]
        for label, row in (
            ("global_best", global_best),
            ("personal_best", personal_best),
            ("role_best", role_best),
        ):
            if row is None:
                lines.append(f"- {label}: none measured yet")
            else:
                lines.append(
                    f"- {label}: proposal={row.proposal_id} utility={row.utility:.4f} "
                    f"split={row.split} note={row.note[:240]}"
                )
        lines.append(
            "Search instruction: retain useful causal structure from measured attractors, "
            "but mutate at least one assumption, representation, or control mechanism. "
            "Prefer a small falsifiable move over convergence by imitation."
        )
        return "\n".join(lines)


class SwarmCouncil:
    """Blinded peer review and deterministic experiment prioritization.

    Reviewers rank what deserves scarce experiment compute. They never promote code. High
    disagreement is preserved as a potentially valuable discriminating experiment rather than
    collapsed by majority vote.
    """

    def __init__(self, proposals: Iterable[Any], reviews: Iterable[ResearchReview] = ()) -> None:
        self.proposals = tuple(proposals)
        self.reviews = tuple(reviews)

    @staticmethod
    def blind_payload(proposal: Any) -> dict[str, str]:
        return {
            "hypothesis": str(proposal.hypothesis),
            "experiment": str(proposal.experiment),
            "target_metric": str(proposal.target_metric),
            "split": str(proposal.split),
            "falsifier": str(proposal.falsifier),
            "implementation": str(proposal.implementation),
            "failure_mode": str(proposal.failure_mode),
        }

    def eligible_proposals(self) -> list[Any]:
        return [
            proposal
            for proposal in self.proposals
            if bool(getattr(proposal, "valid", False))
            and str(getattr(proposal, "split", "")).lower() in {"dev", "validation"}
        ]

    def assign_reviews(
        self,
        reviewer_calls: Iterable[Any],
        *,
        reviews_per_proposal: int = 3,
    ) -> list[ReviewAssignment]:
        reviewers = tuple(reviewer_calls)
        count = max(1, int(reviews_per_proposal))
        assignments: list[ReviewAssignment] = []
        for proposal in self.eligible_proposals():
            pkey = proposal_key(proposal)
            candidates = [
                call
                for call in reviewers
                if not (
                    call.provider_id == proposal.provider_id
                    and call.role_id == proposal.role_id
                )
            ]
            candidates.sort(
                key=lambda call: (
                    int(call.provider_id == proposal.provider_id),
                    int(call.role_id == proposal.role_id),
                    hashlib.sha256(
                        f"{pkey}|{call.provider_id}|{call.role_id}".encode()
                    ).hexdigest(),
                )
            )
            selected: list[Any] = []
            used_providers: set[str] = set()
            used_roles: set[str] = set()
            for candidate in candidates:
                diversity_gain = (
                    candidate.provider_id not in used_providers
                    or candidate.role_id not in used_roles
                )
                if diversity_gain or len(candidates) <= count:
                    selected.append(candidate)
                    used_providers.add(candidate.provider_id)
                    used_roles.add(candidate.role_id)
                if len(selected) >= count:
                    break
            if len(selected) < count:
                for candidate in candidates:
                    if candidate in selected:
                        continue
                    selected.append(candidate)
                    if len(selected) >= count:
                        break
            assignments.extend(
                ReviewAssignment(
                    proposal_key=pkey,
                    reviewer_provider_id=call.provider_id,
                    reviewer_role_id=call.role_id,
                )
                for call in selected
            )
        return assignments

    @staticmethod
    def _review_quality(review: ResearchReview) -> float:
        # Reviewer confidence is deliberately a weak term: confidence is not evidence.
        return (
            0.27 * review.falsifiability
            + 0.27 * review.generalization
            + 0.22 * review.information_gain
            + 0.18 * review.feasibility
            + 0.06 * review.confidence
        )

    def priorities(self) -> list[SwarmPriority]:
        by_proposal: dict[str, list[ResearchReview]] = defaultdict(list)
        for review in self.reviews:
            if review.valid:
                by_proposal[review.proposal_key].append(review)
        rows: list[SwarmPriority] = []
        for proposal in self.eligible_proposals():
            pkey = proposal_key(proposal)
            reviews = by_proposal[pkey]
            if not reviews:
                rows.append(
                    SwarmPriority(
                        proposal_key=pkey,
                        provider_id=proposal.provider_id,
                        role_id=proposal.role_id,
                        review_count=0,
                        mean_quality=0.0,
                        disagreement=1.0,
                        persuasion_risk=1.0,
                        redundancy=1.0,
                        robust_priority=-1.0,
                        dissent_experiment=False,
                    )
                )
                continue
            qualities = [self._review_quality(review) for review in reviews]
            center = mean(qualities)
            disagreement = pstdev(qualities) if len(qualities) > 1 else 0.0
            persuasion = mean(review.persuasion_risk for review in reviews)
            redundancy = mean(review.redundancy for review in reviews)
            info_gain = mean(review.information_gain for review in reviews)
            robust = center - 0.35 * disagreement - 0.12 * persuasion - 0.08 * redundancy
            rows.append(
                SwarmPriority(
                    proposal_key=pkey,
                    provider_id=proposal.provider_id,
                    role_id=proposal.role_id,
                    review_count=len(reviews),
                    mean_quality=center,
                    disagreement=disagreement,
                    persuasion_risk=persuasion,
                    redundancy=redundancy,
                    robust_priority=robust,
                    dissent_experiment=disagreement >= 0.12 and info_gain >= 0.55,
                )
            )
        return sorted(
            rows,
            key=lambda row: (row.robust_priority, row.mean_quality, row.proposal_key),
            reverse=True,
        )

    def select(
        self,
        *,
        max_proposals: int = 8,
        min_reviews: int = 2,
    ) -> list[tuple[Any, SwarmPriority]]:
        limit = max(0, int(max_proposals))
        if limit == 0:
            return []
        priorities = {
            row.proposal_key: row
            for row in self.priorities()
            if row.review_count >= max(1, int(min_reviews))
        }
        candidates = [
            proposal
            for proposal in self.eligible_proposals()
            if proposal_key(proposal) in priorities
        ]
        candidates.sort(
            key=lambda proposal: (
                priorities[proposal_key(proposal)].robust_priority,
                priorities[proposal_key(proposal)].mean_quality,
                proposal_key(proposal),
            ),
            reverse=True,
        )
        selected: list[Any] = []
        used_roles: set[str] = set()
        # First pass protects cognitive diversity from majority collapse.
        for proposal in candidates:
            if proposal.role_id in used_roles:
                continue
            selected.append(proposal)
            used_roles.add(proposal.role_id)
            if len(selected) >= limit:
                break
        # Second pass spends remaining slots on robust priority, including dissent experiments.
        for proposal in candidates:
            if proposal in selected:
                continue
            selected.append(proposal)
            if len(selected) >= limit:
                break
        return [(proposal, priorities[proposal_key(proposal)]) for proposal in selected]

    def battle_plan(
        self,
        *,
        max_proposals: int = 8,
        min_reviews: int = 2,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for index, (proposal, priority) in enumerate(
            self.select(max_proposals=max_proposals, min_reviews=min_reviews),
            start=1,
        ):
            particle = _slug(f"{proposal.role_id}-{proposal.provider_id}")
            rows.append(
                {
                    "proposal_id": f"SWARM-{index:02d}-{particle}",
                    "provider_id": proposal.provider_id,
                    "role_id": proposal.role_id,
                    "hypothesis": proposal.hypothesis,
                    "experiment": proposal.experiment,
                    "target_metric": proposal.target_metric,
                    "selection_split": proposal.split,
                    "falsifier": proposal.falsifier,
                    "implementation": proposal.implementation,
                    "failure_mode": proposal.failure_mode,
                    "review": priority.to_dict(),
                    "disagreement_experiment": priority.dissent_experiment,
                    "suggested_branch": f"experiment/swarm-{index:02d}-{particle}",
                }
            )
        return {
            "phase": "swarm-evidence-battle",
            "selected": rows,
            "selected_count": len(rows),
            "eligible_count": len(self.eligible_proposals()),
            "reviewed_count": sum(row.review_count > 0 for row in self.priorities()),
            "authority": "review prioritizes experiments only; arena outcomes decide promotion",
        }
