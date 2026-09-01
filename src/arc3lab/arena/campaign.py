from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arc3lab.arena.first_tournament import FirstTournamentDirector
from arc3lab.arena.orchestrator import ArenaOrchestrator


@dataclass(frozen=True, slots=True)
class CampaignDecision:
    state: str
    reason: str
    action: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CampaignDirector:
    """Resolve the next V013 action from evidence already present on disk.

    The campaign deliberately stops at external evidence boundaries. It may route a failed
    architecture tournament into the development swarm or prepare an exact candidate artifact,
    but it never invents research-provider outputs, BLIND success, or leaderboard observations.
    """

    def __init__(
        self,
        lab: ArenaOrchestrator,
        *,
        root: str | Path,
        public_registry: str | Path,
        private_registry: str | Path,
        package_dir: str | Path,
    ) -> None:
        self.lab = lab
        self.root = Path(root)
        self.public_registry = Path(public_registry)
        self.private_registry = Path(private_registry)
        self.package_dir = Path(package_dir)

    def _packages(self) -> list[dict[str, Any]]:
        if not self.package_dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.package_dir.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if row.get("status") != "PACKAGED_AND_VERIFIED":
                continue
            row["receipt_path"] = str(path)
            rows.append(row)
        return rows

    def _package_by_contestant(self) -> dict[str, dict[str, Any]]:
        return {
            str(row.get("contestant_id")): row
            for row in self._packages()
            if row.get("contestant_id")
        }

    def _evidence_groups(self, contestant_id: str) -> list[dict[str, Any]]:
        evidence = self.lab.leaderboard_evidence()
        if contestant_id == evidence.get("control_id"):
            return list(evidence.get("control_artifacts") or [])
        candidates = dict(evidence.get("candidate_artifacts") or {})
        return list(candidates.get(contestant_id) or [])

    @staticmethod
    def _group_for_hash(
        groups: list[dict[str, Any]],
        artifact_sha256: str,
    ) -> dict[str, Any] | None:
        artifact_sha256 = artifact_sha256.strip().lower()
        for group in groups:
            if str(group.get("artifact_sha256", "")).strip().lower() == artifact_sha256:
                return group
        return None

    def _next_swarm_generation(self) -> int:
        highest = 0
        pattern = re.compile(r"swarm-cycle-generation-(\d+)\.json$")
        for path in self.root.glob("swarm-cycle-generation-*.json"):
            match = pattern.search(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    def _swarm_decision(self, reason: str, *, falsifier: str) -> CampaignDecision:
        generation = self._next_swarm_generation()
        return CampaignDecision(
            state="NEED_SWARM_RESEARCH",
            reason=reason,
            action=(
                "run one heterogeneous hypothesis-space swarm generation, implement selected "
                "experiments in isolated branches, and measure them against their declared controls"
            ),
            details={
                "generation": generation,
                "providers": "configs/research-providers.nvidia-swarm.json",
                "command": (
                    "python scripts/run_swarm_research_cycle.py "
                    "--providers configs/research-providers.nvidia-swarm.json "
                    f"--generation {generation}"
                ),
                "falsifier": falsifier,
                "authority": "swarm proposes and prioritizes; paired arena evidence updates fitness",
            },
        )

    def _blind_incomplete(self, promoted: list[str]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in self.lab.ledger.read():
            if row.split == "blind":
                counts[row.contestant_id] = counts.get(row.contestant_id, 0) + 1
        required = self.lab.manifest.promotion.min_blind_runs
        missing: list[dict[str, Any]] = []
        for contestant_id in promoted:
            contestant = self.lab._contestant(contestant_id)
            control_id = contestant.control_id
            candidate_runs = counts.get(contestant_id, 0)
            control_runs = counts.get(control_id, 0) if control_id else 0
            if candidate_runs < required or (control_id and control_runs < required):
                missing.append(
                    {
                        "contestant_id": contestant_id,
                        "candidate_blind_runs": candidate_runs,
                        "control_id": control_id,
                        "control_blind_runs": control_runs,
                        "required_runs": required,
                    }
                )
        return missing

    def decide(self) -> CampaignDecision:
        manifest = self.lab.manifest
        if not self.public_registry.exists() or not self.private_registry.exists():
            return CampaignDecision(
                state="NEED_SPLITS",
                reason="DEV/VALIDATION/BLIND registries do not exist yet.",
                action="initialize secret-salted ARC environment splits",
                details={
                    "public_registry": str(self.public_registry),
                    "private_registry": str(self.private_registry),
                },
            )

        tournament = FirstTournamentDirector(self.lab)
        next_stage = tournament.next_stage()
        if next_stage is not None:
            return CampaignDecision(
                state="NEED_TOURNAMENT",
                reason=f"adaptive tournament stage {next_stage.name} still has missing runs",
                action="run the next B/C/D/E tournament stage on the shared verified Qwen server",
                details=next_stage.to_dict(),
            )

        promoted = self.lab.promotion_queue()
        if not promoted:
            return self._swarm_decision(
                "the controlled B/C/D/E tournament produced no internally promoted challenger",
                falsifier="the current architecture family failed repeatable DEV/VALIDATION promotion",
            )

        kaggle_ready = self.lab.kaggle_ready_queue()
        if not kaggle_ready:
            missing_blind = self._blind_incomplete(promoted)
            if missing_blind:
                return CampaignDecision(
                    state="NEED_BLIND_JUDGE",
                    reason="an internal challenger survived but lacks complete private BLIND evidence",
                    action="run promoted challengers and their controls on the private BLIND split",
                    details={"promoted": promoted, "missing_blind": missing_blind},
                )
            # Private values remain judge-owned. Researchers learn only that the gate rejected the
            # lineage, not which private games or mechanics caused the rejection.
            return self._swarm_decision(
                "all internally promoted challengers have complete BLIND evidence but none passed the gate",
                falsifier="private BLIND judge rejected the current promoted lineage",
            )

        packages = self._package_by_contestant()
        top_ready = str(kaggle_ready[0]["contestant_id"])
        package = packages.get(top_ready)
        if package is None:
            return CampaignDecision(
                state="NEED_PACKAGE",
                reason=f"{top_ready} is BLIND-qualified but has no verified exact Kaggle notebook receipt",
                action="package the top Kaggle-ready contestant without changing its policy",
                details={"kaggle_ready": kaggle_ready},
            )

        artifact_sha = str(package.get("notebook_sha256", "")).strip().lower()
        if not artifact_sha:
            return CampaignDecision(
                state="INVALID_PACKAGE",
                reason="candidate package receipt is missing notebook SHA-256 provenance",
                action="rebuild and verify the candidate package",
                details=package,
            )

        control_id = manifest.leaderboard_control_id
        if not control_id:
            return CampaignDecision(
                state="INVALID_MANIFEST",
                reason="no external leaderboard control is configured",
                action="configure one exact Duck/Qwen3.8 control identity",
                details={},
            )

        control_groups = self._evidence_groups(control_id)
        qualified_controls = [
            row
            for row in control_groups
            if int(row.get("runs", 0)) >= manifest.min_leaderboard_control_runs
        ]
        if not qualified_controls:
            return CampaignDecision(
                state="NEED_DUCK_CONTROL",
                reason=(
                    f"the exact Duck control needs at least {manifest.min_leaderboard_control_runs} "
                    "scored runs under one frozen notebook hash"
                ),
                action="run or repeat the same exact Duck/Qwen3.8 notebook and record its provenance",
                details={
                    "control_id": control_id,
                    "observed_control_artifacts": control_groups,
                    "required_runs": manifest.min_leaderboard_control_runs,
                },
            )
        if len(qualified_controls) != 1:
            return CampaignDecision(
                state="AMBIGUOUS_DUCK_CONTROL",
                reason="multiple Duck notebook hashes independently satisfy the control repeat requirement",
                action="select one frozen control artifact identity before comparing challengers",
                details={"qualified_control_artifacts": qualified_controls},
            )

        candidate_group = self._group_for_hash(
            self._evidence_groups(top_ready),
            artifact_sha,
        )
        candidate_runs = int(candidate_group.get("runs", 0)) if candidate_group else 0
        if candidate_runs < manifest.min_leaderboard_candidate_runs:
            return CampaignDecision(
                state="NEED_KAGGLE_CANDIDATE_RUN",
                reason=(
                    f"exact candidate artifact {artifact_sha[:12]} has {candidate_runs} scored run(s); "
                    f"{manifest.min_leaderboard_candidate_runs} are required"
                ),
                action="Save & Run the exact same packaged notebook again and record the score receipt",
                details={
                    "contestant_id": top_ready,
                    "notebook": package.get("notebook"),
                    "artifact_sha256": artifact_sha,
                    "observed_artifact_evidence": candidate_group,
                    "required_runs": manifest.min_leaderboard_candidate_runs,
                },
            )

        nominees = self.lab.leaderboard_queue()
        nominee = next(
            (
                row
                for row in nominees
                if row.get("contestant_id") == top_ready
                and str(row.get("candidate_artifact_sha256", "")).lower() == artifact_sha
            ),
            None,
        )
        if nominee is not None:
            return CampaignDecision(
                state="LEADERBOARD_NOMINEE",
                reason="the exact challenger clears the repeated-artifact Duck-relative uncertainty gate",
                action="retain this artifact as the current champion and begin the next independent research round",
                details={"nominee": nominee, "package": package},
            )

        return CampaignDecision(
            state="NEED_MORE_KAGGLE_EVIDENCE",
            reason=(
                "minimum repeat counts are satisfied but the uncertainty-adjusted challenger/control "
                "difference is still inconclusive"
            ),
            action="repeat the same exact candidate artifact; do not rebuild or cherry-pick a lucky run",
            details={
                "contestant_id": top_ready,
                "artifact_sha256": artifact_sha,
                "candidate_evidence": candidate_group,
                "control_evidence": qualified_controls[0],
            },
        )
