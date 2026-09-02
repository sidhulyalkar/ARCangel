from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from arc3lab.arena.experiment_guard import require_valid_experiment_scope, resolve_ref
from arc3lab.arena.schema import ArenaManifest, ArenaResult, ContestantSpec


PROFILE_IDS = {
    "coding-minimal": "B-coding-minimal",
    "v011": "C-v011-reflective",
    "v012": "D-v012-evidence-first",
    "v012-lite": "E-v012-lite",
}
_ALLOWED_TARGETS = {"coding-minimal", "v012", "v012-lite"}
_ALLOWED_CONTROLS = set(PROFILE_IDS)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:72] or "swarm"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True, slots=True)
class SwarmPromotion:
    contestant_id: str
    control_id: str
    proposal_id: str
    generation: int
    provider_id: str
    role_id: str
    target_profile: str
    control_profile: str
    candidate_git_sha: str
    trusted_base_sha: str
    worktree: str
    fitness_receipt: str
    guard_receipt: str
    robust_delta: float
    mean_delta: float
    paired_runs: int
    status: str = "arena_promoted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SwarmPromotion":
        return cls(**payload)

    def candidate_spec(self, registry_path: str | Path) -> ContestantSpec:
        registry = str(Path(registry_path).resolve())
        command = (
            "python",
            "scripts/run_promoted_swarm_contestant.py",
            "--registry",
            registry,
            "--contestant",
            "{contestant}",
            "--mode",
            "candidate",
            "--split",
            "{split}",
            "--seed",
            "{seed}",
            "--result",
            "{result}",
            "--split-registry",
            "{arena_root}/splits.public.json",
        )
        judge = (
            "python",
            "scripts/run_promoted_swarm_contestant.py",
            "--registry",
            registry,
            "--contestant",
            "{contestant}",
            "--mode",
            "candidate",
            "--split",
            "blind",
            "--seed",
            "{seed}",
            "--result",
            "{result}",
            "--split-registry",
            "{private_registry}",
        )
        return ContestantSpec(
            contestant_id=self.contestant_id,
            family=f"swarm-{self.target_profile}",
            role=f"swarm:{self.role_id}",
            description=(
                f"Measured swarm proposal {self.proposal_id} from {self.provider_id}; "
                f"candidate={self.candidate_git_sha[:12]} base={self.trusted_base_sha[:12]}"
            ),
            command=command,
            judge_command=judge,
            parent=PROFILE_IDS[self.target_profile],
            control_id=self.control_id,
            tags=("swarm-promoted", f"generation-{self.generation}", self.role_id),
            enabled=True,
        )

    def control_spec(self, registry_path: str | Path) -> ContestantSpec:
        registry = str(Path(registry_path).resolve())
        command = (
            "python",
            "scripts/run_promoted_swarm_contestant.py",
            "--registry",
            registry,
            "--contestant",
            self.contestant_id,
            "--result-contestant",
            "{contestant}",
            "--mode",
            "control",
            "--split",
            "{split}",
            "--seed",
            "{seed}",
            "--result",
            "{result}",
            "--split-registry",
            "{arena_root}/splits.public.json",
        )
        judge = (
            "python",
            "scripts/run_promoted_swarm_contestant.py",
            "--registry",
            registry,
            "--contestant",
            self.contestant_id,
            "--result-contestant",
            "{contestant}",
            "--mode",
            "control",
            "--split",
            "blind",
            "--seed",
            "{seed}",
            "--result",
            "{result}",
            "--split-registry",
            "{private_registry}",
        )
        return ContestantSpec(
            contestant_id=self.control_id,
            family=f"frozen-{self.control_profile}",
            role="swarm-frozen-control",
            description=(
                f"Frozen control for {self.contestant_id} at {self.trusted_base_sha[:12]} "
                f"profile={self.control_profile}"
            ),
            command=command,
            judge_command=judge,
            tags=("swarm-control", f"generation-{self.generation}"),
            enabled=True,
        )


class SwarmPromotionRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> list[SwarmPromotion]:
        if not self.path.exists():
            return []
        rows: list[SwarmPromotion] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(SwarmPromotion.from_dict(json.loads(line)))
        return rows

    def append(self, row: SwarmPromotion) -> None:
        existing = self.read()
        if any(item.contestant_id == row.contestant_id for item in existing):
            raise ValueError(f"duplicate promoted contestant {row.contestant_id}")
        if any(item.proposal_id == row.proposal_id for item in existing):
            raise ValueError(f"proposal {row.proposal_id} is already promoted")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")

    def by_contestant(self, contestant_id: str) -> SwarmPromotion:
        for row in self.read():
            if row.contestant_id == contestant_id:
                return row
        raise KeyError(contestant_id)


def build_promotion(
    fitness_receipt: str | Path,
    worktree: str | Path,
    manifest: ArenaManifest,
    *,
    repo_root: str | Path = ".",
) -> SwarmPromotion:
    receipt_path = Path(fitness_receipt).resolve()
    receipt = json.loads(receipt_path.read_text())
    proposal = dict(receipt.get("proposal") or {})
    fitness = dict(receipt.get("fitness") or {})
    outcome = dict(receipt.get("memory_outcome") or {})
    split = str(proposal.get("selection_split", "")).lower()
    if split != "validation":
        raise ValueError("arena promotion requires paired VALIDATION evidence")
    if outcome.get("status") != "measured" or fitness.get("memory_status") != "measured":
        raise ValueError("arena promotion requires repeatable healthy measured fitness")
    robust_delta = float(fitness.get("robust_delta", -1e9))
    if robust_delta < manifest.promotion.min_validation_delta:
        raise ValueError(
            f"swarm robust delta {robust_delta:.4f} < promotion threshold "
            f"{manifest.promotion.min_validation_delta:.4f}"
        )
    if float(fitness.get("candidate_failure_rate", 1.0)) > manifest.promotion.max_failure_rate:
        raise ValueError("candidate failure rate exceeds promotion ceiling")
    if float(fitness.get("candidate_emergency_fraction", 1.0)) > manifest.promotion.max_emergency_fraction:
        raise ValueError("candidate emergency ownership exceeds promotion ceiling")

    target = str(proposal.get("target_profile", ""))
    control = str(proposal.get("control_profile", ""))
    if target not in _ALLOWED_TARGETS:
        raise ValueError(f"unsupported swarm promotion target profile: {target}")
    if control not in _ALLOWED_CONTROLS:
        raise ValueError(f"unsupported swarm promotion control profile: {control}")

    candidate_sha = str(receipt.get("candidate_git_sha", "")).strip()
    trusted_base = str(receipt.get("control_git_sha", "")).strip()
    if len(candidate_sha) < 12 or len(trusted_base) < 12:
        raise ValueError("fitness receipt lacks candidate/control git provenance")
    candidate_root = Path(worktree).resolve()
    if not candidate_root.exists():
        raise FileNotFoundError(candidate_root)
    if resolve_ref(candidate_root, "HEAD") != resolve_ref(candidate_root, candidate_sha):
        raise ValueError("candidate worktree HEAD no longer matches fitness receipt")
    scope = require_valid_experiment_scope(
        repo_root,
        candidate_root,
        base_sha=trusted_base,
        candidate_sha=candidate_sha,
    )

    guard_path = receipt_path.parent / "judge-boundary.json"
    if not guard_path.exists():
        raise ValueError("fitness receipt has no guarded judge-boundary receipt")
    guard = json.loads(guard_path.read_text())
    if not bool((guard.get("scope") or {}).get("valid")):
        raise ValueError("guard receipt does not certify a valid cognition-only experiment")
    if str(guard.get("candidate_head", "")) != scope.candidate_sha:
        raise ValueError("guard candidate SHA disagrees with fitness receipt")
    if str(guard.get("trusted_base_sha", "")) != scope.base_sha:
        raise ValueError("guard base SHA disagrees with fitness receipt")

    proposal_id = str(proposal.get("proposal_id", "")).strip()
    generation = int(proposal.get("generation", 0))
    if not proposal_id or generation < 1:
        raise ValueError("promotion requires generation-stamped swarm proposal identity")
    role_id = str(proposal.get("role_id", "")).strip()
    provider_id = str(proposal.get("provider_id", "")).strip()
    if not role_id or not provider_id:
        raise ValueError("promotion requires provider/role provenance")
    particle = _slug(f"g{generation}-{role_id}-{provider_id}")
    contestant_id = f"SWP-{particle}-{candidate_sha[:10]}"
    control_id = f"SWC-{particle}-{trusted_base[:10]}"
    return SwarmPromotion(
        contestant_id=contestant_id,
        control_id=control_id,
        proposal_id=proposal_id,
        generation=generation,
        provider_id=provider_id,
        role_id=role_id,
        target_profile=target,
        control_profile=control,
        candidate_git_sha=scope.candidate_sha,
        trusted_base_sha=scope.base_sha,
        worktree=str(candidate_root),
        fitness_receipt=str(receipt_path),
        guard_receipt=str(guard_path),
        robust_delta=robust_delta,
        mean_delta=float(fitness.get("mean_delta", 0.0)),
        paired_runs=int(fitness.get("runs", 0)),
    )


def augment_manifest(
    manifest: ArenaManifest,
    registry_path: str | Path,
) -> ArenaManifest:
    registry = SwarmPromotionRegistry(registry_path)
    extras: list[ContestantSpec] = []
    existing = {row.contestant_id for row in manifest.contestants}
    for promotion in registry.read():
        for spec in (promotion.control_spec(registry_path), promotion.candidate_spec(registry_path)):
            if spec.contestant_id in existing:
                raise ValueError(f"promoted contestant collides with manifest id {spec.contestant_id}")
            existing.add(spec.contestant_id)
            extras.append(spec)
    return replace(manifest, contestants=manifest.contestants + tuple(extras))


def import_promotion_validation(
    promotion: SwarmPromotion,
    ledger: Any,
) -> list[ArenaResult]:
    receipt = json.loads(Path(promotion.fitness_receipt).read_text())
    candidate_paths = [Path(path) for path in receipt.get("candidate_results", [])]
    control_paths = [Path(path) for path in receipt.get("control_results", [])]
    if not candidate_paths or len(candidate_paths) != len(control_paths):
        raise ValueError("fitness receipt lacks paired candidate/control result paths")
    imported: list[ArenaResult] = []
    existing = ledger.run_keys()
    for path, contestant_id, kind in [
        *[(path, promotion.contestant_id, "candidate") for path in candidate_paths],
        *[(path, promotion.control_id, "control") for path in control_paths],
    ]:
        row = ArenaResult.from_dict(json.loads(path.read_text()))
        row.contestant_id = contestant_id
        row.metadata["swarm_promotion"] = promotion.proposal_id
        row.metadata["swarm_kind"] = kind
        row.metadata["candidate_git_sha"] = promotion.candidate_git_sha
        row.metadata["trusted_base_sha"] = promotion.trusted_base_sha
        key = (row.contestant_id, row.split, row.seed)
        if key in existing:
            continue
        ledger.append(row)
        existing.add(key)
        imported.append(row)
    return imported
