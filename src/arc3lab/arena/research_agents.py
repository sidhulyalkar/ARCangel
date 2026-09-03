from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests

from arc3lab.arena.provider_transport import build_chat_payload, extract_message_text
from arc3lab.arena.research_packet import DEFAULT_ROLES, ResearchPacketBuilder, ResearchRole
from arc3lab.arena.swarm_intelligence import (
    ResearchReview,
    ReviewAssignment,
    SwarmCouncil,
)


RUNNABLE_PROFILES = ("coding-minimal", "v011", "v012", "v012-lite")


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    provider_id: str
    base_url: str
    model: str
    api_key_env: str
    roles: tuple[str, ...] = ()
    max_tokens: int = 3000
    timeout_seconds: float = 600.0
    request_body: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderSpec":
        request_body = data.get("request_body") or {}
        if not isinstance(request_body, dict):
            raise TypeError(f"provider {data.get('id')} request_body must be an object")
        return cls(
            provider_id=str(data["id"]),
            base_url=str(data["base_url"]).rstrip("/"),
            model=str(data["model"]),
            api_key_env=str(data.get("api_key_env", "")),
            roles=tuple(str(role) for role in data.get("roles", ())),
            max_tokens=max(256, int(data.get("max_tokens", 3000))),
            timeout_seconds=max(10.0, float(data.get("timeout_seconds", 600.0))),
            request_body=dict(request_body),
            enabled=bool(data.get("enabled", True)),
        )

    def transport_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "request_body": self.request_body,
        }


@dataclass(frozen=True, slots=True)
class ResearchCall:
    provider_id: str
    role_id: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider_id}__{self.role_id}"


@dataclass(slots=True)
class ResearchProposal:
    provider_id: str
    role_id: str
    hypothesis: str
    experiment: str
    target_metric: str
    split: str
    falsifier: str
    implementation: str
    failure_mode: str
    target_profile: str = "v012"
    control_profile: str = "coding-minimal"
    raw_text: str = ""
    valid: bool = True

    @classmethod
    def from_text(cls, provider_id: str, role_id: str, text: str) -> "ResearchProposal":
        start, end = text.find("{"), text.rfind("}")
        payload: dict[str, Any] = {}
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = {}
        required = (
            "hypothesis",
            "experiment",
            "target_metric",
            "split",
            "falsifier",
            "implementation",
            "failure_mode",
        )
        valid = all(str(payload.get(key, "")).strip() for key in required)
        return cls(
            provider_id=provider_id,
            role_id=role_id,
            hypothesis=str(payload.get("hypothesis", "")),
            experiment=str(payload.get("experiment", "")),
            target_metric=str(payload.get("target_metric", "")),
            split=str(payload.get("split", "")),
            falsifier=str(payload.get("falsifier", "")),
            implementation=str(payload.get("implementation", "")),
            failure_mode=str(payload.get("failure_mode", "")),
            target_profile=str(payload.get("target_profile", "")).strip(),
            control_profile=str(payload.get("control_profile", "")).strip(),
            raw_text=text,
            valid=valid,
        )

    @property
    def executable_contract_valid(self) -> bool:
        return (
            self.target_profile in RUNNABLE_PROFILES
            and self.control_profile in RUNNABLE_PROFILES
            and self.target_profile != "v011"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "role_id": self.role_id,
            "hypothesis": self.hypothesis,
            "experiment": self.experiment,
            "target_metric": self.target_metric,
            "split": self.split,
            "falsifier": self.falsifier,
            "implementation": self.implementation,
            "failure_mode": self.failure_mode,
            "target_profile": self.target_profile,
            "control_profile": self.control_profile,
            "executable_contract_valid": self.executable_contract_valid,
            "valid": self.valid,
            "raw_text": self.raw_text,
        }


class ResearchSwarm:
    """Run development-only frontier research particles and blinded peer review.

    The swarm may propose and prioritize experiments. It never sees BLIND identities and it never
    promotes a candidate. Promotion remains owned by measured arena evidence.
    """

    def __init__(self, providers: Iterable[ProviderSpec], *, max_workers: int = 4) -> None:
        self.providers = tuple(provider for provider in providers if provider.enabled)
        self.max_workers = max(1, int(max_workers))

    @classmethod
    def load(cls, path: str | Path, *, max_workers: int = 4) -> "ResearchSwarm":
        raw = json.loads(Path(path).read_text())
        providers = [ProviderSpec.from_dict(row) for row in raw.get("providers", [])]
        return cls(providers, max_workers=max_workers)

    def plan(self, roles: Iterable[ResearchRole] = DEFAULT_ROLES) -> list[ResearchCall]:
        role_map = {role.role_id: role for role in roles}
        calls: list[ResearchCall] = []
        for provider in self.providers:
            selected = provider.roles or tuple(role_map)
            for role_id in selected:
                if role_id not in role_map:
                    raise ValueError(f"provider {provider.provider_id} requests unknown role {role_id}")
                calls.append(
                    ResearchCall(
                        provider_id=provider.provider_id,
                        role_id=role_id,
                        model=provider.model,
                    )
                )
        return calls

    def _provider(self, provider_id: str) -> ProviderSpec:
        return next(provider for provider in self.providers if provider.provider_id == provider_id)

    @staticmethod
    def _role(role_id: str, roles: Iterable[ResearchRole]) -> ResearchRole:
        return next(role for role in roles if role.role_id == role_id)

    @staticmethod
    def _missing_proposal(call: ResearchCall, reason: str) -> ResearchProposal:
        return ResearchProposal(
            provider_id=call.provider_id,
            role_id=call.role_id,
            hypothesis="",
            experiment="",
            target_metric="",
            split="",
            falsifier="",
            implementation="",
            failure_mode="",
            target_profile="",
            control_profile="",
            raw_text=reason,
            valid=False,
        )

    def _post_chat(
        self,
        provider: ProviderSpec,
        *,
        system: str,
        user: str,
        temperature: float,
    ) -> str:
        key = os.getenv(provider.api_key_env, "") if provider.api_key_env else ""
        if not key:
            raise RuntimeError(f"MISSING_API_KEY:{provider.api_key_env}")
        payload = build_chat_payload(
            provider.transport_dict(),
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
            temperature=temperature,
            max_tokens=provider.max_tokens,
        )
        response = requests.post(
            f"{provider.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=provider.timeout_seconds,
        )
        response.raise_for_status()
        return extract_message_text(response.json())

    def _call(
        self,
        call: ResearchCall,
        *,
        experiment_id: str,
        context: str,
        roles: Iterable[ResearchRole],
        guidance: str = "",
    ) -> ResearchProposal:
        provider = self._provider(call.provider_id)
        role = self._role(call.role_id, roles)
        system = ResearchPacketBuilder.role_prompt(role, experiment_id)
        user = (
            "Study the blind-safe research context below. Return exactly one JSON object with keys "
            "hypothesis, experiment, target_metric, split, falsifier, implementation, failure_mode, "
            "target_profile, control_profile. target_profile and control_profile must each be one of "
            f"{list(RUNNABLE_PROFILES)}. Do not use v011 as a target_profile because it is retained "
            "only as a historical negative control. Prefer the smallest falsifiable experiment. If "
            "existing runnable profiles can answer the question, explicitly propose a direct comparison "
            "and say in implementation that no cognition patch is required. Only propose a code mutation "
            "when existing profiles cannot falsify the hypothesis; then describe exactly one minimal "
            "mechanism change. Avoid feature bundles. Use DEV for invention and VALIDATION for selection; "
            "never request blind identities.\n\n"
        )
        if guidance:
            user += guidance + "\n\n"
        user += context
        try:
            text = self._post_chat(provider, system=system, user=user, temperature=0.2)
        except Exception as exc:
            return self._missing_proposal(call, f"ERROR:{type(exc).__name__}:{exc}")
        return ResearchProposal.from_text(call.provider_id, call.role_id, text)

    def _review_call(
        self,
        assignment: ReviewAssignment,
        proposal: ResearchProposal,
        *,
        experiment_id: str,
        context: str,
        roles: Iterable[ResearchRole],
    ) -> ResearchReview:
        provider = self._provider(assignment.reviewer_provider_id)
        role = self._role(assignment.reviewer_role_id, roles)
        system = f"""# ARCangel blinded swarm reviewer: {role.role_id}

Experiment: {experiment_id}
Reviewer specialty: {role.mission}
Adversarial question: {role.adversarial_question}

The proposal author identity is deliberately hidden. Do not infer prestige, provider, or author intent.
Judge whether this proposal deserves scarce experimental compute, not whether it sounds persuasive.
No review can promote code. Measured DEV/VALIDATION results remain authoritative.
"""
        proposal_payload = SwarmCouncil.blind_payload(proposal)
        user = (
            "Review the anonymous proposal below against the blind-safe context. Return exactly one JSON object "
            "with numeric scores in [0,1] for falsifiability, generalization, information_gain, feasibility, "
            "redundancy, persuasion_risk, confidence; verdict must be advance, test_disagreement, or reject; "
            "also provide strongest_objection and decisive_test. Check that target_profile/control_profile form "
            "a fair executable comparison. Penalize unnecessary code changes when an existing-profile comparison "
            "would already falsify the hypothesis. Do not request BLIND or Kaggle evidence.\n\n"
            "# ANONYMOUS PROPOSAL\n"
            + json.dumps(proposal_payload, indent=2)
            + "\n\n# RESEARCH CONTEXT\n"
            + context
        )
        try:
            text = self._post_chat(provider, system=system, user=user, temperature=0.0)
        except Exception as exc:
            text = f"ERROR:{type(exc).__name__}:{exc}"
        return ResearchReview.from_text(assignment, text)

    def run_independent_round(
        self,
        *,
        experiment_id: str,
        context: str,
        output_dir: str | Path,
        max_requests: int = 20,
        roles: Iterable[ResearchRole] = DEFAULT_ROLES,
        particle_guidance: dict[str, str] | None = None,
    ) -> list[ResearchProposal]:
        planned = self.plan(roles)
        if len(planned) > max_requests:
            planned = planned[: max(0, int(max_requests))]
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        proposals: list[ResearchProposal] = []
        guidance = particle_guidance or {}
        role_tuple = tuple(roles)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._call,
                    call,
                    experiment_id=experiment_id,
                    context=context,
                    roles=role_tuple,
                    guidance=guidance.get(call.key, ""),
                ): call
                for call in planned
            }
            for future in as_completed(futures):
                call = futures[future]
                try:
                    proposal = future.result()
                except Exception as exc:
                    proposal = self._missing_proposal(
                        call,
                        f"ERROR:{type(exc).__name__}:{exc}",
                    )
                proposals.append(proposal)
                (output / f"{call.key}.json").write_text(
                    json.dumps(proposal.to_dict(), indent=2) + "\n"
                )
        return sorted(proposals, key=lambda proposal: (proposal.provider_id, proposal.role_id))

    def run_review_round(
        self,
        *,
        assignments: Iterable[ReviewAssignment],
        proposals: Iterable[ResearchProposal],
        experiment_id: str,
        context: str,
        output_dir: str | Path,
        max_requests: int = 60,
        roles: Iterable[ResearchRole] = DEFAULT_ROLES,
    ) -> list[ResearchReview]:
        planned = list(assignments)[: max(0, int(max_requests))]
        proposal_map = {
            f"{proposal.provider_id}__{proposal.role_id}": proposal
            for proposal in proposals
        }
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        reviews: list[ResearchReview] = []
        role_tuple = tuple(roles)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._review_call,
                    assignment,
                    proposal_map[assignment.proposal_key],
                    experiment_id=experiment_id,
                    context=context,
                    roles=role_tuple,
                ): assignment
                for assignment in planned
                if assignment.proposal_key in proposal_map
            }
            for future in as_completed(futures):
                assignment = futures[future]
                try:
                    review = future.result()
                except Exception as exc:
                    review = ResearchReview.from_text(
                        assignment,
                        f"ERROR:{type(exc).__name__}:{exc}",
                    )
                reviews.append(review)
                (output / f"{assignment.key}.json").write_text(
                    json.dumps(review.to_dict(), indent=2) + "\n"
                )
        return sorted(
            reviews,
            key=lambda review: (
                review.proposal_key,
                review.reviewer_provider_id,
                review.reviewer_role_id,
            ),
        )
