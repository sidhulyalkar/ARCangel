#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import requests


COGNITION_ROOTS = (
    "src/arc3lab/policy/",
    "src/arc3lab/memory/",
    "src/arc3lab/perception/",
    "src/arc3lab/planning/",
)
SUPPORT_ROOTS = ("tests/", "docs/")
PROFILE_HINTS = {
    "coding-minimal": ("src/arc3lab/policy/coding.py",),
    "v011": ("src/arc3lab/policy/lean_scientist.py",),
    "v012": (
        "src/arc3lab/policy/evidence_first.py",
        "src/arc3lab/policy/evidence_prompt.py",
        "src/arc3lab/policy/evidence_workspace.py",
    ),
    "v012-lite": (
        "src/arc3lab/policy/evidence_first.py",
        "src/arc3lab/policy/evidence_prompt.py",
        "src/arc3lab/policy/evidence_workspace.py",
    ),
}


def _provider(path: Path, provider_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in payload.get("providers", [])
        if str(row.get("id", "")) == provider_id and bool(row.get("enabled", True))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one enabled provider {provider_id!r}; found {len(matches)}")
    return dict(matches[0])


def _read(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return text[: max(0, limit)]


def _context_files(worktree: Path, proposal: dict[str, Any], max_chars: int) -> str:
    target = str(proposal.get("target_profile", ""))
    preferred = list(PROFILE_HINTS.get(target, ()))
    for root in COGNITION_ROOTS:
        base = worktree / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(worktree).as_posix()
            if rel not in preferred:
                preferred.append(rel)

    remaining = max(0, int(max_chars))
    sections: list[str] = []
    for rel in preferred:
        path = worktree / rel
        if not path.exists() or not path.is_file() or remaining <= 0:
            continue
        text = _read(path, remaining)
        section = f"\n\n### FILE {rel}\n{text}"
        sections.append(section)
        remaining -= len(section)
    return "".join(sections)


def _extract_diff(text: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    start = text.find("diff --git ")
    if start < 0:
        start = text.find("--- a/")
    if start < 0:
        return ""
    return text[start:].strip() + "\n"


def _diff_paths(diff: str) -> tuple[str, ...]:
    paths: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for token in parts[2:4]:
                    if token.startswith(("a/", "b/")):
                        paths.add(token[2:])
        elif line.startswith(("+++ ", "--- ")):
            token = line[4:].strip().split("\t", 1)[0]
            if token == "/dev/null":
                continue
            if token.startswith(("a/", "b/")):
                token = token[2:]
            if token:
                paths.add(token)
    return tuple(sorted(paths))


def _allowed(path: str) -> bool:
    if path.startswith("/") or ".." in Path(path).parts:
        return False
    return any(path.startswith(root) for root in COGNITION_ROOTS + SUPPORT_ROOTS)


def _validate_diff(diff: str) -> tuple[str, ...]:
    if not diff.strip():
        raise ValueError("model returned no unified diff")
    paths = _diff_paths(diff)
    if not paths:
        raise ValueError("model diff has no parseable file paths")
    forbidden = [path for path in paths if not _allowed(path)]
    if forbidden:
        raise ValueError(f"model diff touches forbidden paths: {forbidden}")
    if not any(path.startswith(COGNITION_ROOTS) for path in paths):
        raise ValueError("model diff changes support files only; cognition change is required")
    return paths


def _git_apply(worktree: Path, diff: str, *, check_only: bool) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", encoding="utf-8", delete=False) as handle:
        handle.write(diff)
        patch = Path(handle.name)
    try:
        argv = ["git", "apply", "--whitespace=error"]
        if check_only:
            argv.append("--check")
        argv.append(str(patch))
        return subprocess.run(
            argv,
            cwd=worktree,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        patch.unlink(missing_ok=True)


def _chat(provider: dict[str, Any], system: str, user: str, *, max_tokens: int) -> str:
    env_name = str(provider.get("api_key_env", ""))
    key = os.getenv(env_name, "")
    if not key:
        raise RuntimeError(f"missing API key environment variable {env_name}")
    base_url = str(provider["base_url"]).rstrip("/")
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": str(provider["model"]),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max(256, int(max_tokens)),
        },
        timeout=max(30.0, float(provider.get("timeout_seconds", 600.0))),
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def _prompt(proposal: dict[str, Any], context: str, previous_error: str = "") -> tuple[str, str]:
    system = """You are an ARCangel coding-research worker. Implement exactly one falsifiable cognition change.
You are not an evaluator and must not touch scoring, arena, model-serving, competition runner, packaging, CI,
configuration, or secret-handling code. Return only one git-compatible unified diff. Make the smallest patch
that tests the proposal. Changes are allowed only under src/arc3lab/policy, memory, perception, planning,
plus tests/ or docs/. The patch must modify at least one cognition-owned source file. Do not add game-specific
solutions, public game IDs, hidden-game assumptions, or leaderboard-specific logic."""
    user = (
        "# PROPOSAL\n"
        + json.dumps(proposal, indent=2)
        + "\n\n# CURRENT COGNITION CODE\n"
        + context
    )
    if previous_error:
        user += (
            "\n\n# PREVIOUS PATCH REJECTION\n"
            + previous_error[-4000:]
            + "\nRepair only the patch-format/scope/applicability issue while preserving the experiment."
        )
    user += "\n\nReturn only the unified diff, starting with `diff --git`."
    return system, user


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply one bounded frontier-model cognition patch")
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--worktree", default=".")
    ap.add_argument("--providers", default="configs/research-providers.nvidia-swarm.json")
    ap.add_argument("--provider", default="nvidia-deepseek-v4-pro")
    ap.add_argument("--max-context-chars", type=int, default=90000)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--repair-attempts", type=int, default=2)
    ap.add_argument("--receipt", default="")
    args = ap.parse_args()

    worktree = Path(args.worktree).resolve()
    proposal_path = Path(args.proposal).resolve()
    providers_path = Path(args.providers)
    if not providers_path.is_absolute():
        providers_path = worktree / providers_path
    if not worktree.exists():
        raise FileNotFoundError(worktree)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    provider = _provider(providers_path, args.provider)
    context = _context_files(worktree, proposal, args.max_context_chars)
    if not context.strip():
        raise RuntimeError("no cognition code context was collected")

    attempts: list[dict[str, Any]] = []
    last_error = ""
    applied_paths: tuple[str, ...] = ()
    for attempt in range(max(1, int(args.repair_attempts) + 1)):
        system, user = _prompt(proposal, context, last_error)
        text = _chat(provider, system, user, max_tokens=args.max_tokens)
        diff = _extract_diff(text)
        record: dict[str, Any] = {"attempt": attempt + 1, "raw_tail": text[-6000:]}
        try:
            paths = _validate_diff(diff)
            checked = _git_apply(worktree, diff, check_only=True)
            if checked.returncode != 0:
                raise ValueError(f"git apply --check failed: {checked.stderr[-3000:]}")
            applied = _git_apply(worktree, diff, check_only=False)
            if applied.returncode != 0:
                raise ValueError(f"git apply failed after successful check: {applied.stderr[-3000:]}")
            applied_paths = paths
            record.update({"status": "applied", "paths": list(paths)})
            attempts.append(record)
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            record.update({"status": "rejected", "error": last_error})
            attempts.append(record)
    else:
        raise RuntimeError(f"frontier patch worker exhausted repair attempts: {last_error}")

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not status.strip():
        raise RuntimeError("patch reported success but worktree has no changes")
    receipt = {
        "status": "PATCH_APPLIED",
        "provider_id": args.provider,
        "model": provider["model"],
        "proposal_id": proposal.get("proposal_id"),
        "paths": list(applied_paths),
        "attempts": attempts,
        "authority": "patch worker may mutate cognition only; experiment guard and arena remain authoritative",
    }
    receipt_path = Path(args.receipt) if args.receipt else proposal_path.with_suffix(".patch-worker.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
