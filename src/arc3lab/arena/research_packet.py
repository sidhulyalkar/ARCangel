from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from arc3lab.arena.research_context import (
    assert_research_payload_safe,
    sanitize_research_payload,
)


@dataclass(frozen=True, slots=True)
class ResearchRole:
    role_id: str
    mission: str
    adversarial_question: str


DEFAULT_ROLES: tuple[ResearchRole, ...] = (
    ResearchRole("minimalist", "Delete machinery that does not earn held-out score.", "What can be removed without losing competence?"),
    ResearchRole("scientist", "Improve causal hypothesis formation and falsification.", "Which belief is being trusted without enough evidence?"),
    ResearchRole("explorer", "Minimize live actions needed to collapse uncertainty.", "Which single intervention best separates competing mechanics?"),
    ResearchRole("planner", "Compile understood mechanics into efficient execution.", "When should reasoning stop and search begin?"),
    ResearchRole("vision", "Find representation failures across grid, image, delta, and objects.", "What information is hidden by the current representation?"),
    ResearchRole("memory", "Improve long-horizon retrieval while shrinking working context.", "What evidence must stay exact and what should be summarized?"),
    ResearchRole("runtime", "Maximize useful cognition per Kaggle GPU-minute.", "Where is compute spent without changing decisions?"),
    ResearchRole("red_team", "Construct counterexamples to every claimed improvement.", "On which unseen game family should this idea fail?"),
    ResearchRole("generalization", "Reject public-game-specific behavior.", "Would this mechanism still make sense with colors, controls, and geometry permuted?"),
    ResearchRole("integrator", "Combine only independently validated compatible ideas.", "Which interaction effects could make individually good ideas harmful together?"),
)


class ResearchPacketBuilder:
    """Build deterministic, private/leaderboard-safe packets for research agents."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root)

    @staticmethod
    def role_prompt(role: ResearchRole, experiment_id: str) -> str:
        return f"""# ARCangel independent research assignment: {role.role_id}

Experiment: {experiment_id}
Mission: {role.mission}
Adversarial question: {role.adversarial_question}

Work independently before reading other agents' proposals. Treat every current architecture as falsifiable.
Your output must contain exactly one primary architectural hypothesis, the smallest experiment that can test it,
the metric and held-out split that decides the result, the result that would falsify your idea, and a concrete patch
or implementation plan. Do not optimize public game IDs or encode game-specific solutions. Prefer mechanisms that
remain meaningful under color, geometry, action-label, and object-count changes.

Do not claim improvement from unit tests alone. Software correctness, scientific validity, behavioral competence,
and leaderboard performance are separate gates.
"""

    @staticmethod
    def _safe_scorecard(scorecard: dict[str, object]) -> dict[str, object]:
        safe = sanitize_research_payload(json.loads(json.dumps(scorecard)))
        scope = safe.get("evidence_scope")
        if scope != ["dev", "validation"]:
            # A caller may accidentally pass the judge/campaign scorecard. Dynamic private and
            # leaderboard fields are stripped recursively, but its total result_count could still
            # reveal that non-development observations exist. Remove ambiguous aggregate counts.
            safe.pop("result_count", None)
        assert_research_payload_safe(safe)
        return safe

    def build(
        self,
        output_path: str | Path,
        *,
        experiment_id: str,
        scorecard: dict[str, object],
        include_paths: Iterable[str],
        roles: Iterable[ResearchRole] = DEFAULT_ROLES,
    ) -> str:
        files: dict[str, bytes] = {}
        files["arena/scorecard.json"] = (
            json.dumps(self._safe_scorecard(scorecard), indent=2, sort_keys=True) + "\n"
        ).encode()
        files["README_RESEARCH_PACKET.md"] = (
            "# ARCangel research packet\n\n"
            "This packet intentionally excludes private BLIND state and dynamic Kaggle/leaderboard evidence. "
            "Agents should invent against DEV and use VALIDATION only through the development scorecard.\n"
        ).encode()
        for role in roles:
            files[f"prompts/{role.role_id}.md"] = self.role_prompt(role, experiment_id).encode()

        for rel in include_paths:
            path = (self.repo_root / rel).resolve()
            root = self.repo_root.resolve()
            if root not in path.parents and path != root:
                raise ValueError(f"packet path escapes repository: {rel}")
            if not path.exists() or not path.is_file():
                continue
            if "blind" in path.name.lower() or "/blind/" in path.as_posix().lower():
                continue
            files[f"repo/{rel}"] = path.read_bytes()

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
            for name in sorted(files):
                payload = files[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
        raw_tar = tar_buffer.getvalue()
        gzip_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gzip_buffer, mode="wb", mtime=0, filename="") as gz:
            gz.write(raw_tar)
        data = gzip_buffer.getvalue()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
