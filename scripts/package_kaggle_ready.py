from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.schema import ArenaManifest


PROFILE_BY_FAMILY = {
    "coding": "coding-minimal",
    "v011": "v011",
    "v012": "v012",
    "v012-lite": "v012-lite",
}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:80] or "candidate"


def git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Package V013 BLIND-qualified candidates for Kaggle")
    ap.add_argument("--manifest", default="configs/swarm-v013.json")
    ap.add_argument("--arena-root", default="artifacts/arena/v013")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output-dir", default="kaggle/v013_candidates")
    ap.add_argument("--receipt-dir", default="artifacts/arena/v013/packages")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest = ArenaManifest.load(repo_root / args.manifest)
    lab = ArenaOrchestrator(manifest, args.arena_root)
    ready = lab.kaggle_ready_queue()
    if not ready:
        print(json.dumps({"status": "NO_KAGGLE_READY_CANDIDATE"}, indent=2))
        return 2
    selected = ready if args.all else ready[:1]
    head = git_head(repo_root)
    out_dir = repo_root / args.output_dir
    receipt_dir = Path(args.receipt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    packaged = []

    contestants = {row.contestant_id: row for row in manifest.contestants}
    for row in selected:
        contestant_id = str(row["contestant_id"])
        contestant = contestants[contestant_id]
        profile = PROFILE_BY_FAMILY.get(contestant.family)
        if profile is None:
            raise ValueError(
                f"contestant {contestant_id} family {contestant.family!r} has no Kaggle profile"
            )
        build_id = f"V013-{slug(contestant_id).upper()}-{head[:12]}"
        notebook = out_dir / f"ARCangel_{slug(contestant_id)}_Qwen38.ipynb"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_v013_candidate_notebook.py",
                "--profile",
                profile,
                "--contestant-id",
                contestant_id,
                "--build-id",
                build_id,
                "--output",
                str(notebook),
            ],
            cwd=repo_root,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/verify_v013_candidate_notebook.py",
                str(notebook),
                "--profile",
                profile,
                "--contestant-id",
                contestant_id,
                "--build-id",
                build_id,
            ],
            cwd=repo_root,
            check=True,
        )
        notebook_sha = hashlib.sha256(notebook.read_bytes()).hexdigest()
        receipt = {
            "contestant_id": contestant_id,
            "family": contestant.family,
            "profile": profile,
            "build_id": build_id,
            "git_head": head,
            "notebook": str(notebook),
            "notebook_sha256": notebook_sha,
            "blind_gate": row,
            "status": "PACKAGED_AND_VERIFIED",
        }
        receipt_path = receipt_dir / f"{slug(contestant_id)}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        packaged.append(receipt)

    print(json.dumps({"packaged": packaged}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
