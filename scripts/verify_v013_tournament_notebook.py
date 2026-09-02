#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify V013 research-only tournament notebook")
    ap.add_argument("path")
    ap.add_argument("--build-id", required=True)
    args = ap.parse_args()

    path = Path(args.path)
    raw = path.read_bytes()
    nb = json.loads(raw)
    text = json.dumps(nb)
    required = (
        args.build_id,
        "Qwen3.8 27B FP8",
        "run_first_tournament.py",
        "splits.public.json",
        "public_dict()",
        "private_blind_materialized",
        "first-tournament-scorecard.json",
        "Research tournament must not create submission.parquet",
    )
    for marker in required:
        if marker not in text:
            raise AssertionError(f"missing tournament notebook marker: {marker}")

    forbidden = (
        '"--judge"',
        "run_v013_candidate.py",
        "package_kaggle_ready.py",
        "package_promoted_swarm.py",
        "registry.write(",
    )
    for marker in forbidden:
        if marker in text:
            raise AssertionError(f"research tournament contains forbidden authority marker: {marker}")

    bootstrap = "".join(nb["cells"][1]["source"])
    bundle_match = re.search(r'BUNDLE = "([^"]+)"', bootstrap)
    sha_match = re.search(r'EMBEDDED_SOURCE_SHA256 = "([0-9a-f]{64})"', bootstrap)
    if not bundle_match or not sha_match:
        raise AssertionError("embedded tournament bundle metadata missing")
    bundle = base64.b64decode(bundle_match.group(1))
    expected_sha = sha_match.group(1)
    actual_sha = hashlib.sha256(bundle).hexdigest()
    if actual_sha != expected_sha:
        raise AssertionError("embedded tournament source SHA mismatch")

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = set(archive.getnames())
        required_files = {
            "scripts/run_first_tournament.py",
            "scripts/run_arena_contestant.py",
            "configs/swarm-v013.json",
            "src/arc3lab/arena/first_tournament.py",
            "src/arc3lab/arena/splits.py",
            "src/arc3lab/model/server.py",
            "src/arc3lab/policy/coding.py",
            "src/arc3lab/policy/evidence_first.py",
        }
        missing = sorted(required_files - names)
        if missing:
            raise AssertionError(f"embedded tournament files missing: {missing}")

    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")

    print("V013 RESEARCH TOURNAMENT NOTEBOOK VERIFICATION PASS")
    print("notebook_sha256=", hashlib.sha256(raw).hexdigest())
    print("embedded_source_sha256=", actual_sha)
    print("build_id=", args.build_id)


if __name__ == "__main__":
    main()
