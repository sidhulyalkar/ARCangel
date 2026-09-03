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
    parser = argparse.ArgumentParser(description="Verify an exact V013 candidate notebook")
    parser.add_argument("path")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--contestant-id", required=True)
    parser.add_argument("--build-id", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    raw = path.read_bytes()
    nb = json.loads(raw)
    text = json.dumps(nb)
    required = [
        args.build_id,
        args.profile,
        args.contestant_id,
        "Qwen3.8 27B FP8",
        "VLLM_DISABLED_KERNELS",
        "FlashInferFP8ScaledMMLinearKernel",
        "submission.parquet",
        "run_v013_candidate.py",
    ]
    for marker in required:
        if marker not in text:
            raise AssertionError(f"missing notebook marker: {marker}")
    lowered = text.lower()
    if "qwen3.6" in lowered or "allow-qwen36" in lowered:
        raise AssertionError("V013 candidate must not silently fall back to Qwen3.6")

    bootstrap = "".join(nb["cells"][1]["source"])
    bundle_match = re.search(r'BUNDLE = "([^"]+)"', bootstrap)
    sha_match = re.search(r'EMBEDDED_SOURCE_SHA256 = "([0-9a-f]{64})"', bootstrap)
    if not bundle_match or not sha_match:
        raise AssertionError("embedded bundle metadata missing")
    bundle = base64.b64decode(bundle_match.group(1))
    expected_source_sha = sha_match.group(1)
    actual_source_sha = hashlib.sha256(bundle).hexdigest()
    if actual_source_sha != expected_source_sha:
        raise AssertionError("embedded source SHA mismatch")

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = set(archive.getnames())
        required_files = {
            "scripts/run_v013_candidate.py",
            "src/arc3lab/evaluation/runner.py",
            "src/arc3lab/model/server.py",
            "src/arc3lab/policy/coding.py",
            "src/arc3lab/policy/lean_scientist.py",
            "src/arc3lab/policy/evidence_first.py",
        }
        missing = sorted(required_files - names)
        if missing:
            raise AssertionError(f"embedded source files missing: {missing}")
        runner_handle = archive.extractfile("scripts/run_v013_candidate.py")
        if runner_handle is None:
            raise AssertionError("candidate runner is unreadable")
        runner = runner_handle.read().decode("utf-8")
        if 'limit_mm_per_prompt={"image": 2, "video": 0}' not in runner:
            raise AssertionError("candidate runner lost canonical two-image vLLM contract")
        if 'required = ("qwen", "38", "27b", "fp8")' not in runner:
            raise AssertionError("candidate runner lost exact Qwen3.8 model identity gate")

    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{index}", "exec")

    print("V013 CANDIDATE NOTEBOOK VERIFICATION PASS")
    print("notebook_sha256=", hashlib.sha256(raw).hexdigest())
    print("embedded_source_sha256=", actual_source_sha)
    print("profile=", args.profile)
    print("contestant_id=", args.contestant_id)
    print("build_id=", args.build_id)


if __name__ == "__main__":
    main()
