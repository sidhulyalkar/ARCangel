#!/usr/bin/env python3
"""Verify the exact S135 FINAL B Kaggle artifact before upload/submission."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path


BUILD = "S135-FINAL-20260823-B"
NOTEBOOK_SHA256 = "b070dc1d412c49b32f8d91c63c606f21370ad6cd0ad93f292446130868870ddf"
SOURCE_BUNDLE_SHA256 = "0055eaae67277191ead3499b41b5e00181cb1c60775e8b9ae50cb0fae6c90a9f"

REQUIRED = [
    BUILD,
    "SpatialCodingPolicy",
    "V006 SPATIAL ENGINE SYNTHETIC PREFLIGHT PASS",
    "spatial_plan_horizon=12",
    "VLLM_DISABLED_KERNELS",
    "FlashInferFP8ScaledMMLinearKernel",
    "CUDA DRIVER LINKER PASS",
    "DUMMY SUBMISSION PARQUET PASS",
    "submission.parquet",
    "http://gateway:8001/api/games",
]

FORBIDDEN = [
    "MODEL_SCORE < 45",
    "_planned_centers",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(path: Path) -> list[str]:
    failures: list[str] = []
    raw = path.read_bytes()
    actual_sha = sha256(raw)
    if actual_sha != NOTEBOOK_SHA256:
        failures.append(
            f"notebook SHA mismatch: expected {NOTEBOOK_SHA256}, got {actual_sha}"
        )

    try:
        nb = json.loads(raw)
    except Exception as exc:
        return [*failures, f"invalid notebook JSON: {type(exc).__name__}: {exc}"]

    text = json.dumps(nb)
    for marker in REQUIRED:
        if marker not in text:
            failures.append(f"missing marker: {marker}")
    for marker in FORBIDDEN:
        if marker in text:
            failures.append(f"forbidden marker present: {marker}")

    bundle_matches: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        bundle_matches.extend(re.findall(r'BUNDLE = "([A-Za-z0-9+/=]+)"', source))

    if len(bundle_matches) != 1:
        failures.append(f"expected exactly one embedded source BUNDLE, found {len(bundle_matches)}")
    else:
        try:
            bundle = base64.b64decode(bundle_matches[0], validate=True)
            actual_bundle_sha = sha256(bundle)
            if actual_bundle_sha != SOURCE_BUNDLE_SHA256:
                failures.append(
                    "embedded source SHA mismatch: "
                    f"expected {SOURCE_BUNDLE_SHA256}, got {actual_bundle_sha}"
                )
        except Exception as exc:
            failures.append(f"embedded source bundle decode failed: {type(exc).__name__}: {exc}")

    metadata = nb.get("metadata", {}).get("arcangel", {})
    if metadata.get("submission_build") != BUILD:
        failures.append(
            f"metadata build mismatch: {metadata.get('submission_build')!r}"
        )
    if metadata.get("source_bundle_sha256") != SOURCE_BUNDLE_SHA256:
        failures.append(
            f"metadata source hash mismatch: {metadata.get('source_bundle_sha256')!r}"
        )
    if metadata.get("architecture") != "V006-spatial-intelligence":
        failures.append(f"unexpected architecture metadata: {metadata.get('architecture')!r}")

    # All plain Python cells should compile. The generated notebook intentionally uses
    # no shell/IPython magics, so any syntax failure is a release blocker.
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            compile(source, f"{path.name}:cell-{i}", "exec")
        except SyntaxError as exc:
            failures.append(f"cell {i} syntax error: {exc}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    failures = verify(args.notebook)
    if failures:
        print("S135 ARTIFACT VERIFICATION FAILED")
        for failure in failures:
            print(" -", failure)
        return 1

    print("S135 ARTIFACT VERIFICATION PASS")
    print("build:", BUILD)
    print("notebook SHA256:", NOTEBOOK_SHA256)
    print("embedded source SHA256:", SOURCE_BUNDLE_SHA256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
