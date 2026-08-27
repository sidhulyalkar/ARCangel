from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

EXPECTED_BUILD = "S190A-V011-QWEN38-20260826"
EXPECTED_NOTEBOOK = "ARCangel_S190A_V011_Qwen38.ipynb"
EXPECTED_MM_ARG = '{"image":2,"video":0}'


def main(path: str) -> None:
    notebook = Path(path)
    if notebook.name != EXPECTED_NOTEBOOK:
        raise SystemExit(f"unexpected notebook filename: {notebook.name}")

    raw = notebook.read_bytes()
    notebook_sha = hashlib.sha256(raw).hexdigest()
    nb = json.loads(raw)
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    required = [
        EXPECTED_BUILD,
        "Qwen3.8 27B FP8",
        "VLLM_DISABLED_KERNELS",
        "FlashInferFP8ScaledMMLinearKernel",
        "CUDA DRIVER LINKER PASS",
        "run_v011_competition.py",
        '"--workers", "28"',
        '"--max-actions", "1000"',
        '"--max-model-calls", "160"',
        '"--max-tool-calls", "24"',
        '"--time-budget-seconds", "25200"',
        '"--game-time-budget-seconds", "7800"',
        "arcangel_s190a_receipt.json",
        "SAVE/RUN VALIDATION PASS",
        "SUBMISSION FILE READY",
        "submission.parquet",
    ]
    for marker in required:
        if marker not in text:
            raise SystemExit(f"missing marker: {marker}")

    if "--allow-qwen36-fallback" in text and "Intentionally no --allow-qwen36-fallback" not in text:
        raise SystemExit("S190A must not enable Qwen3.6 fallback")

    source_match = re.search(r'EMBEDDED_SOURCE_SHA256 = "([0-9a-f]{64})"', text)
    bundle_match = re.search(r'BUNDLE = "([A-Za-z0-9+/=]+)"', text)
    if source_match is None or bundle_match is None:
        raise SystemExit("embedded source markers missing")
    expected_source_sha = source_match.group(1)
    bundle = base64.b64decode(bundle_match.group(1))
    actual_source_sha = hashlib.sha256(bundle).hexdigest()
    if actual_source_sha != expected_source_sha:
        raise SystemExit(
            f"embedded source SHA mismatch: {actual_source_sha} != {expected_source_sha}"
        )

    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
        names = set(tf.getnames())
    for expected in (
        "src/arc3lab/policy/lean_scientist.py",
        "src/arc3lab/model/server.py",
        "src/arc3lab/evaluation/runner.py",
        "scripts/run_v011_competition.py",
    ):
        if expected not in names:
            raise SystemExit(f"source bundle missing {expected}")

    mm = json.dumps({"image": 2, "video": 0}, separators=(",", ":"), sort_keys=True)
    if mm != EXPECTED_MM_ARG or json.loads(mm) != {"image": 2, "video": 0}:
        raise SystemExit("canonical multimodal JSON contract changed")

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{i}", "exec")

    print("S190A NOTEBOOK VERIFICATION PASS")
    print("notebook_sha256=", notebook_sha)
    print("embedded_source_sha256=", expected_source_sha)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_s190a_notebook.py NOTEBOOK.ipynb")
    main(sys.argv[1])
