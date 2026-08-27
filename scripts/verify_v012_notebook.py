from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

EXPECTED_BUILD = "S210A-V012-EVIDENCE-FIRST-QWEN38-20260827"
EXPECTED_NOTEBOOK = "ARCangel_S210A_V012_EvidenceFirst_Qwen38.ipynb"


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
        "ARC3 vLLM H100 Wheelhouse V3",
        "VLLM_DISABLED_KERNELS",
        "FlashInferFP8ScaledMMLinearKernel",
        "CUDA DRIVER LINKER PASS",
        "run_v012_competition.py",
        '"--workers", "28"',
        '"--max-actions", "900"',
        '"--max-model-calls", "200"',
        '"--max-tool-calls", "96"',
        '"--max-reasoning-rounds", "4"',
        '"--time-budget-seconds", "25200"',
        "arcangel_s210a_receipt.json",
        "SAVE/RUN VALIDATION PASS",
        "SUBMISSION FILE READY",
        "submission.parquet",
    ]
    for marker in required:
        if marker not in text:
            raise SystemExit(f"missing marker: {marker}")

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
        "src/arc3lab/policy/evidence_first.py",
        "src/arc3lab/policy/evidence_workspace.py",
        "src/arc3lab/policy/evidence_prompt.py",
        "src/arc3lab/model/server.py",
        "src/arc3lab/evaluation/runner.py",
        "scripts/run_v012_competition.py",
    ):
        if expected not in names:
            raise SystemExit(f"source bundle missing {expected}")

    # V012's architecture must not regress to the old candidate-arbitration path.
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
        policy_source = tf.extractfile("src/arc3lab/policy/evidence_first.py")
        if policy_source is None:
            raise SystemExit("missing V012 policy source")
        policy_text = policy_source.read().decode("utf-8")
    forbidden = [
        "enumerate_decision_candidates(",
        "_mode_fallback(",
        "frontier_fallback_actions",
    ]
    for marker in forbidden:
        if marker in policy_text:
            raise SystemExit(f"V012 policy contains forbidden inherited authority marker: {marker}")
    if "super(HybridPolicy, self).choose" in policy_text:
        raise SystemExit("V012 must not route normal decisions through inherited heuristic choose()")

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{i}", "exec")

    print("V012 NOTEBOOK VERIFICATION PASS")
    print("notebook_sha256=", notebook_sha)
    print("embedded_source_sha256=", expected_source_sha)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_v012_notebook.py NOTEBOOK.ipynb")
    main(sys.argv[1])
