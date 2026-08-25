from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPECTED_BUILD = "S170-FINAL-20260824-D"
EXPECTED_SHA256 = "8438e8c162f3e7f28dfaab369b51d8c869382530beb3e9d08bc00d6057c71be2"
EXPECTED_SOURCE_SHA256 = "60a425a86ea33239f41d0156ba17d55879cf1d814ff6822ec893cd71ab41cf19"
EXPECTED_MM_ARG = '{"image":2,"video":0}'


def main(path: str) -> None:
    notebook = Path(path)
    digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA mismatch: {digest}")

    nb = json.loads(notebook.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    required = [
        EXPECTED_BUILD,
        EXPECTED_SOURCE_SHA256,
        "V009 SYSTEMATIC FRONTIER PREFLIGHT PASS",
        "V009 PERCEPTUAL SCIENTIST PREFLIGHT PASS",
        "VLLM MULTIMODAL ARG PASS",
        'MM_LIMIT_SPEC = {"image": 2, "video": 0}',
        'MM_LIMIT_JSON = json.dumps(MM_LIMIT_SPEC, separators=(",", ":"))',
        '"--limit-mm-per-prompt", MM_LIMIT_JSON',
        "CUDA DRIVER LINKER PASS",
        "submission.parquet",
    ]
    for marker in required:
        if marker not in text:
            raise SystemExit(f"missing marker: {marker}")

    if '{{"image": 2, "video": 0}}' in text:
        raise SystemExit("malformed doubled-brace multimodal argument still present")

    mm = json.dumps({"image": 2, "video": 0}, separators=(",", ":"))
    if mm != EXPECTED_MM_ARG or json.loads(mm) != {"image": 2, "video": 0}:
        raise SystemExit("multimodal argument round-trip failed")

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{i}", "exec")

    print("S170 FINAL D NOTEBOOK VERIFICATION PASS")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_s170_final_d.py NOTEBOOK.ipynb")
    main(sys.argv[1])
