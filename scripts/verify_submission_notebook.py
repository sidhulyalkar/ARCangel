#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "S115-FINAL-20260822-C": "76bab67d0bd1d97ae5bd8336e1af2df994eaa8cf666f7f1d4b098be7ff2bb5b5",
    "S120-FINAL-20260822-C": "f5db889630e94b8629f216dbf83404d65d92cbae8ef3a93cf3a36a60e2c7392b",
}

REQUIRED_TEXT = [
    "ARCANGEL SUBMISSION BUILD:",
    "Top mounted HF candidates:",
    "MODEL INPUT PASS:",
    "VLLM SERVER PASS",
    "FULL INFRASTRUCTURE PREFLIGHT PASS",
    "SAVE/RUN VALIDATION PASS:",
]

FORBIDDEN_TEXT = [
    "MODEL_SCORE < 45",
    "Best mounted model does not look like Qwen3.6 27B FP8",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--build", choices=sorted(EXPECTED), required=True)
    args = ap.parse_args()

    text = args.notebook.read_text(encoding="utf-8")
    data = json.loads(text)
    if data.get("nbformat") != 4:
        raise SystemExit("FAIL: expected nbformat 4")

    actual = sha256(args.notebook)
    expected = EXPECTED[args.build]
    print("build:", args.build)
    print("sha256:", actual)
    if actual != expected:
        raise SystemExit(f"FAIL: SHA mismatch; expected {expected}")

    if args.build not in text:
        raise SystemExit("FAIL: build ID absent from notebook")
    for marker in REQUIRED_TEXT:
        if marker not in text:
            raise SystemExit(f"FAIL: required marker absent: {marker}")
    for marker in FORBIDDEN_TEXT:
        if marker in text:
            raise SystemExit(f"FAIL: obsolete marker present: {marker}")

    for i, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{i}", "exec")

    print("PASS: exact FINAL C release artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
