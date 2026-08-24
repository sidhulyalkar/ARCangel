from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

EXPECTED_BUILD = "S160-FINAL-20260824-B"
EXPECTED_NOTEBOOK_SHA = "f83666cd2e1082c70eace20e35231f5f9a36599e1e73265a7dd99782be8831e6"
EXPECTED_SOURCE_SHA = "db947cda346279fe0f50891eef17c478dc7b9a53dec4ee84d82bb8ba751e7b91"


def main(path: str) -> None:
    p = Path(path)
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_NOTEBOOK_SHA
    nb = json.loads(raw)
    text = json.dumps(nb)
    for marker in (
        EXPECTED_BUILD,
        "VisualDecisionPolicy",
        "V008 TEMPORAL VISUAL COGNITION PREFLIGHT PASS",
        "image_side=384",
        "visual_packet_calls",
        "submission.parquet",
        "VLLM_DISABLED_KERNELS",
    ):
        assert marker in text, marker
    assert "MODEL_SCORE < 45" not in text
    source_cell = "".join(nb["cells"][3]["source"])
    match = re.search(r'BUNDLE = "([^"]+)"', source_cell)
    assert match
    bundle = base64.b64decode(match.group(1))
    assert hashlib.sha256(bundle).hexdigest() == EXPECTED_SOURCE_SHA
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"cell-{i}", "exec")
    print("S160 NOTEBOOK VERIFICATION PASS")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_s160_notebook.py <notebook.ipynb>")
    main(sys.argv[1])
