from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import arc3lab.arena  # noqa: F401 - deliberately create import caches before packaging


ROOT = Path(__file__).resolve().parents[1]


def build(output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_v013_candidate_notebook.py",
            "--profile",
            "v012-lite",
            "--contestant-id",
            "E-v012-lite",
            "--build-id",
            "TEST-V013-PACKAGE",
            "--seed",
            "7",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_candidate_notebook_is_deterministic_and_cache_free(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"
    build(first)
    build(second)
    assert first.read_bytes() == second.read_bytes()

    nb = json.loads(first.read_text())
    bootstrap = "".join(nb["cells"][1]["source"])
    match = re.search(r'BUNDLE = "([^"]+)"', bootstrap)
    assert match
    bundle = base64.b64decode(match.group(1))
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = archive.getnames()
    assert "scripts/run_v013_candidate.py" in names
    assert all("__pycache__" not in name for name in names)
    assert all(not name.endswith(".pyc") for name in names)

    verified = subprocess.run(
        [
            sys.executable,
            "scripts/verify_v013_candidate_notebook.py",
            str(first),
            "--profile",
            "v012-lite",
            "--contestant-id",
            "E-v012-lite",
            "--build-id",
            "TEST-V013-PACKAGE",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert "V013 CANDIDATE NOTEBOOK VERIFICATION PASS" in verified.stdout
