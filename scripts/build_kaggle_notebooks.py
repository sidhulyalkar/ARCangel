#!/usr/bin/env python
from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bundle_source() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        src = ROOT / "src" / "arc3lab"
        tf.add(src, arcname="src/arc3lab")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def bootstrap_cell(bundle: str) -> str:
    return f'''import base64, glob, io, os, pathlib, subprocess, sys, tarfile

# Kaggle forces competition mode; set it before importing arc_agi.
os.environ["OPERATION_MODE"] = "competition"

wheel_dirs = glob.glob("/kaggle/input/**/arc_agi_3_wheels", recursive=True)
if not wheel_dirs:
    raise FileNotFoundError("Could not locate arc_agi_3_wheels in Kaggle inputs")
WHEELS = wheel_dirs[0]
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
    f"--find-links={{WHEELS}}", "arc-agi==0.9.8", "arcengine==0.9.3",
])

BUNDLE = "{bundle}"
work = pathlib.Path("/kaggle/working/arc3_frontier")
work.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as tf:
    tf.extractall(work)
sys.path.insert(0, str(work / "src"))
print("ARC3 Frontier source ready:", work)
'''


def make_structural(bundle: str) -> dict:
    cells = [
        md_cell(
            "# ARC3 Frontier V001 | Structural Calibration\n\n"
            "Self-contained, no model input required. This is a plumbing/calibration "
            "submission, not the expected winning lane."
        ),
        code_cell(bootstrap_cell(bundle)),
        code_cell(
            '''from arc3lab.evaluation.runner import run_suite
from arc3lab.policy.structural import StructuralPolicy

SEED = 20260820

def factory(game_id):
    return StructuralPolicy(seed=SEED ^ sum(map(ord, game_id)))

out = run_suite(
    policy_factory=factory,
    max_actions=600,
    max_resets=2,
    workers=8,
    time_budget_seconds=3600,
    tags=["frontier-v001", "structural-calibration"],
    output_path="/kaggle/working/arc3_frontier_v001.json",
)
print(out.get("scorecard"))
print(out.get("diagnostics"))
'''
        ),
    ]
    return notebook(cells)


def make_hybrid(bundle: str) -> dict:
    cells = [
        md_cell(
            "# ARC3 Frontier V002 | DuckLite / Qwen\n\n"
            "Attach the public Qwen 3.6 27B FP8 model input used by contemporary Duck "
            "submissions. The notebook auto-discovers a plausible local Qwen model and "
            "launches vLLM on localhost. No internet is used."
        ),
        code_cell(bootstrap_cell(bundle)),
        code_cell(
            '''import glob, os, pathlib, subprocess, sys

# Public Duck-class Kaggle notebooks attach an offline vLLM wheelhouse. Install it
# only when the base image does not already provide vLLM.
try:
    import vllm  # noqa: F401
except Exception:
    vllm_wheels = [
        p for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)
        if pathlib.Path(p).name.lower().startswith("vllm-")
    ]
    if not vllm_wheels:
        raise RuntimeError(
            "vLLM is not installed. Attach the public ARC3 vLLM wheelhouse dataset."
        )
    wheel_dir = str(pathlib.Path(sorted(vllm_wheels)[-1]).parent)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", wheel_dir, "vllm",
    ])

from arc3lab.model import OpenAICompatLocalAdapter, discover_model_path, launch_vllm

MODEL_PATH = os.environ.get("ARC3_MODEL_PATH") or discover_model_path()
if not MODEL_PATH:
    raise FileNotFoundError(
        "No model config found under /kaggle/input. Attach the Qwen 3.6 27B FP8 "
        "model dataset or set ARC3_MODEL_PATH."
    )
print("Model:", MODEL_PATH)
server = launch_vllm(MODEL_PATH, max_model_len=16384, gpu_memory_utilization=0.92)
model = OpenAICompatLocalAdapter(model="arc3", max_tokens=448, timeout=180)
'''
        ),
        code_cell(
            '''from arc3lab.evaluation.runner import run_suite
from arc3lab.policy.hybrid import HybridPolicy

SEED = 20260820

def factory(game_id):
    return HybridPolicy(
        model=model,
        seed=SEED ^ sum(map(ord, game_id)),
        model_every=1,
        max_model_calls=96,
    )

try:
    out = run_suite(
        policy_factory=factory,
        max_actions=1200,
        max_resets=2,
        workers=6,
        time_budget_seconds=29400,
        tags=["frontier-v002", "ducklite-qwen"],
        output_path="/kaggle/working/arc3_frontier_v002.json",
    )
    print(out.get("scorecard"))
    print(out.get("diagnostics"))
finally:
    server.terminate()
    try:
        server.wait(timeout=20)
    except Exception:
        server.kill()
'''
        ),
    ]
    return notebook(cells)


def main() -> None:
    bundle = bundle_source()
    out = ROOT / "kaggle"
    out.mkdir(exist_ok=True)
    (out / "arc3_v001_structural_calibration.ipynb").write_text(
        json.dumps(make_structural(bundle), indent=1), encoding="utf-8"
    )
    (out / "arc3_v002_ducklite_qwen.ipynb").write_text(
        json.dumps(make_hybrid(bundle), indent=1), encoding="utf-8"
    )
    print("generated notebooks in", out)


if __name__ == "__main__":
    main()
