#!/usr/bin/env python
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S190A_BUILD = "S190A-V011-QWEN38-20260826"
S190A_NOTEBOOK = "ARCangel_S190A_V011_Qwen38.ipynb"


def _add_file(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    data = path.read_bytes()
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tf.addfile(info, io.BytesIO(data))


def bundle_source() -> tuple[str, str]:
    """Create a content-deterministic offline source bundle for Kaggle.

    The gzip and tar timestamps are zeroed so the same repository content produces the
    same embedded-source SHA across machines and CI checkouts.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            src = ROOT / "src" / "arc3lab"
            for path in sorted(p for p in src.rglob("*") if p.is_file()):
                rel = path.relative_to(ROOT).as_posix()
                _add_file(tf, path, rel)
            runner = ROOT / "scripts" / "run_v011_competition.py"
            _add_file(tf, runner, "scripts/run_v011_competition.py")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


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


def bootstrap_cell(bundle: str, source_sha: str) -> str:
    return f'''import base64, glob, hashlib, io, os, pathlib, subprocess, sys, tarfile

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
EMBEDDED_SOURCE_SHA256 = "{source_sha}"
raw_bundle = base64.b64decode(BUNDLE)
assert hashlib.sha256(raw_bundle).hexdigest() == EMBEDDED_SOURCE_SHA256
work = pathlib.Path("/kaggle/working/arc3_frontier")
work.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw_bundle), mode="r:gz") as tf:
    tf.extractall(work)
sys.path.insert(0, str(work / "src"))
print("ARC3 Frontier source ready:", work)
print("EMBEDDED SOURCE SHA256:", EMBEDDED_SOURCE_SHA256)
'''


def make_structural(bundle: str, source_sha: str) -> dict:
    cells = [
        md_cell(
            "# ARC3 Frontier V001 | Structural Calibration\n\n"
            "Self-contained, no model input required. This is a plumbing/calibration "
            "submission, not the expected winning lane."
        ),
        code_cell(bootstrap_cell(bundle, source_sha)),
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


def make_hybrid(bundle: str, source_sha: str) -> dict:
    cells = [
        md_cell(
            "# ARC3 Frontier V002 | DuckLite / Qwen\n\n"
            "Attach the public Qwen 3.6 27B FP8 model input used by contemporary Duck "
            "submissions. The notebook auto-discovers a plausible local Qwen model and "
            "launches vLLM on localhost. No internet is used."
        ),
        code_cell(bootstrap_cell(bundle, source_sha)),
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


def make_s190a(bundle: str, source_sha: str) -> dict:
    cells = [
        md_cell(
            f"# ARCangel S190A | V011 Lean Reflective + Qwen3.8 27B FP8\n\n"
            f"Build `{S190A_BUILD}`. This is the primary next contender. It intentionally "
            "requires Qwen3.8-27B-FP8 and will fail rather than silently fall back to Qwen3.6.\n\n"
            "Kaggle settings: RTX PRO 6000, Internet OFF. Attach the official ARC-AGI-3 "
            "competition input, the same Blackwell-qualified vLLM wheelhouse family that "
            "passed S170 FINAL D, and the intended Qwen3.8 27B FP8 model input."
        ),
        code_cell(bootstrap_cell(bundle, source_sha)),
        code_cell(
            f'''import ctypes, glob, os, pathlib, subprocess, sys, tempfile

print("ARCANGEL NOTEBOOK BUILD: {S190A_BUILD}")

# Use the qualified offline vLLM wheelhouse when vLLM is absent. Keep internet OFF.
try:
    import vllm  # noqa: F401
    print("VLLM PACKAGE PASS:", getattr(vllm, "__version__", "unknown"))
except Exception:
    vllm_wheels = [
        p for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)
        if pathlib.Path(p).name.lower().startswith("vllm-")
    ]
    if not vllm_wheels:
        raise RuntimeError("Attach the Blackwell-qualified ARC3 vLLM wheelhouse")
    vllm_wheel = sorted(vllm_wheels)[-1]
    wheel_dir = str(pathlib.Path(vllm_wheel).parent)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", wheel_dir, "vllm",
    ])
    import vllm  # noqa: F401
    print("VLLM PACKAGE PASS:", getattr(vllm, "__version__", "unknown"))

# Reproduce the CUDA linker contract that passed S170 FINAL D on RTX PRO 6000.
libcuda_candidates = [
    pathlib.Path(p) for p in (
        glob.glob("/usr/lib/**/libcuda.so.1", recursive=True)
        + glob.glob("/usr/local/**/libcuda.so.1", recursive=True)
    )
]
real_cuda = next((p for p in libcuda_candidates if p.is_file()), None)
if real_cuda is None:
    raise FileNotFoundError("Could not locate host libcuda.so.1")
ctypes.CDLL(str(real_cuda))
print("CUDA DRIVER RUNTIME LOAD PASS:", real_cuda)

link_dir = pathlib.Path("/kaggle/working/arcangel_cuda_link")
link_dir.mkdir(parents=True, exist_ok=True)
alias = link_dir / "libcuda.so"
if alias.exists() or alias.is_symlink():
    alias.unlink()
alias.symlink_to(real_cuda)
os.environ["LIBRARY_PATH"] = str(link_dir) + os.pathsep + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = str(real_cuda.parent) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
with tempfile.TemporaryDirectory() as td:
    src = pathlib.Path(td) / "probe.cpp"
    exe = pathlib.Path(td) / "probe"
    src.write_text("int main(){{return 0;}}", encoding="utf-8")
    subprocess.check_call(["c++", str(src), "-L", str(link_dir), "-lcuda", "-o", str(exe)])
print("CUDA DRIVER LINKER PASS")

# The FlashInfer FP8 SM120 autotuner crashed in the qualified Kaggle runtime.
os.environ["VLLM_DISABLED_KERNELS"] = "FlashInferFP8ScaledMMLinearKernel"
print("FLASHINFER FP8 LINEAR KERNEL DISABLE PASS")
'''
        ),
        code_cell(
            '''import pathlib, runpy, sys

runner = pathlib.Path("/kaggle/working/arc3_frontier/scripts/run_v011_competition.py")
if not runner.exists():
    raise FileNotFoundError(runner)

# Intentionally no --allow-qwen36-fallback: this slot answers the Qwen3.8 contender question.
sys.argv = [
    str(runner),
    "--workers", "28",
    "--max-actions", "1000",
    "--max-resets", "2",
    "--max-model-calls", "160",
    "--max-tool-calls", "24",
    "--time-budget-seconds", "25200",
    "--game-time-budget-seconds", "7800",
    "--output", "/kaggle/working/arcangel_s190a_receipt.json",
    "--seed", "20260826",
]
runpy.run_path(str(runner), run_name="__main__")
'''
        ),
        code_cell(
            f'''import json, pathlib

receipt = pathlib.Path("/kaggle/working/arcangel_s190a_receipt.json")
submission = pathlib.Path("/kaggle/working/submission.parquet")
if not receipt.exists():
    raise FileNotFoundError("S190A receipt missing")
summary = json.loads(receipt.read_text(encoding="utf-8"))
print("S190A RECEIPT BUILD:", summary.get("build_id"))
print("S190A MODEL FAMILY:", summary.get("model_family"))
print("S190A DIAGNOSTICS:", json.dumps(summary.get("diagnostics", {{}}), sort_keys=True))
print("S190A SEMANTIC DIAGNOSTICS:", json.dumps({{k: v for k, v in summary.get("semantic_diagnostics", {{}}).items() if k != "per_game"}}, sort_keys=True))
if not submission.exists():
    raise FileNotFoundError("Official submission.parquet was not produced")
print("SAVE/RUN VALIDATION PASS: {S190A_BUILD}")
print("SUBMISSION FILE READY:", submission)
'''
        ),
    ]
    return notebook(cells)


def main() -> None:
    bundle, source_sha = bundle_source()
    out = ROOT / "kaggle"
    out.mkdir(exist_ok=True)
    (out / "arc3_v001_structural_calibration.ipynb").write_text(
        json.dumps(make_structural(bundle, source_sha), indent=1), encoding="utf-8"
    )
    (out / "arc3_v002_ducklite_qwen.ipynb").write_text(
        json.dumps(make_hybrid(bundle, source_sha), indent=1), encoding="utf-8"
    )
    s190a_path = out / S190A_NOTEBOOK
    s190a_path.write_text(json.dumps(make_s190a(bundle, source_sha), indent=1), encoding="utf-8")
    print("generated notebooks in", out)
    print("S190A notebook:", s190a_path)
    print("S190A notebook sha256:", hashlib.sha256(s190a_path.read_bytes()).hexdigest())
    print("S190A embedded source sha256:", source_sha)


if __name__ == "__main__":
    main()
