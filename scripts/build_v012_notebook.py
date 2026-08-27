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
BUILD_ID = "S210A-V012-EVIDENCE-FIRST-QWEN38-20260827"
NOTEBOOK_NAME = "ARCangel_S210A_V012_EvidenceFirst_Qwen38.ipynb"


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
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            src = ROOT / "src" / "arc3lab"
            for path in sorted(p for p in src.rglob("*") if p.is_file()):
                _add_file(tf, path, path.relative_to(ROOT).as_posix())
            runner = ROOT / "scripts" / "run_v012_competition.py"
            _add_file(tf, runner, "scripts/run_v012_competition.py")
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


def make_notebook(bundle: str, source_sha: str) -> dict:
    bootstrap = f'''import base64, glob, hashlib, io, os, pathlib, subprocess, sys, tarfile

os.environ["OPERATION_MODE"] = "competition"
os.environ["VLLM_DISABLED_KERNELS"] = "FlashInferFP8ScaledMMLinearKernel"

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
work = pathlib.Path("/kaggle/working/arc3_v012")
work.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw_bundle), mode="r:gz") as tf:
    tf.extractall(work)
sys.path.insert(0, str(work / "src"))
print("ARCANGEL NOTEBOOK BUILD: {BUILD_ID}")
print("EMBEDDED SOURCE SHA256:", EMBEDDED_SOURCE_SHA256)
'''

    runtime = '''import ctypes, glob, os, pathlib, subprocess, sys, tempfile

# Keep the environment variable set before any vLLM import so every process sees the
# Blackwell FP8 kernel exclusion that already passed ARCangel's S170 runtime qualification.
os.environ["VLLM_DISABLED_KERNELS"] = "FlashInferFP8ScaledMMLinearKernel"

try:
    import vllm  # noqa: F401
except Exception:
    vllm_wheels = [
        p for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)
        if pathlib.Path(p).name.lower().startswith("vllm-")
    ]
    if not vllm_wheels:
        raise RuntimeError("Attach ARC3 vLLM H100 Wheelhouse V3")
    wheel_dir = str(pathlib.Path(sorted(vllm_wheels)[-1]).parent)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", wheel_dir, "vllm",
    ])
    import vllm  # noqa: F401
print("VLLM PACKAGE:", getattr(vllm, "__version__", "unknown"))

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
    src.write_text("int main(){return 0;}", encoding="utf-8")
    subprocess.check_call(["c++", str(src), "-L", str(link_dir), "-lcuda", "-o", str(exe)])
print("CUDA DRIVER LINKER PASS")
'''

    run = '''import pathlib, runpy, sys

runner = pathlib.Path("/kaggle/working/arc3_v012/scripts/run_v012_competition.py")
if not runner.exists():
    raise FileNotFoundError(runner)
sys.argv = [
    str(runner),
    "--workers", "28",
    "--max-actions", "900",
    "--max-resets", "2",
    "--max-model-calls", "200",
    "--max-tool-calls", "96",
    "--max-reasoning-rounds", "4",
    "--time-budget-seconds", "25200",
    "--game-time-budget-seconds", "7800",
    "--output", "/kaggle/working/arcangel_s210a_receipt.json",
    "--seed", "20260827",
]
runpy.run_path(str(runner), run_name="__main__")
'''

    receipt = f'''import json, pathlib

receipt = pathlib.Path("/kaggle/working/arcangel_s210a_receipt.json")
submission = pathlib.Path("/kaggle/working/submission.parquet")
if not receipt.exists():
    raise FileNotFoundError("V012 receipt missing")
summary = json.loads(receipt.read_text(encoding="utf-8"))
print("V012 RECEIPT BUILD:", summary.get("build_id"))
print("V012 MODEL FAMILY:", summary.get("model_family"))
print("V012 DIAGNOSTICS:", json.dumps(summary.get("diagnostics", {{}}), sort_keys=True))
evidence = summary.get("evidence_diagnostics", {{}})
print("V012 EVIDENCE DIAGNOSTICS:", json.dumps({{k: v for k, v in evidence.items() if k != "per_game"}}, sort_keys=True))
if not submission.exists():
    raise FileNotFoundError("Official submission.parquet was not produced")
print("SAVE/RUN VALIDATION PASS: {BUILD_ID}")
print("SUBMISSION FILE READY:", submission)
'''

    return notebook(
        [
            md_cell(
                "# ARCangel S210A | V012 Evidence-First Coding Agent\n\n"
                f"Build `{BUILD_ID}`. Architectural reset after S190A=0.17. The model receives exact "
                "evidence plus a Python analysis API; ARCangel no longer ranks semantic candidates or "
                "owns normal actions through heuristic fallbacks.\n\n"
                "Settings: RTX PRO 6000, Internet OFF. Attach ARC Prize 2026 competition input, "
                "ARC3 vLLM H100 Wheelhouse V3, and Qwen3.8 27B FP8 Repacked."
            ),
            code_cell(bootstrap),
            code_cell(runtime),
            code_cell(run),
            code_cell(receipt),
        ]
    )


def main() -> None:
    bundle, source_sha = bundle_source()
    out = ROOT / "kaggle"
    out.mkdir(exist_ok=True)
    path = out / NOTEBOOK_NAME
    path.write_text(json.dumps(make_notebook(bundle, source_sha), indent=1), encoding="utf-8")
    print("V012 notebook:", path)
    print("V012 notebook sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
    print("V012 embedded source sha256:", source_sha)


if __name__ == "__main__":
    main()
