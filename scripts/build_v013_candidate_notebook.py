#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:80] or "candidate"


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
    """Bundle only authoritative source, never interpreter caches or local artifacts."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            src = ROOT / "src" / "arc3lab"
            paths = sorted(
                path
                for path in src.rglob("*.py")
                if path.is_file() and "__pycache__" not in path.parts
            )
            for path in paths:
                _add_file(tf, path, path.relative_to(ROOT).as_posix())
            runner = ROOT / "scripts" / "run_v013_candidate.py"
            _add_file(tf, runner, "scripts/run_v013_candidate.py")
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


def make_notebook(
    *,
    bundle: str,
    source_sha: str,
    profile: str,
    contestant_id: str,
    build_id: str,
    seed: int,
) -> dict:
    slug = _slug(contestant_id).lower()
    receipt_path = f"/kaggle/working/arcangel_v013_{slug}_receipt.json"
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
work = pathlib.Path("/kaggle/working/arc3_v013_candidate")
work.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw_bundle), mode="r:gz") as tf:
    tf.extractall(work)
sys.path.insert(0, str(work / "src"))
print("ARCANGEL NOTEBOOK BUILD: {build_id}")
print("ARCANGEL CONTESTANT: {contestant_id}")
print("ARCANGEL PROFILE: {profile}")
print("EMBEDDED SOURCE SHA256:", EMBEDDED_SOURCE_SHA256)
'''

    runtime = '''import ctypes, glob, os, pathlib, subprocess, sys, tempfile

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

    run = f'''import pathlib, runpy, sys

runner = pathlib.Path("/kaggle/working/arc3_v013_candidate/scripts/run_v013_candidate.py")
if not runner.exists():
    raise FileNotFoundError(runner)
sys.argv = [
    str(runner),
    "--profile", "{profile}",
    "--contestant-id", "{contestant_id}",
    "--build-id", "{build_id}",
    "--workers", "28",
    "--max-actions", "900",
    "--max-resets", "2",
    "--max-model-calls", "200",
    "--max-tool-calls", "96",
    "--time-budget-seconds", "25200",
    "--game-time-budget-seconds", "7800",
    "--output", "{receipt_path}",
    "--seed", "{seed}",
]
runpy.run_path(str(runner), run_name="__main__")
'''

    receipt = f'''import json, pathlib

receipt = pathlib.Path("{receipt_path}")
submission = pathlib.Path("/kaggle/working/submission.parquet")
if not receipt.exists():
    raise FileNotFoundError("V013 candidate receipt missing")
summary = json.loads(receipt.read_text(encoding="utf-8"))
print("V013 RECEIPT BUILD:", summary.get("build_id"))
print("V013 CONTESTANT:", summary.get("contestant_id"))
print("V013 PROFILE:", summary.get("profile"))
print("V013 MODEL FAMILY:", summary.get("model_family"))
print("V013 DIAGNOSTICS:", json.dumps(summary.get("diagnostics", {{}}), sort_keys=True))
if summary.get("build_id") != "{build_id}":
    raise RuntimeError("receipt build id mismatch")
if summary.get("contestant_id") != "{contestant_id}":
    raise RuntimeError("receipt contestant mismatch")
if not submission.exists():
    raise FileNotFoundError("Official submission.parquet was not produced")
print("SAVE/RUN VALIDATION PASS: {build_id}")
print("SUBMISSION FILE READY:", submission)
'''

    return notebook(
        [
            md_cell(
                f"# ARCangel V013 Candidate | {contestant_id}\n\n"
                f"Build `{build_id}` using profile `{profile}`. This notebook is generated only after "
                "the selected architecture has been evaluated through the V013 research arena.\n\n"
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
    parser = argparse.ArgumentParser(description="Build an exact V013 Kaggle candidate notebook")
    parser.add_argument(
        "--profile",
        required=True,
        choices=["coding-minimal", "v011", "v012", "v012-lite"],
    )
    parser.add_argument("--contestant-id", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    bundle, source_sha = bundle_source()
    output = (
        Path(args.output)
        if args.output
        else ROOT / "kaggle" / f"ARCangel_V013_{_slug(args.contestant_id)}_Qwen38.ipynb"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = make_notebook(
        bundle=bundle,
        source_sha=source_sha,
        profile=args.profile,
        contestant_id=args.contestant_id,
        build_id=args.build_id,
        seed=args.seed,
    )
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("V013 candidate notebook:", output)
    print("V013 candidate notebook sha256:", hashlib.sha256(output.read_bytes()).hexdigest())
    print("V013 embedded source sha256:", source_sha)


if __name__ == "__main__":
    main()
