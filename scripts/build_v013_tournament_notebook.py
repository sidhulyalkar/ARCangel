#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RESEARCH_SPLIT_SALT = "arcangel-v013-public-research-split-v1"


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
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            for path in sorted((ROOT / "src" / "arc3lab").rglob("*.py")):
                if path.is_file() and "__pycache__" not in path.parts:
                    _add_file(tf, path, path.relative_to(ROOT).as_posix())
            for rel in (
                "scripts/run_first_tournament.py",
                "scripts/run_arena_contestant.py",
                "configs/swarm-v013.json",
            ):
                _add_file(tf, ROOT / rel, rel)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def _code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def _markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def make_notebook(*, bundle: str, source_sha: str, build_id: str) -> dict[str, object]:
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
work = pathlib.Path("/kaggle/working/arcangel_v013_tournament")
work.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(raw_bundle), mode="r:gz") as tf:
    tf.extractall(work)
sys.path.insert(0, str(work / "src"))
os.chdir(work)
print("ARCANGEL TOURNAMENT BUILD: {build_id}")
print("EMBEDDED SOURCE SHA256:", EMBEDDED_SOURCE_SHA256)
'''

    runtime = '''import ctypes, glob, os, pathlib, subprocess, sys, tempfile

try:
    import vllm  # noqa: F401
except Exception:
    wheels = [
        p for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)
        if pathlib.Path(p).name.lower().startswith("vllm-")
    ]
    if not wheels:
        raise RuntimeError("Attach ARC3 vLLM H100 Wheelhouse V3")
    wheel_dir = str(pathlib.Path(sorted(wheels)[-1]).parent)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--find-links", wheel_dir, "vllm",
    ])
    import vllm  # noqa: F401
print("VLLM PACKAGE:", getattr(vllm, "__version__", "unknown"))

candidates = [
    pathlib.Path(p) for p in (
        glob.glob("/usr/lib/**/libcuda.so.1", recursive=True)
        + glob.glob("/usr/local/**/libcuda.so.1", recursive=True)
    )
]
real_cuda = next((p for p in candidates if p.is_file()), None)
if real_cuda is None:
    raise FileNotFoundError("Could not locate host libcuda.so.1")
ctypes.CDLL(str(real_cuda))
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
print("CUDA DRIVER RUNTIME/LINKER PASS")
'''

    split = f'''import json, pathlib
from arc_agi import Arcade
from arc3lab.arena.splits import SplitRegistry

arc = Arcade()
game_ids = sorted(str(item.game_id) for item in arc.get_environments())
if len(game_ids) < 3:
    raise RuntimeError("Too few public environments for research split")
research_salt = "{PUBLIC_RESEARCH_SPLIT_SALT}"
registry = SplitRegistry.build(game_ids, salt=research_salt, dev_fraction=0.60, validation_fraction=0.20)
root = pathlib.Path("artifacts/arena/v013")
root.mkdir(parents=True, exist_ok=True)
public = root / "splits.public.json"
public.write_text(json.dumps(registry.public_dict(), indent=2) + "\\n", encoding="utf-8")
private = root / "splits.private.json"
if private.exists():
    raise RuntimeError("Research-only notebook must not materialize private BLIND identities")
print("PUBLIC RESEARCH SPLIT:", json.dumps({{
    "split_version": "v1",
    "game_count": len(game_ids),
    "dev_count": len(registry.dev),
    "validation_count": len(registry.validation),
    "reserved_research_holdout_count": len(registry.blind),
    "private_blind_materialized": False,
}}, sort_keys=True))
'''

    run = f'''import pathlib, runpy, sys

runner = pathlib.Path("scripts/run_first_tournament.py")
sys.argv = [
    str(runner),
    "--manifest", "configs/swarm-v013.json",
    "--root", "artifacts/arena/v013",
    "--server-max-sequences", "16",
    "--max-stages", "8",
]
runpy.run_path(str(runner), run_name="__main__")
print("TOURNAMENT EXECUTION COMPLETE: {build_id}")
'''

    receipt = f'''import hashlib, json, pathlib, shutil

root = pathlib.Path("artifacts/arena/v013")
status = root / "first-tournament-status.json"
scorecard = root / "first-tournament-scorecard.json"
if not status.exists() or not scorecard.exists():
    raise FileNotFoundError("Tournament status/scorecard missing")
if (root / "splits.private.json").exists():
    raise RuntimeError("Private BLIND registry unexpectedly exists")
summary = {{
    "build_id": "{build_id}",
    "embedded_source_sha256": "{source_sha}",
    "public_research_split_version": "v1",
    "status": json.loads(status.read_text(encoding="utf-8")),
    "scorecard_sha256": hashlib.sha256(scorecard.read_bytes()).hexdigest(),
    "private_blind_materialized": False,
    "submission_created": pathlib.Path("/kaggle/working/submission.parquet").exists(),
    "authority": "research-only B/C/D/E tournament; no private BLIND and no Kaggle submission",
}}
if summary["submission_created"]:
    raise RuntimeError("Research tournament must not create submission.parquet")
out = pathlib.Path("/kaggle/working/ARCangel_V013_Tournament_Receipt.json")
out.write_text(json.dumps(summary, indent=2) + "\\n", encoding="utf-8")
shutil.copy2(scorecard, "/kaggle/working/ARCangel_V013_Tournament_Scorecard.json")
shutil.copy2(status, "/kaggle/working/ARCangel_V013_Tournament_Status.json")
print(json.dumps(summary, indent=2))
'''

    return {
        "cells": [
            _markdown(
                f"# ARCangel V013 Research Tournament | {build_id}\n\n"
                "Research-only adaptive B/C/D/E architecture tournament. It uses a stable versioned "
                "public DEV/VALIDATION split across code revisions, reserves a research-only holdout, "
                "runs one shared Qwen3.8 27B FP8 server, and exports scorecard/status artifacts. The "
                "true private BLIND registry is separate and is not materialized here. **This notebook "
                "does not create a Kaggle submission.**\n\n"
                "Kaggle settings: RTX PRO 6000, Internet OFF. Attach ARC Prize 2026, ARC3 vLLM H100 "
                "Wheelhouse V3, and Qwen3.8 27B FP8 Repacked."
            ),
            _code(bootstrap),
            _code(runtime),
            _code(split),
            _code(run),
            _code(receipt),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deterministic V013 research tournament notebook")
    ap.add_argument("--build-id", default="V013-TRUTH-PHASE-BCDE-20260902")
    ap.add_argument("--output", default="kaggle/ARCangel_V013_Research_Tournament_Qwen38.ipynb")
    args = ap.parse_args()

    bundle, source_sha = bundle_source()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = make_notebook(bundle=bundle, source_sha=source_sha, build_id=args.build_id)
    output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("V013 tournament notebook:", output)
    print("V013 tournament notebook sha256:", hashlib.sha256(output.read_bytes()).hexdigest())
    print("V013 tournament embedded source sha256:", source_sha)


if __name__ == "__main__":
    main()
