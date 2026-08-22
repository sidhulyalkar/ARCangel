#!/usr/bin/env python
from __future__ import annotations

"""Repair Kaggle's CUDA-driver linker name before FlashInfer/vLLM startup.

Kaggle RTX PRO 6000 images can expose a healthy runtime driver as libcuda.so.1
while FlashInfer's SM120 JIT links against the unversioned -lcuda name. This
utility creates a writable libcuda.so alias, exports build/runtime search paths,
and performs the exact host-linker self-test required by FlashInfer.
"""

import ctypes
import os
import shlex
import subprocess
from pathlib import Path


def _prepend_env_path(name: str, *paths: Path) -> None:
    merged: list[str] = []
    for path in paths:
        if path.exists() and str(path) not in merged:
            merged.append(str(path))
    for raw in os.environ.get(name, "").split(os.pathsep):
        if raw and raw not in merged:
            merged.append(raw)
    os.environ[name] = os.pathsep.join(merged)


def _ldconfig_candidates() -> list[Path]:
    out: list[Path] = []
    try:
        proc = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, check=False, timeout=15
        )
        for line in proc.stdout.splitlines():
            if "libcuda.so" in line and "=>" in line:
                path = Path(line.rsplit("=>", 1)[-1].strip())
                if path.exists():
                    out.append(path.resolve())
    except Exception:
        pass
    return out


def _mapped_candidates() -> list[Path]:
    out: list[Path] = []
    try:
        ctypes.CDLL("libcuda.so.1")
        for line in Path("/proc/self/maps").read_text(errors="replace").splitlines():
            if "libcuda.so" in line:
                path = Path(line.split()[-1])
                if path.exists():
                    out.append(path.resolve())
    except Exception:
        pass
    return out


def prepare_cuda_linker(work: Path = Path("/kaggle/working")) -> Path:
    candidates = _ldconfig_candidates() + _mapped_candidates() + [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib64/libcuda.so.1"),
        Path("/usr/local/nvidia/lib64/libcuda.so.1"),
        Path("/usr/local/cuda/compat/libcuda.so.1"),
    ]
    unique: list[Path] = []
    for path in candidates:
        if path.exists() and path.resolve() not in unique:
            unique.append(path.resolve())
    if not unique:
        raise RuntimeError("Could not resolve a filesystem path for libcuda.so.1")

    driver = unique[0]
    ctypes.CDLL(str(driver))

    link_dir = work / "arcangel_cuda_driver_link"
    link_dir.mkdir(parents=True, exist_ok=True)
    alias = link_dir / "libcuda.so"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(driver)

    cuda_home = Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda")
    os.environ["CUDA_HOME"] = str(cuda_home)
    _prepend_env_path("LIBRARY_PATH", link_dir, driver.parent, cuda_home / "lib64", cuda_home / "targets/x86_64-linux/lib")
    _prepend_env_path("LD_LIBRARY_PATH", link_dir, driver.parent, cuda_home / "lib64", cuda_home / "targets/x86_64-linux/lib")

    extra = os.environ.get("FLASHINFER_EXTRA_LDFLAGS", "").strip()
    flag = f"-L{link_dir}"
    if flag not in extra.split():
        os.environ["FLASHINFER_EXTRA_LDFLAGS"] = (extra + " " + flag).strip()

    for directory in (cuda_home / "lib64/stubs", cuda_home / "lib64"):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "libcuda.so"
            if not target.exists():
                target.symlink_to(driver)
        except Exception:
            pass

    os.environ["FLASHINFER_JIT_DIR"] = str(work / "flashinfer_jit")
    os.environ["FLASHINFER_GEN_SRC_DIR"] = str(work / "flashinfer_generated")
    Path(os.environ["FLASHINFER_JIT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FLASHINFER_GEN_SRC_DIR"]).mkdir(parents=True, exist_ok=True)

    source = work / "arcangel_cuda_link_test.cpp"
    binary = work / "arcangel_cuda_link_test"
    source.write_text(
        'extern "C" int cuInit(unsigned int); int main(){ return (void*)(&cuInit) == nullptr; }\n'
    )
    cmd = [os.environ.get("CXX", "c++"), str(source), f"-L{link_dir}", "-Wl,--no-as-needed", "-lcuda", "-o", str(binary)]
    probe = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(), check=False)
    if probe.returncode:
        raise RuntimeError(
            "CUDA -lcuda linker self-test failed\n"
            + "command: " + " ".join(shlex.quote(x) for x in cmd)
            + "\nstdout:\n" + probe.stdout
            + "\nstderr:\n" + probe.stderr
            + "\nLIBRARY_PATH=" + os.environ.get("LIBRARY_PATH", "")
        )
    return driver


if __name__ == "__main__":
    selected = prepare_cuda_linker()
    print("Selected CUDA driver:", selected)
    print("CUDA DRIVER LINKER PASS")
