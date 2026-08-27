from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def discover_model_path(roots: Iterable[str] = ("/kaggle/input",)) -> str | None:
    """Find the strongest plausible local Qwen 27B FP8 model without relying on a dataset slug.

    The scorer intentionally prefers the newer Qwen3.8 family when both 3.8 and 3.6 snapshots
    are mounted, while remaining compatible with the already-qualified Qwen3.6 runtime.
    """
    candidates: list[tuple[int, Path]] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for cfg in p.glob("**/config.json"):
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                continue
            text = (str(cfg.parent) + " " + json.dumps(data)[:4000]).lower()
            normalized = "".join(ch for ch in text if ch.isalnum())
            score = 0
            score += 8 if "qwen" in text else 0
            score += 5 if "qwen38" in normalized else 0
            score += 2 if "qwen36" in normalized else 0
            score += 4 if "27b" in text or '"27"' in text else 0
            score += 3 if "fp8" in text else 0
            score += 1 if any((cfg.parent / n).exists() for n in ("tokenizer.json", "tokenizer_config.json")) else 0
            candidates.append((score, cfg.parent))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], -len(str(x[1]))), reverse=True)
    return str(candidates[0][1])


def launch_vllm(
    model_path: str,
    *,
    port: int = 8000,
    served_name: str = "arc3",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.92,
    timeout: float = 300.0,
    limit_mm_per_prompt: dict[str, int] | None = None,
    max_num_seqs: int | None = None,
    log_path: str | Path | None = None,
) -> subprocess.Popen:
    """Launch a local vLLM OpenAI server and wait until /v1/models responds.

    Multimodal limits are serialized with json.dumps into one argv element. This deliberately
    prevents the doubled-brace subprocess bug that invalidated S170 FINAL C. When log_path is
    supplied, server output is retained so Kaggle startup failures are diagnosable instead of
    disappearing into DEVNULL.
    """
    import requests

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_name,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--trust-remote-code",
        "--enable-prefix-caching",
    ]
    if limit_mm_per_prompt is not None:
        mm_arg = json.dumps(
            {str(k): int(v) for k, v in limit_mm_per_prompt.items()},
            separators=(",", ":"),
            sort_keys=True,
        )
        # Fail before model loading if the launch argument is ever malformed again.
        json.loads(mm_arg)
        cmd.extend(["--limit-mm-per-prompt", mm_arg])
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max(1, int(max_num_seqs)))])

    log_handle = None
    stdout: int | object = subprocess.DEVNULL
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = path.open("w", encoding="utf-8")
        stdout = log_handle

    proc = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if log_handle is not None:
        # Keep the parent-side handle alive for the lifetime of the subprocess.
        setattr(proc, "_arcangel_log_handle", log_handle)

    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            if log_handle is not None:
                log_handle.flush()
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            if requests.get(url, timeout=2).ok:
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    if log_handle is not None:
        log_handle.flush()
    raise TimeoutError(f"vLLM did not become ready at {url}")
