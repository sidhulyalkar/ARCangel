from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def discover_model_path(roots: Iterable[str] = ("/kaggle/input",)) -> str | None:
    """Find a plausible local Qwen model without relying on a Kaggle dataset slug."""
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
            text = (str(cfg.parent) + " " + json.dumps(data)[:2000]).lower()
            score = 0
            score += 5 if "qwen" in text else 0
            score += 3 if "27b" in text or '"27"' in text else 0
            score += 2 if "fp8" in text else 0
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
) -> subprocess.Popen:
    """Launch a local vLLM OpenAI server and wait until /v1/models responds."""
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
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=os.environ.copy())
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}")
        try:
            if requests.get(url, timeout=2).ok:
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"vLLM did not become ready at {url}")
