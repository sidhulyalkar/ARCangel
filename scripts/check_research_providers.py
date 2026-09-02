#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from arc3lab.arena.provider_transport import build_chat_payload, extract_message_text


_RETRYABLE = {"timeout", "rate_limited", "server_error"}


def _probe_once(row: dict[str, Any], timeout: float) -> dict[str, Any]:
    provider_id = str(row.get("id", ""))
    model = str(row.get("model", ""))
    env_name = str(row.get("api_key_env", ""))
    key = os.getenv(env_name, "")
    if not key:
        return {
            "provider_id": provider_id,
            "model": model,
            "ok": False,
            "classification": "missing_api_key",
            "detail": env_name,
        }
    started = time.monotonic()
    try:
        response = requests.post(
            f"{str(row['base_url']).rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=build_chat_payload(
                row,
                messages=({"role": "user", "content": "Reply with OK."},),
                temperature=0.0,
                max_tokens=32,
            ),
            timeout=timeout,
        )
        elapsed = round(time.monotonic() - started, 3)
        if not response.ok:
            text = response.text[-2000:]
            classification = "http_error"
            lowered = text.lower()
            if response.status_code in {401, 403}:
                classification = "auth_error"
            elif response.status_code == 404 or "model" in lowered and "not found" in lowered:
                classification = "model_unavailable"
            elif response.status_code == 429:
                classification = "rate_limited"
            elif response.status_code >= 500:
                classification = "server_error"
            return {
                "provider_id": provider_id,
                "model": model,
                "ok": False,
                "classification": classification,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed,
                "detail": text,
            }
        payload = response.json()
        text = extract_message_text(payload)
        return {
            "provider_id": provider_id,
            "model": model,
            "ok": True,
            "classification": "healthy",
            "elapsed_seconds": elapsed,
            "response_preview": text[:200],
        }
    except requests.Timeout as exc:
        return {
            "provider_id": provider_id,
            "model": model,
            "ok": False,
            "classification": "timeout",
            "detail": str(exc),
        }
    except Exception as exc:
        return {
            "provider_id": provider_id,
            "model": model,
            "ok": False,
            "classification": type(exc).__name__,
            "detail": str(exc)[:2000],
        }


def _probe(row: dict[str, Any], timeout: float, attempts: int) -> dict[str, Any]:
    effective_timeout = max(5.0, float(row.get("health_timeout_seconds", timeout)))
    history: list[dict[str, Any]] = []
    for attempt in range(1, max(1, attempts) + 1):
        result = _probe_once(row, effective_timeout)
        history.append(
            {
                "attempt": attempt,
                "ok": bool(result.get("ok")),
                "classification": result.get("classification"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )
        if result.get("ok"):
            result["attempts"] = history
            return result
        if str(result.get("classification")) not in _RETRYABLE:
            result["attempts"] = history
            return result
        if attempt < attempts:
            time.sleep(1.0)
    result["attempts"] = history
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe enabled ARCangel research LLM endpoints cheaply")
    ap.add_argument("--providers", default="configs/research-providers.nvidia-swarm.json")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    payload = json.loads(Path(args.providers).read_text(encoding="utf-8"))
    providers = [dict(row) for row in payload.get("providers", []) if bool(row.get("enabled", True))]
    if not providers:
        raise ValueError("provider config contains no enabled research endpoints")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(providers)))) as executor:
        futures = {
            executor.submit(_probe, row, args.timeout, max(1, int(args.attempts))): row
            for row in providers
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: str(row["provider_id"]))
    summary = {
        "status": "HEALTHY" if all(row["ok"] for row in results) else "UNHEALTHY",
        "enabled": len(results),
        "healthy": sum(bool(row["ok"]) for row in results),
        "providers": results,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "HEALTHY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
