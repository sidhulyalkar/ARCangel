from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


_RESERVED_REQUEST_KEYS = {"model", "messages", "temperature", "max_tokens"}


def build_chat_payload(
    provider: Mapping[str, Any],
    *,
    messages: Sequence[Mapping[str, Any]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build one OpenAI-compatible raw HTTP request without surrendering core fields.

    Provider-specific options are useful for heterogeneous NVIDIA NIM models whose
    reasoning controls differ.  They may add fields such as ``top_p``,
    ``reasoning_effort`` or ``chat_template_kwargs``, but may not replace the model,
    messages or caller-owned budget/temperature.
    """

    options = provider.get("request_body") or {}
    if not isinstance(options, Mapping):
        raise TypeError("provider request_body must be an object")
    forbidden = sorted(str(key) for key in options if str(key) in _RESERVED_REQUEST_KEYS)
    if forbidden:
        raise ValueError(f"provider request_body overrides reserved fields: {forbidden}")

    payload = deepcopy(dict(options))
    payload.update(
        {
            "model": str(provider["model"]),
            "messages": [dict(message) for message in messages],
            "temperature": float(temperature),
            "max_tokens": max(1, int(max_tokens)),
            "stream": False,
        }
    )
    return payload


def extract_message_text(payload: Mapping[str, Any]) -> str:
    """Return the best textual model output across OpenAI reasoning variants."""

    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") or {}

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise ValueError("response choice contains neither content nor reasoning text")
