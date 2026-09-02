from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc3lab.arena.provider_transport import build_chat_payload, extract_message_text
from arc3lab.arena.research_agents import ProviderSpec


def test_provider_options_are_merged_without_surrendering_core_fields() -> None:
    provider = {
        "model": "deepseek-ai/example",
        "request_body": {
            "top_p": 0.95,
            "chat_template_kwargs": {"thinking": False},
        },
    }
    payload = build_chat_payload(
        provider,
        messages=({"role": "user", "content": "hello"},),
        temperature=0.2,
        max_tokens=123,
    )

    assert payload["model"] == "deepseek-ai/example"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 123
    assert payload["stream"] is False
    assert payload["top_p"] == 0.95
    assert payload["chat_template_kwargs"] == {"thinking": False}


def test_provider_options_cannot_override_judge_owned_request_identity() -> None:
    with pytest.raises(ValueError, match="reserved fields"):
        build_chat_payload(
            {
                "model": "safe-model",
                "request_body": {"model": "other-model"},
            },
            messages=({"role": "user", "content": "hello"},),
            temperature=0.0,
            max_tokens=16,
        )


def test_message_extraction_supports_reasoning_only_endpoints() -> None:
    assert (
        extract_message_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "reasoned answer",
                        }
                    }
                ]
            }
        )
        == "reasoned answer"
    )
    assert (
        extract_message_text(
            {"choices": [{"message": {"content": "final answer", "reasoning": "hidden"}}]}
        )
        == "final answer"
    )


def test_nvidia_swarm_pins_model_specific_reasoning_modes() -> None:
    payload = json.loads(Path("configs/research-providers.nvidia-swarm.json").read_text())
    rows = {row["id"]: row for row in payload["providers"]}

    nemotron = rows["nvidia-nemotron35-lightning"]
    deepseek = rows["nvidia-deepseek-v4-pro"]
    kimi = rows["nvidia-kimi-k3"]

    assert nemotron["request_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert deepseek["request_body"]["chat_template_kwargs"]["thinking"] is False
    assert deepseek["health_timeout_seconds"] >= 120
    assert kimi["request_body"]["reasoning_effort"] == "low"

    spec = ProviderSpec.from_dict(kimi)
    assert spec.request_body["reasoning_effort"] == "low"
