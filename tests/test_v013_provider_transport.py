from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc3lab.arena.provider_transport import build_chat_payload, extract_message_text
from arc3lab.arena.research_agents import ProviderSpec


def test_provider_options_are_merged_without_surrendering_core_fields() -> None:
    provider = {
        "model": "poolside/example",
        "request_body": {"top_p": 0.95},
    }
    payload = build_chat_payload(
        provider,
        messages=({"role": "user", "content": "hello"},),
        temperature=0.2,
        max_tokens=123,
    )

    assert payload["model"] == "poolside/example"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 123
    assert payload["stream"] is False
    assert payload["top_p"] == 0.95


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


def test_nvidia_swarm_pins_three_distinct_operational_families() -> None:
    payload = json.loads(Path("configs/research-providers.nvidia-swarm.json").read_text())
    rows = {row["id"]: row for row in payload["providers"]}

    nemotron = rows["nvidia-nemotron35-lightning"]
    laguna = rows["nvidia-laguna-xs21"]
    kimi = rows["nvidia-kimi-k3"]

    assert nemotron["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert nemotron["request_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert laguna["model"] == "poolside/laguna-xs-2.1"
    assert set(laguna["roles"]) >= {"scientist", "planner", "memory", "generalization"}
    assert laguna["health_timeout_seconds"] <= 60
    assert kimi["model"] == "moonshotai/kimi-k3"
    assert kimi["request_body"]["reasoning_effort"] == "low"

    families = {row["model"].split("/", 1)[0] for row in payload["providers"]}
    assert families == {"nvidia", "poolside", "moonshotai"}

    spec = ProviderSpec.from_dict(laguna)
    assert spec.request_body["top_p"] == 0.95


def test_nvidia_patch_worker_uses_operational_coding_provider() -> None:
    worker = json.loads(Path("configs/experiment-workers.nvidia.json").read_text())["workers"][0]
    command = worker["command"]
    assert "nvidia-laguna-xs21" in command
    assert "nvidia-deepseek-v4-pro" not in command


def test_research_workflow_uses_required_two_family_active_quorum() -> None:
    text = Path(".github/workflows/swarm-research.yml").read_text()

    assert "--min-healthy 2" in text
    assert "--require-provider nvidia-laguna-xs21" in text
    assert "--active-output artifacts/arena/v013/research-providers.active.json" in text
    assert "--providers artifacts/arena/v013/research-providers.active.json" in text
    assert "research_quorum_met" in text
    assert "len(json.loads(active.read_text()).get('providers', [])) < 2" in text


def test_health_checker_distinguishes_degraded_quorum_from_unhealthy() -> None:
    text = Path("scripts/check_research_providers.py").read_text()

    assert 'return "DEGRADED", missing_required' in text
    assert 'return "UNHEALTHY", missing_required' in text
    assert '"research_quorum_met": status in {"HEALTHY", "DEGRADED"}' in text
    assert 'required_provider_ids' in text
    assert 'unhealthy_provider_ids' in text
