from __future__ import annotations

import json
import tarfile
from pathlib import Path

from arc3lab.arena.evolution import ProposalTournament
from arc3lab.arena.metrics import suite_payload_to_result
from arc3lab.arena.orchestrator import ArenaOrchestrator
from arc3lab.arena.research_agents import ProviderSpec, ResearchProposal, ResearchSwarm
from arc3lab.arena.research_packet import ResearchPacketBuilder
from arc3lab.arena.schema import ArenaManifest, ArenaResult
from arc3lab.arena.scoring import aggregate_results, promotion_decision, score_result
from arc3lab.arena.splits import SplitRegistry


def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "test-swarm",
                "seeds": [1, 2],
                "leaderboard_control_id": "duck",
                "min_leaderboard_delta": 0.0,
                "weights": {"solve_rate": 1.0},
                "promotion": {
                    "min_validation_delta": 0.05,
                    "min_dev_delta": -0.01,
                    "min_validation_runs": 2,
                    "max_emergency_fraction": 0.02,
                    "max_failure_rate": 0.05,
                    "require_control": True,
                },
                "contestants": [
                    {"id": "duck", "family": "duck", "role": "external", "enabled": False},
                    {"id": "control", "family": "coding", "role": "control", "enabled": False},
                    {
                        "id": "challenger",
                        "family": "v012",
                        "role": "scientist",
                        "control_id": "control",
                        "command": [
                            "python",
                            "fake.py",
                            "{contestant}",
                            "{split}",
                            "{seed}",
                            "{result}",
                        ],
                    },
                ],
            }
        )
    )
    return path


def result(
    cid: str,
    split: str,
    seed: int,
    score: float,
    emergency: float = 0.0,
) -> ArenaResult:
    return ArenaResult(
        contestant_id=cid,
        split=split,
        seed=seed,
        metrics={"solve_rate": score, "emergency_fraction": emergency},
    )


def kaggle_result(cid: str, seed: int, score: float) -> ArenaResult:
    return ArenaResult(
        contestant_id=cid,
        split="kaggle",
        seed=seed,
        metrics={"official_score": score},
    )


def proposal(
    provider: str,
    role: str,
    hypothesis: str,
    *,
    split: str = "validation",
) -> ResearchProposal:
    return ResearchProposal(
        provider_id=provider,
        role_id=role,
        hypothesis=hypothesis,
        experiment="run ablation",
        target_metric="solve_rate",
        split=split,
        falsifier="no held-out improvement",
        implementation="small isolated patch",
        failure_mode="overfit",
        valid=True,
    )


def test_promotion_requires_repeatable_validation_control_delta(tmp_path: Path) -> None:
    manifest = ArenaManifest.load(manifest_file(tmp_path))
    rows = [
        result("control", "dev", 1, 0.40),
        result("control", "dev", 2, 0.40),
        result("control", "validation", 1, 0.40),
        result("control", "validation", 2, 0.40),
        result("challenger", "dev", 1, 0.48),
        result("challenger", "dev", 2, 0.47),
        result("challenger", "validation", 1, 0.51),
        result("challenger", "validation", 2, 0.50),
    ]
    aggregates = aggregate_results(rows, manifest)
    challenger = next(c for c in manifest.contestants if c.contestant_id == "challenger")
    decision = promotion_decision(challenger, aggregates, manifest)
    assert decision.promoted
    assert decision.validation_delta is not None and decision.validation_delta > 0.05


def test_emergency_ownership_blocks_promotion(tmp_path: Path) -> None:
    manifest = ArenaManifest.load(manifest_file(tmp_path))
    rows = [
        result("control", "validation", 1, 0.30),
        result("control", "validation", 2, 0.30),
        result("challenger", "validation", 1, 0.70, emergency=0.10),
        result("challenger", "validation", 2, 0.70, emergency=0.10),
    ]
    decision = promotion_decision(
        next(c for c in manifest.contestants if c.contestant_id == "challenger"),
        aggregate_results(rows, manifest),
        manifest,
    )
    assert not decision.promoted
    assert any("emergency" in reason for reason in decision.reasons)


def test_leaderboard_nomination_requires_actual_duck_kaggle_control(tmp_path: Path) -> None:
    manifest = ArenaManifest.load(manifest_file(tmp_path))
    lab = ArenaOrchestrator(manifest, tmp_path / "arena")
    lab.ledger.extend(
        [
            result("control", "dev", 1, 0.30),
            result("control", "dev", 2, 0.30),
            result("control", "validation", 1, 0.30),
            result("control", "validation", 2, 0.30),
            result("challenger", "dev", 1, 0.50),
            result("challenger", "dev", 2, 0.50),
            result("challenger", "validation", 1, 0.50),
            result("challenger", "validation", 2, 0.50),
            kaggle_result("challenger", 1, 2.30),
        ]
    )
    assert lab.promotion_queue() == ["challenger"]
    assert lab.leaderboard_queue() == []
    lab.ledger.append(kaggle_result("duck", 1, 2.23))
    queue = lab.leaderboard_queue()
    assert queue and queue[0]["contestant_id"] == "challenger"
    assert abs(float(queue[0]["kaggle_delta"]) - 0.07) < 1e-9


def test_record_kaggle_score_preserves_artifact_provenance(tmp_path: Path) -> None:
    manifest = ArenaManifest.load(manifest_file(tmp_path))
    lab = ArenaOrchestrator(manifest, tmp_path / "arena")
    row = lab.record_kaggle_score(
        contestant_id="duck",
        score=2.23,
        seed=9,
        source="copy-edit-run",
        artifact_sha256="abc123",
        runtime_seconds=8000,
    )
    assert row.metrics["official_score"] == 2.23
    assert row.metadata["artifact_sha256"] == "abc123"
    assert row.metadata["runtime_seconds"] == 8000.0


def test_split_registry_is_deterministic_and_hides_blind_ids() -> None:
    games = [f"game-{i}" for i in range(100)]
    one = SplitRegistry.build(games, salt="secret")
    two = SplitRegistry.build(reversed(games), salt="secret")
    assert one == two
    assert set(one.dev) | set(one.validation) | set(one.blind) == set(games)
    public = one.public_dict()
    assert "blind" not in public
    assert public["blind_count"] == len(one.blind)


def test_split_registry_keeps_all_three_splits_nonempty_for_three_games() -> None:
    registry = SplitRegistry.build(["a", "b", "c"], salt="secret")
    assert len(registry.dev) == 1
    assert len(registry.validation) == 1
    assert len(registry.blind) == 1


def test_research_packet_is_deterministic_and_excludes_blind(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "visible.md").write_text("visible")
    (repo / "blind-results.json").write_text('{"secret":true}')
    builder = ResearchPacketBuilder(repo)
    kwargs = {
        "experiment_id": "x",
        "scorecard": {"rankings": {"dev": [{"id": "a"}], "blind": [{"id": "secret"}]}},
        "include_paths": ["visible.md", "blind-results.json"],
    }
    first = tmp_path / "a.tar.gz"
    second = tmp_path / "b.tar.gz"
    assert builder.build(first, **kwargs) == builder.build(second, **kwargs)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert "repo/visible.md" in names
        assert all("blind-results" not in name for name in names)
        handle = archive.extractfile("arena/scorecard.json")
        assert handle is not None
        scorecard = json.loads(handle.read())
        assert "blind" not in scorecard["rankings"]


def test_research_proposal_requires_explicit_falsifier_contract() -> None:
    good = ResearchProposal.from_text(
        "p",
        "scientist",
        json.dumps(
            {
                "hypothesis": "exact history helps",
                "experiment": "ablate retrieval",
                "target_metric": "solve_rate",
                "split": "validation",
                "falsifier": "no improvement",
                "implementation": "add selective retrieval",
                "failure_mode": "context overload",
            }
        ),
    )
    bad = ResearchProposal.from_text("p", "scientist", '{"hypothesis":"sounds nice"}')
    assert good.valid
    assert not bad.valid


def test_research_swarm_plans_provider_role_cross_product() -> None:
    swarm = ResearchSwarm(
        [
            ProviderSpec(
                provider_id="model-a",
                base_url="https://example.invalid/v1",
                model="a",
                api_key_env="NO_KEY",
                roles=("scientist", "red_team"),
            )
        ]
    )
    calls = swarm.plan()
    assert [(call.provider_id, call.role_id) for call in calls] == [
        ("model-a", "scientist"),
        ("model-a", "red_team"),
    ]


def test_proposal_tournament_rejects_blind_and_kaggle_requests() -> None:
    tournament = ProposalTournament(
        [
            proposal("a", "scientist", "good", split="validation"),
            proposal("b", "planner", "leak", split="blind"),
            proposal("c", "vision", "leaderboard chase", split="kaggle"),
        ]
    )
    assert [row.hypothesis for row in tournament.eligible()] == ["good"]


def test_proposal_tournament_deduplicates_and_preserves_role_diversity() -> None:
    tournament = ProposalTournament(
        [
            proposal("a", "scientist", "Selective exact memory helps"),
            proposal("b", "scientist", " selective   exact memory HELPS "),
            proposal("c", "planner", "Compile plans after model validation"),
            proposal("d", "vision", "Choose representation adaptively"),
        ]
    )
    selected = tournament.select(max_proposals=3)
    assert len(selected) == 3
    assert len({row.role_id for row in selected}) == 3
    assert sum("memory" in row.hypothesis.lower() for row in selected) == 1


def test_exchange_brief_is_only_measured_public_evidence() -> None:
    tournament = ProposalTournament([proposal("a", "scientist", "test memory")])
    brief = tournament.exchange_brief(
        {
            "rankings": {
                "validation": [
                    {"contestant_id": "D-v012", "robust_score": 0.5, "mean_score": 0.52}
                ]
            },
            "promotion_decisions": [
                {
                    "contestant_id": "E-v012-lite",
                    "promoted": False,
                    "reasons": ["validation delta too small"],
                }
            ],
        }
    )
    assert "D-v012" in brief
    assert "validation delta too small" in brief
    assert "test memory" in brief
    assert "blind" not in brief.lower()


def test_suite_metrics_capture_efficiency_and_scientific_health() -> None:
    payload = {
        "elapsed_seconds": 12,
        "games": [
            {
                "game_id": "g1",
                "state": "WIN",
                "levels_completed": 2,
                "actions": 40,
                "model_calls": 10,
                "error": None,
                "deadline_exhausted": False,
            },
            {
                "game_id": "g2",
                "state": "NOT_FINISHED",
                "levels_completed": 1,
                "actions": 50,
                "model_calls": 20,
                "error": None,
                "deadline_exhausted": False,
            },
        ],
        "diagnostics": {
            "expectation_checks": 10,
            "expectation_mismatches": 1,
            "hypothesis_tests": 8,
            "hypothesis_test_failures": 2,
            "emergency_transport_fallbacks": 1,
        },
    }
    converted = suite_payload_to_result(
        payload,
        contestant_id="v012",
        split="dev",
        seed=1,
    )
    assert converted.metrics["solve_rate"] == 0.5
    assert converted.metrics["prediction_accuracy"] == 0.9
    assert converted.metrics["falsification_health"] == 0.75
    assert 0 < converted.metrics["action_efficiency"] < 1


def test_partial_game_failure_is_penalized_without_destroying_run_information(
    tmp_path: Path,
) -> None:
    manifest = ArenaManifest.load(manifest_file(tmp_path))
    payload = {
        "games": [
            {
                "game_id": "good",
                "state": "WIN",
                "levels_completed": 1,
                "actions": 20,
                "model_calls": 5,
                "error": None,
                "deadline_exhausted": False,
            },
            {
                "game_id": "bad",
                "state": "ERROR",
                "levels_completed": 0,
                "actions": 0,
                "model_calls": 0,
                "error": "boom",
                "deadline_exhausted": False,
            },
        ],
        "diagnostics": {},
    }
    converted = suite_payload_to_result(
        payload,
        contestant_id="challenger",
        split="dev",
        seed=1,
    )
    assert converted.status == "ok"
    assert converted.metrics["failure_rate"] == 0.5
    assert score_result(converted, manifest.weights) > -1.0


def test_plan_expands_tokens_and_skips_existing_runs(tmp_path: Path) -> None:
    manifest = ArenaManifest.load(manifest_file(tmp_path))
    lab = ArenaOrchestrator(manifest, tmp_path / "arena")
    runs = lab.plan(splits=["dev"])
    assert len(runs) == 2
    first = runs[0]
    assert "challenger" in first.command
    assert "dev" in first.command
    lab.ledger.append(result("challenger", "dev", first.seed, 0.5))
    remaining = lab.plan(splits=["dev"])
    assert len(remaining) == 1
