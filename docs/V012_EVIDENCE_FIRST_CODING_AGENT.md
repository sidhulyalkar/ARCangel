# V012 — Evidence-First Coding Agent

## Status

V012 is a clean policy-level architectural reset after S190A / V011 scored **0.17** on the ARC Prize 2026 ARC-AGI-3 Kaggle leaderboard.

That score is treated as falsification evidence, not as a tuning signal. The V009 → V010 → V011 lineage remains in the repository for research continuity, but V012 deliberately removes its main authority structure.

## Why V011 was rejected

S190A ran successfully, used the intended Qwen3.8-27B-FP8 model family, and still scored near baseline. The failure therefore cannot be explained away as deployment plumbing.

The key architectural problems were:

1. **Semantic bottleneck.** ARCangel generated and ranked action candidates using an ontology dominated by actors, spatial progress, target relations, click affordances, risk and information value before the model decided what the game was.
2. **Fallback authority.** When the model did not provide an accepted action, inherited policies still spent real environment actions. On an efficiency-weighted benchmark, a legal but ungrounded action is not safe.
3. **Lossy memory.** Short reflection summaries and recent-outcome packets were easier to consume but could hide exactly the older transition that falsified a current rule.
4. **Scientific vocabulary without scientific enforcement.** Modes named BUILD_MODEL, TEST_HYPOTHESIS and REPAIR_MODEL did not require a hypothesis to survive all compatible historical evidence.
5. **Bureaucratic model contract.** Qwen spent generation budget filling a large ontology instead of choosing what computation or experiment would resolve the uncertainty.
6. **Wrong promotion metrics.** Parser legality, randomized contract tests, and fallback legality proved software properties. They did not prove that the architecture solved unseen games.

## External evidence informing the reset

The first ARC-AGI-3 milestone winner, Tufa Labs' Duck, uses a lightweight Python-driven harness and lets the model choose which representation and computation matter. ARC Prize's milestone write-up explicitly notes that Tufa found hand-built tools could hurt. The current public Qwen3.8 Duck notebook has reached 2.23 on Kaggle using RTX Pro 6000, ARC3 vLLM H100 Wheelhouse V3 and Qwen3.8 27B FP8.

Recent public research systems point in the same direction at a different compute/model scale:

- **Retrodict:** preserve every frame and require proposed rules to retrodict recorded history before spending live actions.
- **Tycho:** allow the actor to delegate an executable world model that is checked against experience before planning through it.

V012 does not copy game-specific rules from any public solver. It extracts the architecture-general lesson: **evidence should constrain the theory; the harness should not constrain the abstraction.**

## Core invariant

> The immutable interaction ledger is ground truth. Everything else is a provisional interpretation.

The agent therefore separates two stores:

### Evidence store

Append-only exact information:

- every numeric frame;
- every action;
- exact before/after frame access;
- level and step;
- changed-cell statistics;
- level completion, game-over and win events;
- optional deterministic component views;
- action-conditioned transition queries.

### Scientific workspace

Mutable theory:

- provisional hypotheses;
- history-consistent hypotheses;
- validated rules;
- falsified rules;
- open questions;
- level notes;
- optional executable world-model source plus its validation receipt.

The theory can be revised. The evidence cannot.

## Control loop

```text
observe exact frame
      ↓
append immutable evidence
      ↓
is a queued action still supported?
 ├─ yes → execute next expected action
 └─ no
      ↓
ask Qwen what remains unexplained
      ↓
can recorded history answer it?
 ├─ yes → write Python query / falsification test
 │          ↓
 │       revise theory for free
 │          ↓
 │       repeat if needed
 └─ no
      ↓
choose ONE discriminating live probe
      ↓
observe consequence
      ↓
update evidence and falsify/confirm theory
      ↓
when mechanics are grounded:
write/search a short plan
      ↓
attach expectation to every action
      ↓
execute until expectation mismatch
      ↓
stop immediately and repair theory
```

There is no normal candidate-registry arbitration layer in this loop.

## Model interface

The model receives:

- the current rendered grid;
- lossless ASCII for the current grid;
- legal action ids;
- latest empirical diff;
- compact persistent workspace;
- a description of the Python evidence API.

It does **not** receive a ranked semantic action menu.

The compact response supports four modes:

- `ANALYZE` — query existing evidence without spending an action;
- `PROBE` — one live discriminating experiment;
- `EXECUTE` — one or more grounded actions;
- `REPAIR` — revise theory after contradiction.

## History falsification

A hypothesis can include `test_python`.

The sandbox runs the test against the current evidence API and expects a compact result like:

```python
result = {
    "consistent": True,
    "support": 4,
    "contradictions": 0,
}
```

A contradiction marks the hypothesis falsified and sharply reduces its authority.

This is intentionally stronger than confidence bookkeeping. A confident model does not get to override contradictory recorded evidence.

## Executable world model

V012 optionally stores `world_model_code` in the persistent workspace.

The program must execute against the complete evidence context and report at least:

```python
result = {
    "checked": N,
    "mismatches": M,
}
```

It is considered validated only when `checked > 0` and `mismatches == 0`.

The model is not required to build a world model on every game. It is an escalation mechanism for environments where explicit transition modeling unlocks search.

## Plan authority

A multi-action plan is accepted only when all of the following hold:

1. the model explicitly marks the plan reliable;
2. every queued action carries an expectation;
3. either a mechanic has survived at least two compatible historical observations, or an executable world model has passed its historical validation.

Otherwise V012 truncates the proposal to one action.

## Expectation checking

Each queued action may predict:

- whether the board should change;
- expected level delta;
- whether game-over should occur;
- whether win should occur;
- minimum meaningful changed cells;
- exact next signature when known.

After the environment responds, V012 compares observation to expectation. Any mismatch:

- clears the plan queue;
- forces re-reasoning;
- records an open repair question;
- prevents stale execution from consuming more score.

## Fallback policy

V012 has **no normal heuristic fallback authority**.

If the model/parse path is exhausted, a minimal emergency transport fallback exists solely to keep the runner contract alive. It is tracked separately as `emergency_transport_fallbacks` and is a promotion failure if it owns a meaningful fraction of actions.

This distinction matters. V011's fallback was part of cognition. V012's emergency fallback is treated as an infrastructure defect signal.

## Useful legacy code that survives

The reset is about authority, not deleting working infrastructure.

The following remain valuable as queryable tools:

- exact grids and frame diffs;
- connected components;
- object signatures;
- transition memory;
- action statistics;
- spatial/topological helpers;
- prediction memory;
- Kaggle/vLLM runtime hardening;
- scorecard runner and telemetry.

A connected component is now evidence the model may inspect, not a declaration that the component is the semantic object the game is about.

## S210A runtime profile

Primary V012 packaged build:

`S210A-V012-EVIDENCE-FIRST-QWEN38-20260827`

Target Kaggle environment:

- RTX PRO 6000;
- Internet OFF;
- ARC Prize 2026 competition input;
- ARC3 vLLM H100 Wheelhouse V3;
- Qwen3.8 27B FP8 Repacked.

Current safety/runtime budget:

- 28 workers;
- 900 action hard ceiling per game;
- 200 model calls per game;
- 96 Python calls per game;
- 4 model/tool reasoning rounds per decision;
- 16 max queued actions;
- 7-hour shared play budget;
- 130-minute per-game hard budget;
- 768 generation-token ceiling.

These are ceilings, not desired usage. Good play should collapse uncertainty and use far less.

## Experiment ladder

V012 is not promoted merely because it compiles.

### S200 — public control

First reproduce the current Duck + Qwen3.8 lane through our runtime. The point is not to claim originality. It establishes that our deployment/model/action plumbing can reach the current public ~2-point regime.

### S210A — V012 evidence-first

Run the packaged architecture and compare against the control using:

- levels solved;
- leaderboard score;
- environment actions per completed level;
- model calls per completed level;
- number of historical queries;
- hypotheses history-confirmed/falsified;
- expectation mismatches;
- queued plan utilization;
- emergency transport ownership.

### Subsequent ablations

Only one major capability should change per experiment:

1. evidence ledger + direct model authority;
2. + history falsification;
3. + persistent cross-level playbook;
4. + optional executable world model;
5. + adaptive compute allocation.

If an addition reduces score, remove it even if it looks intellectually attractive.

## Promotion standard

A V012 submission is credible only if:

- model-authored actions overwhelmingly dominate emergency actions;
- the notebook/runtime path is fully qualified;
- the architecture materially exceeds S190A's 0.17;
- ideally it first beats the public-control reproduction on at least some game families;
- improvements are visible in task-level outcomes, not merely telemetry.

The project objective is no longer to build the most elaborate ARC agent.

It is to build the **smallest evidence-grounded system that reliably turns a few expensive interactions into a reusable executable understanding of a new world.**
