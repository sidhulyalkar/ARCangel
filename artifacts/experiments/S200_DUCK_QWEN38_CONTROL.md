# S200 — Duck / Qwen3.8 public control

Status: **CONTROL DEFINED / KAGGLE COPY-&-RUN PENDING**

## Purpose

S200 is deliberately not an ARCangel invention. It is the external control for the V012 architectural reset.

The question is:

> Can the exact current public Duck/Qwen3.8 notebook family reach the multi-point public leaderboard regime on our Kaggle account/hardware setup before we attribute another low score to model serving, package plumbing, or competition runtime?

Without this control, a V012 score remains ambiguous.

## Reference artifact

Current public reference at time of definition:

- Kaggle notebook: `FOYSAL / LB-9 arc3 duck v12 with Qwen 3.8 27B`
- observed public score: **2.23**
- runtime: approximately **2h 19m 42s**
- accelerator: **GPU RTX Pro 6000**
- notebook license: **Apache 2.0**

Required reference inputs shown by the public notebook:

1. ARC Prize 2026 - ARC-AGI-3 competition input;
2. `ARC3 vLLM H100 Wheelhouse V3`;
3. `TAAF Kaggle Source Bundle`;
4. `Qwen3.8 27B FP8 Repacked`.

Reference URL:

`https://www.kaggle.com/code/foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b`

## Why use the public notebook directly

The control should minimize changes. Reimplementing Duck inside ARCangel would contaminate the control with our own packaging, prompts, API wrappers, scheduling, context handling, and action semantics.

Therefore S200 should be created using Kaggle **Copy & Edit** from the current public notebook version and run with its intended inputs/settings. The result should be recorded verbatim before ARCangel-specific modifications are introduced.

## What S200 does and does not prove

If S200 lands in the same broad regime as the public 2.23 result, it demonstrates that:

- Qwen3.8 27B FP8 is capable of substantially better competition performance than S190A on this runtime family;
- RTX Pro 6000 + current vLLM wheelhouse/model packaging is not inherently trapping us near 0.2;
- V011's failure is primarily behavioral/architectural rather than a generic model-serving ceiling.

It does **not** prove that 2.23 is perfectly reproducible. The benchmark contains stochastic/runtime/model-generation effects and the public notebook may evolve. We care about the regime, not an exact decimal match.

If S200 also scores near baseline, stop V012 leaderboard experiments and diagnose the shared environment/model/runtime path before drawing architectural conclusions.

## Receipt to capture

Record:

- copied notebook source/version ID;
- exact notebook SHA if available after download;
- Qwen model input/version;
- wheelhouse/source bundle versions;
- accelerator;
- total runtime;
- successful/failed game count;
- submission file creation;
- public leaderboard score.

## Decision rule

### Control healthy

Broadly multi-point performance, approximately in the current Duck/Qwen3.8 public regime.

Proceed to S210A V012 and compare task-level behavior.

### Control degraded

Substantially below the public regime but clearly above S190A.

Investigate version/runtime deltas and repeat the control before interpreting V012.

### Control failed

Near-baseline score or runtime failure.

Do **not** spend a V012 leaderboard slot. Repair the common serving/competition pipeline first.
