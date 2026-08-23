# Next Submission Ladder

This document records the post-FINAL-F submission sequence and the source-drift audit that preceded it.

## Deployment baseline

The working deployment stack now includes:

- RTX PRO 6000 detection;
- offline ARC toolkit installation;
- Qwen3.6 27B FP8 model discovery;
- CUDA driver linker alias / `-lcuda` self-test;
- `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` so Blackwell uses the working CUTLASS FP8 kernel;
- localhost vLLM generation smoke;
- official ARC-AGI-3 gateway rerun contract;
- required `/kaggle/working/submission.parquet` Save & Run output.

## Source-drift finding

The earlier `S120-FINAL-20260823-F` notebook is useful as a deployment and partial-predictive receipt, but an audit found that its embedded prompt did not expose the canonical V005 `PERSISTENT GOAL HYPOTHESES` and `TEMPORAL PREDICTIVE MEMORY` sections to Qwen. Its embedded `CodingPolicy` also predated the exact repository behavior for configurable `predictive_history_depth` and actor-gated world-model delegation accounting.

Therefore it should not be treated as the exact canonical V005 comparison point.

## S120G — canonical V005 refresh

Build: `S120-CANONICAL-20260823-G`

Notebook SHA-256:

`4b0e4bf6923ae3ee6d69ddae13e525d4f29dcaa5ddf1a469c5b47fbefb0fc04b`

Purpose: restore the exact V005-shaped prompt/context and h2 configuration while leaving the validated FINAL-F deployment stack unchanged.

Primary question:

> Does the full canonical V005 predictive/goal context improve over the S115 causal-fallback control?

## S125 — expected-effect verification

Build: `S125-FINAL-20260823-C`

Notebook SHA-256:

`05c529fbafabec6c5c44bdeee97fafdea65f19e1fe7b614ce627c8671418134e`

Verified source bundle SHA-256:

`434bcdf97b101b50d3476679d65f7d0787d63b9eb2e93adb84281881051fb49a`

One primary change from S120G:

- each model-proposed action may declare `expected_effect` in `{dead, change, level, unknown}`;
- after the real transition, an explicit mismatch invalidates the remaining action queue and forces immediate repair;
- queue admission otherwise remains unchanged.

Primary question:

> Does explicit per-action effect verification catch wrong execution hypotheses before they cascade into scored action waste?

## S130 — falsifiable plan queue

Build: `S130-FINAL-20260823-C`

Notebook SHA-256:

`68e7906ff17bbec17031cbbd5bcc2c65cb37af9e1962796656ee3c52c27e1c3f`

One primary change from S125:

- a multi-action `plan_reliable` queue is retained only when every queued step has a non-`unknown` expected effect;
- otherwise ARCangel executes only the first action and re-reasons.

Primary question:

> Do multi-action plans generalize better when every queued step is explicitly falsifiable?

## Submission order

1. `S115-FINAL-20260823-F`
2. `S120-CANONICAL-20260823-G`
3. `S125-FINAL-20260823-C`
4. `S130-FINAL-20260823-C`

Do not submit S125 or S130 before its own RTX Save & Run reaches all infrastructure, model, dynamic ARC smoke, and `submission.parquet` validation markers.

## Interpretation gate

- If S120G improves over S115, retain canonical h2 + goal context.
- If S125 improves or is neutral while reducing stale queued execution, promote expected-effect verification.
- If S130 improves action efficiency without hurting later-level continuation, promote falsifiable queues.
- If S130 regresses, keep S125 and treat model-declared `plan_reliable` as useful without requiring every future step to be semantically predicted.

## After S130

Do not immediately stack another heuristic. The next branch should use real hidden receipts to choose between:

1. goal induction / goal contradiction repair;
2. actor-gated persistent executable controller;
3. per-game compute allocation within the constraints of the official gateway framework.

The choice should be driven by observed failure telemetry, not public-game anecdotes.
