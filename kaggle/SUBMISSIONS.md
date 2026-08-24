# Canonical Kaggle submission candidates

The implementation source of truth lives under `src/arc3lab/`. This file tracks the exact generated notebook artifacts that should be imported into Kaggle. **Never submit an older notebook whose title merely looks similar.**

## Current one-per-day ladder

| Order | Build | Hypothesis | Status |
|---|---|---|---|
| 1 | S115 FINAL F | D110R2 causal effect-posterior fallback | baseline / deployed candidate |
| 2 | S120G | canonical V005 h2 predictive state + goals | current predictive baseline |
| 3 | S135 FINAL B | V006 spatial intelligence | next candidate; local qualified, RTX Save & Run required |

The competition workflow currently gives us one private submission per day, so each slot must answer one primary architectural question.

## Runtime contract shared by current candidates

Required Kaggle inputs:

1. **ARC Prize 2026 - ARC-AGI-3**
2. **ARC3 vLLM H100 Wheelhouse V3**
3. **vrfai Qwen3.6 27B FP8 HF Snapshot**

Settings:

- Accelerator: **NVIDIA RTX PRO 6000**
- Internet: **OFF**

Current hardened serving contract:

- punctuation-normalized model discovery accepts the mounted `vrfai-qwen3-6-27b-fp8-hf-snapshot` path;
- resolve the live NVIDIA `libcuda.so.1` and create a writable unversioned linker alias;
- require a real host `c++ ... -lcuda` linker self-test before vLLM startup;
- set `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` because that FlashInfer SM120 FP8 linear autotuner segfaulted in the Kaggle runtime;
- successful validated startup selects `CutlassFP8ScaledMMLinearKernel`;
- Save & Run writes `/kaggle/working/submission.parquet` so Kaggle exposes the notebook version for submission;
- the hidden rerun uses the official `ARC-AGI-3-Agents` framework against `gateway:8001`, which records the actual hidden actions and writes the scored parquet.

## S120G — canonical V005 predictive-state baseline

Build:

`S120-CANONICAL-20260823-G`

Notebook SHA-256:

`4b0e4bf6923ae3ee6d69ddae13e525d4f29dcaa5ddf1a469c5b47fbefb0fc04b`

Primary question:

> Does canonical h2 temporal prediction + persistent goal context improve private transfer over the causal-fallback baseline?

S120G corrects source drift found in an earlier generated S120 artifact so the notebook actually exposes the repository V005 goal and predictive-memory context to Qwen.

## S135 FINAL B — V006 spatial intelligence

Build:

`S135-FINAL-20260823-B`

Notebook SHA-256:

`b070dc1d412c49b32f8d91c63c606f21370ad6cd0ad93f292446130868870ddf`

Embedded source bundle SHA-256:

`0055eaae67277191ead3499b41b5e00181cb1c60775e8b9ae50cb0fae6c90a9f`

Primary question:

> Once ARCangel has causally identified the controlled object and translational action semantics, does exact 360-degree geometry plus guarded shortest-path execution improve level completion and action efficiency?

S135 adds:

- eight-direction spatial relations and raycasts;
- actor-relative geometry and line of sight;
- actor-footprint free-space topology;
- causal controlled-object inference;
- learned action displacement vectors;
- camera/global-motion rejection;
- translation-purity and duplicate-actor gates;
- exact shortest-action BFS;
- Qwen-requested route compilation;
- geometry recheck before every queued route action;
- actor-anchor verification after each route action;
- spatial request/compile/action/mismatch telemetry.

### S135 local qualification

- Python source compile: PASS
- canonical V005 + V006 regression suite: **31 passed**
- randomized shortest-path audit: **1000/1000 matched** independent reference search
- synthetic notebook spatial preflight: embedded

This is **not** enough to spend the private slot. The exact notebook must still pass Kaggle Save & Run.

## Mandatory Save & Run acceptance markers for S135

The exact saved notebook is eligible for submission only if its log contains all of:

1. `ARCANGEL SUBMISSION BUILD: S135-FINAL-20260823-B`
2. `V006 SPATIAL ENGINE SYNTHETIC PREFLIGHT PASS`
3. `ARC TOOLKIT PASS`
4. `VLLM PACKAGE PASS`
5. `MODEL INPUT PASS: S135-FINAL-20260823-B`
6. `CUDA DRIVER RUNTIME LOAD PASS`
7. `CUDA DRIVER LINKER PASS`
8. `VLLM SERVER PASS`
9. `FLASHINFER FP8 LINEAR KERNEL DISABLE PASS`
10. `FULL INFRASTRUCTURE PREFLIGHT PASS`
11. a dynamic public-game smoke with zero harness/model errors
12. `DUMMY SUBMISSION PARQUET PASS`
13. `SAVE/RUN VALIDATION PASS: S135-FINAL-20260823-B`
14. `SUBMISSION FILE READY: /kaggle/working/submission.parquet`

The notebook must **not** contain the retired detector `MODEL_SCORE < 45`.

## Promotion rule

Do not merge/promote V006 solely because the public smoke is green. After RTX validation, use the one-per-day private result plus telemetry:

- if S135 gains materially, promote spatial intelligence and begin executable spatial goal predicates;
- if flat but spatial plans were rarely requested/compiled, treat it as a coverage/gating result rather than immediately deleting the mechanism;
- if plans compiled often but mismatches were high, strengthen control/geometry abstraction before further automation;
- if it regresses despite low mismatch rate, inspect extra model/context/action overhead before stacking new features.

See `docs/V006_SPATIAL_INTELLIGENCE.md` and `docs/EXPERIMENT_PLAN.md`.
