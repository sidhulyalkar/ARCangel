# Canonical Kaggle submission candidates

The implementation source of truth lives under `src/arc3lab/`. This file tracks the exact generated notebook artifacts that should be imported into Kaggle. Do not submit an older notebook whose display title merely looks similar.

Repository baseline for the current release ladder: `4f58bde50ae3160b446a7c5748787d3b951af5ab` (V005 predictive-state merge).

## S115 FINAL C — causal fallback

- build: `S115-FINAL-20260822-C`
- notebook SHA-256: `76bab67d0bd1d97ae5bd8336e1af2df994eaa8cf666f7f1d4b098be7ff2bb5b5`
- primary diff: D110R2-promoted `EffectPosteriorPolicy` beneath the V004 coding/Qwen campaign
- submission order: **1**
- expected first log marker: `ARCANGEL SUBMISSION BUILD: S115-FINAL-20260822-C`

## S120 FINAL C — h2 predictive state

- build: `S120-FINAL-20260822-C`
- notebook SHA-256: `f5db889630e94b8629f216dbf83404d65d92cbae8ef3a93cf3a36a60e2c7392b`
- primary diff: D210R2-promoted h2 predictive-state verification + persistent goals + prediction-error queue invalidation, retaining S115's causal fallback
- D210 architecture gate: **PASSED**
- submission order: **2**
- expected first log marker: `ARCANGEL SUBMISSION BUILD: S120-FINAL-20260822-C`

## Required Kaggle inputs

1. ARC Prize 2026 - ARC-AGI-3
2. ARC3 vLLM H100 Wheelhouse V3
3. vrfai Qwen3.6 27B FP8 HF Snapshot

Settings:

- Accelerator: **NVIDIA RTX PRO 6000**
- Internet: **OFF**

## Mandatory Save & Run acceptance markers

The exact saved notebook is eligible for submission only if its log contains all of:

1. the exact `ARCANGEL SUBMISSION BUILD` above;
2. `ARC TOOLKIT PASS`;
3. `VLLM PACKAGE PASS`;
4. `Top mounted HF candidates:`;
5. `MODEL INPUT PASS: <build id>`;
6. `VLLM SERVER PASS`;
7. `FULL INFRASTRUCTURE PREFLIGHT PASS`;
8. a dynamic public-game smoke with zero harness errors;
9. `SAVE/RUN VALIDATION PASS: <build id>`.

The notebook must **not** contain the retired detector `MODEL_SCORE < 45`. Kaggle may mount the intended Qwen snapshot at a hyphenated path such as `vrfai-qwen3-6-27b-fp8-hf-snapshot`; FINAL C normalizes punctuation and validates Qwen + 27B + FP8 + safetensors evidence.

## Experimental provenance

D110R2 promoted effect-posterior fallback behavior and rejected repeated later-level re-probing, aggressive global dead-action suppression, and a stronger static click prior.

D210R2 promoted h2 temporal predictive state. Mean repeated-key consistency was ~0.9918; held-out exact transition accuracy on covered contexts was ~0.9993 at ~0.766 coverage. Five discovered level-1 plans verified 5/5 with a median action ratio to human of ~0.182. Cross-level effect semantics transferred strongly, but level-2 plans did not automatically transfer, so ARCangel carries mechanics while re-inferring each new level's goal/plan.

See `docs/SUBMISSION_RUNBOOK.md` and `docs/RELEASE_STATUS_2026-08-22.md` for the full release checklist and research state.
