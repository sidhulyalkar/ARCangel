# ARCangel Kaggle submission runbook

This is the release checklist for the post-D110/D210 ladder. Never submit an older imported notebook merely because the title looks similar.

## Required Kaggle inputs

Attach exactly:

1. **ARC Prize 2026 - ARC-AGI-3**
2. **ARC3 vLLM H100 Wheelhouse V3**
3. **vrfai Qwen3.6 27B FP8 HF Snapshot**

Settings:

- **Accelerator:** NVIDIA RTX PRO 6000
- **Internet:** OFF

The intended model may mount at a hyphenated path such as `vrfai-qwen3-6-27b-fp8-hf-snapshot`. FINAL C normalizes punctuation and validates Qwen + 27B + FP8 + safetensors evidence. The retired `MODEL_SCORE < 45` detector must not appear anywhere in the imported notebook.

## S115 FINAL C — causal fallback

Build ID: `S115-FINAL-20260822-C`

Notebook SHA-256: `76bab67d0bd1d97ae5bd8336e1af2df994eaa8cf666f7f1d4b098be7ff2bb5b5`

Primary hypothesis: D110R2's `EffectPosteriorPolicy` improves the V004/Qwen campaign while preserving the higher-level campaign shell.

Recommended private-submission order: **first**.

## S120 FINAL C — h2 predictive state

Build ID: `S120-FINAL-20260822-C`

Notebook SHA-256: `f5db889630e94b8629f216dbf83404d65d92cbae8ef3a93cf3a36a60e2c7392b`

Primary hypothesis: add D210R2-promoted h2 predictive verification, persistent goal hypotheses, prediction-error queue invalidation, and actor-visible temporal memory while retaining S115's causal fallback.

D210R2 architecture gate: **PASSED**.

Recommended private-submission order: **second**.

## Save & Run acceptance criteria

A saved version is eligible for competition submission only if all of the following are true:

1. The first execution cell prints the exact `ARCANGEL SUBMISSION BUILD` above.
2. CUDA reports an RTX PRO 6000 with the expected large-memory device.
3. `ARC TOOLKIT PASS` prints after the mounted ARC wheels install.
4. Embedded ARCangel source imports successfully.
5. `VLLM PACKAGE PASS` prints after vLLM 0.19.0 (or a deliberately reviewed compatible replacement) is available.
6. Model discovery prints `Top mounted HF candidates:` and identifies the intended snapshot.
7. Model discovery prints `MODEL INPUT PASS: <same build id>`.
8. The local server prints `VLLM SERVER PASS`.
9. A real Qwen generation prints `FULL INFRASTRUCTURE PREFLIGHT PASS`.
10. The dynamic public-game smoke completes without harness errors.
11. The notebook ends with `SAVE/RUN VALIDATION PASS: <same build id>`.

If a log contains `if MODEL_SCORE < 45`, lacks a FINAL C build marker, or fails with `Best mounted model does not look like Qwen3.6 27B FP8`, the wrong artifact was imported. Delete that Kaggle notebook and import FINAL C fresh.

## Competition rerun behavior

Validation mode must not open the hidden competition scorecard. Competition rerun mode should:

- open one scorecard;
- call `make()` only once per environment;
- use the continuous worker queue;
- preserve per-game policy memory across level reset/retry;
- use wall-clock budget as the primary campaign limiter;
- retain D110's effect-posterior fallback behavior;
- for S120, use h2 predictive verification and cancel queued actions on high-confidence prediction contradiction;
- emit a durable receipt carrying build/source/serving diagnostics.

## Interpretation discipline

Submit S115 and S120 as two separate hypotheses.

- S110 → S115 asks whether the causal fallback transfers under Qwen.
- S115 → S120 asks whether temporal predictive verification and persistent goal state transfer.

Do not react to one leaderboard number by editing multiple unrelated mechanisms. Preserve each receipt and compare levels, actions, model calls, fallback use, prediction mismatches, goal hypotheses, and elapsed GPU time.

## Next version after S120

S130 should be an **adaptive campaign allocator** calibrated from real S115/S120 receipts. Candidate signals include current level, recent progress velocity, goal confidence, predictive coverage/mismatch rate, controller availability, elapsed GPU time, and model failure rate. Optimize expected marginal private score per GPU-second rather than equal effort per game.

See `docs/RELEASE_STATUS_2026-08-22.md` for the synchronized D110/D210 evidence and current roadmap.
