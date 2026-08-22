# ARCangel Kaggle submission runbook

This file is the release checklist for the post-D110/D210 submission ladder. Do not submit an older imported notebook just because its display title looks similar.

## Required Kaggle inputs

Attach exactly:

1. **ARC Prize 2026 - ARC-AGI-3**
2. **ARC3 vLLM H100 Wheelhouse V3**
3. **vrfai Qwen3.6 27B FP8 HF Snapshot**

Kaggle settings:

- **Accelerator:** NVIDIA RTX PRO 6000
- **Internet:** OFF

The expected model dataset may mount under a hyphenated path such as `vrfai-qwen3-6-27b-fp8-hf-snapshot`. Final notebooks normalize punctuation and validate Qwen + 27B + FP8 + safetensors evidence instead of using the retired brittle `MODEL_SCORE < 45` detector.

## S115 FINAL — causal fallback

Build ID:

```text
S115-FINAL-20260822-A
```

Notebook SHA-256:

```text
e585d9b0c3276e6525237da6b76f52be22c362bebbca4f62da54fc189c2ed9d5
```

Embedded source-bundle SHA-256:

```text
1479f99d16dd56a1fce56281d4c414af9b46ee9f74d2023be7a9edbeef6b857c
```

Primary hypothesis: D110R2's `EffectPosteriorPolicy` improves the V004/Qwen campaign while leaving the higher-level coding/model-serving shell unchanged.

Recommended private-submission order: **first**.

## S120 FINAL — h2 predictive state

Build ID:

```text
S120-FINAL-20260822-A
```

Notebook SHA-256:

```text
fc250554b30ff4254922dae192568a295160ea984859d2bcda48f490d33e9ff2
```

Embedded source-bundle SHA-256:

```text
b7353abdba5cd43b8bfcca0eb21c617450e0c68776970b6048779f507bad2b05
```

Primary hypothesis: add D210R2-promoted h2 predictive-state verification, persistent goal hypotheses, prediction-error queue invalidation and actor-visible temporal memory while retaining S115's causal fallback.

D210R2 research gate: **PASSED**.

Recommended private-submission order: **second**.

## Save & Run acceptance criteria

A development Save & Run is eligible to become a Kaggle competition submission only if all of the following are true:

1. The first contract cell prints the exact `SUBMISSION_BUILD` above.
2. CUDA reports an RTX PRO 6000 with the expected large-memory device.
3. `ARC TOOLKIT PASS` prints after the mounted ARC wheels install.
4. Embedded ARCangel source imports successfully.
5. `vLLM: 0.19.0` (or a deliberately reviewed compatible replacement) imports successfully from the offline wheelhouse.
6. Model discovery prints `Top mounted HF candidates:` followed by the intended snapshot.
7. The detector prints `MODEL INPUT PASS`.
8. The local vLLM server starts and the real Qwen generation smoke succeeds.
9. The dynamic public-game smoke completes without harness errors.
10. The notebook ends with `SAVE/RUN VALIDATION PASS` and the same build ID.

If the log contains:

```text
if MODEL_SCORE < 45
```

or fails with `Best mounted model does not look like Qwen3.6 27B FP8`, the wrong/old notebook revision was executed. Do not debug that obsolete detector; import the final artifact.

## Competition rerun behavior

Save/validation mode must not open the hidden competition scorecard. Competition rerun mode should:

- open one scorecard;
- call `make()` only once per environment;
- use the continuous worker queue;
- preserve per-game policy memory across level reset/retry;
- use wall-clock budget as the primary campaign limiter;
- emit a durable receipt with source/build information and diagnostics.

## Interpretation discipline

Submit S115 and S120 as two separate hypotheses.

- S110 → S115 asks whether the causal fallback transfers under Qwen.
- S115 → S120 asks whether temporal predictive verification and persistent goal state transfer.

Do not react to one leaderboard number by editing multiple unrelated mechanisms. Preserve the receipt, inspect actions/levels/model calls/prediction mismatches, and let the next change answer one question.

## Next version after S120

S130 should be an **adaptive campaign allocator**, calibrated from real S115/S120 receipts. Candidate allocation signals include current level, recent progress velocity, goal confidence, predictive coverage/mismatch rate, controller availability, elapsed GPU time and model failure rate. The objective is expected marginal private score per GPU-second, not equal effort per game.
