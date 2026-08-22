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

The intended model may mount at a hyphenated path such as `vrfai-qwen3-6-27b-fp8-hf-snapshot`. Current releases normalize punctuation and validate Qwen + 27B + FP8 + safetensors evidence. The retired `MODEL_SCORE < 45` detector must not appear anywhere in the imported notebook.

## S115 FINAL D — causal fallback

Build ID: `S115-FINAL-20260822-D`

Notebook SHA-256: `70fe1231a0255b5212c1a06e062c9b1f1ea49698534dc1e2bb95522f805bb939`

Primary hypothesis: D110R2's `EffectPosteriorPolicy` improves the V004/Qwen campaign while preserving the higher-level campaign shell.

Recommended private-submission order: **first**.

## S120 FINAL D — h2 predictive state

Build ID: `S120-FINAL-20260822-D`

Notebook SHA-256: `37680f9fb261ffd6864bb69ef89331282d2b3916548dd54e03af1ccbac23d39d`

Primary hypothesis: add D210R2-promoted h2 predictive verification, persistent goal hypotheses, prediction-error queue invalidation, and actor-visible temporal memory while retaining S115's causal fallback.

D210R2 architecture gate: **PASSED**.

Recommended private-submission order: **second**.

## FINAL C runtime incident

Both FINAL C candidates reached the intended Qwen snapshot and failed identically during FlashInfer 0.6.6 SM120 JIT. CUDA itself was healthy: PyTorch saw the RTX PRO 6000, `nvidia-smi` worked, ARC wheels imported, vLLM 0.19.0 installed, and model discovery confirmed Qwen + 27B + FP8 + safetensors. The final host-link step failed with:

```text
/usr/bin/ld: cannot find -lcuda: No such file or directory
```

This is a build-time linker-name/search-path failure, not a policy or model-selection failure.

## FINAL D CUDA-driver linker hardening

FINAL D performs the following before vLLM starts:

1. resolve the live `libcuda.so.1` using `ldconfig`, `/proc/self/maps`, and known NVIDIA locations;
2. verify runtime loading with `ctypes.CDLL`;
3. create `/kaggle/working/arcangel_cuda_driver_link/libcuda.so` pointing at the live driver;
4. prepend that directory and CUDA runtime directories to `LIBRARY_PATH` and `LD_LIBRARY_PATH`;
5. configure writable `FLASHINFER_JIT_DIR` and `FLASHINFER_GEN_SRC_DIR` locations;
6. place the same linker alias into CUDA's stub directory when permitted;
7. compile and link a real C++ probe against `-lcuda`;
8. require `CUDA DRIVER LINKER PASS` before any long FlashInfer JIT startup;
9. launch the intended high-performance vLLM profile first;
10. retain one conservative FlashAttention/eager fallback for unrelated startup failures.

The reusable preflight implementation is `scripts/kaggle_cuda_linker_preflight.py`.

## Save & Run acceptance criteria

A saved version is eligible for competition submission only if all of the following are true:

1. The first execution cell prints the exact `ARCANGEL SUBMISSION BUILD: ...-D` above.
2. CUDA reports an RTX PRO 6000 with the expected large-memory device.
3. `ARC TOOLKIT PASS` prints after the mounted ARC wheels install.
4. Embedded ARCangel source imports successfully.
5. `VLLM PACKAGE PASS` prints after vLLM 0.19.0 (or a deliberately reviewed compatible replacement) is available.
6. Model discovery prints `Top mounted HF candidates:` and identifies the intended snapshot.
7. Model discovery prints `MODEL INPUT PASS: <same build id>`.
8. The driver preflight prints `CUDA DRIVER RUNTIME LOAD PASS`.
9. The host-linker self-test prints `CUDA DRIVER LINKER PASS`.
10. The local server prints `VLLM SERVER PASS: <profile>`.
11. A real Qwen generation prints `FULL INFRASTRUCTURE PREFLIGHT PASS`.
12. The dynamic public-game smoke completes without harness errors.
13. The notebook ends with `SAVE/RUN VALIDATION PASS: <same build id>`.

If a run lacks the FINAL D build marker, do not debug that older artifact. Import FINAL D fresh. If FINAL D fails, preserve the complete vLLM log tail because the `-lcuda` failure family should already have been eliminated before server launch.

## Competition rerun behavior

Validation mode must not open the hidden competition scorecard. Competition rerun mode should:

- open one scorecard;
- call `make()` only once per environment;
- use the continuous worker queue;
- preserve per-game policy memory across level reset/retry;
- use wall-clock budget as the primary campaign limiter;
- retain D110's effect-posterior fallback behavior;
- for S120, use h2 predictive verification and cancel queued actions on high-confidence prediction contradiction;
- emit a durable receipt carrying build/source/serving diagnostics, including the selected vLLM startup profile.

## Interpretation discipline

Submit S115 and S120 as two separate hypotheses.

- S110 → S115 asks whether the causal fallback transfers under Qwen.
- S115 → S120 asks whether temporal predictive verification and persistent goal state transfer.

Do not react to one leaderboard number by editing multiple unrelated mechanisms. Preserve each receipt and compare levels, actions, model calls, fallback use, prediction mismatches, goal hypotheses, and elapsed GPU time.

## Next version after S120

S130 should be an **adaptive campaign allocator** calibrated from real S115/S120 receipts. Candidate signals include current level, recent progress velocity, goal confidence, predictive coverage/mismatch rate, controller availability, elapsed GPU time, and model failure rate. Optimize expected marginal private score per GPU-second rather than equal effort per game.

See `docs/RELEASE_STATUS_2026-08-22.md` for the synchronized D110/D210 evidence and current roadmap.
