# Canonical Kaggle submission candidates

The implementation source of truth lives under `src/arc3lab/`. This file tracks the exact generated notebook artifacts that should be imported into Kaggle. Do not submit an older notebook whose display title merely looks similar.

Repository baseline for the current release ladder: `4f58bde50ae3160b446a7c5748787d3b951af5ab` (V005 predictive-state merge).

## S115 FINAL D — causal fallback

- build: `S115-FINAL-20260822-D`
- notebook SHA-256: `70fe1231a0255b5212c1a06e062c9b1f1ea49698534dc1e2bb95522f805bb939`
- primary diff: D110R2-promoted `EffectPosteriorPolicy` beneath the V004 coding/Qwen campaign
- infrastructure revision: dynamic CUDA-driver linker preflight for FlashInfer SM120 JIT
- submission order: **1**
- expected first log marker: `ARCANGEL SUBMISSION BUILD: S115-FINAL-20260822-D`

## S120 FINAL D — h2 predictive state

- build: `S120-FINAL-20260822-D`
- notebook SHA-256: `37680f9fb261ffd6864bb69ef89331282d2b3916548dd54e03af1ccbac23d39d`
- primary diff: D210R2-promoted h2 predictive-state verification + persistent goals + prediction-error queue invalidation, retaining S115's causal fallback
- infrastructure revision: same dynamic CUDA-driver linker preflight as S115 FINAL D
- D210 architecture gate: **PASSED**
- submission order: **2**
- expected first log marker: `ARCANGEL SUBMISSION BUILD: S120-FINAL-20260822-D`

## Required Kaggle inputs

1. ARC Prize 2026 - ARC-AGI-3
2. ARC3 vLLM H100 Wheelhouse V3
3. vrfai Qwen3.6 27B FP8 HF Snapshot

Settings:

- Accelerator: **NVIDIA RTX PRO 6000**
- Internet: **OFF**

## FINAL C incident and FINAL D fix

Both FINAL C candidates reached the correct model and then failed identically during FlashInfer 0.6.6 SM120 JIT at the final host-link step:

```text
/usr/bin/ld: cannot find -lcuda: No such file or directory
```

CUDA itself was healthy. FINAL D resolves the live `libcuda.so.1`, creates a writable unversioned `libcuda.so` alias, exports build/runtime linker paths, configures writable FlashInfer JIT directories, and compiles a real `c++ -lcuda` self-test before vLLM startup. The reusable implementation is `scripts/kaggle_cuda_linker_preflight.py`.

## Mandatory Save & Run acceptance markers

The exact saved notebook is eligible for submission only if its log contains all of:

1. the exact `ARCANGEL SUBMISSION BUILD: ...-D` marker above;
2. `ARC TOOLKIT PASS`;
3. `VLLM PACKAGE PASS`;
4. `Top mounted HF candidates:`;
5. `MODEL INPUT PASS: <build id>`;
6. `CUDA DRIVER RUNTIME LOAD PASS`;
7. `CUDA DRIVER LINKER PASS`;
8. `VLLM SERVER PASS: <profile>`;
9. `FULL INFRASTRUCTURE PREFLIGHT PASS`;
10. a dynamic public-game smoke with zero harness errors;
11. `SAVE/RUN VALIDATION PASS: <build id>`.

Do not submit an older C/R2 artifact and do not submit a D artifact that stops before the final validation marker.

## Experimental provenance

D110R2 promoted effect-posterior fallback behavior and rejected repeated later-level re-probing, aggressive global dead-action suppression, and a stronger static click prior.

D210R2 promoted h2 temporal predictive state. Mean repeated-key consistency was ~0.9918; held-out exact transition accuracy on covered contexts was ~0.9993 at ~0.766 coverage. Five discovered level-1 plans verified 5/5 with a median action ratio to human of ~0.182. Cross-level effect semantics transferred strongly, but level-2 plans did not automatically transfer, so ARCangel carries mechanics while re-inferring each new level's goal/plan.

See `docs/SUBMISSION_RUNBOOK.md` and `docs/RELEASE_STATUS_2026-08-22.md` for the full release checklist and research state.
