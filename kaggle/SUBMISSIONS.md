# Canonical Kaggle submission candidates

The source of truth for implementation lives under `src/arc3lab/`; this directory tracks the immutable generated notebook builds used for Kaggle experiments. Do not tune notebook-embedded policy logic independently of the repository source.

## S115 FINAL — causal fallback

- build: `S115-FINAL-20260822-A`
- notebook SHA-256: `e585d9b0c3276e6525237da6b76f52be22c362bebbca4f62da54fc189c2ed9d5`
- embedded source SHA-256: `1479f99d16dd56a1fce56281d4c414af9b46ee9f74d2023be7a9edbeef6b857c`
- primary diff: D110R2-promoted `EffectPosteriorPolicy` beneath the V004 coding/Qwen campaign
- submission order: 1

## S120 FINAL B — h2 predictive state

- build: `S120-FINAL-20260822-B`
- notebook SHA-256: `7d61317a407f2db18e6e117a12b78ca633040367a0c8a607567e489f5021baf6`
- embedded source SHA-256: `e77eac56a8a7f7a242c3a894e0166970609cf8e78fa5295b0f519a83aba664bb`
- primary diff: D210R2-promoted h2 predictive-state verification + persistent goals + prediction-error queue invalidation
- D210 architecture gate: PASSED
- submission order: 2

## Required Kaggle inputs

1. ARC Prize 2026 - ARC-AGI-3
2. ARC3 vLLM H100 Wheelhouse V3
3. vrfai Qwen3.6 27B FP8 HF Snapshot

Settings:

- NVIDIA RTX PRO 6000
- Internet OFF

See `docs/SUBMISSION_RUNBOOK.md` for the complete Save & Run acceptance checklist.

## Why generated notebooks are release artifacts

The notebook is an offline transport envelope: it bootstraps mounted wheels, embeds the exact source tree, discovers the attached model, starts local vLLM, runs validation, and switches to the hidden competition scorecard only under competition rerun mode.

The agent implementation itself should evolve in `src/arc3lab/` with tests and configs. A generated notebook is considered valid only when its build ID and source SHA match this manifest.
