# FINAL D release notes

FINAL D is an infrastructure-only hardening of the S115/S120 submission ladder. It preserves the D110R2/D210R2 policy hypotheses while fixing the Kaggle RTX PRO 6000 / FlashInfer SM120 linker failure observed in both FINAL C Save & Run attempts.

## Builds

- `S115-FINAL-20260822-D` — SHA-256 `70fe1231a0255b5212c1a06e062c9b1f1ea49698534dc1e2bb95522f805bb939`
- `S120-FINAL-20260822-D` — SHA-256 `37680f9fb261ffd6864bb69ef89331282d2b3916548dd54e03af1ccbac23d39d`

## Runtime hardening

Before vLLM startup each notebook now resolves the live CUDA driver, creates an unversioned linker alias, exports build/runtime library search paths, configures writable FlashInfer JIT directories, and compiles a real `-lcuda` linker self-test. The notebook will not spend several minutes compiling FlashInfer kernels unless the linker preflight succeeds first.

The primary vLLM profile remains unchanged. A conservative FlashAttention/eager profile is retained only as a fallback for unrelated startup failures.

## Policy status

No D110R2 or D210R2 conclusion is changed by this release. S115 remains the causal-fallback experiment and S120 remains the h2 predictive-state experiment.
