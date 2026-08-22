# ARCangel release status — 2026-08-22

## Current repository state

V005 is merged to `main` at `4f58bde50ae3160b446a7c5748787d3b951af5ab` via PR #6. The merge promotes the D110R2 causal fallback and D210R2 temporal predictive-state findings into ARCangel and adds goal/prediction telemetry plus experiment/result documentation.

## D110R2 decision

The 72-configuration CPU policy tournament completed cleanly.

Promoted fallback operating point:

- policy: `effect_posterior`
- diagnostic budget: 320 actions
- diagnostic reset allowance: 1
- score: ~0.20219
- progress: 4 levels / 4 games

Interpretation:

- preserve primitive/action semantics learned early rather than blindly re-probing every level;
- use causal effect history as a soft posterior, not a hard action filter;
- do not globally suppress actions merely because they were repeatedly dead in other contexts;
- do not promote a stronger handcrafted ACTION6/object prior;
- the 320/1 optimum applies to the model-free fallback experiment, not as a hard cap on the full Qwen campaign.

## D210R2 decision

The deep predictive-state/planner experiment completed.

Promoted architecture facts:

- recommended default predictive representation: `h2`;
- h2 repeated-key consistency: ~0.99183;
- h3 consistency: ~0.99787 but with greater state complexity;
- held-out exact transition coverage: ~0.76646;
- exact next-state accuracy when covered: ~0.99932;
- plans found: 5;
- plans fully verified: 5/5;
- median action ratio to human: ~0.1818;
- level-2 plans found from transferred level-1 search: 0/5;
- mean cross-level action-effect transfer: 0.90;
- mean cross-level translation transfer: 0.775.

Architecture consequence:

`observe -> infer goal/mechanic -> h2 predictive state -> compile/verify when covered -> execute -> wake the model on prediction error, unseen context, or goal ambiguity`.

Mechanics may transfer across levels, but complete plans/goals must be re-inferred unless independently verified.

## Submission ladder

### S115-FINAL-20260822-C

Purpose: low-risk causal fallback ablation beneath the proven Qwen/vLLM campaign shell.

Notebook SHA-256:

`76bab67d0bd1d97ae5bd8336e1af2df994eaa8cf666f7f1d4b098be7ff2bb5b5`

Submit first after complete Save & Run acceptance.

### S120-FINAL-20260822-C

Purpose: first V005-style h2 predictive-state campaign, retaining the S115 fallback.

Notebook SHA-256:

`f5db889630e94b8629f216dbf83404d65d92cbae8ef3a93cf3a36a60e2c7392b`

Submit second after complete Save & Run acceptance.

## Kaggle release engineering finding

Two recent validation logs executed obsolete S115/S120 imports even after fixed artifacts had been prepared. They were identifiable because the log still contained:

`if MODEL_SCORE < 45`

and lacked an explicit final build identifier.

FINAL C artifacts therefore print the immutable build ID before any install/model work and again at model-discovery/validation completion. The retired numeric model detector is absent. The intended Qwen snapshot may mount as `vrfai-qwen3-6-27b-fp8-hf-snapshot`; punctuation-normalized discovery handles that path.

## Immediate execution order

1. Import only `S115-FINAL-20260822-C` into a fresh Kaggle notebook.
2. Attach the three canonical inputs, RTX PRO 6000, Internet OFF.
3. Save & Run All and require every marker in `kaggle/SUBMISSIONS.md`.
4. Submit that exact saved version if validation passes.
5. Repeat with `S120-FINAL-20260822-C`.
6. Preserve both private receipts before changing strategy.
7. Use S115/S120 receipts to calibrate S130 adaptive campaign allocation rather than tuning public game IDs or mechanics.

## Next research target

Do not restart broad transition-model research. The current bottleneck has moved toward goal induction, pivotal-assumption falsification, contextual ACTION6 affordances, and campaign-level compute allocation.

S130 should estimate expected marginal score gain per GPU-second using level reached, recent progress, goal confidence, predictive coverage/mismatch rate, controller availability, elapsed time, and model failure rate.
