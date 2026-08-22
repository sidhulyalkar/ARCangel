# Experiment Plan

## North-star metric

Private Kaggle score. Public-game score is a development diagnostic and mechanism gate, not the objective.

## Secondary metrics

1. weighted levels completed;
2. action efficiency conditional on completion;
3. first-level solve rate across games;
4. later-level continuation rate after level 1;
5. no-op / repeated-dead-action rate;
6. game-over/reset rate;
7. inference calls and tokens per scored action;
8. wall-clock seconds per game;
9. public seed variance;
10. predictive-state coverage and high-confidence contradiction rate;
11. goal-hypothesis count / level-completion validation;
12. world-model build/use/bypass counts.

## Public evaluation protocol

- Use all 25 games for descriptive evaluation.
- Maintain leave-family-out slices by action interface (`keyboard`, `click`, mixed) and learned visual/mechanic clusters.
- Run at least 3 deterministic seeds for stochastic/model-sampling changes when compute permits.
- Never tune against individual game IDs.
- Record exact git SHA + config + toolkit version in every receipt.
- State the generic hypothesis before inspecting per-game winners/losers.

## Completed foundation

### A0 — structural floor

Completed. The model-free structural layer establishes runner/scorer correctness but is far below the competitive target.

### A1 — local coding-agent campaign shell

Completed through V003/V004:

- Qwen3.6 27B FP8 local actor;
- lossless transition ledger and bounded Python analysis;
- persistent beliefs across retries/levels;
- 28-way continuous campaign queue;
- 132-minute per-game wall-clock allowance;
- shared campaign deadline;
- no arbitrary model/tool-call cap in the campaign configuration;
- first-level primitive discovery carried across later levels.

## D110R2 — fallback policy tournament: COMPLETE

72/72 CPU configurations completed without harness errors.

Promoted mechanism: **soft causal effect posterior** beneath the model.

Key conclusions:

- `effect_posterior` had the best policy-family mean score;
- the promoted model-free operating point was `effect_posterior | 320 | 1`, but its action/reset values apply only to the fallback calibration, not the full Qwen agent;
- re-probing primitives on every later level was uniformly worse than preserving learned semantics;
- global anti-dead suppression regressed;
- a stronger static ACTION6 click prior did not improve aggregate score.

Implementation status: promoted into `EffectPosteriorPolicy` and below `HybridPolicy` on V005.

See `docs/D110_D210_RESULTS.md` and `artifacts/experiments/d110r2-summary.json`.

## D210R2 — predictive state + planner: COMPLETE

Architecture gate: **PASSED**.

Promoted default state: **h2**, the current scene plus the two most recent before-state/action pairs.

Evidence:

- h2 mean repeated-key consistency: 0.9918345;
- held-out transition coverage: 0.7664628;
- exact next-state accuracy conditional on coverage: 0.9993194;
- 5/5 found plans replay-verified;
- median verified plan used 18.18% of human baseline actions;
- mean cross-level effect transfer: 0.90;
- mean cross-level translation transfer: 0.775;
- 0/5 level-2 plans found after replaying level-1 prefixes.

Interpretation:

- temporal context is required for a robust transition cache;
- known temporal state-action contexts can be trusted enough for verification/amortization when supported;
- coverage, not fidelity, is the main transition-model bottleneck;
- transfer mechanics across levels but re-infer later-level goals and plans;
- cover distinct action channels before spending extra probes on uncertainty.

Implementation status: promoted into V005 `PredictiveTransitionMemory`, persistent goal hypotheses and prediction-error queue invalidation.

See `docs/D110_D210_RESULTS.md` and `artifacts/experiments/d210r2-summary.json`.

## V005 — predictive-state coding agent: ACTIVE

The target loop is now:

```text
observe
→ h2 predictive state
→ retrieve known transition / causal evidence
→ infer goal + mechanics when uncertain
→ actor-gated Python/executable analysis when useful
→ execute
→ verify prediction
→ continue or repair on contradiction
```

The model should act as scientist/goal reasoner and debugger rather than a frame-by-frame joystick.

### V005 implementation requirements

- `EffectPosteriorPolicy` as the generic fallback;
- `PredictiveTransitionMemory(history_depth=2)`;
- distributional `(temporal state, action) -> next state` evidence;
- queue cancellation on high-confidence prediction mismatch;
- persistent, separately tracked goal hypotheses;
- full-history programmatic retrieval retained;
- actor-gated world-model analysis, never mandatory compilation;
- per-game predictive/goal/world-model telemetry;
- generic/private-game-safe behavior only.

## Submission ladder

Every private submission must answer one primary question.

### S115 FINAL — causal fallback

Change from V004: promote D110R2 `EffectPosteriorPolicy` beneath the existing coding/Qwen shell.

Question:

> Does a public-CPU-promoted causal fallback transfer when the local model is the primary reasoner?

### S120 FINAL — h2 predictive state

Change from S115: add D210R2-promoted h2 temporal prediction verification, persistent goals and contradiction-triggered queue invalidation.

Question:

> Can known temporal contexts safely amortize model reasoning and prevent stale plans without constraining novel states?

Exact build IDs, notebook hashes, inputs and validation gates live in `docs/SUBMISSION_RUNBOOK.md`.

## S130 — adaptive campaign allocator: NEXT AFTER REAL RECEIPTS

Do not set allocation thresholds from public intuition alone. Build S130 from actual S115/S120 campaign telemetry.

Candidate state for each hidden game:

- weighted levels already completed;
- current level / recent progress velocity;
- elapsed GPU seconds;
- model-call and failure rate;
- goal-hypothesis confidence / validation;
- predictive-state coverage and mismatch rate;
- existence of a reliable queued/compiled controller;
- recent cycle/stall indicators.

Scheduler objective:

```text
expected marginal private score gain / expected remaining GPU-second
```

The scheduler should preferentially continue games with credible marginal reach rather than giving every one of 110 games equal time.

## Later research: goal induction and pivotal-rule verification

Transition prediction is already nearly exact on covered D210 contexts, so another decimal point there is low leverage. The next per-game research question is goal acquisition and play-adequate verification.

Potential experiment:

1. model proposes a falsifiable goal predicate;
2. planner produces a candidate plan under current dynamics;
3. identify the uncertain assumption whose failure would most damage the plan;
4. choose the cheapest discriminating real probe;
5. repair/bypass the controller on counterexample.

This is preferable to generic state novelty because it values information by downstream score risk.

## Kaggle interpretation policy

| Result | Interpretation | Next action |
|---|---|---|
| large gain | mechanism likely transfers | keep; inspect which telemetry moved |
| small gain | possible benefit/noise | retain only if cheap and mechanistically supported |
| flat | public mechanism did not affect private campaign | gate/remove before stacking more complexity |
| regression | action/runtime/context cost may exceed benefit | inspect receipts before retuning |

Do not respond to a bad LB score by adding several unrelated heuristics at once.
