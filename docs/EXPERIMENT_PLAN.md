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
12. world-model build/use/bypass counts;
13. spatial-plan request / compile / execution rate;
14. spatial-route invalidation rate;
15. action-semantic support, confidence and translation purity before route compilation.

## Public evaluation protocol

- Use all 25 games for descriptive evaluation.
- Maintain leave-family-out slices by action interface (`keyboard`, `click`, mixed) and learned visual/mechanic clusters.
- Run at least 3 deterministic seeds for stochastic/model-sampling changes when compute permits.
- Never tune against individual game IDs.
- Record exact git SHA + config + toolkit version in every receipt.
- State the generic hypothesis before inspecting per-game winners/losers.
- Prefer broad autonomous CPU experiment matrices before consuming the one-per-day private submission slot.

## Completed foundation

### A0 — structural floor

Completed. The model-free structural layer establishes runner/scorer correctness but is far below the competitive target.

### A1 — local coding-agent campaign shell

Completed through V003/V004:

- Qwen3.6 27B FP8 local actor;
- lossless transition ledger and bounded Python analysis;
- persistent beliefs across retries/levels;
- campaign scheduling/runtime hardening;
- no arbitrary model/tool-call cap in the campaign configuration;
- first-level primitive discovery carried across later levels.

## D110R2 — fallback policy tournament: COMPLETE

72/72 CPU configurations completed without harness errors.

Promoted mechanism: **soft causal effect posterior** beneath the model.

Key conclusions:

- `effect_posterior` had the best policy-family mean score;
- the promoted model-free operating point was `effect_posterior | 320 | 1`, but its action/reset values apply only to fallback calibration, not the full Qwen agent;
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

## V005 — predictive-state coding agent

The V005 loop is:

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

The model acts as scientist/goal reasoner and debugger rather than a frame-by-frame joystick.

### Canonical V005 mechanisms

- `EffectPosteriorPolicy` generic fallback;
- `PredictiveTransitionMemory(history_depth=2)`;
- distributional `(temporal state, action) -> next state` evidence;
- queue cancellation on high-confidence prediction mismatch;
- persistent, separately tracked goal hypotheses;
- full-history programmatic retrieval;
- actor-gated world-model analysis, never mandatory compilation;
- private-game-safe behavior only.

## V006 — spatial intelligence: QUALIFIED LOCALLY, AWAITING RTX/PRIVATE GATE

The next hypothesis is that many expensive ARC decisions become exact geometric search problems after the agent has learned **which object it controls** and **how actions move it**.

V006 adds:

- exact N/NE/E/SE/S/SW/W/NW spatial relations and raycasts;
- actor-relative distances, touching and line-of-sight;
- full-footprint free-space topology;
- causal controlled-object identification from observed motion;
- learned action displacement vectors, with no assumed button meanings;
- camera/global-motion rejection;
- translation-purity gating to separate locomotion from interactions;
- duplicate-actor ambiguity gating;
- exact shortest-action BFS over the learned displacement model;
- model-requested spatial planning, never unconditional automation;
- pre-action geometry checks and post-action actor-anchor verification;
- spatial sandbox queries and campaign telemetry.

The division of labor is intentional:

```text
Qwen: infer semantic roles / goal / whether navigation is appropriate
exact code: represent geometry / calculate the shortest route
verification: keep executing only while causal and geometric assumptions survive
```

### V006 local qualification

S135 FINAL B source passed:

- source compilation;
- **31** canonical V005 + V006 regression tests;
- **1000/1000** randomized shortest-path comparisons against an independent reference search;
- actor-footprint collision tests;
- camera/global-motion rejection tests;
- duplicate actor ambiguity tests;
- collateral-motion translation-purity tests;
- dynamic queued-route invalidation tests;
- embedded notebook synthetic spatial preflight.

See `docs/V006_SPATIAL_INTELLIGENCE.md`.

## Submission ladder

Every private submission must answer one primary question. The competition currently allows only one private submission per day for this workflow, so a slot should be consumed only after its exact notebook passes Save & Run.

### S115 — causal fallback baseline

Question:

> Does a public-CPU-promoted causal fallback transfer when the local model is the primary reasoner?

### S120G — canonical h2 predictive-state baseline

Question:

> Can known temporal contexts safely amortize model reasoning and prevent stale plans without constraining novel states when the canonical V005 goal/predictive prompt is actually exposed to Qwen?

### S135 FINAL B — V006 spatial intelligence

Build: `S135-FINAL-20260823-B`.

Question:

> Once the controlled object and movement semantics are learned causally, does exact 360° geometry plus guarded shortest-path execution improve private level completion and action efficiency over canonical V005?

S135 is the preferred next architectural submission after S120G because it targets a larger remaining bottleneck than another decimal point of covered transition prediction.

### S140 candidate — executable spatial goal predicates

Only build/promote after S135 telemetry.

Convert model language goals into explicit predicates such as:

- `REACH(actor, target)`;
- `TOUCH(actor, target)`;
- `CONTAIN(actor, region)`;
- `ALIGN(actor, object)`;
- `COLLECT(all members of class)`;
- `ACTIVATE(object)`;
- `EXIT(region)`.

Primary question:

> Can falsifiable goal predicates close the gap between strong transition modeling and weak later-level goal acquisition?

### S145 candidate — symmetry transfer

Use rotations/reflections only as a generic hypothesis over learned geometry/action transforms. Never assume a particular level is rotated.

Primary question:

> Can dihedral-equivalent geometry transfer action semantics across transformed later levels with fewer scored probes?

### S150 candidate — pivotal spatial verification

For a candidate route, identify the uncertain assumption whose failure would most damage expected score and choose the cheapest real discriminating probe.

Approximate value:

```text
P(assumption wrong)
× downstream cost if wrong
× future route dependence
/ probe action cost
```

This should replace generic novelty probing when a concrete route exists.

## Adaptive campaign allocator: DEFER UNTIL REAL RECEIPTS

Do not set allocation thresholds from public intuition alone. Build the allocator from real private receipts after the agent's per-game control layer is strong enough to produce meaningful confidence/plan telemetry.

Candidate state for each hidden game:

- weighted levels already completed;
- current level / recent progress velocity;
- elapsed GPU seconds;
- model-call and failure rate;
- goal-hypothesis confidence / validation;
- predictive-state coverage and mismatch rate;
- spatial-control readiness;
- reliable queued/compiled controller availability;
- recent cycle/stall indicators.

Scheduler objective:

```text
expected marginal private score gain / expected remaining GPU-second
```

## Kaggle interpretation policy

| Result | Interpretation | Next action |
|---|---|---|
| large gain | mechanism likely transfers | keep; inspect which telemetry moved |
| small gain | possible benefit/noise | retain only if cheap and mechanistically supported |
| flat | mechanism may be too rare or too gated | inspect request/compile/use telemetry before loosening gates |
| regression | action/runtime/context cost may exceed benefit | inspect receipts; do not stack another feature blindly |

Do not respond to a bad leaderboard result by adding several unrelated heuristics at once.
