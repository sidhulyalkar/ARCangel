# ARC3 Frontier Lab 🧭

A reproducible research and Kaggle submission harness for **ARC Prize 2026 — ARC-AGI-3**.

ARCangel is built around one constraint: the public 25 games are a diagnostic distribution, not a lookup table. The goal is to compress the strongest ideas from coding agents, retrievable memory, causal action learning, temporal state abstraction and selective executable planning into a **single offline RTX PRO 6000 campaign that generalizes to 110 unseen games**.

## Current status — V005 predictive state

The repository has progressed beyond the original structural/model baseline. The active architecture is now evidence-driven by two completed CPU research programs:

- **D110R2:** 72-config fallback tournament promoted a soft causal `EffectPosteriorPolicy` and rejected hard global anti-dead suppression, per-level primitive re-probing and a stronger static click prior.
- **D210R2:** predictive-state/planner lab promoted **h2 temporal state** as the smallest global representation above 0.99 repeated-key consistency, measured 0.999319 exact held-out next-state accuracy on covered contexts, verified 5/5 executable plans, and found strong but imperfect cross-level mechanic transfer.

The resulting V005 loop is:

```text
observe
  ↓
h2 temporal predictive state
  ↓
known state-action context? ── yes ──→ predict / verify / amortize reliable plan
  │
  no
  ↓
Qwen goal + mechanic reasoning
  ↓
actor-gated Python analysis / executable hypothesis when useful
  ↓
real action
  ↓
prediction contradiction? ── yes ──→ clear stale queue + wake model + repair/bypass
  │
  no
  ↓
continue
```

The model-free floor is no longer raw novelty alone. It softly exploits causal action evidence while leaving rare/conditional actions recoverable.

## Why this architecture

### 1. The current frame is not enough

D210R2 found mean repeated-key state-action consistency of only **0.7933** for visual state alone. Adding temporal context changed the picture:

| State representation | Mean consistency |
|---|---:|
| visual | 0.793337 |
| visual + step mod 2 | 0.934718 |
| visual + step mod 4 | 0.960049 |
| h1 | 0.959275 |
| **h2** | **0.991835** |
| h3 | 0.997865 |
| h1 + action counts | 0.997303 |

The predeclared global promotion target was 0.99, so **h2** is the smallest representation that clears it. ARCangel therefore keys its predictive cache using the current scene plus the two most recent before-state/action pairs.

### 2. Once a context is known, prediction is nearly exact

D210R2 held-out transition prediction achieved:

- coverage: **0.766463**
- exact next-state accuracy on covered transitions: **0.999319**

That makes coverage the dominant problem. ARCangel does not force a world model over unknown states; it trusts supported temporal contexts for verification/amortization and keeps novel contexts model-led.

### 3. Compiled control can be dramatically action-efficient

D210R2 found five level-1 plans and replay-verified all five 3/3. Their median action count was only **18.18%** of the human baseline.

But after replaying those prefixes, the same planner found **0/5** level-2 plans. Mean cross-level effect transfer was still 0.90 and translation transfer 0.775.

The correct transfer unit is therefore:

> **mechanics and confidence, not the previous level's entire plan.**

Later levels inherit verified semantics and re-infer their own goal/route.

### 4. Generic fallback should be causal, but soft

D110R2 completed 72/72 model-free configurations without harness errors. `effect_posterior` had the best policy-family mean score, while re-probing primitives on every later level was uniformly worse than carrying early evidence.

Hard anti-dead suppression regressed materially. That is an important ARC lesson: an action that appears dead in one state may be pivotal in another.

`EffectPosteriorPolicy` therefore prefers channels with observed meaningful effects without permanently banning low-yield actions.

## Core stack

### Perception

Every settled observation can be represented as:

- rendered grid for visual gestalt;
- lossless hexadecimal ASCII for exact inspection;
- 4-connected components with color, size, bounding box, shape identity and edge contact;
- HUD/volatile-edge masking so timers/status strips do not explode the state graph.

### Lossless episodic memory

Every real action preserves:

```text
before state
+ action / ACTION6 target
+ after state
+ changed / meaningful-changed cells
+ level completion / game over / win
```

Summaries are derived views. Raw history is retained for programmatic retrieval.

### Temporal predictive memory

`PredictiveTransitionMemory` stores a distribution over next visual signatures and coarse effects for:

```text
(h2 temporal state, action)
```

A repeated high-confidence prediction is checked against the next actual frame. A contradiction:

1. increments mismatch telemetry;
2. clears queued actions;
3. records a high-confidence contradiction belief;
4. forces the coding model to reason again.

### Goal memory

Goal hypotheses are stored separately from mechanic beliefs. This matters because later levels often preserve action semantics while changing the progress condition.

The Qwen actor sees:

- current grid / objects;
- compact derived memory;
- complete programmatically searchable transition ledger;
- persistent transferable beliefs;
- persistent goal hypotheses;
- temporal predictive-memory diagnostics;
- a `predict(action_id, x=None, y=None)` helper inside the bounded Python sandbox.

### Actor-gated executable reasoning

ARCangel does not force every environment into a simulator. The coding actor may request Python/world-model analysis when it expects that analysis to reduce real-action risk. Otherwise it can act directly or use the causal fallback.

This follows the broader evidence from Duck/PRO-LONG/Tycho-style systems: preserve complete history, make it queryable, and build executable abstractions only when their expected planning value exceeds their orchestration cost.

## Competition runtime

The V004/V005 campaign shell is designed around the actual ARC3 Kaggle constraints:

- **RTX PRO 6000**
- Internet OFF
- local Qwen3.6 27B FP8 through offline vLLM
- 28-way continuous game queue
- 132-minute per-game wall-clock allowance
- shared campaign deadline
- uncapped model/tool calls in the campaign config; wall clock allocates inference
- one scorecard / one `make()` per environment
- fail-soft game isolation
- policy memory retained across competition-mode level RESET/retry
- thread-local ACTION6 `data=` to avoid mutable enum action-data races.

## Submission ladder

Private submissions are treated as experiments, not roulette.

| Candidate | Primary change | Question |
|---|---|---|
| V004 / S110 | campaign/runtime anchor | is the local Qwen coding shell stable? |
| **S115 FINAL** | + D110 effect-posterior fallback | does the causal fallback transfer under Qwen? |
| **S120 FINAL** | + D210 h2 predictive verification + goals | can temporal verification amortize reasoning and stop stale plans? |
| S130 next | adaptive campaign allocator | where should the 9-hour GPU budget go across 110 games? |

Exact final notebook build IDs, SHA-256 hashes, required Kaggle inputs and Save & Run acceptance criteria live in [`docs/SUBMISSION_RUNBOOK.md`](docs/SUBMISSION_RUNBOOK.md).

Research evidence and promotion decisions live in [`docs/D110_D210_RESULTS.md`](docs/D110_D210_RESULTS.md).

## Repository layout

```text
src/arc3lab/
  perception/     segmentation, frame signatures, HUD masking, diffs
  memory/         lossless episode ledger + h2 predictive transition memory
  planning/       observed transition graph / executable-search substrate
  policy/         structural, effect-posterior, hybrid and coding policies
  model/          local model / OpenAI-compatible adapters
  evaluation/     one-shot ARC runner + campaign diagnostics
scripts/
  run_public.py
  run_competition.py
  inspect_dataset.py
  compare_runs.py
configs/
  v001-structural.json
  v002-ducklite.json
  v003-angelcode.json
  v004-campaign-baseline.json
  v005-predictive-state.json
kaggle/
  canonical generated submission candidates
artifacts/experiments/
  machine-readable public research promotion receipts
docs/
  architecture, experiment plan, results and submission runbook
```

## Local evaluation

Point at the ARC environment files and run the generic policy without importing game source into the policy:

```bash
PYTHONPATH=src python scripts/run_public.py \
  --env-dir /path/to/environment_files \
  --policy structural \
  --max-actions 600 \
  --workers 8 \
  --output artifacts/v001.json
```

The public set is used to compare generic mechanisms. Public game IDs, coordinates and source-derived mechanics must never enter policy logic.

## Research chronology

### V003R — AngelCode

Added bounded Python analysis over the full transition ledger, persistent beliefs, model-generated tool queries and short reliable action queues.

### V004C — campaign baseline

Aligned the system with scoring/runtime economics: 28-way continuous scheduling, per-game wall-clock budgets, uncapped inference under the shared deadline and first-level mechanic discovery that persists into later weighted levels.

### D110R2 — fallback selection

Promoted causal effect posterior; rejected repeated re-probing, hard anti-dead suppression and stronger static click bias.

### D210R2 — temporal-state gate

Promoted h2 state, prediction-error reasoning triggers, compiled planning when supported, cross-level semantics with contradiction invalidation and structured probe coverage.

### V005 — active branch

Integrates those promotions into the coding agent while preserving the V004 serving/campaign anchor.

## What to log from every Kaggle run

Every receipt should make a leaderboard result interpretable:

- levels/actions/resets per environment;
- actions per completed level;
- model calls/failures;
- tool calls/failures;
- queued actions used;
- fallback actions;
- prediction mismatches;
- temporal prediction coverage/verification summary;
- goal-hypothesis count;
- actor-gated world-model delegations;
- wall time / deadline exhaustion;
- exact source/build manifest.

A leaderboard point without telemetry is a lottery ticket. A leaderboard point with a controlled diff and receipt is an experiment.

## Next high-leverage work

After real S115/S120 receipts, the next campaign version should optimize **expected marginal score per GPU-second** rather than giving every hidden game equal effort. Candidate scheduler state includes current level, recent progress, elapsed GPU time, goal confidence, predictive coverage/mismatch rate, model failure rate and whether a reliable controller exists.

The next per-game research frontier is **goal acquisition and pivotal-rule verification**, not another decimal point of transition prediction. A useful controller should ask which uncertain rule would most damage the proposed plan and spend the cheapest discriminating real probe on that assumption.

## Guardrails against public-set overfitting

- No game IDs in policy code.
- No imports from `environment_files/*/*.py` in policy code.
- No memorized public coordinates or solution sequences.
- Metadata/human baselines are evaluator-only.
- New heuristics are stated generically before per-game inspection.
- Every major mechanism gets a public ablation/promotion gate and, when quota-worthy, one private LB experiment.
- Cross-level transfer is confidence-weighted and contradiction-sensitive, never blind replay.

## References

- ARC Prize Milestone #1: https://arcprize.org/blog/arc-prize-2026-milestone-1
- Tufa Duck Harness: https://tufalabs.ai/research/duck-harness/
- Duck source: https://github.com/Tufalabs/duck-harness
- PRO-LONG: https://arxiv.org/abs/2607.20064
- Tycho: https://arxiv.org/abs/2607.28287
- ARC toolkit competition mode: https://docs.arcprize.org/toolkit/competition_mode

ARCangel's working thesis is now simple:

> **Learn enough mechanics to predict, infer what winning means, compile behavior only when it earns trust, and spend expensive model reasoning where prediction or purpose is uncertain.**
