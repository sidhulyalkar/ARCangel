# ARC3 Frontier Lab 🧭

A reproducible research and Kaggle submission harness for **ARC Prize 2026 — ARC-AGI-3**.

The goal is not to memorize 25 public games. The goal is to compress the strongest ideas from coding agents, programmatic memory, object-centric perception, and active world-model induction into a **single offline RTX 6000 submission that generalizes to 110 unseen games**.

## Status

**v0.1 starter is executable.** It includes:

- official `arc-agi==0.9.8` / `arcengine==0.9.3` compatible runner;
- exact local scorecard capture and per-level action telemetry;
- lossless transition memory;
- 4-connected object segmentation with translation-invariant shape hashes;
- volatile-edge/HUD masking;
- deterministic state-transition graph and novelty explorer;
- ACTION6 click ranking with rarity/button priors and dead-target memory;
- a reproducible random control;
- an offline model lane using a local OpenAI-compatible server such as vLLM;
- multimodal model prompts (rendered grid + lossless 0-F ASCII + object summary);
- validated short action queues and deterministic fallback;
- automatic local Qwen-model discovery on Kaggle inputs;
- one-scorecard / one-`make()`-per-environment competition execution;
- fail-soft per-game isolation plus a shared wall-clock deadline, so one bad environment or model reply does not destroy the whole submission;
- protection from `GameAction.ACTION6` mutable-enum data races by passing thread-local `data=` directly to `env.step`;
- leakage-resistant policy code: no public game IDs, source-derived rules, or per-game lookup tables.

The first generic structural control scored **0.1755** on the 25 public games at a 200-action cap, completing 2/183 levels. A seeded random control completed 0/183 in the same local setup. This is intentionally a calibration floor, not the candidate we expect to lead Kaggle.

## The important scoring correction

The prose competition description commonly shown in notebooks is slightly behind the uploaded toolkit. The supplied `arc-agi 0.9.8` scorer computes a completed level as:

```text
min((human_actions / agent_actions)^2 * 100, 115)
```

and then caps the game score by the **weighted fraction of levels completed**. Later levels receive larger weights. In practice:

1. **Completion dominates.** A later level is much more valuable than polishing an already-solved early level.
2. **Efficiency still matters quadratically** until the completion cap is saturated.
3. Super-human efficiency can provide up to a 15% per-level cushion, but cannot make a partially completed game score as if more levels were solved.

The toolkit also forces Kaggle into `OperationMode.COMPETITION`: one scorecard, one `make()` per environment, and RESET becomes a **level reset** rather than a full-game restart.

## Why the winning architecture should be model-first

The current evidence points in one direction:

| Approach | What it teaches us | Kaggle / public evidence |
|---|---|---|
| Random | useful plumbing control only | sample notebook ~0.18 |
| Stochastic Goose | online action-effect learning can work, but preview transfer collapsed | current sample ~0.25; preview winner was far stronger on 3 preview games |
| Persistent-memory BFS | deterministic exploration + memory beats pure random cheaply | public notebook 0.46 |
| Reki / Forge | vision model + structured JSON + reflection + action guards works | Milestone #1 places 2/3 |
| Tufa “Duck” | **small local coding model + REPL + multimodal perception + short context** is the strongest proven Kaggle recipe | Milestone #1 winner; recent public Duck notebook ~1.17 |
| PRO-LONG | never throw history away; search it programmatically instead of compressing it into prose | +18 pp average on public 25-game research eval |
| Executable World Models / Tycho | build a simulator only when the actor expects planning value; verification fit alone is not the goal | public 25-game frontier-model saturation |

The public 25-game research results and Kaggle leaderboard must not be conflated. Frontier proprietary models can nearly saturate the public set with unconstrained inference, while the Kaggle runtime is one RTX 6000 for <=9 hours and no internet. That **systems-compression gap** is the competition.

## Proposed winning stack: FRONTIER

**F**rame abstraction  
**R**etrievable lossless memory  
**O**nline action semantics  
**N**ovelty-guided probes  
**T**ool/coding model  
**I**nference-budget gating  
**E**xecutable hypotheses when useful  
**R**eplay-verified iteration

### 1. Perception

Every settled frame is represented three ways:

- pixel image for gestalt/symmetry;
- lossless 0-F ASCII grid for exact inspection;
- object graph for components, shape identity, rarity, edge contact, and click candidates.

Volatile edge cells are treated as likely HUD/timers so a shrinking status strip does not explode the state graph.

### 2. Lossless episodic memory

Every real action stores:

```text
before_state, action, target, after_state,
changed_cells, meaningful_changed_cells,
level_transition, game_over, win
```

Derived summaries can be regenerated. The raw interaction history is never replaced by a prose summary.

### 3. Cheap controller below the LLM

The deterministic layer handles:

- legal-action filtering;
- no-op / dead-target suppression;
- obvious object clicks;
- exact previously observed transitions;
- state-graph novelty;
- batching of a short, high-confidence model plan.

The LLM should reason about **rules, goals, abstractions, and plans**, not waste tokens validating `0 <= x < 64`.

### 4. Local model policy

`HybridPolicy` can talk to an attached model through local vLLM. The model gets a rendered frame, exact ASCII, object summary, and compact transition statistics. It returns 1–4 legal actions, a falsifiable expected change, and its current world-model hypothesis.

The first model target should be the same public Qwen 3.6 27B FP8 family used by Duck, because reproducing a known ~1%+ Kaggle system gives us a trustworthy ladder anchor before novel modifications.

### 5. Active world-model delegation

Do **not** force every game into a simulator. The next major branch should let the actor request an executable model when:

- the goal is understood but sequence planning is hard;
- repeated probes reveal stable deterministic dynamics;
- path/search depth is larger than a few moves;
- the current policy is cycling or repeatedly paying for rediscovery.

The model is verified against the lossless transition ledger, but predictive fit is only a gate. If modeling costs more actions/tokens than direct play, bypass it.

## Repository layout

```text
src/arc3lab/
  perception/     object segmentation, frame signatures, HUD masking, diffs
  memory/         append-only episodic ledger
  planning/       observed transition graph (world-model lane next)
  policy/         random, structural, hybrid model policy
  model/          Transformers + localhost/vLLM adapters
  evaluation/     one-shot runner and exact scorer
scripts/
  run_public.py
  run_competition.py
  inspect_dataset.py
  compare_runs.py
configs/
  v001-structural.json
  v002-ducklite.json
kaggle/
  generated submission notebooks
artifacts/
  local evaluation receipts
```

## Local public evaluation

Point at the uploaded `environment_files` directory:

```bash
PYTHONPATH=src python scripts/run_public.py \
  --env-dir /path/to/environment_files \
  --policy structural \
  --max-actions 600 \
  --workers 8 \
  --output artifacts/v001.json
```

Run a seeded random control:

```bash
PYTHONPATH=src python scripts/run_public.py \
  --env-dir /path/to/environment_files \
  --policy random \
  --max-actions 200 \
  --workers 8 \
  --output artifacts/random.json
```

Compare receipts:

```bash
PYTHONPATH=src python scripts/compare_runs.py artifacts/random.json artifacts/v001.json
```

## Kaggle model submission

Attach:

1. the ARC-AGI-3 competition input;
2. this repository (or use the generated embedded notebook);
3. a public offline model dataset, ideally the Qwen 3.6 27B FP8 family used by Duck.

Then launch the model locally and run:

```bash
PYTHONPATH=src python scripts/run_competition.py \
  --policy hybrid \
  --launch-vllm \
  --max-actions 1200 \
  --workers 6 \
  --time-budget-seconds 29400
```

No external HTTP is used. Model calls go to `127.0.0.1`.

## Submission ladder

Do not mutate five things between Kaggle runs. Each submission should answer one question.

| Version | Change from previous | What the LB teaches us |
|---|---|---|
| V001 | structural instrumented control | toolkit/runner sanity and low-cost floor |
| V002 | exact-ish Duck/Qwen replication | our reproducible competitive anchor |
| V003 | V002 + programmatic lossless-memory retrieval | whether context loss is the dominant failure |
| V004 | V003 + transition/HUD abstraction + action queue verification | whether perception/state aliasing is dominant |
| V005 | V004 + actor-gated executable world model | whether long-horizon planning is dominant |
| V006 | V005 + model-call budget router / specialist portfolio | fit the best reasoning into 9 hours |

The critical comparison is **V002→V003→V004→V005**, not V005 versus random.

## What to log from every Kaggle run

Keep the notebook output artifacts even when the LB score is disappointing:

- total model calls and generated tokens;
- actions per environment and per completed level;
- resets/game-overs;
- no-op action rate;
- illegal-model-output repair rate;
- ACTION6 target types and dead-target rate;
- model confidence calibration;
- number/length of queued plans;
- memory retrieval usage;
- world-model build/repair/use/bypass counts;
- wall time and GPU-memory high-water mark.

The runner also records fail-soft errors, deadline exhaustion, total model calls, and model failures. A leaderboard point without these diagnostics is a lottery ticket. A leaderboard point with them is an experiment.

## Research references

- ARC Prize Milestone #1: https://arcprize.org/blog/arc-prize-2026-milestone-1
- Tufa Duck Harness: https://tufalabs.ai/research/duck-harness/
- Duck source: https://github.com/Tufalabs/duck-harness
- PRO-LONG: https://arxiv.org/abs/2607.20064
- Tycho: https://arxiv.org/abs/2607.28287
- Executable World Models: https://arxiv.org/abs/2605.05138
- ARC toolkit competition mode: https://docs.arcprize.org/toolkit/competition_mode

## Guardrails against public-set overfitting

- No game IDs in policy code.
- No imports from `environment_files/*/*.py` in policy code.
- Metadata baselines are evaluator-only.
- New heuristics must be stated generically before looking at their per-game winners/losers.
- Every major change gets a public ablation and, when worth the quota, one private LB experiment.

That discipline is not bureaucracy. It is our best defense against building a 25-game museum exhibit instead of an agent.
