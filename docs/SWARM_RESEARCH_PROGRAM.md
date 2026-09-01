# V013 Autonomous Research Swarm

## Why this exists

S190A/V011 scored 0.17. That result is treated as an architectural falsification event, not a tuning request. V012 repairs the policy authority model, but V013 changes the *research process itself*: no single architecture, prompt, or agent is presumed to be the answer.

The goal is to search cognitive-architecture space with independent researchers, objective held-out evaluation, explicit controls, adversarial review, and reproducible promotion gates. The lab is designed to optimize the probability of discovering a competitive ARC-AGI-3 agent. It does not promise a leaderboard rank.

## Core principle

**Ensemble researchers, not ordinary environment actions.**

Frontier models are used during development to propose, criticize, implement, and transfer hypotheses. The submitted Kaggle agent should remain compact, offline, and evidence-driven. Constant multi-model voting inside the scored environment is considered an expensive hypothesis that must earn its place experimentally.

## Authority hierarchy

1. Immutable environment evidence.
2. Held-out behavioral results.
3. Repeated-run uncertainty-adjusted score.
4. Explicit control comparison.
5. Research-agent claims.
6. Developer intuition.

A proposal cannot promote itself by sounding convincing. Unit tests establish software correctness only.

## Four qualification gates

### Software

Code imports, tests pass, manifests validate, receipts parse, notebook/runtime contracts hold.

### Scientific

The experiment has an explicit hypothesis, control, target metric, falsifier, and no forbidden blind leakage.

### Behavioral

The challenger improves held-out interactive competence and does not purchase that gain with excessive emergency actions, failures, or pathological action/model-call cost.

### Leaderboard

The exact candidate has real Kaggle evidence and is compared against the imported public Duck/Qwen3.8 control. Internal promotion alone cannot nominate a Kaggle candidate.

## Tournament contestants

`configs/swarm-v013.json` is the source of truth.

- **A Duck/Qwen3.8 public**: external Copy & Edit leaderboard control. ARCangel must not counterfeit this baseline by reimplementing it and calling the result equivalent.
- **B Coding Minimal**: thinnest runnable ARCangel coding-agent control.
- **C V011 Reflective**: retained negative control for the 0.17 lineage.
- **D V012 Evidence First**: full evidence-first contender.
- **E V012 Lite**: deliberate ablation of reasoning/tool/plan capacity.
- **F Duck Memory**: planned selective-memory challenger.
- **G Adaptive Representation**: planned grid/image/delta/object representation chooser.
- **H Executable World Model**: planned conditional model-building escalation.
- **I Information-Gain Explorer**: planned causal probe scheduler.
- **J Sparse Debate**: planned proposal/skeptic/judge escalation at pivotal states only.

Unimplemented contestants stay disabled. A name in the manifest is not evidence that the mechanism works.

## Metrics

Per-run ARC suite receipts are converted into bounded metrics:

- solve rate;
- level progress;
- action efficiency;
- model-call efficiency;
- expectation accuracy;
- falsification health;
- stability;
- official score when exposed by the scorecard;
- failure rate;
- timeout fraction;
- emergency-action fraction.

The arena computes a weighted score and then subtracts a one-standard-error uncertainty term across repeated seeds. Lucky single runs are intentionally disfavored.

## Promotion

A challenger must have at least the configured number of validation runs, beat its explicit internal control by the configured validation margin, avoid material dev regression, and stay below emergency/failure ceilings.

Leaderboard nomination is stricter:

1. the challenger must first pass internal promotion;
2. an actual `kaggle` result for `A-duck-qwen38-public` must exist in the ledger;
3. an actual `kaggle` result for the challenger must exist;
4. the challenger must meet the configured leaderboard delta.

This makes the S200 control a software-enforced dependency rather than a recommendation in prose.

## Split discipline

`SplitRegistry` deterministically partitions game IDs with a secret salt into DEV, VALIDATION, and BLIND.

The public split registry exposes DEV and VALIDATION IDs plus only the count of blind games. The private registry contains blind identities and should stay out of research packets, normal agent prompts, and ordinary development artifacts.

This BLIND split is an internal holdout over available environments, not a substitute for the competition's genuinely hidden games. It exists to detect obvious public-game overfitting before Kaggle does.

## Research swarm roles

Research packets contain ten independent assignments:

- minimalist;
- scientist;
- explorer;
- planner;
- vision;
- memory;
- runtime;
- red team;
- generalization judge;
- integrator.

Round 1 should be independent. Do not share leading-agent advice until each role has produced its own hypothesis. Later rounds may transfer the strongest empirically supported discoveries to lagging agents. This preserves diversity before exploitation.

Each research proposal must state:

1. one primary hypothesis;
2. the smallest experiment capable of testing it;
3. target metric and split;
4. the result that falsifies the proposal;
5. concrete implementation/patch plan;
6. likely generalization failure mode.

## Research packet safety

`ResearchPacketBuilder` creates deterministic `.tar.gz` packets and removes blind scorecard rankings and files whose path identifies them as blind. This is a guardrail, not permission to store blind evidence carelessly elsewhere.

## Typical workflow

```bash
# 1. Validate the tournament contract.
python scripts/run_swarm_lab.py validate

# 2. Build a private/public split from available game IDs.
python scripts/run_swarm_lab.py split \
  --games artifacts/arena/v013/game_ids.txt \
  --salt "$ARCANGEL_SPLIT_SALT"

# 3. Start one local OpenAI-compatible Qwen server, then inspect the battle plan.
python scripts/run_swarm_lab.py plan --splits dev,validation

# 4. Execute enabled contestants. Serial-by-default avoids accidental GPU oversubscription.
python scripts/run_swarm_lab.py run --splits dev,validation

# 5. Score and inspect promotions.
python scripts/run_swarm_lab.py score
python scripts/run_swarm_lab.py promote

# 6. Produce a blind-safe packet for independent frontier research agents.
python scripts/run_swarm_lab.py packet
```

A BLIND run is judge-owned and should use the private split registry explicitly rather than the public contestant command.

## Kaggle result ingestion

Kaggle Save & Run receipts can be converted with:

```bash
python scripts/run_swarm_lab.py ingest-suite \
  --suite path/to/suite-receipt.json \
  --contestant D-v012-evidence-first \
  --split kaggle \
  --seed 20260831
```

The public Duck control should be imported under `A-duck-qwen38-public` after running the exact Copy & Edit control. Do not enter a reported public score as if it were our own experimental receipt.

## What autonomy means here

The lab autonomously handles experiment planning, repeat-run bookkeeping, subprocess execution, receipt ingestion, scoring, uncertainty penalties, promotion decisions, leaderboard gating, and research-packet generation.

External frontier-model calls remain credential/provider dependent. The repository therefore treats external researchers as replaceable proposal producers rather than trusted evaluation authorities. A future provider adapter can automate those calls without changing the arena contract.

## Immediate research sequence

1. Reproduce the Duck/Qwen3.8 Kaggle control and import the receipt.
2. Run B/C/D/E on identical DEV/VALIDATION splits and seeds.
3. Inspect failure families rather than only aggregate score.
4. If D loses to B, freeze full V012 and let E/B guide the next architecture.
5. If D or E wins, BLIND-judge it before building F-J on top.
6. Activate exactly one new mechanism at a time.
7. Only combine individually winning mechanisms in a separate integration experiment.
8. Nominate Kaggle candidates only through the leaderboard gate.

The desired end state is an evolutionary research loop where attractive ideas are cheap to propose, expensive to promote, and easy to kill.
