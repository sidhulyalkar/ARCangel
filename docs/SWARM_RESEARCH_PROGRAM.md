# V013 Autonomous Research Swarm

## Why this exists

S190A/V011 scored 0.17. That result is treated as an architectural falsification event, not a tuning request. V012 repairs the policy authority model, but V013 changes the *research process itself*: no single architecture, prompt, or agent is presumed to be the answer.

The goal is to search cognitive-architecture space with independent researchers, objective held-out evaluation, explicit controls, adversarial review, and reproducible promotion gates. The lab is designed to improve the probability of discovering a competitive ARC-AGI-3 agent. It cannot guarantee a leaderboard rank.

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

## Internal metrics

Per-run ARC suite receipts are converted into bounded internal metrics:

- solve rate;
- level progress;
- action efficiency;
- model-call efficiency;
- expectation accuracy;
- falsification health;
- stability;
- failure rate;
- timeout fraction;
- emergency-action fraction.

The official Kaggle score is retained as provenance when available but is deliberately **not** part of the internal research objective. The arena computes a weighted internal score and subtracts a one-standard-error uncertainty term across repeated seeds. Lucky single runs are intentionally disfavored.

A structurally valid suite remains scoreable if one game fails. Failure and timeout fractions penalize it continuously rather than erasing information from all successful games in the same run. Runner-level failures that do not produce a trustworthy receipt remain hard failures.

## Promotion

A challenger must have at least the configured number of validation runs, beat its explicit internal control by the configured validation margin, avoid material DEV regression, and stay below emergency/failure ceilings.

Leaderboard nomination is stricter:

1. the challenger must first pass internal promotion;
2. an actual Kaggle score for `A-duck-qwen38-public` must exist in the ledger;
3. an actual Kaggle score for the challenger must exist;
4. official score is compared directly with official score;
5. the challenger must meet the configured leaderboard delta.

This makes the S200 control a software-enforced dependency rather than a recommendation in prose.

## Split discipline

`SplitRegistry` deterministically partitions game IDs with a secret salt into DEV, VALIDATION, and BLIND. Exact split counts are allocated after salted hashing, so every split remains nonempty even for small development suites.

The public split registry exposes DEV and VALIDATION IDs plus only the count of blind games. The private registry contains blind identities and the secret salt and should stay out of research packets, normal agent prompts, git history, and ordinary development artifacts. `artifacts/arena/` is gitignored for this reason.

The manifest's `split_salt` field is descriptive only. Never put the real private salt into a committed config. Set `ARCANGEL_SPLIT_SALT` at runtime.

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

Round 1 is independent. Do not share leading-agent advice until each role has produced its own falsifiable hypothesis. `ProposalTournament` then deduplicates near-identical hypotheses, preserves role diversity through round-robin selection, rejects proposals that request BLIND/Kaggle as invention splits, and generates the implementation battle queue.

Only after independent proposals and measured arena results exist does `exchange_brief()` expose leading measured results, failed promotion gates, and peer hypotheses for Round 2. This implements exploration before information sharing and exploitation.

Each research proposal must state:

1. one primary hypothesis;
2. the smallest experiment capable of testing it;
3. target metric and split;
4. the result that falsifies the proposal;
5. concrete implementation/patch plan;
6. likely generalization failure mode.

## Research packet safety

`ResearchPacketBuilder` creates deterministic `.tar.gz` packets and removes blind scorecard rankings and files whose path identifies them as blind. This is a guardrail, not permission to store blind evidence carelessly elsewhere.

## Full autonomous development loop

```bash
# 1. Validate the tournament contract.
python scripts/run_swarm_lab.py validate

# 2. Discover ARC environments and create secret-salted DEV/VALIDATION/BLIND splits.
export ARCANGEL_SPLIT_SALT='<private-random-value>'
python scripts/init_swarm_splits.py

# 3. Start one local OpenAI-compatible Qwen3.8 server, then inspect B/C/D/E battles.
python scripts/run_swarm_lab.py plan --splits dev,validation

# 4. Execute enabled contestants. Serial-by-default avoids accidental GPU oversubscription.
python scripts/run_swarm_lab.py run --splits dev,validation

# 5. Score repeated runs and identify internally promoted challengers.
python scripts/run_swarm_lab.py score
python scripts/run_swarm_lab.py promote

# 6. Create a deterministic blind-safe research packet.
python scripts/run_swarm_lab.py packet

# 7. Run independent frontier-model research roles using configured development providers.
python scripts/run_research_swarm.py \
  --providers configs/research-providers.local.json

# 8. Convert independent proposals into a diverse implementation battle queue.
python scripts/plan_research_round.py

# 9. Implement selected proposals on isolated experiment branches and rerun the arena.
#    A proposal is killed when its declared falsifier fires.

# 10. Only after independent battles have measured outcomes, share the generated
#     round2-exchange.md with the next research round.

# 11. Judge internal survivors on BLIND using the private registry from a judge-owned run.
#     Do not hand blind identities or traces back to research agents.

# 12. Record the exact Duck Copy & Edit Kaggle control score and provenance.
python scripts/run_swarm_lab.py record-kaggle \
  --contestant A-duck-qwen38-public \
  --score <actual-score> \
  --seed <run-seed> \
  --source <saved-run-identifier> \
  --artifact-sha256 <sha256>

# 13. Record an internally promoted candidate only after its exact Kaggle Save & Run.
python scripts/run_swarm_lab.py record-kaggle \
  --contestant D-v012-evidence-first \
  --score <actual-score> \
  --seed <run-seed> \
  --source <saved-run-identifier> \
  --artifact-sha256 <sha256>

# 14. Ask the software gate whether any candidate actually beats the Duck control.
python scripts/run_swarm_lab.py leaderboard
```

A BLIND run is judge-owned and should use `scripts/run_arena_contestant.py` with `splits.private.json` explicitly. The ordinary B/C/D/E commands use only the public registry.

## External research providers

`configs/research-providers.example.json` documents the provider-agnostic OpenAI-compatible contract. Provider credentials are read only from named environment variables. The example configuration is disabled by default and contains no secrets.

`run_research_swarm.py --plan-only` can be used in CI without credentials. Live development calls are capped by `--max-requests` and run in parallel only up to `--max-workers`.

The provider swarm is intentionally replaceable. GPT, Fable/Claude-style agents, NVIDIA-hosted models, Gemini-compatible gateways, or future local research models can participate when an OpenAI-compatible endpoint or adapter is available. Their proposals have no promotion authority.

## What autonomy means here

The repository can autonomously handle:

- experiment planning;
- repeat-run bookkeeping;
- contestant subprocess execution;
- suite receipt normalization;
- uncertainty-adjusted scoring;
- control-relative promotion;
- Kaggle score provenance;
- leaderboard gating;
- deterministic split creation;
- blind-safe packet generation;
- independent provider-role fanout;
- proposal validation and deduplication;
- diversity-preserving battle selection;
- Round 2 exchange brief generation.

The system cannot manufacture GPU availability, Kaggle submissions, or external model credentials. Those external executions must produce receipts that are fed back into the same ledger. Autonomy is therefore bounded by actual compute and account access rather than simulated results.

## Immediate research sequence

1. Reproduce the exact Duck/Qwen3.8 Kaggle control and record the real score/provenance.
2. Run B/C/D/E on identical DEV/VALIDATION splits and three seeds.
3. Inspect failure families rather than only aggregate score.
4. If D loses to B, freeze full V012 and let E/B guide the next architecture.
5. If D or E wins, BLIND-judge it before building F-J on top.
6. Run the independent research swarm against measured failure evidence.
7. Activate exactly one selected mechanism at a time.
8. Only combine individually winning mechanisms in a separate integration experiment.
9. Nominate Kaggle candidates only through the actual-score leaderboard gate.

The desired end state is an evolutionary research loop where attractive ideas are cheap to propose, expensive to promote, and easy to kill.
