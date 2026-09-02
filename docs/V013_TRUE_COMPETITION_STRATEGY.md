# V013 True Competition Strategy

## Objective

ARCangel is not trying to build the most elaborate ARC-AGI-3 agent. It is trying to build the
smallest system that can turn a few expensive interactions with a novel world into reusable,
executable understanding quickly enough to solve hidden games.

The operational north star is:

> **maximize transferable causal knowledge gained per scored interaction, model call, and second.**

A mechanism survives only when paired evidence shows that it improves novel-game behavior. A
beautiful architecture diagram, a persuasive swarm proposal, a passing parser test, or one noisy
leaderboard score is not evidence of intelligence.

---

## 1. What ARC-AGI-3 actually demands

A hidden interactive game forces the agent to solve four coupled inference problems:

1. **Agency discovery** — what can I control and what do actions do?
2. **Mechanics discovery** — what state transitions and causal rules govern the world?
3. **Goal discovery** — which reachable states constitute progress or success?
4. **Execution** — how do I reach those states efficiently while detecting when my model is wrong?

These should be measured separately. A total score alone cannot tell us whether a candidate failed
because it never found the avatar, misunderstood a button, inferred the wrong goal, or planned badly
after learning the rules.

The agent should therefore behave like a compact experimental scientist:

`observe -> hypothesize -> choose discriminating intervention -> predict -> act -> compare -> update -> plan`

The loop matters more than any particular representation or memory format.

---

## 2. The architecture complexity ladder

Complexity is a hypothesis, not a virtue. We will make each additional cognitive layer earn its
runtime and generalization cost.

### External truth control: Duck + Qwen3.8

Use the public Duck implementation as an **external control**, copied essentially unchanged rather
than reconstructed inside ARCangel. Its purpose is to answer whether our entire local methodology
is operating in a credible performance lane.

Do not tune Duck into ARCangel. Once changed, it stops being a control.

### B — Coding Minimal

Model-led direct reasoning with the smallest generic harness. This is the internal Occam control.
If B beats richer architectures, delete complexity instead of explaining it away.

### E — V012 Lite

Retain only the evidence mechanisms that plausibly prevent expensive reasoning errors. This tests
whether a compact scientific scaffold helps without turning the harness into policy authority.

### D — V012 Evidence First

Full persistent evidence ledger, historical falsification, expectation validation, and executable
world-model support. This architecture must beat E enough to pay for its additional calls, tokens,
latency, and failure surface.

### C — V011

Historical negative control. It is useful for explaining what failed at 0.17, not as a target to
resurrect.

### Immediate decision tree

1. **Does V012 beat Coding Minimal?**
   - No: simplify. The burden shifts to minimal/model-led systems.
   - Yes: continue to the next question.
2. **Does V012 beat V012 Lite by enough to justify full machinery?**
   - No: V012 Lite becomes the scientific scaffold.
   - Yes: retain the exact V012 mechanisms responsible for the measured gain.
3. Only after this baseline decision do we add new mechanisms.

Generation 1 independently converged on these comparisons. Two selected proposals asked the same
V012/Coding-Minimal question in opposite directions; ARCangel now deduplicates that pair rather than
spending two experiments.

---

## 3. Measurement before mutation

Every council-selected hypothesis is routed into one of two lanes.

### Lane A — existing-profile measurement

If current runnable profiles can falsify the hypothesis, run the comparison immediately. **Do not
rewrite either treatment first.**

Example:

- hypothesis: exact evidence history improves causal learning;
- target: V012;
- control: V012 Lite;
- action: paired Qwen DEV -> repeated VALIDATION.

No coding model belongs in this loop.

### Lane B — cognition mutation

Only genuinely new mechanisms enter the coding worker. A mutation must:

- change one falsifiable mechanism;
- touch cognition-owned paths only;
- preserve judge/evaluator/configuration authority;
- compile and pass tests;
- enter the same paired DEV/VALIDATION evidence ladder;
- be discarded if the measured gain does not survive.

This prevents the swarm from turning every research question into feature accretion.

---

## 4. The mechanism tournament

After the complexity ladder establishes a credible base architecture, new work is organized into
orthogonal mechanism tracks. One mechanism is changed per experiment.

### Representation

Compete:

- rendered frame only;
- raw grid/ASCII only;
- temporal delta view;
- connected components / topology;
- adaptive model-requested views.

The hypothesis is not that more representations are better. The hypothesis is that the model can
request the representation that reduces uncertainty **when it needs it**, without permanently
paying the token/latency cost.

### Exploration

Compete:

- direct model-selected probes;
- novelty/coverage probes;
- reversible experiments;
- disagreement / information-gain probes;
- counterfactual discrimination between competing mechanics.

A probe should be valued for expected information gained relative to its action/death cost, not
because a hand-written heuristic declares an object interesting.

### Memory

Compete:

- recent context only;
- selective context eviction;
- compact scientific summary;
- exact queryable evidence ledger;
- retrodiction-enforced memory, where a claimed rule must explain prior transitions.

Memory is useful only if it lowers relearning and contradiction while preserving action efficiency.

### World modeling

Compete:

- no explicit model;
- natural-language causal hypotheses;
- executable transition model;
- retrodiction before prospective planning.

A world model is promoted only when its predictions improve future decisions. Code that merely
looks like a simulator does not count.

### Planning

Compete:

- one action at a time;
- short 2–4 action queues;
- longer plans only after mechanics are validated;
- adaptive horizon based on model confidence and expectation accuracy.

Long plans should be rare before causal rules are stable.

---

## 5. The Generalization Suite

Public games are not a trustworthy development objective. ARCangel needs its own hidden-like
curriculum with strict **DEV / VALIDATION / BLIND** separation.

Procedural microgame families should vary the underlying causal concept while changing superficial
appearance aggressively.

### Agency families

- unknown action mappings;
- rotated/permuted controls;
- multiple controllable objects;
- delayed action effects;
- mode-dependent controls.

### Goal families

- touch / avoid / collect;
- activate all / match / transform;
- open / unlock / deliver;
- reach state in a particular temporal phase;
- goals whose visual salience is deliberately misleading.

### Causal-mechanic families

- button -> door;
- key -> lock;
- push chains;
- toggles and reversible transforms;
- resource/state dependencies;
- one-shot versus persistent switches.

### Temporal families

- moving hazards;
- periodic gates;
- delayed rewards;
- cooldowns;
- phase-sensitive actions;
- nonstationary mechanics with observable context.

### Abstract families

- symmetry;
- counting;
- color or symbol remapping;
- spatial correspondence;
- object transformation;
- relational rules that cannot be solved by nearest-object navigation.

### Perturbation protocol

A mechanism that works only under one palette or orientation is not general. Validation should
include held-out transformations such as:

- color permutation;
- rotation/reflection;
- distractor insertion;
- object-size changes;
- action remapping;
- longer horizons;
- sparse versus dense layouts.

Research agents never see BLIND identities or BLIND outcomes.

---

## 6. Metrics that diagnose cognition

### Outcome

- games solved;
- levels solved;
- robust utility across seeds/families.

### Discovery

- actions before controllable agency is identified;
- actions before a stable mechanic is established;
- actions before a plausible goal is identified;
- number of contradicted hypotheses before recovery.

### Scientific behavior

- transition-prediction accuracy;
- expectation accuracy for planned actions;
- proportion of hypotheses that receive an actual discriminating test;
- falsification/revision rate;
- retrodiction consistency.

### Efficiency

- actions per solved level;
- model calls per solved level;
- tool calls per solved level;
- tokens and wall time per game;
- repeated/no-op actions;
- deaths and resets.

### Runtime robustness

- model-call failure rate;
- parser failure rate;
- emergency/fallback fraction;
- games never started;
- per-game timeout rate;
- GPU throughput and memory pressure.

No single metric becomes policy authority. The scorecard exists to reveal *why* a system wins or
loses.

---

## 7. Statistical discipline

ARC-AGI-3 and the public leaderboard are noisy enough that one result is weak evidence.

For internal comparisons:

- use paired seeds whenever possible;
- alternate candidate/control execution order;
- use repeat-aware robust deltas;
- prefer conservative lower estimates over best-run scores;
- promote only when the gain persists across DEV -> VALIDATION;
- retain failure-family breakdowns, not only the aggregate.

For Kaggle:

- notebook bytes are the experiment identity;
- never pool different notebook hashes as one treatment;
- repeat the exact artifact;
- if repeats disagree materially, obtain another repeat rather than declaring victory;
- judge challengers relative to repeated Duck-control evidence, not an absolute public score;
- never tune architecture directly to a single public leaderboard fluctuation.

The historical S190A score of **0.17** remains a falsification anchor. A descendant must not merely
be cleaner software; it must demonstrate a different behavioral regime.

---

## 8. Compute allocation

There are three clocks.

### Research clock — cheap and parallel

NVIDIA-hosted heterogeneous agents generate independent hypotheses and blinded reviews. The swarm
searches **hypothesis space**, not the action space of a live scored game.

### Empirical clock — expensive GPU

One verified Qwen3.8 27B FP8 server stays resident while candidate/control pairs are evaluated.
Use successive halving:

1. one-seed DEV screen;
2. eliminate regressions/unhealthy candidates;
3. repeated VALIDATION for survivors;
4. private BLIND only for repeatable winners.

Do not spend GPU on ideas that can be rejected statically or by an already-measured comparison.

### Leaderboard clock — scarce and noisy

Kaggle is the final external instrument, not the optimizer. Submit only exact artifacts that already
passed the internal evidence ladder.

---

## 9. Failure taxonomy and response

| Failure | Interpretation | Response |
|---|---|---|
| cannot discover control | agency problem | test representation/probing, not planning |
| discovers effects but no reusable rule | mechanics problem | test memory/retrodiction/world model |
| understands mechanics but pursues wrong state | goal problem | test goal inference and progress evidence |
| understands world but wastes actions | planning problem | test horizon/replanning |
| strong early games, starves later games | scheduler problem | runtime/coverage fix, not cognition claim |
| model calls succeed but heuristics own actions | authority failure | remove heuristic policy authority |
| software green, behavior poor | scientific falsification | redesign architecture, do not tune parser |
| LB varies wildly for exact artifact | external measurement noise | repeat exact bytes; do not feature-chase |

This prevents a failure in one layer from triggering unrelated feature additions.

---

## 10. Promotion hierarchy

ARCangel uses five distinct gates:

1. **Software qualification** — code executes, tests pass, scope is valid.
2. **Scientific qualification** — experiment isolates a falsifiable mechanism and has a fair control.
3. **Behavioral qualification** — paired DEV/VALIDATION evidence shows a repeatable gain.
4. **Generalization qualification** — private BLIND / held-out families retain the gain.
5. **Leaderboard qualification** — the exact packaged artifact shows credible repeated external evidence.

Passing an earlier gate says nothing about the next one.

---

## 11. September operating plan

### Sep 2–4 — establish truth

- freeze measurement-first V013 machinery;
- reproduce the exact public Duck/Qwen3.8 lane;
- run the B/E/D complexity tournament on one verified Qwen server;
- resolve V012 vs Coding Minimal and V012 vs V012 Lite by measurement.

### Sep 5–10 — isolate mechanisms

- run representation battle;
- run exploration/probe battle;
- run memory/retrodiction battle;
- kill mechanisms that do not survive paired VALIDATION.

### Sep 11–17 — attack generalization

- expand procedural hidden-like suite;
- run palette/orientation/action-map/distractor transformations;
- red-team finalists for accidental public-game assumptions.

### Sep 18–23 — build finalists

- combine only mechanisms with independently measured value;
- treat integration itself as a new experiment;
- reduce unnecessary tokens/calls/latency.

### Sep 24–27 — private qualification

- repeated VALIDATION;
- private BLIND;
- runtime stress under competition-scale scheduling.

### Sep 28–30 — milestone evidence

- freeze exact finalist bytes;
- repeat exact Duck control and challenger artifacts as budget permits;
- submit the candidate whose internal and external evidence agree;
- preserve a rollback candidate if leaderboard variance is inconclusive.

---

## 12. What winning looks like

The desired final system is not a giant committee and not a library of public-game tricks.

It is likely to be a compact model-led coding/science agent with:

- adaptive access to raw and derived observations;
- a small memory that preserves causal evidence without fossilizing guesses;
- an explicit distinction between observation, hypothesis, and validated rule;
- discriminating probes when uncertainty matters;
- longer plans only after mechanics are trustworthy;
- rapid contradiction detection and repair;
- aggressive context/tool/runtime discipline.

The development swarm may be large. The competition-time agent should remain as small as the
evidence allows.

**Swarm searches. Measurement decides. Simplicity survives until complexity proves itself.**
