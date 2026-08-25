# ARCangel Competitive Architecture Review — 2026-08-24

This document evaluates the strongest public ARC-AGI-3 approaches, recent Kaggle evidence, benchmark-design constraints, and the current ARCangel V009 architecture. Its purpose is not to imitate the public leaderboard. The private competition set is intentionally out-of-distribution relative to the 25 public environments, so the real objective is to identify mechanisms that should remain useful when mechanics, visual layouts, goals, and compositions are genuinely novel.

## Executive thesis

The public evidence does **not** support one universally best harness. Instead it reveals a recurring pattern:

> Strong ARC-AGI-3 agents preserve exact evidence, orient before acting, keep the semantic decision-maker flexible, and invoke heavier abstractions only when uncertainty justifies them.

The best system should therefore not be a monolithic world-model agent, a pure VLM policy, a brute-force explorer, or an over-engineered stack of heuristics. ARCangel should become an **adaptive abstraction operating system** that can fluidly choose between direct action, hypothesis testing, systematic exploration, history analysis, executable modeling, and verified planning.

The flagship target is:

```text
RAW CURRENT FRAME
+ RECENT FRAME HISTORY
+ LOSSLESS INTERACTION LEDGER
        ↓
PERCEPTUAL STATE ESTIMATOR
        ↓
AGENCY / OBJECT / RELATION / TOPOLOGY / TEMPORAL HYPOTHESES
        ↓
MECHANIC HYPOTHESES + EXECUTABLE GOAL HYPOTHESES
        ↓
ACTOR chooses cognitive mode
        ├─ ACT_DIRECTLY
        ├─ TEST_HYPOTHESIS
        ├─ EXPLORE_FRONTIER
        ├─ QUERY_HISTORY / RETRODICT
        ├─ BUILD_OR_REPAIR_MODEL
        └─ EXECUTE_VERIFIED_PLAN
        ↓
REAL ACTION
        ↓
PREDICTION / GOAL / MODEL CHECK
        ↓
UPDATE, REPAIR, OR COMMIT
```

The central design rule is:

> **Use exact code to preserve and test evidence, but let the reasoning model decide what the evidence means.**

---

# 1. What the benchmark itself tells us

The official ARC-AGI-3 technical report is more useful for architecture design than most public-game reverse engineering.

ARC-AGI-3 environments are deliberately built from **Core Knowledge priors** rather than cultural or language conventions. The official design principles explicitly include:

- **objectness**: coherent persistent entities that can move, collide, and be occluded;
- **basic geometry and topology**: symmetry, rotation, inside/outside, connectedness, holes;
- **basic physics**: gravity, momentum, bouncing;
- **agentness**: recognizing that some entities act intentionally;
- no numbers, letters, culturally meaningful clip-art, or assumptions such as green meaning “go”.

This has direct architectural consequences.

We should encode strong priors for **object persistence, geometry, topology, causality, temporal structure, and agency**, but weak priors for semantic labels such as “key”, “enemy”, “door”, or color meaning.

The benchmark is also deliberately composed across levels:

- the first level is tutorial-like and designed to communicate the core interaction pattern;
- every environment contains multiple mechanics;
- later levels combine concepts learned earlier instead of merely scaling one mechanic;
- the private set covers broader mechanics and deeper composition with limited public overlap.

Therefore the correct unit of cross-level transfer is not “repeat the previous solution.” It is:

```text
verified controls
+ verified mechanics
+ object-role hypotheses
+ causal relations
+ goal evidence
```

with the actual goal and route re-inferred on each level.

The benchmark counts only real environment actions. Internal computation, history search, Python analysis, and retries inside the model do not count as scored actions. This strongly favors **expensive cognition before cheap but risky environment interaction**.

### Competitive implication

A contender should act less like a reactive game player and more like a scientist performing costly experiments.

Every real action should ideally purchase one of only three things:

1. direct progress toward a sufficiently credible goal;
2. information that resolves a pivotal uncertainty;
3. execution of a verified low-risk plan.

Anything else is likely wasted score.

---

# 2. Public evidence must be weighted by evaluation relevance

There are several very different kinds of public evidence and they should not be conflated.

## 2.1 Kaggle competition scores

These use the actual competition compute/runtime boundary and are therefore highly relevant operationally. Current public notebook tiles include roughly 1.17–1.22 scores for several Duck-derived or custom submissions, with a copied Duck fast-eval notebook showing a historical best of 1.47.

However, these values are noisy. A public Duck-harness reproduction project reports that one fixed configuration can have a very broad 25-game score distribution, and that single-run A/B comparisons are often not statistically meaningful. Their own ablations also found that adding a state graph regressed one stack while fuller frame presentation helped substantially.

**Lesson:** small leaderboard deltas are weak architectural evidence. Large regime changes and repeated ablations matter more.

## 2.2 Milestone #1 winners

These are especially valuable because they were real open-source competition submissions under the Kaggle constraints.

## 2.3 Public-demo research agents at 99–100%

Tycho, Retrodict, baseline1, Twin, and similar systems expose extremely useful reasoning mechanisms. But many use frontier API models, large token budgets, or local public environment execution that cannot be reproduced inside one RTX PRO 6000 Kaggle submission.

Their mechanisms should be mined. Their absolute scores should not be treated as expected Kaggle performance.

## 2.4 The 25 public environments

The official technical report explicitly says the public set is a demonstration interface and does **not** comprehensively represent private mechanics. The private set is intentionally harder and OOD.

Therefore:

> A mechanism that increases public score by exploiting public regularities may be negative evidence for the private set.

ARCangel should optimize for **invariances and learning procedures**, not for public-game coverage.

---

# 3. The Duck: what the Milestone #1 winner really teaches us

Tufa Labs’ Duck is the most important Kaggle-native reference.

Its architecture is deliberately simple:

- Qwen3.6 27B FP8;
- live Python REPL;
- current rendered image;
- raw ASCII grid;
- segmentation/zoom helpers;
- short rolling context with oldest-message eviction;
- legal-action checks and short action queues.

Tufa’s stated lesson is unusually important: **hand-crafted tools often hurt**. Gains came more from multimodality and base-model quality than from increasing harness complexity.

Their own results are highly uneven by game, which suggests the harness succeeds when the base model can discover the right abstraction and fails badly when it cannot.

### What ARCangel should adopt

- Treat Qwen as a capable scientist, not merely a policy head.
- Give it multiple representations and let it choose which one matters.
- Keep the system prompt stable and generic.
- Keep working context short enough for throughput.
- Preserve Python as a first-class reasoning substrate.
- Avoid forcing every game into the same explicit representation.

### What ARCangel should improve

Duck’s rolling context can forget exact historical evidence. ARCangel should keep a **lossless external interaction ledger** even while keeping the model context short.

Duck also largely relies on the model itself to decide when a proposed rule is trustworthy. ARCangel should add **retrodiction, expectation checking, and explicit uncertainty** without cluttering the model-facing interface.

The competitive goal is therefore:

> Duck-like semantic flexibility + exact external epistemic discipline.

---

# 4. Reki and Forge: vision can work, orchestration can hurt

Reki’s core is a VLM policy:

- render recent frames as labeled images;
- feed them to Gemma-4-31B;
- ask for a structured JSON containing observation, plan, and the next 1–4 actions;
- maintain reflection memory;
- use a lightweight click prior for small/rare/button-like objects.

Forge adds more machinery such as candidate generators, arbiters, and optional confidence mechanisms. The remarkable public lesson is that Forge’s top-scoring profile disabled much of that extra machinery.

### What ARCangel should adopt

- Recent frames should be first-class visual evidence.
- Structured output is useful for legality and plan handling.
- Small generic perception priors can help when they remain soft.

### What ARCangel should avoid

- A hierarchy of agents scoring other agents’ guesses.
- Scalar hand-tuned action scores presented as authority.
- Requiring the model to justify itself through several orchestration layers before every move.

ARCangel V009’s removal of the V008 composite candidate score is therefore directionally correct. The model should see causal evidence, risk, support, and uncertainty, but not be forced to obey a manually invented scalar utility unless that utility has a principled meaning.

---

# 5. PRO-LONG: memory should be lossless and programmatic

PRO-LONG is one of the clearest architectural results in the public literature.

Its main idea is tiny:

- append every observation, action, and outcome to one structured log;
- let the coding agent search that log with Python/grep-like tools;
- do not try to squeeze the entire history into the prompt.

The paper reports large gains over the same base coding agents and substantial token-efficiency improvements.

### Core lesson

> Long-horizon memory should be an external database of evidence, not a prose summary pretending to be evidence.

Summaries should be caches. The log is ground truth.

### ARCangel implication

We should preserve two layers:

**Evidence ledger**

```text
exact settled frame
exact action / click target
exact resulting frame
pixel delta
entity delta
terminal/progress result
```

**Working model**

```text
controls believed known
mechanics believed known
current goal hypotheses
pivotal unresolved questions
current plan
```

The second can be rewritten or compacted. The first must remain immutable and queryable.

---

# 6. Retrodict: the strongest public epistemic discipline

Retrodict is currently one of the most instructive systems because it adds almost no mysterious machinery. It behaves like a scientist with a lab notebook.

Key mechanisms:

- every frame is logged;
- mechanic hypotheses are tested against prior history before live actions are spent;
- supported plans carry explicit predicted board consequences;
- the runner stops a queue at the first prediction mismatch;
- a curated playbook persists across context resets;
- only after prolonged failure does it escalate into constructing a full executable simulator;
- initial vision is used as a hypothesis primer, not accepted as fact.

### What ARCangel should adopt almost directly

**Retrodiction-first reasoning**

A mechanic hypothesis should answer:

```text
Which historical transitions does this explain?
Which ones contradict it?
What prediction does it make that alternatives do not?
```

**Every batched action should have a prediction contract.**

If ARCangel says it understands an action strongly enough to execute three moves without waking Qwen, it should also know what each move is expected to change.

**Context reset should preserve a curated playbook plus exact ledger.**

### Where ARCangel can go beyond Retrodict

Retrodict uses image priming mainly at the beginning and then becomes heavily log/text oriented. ARCangel can maintain **continuous multimodal orientation** on selected high-information turns.

ARCangel should also make **goal hypotheses** as testable as mechanic hypotheses. Retrodiction currently has stronger rule discipline than goal discipline.

This may be one of the largest remaining opportunities.

---

# 7. Executable world models: powerful, but not universally correct

Several systems explore executable world modeling.

The baseline1/Rodionov lineage constructs a Python simulator, verifies it, and plans through it. Verification generally helps, but ablations show that requiring a persistent executable model is not universally beneficial. Stronger base models and more reasoning effort sometimes matter more than individual architecture components.

Tycho provides the more useful orchestration lesson:

> **The actor should decide when a world model is worth building.**

Actor-requested model-builder delegation outperformed forcing the modeling process on every situation.

Twin demonstrates the upside when world modeling succeeds: a coding agent constructs a test-time digital twin, repairs it when real transitions disagree, and plans inside it. Its most revealing conclusion is that **goal inference remains harder than reconstructing dynamics**.

### ARCangel design

World-model construction should be an optional cognitive mode:

```text
BUILD_MODEL
REPAIR_MODEL
BYPASS_MODEL
```

The model should be built when:

- a plan requires multi-step counterfactual reasoning;
- important dynamics are understood but combinatorially hard to search mentally;
- repeated interactions share a compact transition law;
- the expected planning value exceeds the inference/tool cost.

It should be bypassed when:

- the game is simple enough to solve directly;
- goal uncertainty dominates transition uncertainty;
- the interaction is primarily visual/relational and a simulator would be brittle;
- the model’s own prediction errors are concentrated on pivotal dynamics.

---

# 8. Play-adequacy: transition accuracy is the wrong gate by itself

Recent work on verified world models shows that a simulator can reach extremely high average transition accuracy while still losing because the rare mistake occurs on exactly the dynamic that controls the winning plan.

This is a critical warning for ARCangel’s predictive-memory work.

Our D210 result that covered h2 transitions are nearly exact is useful, but it cannot be the final criterion for controller authority.

We need a second concept:

## Pivotality

A hypothesis matters in proportion to the damage caused if it is wrong.

For a proposed plan, approximate:

```text
pivotality(H)
≈ P(H is wrong)
× expected plan damage if wrong
× future score depending on H
```

Before committing a long plan, verify the high-pivotality assumptions rather than sampling rules uniformly.

This leads naturally to **hypothesis-discriminating probes**.

---

# 9. Just Explore: systematic coverage is a superb fallback, not the whole intelligence layer

The graph-based “Explore It Till You Solve It” approach demonstrates how far systematic exploration can go:

- hash visual states;
- record tested/untested actions;
- build transitions;
- choose unexplored edges according to action priorities;
- route back to frontier states rather than repeatedly probing exhausted states.

Its authors report that this outperformed their earlier more LLM-centric ideas in the developer-preview setting.

This is extremely valuable as an **anti-thrashing guarantee**.

### ARCangel should retain

- exact state-action frontier bookkeeping;
- shortest known-safe routes to unresolved states;
- terminal-risk tracking;
- no repeated testing of known deterministic edges unless the context changed.

### ARCangel should not do

- equate novelty with progress;
- brute-force giant private state spaces;
- let frontier search dominate when semantic reasoning has already found a credible goal;
- assume public click-priority heuristics transfer to OOD private mechanics.

Therefore V009’s frontier should remain a fallback epistemic substrate, not the main policy.

---

# 10. Public Duck ablations: perception quality appears more valuable than extra cognition modules

An independent Duck-harness reproduction project reports a useful set of ablations:

- using fuller frame information produced a large gain relative to their frozen baseline;
- an additional HUD code model was approximately flat;
- adding a state graph to an already complex stack regressed performance and was disabled by default;
- run-to-run score variance is large enough that single-run comparisons can be misleading.

These are self-reported public experiments, not official private evidence, but they fit the Milestone #1 story.

### Design implication

Before adding another “intelligence” module, ask whether the model is simply missing the right pixels, temporal state, or exact historical evidence.

ARCangel should prioritize:

1. perceptual data quality;
2. correct task orientation;
3. exact memory;
4. falsifiable hypotheses;

before increasing orchestration depth.

---

# 11. Official model-replay analysis identifies our most important failure modes

ARC Prize’s replay analysis of frontier models identifies three especially relevant failures:

1. **True local effect, false world model**: the agent observes a real local change but extrapolates the wrong global rule.
2. **Wrong level of abstraction from training data**: the model forces the new environment into a familiar game schema.
3. **Solved the level, didn’t learn the game**: success occurs, but the model fails to extract the causal lesson for later levels.

The GPT-5.6 analysis adds another important result: failures are often upstream of code/action execution. Correct orientation is a major determinant of success.

ARCangel should therefore explicitly model four forms of uncertainty:

```text
agency uncertainty
mechanic uncertainty
goal uncertainty
abstraction uncertainty
```

The fourth is important. The system should be able to say:

> “I have evidence for these observations, but I do not yet know whether the correct abstraction is navigation, transformation, matching, temporal synchronization, or some combination.”

This is better than silently choosing the wrong schema.

---

# 12. The architecture ARCangel should become

## Layer 0 — Exact observation substrate

Preserve:

- all settled frames;
- animation frames separately from action states;
- raw grid values;
- current high-resolution render;
- temporal visual packet;
- exact per-action frame diff;
- volatile/HUD candidates without assuming they are irrelevant.

Do not prematurely throw information away.

## Layer 1 — Multi-hypothesis perceptual scene

Maintain several possible decompositions when objectness is ambiguous.

Represent:

```text
pixels
components
multipart groups
persistent tracks
regions
spatial relations
topological relations
symmetries
motion groups
```

A component is evidence for an object, not definitionally an object.

## Layer 2 — Causal interaction graph

Learn relations such as:

```text
action → actor displacement
action → object transformation
click(signature) → remote change
contact(A,B) → change(C)
state predicate → action effectiveness
```

Semantics should emerge from causal role rather than visual stereotypes.

## Layer 3 — Hypothesis registry

Every important hypothesis should become a typed record:

```text
Hypothesis
  type: agency | mechanic | goal | abstraction | temporal
  proposition / executable predicate
  evidence
  counterevidence
  confidence
  predictions
  last_tested
  source_level
  pivotality
```

Confidence should be derived from evidence where possible, not merely supplied by the LLM.

## Layer 4 — Executable goal language

This should be a first-class ARCangel research priority.

Generic goal primitives should include families such as:

```text
TOUCH(A,B)
ENTER(A,R)
EXIT(R)
ALIGN(objects)
MATCH(A,B)
RESTORE_SYMMETRY(axis)
REMOVE_ALL(class)
ACTIVATE_ALL(class)
COLLECT_ALL(class)
COUNT(class) == N
COVER(region)
CLEAR(region)
TRANSFORM(A,signature)
SURVIVE(phase)
REACH(predicate)
SEQUENCE(pattern)
```

These primitives are not public-game rules. They are Core-Knowledge-compatible relational concepts.

The LLM can also write a free-form Python `goal_test(state)` when the generic DSL is insufficient.

Goal hypotheses must be retrodicted against prior level-completion evidence.

## Layer 5 — Active scientist / cognitive-mode controller

The actor chooses:

```text
ACT_DIRECTLY
QUERY_HISTORY
TEST_HYPOTHESIS
EXPLORE_FRONTIER
BUILD_MODEL
REPAIR_MODEL
EXECUTE_VERIFIED_PLAN
```

This controller should use uncertainty structure, not dozens of fixed thresholds.

## Layer 6 — Hypothesis-discriminating probe selection

For unresolved hypotheses H, prefer an action with high expected information gain:

```text
IG(a) = H(H) - E[H(H | observation after a)]
```

Then weight by pivotality and risk:

```text
probe_value(a)
≈ information_gain(a)
× pivotality_of_resolved_question
- game_over_risk(a)
- irreversible_cost(a)
```

This is more principled than “pick the least tried action.”

## Layer 7 — Optional executable twin

When requested by the actor, a model builder creates:

```python
transition(state, action)
candidate_actions(state)
goal_test(state)
```

It must explain the relevant historical transitions before earning authority.

But transition replay alone is insufficient. Before long execution it should also pass a **play-adequacy gate** focused on assumptions used by the proposed route.

## Layer 8 — Verified plan compiler

Once semantics are grounded:

- BFS/A* for deterministic spatial control;
- time-expanded search for periodic mechanics;
- symbolic search for switches/inventory/state variables;
- bounded search inside an executable twin;
- short batched action queues with per-step expectations.

Wake the scientist on the first prediction mismatch.

## Layer 9 — Continual-learning distillation at level completion

Every completed level should trigger an explicit learning pass:

```text
What actions were causally necessary?
Which mechanic hypotheses were confirmed?
Which goal predicate best explains completion?
Which previously assumed rules were irrelevant?
Which knowledge should transfer to the next level?
Which goal/layout details must NOT transfer?
```

Promote mechanics and goals separately.

This directly attacks the official “solved the level, didn’t learn the game” failure.

## Layer 10 — Campaign compute allocator

The Kaggle competition is not merely 110 independent games. It is a fixed GPU-time portfolio.

Allocate inference based on expected marginal score per GPU-second.

Useful features:

- level/depth reached;
- recent progress velocity;
- orientation entropy;
- goal confidence;
- mechanic confidence;
- model-call latency/failure rate;
- existence of a verified controller;
- frontier size;
- estimated probability that more reasoning yields another level.

Do not let one pathological game consume the campaign while other games remain one easy inference call from scoring.

---

# 13. How ARCangel can be better than each public family

| Public family | Best lesson | Main weakness for our setting | ARCangel advantage to build |
|---|---|---|---|
| Duck | lightweight multimodal coding agent | historical evidence can be context-fragile; orientation failures remain uneven | Duck flexibility + lossless ledger + explicit epistemic discipline |
| Reki | recent visual frames can directly drive policy | limited exact causal/history machinery | dual-view VLM + executable evidence |
| Forge | configurable orchestration | extra generator/arbiter machinery hurt top profile | keep evidence rich, arbitration simple |
| PRO-LONG | full history + programmatic retrieval | memory architecture alone does not solve perception/goal acquisition | programmatic memory + continuous multimodal orientation |
| Retrodict | falsify rules on history before real actions | mostly text/log after initial image; goal induction less formal | retrodict mechanics **and goals**, maintain selective visual calls |
| baseline1 | executable verified world model | persistent model not always worth the cost | actor-gated modeling |
| Tycho | active abstraction / actor decides modeling | frontier models and expensive public evaluation | reproduce the control principle under Qwen/Kaggle limits |
| Twin | repairable test-time digital twin | expensive; goal inference still harder | twin only when valuable + first-class goal DSL |
| Just Explore | systematic frontier eliminates thrashing | brute force scales poorly and ignores semantic progress | frontier as safe fallback below a scientist |
| public Duck forks | perception/context tweaks can outperform extra modules | noisy public scores and public-specific ablations | replicate on procedural held-outs before private promotion |

---

# 14. Training and development should target the learning process, not public solutions

The private set is deliberately OOD. Training directly on public-game solutions risks teaching exactly the wrong thing: recognition of a narrow public task manifold.

ARCangel should instead build a procedural **ARCangel Arena** derived from the official design priors.

## Randomize surface form aggressively

- color permutations;
- action-ID permutations;
- translations;
- rotations/reflections;
- sprite shape changes;
- distractors;
- border/HUD variation;
- board dimensions;
- multipart-object renderings.

## Randomize mechanics composition

Mechanic families:

- motion;
- collision;
- pushing/pulling;
- portals;
- toggles;
- gates;
- containment;
- collection;
- transformation;
- matching;
- symmetry;
- temporal cycles;
- moving agents;
- hazards;
- irreversible actions;
- local/global effects.

## Randomize goals independently from mechanics

The same control dynamics should sometimes correspond to different goals. This prevents the model from conflating “how the world works” with “what it wants me to do.”

## Hold out compositions, not seeds

Development might contain:

```text
movement + gate
movement + portal
click + transform
```

while held-out evaluation contains:

```text
movement + gate + portal + transform
```

This better approximates private compositional novelty.

---

# 15. Candidate learning objectives

If we fine-tune or distill an open model, useful targets include:

## Perceptual objectives

- persistent entity correspondence across frames;
- multipart grouping;
- controlled-entity identification;
- topology and bottleneck recognition;
- temporal-vs-causal change separation.

## Mechanic objectives

Given a short trajectory:

- infer action semantics;
- propose competing causal rules;
- predict which historical examples falsify a rule;
- predict the minimum discriminating probe.

## Goal objectives

Given several levels and their completion transitions:

- infer the simplest executable predicate consistent with success;
- distinguish goal evidence from mechanics evidence;
- reject a goal after counterexamples;
- transfer abstract goal families without transferring coordinates/colors.

## Epistemic-control objectives

Choose among:

```text
act
probe
search history
build model
plan
```

based on what information is missing.

## Metamorphic consistency objectives

The same reasoning should survive color/action/coordinate transformations.

## Process-supervision objective

Reward the reasoning process for:

- citing evidence;
- identifying counterevidence;
- distinguishing known from assumed;
- predicting before acting;
- waking on contradiction.

This may be a better training signal than teaching final action sequences.

---

# 16. The experimental program should become adversarial

Before spending private submissions, every architecture should pass three families of tests.

## A. Metamorphic tests

Apply transformations that should preserve the abstract solution:

- recolor;
- rotate;
- reflect;
- translate;
- permute action IDs;
- add inert distractors;
- alter HUD presentation.

Measure whether the inferred mechanics/goals transform appropriately.

## B. Counterfactual fault injection

Deliberately give the agent misleading early evidence:

- action appears deterministic twice then changes contextually;
- visually salient object is irrelevant;
- initially plausible goal predicate is false;
- apparent actor turns out to be autonomous;
- rare pivotal rule appears late.

Measure contradiction recovery.

## C. Composition hold-outs

Evaluate on mechanic combinations never seen together during development.

This should become the main promotion benchmark, not raw performance on the public 25.

---

# 17. Proposed V010 roadmap

V009 is a useful contender experiment, but the long-term target should be called something like **V010 Active Scientist**.

The highest-priority additions are:

1. **Lossless log as an explicit first-class API** rather than only compact memory structures.
2. **Typed hypothesis registry** for agency, mechanics, goals, abstraction, and temporal phase.
3. **Executable goal DSL** with retrodiction against completion history.
4. **Pivotality-aware probe selector** rather than generic novelty.
5. **Actor-selected cognitive mode** including explicit world-model bypass.
6. **Play-adequacy validation** on assumptions used by a proposed plan.
7. **Post-win causal distillation** so a solved tutorial level becomes transferable knowledge.
8. **Procedural ARCangel Arena** for held-out compositional evaluation.
9. **Compute allocator** that optimizes marginal expected score per GPU-second.
10. **Model bakeoff** under the same harness, especially Qwen3.6-27B vs any Kaggle-feasible Gemma-4-class alternative.

---

# 18. What we should explicitly reject

Do not build ARCangel around:

- public game IDs;
- memorized action mappings;
- fixed semantic color meanings;
- public source-code mechanics;
- a mandatory world model;
- a mandatory object segmentation interpretation;
- a giant hand-designed scalar action score;
- repeated primitive probing after mechanics are already established;
- blind transfer of full plans between levels;
- one giant prompt containing the entire trajectory;
- leaderboard deltas from one stochastic public run as a promotion gate;
- transition accuracy as the sole world-model quality measure.

---

# 19. The competitive north star

The strongest public systems each solve only part of the general problem elegantly.

ARCangel should aim to combine their transferable strengths without combining all of their complexity.

The intended cognitive loop is:

```text
SEE
 ↓
ORIENT
 ↓
TRACK WHAT PERSISTS
 ↓
IDENTIFY WHAT IS CONTROLLABLE
 ↓
FORM COMPETING MECHANIC + GOAL HYPOTHESES
 ↓
ASK WHAT CURRENTLY BLOCKS A SOLUTION
 ↓
CAN HISTORY ANSWER IT?
 ├─ yes → RETRODICT
 └─ no  → CHOOSE CHEAPEST PIVOTAL PROBE
 ↓
ENOUGH UNDERSTANDING TO PLAN?
 ├─ no → EXPLORE / MODEL / REPAIR
 └─ yes → COMPILE VERIFIED PLAN
 ↓
EXECUTE CHEAPLY
 ↓
SURPRISE?
 ├─ no → continue
 └─ yes → WAKE SCIENTIST
 ↓
LEVEL COMPLETE
 ↓
DISTILL WHAT WAS ACTUALLY LEARNED
 ↓
TRANSFER MECHANICS, RE-INFER GOAL/ROUTE
```

The point is not to out-feature every competitor.

The point is to build a system with a better **epistemology**:

- exact evidence survives context;
- assumptions are represented as assumptions;
- goals are falsifiable;
- actions are purchased for a reason;
- models earn authority;
- plans carry predictions;
- contradictions cause immediate repair;
- success is converted into reusable knowledge.

That is the architecture most likely to remain useful when the game is genuinely new.

---

# Public references reviewed

- ARC Prize Foundation, **ARC-AGI-3 Technical Report**: https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
- ARC Prize, **Milestone Prize #1**: https://arcprize.org/blog/arc-prize-2026-milestone-1
- Tufa Labs, **Duck Harness**: https://tufalabs.ai/research/duck-harness/
- Tufa Labs Duck source: https://github.com/Tufalabs/duck-harness
- ARC Prize, **GPT-5.6 ARC-AGI results**: https://arcprize.org/results/openai-gpt-5-6
- ARC Prize, **GPT-5.5 / Opus 4.7 replay analysis**: https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis
- PRO-LONG: https://arxiv.org/abs/2607.20064
- Retrodict: https://github.com/ryanbbrown/Retrodict
- Tycho: https://arxiv.org/abs/2607.28287
- Twin: https://arxiv.org/abs/2608.14490
- Executable World Models: https://arxiv.org/abs/2605.05138
- World-model ablations: https://arxiv.org/abs/2607.15439
- Play-adequacy vs prediction accuracy: https://arxiv.org/abs/2607.14169
- Graph exploration: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore
- Independent Duck-harness reproduction/ablations: https://github.com/sonpham-org/arc-3
- Kaggle ARC-AGI-3 discussions and public notebooks: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
