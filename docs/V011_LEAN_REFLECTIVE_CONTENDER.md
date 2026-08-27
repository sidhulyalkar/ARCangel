# V011 Lean Reflective Contender

## Objective

V011 is an orthogonal contender experiment, not an incremental threshold tune. It asks whether ARCangel improves hidden ARC-AGI-3 transfer when the strongest multimodal model owns semantic decisions while code is restricted to exact evidence, memory, legality, verification, and cheap execution.

The design responds to three pieces of evidence:

1. S115's first real private score was 0.10, so downstream planning sophistication did not rescue weak upstream orientation.
2. S170 FINAL D fully qualified the dual-view Qwen3.6 runtime, but its four-action smoke used emergency fallback on 4/4 actions despite three successful multimodal model calls. Semantic control was not demonstrated.
3. Strong public ARC-AGI-3 systems are model-led and comparatively simple. Their reusable lesson is not to copy game-specific tactics, but to avoid forcing the model through a large hand-authored ontology when visual reasoning can decide directly.

V011 therefore keeps V010's semantic-action contract and exact safeguards while shrinking the model-facing control surface.

## Why a Kaggle rerun can take a very long time

ARCangel has two different runtime regimes that can look identical from the Kaggle UI.

### Infrastructure time

Before any hidden game reasoning, the notebook may need to:

- locate and install the offline ARC/vLLM wheels;
- discover and validate the mounted model snapshot;
- initialize CUDA and Blackwell-compatible FP8 kernels;
- start vLLM and load a 27B FP8 checkpoint;
- validate one or two-image generation;
- open the official scorecard/gateway;
- run a public smoke;
- finally enter the hidden campaign.

S170 FINAL C failed before policy execution because `--limit-mm-per-prompt` was passed as malformed JSON. FINAL D fixed this and qualified the full runtime. V011 moves the serialization into the reusable vLLM launcher and retains a server log so this class of error is visible.

### Agent time

The larger risk after startup is an inference amplification loop. V010 allows uncapped model calls. V009's four-action smoke used three 27B multimodal calls for four environment actions. At competition scale, dozens of concurrent games, two images per call, repeated retries, and a 1,200-action safety ceiling can turn uncertainty into thousands of generations.

That behavior is especially dangerous because V009/V010 deliberately reason whenever orientation entropy is high. A difficult game can therefore create the feedback loop:

`uncertain -> call model -> no accepted semantic action -> fallback -> still uncertain -> call model again`

V011 breaks this loop. It reasons densely for the first five level-0 actions, then uses a bounded 2-3 action cadence unless a level boundary, prediction contradiction, or stall justifies immediate reasoning.

## Architectural principle

The semantic actor should answer only the expensive questions code cannot reliably answer:

- What is controlled?
- What visual state appears desirable?
- What did the last action teach us?
- Which mechanic hypothesis best explains the observed transition?
- Is this a progress action, a discriminating experiment, or a verified execution step?

Code should answer exact cheap questions:

- What actions are legal?
- What changed pixel-wise?
- Which objects persist across frames?
- What empirical effects followed each action?
- Is a proposed click in bounds?
- Is a grounded spatial route collision-free?
- Did a high-confidence prediction fail?
- Have we already tested this exact state/action frontier?

This separation is the center of V011.

## Model-facing evidence packet

Each semantic call receives:

1. a 512x512 nearest-neighbor current board;
2. a 384x384 temporal packet containing t-2, t-1, current, and delta;
3. the lossless ASCII grid;
4. a compact scene description;
5. a compact controlled-actor hypothesis, if causal evidence supports one;
6. the last ten real action outcomes;
7. empirical per-action effect posteriors;
8. current executable visual goal hypotheses;
9. typed high-level hypotheses;
10. a compact state-action frontier summary;
11. exact legal candidate records;
12. a bounded persistent reflection.

It intentionally does not receive the long duplicated V009 perceptual instruction block or the entire transition ledger on every turn. Full history remains accessible programmatically when the actor explicitly requests it.

## Persistent reflection

The actor may maintain a compact reflection with four useful fields:

- `goal`: the best current theory of completion;
- `rules`: a bounded set of supported mechanics;
- `avoid`: contradicted or dangerous behavior;
- `next_test`: the highest-value unresolved experiment.

Reflection is not hidden chain-of-thought and is not an unrestricted transcript. It is short structured state intended to prevent repeatedly rediscovering the same lesson. Goal text also feeds ARCangel's persistent goal memory.

## Inference allocation

V011 returns from V010's uncapped inference to explicit safety budgets.

Default contender settings:

- bootstrap semantic reasoning: first 5 level-0 actions;
- later semantic cadence: every 2 actions while actor/goal confidence is weak, otherwise every 3;
- maximum model calls per game: 160;
- maximum Python analyses per game: 24;
- generation cap: 320 tokens;
- 28 continuous workers;
- 1,000 actions/game hard safety ceiling;
- 130-minute game wall-clock ceiling;
- 7-hour shared play ceiling.

These are safety ceilings, not targets. A solved game should finish immediately. The shared ceiling intentionally leaves roughly two hours of Kaggle notebook headroom for model loading, gateway close, parquet creation, and platform variance.

## Model lane

S190 should prefer a Kaggle-legal Qwen3.8 27B FP8 snapshot if the exact attached artifact passes the same offline vLLM/CUDA/multimodal qualification used by S170 FINAL D. Qwen3.6 27B FP8 remains the qualified fallback and the clean architecture-control comparison.

Do not silently change model and policy in the same interpreted experiment. Record both:

- S190A: V011 + Qwen3.8 27B FP8, the contender;
- S190B: V011 + Qwen3.6 27B FP8, architecture-control candidate if a second slot is warranted.

If only one private slot is available, prefer S190A after Save & Run because current public evidence makes the model upgrade a high-value change and the V011 simplification specifically targets ARCangel's known semantic-control failure.

## Promotion gates

### Runtime gate

A candidate is not submit-worthy until it proves:

- exact build marker and source hash;
- intended model discovery;
- CUDA/vLLM startup;
- canonical two-image multimodal argument round-trip;
- successful two-image model generation;
- dynamic public smoke with zero harness errors;
- official gateway close;
- valid `/kaggle/working/submission.parquet`;
- completion safely below the Kaggle runtime ceiling.

### Semantic-control gate

The public diagnostic must prove the model actually controls the agent:

- at least one parsed model response;
- at least one accepted semantic action;
- zero contract-format loops;
- semantic actions exceed emergency fallbacks on a sufficiently long diagnostic slice;
- at least one reflection update or explicit goal proposal;
- zero model transport failures.

S170 FINAL D failed this behavioral gate even though its runtime passed.

### Efficiency gate

Track per game:

- environment actions;
- model calls;
- model calls per action;
- semantic actions;
- emergency fallbacks;
- level completions;
- actions per completed level;
- reflection updates;
- goal proposals;
- queued verified actions;
- prediction mismatches;
- elapsed seconds.

The main runtime efficiency metric is not raw model calls. It is useful semantic progress per model call and completed score per GPU-second.

## Contender decision tree

If S190 improves strongly, promote the lean semantic architecture and next optimize verified multi-action execution after agency/goal grounding.

If S190 behaves well locally but private score remains near S115, simplify further. The likely problem is semantic model capability or observation formatting, not missing heuristic machinery. Benchmark a Duck-style minimal actor under the same model and runtime.

If Qwen3.8 materially beats Qwen3.6 with the same V011 policy, freeze the architecture and focus on model-facing context quality, reflection, and inference allocation rather than adding planners.

If V011 produces many accepted semantic actions but few level completions, prioritize goal acquisition. Expand executable goal representations only from observed failure classes: transformation, count/set, activation, symmetry, sequence, survival, or phase predicates.

If levels complete but action score is weak, shift effort toward controller compilation: infer mechanics early, verify a compact plan, execute without another model call until contradiction.

If runtime approaches the shared ceiling, do not reduce semantic quality uniformly. Allocate inference by expected value: later weighted levels, recent progress, unresolved high-value goals, and games that have demonstrated learnable mechanics should receive more compute than stagnant level-0 loops.

## What V011 deliberately does not add

- no public game IDs or coordinates;
- no source-derived solution rules;
- no assumed color semantics;
- no assumed directional action mapping;
- no large new goal ontology before receipts justify it;
- no global dead-action suppression;
- no mandatory world-model code generation;
- no semantic candidate scalar that overrides the VLM;
- no unlimited model/tool budget.

The wager is deliberately clean: stronger model, leaner semantic interface, persistent compact reflection, exact safeguards, and bounded compute.
