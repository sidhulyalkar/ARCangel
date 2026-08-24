# V008 Visual Belief Search

## Motivation

ARCangel's first real private S115 submission scored **0.10**. That result is treated as evidence that the dominant failure is upstream of action execution: the agent is often not orienting itself to the visual world, controlled entities, causal affordances, and goal evidence well enough before it starts acting.

V008 therefore changes the center of gravity from `current image -> language model -> action` to:

```text
recent visual history
-> persistent object tracks / co-motion
-> causal affordances
-> competing executable visual goal beliefs
-> exact legal counterfactual candidates
-> semantic orientation by Qwen
-> verified action or verified spatial plan
-> re-orient on surprise
```

## Temporal visual packet

Each model call receives one 2x2 image packet containing `t-2`, `t-1`, the current frame, and a diagnostic `t-1 -> current` delta. The live V008 adapter renders this packet at 384x384, while the adapter default remains 256x256 for backward compatibility.

This makes motion, disappearance, appearance, transformation and causal change visible to the VLM without requiring it to reconstruct temporal state from prose.

## Persistent visual entities

`VisualTracker` associates conservative component identities over time. Exact shape/color signatures can translate; mild local transformations can remain linked; ambiguous changes become explicit appearance/disappearance events rather than fabricated object identity.

When multiple differently-colored components move by the same vector under the same action, they are surfaced as a co-moving group. This gives the semantic reasoner evidence for multi-part entities without forcing the geometric auto-planner to trust an unsafe composite actor representation.

## Affordance memory

`AffordanceMemory` learns distributional coarse outcomes for primitive actions and object-grounded ACTION6 targets. Outcomes include `dead`, `change`, `remove`, `spawn`, `global_change`, `game_over`, and `level`.

The memory is intentionally soft. It exposes estimated risk, level progress and information value without permanently suppressing actions that may be context-dependent.

## Visual goal beliefs

`VisualBeliefState` stores falsifiable visual goal hypotheses as a translation-invariant target signature plus an executable generic relation such as `touch`, `inside`, `click`, `remove` or `transform`.

Beliefs carry support, validation and contradiction counts. Spatial predicates can be checked exactly. Level completion can validate a relation that was visually true immediately before the transition.

## Counterfactual decision registry

Every turn, ARCangel enumerates a compact registry of legal choices:

- primitive actions;
- object-grounded ACTION6 click candidates;
- exact spatial plans when V006 causal-control gates are ready.

Each candidate exposes empirical effect support, information value, game-over risk, collision risk, learned movement vector, predicted actor center and progress toward currently grounded visual goals when available.

The numeric score is advisory. Qwen can reject the ranking when the temporal image provides stronger semantic evidence. Candidate IDs keep the model from inventing illegal actions or arbitrary click coordinates.

## Division of labor

Qwen is responsible for semantic orientation: what is controlled, what changed, which objects look important, what visual relation likely represents progress, and what uncertainty blocks a plan.

Code is responsible for exact bookkeeping, legal candidates, object identity, risk statistics, geometry and route calculation. Spatial planning remains an optional grounded tool rather than a mandatory world model.

## Local qualification

The V005/V006/V008 suite passes 40 tests. Additional randomized stress testing passed:

- 1,500 translated-object identity cases;
- 1,000 duplicate-object association cases;
- 1,000 goal-directed counterfactual ranking cases;
- 1,000 ACTION6 candidate legality/uniqueness cases;
- 1,000 arbitrary temporal-packet constructions.

Total randomized checks: **5,500 passed, 0 failed**.

This is structural qualification only. The private leaderboard remains the promotion signal.

## Submission candidate

S160 FINAL B embeds this exact source and uses the validated FINAL-F deployment contract, RTX PRO 6000, the CUTLASS FP8 fallback, official hidden gateway execution, and `/kaggle/working/submission.parquet`.
