# V011 — Evidence-Efficient Learning

Status: research branch, intentionally isolated from the currently running S180 candidate.

## Why this branch exists

V010 fixes the bridge between multimodal reasoning and real action selection. The next performance ceiling is likely not another orchestration layer. It is the number of scored actions ARCangel needs to acquire the right abstraction.

ARC-AGI-3 rewards completion and Relative Human Action Efficiency. Internal reasoning, programmatic analysis, and memory operations are effectively cheaper than environment actions. V011 therefore treats every real action as an experiment that should purchase one of three things:

1. progress toward a credible goal;
2. decisive information about a pivotal uncertainty;
3. execution of a sufficiently verified plan.

Everything else is epistemic waste.

## 1. Separate benchmark guarantees from hidden game rules

ARCangel currently behaves more conservatively than the public ARC-AGI-3 interface requires. The standardized interface documents ACTION1–ACTION4 as directionally mapped to up/down/left/right, ACTION6 as a coordinate interaction, and ACTION7 as undo. The effect of those controls remains game-specific, but the directional vocabulary itself is not latent.

V011 should introduce a small immutable `InterfacePrior` layer containing only benchmark-contract facts. These priors may narrow hypotheses, but must never encode public game identities, solutions, target colors, coordinates, or mechanics.

Important distinction:

- Known: ACTION1 has upward directional semantics.
- Unknown: which object responds, whether it moves one cell, rotates, pushes, steers a camera, or causes no visible effect.

This should remove unnecessary first-level action-ID rediscovery while preserving genuine agency inference.

## 2. Preserve all intra-action frames

The ARC API can return multiple consecutive frames for one environment action while animations or internal dynamics advance before the final settled state. The current `frame_grid()` intentionally takes only `frame[-1]`, so intermediate evidence is discarded.

That is unacceptable for a general interactive learner.

Add an immutable per-transition `FrameSequence` containing:

- pre-action settled frame;
- all returned intermediate frames;
- post-action settled frame;
- per-frame pixel diffs;
- component trajectories;
- appearance/disappearance timing;
- transient colors/components;
- estimated motion vectors;
- periodicity/phase evidence;
- whether a change persisted into the settled state.

The semantic agent should normally see a compact animation summary, not every raw frame. It may request the full sequence or a rendered strip when temporal uncertainty is pivotal.

This allows ARCangel to distinguish:

- action-caused persistent movement;
- transient animation;
- moving hazards;
- projectiles;
- temporary highlighting;
- switch feedback;
- object traversal followed by disappearance;
- clocks and autonomous phase changes.

## 3. Goal inference becomes an executable hypothesis problem

Dynamics inference and goal inference remain separate.

V011 should expand the visual goal representation into a generic executable goal DSL. Candidate predicates should be structural rather than game-specific, for example:

- `TOUCH(A, B)`
- `ADJACENT(A, B)`
- `INSIDE(A, region)`
- `ENTER(region)`
- `REMOVE_ALL(class)`
- `ACTIVATE_ALL(class)`
- `COLLECT_ALL(class)`
- `ALIGN(objects, axis)`
- `MATCH(pattern_a, pattern_b)`
- `RESTORE_SYMMETRY(axis)`
- `COUNT(class) == n`
- `TRANSFORM(A, signature)`
- `REACH_TOPOLOGICAL_REGION(region)`
- `SURVIVE(phase >= n)`
- short conjunctions/disjunctions of the above.

Every goal hypothesis stores evidence, counterevidence, support, contradictions, confidence, first/last level, and an executable predicate when grounded.

### Goal retrodiction

Before trusting a goal, test it against history:

- Was the predicate previously true without completion?
- Did completion coincide with the predicate becoming true?
- Does a simpler predicate explain the same evidence?
- Does the predicate transfer across levels after translation/color/shape changes?

Completion is rare and highly informative. A level transition should trigger a dedicated lesson-distillation pass.

## 4. Post-level lesson distillation

Level completion is a causal supervision event. Do not merely increment `levels_completed`.

On every level completion, infer and store a compact `LevelLesson`:

- likely controlled entity / agency rule;
- verified action semantics/effects;
- likely completion predicate;
- final action and final visual relation;
- persistent mechanics worth transferring;
- mechanics that changed from the prior level;
- shortest verified skill/macros discovered;
- remaining uncertainties.

Later levels should receive these lessons as priors, but must retest them when contradicted.

The goal is concept transfer, not path transfer.

## 5. Hypothesis-disagreement experimental design

Generic novelty/frontier exploration is a safety net. It should not be the primary experiment selector when explicit hypotheses exist.

For a set of competing hypotheses H and legal candidate action a, estimate how differently the hypotheses predict the next observation/effect. Prefer cheap, low-risk actions with high expected discrimination.

Conceptually:

`Value(a) = expected_information_gain(a) * pivotality - terminal_risk - irreversible_risk - action_cost`

where pivotality measures how much the unresolved assumption changes the best plan or likely goal.

Do not compute pseudo-precision from invented probabilities. Use calibrated disagreement bins and empirical support when exact probabilities are unavailable.

## 6. Pivotality-aware verification

Not every uncertainty deserves another action.

A hypothesis is pivotal when:

1. the current best plan depends on it;
2. it is weakly supported or recently contradicted;
3. being wrong would waste many future actions, cause GAME_OVER, or invalidate the goal.

Before a long plan, identify the highest-damage uncertain assumption and verify that one first.

This is more useful than optimizing aggregate transition accuracy.

## 7. Durable skills, not memorized paths

Continual-learning agents report large efficiency gains from skill reuse. ARCangel should compile repeated verified behavior into parameterized skills with explicit preconditions and effects.

Examples:

- navigate controlled footprint to relation with target;
- traverse a verified gate after trigger state;
- click each member of a visual equivalence class;
- undo a reversible failed experiment;
- replay a safe deterministic control sequence from a matched abstract state.

A skill must be rejected if its preconditions do not match the current level. Store abstract relations/signatures, not public coordinates.

## 8. Persistent programmatic scratch state

Strong public coding-agent approaches benefit from a persistent REPL or programmatic workspace. ARCangel already exposes a bounded Python analysis sandbox, but it should eventually support safe per-game persistent helper state:

- reusable analysis functions;
- named derived objects;
- compact causal tables;
- executable transition hypotheses;
- verified goal tests;
- pathfinding helpers.

The model should retrieve and execute these helpers rather than repeatedly regenerate equivalent code.

The immutable evidence ledger remains authoritative; scratch state is disposable derived knowledge.

## 9. Adaptive visual bandwidth

Do not always spend the same visual-token budget.

Suggested policy:

- orientation/high visual uncertainty: current high-res + temporal packet + animation summary;
- local ambiguity: focused crop/segmentation requested by the actor;
- known deterministic execution: no model call;
- contradiction: current + relevant prior frame/sequence;
- ACTION6 ambiguity: object-labeled click candidates and targeted crop.

The actor should request extra visual evidence only when it can change the decision.

## 10. ARCangel Arena and metamorphic validation

Public tasks are demonstrations, not a trustworthy training distribution for a hidden-OOD competition.

Build procedural interactive worlds from generic primitives and hold out *compositions*, not only random seeds.

Randomize:

- colors and sprites;
- grid dimensions;
- positions and topology;
- directional control effects;
- interact/click mechanics;
- gates/switches/portals/hazards;
- periodic dynamics;
- hidden objectives;
- distractors;
- rotations/reflections/translations.

Metamorphic tests should assert that solutions survive irrelevant transformations such as color permutation, translation, reflection, harmless distractors, and sprite replacement.

Primary metrics:

- completion;
- scored actions;
- actions to identify agency;
- actions to identify mechanics;
- actions to identify goal;
- number of repeated uninformative probes;
- contradiction recovery cost;
- cross-level skill reuse;
- calibration of action/hypothesis confidence.

## 11. Model/harness bakeoff only after the architecture is stable

Once S180/V011 control is behaviorally qualified, compare model choices under the exact same harness and budget. Do not change prompt, model, and planner simultaneously.

The correct experiment is a controlled model bakeoff, not another compound submission.

## V011 recommended implementation order

P0 — cheap, high-confidence improvements:

1. benchmark-contract `InterfacePrior`;
2. intra-action frame preservation and animation summaries;
3. separate semantic/fallback telemetry retained from V010;
4. level-completion lesson receipt.

P1 — largest intelligence upside:

5. executable goal DSL + goal retrodiction;
6. hypothesis-disagreement probes;
7. pivotality-aware verification;
8. parameterized verified skills.

P2 — infrastructure for systematic progress:

9. ARCangel Arena;
10. persistent programmatic scratch state;
11. adaptive visual bandwidth;
12. controlled model bakeoff.

## Promotion philosophy

Do not promote V011 because unit tests pass. Promote each capability only when it changes behavior on leakage-resistant held-out worlds or yields a clear Kaggle receipt.

A strong ARCangel should eventually follow this loop:

`observe all evidence -> orient -> maintain competing theories -> identify pivotal uncertainty -> acquire the cheapest decisive observation -> infer executable goal -> verify plan-critical assumptions -> compile exact/parameterized skill -> execute cheaply -> detect surprise -> repair -> distill level lesson`

That is the target architecture: a compact adaptive scientist with exact evidence and selective computation, not a stack of increasingly opinionated arbiters.
