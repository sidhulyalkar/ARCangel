# ARCangel 🧭

**ARCangel is a research and Kaggle submission harness for ARC Prize 2026 — ARC-AGI-3.**

The project is built around one hard constraint:

> **The agent must learn a new interactive world from pixels and consequences, without instructions, memorized game rules, public-game lookup tables, or game-specific code.**

ARC-AGI-3 is not primarily a pathfinding benchmark, an image-classification benchmark, or a language-understanding benchmark. It is an **online abstraction and control problem**. A successful agent must actively discover what matters, infer which parts of the visual scene are persistent and causal, identify what it can control, infer what future state counts as success, and then act efficiently enough that exploration itself does not destroy the score.

The benchmark explicitly targets four capabilities:

1. **Exploration** — obtain missing information through interaction.
2. **Modeling** — turn observations into a predictive representation of the world.
3. **Goal-setting** — infer what desirable or winning state to target without being told.
4. **Planning and execution** — reach that state efficiently and repair the plan when reality disagrees.

ARCangel's objective is to build a single agent that performs all four as parts of one adaptive loop.

---

## Current status — V009 Perceptual Scientist

The first real private ARCangel submission, **S115**, scored **0.10**.

That result is important because it falsified a comforting assumption: having a capable local Qwen model, a long interaction history, causal fallback logic, and a technically robust runtime is **not enough**. The system was still too willing to act before it had correctly oriented itself to the game.

The active V009 architecture is therefore a deliberate pivot away from:

```text
current frame
→ ask model what action to take
→ execute
→ repeat
```

toward:

```text
current high-resolution board
+ recent temporal visual context
+ lossless interaction ledger
        ↓
persistent visual entities
+ causal action effects
+ spatial topology
+ uncertainty estimates
+ competing goal hypotheses
        ↓
what do we know?
what do we not know?
what hypothesis is currently blocking progress?
        ↓
choose a cognitive mode:
  ACT DIRECTLY
  TEST A HYPOTHESIS
  EXPLORE A FRONTIER
  QUERY HISTORY / RETRODICT
  BUILD OR REPAIR A MODEL
  EXECUTE A VERIFIED PLAN
        ↓
real action
        ↓
compare predicted and observed outcome
        ↓
update / repair / replan
```

The current contender artifact is **S170**, built from the V009 branch. S170 FINAL C exposed a deployment-only vLLM argument bug before model startup. S170 FINAL D fixes that runtime issue without changing the embedded V009 policy source.

---

# 1. The problem we are actually trying to solve

A new ARC-AGI-3 task should be treated as an unknown interactive system:

\[
o_t = \text{rendered observation at time }t
\]

\[
a_t = \text{chosen action}
\]

\[
o_{t+1} \sim T(o_t, a_t, z_t)
\]

where \(z_t\) denotes hidden or latent state that may include:

- timers or phases,
- selected modes,
- invisible inventory,
- whether a switch has been activated,
- a multi-step interaction state,
- which object is currently controlled,
- prior actions that alter future dynamics,
- animation state,
- or another task-specific variable.

The agent is not given \(T\), is not given a reward function, and is not given a goal predicate.

It must infer three things jointly:

### A. What exists?

What parts of the visual field correspond to meaningful entities, regions, state variables, or transient animation?

### B. How can the world change?

What does each legal action do, under what context, to which entities, and with what uncertainty?

### C. What future state counts as progress or success?

What visual or latent relation appears to be desirable, and how can we distinguish a true objective from an attractive but irrelevant pattern?

The agent should therefore maintain **beliefs**, not premature facts.

---

# 2. Our central representation: pixels → objects → relations → causes → goals

The most important design decision is how raw pixels become a useful theory of the task.

ARCangel uses several levels of representation simultaneously because no single abstraction is sufficient for every novel game.

## Level 0 — raw pixels

The full 2D grid remains authoritative.

We never want higher-level abstractions to erase information that later becomes important.

The raw grid supports:

- exact color identity,
- exact geometry,
- exact state hashing,
- exact frame differencing,
- exact replay,
- lossless programmatic inspection,
- and recovery when object segmentation is wrong.

A strong agent should be able to abandon an incorrect object theory and return to pixels.

## Level 1 — visual regions and connected components

The scene is segmented into same-color 4-connected components.

For each component we retain features such as:

- color,
- pixel count,
- bounding box,
- centroid / center cell,
- exact cells,
- normalized shape hash,
- edge contact,
- compactness,
- local neighborhood,
- repeated-shape count,
- repeated-color count.

This is deliberately simple. It is a **proposal mechanism**, not a claim that every component is a semantic object.

A multi-color character may contain several components. A single component may contain several conceptual objects. ARCangel must remain able to group or split them later.

## Level 2 — persistent entities

Static segmentation is not enough. Meaning often appears only through time.

ARCangel tracks visual entities across observations using:

- appearance similarity,
- shape similarity,
- relative geometry,
- translational motion,
- persistence,
- transformation events,
- disappearance,
- appearance,
- and co-motion.

A persistent track answers a much more useful question than a fresh component:

> Is this the same thing that moved after my previous action?

This is the first bridge from perception to causality.

## Level 3 — spatial relations

Entities are embedded in an exact 2D relational world.

Generic relations include:

- north / northeast / east / southeast / south / southwest / west / northwest,
- Manhattan distance,
- Chebyshev distance,
- Euclidean distance,
- touching,
- overlap,
- containment,
- adjacency,
- alignment,
- same row / column / diagonal,
- line of sight,
- blockers,
- connected free-space region,
- nearest boundary,
- corridor / room membership,
- and bottleneck relationships.

These relations are preferable to absolute coordinates because they are more likely to survive:

- translation,
- rotation,
- reflection,
- resized layouts,
- color permutation,
- and level-to-level rearrangement.

## Level 4 — causal roles and affordances

An object becomes meaningful when actions systematically affect it.

ARCangel records action-conditioned effects such as:

```text
ACTION2
  moved track 7 by (0,+1)
  support = 4
  collateral motion = low
  game-over observations = 0

ACTION6 on visual signature X
  remove = 3
  change = 1
  dead = 0
  game-over = 0
```

This supports soft role hypotheses:

- **controlled entity**
- **goal-like target**
- **button / trigger**
- **door / gate**
- **hazard**
- **movable object**
- **dynamic entity**
- **projectile**
- **decorative object**
- **status / HUD**
- **unknown**

Roles are hypotheses with evidence, not hard-coded visual labels.

No rule says "green is a goal" or "blue is the player."

## Level 5 — mechanics

Mechanics describe reusable transformations:

```text
ACTION3 translates the controlled entity one cell toward negative row
unless blocked.
```

```text
clicking an isolated object with signature S causes a remote region
to change topology.
```

```text
contact between actor A and object B removes B.
```

```text
the world advances through a 4-state temporal cycle.
```

Mechanics should be:

- context-aware,
- falsifiable,
- supported by observations,
- transferable across levels when still consistent,
- and invalidated when contradicted.

## Level 6 — goal predicates

The hardest abstraction is not always dynamics. It is **what winning means**.

A goal hypothesis should eventually be executable, for example:

```text
TOUCH(actor, target)
```

```text
INSIDE(actor, region)
```

```text
REMOVE_ALL(signature)
```

```text
ACTIVATE_ALL(class)
```

```text
MATCH(pattern_A, pattern_B)
```

```text
RESTORE_SYMMETRY(axis)
```

```text
ALIGN(objects)
```

```text
COUNT(class) == N
```

```text
REACH(exit_region)
```

```text
SURVIVE(phase_count >= N)
```

```text
TRANSFORM(source_signature, target_signature)
```

The representation should be extensible. The model may propose new predicates or compose existing ones.

A goal hypothesis must include:

- predicate,
- arguments,
- supporting evidence,
- counterevidence,
- confidence,
- first proposed step,
- last tested step,
- whether it predicted observed progress,
- whether it predicted a level completion,
- and whether it survived later levels.

---

# 3. A theory of what visual information matters

For a novel game, we want the agent to extract information according to **causal usefulness**, not simply visual salience.

The following categories are especially important.

## Agency

The first question in many environments is:

> **What can I control?**

Agency should be inferred from causal intervention.

If ACTION4 repeatedly moves one persistent entity while everything else stays fixed, that entity becomes a strong controlled-object hypothesis.

If the entire scene translates, that may be camera motion rather than player motion.

If multiple objects respond together, they may be one multipart entity or a coupled system.

Useful agency signals:

- action-conditioned displacement,
- response latency,
- movement consistency,
- uniqueness of the responding entity,
- collateral world motion,
- whether action effects depend on local obstacles,
- and whether the entity's motion predicts future state.

## Salience

Not all unusual pixels matter.

Useful salience should combine:

- uniqueness,
- persistence,
- motion,
- transformation,
- interaction,
- symmetry breaking,
- isolation,
- repeated causal relevance,
- and correlation with progress.

A visually bright but causally inert object should lose priority.

A tiny object that repeatedly opens a region should gain priority.

## Topology

For movement-like worlds, local geometry can hide global structure.

ARCangel extracts:

- reachable anchors for the actor's full footprint,
- connected components of free space,
- rooms,
- corridors,
- enclosed areas,
- narrow passages,
- articulation points,
- chokepoints,
- exits,
- and topology changes caused by actions.

An action that changes no obvious "object" may still be crucial if it connects two previously disconnected regions.

## Symmetry and regularity

Human players often infer intent from structure:

- an almost symmetric board may contain one intentionally misplaced object,
- repeated motifs may imply correspondence,
- paired regions may imply matching,
- a broken pattern may indicate the goal transformation.

ARCangel therefore tracks:

- horizontal symmetry,
- vertical symmetry,
- 180° rotational symmetry,
- diagonal symmetry,
- repeated motifs,
- and localized symmetry violations.

Symmetry is evidence, not a universal objective. The system must test whether making the scene more symmetric predicts progress.

## Temporal structure

A change between frames can mean:

- causal action effect,
- animation,
- autonomous movement,
- timer,
- phase transition,
- stochasticity,
- or hidden-state update.

The agent therefore tracks:

- repeated visual signatures,
- cycle lengths,
- periodic object motion,
- phase-conditioned action effects,
- delayed consequences,
- and differences between settled and intermediate frames.

## Irreversibility

Some actions are far more dangerous than others.

Events such as:

- object disappearance,
- object creation,
- permanent topology change,
- game-over,
- mode switch,
- or irreversible transformation

should increase caution.

The agent should distinguish:

> "I do not know what this does"

from:

> "I do not know what this does and it may permanently destroy the state."

## Progress evidence

Novelty is not progress.

Potential progress signals include:

- level completion,
- new reachable regions,
- removal of obstacles,
- collection-like disappearance,
- increased consistency with a goal hypothesis,
- activation patterns,
- movement toward a plausible target,
- reduction of unresolved goal conditions,
- or visual transitions that repeatedly precede level completion.

Progress is task-relative and should remain probabilistic until validated.

---

# 4. The agent should explicitly represent uncertainty

A major source of poor behavior is acting as if an attractive hypothesis were already known.

ARCangel therefore tries to estimate uncertainty over several domains:

```text
agency uncertainty
goal uncertainty
action-effect uncertainty
object-role uncertainty
temporal-phase uncertainty
topology uncertainty
model-prediction uncertainty
```

These should not all collapse into one scalar internally, but a compact `orientation_entropy` is useful for deciding what kind of reasoning should happen next.

The current V009 reasoning modes are:

```text
IDENTIFY_AGENCY
IDENTIFY_GOAL
DISCRIMINATE_DYNAMICS
REASON_ABOUT_PHASE
MODEL_AND_TEST
PLAN_AND_EXECUTE
REPAIR_MODEL
```

The long-term Actor interface should become even more explicit:

```text
ACT_DIRECTLY
EXPLORE_FRONTIER
TEST_HYPOTHESIS
QUERY_HISTORY
BUILD_OR_REPAIR_MODEL
EXECUTE_VERIFIED_PLAN
```

The core principle is:

> **Do not spend a real action answering a question that is not currently blocking the solution.**

---

# 5. Exploration should be hypothesis-driven

Random exploration wastes scored actions.

Pure novelty exploration can also waste actions because a new state is not necessarily useful.

The ideal exploratory action is one whose possible outcomes best distinguish the hypotheses that currently matter.

Suppose the agent is considering:

```text
H1: touching yellow completes the level
H2: yellow is a switch that opens a door
H3: yellow is irrelevant
```

The best probe is an action for which these hypotheses predict meaningfully different observations.

Conceptually:

\[
a^* =
\arg\max_a
\frac{
    \text{expected reduction in relevant hypothesis uncertainty}
}{
    \text{scored action cost} + \text{risk}
}
\]

A practical approximation can use:

- whether an action is untested in the current state,
- entropy of its observed effect distribution,
- disagreement among candidate models,
- game-over risk,
- irreversibility risk,
- future levels affected by the mechanic,
- and whether the assumption is pivotal to the current plan.

This is **active experimentation**, not generic curiosity.

---

# 6. Systematic frontier exploration is the anti-thrashing backbone

When semantic reasoning is uncertain, ARCangel should still make measurable progress in knowledge.

V009 maintains a level-local state-action graph.

Each node represents an exact settled visual state.

Each grounded action edge records:

- whether it has been tried,
- observed destination states,
- coarse effects,
- terminal outcomes,
- support,
- confidence,
- and uncertainty.

The graph answers:

1. Which actions in the current state have never been tested?
2. Which previously visited state contains unresolved actions?
3. What is the shortest known non-terminal route to such a frontier?
4. Which routes are deterministic enough to replay safely?

The frontier explorer is not the goal system.

Its purpose is to guarantee:

> **If ARCangel does not understand the game, it reduces ignorance systematically instead of repeating itself.**

This idea is supported by the ARC-AGI-3 "Explore It Till You Solve It" work, where a comparatively simple graph-based explorer proved surprisingly competitive.

---

# 7. Retrodiction: a belief must explain the evidence already purchased

Forward prediction is useful, but a hypothesis that only sounds plausible is weak.

Before trusting an important rule, ARCangel should ask whether it explains historical transitions.

For a proposed mechanic \(M\):

```text
retrieve all relevant transitions
→ predict what M says should have happened
→ compare against observation
→ count support and contradictions
```

The bounded Python context exposes tools such as:

```python
retrodict_action(action_id)
```

A future generic interface should support:

```python
retrodict_mechanic(hypothesis)
retrodict_goal(predicate)
retrodict_controller(controller)
```

This creates an important epistemic rule:

> **A world model, mechanic, or goal hypothesis does not earn control merely because it is elegant. It must explain relevant history.**

This principle is consistent with PRO-LONG's emphasis on complete programmatically searchable history, Tycho's model verification, and Twin's requirement that an executable model reproduce previous observed transitions.

---

# 8. World models are optional instruments, not mandatory bureaucracy

One tempting design is to force every game into a complete simulator.

We do not want that.

Some games can be solved directly from visual reasoning and a few causal rules. Building a simulator in those games adds latency and opportunities for error.

Other games have enough hidden state or multi-step dynamics that an executable model becomes extremely valuable.

ARCangel's long-term Actor should choose among:

```text
act directly
analyze history
test one rule
compile a controller
build a world model
repair a world model
bypass the world model
```

A model should earn authority through:

- historical transition fidelity,
- goal consistency,
- coverage of pivotal states,
- and demonstrated planning value.

Transition accuracy alone is not enough. A simulator can reproduce dynamics while still misunderstanding what state should be pursued.

This is one of the central lessons from Tycho and Twin.

---

# 9. Goal acquisition is probably the hardest remaining problem

A visually accurate world model does not guarantee good play.

The agent may understand exactly how the board changes and still not know what constitutes success.

ARCangel therefore treats **mechanic inference and goal inference as separate latent problems**.

We want multiple competing goal hypotheses such as:

```text
reach object X
remove class Y
activate every object matching signature S
restore a repeated pattern
make regions correspond
collect all movable objects
enter an exit-like region
survive until a temporal condition
transform source into target
```

A strong goal system should exploit several kinds of evidence:

## Structural evidence

What configuration appears visually special, incomplete, paired, broken, or target-like?

## Causal evidence

What state changes follow meaningful interactions?

## Progress evidence

What transitions correlate with level completion or other unmistakable progress?

## Cross-level evidence

What goal grammar persists across levels even when layout changes?

## Counterfactual evidence

If a proposed goal were correct, what would we expect to become easier, disappear, activate, align, or complete?

## Negative evidence

What attractive hypotheses have repeatedly failed to produce progress?

The agent should be able to say:

```text
Goal H1:
  predicate: TOUCH(actor, signature S)
  support: 3
  counterevidence: 0
  validated completions: 1
  confidence: 0.87

Goal H2:
  predicate: REMOVE_ALL(signature T)
  support: 1
  counterevidence: 2
  confidence: 0.24
```

rather than keeping only prose such as "probably go to the green object."

---

# 10. Exact planning begins only after abstraction is sufficiently grounded

Once ARCangel knows:

- what is controlled,
- what actions mean,
- what obstacles matter,
- what state is desirable,
- and which important assumptions have been tested,

then language-model reasoning should largely stop paying the action-selection tax.

For deterministic movement:

```text
BFS / A*
```

For uncertain movement:

\[
C(a) =
1 +
\lambda_u (1-P(\text{success}|a))
+
\lambda_d P(\text{game over}|a)
+
\lambda_i P(\text{irreversible error}|a)
\]

For temporal worlds:

```text
state = spatial_state × phase
```

For symbolic interaction worlds:

```text
state =
  controlled-object positions
  movable-object positions
  switches
  doors
  inventory-like variables
  temporal phase
```

The guiding principle is:

> **The model discovers the abstraction; exact code exploits the abstraction.**

Every compiled plan remains conditional. If reality contradicts an expected intermediate state, the queue is invalidated immediately.

---

# 11. Multi-level transfer: transfer mechanics, not coordinates or whole solutions

Later levels are one of the most valuable sources of evidence.

A later level may preserve:

- action semantics,
- object roles,
- interaction rules,
- control mappings,
- goal grammar,
- or temporal dynamics

while changing:

- geometry,
- location,
- color,
- target identity,
- number of objects,
- or exact solution.

ARCangel therefore prefers transferable hypotheses such as:

```text
ACTION2 = translate controlled entity east
```

over:

```text
ACTION2 moves pixel (7,11) to (7,12)
```

and:

```text
touch the unique target-like region
```

over:

```text
go to coordinate (13,4)
```

Cross-level memory should carry:

```text
hypothesis
support
confidence
source levels
contradictions
last validated
```

and be invalidated aggressively when the new level disagrees.

---

# 12. Invariances we want the learned theory to respect

Because evaluation tasks are novel, our abstractions should survive transformations that preserve underlying structure.

A generic mechanism should ideally remain useful under:

- **translation** — all objects moved elsewhere;
- **color permutation** — every color remapped;
- **rotation** — board rotated 90/180/270 degrees;
- **reflection** — horizontal, vertical, diagonal;
- **action permutation** — action IDs relabeled;
- **distractor injection** — inert objects added;
- **layout resizing** — same relational task with different spacing;
- **sprite decomposition** — one semantic entity rendered using multiple components;
- **sprite recomposition** — several components merged visually;
- **animation timing changes** — same mechanic with different intermediate frames;
- **level scaling** — more instances of the same mechanic.

These become **metamorphic tests** for our architecture.

If a policy collapses because "blue became orange", it has learned the wrong abstraction.

---

# 13. How we should "train" the model

ARCangel currently relies primarily on **test-time learning**, not gradient updates on public games.

This distinction is important.

We want three different learning layers.

## Layer A — architectural inductive priors

These are generic assumptions built into the harness:

- pixels form spatial fields;
- persistent entities may exist;
- actions may have causal effects;
- geometry matters;
- time matters;
- goals are latent;
- hypotheses should be falsifiable;
- history should remain queryable;
- repeated evidence should increase confidence;
- contradictions should reduce confidence;
- exact planning is useful after sufficient grounding.

These priors are deliberately broad and should apply across unseen tasks.

## Layer B — within-game test-time learning

This is the main ARCangel learning process.

During a hidden game, the agent learns:

- object identity,
- actor identity,
- action semantics,
- affordances,
- hazards,
- temporal structure,
- topology,
- mechanics,
- goal hypotheses,
- and safe controllers.

This knowledge is built from the current game's own interaction history.

## Layer C — offline procedural curriculum

If we fine-tune or train auxiliary components, the safest direction is **procedurally generated interactive worlds**, not memorization of the public 25.

We want synthetic tasks randomized across independent factors:

| Dimension | Examples |
|---|---|
| control | move, click, select, push, toggle |
| action mapping | random permutation of action IDs |
| actor | single cell, multi-cell, multipart |
| geometry | rooms, corridors, islands, chokepoints |
| interaction | keys, switches, doors, portals |
| goals | reach, collect, remove, align, match, transform |
| temporal behavior | static, moving, periodic, delayed |
| hazards | collision, irreversible deletion, traps |
| visuals | random colors, shapes, distractors |
| symmetries | rotations, reflections, repeated motifs |
| levels | mechanic preservation + goal/layout mutation |

The most important split is not random seed.

It is **held-out composition**.

For example:

```text
development:
  movement + key
  movement + portal
  click + transform

held-out:
  movement + key + portal + transform
```

A model that succeeds there is learning composition, not memorizing surface templates.

---

# 14. What auxiliary models may eventually be worth training

The frozen foundation model should remain the semantic generalist.

Smaller learned modules could specialize in generic perception/control tasks if procedural training demonstrates real held-out transfer.

Potential auxiliary learners:

### Entity association model

Predict whether two components across frames belong to the same persistent entity.

### Agency estimator

Predict which entity or entity-group is causally controlled from a short transition history.

### Interaction classifier

Predict a distribution over coarse effects such as move, remove, spawn, toggle, topology change, terminal.

### Goal proposal model

Generate candidate goal predicates from visual structure and transition history.

### Hypothesis value model

Estimate which experiment would most reduce uncertainty relevant to successful completion.

### Plan-risk model

Predict probability that a compiled action sequence will violate its assumed model.

But these modules should only be promoted if they beat transparent algorithmic baselines on **held-out procedural compositions and metamorphic tests**.

A learned black box is not automatically an improvement.

---

# 15. Memory architecture: preserve facts, compress views

Long-horizon interaction creates a tension:

- keeping everything preserves evidence;
- putting everything in the model prompt becomes intractable.

ARCangel follows a PRO-LONG-compatible principle:

> **Keep the complete structured history, then retrieve the relevant evidence programmatically.**

The lossless ledger stores every real interaction:

```text
before state
action
ACTION6 coordinates / target
after state
changed cells
meaningful changed cells
object events
level completion
game over
win
derived causal observations
```

The model receives compact summaries, but can query exact history through bounded tools.

Important future queries include:

```python
recent(n)
by_action(action_id)
frame(i)
diff_frames(i, j)
action_stats()
level_wins()
visual_track(object_index)
retrodict_action(action_id)
retrodict_goal(goal_id)
frontier_states()
known_route(state_id)
```

The summary is disposable.

The evidence is not.

---

# 16. Visual prompting strategy

V009 uses two complementary views for semantic reasoning.

## Current high-resolution board

A 512×512 nearest-neighbor render preserves exact geometry and small visual elements.

Purpose:

- precise spatial interpretation,
- object identification,
- alignments,
- containment,
- paths,
- and current configuration.

## Temporal context

A separate 384×384 packet contains:

```text
t-2 | t-1
-----------
now | delta
```

Purpose:

- agency,
- motion,
- transformation,
- appearance/disappearance,
- delayed effects,
- and causal interpretation.

The model also receives exact symbolic context:

- lossless ASCII grid,
- object table,
- spatial relations,
- persistent tracks,
- affordance posteriors,
- goal beliefs,
- predictive memory,
- frontier information,
- and complete history access through Python.

The VLM should not have to choose between visual gestalt and exactness. It gets both.

---

# 17. Why we do not want too much handcrafted arbitration

A recurring warning from ARC-AGI-3 systems is that more orchestration is not automatically better.

Milestone #1 systems such as Duck achieved strong results with comparatively lightweight model-centered harnesses, and Forge reported that disabling some extra generator/arbiter machinery improved its best configuration.

ARCangel therefore uses exact algorithms to:

- preserve evidence,
- expose geometry,
- enforce legality,
- verify predictions,
- search history,
- enumerate grounded options,
- and execute trusted plans.

But generic heuristics should not silently become the real policy.

The foundation model should retain semantic authority when the abstraction is uncertain.

The rule is:

> **Code should make the model better informed, not merely more obedient to our guesses.**

---

# 18. Scoring economics: every real action must buy something

ARC-AGI-3 rewards both completion and action efficiency.

That means an action should ideally provide one of three returns:

1. **direct progress toward a sufficiently supported goal;**
2. **information that resolves an uncertainty blocking the solution;**
3. **safe traversal along an already verified plan.**

An action that provides none of these is probably waste.

This motivates the approximate decision objective:

\[
U(a) =
P(\text{completion progress}|a)
+
\lambda_I I(a;\mathcal{H})
-
\lambda_D P(\text{game over}|a)
-
\lambda_R P(\text{irreversible mistake}|a)
-
\lambda_C \text{action cost}
\]

where \(I(a;\mathcal{H})\) is information gain over the currently relevant hypothesis set.

We do **not** want this equation to become a brittle handcrafted scalar policy. It is a conceptual accounting system for deciding what evidence the Actor needs.

---

# 19. Current architecture stack

## Perception

```text
raw grid
→ HUD / volatile-region handling
→ connected components
→ persistent visual tracks
→ co-motion groups
→ spatial relations
→ free-space topology
→ symmetry / anomaly features
→ temporal phase features
```

## Memory

```text
lossless episode ledger
+ persistent visual entity history
+ causal affordance memory
+ h2 temporal predictive memory
+ mechanic beliefs
+ goal beliefs
+ contradiction history
+ frontier graph
```

## Reasoning

```text
dual-view Qwen3.6 27B FP8
+ exact symbolic context
+ bounded Python analysis
+ programmatic history retrieval
+ retrodiction
+ optional world-model construction
```

## Planning

```text
systematic state-action frontier
+ exact spatial shortest path when grounded
+ temporal-state search when needed
+ optional executable controller / simulator
```

## Execution

```text
legality checks
+ expected-outcome verification
+ route pre-action checks
+ contradiction-triggered queue invalidation
+ fail-soft fallback
```

---

# 20. Research chronology

## V003R — AngelCode

Introduced a bounded Python workbench, lossless history, persistent beliefs, and model-generated analysis requests.

## V004C — Campaign Baseline

Aligned the runtime with Kaggle scoring and compute economics.

## D110R2 — Causal Fallback

A 72-configuration tournament promoted soft effect-posterior fallback and rejected hard anti-dead suppression and repeated per-level primitive probing.

## D210R2 — Temporal Predictive State

Promoted h2 temporal context after it cleared the predeclared 0.99 repeated-key consistency gate.

Covered held-out transitions were almost perfectly predictable, indicating that **coverage and goal inference**, rather than raw transition fidelity, were becoming the larger bottlenecks.

## V005 — Predictive Coding Agent

Integrated temporal prediction, persistent goals, prediction-error reasoning triggers, and cross-level mechanic transfer.

## S115 private result — 0.10

This was the critical negative result.

It showed that technical robustness and predictive memory did not solve **orientation**.

## V006 — Spatial Intelligence

Added:

- 360° geometric relationships,
- causal controlled-object identification,
- learned displacement semantics,
- actor-footprint topology,
- exact shortest-path planning,
- camera-motion rejection,
- and guarded route execution.

## V008 — Visual Belief Search

Added:

- temporal visual packets,
- persistent object tracks,
- visual goal beliefs,
- causal click affordances,
- and grounded counterfactual candidate enumeration.

## V009 — Perceptual Scientist + Systematic Frontier

The active architecture.

Adds:

- dual-view visual prompting,
- perceptual state estimation,
- object-role hypotheses,
- goal/action uncertainty,
- topology and articulation analysis,
- symmetry and anomaly evidence,
- temporal-cycle evidence,
- causal interaction graphs,
- retrodiction,
- and systematic state-action frontier exploration.

---

# 21. Current submission line

| Candidate | Purpose |
|---|---|
| S115 | private causal-fallback anchor; scored 0.10 |
| S120 | canonical h2 predictive-state control |
| S135 | spatial-intelligence research candidate |
| S160 | temporal visual-belief candidate |
| **S170** | **V009 perceptual scientist + systematic frontier contender** |

The private leaderboard is treated as a **scarce experimental channel**, not a hyperparameter tuner.

Every submission should answer a meaningful architectural question.

---

# 22. What happens if S170 is weak?

A weak S170 score should not trigger another layer of heuristics.

We use the telemetry and score to choose among architectural regimes.

## Case A — S170 remains near the S115 floor

Interpretation:

> Our orchestration may be interfering with the base model, or our abstractions are failing to transfer.

Response:

Build a **minimal scientist control**:

```text
Qwen/VLM
+ current frame
+ recent frames
+ complete programmatic ledger
+ bounded Python
+ simple frontier fallback
```

Remove most hand-designed intermediate arbitration.

This tests whether simplicity + strong multimodal reasoning beats our structured stack.

## Case B — completion improves but score remains low

Interpretation:

> The agent can understand some tasks but spends too many actions discovering or executing.

Response:

Focus on:

- hypothesis-discriminating probes,
- faster mechanic transfer,
- verified plan compilation,
- safe controller reuse,
- and cross-level amortization.

## Case C — dynamics look good but objectives are wrong

Interpretation:

> Goal acquisition is the bottleneck.

Response:

Build **V010 executable goal inference**:

- richer goal DSL,
- multiple explicit competing goal predicates,
- retrodictive goal evaluation,
- goal-search over learned world models,
- and goal-specific information-gain probes.

## Case D — local/public results are strong but private remains poor

Interpretation:

> Our development distribution is misleading.

Response:

Stop optimizing against public games and prioritize:

- procedural ARCangel Arena,
- held-out mechanic compositions,
- metamorphic transformations,
- and architecture selection based on those synthetic hidden sets.

## Case E — S170 jumps materially

Interpretation:

> Perceptual orientation + systematic exploration is a productive abstraction.

Response:

Preserve the architecture and optimize:

- goal acquisition,
- compute allocation,
- model selection,
- context efficiency,
- planning efficiency,
- and learned auxiliary modules.

---

# 23. ARCangel Arena — the development environment we need

The public 25 should never become our effective training set.

We therefore want a procedural interactive benchmark that generates unknown worlds from generic primitives.

Example grammar:

```text
World
  Geometry
    rooms
    corridors
    islands
    barriers
    portals

  Entities
    actor
    targets
    movable objects
    switches
    hazards
    distractors

  Dynamics
    movement
    pushing
    toggling
    spawning
    deletion
    teleportation
    periodic motion
    delayed effects

  Goals
    reach
    collect
    remove
    align
    match
    activate
    transform
    survive

  Rendering
    colors
    shapes
    rotations
    reflections
    multipart sprites
    distractors
```

The benchmark should support train/dev/test splits by **mechanic composition**, not game seed.

Every architecture proposal should be evaluated on:

- completion rate,
- action efficiency,
- time to identify agency,
- time to identify mechanics,
- time to identify goal,
- uncertainty calibration,
- contradiction recovery,
- plan verification,
- cross-level transfer,
- invariance to metamorphic transforms,
- and compute cost.

This is the closest we can get to a genuine private proxy without contaminating the competition.

---

# 24. Promotion criteria for any new mechanism

A feature should not be promoted because it sounds intelligent.

It should have:

### Hypothesis

What failure mode should this mechanism fix?

### Generic definition

Can the mechanism be stated without mentioning a public game?

### Unit evidence

Does the implementation work on synthetic controlled cases?

### Stress evidence

Does it survive randomized cases?

### Metamorphic evidence

Does it survive irrelevant visual/action transformations?

### Held-out composition evidence

Does it improve procedural tasks with unseen combinations?

### Public ablation

Does it improve or at least preserve generic performance without game-specific tuning?

### Private justification

Is the expected information from a scarce private submission worth the slot?

### Telemetry

If it succeeds or fails, will we know why?

If we cannot answer those questions, the feature is not ready.

---

# 25. Repository layout

```text
src/arc3lab/
  perception/
    scene.py
    visual.py
    spatial.py
    state_estimator.py
    diffs.py

  memory/
    episode.py
    predictive.py
    affordance.py
    visual_belief.py

  planning/
    graph.py
    spatial.py
    counterfactual.py
    frontier.py

  policy/
    structural.py
    effect_posterior.py
    hybrid.py
    coding.py
    spatial_coding.py
    visual_coding.py

  model/
    adapter.py

  evaluation/
    runner.py

configs/
  versioned experiment and submission configs

artifacts/
  experiment receipts
  exact manifests
  validation outputs

docs/
  architecture notes
  experiment plans
  submission runbook
  promotion decisions

kaggle/
  canonical submission metadata / notebooks
```

---

# 26. Runtime principles

The competition runtime must be as disciplined as the research architecture.

Current constraints include:

- NVIDIA RTX PRO 6000,
- Internet OFF,
- local Qwen3.6 27B FP8,
- offline vLLM,
- official ARC-AGI-3 gateway rerun,
- one official scorecard,
- one environment construction per hidden game,
- required `submission.parquet`,
- fail-soft isolation,
- exact build manifests,
- and reproducible runtime preflights.

Infrastructure failures must never be mistaken for policy failures.

Every saved Kaggle version should therefore validate:

```text
exact build ID
model discovery
GPU identity
CUDA runtime
CUDA linker
vLLM startup
known-bad kernel disable
model smoke
actual policy model call
submission.parquet contract
```

---

# 27. Telemetry required from every serious run

A score without telemetry is almost useless.

Per environment we want:

```text
levels completed
actions
resets
actions per completed level

model calls
model failures
tool calls
tool failures

frontier actions
frontier revisits
known-route replays

goal hypotheses
goal entropy
goal validations
goal contradictions

agency confidence
action-effect entropy
learned action semantics

prediction coverage
prediction mismatches
retrodiction failures

spatial plans
plan actions
plan invalidations

visual track counts
appearance/disappearance events
topology changes
temporal cycles

wall time
deadline exhaustion
exact source/build manifest
```

Over time this should let us diagnose **which cognitive stage failed**, not just which game failed.

---

# 28. Guardrails against public-set overfitting

- No public game IDs in policy code.
- No memorized public coordinates.
- No public solution sequences.
- No source-derived mechanics in the policy.
- No rule tied to a named public environment.
- No hard-coded semantic color roles.
- No assumed action meanings.
- Every heuristic must be stated generically before per-game inspection.
- Prefer relative relations over absolute positions.
- Prefer causal evidence over visual stereotypes.
- Prefer hypotheses with contradiction handling over irreversible rules.
- Treat public games as diagnostics, not curriculum targets.
- Prefer procedural held-out composition tests for architecture selection.

---

# 29. Research principles

## Evidence before confidence

A hypothesis is not knowledge because the model stated it fluently.

## Pixels remain recoverable

Higher abstractions must never destroy the ability to return to exact observations.

## Dynamics and goals are separate

Knowing how a world changes is not the same as knowing what state is desirable.

## Novelty is not progress

Exploration should reduce relevant uncertainty.

## Mechanics transfer more safely than solutions

Carry causal rules and confidence, not whole trajectories.

## Models must earn authority

Use executable models when they reproduce relevant history and improve planning.

## Exact code should exploit known structure

Do not ask a language model to repeatedly solve a shortest-path problem once movement and goal geometry are known.

## Contradictions are valuable

A failed prediction is evidence that should wake the agent and repair its abstraction.

## Simplicity is a real competitor

If added machinery does not improve held-out learning efficiency, remove it.

## The private leaderboard is not a debugger

A scarce submission should test an architectural hypothesis.

---

# 30. Working thesis

ARCangel is trying to solve ARC-AGI-3 with the following general strategy:

> **Observe without prematurely interpreting. Track what persists. Intervene to discover causality. Represent the world in objects, relations, and latent hypotheses while preserving raw pixels. Infer what is controllable. Infer what future visual relation constitutes progress. Spend real actions only on progress, pivotal information, or verified execution. Build an executable model only when uncertainty justifies it. Retrodict important hypotheses against history. Once mechanics and goals are grounded, compile the cheapest safe plan. Wake the scientist again whenever reality disagrees.**

Or, more compactly:

```text
SEE
→ TRACK
→ INTERVENE
→ EXPLAIN
→ INFER GOAL
→ TEST PIVOTAL UNCERTAINTY
→ PLAN
→ EXECUTE
→ VERIFY
→ REPAIR
```

That is the behavior we ultimately want to generalize to environments ARCangel has never seen before.

---

# References

### Official ARC-AGI-3

- ARC-AGI-3 Competition: https://arcprize.org/competitions/2026/arc-agi-3
- ARC-AGI-3 launch / benchmark motivation: https://arcprize.org/blog/arc-agi-3-launch
- ARC-AGI-3 Technical Report: https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
- Milestone #1 results: https://arcprize.org/blog/arc-prize-2026-milestone-1

### Relevant agent research

- PRO-LONG — Programmatic Memory Enables Long-Horizon Reasoning: https://arxiv.org/abs/2607.20064
- Tycho — Active Abstraction with Programmatic World Models: https://arxiv.org/abs/2607.28287
- Twin — Playing an Unknown Game with a Test-Time Digital Twin: https://arxiv.org/abs/2608.14490
- Explore It Till You Solve It / graph exploration: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore

### ARCangel documents

- Submission runbook: [`docs/SUBMISSION_RUNBOOK.md`](docs/SUBMISSION_RUNBOOK.md)
- V009 design / perceptual scientist notes: [`V009_PERCEPTUAL_SCIENTIST.md`](V009_PERCEPTUAL_SCIENTIST.md)
- Active V009 PR: https://github.com/sidhulyalkar/ARCangel/pull/13

---

ARCangel is an experiment in **learning the right abstraction quickly enough that optimal behavior becomes cheap**.
