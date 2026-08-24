# V006 Spatial Intelligence

## Thesis

ARCangel should not ask a language model to repeatedly rediscover exact navigation geometry from pixels. The model should infer **what the world means**; deterministic code should compute **what route is shortest and currently safe** once the relevant semantics have enough causal support.

V006 therefore inserts an exact geometric layer between V005 perception/memory and action execution:

```text
frame
→ components + h2 temporal memory
→ causal controlled-object hypothesis
→ 360° relational geometry + footprint topology
→ learned action displacement vectors
→ model goal/target semantics
→ exact shortest-action spatial plan when sufficiently supported
→ pre-action geometry guard
→ execute one step
→ verify predicted actor anchor
→ continue or wake Qwen on surprise
```

This is intentionally not a hand-authored game solver. No action meaning, player color, goal color, public game ID, coordinate, source rule, or public solution is assumed.

## Exact 360° representation

`arc3lab.perception.spatial` adds:

- eight directional relations: N, NE, E, SE, S, SW, W, NW;
- signed row/column deltas;
- Manhattan, Chebyshev and Euclidean separation;
- bounding-box gaps, touching, row/column overlap;
- eight-direction raycasts from the inferred controlled object;
- exact line-of-sight with obstacle checks;
- actor-relative object summaries;
- full-footprint valid-anchor masks;
- connected free-space regions;
- reachable anchors under learned movements;
- narrow-anchor/topology diagnostics;
- dihedral transforms of learned movement vectors as a future symmetry-transfer hook.

The actor footprint is the complete connected component. A 2×2 or irregular controlled object cannot be routed through a corridor merely because its center cell would fit.

## Causal control inference

`SpatialControlModel` identifies the likely controlled object from observed transitions rather than visual priors.

For simple actions it matches shape/color-identical components across frames and records translation evidence. Planner readiness is conservative:

- actor confidence >= 0.72;
- a unique current actor candidate;
- >=3 actor-motion observations;
- >=2 distinct actions associated with actor motion;
- >=2 learned translational action vectors;
- each used action vector has confidence >=0.80;
- translation purity >=0.66.

### Important failure guards

**Global/camera motion.** If at least three matched objects share a dominant translation and that translation accounts for >=65% of matched objects, the transition is treated as possible camera/world scrolling and is excluded from control learning.

**Duplicate actor-like objects.** If the learned actor signature matches several current components, the best candidate can remain available for descriptive reasoning, but its confidence is penalized and automatic planning is disabled.

**Collateral motion.** A movement vector earns `pure_support` only when the actor is the sole translated object and component count remains stable. This keeps switches, pushes and other interactions from being silently promoted into pure locomotion actions.

## Spatial planner

`arc3lab.planning.spatial.shortest_spatial_plan` performs exact breadth-first search in **action space**, not pixel distance.

The planner receives:

- the current scene;
- inferred actor index;
- model-selected target object;
- learned action displacement vectors;
- requested spatial relation.

Supported relations:

- `touch` (default);
- `adjacent8`;
- `overlap` / `enter`;
- `center` / `center_on`;
- `inside` / `contain`.

The target is treated as passable only for relations that require entry/overlap. Other visible components remain obstacles. The result contains both action IDs and exact expected actor anchors after every step.

## Qwen/code division of labor

`SpatialCodingPolicy` keeps Qwen responsible for semantic uncertainty. The prompt exposes the exact spatial summary and allows Qwen to request:

```json
{
  "spatial_plan": {
    "target_object": 4,
    "relation": "touch",
    "execute": true
  }
}
```

Object indices are explicitly local observation references, never persistent game identities.

The bounded Python sandbox additionally receives:

- `spatial` — exact current spatial summary;
- `spatial_relations(object_index)` — all pairwise relations from an object;
- `spatial_plan(target_index, relation="touch", max_steps=128)` — advisory exact path query.

A model request does **not** guarantee execution. The route is compiled only if the causal-control readiness gate clears and the model confidence remains high enough.

## Safe route execution

Compiled routes are never fire-and-forget.

Each queued route step stores its expected actor anchor. Before every queued action, ARCangel recomputes current geometry and verifies:

1. the controlled object is still identifiable;
2. the action still has a high-confidence, sufficiently pure translation vector;
3. applying that vector reaches the stored expected anchor;
4. the actor's full footprint can occupy that anchor under the current obstacle map.

After the real action, the observed actor anchor is compared with the expected anchor. Any pre-action or post-action mismatch clears the route and wakes the reasoner.

This is the key private-safety boundary: deterministic planning is allowed to save actions only while its assumptions continue to survive falsification.

## Telemetry

Per game and campaign receipts include:

- `spatial_plans_requested`;
- `spatial_plans_compiled`;
- `spatial_plan_actions`;
- `spatial_plan_mismatches`.

These distinguish four important outcomes:

1. Qwen rarely sees navigation opportunities;
2. Qwen requests routes but causal control evidence is insufficient;
3. routes compile and execute safely;
4. routes frequently invalidate, indicating the geometry/action abstraction is too weak for that environment family.

## Local qualification

The S135 FINAL B source was qualified with:

- Python source compilation: PASS;
- canonical V005 + V006 regression suite: **31 passed**;
- randomized shortest-path audit: **1000/1000** agreement with an independent reference search on randomized obstacle fields;
- exact eight-direction relation tests;
- raycast/line-of-sight tests;
- causal actor-learning tests with arbitrary actor color;
- camera/global-motion rejection;
- duplicate-actor ambiguity rejection;
- collateral-motion purity rejection;
- full-footprint collision tests;
- dynamic queued-route invalidation;
- notebook-level synthetic spatial preflight.

Kaggle RTX Save & Run remains the authoritative integration gate for the model server, ARC runtime and official competition gateway.

## Submission candidate

S135 FINAL B:

- build: `S135-FINAL-20260823-B`;
- notebook SHA-256: `b070dc1d412c49b32f8d91c63c606f21370ad6cd0ad93f292446130868870ddf`;
- embedded source SHA-256: `0055eaae67277191ead3499b41b5e00181cb1c60775e8b9ae50cb0fae6c90a9f`.

Primary question:

> Once the controlled object and movement semantics are learned causally, does exact 360° geometry plus guarded shortest-path execution improve private level completion and action efficiency over canonical V005?

Do not merge/promote V006 solely from local tests. First require the exact notebook to pass the RTX Save & Run acceptance markers in `kaggle/SUBMISSIONS.md`; then use its one-per-day private result as the promotion signal.

## Next research if V006 earns promotion

1. **Executable spatial goal predicates:** convert language goals into falsifiable relations such as REACH, TOUCH, CONTAIN, ALIGN and COLLECT.
2. **Symmetry transfer:** use rotations/reflections to carry learned action semantics across transformed later levels.
3. **Pivotal spatial verification:** choose the cheapest real probe for the uncertain assumption with the largest downstream route cost.
4. **Adaptive campaign allocation:** only after real hidden receipts show which games benefit from compiled spatial control.
