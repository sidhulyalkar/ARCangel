# Experiment Plan

## North-star metric

Private Kaggle score. Public-game score is a development diagnostic, not the objective.

## Secondary metrics

1. weighted levels completed;
2. action efficiency conditional on completion;
3. first-level solve rate across games;
4. later-level continuation rate after level 1;
5. no-op rate;
6. game-over/reset rate;
7. inference calls and tokens per scored action;
8. wall-clock seconds per game;
9. public seed variance.

## Public evaluation protocol

- Use all 25 games for descriptive evaluation.
- Maintain leave-family-out slices by action interface (`keyboard`, `click`, mixed) and learned visual/mechanic clusters.
- Run at least 3 deterministic seeds for stochastic/model sampling changes.
- Never tune against individual game IDs.
- Record exact git SHA + config + toolkit version in every receipt.

## Phase A — reproduce the contemporary Kaggle frontier

**A0. Structural floor**: done. 0.1755 local public score at 200 action cap, 2/183 levels.

**A1. Duck replication**: attach the same Qwen 3.6 27B FP8 family and reproduce a ~1%+ Kaggle-class system. Do not add novel mechanisms until runtime/model serving is stable.

Success gate: comparable public/Kaggle order of magnitude to the public Duck notebooks, with no crashes and complete diagnostics.

## Phase B — context/memory

Add queryable programmatic memory:

- transition filters by action/object/effect;
- nearest structural-state retrieval;
- “show me all times ACTION5 changed a non-HUD object” queries;
- level-boundary summaries derived from the raw ledger;
- model-triggered retrieval instead of injecting the entire past.

Ablation: same model, same prompt except memory tool on/off.

## Phase C — perception/state abstraction

- distinguish settled state from animation frames;
- improve HUD/timer masking;
- object containment/adjacency graph;
- cross-frame object identity by shape/color/overlap;
- movement/action semantic induction;
- detect controllable object candidates;
- detect click affordances and dead signatures.

Ablation: same model/memory with raw-only vs structured perception.

## Phase D — active executable world models

Actor emits `delegate_world_model=true` only when planning value is high. Model builder produces a compact transition program over abstract state. Verifier replays all relevant recorded transitions. Planner may use the model only above a calibrated reliability threshold.

Critically, verification failure should not automatically trigger repeated expensive repairs. The actor chooses build / repair / use / bypass.

## Phase E — inference budget router

For 110 games under 9 hours, assign reasoning budget dynamically:

- cheap first probe / interface discovery;
- more tokens after meaningful progress;
- stop spending model calls on hopeless cycles;
- prioritize games with clear progress and unsolved later levels;
- use queued actions for reliable local plans;
- batch model requests across concurrent games if the serving backend supports it.

## Kaggle submission policy

Every submission must have a single primary hypothesis and a pre-written interpretation table:

| Result | Interpretation | Next action |
|---|---|---|
| large gain | mechanism likely transfers | keep, ablate interaction terms |
| small gain | possible benefit/noise | repeat locally; combine only if cheap |
| flat | public gain did not transfer | remove or gate mechanism |
| regression | mechanism consumes actions/runtime or aliases state | inspect telemetry before retuning |

Do not respond to a bad LB score by adding three more heuristics at once.
