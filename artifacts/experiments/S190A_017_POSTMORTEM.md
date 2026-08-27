# S190A / V011 leaderboard postmortem

Status: **BEHAVIORAL FAILURE / ARCHITECTURE NOT PROMOTED**

Observed Kaggle public leaderboard score: **0.17**.

## Interpretation

S190A successfully moved ARCangel onto the intended Qwen3.8 27B FP8 runtime lane, but the resulting score is near baseline. This is treated as an architectural falsification result rather than a request for another threshold/cadence tune.

The V009 → V010 → V011 controller family should therefore remain available for historical comparison but should not receive another scarce leaderboard slot without a fundamentally different experiment.

## What the result falsifies

The result argues against the combined hypothesis that competitive unseen-game performance would emerge from:

- deterministic component/object preprocessing;
- actor/goal/spatial candidate enumeration;
- fixed candidate utility priors;
- a compact model-facing semantic packet;
- a large structured hypothesis/action JSON contract;
- periodic model reasoning separated by heuristic actions;
- emergency/frontier/effect-posterior fallback as a normal control path;
- short reflection memory as the primary persistent theory.

Any one of these ingredients may still be useful as an optional tool. Their combination as the **authority hierarchy** did not work.

## Pre-existing warning that we underweighted

S170 FINAL D had already demonstrated a structural warning: three successful multimodal model calls occurred during the four-action validation, yet all four real actions were owned by fallback/frontier behavior and no executable goal hypothesis was produced.

V010/V011 improved observability and model semantic ownership, but the deeper architecture still positioned hand-authored abstractions between raw evidence and action. S190A shows that making this hierarchy cleaner did not make it competitive.

## Process error

ARCangel's internal qualification increasingly answered software questions:

- Is the JSON contract valid?
- Is the candidate legal?
- Is fallback deterministic?
- Does the registry preserve hypotheses?
- Does the notebook launch?

Those are necessary engineering checks, but they were allowed to masquerade as evidence of task competence.

Going forward, promotion must distinguish:

1. **software qualification** — implementation behaves as specified;
2. **scientific qualification** — hypotheses are actually constrained by evidence;
3. **behavioral qualification** — levels are solved efficiently on held-out/public diagnostic games;
4. **leaderboard qualification** — the exact Kaggle artifact earns a competitive score.

Passing category 1 must never imply categories 2–4.

## Architectural response

V012 is a clean policy reset around a different invariant:

> The complete interaction ledger is ground truth. Harness abstractions are optional queries. Persistent beliefs remain provisional until historical evidence and real progress support them.

V012 therefore introduces:

- complete immutable evidence access;
- free historical Python interrogation before live probing;
- explicit hypothesis falsification;
- persistent evidence/theory separation;
- model-authored actions without semantic candidate arbitration;
- expectation-checked action queues;
- explicit support IDs tying long plans to tested hypotheses;
- optional executable world-model validation;
- no normal heuristic fallback authority.

## Next experimental requirement

Before interpreting another custom ARCangel score, establish a public control lane using the current Duck/Qwen3.8 notebook family. The control exists to prove that the same Kaggle hardware/model serving environment can reach the current multi-point public regime when ARCangel cognition is absent.

V012 only earns promotion if it demonstrates task-level value beyond the 0.17 failure and is evaluated relative to that control rather than relative to internal parser/telemetry metrics.
