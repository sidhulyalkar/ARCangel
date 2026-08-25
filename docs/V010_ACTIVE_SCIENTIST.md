# V010 Active Scientist

V010 exists because S170 FINAL D qualified the complete V009 runtime but exposed a semantic-control failure mode: three successful multimodal reasoning calls produced no provably model-directed action, while all four real validation actions came from the systematic frontier fallback.

## Objective

Make the foundation model the semantic scientist and deterministic code the evidence/computation substrate.

The policy distinguishes confidence that a theory is true from confidence that a proposed experiment is the best next move. It also distinguishes deliberate model-directed exploration from emergency fallback.

Target loop:

`observe -> hypothesize -> choose pivotal experiment -> update causal evidence -> ground goal -> compile verified plan -> execute cheaply -> repair on contradiction`

## Canonical decision contract

V010 replaces the inherited legacy/extension schema split with one system-level JSON contract. It includes orientation, typed agency/mechanics/goal/abstraction hypotheses, executable visual goals, one cognitive mode, grounded candidate/direct action, separate hypothesis/action confidence, an explicit experiment, optional Python analysis, and plan metadata.

Low theory confidence must not suppress a high-confidence information-gathering action.

## Typed hypotheses

`HypothesisRegistry` stores agency, mechanics, goals and abstraction theories. Repeated evidence raises calibrated support softly. Prediction/counterfactual contradictions penalize recent agency/mechanics theories. The registry preserves evidence continuity and does not choose actions.

## Mode-conditioned experimentation

Emergency fallback is no longer generic frontier everywhere:

- `IDENTIFY_AGENCY`: replicate candidate motion when possible, otherwise a low-risk primitive probe.
- `IDENTIFY_GOAL`: low-risk, high-information grounded probe.
- `DISCRIMINATE_DYNAMICS`: information-rich probe with low terminal risk.
- `REASON_ABOUT_PHASE`: known low-risk quasi-no-op when possible.
- `MODEL_AND_TEST`: grounded information probe.
- `REPAIR_MODEL`: selectively retest the contradicted grounded action when safe.
- otherwise: systematic state-action frontier.

## Semantic telemetry

V010 records parse successes, contract errors, semantic actions, semantic candidate/direct actions, model-directed frontier actions, emergency fallback actions, low-action-confidence rejections, goal proposals, typed-hypothesis updates/contradictions, decision/fallback modes and one structured decision trace. It does not record private chain-of-thought.

## Qualification

Focused V010 suite: **23/23 PASS**.

Randomized audits:

- decision-contract normalization: 20,000 / 20,000;
- mode-fallback legality: 10,000 / 10,000;
- typed-hypothesis calibration/pruning: 5,000 / 5,000.

Total randomized cases: **35,000**, zero failures.

An end-to-end synthetic control audit also passes: arbitrary movement semantics are learned through four semantic probes, an executable touch goal is proposed, the exact spatial planner compiles a route, and later route actions execute from the verified queue without additional model calls.

These tests establish control-path correctness and generic invariants, not hidden leaderboard performance.
