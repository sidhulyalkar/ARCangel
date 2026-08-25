# S170 FINAL D Kaggle Save & Run Receipt

Build: `S170-FINAL-20260824-D`

Date: 2026-08-24 (America/Los_Angeles)

Status: **RUNTIME QUALIFIED / BEHAVIORAL PROMOTION WITHHELD**

## What passed

The exact FINAL D Kaggle Save & Run reached every deployment gate that previously failed:

- ARC toolkit bootstrap
- embedded ARCangel V006 spatial preflight
- V009 systematic frontier preflight
- V009 perceptual scientist preflight
- official gateway wrapper
- Qwen3.6 27B FP8 model discovery
- CUDA driver runtime load
- CUDA driver linker self-test
- canonical multimodal argv: `{"image":2,"video":0}`
- vLLM 0.19.0 server startup on RTX PRO 6000 Blackwell
- `CutlassFP8ScaledMMLinearKernel` selected
- `FlashInferFP8ScaledMMLinearKernel` disabled
- model smoke
- full infrastructure preflight
- dynamic public validation game through official scorecard runner
- multi-view model inference
- dummy `submission.parquet`
- Save & Run validation and final submission-file creation

## Validation game receipt

Dynamically selected game: `sk48-d8078629`

Validation limits intentionally allowed only four environment actions.

Observed diagnostics:

```json
{
  "errors": 0,
  "deadline_exhausted_games": 0,
  "model_calls": 3,
  "model_failures": 0,
  "reasoning_cycles": 3,
  "tool_calls": 0,
  "tool_failures": 0,
  "queued_actions_used": 0,
  "fallback_actions": 4,
  "prediction_mismatches": 0,
  "spatial_plans_requested": 0,
  "spatial_plans_compiled": 0,
  "spatial_plan_actions": 0,
  "spatial_plan_mismatches": 0,
  "visual_packet_calls": 3,
  "multiview_calls": 3,
  "frontier_fallback_actions": 4,
  "frontier_known_states": 5,
  "visual_candidate_selections": 4,
  "visual_goal_updates": 0,
  "visual_affordance_observations": 4,
  "visual_expectation_mismatches": 1,
  "mean_final_orientation_entropy": 0.7918,
  "perceptual_modes": {"IDENTIFY_AGENCY": 1},
  "world_model_delegations": 0,
  "goal_hypotheses": 0
}
```

No levels were expected to be solved in this four-action infrastructure smoke; scorecard score was 0.

## Important interpretation

The runtime is now qualified. The **behavioral control path is not yet qualified**.

`visual_candidate_selections == 4` must not be interpreted as four model-selected actions. `PerceptualDecisionPolicy.choose()` routes the systematic frontier fallback through `_candidate_action()`, which increments `visual_candidate_selections`. Because `frontier_fallback_actions == 4`, all four counted selections can be explained by the frontier fallback itself.

Therefore the three successful multimodal model calls may have produced no executable action accepted by the policy. Supporting signals are:

- all four real actions were fallback actions;
- all four were frontier fallback actions;
- zero visual goal updates;
- zero persistent goal hypotheses;
- zero tool calls;
- zero world-model delegations;
- zero spatial-plan requests;
- final orientation entropy remained high at 0.7918.

This is not proof that the model is behaviorally broken: the smoke is only four actions on one public game and begins in `IDENTIFY_AGENCY`. But it is enough to withhold behavioral promotion until we can distinguish:

1. parsed model response contained no action/candidate;
2. model action existed but confidence was below the `0.35` acceptance gate;
3. action/candidate was malformed or illegal;
4. candidate ID was absent/not found after re-enumeration;
5. semantic output was valid but the validation horizon was simply too short.

## Required next diagnostic

Before treating S170 as the scarce private-slot candidate, add non-chain-of-thought structured decision telemetry:

- parse success/failure
- model-proposed primitive action count
- model-proposed candidate ID count
- candidate resolution success/failure
- model confidence
- low-confidence rejection count
- model goal-string presence
- executable visual-goal proposal count
- model action accepted count
- reason semantic action was rejected
- frontier fallback reason

Run a longer Save & Run validation on a small generic public diagnostic slice. The objective is not public score tuning. The gate is that the semantic layer demonstrably produces legal accepted actions and/or explicit goal hypotheses while the fallback remains available rather than owning 100% of control.

## Promotion status

- Deployment: **PASS**
- Multi-view inference: **PASS**
- Frontier fallback: **PASS**
- Semantic model steering: **UNPROVEN**
- Goal acquisition: **UNPROVEN**
- Private submission recommendation: **HOLD until semantic-control telemetry is qualified**
