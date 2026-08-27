# S190A Kaggle Save & Run checklist

This checklist is the qualification boundary for V011 + Qwen3.8 27B FP8. The private submission slot should be spent only on an exact saved notebook version that produces all required receipts below.

## 1. Notebook settings

- Accelerator: NVIDIA RTX PRO 6000.
- Internet: OFF.
- Competition data attached through the official ARC-AGI-3 competition input.
- Offline vLLM/CUDA wheelhouse: use the same Blackwell-qualified dependency family that passed S170 FINAL D unless a new wheelhouse has itself been qualified.
- Model input: attach the intended Qwen3.8 27B FP8 repack. Do not allow Kaggle input discovery to silently choose a different model.
- Keep the exact notebook source/version immutable after qualification. If any cell, input, package, or model mount changes, treat it as a new candidate.

## 2. Cheap pre-scorecard checks

All failures in this section should happen before the official scorecard is opened.

Expected markers, in order:

1. `ARCANGEL SUBMISSION BUILD: S190A-V011-QWEN38-20260826`
2. `MODEL INPUT PASS: S190A-V011-QWEN38-20260826: <path>`
3. `VLLM SERVER PASS`
4. `FULL INFRASTRUCTURE PREFLIGHT PASS`

The preflight includes a real two-image generation against the local OpenAI-compatible endpoint. If any marker is absent, do not wait for a hidden campaign timeout. Inspect `/kaggle/working/arcangel_vllm.log` and repair the infrastructure first.

## 3. Hidden-campaign liveness

After preflight, the notebook must print an `ARCANGEL HEARTBEAT` approximately once per minute. Each line contains:

- elapsed seconds;
- instantiated games;
- model calls;
- accepted semantic actions;
- emergency fallback actions.

Interpretation:

- No heartbeat after preflight: suspect gateway/runner initialization or a deadlock.
- `model_calls` rises but `semantic_actions` remains near zero: the semantic contract is failing and the agent is paying for inference without gaining control.
- `emergency_fallbacks` rises much faster than `semantic_actions`: do not promote. This recreates the S170 behavioral failure even if the notebook finishes.
- Heartbeats continue but model calls grow nearly one-for-one with environment actions after bootstrap: inference gating is not working as intended.
- Heartbeats stop for several minutes while GPU utilization remains high: inspect vLLM batching/queueing and the server log.

## 4. Runtime ceilings

S190A defaults are intentionally below Kaggle's notebook ceiling:

- shared hidden play budget: 25,200 s (7 h);
- per-game wall-clock budget: 7,800 s (130 min);
- maximum actions per game: 1,000;
- maximum model calls per game: 160;
- maximum Python analyses per game: 24;
- workers: 28.

These are safety ceilings, not goals. Solved games should finish immediately. A candidate that routinely reaches these limits is not healthy even if it technically creates a submission.

## 5. Behavioral promotion gate

Before spending a private slot, inspect the saved receipt at `/kaggle/working/arcangel_s190a_receipt.json` and require:

- `model_failures == 0`;
- `model_parse_contract_errors == 0`;
- `model_parse_successes >= 1`;
- `semantic_actions >= 1`;
- semantic actions clearly exceed emergency fallback ownership on a meaningful diagnostic slice;
- `reflection_updates >= 1` or `goal_proposals >= 1`;
- model calls per action materially below the S170 diagnostic ratio after bootstrap;
- no game burns its full model-call budget without progress or an explicit stall receipt.

The key metric is useful progress per model call, not raw inference volume.

## 6. Submission artifact gate

The saved notebook must finish normally, close the official scorecard/gateway, and leave the competition-generated `submission.parquet` in `/kaggle/working`.

Before clicking Submit:

- confirm the file exists;
- confirm the notebook status is complete, not cancelled or timed out;
- confirm the exact build marker and model identity in output;
- keep the saved notebook version ID with the leaderboard result;
- record elapsed runtime, model family, source commit, and the semantic receipt alongside the public/private score.

## 7. What to do if the current resubmission is already running

Use its output position to diagnose rather than judging by wall time alone:

- Still queued/pending with no notebook output: this is platform/accelerator provisioning, not ARCangel policy execution.
- Running before model-server readiness: dependency/model load or CUDA/vLLM startup is the likely bottleneck.
- Running after model readiness with repeated expensive generations: the old V010 uncapped reasoning loop is the likely bottleneck.
- Running with no new output for a long interval: treat it as a possible deadlock or blocked generation rather than assuming productive search.

If the current notebook is the known S170 FINAL C artifact, do not use it. That artifact had the malformed multimodal CLI argument. FINAL D is the qualified S170 runtime boundary.

## 8. Submission ladder after S190A

Do not spend subsequent slots on tiny threshold changes. Use orthogonal experiments that identify the limiting layer:

1. **S190A:** V011 + Qwen3.8 27B FP8. Primary contender.
2. **S190B:** V011 + Qwen3.6 27B FP8 only if an architecture-control result is worth the slot.
3. **S195:** Minimal Qwen3.8 model-led baseline with only legal-action and safety guards. This tests whether ARCangel still has too much scaffolding.
4. **S200:** V011 plus adaptive global inference allocation, giving more compute to games that demonstrate progress and cutting stagnant loops earlier.
5. **S210:** Verified controller compilation, where the model infers a reliable mechanic/goal and code executes a short plan without additional model calls until contradiction.

Each private submission should answer one scientific question. A slot that changes model, prompt, planner, budgets, and perception simultaneously produces a leaderboard number but weak learning.
