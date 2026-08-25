# V009 Perceptual Scientist

S115's 0.10 private score is treated as an upstream orientation failure, not as evidence that ARCangel merely needs more downstream action heuristics.

V009 therefore stabilizes the latent facts a strong human player extracts before planning:

1. **Agency** — which persistent entity is causally controlled, how confidently, and by which learned action vectors.
2. **Object roles** — soft posteriors for controlled entity, goal target, trigger/button, hazard, dynamic entity, or decoration.
3. **Causal interaction graph** — which actions recently moved, transformed, created, or removed which persistent tracks.
4. **Goal uncertainty** — entropy over executable visual goal hypotheses rather than a single prose guess.
5. **Action-model uncertainty** — entropy over empirical effect distributions, with explicit information value and game-over risk.
6. **Topology** — actor-footprint-valid anchors, reachable area, exact articulation points, and action-conditioned topology deltas that reveal gates/doors/region changes.
7. **Symmetry** — horizontal/vertical/rotational/diagonal scores plus localization of the objects that violate the strongest symmetry.
8. **Temporal phase** — repeated-state cycle detection to avoid confusing periodic animation with causal change.
9. **Novelty and irreversibility** — new tracks plus appearance/disappearance/transformation events that deserve extra caution.
10. **Retrodiction** — `retrodict_action(action_id)` exposes exact historical transitions and visual events before a mechanic is trusted.

## Multi-view vision

Every model reasoning call gets two independent views:

- **CURRENT HIGH-RESOLUTION BOARD**, 512×512, preserving spatial detail.
- **TEMPORAL CONTEXT**, 384×384, containing t-2, t-1, current, and a diagnostic delta.

This avoids the V008 compromise where four panels shared a single image's resolution.

## Avoiding over-engineering

V009 deliberately removes the scalar candidate score from the model-facing registry. Exact code still enumerates legal primitive actions, object-grounded ACTION6 clicks, and supported spatial plans, but Qwen sees their evidence rather than a hand-tuned ranking. This follows the evidence from strong public ARC-AGI-3 agents that rich perception/history and selective world modeling matter more than piling on arbiters.

## Decision modes

The perceptual estimator recommends only a *reasoning mode*, never a game-specific action:

- `IDENTIFY_AGENCY`
- `IDENTIFY_GOAL`
- `DISCRIMINATE_DYNAMICS`
- `REASON_ABOUT_PHASE`
- `MODEL_AND_TEST`
- `PLAN_AND_EXECUTE`
- `REPAIR_MODEL`

High orientation entropy forces denser model reasoning. Once the world is sufficiently understood, exact plans can execute cheaply and are revalidated on surprise.

## Local qualification

- focused V009 tests: 20/20
- randomized perceptual scenes: 5,000/5,000
- exact articulation-point comparisons against independent brute force: 1,477/1,477
- full 64×64 open topology stress: pass
- multi-view OpenAI-compatible request construction: pass
- runner multiview telemetry: pass
- frontier shortest-route comparisons against an independent reference: 2,500/2,500

These tests establish structural correctness only. Kaggle RTX Save & Run and the private leaderboard remain the promotion boundary.

## Systematic state-action frontier

A strong fallback should not collapse to repeated local heuristics when semantic orientation is still uncertain. V009 therefore maintains a level-local directed graph of exact visual states and grounded real actions. Each node records which candidate actions were available and which were actually tested. Tested transitions retain destination distributions and terminal risk.

The graph answers two generic questions without encoding any game-specific rule:

1. Which actions at the current state remain genuinely untested?
2. If the current state is exhausted, what is the shortest previously observed, non-terminal, high-confidence route to a state with something still untested?

The VLM receives this frontier evidence but is not forced to follow it. If the model fails to return a usable action, ARCangel uses the lowest-risk local frontier candidate or a known-safe first edge toward a remote frontier. Only then does it fall through to the D110R2 effect posterior. Frontier novelty is never equated with goal progress.
