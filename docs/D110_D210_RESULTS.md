# D110R2 + D210R2: promoted architecture evidence

These experiments are public-set diagnostics. They are used to select generic mechanisms, not to encode public-game solutions.

## D110R2 — model-free fallback tournament

D110R2 evaluated 72 configurations: six generic fallback policies crossed with four action ceilings and three reset limits. All 72 configurations completed without harness errors.

### Promotion

The strongest operating point was:

```text
effect_posterior | 320 action safety ceiling | 1 reset
score: 0.2021902998
levels completed: 4
games with progress: 4
```

Across the full factorial, `effect_posterior` also had the best policy-family mean score (0.185896), narrowly ahead of `exploit_change` and above the structural baseline (0.179964).

The correct conclusion is **not** that the full Qwen agent should stop after 320 actions or one reset. Those values calibrated only the model-free fallback. The promoted mechanism is the soft causal action posterior beneath the model.

### Cross-level carry result

The deliberate `reprobe_each_level` ablation scored 0.015460 below the carry-preserving baseline in every matched budget/reset cell. Primitive action semantics should therefore be learned early and preserved across later levels until contradicted.

### Negative results worth keeping

- Hard/global anti-dead suppression was materially worse. An action that is dead in one context may be pivotal in another.
- A stronger static ACTION6 click prior was aggregate-identical to baseline. Click affordance should be contextual, not another handcrafted global rarity rule.
- More generic fallback action budget after the response-surface elbow mostly burned actions without new reach.

Machine-readable summary: `artifacts/experiments/d110r2-summary.json`.

## D210R2 — predictive state and planner

D210R2 tested temporal state representations, randomized probe learning curves, exact held-out transition prediction, adaptive probe schedules, executable search, plan verification, and cross-level transfer.

### Temporal-state gate

Mean repeated-key state-action consistency:

| Representation | Consistency |
|---|---:|
| visual | 0.793337 |
| visual + step mod 2 | 0.934718 |
| visual + step mod 4 | 0.960049 |
| h1 | 0.959275 |
| **h2** | **0.991835** |
| h3 | 0.997865 |
| action counts | 0.892982 |
| h1 + counts | 0.997303 |

The predeclared target was 0.99. **h2 is the smallest global representation that clears it**, so V005 uses the current scene plus the two most recent before-state/action pairs as its temporal state key.

### Prediction quality

On held-out transitions:

- coverage: **0.766463**
- exact next-state accuracy conditional on coverage: **0.999319**

This changes the controller design. The challenge is mostly **coverage**, not fidelity once a temporal context is known. V005 therefore uses the table as a verification/cache layer and keeps uncovered contexts model-led.

### Probe scheduling

At six probes, random sampling covered only 0.782 of action channels, while round-robin and uncertainty-first each reached about 0.994. At eight probes both structured strategies reached 1.0 coverage; uncertainty-first mode accuracy was about 0.926 and reached about 0.941 by twelve probes.

Policy implication:

1. cover distinct action channels first;
2. spend additional scored probes only where uncertainty remains;
3. do not assume a fixed six-probe random prefix identifies the interface.

### Executable plan evidence

The planner found five level-1 plans. All five replayed successfully 3/3 times. The median plan used only 18.18% as many actions as the corresponding human baseline.

This proves that verified compilation can deliver enormous action-efficiency gains when a usable local model exists.

However, after replaying those level-1 prefixes, the planner found **0/5 level-2 plans** under the same search setup. Cross-level effect semantics still transferred at 0.90 on average and translation semantics at 0.775.

The correct transfer unit is therefore **mechanics and confidence, not the whole prior-level plan**. Later levels should re-infer their goal and route.

Machine-readable summary: `artifacts/experiments/d210r2-summary.json`.

## V005 architecture promoted from the two experiments

```text
observation
  ↓
h2 predictive state
  ↓
known repeated context? ── yes ──→ verify / cheaply amortize plan
  │
  no
  ↓
Qwen goal + mechanic reasoning / targeted probe
  ↓
actor-gated Python analysis or executable hypothesis when useful
  ↓
real action
  ↓
prediction matches? ── no ──→ clear queue, record contradiction, wake model
  │
  yes
  ↓
continue
```

The model-free floor beneath this stack is D110R2's `EffectPosteriorPolicy`. Goal hypotheses are stored separately from transition hypotheses because later levels frequently preserve mechanics while changing what constitutes progress.

## Submission ladder

- **S115 FINAL** isolates the D110 promotion: V004/Qwen campaign + effect-posterior fallback.
- **S120 FINAL** adds the D210 promotion: h2 predictive-state verification + persistent goal hypotheses.

The next campaign-level experiment after these receipts should be adaptive compute allocation across the 110 hidden games, not another broad public-game mechanics sweep.
