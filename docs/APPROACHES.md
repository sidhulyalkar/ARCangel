# Public Approach Review — 2026-08-20 snapshot

## Kaggle-constrained approaches

### Random agent

The official sample is a plumbing baseline. Its score is around 0.18 on the current Kaggle code page. It establishes that sparse accidental level completion exists, but carries almost no transferable strategy.

### Stochastic Goose

The Developer Preview winner learned from interaction: a CNN predicted whether actions would change frames, with action/coordinate heads and dynamically reset models. It was excellent on the tiny preview distribution, but the current competition sample is around 0.25. The lesson is not “CNNs fail”; it is that **action-effect prediction without semantic goal induction transfers poorly** when the environment family expands.

### Persistent Memory BFS

A public notebook reports 0.46 with a very short runtime. This is a useful lower-cost anchor: explicit memory and search beat pure randomness. Search remains bottlenecked by state aliasing, action semantics, and unknown goals.

### Reki

Milestone #1 second place used a vision-language policy with structured JSON, legal-action constraints, JSON repair, a 1–4 action queue, reflection memory, and click heuristics. It is an important systems lesson: robust action plumbing and bounded plans matter nearly as much as prompt cleverness.

### Forge

Milestone #1 third place generalized the Reki pattern into a profile-driven framework with candidate generation/arbiter/confidence machinery. The strongest reported run disabled much of the extra machinery, reinforcing the idea that complicated orchestration can consume model capacity without increasing intelligence.

### Tufa Labs — The Duck

Milestone #1 winner. The model is treated as a coding agent with a live Python REPL. It sees image, ASCII, segmentation, structured transition state, and can execute code/actions. Context stays bounded via eviction. Tufa reports that better models and multimodality drove gains while additional handcrafted tools could hurt.

The open Duck write-up reports Qwen 3.6 27B FP8 and a mean 1.6002 ± 0.4475 on repeated runs over the 25 public games. This variance is a warning: a one-shot Kaggle score is noisy, so we need controlled ablations and multiple local seeds.

## Research systems not directly comparable to Kaggle

### PRO-LONG

Core idea: keep a complete structured log and let a coding agent search it programmatically. On the public 25-game research setup, the paper reports +18.0 percentage points over base coding agents, up to 76.1% pass@1, with materially lower token use. This is highly relevant to Kaggle because memory architecture is cheap compared with scaling the base model.

### Executable World Models

A coding agent writes a game-specific simulator, verifies it against interaction, simplifies it, and plans through it. Strong proof that explicit models can help, but follow-up work found the gain depends strongly on the underlying model and orchestration policy.

### Tycho / Active Abstraction

The most important new idea is *not* “always build a world model.” Tycho lets the acting agent decide when to delegate to a model builder. On the public set with frontier models, actor-requested delegation outperformed automatic modeling/repair. That is the design we should port to Kaggle: model-building is an investment with an expected value, not a ritual.

## Synthesis

The strongest Kaggle-ready hypothesis is:

> A Duck-like local coding/vision agent, upgraded with PRO-LONG-style lossless searchable memory and Tycho-style selective world-model delegation, should dominate either hand-built BFS or a pure JSON policy at equal model capability.

The hard engineering problem is inference budgeting. We need to spend 27B-model tokens on uncertainty reduction and planning, while deterministic code handles legality, state bookkeeping, exact diffs, known transitions, and obvious repetitions.
