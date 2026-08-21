# Local baseline receipt

Date: 2026-08-20
Toolkit: uploaded `arc-agi==0.9.8`, `arcengine==0.9.3`
Public suite: 25 environments, 183 levels

| Policy | Public score | Levels | Total actions | Purpose |
|---|---:|---:|---:|---|
| seeded random | 0.0000 | 0 / 183 | 2,669 | plumbing/control |
| FRONTIER V001 structural | **0.1755339** | **2 / 183** | **3,036** | deterministic calibration floor |

V001 completed:

- `lf52-*` level 1 in 34 actions versus a 32-action human baseline.
- `lp85-*` level 1 in 5 actions versus a 17-action human baseline. Under toolkit 0.9.8 this level receives the 115-point per-level efficiency cap, while the game itself remains capped by weighted completion.

These numbers are **not a claim of competitiveness**. The structural policy is intentionally generic and model-free. Its role is to verify runner semantics, scoring, perception, action telemetry, and regression behavior before spending Kaggle quota on the model lane.

Raw local scorecards remain development artifacts and are gitignored. The values above are the tracked release receipt.
