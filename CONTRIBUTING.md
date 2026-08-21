# Contributing experiments

Treat leaderboard submissions as controlled experiments.

1. Create one config/version for one primary hypothesis.
2. Run `make check` before publishing or submitting.
3. Do not add public game IDs, source-derived mechanics, human baseline metadata, or per-game lookup tables to policy code.
4. Record toolkit/model/config/git SHA and keep the Kaggle output receipt.
5. Prefer generic mechanisms that can be stated before inspecting game-specific wins and failures.
6. Compare against the nearest parent version, not against random.
7. If a change improves public games but hurts Kaggle, gate or remove it rather than stacking another heuristic on top.

Version naming uses `vNNN-short-hypothesis`. A leaderboard score without telemetry is not considered an experiment result.
