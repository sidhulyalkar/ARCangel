# Kaggle Landscape Snapshot — 2026-08-20

This file records only externally observable public evidence. Kaggle's live leaderboard table is client-rendered and may not be reliably crawlable, so notebook scores are **not claimed to equal the exact current #1 score**.

Observed on the competition Code page recently:

- official Random sample: ~0.18
- official Stochastic Goose sample: ~0.25
- Persistent Memory BFS notebook: 0.46
- Gorilla Eval (New): 0.87
- Qwen3.6 Duck full-code notebook: 1.17
- Tough Guard V2: 1.18
- Sandwich: 1.22

A Kaggle discussion posted roughly 12 days before this snapshot described the live top as still **below 2%**. Another discussion documents notebook-page vs leaderboard-score mismatches, so we should treat notebook tiles as noisy public anchors.

Research results on the 25 public games are much higher and not hardware-comparable. Tycho/frontier-model work reports public-set saturation, while the Kaggle code competition permits only a single <=9-hour CPU/GPU notebook, no internet, and currently offers RTX 6000 machines.

## Operational targets

- `>0.46`: clear hand-search baseline territory
- `~1.0`: credible local-model agent
- `1.1–1.3`: contemporary strong public/open notebook range visible in the current crawl
- `>2.0`: meaningful current frontier break if confirmed on the live leaderboard
- `5+`: new regime; likely requires a real generalization/inference-budget breakthrough

Sources:
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/code
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728934
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717055
