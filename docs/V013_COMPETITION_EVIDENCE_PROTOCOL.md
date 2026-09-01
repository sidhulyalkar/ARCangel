# V013 Competition Evidence Protocol

## Purpose

ARCangel treats a Kaggle score as a noisy external measurement, not as a deterministic property of an architecture. The final experimental unit is therefore:

`contestant ID + exact notebook SHA-256 + exact runtime configuration + repeated Kaggle executions`.

Never average scores from different notebook hashes and call the result one model.

## Competition topology

Current ARC Prize 2026 competition guidance matters for runtime design:

- scored notebooks evaluate both hidden halves, for 110 games total;
- only the public half contributes to the displayed public leaderboard during the competition;
- the notebook GPU runtime limit is 9 hours;
- private scores are calculated from the original scored run rather than by rerunning the notebook after the competition.

Primary references:

- Kaggle competition overview: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview
- Host discussion on 110-game evaluation: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion

The important engineering consequence is that optimizing only the displayed public half is not enough. Every scored notebook must budget compute across the full hidden game set.

## Why one leaderboard score is insufficient

Public discussion has reported materially different scores from repeated executions of the same submission, including a reported 2.11 followed by 0.89. Whether the variation comes from environment/runtime/model nondeterminism or evaluation details, the operational lesson is the same: one displayed score is not strong enough evidence to declare an architecture superior.

V013 therefore uses repeated exact-artifact evidence.

## Exact-artifact evidence rule

Production `configs/swarm-v013.json` requires:

- at least 2 Kaggle runs for the exact Duck control notebook hash;
- at least 2 Kaggle runs for the exact challenger notebook hash;
- a provenance label for every run;
- the exact notebook SHA-256 for every run;
- no pooling across hashes.

A third or later run may be added without changing the candidate identity. This is useful when the first two runs are too variable to resolve the comparison.

The control should also remain one exact artifact. If two distinct Duck notebook hashes each accumulate enough evidence to qualify, V013 treats the control as ambiguous and refuses to nominate a challenger until the control identity is resolved.

## Leaderboard comparison

For each exact artifact, V013 records:

- raw scores;
- number of runs;
- mean score;
- population standard deviation;
- standard error;
- uncertainty-adjusted lower/upper bounds;
- mean runtime when supplied.

The challenger/control comparison uses:

`mean_delta - confidence_se * sqrt(candidate_SE^2 + control_SE^2)`

The production default is one combined standard error. A candidate is not nominated unless this lower bound clears `min_leaderboard_delta`.

This intentionally prefers a stable 2.0 over a 2.5/0.5 lottery whose evidence is still unresolved.

## Sequential submission policy

Treat scarce Kaggle submissions as sequential experiments:

1. Establish one exact Duck control artifact.
2. Obtain the configured minimum repeat evidence for it.
3. Submit only a challenger that has already passed DEV, VALIDATION and private BLIND.
4. Repeat that *same challenger artifact* until either:
   - its uncertainty-adjusted delta clears the control; or
   - additional evidence makes superiority implausible enough to stop spending slots.
5. Change notebook bytes only when starting a new experimental candidate. A changed hash resets that artifact's evidence group.

Do not cherry-pick the best run from an artifact.

## Runtime coverage contract

The V013 Kaggle candidate runner carries two runtime receipts.

### Competition envelope

Before vLLM launch, the runner audits the expected scored configuration against:

- expected games: 110;
- workers: 28;
- notebook limit: 32,400 seconds;
- setup reserve: 3,600 seconds;
- ARC campaign budget: 25,200 seconds;
- requested per-game cap: 7,800 seconds;
- coverage reserve: 5%.

For 110 games and 28 workers there are four scheduling waves. A 7,800-second worst-case game cap would allow four waves to require 31,200 seconds, exceeding the 25,200-second campaign window.

The coverage-safe cap is therefore:

`25200 * 0.95 / 4 = 5985 seconds`

for a 110-game run.

### Actual discovered environment

`run_suite()` also computes the same scheduling math from the number of games actually exposed to the notebook.

This is adaptive:

- a 25-game public run with 28 workers has one wave and keeps the requested 7,800-second cap;
- a 110-game scored run has four waves and tightens the cap to protect full-suite coverage.

The purpose is not to make every game equally easy. It is to prevent a few early difficult games from starving unseen later games.

## Deadline semantics

V013 distinguishes three cases:

- `game_budget`: one started game used its allotted cap. This is allowed and recorded.
- `global`: a started game was still active when the shared campaign deadline expired. This invalidates the coverage contract.
- `before_start`: a discovered game never started because the suite deadline was already exhausted. This invalidates the coverage contract.

A candidate notebook fails its runtime qualification if any game is globally starved or skipped before start. A deliberately capped hard game does not fail the notebook by itself.

## Required receipt fields

A trustworthy Kaggle candidate receipt should contain at least:

- build ID;
- contestant ID;
- policy profile;
- model family and model path;
- `competition_runtime_envelope`;
- actual `runtime_budget`;
- per-game results;
- model/runtime diagnostics;
- official `submission.parquet` existence check.

After Save & Run, record the displayed Kaggle score separately with:

```bash
python scripts/run_swarm_lab.py record-kaggle \
  --contestant <contestant-id> \
  --score <displayed-score> \
  --seed <unique-run-receipt-id> \
  --source <kaggle-run-or-version-id> \
  --artifact-sha256 <exact-notebook-sha256> \
  --runtime-seconds <runtime>
```

`--seed` is used as the unique ledger run key for this external observation. It should be unique for repeated scored executions even when the notebook's internal policy seed is unchanged.

Then inspect:

```bash
python scripts/run_swarm_lab.py leaderboard-evidence
python scripts/run_swarm_lab.py leaderboard
```

## Invalid evidence

Do not use any of the following as proof of leaderboard superiority:

- a local suite labeled as `kaggle`;
- a reported public score without artifact provenance when strict mode is enabled;
- a single lucky run under the production V013 manifest;
- scores pooled across notebook hashes;
- a candidate that never passed private BLIND;
- a scored run that skipped hidden games because the shared deadline was exhausted;
- an internal composite score compared numerically to a Kaggle official score.

The purpose of these constraints is not bureaucracy. They make it harder for an autonomous research system to manufacture a false victory from noisy measurements.
