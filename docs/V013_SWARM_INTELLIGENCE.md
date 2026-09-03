# V013 Hypothesis-Space Swarm Intelligence

## Objective

ARCangel should use many agents to search the architecture space without giving social consensus authority over scientific truth.

The swarm therefore operates primarily during development. Each model/role pair behaves like a particle exploring a different cognitive niche. Measured ARC arena outcomes provide the utility signal. The submitted Kaggle policy remains compact and offline unless a specific multi-agent inference mechanism independently earns its place through controlled evaluation.

## Why not ordinary multi-agent debate

Recent multi-agent research provides two important lessons:

1. Diversity can improve search and reasoning.
2. Free-form debate and majority consensus are fragile. Persuasive agents can pull other agents toward wrong answers, majority pressure can suppress useful minority reasoning, and much of the apparent gain from debate can come from simple ensembling rather than genuine correction.

ARCangel therefore separates proposal generation, peer review, and empirical authority.

Reviewers decide which hypotheses deserve scarce experiment compute. They cannot promote code. DEV/VALIDATION/BLIND and Kaggle gates remain authoritative.

## Swarm lifecycle

```text
measured arena history
        |
        v
personal / role / global attractors
        |
        v
independent heterogeneous particles
        |
        v
anonymous proposals
        |
        v
blinded cross-review
        |
        v
evidence-weighted experiment priority
        |
        v
isolated implementation branches
        |
        v
controlled ARC arena battles
        |
        v
measured utility -> swarm memory
        |
        +--------------------------> next generation
```

This resembles particle-swarm optimization conceptually, but the search space is architectural hypotheses rather than numeric weights.

## Particles

A particle is a `(provider, role)` pair.

The current roles are:

- minimalist
- scientist
- explorer
- planner
- vision
- memory
- runtime
- red team
- generalization
- integrator

The default heterogeneous NVIDIA swarm uses three independent model families through the same OpenAI-compatible NVIDIA endpoint:

- `nvidia/nemotron-3.5-lightning-30b-a3b`: minimalist, runtime, integrator
- `deepseek-ai/deepseek-v4-pro-0813`: scientist, planner, memory, generalization
- `z-ai/glm-5.2`: explorer, vision, red team

All credentials are read from `NVIDIA_API_KEY`. No key is stored in repository files.

## Independent generation first

Particles receive the same blind-safe research context but different role missions. They do not see peer proposals during initial generation.

From generation 2 onward, each particle also receives measured swarm guidance:

- global best measured proposal;
- its own personal best measured proposal;
- the best measured proposal from its cognitive role.

These are explicitly described as **attractors for mutation, not truths**. Each particle is instructed to preserve useful causal structure while mutating at least one assumption, representation, or control mechanism.

This provides exploitation without turning the swarm into a cloning machine.

## Blinded cross-review

Proposal author/provider identity is removed before review.

Review assignments are deterministic and favor reviewers from different providers and different roles. Exact self-review is forbidden.

Each review returns bounded scores for:

- falsifiability;
- generalization;
- information gain;
- implementation feasibility;
- redundancy;
- persuasion risk;
- reviewer confidence.

It must also state:

- strongest objection;
- decisive experiment;
- verdict: `advance`, `test_disagreement`, or `reject`.

Reviewer confidence has deliberately low weight. Confidence is not evidence.

## Robust priority rather than majority vote

The council computes a robust experiment priority from review dimensions.

Positive terms emphasize falsifiability, generalization, information gain, and feasibility. Priority is penalized for:

- reviewer disagreement;
- persuasion risk;
- redundancy.

The output is a **compute-priority ranking**, not an architectural promotion ranking.

### Minority preservation

Selection first preserves role diversity before filling remaining slots by robust priority.

High reviewer disagreement is not automatically bad. If disagreement is high while expected information gain remains high, the proposal is tagged as a `disagreement_experiment`. That means the system should run the smallest experiment capable of resolving the disagreement instead of suppressing the minority hypothesis.

## Measured swarm memory

`SwarmMemory` is append-only and accepts only DEV or VALIDATION outcomes.

BLIND and Kaggle evidence are forbidden from entering development swarm memory. This prevents private holdout information from leaking into architecture invention.

One outcome records:

- proposal ID;
- provider;
- role;
- split;
- measured utility;
- source receipt;
- optional note.

The next generation uses these outcomes as personal, role, and global attractors.

## Heterogeneous NVIDIA research cycle

With `NVIDIA_API_KEY` available:

```bash
python scripts/run_swarm_research_cycle.py \
  --providers configs/research-providers.nvidia-swarm.json \
  --generation 1
```

This runs:

1. independent proposal generation;
2. deterministic blinded reviewer assignment;
3. heterogeneous cross-review;
4. robust swarm priority;
5. generation battle-plan creation.

If coding-agent workers are configured:

```bash
python scripts/run_swarm_research_cycle.py \
  --providers configs/research-providers.nvidia-swarm.json \
  --workers configs/experiment-workers.local.json \
  --generation 1 \
  --execute-workers
```

The workers create isolated branches and qualify software changes. They still do not establish ARC competence.

## Closing the loop

After a selected proposal is evaluated against its explicit control in the arena, record its measured utility:

```bash
python scripts/record_swarm_outcome.py \
  --proposal-id SWARM-01-example \
  --provider-id nvidia-deepseek-v4-pro \
  --role-id scientist \
  --split validation \
  --utility 0.037 \
  --source artifacts/arena/v013/example-result.json \
  --note "repeatable held-out improvement"
```

Then run the next generation:

```bash
python scripts/run_swarm_research_cycle.py \
  --providers configs/research-providers.nvidia-swarm.json \
  --generation 2
```

Generation 2 can exploit measured winners while retaining heterogeneous exploration.

## Authority hierarchy

The swarm must obey this order:

1. immutable environment evidence;
2. repeated controlled arena outcomes;
3. BLIND judge results;
4. exact-artifact Kaggle evidence;
5. deterministic swarm priority;
6. reviewer opinions;
7. proposal rhetoric.

A lower layer cannot override a higher layer.

## Relationship to J-sparse-debate

`J-sparse-debate` remains disabled.

The development swarm should first establish whether multi-agent reasoning produces mechanisms that improve controlled ARC behavior. Only then should we test sparse multi-agent inference inside the scored agent.

If J is activated later, it should use the same lessons:

- trigger only at pivotal uncertainty;
- generate independent candidate theories before information sharing;
- anonymize proposals;
- use an evidence-aware arbiter rather than majority vote;
- stop debate when an executable discriminating test is available;
- measure token/GPU cost against the single-agent control.

## Falsification criteria

The swarm approach should be reduced or abandoned if any of the following occurs repeatedly:

- proposal diversity collapses despite role/model heterogeneity;
- council ranking does not predict which experiments improve held-out behavior;
- review cost exceeds the value of avoided bad experiments;
- measured swarm generations converge without improving arena utility;
- simple single-agent research produces equally strong experiments at lower cost;
- external-model research systematically encourages public-game-specific mechanisms.

Swarm intelligence is useful only if it improves the efficiency of scientific search. The swarm itself is not the product.
