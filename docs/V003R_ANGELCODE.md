# V003R AngelCode

V003R is ARCangel's first true coding-agent submission candidate.

## Primary hypothesis

A Qwen3.6 27B FP8 actor should perform better when it can write bounded Python over a complete structured interaction ledger than when it must reason only inside one model response and immediately emit actions.

This is the cleanest next ablation after V002. It combines three ideas that have independent public evidence without adding public-game rules:

1. **Duck-style coding:** exact calculations/search can be delegated to code.
2. **PRO-LONG-style memory:** the raw transition ledger remains complete and Python can query it without stuffing the full history into model context.
3. **Tycho-style selective modeling:** the actor requests code only when an exact computation or executable hypothesis is worth the cost; world modeling is not mandatory on every game.

## Tool boundary

Model-written Python is an analysis tool only. It cannot directly call the ARC environment. The sandbox blocks arbitrary imports, files, network access, eval/exec, dunder access, and unbounded Python execution. It exposes the current grid/object graph, complete transition records, recent frames, beliefs, and compact retrieval helpers.

Every real ARC action is still parsed, bounds-checked, checked against `available_actions`, and executed by the normal competition runner.

## Runtime allocation

- RTX Pro 6000, Internet OFF
- Qwen3.6 27B FP8
- 12 concurrent games
- vLLM `max_model_len=8192`, `max_num_seqs=16`, prefix caching
- 8-hour play budget inside Kaggle's 9-hour hard limit
- 96 model generations/game maximum
- 32 Python analyses/game maximum
- later levels: dense reasoning
- level 0 after 150/400 actions without progress: progressively throttled model use
- structural explorer remains available as fail-soft fallback

## Required Kaggle inputs

- ARC Prize 2026 - ARC-AGI-3
- ARC3 vLLM H100 Wheelhouse V3
- vrfai Qwen3.6 27B FP8 Hugging Face Snapshot

The generated notebook embeds ARCangel source and does not require the TAAF source dataset.

## What to retain after a run

Preserve the full notebook log and `arc3_frontier_v003r.json`. In addition to the scorecard, the receipt records model calls/failures, reasoning cycles, Python tool calls/failures, queued-plan actions, structural fallbacks, resets, errors, and deadline exhaustion.
