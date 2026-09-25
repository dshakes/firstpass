# Pre-registration — decision-model prior, replayed on real MBPP matrices

Written 2026-09-25, **before any prior was fetched or any arm was scored.** It is committed first so the
verdict can't be tuned to the result.

## Question

On real model outcomes, does a pre-generation decision-model prior on the start rung (ADR 0013) lower $/success
compared with plain first-pass, without serving more wrong answers?

## Data (already recorded, no new generation)

These are 974-task MBPP replay matrices. Each has per-rung `gate_full_pass`, `oracle_correct` and real `cost_usd`:

- `fp-mbpp-974-sonnet-k5.jsonl`: haiku-4-5 → sonnet-5
- `fp-mbpp-974-opus-k5.jsonl`: haiku-4-5 → opus-4-8
- `fp-mbpp-974-openai.jsonl`: gpt-4.1-mini → gpt-5.5

The prompts are canonical MBPP (`task_id` N ↔ `mbpp-N`).

## Prior source

This uses **OpenJev** (Apache-2.0, DiffusionGemma 26B-A4B), served locally with an API compatible with
`POST /v1/systemone`. **It is not TypeSafe's Jev**, so results say nothing about Jev's own accuracy. Each task gets one
`choice` question, with one rubric line per rung ("rung r is the least capable tier that fully handles this").
Probabilities are converted with `firstpass_core::cumulative_pass`. Priors are fetched once, saved to a JSONL file and
replayed, so the scoring is deterministic.

## Arms

All gated arms serve the first rung at or above the start rung that passes the gate. Their semantics match the
existing `costaware::first_pass`.

1. `first-pass`: start at rung 0.
2. `prior`: start at the expected-cost argmin under the prior. Rung price is **cross-fitted mean cost per rung**,
   which is known before generation and never uses the task's own cost.
3. `prior-unverified`: the argmin rung is served without a gate. This is the Jev-router stand-in.
4. Reference arms: `always-cheap`, `always-top`, `cost-aware (learned p)` (existing), `cost-aware (oracle p)`
   (upper bound).

## Metrics

Success means the served output has `oracle_correct`. Also reported: $/success, served-failure and escalation rate,
each with seeded bootstrap 95% CIs from `stats.rs`.

## Kill criterion (per ladder, and pooled)

**PROCEED** only if, on the pooled three-ladder result, `prior` has lower $/success than `first-pass` with the
bootstrap CI of the difference excluding 0, **and** its served-failure rate is not higher than `first-pass`'s by more
than 1 percentage point. **Otherwise PRIOR=STOP**, which confirms ADR 0013's simulation verdict. Per-ladder results
are reported but cannot override the pooled verdict.

## Degeneracy guard

If more than 95% of tasks get the same argmin rung under the prior, the run is reported as **DEGENERATE** (the prior
carries no per-query signal) and counts as STOP.
