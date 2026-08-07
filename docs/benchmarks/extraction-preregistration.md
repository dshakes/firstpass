# Pre-registration: the structured-extraction study

Written **before** the study runs. Everything below — the hypothesis, the sample size, and the
condition under which the result counts as a failure — is fixed now precisely so it cannot be
adjusted after seeing the numbers. That is the only thing separating a benchmark from a
demonstration.

## The objection this answers

Every measurement Firstpass publishes comes from **coding tasks with executable tests** — MBPP,
scored by the same harness that runs BigCodeBench. (A SWE-bench harness exists in the tree but has
no CLI entry point, so nothing published rests on it — see ADR 0010.) That is the single domain
where a gate is nearly free, and it is fair to ask whether verification-gated routing works
anywhere a gate is harder to write.

## Task

Extractive question answering as structured output, from **SQuAD v2 validation**
(`rajpurkar/squad_v2`), answerable rows only.

The model is given a passage and a question and must reply with `{"answer": "..."}`. The answer is
an exact span of the passage with a published human label, so the oracle is a string comparison
against ground truth — **not** a model judging a model. A domain proof whose oracle is itself an
LLM would prove nothing about LLM reliability.

Unanswerable rows are excluded. Correctly declining to answer is a real capability, but it needs
its own gate semantics, and mixing it in would make one accuracy number mean two different things.

## Gate vs oracle

| | Checks |
|---|---|
| **Gate** (what the router acts on) | reply parses as JSON, and carries the `answer` key |
| **Oracle** (ground truth, never shown to the router) | the value equals the human-labelled span |

The gap is deliberate and load-bearing: a model can emit perfectly-formed `{"answer": "..."}` with
the wrong span in it, pass the gate, and be wrong. Without that gap every gate pass would be a
correct answer by construction, the served-failure bound would be trivially zero, and the study
would be circular.

## Pre-registered parameters

- **n = 500** answerable rows, taken as a **prefix in dataset order** (a prefix reproduces without
  recording a seed; the manifest records exactly which ids were used)
- **Ladder**: `anthropic/claude-haiku-4-5` → `anthropic/claude-sonnet-5`
- **Policies compared**: always-cheap, always-top, first-pass — all replayed over one shared
  measurement matrix, so they are judged on identical outcomes
- **Statistic**: paired bootstrap, 95% CI. Paired is the correct test because both arms are the
  same tasks decided two ways; marginal intervals overlap on this design even when every
  disagreement points one way

## Kill criterion

**The claim fails if the paired improvement of first-pass over always-cheap has a 95% CI whose
lower bound is ≤ 0.**

That is the same criterion the MBPP study used. If it fires, the honest reading is that
verification-gated routing does not transfer from executable-test domains to schema-gated ones, and
that gets published as prominently as a positive result would.

A secondary observation, recorded but **not** a pass/fail condition: how often the gate passes an
answer the oracle rejects. On MBPP that number was zero — no correct cheap answer was ever
rejected — and there is no reason to expect it to hold here. If it is materially non-zero, the
"zero regressions" language stays scoped to coding and is not generalised.

## Running it

```bash
python3 scripts/fetch-coding-dataset.py --dataset extraction --limit 500 --out extraction-500.jsonl
export ANTHROPIC_API_KEY=...            # real spend
export FIRSTPASS_CODING_DATASET=extraction-500.jsonl
cargo run -p firstpass-bench --release -- --coding-policy
```

The artifact and its manifest get committed under `docs/benchmarks/` whatever the outcome.
