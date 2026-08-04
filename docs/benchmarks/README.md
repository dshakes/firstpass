# Benchmark result artifacts

Every number Firstpass publishes must regenerate from a committed command — this directory
holds the raw output of the runs the docs cite, exactly as the harness printed it. If a
number appears in the README or on the site and its artifact isn't here, treat the number
as stale and the artifact as the bug to file.

| Artifact | Claim it backs | Command | Cost |
|---|---|---|---|
| [`mbpp-live-base.txt`](mbpp-live-base.txt) ✅ | MBPP served-failure bound, base (test) gate | `FIRSTPASS_CODING_DATASET=./mbpp.jsonl cargo run --release -p firstpass-bench -- --coding-live` | your key + Docker, ~$5 |
| `mbpp-live-judged.txt` | MBPP bound with an LLM judge on the gate | same, plus `FIRSTPASS_CODING_JUDGE=<model>` | your key + Docker, ~$10 |
| [`mbpp-policy-974.txt`](mbpp-policy-974.txt) ✅ | Routing policy vs always-cheap / always-top, haiku→sonnet ladder | `FIRSTPASS_CODING_DATASET=./mbpp-974.jsonl cargo run --release -p firstpass-bench -- --coding-policy` | your key + Docker, ~$3 |
| [`mbpp-policy-974-opus.txt`](mbpp-policy-974-opus.txt) ✅ | Same comparison on a ~15× ladder, haiku→opus | same, plus `FIRSTPASS_CODING_LADDER="anthropic/claude-haiku-4-5,anthropic/claude-opus-4-8"` | your key + Docker, ~$9 |
| [`elastic-validation.txt`](elastic-validation.txt) ✅ | Elastic-verification Phase 3: cost saved + held-out served-failure bound | `cargo run --release -p firstpass-bench -- --elastic` | free (offline) |
| `live-200.txt` | 200-task live cost/success table | `cargo run -p firstpass-bench -- --live` | your key, ~a few $ |

Provenance rules:

- Artifacts are the harness's stdout, unedited. The harness self-labels simulation vs live.
- The MBPP dataset is not vendored (license + size); the command above fetches the same
  public `mbpp.jsonl` the published runs used (974 rows; tasks that fail the loader's
  strict assert-conversion are skipped and counted in the artifact).
- Model prices drift; `$/success` comparisons inside one artifact are internally consistent
  (same price table snapshot), but cross-artifact dollar comparisons are not meaningful.
- **A saving is meaningless without its ladder.** The two policy artifacts run the same policy
  on the same 974 tasks and differ only in the top rung, and they report 12% and 57%. Quote
  either number with the ladder attached, or it says nothing.
- The policy artifacts compare policies **paired**, on one shared matrix of measurements: each
  task is solved once per rung and every policy is replayed over those same outcomes. Marginal
  confidence intervals are the wrong test for that design — they carry the same task-difficulty
  variance twice and overlap even when every disagreement points one way. The per-task
  difference is what the artifacts report a CI on.
- A re-run on your account will differ (model updates, sampling): the *bound machinery* is
  the reproducible part — the harness recomputes the conformal bound from your own run's
  gate/oracle outcomes with the same pre-registered α/δ.
