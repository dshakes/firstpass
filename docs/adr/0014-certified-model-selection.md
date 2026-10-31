# ADR 0014 — Certified model selection: choose the route's model from its own outcomes, with an anytime-valid guarantee

Status: **accepted — stage 2 (paired shadow mode, shared state via Redis, `observe_pair`)**.
Stage 1: live challenger arm, halt guard, `firstpass experiment` report.
Date: 2026-10-30.

## Context

Firstpass's per-request cascade wins where a precise gate exists. On MBPP the audited head-to-head
holds (+3.3pp / +5.1pp against the best-tuned judge) because the gate is a test run.

Agentic traffic is different, and the evidence is now direct rather than inferred.
`docs/benchmarks/agentic-real-runs.md` replays 500 real SWE-bench Verified agent runs per model, for
11 models, at $0. Three findings:

1. **No gate available on a finished agent run was precise enough for escalation to beat
   always-top.** That held for step counts on four ladders and for patch agreement, which failed
   replication. An oracle gate would win by 2–6pp, so the loss is the gate's, not the cascade's.
2. **The money is in choosing the model for the workload.** Cross-fit selection of the cheapest
   model on par with opus-4.5 resolved 374 vs 384 (difference not significant) at $0.219 vs $0.754
   per task, 71% cheaper.
3. That selection was made on a public benchmark. **An operator needs it made on their own traffic,
   and the only honest evidence for an agent task is the task outcome itself.**

Routers that pick a model per request from a learned predictor (Not Diamond, Martian, RouteLLM-style)
predict quality; none of them prove it on the customer's own outcomes. Firstpass's wedge has always
been proof over prediction, so model choice should be proved the same way.

## Decision

A route may declare a **challenger ladder**. A configured fraction of sessions is randomized to it and
served through the normal enforce path, so gates, budget, escalation and receipts are unchanged. Task
outcomes arrive through the existing feedback API. A sequential non-inferiority test
(`firstpass_core::select`) then decides, at any moment anyone reads it, one of three states:

| status | meaning (error ≤ `alpha` for each conclusion) | what happens |
|---|---|---|
| `certified` | challenger is no worse than the incumbent by more than `margin` | `firstpass experiment` prints the ladder to adopt **if it is also cheaper**; a human adopts it |
| `halted` | challenger is worse by more than `margin` | new sessions stop going to the challenger, automatically, with an error log and a metric |
| `undecided` | not enough evidence | keep running |

```toml
[[route]]
mode = "enforce"
ladder = ["anthropic/claude-opus-4-5"]

[route.challenger]
ladder  = ["openrouter/minimax-m2.5"]
percent = 10        # % of sessions served by the challenger
margin  = 0.05      # default; challenger may be at most 5 points worse
alpha   = 0.05      # default; error rate of each conclusion
```

### The test

Arms are assigned per session by salted hash, under a domain tag independent of the rollout and
shadow sampling. The `k`-th incumbent outcome is paired with the `k`-th challenger outcome. Take
`D = X_inc − X_ch` and run two grid-mixture betting e-processes:

```text
certify : H0 E[D] ≥ m     wealth ·= 1 + λ (m − D)
halt    : H0 E[D] ≤ m     wealth ·= 1 + λ (D − m)
```

Ville's inequality bounds the chance that either one ever crosses `1/alpha` under its null by `alpha`.
That bound holds **however often it is read**. This is the property a live experiment needs. A
fixed-sample test that is re-run as data arrives, with the first "significant" result adopted, is
optional stopping. That is the failure mode `calibrate.rs` already documents for thresholds.

Only the first outcome of each session counts. Outcomes within a session are correlated, and
counting them all would break the pairing argument. The experiment is per (tenant, route), so one
tenant's outcomes never decide another's model (ADR 0004).

### Why these defaults

The design was replayed on the real runs before any of it was written
(`scripts/sim-certified-selection.py`). The replay uses the same grid and stopping rule as
`select.rs`, with incumbent opus-4.5, a 50/50 split, 400 replays, and a cap of 20k sessions:

| challenger | true gap | cost ratio | m = 0.03: certified / halted | m = 0.05: certified / halted | m = 0.05 median sessions |
|---|---|---|---|---|---|
| minimax-2.5 | +0.010 | 0.10 | 0.63 / 0.01 | **0.99 / 0.01** | 3,667 |
| gemini-3-flash | +0.010 | 0.47 | 0.63 / 0.01 | **0.99 / 0.01** | 3,600 |
| opus-4.6 | +0.012 | 0.73 | 0.50 / 0.01 | **0.99 / 0.01** | 4,154 |
| kimi-k2.5 | +0.060 | 0.19 | 0.02 / 0.92 | 0.03 / 0.15 | 1,481 |
| haiku-4.5 | +0.102 | 0.44 | 0.01 / 0.99 | 0.01 / **0.99** | 2,206 |
| gpt-5-mini | +0.206 | 0.06 | 0.00 / 1.00 | 0.00 / **1.00** | 223 |
| synthetic, exactly `m` worse | `m` | — | 0.055 / 0.033 | 0.045 / 0.058 | — |

At `m = 0.03` the good challengers needed about 9k sessions and certified in only 63% of replays
within 20k. That is longer than most operators will run an experiment, so the default is `0.05`. The
dominant cost is information, not the method. Two unpaired Bernoulli arms at ~0.76 give `sd(D) ≈ 0.6`,
and a fixed-sample test of a 4-point gap needs about the same order of sessions.

### What it does not claim

- **Outcome non-inferiority on the traffic it saw.** A workload shift after certification is not
  covered; a fresh experiment is.
- **Cost is reported, not tested.** Adopting is advised only when the challenger's $/request is lower.
  Cost has no natural bound, so a betting test on it would need a budget-derived bound. That is not
  worth its complexity while the cost gap is usually 2–10×.
- **Never auto-promotes.** Changing what a route serves is the irreversible step, so a human makes it.
  Halting is automatic because it only ever moves traffic back to the incumbent.
- **Halt state is rebuilt from receipts at startup.** Between a halt and a restart it lives in
  memory. The rebuild replays the same receipts the CLI reads, so the two cannot disagree.

## Consequences

- The challenger arm skips start-rung learning that is indexed by the incumbent ladder (bandit,
  `[escalation.prior]`, session promotion). Their state is per incumbent rung and means nothing
  on another ladder.
- Receipts gain an optional `experiment` field `{arm, percent, bucket}`. It is absent when no
  challenger is configured, so earlier traces hash identically and the chain stays re-derivable.
- An operator without outcome feedback cannot use this. That is deliberate: without outcomes there
  is nothing to certify, and certifying on gate verdicts alone would repeat finding 1.

## Stage 2 — Paired shadow mode and shared state (this branch)

### Paired shadow mode (`mode = "paired"`)

In paired mode no session is ever served by the challenger ladder. Instead, after the incumbent
response is returned, a detached task re-runs the challenger ladder on **the same request**. The
incumbent gate verdict (`x_inc`) and the challenger gate verdict (`x_ch`) form a pair:

- `x_inc = 1.0` if the incumbent's final `served_from` is `Attempt` (gate passed)
- `x_inc = 0.0` if `BestAttempt` (gate failed, served anyway)
- `Cache` or `Error` → pair dropped (no fresh gate evaluation)
- `x_ch = 1.0` if the challenger's ladder produced a gate-passing answer; `0.0` otherwise

Each pair feeds `Experiment::observe_pair`, which bypasses the FIFO queuing used in live mode
(both outcomes arrive simultaneously, so FIFO would introduce incorrect buffering).

**Why this certifies gate-pass non-inferiority, not outcome non-inferiority.** The incumbent and
challenger each run a gate (a deterministic function of the output), not a task outcome (which
requires a human evaluation or a symbolic test that only exists for some workloads). The certificate
therefore reads "the challenger's gate-pass rate is within ε of the incumbent's" — valid for
single-turn routes where the gate is a meaningful quality proxy. For agentic sessions (finding 1 of
the evidence above) where no gate is precise, live mode with real task outcomes is the right tool.

**Why paired mode is only allowed on single-turn routes.** An agent session diverges after the first
tool call: the challenger's second turn sees the incumbent's first-turn tool results, not its own.
The gate verdicts of those turns are not comparable. The config validator rejects paired mode on any
route whose gates are all `non-empty` (a pass-everything gate makes non-inferiority meaningless).

**Spend ceiling is required.** Paired mode makes real model calls on every (sampled) enforce
request. `max_usd_per_day` is enforced via the same `ShadowLedger` used by shadow scoring — it
stops new evaluations for the rest of the UTC day once the ceiling is reached. The stop is logged,
never silent.

### Shared state across replicas (`redis_url`)

When `redis_url` is set under `[route.challenger]`, every outcome (live mode) and pair (paired mode)
is appended to a Redis list keyed `firstpass:exp:{tenant}:{route_ix}:{fingerprint}`. Each replica
rebuilds its local `Experiment` from that list via `sync_from_backend`:

- Called after every feedback event (outcome or pair)
- Called on a background timer ≤ 5 s (`start_sync_task`)
- Uses `LRANGE offset -1` so each call is cheap when there are no new records
- Live mode session dedup is cross-replica: `SET NX` on a per-session key (1-day TTL) ensures
  one session is counted at most once across all replicas

Without `redis_url` behavior is unchanged (no append, no sync timer). The backend is behind the
`redis-cache` Cargo feature; a URL set without that feature is rejected at startup.

## Follow-ons (not in stage 2)

- **Several challengers at once,** with an alpha split across them, or a best-arm-identification
  design.
- **A live run.** Every number above comes from public recorded runs. A paid live experiment on a real
  route is the claim's final proof and needs the operator's spend approval.
