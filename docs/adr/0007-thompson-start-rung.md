# ADR 0007 — Thompson-sampling start-rung with discounting and a speculative-deferral band

Status: accepted · 2026-07-20

## Context

The start-rung bandit shipped as deterministic UCB1 (`bandit@v1`). Deterministic selection has
degenerate (0/1) propensities, so off-policy evaluation of start-rung policies (`firstpass ope
--start-rung`) required the epsilon-greedy overlay as a bolted-on randomiser. Separately, model
churn and workload drift make stale arm counts actively harmful, and unconditional speculative
prefetch (`speculation > 0`) pays parallel-token waste on requests where the outcome was never
in doubt.

## Decision

1. **`[escalation.bandit] algorithm = "ucb1" | "thompson"`** (default `ucb1` — byte-identical
   for existing deployments; the default flips only after a live A/B shows a win, mirroring how
   `bandit@v1` itself was validated).
2. **Thompson sampling** (`bandit@v2-ts` policy stamp): per `(context, rung)` Beta posteriors
   over gate-pass, sampled per decision; the sampled pass-probabilities drive the *same*
   expected-cost argmin as UCB1 — only estimation/selection changes, never the objective.
   Beta draws come from a hand-rolled xorshift64* + Box–Muller + Marsaglia–Tsang Gamma sampler
   (no new dependencies; the `+1` uniform prior keeps both shapes ≥ 1, the method's domain).
3. **Propensities are Monte-Carlo estimated** (M = 64 re-draws; logged propensity = fraction of
   re-draws agreeing with the taken decision, floored at `1/(2M)` so IPS never divides by 0).
   Exact TS selection probabilities are analytically intractable; the MC estimator is the
   standard practice and its error is visible in `firstpass ope`'s bootstrap CIs. When
   `algorithm = "thompson"`, the epsilon overlay is redundant and ignored (warned once): the
   policy is already stochastic and self-logging.
4. **Discounting** (`discount ∈ (0, 1]`, default 1.0): every observation multiplies the
   context's counts by `discount` first — geometric forgetting, the textbook non-stationarity
   fix for TS. `0.99` ≈ an effective window of ~100 observations.
5. **Speculative-deferral band** (`[escalation.speculation] speculation_band = [low, high]`):
   with `speculation > 0` and a warm bandit, prefetch fires only when the posterior-mean
   gate-pass estimate of the chosen start rung lies inside the band — the marginal zone where
   the next rung is probably-but-not-certainly needed, which is where parallel spend buys the
   most latency per wasted token (the deferral-rule insight from speculative cascades).
   Outside the band the request runs serial; `firstpass_speculation_skipped_total` counts the
   saves. Cold context or no bandit ⇒ configured `speculation` applies unchanged.

## Invariants

- Prediction still only chooses where the ladder *starts* (and whether prefetch fires); the
  gate decides what is *served*. A wrong bandit costs money or latency, never correctness.
- Default-off: absent config keys reproduce v1 behavior byte-for-byte.
- Receipts stay the durable bandit state: warm-start replays the trace store on boot; ladder
  changes create fresh contexts rather than reusing stale arms.

## Consequences

- `firstpass ope --start-rung N` gets non-degenerate logged propensities natively under
  Thompson — IPS/SNIPS/DR become first-class without epsilon.
- The MC propensity adds ~64 posterior draws per routed request (microseconds; no model calls).
- A live A/B (thompson vs ucb1, same workload) is the promotion gate for changing the default.
