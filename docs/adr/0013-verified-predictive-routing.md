# ADR 0013 — Verified predictive routing: a decision-model prior on the start rung, never on what is served

Status: **accepted (pending live measurement)** · 2026-09-25

## Context

A new class of "System One" decision models — TypeSafe AI's **Jev** (launched 2026-09-15;
`POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`) answers a single closed-form
question (`noul`: P(yes); `choice`: probabilities + confidence over fixed options; `score`) in one
cheap call — $0.042 per 1M input tokens, $0 output, no text generation, cited at ~10 decisions/s.
Products built on it (`jev-router`, `prismhq/jev-router`) route a query to a model tier by asking
Jev to pick the tier directly, then **serve that tier's output unverified**.

That is exactly the "decide before the output exists and hope" pattern `docs/related-work.md`
positions Firstpass against. But the underlying signal — a cheap, per-query estimate of which tier
is likely sufficient — is real, and Firstpass's own start-rung bandit (ADR 0007) already has a spot
built for exactly this kind of estimate: it just has no information before `min_observations`
gate-verdicts accumulate in a context bucket, so every new context starts at rung 0 regardless of
how obviously hard or easy the query is. A per-query prior fills that gap without touching what the
gate is allowed to serve.

Jev has no published accuracy benchmarks and is not deterministic: the same query asked twice can
return different probabilities. Analysts also flag a weak moat — frontier labs are expected to ship
their own decision models, so this ADR treats Jev as an interchangeable HTTP dependency, not a
platform bet.

## Decision

Add an optional `[escalation.prior]` config block. When set (and `[escalation.bandit]` is also
set — the prior blends into the bandit's posterior and is a no-op without one), the router asks Jev
one `choice` question per request, before generation, and blends the answer into the start-rung
bandit as a Beta prior. **The gate still verifies every output, unchanged.** The prior only ever
moves where the ladder starts.

```toml
[escalation.bandit]
min_observations = 50

[escalation.prior]
endpoint   = "https://api.typesafe.ai/v1/systemone"   # default; override for self-hosted/mock Jev
model      = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"                      # env var name only — the key itself is never logged
timeout_ms = 150                                       # fail-open below
strength   = 20                                        # Beta pseudo-count weight of the prior
rubric = [
  "rung 0 (claude-haiku) is the least capable tier that fully handles this request",
  "rung 1 (claude-sonnet) is the least capable tier that fully handles this request",
  "rung 2 (claude-opus) is the least capable tier that fully handles this request",
]
```

`rubric` has one line per ladder rung (`r0..rN`); `Config::parse` rejects a `rubric.len()` that
does not match the ladder length, the same class of check as the existing per-rung price
requirement.

**The math.** The `choice` question's options are the rubric lines, each phrased as "rung `r` is
the *least capable* tier that fully handles this" — a partition over "which rung is the cheapest
sufficient one", not an independent pass/fail per rung. Jev returns `P(option r)` for each `r`.
Because the options partition the same event space,

```
P(pass at rung r) = Σ_{i=0..r} P(option i)
```

— a cumulative sum of non-negative probabilities, **monotone in `r` by construction**. No
post-hoc isotonic fix-up is needed to guarantee "a higher rung is never estimated less likely to
pass than a lower one."

**The blend.** For each rung, `P(pass at rung r)` from Jev becomes a Beta(`strength · p_r`,
`strength · (1 − p_r)`) pseudo-observation added to that context's real Beta posterior (`bandit@v2-ts`)
or count (`bandit@v1`) before sampling/selection. `strength` pseudo-counts is a fixed weight; real
gate verdicts are exact counts, so as they accumulate they dominate the fixed-size prior
automatically — the prior recedes, it is never pinned. This removes the cold-start cliff described
in `BanditConfig::min_observations`: today, a context with zero history starts at rung 0
unconditionally; with a prior it starts at the per-query estimate instead, without waiting for
`min_observations` verdicts that a rarely-seen context may never accumulate.

**Feeds the existing objective.** The blended `P(pass at rung r)` values are handed to the same
expected-cost argmin the bandit already runs (ADR 0007) — the objective doesn't change, only the
estimator that feeds it gains a per-query term. This is the same shape of change ADR 0007 made
going from UCB1 to Thompson: estimation changes, the decision rule does not.

**Policy stamp.** `bandit@v3-prior` — a new stamp distinct from `bandit@v1` (UCB1) and
`bandit@v2-ts` (Thompson), recorded on every decision's `PolicyRef.id` so a receipt or an
off-policy evaluation can tell which estimator produced the start rung.

## Invariants preserved

- **A failed rung is never served.** The prior chooses where the ladder *starts*; the gate decides
  what ships, exactly as in every prior ADR in this chain (0007, 0008, 0012). Jev's non-determinism
  — the same query can get a different prior on a re-ask — therefore affects **only cost** (which
  rung the ladder opens on), never correctness. If the prior is wrong, the ladder escalates the same
  way it always does on a gate failure.
- **Fail-open.** Timeout (default 150ms), a network error, a non-2xx response, or a response that
  doesn't parse as the expected `choice` shape all fall back to today's behavior — the bandit's own
  posterior with no prior term — and increment `firstpass_prior_errors_total`. A slow or broken Jev
  degrades to "the bandit as it already worked," never to a stall or a wrong serve.
- **No lock-in.** Deleting `[escalation.prior]` is byte-identical to pre-ADR-0013 behavior: no
  outbound call, no blended term, `bandit@v2-ts`/`bandit@v1` unchanged. There is no state written
  anywhere that depends on the prior having run.
- **Hash chain re-derivable.** The receipt gains one new optional field, `decision_prior` (the raw
  per-rung probability vector Jev returned), on the same struct as `predicted_pass` (ADR 0008) and
  `mode_profile`, with `#[serde(default, skip_serializing_if = "Option::is_none")]`. Absent when
  the block is unconfigured, so old traces and traces from deployments that never enable this
  reserialize byte-for-byte, and the hash chain re-derives exactly as before.
- **No `FEATURE_VERSION` bump.** `FEATURE_VERSION` (`crates/firstpass-core/src/features.rs`) governs
  the deterministic feature vector used to key bandit contexts (`ContextBucket`) — it bumps when
  *how a context is computed* changes, because mixing vectors computed two different ways under one
  key is the exact failure ADR 0012 bumped it to prevent. `decision_prior` does not touch
  `Features` or `ContextBucket`: it is parallel telemetry recorded alongside a decision, the same
  category as `predicted_pass` and `ElasticDecision` (ADR 0008), neither of which bumped it either.
- **The API key is never logged or recorded.** `[escalation.prior].api_key_env` names an
  environment variable, matching every existing `[[provider]]` block's `api_key_env`; the raw key
  is read once at startup and never appears in a receipt, a log line, or an error message. A prior
  fetch failure logs the HTTP status and elapsed time, not headers or body.

## Alternatives considered

- **(a) Serve the predicted tier unverified** — what `jev-router` / `prismhq/jev-router` do today.
  Rejected: it discards the served-failure guarantee that is the whole reason Firstpass exists, in
  exchange for a call whose own vendor publishes no accuracy numbers.
- **(b) Promote the existing shadow `PassPredictor`** (ADR 0008 Phase 2 — an online logistic
  regression trained from receipts). Complementary, not competing: it improves as traffic
  accumulates and has no cold-start gap of its own to fix, but it has *no* signal on a context it
  has never seen, which is precisely where a pre-generation prior helps most. Both can run; the
  prior fills the gap the learned predictor cannot by definition close on day one.
- **(c) Use Jev as a gate** (a `noul` judge, at roughly $0.042/M input against frontier
  judge pricing). Deferred, not rejected: gates are safety-critical and this repo's own measurements
  (`docs/related-work.md`'s open caveat) show judge precision/recall has to be measured before it
  is trusted with a serve/escalate decision — using an unbenchmarked decision model as the thing
  that decides pass/fail is a materially bigger bet than using it to pick a start rung a failing
  gate can still overrule. Next batch, after the prior itself clears its own bar below.

## Consequences

**This ships as a design decision, not a validated win.** Two gates stand between "accepted" and
"default-on," in order:

1. **Simulation, pre-registered.** A σ-sweep in `firstpass-bench` over synthetic prior noise
   (Jev's own accuracy is unpublished, so the sweep spans plausible calibration error rather than
   assuming a value): `firstpass+prior` must beat plain `firstpass` on **$/success** at `σ ≤ 0.2`
   **without raising served-failure**, or the pre-registered kill criterion fires — `PRIOR = STOP`,
   written up as a negative result exactly as ADR 0012's trajectory-hint was, and the block stays
   default-off.
2. **Live A/B, if simulation clears.** MBPP, haiku→sonnet ladder, prior on vs off: an n=20 pilot to
   catch integration bugs cheaply, then n≥135 for a properly powered paired-bootstrap comparison on
   $/success with served-failure held at parity, following the same measurement discipline as ADR
   0007's Thompson-vs-UCB1 promotion gate and ADR 0012's MBPP/SWE-bench kill runs.

Until both run, every number in this ADR is a design argument, not a result — **UNVERIFIED**. If
either gate fails, `[escalation.prior]` follows ADR 0012's precedent: the plumbing (config,
blending math, `decision_prior` receipt field) stays because it costs nothing when unconfigured,
and the claim that it pays gets withdrawn in an addendum rather than silently dropped.

Related: ADR 0007 (the start-rung bandit and `bandit@v2-ts` this extends), ADR 0008 (`PassPredictor`
shadow, the complementary learned signal), ADR 0012 (the measurement discipline and addendum format
this ADR commits to following), `docs/related-work.md` (where Jev-based routers sit in the field).
