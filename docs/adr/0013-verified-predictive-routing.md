# ADR 0013 — Verified predictive routing: a decision-model prior on the start rung, never on what is served

Status: **accepted — prior: PROCEED on real-data replay (OpenJev); prior+learned blend: NEUTRAL;
decision gate: NOT-RECOMMENDED (OpenJev); hosted-Jev unmeasured** · 2026-09-25

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

Add an optional `[escalation.prior]` config block. When set, the router asks Jev
one `choice` question per request, before generation, and blends the answer into the start-rung
bandit as a Beta prior. **The gate still verifies every output, unchanged.** The prior only ever
moves where the ladder starts.

```toml
[escalation.prior]
provider   = "typesafe"                                # the only accepted value today
base_url   = "https://api.typesafe.ai"                 # default; override for self-hosted/mock Jev
model      = "jev-latest"
api_key_env = "TYPESAFE_API_KEY"                      # env var name only — the key itself is never logged
timeout_ms = 150                                       # fail-open below
strength   = 10                                        # Beta pseudo-count weight of the prior
rungs = [
  "rung 0 (claude-haiku) is the least capable tier that fully handles this request",
  "rung 1 (claude-sonnet) is the least capable tier that fully handles this request",
  "rung 2 (claude-opus) is the least capable tier that fully handles this request",
]
```

`rungs` has one rubric line per ladder rung (`r0..rN`); `Config::parse` rejects a `rungs.len()` that
does not match any enforce route's ladder length, the same class of check as the existing per-rung price
requirement.

**The math.** The `choice` question's options are the `rungs` rubric lines, each phrased as "rung `r` is
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

## Addendum 2026-09-25 — simulation gate: PRIOR=STOP on the default suite

The pre-registered σ-sweep ran (`cargo run -p firstpass-bench`, n=500, seed 20260708, **simulation**):

| σ | firstpass $/success | firstpass+prior $/success | predictive-prior (unverified) served-fail |
|---|---|---|---|
| 0.00 | 0.0469 | 0.0469 | 0.562 |
| 0.10 | 0.0469 | 0.0472 | 0.556 |
| 0.20 | 0.0469 | 0.0490 | 0.550 |
| 0.40 | 0.0469 | 0.0534 | 0.506 |

The kill criterion required *strictly* lower $/success at σ ≤ 0.2; the fused arm ties at σ=0 and
loses beyond it. **Verdict: PRIOR=STOP on this suite.** The reason is structural, not noise: the
sim's rung-0 clearance is `strength − 0.45·difficulty ≥ ~0.10` at max difficulty, above the
cheap/next price ratio, so the expected-cost rule never skips rung 0 even with a perfect prior —
there is no gap for a prior to close. The same table is the strongest evidence yet for the
*verification* half: the unverified Jev-style arm serves a wrong answer on 51–56% of traffic at
every σ, while the gated arms hold their served-failure rate.

What this does and does not settle. It does not refute the prior where the real workload differs
from the sim — the live MBPP measurement in `costaware.rs` (n=135) found escalation adversely
selected 2.16x and a per-query P(pass) start rule 22% cheaper than first-pass. **[RETRACTED
2026-09-25 — see the hindsight-leak correction addendum below.]** That 22% figure came from the
same realized-cost arm now labeled `learned-p (hindsight)`; it decides and buckets on each task's
own post-generation cost, which is optimism, not signal a router has before generating. It does
mean the feature stays **default-off and opt-in**, and is not claimed as a win, until the live A/B
gate below passes. If it fails live, the ADR 0012 precedent applies: remove the acting code, keep
the receipt field.

The prior also works without `[escalation.bandit]`: the proxy then scores it against a
zero-observation bandit per request, so the start rung is decided by the prior alone.

### Exploratory follow-up (post-hoc, not a gate): adverse-selection workload

Same seed, every 10th task pushed into a hard tail (difficulty +2.0, past rung 0's clearance floor)
with tokens scaled by the 2.165x multiplier `costaware.rs` measured on real MBPP. The prior now does
act — escalations fall 0.91 → 0.74 at σ=0 — but $/success is 0.0736 vs 0.0738 (inside each other's
bootstrap CIs) and served-failure edges up 0.240 → 0.244; at σ ≥ 0.2 the fused arm is worse. A wash,
not a win. **PRIOR=STOP stands**; only the live A/B can change it. The unverified Jev-style arm again
serves a wrong answer on 55–60% of traffic.

## Addendum 2026-09-25 — real-data replay: PROCEED

The live A/B gate ran — not against hosted Jev, but against **OpenJev** (Apache-2.0, DiffusionGemma
26B-A4B), served locally via `POST /v1/systemone`. Pre-registration: `specs/openjev-prior-replay-ab.md`.
Full tables: `docs/benchmarks/openjev-prior-replay.md`.

Three real MBPP outcome matrices (already-recorded model+gate results, no new generation), scored
with a prior fetched once from OpenJev and replayed deterministically:

- `fp-mbpp-974-sonnet-k5.jsonl` (haiku-4-5 → sonnet-5, n=974)
- `fp-mbpp-974-opus-k5.jsonl` (haiku-4-5 → opus-4-8, n=470)
- `fp-mbpp-974-openai.jsonl` (gpt-4.1-mini → gpt-5.5, n=974)

**Pooled (n=2418): $/success $0.01126 → $0.01075 (−4.6%)**, paired-bootstrap CI of the difference
`[-0.00074, -0.00031]` — excludes 0. Success 0.9214 → 0.9222; served-failure 0.0786 → 0.0778 (down,
not up). Degeneracy guard: 93.4% of prior-covered decisions share the mode start rung, below the 95%
threshold, so the run is not degenerate. **The pre-registered kill criterion passes: PROCEED.**

Per ladder, the win is real but ladder-dependent:

| ladder | first-pass $/success | prior $/success | Δ |
|---|---|---|---|
| haiku→sonnet (n=974) | $0.01697 | $0.01590 | −6.3% |
| haiku→opus (n=470) | $0.02003 | $0.01955 | −2.4% |
| gpt-4.1-mini→gpt-5.5 (n=974) | $0.00136 | $0.00136 | 0% |

The gpt-4.1-mini→gpt-5.5 ladder never moves: at that ladder's ~20x price ratio between rungs,
skipping the cheap rung never pays even when the prior is confident the cheap rung will fail — the
same "no gap for a prior to close" structure the simulation addendum above found, just realized in
price ratio instead of a synthetic clearance floor.

**Why the sim said STOP and real data says PROCEED.** The simulation's rung-0 clearance was
constructed so it never fell below the cheap/next price ratio, so the expected-cost rule never had a
reason to skip rung 0 even with a perfect prior. Real MBPP is not like that: there are tasks the
cheap model reliably fails, the prior can see them from the prompt alone, and skipping straight to
the next rung is cheaper than paying for a doomed cheap attempt plus the escalation. The sim wasn't
wrong about its own construction; its construction wasn't representative of this workload.

**Caveats that stand alongside the win:**

1. **OpenJev ≠ Jev.** This says nothing about hosted Jev's own accuracy — a different model, a
   different vendor, run locally rather than over TypeSafe's API. The `[escalation.prior]` block
   defaults still point at `https://api.typesafe.ai`; using OpenJev is a `base_url` override (below).
2. **The win is modest and ladder-dependent** (table above) — real, but not the >20% swings a
   headline number might suggest.
3. **A bench-only cost-aware learned-p estimator (cross-fitted on traffic) still beats the prior**:
   $0.00919 pooled vs. the prior's $0.01075. **[RETRACTED 2026-09-25 — see the hindsight-leak
   correction addendum below.]** $0.00919 is the realized-cost `learned-p (hindsight)` arm; it is
   optimistic by $0.00190/success. The honest ex-ante version is $0.01109 pooled — only ~1.5% under
   first-pass's $0.01126, not a beat of the prior. Blending the two was measured (Study A, below):
   the blend is **worse** than the prior alone, not better.
4. **The unverified Jev-router stand-in served a wrong answer on 18.16% of pooled traffic**, against
   7.78% for the gated `prior` arm. Verification is still the reason this is safe to ship — an
   unverified decision-model router more than doubles served-failure versus the gated version of the
   same signal.
5. **`always-top` beats `first-pass` outright on haiku→sonnet** ($0.01500 vs $0.01697) — adverse
   selection, a finding already known from `costaware.rs`'s live measurement, not new here.

**What changes as a result.** `[escalation.prior]` remains **default-off** — this replay used
OpenJev, not hosted Jev, so it is not evidence about the vendor the ADR was written against. But the
pre-registered kill criterion this ADR set for "does the prior mechanism itself pay for itself on
real outcomes" now reads **PROCEED**, reversing the simulation's PRIOR=STOP on that narrower question.
Two bugs were found and fixed getting here, both worth recording:

- The `decision` gate read `probability`/`p`/`value` from a `/v1/systemone` reply; the real wire
  field is `noul`. It would have abstained on every real answer. Fixed in `crates/firstpass-proxy/src/decision.rs`.
- Live, opt-in wire tests now exist and are not run by default:
  `OPENJEV_URL=http://127.0.0.1:8080 cargo test -p firstpass-proxy -- --ignored live_`.

The `decision` gate's own precision/recall against real failures is **still unmeasured** — this
replay scored the *prior*, not the gate — and stays labeled that way. **[UPDATE 2026-09-25 — the
gate itself was measured next; see the Study B addendum below: NOT-RECOMMENDED with OpenJev as the
verifier. Hosted Jev as a gate remains unmeasured.]**

## Addendum 2026-09-25 — correction: the cost-aware learned-p savings claim is withdrawn (hindsight leak)

Pre-registration: `specs/prior-blend-and-decision-gate.md`. `costaware::serve` (`costaware.rs:145`)
decides and buckets `PassPredictor` on the task's own **realized** `c0.cost_usd`/`c1.cost_usd` —
values that include output tokens that exist only after generation, and hard tasks produce longer
outputs. That arm is hindsight, not a signal a router has before generating, and its $/success is
optimistic by **$0.00190/success** (pooled) versus the honest ex-ante version. It is now labeled
`learned-p (hindsight)` in every report and must never be the comparison target.

This retracts, without deleting, two claims made earlier in this ADR (both annotated in place
above, per the ADR 0012 addendum convention):

- The "22% cheaper than first-pass" figure in the "What this does and does not settle" paragraph
  under the simulation addendum — came from the leaking arm.
- Caveat 3 above ("a bench-only cost-aware learned-p estimator ... still beats the prior: $0.00919
  pooled") — $0.00919 is the hindsight number. The honest ex-ante `learned-p (ex-ante)` is
  **$0.01109 pooled**, only ~1.5% under first-pass's $0.01126, not a beat of the prior's $0.01075.

## Addendum 2026-09-25 — Study A: prior+learned blend is BLEND-NEUTRAL

Pre-registration: `specs/prior-blend-and-decision-gate.md`. Full tables:
`docs/benchmarks/prior-blend-replay.md`.

Same three MBPP outcome matrices and committed OpenJev priors as the PROCEED replay above, plus a
cross-fitted ex-ante learned-p arm (bucketed on the MBPP prompt's character length — known before
generation — using the calibration fold's quartiles) and a `prior+learned` blend using the proxy's
own posterior-mean formula: `(s·prior_r0 + passes_b)/(s + seen_b)`, `s = 10` (the
`[escalation.prior] strength` default), `passes_b`/`seen_b` from the calibration-fold bucket.

Pooled (n=2418): `prior+learned` **$0.01094** vs `prior` **$0.01075**; paired bootstrap diff
**+0.00019 [+0.00004, +0.00036]** — excludes 0, the blend is worse. Honest ex-ante `learned-p
(ex-ante)` is $0.01109 pooled, only ~1.5% under first-pass's $0.01126.

**Verdict: BLEND-NEUTRAL. The prior alone stays the recommendation** — blending the learned signal
into the prior does not pay for itself on this data.

## Addendum 2026-09-25 — Study B: decision gate as a second gate is NOT-RECOMMENDED (OpenJev)

Pre-registration: `specs/prior-blend-and-decision-gate.md`. Full tables:
`docs/benchmarks/decision-gate-study.md`.

974 served MBPP answers (`~/vrb-cascade.jsonl`) that already passed the existing gate, labeled by
VRBench's hidden-test oracle in the fail-closed sandbox (111 oracle-wrong, 863 oracle-right), scored
by the proxy's `DecisionGate` request shape against **local OpenJev** at `http://127.0.0.1:8080` —
this measures OpenJev, not hosted Jev.

| metric | point | 95% CI |
|---|---|---|
| catch rate (τ=0.5) | 0.2162 | [0.1441, 0.2973] |
| collateral (τ=0.5) | 0.1031 | [0.0834, 0.1228] |
| AUC | 0.6310 | [0.5711, 0.6886] |
| abstain rate | 0.0000 | [0.0000, 0.0000] |

The pre-registered bar was catch rate ≥ 0.30 **and** collateral ≤ 0.05; both missed on point
estimates. **Verdict: NOT-RECOMMENDED** at τ=0.5 (an exploratory τ sweep in the full report cannot
change this verdict). The gate code stays available — this is a design decision, not a validated
win, per the ADR 0012 precedent — but the docs must not suggest pairing it with OpenJev as a
verifier. Hosted Jev as a gate remains unmeasured.
