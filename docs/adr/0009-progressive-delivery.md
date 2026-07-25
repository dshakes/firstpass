# ADR 0009 — Progressive delivery: shadow scoring, percentage rollout, and the guardrail

- Status: Proposed — design only; no code written against it yet
- Date: 2026-07-25
- Related: ADR 0003 (GA readiness), ADR 0008 (elastic verification), SPEC §10.1
  (conformal serving threshold), `docs/runbooks/soak.md`

## Context

Firstpass asks an operator to let it change what their users are served. The
product's answer to "why should I trust that" is currently: run it in observe
mode, read the receipts, then flip to enforce. That answer has three holes, and
they are the same hole seen from three angles — **there is no safe path between
"watching" and "serving".**

1. **Observe does not actually score anything.** `observe_passthrough` forwards
   the request upstream and records a trace. It never runs the ladder. So a
   session in observe tells the operator what their traffic looked like, not
   what Firstpass would have decided, what it would have served, or what it
   would have cost. The one question observe exists to answer is the one it
   cannot answer. `deferred_gates` scores the *upstream's* answer after the
   fact, which is a different question again.

2. **Enforce is all-or-nothing.** `Route.match_` slices traffic by request
   *features* (task kind, tenant, repo fingerprint). There is no way to say
   "5% of otherwise-identical traffic". The only ramp available is to hand-craft
   a matcher that happens to select a small slice, which correlates the
   experiment with whatever that matcher keys on and destroys the comparison.

3. **Nothing watches the thing being promised.** `/metrics` exposes counters and
   `GateHealthRegistry` trips individual gates on their own error budget, but no
   component watches the *served-failure rate* — the quantity the product's
   headline guarantee is about — and reacts when it degrades. The guarantee is
   calibrated, then trusted indefinitely.

The through-line: Firstpass can compute a distribution-free bound on served
failures, and that is genuinely its differentiator, but it has no mechanism to
**earn** that bound incrementally on a specific deployment's traffic. Every
operator's first enforce request is a step off a cliff.

## Decision

Three additive mechanisms, each independently useful, which compose into one
adoption path: **shadow → ramp → guard**.

All three are **default-off**. A config that does not mention them behaves
byte-identically to today. This is non-negotiable: these features exist to make
adoption safer, and a feature that changes behavior on upgrade would do the
opposite.

---

### D1 — Percentage rollout with stable bucketing

Add an optional `rollout` to a route:

```toml
[[route]]
match  = {}
mode   = "enforce"
ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
gates  = ["json-valid"]

[route.rollout]
percent = 5.0          # of matched traffic actually enforced; the rest observes
key     = "session"    # session | request | tenant
```

A matched request is enforced iff it falls in the bucket; otherwise it takes the
existing observe passthrough. The route's `mode` still gates whether enforcement
is possible at all — `rollout` only ever *subtracts*.

**Bucketing must be stable, and this is the load-bearing decision.** The obvious
implementation — draw a random number per request — is wrong in a way that is
easy to miss:

- A multi-turn agent conversation would flip between enforced and observed
  mid-thread. The user sees a model change mid-task for no reason they can
  perceive, and the operator sees an incoherent transcript.
- Worse, it silently corrupts the guarantee. The conformal bound is computed
  over *the population that was served*. If membership in that population is
  re-drawn per request, the population is not a stable sample of anything, and
  the bound describes a cohort that never existed.

So the bucket is a deterministic function of a **stable key**:

```
bucket = first_4_bytes_of( sha256( prompt_salt || key_kind || key_value ) ) % 10_000
enforced = bucket < percent * 100
```

- `session` (default) — the `x-firstpass-session` header. The correct default:
  a conversation is the unit a user experiences, so it is the unit that should
  be held constant. Falls back to `request` when the header is absent.
- `request` — hashes the request's own identity. For stateless single-shot
  traffic where no session exists.
- `tenant` — for hosted deployments ramping tenant-by-tenant.

Reusing `prompt_salt` means the assignment is deterministic across restarts and
across replicas with the same config, needs no shared state, and leaks nothing:
it is the same salted-hash discipline `repo_fingerprint` already uses.

**The receipt records the decision, not just the outcome.** Every trace gains:

```json
"rollout": { "percent": 5.0, "key": "session", "bucket": 731, "enforced": true }
```

Without this, a mixed-population receipt log cannot be split back into "the
enforced arm" and "the observed arm", and every savings or failure number
computed over it is a blend of two different regimes. With it, an auditor can
re-derive arm membership from the receipt alone — the same standard the hash
chain already meets.

---

### D2 — Shadow scoring

Add an optional `shadow` to a route, valid in observe mode:

```toml
[route.shadow]
sample_rate = 0.10        # fraction of observed requests scored counterfactually
max_usd_per_day = 5.00    # hard ceiling; shadow work stops when exhausted
```

For a sampled request: serve the upstream response **verbatim, as today**, and
*afterwards, off the hot path*, run the ladder and gates against the same input
to record what Firstpass would have done.

The receipt gains a shadow block:

```json
"shadow": {
  "would_serve_rung": 0,
  "would_pass": true,
  "projected_cost_usd": 0.0009,
  "actual_cost_usd": 0.0031,
  "gates": [{ "id": "json-valid", "verdict": "pass" }]
}
```

That turns observe from "we recorded your traffic" into **"on your traffic, the
cheap rung would have cleared your gate 82% of the time and cost 71% less"** —
which is the number an operator actually needs before flipping, and the number
the landing page currently asks them to take on faith.

Non-negotiable properties:

- **It never touches the response.** The served bytes, headers, and timing are
  byte-identical to observe today. Shadow work is spawned detached, after the
  response is handed back.
- **It costs real money and must say so.** Shadow makes genuine model calls at
  `sample_rate`. Hence an explicit sample rate with no default-on value, and a
  hard daily ceiling. Exceeding the ceiling stops shadow work and records that
  it was stopped — silently degrading a measurement is worse than not taking it.
- **Its failures are invisible to the caller.** A shadow error is recorded on the
  trace and never propagates. Shadow must not be able to take down a request path
  that would otherwise have succeeded.
- **It is excluded from the guarantee.** Shadow verdicts describe a counterfactual
  that was never served, so they must never enter the conformal calibration set
  for served failures. They are a separate estimate with their own sample size.

---

### D3 — The guardrail

Add an optional guardrail, per route or global:

```toml
[guardrail]
alpha    = 0.10      # the served-failure target being defended
window   = 500       # sliding window of served, feedback-resolved decisions
min_n    = 100       # never act on fewer than this
action   = "demote"  # demote | alarm
cooldown = 3600      # seconds before an automatic re-promote may be considered
```

The guardrail watches the **lower confidence bound** on served failures over the
trailing window, using the same Hoeffding machinery in `firstpass-core` that
produces the published bound. When the bound exceeds `alpha` with at least
`min_n` resolved samples, it acts:

- `demote` — the route reverts to observe. Traffic is served exactly as it would
  have been without Firstpass. This is the safe direction: the failure mode of
  the guardrail is "you stop saving money", never "you serve worse answers".
- `alarm` — emit the event and a metric, change nothing. For operators who want
  a human in the loop.

Design constraints that matter more than the feature itself:

- **It acts on resolved outcomes, not gate verdicts alone.** A gate verdict is
  Firstpass's own opinion; the guarantee is about real downstream outcomes, which
  arrive via `/v1/feedback`. Tripping on self-reported verdicts would let the
  system grade its own homework.
- **Minimum sample size is mandatory.** Without `min_n`, a route trips on the
  first two failures of the day. The bound is only meaningful with enough
  resolved samples, and acting early is how a guardrail becomes noise an operator
  disables.
- **Demotion is sticky and cooldown-gated.** A guardrail that re-promotes as soon
  as the window recovers oscillates, and each oscillation is a visible behavior
  change for users. Re-promotion is deliberately conservative.
- **Tripping is an audit event.** It goes into the receipt stream with the window,
  the bound, and the sample size, because "why did this route stop enforcing at
  0300" must be answerable from the record alone.

---

## Invariants that must not regress

1. **Observe never changes served bytes.** Shadow work is strictly post-response
   and detached.
2. **The hash chain stays re-derivable** by an external auditor from stored
   records alone. New fields are inside hashed bodies; nothing is mutated after
   sealing.
3. **Arm membership is reconstructible from the receipt.** Given a trace, an
   auditor can recompute the bucket and confirm which arm it belonged to.
4. **Default-off.** A config not mentioning `rollout`, `shadow`, or `guardrail`
   produces byte-identical behavior to the previous release.
5. **The guarantee is computed per population.** Enforced, observed, and shadow
   traces are never pooled into one bound.
6. **Offboarding is still one env var.**

## Phasing

| Batch | Scope | Gate to proceed |
| --- | --- | --- |
| 1 | D1 bucketing, receipt field, dispatch | Bucketing proven deterministic, uniform, and stable across restarts |
| 2 | D2 shadow scoring, cost ceiling | Byte-identical passthrough proven under shadow load |
| 3 | D3 guardrail, demotion, audit event | Trip and no-trip both proven on synthetic outcome streams |
| 4 | Config reference, guide, site, CLI/MCP surfaces | Docs match behavior; generated presets still pinned |

Batches 1–3 each land default-off and independently revertable.

## Alternatives considered

- **Random per-request bucketing.** Rejected: breaks multi-turn coherence and
  makes the served population unstable, which is precisely the thing the bound
  is computed over.
- **Shadow inside the request path.** Rejected: doubles latency on every sampled
  request and puts a measurement in a position to fail a user's call.
- **Guardrail on gate verdicts only.** Rejected: the system would be grading its
  own homework. Real outcomes arrive via feedback and that is what the guarantee
  is about.
- **Reusing `escalation.probe.sample_rate` for shadow.** Rejected: probe samples
  *within* an enforced decision to improve routing; shadow evaluates a
  counterfactual for an *observed* one. Same word, different populations —
  conflating them would corrupt both.

## Open questions

- Should `rollout.percent` be ramp-able on a schedule (`5 → 25 → 50` over days),
  or is that an operator's deployment-tooling concern? Leaning operator's, since
  encoding time in routing config invites clock-skew bugs across replicas.
- Should the guardrail's demotion survive a restart? It currently would not.
  Persisting it means a demoted route stays demoted through a deploy, which is
  probably right, but it puts operational state in the trace store.
