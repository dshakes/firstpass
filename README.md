<div align="center">

<img src="assets/hero.svg" alt="Firstpass — the cheapest model takes the first pass, proven not guessed" width="840">

# Firstpass

### The cheapest model takes the first pass. It clears your gate — proven, not guessed — or the request escalates.

**Firstpass routes every LLM request to the cheapest model that _provably_ passes your quality gate, and hands you a signed receipt for the decision.**

Proof over prediction. Built for agent fleets.

[![status](https://img.shields.io/badge/status-building%20M1-f5a623)](SPEC.md)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![rust](https://img.shields.io/badge/rust-1.93%2B-orange)](rust-toolchain.toml)
[![spec](https://img.shields.io/badge/spec-v0.1-informational)](SPEC.md)

</div>

---

> **Status — founding stage, actively building.** The founding spec ([SPEC.md](SPEC.md)) is done and the
> domain contract (`firstpass-core`: trace schema, verdicts, tamper-evident hash chain, routing config,
> cost model) is landing now. The M1 proxy is next. This README describes the product being built and the
> DX we're building toward; nothing here claims measured results yet — that's what M0's benchmark proves.

## The one-paragraph pitch

Every model router on the market routes by **prediction** — a learned policy guesses which model will answer
well, sends the request there, and never checks. They all decide _before_ generation and ask you to trust the
guess. Firstpass routes by **proof**: it sends each
request to the cheapest plausible model, runs a real gate on the actual output — tests, typecheck, schema,
a fresh-context judge — and escalates one rung only when the gate fails. Every decision becomes a
tamper-evident trace you can audit, and every downstream outcome (did the tests pass an hour later?) flows
back to sharpen the gate. Prediction is a black box you have to trust. **Proof is a receipt you can read.**

## Why now

Agent fleets — coding agents, CI bots, review agents — send **everything** to a top-tier model because
nobody can tell, per request, when a cheaper one would have done the job. The price gap between tiers is
**10–30×**. The fraction of requests that actually _need_ the top tier is unknown — which is precisely the
problem. Meanwhile those same agents already run tests, typechecks, and lint on every step: a free,
ground-truth signal for "was this model good enough" that today just evaporates.

Firstpass turns that thrown-away signal into a routing decision. In agent and coding workloads, "correct" is
**checkable for free and objectively** — does it compile, do the tests pass, is the diff applicable, is the
tool-call schema-valid. That's the one place a verifier beats a predictor outright, because the predictor
can't see whether the code runs, and you can.

## The receipt

Every routed request emits an append-only, hash-chained trace. This is the artifact no predictive router
produces — the thing your security team, your finance team, and your incident review all actually want:

<div align="center">
<img src="assets/receipt.svg" alt="Firstpass audit receipt: haiku failed the test gate, sonnet passed, served at 80% savings versus always-top-tier" width="620">
</div>

And the same trace as JSON your tools can parse and re-derive:

```jsonc
{
  "trace_id": "0192f3a1-7c4e-7abc-9d21-4e8b1f0a2c33",
  "prev_hash": "9f2c…a1b7",                      // chains to the previous decision — tamper-evident
  "session_id": "agent-run-4417",
  "mode": "enforce",
  "request": { "task_kind": "code_edit", "language": "rust", "features": "features@v1" },
  "attempts": [
    { "rung": 0, "model": "anthropic/claude-haiku-4-5", "cost_usd": 0.0007,
      "gates": [ { "gate_id": "cargo-test", "verdict": "fail", "score": 0.0, "ms": 3100 } ],
      "verdict": "fail" },                        // cheap model tried first — the test gate caught it
    { "rung": 1, "model": "anthropic/claude-sonnet-5", "cost_usd": 0.0121,
      "gates": [ { "gate_id": "cargo-test", "verdict": "pass", "score": 1.0, "ms": 2950 } ],
      "verdict": "pass" }                         // escalated one rung, proven to pass, served
  ],
  "final": {
    "served_rung": 1, "served_from": "attempt",
    "total_cost_usd": 0.0128,
    "counterfactual_baseline_usd": 0.0630,        // what always-top-tier would have cost
    "savings_usd": 0.0502                         // 80% cheaper, at proven quality parity
  }
}
```

You can re-derive the hash chain yourself. You can point at any request and answer *why did this go to that
model, and what did it cost.* No router on the market can.

## How it works

<div align="center">
<img src="assets/flow.svg" alt="Routing flow: a request hits the cheapest rung, fails the gate, escalates one rung, passes, and is served — every decision logged to the audit trace" width="840">
</div>

1. **Send** the request to the cheapest plausible model tier.
2. **Gate** the output with a real check — tests, typecheck, schema, or a fresh-context judge (maker ≠ checker).
3. **Escalate** exactly one rung on gate failure; serve the first output that passes, budget-capped.
4. **Log** every decision as a tamper-evident trace; downstream outcomes feed back and sharpen the gate.

## Install & integrate — the target experience (M1)

> These are the one-liners we're building toward, shown so the DX is unambiguous. The proxy binary lands in M1;
> follow [SPEC.md §16](SPEC.md) for status.

A single static binary, no runtime dependencies, running in seconds — any of:

```bash
curl -LsSf https://firstpass.dev/install.sh | sh   # or: brew install firstpass/tap/firstpass
docker run -p 8080:8080 firstpass/firstpass         # or: cargo install firstpass
```

Then adopt it with **one environment variable** — no SDK, no code change:

```bash
firstpass up                                        # starts in observe mode with sane defaults
export ANTHROPIC_BASE_URL="http://localhost:8080"   # your agent already speaks this wire format
# ...run your agent exactly as before. Offboard anytime: unset ANTHROPIC_BASE_URL
```

**Multi-provider from day one:** first-class Anthropic, OpenAI, and Google clients, plus a generic
OpenAI-compatible client that covers any compatible endpoint — hosted aggregators, third-party inference
providers, and local runtimes alike. A model is just a `provider/model` line in your ladder — adding one never means a rebuild or a retrain.

## Plugs into anything that talks to an LLM

Firstpass is **wire-compatible** — it speaks the provider APIs verbatim, so whatever you already run plugs in
without a code change and unplugs the same way:

- **Coding agents & IDE extensions** — one `base_url` env var (`ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` / `GOOGLE_GEMINI_BASE_URL`).
- **Headless & CI agents, serverless** — drop a Firstpass sidecar in front of provider traffic.
- **In-process** — link the router as an embedded library instead of a network hop.
- **Agent-native** — an MCP server exposes traces, capabilities, and the feedback API as tools.
- **Humans & scripts** — a CLI (`firstpass up` / `doctor` / `trace`).

It exposes the **Anthropic Messages**, **OpenAI Chat + Responses**, and **Google Gemini** surfaces, and passes
through everything you send — **streaming (SSE), tool/function calling, multimodal, structured outputs** —
faithfully. Gates run on the assembled output; the wire contract your agent uses never changes. BYOK, zero markup.

## Proof over prediction — how Firstpass differs

| | Decides… | Checks the actual output? | Audit receipt | New model onboarding |
|---|---|---|---|---|
| **Predictive routers** | before generation (predict) | ❌ never | ❌ | retrain / re-eval per model |
| **Black-box orchestrators** | before generation (predict) | ❌ opaque | ❌ | proprietary |
| **Model gateways** | by price/uptime rules | ❌ | partial logs | n/a (passthrough) |
| **Firstpass** | **after** generation (verify) | ✅ **every request** | ✅ **tamper-evident** | **one line in the ladder, no retrain** |

## What's actually new — stated honestly

The cascade mechanism itself is **not** novel — "cheapest-first, verify, escalate" is well-established in the
model-cascade research literature. We won't pretend otherwise. What no paper and no product has put together,
and what Firstpass is built around, is three things:

1. **Tamper-evident audit** of every routing decision and gate verdict — the exact thing black-box
   orchestrators structurally cannot give a regulated buyer.
2. **Outcome-feedback calibration** — the downstream truth (did the tests pass later?) flows back through a
   feedback API and auto-tunes the gates. A static cascade becomes a self-improving one.
3. **Zero-retrain onboarding** — a new frontier model ships monthly; predictive routers all need retraining.
   Firstpass adds it as one line in the ladder and lets the gate decide.

Lead with those, or a reviewer who knows the literature rightly calls it a cascade with a nicer UI. We lead with those.

## Built agent-first

Firstpass's primary user is an agent, and every surface is designed for one — not a human clicking a dashboard.
Adoption is a one-env-var `base_url` swap on an OpenAI/Anthropic-wire-compatible proxy; config is declarative;
traces are structured JSON with a re-derivable hash chain; verdicts are typed; errors are structured, never prose.
A `GET /v1/capabilities` endpoint lets an agent learn the ladder, gates, and limits at runtime, and an optional
MCP server exposes traces, verdicts, and the feedback API as tools so an agent can inspect and correct its own
routing. Docs are agent-consumable by default ([`llms.txt`](llms.txt), [`AGENTS.md`](AGENTS.md)). Onboarding is
self-serve and programmatic; offboarding is one reversible env var. See [SPEC.md §0.2](SPEC.md).

## No lock-in, ever

Firstpass sits *below* your harness as a `base_url` — it never replaces your agent framework. If the proxy
is down or you don't like it, **unset one environment variable** and your harness talks to the providers
directly again. BYOK (bring your own keys); we never mark up tokens. The escape hatch is a front-page promise,
not fine print.

## Roadmap (see [SPEC.md §16](SPEC.md))

- **M0** — tier-clearance benchmark on real agentic coding tasks — the go/no-go gate on the whole bet.
- **M1** — Rust proxy MVP: Anthropic + OpenAI endpoints, static ladder, observe mode, SQLite trace store.
- **M2** — gate framework: plugin contract, reference gates, enforce mode, feedback API, error-budget auto-disable.
- **M3** — dogfood GA on real agent traffic (compass subagent fleet is customer #1).
- **M4 / M5** — learned routing (contextual bandit): shadow-eval → live, promotion gated on logged traces.
- **M6** — OSS launch.

## Non-goals (what we deliberately don't do)

Not a universal router (v0–v1 target workloads *where gates exist*). Not an inference provider or reseller
(BYOK, zero token markup). Not an eval platform, not an agent framework, not a day-one trained coordinator.
Firstpass does one thing: route to the cheapest model that provably passes, and prove it.

---

<div align="center">

**[Read the full spec →](SPEC.md)**

*Proof over prediction.*

</div>
