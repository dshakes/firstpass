<div align="center">

<img src="assets/hero.svg" alt="Firstpass — the cheapest model takes the first pass, proven not guessed" width="840">

# Firstpass

**Route every LLM request to the cheapest model that _provably_ passes your quality gate — and get a signed receipt for the decision.**

Proof over prediction. Built for agent fleets.

[![CI](https://github.com/dshakes/firstpass/actions/workflows/ci.yml/badge.svg)](https://github.com/dshakes/firstpass/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![spec](https://img.shields.io/badge/spec-v0.1-informational)](SPEC.md)
[![site](https://img.shields.io/badge/site-live-19E3B1)](https://dshakes.github.io/firstpass)

**[Website](https://dshakes.github.io/firstpass)** · [Quickstart](#quickstart) · [Install](#install) · [How it works](#how-it-works) · [Configure](#configuration) · [SPEC](SPEC.md)

</div>

---

Firstpass is a **drop-in, Anthropic-compatible proxy**. Point your agent's `base_url` at it and every request is routed to the cheapest model first, its **real output** is checked by a gate you define (tests, typecheck, schema, a judge), and it escalates one rung only when the gate fails — writing a tamper-evident audit trace for every decision.

> **Honestly scoped.** The proxy routes, gates, escalates, fails over, audits, and learns end-to-end over real HTTP — no test doubles in the plane, and the enforce path is [live-verified](#proof-not-adjectives) against real Anthropic. The [proof harness](#proof-not-adjectives) has a **200-task live result** below; still ahead: a coding-with-tests benchmark (to earn the conformal guarantee), a 30-day dogfood, and prebuilt binaries. See the [roadmap](#roadmap). Nothing here is claimed as measured that isn't.

## Quickstart

See the whole loop in ~10 seconds — **no API keys**:

```bash
git clone https://github.com/dshakes/firstpass && cd firstpass
cargo run -p firstpass-proxy --example demo
```

It stands up a mock upstream and the proxy, drives one real decision (cheap model fails the gate → escalates → passes), prints the [receipt](#the-receipt), submits feedback, and re-verifies the sealed chain.

**Then use it in front of your own agent:**

```bash
firstpass-proxy                                    # 1. run it (observe mode: logs, changes nothing)
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"  # 2. point your agent at it — same wire format
#   … use your agent normally; Firstpass records a receipt per call …
unset ANTHROPIC_BASE_URL                            # 3. offboard anytime — one env var
```

To turn on cheapest-first routing + gating, copy [`firstpass.example.toml`](firstpass.example.toml) and run in enforce mode — see [Configuration](#configuration).

## Install

| Method | Command |
| --- | --- |
| **curl \| sh** | `curl --proto '=https' --tlsv1.2 -LsSf https://github.com/dshakes/firstpass/releases/latest/download/firstpass-proxy-installer.sh \| sh` |
| **Docker** | `docker run -p 8080:8080 -e FIRSTPASS_BIND=0.0.0.0:8080 ghcr.io/dshakes/firstpass:latest` |
| **From source** | `cargo install --git https://github.com/dshakes/firstpass firstpass-proxy` |

> The `curl \| sh` installer and prebuilt binaries (macOS · Linux · Windows, all checksummed) are produced by [cargo-dist](https://opensource.axo.dev/cargo-dist/) and attach to each [GitHub Release](https://github.com/dshakes/firstpass/releases). The container image is published to [GHCR](https://github.com/dshakes/firstpass/pkgs/container/firstpass) on every push to `main`. Homebrew is planned (needs a formula tap). `cargo install --git` and Docker work today.

## Prediction vs. proof

<div align="center"><img src="assets/compare.svg" alt="Prediction guesses a model up front and never checks; Firstpass sends cheap, gates the real output, escalates only on failure, and emits a receipt" width="880"></div>

Model routers on the market route by **prediction** — a learned policy guesses which model will answer well, sends the request there, and never checks. Firstpass routes by **proof**: it runs a real gate on the real output, and the decision is a record you can audit — not a guess you hope was right.

## How it works

<div align="center"><img src="assets/how.svg" alt="A request hits the cheapest rung, fails the gate, escalates one rung, passes, and is served — every decision logged to the audit trace" width="840"></div>

1. **Route** to the cheapest rung of a declarative ladder. BYOK — your keys pass through, redacted from every log.
2. **Gate** the real output: inline (non-empty, JSON, [JSON-Schema](SPEC.md)) or subprocess plugins (your tests/linter/judge) that read the candidate on **stdin, never argv** — injection-resistant. Per-gate error budgets auto-disable a flaky gate.
3. **Escalate** exactly one rung on gate failure — budget-capped; a provider 5xx fails over cross-provider. Serve the first output that passes, and record everything.

## The receipt

Not a dashboard number — a decision you can audit. Every call becomes a hash-chained JSON trace an external auditor can re-derive.

<div align="center"><img src="assets/receipt.svg" alt="Firstpass audit receipt: the cheap model failed the test gate, the next rung passed, served at a real saving versus always-top-tier" width="620"></div>

```jsonc
{
  "trace_id": "0192f3a1-7c4e-7abc-9d21-4e8b1f0a2c33",
  "prev_hash": "9f2c…a1b7",                         // chains to the previous decision — tamper-evident
  "mode": "enforce",
  "attempts": [
    { "rung": 0, "model": "anthropic/claude-haiku-4-5", "cost_usd": 0.0007,
      "gates": [{ "gate_id": "cargo-test", "verdict": "fail", "score": 0.0 }],
      "verdict": "fail" },                           // cheap model tried first — the test gate caught it
    { "rung": 1, "model": "anthropic/claude-sonnet-5", "cost_usd": 0.0121,
      "gates": [{ "gate_id": "cargo-test", "verdict": "pass", "score": 1.0 }],
      "verdict": "pass" }                            // escalated one rung, proven to pass, served
  ],
  "final": { "served_rung": 1, "total_cost_usd": 0.0128,
             "counterfactual_baseline_usd": 0.0630, "savings_usd": 0.0502 }
}
```

Point at any request and answer *why did this go to that model, and what did it cost* — and re-derive the hash chain yourself. Downstream outcomes flow back via [`POST /v1/feedback`](SPEC.md) onto a deferred-verdict side table that never alters the sealed record.

## Configuration

Firstpass is configured through the environment (12-factor) — run `firstpass-proxy --help` for the full reference:

| Variable | Purpose | Default |
| --- | --- | --- |
| `FIRSTPASS_MODE` | `observe` \| `enforce` | `observe` |
| `FIRSTPASS_BIND` | listen address | `127.0.0.1:8080` |
| `FIRSTPASS_CONFIG` | path to `firstpass.toml` (routes, ladders, gates) | — |
| `FIRSTPASS_DB` | trace store path | `firstpass.db` |
| `FIRSTPASS_UPSTREAM_ANTHROPIC` | upstream base URL | `https://api.anthropic.com` |
| `FIRSTPASS_UPSTREAM_OPENAI` | upstream base URL | `https://api.openai.com` |

Routing itself is declarative TOML — routes → mode, a cheapest-first model ladder, and the gates that must pass. Start from [`firstpass.example.toml`](firstpass.example.toml):

```bash
cp firstpass.example.toml firstpass.toml
FIRSTPASS_MODE=enforce FIRSTPASS_CONFIG=./firstpass.toml firstpass-proxy
```

**Endpoints:** `POST /v1/messages` (drop-in) · `POST /v1/feedback` · `GET /v1/capabilities` · `GET /healthz`.

## Proof, not adjectives

<div align="center"><img src="assets/proof.svg" alt="Proof-harness simulation: ~65% cheaper at equal-or-higher success, 0.16 vs 0.46 served-failure versus a predictive router, ≤10% conformal served-failure at 95% confidence, kill criterion reads PROCEED" width="880"></div>

The harness (`cargo run -p firstpass-bench`) runs the full methodology — baselines, bootstrap confidence intervals, a split-conformal served-failure guarantee, and a **pre-registered kill criterion** that says *stop* if the thesis fails — behind real-backend trait seams. `--live` swaps the simulated backend for real providers (BYOK).

**Live — real Anthropic, 200 graded verifiable tasks** (`--live`, difficulty-graded arithmetic; reproducible, deterministic):

| policy | success | $/success | served-failure |
|---|---|---|---|
| always-cheap (Haiku) | 0.62 `[0.55, 0.69]` | $0.0001 | 0.38 |
| always-top (Opus) | 0.98 `[0.96, 0.99]` | $0.0023 | 0.02 |
| predictive router | 0.88 | $0.0007 | 0.12 |
| **firstpass** | **1.00** `[1.00, 1.00]` | **$0.0003** | **0.00** |

Firstpass served at **~85% lower $/success than always-top, at parity-or-better quality**, with **served-failure 0.00 vs the predictive router's 0.12** — verification catches what prediction serves blind. The cheap tier cleared 62% of tasks against a 13% break-even, so **the pre-registered kill criterion reads PROCEED**.

**Reproduced with a real LLM-judge gate, not just a deterministic checker** (`FIRSTPASS_GATE=judge`, Sonnet grading each answer *without seeing the ground truth*, n=200): firstpass **1.00 success, ~84% cheaper than Opus, served-failure 0.00, PROCEED** — the win holds with the gate you'd actually deploy.

Stated plainly on the one thing this did **not** show at the time: the **conformal served-failure guarantee stayed degenerate** on self-checking tasks — the gate is either near-perfect (nothing to bound) or, if you weaken the judge, it false-rejects correct answers and the cost advantage flips. Earning the guarantee needed a domain where the *best practical* gate is still imperfect — a **coding-with-tests benchmark** (real test suites have coverage gaps), which needed a sandbox for untrusted code. **Both are now built**: a fail-closed code-execution sandbox ([ADR 0002](docs/adr/0002-bench-code-execution-sandbox.md) — gVisor `runsc` first, `runc` fallback with a warning, `--network none`, capability drops, no host mounts) and a coding-with-tests benchmark with a **continuous gate score**, so conformal calibrates on `(gate_score, oracle_correct)` with real error instead of a zero-error self-checker. A live-provider run of that benchmark hasn't been reported here yet.

## Roadmap

- **M0 ✓** — proof harness: baselines, bootstrap CIs, conformal guarantee, pre-registered kill criterion.
- **M1 ✓** — Rust proxy: Anthropic + OpenAI clients, observe **and** enforce, escalation, cross-provider failover, SQLite trace store — over real HTTP.
- **M2 ✓** — gate framework: subprocess plugins, inline + schema gates, **native LLM-judge gate** (maker≠checker, candidate-as-data), error-budget auto-disable, feedback API + deferred verdicts.
- **M2.5 ✓** — real-traffic proxy: **SSE streaming passthrough**, tool/multimodal-safe enforce; **`firstpass` CLI** (`up` / `doctor` / `trace`) + **MCP server**; live-provider proof harness (`--live`).
- **M3 ✓** — 200-task live benchmark (cost/success proof at scale) **and** a real LLM-judge gate reproduction — both done against real Anthropic. Only **published binaries + a Homebrew tap** remain from this milestone (the cargo-dist `release.yml` has never been run — see [`docs/runbooks/release.md`](docs/runbooks/release.md)).
- **Speculative escalation ✓** — cheap and next-rung run in parallel for latency, serving a byte-identical result to sequential escalation.
- **Sandbox + coding-with-tests conformal ✓** — the fail-closed code-execution sandbox and continuous-gate benchmark described above (ADR 0002).
- **Learning loop ✓** — `firstpass calibrate` recalibrates the serving threshold from real deferred feedback; conformal moved to `firstpass-core` so proxy and bench share it.
- **Prod hardening ✓** — HTTP client timeouts, a bounded trace channel with load-shedding, opaque error responses, an explicit request-body cap, sandbox shell-quoting; see [ADR 0003](docs/adr/0003-production-ga-readiness.md) for the full GA-readiness gap analysis.
- **Not yet done** — published binaries + Homebrew tap, an external security audit, SOC 2 (see [`docs/compliance/soc2-controls.md`](docs/compliance/soc2-controls.md) for the readiness map, not a claim), a hosted multi-tenant plane, and a real 30-day soak (procedure in [`docs/runbooks/soak.md`](docs/runbooks/soak.md); hasn't been run and closed out).
- **Hosted GA →** — multi-tenant control plane, BYOK KMS envelope encryption, sandboxed gate execution — designed in [ADR 0001](docs/adr/0001-hosted-ga-architecture.md), gated behind dogfood GA.

## Links

[Website](https://dshakes.github.io/firstpass) · [SPEC](SPEC.md) · [Example config](firstpass.example.toml) · [Agent guide](AGENTS.md) · [llms.txt](llms.txt) · [License](LICENSE)

<div align="center"><sub>Proof over prediction. No competitor products named — Firstpass competes on evidence, not adjectives.</sub></div>
