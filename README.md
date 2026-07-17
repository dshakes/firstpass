<div align="center">

<img src="assets/hero.svg" alt="Firstpass sends each request to the cheapest model first, proves the output passes your gate, and pays for a stronger model only when the cheap one fails — with a guaranteed ceiling on wrong answers served" width="880">

# Firstpass

**Try cheap. Prove it. Escalate only on failure.**

The adaptive LLM router that cuts your model spend without gambling on quality —
the only router with a **mathematical ceiling on wrong answers served**.

[![CI](https://github.com/dshakes/firstpass/actions/workflows/ci.yml/badge.svg)](https://github.com/dshakes/firstpass/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/firstpass-proxy)](https://crates.io/crates/firstpass-proxy)
[![PyPI](https://img.shields.io/pypi/v/firstpass)](https://pypi.org/project/firstpass/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**[Website](https://dshakes.github.io/firstpass)** · [Quickstart](#quickstart) · [Install](#install) · [How it works](#how-it-works) · [Providers](#every-provider-including-open-source) · [Proof](#the-numbers) · [Docs](https://dshakes.github.io/firstpass/usage.html)

</div>

---

Every AI product overpays: simple requests go to your most expensive model because a bad answer is worse than a big bill. Routers that promise savings **guess** which model to use from the prompt — and a wrong guess ships to your user unchecked.

Firstpass doesn't guess. It sends each request to the **cheapest model first**, runs a real check on the **actual output** — your unit tests, a schema, or an LLM judge — and escalates to a stronger model **only when the check fails**. The first answer that passes is served, and every decision leaves a signed, auditable receipt.

|  |  |
|---|---|
| 💸 **Genuinely cheaper** | Pay frontier prices only on proven need. Every trace records what always-top-tier *would* have cost — your savings are measured, not promised. |
| 🛡️ **Never trades accuracy** | A distribution-free guarantee: **≤10% wrong answers served at 95% confidence**, earned live on 964 real coding tasks. The gate is the floor — savings can't undercut it. |
| 🔄 **Tunes itself** | The serve threshold self-adjusts from live outcomes (online conformal). Set it in front of your traffic; it stays calibrated as your workload drifts. |
| 🔌 **Drop-in, walk-out** | One env var to onboard, one to leave. Speaks the Anthropic wire format your agent already uses. |
| 🌐 **Every provider** | Anthropic, OpenAI, Gemini, Bedrock, Vertex, Groq, DeepSeek, Together, OpenRouter, Azure — or open-source models on your own Ollama/vLLM box. Mix them in one ladder. |

## Quickstart

Watch a full routing decision in ~10 seconds — **no API keys**:

```bash
git clone https://github.com/dshakes/firstpass && cd firstpass
cargo run -p firstpass-proxy --example demo
```

The demo stands up a mock provider and drives one real decision — cheap model fails the gate → escalates → passes → served — then prints the [receipt](#the-receipt) and re-verifies the tamper-evident chain.

**In front of your own agent** (zero config, zero risk):

```bash
firstpass-proxy                                     # observe mode: watches, changes nothing
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"   # your agent now routes through firstpass
# … use your agent normally — firstpass records what it *would* save, per request
unset ANTHROPIC_BASE_URL                            # offboard anytime
```

When the numbers convince you, switch on routing: `cp firstpass.example.toml firstpass.toml`, then `FIRSTPASS_MODE=enforce FIRSTPASS_CONFIG=./firstpass.toml firstpass-proxy`.

## Install

All channels publish automatically on every release and stay in sync:

| | |
|---|---|
| **pip / uvx** | `pip install firstpass` · `uvx firstpass` |
| **Homebrew** | `brew install dshakes/tap/firstpass` |
| **npm** | `npx @dshakesnotbot/firstpass` |
| **Cargo** | `cargo install firstpass-proxy` |
| **Docker** | `docker run -p 8080:8080 -e FIRSTPASS_BIND=0.0.0.0:8080 ghcr.io/dshakes/firstpass:latest` |
| **curl \| sh** | `curl --proto '=https' --tlsv1.2 -LsSf https://github.com/dshakes/firstpass/releases/latest/download/firstpass-proxy-installer.sh \| sh` |
| **Binaries** | macOS · Linux · Windows, checksummed, on the [Release page](https://github.com/dshakes/firstpass/releases) — with a built-in self-updater (`firstpass-proxy-update`) |

## How it works

<div align="center"><img src="assets/how.svg" alt="A request hits the cheapest rung, fails the gate, escalates one rung, passes, and is served — every decision logged to the audit trace" width="840"></div>

1. **Route** — the request goes to the cheapest rung of your model ladder. BYOK; keys are redacted from every log.
2. **Gate** — the *real output* is checked: built-ins (`non-empty`, `json-valid`, JSON-Schema), your own command (tests, linter — candidate on **stdin, never argv**), or an LLM judge (maker ≠ checker, candidate treated as data).
3. **Escalate** — on failure, exactly one rung up, budget-capped, with cross-provider failover on a 5xx. The first output that passes is served.
4. **Learn** — outcomes flow back via `/v1/feedback`; the serve threshold recalibrates itself so the guarantee tracks your live traffic.

**Who decides a request needs the expensive model? The gate does — from the cheap model's actual answer.** Never a classifier guessing from the prompt. To change routing behavior, edit a gate; there is no policy model to retrain.

## "Do I have to write gates?"

No. Meet it where you are:

| Effort | You get |
|---|---|
| **None** — observe mode | Firstpass watches your traffic and reports what it *would* route and save. Nothing changes. |
| **One sentence** — judge gate | A second model grades each answer against your plain-English rubric. No code. |
| **Your existing tests** | The strongest gate: generated code ships only if your test suite actually passes. |

Flaky gates auto-disable on an error budget, so one bad check can't take down a route.

## Every provider, including open-source

`anthropic` and `openai` are built in. One `[[provider]]` block adds anything else — a ladder rung is `<id>/<model>`, so a route can open on a free local model and escalate to a frontier model only on proven need:

```toml
[[provider]]
id = "groq"                                  # any OpenAI-compatible host: Groq, Together,
dialect = "openai"                           # DeepSeek, Mistral, xAI, OpenRouter, Azure,
base_url = "https://api.groq.com/openai"     # or your own Ollama / vLLM server
api_key_env = "GROQ_API_KEY"

[[route]]
match = {}
mode = "enforce"
ladder = ["groq/llama-3.3-70b-versatile", "anthropic/claude-sonnet-5"]
gates = ["unit-tests"]
```

Gemini (`dialect = "gemini"`), AWS Bedrock (`auth = "aws_sigv4"`), and Google Vertex (`auth = "gcp_oauth"`) use the same shape — every documented variant lives in [`firstpass.example.toml`](firstpass.example.toml) and is validated by a parse test. Full walkthrough on the [usage page](https://dshakes.github.io/firstpass/usage.html#providers).

> [!NOTE]
> Bedrock and Vertex auth is delegated to maintained crates (`aws-sigv4`, `gcp_auth`) — never hand-rolled. Both are offline-tested and **live-unverified**: verify with real AWS/GCP credentials before production.

## The receipt

Every call becomes a hash-chained JSON trace an external auditor can re-derive — point at any request and answer *why did this go to that model, and what did it cost*:

```jsonc
{
  "trace_id": "0192f3a1-7c4e-7abc-9d21-4e8b1f0a2c33",
  "prev_hash": "9f2c…a1b7",                          // chains to the prior decision — tamper-evident
  "attempts": [
    { "rung": 0, "model": "anthropic/claude-haiku-4-5", "cost_usd": 0.0007,
      "gates": [{ "gate_id": "cargo-test", "verdict": "fail" }] },   // cheap tried first — gate caught it
    { "rung": 1, "model": "anthropic/claude-sonnet-5", "cost_usd": 0.0121,
      "gates": [{ "gate_id": "cargo-test", "verdict": "pass" }] }    // escalated, proven, served
  ],
  "final": { "served_rung": 1, "total_cost_usd": 0.0128,
             "counterfactual_baseline_usd": 0.0630, "savings_usd": 0.0502 }
}
```

## The numbers

Everything below is reproducible from this repo (`cargo run -p firstpass-bench`; `--live` uses real providers, BYOK).

| Result | Where it comes from |
|---|---|
| **~65% cheaper** at equal-or-higher success vs a predictive router | Proof-harness simulation — the methodology, with bootstrap CIs and a pre-registered kill criterion |
| **~85% lower $/success** than always-Opus, **1.00 success, 0.00 served-failure** (vs 0.12 for a predictive router) | **Live**, real Anthropic, 200 graded tasks — reproduced with a real LLM-judge gate |
| **≤10% wrong answers served, 95% confidence** — empirically 7.6%, and 5.9% with a judge on the gate, while serving 82% from the cheap tier | **Live**, 964 real MBPP coding tasks in a fail-closed sandbox — a distribution-free conformal bound **no other router offers** |

Your savings depend on how often *your* cheap model clears *your* gate — which is why every trace records the always-top counterfactual, so you measure your number instead of trusting ours. The guarantee is the claim that doesn't depend on your workload.

<div align="center"><img src="assets/proof.svg" alt="Proof-harness results: ~65% cheaper at equal-or-higher success, served-failure 0.16 vs 0.46 for prediction, ≤10% conformal served-failure at 95% confidence, kill criterion reads PROCEED" width="880"></div>

## Firstpass vs. predictive routers

| | Predictive routers | **Firstpass** |
|---|---|---|
| Decides by | guessing from the prompt | **proving the real output** |
| A wrong answer | ships silently | **is caught by the gate and escalated** |
| Quality guarantee | none | **≤10% served-failure @ 95%, earned live** |
| Adapts by | retraining a policy model | **self-tuning threshold + edit a gate** |
| Audit trail | a dashboard number | **hash-chained receipt per decision** |

## Configuration

12-factor, env-driven — `firstpass-proxy --help` for the full reference:

| Variable | Purpose | Default |
|---|---|---|
| `FIRSTPASS_MODE` | `observe` \| `enforce` | `observe` |
| `FIRSTPASS_BIND` | listen address | `127.0.0.1:8080` |
| `FIRSTPASS_CONFIG` | path to `firstpass.toml` (routes, ladders, gates, providers) | — |
| `FIRSTPASS_DB` | trace store path | `firstpass.db` |

**Endpoints:** `POST /v1/messages` (drop-in) · `POST /v1/feedback` · `GET /v1/capabilities` · `GET /healthz` · `GET /metrics`.

Multi-tenant deployments get per-tenant auth (Argon2id), rate limits, gate-health scoping, and AES-256-GCM key custody — all opt-in, default-off ([ADR 0004](docs/adr/0004-hosted-multitenant-plane.md)).

## Status

**v0.1.6 · GA-ready core, shipped in the open.** Observe + enforce over real HTTP, escalation + cross-provider failover, gate framework with LLM-judge, speculative escalation (~2× p95 win), the earned conformal guarantee, self-tuning threshold, tool/multimodal/streaming enforce, all providers, every install channel auto-published. Still ahead — and tracked honestly on the [roadmap](https://dshakes.github.io/firstpass/#roadmap): a 30-day soak, an external security audit, live-verifying Bedrock/Vertex, and the hosted multi-tenant plane ([ADR 0001](docs/adr/0001-hosted-ga-architecture.md)).

## Links

[Website](https://dshakes.github.io/firstpass) · [Usage guide](https://dshakes.github.io/firstpass/usage.html) · [SPEC](SPEC.md) · [Example config](firstpass.example.toml) · [ADRs](docs/adr) · [Agent guide](AGENTS.md) · [llms.txt](llms.txt) · [License](LICENSE)

<div align="center"><sub><b>Cheapest-first. Proven before served.</b> — proof over prediction, receipts over adjectives.</sub></div>
