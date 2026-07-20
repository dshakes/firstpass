<div align="center">

# ⚡ Firstpass

### Cheap until proven otherwise.

The adaptive LLM router that checks **every answer** with your gate, pays for a big model **only on proof of need**, and caps wrong answers with a **mathematical guarantee**.

[![CI](https://github.com/dshakes/firstpass/actions/workflows/ci.yml/badge.svg)](https://github.com/dshakes/firstpass/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/dshakes/firstpass)](https://github.com/dshakes/firstpass/releases)
[![PyPI](https://img.shields.io/pypi/v/firstpass)](https://pypi.org/project/firstpass/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**[Website](https://dshakes.github.io/firstpass)** · [Install](#install) · [Quickstart](#quickstart) · [How it works](#how-it-works) · [Benchmarks](#benchmarks) · [Docs](https://dshakes.github.io/firstpass/usage.html)

<img src="assets/hero.svg" alt="Firstpass cuts your AI bill — from $$$$ to $ — at the same quality, guaranteed: it checks every answer and pays for a big model only when truly needed" width="900">

</div>

## Highlights

- 💸 **Cheapest model first, always** — you pay frontier prices only when a real check proves you must.
- 🛡️ **A guarantee, not a vibe** — ≤10% wrong answers served at 95% confidence, earned live on 974 real coding tasks ([artifact](docs/benchmarks/mbpp-live-base.txt)).
- 🧠 **Self-tuning** — the serve threshold recalibrates from live outcomes as your traffic drifts. No retraining, ever.
- 🎯 **Predict-to-start, verify-to-serve** — a UCB1 bandit learns which rung to *start* on per context; the gate still checks the output before it ships.
- 🔬 **Measured confidence** — the self-consistency gate resamples the model k times; agreement on the *actual output* is a calibrated confidence score, not a guess about the prompt.
- 🧪 **Rehearse before you enforce** — `firstpass ope` replays your logged receipts against a candidate ladder: estimated cost and served-failure, with confidence intervals, before anything changes.
- 🔍 **Proof, not prediction** — the gate checks the *actual output*; a wrong answer is caught, never shipped on a guess.
- 🧾 **A receipt per decision** — hash-chained, tamper-evident, auditable: *why this model, what did it cost, what did it save*.
- 🌐 **Every provider** — Anthropic, OpenAI, Gemini, Bedrock, Vertex, Groq, DeepSeek, OpenRouter, Azure, local Ollama/vLLM.
- 🪶 **Drop-in, walk-out** — one env var in, one env var out. Speaks the wire format your agent already uses.

## Install

No Rust, no toolchain — grab a binary and go:

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/dshakes/firstpass/releases/latest/download/firstpass-proxy-installer.sh | sh
```

Or through your package manager — each row below is live and republishes on every release:

| | |
|---|---|
| 🐍 **pip / uvx** | `pip install firstpass` · `uvx --from firstpass firstpass-proxy` |
| 🍺 **Homebrew** | `brew install dshakes/tap/firstpass-proxy` |
| 🐳 **Docker** | `docker run -p 8080:8080 -e FIRSTPASS_BIND=0.0.0.0:8080 ghcr.io/dshakes/firstpass:latest` |
| 🦀 **Cargo** | `cargo install --git https://github.com/dshakes/firstpass firstpass-proxy` (needs a Rust toolchain; crates.io publish pending) |
| ⬇️ **Binaries** | macOS · Linux · Windows, checksummed, self-updating (`firstpass-proxy-update`) — [Releases](https://github.com/dshakes/firstpass/releases) |

## Quickstart

Three lines. Zero config. Zero risk — observe mode changes nothing:

```bash
firstpass-proxy                                     # watches your traffic, touches nothing
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"   # your agent now routes through firstpass
# … use your agent normally — every call gets a receipt: what it'd route, what you'd save
```

Convinced by your own numbers? Switch on routing:

```bash
cp firstpass.example.toml firstpass.toml
FIRSTPASS_MODE=enforce FIRSTPASS_CONFIG=./firstpass.toml firstpass-proxy
```

Leaving is `unset ANTHROPIC_BASE_URL`. That's the whole offboarding story.

## 🤖 Agentic onboarding — one command does everything

Don't follow docs. Firstpass detects your machine, plans the setup, executes it, and verifies itself:

```console
$ firstpass onboard --apply
detected: shell=zsh · proxy_running=false · routed=false · claude_cli=true

✓ proxy started (pid 17005, observe mode) — log: firstpass-proxy.log
✓ wired ~/.zshrc — export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
→ optional: claude mcp add firstpass -- firstpass mcp
✓ verified — proxy healthy · capabilities live
```

It auto-detects your shell (zsh/bash/fish), whether the proxy is running, whether you're already routed, and which agents you have — then does only what's missing. **Idempotent** (re-run any time), **transparent** (`firstpass onboard` alone is a dry run showing the exact plan), and **reversible**: `firstpass offboard` strips the shell line, stops the proxy, and prints the unset — the whole exit in one command.

For agents onboarding *themselves*: [`llms.txt`](llms.txt) + [`AGENTS.md`](AGENTS.md) ship machine-readable setup, `GET /v1/capabilities` gives runtime discovery, and `firstpass mcp` exposes traces and savings as tools.

## Benchmarks

<div align="center"><img src="assets/bench.svg" alt="Cost per successful task, live on 200 graded tasks: always-top $0.0023 at 0.98 success; predictive router $0.0007 at 0.88 success while silently serving wrong answers 12% of the time; always-cheap $0.0001 but 0.62 success; firstpass $0.0003 at 1.00 success with zero wrong answers served" width="900"></div>

And the claim no other router makes: on **974 real MBPP coding tasks** (fail-closed sandbox, real test gates — [committed artifact](docs/benchmarks/mbpp-live-base.txt), re-earned 2026-07-20), firstpass earned a **distribution-free bound of ≤10% wrong answers served at 95% confidence** — calibrated risk 5.5%, realized served-failure 7.7% at the threshold, while serving **82%** of requests from the cheap tier. An LLM judge on the gate tightens it further (judged artifact lands in the same directory). Your savings depend on your workload — which is why every trace records the always-top counterfactual, **so you measure your number instead of trusting ours.**

Reproduce it — each command labels itself and states what it costs:

```bash
cargo run -p firstpass-bench                    # simulation harness (free, self-labeled SIMULATION)
cargo run -p firstpass-bench -- --live          # the 200-task live benchmark (your key, ~a few $)
curl -sLO https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
FIRSTPASS_CODING_DATASET=./mbpp.jsonl \
  cargo run --release -p firstpass-bench -- --coding-live   # the MBPP bound (your key + Docker, ~$5)
```

Result artifacts for the published numbers live in [`docs/benchmarks/`](docs/benchmarks/) ([methodology](https://dshakes.github.io/firstpass/#proof), pre-registered kill criterion included).

## How it works

<div align="center"><img src="assets/demo.svg" alt="A live routing decision: the cheap model's answer fails the real test gate, firstpass escalates one rung, the stronger model passes, and the answer is served with a sealed receipt showing the saving" width="900"></div>

1. **Route** — every request opens on the cheapest rung of your model ladder.
2. **Prove** — a *gate* checks the actual output: your unit tests, a JSON schema, or an LLM judge (maker ≠ checker).
3. **Escalate** — only on gate failure: one rung up, budget-capped, cross-provider failover on a 5xx.
4. **Learn** — outcomes feed back; the serve threshold self-tunes so the guarantee tracks your live traffic.

> **Who decides a request needs the expensive model?** The gate — from the cheap model's *actual answer*. Never a classifier guessing from the prompt. Change what "good" means by editing a gate; there is no policy model to retrain.

## "Do I have to write gates?"

No. Meet it where you are:

| Effort | You get |
|---|---|
| **None** — observe mode | Firstpass reports what it *would* route and save. Nothing changes. |
| **One sentence** — judge gate | A second model grades every answer against your plain-English rubric. |
| **One config line** — consistency gate | The model answers k times; agreement is measured confidence (self-consistency, Wang et al. 2022). |
| **Your existing tests** | The strongest gate: generated code ships only if your suite actually passes. |

Flaky gates auto-disable on an error budget — one bad check can't take down a route.

## Every provider, including open-source

A ladder rung is `<id>/<model>` — open on a free local model, escalate to a frontier model only on proven need:

```toml
[[provider]]
id = "groq"                                  # any OpenAI-compatible host — Groq, Together,
dialect = "openai"                           # DeepSeek, Mistral, xAI, OpenRouter, Azure —
base_url = "https://api.groq.com/openai"     # or your own Ollama / vLLM box
api_key_env = "GROQ_API_KEY"

[[route]]
match  = {}
mode   = "enforce"
ladder = ["groq/llama-3.3-70b-versatile", "anthropic/claude-sonnet-5"]
gates  = ["unit-tests"]
```

`anthropic` and `openai` are built in; Gemini (`dialect = "gemini"`), AWS Bedrock (`auth = "aws_sigv4"`), and Google Vertex (`auth = "gcp_oauth"`) use the same shape. Every variant ships in [`firstpass.example.toml`](firstpass.example.toml), guarded by a parse test — full walkthrough on the [usage page](https://dshakes.github.io/firstpass/usage.html#providers).

**Verification status, stated plainly:** the Anthropic path is **live-verified end-to-end** (real traffic through the running proxy). The OpenAI-compatible, Gemini, Bedrock, and Vertex adapters are **implemented and offline-tested against recorded wire shapes, pending live verification** — each flips to *verified* only when a key-gated CI smoke test exercises it against the real endpoint ([roadmap](docs/roadmap.md), Phase 1). If an unverified path misbehaves on your account, that's a bug we want: open an issue with the receipt.

<details>
<summary><b>🧾 The receipt</b> — every decision is a hash-chained trace an auditor can re-derive</summary>

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

Downstream outcomes flow back via `POST /v1/feedback` onto a deferred-verdict side table that never alters the sealed record.
</details>

<details>
<summary><b>⚙️ Configuration</b> — 12-factor, env-driven</summary>

| Variable | Purpose | Default |
|---|---|---|
| `FIRSTPASS_MODE` | `observe` \| `enforce` | `observe` |
| `FIRSTPASS_BIND` | listen address | `127.0.0.1:8080` |
| `FIRSTPASS_CONFIG` | path to `firstpass.toml` (routes, ladders, gates, providers) | — |
| `FIRSTPASS_DB` | trace store path | `firstpass.db` |

**Endpoints:** `POST /v1/messages` (Anthropic drop-in) · `POST /v1/chat/completions` (OpenAI drop-in) · `POST /v1/feedback` · `GET /v1/capabilities` · `GET /healthz` · `GET /metrics`.

Multi-tenant deployments add per-tenant auth (Argon2id), rate limits, gate-health scoping, and AES-256-GCM key custody — all opt-in, default-off ([ADR 0004](docs/adr/0004-hosted-multitenant-plane.md)).
</details>

## Firstpass vs. predictive routers

| | Predictive routers | ⚡ **Firstpass** |
|---|---|---|
| Decides by | guessing from the prompt | **proving the real output** |
| A wrong answer | ships silently | **caught by the gate, escalated** |
| Quality guarantee | none | **≤10% served-failure @ 95%, earned live** |
| Adapts by | retraining a policy model | **self-tuning threshold + edit a gate** |
| Audit trail | a dashboard number | **hash-chained receipt per decision** |
| Policy changes | deploy and hope | **rehearsed first: `firstpass ope` replays your logs with CIs** |

And the one good idea predictive routers had — starting on the right model — is *inside* firstpass now: the bandit picks the starting rung, prediction errors cost only latency, and the gate still decides what ships.

## Status

**v0.1.7 — pre-GA, shipped in the open.** Working today: enforce + observe over real HTTP (Anthropic path live-verified), cross-provider failover, schema + subprocess + LLM-judge + self-consistency gates with per-gate `on_abstain` policy, bandit start-rung selection, speculative escalation (~2× p95 offline-proven), the earned conformal guarantee, self-tuning threshold, offline policy replay (`firstpass ope`), `firstpass savings` from your own receipts. Honest limits, tracked on the [roadmap](docs/roadmap.md): structured (tools/images) enforce is **default-on with a fidelity guard** — it routes only through ladders that carry content verbatim (Anthropic-dialect today; OpenAI/Gemini structured translation is next) and serves streaming clients buffered-then-SSE (verification needs the whole candidate); four of five provider dialects await live wire verification; 30-day soak, external security audit, and the hosted multi-tenant plane are ahead of us, not behind us. GA is a checklist we publish ([ADR 0003](docs/adr/0003-ga-readiness.md)), not an adjective.

## Links

[Website](https://dshakes.github.io/firstpass) · [Usage guide](https://dshakes.github.io/firstpass/usage.html) · [SPEC](SPEC.md) · [Example config](firstpass.example.toml) · [ADRs](docs/adr) · [Agent guide](AGENTS.md) · [llms.txt](llms.txt) · [License](LICENSE)

<div align="center">

**Try cheap. Prove it. Escalate only on failure.**

<sub>proof over prediction · receipts over adjectives</sub>

</div>
