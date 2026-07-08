# Firstpass — Founding Spec

**Version:** 0.1 (founding draft)
**Date:** 2026-07-07
**Status:** Draft for founder review
**One-liner:** Firstpass routes every LLM request to the cheapest model that **provably** clears your quality gate — verification-gated escalation with a full audit trail, learning your workload from its own verdicts.

> The cheapest model takes the **first pass** at every request — and either clears your
> gate (proven, not guessed) or the request escalates to the next tier. You pay up only
> when the proof says you must.

---

## 0. Positioning in one paragraph

Every model router on the market routes by **prediction**: a learned policy guesses which
model will answer well, sends the request there, and never checks. Firstpass routes by
**proof**: send the request to the cheapest plausible tier, run a real gate on the output
(tests, typecheck, schema, fresh-context judge), and escalate one rung only on gate
failure. Every decision is logged as an auditable trace, and those gate verdicts — ground
truth, not guesses — become the training signal that makes the routing policy smarter over
time. Prediction is a black box you have to trust. Proof is a receipt you can read.

---

## 0.1 Prior art & what's actually new (stated honestly)

The escalation *mechanism* is not novel, and the pitch must not pretend it is. The model-cascade
research literature established cheapest-first querying with a learned scorer that escalates on low
confidence, and a line of verifier/judge-driven deferral work has refined it since. The mechanism is
proven — and left un-productized. The standing academic critique of every cascade is that it has *no
reliability guarantee* and treats the deferral threshold as an unprincipled hyperparameter.

The competing products all route by **prediction**, deciding before generation and never checking the
output. **Predictive routers** learn a policy — from model internals, from eval data, or from
preference data — and call the single model it points to; they need retraining or re-evaluation for
every new model. **Black-box orchestrators** put a predictive head on a hidden state and compose
frontier models opaquely, with unpredictable per-request cost. **Model gateways** route on price and
uptime rules, not output quality — they are complements Firstpass sits on top of, not rivals. None of
them verify the output they actually served.

Firstpass's defensible novelty is **not** the cascade. It is three things no paper and no product
has put together:

1. **Tamper-evident audit** of every routing decision + gate verdict (§9) — the exact artifact
   black-box orchestrators cannot give a regulated buyer, and the transparency axis to win on.
2. **Outcome-feedback calibration** (§8.3, feedback API) — downstream ground truth ("did the tests
   pass an hour later?") flows back and auto-tunes the gate, answering the "no reliability guarantee"
   critique that every static cascade carries.
3. **Zero-retrain onboarding** — add a model as one ladder rung; the gate decides. Predictive routers
   structurally require retraining/re-eval per new frontier model, in a market shipping them monthly.

The founding bet stands on those three plus the wedge in §2 (agent/coding traffic, where gates are
cheap and objective). Pitched as a generic cheap-router, Firstpass loses to prior art; pitched as
the verified, auditable, self-correcting layer for agent fleets, it occupies ground no incumbent holds.

---

## 0.2 Agent-first by construction

Firstpass's primary user is not a human clicking a dashboard — it is an **agent** (a coding agent, a
CI bot, a subagent in a fleet). That is a design constraint on *every* surface, not a marketing line:

- **Product.** The data plane is a drop-in `base_url`. An agent adopts Firstpass by changing one
  environment variable and offboards by unsetting it — no SDK, no code change, no rebuild. The proxy
  is OpenAI/Anthropic wire-compatible so any agent that speaks those APIs already speaks Firstpass.
- **Components.** Every surface is machine-first and self-describing: config is declarative
  (TOML/JSON), traces are structured JSON with a re-derivable hash chain, verdicts are typed
  (`pass`/`fail`/`abstain` + score), errors are structured (never prose an agent must parse), and the
  feedback channel is a plain HTTP API. A **discovery endpoint** (`GET /v1/capabilities`) lets an agent
  learn the ladder, gates, modes, and limits at runtime, and an optional **MCP server** exposes traces,
  verdicts, and feedback as tools so an agent can inspect and correct its own routing.
- **Comms & docs.** Documentation is agent-consumable by default: [`llms.txt`](llms.txt) at the root,
  an [`AGENTS.md`](AGENTS.md) contributor/onboarding manifest, machine-readable schemas for the trace
  and config. A human-readable page is a rendering of the machine-readable source of truth, never the
  other way round.
- **Onboarding.** Self-serve and programmatic: point `base_url`, GET `/v1/capabilities`, start in
  `observe` mode (serve immediately, gate asynchronously) so nothing is at risk while the agent's own
  traffic teaches the router. No human in the setup loop.
- **Offboarding.** One env var, instantly reversible, zero lock-in at the data plane (§7.2). BYOK means
  the agent keeps its own provider relationship; there is nothing to migrate off.

Concretely: `GET /v1/capabilities`, the MCP server, `llms.txt`, and `AGENTS.md` are first-class M1/M2
deliverables, not afterthoughts. Where a choice is "nice human UI" vs "clean agent contract," the agent
contract wins and the UI renders it.

---

## 1. Problem

1. **Model spend is misallocated.** Agentic coding workloads send everything to a
   top-tier model because nobody can tell, per request, when a cheaper one would have
   sufficed. The cost delta between tiers is 10–30x; the fraction of requests that
   actually *need* the top tier is unknown — which is exactly the problem.
2. **Existing routers can't be trusted with quality.** Predictive routers (route-by-guess)
   have no feedback loop to catch their own mistakes; when they route down and the output
   is bad, the customer eats it silently. That's why adoption stalls: nobody puts a black
   box between their agent and their model.
3. **Nobody can answer "why did this request go to that model?"** Enterprises adopting AI
   need routing decisions to be explainable and auditable — for cost governance, for
   incident review, for compliance. No current router produces evidence.
4. **The feedback signal already exists and is thrown away.** Agent harnesses run tests,
   typechecks, and reviews constantly. Those verdicts are a free, ground-truth label for
   "was this model good enough for this task" — and today they evaporate.

## 2. Thesis

- **Verification-gated escalation beats predictive routing** wherever a gate exists,
  because it converts routing from a trust problem into an evidence problem.
- **Gates exist for code.** Agentic coding is the wedge market: tests, typecheck, lint,
  build, structured-output schemas, and diff review are all machine-checkable.
- **The trace log is the moat.** Every routed request emits a labeled example
  (*task features → tier tried → verdict → cost*). Competitors with predictive routers
  must synthesize training data; Firstpass's customers generate it as exhaust. v1's
  learned policy trains on v0's dogfood traces; v2's learned coordinator trains on v1's.
- **Deterministic first, learned second.** Anything deterministic logic can decide never
  goes to a model. The ladder, the budget caps, the gate execution — all deterministic.
  Learning enters only where judgment is required (which rung to *start* on), and only
  after shadow evaluation against logged traces.

## 3. Non-goals

- **Not a universal router.** v0–v1 target agentic/coding workloads where gates exist.
  Chat, creative writing, and open-ended Q&A have no gate; we do not pretend to serve
  them (observe-mode data collection only).
- **Not an inference provider or a reseller.** BYOK (bring your own keys). We never mark
  up tokens; we charge for the routing intelligence and the audit plane.
- **Not a trained-coordinator clone.** We do not start with offline RL / evolved
  coordinators (that is v2, gated on v1 traction and trace volume). Competing on learned
  coordination from day one is competing on a research lab's terms with none of their
  resources.
- **Not an eval platform.** Gates plug in; we ship reference gates, but building a
  general evals product is someone else's business.
- **Not an agent framework.** Firstpass sits *below* the harness (a base_url), never
  replaces it.

## 4. Who it's for (ICP, in order)

1. **AI-native dev-tool companies** running agent fleets (coding agents, CI agents,
   review bots) with real token bills and existing test gates. Pain: spend. Hook: verified
   savings at quality parity.
2. **Platform/infra teams at enterprises** adopting agentic coding. Pain: governance —
   "which model saw what, why, and what did it cost." Hook: the audit trail.
3. **Individual power users / small teams** running Claude Code, Codex, Gemini CLI and
   similar harnesses hard. Hook: OSS proxy, one env var, immediate savings dashboard.

First customer: **compass** (founder's own agent-config toolkit) routing its subagent
fleet. Second: **lantern** (production AI-infra workload). Dogfooding is the GTM seed and
the first trace dataset.

---

## 5. Core concepts (the domain model)

| Concept | Definition |
|---|---|
| **Rung** | One (provider, model) pair in an ordered ladder, cheapest first. |
| **Ladder** | Ordered list of rungs for a given route/task-kind. E.g. `haiku → sonnet → opus`. Cross-provider ladders are first-class. |
| **Gate** | A checker that examines a candidate response (or a task outcome) and returns a **Verdict**. Deterministic where possible; judge-model where not. |
| **Verdict** | `pass` / `fail` / `abstain` + score ∈ [0,1] + evidence. The atomic unit of ground truth. |
| **Attempt** | One request sent to one rung, with its gate verdicts. |
| **Trace** | The full record of a routed request: features, attempts, verdicts, costs, latencies, final outcome. Append-only, hash-chained. |
| **Policy** | The function `features → starting rung (+ escalation rules)`. v0: static config. v1: contextual bandit. v2: learned coordinator. |
| **Mode** | `enforce` (gate before serving, escalate on fail) or `observe` (serve immediately, gate asynchronously — verdicts feed learning only). |
| **Feedback** | A deferred verdict reported by the harness *after* serving (e.g. "the tests passed 40 s later"), attached to the trace via feedback API. |

### 5.1 The two gate scopes (important design decision)

Not all gates can run inside the proxy:

- **Inline gates** (request-scoped): run synchronously on a single candidate response
  before it is served. Examples: schema validation, lint-on-diff, fast judge. These power
  `enforce` mode.
- **Deferred gates** (task-scoped): the real outcome of a coding task — *do the tests
  pass after the agent applied the edits* — is only known later, in the harness, possibly
  spanning many completions. These arrive via the **Feedback API**: the harness (e.g. a
  compass hook) posts a verdict tied to a `session_id`/`trace_id`. Deferred verdicts
  cannot un-serve a response; they are the highest-quality training labels and can trigger
  *session-level* escalation ("this session's rung-0 attempts keep failing tests — start
  subsequent requests at rung 1").

This split is the honest answer to "you can't run `cargo test` in a proxy" — and the
Feedback API is what no predictive router has.

---

## 6. Product surface

### 6.1 The proxy (data plane)

- **Anthropic-compatible** `/v1/messages` and **OpenAI-compatible** `/v1/chat/completions`
  endpoints. A harness adopts Firstpass by changing one base URL
  (`ANTHROPIC_BASE_URL=https://localhost:4130` or the hosted equivalent) — the same
  insertion pattern proven by compression/caching proxies.
- Transparent pass-through of everything the policy doesn't touch: tool definitions,
  system prompts, images, tool results. Firstpass rewrites exactly one thing: the
  `model` field (plus optional per-rung prompt adaptations in v2).
- **Streaming semantics:**
  - `observe` mode: stream through untouched; gates run asynchronously on the buffered
    copy; verdicts are recorded for learning. Zero added latency. This is the default
    onboarding mode and the permanent mode for ungated traffic.
  - `enforce` mode: response is buffered, gated, then served (re-chunked as a stream if
    the client requested streaming). Only sane for non-interactive traffic (subagents,
    batch, CI) or fast inline gates. Config chooses per route.
- Single static binary. Runs as a sidecar, a laptop daemon, or a hosted endpoint.

### 6.2 The feedback API

- `POST /v1/feedback` — `{ trace_id | session_id, gate_id, verdict, score?, evidence?,
  reporter }`. Authenticated per tenant. Idempotent.
- Reference reporters shipped day one: a **Claude Code hook** (compass), a **CI step**
  (GitHub Actions), and a **shell one-liner** (`firstpass report --gate tests --verdict
  pass`).

### 6.3 The control plane (paid, later)

- Dashboard: savings vs. baseline, escalation rates, gate precision, per-route
  cost/latency curves, trace explorer ("why did this request go there" → the receipt).
- Policy management: shadow → canary → live promotion for learned policies.
- Audit export: hash-chained trace log, retention controls, SIEM export.

### 6.4 The CLI

- `firstpass init` — scaffold config, detect harness, print the one env var.
- `firstpass bench` — the tier-clearance benchmark (§10) against your own repo/tasks.
- `firstpass traces` / `firstpass why <trace_id>` — local trace inspection.
- `firstpass gate test <gate_id>` — run a gate against a fixture.

---

## 7. Architecture (v0)

```
 harness (Claude Code / Codex / any SDK)
   │  base_url
   ▼
┌─────────────────────────────────────────────┐
│ firstpass proxy (single Rust binary)        │
│                                             │
│  router core ── policy engine (static v0)   │
│      │              │                       │
│      ▼              ▼                       │
│  provider clients  gate runner              │
│  (anthropic,       (inline gates as         │
│   openai, google,   subprocess plugins)     │
│   oai-compatible)                           │
│      │                                      │
│      ▼                                      │
│  trace store (SQLite → Postgres)            │
│  feedback API (deferred verdicts)           │
└─────────────────────────────────────────────┘
```

**Stack decisions (v0):**

- **Rust** for the proxy (axum/hyper/tokio). It's a latency-sensitive hot path handling
  streamed bytes; the added p50 overhead budget is **< 5 ms** for pass-through and the
  binary must be trivially distributable. (Founder's strongest hot-path language; matches
  the rest of the fleet's conventions: clippy-clean, no unwrap on fallible paths, typed
  errors.)
- **SQLite (WAL)** for the local/single-node trace store; **Postgres** for hosted.
  Same schema, `sqlx` both ways.
- **Gates are subprocess plugins**: language-agnostic exec contract (§8). No dynamic
  linking, no WASM in v0 (revisit for hosted multi-tenant sandboxing — likely WASI or
  Firecracker in the hosted gate-runner).
- **Config is TOML**, versioned in the customer's repo. The policy that routed a request
  is content-hashed into its trace — reproducibility from day one.
- **No queue, no k8s, no microservices in v0.** One binary + one DB file. The hosted
  control plane comes after dogfooding proves the loop, not before.

### 7.1 Request lifecycle (enforce mode)

1. Ingest request → extract **features** (§9.2) → match a **route** in config.
2. Policy selects starting rung (v0: static per route; v1: bandit).
3. Send to rung *r*. Stream into buffer.
4. Run inline gates (parallel where independent). Aggregate verdict:
   - all pass → serve response; write trace; done.
   - any fail → check budget + ladder: escalate to rung *r+1*, goto 3.
   - abstain → per-gate config: treat as pass (fail-open) or fail (fail-closed).
5. Ladder exhausted or budget exhausted → per-route config: serve **best-scoring
   attempt** (default) or return a structured error.
6. Deferred verdicts attach to the trace whenever they arrive.

### 7.2 Failure semantics (prod-grade requirements)

- **Provider outage on rung r** → treat as `abstain` with reason `provider_error`,
  escalate (an outage must never take the customer down — the ladder doubles as
  failover, which is a sellable feature by itself).
- **Gate crash/timeout** → `abstain` + configured fail-open/closed. Gate errors are
  metered; a gate exceeding an error budget is auto-disabled with an alert (a broken
  gate silently failing closed would burn money; silently failing open would burn trust
  — both are alarms, not defaults).
- **Proxy crash** → the harness talks to providers directly again by unsetting one env
  var. Escape hatch is documented on the front page. No lock-in at the data plane, ever.
- **Trace store unavailable** → serve traffic, buffer traces to disk, backfill. Losing
  a trace is losing money (training data), but never availability.

---

### 7.3 Distribution & packaging (world-class artifacts)

The product ships as a **single statically-linked binary with zero runtime dependencies** — the natural
payoff of the Rust choice. Time-to-first-routed-request is measured in seconds, not a setup guide.

- **Install in one line, several ways** (built with [`dist`](https://opensource.axo.dev/cargo-dist/), which
  produces all of the below from one release):
  - `curl -LsSf https://firstpass.dev/install.sh | sh` (and a PowerShell equivalent for Windows)
  - `brew install firstpass/tap/firstpass`
  - `docker run firstpass/firstpass` (multi-arch image, small distroless base)
  - `cargo install firstpass` (from crates.io, for the Rust-native)
- **Prebuilt multi-arch releases** on every tag: macOS (arm64 + x86_64), Linux (arm64 + x86_64,
  gnu + musl for static), Windows x86_64 — with checksums and cosign signatures. No "compile from source" step required.
- **Config-optional start.** `firstpass up` with provider keys in the environment routes immediately in
  `observe` mode with sane defaults; a `firstpass.toml` refines it. `firstpass doctor` validates keys, config,
  and gate binaries before you rely on it.
- **Multi-provider / multi-LLM from day one.** First-class clients for **Anthropic, OpenAI, and Google**,
  plus a generic **OpenAI-compatible** client that covers any compatible endpoint — hosted aggregators,
  third-party inference providers, and local runtimes alike. A model is a `provider/model` string in the ladder — adding one is a
  config line, never a rebuild (this is the zero-retrain onboarding of §0.1, at the packaging layer).
- **Reproducible & verifiable.** Pinned toolchain, locked dependencies, `cargo-deny` in CI (license + advisory
  gate), SBOM attached to releases. The install script is auditable and versioned.

Distribution quality is a feature, not an afterthought: a router nobody can install in a minute does not get
adopted by the agent fleets that are the whole market.

---

### 7.4 Integration surface (pluggable into everything that talks to an LLM)

The whole value of a data-plane router is that it disappears into the tools you already run. Firstpass is
**wire-compatible**: it speaks the provider APIs verbatim, so anything that talks to an LLM plugs in without a
code change and unplugs the same way.

- **Wire-compatible endpoints.** Firstpass exposes the **Anthropic Messages**, **OpenAI Chat Completions +
  Responses**, and **Google Gemini `generateContent`** surfaces. A caller points its base URL at Firstpass and
  everything it already sends — **streaming (SSE), tool/function calling, multimodal inputs, structured
  outputs** — passes through faithfully. Gates run on the *assembled* output; streaming is preserved to the
  caller (buffered only when a pre-serve gate in enforce mode requires the full response).
- **Every way to plug in:**
  1. **`base_url` env swap** — coding agents, IDE extensions, and SDKs that honour `ANTHROPIC_BASE_URL` /
     `OPENAI_BASE_URL` / `OPENAI_API_BASE` / `GOOGLE_GEMINI_BASE_URL`. Zero code change; the whole integration
     is one environment variable.
  2. **Sidecar / reverse proxy** — headless agents, batch jobs, CI runners, and serverless functions route
     provider traffic through a Firstpass container in front of them.
  3. **Embedded library** — the router as a linkable crate (plus a thin HTTP/FFI shim) for teams that want it
     in-process rather than as a hop.
  4. **MCP server** — the agent-native surface: an agent calls tools to read its own traces, query
     `/v1/capabilities`, and submit outcome feedback, so it can inspect and correct its own routing.
  5. **CLI** — `firstpass up` / `doctor` / `trace` for humans, scripts, and Makefiles.
- **BYOK passthrough.** Provider keys stay the caller's; Firstpass forwards them and never marks up tokens, so
  the caller keeps its own provider relationship and ToS.
- **The invariant:** Firstpass adds a routing + audit + feedback layer *around* the wire contract the agent
  already uses; it never changes that contract. Pluggable by construction, reversible by construction (§7.2).

M1 ships the Anthropic + OpenAI wire endpoints and the `base_url`/sidecar paths; Gemini, the MCP server, and
the embedded-library mode follow in M2+.

---

## 8. The gate framework (the moat — most design effort lives here)

### 8.1 Gate plugin contract

A gate is any executable honoring:

- **stdin** (JSON): `{ "request": {...}, "candidate": {...}, "features": {...},
  "config": {...}, "context": { "workdir": "...?", "env": {allowlisted} } }`
- **stdout** (JSON): `{ "verdict": "pass|fail|abstain", "score": 0.0-1.0,
  "reasons": ["..."], "evidence": {...} }`
- **exit ≠ 0** → gate error → abstain (`provider_error` reasons captured from stderr).
- Hard timeout per gate (config, default 5 s inline / 15 min deferred).

Language-agnostic by construction: a gate can be a 10-line Python script, a compiled
binary, or `bash -c 'jq ... | ...'`.

### 8.2 Reference gates shipped in v0

| Gate | Type | Scope | What it checks |
|---|---|---|---|
| `schema` | deterministic | inline | Structured output matches a JSON Schema (tool-call args, extraction tasks). |
| `patch-applies` | deterministic | inline | Proposed diff applies cleanly to the workspace snapshot. |
| `lint-diff` | deterministic | inline | Changed lines pass the repo's linter (ruff/clippy/eslint autodetected). |
| `compiles` | deterministic | deferred | Typecheck/build passes after edits (reported by harness hook). |
| `tests` | deterministic | deferred | The task's test command exits 0 (reported by harness hook / CI). |
| `judge-diff` | model | inline | Fresh-context LLM judge reviews the candidate against the request. **Maker ≠ checker enforced structurally:** the judge model must differ from the rung that produced the candidate; cross-provider judging is recommended default. |
| `self-consistency` | model | inline | k cheap samples; disagreement above threshold → fail (escalation by uncertainty). |

### 8.3 Anti-gaming / anti-Goodhart requirements

The gate is the product's integrity. A gamed gate poisons both served quality *and* the
training data. Hard rules:

1. **Maker ≠ checker.** A judge gate never runs on the model (ideally not even the
   provider) that produced the candidate. Enforced by the runner, not by convention.
2. **Candidate is data, not instructions.** Judge prompts wrap the candidate in
   delimited, escaped blocks; judges run with **no tools**, no network, and a pinned
   system prompt. Prompt-injection attempts inside a candidate must not be able to flip
   a verdict; this is red-teamed with a fixture suite (compass's red-team corpus seeds it).
3. **Gates can't see the ladder.** A gate never knows which rung produced the candidate
   or what escalation costs — no "it's from the cheap model so grade easier."
4. **Gate quality is itself measured.** Deferred ground truth (tests) grades inline gates
   (judges): a judge whose `pass` verdicts keep failing tests later has measurable
   false-pass rate → surfaced on the dashboard, auto-flagged past threshold. **Gate
   precision/recall is a first-class product metric** (§14).
5. **Score inflation is detected** by monitoring per-gate score distributions for drift.

### 8.4 Escalation policy config (v0 sketch)

```toml
[[route]]
match = { agent = "claude-code", subagent = ["test-runner", "explore"] }
mode  = "enforce"
ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
gates  = ["schema", "judge-diff"]

[[route]]
match = { task_kind = "code_edit" }
mode  = "enforce"
ladder = ["anthropic/claude-sonnet-5", "anthropic/claude-opus-4-8"]
gates  = ["patch-applies", "lint-diff", "judge-diff"]
deferred_gates = ["compiles", "tests"]

[[route]]                      # everything else: learn, don't touch
match = {}
mode  = "observe"
ladder = ["anthropic/claude-opus-4-8"]

[budget]
per_request_usd = 0.50
per_session_usd = 10.00
per_day_usd     = 250.00
on_exhausted    = "serve_best_attempt"   # never brick the customer

[escalation]
max_rungs_per_request = 3
session_promotion = { after_failures = 3, window = "30m" }  # start higher for the rest of a struggling session
```

---

## 9. The trace schema (designed for v1/v2 training from the first record)

### 9.1 Record (JSON, one per routed request)

```jsonc
{
  "trace_id": "uuid7",                    // time-ordered
  "prev_hash": "sha256(...)",             // hash chain → tamper-evident audit log
  "tenant_id": "t_...",
  "session_id": "s_...",                  // groups an agent session
  "ts": "2026-07-07T18:04:11Z",
  "mode": "enforce",
  "policy": { "id": "static@sha256:ab12…", "explore": false },
  "request": {
    "api": "anthropic.messages",
    "prompt_hash": "sha256",              // bodies stored only if retention allows
    "features": { /* §9.2 */ }
  },
  "attempts": [
    {
      "rung": 0, "model": "claude-haiku-4-5", "provider": "anthropic",
      "in_tokens": 8123, "out_tokens": 512,
      "cost_usd": 0.0031, "latency_ms": 1740,
      "gates": [
        { "gate_id": "schema@v1",     "verdict": "pass", "score": 1.0,  "cost_usd": 0, "ms": 2 },
        { "gate_id": "judge-diff@v2", "verdict": "fail", "score": 0.31, "cost_usd": 0.004, "ms": 2600,
          "evidence_ref": "ev_9f2…" }
      ],
      "verdict": "fail"
    },
    { "rung": 1, "model": "claude-sonnet-5", "...": "...", "verdict": "pass" }
  ],
  "deferred": [
    { "gate_id": "tests@v1", "verdict": "pass", "reported_at": "…",
      "reporter": "compass-hook/1.0" }
  ],
  "final": {
    "served_rung": 1, "served_from": "attempt",
    "total_cost_usd": 0.0192, "gate_cost_usd": 0.008,
    "total_latency_ms": 6400, "escalations": 1,
    "counterfactual_baseline_usd": 0.0410   // what always-top-rung would have cost → savings math
  }
}
```

### 9.2 Feature vector (the bandit's context; privacy-safe by construction)

No prompt text — only derived features:
`task_kind` (classified: code_edit | test_gen | explore | review | extract | chat | …),
`language`, `agent` + `subagent`, `prompt_token_bucket`, `tool_count`,
`has_images`, `repo_fingerprint` (salted hash), `session_failure_count`,
`hour_bucket`, `prior_rung_clearance` (per-bucket rolling stats).
Feature extraction is deterministic and versioned (`features@vN` recorded per trace) —
a policy trained on v3 features never silently scores v4 vectors.

### 9.3 Privacy & retention

- **Zero-retention mode** (default for hosted): store features + verdicts + costs only;
  prompt/response bodies are never persisted. Self-host can opt into body retention for
  debugging.
- Evidence blobs (judge rationales) stored separately with their own retention clock.
- **No training on customer traces across tenants without explicit opt-in.** Per-tenant
  policies (§11) make this the default architecture, not a promise.

---

## 10. The tier-clearance benchmark (Milestone 0 — before any product code beyond a harness)

The whole business rests on one empirical question: **what fraction of real agentic
requests does each rung clear?** If cheap-tier clearance is low, the ladder is slower
*and* pricier than routing straight to mid-tier. Measure before believing.

**Method:**
1. Task corpus: 200+ real tasks harvested from compass's repos and lantern's workload
   (bug fixes, test writing, refactors, doc edits, review passes) — not synthetic
   benchmarks; those are exactly what predictive routers overfit.
2. Run each task at each rung independently (Haiku, Sonnet, Opus; plus one non-Anthropic
   mid-tier for cross-provider data). Grade with the deferred gates (tests/typecheck) —
   ground truth, not judges.
3. Report per-(task_kind, rung): clearance rate *p*, cost *c*, latency; judge-gate
   agreement vs. ground truth (this simultaneously calibrates the judge gates).

**Decision math** (published in the report):
expected ladder cost `E = (c₀+g₀) + (1−p₀)(c₁+g₁) + (1−p₀)(1−p₁)(c₂+g₂)…`
The ladder beats direct-to-rung-1 for a task class iff `E < c₁`, i.e. roughly when
`p₀ > (c₀+g₀)/c₁`. With a 10–20x inter-tier price gap, rung-0 clearance as low as
**10–15% already breaks even**; anything above is profit. The benchmark fills in the real
numbers per task class, and those numbers become v0's static routing table.

**Kill criterion (pre-registered):** if no task class shows cheap-tier clearance above
break-even *and* the audit-trail story alone doesn't justify the proxy, stop — publish
the negative result and fold the learnings back into compass instead. Ambition includes
knowing the exit.

---

## 11. Learning roadmap

### v1 — Contextual bandit (learned routing, no GPUs)

- **Model:** per-bucket Thompson sampling. Bucket = coarse feature tuple
  (task_kind × language × prompt_bucket × agent). Per bucket per rung, maintain a Beta
  posterior over clearance probability, updated by gate verdicts (deferred verdicts
  weighted highest). Choose starting rung minimizing expected total cost given posteriors,
  subject to constraints.
- **Constraints (hard, deterministic — never learned):** budget caps; exploration rate
  cap (≤ 5% of requests, only in `enforce` mode where a failed exploration is caught by
  the gate, *never* on routes flagged `critical`); floor/ceiling rungs per route.
- **Promotion pipeline:** every candidate policy is evaluated **offline first** by replay
  against logged traces (IPS/doubly-robust estimators), then **shadow mode** (policy
  logs what it *would* have done alongside the live static policy), then canary %, then
  live. Rollback is one config flip; the trace records which policy routed every request.
- Why bandit and not RL: the reward is immediate (gate verdict), the action space is tiny
  (which rung), and posteriors are **inspectable** — you can print *why* the policy
  believes Haiku clears 62% of Rust test-gen. Auditability survives the learning.

### v2 — Learned coordinator (topology, not just tier)

Only after v1 shows demand + trace volume. Scope: policy also chooses the **scaffold** —
one-shot vs. plan→build→verify triad vs. k-sample self-consistency vs. decompose —
per request, trained on accumulated traces (now including which topologies won). This is
the research-lab territory (evolved/RL coordinators); we enter it with a proprietary
dataset of gate-labeled production traces that cannot be bought — that's the earned
right to compete there.

### v2.5 — Feature ideas from the founder's backlog (kept honest)

- **Session-level promotion** (already v0): a struggling session starts higher.
- **Cross-provider failover as a tier-0 feature** (falls out of §7.2 for free).
- **Prompt adaptation per rung** (small models often need different prompt shapes —
  Conductor-style rewriting; v2, needs eval).
- **Verified-savings billing** (§13) — the counterfactual baseline field exists in every
  trace from day one precisely to make this billable later.

---

## 12. Security & trust model

Firstpass sits in the most sensitive position possible: between customers' prompts and
their model providers. Trust is the product; these are requirements, not aspirations.

1. **BYOK, keys never at rest in plaintext.** Self-host: keys stay in the customer's env.
   Hosted: envelope encryption (KMS), decrypted in memory per request, never logged,
   never in traces.
2. **Tenant isolation end-to-end:** per-tenant policies, per-tenant posteriors,
   per-tenant traces. No cross-tenant learning without explicit opt-in (§9.3). The
   multi-tenant boundary is sacred — never widened "to make something work."
3. **Prompt-injection resistance in judge gates** (§8.3.2) with a maintained red-team
   fixture suite; a release that regresses the injection suite does not ship.
4. **Hash-chained trace log** → tamper-evident audit; export with chain verification.
5. **Hosted gate execution is sandboxed** (WASI/Firecracker, no network by default) —
   customer gates are untrusted code by definition.
6. **Supply chain:** signed releases, SBOM, pinned deps, provenance attestation —
   (the compass `harden` pipeline, applied to Firstpass's own CI).
7. **Compliance path:** SOC 2 Type I within 12 months of hosted GA; zero-retention mode
   is the default posture that makes early enterprise conversations possible before that.

---

## 13. Business model & go-to-market

- **Open-core.**
  - **OSS (Apache-2.0): the proxy, gate runner, reference gates, CLI, trace schema.**
    The auditability claim demands source visibility; adoption at the data plane must be
    frictionless; and OSS is the wedge against closed predictive routers.
  - **Paid:** hosted control plane (dashboard, trace explorer, audit export, SSO),
    learned policies (the bandit and everything after), team features, support.
- **Pricing hypotheses (validate with dogfooding + first 5 design partners):**
  1. Platform fee per seat/agent + **% of verified savings** (the counterfactual baseline
     in every trace makes savings measurable, not estimated — an aligned-incentive
     pricing no predictive router can honestly offer).
  2. Fallback: flat per-request routing fee (simpler procurement).
- **GTM sequence:** dogfood (compass + lantern) → publish the tier-clearance benchmark as
  the launch content (real numbers on "what % of coding-agent traffic actually needs the
  top tier" is link-bait *and* the sales deck) → OSS launch with the one-env-var demo →
  design partners from the agent-harness ecosystem.
- **Competitive map (why we win each fight):**

| Competitor class | Their game | Our counter |
|---|---|---|
| Plumbing routers / gateways | Unified API, uptime, price/uptime routing | We're not plumbing; we compose with them (a rung can *be* one of their models). |
| Predictive routers | Route by predicted quality, before generation | No verification, no receipts, no per-customer learning. Proof beats prediction where gates exist. |
| Trained coordinators | Learned multi-model orchestration | Black-box, one frozen policy, benchmark-trained. We're auditable, per-tenant, production-trained — and we'll meet them at v2 with data they can't buy. |
| Academic cascades | Cost-quality cascades on benchmarks | No product, no deferred ground truth, no harness integration. |
| DIY / in-house | A bash ladder | The gate framework, anti-gaming, trace/audit plane, and learning are the 90% under the waterline. |

- **The one-sentence answer to "why not X":** *"Everyone else routes by predicting
  quality; Firstpass routes by verifying it — and hands you the receipt."*

---

## 13.5 The evidence program (how we beat incumbents: published, reproducible data)

The strategy against every incumbent — predictive routers, trained coordinators,
plumbing — is the same: **make claims they can't make, backed by artifacts anyone can
re-run.** Their numbers are self-reported on frozen benchmarks; ours are reproducible on
the reader's own repo. This is a standing program, not launch content:

1. **Open benchmark, runnable by anyone.** `firstpass bench` (§6.4) *is* the benchmark:
   it runs the tier-clearance methodology (§10) against the user's own codebase and
   prints their own break-even table. The strongest possible sales argument is the
   prospect's own data — and it's unfakeable by us, which is the point.
2. **Head-to-head evals with published methodology.** Reproducible harness comparing
   Firstpass vs. always-top-model vs. predictive routing (via public APIs where
   available) on the open task corpus: cost, end-task success (ground-truth gates, never
   judge-only), latency, and — uniquely — **auditability** (can the system produce a
   receipt per decision: yes/no). Methodology, corpus, and raw traces published; every
   number in marketing links to the run that produced it.
3. **Pre-registration discipline.** Experiments state their hypothesis, metric, and
   success threshold *before* running (M0's kill criterion is the first instance).
   Negative results get published too — credibility compounds, cherry-picking bankrupts.
4. **Continuous public dogfood dashboard.** Live verified-savings and quality-parity
   numbers from compass + lantern traffic (features/aggregates only, per §9.3). A
   routing product that publishes its own live false-pass rate is making a claim no
   black-box competitor can copy without re-architecting.
5. **Research posture.** The trace corpus + bandit/coordinator results are written up as
   technical reports (arXiv) as they mature — the v2 coordinator work is publishable, and
   publication is both recruiting and moat-signaling. Rule: no claim in a report that
   isn't backed by a committed, hash-referenced run artifact.
6. **Third-party verification path.** Once design partners exist, invite an external
   audit of the savings methodology — "verified by" beats "we claim."

Cost of the program: it slows marketing down. That's accepted; the entire brand is
*proof over prediction*, and it must hold for our own claims or the product is a lie.

## 14. Success metrics

- **North star: verified $ saved at quality parity** (counterfactual baseline − actual,
  summed over traces whose deferred gates passed).
- **Quality guardrails (these gate the north star, same philosophy as the product):**
  - Gate false-pass rate (inline `pass` later contradicted by deferred ground truth) —
    target < 2%, alarmed at 5%.
  - End-task success delta vs. always-top-rung baseline — must be statistically
    indistinguishable (parity), measured continuously on dogfood.
- **Adoption:** routed requests/day; % of traffic in `enforce` vs `observe`; escalation
  rate per route (too high = mis-set ladder; too low = ladder could start cheaper).
- **Performance:** added p50 latency < 5 ms pass-through, < gate-cost budget in enforce.
- **Business:** design partners signed; % savings fee realized; OSS stars/installs as
  top-of-funnel only (per the compass launch lesson: stars are a gate, not a goal).

## 15. Risks (ranked) & mitigations

1. **Cheap-tier clearance too low** → ladder uneconomical. *Mitigation:* Milestone 0
   benchmark with a pre-registered kill criterion; observe-mode still yields the audit
   product even if escalation flops.
2. **Gate gaming / false passes** → silent quality erosion, poisoned training data.
   *Mitigation:* §8.3 in full; gate precision as a first-class alarmed metric.
3. **Latency overhead unacceptable for interactive use.** *Mitigation:* observe mode as
   default; enforce mode scoped to subagent/batch/CI traffic; inline gates budgeted.
4. **Fast-follow by a big router** (an incumbent gateway bolts on gates). *Mitigation:* the deferred-
   feedback harness integrations + per-tenant learned policies + audit plane are the
   compound moat; move fast on the harness ecosystem where we have home-field advantage.
5. **Provider ToS friction** (proxying, model-output-judging-model). *Mitigation:* BYOK
   (customer's own keys, their ToS relationship); legal review before hosted GA.
6. **Single-founder bandwidth.** *Mitigation:* ruthless v0 scope (one binary, one DB, no
   control plane); compass's own agent fleet as the workforce; milestones sized in weeks.
7. **Trust ("why put you between me and my model?").* *Mitigation:* OSS data plane,
   escape hatch = unset one env var, zero-retention default, hash-chained receipts.

## 16. Milestones

| # | Milestone | Deliverable | Acceptance criteria |
|---|---|---|---|
| M0 | **Tier-clearance benchmark** | Harness + report + routing table | ≥200 real tasks, 3+ rungs, ground-truth gating; break-even analysis per task class; go/no-go per §10 kill criterion |
| M1 | **Proxy MVP** | Rust binary: Anthropic+OpenAI endpoints, static ladder, observe mode, SQLite traces | compass subagent traffic routed for 1 week, zero request loss, p50 overhead <5 ms, traces queryable |
| M2 | **Gate framework** | Plugin contract, 7 reference gates, enforce mode, feedback API + compass hook reporter | injection fixture suite passing; deferred verdicts attaching; gate error-budget auto-disable working |
| M3 | **Dogfood GA** | compass + lantern in enforce mode on scoped routes; savings dashboard (CLI) | 30 days continuous; verified savings > 0 at quality parity; false-pass < 2% |
| M4 | **Bandit (shadow)** | Feature extraction v1, Thompson policy, offline replay eval, shadow logging | shadow policy beats static on replay ≥X% cost at parity (X set from M0 data) |
| M5 | **Bandit (live)** | Canary → live promotion pipeline, rollback flip | live cost < static baseline at parity over 30 days |
| M6 | **OSS launch** | Repo public, docs, one-env-var quickstart, benchmark report published | external installs; ≥3 design-partner conversations |

## 17. Open questions (founder decisions, not blockers for M0/M1)

1. **Name/trademark:** "Firstpass" (working name; superseded "Switchyard"). Caveat:
   common English phrase → hard to trademark/own and weak for SEO. Distinctive coined
   alternates surfaced in the naming search if a commercial rebrand is wanted later
   (Greek: Basano, Tekmo; Sanskrit: Viveka, Nikasha). Working name stands until
   incorporation forces the check.
2. **License:** Apache-2.0 assumed for OSS core (patent grant matters here); confirm.
3. **First non-Anthropic rung** for cross-provider data in M0: which provider/model.
4. **Hosted-first vs self-host-first** for design partners (spec assumes self-host first).
5. **Company formation timing** — before or after M3 dogfood proof.

## 18. Naming & attribution note

Public docs describe the ideology generically — verification-gated escalation, learned
routing, proof over prediction. Research lineage (cascade literature, coordinator-model
work) is cited in a single ACKNOWLEDGMENTS section when the repo goes public, not
threaded through product copy.

---

*Next actions: M0 benchmark harness design doc → `docs/m0-benchmark.md`; repo scaffolding
(CI, lint, release provenance via the compass `harden` pipeline); compass integration
hook sketch.*
