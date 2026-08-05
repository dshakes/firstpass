# Changelog

## [0.4.0]

Feature parity with the rest of the proxy-router category, reached mostly by **fixing silent wrong
numbers** rather than adding features. Three of the items below were bugs that produced a confident
wrong answer or a fabricated figure with no error to notice: a config block that parsed and did
nothing, cached prompts billed as free, and a long conversation failing outright when the rung above
would have served it.

Not breaking: every new config field is optional and every new receipt field is omitted when zero,
so existing receipts re-derive the same hash. **Reported costs will go up** on cached traffic —
that is the fix, not a regression.

### Fixed: cached prompts were billed as if they were free

Token usage was read as `usage.input_tokens` alone. Providers that support prompt caching do not
report the prompt there — they split it:

| field | what it is | billed at |
| --- | --- | --- |
| `input_tokens` | the uncached remainder | 1× |
| `cache_creation_input_tokens` | prompt written to cache | **1.25×** |
| `cache_read_input_tokens` | prompt served from cache | **0.1×** |

Firstpass counted only the first. A caller using prompt caching — which coding agents do by
default — sends a large stable prefix, so a turn with a 190k-token cached prompt arrives as an
`input_tokens` of about **20**. The receipt recorded roughly **$0.0001 for a call that cost about
$0.72**, and `[budget]` caps, which are fed by that same total, could not trip on it.

This is the same defect class as the unpriced-rung bug fixed in 0.3.0: not a missing number, a
**fabricated** one, in the record whose whole purpose is being auditable.

Cache traffic is now parsed, priced at its own multipliers, and recorded on the receipt as
`cache_write_tokens` / `cache_read_tokens`. Both the enforce path and observe-mode passthrough are
fixed — observe reads usage off the same response body and had the same gap.

**The hash chain is unaffected.** Both fields are omitted from canonical JSON when zero, so every
receipt written before this change serializes byte-identically and re-derives the same hash, and
older receipts still deserialize. Verified directly, not assumed.

`Attempt::billable_input()` returns the true prompt size (uncached + written + read); prefer it
over `in_tokens` for anything cost- or volume-shaped.

Every path that prices a real provider response is fixed, not only the gated ladder: speculative
calls that completed before cancellation, consistency samples, judge calls, and shadow probes.
Consistency and probes matter most — both re-send one prompt k times, which is exactly the shape
prompt caching is built for, so on a cached prompt nearly all of their spend lands in the cache
counters. `PriceTable::cost_usd` now documents that it is only correct with no cache traffic.

### Fixed: `[escalation.session_promotion]` parsed and did nothing

The block has been in the config schema — and exported from `firstpass-core` — since before this
release, and **nothing ever read it**. An operator could set `after_failures` and `window`, get no
parse error, and receive none of the behaviour. A config that parses and no-ops is the same class
of defect as a receipt that records `$0.00`: the file states one thing and the system does another,
with nothing to notice it by.

It is now implemented. A session that had to escalate starts its next turn on the rung it actually
needed, instead of re-paying for the rung that already failed it — aimed at the escalation tax
measured at 18% of first-pass spend.

Two additions keep it from becoming a cost regression:

- **`probe_every` (default 5)** — every 5th turn deliberately starts one rung *lower* to test
  whether the promotion is still earned. Without it, promotion is a ratchet: a conversation that
  escalates once stays expensive forever, including the trivial turns at the end. `probe_every = 0`
  is **rejected at parse** rather than silently pinning a session permanently.
- **`max_sessions` (default 10000)** — hard cap, oldest evicted first. A routing optimisation must
  not be able to exhaust memory on a proxy under load.

Promotion is per `(tenant, session)` and never crosses a tenant boundary. It chooses only where the
ladder starts; the gate still verifies whatever is served, so a wrong promotion costs money, never
correctness.

### Added

- **Tool calls translate both ways on `/v1/responses`**, so tool-using agentic requests are
  **gated** rather than passed through un-verified. A tool definition becomes Chat's nested
  `function` shape, a tool call becomes an `assistant` message with `tool_calls`, a result becomes a
  `tool` message, and a tool call in the reply returns as a `function_call` item. `tool_choice` is
  translated too — Responses spells it `{type, name}` and Chat `{type, function: {name}}`, and
  forwarding it unchanged does not error, it just lets the model ignore a tool the caller demanded.
  A response carrying a pending tool call is `status: "completed"` (generation finished — the agent
  loop being unfinished is not the same thing); `incomplete` is now reported for its real cause, a
  truncated reply, with `incomplete_details.reason`. Hosted tools (`web_search`, `file_search`,
  `computer_use`), reasoning items, and `previous_response_id` threading still take passthrough.
- **`POST /v1/responses`** — the OpenAI Responses API. Newer OpenAI-family agents speak Responses
  rather than Chat Completions, and without this they could not point at Firstpass at all. Served
  by translating to and from the Chat Completions path, so the gate, ladder, budget, and receipt are
  the **same** ones — not a parallel implementation that could drift on the parts that matter.
  Buffered only: `enforce` cannot stream, because the gate must see the whole candidate before it
  can judge it, so a streaming Responses request takes the same observe passthrough that Chat
  Completions already uses for a request it cannot gate faithfully.
- **`[escalation] prompt_cache`** — insert Anthropic prompt-cache breakpoints on the stable prefix
  (system prompt + tool definitions), the part of an agent request that repeats byte-for-byte every
  turn. **Off by default, and deliberately so**: a cache write costs 1.25× base input and a read
  0.1×, so roughly one reuse of the prefix covers the premium and everything after saves heavily —
  but single-shot traffic never reuses and would pay a 25% surcharge for nothing. Whether that
  trade pays is a fact about your traffic, not something this code can infer. A caller that already
  places its own `cache_control` is left untouched, and a prefix too small to qualify is never
  marked — Anthropic refuses to cache below a per-model minimum (1024 tokens; 2048 on Haiku), so
  marking a shorter one is a request the API can reject rather than merely wasted spend.
- **`[escalation.condense]`** — last-resort context condensing. When a conversation has overflowed
  the context window of *every* rung, the middle of the history is dropped (head and tail kept, with
  a marker turn telling the model its history is incomplete) and the top rung is retried **once**.
  Deliberately not a general trimming knob: condensing routinely would mean gating an answer
  produced from a prompt the client never sent. It fires only where the alternative is no answer at
  all, and the overflow attempts stay on the receipt so the elision is visible. Absent by default.
- **Context overflow now escalates instead of failing the request.** A prompt too large for a rung
  is a 400, and every 400 aborted the ladder — so a long agent conversation that outgrew the
  *cheapest* rung's context window failed outright, even with a larger rung configured directly
  above it. It now climbs, and records the attempt as `context_overflow` rather than a gate
  failure, so capacity-forced escalations are not pooled with quality ones in the statistics. An
  unrecognised 400 still aborts: escalating a genuinely malformed request only buys the same
  rejection at a higher price.
- **`GET /v1/models`** — OpenAI-shaped discovery so agent CLIs can populate a model picker.
  Reports distinct ladder rungs in ladder order (that order is the cost gradient) with the real
  per-1M prices from the table that bills the receipt.
- **`firstpass launch claude|codex|openai -- <cmd>`** — start a coding agent already pointed at the
  proxy. Refuses when nothing is listening, and refuses when something that is *not* Firstpass is.
- **`firstpass export --format rl`** — receipts reshaped as flat training rows (context, action,
  reward, propensity) for offline learning. The propensity is the part that matters: it is already
  logged for IPS/SNIPS/DR off-policy evaluation and is exactly what a routing log normally lacks.
  A deterministic decision exports `null` rather than a defaulted `1.0`, which would look like a
  uniformly-sampled row and quietly bias anything built on it.
- **`docs/parity.md`** — feature-parity audit against comparable routing proxies, and the order the
  remaining gaps get closed.

### Fixed

- **Cross-vendor reasoning effort.** A ladder is routinely cross-vendor, so an OpenAI client that
  set `reasoning_effort` got a **400 from Anthropic** the moment the router escalated it onto an
  Anthropic rung — a correct request broken by the ladder moving underneath it. The caller's
  request is now translated between `reasoning_effort`, `thinking.budget_tokens`, and
  `thinkingConfig.thinkingBudget`, honouring Anthropic's constraints (a budget `max_tokens` cannot
  fit is dropped rather than sent invalid; enabling thinking clears a now-invalid `temperature`).
  Same-dialect bodies are never rewritten — ADR 0005 promises verbatim passthrough, and that covers
  the caller's own mistakes.
- **`GET /healthz` now reports `service` and `version`.** A bare `{"status":"ok"}` cannot
  distinguish this proxy from anything else holding the port, which made it useless for the one
  check that needs it — `firstpass launch` pointed an agent at an unrelated dev server that answered
  200, which would have routed every request past the gate to a stranger that replies.

## [0.3.0]

**Breaking.** Two config changes can stop an existing deployment from starting. Both are cases
where the previous behaviour was silently producing a wrong number, so failing at startup with a
message is the intended upgrade path — read the two fixes below before upgrading.

### Breaking: an unpriced ladder rung is now rejected at parse

`cost_usd()` returns an error for a model with no price entry, and every call site was swallowing
it with `.unwrap_or(0.0)`. The built-in table knows only the seven first-party models, so on the
entire bring-your-own-provider path — Groq, DeepSeek, OpenRouter, a local Ollama rung — this had
two silent consequences:

- the **receipt recorded `cost_usd: 0.0`** for calls that really cost money. In a product whose
  claim is a tamper-evident audit record, that is a fabricated figure, not a missing one.
- **`[budget]` caps could never trip**, because they are fed by that same total. The budget
  guardrail was inert for exactly the operators who configured a provider by hand.

If your ladder names a model outside the built-in table, the proxy now refuses to start and prints
the block to paste:

```toml
[[price]]
model = "groq/llama-3.3-70b-versatile"
input_per_mtok = 0.59
output_per_mtok = 0.79
```

A model you host yourself really is free — declare it as an explicit `0.0`, so the `$0.00` in the
receipt is a stated fact rather than an inferred one.

### Breaking: two Google models were removed because they never existed

`google/gemini-3.1-flash` and `google/gemini-3.1-pro` were in the price table and in the ladder
`firstpass onboard` generated, and neither is a real model — `generateContent` answers
`models/gemini-3.1-flash is not found for API version v1beta`. Every Google user's first request
failed. Replaced with GA ids at Google's published rates:

| id | input / 1M | output / 1M |
| --- | --- | --- |
| `google/gemini-3.5-flash-lite` | $0.30 | $2.50 |
| `google/gemini-3.6-flash` | $1.50 | $7.50 |
| `google/gemini-2.5-pro` | $1.25 | $10.00 |

The generated Google ladder is now Flash-Lite → Flash. It was Flash → Pro, which at $1.50/$7.50
against $1.25/$10.00 is an ~8% price gradient — escalation bought nothing.

### Added

- **`firstpass kiosk`** — one press, no questions: finds the provider key already in your
  environment, writes a config, puts one real request through the ladder, prints the receipt. With
  no key it runs the keyless demo, so the button always produces a working receipt.
- **`firstpass handshake`** — the same for a caller with no terminal: one JSON document of keys
  found, the config that would run, and a **keyless** end-to-end self-test. Reports; never writes.
  Also an MCP tool.
- **`GET /panel`** — live receipts served by the binary itself, no CDN. Backed by
  `GET /v1/receipts`, which stays inside the authed group and is tenant-scoped.
- **Benchmarks**: `--coding-policy` compares routing policies on identical measurements;
  BigCodeBench and SWE-bench harnesses; `scripts/fetch-coding-dataset.py`.

### Fixed

- Provider HTTP errors log the upstream body server-side. `"upstream http 404"` alone hid the
  Gemini bug above for weeks; the body named the exact model that did not exist.
- `provider-smoke` jobs that skip for a missing secret no longer read as verified — each job
  writes its real state into the run summary.
- Judge model calls are metered and priced, so a judged policy can report a real `$/success`
  instead of one that omits the calls that produced it.

### Measured

On 974 real MBPP coding tasks (committed artifacts under `docs/benchmarks/`): gating recovered
**+15.2 points** of quality over serving the cheap model alone (paired, 95% CI [+12.9, +17.5]) and
made **zero of those 974 tasks worse**, at 12–57% lower cost per success **depending on the
ladder** — 12% against a `sonnet` ceiling, 57% against an `opus` one.

## [0.2.7]

Restored a dual-arch `latest` image: `latest` now tracks release tags rather than every push to
`main`, so it is built for amd64 and arm64.
