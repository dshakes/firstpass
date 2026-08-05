# Feature parity: proxy-layer routers

What comparable open-source LLM routing proxies do that Firstpass does not yet, and the order we
close it. Written from a read of the published source of the category's most complete
implementation (an NVIDIA-authored routing proxy, Apache-2.0) plus the public API surface of two
hosted routers.

**Read-only audit.** Features were identified by reading published source to understand *what
capability exists*. No third-party code was copied, adapted, or translated — every item below is
implemented from its behavioural description against our own types.

## The thing that does not appear on this page

Parity is a floor, not a claim. The routers surveyed all decide **before** generation — from a
prompt classifier, a trajectory judge, request headers, or static weights. Their response-side
processors log: RL traces, routing records, stats. None of them can reject a produced answer and
re-route.

Firstpass decides **after** the output exists, and that is not on the parity list because nothing
surveyed has it to be at parity with. Everything below is table stakes we owe users so that
choosing verification does not cost them a feature they already had.

## Verified gaps

Each row was confirmed by absence of a route or capability, then re-checked against our actual
config surface — two items on the first pass turned out to already exist and were struck.

| # | Capability | Why it matters | Size | Status |
|---|---|---|---|---|
| 1 | `/v1/responses` (OpenAI Responses API) | Newer OpenAI agents speak Responses, not Chat Completions; without it they cannot point at us at all | M | **done** |
| 2 | `/v1/models` catalog | Agent CLIs call it to populate a model picker; its absence shows an empty list | S | **done** |
| 3 | Session affinity | A multi-turn session that escalated re-pays the gate cycle from rung 0 every turn | M | **done** |
| 4 | Agent launchers (`firstpass launch …`) | One command beats a page of env-var instructions | S | **done** |
| 5 | Reasoning-effort normalization | Providers spell effort differently; agents should not care which is behind us | S | **done** |
| 6 | Prompt-cache breakpoints | Real money on long agent conversations against Anthropic | S | **done** (accounting + opt-in writes) |
| 7 | Message condensing | Long conversations blow the cheap rung's context and force escalation for the wrong reason | M | **done** (last-resort only) |
| 8 | RL / training-trace export | Traces are already recorded; the gap is an export shape a trainer can consume | S | **done** |

Three of the four shipped items turned out to be fixing a bug rather than adding a feature,
which is the usual shape of parity work:

- **Reasoning effort** was a live **400**. A ladder is routinely cross-vendor, so an OpenAI client
  that set `reasoning_effort` got an Anthropic rejection the moment the router escalated it —
  correct request, ladder moved underneath it.
- **`launch`** exposed that `/healthz` answered a bare `{"status":"ok"}`, indistinguishable from
  any other service holding the port. The first version of the launcher pointed an agent at an
  unrelated dev server on :8080 that answered 200. Every request would have bypassed the gate and
  reached something that replies — a failure that looks like success.
- **Session affinity** was already in the config schema as `[escalation.session_promotion]`,
  parsing cleanly and doing **nothing** — no code read it. Implementing it was the work; the
  feature had been advertised to operators for releases.

### Struck on re-check — already present

- **Subagent-aware routing.** `[route.match]` already matches on `subagent`, and also on `agent`,
  `task_kind`, and `language`. Wider than the surveyed equivalent, which keys on subagent alone.
- **Stage router.** The gated ladder *is* staged escalation, with the stage boundary decided by a
  gate on real output rather than a prediction. Composing profiles is a config-ergonomics nicety,
  not a missing capability.

Both were on the first-pass list because the module names were absent. Module absence is not
feature absence — the re-check is why this table has 8 rows and not 10.

## Two that ship off by default, and why

**All 8 are implemented.** Two are opt-in, for reasons about correctness and cost rather than
caution — a default that quietly costs money or quietly answers a different question is worse than
a knob.

### Prompt-cache writes (`[escalation] prompt_cache`)

Marks the stable prefix — system prompt plus tool definitions. Off by default because it is a bet
on your traffic, not a free optimisation:

- A cache **write** costs 1.25× base input and a **read** 0.1×, so roughly one reuse of the prefix
  covers the premium and everything after it saves heavily. A long agent session with a fixed
  system prompt and tool set reuses constantly; single-shot traffic never does and pays a 25%
  surcharge on the prefix for nothing.
- The one place reuse looks structural — a consistency gate or shadow probe drawing k samples of
  one prompt — does **not** currently benefit, because those k calls are issued **concurrently**
  (`consistency.rs`), and a cache entry is not available to a request that starts before the first
  write completes. Capturing that saving would mean serializing sample 1 and parallelising the
  rest: roughly 67% of the gate's input cost, bought with an extra round trip of latency **in the
  serving path**.

That second point is a genuine cost/latency tradeoff rather than an optimisation, and it applies
only to k-sample gates — **not** to the routed request, where an agent's turn 2 naturally reuses the
prefix written by turn 1 because the calls are sequential. So the two halves of the default-on
question separate cleanly:

- **Routed caller requests** — no latency cost at all, and break-even is exactly **one reuse**.
  Any multi-turn session with a stable system prompt and tool set is past break-even by its second
  turn.
- **k-sample gates** — would need sample 1 serialized, which is the tradeoff above. Not implemented,
  and it is what a `--coding-policy` run reporting `$/success` and p95 latency would settle.

It stays opt-in nonetheless, because single-shot traffic is past no break-even and would pay the 25%
write premium for nothing. That is a fact about your traffic, not about this code. What has changed
is that the guard below makes leaving it on materially safer:

**A prefix too small to qualify is never marked.** Anthropic refuses to cache below a per-model
minimum (1024 tokens; 2048 on Haiku), so marking a shorter prefix is not merely wasted — it is a
request the API can reject. The estimate is deliberately conservative, skipping only prefixes that
could not possibly qualify.

Callers that place their own `cache_control` are left untouched: they know their prefix better than
this code does, and extra breakpoints would exceed Anthropic's limit and re-bill the write premium.

### Condensing (`[escalation.condense]`)

Fires only once a prompt has overflowed the context window of **every** rung. Condensing routinely
would mean the gate verifies an answer produced from a prompt the client never sent, and the
receipt would attest to a decision about a different question than the one asked — which is the one
thing this product cannot afford to get wrong.

Restricted to the exhausted-ladder case the trade is no longer "faithful vs degraded" but "degraded
vs no answer at all", and the overflow attempts stay on the receipt so the elision is visible rather
than inferred.

## Tool-using Responses requests are gated

Function tools and the `function_call` / `function_call_output` round trip translate in **both**
directions: a tool definition becomes Chat's nested `function` shape, a tool call becomes an
`assistant` message with `tool_calls`, a result becomes a `tool` message, and a tool call in the
reply comes back as a `function_call` item rather than vanishing. A turn awaiting a tool result
reports `status: "incomplete"`, because telling an agent the turn is `completed` while the model
waits on it is its own wrong answer.

That closes the limitation this section previously recorded — agentic clients, the ones this
endpoint exists for, are now verified rather than passed through.

Still un-gated, and these are genuine rather than provisional:

- **Hosted tools** (`web_search`, `file_search`, `computer_use`) run inside the provider and have no
  Chat equivalent. Sending a partial tool set is worse than not routing at all: a model missing one
  of its tools produces a confidently wrong plan.
- **A `tool_choice` naming a hosted tool**, for the same reason.
- **Reasoning items**, which carry provider-internal state.
- **`previous_response_id`** threading, where the upstream holds history we do not have — a
  translated request is then not the conversation the client is continuing.
- **A malformed tool item** — a `function_call` missing its `call_id`, say. The gate asks the
  translator rather than restating its rules, so anything that would translate to nothing routes to
  passthrough instead of silently losing the turn.

The check is an allow-list rather than a deny-list because the deny-list lost three times running:
images, then files, then tool calls were each silently dropped, each producing a confidently wrong
answer rather than an error. Enumerating what breaks loses that game whenever a provider adds a
field.

## Where this list came from

Audited surface: 269 Python files (~54k LOC) plus ~53k LOC of Rust across eight crates; routing
profiles, request/response processors, backends, endpoints, and the launcher tree. The two hosted
routers were surveyed from public API documentation only.
