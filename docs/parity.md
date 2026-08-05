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

That second point is a genuine cost/latency tradeoff rather than an optimisation, and it is why
turning this on by default still needs a measurement — a `--coding-policy` run with and without,
reporting both `$/success` and p95 latency. The knob shipping is not the same decision as the
default changing.

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

## Known limitation: tool-using Responses requests are not gated

`/v1/responses` translates only what provably round-trips — plain text/image conversations with no
tools. A model given tools may reply with a tool call, which has no representation on the way back,
so those requests take the un-gated passthrough.

This is an allow-list because the deny-list version lost three times running: images, then files,
then tool calls were each silently dropped, each producing a confidently wrong answer rather than an
error. Enumerating what breaks loses that game whenever a provider adds a field.

It costs gating for exactly the agentic clients the endpoint targets, which is a real limitation.
Lifting it needs tool-call translation in both directions.

## Where this list came from

Audited surface: 269 Python files (~54k LOC) plus ~53k LOC of Rust across eight crates; routing
profiles, request/response processors, backends, endpoints, and the launcher tree. The two hosted
routers were surveyed from public API documentation only.
