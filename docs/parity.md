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
| 1 | `/v1/responses` (OpenAI Responses API) | Newer OpenAI agents speak Responses, not Chat Completions; without it they cannot point at us at all | M | planned |
| 2 | `/v1/models` catalog | Agent CLIs call it to populate a model picker; its absence shows an empty list | S | **done** |
| 3 | Session affinity | A multi-turn session that escalated re-pays the gate cycle from rung 0 every turn | M | **done** |
| 4 | Agent launchers (`firstpass launch …`) | One command beats a page of env-var instructions | S | **done** |
| 5 | Reasoning-effort normalization | Providers spell effort differently; agents should not care which is behind us | S | **done** |
| 6 | Prompt-cache breakpoints | Real money on long agent conversations against Anthropic | S | accounting fixed; write-side pending |
| 7 | Message condensing | Long conversations blow the cheap rung's context and force escalation for the wrong reason | M | planned |
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

## Order

1. **Cheap and self-contained** — `/v1/models` (done), reasoning effort, cache breakpoints,
   launchers. Each is a day or less and removes a checkbox someone can point at.
2. **Session affinity.** Ranked above the Responses API because it is not only parity: a session
   that already escalated should start where it landed, which attacks the escalation tax measured
   at 18% of first-pass spend.
3. **`/v1/responses`.** Largest of the wire-format items and needs its own streaming translation.
4. **Message condensing, trace export.**

## Where this list came from

Audited surface: 269 Python files (~54k LOC) plus ~53k LOC of Rust across eight crates; routing
profiles, request/response processors, backends, endpoints, and the launcher tree. The two hosted
routers were surveyed from public API documentation only.
