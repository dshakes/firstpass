# ADR 0012 — Trajectory features: routing on the agent's own conversation, before spending anything

Status: **accepted — implemented and wired; not yet validated against a pre-registered bar** · 2026-08-14

Bumps `FEATURE_VERSION` 1 → 2. Related: `crates/firstpass-core/src/features.rs`,
`crates/firstpass-proxy/src/{proxy,bandit,affinity}.rs`, SPEC §8.4/§9.2.

## Context

A competitive audit against **NVIDIA NeMo Switchyard** (Apache 2.0, pre-alpha, explicitly "not for
production use") found Firstpass ahead or equal on every advertised capability except one.

Switchyard's `stage_router` reads signals **already present in the agent's conversation** — failing
tool results, repeated unproductive work, prolonged exploration — and routes each turn accordingly,
with no extra model call. Firstpass's only cross-turn signal is `affinity.rs`, which counts **our
own gate's failures** and promotes a session's start rung after `after_failures` of them.

The asymmetry is the point:

|  | `affinity.rs` (today) | trajectory features (this ADR) |
|---|---|---|
| whose failures | **ours** — the gate's | **the agent's** — its tools' |
| when available | *after* paying for a failed attempt | *before* spending a token |
| on turns our gate would pass | invisible | still visible |

On a long coding session Firstpass currently learns a session is hard only by buying failures.
Switchyard knows from turn 3's failing test output that turn 4 is hard, for free. That is a genuine
cost-efficiency edge on precisely the workload that is growing fastest — and `Features` already
carries `agent` and `subagent` fields, so the proxy knows it is serving agents while extracting
nothing from the transcript they send.

## Decision

Extract three deterministic, zero-cost signals from the inbound message array, collapse them into
one ordinal hint, and let the existing bandit learn what to do with it.

**Where.** `request_features` already parses the body and walks `messages` (that is how
`has_images` works), so extraction adds one more pass over data already deserialized and warm in
cache. Wire-format walking lives in `proxy.rs`; **scoring lives in `firstpass-core`** so it stays
deterministic, version-stamped, and testable without a server.

**What.** `TrajectorySignals { tool_errors, tool_results, assistant_turns, repeated_tool_calls }`
→ `DifficultyHint::{None, Low, Medium, High}` → `Features::difficulty_hint: u8`.

**Both dialects, non-negotiably.** Anthropic carries `tool_result` blocks with an explicit
`is_error` flag; OpenAI uses `role: "tool"` messages and assistant `tool_calls` arrays with no such
flag. A router that walks only one shape reports "no signal" for half its traffic — which is
indistinguishable from a healthy session, the worst possible failure mode for a feature whose entire
job is spotting unhealthy ones.

**One dimension, not three.** The hint enters `ContextBucket`, which keys the bandit's per-arm
statistics. Every dimension multiplies the arm count, and arms that never accumulate traffic never
learn: a bandit with a beautifully descriptive key and four samples per arm is worse than one with a
crude key and four hundred. Three signals collapse to one 4-level ordinal — 4× cardinality, not 48×.

### Deliberate design constraints

- **Privacy holds.** Counts, never content. Repetition is detected by comparing a *hash* of
  (tool name, arguments), so no argument text ever reaches a feature vector or a receipt.
- **Bounded window** (12 messages). Unbounded, ancient failures accumulate forever and ratchet every
  long conversation to maximum difficulty permanently — pinning it to the top rung long after it
  recovered. A cost regression wearing the costume of a signal.
- **Under-count rather than over-count.** OpenAI has no error flag, so failure detection there is a
  heuristic on anchored prefixes, not a substring search for "error". A linter reporting `0 errors`,
  a test runner printing a summary, a log reader — all succeed while mentioning errors. Under-counting
  costs a little routing signal; over-counting pushes healthy sessions to expensive rungs and costs
  money.
- **"No signal" ≠ "signal is fine".** A request with no tool activity scores `None`, not `Low`.
  Conflating them would make every single-shot request look like a healthy agent session and pollute
  the bandit's healthiest bucket with traffic carrying no trajectory information at all.
- **Never fails a request.** Malformed, truncated, or unfamiliar bodies yield "no signal". The
  extract path must never reject something the upstream would have served.
- **A hint, never a verdict.** It may choose which rung to *start* on. It may never decide what is
  fit to *serve* — only a gate does that. This is the `predict-to-start, verify-to-serve` invariant
  from `bandit.rs`, and it is the line between Firstpass and everything that routes on a guess.

## Why `FEATURE_VERSION` 1 → 2

The field is `#[serde(default)]` and v1 traces still deserialize, so the bump is not needed for
parsing. It is needed because the version records *how a vector was computed*. A v1 trace genuinely
had no hint available; leaving the version at 1 would let a policy fitted on v2 traffic be replayed
against v1 traces as though the missing hints were real zeroes — silently mixing "no signal" with
"not measured". That is exactly the error the version field exists to prevent.

## Consequences

**Default-off in effect, not just in config.** The hint is `0` for every non-agent request, so a
deployment seeing no agent traffic keys exactly as before and its learned bandit statistics remain
valid. No existing operator's routing changes.

**Not yet validated.** The feature is wired, but per `report.rs` discipline it does not get to claim
a win without clearing a **pre-registered bar**: it must beat the current bandit on **$/success**
with **no increase in served-failure rate** (the conformal bound must still hold), paired-bootstrap
CI via `stats.rs`, on a multi-turn task family. That harness is the next deliverable and needs real
API spend. Until it runs, this ADR claims *plumbing*, not *benefit* — and if the bar is missed, the
negative result gets written up and the feature comes back out.

## Evidence so far

Deterministic unit tests, no network: both dialects extracted; clean-but-error-mentioning output not
miscounted; repetition detected without errors present; window bound enforced; malformed bodies
never panic; the hint reaches `Features` and the bandit key for both dialects; v1 traces still
deserialize.

**All mutation-tested.** Disabling `is_error` reading, dropping the window bound, switching to a
substring match, ignoring repeats, and scoring no-activity as `Low` each fail exactly their own test.
Zeroing the hint inside the bandit key initially **survived** — the wiring was untested, so the
feature could have been extracted, traced, and exported while being silently ignored by its only
consumer, with the telemetry insisting it worked. `the_difficulty_hint_reaches_the_bandit_key` now
covers it.

---

## Addendum, 2026-08-14: measured, and refuted on MBPP

The pre-registered bar was run at full scale. **The feature did not clear it.**

`firstpass-bench --agentic-multiturn`, 974 MBPP tasks, 1228 turns, $8.05 of real spend, genuine
multi-turn trajectories (model writes code -> sandbox runs visible tests -> real failures return as
real error text -> model retries):

| policy | success | $/success |
|---|---|---|
| baseline (always start cheap) | 0.895 | $0.00529 |
| trajectory-informed start | 0.897 | **$0.00547** |

**−3.4% cost improvement against a required +5%.** Quality was fine (+0.0024, CI [−0.0008,
+0.0065]). Verdict: `KILLED`. Full artifact and caveats in
`docs/benchmarks/mbpp-974-agentic-multiturn.txt`.

**Status change: the feature remains default-off and does not ship on this evidence.** The
extraction, the audit-contract bump, and the bandit wiring stay — they are correct, tested, and cost
nothing when the hint is `None`, which it is for all non-agent traffic. What is withdrawn is any
claim that acting on the hint pays.

### Why, and why that is informative rather than embarrassing

MBPP retries are cheap. Starting higher when the conversation looks troubled skips a cheap rung that
would usually have passed, and the savings on genuinely hard tasks do not cover it. This is the
real-data version of the −66.7% regression the harness caught in its own synthetic check.

The finding is workload-specific, and the boundary is the interesting part: the trade should invert
where retries are expensive. Agentic SWE-bench measures exactly that and is reported separately.

### Two corrections to this ADR's original claims

1. **`deep >= 8` was unreachable.** Observed depth capped at 2 on MBPP and 7 on agentic SWE-bench
   (8-turn cap) — `High` was dead code in every workload measured. A level that never fires looks
   like coverage while contributing nothing, and the bandit can never learn an arm it is never
   given. Lowered to `>= 5`.
2. **`Low` also never fires** in either harness, because the retry loop only continues when the
   cheap rung fails, so every recorded tool result is a failure. Not yet fixed: it needs the loop to
   record successful tool calls too, which changes what a "turn" means. Recorded here rather than
   quietly left as a gap.
