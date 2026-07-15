# ADR 0005 — Enforce through tool-calling and multimodal requests

- Status: Accepted — P1+P2 implemented (default-off), P3 queued
- Date: 2026-07-15
- Supersedes: —
- Related: SPEC §7.1 (enforce), §7.4 (pluggable), ADR 0001 (hosted plane)

## Context

The enforce path (route → gate → escalate) only engages when `enforce_can_handle`
is true:

```rust
features.tool_count == 0 && !features.has_images && !messages_have_tool_blocks(body)
```

Any request that declares `tools`, carries images, or contains `tool_use` /
`tool_result` content blocks falls back to **observe passthrough** — transparently
forwarded and logged, but **not routed** to a cheaper model. That fallback is
correct and safe (it never corrupts the call), but it means the wedge — proven
cost-routing — does not apply to the traffic that matters most: **Claude Code
sends tools on essentially every turn**, so today CC gets full observability from
firstpass and cost-routing on almost nothing.

Root cause, from the code:

- `ChatMessage.content` is a **`String`** — the enforce path collapses structured
  content (tool_use / tool_result / image blocks) to plain text. A model that
  can't see the prior `tool_use`/`tool_result` blocks can't continue a tool
  conversation, so enforcing such a request would corrupt it.
- `ModelRequest.tools` is **already carried** (opaque `Value`, forwarded as-is).
- `ModelResponse` **already keeps `raw`** (the full wire response) alongside the
  extracted `text` — so *response*-side tool_use blocks are already preserved as
  long as serving returns `raw`.

So the only real fidelity gap is the **request side** (`ChatMessage.content`).

## Decision

Make the enforce request/response path **fidelity-preserving** for structured
content, and gate turning it on behind an **opt-in flag, default-off**, with the
observe fallback remaining the safety net. Production enablement is gated on
**live verification against real tool round-trips** — never flipped on blind.

Concretely:

1. `ChatMessage.content` becomes a `serde_json::Value` that is *either* a plain
   string (today's shape, byte-identical on the wire) *or* the original array of
   content blocks (`text`, `tool_use`, `tool_result`, `image`). The adapters
   forward it as-is — both Anthropic and OpenAI accept string-or-array content.
2. The enforce request builder preserves the incoming message content verbatim
   instead of flattening it to text. Gates keep operating on an extracted **text
   view** (a helper that concatenates the text blocks), so gate behavior is
   unchanged.
3. Serving returns `ModelResponse.raw` (the full wire response), so `tool_use`
   blocks the chosen model emits reach the caller intact.
4. `enforce_can_handle` relaxes to accept tool/image/tool_block requests **only
   when the opt-in `[escalation] enforce_structured = true` is set**. Default
   (unset) → observe fallback, byte-identical to today.

## Consequences / Invariants (must never regress)

- **I1 — Default-off is byte-identical.** With `enforce_structured` unset, every
  existing test passes unchanged and behavior is exactly today's.
- **I2 — No content block is dropped.** A round-tripped request preserves every
  `tool_use` / `tool_result` / `image` block; the served response preserves every
  `tool_use` block the model produced.
- **I3 — Live-verification gate.** Before an operator relies on it in production,
  they drive real tool turns through enforce and byte-compare the served
  `tool_use` blocks against a direct call. The flag ships **off** and documented
  as "verify with your tool workload first."
- **I4 — Observe remains the safety net.** Any parse ambiguity or adapter that
  can't faithfully round-trip a request falls back to observe, never a lossy
  enforce.

## Phases

- **P1 — request-side fidelity + tests. ✅ Done.** `ChatMessage.content → Value`;
  adapters forward it as-is; `parse_model_request` preserves incoming content
  verbatim; gates read `ChatMessage::text_view()`. Tests prove the invariants:
  `text_message_serializes_byte_identical_to_a_plain_string` (I1),
  `tool_and_image_blocks_survive_the_request_round_trip` and
  `parse_model_request_preserves_content_verbatim_and_projects_text` (I2).
- **P2 — flag + relaxed gate. ✅ Done (default-off).** `[escalation]
  enforce_structured` added (default `false`). When on, `enforce_can_handle`
  routes tool/image/tool-block requests through enforce — **except streaming**,
  which still falls to observe (that's P3). Guarded by
  `structured_enforce_routes_tools_but_not_streaming`. **Live-verify (I3) is still
  required before an operator turns it on in production** — the flag ships off and
  documented as such; the code path is proven offline, not yet against a live tool
  workload.
- **P3 — streaming tool round-trip. Queued.** Extend to SSE tool_use deltas (the
  observe stream path already relays SSE; enforce streaming with tools is the last
  piece). Until then, `enforce_structured` deliberately excludes `stream:true`.

## Risks

- **Tool corruption** — the reason for the phased, default-off, live-verified
  rollout. P1 changes no default behavior; P2 is opt-in and gated on verification.
- **OpenAI tool-format divergence** — Anthropic and OpenAI differ on tool/message
  shapes. P1 keeps content opaque (forwarded as-is per adapter); cross-provider
  tool routing is validated per-adapter in P2.
