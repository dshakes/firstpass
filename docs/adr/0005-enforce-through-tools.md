# ADR 0005 — Enforce through tool-calling and multimodal requests

- Status: Accepted — P1+P2+P3 implemented (default-off); P4+P5 default-on + fidelity guard; M1 OpenAI inbound
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
  Response-side fidelity is enforced in `anthropic_response_json`: served content
  blocks come **verbatim** from the upstream response (`resp.raw`), so `tool_use`
  reaches the caller intact instead of being reconstructed as a single text block.
- **P2 — flag + relaxed gate. ✅ Done (default-off).** `[escalation]
  enforce_structured` added (default `false`). When on, `enforce_can_handle`
  routes tool/image/streaming requests through enforce. Guarded by
  `structured_enforce_routes_tools_and_streaming`.
- **P3 — streaming through enforce. ✅ Done.** A `stream:true` request that reaches
  enforce is served the gated result **re-emitted as SSE** (`anthropic_sse_from_message`):
  the gate needs the whole candidate, so enforce buffers, gates, then streams the
  served blocks out — one delta per block, with `tool_use` input carried as an
  `input_json_delta`. Proven by `enforce_sse_reemission_preserves_text_and_tool_use`.

**On I3 (live-verify).** Fidelity is now enforced *in code* and proven offline on
both sides — request wire body (`anthropic_wire_forwards_tool_and_image_content_verbatim`),
response blocks (verbatim from `raw`), and the SSE round trip. The remaining live-verify
step is an operator's final confidence check against their own tool workload before
enabling the flag in production; it is no longer the *only* thing standing between the
code and correctness. The flag still ships **off** by default.

## Risks

- **Tool corruption** — the reason for the phased, default-off, live-verified
  rollout. P1 changes no default behavior; P2 is opt-in and gated on verification.
- **OpenAI tool-format divergence** — Anthropic and OpenAI differ on tool/message
  shapes. P1 keeps content opaque (forwarded as-is per adapter); cross-provider
  tool routing is validated per-adapter in P2.

## Addendum (P1 roadmap, post-v0.1.7): default-on + verbatim raw carry + fidelity guard

- **P4 — raw-body carry.** `ModelRequest.raw` now holds the full original inbound JSON;
  Anthropic-dialect adapters (anthropic / bedrock / vertex) send it **verbatim** with only
  `model` swapped and `stream` stripped — so `tools`, `tool_choice`, `temperature`,
  `thinking`, `stop_sequences`, and any future field survive the rung without a translation
  layer. Proven by `anthropic_wire_body_carries_raw_request_verbatim` and
  `bedrock_vertex_body_carries_raw_minus_model_plus_version`.
- **P5 — default flip, guarded.** `enforce_structured` now defaults **true**. I3's
  operator-verification burden is replaced by a structural **fidelity guard**: a structured
  request routes only when every ladder rung's provider reports
  `carries_structured_verbatim()`; a ladder containing a dialect that would need
  (not-yet-built) translation — OpenAI-compatible, Gemini — falls back to transparent
  observe passthrough (I4 preserved). Proven by
  `fidelity_guard_blocks_structured_on_non_verbatim_ladder` and the default-on halves of
  `enforce_falls_back_to_observe_for_tool_requests`.
- I1 is superseded for defaults (default behavior is now route-structured) but preserved
  under `enforce_structured = false`.

## Addendum (M1, post-P5): OpenAI-compatible inbound endpoint

- **POST /v1/chat/completions** added alongside the existing `POST /v1/messages`. The
  endpoint stamps `api: "openai.chat_completions"` on `EnforceCtx` and then runs the
  same escalation engine.
- **Fidelity guard extended.** `enforce_can_handle` now receives `inbound: Dialect` and
  evaluates two gating paths for OpenAI inbound:
  - *All-OpenAI ladder*: every rung's provider returns `carries_structured_verbatim(Dialect::Openai)`;
    the raw body is forwarded verbatim (only `model` swapped, `stream` stripped). Proven by
    `enforce_can_handle_openai_inbound_all_openai_ladder`.
  - *Translation path*: OpenAI inbound + all-Anthropic ladder + no http image URLs; the body is
    translated to the internal `ModelRequest` shape (tool_calls → tool_use blocks; role:"tool" →
    tool_result blocks; data: URI images → Anthropic base64 blocks). Proven by
    `enforce_can_handle_openai_inbound_all_anthropic_ladder_no_http_image`.
- **HTTP-image fallback.** `image_url` content with an `http(s)://` URL is non-translatable
  (Anthropic Vision needs base64 or a managed fetch). Those requests fall back to observe
  passthrough. Proven by `enforce_can_handle_openai_inbound_http_image_falls_back`.
- **Response rendering.** `openai_response_json` renders the served `ModelResponse` as a
  `chat.completion` envelope. `openai_sse_from_message` re-emits it as `chat.completion.chunk`
  SSE frames ending with `data: [DONE]` for `stream: true` clients.
- Existing `POST /v1/messages` behavior is unchanged.
