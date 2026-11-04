//! OpenAI **Responses API** (`POST /v1/responses`) as a translation over the Chat Completions
//! path.
//!
//! Newer OpenAI-family agents speak Responses rather than Chat Completions. Without this endpoint
//! they cannot point at Firstpass at all — not "route less well", but fail to connect. That is the
//! only parity gap on the list that locks a whole class of client out.
//!
//! ## Why a translation rather than a second engine
//!
//! The two APIs differ in shape, not in semantics: a list of turns in, a model reply out. Gating,
//! escalation, budgets, and receipts have nothing to do with which envelope carried the turns. So
//! this converts the request into the Chat Completions shape, runs the **existing** enforce
//! pipeline unchanged, and converts the reply back. One translation layer instead of a parallel
//! implementation that could drift on the parts that matter.
//!
//! ## Streaming
//!
//! `enforce` is inherently buffered — the gate has to see the whole candidate before it can judge
//! it — so a `stream: true` request runs the ladder to a gated answer and [`sse_from_response`]
//! then re-emits that answer as a Responses event stream: one delta per text part / argument
//! string, not token-by-token.
//!
//! ## Reasoning items are dropped, not carried
//!
//! A `reasoning` input item is the model's hidden chain of thought from a prior turn, usually
//! `encrypted_content` only the issuing OpenAI deployment can read. Chat Completions has no field
//! for it, so the translation drops it — the same history every Chat client already sends — and
//! the gate judges the new answer on its own. The served answer has no reasoning item; Codex treats
//! reasoning output as optional (`codex-api/src/sse/responses.rs` parses each
//! `output_item.done` independently; nothing waits for a reasoning item).
//!
//! ## What it does not do yet
//!
//! Stateful `previous_response_id` threading, hosted / namespaced tools, compaction items, and
//! structured `text.format` output. A request carrying any takes the passthrough (see
//! [`has_untranslatable_content`]) — unless the native path below applies.
//!
//! ## Native path (no translation)
//!
//! When every rung of the route (and its challenger) is a provider that serves `/v1/responses`
//! itself ([`crate::provider::Provider::serves_responses`]), there is nothing to translate: each
//! rung gets the client's Responses body with only `model` swapped ([`native_wire_body`]), so
//! hosted tools (`web_search`), encrypted `reasoning` items, `include`, `store`, compaction items,
//! and `text.format` all reach the model exactly as Codex sent them. The gates read the rung's
//! `output[]` (`message` text via [`output_text`]; `function_call` / `custom_tool_call` via the
//! gate layer's tool-call extraction; hosted `*_call` items are provider-side and need no
//! argument check), and the winning rung's Responses object is returned verbatim. Eligibility is
//! [`native_eligible`].
//!
//! **`previous_response_id` / `conversation` stay on passthrough.** A native turn returns a real
//! upstream response id (stored if the client asked for `store`), so a later turn threading onto it
//! is valid against OpenAI with the same credentials. Enforcing *that* later turn, though, would
//! replay the server-held thread on every rung: the history is invisible to the gates, it may have
//! been stored under a different key than the rung's (`api_key_env` on the provider), and whether
//! a stored thread's reasoning state replays on a different model is not something this proxy has
//! verified. Unsure, so conservative: those turns are relayed un-gated, as before. Codex sends
//! `store: false` and the full history every turn, so its default turns are unaffected.
//!
//! With `store: true` (or absent — the API default), every rung that was tried stores its
//! response upstream, including the ones whose answer failed the gate. They are never returned to
//! the client, so nothing threads onto them; the cost is upstream retention only.

use serde_json::{Map, Value};

/// Convert a Responses request body into a Chat Completions one.
///
/// The pieces that actually move:
/// - `instructions` becomes a leading `system` message.
/// - `input` becomes `messages`. It may be a bare string, a list of turns, or a list of turns whose
///   `content` is itself a list of typed parts — all three are in the wild.
/// - `max_output_tokens` becomes `max_tokens`.
///
/// Unknown top-level fields are carried across untouched: this is a proxy, and a field it does not
/// understand is far more likely to be newer than this code than to be wrong.
#[must_use]
pub fn request_to_chat(body: &Value) -> Value {
    let Some(obj) = body.as_object() else {
        return body.clone();
    };
    let mut out = obj.clone();

    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = obj.get("instructions").and_then(Value::as_str) {
        messages.push(serde_json::json!({ "role": "system", "content": instructions }));
    }
    match obj.get("input") {
        // `input: "..."` — the single-turn shorthand.
        Some(Value::String(s)) => {
            messages.push(serde_json::json!({ "role": "user", "content": s }));
        }
        Some(Value::Array(turns)) => {
            for turn in turns.iter().filter(|t| !is_dropped_item(t)) {
                if let Some(m) = turn_to_message(turn) {
                    push_message(&mut messages, m);
                }
            }
        }
        _ => {}
    }
    // Tool definitions: Responses puts name/parameters on the item; Chat nests them under
    // `function`.
    if let Some(Value::Array(tools)) = obj.get("tools") {
        let converted: Vec<Value> = tools.iter().filter_map(tool_to_chat).collect();
        if converted.len() == tools.len() {
            out.insert("tools".to_owned(), Value::Array(converted));
        }
    }
    // `tool_choice` naming a specific function differs in shape between the two APIs. Passing the
    // Responses spelling through unchanged does not error — Chat simply does not recognise it, so
    // the model is free to ignore a tool the caller demanded. Silent, and exactly the failure this
    // file keeps having to design against.
    if let Some(tc) = obj.get("tool_choice") {
        match tool_choice_to_chat(tc) {
            Some(v) => {
                out.insert("tool_choice".to_owned(), v);
            }
            // Unrepresentable (a hosted-tool choice). `has_untranslatable_content` routes these to
            // passthrough, so dropping here is belt-and-braces rather than the real guard.
            None => {
                out.remove("tool_choice");
            }
        }
    }
    // A caller that sent `messages` directly (some SDKs accept both) keeps them.
    if !messages.is_empty() {
        out.insert("messages".to_owned(), Value::Array(messages));
    }
    out.remove("input");
    out.remove("instructions");
    if let Some(max) = out.remove("max_output_tokens") {
        out.insert("max_tokens".to_owned(), max);
    }
    // Responses spells reasoning effort as a nested object; the provider layer normalizes dialects
    // from the flat field, so hand it the shape it already knows.
    if let Some(effort) = obj
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str)
    {
        out.insert(
            "reasoning_effort".to_owned(),
            Value::String(effort.to_owned()),
        );
    }
    // Responses-only request controls. Raw bodies are carried verbatim to OpenAI-dialect rungs
    // (`provider::openai_wire_body`), where Chat Completions rejects each of these as an unknown
    // or mistyped argument — a 400 on every rung for a turn that was perfectly answerable. None of
    // them changes what a correct answer is:
    // - `reasoning` (beyond `effort`, flattened above): `summary`/`context` shape reasoning
    //   *output*, and the Chat path returns no reasoning items. Codex sends `reasoning: null` for
    //   models without reasoning, which would otherwise ride along.
    // - `include`: asks for encrypted reasoning/output items Chat never produces.
    // - `store`: server-side retention for `previous_response_id`, which already routes to
    //   passthrough. Chat's own default is not to store.
    // - `prompt_cache_key`, `service_tier`, `client_metadata`, `stream_options`: cache routing,
    //   billing tier, telemetry, and stream framing — cost/latency knobs, not answer content.
    // - `text`: only reaches here as `verbosity` (or the no-op `format: text`); a structured
    //   `format` is routed to passthrough by `has_untranslatable_content`. Verbosity is a length
    //   hint that non-GPT-5 Chat models reject outright.
    // ponytail: stripping `prompt_cache_key` costs OpenAI rungs some prefix-cache hits; carry it
    // per-provider if a measured cache-hit drop shows up.
    for k in [
        "reasoning",
        "include",
        "store",
        "prompt_cache_key",
        "service_tier",
        "client_metadata",
        "stream_options",
        "text",
    ] {
        out.remove(k);
    }
    // Meaningful (false forbids parallel calls), but OpenAI 400s it on a request with no tools.
    // `tools: []` / `null` counts as none.
    if out
        .get("tools")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        out.remove("parallel_tool_calls");
    }
    Value::Object(out)
}

/// One Responses `input` turn → one Chat Completions message.
fn turn_to_message(turn: &Value) -> Option<Value> {
    let obj = turn.as_object()?;
    // Typed items carry no role/content pair — they are the tool round trip.
    match obj.get("type").and_then(Value::as_str) {
        Some("function_call") => return function_call_to_chat(obj),
        Some("custom_tool_call") => return custom_call_to_chat(obj),
        // Both outputs share one wire encoding (codex `protocol/src/models.rs`,
        // `CustomToolCallOutput`), and Chat has one `tool` message for either.
        Some("function_call_output" | "custom_tool_call_output") => {
            return function_output_to_chat(obj);
        }
        None | Some("message") => {}
        // Any other typed item is not a chat turn, even when it has a `content` key (Codex's
        // `agent_message` does). Falling through would translate it as a blank user message.
        Some(_) => return None,
    }
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
    let content = match obj.get("content") {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            // All-text collapses to a plain string (what a Chat client would have sent). Mixed
            // content keeps the typed-part array, with images carried across in Chat's spelling.
            // Dropping them here is not an option: `has_images` is computed from the TRANSLATED
            // body, so a dropped image also erases the signal that would have routed the request
            // to passthrough — and a multimodal request would be gated and served as text-only,
            // silently, having thrown away the thing it was asking about.
            if parts
                .iter()
                .all(|p| p.get("type").and_then(Value::as_str) != Some("input_image"))
            {
                let text: String = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("");
                Value::String(text)
            } else {
                Value::Array(parts.iter().filter_map(part_to_chat).collect())
            }
        }
        _ => return None,
    };
    Some(serde_json::json!({ "role": role, "content": content }))
}

/// Input items the translation drops on purpose, as opposed to failing to translate.
///
/// Only `reasoning`: prior hidden reasoning that Chat Completions cannot carry and that no Chat
/// client sends either (see the module docs). Everything else either translates or routes the
/// request to passthrough.
fn is_dropped_item(turn: &Value) -> bool {
    turn.get("type").and_then(Value::as_str) == Some("reasoning")
}

/// Append `m`, folding a tool-call assistant message into the assistant message before it.
///
/// Responses spells parallel calls (and text-then-call) as separate items; translated one-to-one
/// that is `assistant(call A), assistant(call B), tool(A), tool(B)`, which Chat rejects — every
/// assistant message with `tool_calls` must be followed directly by its tool results.
fn push_message(messages: &mut Vec<Value>, mut m: Value) {
    if let Some(prev) = messages.last_mut()
        && prev.get("role").and_then(Value::as_str) == Some("assistant")
        && let Some(Value::Array(calls)) = m.get_mut("tool_calls").map(Value::take)
    {
        match prev.get_mut("tool_calls") {
            Some(Value::Array(prev_calls)) => prev_calls.extend(calls),
            _ => {
                prev["tool_calls"] = Value::Array(calls);
            }
        }
        return;
    }
    messages.push(m);
}

/// One Responses tool definition → its Chat Completions spelling.
///
/// `{type: function, name, parameters}` → `{type: function, function: {name, parameters}}`.
/// Returns `None` for a tool type with no Chat equivalent (hosted tools like `web_search`), which
/// makes the caller route the whole request to passthrough rather than send a partial tool set —
/// a model missing one of its tools produces a confidently wrong plan.
fn tool_to_chat(tool: &Value) -> Option<Value> {
    let obj = tool.as_object()?;
    // Already in Chat shape (some SDKs send it): pass through.
    if obj.contains_key("function") {
        return Some(tool.clone());
    }
    match obj.get("type").and_then(Value::as_str) {
        Some("function") => {}
        Some("custom") => return custom_tool_to_chat(obj),
        _ => return None,
    }
    let mut f = serde_json::Map::new();
    f.insert("name".to_owned(), obj.get("name")?.clone());
    if let Some(d) = obj.get("description") {
        f.insert("description".to_owned(), d.clone());
    }
    if let Some(p) = obj.get("parameters") {
        f.insert("parameters".to_owned(), p.clone());
    }
    if let Some(st) = obj.get("strict") {
        f.insert("strict".to_owned(), st.clone());
    }
    Some(serde_json::json!({ "type": "function", "function": Value::Object(f) }))
}

/// The single string argument a `custom` (freeform) tool's input rides in on the Chat side.
pub(crate) const CUSTOM_INPUT_ARG: &str = "input";

/// Envelope of a Codex patch (`apply_patch` lark grammar: `start: begin_patch hunk+ end_patch`):
/// the begin marker, at least one file operation, the end marker.
/// ECMA-262 syntax (`[\s\S]`, no inline flags): providers validate tool schemas, and JSON Schema
/// `pattern` is ECMA-262, so a Rust-only construct would 400 upstream.
pub(crate) const PATCH_ENVELOPE_PATTERN: &str = r"^\s*\*\*\* Begin Patch\r?\n[\s\S]*?\*\*\* (Add|Delete|Update) File: [\s\S]*\*\*\* End Patch\s*$";

/// A Responses `custom` (freeform) tool → a Chat function tool taking one string, `input`.
///
/// Codex declares `apply_patch` this way: `{type: "custom", name, description, format: {type:
/// "grammar", syntax: "lark", definition}}` (codex `core/src/tools/handlers/apply_patch_spec.rs`).
/// Chat function tools are JSON-only, so the freeform text becomes the `input` argument and
/// [`response_from_chat`] unwraps it back into a `custom_tool_call`.
///
/// The grammar cannot be enforced as constrained decoding on the Chat side, so it travels in the
/// description instead — dropping it would hide the input contract from the model. The tool's own
/// description may say "do not wrap in JSON", which is true of the caller's API and false of this
/// one, so the appended note says which wins.
fn custom_tool_to_chat(obj: &Map<String, Value>) -> Option<Value> {
    let name = obj.get("name")?.clone();
    let mut input_pattern: Option<&str> = None;
    let mut description = obj
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if !description.is_empty() {
        description.push_str("\n\n");
    }
    description.push_str(
        "Pass the raw freeform text as the `input` string argument; it is delivered unchanged.",
    );
    match obj.get("format") {
        // Absent or `text`: unconstrained freeform text, nothing more to carry.
        None => {}
        Some(f) if f.get("type").and_then(Value::as_str) == Some("text") => {}
        Some(f) if f.get("type").and_then(Value::as_str) == Some("grammar") => {
            let syntax = f.get("syntax").and_then(Value::as_str)?;
            let definition = f.get("definition").and_then(Value::as_str)?;
            // The gate cannot run a lark grammar, and a grammar is a contract: serving an input
            // that breaks it would fail-open. Only the patch grammar (Codex `apply_patch`) is
            // translated, with its envelope as a JSON Schema `pattern` that `tool-args-valid`
            // checks, so a cheap rung's malformed patch fails the gate and escalates. Any other
            // grammar returns None → the request takes passthrough.
            // ponytail: envelope only (begin/end markers + a file op), not every hunk line.
            if !(definition.contains("\"*** Begin Patch\"")
                && definition.contains("\"*** End Patch\""))
            {
                return None;
            }
            input_pattern = Some(PATCH_ENVELOPE_PATTERN);
            description.push_str("\n\nThe `input` text must match this ");
            description.push_str(syntax);
            description.push_str(" grammar:\n");
            description.push_str(definition);
        }
        // A format this code does not know is a contract it would silently drop.
        Some(_) => return None,
    }
    let mut input_schema = serde_json::json!({ "type": "string" });
    if let Some(pat) = input_pattern {
        input_schema["pattern"] = Value::String(pat.to_owned());
    }
    Some(serde_json::json!({ "type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": { CUSTOM_INPUT_ARG: input_schema },
            "required": [CUSTOM_INPUT_ARG],
            "additionalProperties": false,
        },
    }}))
}

/// Names of the request's `custom` (freeform) tools: a served call to one of these must go back
/// as a `custom_tool_call`, not a `function_call` — Codex dispatches `apply_patch` only on the
/// custom payload. Empty (no allocation) for a request without custom tools.
#[must_use]
pub fn custom_tool_names(body: &Value) -> Vec<String> {
    body.get("tools")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str) == Some("custom"))
        .filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect()
}

/// A Responses `tool_choice` → its Chat spelling.
///
/// The string forms (`auto` / `none` / `required`) are identical. Naming a specific function is
/// not: Responses says `{type: "function", name: "f"}`, Chat says
/// `{type: "function", function: {name: "f"}}`.
fn tool_choice_to_chat(tc: &Value) -> Option<Value> {
    match tc {
        Value::String(_) => Some(tc.clone()),
        Value::Object(o) => {
            if o.contains_key("function") {
                return Some(tc.clone()); // already Chat-shaped
            }
            // A custom tool is a function on the Chat side (`custom_tool_to_chat`).
            if !matches!(
                o.get("type").and_then(Value::as_str),
                Some("function" | "custom")
            ) {
                return None; // hosted-tool choice — no Chat equivalent
            }
            Some(serde_json::json!({
                "type": "function", "function": { "name": o.get("name")?.clone() }
            }))
        }
        _ => None,
    }
}

/// A Responses tool-call item → the Chat `assistant` message that carries it.
fn function_call_to_chat(item: &serde_json::Map<String, Value>) -> Option<Value> {
    // A namespaced call (`namespace` + `name`, codex `ResponseItem::FunctionCall`) names a tool
    // inside a `namespace` tool, which has no Chat spelling; the bare name alone could resolve to
    // a different tool.
    if item.get("namespace").is_some_and(|n| !n.is_null()) {
        return None;
    }
    Some(serde_json::json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": [{
            "id": item.get("call_id").or_else(|| item.get("id"))?.clone(),
            "type": "function",
            "function": {
                "name": item.get("name")?.clone(),
                // Chat wants a JSON string; Responses already sends one.
                "arguments": item.get("arguments").cloned()
                    .unwrap_or_else(|| Value::String("{}".to_owned())),
            },
        }],
    }))
}

/// A Responses `custom_tool_call` (freeform input, e.g. Codex's `apply_patch`) → the Chat
/// `assistant` tool call that [`custom_tool_to_chat`]'s function spelling would have produced.
fn custom_call_to_chat(item: &serde_json::Map<String, Value>) -> Option<Value> {
    if item.get("namespace").is_some_and(|n| !n.is_null()) {
        return None;
    }
    let input = item.get("input")?.as_str()?;
    Some(serde_json::json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": [{
            "id": item.get("call_id")?.clone(),
            "type": "function",
            "function": {
                "name": item.get("name")?.clone(),
                "arguments": serde_json::json!({ CUSTOM_INPUT_ARG: input }).to_string(),
            },
        }],
    }))
}

/// A Responses tool-result item (`function_call_output` or `custom_tool_call_output`) → the Chat
/// `tool` message that carries it.
///
/// `output` is a string, or a list of content items (codex `FunctionCallOutputPayload`). An
/// all-text list joins to its text. Anything else in the list — an image from Codex's
/// `view_image`, audio, an `encrypted_content` item — has no place in a Chat `tool` message, so it
/// translates to nothing and the request takes passthrough; stringifying it would hand the model a
/// base64 blob as "text".
fn function_output_to_chat(item: &serde_json::Map<String, Value>) -> Option<Value> {
    let output = match item.get("output")? {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut text = String::new();
            for p in parts {
                if !matches!(
                    p.get("type").and_then(Value::as_str),
                    Some("input_text" | "output_text" | "text")
                ) {
                    return None;
                }
                text.push_str(p.get("text").and_then(Value::as_str)?);
            }
            text
        }
        other => other.to_string(),
    };
    Some(serde_json::json!({
        "role": "tool",
        "tool_call_id": item.get("call_id")?.clone(),
        "content": output,
    }))
}

/// One Responses content part → its Chat Completions spelling.
fn part_to_chat(part: &Value) -> Option<Value> {
    match part.get("type").and_then(Value::as_str) {
        Some("input_text" | "output_text" | "text") => Some(serde_json::json!({
            "type": "text",
            "text": part.get("text").and_then(Value::as_str).unwrap_or_default(),
        })),
        Some("input_image") => {
            // Responses puts the URL directly on the part; Chat nests it under `image_url`.
            let url = part.get("image_url").and_then(|u| {
                u.as_str()
                    .map(str::to_owned)
                    .or_else(|| u.get("url").and_then(Value::as_str).map(str::to_owned))
            })?;
            Some(serde_json::json!({ "type": "image_url", "image_url": { "url": url } }))
        }
        _ => None,
    }
}

/// Whether this request contains anything the translation cannot faithfully round-trip.
///
/// **Allow-list, not deny-list, and deliberately so.** Three separate content types were found to
/// be silently dropped by an earlier deny-list version of this check — images, files, and tool
/// calls — each producing a confidently wrong answer rather than an error. Enumerating what breaks
/// loses that game every time a provider adds a field; enumerating what provably round-trips does
/// not. Anything outside the list takes the passthrough path un-gated.
///
/// Un-gated is a real limitation. Being answered about content that was thrown away is a wrong
/// answer, and between the two this proxy takes the limitation.
///
/// Currently translatable: plain `user`/`assistant`/`system`/`developer` turns whose content is
/// text or text-and-image parts, function and `custom` (freeform) tool definitions, the
/// `function_call` / `function_call_output` and `custom_tool_call` / `custom_tool_call_output`
/// round trips — in both directions, so a tool call in the reply comes back as the item type its
/// tool was declared as rather than vanishing — and `reasoning` items, which are dropped on purpose
/// (see the module docs).
///
/// Still excluded: **hosted** tools (`web_search`, `file_search`, `computer_use`), which run inside
/// the provider and have no Chat equivalent; `namespace` / `tool_search` tools; `compaction` and
/// other typed items (a compaction item IS the compressed history — dropping it would answer a
/// conversation with its past erased); tool outputs carrying images; a structured `text.format`;
/// and `previous_response_id` threading, where the upstream holds history we do not have, so a
/// translated request is not the conversation the client is continuing.
#[must_use]
pub fn has_untranslatable_content(body: &Value) -> bool {
    // A served call is emitted as `custom_tool_call` by name. If the same name is also declared
    // as a function tool, the reply type would be a guess, and Codex dispatches on the type.
    let custom = custom_tool_names(body);
    if !custom.is_empty()
        && body
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|ts| {
                ts.iter().any(|t| {
                    t.get("type").and_then(Value::as_str) == Some("function")
                        && t.get("name")
                            .or_else(|| t.pointer("/function/name"))
                            .and_then(Value::as_str)
                            .is_some_and(|n| custom.iter().any(|c| c == n))
                })
            })
    {
        return true;
    }
    // Hosted tools (web_search, file_search, computer_use) run inside the provider and have no
    // Chat Completions equivalent. Sending a partial tool set is worse than not routing: a model
    // missing one of its tools produces a confidently wrong plan.
    if let Some(Value::Array(tools)) = body.get("tools")
        && tools.iter().any(|t| tool_to_chat(t).is_none())
    {
        return true;
    }
    // A `tool_choice` we cannot express would let the model ignore a tool the caller demanded.
    if let Some(tc) = body.get("tool_choice")
        && tool_choice_to_chat(tc).is_none()
    {
        return true;
    }
    // Stateful threading: the upstream holds history we do not have, so a translated request is
    // not the conversation the client is actually continuing. `conversation` is the newer,
    // mutually exclusive form of the same server-side state.
    // An explicit `store: true` asks the upstream to persist this response for a later
    // `previous_response_id` turn; a synthetic reply built from Chat has nothing stored, so that
    // continuation would fail or thread onto the wrong state. Codex sends `store: false`.
    // ponytail: an absent `store` (API default true) stays translated, as before this change;
    // a client that omits it and then threads by id gets passthrough on that next turn anyway.
    if body.get("previous_response_id").is_some()
        || body.get("conversation").is_some()
        || body.get("store").and_then(Value::as_bool) == Some(true)
    {
        return true;
    }
    // A structured-output contract (`codex exec --output-schema` sends `text.format:
    // {type: json_schema, ...}`). Chat spells it `response_format`, which the Anthropic
    // translation path does not carry — dropped, the schema the caller demanded silently vanishes.
    if let Some(f) = body.pointer("/text/format")
        && f.get("type").and_then(Value::as_str) != Some("text")
    {
        return true;
    }
    let Some(Value::Array(turns)) = body.get("input") else {
        return false;
    };
    // Nothing left once reasoning items are dropped: an empty Chat request would 400.
    if !turns.is_empty() && turns.iter().all(is_dropped_item) {
        return true;
    }
    turns.iter().any(|turn| {
        if is_dropped_item(turn) {
            return false;
        }
        let Some(obj) = turn.as_object() else {
            return true;
        };
        // Typed items: ask the translator itself rather than restating its rules here. A
        // `function_call` missing its `call_id` translates to nothing, and a check that only
        // looked at `type` would wave it through and let the turn vanish — the exact silent drop
        // this allow-list exists to prevent. One source of truth cannot drift from itself.
        if obj
            .get("type")
            .is_some_and(|t| t.as_str() != Some("message"))
        {
            return turn_to_message(turn).is_none();
        }
        if !matches!(
            obj.get("role").and_then(Value::as_str),
            Some("user" | "assistant" | "system" | "developer")
        ) {
            return true;
        }
        match obj.get("content") {
            Some(Value::String(_)) => false,
            Some(Value::Array(parts)) => parts.iter().any(|p| {
                match p.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text" | "text") => false,
                    // Ask the translator: an image by `file_id` has no URL and would be dropped.
                    Some("input_image") => part_to_chat(p).is_none(),
                    _ => true,
                }
            }),
            _ => true,
        }
    })
}

/// Hosted tools: executed inside the provider, so their `*_call` output items are already-run
/// provider work, not calls the client must make — the gate has no arguments to validate.
/// Allow-list (see [`has_untranslatable_content`] for why): `mcp`, `computer_use_preview`,
/// `local_shell` (client-executed despite the name) and anything newer stay on passthrough.
const NATIVE_HOSTED_TOOLS: &[&str] = &[
    "web_search",
    "web_search_preview",
    "file_search",
    "code_interpreter",
    "image_generation",
];

/// Whether a Responses request can take the native enforce path (forwarded unchanged to a rung
/// that serves `/v1/responses`). The ladder half of the decision is the caller's
/// (`Provider::serves_responses`); this is the request half.
///
/// Every tool must be either one the gate can validate ([`tool_to_chat`] translates it) or a
/// hosted tool from [`NATIVE_HOSTED_TOOLS`]. Excluded: `previous_response_id` / `conversation`
/// threading (module docs: server-held history the gate cannot see, and cross-model replay is
/// unverified), and `background: true` (the reply is a queued shell, not an answer to gate).
#[must_use]
pub fn native_eligible(body: &Value) -> bool {
    let Some(obj) = body.as_object() else {
        return false;
    };
    if obj
        .get("previous_response_id")
        .is_some_and(|v| !v.is_null())
        || obj.get("conversation").is_some_and(|v| !v.is_null())
        || obj.get("background").and_then(Value::as_bool) == Some(true)
    {
        return false;
    }
    obj.get("tools")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .all(|t| {
            tool_to_chat(t).is_some()
                || t.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|ty| NATIVE_HOSTED_TOOLS.contains(&ty))
        })
}

/// The tool list the gates validate a native turn against: the request's function and custom tools
/// in Chat spelling (a `custom` patch tool carries its envelope `pattern`, as on the translated
/// path), hosted tools left out — they have no client-side arguments.
#[must_use]
pub fn native_gate_tools(body: &Value) -> Value {
    match body.get("tools").and_then(Value::as_array) {
        Some(tools) => Value::Array(tools.iter().filter_map(tool_to_chat).collect()),
        None => Value::Null,
    }
}

/// Whether a raw request body is Responses-shaped (`input`, no `messages`). Only the native path
/// puts one on `ModelRequest::raw`; the Chat path's raw body always has `messages` (it fails to
/// parse without them), so the two cannot be confused.
#[must_use]
pub fn is_responses_body(raw: &Value) -> bool {
    raw.get("input").is_some() && raw.get("messages").is_none()
}

/// The wire body for one native rung: the client's Responses body byte-for-byte except `model`
/// (the rung's bare id) and the stream controls — the enforce path buffers, so the upstream call is
/// non-streaming and `stream_options` without `stream` would be rejected. `reasoning` items,
/// hosted tools, `include`, `store`, `text`, and every unknown field are untouched.
#[must_use]
pub fn native_wire_body(raw: &Map<String, Value>, wire_model: &str) -> Value {
    let mut body = raw.clone();
    body.insert("model".to_owned(), Value::String(wire_model.to_owned()));
    body.remove("stream");
    body.remove("stream_options");
    Value::Object(body)
}

/// The text a Responses object answered with: every `output_text` part of every `message` item,
/// joined. (`output_text` at the top level is an SDK convenience the wire does not send.)
#[must_use]
pub fn output_text(resp: &Value) -> String {
    let mut text = String::new();
    for item in resp
        .get("output")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
        .iter()
        .filter(|i| i.get("type").and_then(Value::as_str) == Some("message"))
    {
        for part in item
            .get("content")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
        {
            if part.get("type").and_then(Value::as_str) == Some("output_text")
                && let Some(t) = part.get("text").and_then(Value::as_str)
            {
                text.push_str(t);
            }
        }
    }
    text
}

/// The client-executed tool calls in a Responses object's `output`, as the gate layer's
/// `(name, json_args)` pairs: a `function_call`'s `arguments` string, or a `custom_tool_call`'s
/// freeform `input` wrapped as `{"input": ...}` — the shape [`native_gate_tools`] declares for it.
/// Hosted `*_call` items are not included (provider-side, nothing to validate). `None` when `raw`
/// is not a Responses object, so callers can fall through to the other dialects.
#[must_use]
pub fn output_tool_calls(raw: &Value) -> Option<Vec<(String, String)>> {
    if raw.get("object").and_then(Value::as_str) != Some("response") {
        return None;
    }
    let items = raw.get("output").and_then(Value::as_array)?;
    let name = |i: &Value| {
        i.get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    Some(
        items
            .iter()
            .filter_map(|i| match i.get("type").and_then(Value::as_str) {
                // Missing/non-string arguments → "" fails the gate's JSON parse, not a fake `{}`.
                Some("function_call") => Some((
                    name(i),
                    i.get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                )),
                Some("custom_tool_call") => Some((
                    name(i),
                    i.get("input")
                        .and_then(Value::as_str)
                        .map(|s| serde_json::json!({ CUSTOM_INPUT_ARG: s }).to_string())
                        .unwrap_or_default(),
                )),
                _ => None,
            })
            .collect(),
    )
}

/// Normalize a native rung's Responses object into the provider-agnostic response the router and
/// gates consume. `raw` is kept whole — it is what the client is served.
///
/// ponytail: `in_tokens` includes cached input tokens (no cache split), matching the Chat-path
/// `OpenAiProvider`; report `input_tokens_details.cached_tokens` once OpenAI cache pricing is in
/// the price table, or Codex's heavily cached turns are over-costed on the receipt.
#[must_use]
pub fn model_response(model: &str, raw: Value) -> crate::provider::ModelResponse {
    let usage = |k: &str| {
        raw.pointer(&format!("/usage/{k}"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    crate::provider::ModelResponse {
        model: model.to_owned(),
        text: output_text(&raw),
        in_tokens: usage("input_tokens"),
        cache_write_tokens: 0,
        cache_read_tokens: 0,
        out_tokens: usage("output_tokens"),
        raw,
    }
}

/// Convert a Chat Completions response into the Responses shape.
///
/// Reports `usage` with Responses' own field names (`input_tokens` / `output_tokens`), which is
/// what a client reads to display cost — leaving Chat's `prompt_tokens` spelling would make a
/// served request look like it consumed nothing.
///
/// `custom_tools` ([`custom_tool_names`] of the original request) are the tools declared
/// `custom`: a call to one comes back as a `custom_tool_call` with the freeform `input`, the only
/// shape the caller's tool accepts.
#[must_use]
pub fn response_from_chat(chat: &Value, custom_tools: &[String]) -> Value {
    let text = chat
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let model = chat
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let id = chat.get("id").and_then(Value::as_str).unwrap_or("resp_0");
    let in_tokens = chat
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let out_tokens = chat
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let mut resp = Map::new();
    resp.insert("id".to_owned(), Value::String(id.to_owned()));
    resp.insert("object".to_owned(), Value::String("response".to_owned()));
    // Required by the SDK's `Response` type; Chat's `created` is the same unix-seconds instant.
    resp.insert(
        "created_at".to_owned(),
        chat.get("created").cloned().unwrap_or(Value::from(0)),
    );
    resp.insert("model".to_owned(), Value::String(model.to_owned()));

    // A reply may carry text, tool calls, or both. Dropping the tool calls here would hand an
    // agent a plausible-looking message where it expected a call to make — the failure being
    // silent is what makes it dangerous.
    let mut output: Vec<Value> = Vec::new();
    if !text.is_empty() {
        output.push(serde_json::json!({
            "type": "message",
            "id": format!("msg_{id}"),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
        }));
    }
    let tool_calls = chat
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array);
    if let Some(calls) = tool_calls {
        for (i, call) in calls.iter().enumerate() {
            let name = call.pointer("/function/name").and_then(Value::as_str);
            if let Some(name) = name.filter(|n| custom_tools.iter().any(|c| c == n)) {
                let args = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                output.push(serde_json::json!({
                    "type": "custom_tool_call",
                    "id": format!("ctc_{id}_{i}"),
                    "status": "completed",
                    "call_id": call.get("id").cloned().unwrap_or(Value::Null),
                    "name": name,
                    "input": custom_input(args),
                }));
                continue;
            }
            output.push(serde_json::json!({
                "type": "function_call",
                "id": format!("fc_{id}_{i}"),
                "status": "completed",
                "call_id": call.get("id").cloned().unwrap_or(Value::Null),
                "name": call.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "arguments": call
                    .pointer("/function/arguments")
                    .cloned()
                    .unwrap_or_else(|| Value::String("{}".to_owned())),
            }));
        }
    }
    // A reply with neither text nor calls still needs a well-formed message item.
    if output.is_empty() {
        output.push(serde_json::json!({
            "type": "message",
            "id": format!("msg_{id}"),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "", "annotations": [] }],
        }));
    }
    // `completed` means GENERATION finished, not that the agent loop is done — a response is
    // `completed` even when `output` carries a `function_call` the caller still has to run. The
    // Responses API reserves `incomplete` for a TRUNCATED response, with `incomplete_details`
    // naming the reason.
    //
    // An earlier version of this reported a tool-call turn as `incomplete` by analogy: the model
    // is waiting, so surely it is not complete. That is wrong against the spec and would stall any
    // client that checks `status == "completed"` before reading `output`.
    let truncated = chat
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        == Some("length");
    if truncated {
        resp.insert("status".to_owned(), Value::String("incomplete".to_owned()));
        resp.insert(
            "incomplete_details".to_owned(),
            serde_json::json!({ "reason": "max_output_tokens" }),
        );
    } else {
        resp.insert("status".to_owned(), Value::String("completed".to_owned()));
    }
    resp.insert("output".to_owned(), Value::Array(output));
    // The SDK convenience accessor. Populating it means `response.output_text` works rather than
    // returning nothing on a response this proxy produced.
    resp.insert("output_text".to_owned(), Value::String(text.to_owned()));
    resp.insert(
        "usage".to_owned(),
        serde_json::json!({
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        }),
    );
    Value::Object(resp)
}

/// The freeform `input` of a served custom-tool call: the `input` string out of the Chat
/// arguments `{"input": "..."}`.
///
/// Arguments that are not that shape (a model that ignored the wrapper and emitted the patch
/// itself) are delivered as-is rather than invented around. A `tool-args-valid` gate fails these
/// before they are served; without one, the caller's tool sees the raw text and reports its own
/// parse error to the agent loop — visible, not silent.
fn custom_input(args: &str) -> String {
    match serde_json::from_str::<Value>(args) {
        Ok(Value::Object(mut m)) => match m.remove(CUSTOM_INPUT_ARG) {
            Some(Value::String(s)) => s,
            _ => args.to_owned(),
        },
        _ => args.to_owned(),
    }
}

/// Append one `event: <type>` SSE frame: `data` (a JSON object) stamped with `type` and the next
/// `sequence_number`.
fn push_event(out: &mut String, seq: &mut u64, kind: &str, data: Value) {
    let mut data = match data {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    data.insert("type".to_owned(), Value::String(kind.to_owned()));
    data.insert("sequence_number".to_owned(), Value::from(*seq));
    *seq += 1;
    out.push_str("event: ");
    out.push_str(kind);
    out.push_str("\ndata: ");
    out.push_str(&Value::Object(data).to_string());
    out.push_str("\n\n");
}

/// The `response` carried by `response.created` / `response.in_progress`: the final object's
/// identity (`id`, `model`, `created_at`) with nothing generated yet. The SDK's stream accumulator
/// raises if a lifecycle event's `response.id` differs from the one `response.created` announced.
fn in_progress_snapshot(resp: &Value) -> Value {
    let mut snap = resp.clone();
    if let Some(m) = snap.as_object_mut() {
        m.insert("status".to_owned(), "in_progress".into());
        m.insert("output".to_owned(), Value::Array(Vec::new()));
        m.insert("output_text".to_owned(), "".into());
        m.insert("usage".to_owned(), Value::Null);
        m.remove("incomplete_details");
    }
    snap
}

/// Re-emit a complete Responses object (from [`response_from_chat`]) as the Responses SSE event
/// sequence the OpenAI SDK's stream accumulator and Codex's `process_responses_event` parse.
///
/// `response.created` → `response.in_progress` → per output item: `response.output_item.added`;
/// for a `message`, per content part `response.content_part.added` → `response.output_text.delta`
/// → `response.output_text.done` → `response.content_part.done`; for a `function_call`,
/// `response.function_call_arguments.delta` → `.done` (for a `custom_tool_call`,
/// `response.custom_tool_call_input.delta` → `.done`); then `response.output_item.done` → finally
/// `response.completed` (`response.incomplete` for a truncated answer) carrying the full object
/// with `usage`. Event item ids are the final object's ids: the SDK raises if an item's identity
/// changes between `added`, `done`, and `completed`.
///
/// Any other item type (none today — the Chat path has no reasoning items) passes through as an
/// `output_item.added` / `.done` pair rather than being dropped.
#[must_use]
pub fn sse_from_response(resp: &Value) -> String {
    use serde_json::json;
    let mut out = String::new();
    let mut seq = 0_u64;
    let snapshot = in_progress_snapshot(resp);
    for kind in ["response.created", "response.in_progress"] {
        push_event(&mut out, &mut seq, kind, json!({ "response": snapshot }));
    }
    let items = resp
        .get("output")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for (output_index, item) in items.iter().enumerate() {
        let item_id = item.get("id").cloned().unwrap_or(Value::Null);
        let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
        // The `added` item is the in-progress shell; its content arrives in the events after it.
        let mut shell = item.clone();
        if let Some(m) = shell.as_object_mut() {
            match kind {
                "message" => {
                    m.insert("content".to_owned(), json!([]));
                }
                "function_call" => {
                    m.insert("arguments".to_owned(), "".into());
                }
                // Codex parses the `added` item as a `ResponseItem`, whose `input` is required;
                // an item that fails to parse is dropped with only a debug log.
                "custom_tool_call" => {
                    m.insert("input".to_owned(), "".into());
                }
                _ => {}
            }
            if m.contains_key("status") {
                m.insert("status".to_owned(), "in_progress".into());
            }
        }
        push_event(
            &mut out,
            &mut seq,
            "response.output_item.added",
            json!({ "output_index": output_index, "item": shell }),
        );
        match kind {
            "message" => {
                let parts = item
                    .get("content")
                    .and_then(Value::as_array)
                    .map_or(&[][..], Vec::as_slice);
                for (content_index, part) in parts.iter().enumerate() {
                    let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                    let mut empty = part.clone();
                    if let Some(m) = empty.as_object_mut() {
                        m.insert("text".to_owned(), "".into());
                    }
                    let at = |mut extra: Value| {
                        if let Some(m) = extra.as_object_mut() {
                            m.insert("item_id".to_owned(), item_id.clone());
                            m.insert("output_index".to_owned(), output_index.into());
                            m.insert("content_index".to_owned(), content_index.into());
                        }
                        extra
                    };
                    let kinds_and_data = [
                        ("response.content_part.added", json!({ "part": empty })),
                        (
                            "response.output_text.delta",
                            json!({ "delta": text, "logprobs": [] }),
                        ),
                        (
                            "response.output_text.done",
                            json!({ "text": text, "logprobs": [] }),
                        ),
                        ("response.content_part.done", json!({ "part": part })),
                    ];
                    for (k, data) in kinds_and_data {
                        // An empty part has no delta to send; `done` still carries its (empty) text.
                        if k == "response.output_text.delta" && text.is_empty() {
                            continue;
                        }
                        push_event(&mut out, &mut seq, k, at(data));
                    }
                }
            }
            "function_call" | "custom_tool_call" => {
                // Same framing, different names: a custom call streams its freeform `input`
                // (`response.custom_tool_call_input.*`, which Codex's `process_responses_event`
                // turns into `ToolCallInputDelta`) where a function call streams `arguments`.
                let (field, delta_kind, done_kind) = if kind == "custom_tool_call" {
                    (
                        "input",
                        "response.custom_tool_call_input.delta",
                        "response.custom_tool_call_input.done",
                    )
                } else {
                    (
                        "arguments",
                        "response.function_call_arguments.delta",
                        "response.function_call_arguments.done",
                    )
                };
                let args = item.get(field).and_then(Value::as_str).unwrap_or_default();
                for (k, data) in [
                    (delta_kind, json!({ "delta": args })),
                    (done_kind, json!({ field: args })),
                ] {
                    let mut data = data;
                    if let Some(m) = data.as_object_mut() {
                        m.insert("item_id".to_owned(), item_id.clone());
                        m.insert("output_index".to_owned(), output_index.into());
                    }
                    push_event(&mut out, &mut seq, k, data);
                }
            }
            _ => {}
        }
        push_event(
            &mut out,
            &mut seq,
            "response.output_item.done",
            json!({ "output_index": output_index, "item": item }),
        );
    }
    let terminal = if resp.get("status").and_then(Value::as_str) == Some("incomplete") {
        "response.incomplete"
    } else {
        "response.completed"
    };
    push_event(&mut out, &mut seq, terminal, json!({ "response": resp }));
    out
}

/// A Responses stream that failed before any answer could be served (the ladder exhausted, a
/// budget tripped): `response.created` then `response.failed`, both under `id`. The SDK requires
/// `response.created` first; Codex reads `response.error.message` from `response.failed`.
#[must_use]
pub fn sse_failed(id: &str, message: &str) -> String {
    let mut out = String::new();
    let mut seq = 0_u64;
    let base = serde_json::json!({
        "id": id, "object": "response", "created_at": 0, "model": "", "output": [],
        "usage": null, "error": null,
    });
    let snapshot = in_progress_snapshot(&base);
    push_event(
        &mut out,
        &mut seq,
        "response.created",
        serde_json::json!({ "response": snapshot }),
    );
    let mut failed = base;
    failed["status"] = "failed".into();
    failed["error"] = serde_json::json!({ "code": "server_error", "message": message });
    push_event(
        &mut out,
        &mut seq,
        "response.failed",
        serde_json::json!({ "response": failed }),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Split an SSE body into its `data` objects, asserting each frame's `event:` line names the
    /// same type as its payload and that `sequence_number` counts 0, 1, 2, … with no gap.
    fn frames(sse: &str) -> Vec<Value> {
        let out: Vec<Value> = sse
            .split("\n\n")
            .filter(|f| !f.is_empty())
            .map(|f| {
                let (ev, data) = f.split_once('\n').unwrap();
                let v: Value = serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap();
                assert_eq!(ev.strip_prefix("event: "), v["type"].as_str());
                v
            })
            .collect();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v["sequence_number"], i, "{v}");
        }
        out
    }

    fn types(events: &[Value]) -> Vec<&str> {
        events.iter().map(|e| e["type"].as_str().unwrap()).collect()
    }

    fn chat(content: Value, tool_calls: Option<Value>) -> Value {
        let mut msg = serde_json::json!({ "role": "assistant", "content": content });
        if let Some(tc) = tool_calls {
            msg["tool_calls"] = tc;
        }
        serde_json::json!({
            "id": "chatcmpl-1", "created": 1_700_000_000, "model": "openai/gpt-5.5",
            "choices": [{ "index": 0, "message": msg, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18 },
        })
    }

    fn call() -> Value {
        serde_json::json!([{ "id": "call_1", "type": "function",
            "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } }])
    }

    const TEXT_EVENTS: [&str; 6] = [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ];
    const CALL_EVENTS: [&str; 4] = [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
    ];

    fn expected(items: &[&[&str]]) -> Vec<String> {
        let mut v = vec!["response.created", "response.in_progress"];
        for i in items {
            v.extend_from_slice(i);
        }
        v.push("response.completed");
        v.into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn text_response_streams_the_exact_event_sequence() {
        let resp = response_from_chat(&chat(Value::from("hello world"), None), &[]);
        let ev = frames(&sse_from_response(&resp));
        assert_eq!(types(&ev), expected(&[&TEXT_EVENTS]));
        // Lifecycle events share the final response id; created/in_progress carry no output yet.
        for e in [&ev[0], &ev[1]] {
            assert_eq!(e["response"]["id"], "chatcmpl-1");
            assert_eq!(e["response"]["status"], "in_progress");
            assert_eq!(e["response"]["output"], serde_json::json!([]));
        }
        let item_id = ev[2]["item"]["id"].clone();
        assert_eq!(item_id, "msg_chatcmpl-1");
        assert_eq!(ev[2]["item"]["content"], serde_json::json!([]));
        for e in &ev[3..7] {
            assert_eq!(e["item_id"], item_id);
            assert_eq!(e["output_index"], 0);
            assert_eq!(e["content_index"], 0);
        }
        assert_eq!(ev[4]["delta"], "hello world");
        assert_eq!(ev[5]["text"], "hello world");
        assert_eq!(ev[6]["part"]["text"], "hello world");
        assert_eq!(ev[7]["item"], resp["output"][0]);
        let done = &ev[8]["response"];
        assert_eq!(done["status"], "completed");
        assert_eq!(done["created_at"], 1_700_000_000);
        assert_eq!(
            done["usage"],
            serde_json::json!({ "input_tokens": 11, "output_tokens": 7, "total_tokens": 18 })
        );
        assert_eq!(done["output"][0]["content"][0]["text"], "hello world");
    }

    #[test]
    fn function_call_response_streams_its_arguments() {
        let resp = response_from_chat(&chat(Value::Null, Some(call())), &[]);
        let ev = frames(&sse_from_response(&resp));
        assert_eq!(types(&ev), expected(&[&CALL_EVENTS]));
        let added = &ev[2]["item"];
        assert_eq!(added["type"], "function_call");
        assert_eq!(added["call_id"], "call_1");
        assert_eq!(added["name"], "get_weather");
        assert_eq!(added["arguments"], "");
        for e in &ev[3..5] {
            assert_eq!(e["item_id"], added["id"]);
            assert_eq!(e["output_index"], 0);
        }
        assert_eq!(ev[3]["delta"], r#"{"city":"Paris"}"#);
        assert_eq!(ev[4]["arguments"], r#"{"city":"Paris"}"#);
        assert_eq!(ev[5]["item"]["arguments"], r#"{"city":"Paris"}"#);
        assert_eq!(ev[6]["response"]["usage"]["total_tokens"], 18);
    }

    #[test]
    fn text_then_function_call_use_consecutive_output_indexes() {
        let resp = response_from_chat(&chat(Value::from("checking"), Some(call())), &[]);
        let ev = frames(&sse_from_response(&resp));
        assert_eq!(types(&ev), expected(&[&TEXT_EVENTS, &CALL_EVENTS]));
        assert_eq!(ev[2]["output_index"], 0);
        assert_eq!(ev[8]["output_index"], 1);
        assert_eq!(ev[8]["item"]["type"], "function_call");
        let done = &ev[12]["response"];
        assert_eq!(done["output"].as_array().unwrap().len(), 2);
        assert_eq!(done["usage"]["input_tokens"], 11);
        assert_eq!(done["usage"]["output_tokens"], 7);
    }

    #[test]
    fn truncated_response_ends_with_incomplete_not_completed() {
        let mut c = chat(Value::from("cut"), None);
        c["choices"][0]["finish_reason"] = "length".into();
        let ev = frames(&sse_from_response(&response_from_chat(&c, &[])));
        assert_eq!(ev.last().unwrap()["type"], "response.incomplete");
    }

    #[test]
    fn non_message_items_pass_through_as_added_and_done() {
        let resp = serde_json::json!({ "id": "r", "status": "completed", "output": [
            { "type": "reasoning", "id": "rs_1", "summary": [] } ] });
        let ev = frames(&sse_from_response(&resp));
        assert_eq!(
            types(&ev),
            expected(&[&["response.output_item.added", "response.output_item.done"]])
        );
        assert_eq!(ev[3]["item"]["id"], "rs_1");
    }

    #[test]
    fn failure_is_created_then_failed_under_one_id() {
        let ev = frames(&sse_failed("resp_x", "all rungs failed"));
        assert_eq!(types(&ev), ["response.created", "response.failed"]);
        assert_eq!(ev[0]["response"]["id"], "resp_x");
        assert_eq!(ev[1]["response"]["id"], "resp_x");
        assert_eq!(ev[1]["response"]["status"], "failed");
        assert_eq!(ev[1]["response"]["error"]["code"], "server_error");
        assert_eq!(ev[1]["response"]["error"]["message"], "all rungs failed");
    }

    #[test]
    fn the_single_turn_string_shorthand_becomes_a_user_message() {
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5", "input": "write a haiku",
        }));
        assert_eq!(out["messages"][0]["role"], "user");
        assert_eq!(out["messages"][0]["content"], "write a haiku");
        assert!(
            out.get("input").is_none(),
            "the Responses field must not leak downstream"
        );
    }

    #[test]
    fn instructions_become_a_leading_system_message() {
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5",
            "instructions": "be terse",
            "input": [{ "role": "user", "content": "hi" }],
        }));
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][0]["content"], "be terse");
        assert_eq!(out["messages"][1]["role"], "user");
        assert!(out.get("instructions").is_none());
    }

    #[test]
    fn an_image_survives_translation_instead_of_being_silently_dropped() {
        // Caught in review. `has_images` is computed from the TRANSLATED body, so dropping the
        // image here also erased the signal that routes multimodal requests to passthrough — a
        // request with a picture would be gated and answered as text-only, having discarded the
        // very thing it asked about.
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5",
            "input": [{ "role": "user", "content": [
                { "type": "input_text", "text": "what is this?" },
                { "type": "input_image", "image_url": "data:image/png;base64,AAAA" },
            ]}],
        }));

        let parts = out["messages"][0]["content"]
            .as_array()
            .expect("mixed content stays a typed array");
        assert_eq!(parts.len(), 2, "{parts:?}");
        assert_eq!(parts[0]["type"], "text");
        // Chat nests the url; Responses puts it directly on the part.
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn content_we_cannot_represent_is_flagged_for_passthrough() {
        // A file part has no Chat equivalent here. Translating it would quietly produce a smaller
        // request than the client sent, so the caller routes it to passthrough instead.
        let with_file = serde_json::json!({
            "input": [{ "role": "user", "content": [
                { "type": "input_text", "text": "summarise" },
                { "type": "input_file", "file_id": "file_123" },
            ]}],
        });
        assert!(has_untranslatable_content(&with_file));

        let text_and_image = serde_json::json!({
            "input": [{ "role": "user", "content": [
                { "type": "input_text", "text": "hi" },
                { "type": "input_image", "image_url": "data:..." },
            ]}],
        });
        assert!(
            !has_untranslatable_content(&text_and_image),
            "text and images ARE representable — they must still be gated"
        );
        assert!(!has_untranslatable_content(
            &serde_json::json!({ "input": "plain" })
        ));
    }

    #[test]
    fn a_tool_round_trip_survives_in_both_directions() {
        // This is what unblocks gating for agentic clients. Previously the whole request took
        // un-gated passthrough because a tool call had no representation coming back.
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5",
            "tools": [{
                "type": "function", "name": "get_weather",
                "description": "look up weather",
                "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
            }],
            "input": [
                { "role": "user", "content": "weather in Paris?" },
                { "type": "function_call", "call_id": "call_1", "name": "get_weather",
                  "arguments": "{\"city\":\"Paris\"}" },
                { "type": "function_call_output", "call_id": "call_1", "output": "22C" },
            ],
        }));

        // Tool definition: Responses puts name/parameters on the item, Chat nests under `function`.
        assert_eq!(out["tools"][0]["type"], "function");
        assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(out["tools"][0]["function"]["parameters"]["type"], "object");

        let msgs = out["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
        // The tool result becomes a `tool` message keyed by the call id.
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_1");
        assert_eq!(msgs[2]["content"], "22C");
    }

    #[test]
    fn a_tool_call_in_the_reply_comes_back_as_a_function_call_item() {
        // Dropping this would hand an agent a plausible-looking message where it expected a call
        // to make. Silent, and therefore dangerous.
        let chat = serde_json::json!({
            "id": "chatcmpl_1", "model": "openai/gpt-5.5",
            "choices": [{ "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{
                    "id": "call_9", "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
                }],
            }}],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 },
        });

        let out = response_from_chat(&chat, &[]);

        let item = &out["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_9");
        assert_eq!(item["name"], "get_weather");
        assert_eq!(item["arguments"], "{\"city\":\"Paris\"}");
        // `completed` is correct even with a tool call pending: it means GENERATION finished, not
        // that the agent loop is done. An earlier version reported `incomplete` here by analogy,
        // which is wrong against the spec and stalls any client that gates on `completed`.
        assert_eq!(out["status"], "completed");
    }

    #[test]
    fn text_alongside_a_tool_call_keeps_both() {
        let chat = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{ "message": {
                "content": "let me check",
                "tool_calls": [{ "id": "c1", "function": { "name": "f", "arguments": "{}" } }],
            }}],
        });

        let out = response_from_chat(&chat, &[]);

        assert_eq!(out["output"][0]["type"], "message");
        assert_eq!(out["output"][0]["content"][0]["text"], "let me check");
        assert_eq!(out["output"][1]["type"], "function_call");
    }

    #[test]
    fn a_truncated_reply_is_the_only_incomplete_one() {
        // The Responses API reserves `incomplete` for truncation, with `incomplete_details` naming
        // the reason — which is what `finish_reason: "length"` means on the Chat side.
        let chat = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{ "finish_reason": "length", "message": { "content": "cut off mid-" } }],
        });

        let out = response_from_chat(&chat, &[]);

        assert_eq!(out["status"], "incomplete");
        assert_eq!(out["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn tool_choice_is_translated_not_passed_through_in_the_wrong_shape() {
        // Responses: {type, name}. Chat: {type, function: {name}}. Forwarded unchanged this does
        // not error — Chat just does not recognise it, so the model is free to ignore a tool the
        // caller demanded. Silent, which is the whole problem.
        let out = request_to_chat(&serde_json::json!({
            "input": "hi",
            "tools": [{ "type": "function", "name": "f" }],
            "tool_choice": { "type": "function", "name": "f" },
        }));
        assert_eq!(out["tool_choice"]["type"], "function");
        assert_eq!(out["tool_choice"]["function"]["name"], "f");

        // The string forms are identical in both APIs.
        for word in ["auto", "none", "required"] {
            let out = request_to_chat(&serde_json::json!({
                "input": "hi", "tool_choice": word,
            }));
            assert_eq!(out["tool_choice"], word);
        }
    }

    #[test]
    fn a_tool_choice_we_cannot_express_routes_to_passthrough() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "hi",
            "tool_choice": { "type": "file_search" },
        })));
        assert!(!has_untranslatable_content(&serde_json::json!({
            "input": "hi",
            "tools": [{ "type": "function", "name": "f" }],
            "tool_choice": { "type": "function", "name": "f" },
        })));
    }

    #[test]
    fn hosted_tools_still_take_passthrough() {
        // web_search / file_search run inside the provider. Sending a partial tool set is worse
        // than not routing: a model missing one of its tools makes a confidently wrong plan.
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "search for it",
            "tools": [
                { "type": "function", "name": "f" },
                { "type": "web_search" },
            ],
        })));
    }

    #[test]
    fn function_tools_are_now_gated_rather_than_passed_through() {
        assert!(!has_untranslatable_content(&serde_json::json!({
            "input": [
                { "role": "user", "content": "weather?" },
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "c1", "output": "22C" },
            ],
            "tools": [{ "type": "function", "name": "f" }],
        })));
    }

    #[test]
    fn a_malformed_tool_item_routes_to_passthrough_rather_than_vanishing() {
        // Found in review: the gate said `function_call` translates while the translator returned
        // None for one missing its `call_id`, so the turn was silently dropped. The gate now asks
        // the translator, so the two cannot disagree.
        let no_call_id = serde_json::json!({
            "input": [{ "type": "function_call", "name": "f", "arguments": "{}" }],
        });
        assert!(has_untranslatable_content(&no_call_id));

        let no_name = serde_json::json!({
            "input": [{ "type": "function_call", "call_id": "c1", "arguments": "{}" }],
        });
        assert!(has_untranslatable_content(&no_name));

        let no_output = serde_json::json!({
            "input": [{ "type": "function_call_output", "call_id": "c1" }],
        });
        assert!(has_untranslatable_content(&no_output));

        // The well-formed pair still translates and is still gated.
        let ok = serde_json::json!({
            "input": [
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "c1", "output": "42" },
            ],
        });
        assert!(!has_untranslatable_content(&ok));
    }

    #[test]
    fn every_allowed_turn_actually_produces_a_message() {
        // The invariant behind the fix, stated directly: anything the gate allows must translate
        // to something. If these two ever disagree again, a turn goes missing with no error.
        // Text-then-call is folded into one assistant message (`push_message`), so the
        // turns are kept apart here by a user message; the folded case is tested on its own.
        let body = serde_json::json!({
            "input": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": [{ "type": "output_text", "text": "yes" }] },
                { "role": "user", "content": "go" },
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "c1", "output": "42" },
                { "type": "custom_tool_call", "call_id": "c2", "name": "p", "input": "x" },
                { "type": "custom_tool_call_output", "call_id": "c2", "output": "ok" },
            ],
        });
        assert!(!has_untranslatable_content(&body));

        let turns = body["input"].as_array().expect("array");
        let out = request_to_chat(&body);
        assert_eq!(
            out["messages"].as_array().expect("messages").len(),
            turns.len(),
            "every allowed turn must survive translation"
        );
    }

    #[test]
    fn reasoning_items_are_dropped_and_nothing_else_is() {
        // Prior encrypted reasoning: Chat cannot carry it and no Chat client sends it, so it is
        // dropped and the turn is gated. Before this, any Codex turn after the first reasoning
        // model reply took un-gated passthrough.
        let body = serde_json::json!({
            "input": [
                { "role": "user", "content": "hi" },
                { "type": "reasoning", "id": "rs_1", "summary": [],
                  "encrypted_content": "gAAAAB..." },
                { "role": "assistant", "content": "hello" },
            ],
        });
        assert!(!has_untranslatable_content(&body));
        let msgs = request_to_chat(&body)["messages"].clone();
        assert_eq!(msgs.as_array().expect("messages").len(), 2, "{msgs}");
        assert_eq!(msgs[0]["content"], "hi");
        assert_eq!(msgs[1]["content"], "hello");
        assert!(!msgs.to_string().contains("gAAAAB"), "{msgs}");

        // Only reasoning: a compaction item IS the compressed history and must not be dropped.
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": [{ "type": "compaction", "encrypted_content": "gAAAAB..." }],
        })));
    }

    #[test]
    fn parallel_calls_and_text_then_call_fold_into_one_assistant_message() {
        // Chat requires an assistant message with `tool_calls` to be followed directly by its
        // tool results; one-to-one translation of Responses' parallel calls breaks that.
        let out = request_to_chat(&serde_json::json!({
            "input": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "content": "checking both" },
                { "type": "function_call", "call_id": "a", "name": "f", "arguments": "{}" },
                { "type": "custom_tool_call", "call_id": "b", "name": "p", "input": "x" },
                { "type": "function_call_output", "call_id": "a", "output": "1" },
                { "type": "custom_tool_call_output", "call_id": "b", "output": "2" },
            ],
        }));
        let msgs = out["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 4, "{msgs:?}");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "checking both");
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "a");
        assert_eq!(msgs[1]["tool_calls"][1]["id"], "b");
        assert_eq!(msgs[2]["tool_call_id"], "a");
        assert_eq!(msgs[3]["tool_call_id"], "b");
    }

    #[test]
    fn a_tool_using_request_is_not_translated_when_hosted() {
        // Function tools now translate both ways, so they are gated. Only provider-hosted tools
        // remain untranslatable.
        assert!(!has_untranslatable_content(&serde_json::json!({
            "input": "hi",
            "tools": [{ "type": "function", "name": "get_weather" }],
        })));
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "hi",
            "tools": [{ "type": "computer_use_preview" }],
        })));
    }

    #[test]
    fn typed_item_turns_are_not_translated() {
        // function_call / function_call_output / reasoning items carry no role+content pair.
        // The earlier deny-list version returned None for these and dropped the turn entirely.
        // function_call(_output), custom_tool_call(_output) translate and reasoning is dropped;
        // every other Codex `ResponseItem` type (codex `protocol/src/models.rs`) still routes to
        // passthrough, as do the malformed / lossy forms of the translatable ones.
        for item in [
            serde_json::json!({ "type": "web_search_call", "id": "ws_1" }),
            serde_json::json!({ "type": "local_shell_call", "call_id": "c", "status": "completed",
                "action": { "type": "exec", "command": ["ls"] } }),
            serde_json::json!({ "type": "tool_search_call", "call_id": "c", "execution": "x",
                "arguments": {} }),
            serde_json::json!({ "type": "image_generation_call", "status": "completed",
                "result": "..." }),
            serde_json::json!({ "type": "compaction", "encrypted_content": "e" }),
            serde_json::json!({ "type": "context_compaction", "encrypted_content": "e" }),
            serde_json::json!({ "type": "agent_message", "author": "a", "recipient": "b",
                "content": [] }),
            // Typed, with a role and content, but not a `message`: still not a chat turn.
            serde_json::json!({ "type": "future_item", "role": "user", "content": "x" }),
            // Codex Responses-Lite tool declarations ride in the input as a developer item.
            serde_json::json!({ "type": "additional_tools", "role": "developer", "tools": [] }),
            // A namespaced call: the bare name could resolve to a different tool.
            serde_json::json!({ "type": "function_call", "call_id": "c", "name": "f",
                "namespace": "mcp__srv", "arguments": "{}" }),
            // A custom call whose input is not a string, or is missing.
            serde_json::json!({ "type": "custom_tool_call", "call_id": "c", "name": "p" }),
            serde_json::json!({ "type": "custom_tool_call", "call_id": "c", "name": "p",
                "input": { "x": 1 } }),
            // Tool output carrying an image (Codex `view_image`): no place in a Chat tool message.
            serde_json::json!({ "type": "function_call_output", "call_id": "c", "output": [
                { "type": "input_text", "text": "see" },
                { "type": "input_image", "image_url": "data:image/png;base64,AAAA" } ] }),
        ] {
            assert!(
                has_untranslatable_content(&serde_json::json!({ "input": [item.clone()] })),
                "must not translate {item}"
            );
        }
    }

    #[test]
    fn patch_grammar_is_gated_by_its_envelope_and_other_grammars_pass_through() {
        let tool = |definition: &str| {
            serde_json::json!({"model": "m", "input": "hi", "store": false, "tools": [{
                "type": "custom", "name": "apply_patch", "description": "d",
                "format": {"type": "grammar", "syntax": "lark", "definition": definition}
            }]})
        };
        let patch = r#"start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?"#;
        assert!(has_untranslatable_content(&tool(r#"start: "SELECT" /.+/"#)));
        assert!(!has_untranslatable_content(&tool(patch)));
        // The gate's half (pattern enforced by tool-args-valid) is tested in gate.rs.
        let chat = request_to_chat(&tool(patch));
        assert_eq!(
            chat["tools"][0]["function"]["parameters"]["properties"][CUSTOM_INPUT_ARG]["pattern"],
            PATCH_ENVELOPE_PATTERN
        );
    }

    #[test]
    fn reasoning_only_input_is_not_translated() {
        let body = serde_json::json!({"model": "m", "store": false, "input": [
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "x"}
        ]});
        assert!(has_untranslatable_content(&body));
    }

    #[test]
    fn patch_envelope_pattern_is_ecma_262_compatible() {
        // No Rust-only inline flags: providers validate `pattern` as ECMA-262.
        assert!(!PATCH_ENVELOPE_PATTERN.contains("(?"));
    }

    #[test]
    fn empty_tools_still_drop_parallel_tool_calls() {
        let chat = request_to_chat(&serde_json::json!({
            "model": "m", "input": "hi", "tools": [], "parallel_tool_calls": true
        }));
        assert!(chat.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn a_name_declared_as_both_custom_and_function_is_not_translated() {
        let body = |fname: &str| {
            serde_json::json!({"model": "m", "input": "hi", "store": false, "tools": [
                {"type": "custom", "name": "apply_patch", "description": "d"},
                {"type": "function", "name": fname, "parameters": {"type": "object"}}
            ]})
        };
        assert!(has_untranslatable_content(&body("apply_patch")));
        assert!(!has_untranslatable_content(&body("shell")));
    }

    #[test]
    fn explicit_store_true_is_not_translated_but_store_false_is() {
        let body = |store: bool| serde_json::json!({"model": "m", "input": "hi", "store": store});
        assert!(has_untranslatable_content(&body(true)));
        assert!(!has_untranslatable_content(&body(false)));
    }

    #[test]
    fn stateful_threading_is_not_translated() {
        // The upstream holds history we do not have, so a translated request is not the
        // conversation the client is actually continuing.
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "and then?", "previous_response_id": "resp_abc",
        })));
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "and then?", "conversation": "conv_abc", "stream": true,
        })));
    }

    #[test]
    fn plain_conversations_are_still_translated_and_gated() {
        // The allow-list must not be so tight that nothing gets verified.
        assert!(!has_untranslatable_content(
            &serde_json::json!({ "input": "hi" })
        ));
        assert!(!has_untranslatable_content(&serde_json::json!({
            "input": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "hello" },
                { "role": "user", "content": [
                    { "type": "input_text", "text": "and this?" },
                    { "type": "input_image", "image_url": "data:..." },
                ]},
            ],
        })));
    }

    #[test]
    fn typed_content_parts_are_flattened_to_text() {
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5",
            "input": [{ "role": "user", "content": [
                { "type": "input_text", "text": "part one " },
                { "type": "input_text", "text": "part two" },
            ]}],
        }));
        assert_eq!(out["messages"][0]["content"], "part one part two");
    }

    #[test]
    fn max_output_tokens_is_renamed_not_dropped() {
        // Dropped, the request would silently run to the provider's default ceiling — a caller
        // capping output would be billed for far more than it asked for.
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5", "input": "hi", "max_output_tokens": 256,
        }));
        assert_eq!(out["max_tokens"], 256);
        assert!(out.get("max_output_tokens").is_none());
    }

    #[test]
    fn nested_reasoning_effort_is_flattened_for_the_provider_layer() {
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5", "input": "hi", "reasoning": { "effort": "high" },
        }));
        assert_eq!(out["reasoning_effort"], "high");
        assert!(out.get("reasoning").is_none());
    }

    #[test]
    fn unknown_fields_survive_the_crossing() {
        // A field this code does not know is likelier to be newer than this code than to be wrong.
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5", "input": "hi", "metadata": { "run": "abc" },
        }));
        assert_eq!(out["metadata"]["run"], "abc");
    }

    #[test]
    fn responses_only_controls_do_not_reach_a_chat_provider() {
        // Raw bodies are carried verbatim to OpenAI-dialect rungs; Chat Completions 400s on each
        // of these. Values are what Codex sends (codex `core/src/client.rs`
        // `build_responses_request`).
        let out = request_to_chat(&serde_json::json!({
            "model": "gpt-5.5", "input": "hi", "stream": true,
            "reasoning": { "effort": "medium", "summary": "auto" },
            "store": false, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "019a-thread", "service_tier": "priority",
            "text": { "verbosity": "low" }, "client_metadata": { "x": "y" },
            "stream_options": { "reasoning_summary_delivery": "sequential_cutoff" },
            "parallel_tool_calls": true,
        }));
        for k in [
            "reasoning",
            "store",
            "include",
            "prompt_cache_key",
            "service_tier",
            "text",
            "client_metadata",
            "stream_options",
            // No tools: OpenAI rejects `parallel_tool_calls` without them.
            "parallel_tool_calls",
        ] {
            assert!(out.get(k).is_none(), "{k} leaked: {out}");
        }
        assert_eq!(out["reasoning_effort"], "medium");

        // Codex's `reasoning: null` (a model without reasoning) must not ride along either.
        let out = request_to_chat(&serde_json::json!({ "input": "hi", "reasoning": null }));
        assert!(out.get("reasoning").is_none(), "{out}");

        // With tools, parallel_tool_calls is meaningful (false forbids parallel calls) and stays.
        let out = request_to_chat(&serde_json::json!({
            "input": "hi", "tools": [{ "type": "function", "name": "f" }],
            "parallel_tool_calls": false,
        }));
        assert_eq!(out["parallel_tool_calls"], false);
    }

    #[test]
    fn a_structured_text_format_takes_passthrough_but_verbosity_does_not() {
        // `codex exec --output-schema`: a schema contract the Anthropic path would drop.
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "hi",
            "text": { "format": { "type": "json_schema", "name": "out", "strict": true,
                "schema": { "type": "object" } } },
        })));
        assert!(!has_untranslatable_content(&serde_json::json!({
            "input": "hi", "text": { "verbosity": "low" },
        })));
        assert!(!has_untranslatable_content(&serde_json::json!({
            "input": "hi", "text": { "format": { "type": "text" } },
        })));
    }

    #[test]
    fn an_image_without_a_url_takes_passthrough_instead_of_vanishing() {
        // `part_to_chat` has no Chat spelling for a `file_id` image and returned None, so the
        // allow-list waved the part through and the translation dropped it.
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": [{ "role": "user", "content": [
                { "type": "input_text", "text": "what is this?" },
                { "type": "input_image", "file_id": "file_123" },
            ]}],
        })));
    }

    /// Codex's `apply_patch` declaration, verbatim shape from codex
    /// `core/src/tools/handlers/apply_patch_spec.rs` (grammar shortened).
    fn apply_patch_tool() -> Value {
        serde_json::json!({
            "type": "custom", "name": "apply_patch",
            "description": "The `apply_patch` tool can be used to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
            "format": { "type": "grammar", "syntax": "lark",
                "definition": "start: begin_patch hunk+ end_patch\nbegin_patch: \"*** Begin Patch\" LF\nend_patch: \"*** End Patch\" LF?" },
        })
    }

    #[test]
    fn a_custom_tool_becomes_a_one_string_function_carrying_its_grammar() {
        let out = request_to_chat(&serde_json::json!({
            "input": "edit it", "tools": [apply_patch_tool()],
            "tool_choice": { "type": "custom", "name": "apply_patch" },
        }));
        let f = &out["tools"][0];
        assert_eq!(f["type"], "function");
        assert_eq!(f["function"]["name"], "apply_patch");
        assert_eq!(f["function"]["parameters"]["required"][0], "input");
        assert_eq!(
            f["function"]["parameters"]["properties"]["input"]["type"],
            "string"
        );
        let desc = f["function"]["description"].as_str().expect("description");
        assert!(desc.contains("FREEFORM tool"), "{desc}");
        assert!(desc.contains("lark grammar"), "{desc}");
        assert!(
            desc.contains("start: begin_patch hunk+ end_patch"),
            "the grammar is the input contract: {desc}"
        );
        // A custom tool_choice is a function choice on the Chat side.
        assert_eq!(out["tool_choice"]["function"]["name"], "apply_patch");

        // An unknown format is a contract this code would drop: passthrough.
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "x",
            "tools": [{ "type": "custom", "name": "q", "format": { "type": "regex" } }],
        })));
        // An unconstrained (`text`) custom tool translates.
        assert!(!has_untranslatable_content(&serde_json::json!({
            "input": "x",
            "tools": [{ "type": "custom", "name": "q", "format": { "type": "text" } }],
        })));
    }

    #[test]
    fn custom_tool_history_becomes_a_chat_tool_round_trip() {
        let out = request_to_chat(&serde_json::json!({
            "input": [
                { "role": "user", "content": "fix it" },
                { "type": "custom_tool_call", "status": "completed", "call_id": "call_p",
                  "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch\n" },
                { "type": "custom_tool_call_output", "call_id": "call_p",
                  "output": [{ "type": "input_text", "text": "Success. " },
                             { "type": "input_text", "text": "Updated a.txt" }] },
            ],
        }));
        let msgs = out["messages"].as_array().expect("messages");
        assert_eq!(msgs.len(), 3);
        let tc = &msgs[1]["tool_calls"][0];
        assert_eq!(tc["id"], "call_p");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "apply_patch");
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().expect("string args"))
                .expect("json args");
        assert_eq!(args["input"], "*** Begin Patch\n*** End Patch\n");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_p");
        // An all-text content-item output joins to its text, not a JSON dump of the list.
        assert_eq!(msgs[2]["content"], "Success. Updated a.txt");
    }

    fn custom_chat_call(args: &str) -> Value {
        chat(
            Value::Null,
            Some(serde_json::json!([{ "id": "call_p", "type": "function",
                "function": { "name": "apply_patch", "arguments": args } }])),
        )
    }

    #[test]
    fn a_served_call_to_a_custom_tool_comes_back_as_a_custom_tool_call() {
        let patch = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch\n";
        let args = serde_json::json!({ "input": patch }).to_string();
        let custom = vec!["apply_patch".to_owned()];
        let out = response_from_chat(&custom_chat_call(&args), &custom);
        let item = &out["output"][0];
        // The exact shape codex `ResponseItem::CustomToolCall` deserializes.
        assert_eq!(item["type"], "custom_tool_call");
        assert_eq!(item["call_id"], "call_p");
        assert_eq!(item["name"], "apply_patch");
        assert_eq!(item["input"], patch);
        assert!(item.get("arguments").is_none(), "{item}");

        // Not declared custom: stays a function_call.
        let out = response_from_chat(&custom_chat_call(&args), &[]);
        assert_eq!(out["output"][0]["type"], "function_call");

        // Arguments not in the wrapper shape are delivered as-is, not invented around.
        let out = response_from_chat(&custom_chat_call(patch), &custom);
        assert_eq!(out["output"][0]["input"], patch);
    }

    #[test]
    fn a_custom_tool_call_streams_its_input() {
        let patch = "*** Begin Patch\n*** End Patch\n";
        let args = serde_json::json!({ "input": patch }).to_string();
        let resp = response_from_chat(&custom_chat_call(&args), &["apply_patch".to_owned()]);
        let ev = frames(&sse_from_response(&resp));
        assert_eq!(
            types(&ev),
            expected(&[&[
                "response.output_item.added",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "response.output_item.done",
            ]])
        );
        let added = &ev[2]["item"];
        assert_eq!(added["type"], "custom_tool_call");
        // Codex parses the shell as a ResponseItem; `input` is required.
        assert_eq!(added["input"], "");
        assert_eq!(added["status"], "in_progress");
        for e in &ev[3..5] {
            assert_eq!(e["item_id"], added["id"]);
            assert_eq!(e["output_index"], 0);
        }
        assert_eq!(ev[3]["delta"], patch);
        assert_eq!(ev[4]["input"], patch);
        assert!(ev[4].get("arguments").is_none());
        assert_eq!(ev[5]["item"]["input"], patch);
        assert_eq!(ev[6]["response"]["output"][0]["type"], "custom_tool_call");
    }

    /// A request in the shape current Codex sends (codex 0ddee3e, `core/src/client.rs`
    /// `build_responses_request` + `protocol/src/models.rs` `ResponseItem`): base instructions as
    /// a leading `developer` message rather than `instructions`, encrypted reasoning from prior
    /// turns, an `exec_command` function round trip, an `apply_patch` custom round trip, and the
    /// Responses-only controls. `web_search` is absent — Codex sends it unless the user sets
    /// `web_search = "disabled"`, and with it the turn still (correctly) takes passthrough.
    fn codex_turn() -> Value {
        serde_json::json!({
            "model": "gpt-5.5", "stream": true,
            "input": [
                { "type": "message", "id": "msg_base", "role": "developer",
                  "content": [{ "type": "input_text", "text": "You are Codex, a coding agent." }] },
                { "type": "message", "role": "user", "content": [{ "type": "input_text",
                  "text": "<environment_context><cwd>/repo</cwd></environment_context>" }] },
                { "type": "message", "role": "user",
                  "content": [{ "type": "input_text", "text": "fix the typo in README" }] },
                { "type": "reasoning", "id": "rs_1",
                  "summary": [{ "type": "summary_text", "text": "Read the file first." }],
                  "encrypted_content": "gAAAAB-enc-1" },
                { "type": "function_call", "name": "exec_command",
                  "arguments": "{\"cmd\":\"cat README.md\"}", "call_id": "call_a" },
                { "type": "function_call_output", "call_id": "call_a", "output": "Helo world" },
                { "type": "reasoning", "id": "rs_2", "summary": [],
                  "encrypted_content": "gAAAAB-enc-2" },
                { "type": "custom_tool_call", "status": "completed", "call_id": "call_b",
                  "name": "apply_patch",
                  "input": "*** Begin Patch\n*** Update File: README.md\n-Helo world\n+Hello world\n*** End Patch\n" },
                { "type": "custom_tool_call_output", "call_id": "call_b",
                  "output": "Success. Updated the following files:\nM README.md\n" },
                { "type": "message", "role": "assistant", "phase": "final_answer",
                  "content": [{ "type": "output_text", "text": "Fixed the typo." }] },
                { "type": "message", "role": "user",
                  "content": [{ "type": "input_text", "text": "also add a trailing newline" }] },
            ],
            "tools": [
                { "type": "function", "name": "exec_command", "description": "Run a command.",
                  "strict": false, "parameters": { "type": "object",
                    "properties": { "cmd": { "type": "string" } }, "required": ["cmd"],
                    "additionalProperties": false } },
                apply_patch_tool(),
            ],
            "tool_choice": "auto", "parallel_tool_calls": true,
            "reasoning": { "effort": "medium", "summary": "auto" },
            "store": false, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "019a5f3e-thread", "text": { "verbosity": "low" },
            "client_metadata": { "x-codex-installation-id": "i" },
        })
    }

    #[test]
    fn a_realistic_codex_turn_is_gated_and_round_trips() {
        let body = codex_turn();
        assert!(
            !has_untranslatable_content(&body),
            "a default-shaped Codex turn must be gated"
        );

        let out = request_to_chat(&body);
        let msgs = out["messages"].as_array().expect("messages");
        let roles: Vec<&str> = msgs.iter().filter_map(|m| m["role"].as_str()).collect();
        assert_eq!(
            roles,
            [
                "developer",
                "user",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "user"
            ],
            "9 = 11 input items - 2 reasoning"
        );
        assert_eq!(msgs[0]["content"], "You are Codex, a coding agent.");
        assert_eq!(msgs[3]["tool_calls"][0]["function"]["name"], "exec_command");
        assert_eq!(msgs[5]["tool_calls"][0]["function"]["name"], "apply_patch");
        assert_eq!(msgs[6]["tool_call_id"], "call_b");
        assert_eq!(msgs[7]["content"], "Fixed the typo.");
        assert_eq!(msgs[8]["content"], "also add a trailing newline");
        assert!(!out.to_string().contains("gAAAAB"), "reasoning dropped");
        assert_eq!(out["tools"][1]["function"]["name"], "apply_patch");
        assert_eq!(out["reasoning_effort"], "medium");
        for k in [
            "input",
            "include",
            "store",
            "text",
            "prompt_cache_key",
            "client_metadata",
        ] {
            assert!(out.get(k).is_none(), "{k} leaked");
        }

        // The gated answer is an apply_patch call: it goes back as a custom_tool_call.
        let args = serde_json::json!({ "input": "*** Begin Patch\n*** End Patch\n" });
        let resp = response_from_chat(
            &custom_chat_call(&args.to_string()),
            &custom_tool_names(&body),
        );
        assert_eq!(resp["output"][0]["type"], "custom_tool_call");
        assert_eq!(
            resp["output"][0]["input"],
            "*** Begin Patch\n*** End Patch\n"
        );
        assert_eq!(resp["status"], "completed");

        // Codex's default tool list also carries hosted web_search: still passthrough.
        let mut with_search = body;
        with_search["tools"]
            .as_array_mut()
            .expect("tools")
            .push(serde_json::json!({ "type": "web_search", "external_web_access": false }));
        assert!(has_untranslatable_content(&with_search));
    }

    #[test]
    fn custom_tool_names_lists_only_custom_tools() {
        assert_eq!(
            custom_tool_names(&codex_turn()),
            vec!["apply_patch".to_owned()]
        );
        assert!(custom_tool_names(&serde_json::json!({ "input": "hi" })).is_empty());
    }

    #[test]
    fn the_reply_carries_text_in_both_places_a_client_looks() {
        let chat = serde_json::json!({
            "id": "chatcmpl_1", "model": "openai/gpt-5.5",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": "42" } }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 3 },
        });

        let out = response_from_chat(&chat, &[]);

        assert_eq!(out["object"], "response");
        assert_eq!(out["status"], "completed");
        assert_eq!(out["output"][0]["content"][0]["text"], "42");
        assert_eq!(out["output"][0]["content"][0]["type"], "output_text");
        // The SDK convenience accessor — without it `response.output_text` is empty.
        assert_eq!(out["output_text"], "42");
    }

    #[test]
    fn usage_is_renamed_or_the_call_looks_free() {
        // Responses clients read `input_tokens`/`output_tokens`. Left in Chat's spelling, a served
        // request displays as having consumed nothing.
        let chat = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{ "message": { "content": "x" } }],
            "usage": { "prompt_tokens": 1200, "completion_tokens": 300 },
        });

        let out = response_from_chat(&chat, &[]);

        assert_eq!(out["usage"]["input_tokens"], 1200);
        assert_eq!(out["usage"]["output_tokens"], 300);
        assert_eq!(out["usage"]["total_tokens"], 1500);
    }

    #[test]
    fn a_round_trip_preserves_the_conversation() {
        let original = serde_json::json!({
            "model": "gpt-5.5",
            "instructions": "be terse",
            "input": [
                { "role": "user", "content": "first" },
                { "role": "assistant", "content": "ok" },
                { "role": "user", "content": "second" },
            ],
        });

        let chat = request_to_chat(&original);
        let msgs = chat["messages"].as_array().expect("messages");

        assert_eq!(msgs.len(), 4, "system + three turns");
        assert_eq!(msgs[3]["content"], "second");
        assert_eq!(msgs[2]["role"], "assistant");
    }

    /// Native eligibility: hosted tools the gate can treat as provider-side are fine; server-held
    /// threading, background mode, and tools with no known semantics are not.
    #[test]
    fn native_eligible_allows_hosted_web_search_but_not_threading() {
        let base = serde_json::json!({
            "model": "gpt-5", "input": "hi", "store": false,
            "tools": [{ "type": "web_search" },
                      { "type": "function", "name": "shell", "parameters": {} }],
        });
        assert!(native_eligible(&base));
        assert!(
            has_untranslatable_content(&base),
            "translation path cannot carry it"
        );
        for (k, v) in [
            ("previous_response_id", serde_json::json!("resp_1")),
            ("conversation", serde_json::json!("conv_1")),
            ("background", serde_json::json!(true)),
            (
                "tools",
                serde_json::json!([{ "type": "mcp", "server_label": "x" }]),
            ),
        ] {
            let mut b = base.clone();
            b[k] = v;
            assert!(!native_eligible(&b), "{k} must not be native");
        }
    }

    /// The native wire body is the client's body with only `model` and the stream controls changed.
    #[test]
    fn native_wire_body_changes_only_model_and_stream() {
        let raw = serde_json::json!({
            "model": "gpt-5-codex", "stream": true, "stream_options": {},
            "store": true, "include": ["reasoning.encrypted_content"],
            "tools": [{ "type": "web_search" }],
            "input": [{ "type": "reasoning", "id": "rs", "summary": [],
                        "encrypted_content": "opaque==" }],
            "some_future_field": 1,
        });
        let out = native_wire_body(raw.as_object().unwrap(), "gpt-5-mini");
        let mut expected = raw.clone();
        expected["model"] = "gpt-5-mini".into();
        expected.as_object_mut().unwrap().remove("stream");
        expected.as_object_mut().unwrap().remove("stream_options");
        assert_eq!(out, expected);
        assert!(is_responses_body(&raw));
        assert!(!is_responses_body(
            &serde_json::json!({ "messages": [], "input": 1 })
        ));
    }

    /// Output extraction: text from `message` parts only; client tool calls (function and custom)
    /// in gate shape; reasoning and hosted calls ignored; a non-Responses object is `None`.
    #[test]
    fn native_output_extraction() {
        let raw = serde_json::json!({ "object": "response", "output": [
            { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "hidden" }] },
            { "type": "web_search_call", "id": "ws" },
            { "type": "message", "content": [
                { "type": "output_text", "text": "a" }, { "type": "output_text", "text": "b" }] },
            { "type": "function_call", "name": "f", "arguments": "{\"x\":1}" },
            { "type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch" },
        ]});
        assert_eq!(output_text(&raw), "ab");
        assert_eq!(
            output_tool_calls(&raw).unwrap(),
            [
                ("f".to_owned(), "{\"x\":1}".to_owned()),
                (
                    "apply_patch".to_owned(),
                    r#"{"input":"*** Begin Patch"}"#.to_owned()
                ),
            ]
        );
        assert!(output_tool_calls(&serde_json::json!({ "choices": [] })).is_none());
    }
}
