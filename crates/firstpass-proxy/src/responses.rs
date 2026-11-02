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
//! ## What it does not do yet
//!
//! Reasoning items and stateful `previous_response_id` threading. A request carrying either takes
//! the passthrough (see [`has_untranslatable_content`]); the enforce path runs over Chat
//! Completions, which has no reasoning items, so a served answer never contains one.

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
            for turn in turns {
                if let Some(m) = turn_to_message(turn) {
                    messages.push(m);
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
        out.remove("reasoning");
    }
    Value::Object(out)
}

/// One Responses `input` turn → one Chat Completions message.
fn turn_to_message(turn: &Value) -> Option<Value> {
    let obj = turn.as_object()?;
    // Typed items carry no role/content pair — they are the tool round trip.
    match obj.get("type").and_then(Value::as_str) {
        Some("function_call") => return function_call_to_chat(obj),
        Some("function_call_output") => return function_output_to_chat(obj),
        _ => {}
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
    if obj.get("type").and_then(Value::as_str) != Some("function") {
        return None;
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
            if o.get("type").and_then(Value::as_str) != Some("function") {
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

/// A Responses tool-result item → the Chat `tool` message that carries it.
fn function_output_to_chat(item: &serde_json::Map<String, Value>) -> Option<Value> {
    let output = match item.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => return None,
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
/// text or text-and-image parts, function tool definitions, and the `function_call` /
/// `function_call_output` round trip — in both directions, so a tool call in the reply comes back
/// as a `function_call` item rather than vanishing.
///
/// Still excluded: **hosted** tools (`web_search`, `file_search`, `computer_use`), which run inside
/// the provider and have no Chat equivalent; reasoning items; and `previous_response_id` threading,
/// where the upstream holds history we do not have, so a translated request is not the conversation
/// the client is continuing.
#[must_use]
pub fn has_untranslatable_content(body: &Value) -> bool {
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
    // not the conversation the client is actually continuing.
    if body.get("previous_response_id").is_some() {
        return true;
    }
    let Some(Value::Array(turns)) = body.get("input") else {
        return false;
    };
    turns.iter().any(|turn| {
        let Some(obj) = turn.as_object() else {
            return true;
        };
        // Typed items: ask the translator itself rather than restating its rules here. A
        // `function_call` missing its `call_id` translates to nothing, and a check that only
        // looked at `type` would wave it through and let the turn vanish — the exact silent drop
        // this allow-list exists to prevent. One source of truth cannot drift from itself.
        if obj.contains_key("type") && !obj.contains_key("role") {
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
                !matches!(
                    p.get("type").and_then(Value::as_str),
                    Some("input_text" | "output_text" | "text" | "input_image")
                )
            }),
            _ => true,
        }
    })
}

/// Convert a Chat Completions response into the Responses shape.
///
/// Reports `usage` with Responses' own field names (`input_tokens` / `output_tokens`), which is
/// what a client reads to display cost — leaving Chat's `prompt_tokens` spelling would make a
/// served request look like it consumed nothing.
#[must_use]
pub fn response_from_chat(chat: &Value) -> Value {
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
/// `response.function_call_arguments.delta` → `.done`; then `response.output_item.done` → finally
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
            "function_call" => {
                let args = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                for (k, data) in [
                    (
                        "response.function_call_arguments.delta",
                        json!({ "delta": args }),
                    ),
                    (
                        "response.function_call_arguments.done",
                        json!({ "arguments": args }),
                    ),
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
        let resp = response_from_chat(&chat(Value::from("hello world"), None));
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
        let resp = response_from_chat(&chat(Value::Null, Some(call())));
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
        let resp = response_from_chat(&chat(Value::from("checking"), Some(call())));
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
        let ev = frames(&sse_from_response(&response_from_chat(&c)));
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

        let out = response_from_chat(&chat);

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

        let out = response_from_chat(&chat);

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

        let out = response_from_chat(&chat);

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
        let body = serde_json::json!({
            "input": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": [{ "type": "output_text", "text": "yes" }] },
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "c1", "output": "42" },
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
    fn reasoning_items_still_take_passthrough() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": [{ "type": "reasoning", "summary": [] }],
        })));
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
        for item in [
            // function_call / function_call_output now translate; only these remain.
            serde_json::json!({ "type": "reasoning", "summary": [] }),
            serde_json::json!({ "type": "web_search_call", "id": "ws_1" }),
        ] {
            assert!(
                has_untranslatable_content(&serde_json::json!({ "input": [item.clone()] })),
                "must not translate {item}"
            );
        }
    }

    #[test]
    fn stateful_threading_is_not_translated() {
        // The upstream holds history we do not have, so a translated request is not the
        // conversation the client is actually continuing.
        assert!(has_untranslatable_content(&serde_json::json!({
            "input": "and then?", "previous_response_id": "resp_abc",
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
            "model": "gpt-5.5", "input": "hi", "metadata": { "run": "abc" }, "store": false,
        }));
        assert_eq!(out["metadata"]["run"], "abc");
        assert_eq!(out["store"], false);
    }

    #[test]
    fn the_reply_carries_text_in_both_places_a_client_looks() {
        let chat = serde_json::json!({
            "id": "chatcmpl_1", "model": "openai/gpt-5.5",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": "42" } }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 3 },
        });

        let out = response_from_chat(&chat);

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

        let out = response_from_chat(&chat);

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
}
