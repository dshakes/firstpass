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
//! ## What it does not do yet
//!
//! Streaming. `enforce` is inherently buffered — the gate has to see the whole candidate before it
//! can judge it — so a streaming Responses request is served through the observe passthrough, the
//! same fallback the Chat Completions path uses for a request it cannot gate faithfully. Reasoning
//! items, tool-call round trips, and stateful `previous_response_id` threading are also out of
//! scope here; they are passed through where harmless and otherwise left alone rather than
//! half-translated.

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
        // Typed items: the tool round trip translates; anything else (reasoning items, hosted-tool
        // calls) does not.
        if let Some(kind) = obj.get("type").and_then(Value::as_str)
            && !obj.contains_key("role")
        {
            return !matches!(kind, "function_call" | "function_call_output");
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
    // Responses reports a tool-call turn as `incomplete`, not `completed`: the model is waiting
    // for the caller to run the tool. Reporting `completed` would tell an agent the turn is over.
    let awaiting_tool = tool_calls.is_some_and(|c| !c.is_empty());
    resp.insert(
        "status".to_owned(),
        Value::String(
            if awaiting_tool {
                "incomplete"
            } else {
                "completed"
            }
            .to_owned(),
        ),
    );
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // A turn awaiting a tool result is NOT complete — reporting `completed` would tell an
        // agent the turn is over when the model is waiting on it.
        assert_eq!(out["status"], "incomplete");
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
