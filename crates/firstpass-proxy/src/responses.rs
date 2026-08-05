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
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
    let content = match obj.get("content") {
        Some(Value::String(s)) => Value::String(s.clone()),
        // Typed parts: concatenate the text ones. Anything else (images, files) is not represented
        // in a plain string, so it is dropped here rather than being mangled into one — those
        // requests should take the passthrough path.
        Some(Value::Array(parts)) => {
            let text: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("");
            Value::String(text)
        }
        _ => return None,
    };
    Some(serde_json::json!({ "role": role, "content": content }))
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
    resp.insert("status".to_owned(), Value::String("completed".to_owned()));
    resp.insert("model".to_owned(), Value::String(model.to_owned()));
    resp.insert(
        "output".to_owned(),
        serde_json::json!([{
            "type": "message",
            "id": format!("msg_{id}"),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }],
        }]),
    );
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
