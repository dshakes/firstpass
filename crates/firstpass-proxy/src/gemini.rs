//! Google Gemini `generateContent` inbound protocol — translation to/from Chat Completions.
//!
//! Verified sources (2026-11-02):
//! - `google-gemini/gemini-cli` `packages/core/src/core/contentGenerator.ts`:
//!   `GOOGLE_GEMINI_BASE_URL` env var (line 87-88) triggers `AuthType.GATEWAY`.
//!   API key relayed via `x-goog-api-key` header (from `@google/genai`).
//! - `googleapis/js-genai` `src/_api_client.ts`: default apiVersion = `v1beta` (line 355-356),
//!   baseUrl = `https://generativelanguage.googleapis.com/` (line 357).
//!   Full URL: `{baseUrl}/v1beta/models/{model}:generateContent`.
//! - `googleapis/js-genai` `src/node/_node_auth.ts` line 63:
//!   `headers.append(GOOGLE_API_KEY_HEADER, this.apiKey)` where `GOOGLE_API_KEY_HEADER = "x-goog-api-key"`.
//! - SSE stream format: `data: {json}\n\n` delimiters (`_api_client.ts` line 751).
//!
//! ## Translation scope (allow-list; `has_untranslatable_content` guards the rest)
//!
//! | Gemini field | Chat equivalent |
//! |---|---|
//! | `systemInstruction.parts[].text` | `messages[{role:"system"}]` |
//! | `contents[].role = "user"/"model"` | `messages[].role = "user"/"assistant"` |
//! | `contents[].parts[].text` | message content string |
//! | `contents[].parts[].functionCall` | `tool_calls` in assistant message |
//! | `contents[].parts[].functionResponse` | `{role:"tool", tool_call_id, content}` |
//! | `tools[].functionDeclarations[]` | `tools[{type:"function", function:{...}}]` |
//! | `generationConfig.maxOutputTokens` | `max_tokens` |
//! | `usageMetadata.promptTokenCount` | `usage.prompt_tokens` |
//! | `usageMetadata.candidatesTokenCount` | `usage.completion_tokens` |
//! | `finishReason: "STOP"/"MAX_TOKENS"` | `finish_reason: "stop"/"length"` |
//!
//! ## Not translated → passthrough
//!
//! `inlineData` (binary), `executableCode` / `codeExecutionResult`, `fileData`, `videoMetadata`,
//! `thoughtSignature`, `cachedContent`, and any part type not in the allow-list above.

use serde_json::{Map, Value};

// ── inbound (Gemini → Chat) ───────────────────────────────────────────────────

/// Convert a Gemini `generateContent` request body to Chat Completions.
///
/// `model` is the wire model name (extracted from the URL path by the caller, absent from body).
/// A result with `"messages": []` means the body had no translatable content; callers should check
/// `has_untranslatable_content` before calling this.
#[must_use]
pub fn request_to_chat(body: &Value, model: &str) -> Value {
    let Some(obj) = body.as_object() else {
        return body.clone();
    };
    let mut out = Map::new();
    out.insert("model".to_owned(), Value::String(model.to_owned()));

    // systemInstruction → leading system message
    let mut messages: Vec<Value> = Vec::new();
    if let Some(si) = obj.get("systemInstruction")
        && let Some(text) = parts_to_text(si)
        && !text.is_empty()
    {
        messages.push(serde_json::json!({"role": "system", "content": text}));
    }

    // contents → messages
    if let Some(Value::Array(contents)) = obj.get("contents") {
        for (turn, c) in contents.iter().enumerate() {
            messages.extend(content_to_messages(c, turn));
        }
    }
    out.insert("messages".to_owned(), Value::Array(messages));

    // generationConfig → top-level fields Chat understands
    if let Some(Value::Object(gen_cfg)) = obj.get("generationConfig") {
        if let Some(v) = gen_cfg.get("maxOutputTokens") {
            out.insert("max_tokens".to_owned(), v.clone());
        }
        if let Some(v) = gen_cfg.get("temperature") {
            out.insert("temperature".to_owned(), v.clone());
        }
        if let Some(v) = gen_cfg.get("topP") {
            out.insert("top_p".to_owned(), v.clone());
        }
    }

    // tools: flatten functionDeclarations → Chat tool list
    if let Some(Value::Array(tools)) = obj.get("tools") {
        let converted: Vec<Value> = tools
            .iter()
            .flat_map(|t| {
                t.get("functionDeclarations")
                    .and_then(Value::as_array)
                    .map_or(vec![], |fds| fds.iter().filter_map(fd_to_chat).collect())
            })
            .collect();
        if !converted.is_empty() {
            out.insert("tools".to_owned(), Value::Array(converted));
        }
    }

    Value::Object(out)
}

/// One Gemini `content` object → zero or more Chat messages.
///
/// A `functionResponse` content (the user/tool side returning results) may have multiple parts
/// when the model issued parallel calls. Each response is emitted as its own `{role:"tool"}`
/// message so downstream providers see one message per call ID, matching the OpenAI wire format.
///
/// Tool call IDs are positional (`call_{turn}_{idx}`) so two parallel calls to the
/// same function get distinct IDs and neither clobbers the other's result.
/// `turn` is this content's index in `contents`. Call ids embed the turn so they are unique
/// across the conversation (Anthropic rejects a repeated `tool_use` id); a functionResponse turn
/// answers the model turn right before it (`function_pairing_is_exact` guarantees that), so it
/// uses `turn - 1`.
fn content_to_messages(content: &Value, turn: usize) -> Vec<Value> {
    let Some(obj) = content.as_object() else {
        return vec![];
    };
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
    let chat_role = if role == "model" { "assistant" } else { role };
    let Some(parts) = obj.get("parts").and_then(Value::as_array) else {
        return vec![];
    };

    // functionCall parts → single assistant message with tool_calls list.
    // Positional IDs: call_{turn}_{idx} so parallel calls to the same function differ.
    // The index counts functionCall parts only (not text), so it matches the ordinal of the
    // paired functionResponse in the next turn; `has_untranslatable_content` rejects any
    // history where that pairing is not exact.
    let calls: Vec<Value> = parts
        .iter()
        .filter_map(|p| p.as_object()?.get("functionCall")?.as_object())
        .enumerate()
        .filter_map(|(idx, fc)| {
            let name = fc.get("name")?.clone();
            let args = fc
                .get("args")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            Some(serde_json::json!({
                "id": format!("call_{turn}_{idx}"),
                "type": "function",
                "function": { "name": name, "arguments": args.to_string() },
            }))
        })
        .collect();
    if !calls.is_empty() {
        let text = text_from_parts(parts);
        let content_val = if text.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        };
        return vec![serde_json::json!({
            "role": "assistant",
            "content": content_val,
            "tool_calls": calls,
        })];
    }

    // functionResponse parts → one tool message per part (parallel tool results).
    // Each part contributes its own {role:"tool"} message with a positional call ID.
    let tool_msgs: Vec<Value> = parts
        .iter()
        .filter_map(|p| p.as_object()?.get("functionResponse")?.as_object())
        .enumerate()
        .filter_map(|(idx, fr)| {
            fr.get("name").and_then(Value::as_str)?;
            let output = fr
                .get("response")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()))
                .to_string();
            Some(serde_json::json!({
                "role": "tool",
                "tool_call_id": format!("call_{}_{idx}", turn.saturating_sub(1)),
                "content": output,
            }))
        })
        .collect();
    if !tool_msgs.is_empty() {
        return tool_msgs;
    }

    // Plain text message
    let text = text_from_parts(parts);
    vec![serde_json::json!({ "role": chat_role, "content": text })]
}

/// Concatenate text parts from a `parts` array.
fn text_from_parts(parts: &[Value]) -> String {
    parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

/// Concatenated text from a container with a `parts` array (used for `systemInstruction`).
fn parts_to_text(v: &Value) -> Option<String> {
    Some(text_from_parts(v.get("parts").and_then(Value::as_array)?))
}

/// One Gemini `functionDeclaration` → its Chat `{type: "function", function: {...}}` wrapper.
fn fd_to_chat(fd: &Value) -> Option<Value> {
    let mut f = Map::new();
    f.insert("name".to_owned(), fd.get("name")?.clone());
    if let Some(d) = fd.get("description") {
        f.insert("description".to_owned(), d.clone());
    }
    if let Some(p) = fd.get("parameters") {
        f.insert("parameters".to_owned(), p.clone());
    }
    Some(serde_json::json!({"type": "function", "function": Value::Object(f)}))
}

/// Chat pairs a tool result to its call by id; Gemini pairs them by order. The positional ids
/// `content_to_messages` assigns are only right when every turn of functionResponses answers the
/// functionCalls of the model turn right before it, same names in the same order, and does not
/// mix in other parts. Anything else is ambiguous → the caller sends it to passthrough.
fn function_pairing_is_exact(contents: Option<&Value>) -> bool {
    let Some(Value::Array(contents)) = contents else {
        return true;
    };
    let names = |c: &Value, key: &str| -> Vec<String> {
        c.get("parts")
            .and_then(Value::as_array)
            .map(|ps| {
                ps.iter()
                    .filter_map(|p| p.get(key)?.get("name")?.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut pending: Vec<String> = Vec::new();
    for c in contents {
        let responses = names(c, "functionResponse");
        if !responses.is_empty() {
            let all_parts = c.get("parts").and_then(Value::as_array).map_or(0, Vec::len);
            if responses != pending || all_parts != responses.len() {
                return false;
            }
            pending.clear();
            continue;
        }
        if !pending.is_empty() {
            return false; // calls left unanswered before the next turn
        }
        pending = names(c, "functionCall");
    }
    true
}

/// Whether this `generateContent` body contains anything the translation cannot faithfully
/// represent. Allow-list, not deny-list: only known-good content is gated; anything outside
/// takes the passthrough path un-gated.
///
/// **Request-level**: `cachedContent`, `toolConfig`, `safetySettings`, non-`functionDeclarations`
/// tool entries (e.g. `googleSearch`, `codeExecution`), `systemInstruction` with non-text parts,
/// and unmapped `generationConfig` fields trigger passthrough.
///
/// **Part-level**: plain text, `functionCall`, `functionResponse` are translatable. Anything
/// else — `inlineData`, `executableCode`, `codeExecutionResult`, `fileData`, `videoMetadata`,
/// `thoughtSignature`, or any unknown part key — is not.
#[must_use]
pub fn has_untranslatable_content(body: &Value) -> bool {
    let Some(obj) = body.as_object() else {
        return false;
    };

    // ── request-level allow-list ──────────────────────────────────────────────
    // Only the fields `request_to_chat` maps; any other (cachedContent, toolConfig,
    // safetySettings, labels, a future field) would be silently dropped → passthrough.
    if obj.keys().any(|k| {
        !matches!(
            k.as_str(),
            "contents" | "systemInstruction" | "generationConfig" | "tools"
        )
    }) {
        return true;
    }
    if !function_pairing_is_exact(obj.get("contents")) {
        return true;
    }
    if obj.contains_key("cachedContent")
        || obj.contains_key("toolConfig")
        || obj.contains_key("safetySettings")
    {
        return true;
    }

    // tools: each entry must only carry functionDeclarations; googleSearch, codeExecution,
    // retrieval, etc. have no Chat equivalent and would be silently dropped.
    if let Some(Value::Array(tools)) = obj.get("tools") {
        for tool in tools {
            let is_only_fn_decls = tool
                .as_object()
                .is_some_and(|m| m.keys().all(|k| k == "functionDeclarations"));
            if !is_only_fn_decls {
                return true;
            }
        }
    }

    // systemInstruction: only pure-text parts are translatable.
    if let Some(si) = obj.get("systemInstruction")
        && let Some(Value::Array(parts)) = si.get("parts")
    {
        for p in parts {
            let has_only_text = p
                .as_object()
                .is_some_and(|m| m.len() == 1 && m.contains_key("text"));
            if !has_only_text {
                return true;
            }
        }
    }

    // generationConfig: only the mapped subset (maxOutputTokens, temperature, topP).
    if let Some(Value::Object(gen_cfg)) = obj.get("generationConfig") {
        for k in gen_cfg.keys() {
            if !matches!(k.as_str(), "maxOutputTokens" | "temperature" | "topP") {
                return true;
            }
        }
    }

    // ── part-level allow-list ─────────────────────────────────────────────────
    let Some(Value::Array(contents)) = obj.get("contents") else {
        // No contents at all is fine — caller decides whether to reject that.
        return false;
    };
    contents.iter().any(|content| {
        let Some(content_obj) = content.as_object() else {
            return true;
        };
        let Some(Value::Array(parts)) = content_obj.get("parts") else {
            return true;
        };
        parts.iter().any(|part| {
            let Some(part_obj) = part.as_object() else {
                return true;
            };
            // Strict allow-list: every field on the part must translate faithfully. Anything
            // else (inlineData, `thought: true`, and `thoughtSignature`, an opaque signed
            // token Gemini 2.5+/3 needs replayed) would be silently dropped. gemini-cli
            // replays thoughtSignature in history after the first tool call, so for those
            // models most agentic tool turns take the observe passthrough.
            part_obj.is_empty()
                || part_obj
                    .keys()
                    .any(|k| !matches!(k.as_str(), "text" | "functionCall" | "functionResponse"))
        })
    })
}

// ── outbound (Chat → Gemini) ──────────────────────────────────────────────────

/// Convert a Chat Completions response to a Gemini `generateContent` response shape.
///
/// `finish_reason: "stop"` → `finishReason: "STOP"`.
/// `finish_reason: "length"` → `finishReason: "MAX_TOKENS"` (the only `incomplete` reason Gemini
/// spells differently from OpenAI).
/// `finish_reason: "tool_calls"` → `finishReason: "STOP"` (generation finished, call pending run).
#[must_use]
pub fn response_from_chat(chat: &Value) -> Value {
    let finish_reason = chat
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .map(chat_finish_reason_to_gemini)
        .unwrap_or("STOP");

    let mut parts: Vec<Value> = Vec::new();

    let text = chat
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !text.is_empty() {
        parts.push(serde_json::json!({"text": text}));
    }

    if let Some(calls) = chat
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
    {
        for tc in calls {
            let name = tc
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args_str = tc
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let args: Value =
                serde_json::from_str(args_str).unwrap_or_else(|_| Value::Object(Map::new()));
            parts.push(serde_json::json!({
                "functionCall": { "name": name, "args": args }
            }));
        }
    }

    if parts.is_empty() {
        parts.push(serde_json::json!({"text": ""}));
    }

    let in_tok = chat
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let out_tok = chat
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    serde_json::json!({
        "candidates": [{
            "content": { "role": "model", "parts": parts },
            "finishReason": finish_reason,
        }],
        "usageMetadata": {
            "promptTokenCount": in_tok,
            "candidatesTokenCount": out_tok,
            "totalTokenCount": in_tok + out_tok,
        },
    })
}

fn chat_finish_reason_to_gemini(reason: &str) -> &'static str {
    match reason {
        "length" => "MAX_TOKENS",
        // "stop" | "tool_calls" | anything else → STOP (generation finished)
        _ => "STOP",
    }
}

/// Re-emit a gated `generateContent` response as the SSE stream `@google/genai` expects.
///
/// The SDK parser splits on `\n\n` and looks for lines starting with `data:` (`_api_client.ts`
/// lines 751–794). A single `data: {json}\n\n` chunk is sufficient — the SDK accumulates chunks
/// and the final state is what the caller sees.
#[must_use]
pub fn sse_from_response(resp: &Value) -> String {
    format!("data: {resp}\n\n")
}

/// A Gemini SSE stream that failed before any answer could be served — emits an error response
/// the `@google/genai` SDK will surface as a thrown `Error`.
#[must_use]
pub fn sse_error_response(message: &str) -> String {
    let resp = serde_json::json!({
        "error": { "code": 500, "message": message, "status": "INTERNAL" }
    });
    format!("data: {resp}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── request_to_chat ─────────────────────────────────────────────────────

    #[test]
    fn plain_text_conversation_translates() {
        let body = serde_json::json!({
            "contents": [
                { "role": "user", "parts": [{ "text": "hello" }] },
                { "role": "model", "parts": [{ "text": "hi there" }] },
                { "role": "user", "parts": [{ "text": "tell me more" }] },
            ],
            "generationConfig": { "maxOutputTokens": 1024, "temperature": 0.5 },
        });
        let out = request_to_chat(&body, "gemini-2.0-flash");
        assert_eq!(out["model"], "gemini-2.0-flash");
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "hi there");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(out["max_tokens"], 1024);
        assert_eq!(out["temperature"], 0.5);
        // mutant: if we remove the role mapping `model` → `assistant`, msgs[1]["role"] would be "model"
        // (confirmed fail)
    }

    #[test]
    fn system_instruction_becomes_system_message() {
        let body = serde_json::json!({
            "systemInstruction": { "parts": [{ "text": "be terse" }] },
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        });
        let out = request_to_chat(&body, "m");
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be terse");
        assert_eq!(msgs[1]["role"], "user");
        // mutant: if we skip systemInstruction, msgs[0]["role"] would be "user"
    }

    #[test]
    fn function_declarations_become_chat_tools() {
        let body = serde_json::json!({
            "contents": [{ "role": "user", "parts": [{ "text": "weather?" }] }],
            "tools": [{ "functionDeclarations": [{
                "name": "get_weather",
                "description": "look up weather",
                "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
            }] }],
        });
        let out = request_to_chat(&body, "m");
        let tools = out["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["description"], "look up weather");
        // mutant: if we forget to set function.name, this assertion fails
    }

    #[test]
    fn function_call_in_request_becomes_tool_calls() {
        let body = serde_json::json!({
            "contents": [
                { "role": "user", "parts": [{ "text": "weather in Paris?" }] },
                { "role": "model", "parts": [{ "functionCall": {
                    "name": "get_weather", "args": { "city": "Paris" }
                } }] },
            ],
        });
        let out = request_to_chat(&body, "m");
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["role"], "assistant");
        let calls = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // mutant: if we fail to emit tool_calls, this fails
    }

    #[test]
    fn function_response_becomes_tool_message() {
        let body = serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [{ "functionResponse": {
                    "name": "get_weather",
                    "response": { "temperature": "22C" }
                } }],
            }],
        });
        let out = request_to_chat(&body, "m");
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        // positional ID: the functionResponse is at index 0
        assert_eq!(msgs[0]["tool_call_id"], "call_0_0");
        assert!(msgs[0]["content"].as_str().is_some());
        // mutant: if we emit role "user" instead of "tool", assertion fails
    }

    #[test]
    fn parallel_function_responses_each_become_own_tool_message() {
        // Two functionResponse parts in one content → two separate tool messages with
        // positional call IDs. Dropping the second part would lose a tool result.
        let body = serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [
                    { "functionResponse": { "name": "lookup", "response": { "a": 1 } } },
                    { "functionResponse": { "name": "lookup", "response": { "b": 2 } } },
                ],
            }],
        });
        let out = request_to_chat(&body, "m");
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(
            msgs.len(),
            2,
            "must emit one message per functionResponse part"
        );
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "call_0_0");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], "call_0_1");
        // mutant: revert to old `return Some(...)` on first match → msgs.len() == 1
    }

    #[test]
    fn parallel_function_calls_get_positional_ids() {
        // Two functionCall parts for the same function name must get distinct IDs so neither
        // result overwrites the other when the model returns its tool responses.
        let body = serde_json::json!({
            "contents": [{
                "role": "model",
                "parts": [
                    { "functionCall": { "name": "search", "args": { "q": "foo" } } },
                    { "functionCall": { "name": "search", "args": { "q": "bar" } } },
                ],
            }],
        });
        let out = request_to_chat(&body, "m");
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        let calls = msgs[0]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(
            calls[0]["id"], calls[1]["id"],
            "parallel calls must have distinct IDs"
        );
        assert_eq!(calls[0]["id"], "call_0_0");
        assert_eq!(calls[1]["id"], "call_0_1");
        // mutant: use name-only ID → both get "call_0" → ids are equal → assertion fails
    }

    // ── response_from_chat ──────────────────────────────────────────────────

    #[test]
    fn text_response_produces_candidates_shape() {
        let chat = serde_json::json!({
            "id": "c", "model": "m",
            "choices": [{ "finish_reason": "stop", "message": { "content": "answer" } }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 },
        });
        let out = response_from_chat(&chat);
        assert_eq!(
            out["candidates"][0]["content"]["parts"][0]["text"],
            "answer"
        );
        assert_eq!(out["candidates"][0]["content"]["role"], "model");
        assert_eq!(out["candidates"][0]["finishReason"], "STOP");
        assert_eq!(out["usageMetadata"]["promptTokenCount"], 10);
        assert_eq!(out["usageMetadata"]["candidatesTokenCount"], 5);
        assert_eq!(out["usageMetadata"]["totalTokenCount"], 15);
        // mutant: if we set finishReason to "stop" (lowercase), assertion fails
    }

    #[test]
    fn length_finish_reason_maps_to_max_tokens() {
        let chat = serde_json::json!({
            "choices": [{ "finish_reason": "length", "message": { "content": "cut off" } }],
        });
        let out = response_from_chat(&chat);
        assert_eq!(out["candidates"][0]["finishReason"], "MAX_TOKENS");
        // mutant: if we map "length" to "STOP", assertion fails
    }

    #[test]
    fn tool_calls_in_response_become_function_call_parts() {
        let chat = serde_json::json!({
            "choices": [{ "finish_reason": "tool_calls", "message": {
                "content": null,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
                }],
            }}],
            "usage": { "prompt_tokens": 8, "completion_tokens": 4 },
        });
        let out = response_from_chat(&chat);
        let parts = out["candidates"][0]["content"]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "get_weather");
        assert_eq!(parts[0]["functionCall"]["args"]["city"], "Paris");
        // finish_reason "tool_calls" → Gemini STOP (generation finished, call pending)
        assert_eq!(out["candidates"][0]["finishReason"], "STOP");
        // mutant: if we keep finish_reason "tool_calls" as-is, finishReason != "STOP"
    }

    #[test]
    fn empty_response_still_has_a_parts_array() {
        // An empty content response must still have a well-formed candidates shape.
        let chat = serde_json::json!({
            "choices": [{ "message": { "content": null } }],
        });
        let out = response_from_chat(&chat);
        assert!(out["candidates"][0]["content"]["parts"].is_array());
        // mutant: if we omit parts on empty content, is_array() fails
    }

    // ── has_untranslatable_content ──────────────────────────────────────────

    #[test]
    fn plain_text_is_translatable() {
        assert!(!has_untranslatable_content(&serde_json::json!({
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // mutant: if we return true for plain text, this fails
    }

    #[test]
    fn inline_data_is_not_translatable() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "contents": [{ "role": "user", "parts": [
                { "inlineData": { "mimeType": "image/png", "data": "AAAA" } }
            ]}],
        })));
        // mutant: if we remove the inlineData check, this fails
    }

    #[test]
    fn executable_code_is_not_translatable() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "contents": [{ "role": "user", "parts": [
                { "executableCode": { "language": "python", "code": "print(1)" } }
            ]}],
        })));
    }

    #[test]
    fn function_call_and_response_are_translatable() {
        assert!(!has_untranslatable_content(&serde_json::json!({
            "contents": [
                { "role": "model", "parts": [{ "functionCall": { "name": "f", "args": {} } }] },
                { "role": "user", "parts": [{ "functionResponse": { "name": "f", "response": {} } }] },
            ],
        })));
        // mutant: if we flag functionCall as untranslatable, this fails
    }

    #[test]
    fn function_call_with_thought_signature_is_untranslatable() {
        // Gemini 2.5+/3 returns `thoughtSignature` on functionCall parts. gemini-cli
        // replays it in every subsequent history turn, so after the first tool call
        // on those models most agentic turns will have untranslatable content and fall
        // through to observe passthrough.
        assert!(has_untranslatable_content(&serde_json::json!({
            "contents": [{
                "role": "model",
                "parts": [{
                    "functionCall": { "name": "f", "args": {} },
                    "thoughtSignature": "Aef8xyz=="
                }],
            }],
        })));
        // mutant: if we remove `obj.contains_key("thoughtSignature")` from the guard,
        // has_untranslatable_content returns false (the functionCall makes it "translatable")
        // and this assertion fails — the thoughtSignature would be silently dropped.
    }

    #[test]
    fn a_translatable_part_with_any_extra_field_is_untranslatable() {
        // Presence of `text` must not launder a sibling field the translation drops.
        for part in [
            serde_json::json!({ "text": "hi", "inlineData": { "mimeType": "image/png", "data": "AA==" } }),
            serde_json::json!({ "text": "pondering", "thought": true }),
            serde_json::json!({ "text": "ok", "thoughtSignature": "sig" }),
            serde_json::json!({}),
        ] {
            assert!(
                has_untranslatable_content(&serde_json::json!({
                    "contents": [{ "role": "user", "parts": [part.clone()] }]
                })),
                "must not translate {part}"
            );
        }
        assert!(!has_untranslatable_content(&serde_json::json!({
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }]
        })));
    }

    #[test]
    fn tool_call_ids_pair_across_turns_when_text_precedes_the_call() {
        // The normal Gemini shape: the model says something, then calls. The call id and the
        // result id must still match, or the tool result is orphaned upstream.
        let body = serde_json::json!({ "contents": [
            { "role": "user", "parts": [{ "text": "list files" }] },
            { "role": "model", "parts": [
                { "text": "Let me look." },
                { "functionCall": { "name": "ls", "args": {} } },
                { "functionCall": { "name": "ls", "args": { "path": "src" } } }
            ] },
            { "role": "user", "parts": [
                { "functionResponse": { "name": "ls", "response": { "out": "a" } } },
                { "functionResponse": { "name": "ls", "response": { "out": "b" } } }
            ] }
        ] });
        assert!(!has_untranslatable_content(&body));
        let chat = request_to_chat(&body, "gemini-2.5-flash");
        let msgs = chat["messages"].as_array().unwrap();
        let call_ids: Vec<&str> = msgs
            .iter()
            .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten())
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        let result_ids: Vec<&str> = msgs
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["tool_call_id"].as_str().unwrap())
            .collect();
        assert_eq!(call_ids.len(), 2);
        assert_eq!(call_ids, result_ids);
        assert_ne!(call_ids[0], call_ids[1]);
    }

    #[test]
    fn tool_call_ids_are_unique_across_turns() {
        // Two turns that each call `ls` first: a repeated id is rejected by Anthropic.
        let call = serde_json::json!({ "functionCall": { "name": "ls", "args": {} } });
        let resp = serde_json::json!({ "functionResponse": { "name": "ls", "response": {} } });
        let body = serde_json::json!({ "contents": [
            { "role": "user", "parts": [{ "text": "go" }] },
            { "role": "model", "parts": [call.clone()] },
            { "role": "user", "parts": [resp.clone()] },
            { "role": "model", "parts": [call] },
            { "role": "user", "parts": [resp] }
        ] });
        assert!(!has_untranslatable_content(&body));
        let chat = request_to_chat(&body, "gemini-2.5-flash");
        let msgs = chat["messages"].as_array().unwrap();
        let call_ids: Vec<&str> = msgs
            .iter()
            .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten())
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        let result_ids: Vec<&str> = msgs
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(|m| m["tool_call_id"].as_str().unwrap())
            .collect();
        assert_eq!(call_ids.len(), 2);
        assert_ne!(call_ids[0], call_ids[1]);
        assert_eq!(call_ids, result_ids);
    }

    #[test]
    fn ambiguous_function_pairing_and_unknown_fields_are_untranslatable() {
        let call = |n: &str| serde_json::json!({ "functionCall": { "name": n, "args": {} } });
        let resp =
            |n: &str| serde_json::json!({ "functionResponse": { "name": n, "response": {} } });
        let turns = |model: Vec<Value>, user: Vec<Value>| {
            serde_json::json!({ "contents": [
            { "role": "user", "parts": [{ "text": "go" }] },
            { "role": "model", "parts": model },
            { "role": "user", "parts": user }
        ] })
        };
        // Out of order, missing one, extra text alongside results, response with no call.
        for body in [
            turns(vec![call("a"), call("b")], vec![resp("b"), resp("a")]),
            turns(vec![call("a"), call("b")], vec![resp("a")]),
            turns(
                vec![call("a")],
                vec![resp("a"), serde_json::json!({ "text": "also" })],
            ),
            turns(vec![serde_json::json!({ "text": "hi" })], vec![resp("a")]),
            serde_json::json!({ "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }], "labels": { "a": "b" } }),
        ] {
            assert!(
                has_untranslatable_content(&body),
                "must not translate {body}"
            );
        }
        assert!(!has_untranslatable_content(&turns(
            vec![call("a")],
            vec![resp("a")]
        )));
    }

    #[test]
    fn no_contents_is_not_flagged() {
        // A body with no contents (just model/generationConfig) should not be considered
        // untranslatable — the caller decides whether it's a valid request.
        assert!(!has_untranslatable_content(&serde_json::json!({
            "generationConfig": { "maxOutputTokens": 128 },
        })));
    }

    #[test]
    fn cached_content_is_not_translatable() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "cachedContent": "projects/p/cachedContents/c",
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // mutant: remove `cachedContent` check → returns false
    }

    #[test]
    fn tool_config_is_not_translatable() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "toolConfig": { "functionCallingConfig": { "mode": "ANY" } },
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // mutant: remove `toolConfig` check → returns false
    }

    #[test]
    fn safety_settings_are_not_translatable() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "safetySettings": [{ "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" }],
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // mutant: remove `safetySettings` check → returns false
    }

    #[test]
    fn non_function_declaration_tool_is_not_translatable() {
        // googleSearch has no Chat equivalent — it must be passed through unmodified.
        assert!(has_untranslatable_content(&serde_json::json!({
            "tools": [{ "googleSearch": {} }],
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // A mix of functionDeclarations and googleSearch is also not translatable.
        assert!(has_untranslatable_content(&serde_json::json!({
            "tools": [{ "functionDeclarations": [{ "name": "f" }], "googleSearch": {} }],
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // Pure functionDeclarations is translatable.
        assert!(!has_untranslatable_content(&serde_json::json!({
            "tools": [{ "functionDeclarations": [{ "name": "f" }] }],
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // mutant: remove the tools check → first assert returns false
    }

    #[test]
    fn system_instruction_with_non_text_part_is_not_translatable() {
        assert!(has_untranslatable_content(&serde_json::json!({
            "systemInstruction": { "parts": [{ "inlineData": { "mimeType": "image/png", "data": "AA==" } }] },
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // Text-only systemInstruction is fine.
        assert!(!has_untranslatable_content(&serde_json::json!({
            "systemInstruction": { "parts": [{ "text": "be helpful" }] },
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // mutant: remove the systemInstruction part check → first assert returns false
    }

    #[test]
    fn unmapped_generation_config_field_is_not_translatable() {
        // responseMimeType has no direct Chat equivalent.
        assert!(has_untranslatable_content(&serde_json::json!({
            "generationConfig": { "responseMimeType": "application/json" },
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // topK is not in the mapped set.
        assert!(has_untranslatable_content(&serde_json::json!({
            "generationConfig": { "maxOutputTokens": 512, "topK": 40 },
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // All mapped fields together are fine.
        assert!(!has_untranslatable_content(&serde_json::json!({
            "generationConfig": { "maxOutputTokens": 512, "temperature": 0.7, "topP": 0.9 },
            "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }],
        })));
        // mutant: remove the generationConfig check → first assert returns false
    }

    // ── SSE format ───────────────────────────────────────────────────────────

    #[test]
    fn sse_from_response_is_parseable_data_line() {
        let resp = serde_json::json!({ "candidates": [{}], "usageMetadata": {} });
        let sse = sse_from_response(&resp);
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
        // The content after "data: " must be valid JSON.
        let json_str = sse.strip_prefix("data: ").unwrap().trim_end();
        serde_json::from_str::<Value>(json_str).expect("SSE data must be valid JSON");
        // mutant: if we output "data:" without the space, the SDK would still parse it
        // (it trims); but the test confirms the exact format is well-formed
    }

    #[test]
    fn sse_error_response_has_error_field() {
        let sse = sse_error_response("all rungs failed");
        let json_str = sse.strip_prefix("data: ").unwrap().trim_end();
        let v: Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(v["error"]["message"], "all rungs failed");
        // mutant: if we omit the error field, v["error"]["message"] is null
    }

    // ── credential scoping ───────────────────────────────────────────────────

    #[test]
    fn gemini_key_only_populated_on_gemini_dialect() {
        use crate::provider::Auth;
        use axum::http::HeaderMap;
        use firstpass_core::Dialect;

        let mut h = HeaderMap::new();
        h.insert("x-goog-api-key", "AIza-google-key".parse().unwrap());

        // On the Gemini path: x-goog-api-key → gemini_key only
        let auth = Auth::from_headers(&h, Dialect::Gemini);
        assert_eq!(auth.gemini_key.as_deref(), Some("AIza-google-key"));
        assert!(
            auth.anthropic_key.is_none(),
            "Google key must not go to Anthropic"
        );
        // openai_key populated from OPENAI_API_KEY env for mixed-ladder rungs — not from x-goog-api-key
        assert_ne!(
            auth.openai_key.as_deref(),
            Some("AIza-google-key"),
            "Google key must not go to OpenAI path"
        );

        // On Anthropic/OpenAI paths: x-goog-api-key is ignored
        let auth_ant = Auth::from_headers(&h, Dialect::Anthropic);
        assert!(
            auth_ant.gemini_key.is_none(),
            "Gemini key must not be set on Anthropic path"
        );

        let auth_oai = Auth::from_headers(&h, Dialect::Openai);
        assert!(
            auth_oai.gemini_key.is_none(),
            "Gemini key must not be set on OpenAI path"
        );
        // mutant: if from_headers sets gemini_key on Anthropic/OpenAI, assertion fails
    }
}
