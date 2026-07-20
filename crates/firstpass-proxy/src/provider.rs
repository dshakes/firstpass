//! Normalized model access: a provider-agnostic request/response shape, and the wire adapters
//! (Anthropic Messages, OpenAI Chat Completions, Google Gemini) that speak it. The router
//! ([`crate::router`]) only ever talks to [`Provider`]; it never knows which wire format is
//! behind a given rung.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One message in a normalized chat conversation. `content` is either a plain string (the common
/// case — serializes byte-identically to a text message) OR the original array of content blocks
/// (`text` / `tool_use` / `tool_result` / `image`), forwarded verbatim so tool and multimodal turns
/// survive the enforce path (ADR 0005). Use [`ChatMessage::text_view`] to read a text projection for
/// gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `"user"`, `"assistant"`, or `"system"`.
    pub role: String,
    /// Message content — a string, or an array of content blocks, forwarded as-is to the provider.
    pub content: Value,
}

impl ChatMessage {
    /// A text-only message (the common path).
    #[must_use]
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Value::String(content.into()),
        }
    }

    /// Concatenate the text this message carries, for gating. A string is itself; an array yields the
    /// joined `text` blocks (tool_use/tool_result/image blocks contribute no gating text).
    #[must_use]
    pub fn text_view(&self) -> String {
        match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }
}

/// A provider-agnostic model request, built once per incoming call and re-used (with
/// `model` swapped) across every rung of the ladder.
///
// Message content is carried verbatim (string or content-block array, ADR 0005); gates read a text
// projection via `ChatMessage::text_view`. `tools` is opaque passthrough.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    /// `provider/model`, e.g. `"anthropic/claude-haiku-4-5"`.
    pub model: String,
    /// System prompt, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Conversation turns.
    pub messages: Vec<ChatMessage>,
    /// Max tokens to generate.
    pub max_tokens: u32,
    /// Opaque tool/function-calling passthrough, forwarded as-is to the wire provider.
    #[serde(default)]
    pub tools: Value,
    /// The **full original inbound request JSON**, when this request came off the proxy's wire
    /// (ADR 0005). Anthropic-dialect providers send it verbatim with only `model` swapped (and
    /// `stream` stripped), so every field — `tools`, `tool_choice`, `temperature`, `thinking`,
    /// `stop_sequences`, … — survives the rung exactly as the caller wrote it. `Null` for
    /// synthesized requests (judge / consistency samples), which use the normalized fields.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub raw: Value,
}

/// A provider-agnostic model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// `provider/model` that produced this response.
    pub model: String,
    /// Concatenated text output.
    pub text: String,
    /// Input tokens billed.
    pub in_tokens: u64,
    /// Output tokens billed.
    pub out_tokens: u64,
    /// The raw wire response, kept for debugging/audit — never logged wholesale.
    pub raw: Value,
}

/// Failure modes of a provider call.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProviderError {
    /// The request never got a response (connection failure, timeout).
    #[error("transport error: {0}")]
    Transport(String),
    /// The provider responded with a non-2xx status.
    #[error("http {status}: {body}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Response body (truncated upstream, not by us).
        body: String,
    },
    /// The response body didn't parse into the shape we expected.
    #[error("decode error: {0}")]
    Decode(String),
}

impl ProviderError {
    /// Whether this failure should trigger cross-rung/cross-provider failover (transport
    /// errors and 5xx) rather than being treated as a hard, non-retryable error (4xx, decode).
    #[must_use]
    pub fn is_failover_eligible(&self) -> bool {
        match self {
            ProviderError::Transport(_) => true,
            ProviderError::Http { status, .. } => *status >= 500,
            ProviderError::Decode(_) => false,
        }
    }
}

/// BYOK credentials for one request, extracted from headers with env-var fallback.
///
/// Never logged or persisted — [`std::fmt::Debug`] redacts both fields.
#[derive(Clone, Default)]
pub struct Auth {
    /// Anthropic API key (`x-api-key` header, or `ANTHROPIC_API_KEY`).
    pub anthropic_key: Option<String>,
    /// OpenAI API key (`authorization: Bearer ...` header, or `OPENAI_API_KEY`).
    pub openai_key: Option<String>,
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Auth")
            .field("anthropic_key", &self.anthropic_key.as_ref().map(|_| "***"))
            .field("openai_key", &self.openai_key.as_ref().map(|_| "***"))
            .finish()
    }
}

impl Auth {
    /// Extract BYOK credentials from request headers, falling back to `ANTHROPIC_API_KEY` /
    /// `OPENAI_API_KEY` environment variables.
    #[must_use]
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let anthropic_key = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());
        let openai_key = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(str::to_owned)
            .or_else(|| std::env::var("OPENAI_API_KEY").ok());
        Self {
            anthropic_key,
            openai_key,
        }
    }
}

/// A normalized model backend. Implementations translate [`ModelRequest`]/[`ModelResponse`]
/// to/from one wire API.
#[async_trait]
pub trait Provider: Send + Sync + std::fmt::Debug {
    /// Call the model and normalize the result.
    ///
    /// # Errors
    /// Returns [`ProviderError`] on transport failure, a non-2xx response, or a response
    /// that doesn't decode into the expected shape.
    async fn complete(
        &self,
        req: &ModelRequest,
        auth: &Auth,
    ) -> Result<ModelResponse, ProviderError>;

    /// Provider identity, e.g. `"anthropic"`.
    fn id(&self) -> &str;

    /// Whether this provider carries Anthropic-shaped structured content (tool_use / tool_result /
    /// image blocks and top-level `tools`) **verbatim** on the wire. The enforce path's fidelity
    /// guard (ADR 0005) only routes structured requests through ladders where every rung's
    /// provider returns `true`; a dialect that would need translation returns `false` until that
    /// translation exists — falling back to transparent passthrough is always safe, corrupting a
    /// tool turn never is.
    fn carries_structured_verbatim(&self) -> bool {
        false
    }
}

/// Strip the `provider/` prefix from a ladder model id for the provider's wire API — Anthropic and
/// OpenAI expect the bare model (`claude-haiku-4-5`), not `anthropic/claude-haiku-4-5`. The full
/// prefixed id is still what the ladder/trace use; only the wire call is stripped.
fn wire_model(model: &str) -> &str {
    model.split_once('/').map_or(model, |(_, m)| m)
}

/// Resolve the API key for a provider call. A configured `api_key_env` wins (that env var is *this*
/// provider's key — e.g. `GROQ_API_KEY` for a Groq rung); otherwise fall back to the per-request
/// BYOK override from headers/env (the built-in `anthropic`/`openai` path). Empty string when
/// neither is set (a keyless local endpoint).
fn resolve_api_key(api_key_env: Option<&str>, byok_override: Option<&str>) -> String {
    api_key_env
        .and_then(|e| std::env::var(e).ok())
        .or_else(|| byok_override.map(str::to_owned))
        .unwrap_or_default()
}

/// Build the Anthropic Messages wire body for `req`: the original inbound JSON **verbatim** with
/// only `model` swapped to the rung's wire id and `stream` stripped (`complete` is the buffered
/// call — the gate needs the whole candidate). Falls back to the normalized fields when there is
/// no raw body (synthesized judge/consistency requests). Pure, so fidelity is unit-testable.
#[must_use]
pub fn anthropic_wire_body(req: &ModelRequest) -> Value {
    if let Value::Object(raw) = &req.raw {
        let mut body = raw.clone();
        body.insert(
            "model".to_owned(),
            Value::String(wire_model(&req.model).to_owned()),
        );
        body.remove("stream");
        return Value::Object(body);
    }
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();
    let mut body = serde_json::json!({
        "model": wire_model(&req.model),
        "max_tokens": req.max_tokens,
        "messages": messages,
    });
    if let Some(system) = req.system.as_deref() {
        body["system"] = serde_json::json!(system);
    }
    if !req.tools.is_null() {
        body["tools"] = req.tools.clone();
    }
    body
}

/// Speaks `POST {base}/v1/messages` (Anthropic Messages API).
///
// LIVE-VERIFIED (2026-07-10): exercised against real Anthropic through the running proxy's enforce
// path — a haiku completion served end-to-end. The `anthropic/` prefix must be stripped for the
// wire call (see `wire_model`); sending it verbatim 404s.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    /// Ladder prefix / trace label for this provider (usually `"anthropic"`).
    pub id: String,
    /// Base URL, e.g. `https://api.anthropic.com`.
    pub base_url: String,
    /// Env var the API key is read from when no per-request BYOK header is present. `None` for the
    /// built-in provider, which resolves the key via [`Auth`] (`x-api-key` header or env).
    pub api_key_env: Option<String>,
    /// Shared, connection-pooled HTTP client.
    pub http: reqwest::Client,
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn carries_structured_verbatim(&self) -> bool {
        true
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        let key = resolve_api_key(self.api_key_env.as_deref(), auth.anthropic_key.as_deref());
        let body = anthropic_wire_body(req);

        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        if !status.is_success() {
            return Err(ProviderError::Http {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }

        let json: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Decode(e.to_string()))?;
        let (text, in_tokens, out_tokens) = anthropic_parse_response(&json)?;

        Ok(ModelResponse {
            model: req.model.clone(),
            text,
            in_tokens,
            out_tokens,
            raw: json,
        })
    }
}

/// Extract `(text, in_tokens, out_tokens)` from an Anthropic Messages API response. Shared by
/// [`AnthropicProvider`] and every auth scheme that wraps the Anthropic body shape (Bedrock,
/// Vertex — ADR 0006): they all get the same response back, only the request's auth/URL differ.
fn anthropic_parse_response(json: &Value) -> Result<(String, u64, u64), ProviderError> {
    let text = json
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .ok_or_else(|| ProviderError::Decode("missing content[].text".to_owned()))?;

    let in_tokens = json
        .pointer("/usage/input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let out_tokens = json
        .pointer("/usage/output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    Ok((text, in_tokens, out_tokens))
}

/// Build an Anthropic Messages-shaped request body with the model **omitted** — Bedrock and Vertex
/// both put the model in the URL, not the body (ADR 0006 P2/P3), unlike direct Anthropic API calls
/// which need `model` in the body ([`AnthropicWireRequest`]). `anthropic_version` is the dialect
/// version string each host expects (`bedrock-2023-05-31` / `vertex-2023-10-16`).
fn anthropic_messages_body(req: &ModelRequest, anthropic_version: &str) -> Value {
    // Same verbatim-carry rule as `anthropic_wire_body`, adapted to hosts that put the model in
    // the URL: original inbound JSON minus `model`/`stream`, plus the host's version string.
    if let Value::Object(raw) = &req.raw {
        let mut body = raw.clone();
        body.remove("model");
        body.remove("stream");
        body.insert(
            "anthropic_version".to_owned(),
            Value::String(anthropic_version.to_owned()),
        );
        return Value::Object(body);
    }
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();
    let mut body = serde_json::json!({
        "anthropic_version": anthropic_version,
        "max_tokens": req.max_tokens,
        "messages": messages,
    });
    if let Some(system) = req.system.as_deref() {
        body["system"] = serde_json::json!(system);
    }
    if !req.tools.is_null() {
        body["tools"] = req.tools.clone();
    }
    body
}

#[derive(Serialize)]
struct OpenAiWireMessage<'a> {
    role: &'a str,
    content: Value,
}

#[derive(Serialize)]
struct OpenAiWireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<OpenAiWireMessage<'a>>,
}

/// Speaks `POST {base}/v1/chat/completions` (OpenAI Chat Completions API).
///
// LIVE-UNVERIFIED: the `wire_model` prefix-strip fix is applied here too, but the OpenAI path has
// not yet been exercised against a real endpoint (only Anthropic has). Verify against a real key
// before relying on it in production.
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    /// Ladder prefix / trace label for this provider (`"openai"`, `"groq"`, `"together"`, …).
    pub id: String,
    /// Base URL, e.g. `https://api.openai.com` or `https://api.groq.com/openai`.
    pub base_url: String,
    /// Env var the API key is read from, e.g. `"GROQ_API_KEY"`. `None` for the built-in `openai`
    /// provider (resolves via [`Auth`]: `authorization` header or `OPENAI_API_KEY`) and for keyless
    /// local endpoints (Ollama / vLLM).
    pub api_key_env: Option<String>,
    /// Shared, connection-pooled HTTP client.
    pub http: reqwest::Client,
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        let key = resolve_api_key(self.api_key_env.as_deref(), auth.openai_key.as_deref());
        let mut messages = Vec::with_capacity(req.messages.len() + 1);
        if let Some(system) = req.system.as_deref() {
            messages.push(OpenAiWireMessage {
                role: "system",
                content: Value::String(system.to_owned()),
            });
        }
        messages.extend(req.messages.iter().map(|m| OpenAiWireMessage {
            role: &m.role,
            content: m.content.clone(),
        }));
        let body = OpenAiWireRequest {
            model: wire_model(&req.model),
            max_tokens: req.max_tokens,
            messages,
        };

        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let resp = self
            .http
            .post(url)
            .header(axum::http::header::AUTHORIZATION, format!("Bearer {key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        if !status.is_success() {
            return Err(ProviderError::Http {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }

        let json: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Decode(e.to_string()))?;

        let text = json
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| ProviderError::Decode("missing choices[0].message.content".to_owned()))?
            .to_owned();

        let in_tokens = json
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let out_tokens = json
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);

        Ok(ModelResponse {
            model: req.model.clone(),
            text,
            in_tokens,
            out_tokens,
            raw: json,
        })
    }
}

/// Build the Gemini `generateContent` request body from a normalized [`ModelRequest`]. Split out
/// (pure, no I/O) so the translation is unit-tested offline. Gemini uses `contents` with roles
/// `user` / `model` (not `assistant`) and a separate `system_instruction`; the system prompt and the
/// per-message text projection ([`ChatMessage::text_view`]) are what we send. Tool/multimodal blocks
/// for Gemini (its `functionCall` / `functionResponse` / `inlineData` shapes) are a follow-on — this
/// adapter routes text, matching the OpenAI adapter's current scope.
fn gemini_request_body(req: &ModelRequest) -> Value {
    let contents: Vec<Value> = req
        .messages
        .iter()
        .map(|m| {
            let role = if m.role == "assistant" {
                "model"
            } else {
                "user"
            };
            serde_json::json!({ "role": role, "parts": [{ "text": m.text_view() }] })
        })
        .collect();
    let mut body = serde_json::json!({
        "contents": contents,
        "generationConfig": { "maxOutputTokens": req.max_tokens },
    });
    if let Some(system) = req.system.as_deref() {
        body["system_instruction"] = serde_json::json!({ "parts": [{ "text": system }] });
    }
    body
}

/// Extract `(text, in_tokens, out_tokens)` from a Gemini `generateContent` response. `text` is the
/// concatenation of the first candidate's text parts; token counts come from `usageMetadata`.
fn gemini_parse_response(json: &Value) -> Result<(String, u64, u64), ProviderError> {
    let parts = json
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .ok_or_else(|| ProviderError::Decode("missing candidates[0].content.parts".to_owned()))?;
    let text = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("");
    let in_tokens = json
        .pointer("/usageMetadata/promptTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let out_tokens = json
        .pointer("/usageMetadata/candidatesTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Ok((text, in_tokens, out_tokens))
}

/// Speaks `POST {base}/v1beta/models/{model}:generateContent` (Google Gemini Generative Language
/// API). The API key goes in the `x-goog-api-key` header — never the URL query string, so it stays
/// out of logs and proxies.
///
// LIVE-UNVERIFIED: the request/response translation is unit-tested offline; it has not yet been
// exercised against a real Gemini endpoint. Verify against a real key before relying on it.
#[derive(Debug, Clone)]
pub struct GeminiProvider {
    /// Ladder prefix / trace label (usually `"gemini"` or `"google"`).
    pub id: String,
    /// Base URL, e.g. `https://generativelanguage.googleapis.com`.
    pub base_url: String,
    /// Env var the API key is read from, e.g. `"GEMINI_API_KEY"`.
    pub api_key_env: Option<String>,
    /// Shared, connection-pooled HTTP client.
    pub http: reqwest::Client,
}

#[async_trait]
impl Provider for GeminiProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        let key = resolve_api_key(self.api_key_env.as_deref(), auth.openai_key.as_deref());
        let body = gemini_request_body(req);
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            wire_model(&req.model),
        );
        let resp = self
            .http
            .post(url)
            .header("x-goog-api-key", key)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(ProviderError::Http {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }

        let json: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Decode(e.to_string()))?;
        let (text, in_tokens, out_tokens) = gemini_parse_response(&json)?;
        Ok(ModelResponse {
            model: req.model.clone(),
            text,
            in_tokens,
            out_tokens,
            raw: json,
        })
    }
}

/// AWS credentials read fresh from the standard env vars at call time — never cached, never
/// logged (ADR 0006 P2). `AWS_SESSION_TOKEN` is optional (long-lived IAM user keys have none;
/// STS-issued temporary credentials do).
struct AwsEnvCredentials {
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
}

impl AwsEnvCredentials {
    fn from_env() -> Result<Self, ProviderError> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| ProviderError::Transport("AWS_ACCESS_KEY_ID is not set".to_owned()))?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| ProviderError::Transport("AWS_SECRET_ACCESS_KEY is not set".to_owned()))?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        Ok(Self {
            access_key_id,
            secret_access_key,
            session_token,
        })
    }
}

/// Build the Bedrock `invoke` URL for a region + wire model id.
fn bedrock_url(region: &str, model: &str) -> String {
    format!("https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke")
}

/// SigV4-sign a Bedrock `InvokeModel` request, returning the fully-built (headers + body) HTTP
/// request ready to hand to reqwest. Delegates all canonical-request construction and HMAC signing
/// to `aws-sigv4` (ADR 0006 I3) — this function only wires host/service/region into that crate's
/// API and applies the resulting signature. Split out from [`BedrockProvider::complete`] so the
/// signing shape (not its cryptographic validity, which only a real AWS call can prove) is
/// unit-testable offline with dummy credentials.
///
// LIVE-UNVERIFIED: tests assert the produced request carries an `AWS4-HMAC-SHA256` authorization
// header and the expected host, against dummy credentials. Signature validity is only provable
// against a real Bedrock endpoint.
fn sign_bedrock(
    url: &str,
    region: &str,
    body: &[u8],
    creds: &AwsEnvCredentials,
) -> Result<http::Request<Vec<u8>>, ProviderError> {
    let host = url
        .parse::<http::Uri>()
        .ok()
        .and_then(|u| u.host().map(str::to_owned))
        .ok_or_else(|| ProviderError::Transport(format!("invalid bedrock URL: {url}")))?;

    let identity: aws_smithy_runtime_api::client::identity::Identity =
        aws_credential_types::Credentials::new(
            creds.access_key_id.clone(),
            creds.secret_access_key.clone(),
            creds.session_token.clone(),
            None,
            "firstpass",
        )
        .into();

    let signing_params: aws_sigv4::http_request::SigningParams<'_> =
        aws_sigv4::sign::v4::SigningParams::builder()
            .identity(&identity)
            .region(region)
            .name("bedrock")
            .time(SystemTime::now())
            .settings(aws_sigv4::http_request::SigningSettings::default())
            .build()
            .map_err(|e| ProviderError::Transport(format!("sigv4 signing params: {e}")))?
            .into();

    let headers = [
        ("host", host.as_str()),
        ("content-type", "application/json"),
    ];
    let signable = aws_sigv4::http_request::SignableRequest::new(
        "POST",
        url,
        headers.into_iter(),
        aws_sigv4::http_request::SignableBody::Bytes(body),
    )
    .map_err(|e| ProviderError::Transport(format!("sigv4 signable request: {e}")))?;

    let (instructions, _signature) = aws_sigv4::http_request::sign(signable, &signing_params)
        .map_err(|e| ProviderError::Transport(format!("sigv4 sign: {e}")))?
        .into_parts();

    let mut req = http::Request::builder()
        .method("POST")
        .uri(url)
        .header("host", host)
        .header("content-type", "application/json")
        .body(body.to_vec())
        .map_err(|e| ProviderError::Transport(format!("build bedrock request: {e}")))?;
    instructions.apply_to_request_http1x(&mut req);
    Ok(req)
}

/// Speaks `POST https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke` (Claude on AWS
/// Bedrock) — an Anthropic-shaped body ([`anthropic_messages_body`]) authenticated with AWS SigV4
/// request signing rather than an API key (ADR 0006 P2). Credentials come from the standard
/// `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` env vars, read fresh per
/// call — never logged, never put in the URL (I2).
///
// LIVE-UNVERIFIED: the request/response translation and signing call shape are unit-tested
// offline; this has not been exercised against a real Bedrock endpoint. Verify against real AWS
// credentials before relying on it in production.
#[derive(Debug, Clone)]
pub struct BedrockProvider {
    /// Ladder prefix / trace label (usually `"bedrock"`).
    pub id: String,
    /// AWS region Bedrock is called in, e.g. `"us-east-1"`.
    pub region: Option<String>,
    /// Shared, connection-pooled HTTP client.
    pub http: reqwest::Client,
}

#[async_trait]
impl Provider for BedrockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn carries_structured_verbatim(&self) -> bool {
        true
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        _auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        let region = self.region.as_deref().ok_or_else(|| {
            ProviderError::Transport("bedrock provider requires a region".to_owned())
        })?;
        let model = wire_model(&req.model);
        let url = bedrock_url(region, model);
        let body = anthropic_messages_body(req, "bedrock-2023-05-31");
        let body_bytes =
            serde_json::to_vec(&body).map_err(|e| ProviderError::Decode(e.to_string()))?;

        let creds = AwsEnvCredentials::from_env()?;
        let signed = sign_bedrock(&url, region, &body_bytes, &creds)?;
        let http_req = reqwest::Request::try_from(signed)
            .map_err(|e| ProviderError::Transport(format!("build reqwest request: {e}")))?;

        let resp = self
            .http
            .execute(http_req)
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(ProviderError::Http {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }

        let json: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Decode(e.to_string()))?;
        let (text, in_tokens, out_tokens) = anthropic_parse_response(&json)?;
        Ok(ModelResponse {
            model: req.model.clone(),
            text,
            in_tokens,
            out_tokens,
            raw: json,
        })
    }
}

/// Build the Vertex AI `rawPredict` URL for a region + project + wire model id (Claude on Vertex).
fn vertex_url(region: &str, project: &str, model: &str) -> String {
    format!(
        "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:rawPredict"
    )
}

/// Speaks `POST https://{region}-aiplatform.googleapis.com/.../publishers/anthropic/models/{model}:rawPredict`
/// (Claude on Google Vertex AI) — an Anthropic-shaped body ([`anthropic_messages_body`])
/// authenticated with a GCP OAuth2 bearer token rather than an API key (ADR 0006 P3). The token is
/// minted and cached by `gcp_auth` from `GOOGLE_APPLICATION_CREDENTIALS` or the ambient GCP
/// environment — this adapter never caches or logs it, and it never goes in the URL (I2).
///
// LIVE-UNVERIFIED: the request/response translation is unit-tested offline; this has not been
// exercised against a real Vertex endpoint. Verify against a real service account before relying
// on it in production.
#[derive(Debug, Clone)]
pub struct VertexProvider {
    /// Ladder prefix / trace label (usually `"vertex"`).
    pub id: String,
    /// GCP region Vertex is called in, e.g. `"us-central1"`.
    pub region: Option<String>,
    /// GCP project id.
    pub project: Option<String>,
    /// Shared, connection-pooled HTTP client.
    pub http: reqwest::Client,
}

#[async_trait]
impl Provider for VertexProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn carries_structured_verbatim(&self) -> bool {
        true
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        _auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        let region = self.region.as_deref().ok_or_else(|| {
            ProviderError::Transport("vertex provider requires a region".to_owned())
        })?;
        let project = self.project.as_deref().ok_or_else(|| {
            ProviderError::Transport("vertex provider requires a project".to_owned())
        })?;
        let model = wire_model(&req.model);
        let url = vertex_url(region, project, model);
        let body = anthropic_messages_body(req, "vertex-2023-10-16");

        // ponytail: gcp_auth re-detects the auth method per call (the token itself is cached inside
        // the provider). Cache the provider in a OnceCell if Vertex ever becomes a hot path.
        let provider = gcp_auth::provider()
            .await
            .map_err(|e| ProviderError::Transport(format!("gcp_auth provider: {e}")))?;
        let token = provider
            .token(&["https://www.googleapis.com/auth/cloud-platform"])
            .await
            .map_err(|e| ProviderError::Transport(format!("gcp_auth token: {e}")))?;

        let resp = self
            .http
            .post(url)
            .header(
                axum::http::header::AUTHORIZATION,
                format!("Bearer {}", token.as_str()),
            )
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = resp.status();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(ProviderError::Http {
                status: status.as_u16(),
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }

        let json: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Decode(e.to_string()))?;
        let (text, in_tokens, out_tokens) = anthropic_parse_response(&json)?;
        Ok(ModelResponse {
            model: req.model.clone(),
            text,
            in_tokens,
            out_tokens,
            raw: json,
        })
    }
}

/// Lookup from provider id (`"anthropic"`, `"openai"`, ...) to the [`Provider`] that serves it.
#[derive(Clone)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRegistry")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ProviderRegistry {
    /// One HTTP client shared by every provider. The enforce path is request/response (never
    /// streamed through the adapter), so it carries a total request timeout as well as a connect
    /// timeout — a hung or slow upstream can't pin a routing decision indefinitely. Falls back to a
    /// default client if the builder fails (only on TLS backend init, which is fatal anyway).
    fn build_http_client() -> reqwest::Client {
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    }

    /// Build the standard registry: the built-in `anthropic` + `openai` providers.
    #[must_use]
    pub fn new(anthropic_base: impl Into<String>, openai_base: impl Into<String>) -> Self {
        Self::from_config(&[], anthropic_base, openai_base)
    }

    /// Build the registry with the built-in `anthropic` / `openai` providers plus every configured
    /// `[[provider]]` entry — so a ladder can route to any OpenAI-compatible or Anthropic-compatible
    /// endpoint (Groq, Together, Fireworks, DeepSeek, Mistral, xAI, OpenRouter, Ollama, vLLM, Azure,
    /// …) by id. A `[[provider]]` whose id is `anthropic` or `openai` overrides the built-in default
    /// (e.g. to point `openai` at Azure). Shares one HTTP client across all of them.
    #[must_use]
    pub fn from_config(
        defs: &[firstpass_core::ProviderDef],
        anthropic_base: impl Into<String>,
        openai_base: impl Into<String>,
    ) -> Self {
        let http = Self::build_http_client();
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            Arc::new(AnthropicProvider {
                id: "anthropic".to_owned(),
                base_url: anthropic_base.into(),
                api_key_env: None,
                http: http.clone(),
            }),
        );
        providers.insert(
            "openai".to_owned(),
            Arc::new(OpenAiProvider {
                id: "openai".to_owned(),
                base_url: openai_base.into(),
                api_key_env: None,
                http: http.clone(),
            }),
        );
        for def in defs {
            // Auth scheme comes first (ADR 0006): `aws_sigv4`/`gcp_oauth` are bespoke-auth
            // backends that wrap the Anthropic body shape regardless of `dialect`; `api_key`
            // (the default) is today's dialect-driven dispatch, unchanged.
            let provider: Arc<dyn Provider> = match def.auth {
                firstpass_core::AuthScheme::AwsSigv4 => Arc::new(BedrockProvider {
                    id: def.id.clone(),
                    region: def.region.clone(),
                    http: http.clone(),
                }),
                firstpass_core::AuthScheme::GcpOauth => Arc::new(VertexProvider {
                    id: def.id.clone(),
                    region: def.region.clone(),
                    project: def.project.clone(),
                    http: http.clone(),
                }),
                firstpass_core::AuthScheme::ApiKey => match def.dialect {
                    firstpass_core::Dialect::Anthropic => Arc::new(AnthropicProvider {
                        id: def.id.clone(),
                        base_url: def.base_url.clone(),
                        api_key_env: def.api_key_env.clone(),
                        http: http.clone(),
                    }),
                    firstpass_core::Dialect::Openai => Arc::new(OpenAiProvider {
                        id: def.id.clone(),
                        base_url: def.base_url.clone(),
                        api_key_env: def.api_key_env.clone(),
                        http: http.clone(),
                    }),
                    firstpass_core::Dialect::Gemini => Arc::new(GeminiProvider {
                        id: def.id.clone(),
                        base_url: def.base_url.clone(),
                        api_key_env: def.api_key_env.clone(),
                        http: http.clone(),
                    }),
                },
            };
            providers.insert(def.id.clone(), provider);
        }
        Self { providers }
    }

    /// Build a registry from arbitrary providers — used to wire up [`MockProvider`]s in tests.
    #[must_use]
    pub fn from_map(providers: HashMap<String, Arc<dyn Provider>>) -> Self {
        Self { providers }
    }

    /// Look up a provider by id.
    #[must_use]
    pub fn get(&self, provider_id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(provider_id).cloned()
    }
}

/// Test-only provider: returns a pre-programmed outcome per model, deterministically.
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub struct MockProvider {
    id: String,
    outcomes: HashMap<String, Result<ModelResponse, ProviderError>>,
    /// Every model string `complete()` was called with — lets speculation tests assert which rungs
    /// were actually fired. Shared (`Arc`) so a clone taken before boxing still observes the calls.
    calls: Arc<std::sync::Mutex<Vec<String>>>,
    /// Simulated per-call latency (0 = respond instantly). Lets a test measure the wall-clock win
    /// speculation buys by overlapping rung calls that would otherwise run serially.
    delay_ms: u64,
}

#[cfg(test)]
impl MockProvider {
    /// Build a mock provider that answers `outcomes[model]` for `complete()`.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        outcomes: HashMap<String, Result<ModelResponse, ProviderError>>,
    ) -> Self {
        Self {
            id: id.into(),
            outcomes,
            calls: Arc::default(),
            delay_ms: 0,
        }
    }

    /// Make `complete()` sleep `ms` before responding, to simulate real per-call latency.
    #[must_use]
    pub fn with_delay(mut self, ms: u64) -> Self {
        self.delay_ms = ms;
        self
    }

    /// A handle to the shared call log; clone it before boxing the provider into a registry, then
    /// inspect the models `complete()` saw after the engine runs.
    #[must_use]
    pub fn call_log(&self) -> Arc<std::sync::Mutex<Vec<String>>> {
        Arc::clone(&self.calls)
    }
}

#[cfg(test)]
#[async_trait]
impl Provider for MockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn carries_structured_verbatim(&self) -> bool {
        true
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        _auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        self.calls.lock().unwrap().push(req.model.clone());
        if self.delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
        }
        self.outcomes.get(&req.model).cloned().unwrap_or_else(|| {
            Err(ProviderError::Decode(format!(
                "no mock outcome configured for {}",
                req.model
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_model_strips_the_provider_prefix() {
        // Regression: sending "anthropic/claude-haiku-4-5" verbatim 404s at the provider.
        assert_eq!(wire_model("anthropic/claude-haiku-4-5"), "claude-haiku-4-5");
        assert_eq!(wire_model("openai/gpt-5.5"), "gpt-5.5");
        assert_eq!(wire_model("claude-opus-4-8"), "claude-opus-4-8"); // no prefix → unchanged
    }

    #[test]
    fn gemini_request_maps_roles_and_system_instruction() {
        let req = ModelRequest {
            model: "gemini/gemini-2.0-flash".to_owned(),
            system: Some("be terse".to_owned()),
            messages: vec![
                ChatMessage::text("user", "hi"),
                ChatMessage::text("assistant", "hello"),
            ],
            max_tokens: 256,
            tools: Value::Null,
            raw: Value::Null,
        };
        let body = gemini_request_body(&req);
        // System prompt goes in system_instruction, not contents.
        assert_eq!(body["system_instruction"]["parts"][0]["text"], "be terse");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 256);
        // Anthropic's "assistant" role becomes Gemini's "model"; "user" stays "user".
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
        assert_eq!(body["contents"][1]["role"], "model");
        assert_eq!(body["contents"][1]["parts"][0]["text"], "hello");
    }

    #[test]
    fn gemini_response_parses_text_and_usage() {
        let json = serde_json::json!({
            "candidates": [{ "content": { "role": "model", "parts": [
                { "text": "the answer " }, { "text": "is 42" }
            ] } }],
            "usageMetadata": { "promptTokenCount": 11, "candidatesTokenCount": 4 }
        });
        let (text, in_tok, out_tok) = gemini_parse_response(&json).unwrap();
        assert_eq!(text, "the answer is 42");
        assert_eq!(in_tok, 11);
        assert_eq!(out_tok, 4);
        // A response with no candidates is a decode error, not a fabricated empty answer.
        assert!(gemini_parse_response(&serde_json::json!({ "candidates": [] })).is_err());
    }

    #[test]
    fn from_config_wires_the_gemini_dialect() {
        let defs = vec![firstpass_core::ProviderDef {
            id: "gemini".to_owned(),
            dialect: firstpass_core::Dialect::Gemini,
            base_url: "https://generativelanguage.googleapis.com".to_owned(),
            api_key_env: Some("GEMINI_API_KEY".to_owned()),
            auth: firstpass_core::AuthScheme::ApiKey,
            region: None,
            project: None,
        }];
        let reg = ProviderRegistry::from_config(
            &defs,
            "https://api.anthropic.com",
            "https://api.openai.com",
        );
        assert_eq!(reg.get("gemini").unwrap().id(), "gemini");
    }

    #[test]
    fn from_config_registers_custom_providers_alongside_builtins() {
        let defs = vec![
            firstpass_core::ProviderDef {
                id: "groq".to_owned(),
                dialect: firstpass_core::Dialect::Openai,
                base_url: "https://api.groq.com/openai".to_owned(),
                api_key_env: Some("GROQ_API_KEY".to_owned()),
                auth: firstpass_core::AuthScheme::ApiKey,
                region: None,
                project: None,
            },
            // A custom provider may override a built-in id (e.g. point `openai` at Azure).
            firstpass_core::ProviderDef {
                id: "openai".to_owned(),
                dialect: firstpass_core::Dialect::Openai,
                base_url: "https://my-azure.openai.azure.com".to_owned(),
                api_key_env: Some("AZURE_OPENAI_KEY".to_owned()),
                auth: firstpass_core::AuthScheme::ApiKey,
                region: None,
                project: None,
            },
        ];
        let reg = ProviderRegistry::from_config(
            &defs,
            "https://api.anthropic.com",
            "https://api.openai.com",
        );
        // Built-in anthropic is still present; the custom groq resolves and labels itself "groq".
        assert_eq!(reg.get("anthropic").unwrap().id(), "anthropic");
        assert_eq!(reg.get("groq").unwrap().id(), "groq");
        // Unknown provider → None (router fails over rather than guessing).
        assert!(reg.get("does-not-exist").is_none());
    }

    #[test]
    fn anthropic_messages_body_omits_model_and_includes_system_only_when_set() {
        let req = ModelRequest {
            model: "bedrock/anthropic.claude-3-5-haiku".to_owned(),
            system: Some("be terse".to_owned()),
            messages: vec![
                ChatMessage::text("user", "hi"),
                ChatMessage {
                    role: "assistant".to_owned(),
                    content: serde_json::json!([{ "type": "text", "text": "hello" }]),
                },
            ],
            max_tokens: 128,
            tools: Value::Null,
            raw: Value::Null,
        };
        let body = anthropic_messages_body(&req, "bedrock-2023-05-31");
        // Model goes in the URL for Bedrock/Vertex, never the body.
        assert!(body.get("model").is_none());
        assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["messages"][0]["content"], serde_json::json!("hi"));
        // Content-block arrays forward verbatim, same as the direct Anthropic adapter (ADR 0005).
        assert_eq!(
            body["messages"][1]["content"],
            serde_json::json!([{ "type": "text", "text": "hello" }])
        );

        // No system prompt => no `system` key at all (not `null`).
        let req_no_system = ModelRequest {
            system: None,
            ..req
        };
        let body2 = anthropic_messages_body(&req_no_system, "vertex-2023-10-16");
        assert!(body2.get("system").is_none());
    }

    #[test]
    fn bedrock_url_construction_and_missing_region() {
        assert_eq!(
            bedrock_url("us-east-1", "anthropic.claude-3-5-haiku-20241022-v1:0"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-haiku-20241022-v1:0/invoke"
        );
    }

    #[tokio::test]
    async fn bedrock_complete_errors_without_a_region() {
        let provider = BedrockProvider {
            id: "bedrock".to_owned(),
            region: None,
            http: reqwest::Client::new(),
        };
        let req = ModelRequest {
            model: "bedrock/anthropic.claude-3-5-haiku".to_owned(),
            system: None,
            messages: vec![],
            max_tokens: 16,
            tools: Value::Null,
            raw: Value::Null,
        };
        let err = provider.complete(&req, &Auth::default()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Transport(_)));
    }

    #[test]
    fn bedrock_signing_produces_a_sigv4_authorization_header() {
        // Dummy, non-production credentials — this only proves the *shape* of the signed request
        // (algorithm header + host), not cryptographic validity against real AWS (LIVE-UNVERIFIED).
        let creds = AwsEnvCredentials {
            access_key_id: "AKIDEXAMPLE".to_owned(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_owned(),
            session_token: None,
        };
        let url = bedrock_url("us-east-1", "anthropic.claude-3-5-haiku");
        let body = br#"{"anthropic_version":"bedrock-2023-05-31"}"#;
        let signed = sign_bedrock(&url, "us-east-1", body, &creds).unwrap();

        let auth_header = signed
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .expect("authorization header present");
        assert!(auth_header.starts_with("AWS4-HMAC-SHA256"));

        let host_header = signed
            .headers()
            .get("host")
            .and_then(|v| v.to_str().ok())
            .expect("host header present");
        assert_eq!(host_header, "bedrock-runtime.us-east-1.amazonaws.com");
    }

    #[test]
    fn vertex_url_construction_and_missing_project() {
        assert_eq!(
            vertex_url("us-central1", "my-project", "claude-3-5-sonnet"),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet:rawPredict"
        );
    }

    #[tokio::test]
    async fn vertex_complete_errors_without_a_project() {
        let provider = VertexProvider {
            id: "vertex".to_owned(),
            region: Some("us-central1".to_owned()),
            project: None,
            http: reqwest::Client::new(),
        };
        let req = ModelRequest {
            model: "vertex/claude-3-5-sonnet".to_owned(),
            system: None,
            messages: vec![],
            max_tokens: 16,
            tools: Value::Null,
            raw: Value::Null,
        };
        let err = provider.complete(&req, &Auth::default()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Transport(_)));
    }

    #[test]
    fn from_config_wires_bedrock_and_vertex_auth_schemes() {
        let defs = vec![
            firstpass_core::ProviderDef {
                id: "bedrock".to_owned(),
                dialect: firstpass_core::Dialect::Anthropic,
                base_url: String::new(),
                api_key_env: None,
                auth: firstpass_core::AuthScheme::AwsSigv4,
                region: Some("us-east-1".to_owned()),
                project: None,
            },
            firstpass_core::ProviderDef {
                id: "vertex".to_owned(),
                dialect: firstpass_core::Dialect::Anthropic,
                base_url: String::new(),
                api_key_env: None,
                auth: firstpass_core::AuthScheme::GcpOauth,
                region: Some("us-central1".to_owned()),
                project: Some("my-project".to_owned()),
            },
        ];
        let reg = ProviderRegistry::from_config(
            &defs,
            "https://api.anthropic.com",
            "https://api.openai.com",
        );
        assert_eq!(reg.get("bedrock").unwrap().id(), "bedrock");
        assert_eq!(reg.get("vertex").unwrap().id(), "vertex");
    }

    #[test]
    fn resolve_api_key_prefers_configured_env_then_byok() {
        // Use PATH (always present) to exercise the env branch without mutating process env.
        let path = std::env::var("PATH").expect("PATH is set");
        // Configured env wins over any BYOK override (the env var is *this* provider's key).
        assert_eq!(resolve_api_key(Some("PATH"), Some("byok")), path);
        // An unset configured env → fall back to the per-request BYOK override.
        assert_eq!(
            resolve_api_key(Some("FIRSTPASS_DEFINITELY_UNSET_KEY"), Some("byok")),
            "byok"
        );
        // No configured env → fall back to BYOK.
        assert_eq!(resolve_api_key(None, Some("byok")), "byok");
        // Neither → empty (keyless local endpoint).
        assert_eq!(resolve_api_key(None, None), "");
    }

    #[test]
    fn anthropic_wire_forwards_tool_and_image_content_verbatim() {
        // ADR 0005 I3 (request side): the Anthropic adapter serializes tool_use / tool_result /
        // image content blocks byte-for-byte into the wire body — enforce forwards them, it does not
        // flatten them. A plain-string message still serializes as a bare string (I1).
        let messages = vec![
            ChatMessage::text("user", "hi"),
            ChatMessage {
                role: "assistant".to_owned(),
                content: serde_json::json!([
                    { "type": "tool_use", "id": "t1", "name": "calc", "input": { "x": 1 } }
                ]),
            },
            ChatMessage {
                role: "user".to_owned(),
                content: serde_json::json!([
                    { "type": "tool_result", "tool_use_id": "t1", "content": "2" },
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AA==" } }
                ]),
            },
        ];
        let wire = anthropic_wire_body(&ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: None,
            messages,
            max_tokens: 64,
            tools: Value::Null,
            raw: Value::Null,
        });
        assert_eq!(wire["messages"][0]["content"], serde_json::json!("hi"));
        assert_eq!(
            wire["messages"][1]["content"],
            serde_json::json!([{ "type": "tool_use", "id": "t1", "name": "calc", "input": { "x": 1 } }])
        );
        assert_eq!(
            wire["messages"][2]["content"],
            serde_json::json!([
                { "type": "tool_result", "tool_use_id": "t1", "content": "2" },
                { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "AA==" } }
            ])
        );
    }

    fn resp(model: &str, text: &str) -> ModelResponse {
        ModelResponse {
            model: model.to_owned(),
            text: text.to_owned(),
            in_tokens: 10,
            out_tokens: 5,
            raw: Value::Null,
        }
    }

    #[test]
    fn transport_and_5xx_are_failover_eligible() {
        assert!(ProviderError::Transport("boom".into()).is_failover_eligible());
        assert!(
            ProviderError::Http {
                status: 503,
                body: String::new()
            }
            .is_failover_eligible()
        );
    }

    #[test]
    fn client_errors_and_decode_failures_are_hard() {
        assert!(
            !ProviderError::Http {
                status: 400,
                body: String::new()
            }
            .is_failover_eligible()
        );
        assert!(!ProviderError::Decode("bad json".into()).is_failover_eligible());
    }

    #[test]
    fn auth_debug_never_prints_key_material() {
        let auth = Auth {
            anthropic_key: Some("sk-ant-super-secret".to_owned()),
            openai_key: Some("sk-oai-super-secret".to_owned()),
        };
        let debug = format!("{auth:?}");
        assert!(!debug.contains("super-secret"));
    }

    #[tokio::test]
    async fn mock_provider_returns_configured_outcome() {
        let mut outcomes = HashMap::new();
        outcomes.insert(
            "anthropic/claude-haiku-4-5".to_owned(),
            Ok(resp("anthropic/claude-haiku-4-5", "hello")),
        );
        let provider = MockProvider::new("anthropic", outcomes);
        let req = ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: None,
            messages: vec![],
            max_tokens: 100,
            tools: Value::Null,
            raw: Value::Null,
        };
        let out = provider.complete(&req, &Auth::default()).await.unwrap();
        assert_eq!(out.text, "hello");
    }

    #[test]
    fn anthropic_wire_body_carries_raw_request_verbatim() {
        // ADR 0005: with a raw inbound body present, the wire request IS that body — every field
        // the caller set (tools, tool_choice, temperature, thinking, ...) survives; only `model`
        // is swapped to the rung's wire id and `stream` is stripped (complete() buffers to gate).
        let raw = serde_json::json!({
            "model": "claude-opus-4-8",
            "stream": true,
            "max_tokens": 512,
            "temperature": 0.2,
            "tool_choice": { "type": "auto" },
            "tools": [{ "name": "get_weather", "input_schema": { "type": "object" } }],
            "thinking": { "type": "enabled", "budget_tokens": 1024 },
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: None,
            messages: vec![ChatMessage::text("user", "hi")],
            max_tokens: 512,
            tools: raw["tools"].clone(),
            raw: raw.clone(),
        };
        let wire = anthropic_wire_body(&req);
        assert_eq!(wire["model"], "claude-haiku-4-5", "rung model swapped in");
        assert!(wire.get("stream").is_none(), "stream stripped");
        for field in [
            "temperature",
            "tool_choice",
            "tools",
            "thinking",
            "max_tokens",
            "messages",
        ] {
            assert_eq!(
                wire[field], raw[field],
                "field {field} must survive verbatim"
            );
        }
    }

    #[test]
    fn bedrock_vertex_body_carries_raw_minus_model_plus_version() {
        let raw = serde_json::json!({
            "model": "claude-haiku-4-5",
            "stream": true,
            "max_tokens": 64,
            "tools": [{ "name": "t" }],
            "messages": [{ "role": "user", "content": "hi" }]
        });
        let req = ModelRequest {
            model: "bedrock/anthropic.claude-haiku".to_owned(),
            system: None,
            messages: vec![ChatMessage::text("user", "hi")],
            max_tokens: 64,
            tools: raw["tools"].clone(),
            raw: raw.clone(),
        };
        let body = anthropic_messages_body(&req, "bedrock-2023-05-31");
        assert!(body.get("model").is_none(), "model lives in the URL");
        assert!(body.get("stream").is_none());
        assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
        assert_eq!(body["tools"], raw["tools"]);
        assert_eq!(body["messages"], raw["messages"]);
    }

    #[test]
    fn verbatim_carriers_are_anthropic_shaped_dialects_only() {
        let reg = ProviderRegistry::new("http://localhost", "http://localhost");
        assert!(reg.get("anthropic").unwrap().carries_structured_verbatim());
        assert!(!reg.get("openai").unwrap().carries_structured_verbatim());
    }
}
