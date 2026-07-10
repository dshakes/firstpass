//! Normalized model access: a provider-agnostic request/response shape, and the two wire
//! adapters (Anthropic Messages, OpenAI Chat Completions) that speak it. The router
//! ([`crate::router`]) only ever talks to [`Provider`]; it never knows which wire format is
//! behind a given rung.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One message in a normalized chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `"user"`, `"assistant"`, or `"system"`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// A provider-agnostic model request, built once per incoming call and re-used (with
/// `model` swapped) across every rung of the ladder.
///
// ponytail: multimodal content and tool-result blocks are collapsed to plain text for now —
// enough to route and gate on. A richer content-block model is a later batch.
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
}

#[derive(Serialize)]
struct AnthropicWireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Strip the `provider/` prefix from a ladder model id for the provider's wire API — Anthropic and
/// OpenAI expect the bare model (`claude-haiku-4-5`), not `anthropic/claude-haiku-4-5`. The full
/// prefixed id is still what the ladder/trace use; only the wire call is stripped.
fn wire_model(model: &str) -> &str {
    model.split_once('/').map_or(model, |(_, m)| m)
}

#[derive(Serialize)]
struct AnthropicWireRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    max_tokens: u32,
    messages: Vec<AnthropicWireMessage<'a>>,
}

/// Speaks `POST {base}/v1/messages` (Anthropic Messages API).
///
// LIVE-VERIFIED (2026-07-10): exercised against real Anthropic through the running proxy's enforce
// path — a haiku completion served end-to-end. The `anthropic/` prefix must be stripped for the
// wire call (see `wire_model`); sending it verbatim 404s.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    /// Base URL, e.g. `https://api.anthropic.com`.
    pub base_url: String,
    /// Shared, connection-pooled HTTP client.
    pub http: reqwest::Client,
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn id(&self) -> &str {
        "anthropic"
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        let key = auth.anthropic_key.as_deref().unwrap_or_default();
        let body = AnthropicWireRequest {
            model: wire_model(&req.model),
            system: req.system.as_deref(),
            max_tokens: req.max_tokens,
            messages: req
                .messages
                .iter()
                .map(|m| AnthropicWireMessage {
                    role: &m.role,
                    content: &m.content,
                })
                .collect(),
        };

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

        Ok(ModelResponse {
            model: req.model.clone(),
            text,
            in_tokens,
            out_tokens,
            raw: json,
        })
    }
}

#[derive(Serialize)]
struct OpenAiWireMessage<'a> {
    role: &'a str,
    content: &'a str,
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
    /// Base URL, e.g. `https://api.openai.com`.
    pub base_url: String,
    /// Shared, connection-pooled HTTP client.
    pub http: reqwest::Client,
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn id(&self) -> &str {
        "openai"
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
        let key = auth.openai_key.as_deref().unwrap_or_default();
        let mut messages = Vec::with_capacity(req.messages.len() + 1);
        if let Some(system) = req.system.as_deref() {
            messages.push(OpenAiWireMessage {
                role: "system",
                content: system,
            });
        }
        messages.extend(req.messages.iter().map(|m| OpenAiWireMessage {
            role: &m.role,
            content: &m.content,
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
    /// Build the standard registry: Anthropic + OpenAI, sharing one HTTP client.
    #[must_use]
    pub fn new(anthropic_base: impl Into<String>, openai_base: impl Into<String>) -> Self {
        let http = reqwest::Client::new();
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(
            "anthropic".to_owned(),
            Arc::new(AnthropicProvider {
                base_url: anthropic_base.into(),
                http: http.clone(),
            }),
        );
        providers.insert(
            "openai".to_owned(),
            Arc::new(OpenAiProvider {
                base_url: openai_base.into(),
                http,
            }),
        );
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
        }
    }
}

#[cfg(test)]
#[async_trait]
impl Provider for MockProvider {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(
        &self,
        req: &ModelRequest,
        _auth: &Auth,
    ) -> Result<ModelResponse, ProviderError> {
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
        };
        let out = provider.complete(&req, &Auth::default()).await.unwrap();
        assert_eq!(out.text, "hello");
    }
}
