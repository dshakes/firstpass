//! Structured, no-leak errors (SPEC §7.4: "errors are structured, never prose an agent
//! must parse").
//!
//! Every error the proxy returns to a caller is `{"error": {"type": "...", "message": "..."}}`.
//! Nothing here ever includes a stack trace, an API key, or raw prompt text.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

/// Errors that can surface on the request path.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// The upstream provider could not be reached, or returned something unusable
    /// (connection failure, timeout, decode error).
    #[error("upstream request failed: {0}")]
    Upstream(#[from] reqwest::Error),

    /// The inbound request body could not be read (e.g. the connection dropped mid-body).
    #[error("failed to read request body")]
    BadRequestBody,

    /// The request body was malformed for the target wire API (enforce mode parses it).
    #[error("{0}")]
    BadRequest(String),

    /// The escalation engine could not serve any output (every rung errored).
    #[error("{0}")]
    Engine(String),

    /// A referenced resource does not exist (e.g. feedback for an unknown trace id).
    #[error("{0}")]
    NotFound(String),

    /// Authentication failed: a missing or invalid tenant API key when `require_auth` is on
    /// (ADR 0004 §D1). The body is intentionally opaque — no "unknown tenant" oracle.
    #[error("unauthorized")]
    Unauthorized,

    /// An internal failure (e.g. the trace store errored). Never leaks internals to the caller.
    #[error("internal error")]
    Internal(String),

    /// The tenant exceeded its configured request rate (ADR 0004 §D6). The body is intentionally
    /// opaque — no bucket state or limit value is disclosed.
    #[error("rate limited")]
    RateLimited,
}

impl ProxyError {
    fn kind(&self) -> &'static str {
        match self {
            ProxyError::Upstream(_) => "upstream_error",
            ProxyError::BadRequestBody | ProxyError::BadRequest(_) => "bad_request",
            ProxyError::Engine(_) => "engine_error",
            ProxyError::NotFound(_) => "not_found",
            ProxyError::Unauthorized => "unauthorized",
            ProxyError::Internal(_) => "internal_error",
            ProxyError::RateLimited => "rate_limited",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            ProxyError::Upstream(_) | ProxyError::Engine(_) => StatusCode::BAD_GATEWAY,
            ProxyError::BadRequestBody | ProxyError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ProxyError::NotFound(_) => StatusCode::NOT_FOUND,
            ProxyError::Unauthorized => StatusCode::UNAUTHORIZED,
            ProxyError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::RateLimited => StatusCode::TOO_MANY_REQUESTS,
        }
    }

    /// The message returned to the caller. Client-actionable errors (validation, not-found) return
    /// their real message; upstream / engine / internal detail is generalized so the response never
    /// leaks the upstream's identity, an internal error string, or server state (the full detail is
    /// logged server-side instead — see [`IntoResponse`]).
    pub(crate) fn client_message(&self) -> String {
        match self {
            ProxyError::BadRequest(m) | ProxyError::NotFound(m) => m.clone(),
            ProxyError::BadRequestBody => "failed to read request body".to_owned(),
            ProxyError::Upstream(_) => "upstream request failed".to_owned(),
            ProxyError::Engine(_) => "no rung could serve a valid response".to_owned(),
            ProxyError::Unauthorized => "unauthorized".to_owned(),
            ProxyError::Internal(_) => "internal error".to_owned(),
            ProxyError::RateLimited => "rate limited".to_owned(),
        }
    }

    /// Whether the full detail must be kept server-side (logged, not returned).
    fn detail_is_internal(&self) -> bool {
        matches!(
            self,
            ProxyError::Upstream(_) | ProxyError::Engine(_) | ProxyError::Internal(_)
        )
    }
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    message: &'a str,
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status();
        let kind = self.kind();
        // Log the full detail server-side for the variants whose detail must not reach the caller,
        // so operators keep the diagnostic while clients get an opaque message.
        if self.detail_is_internal() {
            tracing::warn!(kind, detail = %self, "request error");
        }
        let message = self.client_message();
        let body = ErrorBody {
            error: ErrorDetail {
                kind,
                message: &message,
            },
        };
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bad_request_body_maps_to_400_with_structured_json() {
        let response = ProxyError::BadRequestBody.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "bad_request");
    }

    #[tokio::test]
    async fn engine_error_does_not_leak_internal_detail_to_caller() {
        // The engine detail names a specific upstream host — it must not reach the client body.
        let leaky = "connection refused to https://internal-upstream.acme:8443/v1/messages";
        let response = ProxyError::Engine(leaky.to_owned()).into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "engine_error");
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(
            !msg.contains("internal-upstream"),
            "leaked upstream host: {msg}"
        );
        assert!(!msg.contains("acme"), "leaked internal detail: {msg}");
    }

    #[tokio::test]
    async fn bad_request_keeps_its_client_actionable_message() {
        // Validation messages ARE client-actionable and should be preserved.
        let response =
            ProxyError::BadRequest("field `model` is required".to_owned()).into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"].as_str().unwrap().contains("model"));
    }
}
