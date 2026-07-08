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
}

impl ProxyError {
    fn kind(&self) -> &'static str {
        match self {
            ProxyError::Upstream(_) => "upstream_error",
            ProxyError::BadRequestBody | ProxyError::BadRequest(_) => "bad_request",
            ProxyError::Engine(_) => "engine_error",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            ProxyError::Upstream(_) | ProxyError::Engine(_) => StatusCode::BAD_GATEWAY,
            ProxyError::BadRequestBody | ProxyError::BadRequest(_) => StatusCode::BAD_REQUEST,
        }
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
        let message = self.to_string();
        let body = ErrorBody {
            error: ErrorDetail {
                kind: self.kind(),
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
}
