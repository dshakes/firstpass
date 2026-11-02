//! Forwarding to the upstream Anthropic-compatible provider.
//!
//! Observe mode is a transparent proxy: the request body and BYOK auth headers go upstream
//! byte-for-byte, and the response comes back byte-for-byte. Firstpass never inspects,
//! injects, or logs the API key.

use axum::http::{HeaderMap, StatusCode, header};
use bytes::Bytes;

use crate::error::ProxyError;

/// Request headers forwarded verbatim to the upstream provider. Anything not in this list
/// (in particular hop-by-hop headers) is dropped rather than relayed.
const FORWARDED_REQUEST_HEADERS: &[&str] = &[
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "content-type",
];

/// Build an upstream POST request to `{base}{path}`, forwarding the allow-listed headers and the
/// caller's body byte-for-byte. Shared by the buffered and streaming paths, and by both the
/// Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`) observe-passthrough paths.
fn build_request(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> reqwest::RequestBuilder {
    let url = format!("{}{path}", base.trim_end_matches('/'));
    let mut req = client.post(url);
    for name in FORWARDED_REQUEST_HEADERS {
        if let Some(value) = headers.get(*name) {
            req = req.header(*name, value.clone());
        }
    }
    req.body(body)
}

/// Response headers observe mode relays verbatim. Beyond content-type, agents need the upstream's
/// back-off and correlation signals: Claude Code, Codex and the SDKs honour `retry-after` and the
/// rate-limit headers on a 429/529, and a `request-id` is what a provider's support asks for.
/// Framing headers (content-length, transfer-encoding, content-encoding) are deliberately not
/// relayed: the body is re-framed by this server.
const RELAYED_RESPONSE_HEADERS: &[&str] = &["retry-after", "request-id", "x-request-id"];
const RELAYED_RESPONSE_HEADER_PREFIXES: &[&str] = &["anthropic-ratelimit-", "x-ratelimit-"];

fn relayed(name: &str) -> bool {
    RELAYED_RESPONSE_HEADERS.contains(&name)
        || RELAYED_RESPONSE_HEADER_PREFIXES
            .iter()
            .any(|p| name.starts_with(p))
}

/// Copy the response headers observe mode relays (see [`RELAYED_RESPONSE_HEADERS`]).
fn passthrough_headers(response: &reqwest::Response) -> HeaderMap {
    // reqwest and axum both build on the same `http` crate, so header types carry over
    // without re-encoding.
    let mut out = HeaderMap::new();
    if let Some(content_type) = response.headers().get(header::CONTENT_TYPE) {
        out.insert(header::CONTENT_TYPE, content_type.clone());
    }
    for (name, value) in response.headers() {
        if relayed(name.as_str()) {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

/// Forward one `POST /v1/messages` request and return its response **buffered** (full body).
/// Used for non-streaming requests, where the assembled body also feeds the audit trace.
///
/// # Errors
/// Returns [`ProxyError::Upstream`] if the upstream request fails at the transport level
/// (DNS, connect, timeout). A non-2xx HTTP response from upstream is *not* an error here —
/// it is returned to the caller as-is, matching observe mode's "forward unchanged" contract.
pub async fn forward_anthropic(
    client: &reqwest::Client,
    base: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, HeaderMap, Bytes), ProxyError> {
    forward_anthropic_path(client, base, "/v1/messages", headers, body).await
}

/// Forward one `POST /v1/messages` request and return the raw response for **streaming**
/// relay (`stream: true`) — the caller pipes `response.bytes_stream()` to the client without
/// buffering, preserving SSE chunk-by-chunk and keeping time-to-first-byte low.
///
/// # Errors
/// Returns [`ProxyError::Upstream`] on a transport-level failure (see [`forward_anthropic`]).
pub async fn forward_anthropic_streaming(
    client: &reqwest::Client,
    base: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, HeaderMap, reqwest::Response), ProxyError> {
    let response = build_request(client, base, "/v1/messages", headers, body)
        .send()
        .await?;
    let status = response.status();
    let out_headers = passthrough_headers(&response);
    Ok((status, out_headers, response))
}

/// Forward one Anthropic request to an explicit upstream path and return its response **buffered**.
///
/// Factored out so sub-paths like `/v1/messages/count_tokens` can be forwarded byte-for-byte
/// without duplicating `build_request` or the header allow-list.
///
/// # Errors
/// Returns [`ProxyError::Upstream`] on a transport-level failure.
pub async fn forward_anthropic_path(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, HeaderMap, Bytes), ProxyError> {
    let response = build_request(client, base, path, headers, body)
        .send()
        .await?;
    let status = response.status();
    let out_headers = passthrough_headers(&response);
    let body = response.bytes().await?;
    Ok((status, out_headers, body))
}

/// The OpenAI-compatible inbound paths this proxy relays for.
pub const OPENAI_CHAT_PATH: &str = "/v1/chat/completions";
/// The Responses API path. A Responses-shaped body relayed to [`OPENAI_CHAT_PATH`] is rejected by
/// the upstream, so the passthrough has to preserve which endpoint the client actually called.
pub const OPENAI_RESPONSES_PATH: &str = "/v1/responses";

/// Forward one OpenAI-compatible request to an explicit upstream path and return its response
/// **buffered**. Observe-mode passthrough — a byte-identical relay.
///
/// The path is explicit because a Responses-shaped body relayed to [`OPENAI_CHAT_PATH`] is
/// rejected upstream; the caller knows which endpoint the client actually called.
///
/// # Errors
/// Returns [`ProxyError::Upstream`] on a transport-level failure.
pub async fn forward_openai_path(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, HeaderMap, Bytes), ProxyError> {
    let response = build_request(client, base, path, headers, body)
        .send()
        .await?;
    let status = response.status();
    let out_headers = passthrough_headers(&response);
    let body = response.bytes().await?;
    Ok((status, out_headers, body))
}

/// Forward one OpenAI-compatible request to an explicit upstream path and return the raw response
/// for streaming relay.
///
/// # Errors
/// Returns [`ProxyError::Upstream`] on a transport-level failure.
pub async fn forward_openai_streaming_path(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, HeaderMap, reqwest::Response), ProxyError> {
    let response = build_request(client, base, path, headers, body)
        .send()
        .await?;
    let status = response.status();
    let out_headers = passthrough_headers(&response);
    Ok((status, out_headers, response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_relays_backoff_and_correlation_headers_only() {
        let upstream = axum::http::Response::builder()
            .status(429)
            .header("content-type", "application/json")
            .header("retry-after", "7")
            .header("request-id", "req_123")
            .header("anthropic-ratelimit-requests-remaining", "0")
            .header("x-ratelimit-reset-tokens", "6s")
            .header("content-length", "2")
            .header("set-cookie", "a=b")
            .body("{}")
            .unwrap_or_default();
        let out = passthrough_headers(&reqwest::Response::from(upstream));
        for (k, v) in [
            ("content-type", "application/json"),
            ("retry-after", "7"),
            ("request-id", "req_123"),
            ("anthropic-ratelimit-requests-remaining", "0"),
            ("x-ratelimit-reset-tokens", "6s"),
        ] {
            assert_eq!(out.get(k).and_then(|h| h.to_str().ok()), Some(v), "{k}");
        }
        assert!(
            out.get("content-length").is_none(),
            "framing is not relayed"
        );
        assert!(
            out.get("set-cookie").is_none(),
            "unlisted headers are not relayed"
        );
    }
}
