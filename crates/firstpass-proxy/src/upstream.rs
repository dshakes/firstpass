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

/// Build the upstream `POST /v1/messages` request, forwarding the allow-listed headers and the
/// caller's body byte-for-byte. Shared by the buffered and streaming paths.
fn build_request(
    client: &reqwest::Client,
    base: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> reqwest::RequestBuilder {
    let url = format!("{}/v1/messages", base.trim_end_matches('/'));
    let mut req = client.post(url);
    for name in FORWARDED_REQUEST_HEADERS {
        if let Some(value) = headers.get(*name) {
            req = req.header(*name, value.clone());
        }
    }
    req.body(body)
}

/// Copy the response's content-type (the only response header observe mode relays).
fn passthrough_headers(response: &reqwest::Response) -> HeaderMap {
    // reqwest and axum both build on the same `http` crate, so header types carry over
    // without re-encoding.
    let mut out = HeaderMap::new();
    if let Some(content_type) = response.headers().get(header::CONTENT_TYPE) {
        out.insert(header::CONTENT_TYPE, content_type.clone());
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
    let response = build_request(client, base, headers, body).send().await?;
    let status = response.status();
    let out_headers = passthrough_headers(&response);
    let body = response.bytes().await?;
    Ok((status, out_headers, body))
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
    let response = build_request(client, base, headers, body).send().await?;
    let status = response.status();
    let out_headers = passthrough_headers(&response);
    Ok((status, out_headers, response))
}
