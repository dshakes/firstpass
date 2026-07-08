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

// ponytail: buffered (non-streaming) passthrough only. SSE streaming passthrough — proxying
// `stream: true` messages chunk-by-chunk instead of buffering the full body — is a follow-up
// once observe mode is proven; it needs its own latency test, not a speculative build now.

/// Forward one `POST /v1/messages` request to the upstream provider and return its response
/// unchanged.
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
    let url = format!("{}/v1/messages", base.trim_end_matches('/'));
    let mut req = client.post(url);

    for name in FORWARDED_REQUEST_HEADERS {
        if let Some(value) = headers.get(*name) {
            req = req.header(*name, value.clone());
        }
    }

    let response = req.body(body).send().await?;
    let status = response.status();

    // reqwest and axum both build on the same `http` crate, so header types carry over
    // without re-encoding.
    let mut out_headers = HeaderMap::new();
    if let Some(content_type) = response.headers().get(header::CONTENT_TYPE) {
        out_headers.insert(header::CONTENT_TYPE, content_type.clone());
    }

    let body = response.bytes().await?;
    Ok((status, out_headers, body))
}
