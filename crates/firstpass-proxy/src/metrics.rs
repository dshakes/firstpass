//! Prometheus metrics: install the global recorder once per process and serve `GET /metrics`
//! for scraping. Real signals only — latency, escalations, what got served, what dropped.

use std::sync::OnceLock;

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use crate::error::ProxyError;

/// Caches the install outcome so every caller (the real server, every test that builds an
/// [`crate::proxy::app`]) observes the same result without racing to install the recorder twice.
static HANDLE: OnceLock<Result<PrometheusHandle, String>> = OnceLock::new();

/// Install the global Prometheus recorder, once per process, and return a handle that renders
/// the scrape payload. Safe to call more than once: `OnceLock::get_or_init` runs the install
/// exactly once even under concurrent callers, so repeat calls (e.g. from every test that builds
/// an [`crate::proxy::app`]) just reuse the cached handle instead of re-installing.
///
/// # Errors
/// Returns [`ProxyError::Internal`] if the first install fails (e.g. a different metrics
/// recorder is already installed in-process).
pub fn install() -> Result<PrometheusHandle, ProxyError> {
    HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(ProxyError::Internal)
}

/// `GET /metrics` — render the current Prometheus scrape payload.
pub async fn handler() -> Response {
    match install() {
        Ok(handle) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
            handle.render(),
        )
            .into_response(),
        Err(err) => err.into_response(),
    }
}
