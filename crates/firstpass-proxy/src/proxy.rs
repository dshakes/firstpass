//! Axum wiring: routes, request/response shapes, and observe-mode trace construction
//! (SPEC §7.1, §7.1a — forward unchanged, record asynchronously, zero added latency).

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use firstpass_core::features::{hour_bucket, token_bucket};
use firstpass_core::hashchain::sha256_hex;
use firstpass_core::{
    Attempt, FEATURE_VERSION, Features, FinalOutcome, GENESIS_HASH, Mode, PolicyRef, RequestInfo,
    ServedFrom, TaskKind, Trace, Verdict,
};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

use crate::config::ProxyConfig;
use crate::upstream::forward_anthropic;

/// Shared state handed to every request handler. Cheap to clone: an `Arc`ed config, a
/// pooled HTTP client, and an unbounded channel sender.
#[derive(Clone)]
pub struct AppState {
    /// Static proxy configuration.
    pub config: Arc<ProxyConfig>,
    /// Shared, connection-pooled HTTP client used to call upstream.
    pub http: reqwest::Client,
    /// Fire-and-forget sender to the background trace writer.
    pub traces: UnboundedSender<Trace>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Build the axum router: `POST /v1/messages`, `GET /v1/capabilities`, `GET /healthz`.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/capabilities", get(capabilities))
        .route("/healthz", get(healthz))
        .with_state(state)
}

/// `GET /healthz` — liveness probe.
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// `GET /v1/capabilities` — agent-first discovery (SPEC §0.2, §7.4): what this proxy speaks,
/// what mode it's in, and how to turn it off.
async fn capabilities() -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "firstpass",
        "version": env!("CARGO_PKG_VERSION"),
        "feature_version": FEATURE_VERSION,
        "modes": ["observe"],
        "wire_apis": ["anthropic.messages"],
        // ponytail: no firstpass.toml route table is loaded yet (that's the M2 gating
        // config), so the ladder is always empty in this observe-only skeleton.
        "ladder": [],
        "gates": [],
        "offboarding": "unset ANTHROPIC_BASE_URL",
    }))
}

/// The header a caller may set to group requests into a session for the audit trail. When
/// absent, each request is its own session (keyed by its own trace id).
const SESSION_HEADER: &str = "x-firstpass-session";

/// `POST /v1/messages` — forward unchanged to the upstream provider, return unchanged, and
/// record a trace asynchronously. The trace write never delays the response.
async fn messages(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let start = Instant::now();
    let session_header = headers
        .get(SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let result = forward_anthropic(
        &state.http,
        &state.config.upstream_anthropic,
        &headers,
        body.clone(),
    )
    .await;
    let latency_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    match result {
        Ok((status, resp_headers, resp_body)) => {
            // Build + record the trace on a detached task so neither JSON parsing nor the
            // channel send touches the response path: observe mode adds zero latency to what
            // the caller sees (SPEC §7.1a). `Bytes` clones are cheap (refcounted).
            spawn_trace(
                &state,
                body,
                Some(resp_body.clone()),
                latency_ms,
                session_header,
            );
            (status, resp_headers, resp_body).into_response()
        }
        Err(err) => {
            spawn_trace(&state, body, None, latency_ms, session_header);
            err.into_response()
        }
    }
}

/// Construct the trace and enqueue it for the background writer, entirely off the response
/// path. Fire-and-forget: if the writer has shut down we log rather than propagate — recording
/// must never affect what the caller sees. `resp_body` is `Some` for a forwarded response and
/// `None` when the upstream call failed outright.
fn spawn_trace(
    state: &AppState,
    req_body: Bytes,
    resp_body: Option<Bytes>,
    latency_ms: u64,
    session_header: Option<String>,
) {
    let config = state.config.clone();
    let traces = state.traces.clone();
    tokio::spawn(async move {
        let trace = match resp_body {
            Some(resp) => build_trace(
                &config,
                &req_body,
                &resp,
                latency_ms,
                session_header.as_deref(),
            ),
            None => build_error_trace(&config, &req_body, latency_ms, session_header.as_deref()),
        };
        if traces.send(trace).is_err() {
            tracing::warn!("trace writer is gone; dropping trace");
        }
    });
}

/// Session id for the trace: the caller-supplied header, or the trace's own id when absent.
fn session_id(session_header: Option<&str>, trace_id: Uuid) -> String {
    session_header
        .map(str::to_owned)
        .unwrap_or_else(|| trace_id.to_string())
}

/// Salted hash of the raw request body — the only trace of the prompt that ever touches
/// storage (SPEC: never log or persist raw prompt text).
fn prompt_hash(salt: &str, body: &[u8]) -> String {
    let mut salted = Vec::with_capacity(salt.len() + body.len());
    salted.extend_from_slice(salt.as_bytes());
    salted.extend_from_slice(body);
    sha256_hex(&salted)
}

/// Best-effort request-side feature extraction: model name, tool count, and whether any
/// message carries image content. Malformed/absent fields fall back to safe defaults rather
/// than failing the request — this is telemetry, not the served response.
fn request_features(body: &[u8]) -> (Option<String>, u32, bool) {
    let Ok(json) = serde_json::from_slice::<Value>(body) else {
        return (None, 0, false);
    };
    let model = json.get("model").and_then(Value::as_str).map(str::to_owned);
    let tool_count = json
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, |tools| u32::try_from(tools.len()).unwrap_or(u32::MAX));
    let has_images = json
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| messages.iter().any(message_has_image));
    (model, tool_count, has_images)
}

/// Whether a single message's content contains an image block (`{"type": "image", ...}`).
fn message_has_image(message: &Value) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks
                .iter()
                .any(|block| block.get("type").and_then(Value::as_str) == Some("image"))
        })
}

/// Response-side usage extraction: model name and token counts, defaulting to `0` when the
/// upstream response doesn't carry them (e.g. an error body).
fn response_usage(body: &[u8]) -> (Option<String>, u64, u64) {
    let Ok(json) = serde_json::from_slice::<Value>(body) else {
        return (None, 0, 0);
    };
    let model = json.get("model").and_then(Value::as_str).map(str::to_owned);
    let in_tokens = json
        .pointer("/usage/input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let out_tokens = json
        .pointer("/usage/output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    (model, in_tokens, out_tokens)
}

/// Build the observe-mode trace for a request that was successfully forwarded and answered.
fn build_trace(
    config: &ProxyConfig,
    req_body: &Bytes,
    resp_body: &Bytes,
    latency_ms: u64,
    session_header: Option<&str>,
) -> Trace {
    let (req_model, tool_count, has_images) = request_features(req_body);
    let (resp_model, in_tokens, out_tokens) = response_usage(resp_body);
    let model = resp_model
        .or(req_model)
        .unwrap_or_else(|| "unknown".to_owned());

    let cost_usd = config
        .prices
        .cost_usd(&format!("anthropic/{model}"), in_tokens, out_tokens)
        .unwrap_or(0.0);

    let attempt = Attempt {
        rung: 0,
        model,
        provider: "anthropic".to_owned(),
        in_tokens,
        out_tokens,
        cost_usd,
        latency_ms,
        gates: Vec::new(),
        verdict: Verdict::Pass,
    };

    let mut trace = base_trace(config, req_body, latency_ms, session_header);
    trace.request.features.prompt_token_bucket = token_bucket(in_tokens);
    trace.request.features.tool_count = tool_count;
    trace.request.features.has_images = has_images;
    trace.attempts.push(attempt);
    trace.final_ = FinalOutcome {
        served_rung: Some(0),
        served_from: ServedFrom::Attempt,
        total_cost_usd: cost_usd,
        gate_cost_usd: 0.0,
        total_latency_ms: latency_ms,
        escalations: 0,
        counterfactual_baseline_usd: cost_usd,
        savings_usd: 0.0,
    };
    trace.recompute_savings();
    trace
}

/// Build the observe-mode trace for a request whose upstream call failed outright (no
/// response to report usage from). Recorded with `served_from: Error` and no attempts —
/// keep the audit trail honest that nothing was served.
fn build_error_trace(
    config: &ProxyConfig,
    req_body: &Bytes,
    latency_ms: u64,
    session_header: Option<&str>,
) -> Trace {
    let (_, tool_count, has_images) = request_features(req_body);
    let mut trace = base_trace(config, req_body, latency_ms, session_header);
    trace.request.features.tool_count = tool_count;
    trace.request.features.has_images = has_images;
    trace.final_ = FinalOutcome {
        served_rung: None,
        served_from: ServedFrom::Error,
        total_cost_usd: 0.0,
        gate_cost_usd: 0.0,
        total_latency_ms: latency_ms,
        escalations: 0,
        counterfactual_baseline_usd: 0.0,
        savings_usd: 0.0,
    };
    trace.recompute_savings();
    trace
}

/// The parts of a trace that don't depend on whether the call succeeded: identity, policy,
/// and the request-side feature vector minus token bucket (which needs response usage).
fn base_trace(
    config: &ProxyConfig,
    req_body: &Bytes,
    latency_ms: u64,
    session_header: Option<&str>,
) -> Trace {
    let trace_id = Uuid::now_v7();
    let mut features = Features::new(TaskKind::Other);
    features.hour_bucket = hour_bucket(jiff::Timestamp::now());

    Trace {
        trace_id,
        prev_hash: GENESIS_HASH.to_owned(),
        tenant_id: config.tenant_id.clone(),
        session_id: session_id(session_header, trace_id),
        ts: jiff::Timestamp::now(),
        mode: Mode::Observe,
        policy: PolicyRef {
            id: "observe-passthrough@v0".to_owned(),
            explore: false,
        },
        request: RequestInfo {
            api: "anthropic.messages".to_owned(),
            prompt_hash: prompt_hash(&config.prompt_salt, req_body),
            features,
        },
        attempts: Vec::new(),
        deferred: Vec::new(),
        final_: FinalOutcome {
            served_rung: None,
            served_from: ServedFrom::Error,
            total_cost_usd: 0.0,
            gate_cost_usd: 0.0,
            total_latency_ms: latency_ms,
            escalations: 0,
            counterfactual_baseline_usd: 0.0,
            savings_usd: 0.0,
        },
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn test_config() -> ProxyConfig {
        ProxyConfig::from_lookup(|_| None).unwrap()
    }

    #[test]
    fn build_trace_maps_request_and_response_fields() {
        let config = test_config();
        let req = Bytes::from_static(
            br#"{"model":"claude-haiku-4-5","tools":[{"name":"a"}],"messages":[]}"#,
        );
        let resp = Bytes::from_static(
            br#"{"model":"claude-haiku-4-5","usage":{"input_tokens":1200,"output_tokens":300}}"#,
        );

        let trace = build_trace(&config, &req, &resp, 42, Some("sess-1"));

        assert_eq!(trace.request.api, "anthropic.messages");
        assert_eq!(trace.session_id, "sess-1");
        assert_eq!(trace.attempts.len(), 1);
        let attempt = &trace.attempts[0];
        assert_eq!(attempt.model, "claude-haiku-4-5");
        assert_eq!(attempt.provider, "anthropic");
        assert_eq!(attempt.in_tokens, 1200);
        assert_eq!(attempt.out_tokens, 300);
        assert!(attempt.cost_usd > 0.0);
        assert_eq!(trace.request.features.tool_count, 1);
        assert!(!trace.request.features.has_images);
        assert_eq!(trace.final_.served_rung, Some(0));
    }

    #[test]
    fn build_trace_falls_back_to_trace_id_session_when_header_absent() {
        let config = test_config();
        let req = Bytes::from_static(b"{}");
        let resp = Bytes::from_static(b"{}");

        let trace = build_trace(&config, &req, &resp, 1, None);

        assert_eq!(trace.session_id, trace.trace_id.to_string());
    }

    #[test]
    fn build_error_trace_has_no_attempts_and_served_from_error() {
        let config = test_config();
        let req = Bytes::from_static(br#"{"model":"claude-haiku-4-5"}"#);

        let trace = build_error_trace(&config, &req, 7, None);

        assert!(trace.attempts.is_empty());
        assert_eq!(trace.final_.served_from, ServedFrom::Error);
        assert_eq!(trace.final_.served_rung, None);
    }

    #[test]
    fn message_with_image_block_sets_has_images() {
        let req = br#"{"messages":[{"role":"user","content":[{"type":"image"}]}]}"#;
        let (_, _, has_images) = request_features(req);
        assert!(has_images);
    }

    #[test]
    fn prompt_hash_never_contains_raw_prompt_text() {
        let hash = prompt_hash("salt", b"super secret prompt");
        assert!(!hash.contains("secret"));
        assert_eq!(hash.len(), 64);
    }
}
