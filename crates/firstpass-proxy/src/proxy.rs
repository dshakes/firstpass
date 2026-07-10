//! Axum wiring: routes, request/response shapes, and observe-mode trace construction
//! (SPEC §7.1, §7.1a — forward unchanged, record asynchronously, zero added latency).

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use firstpass_core::features::{hour_bucket, token_bucket};
use firstpass_core::hashchain::sha256_hex;
use firstpass_core::{
    Attempt, DeferredVerdict, FEATURE_VERSION, Features, FinalOutcome, GENESIS_HASH, Mode,
    PolicyRef, RequestInfo, Score, ServedFrom, TaskKind, Trace, Verdict,
};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;

use crate::config::ProxyConfig;
use crate::error::ProxyError;
use crate::gate::{GateHealthRegistry, resolve_gates};
use crate::provider::{Auth, ChatMessage, ModelRequest, ModelResponse, ProviderRegistry};
use crate::router::{EnforceCtx, EngineOutcome, route_enforce};
use crate::store;
use crate::upstream::{forward_anthropic, forward_anthropic_streaming};
use firstpass_core::Route;

/// Shared state handed to every request handler. Cheap to clone: an `Arc`ed config, a
/// pooled HTTP client, and an unbounded channel sender.
#[derive(Clone)]
pub struct AppState {
    /// Static proxy configuration.
    pub config: Arc<ProxyConfig>,
    /// Shared, connection-pooled HTTP client used to call upstream (observe passthrough).
    pub http: reqwest::Client,
    /// Multi-provider registry used by the enforce-mode escalation engine.
    pub providers: ProviderRegistry,
    /// Per-gate error budgets (auto-disable), shared across requests.
    pub gate_health: Arc<GateHealthRegistry>,
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
        .route("/v1/feedback", post(feedback))
        .route("/v1/capabilities", get(capabilities))
        .route("/healthz", get(healthz))
        .with_state(state)
}

/// `GET /healthz` — liveness probe.
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// `GET /v1/capabilities` — agent-first discovery (SPEC §0.2, §7.4): what this proxy speaks,
/// which modes are live, the first enforce route's ladder/gates, and how to turn it off.
async fn capabilities(State(state): State<AppState>) -> impl IntoResponse {
    // Report the first enforce route's ladder + gates, so an agent can discover what it's routed
    // through. Empty when no routing config is loaded (pure observe deployment).
    let (ladder, gates) = state
        .config
        .routing
        .as_ref()
        .and_then(|c| c.routes.iter().find(|r| r.mode == Mode::Enforce))
        .map(|r| (r.ladder.clone(), r.gates.clone()))
        .unwrap_or_default();
    Json(serde_json::json!({
        "service": "firstpass",
        "version": env!("CARGO_PKG_VERSION"),
        "feature_version": FEATURE_VERSION,
        "modes": ["observe", "enforce"],
        "wire_apis": ["anthropic.messages"],
        "ladder": ladder,
        "gates": gates,
        "feedback_api": "POST /v1/feedback",
        "offboarding": "unset ANTHROPIC_BASE_URL",
    }))
}

/// Body of `POST /v1/feedback`: a downstream outcome reported for a past decision.
#[derive(Debug, Deserialize)]
struct FeedbackRequest {
    /// The `trace_id` of the decision this outcome is about.
    trace_id: String,
    /// The gate/source id, e.g. `"tests"` or `"feedback:ci"`.
    gate_id: String,
    /// `"pass"` | `"fail"` | `"abstain"`.
    verdict: String,
    /// Optional confidence in `[0, 1]`.
    #[serde(default)]
    score: Option<f64>,
    /// Who reported it (a CI system, a human reviewer, a deferred gate).
    reporter: String,
}

/// `POST /v1/feedback` — attach a downstream outcome (deferred verdict) to a past trace, closing
/// the outcome-feedback loop (SPEC §8.3.4). The verdict is stored in a **separate** table keyed
/// by `trace_id`; the sealed, hashed trace is never mutated, so the audit chain stays verifiable.
/// Returns `202 Accepted`. This is the signal that later calibrates the gates.
async fn feedback(State(state): State<AppState>, body: Bytes) -> Response {
    let req: FeedbackRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return ProxyError::BadRequest(format!("invalid feedback body: {e}")).into_response();
        }
    };
    let verdict = match req.verdict.as_str() {
        "pass" => Verdict::Pass,
        "fail" => Verdict::Fail,
        "abstain" => Verdict::Abstain,
        other => {
            return ProxyError::BadRequest(format!("unknown verdict {other:?}")).into_response();
        }
    };
    let score = match req.score {
        Some(s) => match Score::new(s) {
            Ok(sc) => Some(sc),
            Err(_) => {
                return ProxyError::BadRequest(format!("score {s} out of range [0,1]"))
                    .into_response();
            }
        },
        None => None,
    };

    let db = state.config.db_path.clone();

    // Reject feedback for an unknown trace, so orphan outcomes can't accumulate.
    let (db_check, tid_check) = (db.clone(), req.trace_id.clone());
    match tokio::task::spawn_blocking(move || store::trace_exists(&db_check, &tid_check)).await {
        Ok(Ok(true)) => {}
        Ok(Ok(false)) => {
            return ProxyError::NotFound(format!("unknown trace_id {:?}", req.trace_id))
                .into_response();
        }
        Ok(Err(e)) => {
            tracing::error!(%e, "feedback: trace_exists check failed");
            return ProxyError::Internal(e.to_string()).into_response();
        }
        Err(e) => {
            tracing::error!(%e, "feedback: trace_exists task panicked");
            return ProxyError::Internal(e.to_string()).into_response();
        }
    }

    let dv = DeferredVerdict {
        gate_id: req.gate_id,
        verdict,
        score,
        reported_at: jiff::Timestamp::now(),
        reporter: req.reporter,
    };
    let trace_id = req.trace_id.clone();
    match tokio::task::spawn_blocking(move || store::append_deferred(&db, &req.trace_id, &dv)).await
    {
        Ok(Ok(())) => (
            axum::http::StatusCode::ACCEPTED,
            Json(serde_json::json!({ "status": "recorded", "trace_id": trace_id })),
        )
            .into_response(),
        Ok(Err(e)) => {
            tracing::error!(%e, "feedback: append_deferred failed");
            ProxyError::Internal(e.to_string()).into_response()
        }
        Err(e) => {
            tracing::error!(%e, "feedback: append_deferred task panicked");
            ProxyError::Internal(e.to_string()).into_response()
        }
    }
}

/// The header a caller may set to group requests into a session for the audit trail. When
/// absent, each request is its own session (keyed by its own trace id).
const SESSION_HEADER: &str = "x-firstpass-session";

/// Header carrying the calling agent identity (feature/routing signal).
const AGENT_HEADER: &str = "x-firstpass-agent";
/// Header carrying the calling subagent identity.
const SUBAGENT_HEADER: &str = "x-firstpass-subagent";

/// `POST /v1/messages` — dispatch on the matched route's mode. **Enforce** routes run the
/// escalation engine (gate + escalate + failover); everything else is an **observe**
/// passthrough (forward unchanged, trace asynchronously). Either way the trace is recorded
/// off the response path.
async fn messages(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let session_header = header_str(&headers, SESSION_HEADER);

    // Only parse the request for routing when a routing config is loaded — an observe-only
    // deployment does zero on-path parsing and keeps its zero-added-latency guarantee.
    if let Some(routing) = state.config.routing.as_ref() {
        let features = extract_features(&headers, &body);
        if let Some(route) = routing
            .route_for(&features)
            .filter(|r| r.mode == Mode::Enforce && !r.ladder.is_empty())
        {
            // Clone the matched route so no borrow of `state.config` is held across the await;
            // routes are tiny (a handful of strings).
            let route = route.clone();
            if enforce_can_handle(&features, &body) {
                return handle_enforce(&state, &headers, &body, features, &route, session_header)
                    .await;
            }
            // The request carries tools / images / tool-result blocks the text-only enforce path
            // can't round-trip faithfully yet. Rather than drop them and serve corrupted output,
            // fall through to transparent observe passthrough (correct, un-gated) for this request.
            tracing::info!(
                "enforce route matched but request has tool/image content; serving via observe passthrough"
            );
        }
    }
    observe_passthrough(state, headers, body, session_header).await
}

/// Whether the enforce path can faithfully handle this request. Enforce normalizes content to
/// text and re-synthesizes a text response, so it cannot round-trip tool calls or images — a
/// request that declares tools, carries images, or contains tool_use/tool_result blocks is served
/// via transparent observe passthrough instead of being silently corrupted.
///
// ponytail: full tool/multimodal round-tripping through the ladder is the follow-on (needs
// provider-adapter work + live verification); this guard just refuses to corrupt in the meantime.
fn enforce_can_handle(features: &Features, body: &[u8]) -> bool {
    features.tool_count == 0 && !features.has_images && !messages_have_tool_blocks(body)
}

/// Whether any message carries a `tool_use` or `tool_result` content block (a multi-turn tool
/// conversation), which the text-only enforce normalization would drop.
fn messages_have_tool_blocks(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|json| {
            json.get("messages")
                .and_then(Value::as_array)
                .map(|messages| messages.iter().any(message_has_tool_block))
        })
        .unwrap_or(false)
}

/// Whether a single message's content contains a `tool_use` or `tool_result` block.
fn message_has_tool_block(message: &Value) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks.iter().any(|block| {
                matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("tool_use" | "tool_result")
                )
            })
        })
}

/// Whether the request opts into server-sent-events streaming (`"stream": true`).
fn is_stream_request(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|json| json.get("stream").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// Read a header as an owned `String`, if present and valid UTF-8.
fn header_str(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

/// Build the routing/telemetry feature vector from request headers + body (best-effort;
/// malformed fields fall back to safe defaults — this must never fail a request).
fn extract_features(headers: &HeaderMap, body: &[u8]) -> Features {
    let (_model, tool_count, has_images) = request_features(body);
    let mut f = Features::new(TaskKind::Other);
    f.agent = header_str(headers, AGENT_HEADER);
    f.subagent = header_str(headers, SUBAGENT_HEADER);
    f.tool_count = tool_count;
    f.has_images = has_images;
    // Pre-call we don't know the token count, so bucket by request byte size — a coarse,
    // monotonic proxy that never exposes the exact prompt (matches the privacy contract).
    f.prompt_token_bucket = token_bucket(body.len() as u64);
    f.hour_bucket = hour_bucket(jiff::Timestamp::now());
    f
}

/// Enforce mode (SPEC §7.1): run the escalation engine and serve the first output that clears
/// the route's gates, escalating on failure with cross-provider failover.
async fn handle_enforce(
    state: &AppState,
    headers: &HeaderMap,
    body: &Bytes,
    features: Features,
    route: &Route,
    session_header: Option<String>,
) -> Response {
    let Some(base_request) = parse_model_request(body) else {
        return ProxyError::BadRequest(
            "request body is not a valid Anthropic Messages request".to_owned(),
        )
        .into_response();
    };
    let auth = Auth::from_headers(headers);
    let gate_defs = state
        .config
        .routing
        .as_ref()
        .map_or(&[][..], |cfg| &cfg.gate_defs);
    let gates = resolve_gates(&route.gates, gate_defs);
    let session_id = session_header.unwrap_or_else(|| Uuid::now_v7().to_string());
    let (budget, max_rungs) = match state.config.routing.as_ref() {
        Some(cfg) => (
            cfg.budget.per_request_usd,
            cfg.escalation.max_rungs_per_request,
        ),
        None => (None, 3),
    };

    let ctx = EnforceCtx {
        ladder: &route.ladder,
        gates: &gates,
        health: &state.gate_health,
        base_request: &base_request,
        providers: &state.providers,
        auth: &auth,
        prices: &state.config.prices,
        budget_per_request_usd: budget,
        max_rungs,
        features,
        tenant_id: state.config.tenant_id.clone(),
        session_id,
        prompt_hash: prompt_hash(&state.config.prompt_salt, body),
        api: "anthropic.messages".to_owned(),
        policy_id: "static-ladder@v0".to_owned(),
    };

    let (outcome, trace) = route_enforce(ctx).await;
    // The trace is already built; enqueue it off-path (send is non-blocking on an unbounded
    // channel, so no spawn needed).
    if state.traces.send(trace).is_err() {
        tracing::warn!("trace writer is gone; dropping enforce trace");
    }

    match outcome {
        EngineOutcome::Served(resp) => (
            axum::http::StatusCode::OK,
            Json(anthropic_response_json(&resp)),
        )
            .into_response(),
        EngineOutcome::Failed(msg) => ProxyError::Engine(msg).into_response(),
    }
}

/// Parse an Anthropic Messages request body into the normalized [`ModelRequest`]. Returns
/// `None` if the body isn't valid JSON or lacks a `messages` array.
///
// Content blocks are collapsed to their concatenated text. Requests carrying tool_use/tool_result
// or image blocks never reach this function — `enforce_can_handle` routes them to transparent
// observe passthrough instead, so nothing is silently dropped here. Full multimodal round-tripping
// through the ladder is the follow-up; the enforce beachhead is text/code.
fn parse_model_request(body: &[u8]) -> Option<ModelRequest> {
    let json: Value = serde_json::from_slice(body).ok()?;
    let messages_json = json.get("messages")?.as_array()?;
    let messages = messages_json
        .iter()
        .map(|m| ChatMessage {
            role: m
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_owned(),
            content: content_to_text(m.get("content")),
        })
        .collect();
    let system = json
        .get("system")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let max_tokens = json
        .get("max_tokens")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(1024);
    let tools = json.get("tools").cloned().unwrap_or(Value::Null);
    Some(ModelRequest {
        model: json
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        system,
        messages,
        max_tokens,
        tools,
    })
}

/// Flatten Anthropic message content (a string, or an array of `{type,text}` blocks) to text.
fn content_to_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Render a served [`ModelResponse`] back into an Anthropic Messages response envelope, so the
/// caller sees the same wire shape regardless of which provider actually answered.
fn anthropic_response_json(resp: &ModelResponse) -> Value {
    serde_json::json!({
        "id": format!("msg_{}", Uuid::now_v7()),
        "type": "message",
        "role": "assistant",
        "model": resp.model,
        "content": [{ "type": "text", "text": resp.text }],
        "usage": { "input_tokens": resp.in_tokens, "output_tokens": resp.out_tokens },
    })
}

/// Observe mode (SPEC §7.1a): forward unchanged, return unchanged, trace asynchronously.
async fn observe_passthrough(
    state: AppState,
    headers: HeaderMap,
    body: Bytes,
    session_header: Option<String>,
) -> Response {
    // Streaming requests are relayed chunk-by-chunk rather than buffered (SPEC §7.4).
    if is_stream_request(&body) {
        return observe_stream(state, headers, body, session_header).await;
    }
    let start = Instant::now();
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

/// Observe mode for a streaming request (`stream: true`): relay the upstream SSE response
/// chunk-by-chunk instead of buffering, so streaming is preserved to the caller and
/// time-to-first-byte stays low. `latency_ms` is time-to-response-headers (the added-latency
/// figure that matters), recorded off the response path.
///
// ponytail: streamed-response token usage lives in the SSE `message_start`/`message_delta` events
// we don't buffer, so the trace records request-side features + latency now; parsing usage from a
// teed SSE stream is the follow-on.
async fn observe_stream(
    state: AppState,
    headers: HeaderMap,
    body: Bytes,
    session_header: Option<String>,
) -> Response {
    let start = Instant::now();
    let result = forward_anthropic_streaming(
        &state.http,
        &state.config.upstream_anthropic,
        &headers,
        body.clone(),
    )
    .await;
    let latency_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    match result {
        Ok((status, resp_headers, response)) => {
            spawn_stream_trace(&state, body, latency_ms, session_header);
            let stream_body = Body::from_stream(response.bytes_stream());
            (status, resp_headers, stream_body).into_response()
        }
        Err(err) => {
            spawn_trace(&state, body, None, latency_ms, session_header);
            err.into_response()
        }
    }
}

/// Enqueue a request-side trace for a streamed observe response, off the response path.
fn spawn_stream_trace(
    state: &AppState,
    req_body: Bytes,
    latency_ms: u64,
    session_header: Option<String>,
) {
    let config = state.config.clone();
    let traces = state.traces.clone();
    tokio::spawn(async move {
        let trace = build_stream_trace(&config, &req_body, latency_ms, session_header.as_deref());
        if traces.send(trace).is_err() {
            tracing::warn!("trace writer is gone; dropping stream trace");
        }
    });
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

/// Build the observe-mode trace for a **streamed** response: we relayed real bytes to the caller,
/// but the token usage lives in the SSE events we didn't buffer, so it's recorded as served with
/// unknown (zero) usage — honest about what we served without inventing token counts.
fn build_stream_trace(
    config: &ProxyConfig,
    req_body: &Bytes,
    latency_ms: u64,
    session_header: Option<&str>,
) -> Trace {
    let (req_model, tool_count, has_images) = request_features(req_body);
    let model = req_model.unwrap_or_else(|| "unknown".to_owned());

    let attempt = Attempt {
        rung: 0,
        model,
        provider: "anthropic".to_owned(),
        in_tokens: 0,
        out_tokens: 0,
        cost_usd: 0.0,
        latency_ms,
        gates: Vec::new(),
        verdict: Verdict::Pass,
    };

    let mut trace = base_trace(config, req_body, latency_ms, session_header);
    trace.request.features.tool_count = tool_count;
    trace.request.features.has_images = has_images;
    trace.attempts.push(attempt);
    trace.final_ = FinalOutcome {
        served_rung: Some(0),
        served_from: ServedFrom::Attempt,
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

    #[test]
    fn parse_model_request_flattens_content_blocks() {
        let body = br#"{"model":"m","system":"sys","max_tokens":50,
            "messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
                        {"role":"assistant","content":"c"}]}"#;
        let req = parse_model_request(body).unwrap();
        assert_eq!(req.system.as_deref(), Some("sys"));
        assert_eq!(req.max_tokens, 50);
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].content, "a\nb");
        assert_eq!(req.messages[1].content, "c");
    }

    #[test]
    fn parse_model_request_rejects_non_message_bodies() {
        assert!(parse_model_request(b"not json").is_none());
        assert!(parse_model_request(br#"{"no":"messages"}"#).is_none());
    }

    // --- Enforce-path handler tests (drive `messages` end-to-end with mock providers) ---

    use crate::provider::{MockProvider, ModelResponse, Provider, ProviderError, ProviderRegistry};
    use axum::extract::State;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    fn model_resp(model: &str, text: &str) -> ModelResponse {
        ModelResponse {
            model: model.to_owned(),
            text: text.to_owned(),
            in_tokens: 1000,
            out_tokens: 400,
            raw: serde_json::Value::Null,
        }
    }

    /// Build an `AppState` whose anthropic provider answers the given per-model outcomes, with an
    /// enforce route over `ladder`/`gates`. Returns the state and the trace receiver.
    fn enforce_state(
        ladder: &[&str],
        gates: &[&str],
        outcomes: Vec<(&str, Result<ModelResponse, ProviderError>)>,
    ) -> (AppState, mpsc::UnboundedReceiver<Trace>) {
        let toml = format!(
            "[[route]]\nmatch = {{}}\nmode = \"enforce\"\nladder = [{}]\ngates = [{}]\n",
            ladder
                .iter()
                .map(|m| format!("\"{m}\""))
                .collect::<Vec<_>>()
                .join(", "),
            gates
                .iter()
                .map(|g| format!("\"{g}\""))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let config = ProxyConfig::from_lookup(|k| match k {
            "FIRSTPASS_CONFIG_TOML" => Some(toml.clone()),
            "FIRSTPASS_MODE" => Some("enforce".to_owned()),
            _ => None,
        })
        .unwrap();

        let mut outs = HashMap::new();
        for (model, out) in outcomes {
            outs.insert(model.to_owned(), out);
        }
        let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        map.insert(
            "anthropic".to_owned(),
            Arc::new(MockProvider::new("anthropic", outs)),
        );
        let providers = ProviderRegistry::from_map(map);

        let (traces, rx) = mpsc::unbounded_channel();
        let state = AppState {
            config: Arc::new(config),
            http: reqwest::Client::new(),
            providers,
            gate_health: Arc::new(GateHealthRegistry::new()),
            traces,
        };
        (state, rx)
    }

    fn user_body() -> Bytes {
        Bytes::from_static(
            br#"{"model":"ignored","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}"#,
        )
    }

    async fn body_json(resp: Response) -> Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn enforce_serves_first_pass_and_returns_anthropic_shape() {
        let (state, mut rx) = enforce_state(
            &["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"],
            &["non-empty"],
            vec![(
                "anthropic/claude-haiku-4-5",
                Ok(model_resp("anthropic/claude-haiku-4-5", "hello")),
            )],
        );
        let resp = messages(State(state), HeaderMap::new(), user_body()).await;
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["type"], "message");
        assert_eq!(json["content"][0]["text"], "hello");
        assert_eq!(json["model"], "anthropic/claude-haiku-4-5");

        let trace = rx.try_recv().expect("a trace was enqueued");
        assert_eq!(trace.mode, Mode::Enforce);
        assert_eq!(trace.final_.served_rung, Some(0));
        assert_eq!(trace.attempts.len(), 1);
    }

    #[tokio::test]
    async fn enforce_escalates_then_serves_and_traces_two_attempts() {
        let (state, mut rx) = enforce_state(
            &["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"],
            &["non-empty"],
            vec![
                (
                    "anthropic/claude-haiku-4-5",
                    Ok(model_resp("anthropic/claude-haiku-4-5", "   ")),
                ), // fails
                (
                    "anthropic/claude-sonnet-5",
                    Ok(model_resp("anthropic/claude-sonnet-5", "answer")),
                ),
            ],
        );
        let resp = messages(State(state), HeaderMap::new(), user_body()).await;
        let json = body_json(resp).await;
        assert_eq!(json["content"][0]["text"], "answer");

        let trace = rx.try_recv().expect("trace enqueued");
        assert_eq!(trace.attempts.len(), 2);
        assert_eq!(trace.final_.escalations, 1);
        assert_eq!(trace.final_.served_rung, Some(1));
    }

    #[tokio::test]
    async fn enforce_all_rungs_error_returns_502() {
        let (state, mut rx) = enforce_state(
            &["anthropic/claude-haiku-4-5"],
            &["non-empty"],
            vec![(
                "anthropic/claude-haiku-4-5",
                Err(ProviderError::Transport("down".into())),
            )],
        );
        let resp = messages(State(state), HeaderMap::new(), user_body()).await;
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_GATEWAY);
        // A trace is still recorded for the failed decision.
        assert!(rx.try_recv().is_ok());
    }

    #[tokio::test]
    async fn no_routing_config_falls_through_to_observe_not_enforce() {
        // config with no routing => enforce path never runs; observe attempts a real upstream
        // call which fails fast against an unroutable host. We only assert it did NOT take the
        // enforce branch (which would have used the mock and returned 200 with our text).
        let config = ProxyConfig::from_lookup(|k| match k {
            "FIRSTPASS_UPSTREAM_ANTHROPIC" => Some("http://127.0.0.1:1".to_owned()),
            _ => None,
        })
        .unwrap();
        let (traces, _rx) = mpsc::unbounded_channel();
        let state = AppState {
            config: Arc::new(config),
            http: reqwest::Client::new(),
            providers: ProviderRegistry::new("http://127.0.0.1:1", "http://127.0.0.1:1"),
            gate_health: Arc::new(GateHealthRegistry::new()),
            traces,
        };
        let resp = messages(State(state), HeaderMap::new(), user_body()).await;
        // Observe path forwards upstream; the bogus host yields a gateway error, not our 200.
        assert_ne!(resp.status(), axum::http::StatusCode::OK);
    }

    #[test]
    fn detects_stream_requests() {
        assert!(is_stream_request(br#"{"stream": true}"#));
        assert!(!is_stream_request(br#"{"stream": false}"#));
        assert!(!is_stream_request(br#"{"model":"m"}"#));
        assert!(!is_stream_request(b"not json"));
    }

    #[test]
    fn detects_tool_blocks_in_messages() {
        let with =
            br#"{"messages":[{"role":"user","content":[{"type":"tool_result","content":"42"}]}]}"#;
        let without = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        assert!(messages_have_tool_blocks(with));
        assert!(!messages_have_tool_blocks(without));
    }

    #[test]
    fn enforce_only_handles_plain_text() {
        let plain =
            Bytes::from_static(br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let tools = Bytes::from_static(
            br#"{"model":"m","tools":[{"name":"t"}],"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let f_plain = extract_features(&HeaderMap::new(), &plain);
        let f_tools = extract_features(&HeaderMap::new(), &tools);
        assert!(enforce_can_handle(&f_plain, &plain));
        assert!(!enforce_can_handle(&f_tools, &tools));
    }

    /// B2: an enforce route serves plain text (200 from the mock) but falls back to transparent
    /// observe passthrough for tool/image requests rather than dropping blocks — proven by the
    /// tool request hitting the (bogus) upstream instead of the enforcing mock.
    #[tokio::test]
    async fn enforce_falls_back_to_observe_for_tool_requests() {
        let toml = "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/m\"]\ngates = [\"non-empty\"]\n";
        let config = ProxyConfig::from_lookup(|k| match k {
            "FIRSTPASS_CONFIG_TOML" => Some(toml.to_owned()),
            "FIRSTPASS_MODE" => Some("enforce".to_owned()),
            "FIRSTPASS_UPSTREAM_ANTHROPIC" => Some("http://127.0.0.1:1".to_owned()),
            _ => None,
        })
        .unwrap();
        let mut outs = HashMap::new();
        outs.insert(
            "anthropic/m".to_owned(),
            Ok(model_resp("anthropic/m", "hello")),
        );
        let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        map.insert(
            "anthropic".to_owned(),
            Arc::new(MockProvider::new("anthropic", outs)),
        );
        let (traces, _rx) = mpsc::unbounded_channel();
        let state = AppState {
            config: Arc::new(config),
            http: reqwest::Client::new(),
            providers: ProviderRegistry::from_map(map),
            gate_health: Arc::new(GateHealthRegistry::new()),
            traces,
        };

        // Plain text enforces: the mock serves 200.
        let plain =
            Bytes::from_static(br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let resp = messages(State(state.clone()), HeaderMap::new(), plain).await;
        assert_eq!(
            resp.status(),
            axum::http::StatusCode::OK,
            "plain text should enforce"
        );

        // Declares tools => cannot enforce faithfully => observe fallback => bogus upstream => not 200.
        let tools = Bytes::from_static(
            br#"{"model":"m","tools":[{"name":"get_weather"}],"messages":[{"role":"user","content":"hi"}]}"#,
        );
        let resp = messages(State(state.clone()), HeaderMap::new(), tools).await;
        assert_ne!(
            resp.status(),
            axum::http::StatusCode::OK,
            "tool request must fall back to observe, not enforce"
        );

        // tool_result block in a message => same fallback.
        let toolres = Bytes::from_static(
            br#"{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x","content":"42"}]}]}"#,
        );
        let resp = messages(State(state), HeaderMap::new(), toolres).await;
        assert_ne!(
            resp.status(),
            axum::http::StatusCode::OK,
            "tool_result request must fall back to observe, not enforce"
        );
    }

    // --- Feedback API tests (drive `feedback` against a real temp trace store) ---

    /// Persist one trace to a fresh temp DB and return (state, db_path, trace_id).
    async fn feedback_state() -> (AppState, std::path::PathBuf, String) {
        let db = std::env::temp_dir().join(format!("firstpass-feedback-{}.db", Uuid::now_v7()));
        let (tx, handle) = crate::store::open(&db).unwrap();

        let mut trace = build_error_trace(
            &ProxyConfig::from_lookup(|_| None).unwrap(),
            &Bytes::from_static(b"{}"),
            5,
            Some("sess-fb"),
        );
        trace.attempts.push(Attempt {
            rung: 0,
            model: "anthropic/claude-haiku-4-5".into(),
            provider: "anthropic".into(),
            in_tokens: 10,
            out_tokens: 5,
            cost_usd: 0.001,
            latency_ms: 5,
            gates: vec![],
            verdict: Verdict::Pass,
        });
        let trace_id = trace.trace_id.to_string();
        tx.send(trace).unwrap();
        drop(tx);
        handle.await.unwrap();

        let db_str = db.to_string_lossy().into_owned();
        let config = ProxyConfig::from_lookup(move |k| match k {
            "FIRSTPASS_DB" => Some(db_str.clone()),
            _ => None,
        })
        .unwrap();
        let (traces, _rx) = mpsc::unbounded_channel();
        let state = AppState {
            config: Arc::new(config),
            http: reqwest::Client::new(),
            providers: ProviderRegistry::new("http://127.0.0.1:1", "http://127.0.0.1:1"),
            gate_health: Arc::new(GateHealthRegistry::new()),
            traces,
        };
        (state, db, trace_id)
    }

    #[tokio::test]
    async fn feedback_records_a_deferred_verdict_without_breaking_the_chain() {
        let (state, db, trace_id) = feedback_state().await;
        let body = Bytes::from(
            serde_json::json!({
                "trace_id": trace_id,
                "gate_id": "tests",
                "verdict": "pass",
                "score": 1.0,
                "reporter": "ci",
            })
            .to_string(),
        );
        let resp = feedback(State(state), body).await;
        assert_eq!(resp.status(), axum::http::StatusCode::ACCEPTED);

        // The deferred verdict is visible on the trace view...
        let view = crate::store::load_trace_view(&db, &trace_id)
            .unwrap()
            .unwrap();
        assert_eq!(view.deferred.len(), 1);
        assert_eq!(view.deferred[0].gate_id, "tests");
        // ...and the sealed chain still verifies (the outcome didn't mutate the trace).
        let traces = crate::store::load_all_traces(&db).unwrap();
        firstpass_core::verify_chain(&traces, GENESIS_HASH).unwrap();

        let _ = std::fs::remove_file(&db);
    }

    #[tokio::test]
    async fn feedback_for_unknown_trace_is_404() {
        let (state, db, _trace_id) = feedback_state().await;
        let body = Bytes::from(
            serde_json::json!({
                "trace_id": "does-not-exist",
                "gate_id": "tests",
                "verdict": "pass",
                "reporter": "ci",
            })
            .to_string(),
        );
        let resp = feedback(State(state), body).await;
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
        let _ = std::fs::remove_file(&db);
    }

    #[tokio::test]
    async fn feedback_rejects_bad_verdict_and_score() {
        let (state, db, trace_id) = feedback_state().await;
        let bad_verdict = Bytes::from(
            serde_json::json!({ "trace_id": trace_id, "gate_id": "g", "verdict": "maybe", "reporter": "x" })
                .to_string(),
        );
        assert_eq!(
            feedback(State(state.clone()), bad_verdict).await.status(),
            axum::http::StatusCode::BAD_REQUEST
        );
        let bad_score = Bytes::from(
            serde_json::json!({ "trace_id": trace_id, "gate_id": "g", "verdict": "pass", "score": 9.0, "reporter": "x" })
                .to_string(),
        );
        assert_eq!(
            feedback(State(state), bad_score).await.status(),
            axum::http::StatusCode::BAD_REQUEST
        );
        let _ = std::fs::remove_file(&db);
    }
}
