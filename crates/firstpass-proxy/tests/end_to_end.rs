//! End-to-end, NO-MOCK integration test.
//!
//! This drives the entire stack over real HTTP with zero test doubles in the plane: the real
//! `ProviderRegistry` (real `AnthropicProvider`/`OpenAiProvider` reqwest clients) talks to a
//! local server that faithfully implements the Anthropic Messages and OpenAI Chat Completions
//! wire protocols. A request flows: reqwest client → proxy `/v1/messages` → real HTTP call to the
//! upstream → JSON decode → gates → escalate → serve → async trace persist → `/v1/feedback` →
//! deferred verdict → chain verification. If any wiring is wrong, this fails.
//!
//! It's the standard way to verify an API client without hitting a paid cloud endpoint: point it
//! at a spec-accurate local server. That's what lifts the adapters from "compiles" to "works".
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use bytes::Bytes;
use firstpass_core::{GENESIS_HASH, ServedFrom, Verdict, verify_chain};
use firstpass_proxy::provider::ProviderRegistry;
use firstpass_proxy::proxy::AppState;
use firstpass_proxy::{ProxyConfig, app, store};
use serde_json::{Value, json};

/// A faithful local stand-in for the frontier providers. Behaviour is keyed on the requested
/// model so a single server can drive escalation and failover scenarios:
/// - model contains `haiku`  → returns EMPTY content (fails the non-empty gate)
/// - model contains `fail5xx`→ returns HTTP 503 (failover-eligible)
/// - anything else           → returns good content (passes the gate)
async fn spawn_upstream() -> String {
    let router = Router::new()
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/responses", post(openai_responses))
        // Gemini's `{model}:generateContent` is a single path segment.
        .route("/v1beta/models/{model_action}", post(gemini_generate));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// One recorded Gemini upstream request: `(path_and_query, x_goog_api_key_header)`.
type GeminiLogEntry = (String, Option<String>);

/// A recording Gemini upstream: logs every request's path+query and `x-goog-api-key` header so
/// tests can assert that the proxy forwarded the query correctly and scrubbed `?key=`.
#[derive(Clone, Default)]
struct GeminiRequestLog(Arc<std::sync::Mutex<Vec<GeminiLogEntry>>>);

async fn recording_gemini_handler(
    State(log): State<GeminiRequestLog>,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    _body: Bytes,
) -> Response {
    let pq = uri
        .path_and_query()
        .map(|pq| pq.to_string())
        .unwrap_or_default();
    let key = headers
        .get("x-goog-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    log.0.lock().unwrap().push((pq, key));
    Json(json!({
        "candidates": [{ "content": { "role": "model", "parts": [{ "text": "recorded" }] } }],
        "usageMetadata": { "promptTokenCount": 5, "candidatesTokenCount": 3 },
    }))
    .into_response()
}

async fn spawn_recording_gemini_upstream() -> (String, GeminiRequestLog) {
    let log = GeminiRequestLog::default();
    let log_clone = log.clone();
    let router = Router::new()
        .route(
            "/v1beta/models/{model_action}",
            post(recording_gemini_handler),
        )
        .with_state(log_clone);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("http://{addr}"), log)
}

fn requested_model(body: &Bytes) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_default()
}

/// Anthropic Messages wire response (or a 503 for the failover case).
async fn anthropic_messages(body: Bytes) -> Response {
    let model = requested_model(&body);
    if model.contains("fail5xx") {
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, "upstream down").into_response();
    }
    // A tool-using answer: preamble text plus one `tool_use` block.
    if model.contains("tools") {
        return Json(json!({
            "id": "msg_e2e_tools",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                { "type": "text", "text": "Checking the weather." },
                { "type": "tool_use", "id": "toolu_1", "name": "get_weather",
                  "input": { "city": "Paris" } },
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 1000, "output_tokens": 200 },
        }))
        .into_response();
    }
    let text = if model.contains("haiku") {
        ""
    } else {
        "fn main() {} // compiles"
    };
    Json(json!({
        "id": "msg_e2e",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{ "type": "text", "text": text }],
        "usage": { "input_tokens": 1000, "output_tokens": 200 },
    }))
    .into_response()
}

/// OpenAI Chat Completions wire response (always good — used as the failover target).
async fn openai_chat(body: Bytes) -> Response {
    let model = requested_model(&body);
    Json(json!({
        "id": "chatcmpl_e2e",
        "object": "chat.completion",
        "model": model,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": "answer via openai" }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 1000, "completion_tokens": 200 },
    }))
    .into_response()
}

/// Responses-API wire response. Exists so a relay that targets the WRONG upstream path is
/// observable: a Responses-shaped body sent to /v1/chat/completions would land in `openai_chat`
/// and come back Chat-shaped, which is exactly the bug this endpoint had.
async fn openai_responses(body: Bytes) -> Response {
    let model = requested_model(&body);
    Json(json!({
        "id": "resp_e2e",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [{
            "type": "message", "id": "msg_e2e", "status": "completed", "role": "assistant",
            "content": [{ "type": "output_text", "text": "streamed via responses", "annotations": [] }],
        }],
        "output_text": "streamed via responses",
        "usage": { "input_tokens": 7, "output_tokens": 3, "total_tokens": 10 },
    }))
    .into_response()
}

/// Gemini `generateContent` wire response — a `candidates[].content.parts[].text` shape with
/// `usageMetadata`. The API key rides in the `x-goog-api-key` header (the mock doesn't check it).
///
/// Behaviour is keyed on the model in the URL path segment:
/// - model contains `haiku` → returns empty text (fails the non-empty gate)
/// - anything else          → returns "gemini says hi"
async fn gemini_generate(
    axum::extract::Path(model_action): axum::extract::Path<String>,
    body: Bytes,
) -> Response {
    let _ = &body;
    let model = model_action.split(':').next().unwrap_or_default();
    let text = if model.contains("haiku") {
        ""
    } else {
        "gemini says hi"
    };
    Json(json!({
        "candidates": [{ "content": { "role": "model", "parts": [{ "text": text }] } }],
        "usageMetadata": { "promptTokenCount": 5, "candidatesTokenCount": 3 },
    }))
    .into_response()
}

/// Start the real proxy with an enforce route over `ladder`, pointed at `upstream`. `providers_toml`
/// is extra `[[provider]]` TOML (empty for the built-in anthropic/openai path). Returns the proxy
/// base URL and the temp DB path.
async fn spawn_proxy(
    upstream: &str,
    ladder: &[&str],
    providers_toml: &str,
) -> (String, std::path::PathBuf) {
    spawn_proxy_with(upstream, ladder, providers_toml, "").await
}

/// As [`spawn_proxy`], plus arbitrary extra routing TOML (e.g. an `[escalation.*]` block), so a
/// test can exercise a feature exactly as an operator would configure it rather than by reaching
/// into `AppState`.
async fn spawn_proxy_with(
    upstream: &str,
    ladder: &[&str],
    providers_toml: &str,
    extra_toml: &str,
) -> (String, std::path::PathBuf) {
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let ladder_toml = ladder
        .iter()
        .map(|m| format!("\"{m}\""))
        .collect::<Vec<_>>()
        .join(", ");
    // These fixtures use synthetic model ids (`anthropic/claude-fail5xx`) that no price table
    // knows, and config now rejects an unpriced rung rather than let it bank a $0.00 receipt.
    // Declare a nominal price for each so the fixture states its cost instead of inferring it.
    // Only the synthetic ones: overriding a real model would flatten the ladder's price
    // gradient, and the escalation test asserts savings > 0.
    let builtin = firstpass_core::cost::PriceTable::defaults();
    let price_toml: String = ladder
        .iter()
        .filter(|m| builtin.get(m).is_none())
        .map(|m| {
            format!("[[price]]\nmodel = \"{m}\"\ninput_per_mtok = 1.0\noutput_per_mtok = 5.0\n")
        })
        .collect();
    let routing = format!(
        "{providers_toml}\n{price_toml}\n{extra_toml}\n[[route]]\nmatch = {{}}\nmode = \"enforce\"\nladder = [{ladder_toml}]\ngates = [\"non-empty\"]\n"
    );
    let upstream = upstream.to_owned();
    let db_str = db_path.to_string_lossy().into_owned();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_ANTHROPIC" => Some(upstream.clone()),
        "FIRSTPASS_UPSTREAM_OPENAI" => Some(upstream.clone()),
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream.clone()),
        "FIRSTPASS_MODE" => Some("enforce".to_owned()),
        "FIRSTPASS_CONFIG_TOML" => Some(routing.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();

    // The REAL registry, built exactly as production does — built-in anthropic/openai plus any
    // configured `[[provider]]` (e.g. gemini), all real reqwest clients pointed at the local server.
    let provider_defs = config
        .routing
        .as_ref()
        .map(|r| r.providers.as_slice())
        .unwrap_or_default();
    let providers = ProviderRegistry::from_config(
        provider_defs,
        &config.upstream_anthropic,
        &config.upstream_openai,
    );
    let (traces, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers,
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    (format!("http://{addr}"), db_path)
}

async fn wait_for_traces(db_path: &std::path::Path, want: usize) -> Vec<firstpass_core::Trace> {
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        let traces = store::load_all_traces(db_path).unwrap_or_default();
        if traces.len() >= want || std::time::Instant::now() >= deadline {
            return traces;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn enforce_escalates_over_real_http_then_feedback_attaches() {
    let upstream = spawn_upstream().await;
    // 3-rung ladder: haiku fails the gate, sonnet passes and is served (rung 1 < top rung 2),
    // so the counterfactual (opus) makes savings strictly positive.
    let (proxy, db) = spawn_proxy(
        &upstream,
        &[
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-4-8",
        ],
        "",
    )
    .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{proxy}/v1/messages"))
        .header("x-api-key", "byok-test")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 256,
            "messages": [{ "role": "user", "content": "write a hello world" }],
        }))
        .send()
        .await
        .unwrap();

    // The proxy served the escalated (sonnet) output, in Anthropic wire shape.
    assert_eq!(resp.status(), 200);
    let served: Value = resp.json().await.unwrap();
    assert_eq!(served["type"], "message");
    assert_eq!(served["content"][0]["text"], "fn main() {} // compiles");
    assert_eq!(served["model"], "anthropic/claude-sonnet-5");

    // The audit trace records the real escalation and positive savings; the chain verifies.
    let traces = wait_for_traces(&db, 1).await;
    assert_eq!(traces.len(), 1);
    let trace = &traces[0];
    assert_eq!(trace.attempts.len(), 2, "haiku failed, sonnet passed");
    assert_eq!(trace.attempts[0].verdict, Verdict::Fail);
    assert_eq!(trace.attempts[1].verdict, Verdict::Pass);
    assert_eq!(trace.final_.served_rung, Some(1));
    assert_eq!(trace.final_.served_from, ServedFrom::Attempt);
    assert!(
        trace.final_.savings_usd > 0.0,
        "served below top rung => real savings"
    );
    // real token usage decoded from the upstream response
    assert_eq!(trace.attempts[1].in_tokens, 1000);
    assert_eq!(trace.attempts[1].out_tokens, 200);
    verify_chain(&traces, GENESIS_HASH).unwrap();

    // The outcome loop: report that downstream tests passed, and confirm it attaches without
    // breaking the chain.
    let trace_id = trace.trace_id.to_string();
    let fb = client
        .post(format!("{proxy}/v1/feedback"))
        .json(&json!({
            "trace_id": trace_id,
            "gate_id": "tests",
            "verdict": "pass",
            "score": 1.0,
            "reporter": "ci",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(fb.status(), 202);

    let view = store::load_trace_view(&db, "default", &trace_id)
        .unwrap()
        .unwrap();
    assert_eq!(view.deferred.len(), 1);
    assert_eq!(view.deferred[0].gate_id, "tests");
    // Sealed bodies untouched: chain still verifies after the late outcome.
    verify_chain(&store::load_all_traces(&db).unwrap(), GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db);
}

#[tokio::test]
async fn cross_provider_failover_over_real_http() {
    let upstream = spawn_upstream().await;
    // Rung 0 (anthropic) 503s → failover to rung 1 (openai), whose REAL client decodes the
    // OpenAI wire response. Proves the Anthropic→OpenAI failover path end to end.
    let (proxy, db) = spawn_proxy(
        &upstream,
        &["anthropic/claude-fail5xx", "openai/gpt-5.5"],
        "",
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/messages"))
        .header("x-api-key", "byok-test")
        .json(&json!({
            "model": "claude-fail5xx",
            "max_tokens": 128,
            "messages": [{ "role": "user", "content": "hi" }],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let served: Value = resp.json().await.unwrap();
    assert_eq!(served["content"][0]["text"], "answer via openai");
    assert_eq!(served["model"], "openai/gpt-5.5");

    let traces = wait_for_traces(&db, 1).await;
    let trace = &traces[0];
    assert_eq!(
        trace.attempts[0].verdict,
        Verdict::Abstain,
        "anthropic 503 → abstain"
    );
    assert_eq!(trace.attempts[1].verdict, Verdict::Pass, "openai served");
    assert_eq!(trace.attempts[1].provider, "openai");
    assert_eq!(trace.final_.served_rung, Some(1));
    verify_chain(&traces, GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db);
}

#[tokio::test]
async fn gemini_dialect_serves_over_real_http() {
    let upstream = spawn_upstream().await;
    // A `[[provider]] gemini` pointed at the local server, with a gemini rung in the ladder. The
    // REAL GeminiProvider reqwest client translates the request, POSTs to
    // `/v1beta/models/{model}:generateContent`, and decodes the generateContent response — proving
    // the whole Gemini wire path end to end, not just the offline translation unit tests.
    let providers =
        format!("[[provider]]\nid = \"gemini\"\ndialect = \"gemini\"\nbase_url = \"{upstream}\"\n");
    let (proxy, db) = spawn_proxy(&upstream, &["gemini/gemini-2.0-flash"], &providers).await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/messages"))
        .header("x-api-key", "byok-test")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 128,
            "messages": [{ "role": "user", "content": "hi gemini" }],
        }))
        .send()
        .await
        .unwrap();

    // The proxy served the Gemini output, re-rendered in the Anthropic wire shape the caller speaks.
    assert_eq!(resp.status(), 200);
    let served: Value = resp.json().await.unwrap();
    assert_eq!(served["content"][0]["text"], "gemini says hi");
    assert_eq!(served["model"], "gemini/gemini-2.0-flash");

    let traces = wait_for_traces(&db, 1).await;
    let trace = &traces[0];
    assert_eq!(trace.attempts[0].verdict, Verdict::Pass);
    assert_eq!(trace.attempts[0].provider, "gemini");
    // Token usage came from the Gemini `usageMetadata`, proving the response decode wired through.
    assert_eq!(trace.attempts[0].in_tokens, 5);
    assert_eq!(trace.attempts[0].out_tokens, 3);
    verify_chain(&traces, GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db);
}

/// Session promotion, end to end over real HTTP and configured exactly as an operator would.
///
/// This is the wiring test, and it exists because `[escalation.session_promotion]` shipped as a
/// block that parsed and did nothing at all. Unit tests on the promoter would have passed happily
/// against that same dead config, so the assertion that matters is not "the promoter works" but
/// "a second request on this session really did skip the rung that already failed".
#[tokio::test]
async fn session_promotion_skips_the_rung_that_already_failed() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy_with(
        &upstream,
        &[
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-4-8",
        ],
        "",
        "[escalation.session_promotion]\nafter_failures = 1\nwindow = \"30m\"\n",
    )
    .await;

    let client = reqwest::Client::new();
    let send = |session: &str| {
        let client = client.clone();
        let proxy = proxy.clone();
        let session = session.to_owned();
        async move {
            client
                .post(format!("{proxy}/v1/messages"))
                .header("x-api-key", "byok-test")
                .header("x-firstpass-session", session)
                .json(&json!({
                    "model": "claude-haiku-4-5",
                    "max_tokens": 256,
                    "messages": [{ "role": "user", "content": "write a hello world" }],
                }))
                .send()
                .await
                .unwrap()
        }
    };

    // Turn 1: haiku fails the gate, the ladder climbs, sonnet serves.
    assert_eq!(send("sess-promote").await.status(), 200);
    let first = wait_for_traces(&db, 1).await;
    assert_eq!(first[0].attempts.len(), 2, "turn 1 must climb 0 → 1");
    assert_eq!(first[0].attempts[0].rung, 0);

    // Turn 2, same session: the promotion should start it at rung 1, so haiku — which is still
    // configured, still cheapest, and still fails — is never called at all.
    assert_eq!(send("sess-promote").await.status(), 200);
    let both = wait_for_traces(&db, 2).await;
    let second = both
        .iter()
        .find(|t| t.trace_id != first[0].trace_id)
        .expect("second trace");

    assert_eq!(
        second.attempts.len(),
        1,
        "promoted turn must not re-attempt the failing rung: {:?}",
        second.attempts.iter().map(|a| a.rung).collect::<Vec<_>>()
    );
    assert_eq!(second.attempts[0].rung, 1, "started on the promoted rung");
    assert_eq!(second.final_.served_rung, Some(1));

    // A different session must not inherit it — promotion is per conversation, not global.
    assert_eq!(send("sess-other").await.status(), 200);
    let all = wait_for_traces(&db, 3).await;
    let other = all
        .iter()
        .find(|t| t.trace_id != first[0].trace_id && t.trace_id != second.trace_id)
        .expect("third trace");
    assert_eq!(
        other.attempts.len(),
        2,
        "an unrelated session starts cold and climbs on its own"
    );
}

/// A coding agent sends no `x-firstpass-session`; its turns must still group (derived from the
/// stable conversation prefix) so promotion carries turn 1's failure into turn 2.
#[tokio::test]
async fn agent_turns_without_a_session_header_share_a_promotion() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy_with(
        &upstream,
        &[
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-4-8",
        ],
        "",
        "[escalation.session_promotion]\nafter_failures = 1\nwindow = \"30m\"\n",
    )
    .await;
    let client = reqwest::Client::new();
    let tools =
        json!([{ "name": "bash", "description": "run", "input_schema": { "type": "object" } }]);
    let send = |messages: serde_json::Value| {
        let client = client.clone();
        let url = format!("{proxy}/v1/messages");
        let tools = tools.clone();
        async move {
            client
                .post(url)
                .header("x-api-key", "byok-test")
                .json(&json!({
                    "model": "claude-haiku-4-5", "max_tokens": 256,
                    "tools": tools, "messages": messages,
                }))
                .send()
                .await
                .unwrap()
        }
    };
    let first_user = json!({ "role": "user", "content": "write a hello world" });

    assert_eq!(send(json!([first_user])).await.status(), 200);
    let first = wait_for_traces(&db, 1).await;
    assert_eq!(first[0].attempts.len(), 2, "turn 1 must climb 0 -> 1");

    // Turn 2 appends assistant + user; no session header anywhere.
    let turn2 = json!([
        first_user,
        { "role": "assistant", "content": "ok" },
        { "role": "user", "content": "now add a test" },
    ]);
    assert_eq!(send(turn2).await.status(), 200);
    let both = wait_for_traces(&db, 2).await;
    let second = both
        .iter()
        .find(|t| t.trace_id != first[0].trace_id)
        .expect("second trace");
    assert_eq!(second.attempts.len(), 1, "turn 2 must skip the failed rung");
    assert_eq!(second.attempts[0].rung, 1, "started on the promoted rung");
}

/// The Responses API served through the real gated ladder over real HTTP.
///
/// The point is not the translation — that is unit-tested. It is that a Responses client gets the
/// *same* engine: the cheap rung is tried, its empty answer fails the gate, the ladder escalates,
/// and the receipt records both attempts. A parallel implementation would be the easy mistake here,
/// and it would drift on exactly the parts that matter.
#[tokio::test]
async fn responses_api_is_gated_by_the_same_ladder() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy(
        &upstream,
        &["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"],
        "",
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .header("x-api-key", "byok-test")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "instructions": "be terse",
            "input": [{ "role": "user", "content": [{ "type": "input_text", "text": "write a hello world" }] }],
            "max_output_tokens": 256,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();

    // Responses shape, not Chat shape leaking through.
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(
        body["output"][0]["content"][0]["text"],
        "fn main() {} // compiles"
    );
    assert_eq!(body["output_text"], "fn main() {} // compiles");
    // Responses spells usage differently; Chat's spelling here would read as a free call.
    assert!(
        body["usage"]["input_tokens"].as_u64().unwrap() > 0,
        "{}",
        body["usage"]
    );
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() > 0);

    // And it went through the real ladder: haiku failed the gate, sonnet served.
    let traces = wait_for_traces(&db, 1).await;
    assert_eq!(traces[0].attempts.len(), 2, "same gate, same escalation");
    assert_eq!(traces[0].attempts[0].verdict, Verdict::Fail);
    assert_eq!(traces[0].final_.served_rung, Some(1));
    assert!(
        verify_chain(&traces, GENESIS_HASH).is_ok(),
        "receipt chain intact"
    );
}

/// A Responses request that cannot be gated must still reach the **Responses** upstream endpoint.
///
/// Caught in review: the fallback relayed a Responses-shaped body to `/v1/chat/completions`, which
/// the upstream rejects — breaking the exact client class the endpoint exists to serve.
///
/// Streaming no longer makes a request un-gateable (it is buffered and re-streamed on enforce
/// routes), so this uses `previous_response_id` — stateful threading the translation cannot carry —
/// to force the streaming passthrough fallback.
#[tokio::test]
async fn a_responses_request_that_cannot_be_gated_still_reaches_the_responses_endpoint() {
    let upstream = spawn_upstream().await;
    let (proxy, _db) = spawn_proxy_with(&upstream, &["anthropic/claude-haiku-4-5"], "", "").await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .header("x-api-key", "byok-test")
        .json(&json!({
            "model": "gpt-5.5", "input": "hi", "stream": true,
            "previous_response_id": "resp_prev",
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(
        text.contains("streamed via responses"),
        "relay must hit /v1/responses upstream, not /v1/chat/completions: {text}"
    );
    assert!(
        !text.contains("answer via openai"),
        "landing in the chat handler means the wrong upstream path: {text}"
    );
}

/// The receipt must describe the request the CLIENT made, not the translated one it was routed as.
#[tokio::test]
async fn a_responses_receipt_names_the_api_the_client_actually_called() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy(
        &upstream,
        &["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"],
        "",
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .header("x-api-key", "byok-test")
        .json(&json!({ "model": "claude-haiku-4-5", "input": "write a hello world" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let traces = wait_for_traces(&db, 1).await;
    assert_eq!(
        traces[0].request.api, "openai.responses",
        "an audit record that misnames the API cannot be reconciled against the client's logs"
    );
}

/// The verified cache, end to end over real HTTP.
///
/// This is the wiring test, and it exists because a module with passing unit tests and no caller is
/// exactly the defect this release already had to fix once. The assertions that matter are not
/// "the cache works" but "the second request never reached the provider, and its receipt names the
/// decision that proved the answer".
#[tokio::test]
async fn a_verified_answer_is_replayed_with_the_receipt_that_proved_it() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy_with(
        &upstream,
        &["anthropic/claude-sonnet-5"],
        "",
        "[escalation.verified_cache]\nttl_secs = 300\n",
    )
    .await;

    let client = reqwest::Client::new();
    let send = || {
        let (client, proxy) = (client.clone(), proxy.clone());
        async move {
            client
                .post(format!("{proxy}/v1/messages"))
                .header("x-api-key", "byok-test")
                .json(&json!({
                    "model": "claude-sonnet-5",
                    "max_tokens": 256,
                    "messages": [{ "role": "user", "content": "write a hello world" }],
                }))
                .send()
                .await
                .unwrap()
        }
    };

    // Turn 1: runs the ladder, passes the gate, and is cached.
    let first = send().await;
    assert_eq!(first.status(), 200);
    let first_body: Value = first.json().await.unwrap();
    assert_eq!(first_body["content"][0]["text"], "fn main() {} // compiles");

    let one = wait_for_traces(&db, 1).await;
    assert_eq!(one[0].final_.served_from, ServedFrom::Attempt);
    assert!(one[0].final_.cache_source.is_none());

    // Turn 2: identical prompt. Must be replayed, not re-routed.
    let second = send().await;
    assert_eq!(second.status(), 200);
    let second_body: Value = second.json().await.unwrap();
    assert_eq!(
        second_body["content"][0]["text"], "fn main() {} // compiles",
        "the replay must be the same answer"
    );

    let both = wait_for_traces(&db, 2).await;
    let hit = both
        .iter()
        .find(|t| t.final_.served_from == ServedFrom::Cache)
        .expect("second request should have been served from cache");

    // No model was called and no gate ran today — synthesizing an attempt here would claim a
    // verification that did not happen and inflate every rate computed over receipts.
    assert!(hit.attempts.is_empty(), "{:?}", hit.attempts);
    // What makes the serve defensible: the receipt NAMES the decision whose gate passed.
    assert_eq!(
        hit.final_.cache_source,
        Some(one[0].trace_id),
        "a hit must link to the decision that proved it, not merely say 'cached'"
    );
    assert_eq!(hit.final_.total_cost_usd, 0.0, "a replay spends nothing");
    assert!(
        hit.final_.savings_usd > 0.0,
        "and the saving is the avoided call"
    );

    // The chain still verifies with a cache receipt in it.
    assert!(verify_chain(&both, GENESIS_HASH).is_ok());
}

/// A `/v1/feedback` Fail must stop the cache replaying the answer it disproved.
///
/// Found in review: the insert-time deferred check can never see a verdict that arrives later, so
/// on its own it is a no-op in production. This exercises the half that actually runs.
#[tokio::test]
async fn a_failing_outcome_stops_the_cache_replaying_that_answer() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy_with(
        &upstream,
        &["anthropic/claude-sonnet-5"],
        "",
        "[escalation.verified_cache]\nttl_secs = 600\n",
    )
    .await;

    let client = reqwest::Client::new();
    let ask = || {
        let (client, proxy) = (client.clone(), proxy.clone());
        async move {
            client
                .post(format!("{proxy}/v1/messages"))
                .header("x-api-key", "byok-test")
                .json(&json!({
                    "model": "claude-sonnet-5",
                    "max_tokens": 256,
                    "messages": [{ "role": "user", "content": "write a hello world" }],
                }))
                .send()
                .await
                .unwrap()
        }
    };

    assert_eq!(ask().await.status(), 200);
    let first = wait_for_traces(&db, 1).await;
    let proven = first[0].trace_id;

    // Confirm it IS being cached before retracting — otherwise the assertion below could pass
    // simply because caching never worked.
    assert_eq!(ask().await.status(), 200);
    let two = wait_for_traces(&db, 2).await;
    assert!(
        two.iter()
            .any(|t| t.final_.served_from == ServedFrom::Cache),
        "precondition: the answer must be cached before a retraction can matter"
    );

    // CI ran the real tests and the answer was wrong after all.
    //
    // Reported against the REPLAY's trace id, not the original decision's — which is what a real
    // caller does, because the replay is the response it received. A retraction keyed only on the
    // reported id would match nothing here (a hit is logged under a new trace id), and the
    // disproven answer would keep serving until TTL. Found in review.
    let replay_id = two
        .iter()
        .find(|t| t.final_.served_from == ServedFrom::Cache)
        .map(|t| t.trace_id)
        .expect("a cache replay");
    assert_ne!(
        replay_id, proven,
        "a replay is logged under its own trace id"
    );
    let fb = client
        .post(format!("{proxy}/v1/feedback"))
        .header("x-api-key", "byok-test")
        .json(&json!({
            "trace_id": replay_id.to_string(),
            "gate_id": "tests",
            "verdict": "fail",
            "reporter": "ci",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(fb.status(), 202, "feedback rejected: {:?}", fb.text().await);

    // The next identical request must run the ladder again, not replay the disproven answer.
    assert_eq!(ask().await.status(), 200);
    let all = wait_for_traces(&db, 4).await;
    let latest = all.iter().max_by_key(|t| t.ts).expect("a latest trace");
    assert_ne!(
        latest.final_.served_from,
        ServedFrom::Cache,
        "an answer whose proof was retracted must not keep being replayed"
    );
}

/// The receipt records the guarantee the answer was served under. With a fixed
/// `serve_threshold` configured (no e-process, no ACI), the stored, hash-chained trace must carry a
/// `fixed` certificate at exactly that threshold — read back from the real store, and the chain
/// must still verify with it inside.
#[tokio::test]
async fn receipt_carries_the_risk_certificate_it_was_served_under() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy_with(
        &upstream,
        &["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"],
        "",
        "[escalation]\nserve_threshold = 0.5\n",
    )
    .await;
    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/messages"))
        .header("x-api-key", "byok-test")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 256,
            "messages": [{ "role": "user", "content": "write a hello world" }],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let traces = wait_for_traces(&db, 1).await;
    let risk = traces[0].risk.as_ref().expect("certificate on the receipt");
    assert_eq!(risk.source, firstpass_core::trace::ThresholdSource::Fixed);
    assert!((risk.threshold - 0.5).abs() < 1e-12);
    assert_eq!(risk.alpha, None, "a fixed threshold claims no guarantee");
    verify_chain(&store::load_all_traces(&db).unwrap(), GENESIS_HASH).unwrap();
}

/// Strict parser for an OpenAI Responses SSE body: every frame is `event: <type>` + `data: <json>`
/// whose `type` matches, every type is one this test knows, `sequence_number` strictly increases,
/// the first event is `response.created`, and the stream ends in exactly one terminal event.
/// Keepalive comments (`: ...`) are skipped as any EventSource parser would.
fn parse_responses_sse(body: &str) -> Vec<Value> {
    const KNOWN: &[&str] = &[
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
        "response.failed",
    ];
    let mut events: Vec<Value> = Vec::new();
    for frame in body.split("\n\n").filter(|f| !f.trim().is_empty()) {
        if frame.starts_with(':') {
            continue;
        }
        let mut lines = frame.lines();
        let ev = lines.next().unwrap().strip_prefix("event: ").unwrap();
        let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
        assert!(lines.next().is_none(), "one data line per frame: {frame}");
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["type"], ev, "event line and payload type agree");
        assert!(KNOWN.contains(&ev), "unknown event type {ev}");
        let seq = v["sequence_number"].as_u64().unwrap();
        if let Some(prev) = events.last() {
            assert!(seq > prev["sequence_number"].as_u64().unwrap(), "{seq}");
        }
        events.push(v);
    }
    assert_eq!(events.first().unwrap()["type"], "response.created");
    let terminal = |e: &Value| {
        matches!(
            e["type"].as_str(),
            Some("response.completed" | "response.failed")
        )
    };
    assert!(
        terminal(events.last().unwrap()),
        "stream ends in a terminal event"
    );
    assert_eq!(events.iter().filter(|e| terminal(e)).count(), 1);
    events
}

/// Codex's case, end to end: a `stream: true` Responses request on an enforce route. The cheap
/// rung returns empty text and fails `non-empty`; the next (Anthropic) rung answers with text plus
/// a `tool_use`, passes, and is re-emitted as a Responses event stream whose deltas reassemble
/// exactly the served answer. Before this change the request bypassed enforce entirely (relayed
/// un-gated to the upstream's /v1/responses) and no receipt recorded an escalation.
#[tokio::test]
async fn streaming_responses_request_is_gated_and_restreamed() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy(
        &upstream,
        &["anthropic/claude-haiku-4-5", "anthropic/claude-tools-5"],
        "",
    )
    .await;
    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .header("authorization", "Bearer byok-test")
        .json(&json!({
            "model": "gpt-5.5",
            "stream": true,
            "input": [{ "role": "user", "content": "weather in Paris?" }],
            "tools": [{ "type": "function", "name": "get_weather", "parameters": {
                "type": "object", "required": ["city"],
                "properties": { "city": { "type": "string" } } } }],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );
    let events = parse_responses_sse(&resp.text().await.unwrap());

    let text: String = events
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    let args: String = events
        .iter()
        .filter(|e| e["type"] == "response.function_call_arguments.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(text, "Checking the weather.");
    assert_eq!(
        serde_json::from_str::<Value>(&args).unwrap(),
        json!({ "city": "Paris" })
    );

    let done = &events.last().unwrap()["response"];
    assert_eq!(events.last().unwrap()["type"], "response.completed");
    assert_eq!(done["status"], "completed");
    assert_eq!(done["model"], "anthropic/claude-tools-5");
    assert_eq!(done["output"][0]["content"][0]["text"], text);
    assert_eq!(done["output"][1]["type"], "function_call");
    assert_eq!(done["output"][1]["call_id"], "toolu_1");
    assert_eq!(done["output"][1]["name"], "get_weather");
    assert_eq!(done["output"][1]["arguments"], args);
    assert_eq!(
        done["usage"],
        json!({ "input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200 })
    );

    let traces = wait_for_traces(&db, 1).await;
    assert_eq!(traces.len(), 1);
    let trace = &traces[0];
    assert_eq!(trace.request.api, "openai.responses");
    assert_eq!(
        trace.attempts.len(),
        2,
        "haiku failed the gate, tools rung served"
    );
    assert_eq!(trace.attempts[0].verdict, Verdict::Fail);
    assert_eq!(trace.attempts[1].verdict, Verdict::Pass);
    assert_eq!(trace.final_.served_rung, Some(1));
    verify_chain(&traces, GENESIS_HASH).unwrap();
    let _ = std::fs::remove_file(&db);
}

/// A streaming Responses request whose ladder fails outright ends in `response.failed`, the
/// terminal event a Responses client recognises — not an Anthropic-shaped `error` frame it ignores.
#[tokio::test]
async fn streaming_responses_ladder_failure_ends_in_response_failed() {
    let upstream = spawn_upstream().await;
    let (proxy, db) = spawn_proxy(&upstream, &["anthropic/claude-fail5xx"], "").await;
    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .header("authorization", "Bearer byok-test")
        .json(&json!({ "model": "gpt-5.5", "stream": true, "input": "hi" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let events = parse_responses_sse(&resp.text().await.unwrap());
    let failed = events.last().unwrap();
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(failed["response"]["status"], "failed");
    assert_eq!(failed["response"]["id"], events[0]["response"]["id"]);
    assert!(failed["response"]["error"]["message"].is_string());
    let _ = std::fs::remove_file(&db);
}

// ─── Gemini inbound tests ─────────────────────────────────────────────────────

/// Parse the Gemini SSE stream format (`data: {json}\n\n`) and collect the data payloads.
fn parse_gemini_sse(text: &str) -> Vec<Value> {
    text.split("\n\n")
        .filter_map(|chunk| {
            let line = chunk.trim();
            if line.is_empty() {
                return None;
            }
            line.strip_prefix("data: ")
                .and_then(|s| serde_json::from_str(s).ok())
        })
        .collect()
}

/// `POST /v1beta/models/{model}:generateContent` observe mode: the proxy relays the request
/// byte-for-byte to `FIRSTPASS_UPSTREAM_GEMINI` and returns the upstream response unchanged.
///
/// Mutation check: if `FIRSTPASS_UPSTREAM_GEMINI` is not wired into the config the request
/// goes to the default `https://generativelanguage.googleapis.com` and fails to connect.
#[tokio::test]
async fn gemini_observe_non_streaming() {
    let upstream = spawn_upstream().await;
    // Observe-only proxy: no routing config, just FIRSTPASS_UPSTREAM_GEMINI.
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();
    let (traces, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers: ProviderRegistry::from_config(&[], &upstream, &upstream),
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:generateContent"
        ))
        .header("x-goog-api-key", "AIza-test-key")
        .json(&json!({
            "contents": [{ "role": "user", "parts": [{ "text": "hello" }] }],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    // The upstream returned the Gemini-shaped response; the proxy relayed it byte-for-byte.
    assert_eq!(
        body["candidates"][0]["content"]["parts"][0]["text"],
        "gemini says hi"
    );
    assert_eq!(body["usageMetadata"]["promptTokenCount"], 5);
    // mutant: if FIRSTPASS_UPSTREAM_GEMINI is not wired, the proxy would try
    // generativelanguage.googleapis.com and return an error, making this assertion fail.

    let traces = wait_for_traces(&db_path, 1).await;
    assert_eq!(traces.len(), 1, "observe trace written");
    verify_chain(&traces, GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db_path);
}

/// Observe streaming: `streamGenerateContent` is relayed as an SSE stream and the proxy does NOT
/// buffer the response — chunks flow directly from upstream to the client.
#[tokio::test]
async fn gemini_observe_streaming() {
    let upstream = spawn_upstream().await;
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();
    let (traces, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers: ProviderRegistry::from_config(&[], &upstream, &upstream),
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    // streamGenerateContent → proxy relays SSE.
    // The fake upstream returns a plain JSON response; the proxy pipes it through as-is.
    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        ))
        .header("x-goog-api-key", "AIza-test-key")
        .json(&json!({
            "contents": [{ "role": "user", "parts": [{ "text": "stream me" }] }],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    // The upstream returned JSON; just verify we got a non-empty response and no error.
    let body_text = resp.text().await.unwrap();
    assert!(
        !body_text.is_empty(),
        "streaming response must not be empty"
    );
    // The body contains the Gemini-shaped candidates from the upstream.
    assert!(
        body_text.contains("gemini says hi"),
        "upstream text must be present in stream"
    );
    // mutant: if the proxy tried to call :generateContent instead of :streamGenerateContent,
    // the upstream would still return the same response body, but the path assertion in
    // future tests would fail (path matters for real Google API).

    let _ = std::fs::remove_file(&db_path);
}

/// Enforce mode with Gemini inbound: a cheap rung fails the gate, the ladder escalates to the next
/// rung, and the caller receives a Gemini-shaped `generateContent` response (not Chat Completions).
/// The audit trail records two attempts and the receipt chain verifies.
///
/// Mutation check: if `response_from_chat` returns Chat shape instead of Gemini shape, the
/// `candidates[0]` assertion fails.
#[tokio::test]
async fn gemini_enforce_escalates_and_returns_gemini_shape() {
    let upstream = spawn_upstream().await;
    // Ladder: gemini-haiku fails the non-empty gate → escalate to gemini-flash which passes.
    let providers =
        format!("[[provider]]\nid = \"gemini\"\ndialect = \"gemini\"\nbase_url = \"{upstream}\"\n");
    let (proxy, db) = spawn_proxy(
        &upstream,
        &["gemini/gemini-haiku", "gemini/gemini-2.0-flash"],
        &providers,
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:generateContent"
        ))
        .header("x-goog-api-key", "AIza-test-key")
        .json(&json!({
            "contents": [{ "role": "user", "parts": [{ "text": "write code" }] }],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let served: Value = resp.json().await.unwrap();

    // The response must be Gemini-shaped (candidates, not Chat "choices").
    assert!(
        served.get("candidates").is_some(),
        "response must be Gemini-shaped (candidates), got: {served}"
    );
    assert!(
        served.get("choices").is_none(),
        "response must NOT be Chat-shaped (choices), got: {served}"
    );
    // The escalated rung (flash) returned non-empty text.
    assert_eq!(
        served["candidates"][0]["content"]["parts"][0]["text"],
        "gemini says hi"
    );
    assert!(
        served.get("usageMetadata").is_some(),
        "usageMetadata present"
    );
    // mutant: if response_from_chat is not called, served["candidates"] would be null.

    let traces = wait_for_traces(&db, 1).await;
    let trace = &traces[0];
    assert_eq!(trace.attempts.len(), 2, "haiku failed, flash passed");
    assert_eq!(
        trace.attempts[0].verdict,
        Verdict::Fail,
        "haiku fails non-empty gate"
    );
    assert_eq!(
        trace.attempts[1].verdict,
        Verdict::Pass,
        "flash passes non-empty gate"
    );
    assert_eq!(trace.final_.served_rung, Some(1));
    assert_eq!(trace.request.api, "gemini.generate_content");
    // mutant: if the api label is wrong, this assertion fails.
    verify_chain(&traces, GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db);
}

/// Streaming enforce: same escalation path, but the caller requests
/// `:streamGenerateContent`. The response is an SSE stream of Gemini data frames.
#[tokio::test]
async fn gemini_enforce_streaming_returns_gemini_sse() {
    let upstream = spawn_upstream().await;
    let providers =
        format!("[[provider]]\nid = \"gemini\"\ndialect = \"gemini\"\nbase_url = \"{upstream}\"\n");
    let (proxy, db) = spawn_proxy(
        &upstream,
        &["gemini/gemini-haiku", "gemini/gemini-2.0-flash"],
        &providers,
    )
    .await;

    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        ))
        .header("x-goog-api-key", "AIza-test-key")
        .json(&json!({
            "contents": [{ "role": "user", "parts": [{ "text": "stream code" }] }],
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body_text = resp.text().await.unwrap();
    let frames = parse_gemini_sse(&body_text);
    assert!(!frames.is_empty(), "must have at least one SSE frame");

    // The final frame must be Gemini-shaped (candidates), not an error frame.
    let last = frames.last().unwrap();
    assert!(
        last.get("candidates").is_some(),
        "final SSE frame must be Gemini-shaped, got: {last}"
    );
    assert!(
        last.get("error").is_none(),
        "final SSE frame must not be an error, got: {last}"
    );
    assert_eq!(
        last["candidates"][0]["content"]["parts"][0]["text"],
        "gemini says hi"
    );
    // mutant: if sse_from_response is bypassed, the body would be Chat JSON with "choices", not SSE.

    let traces = wait_for_traces(&db, 1).await;
    let trace = &traces[0];
    assert_eq!(
        trace.attempts.len(),
        2,
        "haiku failed, flash served via SSE"
    );
    assert_eq!(trace.final_.served_rung, Some(1));
    verify_chain(&traces, GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db);
}

/// `?alt=sse` (and any other query params) must reach the upstream verbatim.
/// Without this, `streamGenerateContent` returns a JSON array instead of SSE, breaking
/// Gemini CLI streaming in observe mode.
///
/// Mutation check: comment out the `?{scrubbed_query}` append in `observe_passthrough_gemini`
/// → upstream receives no query, the recorded path has no `alt=sse`, assertion fails.
#[tokio::test]
async fn gemini_query_alt_sse_forwarded() {
    let (upstream, log) = spawn_recording_gemini_upstream().await;
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();
    let (traces_store, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers: ProviderRegistry::from_config(&[], &upstream, &upstream),
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces: traces_store,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    // Send with ?alt=sse as real Gemini CLI does for streamGenerateContent.
    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
        ))
        .header("x-goog-api-key", "AIza-test")
        .json(&json!({ "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Give the upstream a moment to record the request.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let recorded = log.0.lock().unwrap().clone();
    assert!(
        !recorded.is_empty(),
        "upstream must have received a request"
    );
    let (path_and_query, _key) = &recorded[0];
    assert!(
        path_and_query.contains("alt=sse"),
        "upstream must have received ?alt=sse; got path_and_query={path_and_query}"
    );

    let _ = std::fs::remove_file(&db_path);
}

/// `?key=` is an alternate way Gemini clients pass the API key. The proxy must:
/// 1. Forward the key value to the upstream as `x-goog-api-key` header (not in the URL).
/// 2. NOT include the key value in the upstream URL (which could appear in server-side logs).
/// 3. NOT store the key value in the receipt / trace.
///
/// Mutation check: comment out the `key_from_query` injection in `gemini_generate_content`
/// → upstream receives no `x-goog-api-key` header (assertion 1 fails), or if `key=` is left
/// in the scrubbed query the URL would carry it and assertion 2 fails.
#[tokio::test]
async fn gemini_query_key_not_in_receipt() {
    let (upstream, log) = spawn_recording_gemini_upstream().await;
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();
    let (traces_store, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers: ProviderRegistry::from_config(&[], &upstream, &upstream),
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces: traces_store,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    // Send with ?key= (no x-goog-api-key header), using a recognisable sentinel value.
    let sentinel = "dummy-secret-q";
    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:generateContent?key={sentinel}"
        ))
        .json(&json!({ "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Give the upstream a moment to record the request.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let recorded = log.0.lock().unwrap().clone();
    assert!(
        !recorded.is_empty(),
        "upstream must have received a request"
    );
    let (path_and_query, api_key_header) = &recorded[0];

    // 1. Key must have been forwarded as x-goog-api-key header.
    assert_eq!(
        api_key_header.as_deref(),
        Some(sentinel),
        "upstream must receive the key as x-goog-api-key header"
    );

    // 2. Upstream URL must NOT contain the key value.
    assert!(
        !path_and_query.contains(sentinel),
        "upstream URL must not contain the key; got path_and_query={path_and_query}"
    );

    // 3. Stored receipt must NOT contain the key value.
    let traces = wait_for_traces(&db_path, 1).await;
    assert!(!traces.is_empty(), "trace must be written");
    let trace_json = serde_json::to_string(&traces[0]).unwrap();
    assert!(
        !trace_json.contains(sentinel),
        "receipt must not contain the key value; trace_json snippet: {}",
        &trace_json[..trace_json.len().min(400)]
    );

    let _ = std::fs::remove_file(&db_path);
}

/// `countTokens` is a metadata endpoint: the proxy relays it transparently but writes NO receipt.
///
/// Mutation check: restore `spawn_trace` in the countTokens branch → a trace is written and
/// the `is_empty()` assertion fails.
#[tokio::test]
async fn gemini_count_tokens_writes_no_receipt() {
    let (upstream, log) = spawn_recording_gemini_upstream().await;
    drop(log); // not needed here
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();
    let (traces_store, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers: ProviderRegistry::from_config(&[], &upstream, &upstream),
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces: traces_store,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:countTokens"
        ))
        .header("x-goog-api-key", "AIza-test")
        .json(&json!({ "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "countTokens must relay 200");

    // Wait briefly; any spawned trace would land within this window.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let traces = store::load_all_traces(&db_path).unwrap_or_default();
    assert!(
        traces.is_empty(),
        "countTokens must not write a receipt; got {} traces",
        traces.len()
    );

    let _ = std::fs::remove_file(&db_path);
}

/// A Gemini observe-mode trace records `api = "gemini.generate_content"`, the model from the
/// URL path, and token counts from `usageMetadata` (not `usage.prompt_tokens`).
///
/// Mutation check: revert `response_usage` Gemini branch → in_tokens/out_tokens stay at 0.
/// Mutation check: revert `base_trace` api param → api stays `"anthropic.messages"`.
/// Mutation check: remove `model_hint` → model stays `"unknown"`.
#[tokio::test]
async fn gemini_observe_receipt_has_correct_api_model_tokens() {
    let (upstream, _log) = spawn_recording_gemini_upstream().await;
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();
    let (traces_store, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers: ProviderRegistry::from_config(&[], &upstream, &upstream),
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces: traces_store,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    // The recording upstream returns usageMetadata: promptTokenCount=5, candidatesTokenCount=3.
    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.5-pro:generateContent"
        ))
        .header("x-goog-api-key", "AIza-test")
        .json(&json!({ "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let traces = wait_for_traces(&db_path, 1).await;
    assert!(!traces.is_empty(), "a trace must be written");
    let t = &traces[0];

    assert_eq!(
        t.request.api, "gemini.generate_content",
        "api must be gemini.generate_content, got {:?}",
        t.request.api
    );
    assert_eq!(
        t.attempts[0].model, "gemini-2.5-pro",
        "model must come from URL path, got {:?}",
        t.attempts[0].model
    );
    assert_eq!(
        t.attempts[0].in_tokens, 5,
        "in_tokens from usageMetadata.promptTokenCount"
    );
    assert_eq!(
        t.attempts[0].out_tokens, 3,
        "out_tokens from usageMetadata.candidatesTokenCount"
    );

    // Priced from the built-in `google/` table, not a silent $0 receipt.
    assert!(
        t.attempts[0].cost_usd > 0.0,
        "a priced Gemini model must not record $0, got {}",
        t.attempts[0].cost_usd
    );

    let _ = std::fs::remove_file(&db_path);
}

/// A `?key=` value containing percent-encoded characters (e.g. `%3D` for `=`) must be decoded
/// before it is forwarded as the `x-goog-api-key` header.
///
/// Mutation check: remove `percent_decode_ascii` call → header carries `"AIza%3Dkey"` (raw) →
/// assert_eq fails.
#[tokio::test]
async fn gemini_query_key_percent_decoded() {
    let (upstream, log) = spawn_recording_gemini_upstream().await;
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        _ => None,
    })
    .unwrap();
    let (traces_store, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers: ProviderRegistry::from_config(&[], &upstream, &upstream),
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces: traces_store,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    // `AIza%3Dkey` encodes `=`; the upstream header must receive the decoded `AIza=key`.
    let resp = reqwest::Client::new()
        .post(format!(
            "{proxy}/v1beta/models/gemini-2.0-flash:generateContent?key=AIza%3Dkey"
        ))
        .json(&json!({ "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let recorded = log.0.lock().unwrap().clone();
    assert!(
        !recorded.is_empty(),
        "upstream must have received a request"
    );
    let (_, api_key_header) = &recorded[0];
    assert_eq!(
        api_key_header.as_deref(),
        Some("AIza=key"),
        "percent-encoded key must be decoded; got {:?}",
        api_key_header
    );

    let _ = std::fs::remove_file(&db_path);
}

/// Different Gemini tool-bearing requests get different session IDs (session collision fix).
/// When `body_session` was called with the raw Gemini body, all tool requests hashed to the
/// same `conv:` prefix because Gemini bodies have no `messages` field — only `contents`.
///
/// Mutation check: revert to `body_session(&parsed, ...)` → both sessions are identical →
/// `assert_ne!` fails.
#[tokio::test]
async fn gemini_enforce_different_tool_requests_get_different_sessions() {
    // We need an enforce route with a Gemini ladder. Use the recording upstream as both Gemini
    // and the ladder upstream (it just echoes back a valid Gemini response).
    let (upstream, _log) = spawn_recording_gemini_upstream().await;
    let db_path = std::env::temp_dir().join(format!("firstpass-e2e-{}.db", uuid::Uuid::now_v7()));
    let db_str = db_path.to_string_lossy().into_owned();
    let upstream_c = upstream.clone();

    // Minimal Gemini provider config so `gemini/gemini-2.0-flash` resolves.
    let providers_toml =
        format!("[[provider]]\nid = \"gemini\"\ndialect = \"gemini\"\nbase_url = \"{upstream}\"\n");
    let price_toml = "[[price]]\nmodel = \"gemini/gemini-2.0-flash\"\ninput_per_mtok = 0.1\noutput_per_mtok = 0.4\n";
    let routing_toml = format!(
        "{providers_toml}{price_toml}\n[[route]]\nmatch = {{}}\nmode = \"enforce\"\nladder = [\"gemini/gemini-2.0-flash\"]\ngates = [\"non-empty\"]\n"
    );
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_GEMINI" => Some(upstream_c.clone()),
        "FIRSTPASS_DB" => Some(db_str.clone()),
        "FIRSTPASS_CONFIG_TOML" => Some(routing_toml.clone()),
        _ => None,
    })
    .unwrap();
    let provider_defs = config
        .routing
        .as_ref()
        .map(|r| r.providers.as_slice())
        .unwrap_or_default();
    let providers = ProviderRegistry::from_config(provider_defs, &upstream, &upstream);
    let (traces_store, _writer) = store::open(&db_path).unwrap();
    let promoter = firstpass_proxy::proxy::build_promoter_async(&config)
        .await
        .unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config)
        .await
        .unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers,
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        paired_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        experiments: std::sync::Arc::default(),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces: traces_store,
        adaptive: None,
        eprocess: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
        prior: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state).unwrap()).await.unwrap();
    });
    let proxy = format!("http://{addr}");

    // Two different tool-bearing requests: different system prompts + different tool names.
    let req_a = json!({
        "systemInstruction": { "parts": [{ "text": "You are agent A" }] },
        "contents": [{ "role": "user", "parts": [{ "text": "query A" }] }],
        "tools": [{ "functionDeclarations": [{ "name": "tool_a", "description": "a" }] }],
    });
    let req_b = json!({
        "systemInstruction": { "parts": [{ "text": "You are agent B" }] },
        "contents": [{ "role": "user", "parts": [{ "text": "query B" }] }],
        "tools": [{ "functionDeclarations": [{ "name": "tool_b", "description": "b" }] }],
    });

    let client = reqwest::Client::new();
    let url = format!("{proxy}/v1beta/models/gemini-2.0-flash:generateContent");

    for body in [&req_a, &req_b] {
        client
            .post(&url)
            .header("x-goog-api-key", "AIza-test")
            .json(body)
            .send()
            .await
            .unwrap();
    }

    let traces = wait_for_traces(&db_path, 2).await;
    assert_eq!(traces.len(), 2, "expected 2 traces, got {}", traces.len());

    let sess_a = &traces[0].session_id;
    let sess_b = &traces[1].session_id;
    assert_ne!(
        sess_a, sess_b,
        "different tool-bearing requests must get different session IDs (session collision fix)"
    );

    let _ = std::fs::remove_file(&db_path);
}
