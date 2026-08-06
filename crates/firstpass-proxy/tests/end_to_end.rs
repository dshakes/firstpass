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
async fn gemini_generate(body: Bytes) -> Response {
    // The incoming Gemini body has no top-level `model` (it's in the URL), so just echo a good answer.
    let _ = &body;
    Json(json!({
        "candidates": [{ "content": { "role": "model", "parts": [{ "text": "gemini says hi" }] } }],
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
    let promoter = firstpass_proxy::proxy::build_promoter(&config).unwrap();
    let verified_cache = firstpass_proxy::proxy::build_verified_cache(&config);
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers,
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        shadow_ledger: std::sync::Arc::new(firstpass_proxy::shadow::ShadowLedger::new()),
        guardrails: std::sync::Arc::new(firstpass_proxy::guard::GuardrailRegistry::new()),
        traces,
        adaptive: None,
        bandit: None,
        promoter,
        verified_cache,
        predictor: None,
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
/// the upstream rejects — breaking the exact client class the endpoint exists to serve. Observe
/// mode is the reachable fallback here; streaming takes the same path.
#[tokio::test]
async fn a_responses_request_that_cannot_be_gated_still_reaches_the_responses_endpoint() {
    let upstream = spawn_upstream().await;
    // No enforce route → every request falls through to observe passthrough.
    let (proxy, _db) = spawn_proxy_with(&upstream, &["anthropic/claude-haiku-4-5"], "", "").await;

    let resp = reqwest::Client::new()
        .post(format!("{proxy}/v1/responses"))
        .header("x-api-key", "byok-test")
        .json(&json!({ "model": "gpt-5.5", "input": "hi", "stream": true }))
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
