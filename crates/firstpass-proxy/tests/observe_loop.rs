//! Offline integration test (SPEC §7.1/§9.1): no live network. A mock upstream stands in
//! for Anthropic; the test proves the proxy forwards responses byte-for-byte and records a
//! valid, tamper-evident trace chain asynchronously, without the caller waiting on it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::routing::post;
use firstpass_core::{GENESIS_HASH, verify_chain};
use firstpass_proxy::proxy::AppState;
use firstpass_proxy::{ProxyConfig, app, load_all_traces, store};
use serde_json::{Value, json};

const CANNED_RESPONSE: &str = r#"{"id":"msg_1","model":"claude-haiku-4-5","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":1200,"output_tokens":300}}"#;

/// Start a mock Anthropic upstream on an ephemeral port; returns its base URL.
async fn spawn_mock_upstream() -> String {
    let router = axum::Router::new().route(
        "/v1/messages",
        post(|| async {
            let body: Value = serde_json::from_str(CANNED_RESPONSE).unwrap();
            Json(body)
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// Start the Firstpass proxy pointed at `upstream`, backed by a fresh temp SQLite file.
/// Returns the proxy's base URL and the temp DB path (caller cleans it up).
async fn spawn_proxy(upstream: &str) -> (String, std::path::PathBuf) {
    let db_path = std::env::temp_dir().join(format!(
        "firstpass-observe-loop-{}.db",
        uuid::Uuid::now_v7()
    ));
    let upstream = upstream.to_owned();
    let config = ProxyConfig::from_lookup(move |key| match key {
        "FIRSTPASS_UPSTREAM_ANTHROPIC" => Some(upstream.clone()),
        _ => None,
    })
    .unwrap();

    let (traces, _writer) = store::open(&db_path).unwrap();
    let providers = firstpass_proxy::provider::ProviderRegistry::new(
        "https://api.anthropic.com",
        "https://api.openai.com",
    );
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers,
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        traces,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(state)).await.unwrap();
    });
    (format!("http://{addr}"), db_path)
}

/// Poll the trace DB until `want` rows exist or the timeout elapses.
async fn wait_for_traces(db_path: &std::path::Path, want: usize) -> Vec<firstpass_core::Trace> {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        let traces = load_all_traces(db_path).unwrap_or_default();
        if traces.len() >= want || std::time::Instant::now() >= deadline {
            return traces;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn observe_mode_forwards_unchanged_and_records_a_valid_trace() {
    let upstream = spawn_mock_upstream().await;
    let (proxy, db_path) = spawn_proxy(&upstream).await;

    let client = reqwest::Client::new();
    let request_body = json!({
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "get_weather"}],
    });

    let response = client
        .post(format!("{proxy}/v1/messages"))
        .header("x-api-key", "test")
        .json(&request_body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    let canned: Value = serde_json::from_str(CANNED_RESPONSE).unwrap();
    assert_eq!(
        body, canned,
        "proxy must return the upstream body unchanged"
    );

    let traces = wait_for_traces(&db_path, 1).await;
    assert_eq!(traces.len(), 1, "expected exactly one recorded trace");

    let trace = &traces[0];
    assert_eq!(trace.request.api, "anthropic.messages");
    assert_eq!(trace.attempts.len(), 1);
    assert_eq!(trace.attempts[0].model, "claude-haiku-4-5");
    assert_eq!(trace.attempts[0].in_tokens, 1200);
    assert_eq!(trace.attempts[0].out_tokens, 300);
    assert!(trace.attempts[0].cost_usd > 0.0);

    verify_chain(&traces, GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db_path);
}

#[tokio::test]
async fn two_requests_form_a_valid_two_link_chain() {
    let upstream = spawn_mock_upstream().await;
    let (proxy, db_path) = spawn_proxy(&upstream).await;

    let client = reqwest::Client::new();
    for _ in 0..2 {
        let response = client
            .post(format!("{proxy}/v1/messages"))
            .header("x-api-key", "test")
            .json(&json!({"model": "claude-haiku-4-5", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    let traces = wait_for_traces(&db_path, 2).await;
    assert_eq!(traces.len(), 2, "expected exactly two recorded traces");
    assert_eq!(traces[1].prev_hash, traces[0].hash().unwrap());
    verify_chain(&traces, GENESIS_HASH).unwrap();

    let _ = std::fs::remove_file(&db_path);
}
