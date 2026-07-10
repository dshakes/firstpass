//! `cargo run -p firstpass-proxy --example demo`
//!
//! A self-contained, no-keys, real-HTTP demonstration of the whole Firstpass loop. It stands up
//! a local server that speaks the Anthropic wire protocol (the cheap model returns a weak answer,
//! the next rung a good one), runs one enforce-mode decision through the real proxy, and prints
//! the audit receipt — which model was tried, which gate caught what, what it cost, and what it
//! saved versus always calling the top tier. Then it reports a downstream outcome via the feedback
//! API and shows it attach without breaking the tamper-evident chain.
//!
//! Everything here is real code over real HTTP; only the upstream is local so no API keys are
//! needed. Point the same proxy at real providers (set the base URLs + BYOK) and it behaves
//! identically.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use axum::routing::post;
use axum::{Json, Router};
use bytes::Bytes;
use firstpass_core::{GENESIS_HASH, verify_chain};
use firstpass_proxy::provider::ProviderRegistry;
use firstpass_proxy::proxy::AppState;
use firstpass_proxy::{ProxyConfig, app, store};
use serde_json::{Value, json};

#[tokio::main]
async fn main() {
    // 1. A faithful local Anthropic upstream: haiku answers weakly (empty), sonnet answers well.
    let upstream = spawn_upstream().await;

    // 2. The real proxy, enforce route haiku → sonnet → opus, gated on non-empty output.
    let db = std::env::temp_dir().join(format!("firstpass-demo-{}.db", uuid::Uuid::now_v7()));
    let proxy = spawn_proxy(&upstream, &db).await;

    println!(
        "\n\x1b[1mFirstpass demo\x1b[0m — routing one request through a real enforce ladder\n"
    );

    // 3. Send a request, exactly as a coding agent would (Anthropic wire format, BYOK header).
    let client = reqwest::Client::new();
    let served: Value = client
        .post(format!("{proxy}/v1/messages"))
        .header("x-api-key", "byok-demo")
        .json(&json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 256,
            "messages": [{ "role": "user", "content": "write a hello world in rust" }],
        }))
        .send()
        .await
        .expect("request")
        .json()
        .await
        .expect("json");

    println!("served output : {}", served["content"][0]["text"]);
    println!("served model  : {}\n", served["model"]);

    // 4. Read back the audit trace and print the receipt.
    let trace = wait_for_trace(&db).await;
    println!("\x1b[1m── audit receipt ──────────────────────────────────\x1b[0m");
    for a in &trace.attempts {
        let verdict = match a.verdict {
            firstpass_core::Verdict::Pass => "\x1b[32mPASS\x1b[0m",
            firstpass_core::Verdict::Fail => "\x1b[31mFAIL\x1b[0m",
            firstpass_core::Verdict::Abstain => "\x1b[33mABSTAIN\x1b[0m",
        };
        println!(
            "  rung {} · {:<28} · {verdict} · ${:.4}",
            a.rung, a.model, a.cost_usd
        );
    }
    let f = &trace.final_;
    println!("  ─────────────────────────────────────────────────");
    println!("  total     ${:.4}", f.total_cost_usd);
    println!(
        "  baseline  ${:.4}   (always top-tier)",
        f.counterfactual_baseline_usd
    );
    println!(
        "  \x1b[32mSAVED     ${:.4}   ({:.0}% cheaper at proven quality)\x1b[0m",
        f.savings_usd,
        if f.counterfactual_baseline_usd > 0.0 {
            f.savings_usd / f.counterfactual_baseline_usd * 100.0
        } else {
            0.0
        }
    );
    println!("  trace_id  {}", trace.trace_id);
    println!(
        "  chain     {}\n",
        if verify_chain(std::slice::from_ref(&trace), &trace.prev_hash).is_ok() {
            "verified ✓"
        } else {
            "BROKEN"
        }
    );

    // 5. The outcome loop: report that the served code actually passed CI, and show it attach.
    let trace_id = trace.trace_id.to_string();
    let fb = client
        .post(format!("{proxy}/v1/feedback"))
        .json(&json!({ "trace_id": trace_id, "gate_id": "ci-tests", "verdict": "pass", "reporter": "github-actions" }))
        .send()
        .await
        .expect("feedback");
    println!(
        "feedback POST /v1/feedback → {} (downstream outcome recorded)",
        fb.status()
    );

    let view = store::load_trace_view(&db, &trace_id)
        .expect("view")
        .expect("trace");
    println!(
        "deferred verdicts on trace: {} ({} reported it)",
        view.deferred.len(),
        view.deferred[0].reporter
    );
    let all = store::load_all_traces(&db).expect("all");
    println!(
        "audit chain after feedback : {}\n",
        if verify_chain(&all, GENESIS_HASH).is_ok() {
            "\x1b[32mstill verified ✓\x1b[0m — the sealed record never changed"
        } else {
            "BROKEN"
        }
    );

    let _ = std::fs::remove_file(&db);
}

async fn spawn_upstream() -> String {
    async fn messages(body: Bytes) -> Json<Value> {
        let model = serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|v| v.get("model").and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_default();
        let text = if model.contains("haiku") {
            ""
        } else {
            "fn main() { println!(\"hello world\"); }"
        };
        Json(json!({
            "id": "msg_demo", "type": "message", "role": "assistant", "model": model,
            "content": [{ "type": "text", "text": text }],
            "usage": { "input_tokens": 1200, "output_tokens": 220 },
        }))
    }
    let router = Router::new().route("/v1/messages", post(messages));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

async fn spawn_proxy(upstream: &str, db: &std::path::Path) -> String {
    let routing = "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\", \"anthropic/claude-sonnet-5\", \"anthropic/claude-opus-4-8\"]\ngates = [\"non-empty\"]\n";
    let (up, dbs) = (upstream.to_owned(), db.to_string_lossy().into_owned());
    let config = ProxyConfig::from_lookup(move |k| match k {
        "FIRSTPASS_UPSTREAM_ANTHROPIC" => Some(up.clone()),
        "FIRSTPASS_MODE" => Some("enforce".to_owned()),
        "FIRSTPASS_CONFIG_TOML" => Some(routing.to_owned()),
        "FIRSTPASS_DB" => Some(dbs.clone()),
        _ => None,
    })
    .unwrap();
    let providers = ProviderRegistry::new(&config.upstream_anthropic, &config.upstream_openai);
    let (traces, _writer) = store::open(db).unwrap();
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers,
        gate_health: Arc::new(firstpass_proxy::gate::GateHealthRegistry::new()),
        traces,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app(state).unwrap()).await.unwrap() });
    format!("http://{addr}")
}

async fn wait_for_trace(db: &std::path::Path) -> firstpass_core::Trace {
    for _ in 0..150 {
        if let Ok(t) = store::load_all_traces(db)
            && let Some(first) = t.into_iter().next()
        {
            return first;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("no trace recorded");
}
