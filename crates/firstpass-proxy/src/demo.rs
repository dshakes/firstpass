//! The no-keys demo: `firstpass demo`.
//!
//! A self-contained, real-HTTP demonstration of the whole Firstpass loop. It stands up a local
//! server that speaks the Anthropic wire protocol (the cheap model returns a weak answer, the next
//! rung a good one), runs one enforce-mode decision through the real proxy, and prints the audit
//! receipt — which model was tried, which gate caught what, what it cost, and what it saved versus
//! always calling the top tier. Then it reports a downstream outcome via the feedback API and
//! shows it attach without breaking the tamper-evident chain.
//!
//! Everything here is real code over real HTTP; only the upstream is local, so no API keys are
//! needed. Point the same proxy at real providers (base URLs + BYOK) and it behaves identically.
//!
//! This lives in the library, not in `examples/`, so it ships inside the `firstpass` binary and is
//! therefore reachable from every install channel — `uvx firstpass demo` works without a Rust
//! toolchain. `examples/demo.rs` is a shim over this same code so the cargo invocation in older
//! docs keeps working.

use std::sync::Arc;
use std::time::Duration;

use axum::routing::post;
use axum::{Json, Router};
use bytes::Bytes;
use firstpass_core::{GENESIS_HASH, verify_chain};
use serde_json::{Value, json};

use crate::provider::ProviderRegistry;
use crate::proxy::AppState;
use crate::{ProxyConfig, app, store};

/// Anything that can go wrong in the demo. It is a demonstration, so a failure is reported plainly
/// rather than panicking — the binary is the same one operators run in production.
type Fail = Box<dyn std::error::Error>;

/// Run the demo end to end. Prints to stdout; leaves no state behind (the trace DB is a temp file).
///
/// # Errors
/// If the local upstream or proxy cannot bind, the request fails, or no trace is recorded.
pub async fn run() -> Result<(), Fail> {
    // 1. A faithful local Anthropic upstream: haiku answers weakly (empty), sonnet answers well.
    let upstream = spawn_upstream().await?;

    // 2. The real proxy, enforce route haiku → sonnet → opus, gated on non-empty output.
    let db = std::env::temp_dir().join(format!("firstpass-demo-{}.db", uuid::Uuid::now_v7()));
    let (proxy, _writer) = spawn_proxy(&upstream, &db).await?;

    println!("\n\x1b[1mFirstpass demo\x1b[0m — routing one request through a real enforce ladder");
    println!("no API keys, no config: the upstream is local, everything else is the real proxy\n");

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
        .await?
        .json()
        .await?;

    println!("served output : {}", served["content"][0]["text"]);
    println!("served model  : {}\n", served["model"]);

    // 4. Read back the audit trace and print the receipt.
    let trace = wait_for_trace(&db).await.ok_or("no trace recorded")?;
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
    let pct = if f.counterfactual_baseline_usd > 0.0 {
        f.savings_usd / f.counterfactual_baseline_usd * 100.0
    } else {
        0.0
    };
    println!(
        "  \x1b[32mSAVED     ${:.4}   ({pct:.0}% cheaper at proven quality)\x1b[0m",
        f.savings_usd
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
        .json(&json!({
            "trace_id": trace_id, "gate_id": "ci-tests",
            "verdict": "pass", "reporter": "github-actions",
        }))
        .send()
        .await?;
    println!(
        "feedback POST /v1/feedback → {} (downstream outcome recorded)",
        fb.status()
    );

    let view = store::load_trace_view(&db, "default", &trace_id)?.ok_or("trace view missing")?;
    let reporter = view
        .deferred
        .first()
        .map_or("(none)", |d| d.reporter.as_str());
    println!(
        "deferred verdicts on trace: {} ({reporter} reported it)",
        view.deferred.len()
    );
    let all = store::load_all_traces(&db)?;
    println!(
        "audit chain after feedback : {}\n",
        if verify_chain(&all, GENESIS_HASH).is_ok() {
            "\x1b[32mstill verified ✓\x1b[0m — the sealed record never changed"
        } else {
            "BROKEN"
        }
    );
    println!("next: `firstpass onboard` wires your own agent through this in observe mode.\n");

    let _ = std::fs::remove_file(&db);
    Ok(())
}

/// A local stand-in for the Anthropic API: haiku returns an empty answer (so the gate fails and the
/// ladder escalates), anything above it returns real code.
pub(crate) async fn spawn_upstream() -> Result<String, Fail> {
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
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    Ok(format!("http://{addr}"))
}

/// Stand up the real proxy against the local upstream. Returns its base URL and the trace-writer
/// task handle, kept alive alongside the server for the life of the demo.
pub(crate) async fn spawn_proxy(
    upstream: &str,
    db: &std::path::Path,
) -> Result<(String, tokio::task::JoinHandle<()>), Fail> {
    let routing = "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\", \"anthropic/claude-sonnet-5\", \"anthropic/claude-opus-4-8\"]\ngates = [\"non-empty\"]\n";
    let (up, dbs) = (upstream.to_owned(), db.to_string_lossy().into_owned());
    let config = ProxyConfig::from_lookup(move |k| match k {
        "FIRSTPASS_UPSTREAM_ANTHROPIC" => Some(up.clone()),
        "FIRSTPASS_MODE" => Some("enforce".to_owned()),
        "FIRSTPASS_CONFIG_TOML" => Some(routing.to_owned()),
        "FIRSTPASS_DB" => Some(dbs.clone()),
        _ => None,
    })?;
    let providers = ProviderRegistry::new(&config.upstream_anthropic, &config.upstream_openai);
    let (traces, writer) = store::open(db)?;
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
        providers,
        gate_health: Arc::new(crate::gate::GateHealthRegistry::new()),
        shadow_ledger: Arc::new(crate::shadow::ShadowLedger::new()),
        guardrails: Arc::new(crate::guard::GuardrailRegistry::new()),
        traces,
        adaptive: None,
        bandit: None,
        predictor: None,
        tenant_rate_limiter: None,
        spill: None,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let router = app(state)?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    Ok((format!("http://{addr}"), writer))
}

/// Traces are written off the hot path, so poll briefly for the first one to land.
pub(crate) async fn wait_for_trace(db: &std::path::Path) -> Option<firstpass_core::Trace> {
    for _ in 0..150 {
        if let Ok(t) = store::load_all_traces(db)
            && let Some(first) = t.into_iter().next()
        {
            return Some(first);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    None
}
