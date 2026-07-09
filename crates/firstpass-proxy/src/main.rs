//! Firstpass proxy binary: loads config, opens the trace store, and serves the observe- and
//! enforce-mode HTTP proxy until Ctrl-C.

use std::sync::Arc;

use firstpass_proxy::gate::GateHealthRegistry;
use firstpass_proxy::provider::ProviderRegistry;
use firstpass_proxy::{AppState, ProxyConfig, app, store};

/// Usage text for `--help`. The proxy is configured entirely through the environment
/// (12-factor), so `--help` doubles as the config reference — there are no subcommands.
const HELP: &str = "\
firstpass-proxy — drop-in, Anthropic-compatible LLM proxy that routes to the cheapest
model that provably passes your gate, and writes a tamper-evident receipt for every call.

USAGE:
    firstpass-proxy [--help] [--version]
    <configured via environment variables; then point your agent's ANTHROPIC_BASE_URL at it>

ENVIRONMENT:
    FIRSTPASS_MODE                observe (default) | enforce
    FIRSTPASS_BIND               listen address           [default 127.0.0.1:8080]
    FIRSTPASS_CONFIG             path to firstpass.toml (routes, ladders, gates)
    FIRSTPASS_DB                 trace store path         [default firstpass.db]
    FIRSTPASS_UPSTREAM_ANTHROPIC upstream base URL        [default https://api.anthropic.com]
    FIRSTPASS_UPSTREAM_OPENAI    upstream base URL        [default https://api.openai.com]
    FIRSTPASS_TENANT             tenant id for the trace  [default default]
    FIRSTPASS_PROMPT_SALT        salt for prompt hashing in traces
    RUST_LOG                     tracing filter           [default info]

QUICKSTART:
    firstpass-proxy                                   # observe mode, no behavior change
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8080   # point your agent at it
    # offboard anytime:  unset ANTHROPIC_BASE_URL

DOCS:  https://dshakes.github.io/firstpass  ·  SPEC: https://github.com/dshakes/firstpass/blob/main/SPEC.md";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Zero-dependency arg handling: the real interface is env vars, so we only field
    // the two flags every CLI is expected to answer.
    if let Some(flag) = std::env::args().nth(1) {
        match flag.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            "--version" | "-V" => {
                println!("firstpass-proxy {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => {
                eprintln!("firstpass-proxy: unrecognized argument `{other}`\n\n{HELP}");
                std::process::exit(2);
            }
        }
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = ProxyConfig::from_env()?;
    let (traces, writer) = store::open(&config.db_path)?;

    let bind = config.bind.clone();
    let providers = ProviderRegistry::new(&config.upstream_anthropic, &config.upstream_openai);

    // Register a default error budget for every gate named across enforce routes: auto-disable
    // a gate whose abstain rate exceeds 25% over its last 50 runs (SPEC §7.2).
    let mut gate_health = GateHealthRegistry::new();
    if let Some(routing) = config.routing.as_ref() {
        let mut seen = std::collections::HashSet::new();
        for route in &routing.routes {
            for gate in route.gates.iter().chain(&route.deferred_gates) {
                if seen.insert(gate.clone()) {
                    gate_health = gate_health.with_budget(gate.clone(), 50, 0.25);
                }
            }
        }
    }

    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::builder().build()?,
        providers,
        gate_health: Arc::new(gate_health),
        traces,
    };

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(%bind, "firstpass-proxy listening");
    tracing::info!("offboard: unset ANTHROPIC_BASE_URL");

    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Dropping `traces` (via `state`, already gone out of scope with the server) closes the
    // channel; wait for the writer to flush and exit before the process ends.
    drop(writer);
    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::warn!(%err, "failed to install Ctrl-C handler; shutting down anyway");
    }
}
