//! Firstpass proxy binary: loads config, opens the trace store, and serves the observe-mode
//! HTTP proxy until Ctrl-C.

use std::sync::Arc;

use firstpass_proxy::gate::GateHealthRegistry;
use firstpass_proxy::provider::ProviderRegistry;
use firstpass_proxy::{AppState, ProxyConfig, app, store};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
