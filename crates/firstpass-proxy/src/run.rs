//! Server bootstrap shared by the `firstpass` and `firstpass-proxy` binaries: build state from
//! config, open the trace store, and serve until Ctrl-C. Keeping this in the lib means both the
//! unified CLI (`firstpass up`) and the bare proxy binary start the server the exact same way.

use std::sync::Arc;

use crate::gate::GateHealthRegistry;
use crate::provider::ProviderRegistry;
use crate::{AppState, ProxyConfig, app, store};

/// Initialize the global tracing subscriber from `RUST_LOG` (default `info`). Called by the
/// binaries, not the library internals.
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
}

/// Register a default error budget for every gate named across enforce routes: auto-disable a gate
/// whose abstain rate exceeds 25% over its last 50 runs (SPEC §7.2).
#[must_use]
pub fn build_gate_health(config: &ProxyConfig) -> GateHealthRegistry {
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
    gate_health
}

/// Open the trace store, build [`AppState`], and serve the HTTP proxy until Ctrl-C.
///
/// # Errors
/// Returns any error from opening the store, building the HTTP client, binding the listener, or
/// serving.
pub async fn serve(config: ProxyConfig) -> Result<(), Box<dyn std::error::Error>> {
    let (traces, writer) = store::open(&config.db_path)?;
    let bind = config.bind.clone();
    let providers = ProviderRegistry::new(&config.upstream_anthropic, &config.upstream_openai);
    let gate_health = build_gate_health(&config);

    let state = AppState {
        config: Arc::new(config),
        // Observe passthrough may stream SSE, so only bound the CONNECT phase here — a total or
        // read timeout would sever a long-lived stream. (The enforce providers, which never stream
        // through the adapter, carry a full request timeout — see `ProviderRegistry::new`.)
        http: reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()?,
        providers,
        gate_health: Arc::new(gate_health),
        traces,
    };

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(%bind, "firstpass listening");
    tracing::info!("offboard: unset ANTHROPIC_BASE_URL");

    axum::serve(listener, app(state)?)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Dropping `traces` (via `state`, already out of scope) closes the channel; wait for the
    // writer to flush and exit before the process ends.
    drop(writer);
    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::warn!(%err, "failed to install Ctrl-C handler; shutting down anyway");
    }
}
