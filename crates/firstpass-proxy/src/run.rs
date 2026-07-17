//! Server bootstrap shared by the `firstpass` and `firstpass-proxy` binaries: build state from
//! config, open the trace store, and serve until Ctrl-C. Keeping this in the lib means both the
//! unified CLI (`firstpass up`) and the bare proxy binary start the server the exact same way.

use std::sync::Arc;

use crate::bandit::{ContextBucket, StartRungBandit};
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
    // Build the provider registry from any `[[provider]]` entries in the routing config (Groq,
    // Together, Ollama, …), on top of the built-in anthropic/openai defaults. No config => defaults.
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
    let gate_health = build_gate_health(&config);
    // Online adaptive conformal (opt-in): seed the live threshold from the fixed one (or 0.5) and
    // let /v1/feedback track it. Absent config => None => fixed-threshold behavior, byte-identical.
    let adaptive = config
        .routing
        .as_ref()
        .and_then(|r| r.escalation.adaptive.as_ref())
        .map(|a| {
            let init = config
                .routing
                .as_ref()
                .and_then(|r| r.escalation.serve_threshold)
                .unwrap_or(0.5);
            Arc::new(std::sync::Mutex::new(
                firstpass_core::conformal::AdaptiveConformal::new(a.alpha, a.gamma, init),
            ))
        });
    let tenant_rate_limiter = crate::proxy::build_tenant_rate_limiter(&config);

    // UCB1 start-rung bandit (opt-in): warm-start from stored traces so learning survives
    // restarts. Forgiving: an unreadable or absent store simply yields an empty bandit.
    // ponytail: operator-wide load (load_all_traces) is correct here — the bandit is also
    // operator-wide, keyed by context bucket, not by tenant.
    let bandit = config
        .routing
        .as_ref()
        .and_then(|r| r.escalation.bandit.as_ref())
        .map(|bc| {
            let mut b = StartRungBandit::new(bc.min_observations, bc.exploration);
            if let Ok(traces_history) = store::load_all_traces(&config.db_path) {
                for trace in &traces_history {
                    let ctx = ContextBucket::from_features(&trace.request.features);
                    b.feed_trace_attempts(&ctx, &trace.attempts);
                }
                tracing::info!(
                    n = traces_history.len(),
                    "bandit warm-started from trace store"
                );
            }
            Arc::new(std::sync::Mutex::new(b))
        });

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
        adaptive,
        bandit,
        tenant_rate_limiter,
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
