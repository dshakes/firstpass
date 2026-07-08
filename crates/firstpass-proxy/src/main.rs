//! Firstpass proxy binary: loads config, opens the trace store, and serves the observe-mode
//! HTTP proxy until Ctrl-C.

use std::sync::Arc;

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
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::builder().build()?,
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
