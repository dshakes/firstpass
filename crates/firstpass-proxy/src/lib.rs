//! # firstpass-proxy
//!
//! An HTTP proxy that sits as a drop-in `base_url` in front of Anthropic/OpenAI-compatible
//! providers. In **observe** mode it forwards each request unchanged and records a
//! tamper-evident audit trace asynchronously (zero added latency). In **enforce** mode it runs
//! the escalation engine — cheapest model first, gate the output, escalate one rung on failure,
//! serve the first output that passes — with cross-provider failover (SPEC §7.1, §7.1a, §9.1).
//!
//! - [`config`] — [`ProxyConfig`], loaded from the environment.
//! - [`store`] — the background SQLite trace writer.
//! - [`calibrate`] — recalibrate the conformal serving threshold from deferred feedback.
//! - [`upstream`] — BYOK passthrough to the upstream provider (observe mode).
//! - [`provider`] — normalized multi-provider model access (Anthropic, OpenAI).
//! - [`gate`] — runtime verification gates (Batch 3 inline set).
//! - [`router`] — the enforce-mode escalation engine.
//! - [`proxy`] — axum routing, observe passthrough, and enforce dispatch.
//! - [`error`] — structured, no-leak error responses.
//! - [`key_custody`] — per-tenant AES-256-GCM envelope key custody (ADR 0004 §D5, pre-review).
//! - [`metrics`] — Prometheus recorder install + `GET /metrics`.
//! - [`cli`] — `firstpass doctor` / `trace` logic (validate a setup, read the store).
//! - [`mcp`] — minimal MCP stdio server so an agent can read its traces and submit feedback.
//! - [`tenant_auth`] — experimental multi-tenant API-key auth (ADR 0004 §D1, default-off).
//! - [`run`] — shared server bootstrap for the `firstpass` and `firstpass-proxy` binaries.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod calibrate;
pub mod cli;
pub mod config;
pub mod consistency;
pub mod error;
pub mod gate;
pub mod judge;
pub mod key_custody;
pub mod mcp;
pub mod metrics;
pub mod onboard;
pub mod ope;
pub mod provider;
pub mod proxy;
pub mod router;
pub mod run;
pub mod store;
pub mod subprocess;
pub mod tenant_auth;
pub mod upstream;

pub use config::ProxyConfig;
pub use error::ProxyError;
pub use proxy::{AppState, app};
pub use store::{TraceSender, load_all_traces};
pub use tenant_auth::{TenantId, TenantKeys};
