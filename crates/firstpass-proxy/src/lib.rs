//! # firstpass-proxy
//!
//! An HTTP proxy that sits as a drop-in `base_url` in front of Anthropic. In **observe
//! mode** (the only mode this build serves — gating is M2) it forwards each request to the
//! upstream provider unchanged, returns the response unchanged, and asynchronously records
//! a tamper-evident audit trace to SQLite (SPEC §7.1, §7.1a, §9.1).
//!
//! - [`config`] — [`ProxyConfig`], loaded from the environment.
//! - [`store`] — the background SQLite trace writer.
//! - [`upstream`] — BYOK passthrough to the upstream provider.
//! - [`proxy`] — axum routing and observe-mode trace construction.
//! - [`error`] — structured, no-leak error responses.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod config;
pub mod error;
pub mod proxy;
pub mod store;
pub mod upstream;

pub use config::ProxyConfig;
pub use error::ProxyError;
pub use proxy::{AppState, app};
pub use store::{TraceSender, load_all_traces};
