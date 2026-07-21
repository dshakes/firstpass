//! # firstpass-core
//!
//! The Firstpass domain contract — pure, with **no I/O** (no filesystem, network, clock, or
//! env access). Everything here is deterministic so the audit hash chain and feature
//! extraction are reproducible and testable in isolation; all I/O lives in `firstpass-proxy`.
//!
//! This crate is the versioned thing every other component (gates, bandit, CLI, auditors)
//! depends on. The serde field names on [`Trace`], [`Config`], [`Verdict`], and friends are
//! the wire/audit contract — see each module for the "don't rename silently" rule.
//!
//! ## Modules
//! - [`verdict`] — [`Verdict`], validated [`Score`], [`GateResult`] (the unit of ground truth).
//! - [`features`] — the versioned, privacy-preserving request [`Features`] vector.
//! - [`trace`] — the [`Trace`] audit record (SPEC §9.1).
//! - [`hashchain`] — tamper-evident, auditor-re-derivable hashing.
//! - [`config`] — declarative routing [`Config`] (SPEC §8.4).
//! - [`cost`] — model pricing and the counterfactual baseline.
//! - [`conformal`] — split-conformal risk control on the gate threshold (SPEC §10.1).
//! - [`ltt`] — Learn-then-Test threshold calibration (RCPS, Angelopoulos et al. 2021).
//! - [`error`] — the crate [`Error`] type.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![doc(html_root_url = "https://docs.rs/firstpass-core")]

pub mod config;
pub mod conformal;
pub mod cost;
pub mod error;
pub mod features;
pub mod hashchain;
pub mod ltt;
pub mod trace;
pub mod verdict;

pub use config::{
    AbstainPolicy, AuthScheme, BanditAlgorithm, BanditConfig, Budget, Config, ConsistencyDef,
    Dialect, Escalation, GateDef, JudgeDef, Mode, ModePreset, ModelRef, OnExhausted, PriceDef,
    ProbeConfig, ProviderDef, Route, RoutingMode, SessionPromotion,
};
pub use conformal::{ConformalResult, calibrate, served_failure_rate};
pub use cost::{ModelPrice, PriceTable};
pub use error::{Error, Result};
pub use features::{FEATURE_VERSION, Features, TaskKind};
pub use hashchain::{Chained, GENESIS_HASH, canonical_json, record_hash, verify_chain};
pub use ltt::{LttDiagnostic, LttResult};
pub use trace::{
    Attempt, DeferredVerdict, FinalOutcome, PolicyRef, ProbeRegime, ProbeSignal, RequestInfo,
    ServedFrom, Trace,
};
pub use verdict::{GateResult, Score, Verdict};
