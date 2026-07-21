//! The audit trace (SPEC §9.1) — the tamper-evident record of one routing decision.
//!
//! A trace captures *why* a request went where it did and *what it cost*: the features it was
//! routed on, every attempt and its gate verdicts, the final served outcome, and the savings
//! versus always calling the top rung. Each record links to the previous via [`Trace::prev_hash`]
//! (see [`crate::hashchain`]), so the log is append-only and any tampering is detectable.
//!
//! **The serde field names in this module are the wire/audit contract.** External auditors
//! parse them and re-derive the hash chain; renaming one is a breaking change that requires a
//! schema/version bump, never a silent edit.

use crate::Result;
use crate::config::Mode;
use crate::features::Features;
use crate::hashchain::{Chained, record_hash};
use crate::verdict::{GateResult, Score, Verdict};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single routing decision, start to finish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    /// Time-ordered unique id (UUIDv7).
    pub trace_id: Uuid,
    /// Hash of the previous record in this chain (or [`crate::hashchain::GENESIS_HASH`]).
    pub prev_hash: String,
    /// Tenant this trace belongs to.
    pub tenant_id: String,
    /// Session (e.g. an agent run) this request is part of.
    pub session_id: String,
    /// When the decision was made.
    pub ts: Timestamp,
    /// Serving mode in effect.
    pub mode: Mode,
    /// Which policy produced this decision.
    pub policy: PolicyRef,
    /// The request that was routed.
    pub request: RequestInfo,
    /// Every attempt made, cheapest rung first.
    pub attempts: Vec<Attempt>,
    /// Verdicts that arrived after serving (deferred gates, feedback API). Attach over time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deferred: Vec<DeferredVerdict>,
    /// The final served outcome and its economics.
    #[serde(rename = "final")]
    pub final_: FinalOutcome,
}

/// Reference to the policy that produced a decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyRef {
    /// Policy identity/version, e.g. `"static@v0"` or `"bandit@v3"`.
    pub id: String,
    /// Whether this decision was a deliberate exploration draw (bounded, §, only in enforce).
    #[serde(default)]
    pub explore: bool,
    /// The probability the logging policy assigned to the start rung it chose — the
    /// Horvitz-Thompson denominator for IPS / SNIPS off-policy evaluation.
    ///
    /// `None` for deterministic (non-exploring) policies. `skip_serializing_if` keeps old
    /// trace bytes byte-identical when `None`, preserving hash-chain compatibility with
    /// existing logs. Only populated when `[escalation.exploration]` is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propensity: Option<f64>,
    /// Resolved routing-mode profile when it was explicitly set (header / route / global env).
    /// `None` means `Balanced` was in effect (the default, no override active). Absent from
    /// the JSON when `None` → byte-identical serialization for all existing traces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_profile: Option<String>,
}

/// The routed request, described without its raw content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestInfo {
    /// Wire API the caller used, e.g. `"anthropic.messages"` / `"openai.chat"`.
    pub api: String,
    /// Salted hash of the prompt — never the prompt text itself.
    pub prompt_hash: String,
    /// The versioned feature vector routing keyed on.
    pub features: Features,
}

/// One attempt at a rung: the model call plus the gates run against its output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    /// Ladder rung (0 = cheapest).
    pub rung: u32,
    /// Model called, `provider/model`.
    pub model: String,
    /// Provider segment (denormalized for cheap querying).
    pub provider: String,
    /// Input tokens consumed.
    pub in_tokens: u64,
    /// Output tokens produced.
    pub out_tokens: u64,
    /// USD cost of this model call (excludes gate cost).
    pub cost_usd: f64,
    /// Wall-clock latency of the model call.
    pub latency_ms: u64,
    /// Gate verdicts for this attempt's output.
    pub gates: Vec<GateResult>,
    /// The attempt's overall verdict (the aggregate that drove escalate-or-serve).
    pub verdict: Verdict,
}

/// A verdict that arrived after the response was served (deferred gate or downstream feedback).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeferredVerdict {
    /// Gate/source identity, e.g. `"tests"` or `"feedback:ci"`.
    pub gate_id: String,
    /// The late verdict.
    pub verdict: Verdict,
    /// Optional score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<Score>,
    /// When it arrived.
    pub reported_at: Timestamp,
    /// Who reported it (a deferred gate, or a feedback-API caller identity).
    pub reporter: String,
}

/// Where the served response came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServedFrom {
    /// A passing attempt (the normal path).
    Attempt,
    /// The best attempt seen, served because budget/ladder was exhausted without a pass.
    BestAttempt,
    /// No output served; a structured error was returned.
    Error,
}

/// The final outcome and economics of a decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalOutcome {
    /// Rung whose output was served (`None` if `served_from = error`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_rung: Option<u32>,
    /// Provenance of the served response.
    pub served_from: ServedFrom,
    /// Total USD spent (all model calls + all gates).
    pub total_cost_usd: f64,
    /// USD portion spent on gates alone.
    pub gate_cost_usd: f64,
    /// Total wall-clock latency the caller experienced.
    pub total_latency_ms: u64,
    /// Number of rung escalations taken.
    pub escalations: u32,
    /// What always calling the top rung would have cost (§9.1 counterfactual).
    pub counterfactual_baseline_usd: f64,
    /// `counterfactual_baseline_usd - total_cost_usd` — the savings this decision proves.
    pub savings_usd: f64,
}

impl Chained for Trace {
    fn prev_hash(&self) -> &str {
        &self.prev_hash
    }
}

impl Trace {
    /// This record's own hash (SHA-256 over its canonical JSON) — the value the *next*
    /// record stores as its `prev_hash`.
    ///
    /// # Errors
    /// Returns [`crate::Error::Json`] if the trace cannot be serialized.
    pub fn hash(&self) -> Result<String> {
        record_hash(self)
    }

    /// Compute `savings_usd = baseline - total` and store it, keeping the field consistent
    /// with the two it is derived from. Returns the savings.
    pub fn recompute_savings(&mut self) -> f64 {
        let s = self.final_.counterfactual_baseline_usd - self.final_.total_cost_usd;
        self.final_.savings_usd = s;
        s
    }
}

impl std::fmt::Display for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trace {} rung {:?} served_from {:?} cost ${:.4} saved ${:.4}",
            self.trace_id,
            self.final_.served_rung,
            self.final_.served_from,
            self.final_.total_cost_usd,
            self.final_.savings_usd,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{Features, TaskKind};
    use crate::hashchain::{GENESIS_HASH, verify_chain};
    use crate::verdict::Verdict;

    fn sample_trace(prev_hash: &str, id: u128) -> Trace {
        let mut t = Trace {
            trace_id: Uuid::from_u128(id),
            prev_hash: prev_hash.to_owned(),
            tenant_id: "acme".into(),
            session_id: "agent-run-4417".into(),
            ts: "2026-07-08T15:04:05Z".parse().unwrap(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "static@v0".into(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".into(),
                prompt_hash: "deadbeef".into(),
                features: Features::new(TaskKind::CodeEdit),
            },
            attempts: vec![
                Attempt {
                    rung: 0,
                    model: "anthropic/claude-haiku-4-5".into(),
                    provider: "anthropic".into(),
                    in_tokens: 2000,
                    out_tokens: 700,
                    cost_usd: 0.0007,
                    latency_ms: 900,
                    gates: vec![GateResult::deterministic("cargo-test", Verdict::Fail, 3100)],
                    verdict: Verdict::Fail,
                },
                Attempt {
                    rung: 1,
                    model: "anthropic/claude-sonnet-5".into(),
                    provider: "anthropic".into(),
                    in_tokens: 2000,
                    out_tokens: 800,
                    cost_usd: 0.0121,
                    latency_ms: 1200,
                    gates: vec![GateResult::deterministic("cargo-test", Verdict::Pass, 2950)],
                    verdict: Verdict::Pass,
                },
            ],
            deferred: vec![],
            final_: FinalOutcome {
                served_rung: Some(1),
                served_from: ServedFrom::Attempt,
                total_cost_usd: 0.0128,
                gate_cost_usd: 0.0,
                total_latency_ms: 2100,
                escalations: 1,
                counterfactual_baseline_usd: 0.0630,
                savings_usd: 0.0,
            },
        };
        t.recompute_savings();
        t
    }

    #[test]
    fn wire_field_names_are_the_contract() {
        let t = sample_trace(GENESIS_HASH, 1);
        let j = serde_json::to_string(&t).unwrap();
        assert!(j.contains("\"prev_hash\":"));
        assert!(
            j.contains("\"final\":"),
            "the outcome key must serialize as `final`"
        );
        assert!(j.contains("\"served_from\":\"attempt\""));
        assert!(j.contains("\"verdict\":\"fail\""));
        assert!(j.contains("\"verdict\":\"pass\""));
        assert!(j.contains("\"counterfactual_baseline_usd\":"));
    }

    #[test]
    fn savings_is_baseline_minus_total() {
        let t = sample_trace(GENESIS_HASH, 1);
        assert!((t.final_.savings_usd - (0.0630 - 0.0128)).abs() < 1e-12);
    }

    #[test]
    fn traces_chain_and_verify() {
        let t0 = sample_trace(GENESIS_HASH, 1);
        let t1 = sample_trace(&t0.hash().unwrap(), 2);
        let chain = [t0, t1];
        assert!(verify_chain(&chain, GENESIS_HASH).is_ok());
    }

    #[test]
    fn tampering_a_served_trace_is_detectable() {
        let t0 = sample_trace(GENESIS_HASH, 1);
        let t1 = sample_trace(&t0.hash().unwrap(), 2);
        let mut chain = [t0, t1];
        chain[0].final_.total_cost_usd = 0.0; // forge a cheaper decision after the fact
        assert!(verify_chain(&chain, GENESIS_HASH).is_err());
    }

    #[test]
    fn roundtrips_through_json() {
        let t = sample_trace(GENESIS_HASH, 7);
        let j = serde_json::to_string(&t).unwrap();
        let back: Trace = serde_json::from_str(&j).unwrap();
        assert_eq!(t, back);
    }

    // ── propensity backward-compat ────────────────────────────────────────────

    /// Traces where propensity is None must serialize byte-identically to pre-field traces:
    /// the field must be absent from the JSON, not serialized as `null`.
    #[test]
    fn propensity_none_absent_from_json() {
        let pr = PolicyRef {
            id: "static@v0".into(),
            explore: false,
            propensity: None,
            mode_profile: None,
        };
        let j = serde_json::to_string(&pr).unwrap();
        assert!(
            !j.contains("propensity"),
            "propensity=None must be omitted (skip_serializing_if): {j}"
        );
    }

    /// Old JSON without a `propensity` field must deserialize to `propensity: None`
    /// (via `#[serde(default)]`), keeping old traces readable without schema migration.
    #[test]
    fn old_trace_without_propensity_deserializes_to_none() {
        let old_json = r#"{"id":"static@v0","explore":false}"#;
        let pr: PolicyRef = serde_json::from_str(old_json).unwrap();
        assert_eq!(pr.propensity, None);
    }

    /// New JSON with a propensity value round-trips correctly.
    #[test]
    fn propensity_some_roundtrips() {
        let pr = PolicyRef {
            id: "bandit@v1+eps".into(),
            explore: true,
            propensity: Some(0.3),
            mode_profile: None,
        };
        let j = serde_json::to_string(&pr).unwrap();
        assert!(
            j.contains("\"propensity\":0.3"),
            "expected propensity in: {j}"
        );
        let back: PolicyRef = serde_json::from_str(&j).unwrap();
        assert_eq!(back, pr);
    }

    // ── mode_profile backward-compat ─────────────────────────────────────────

    /// `mode_profile = None` must not appear in the serialized JSON — byte-identical for
    /// existing traces (Balanced is the default and must be invisible).
    #[test]
    fn mode_profile_none_absent_from_json() {
        let pr = PolicyRef {
            id: "static@v0".into(),
            explore: false,
            propensity: None,
            mode_profile: None,
        };
        let j = serde_json::to_string(&pr).unwrap();
        assert!(
            !j.contains("mode_profile"),
            "mode_profile=None must be omitted (skip_serializing_if): {j}"
        );
    }

    /// Old JSON without a `mode_profile` field deserializes to `mode_profile: None`.
    #[test]
    fn old_trace_without_mode_profile_deserializes_to_none() {
        let old_json = r#"{"id":"static@v0","explore":false}"#;
        let pr: PolicyRef = serde_json::from_str(old_json).unwrap();
        assert_eq!(pr.mode_profile, None);
    }

    /// `mode_profile = Some(...)` round-trips correctly.
    #[test]
    fn mode_profile_some_roundtrips() {
        let pr = PolicyRef {
            id: "static@v0".into(),
            explore: false,
            propensity: None,
            mode_profile: Some("quality".into()),
        };
        let j = serde_json::to_string(&pr).unwrap();
        assert!(
            j.contains("\"mode_profile\":\"quality\""),
            "expected mode_profile in: {j}"
        );
        let back: PolicyRef = serde_json::from_str(&j).unwrap();
        assert_eq!(back, pr);
    }
}
