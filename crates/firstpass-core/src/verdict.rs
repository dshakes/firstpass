//! Verdicts and scores — the atomic unit of ground truth (SPEC §7.1, §9).
//!
//! A gate emits a [`Verdict`] (`pass`/`fail`/`abstain`) plus an optional [`Score`]
//! in `[0, 1]`, its cost, its latency, and optional evidence. These types are the
//! wire/audit contract: their serde field names appear verbatim in the trace (§9.1),
//! so renaming one is a breaking change.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// A gate's judgement of an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// The output cleared the bar — safe to serve.
    Pass,
    /// The output failed the bar — escalate (or, for deferred gates, a learning signal).
    Fail,
    /// The gate could not decide (provider/gate error, timeout). Policy resolves it
    /// fail-open or fail-closed (§7.2); an abstain is never silently a pass.
    Abstain,
}

impl Verdict {
    /// True only for [`Verdict::Pass`].
    #[must_use]
    pub const fn is_pass(self) -> bool {
        matches!(self, Verdict::Pass)
    }

    /// Its lowercase wire form (`"pass"`/`"fail"`/`"abstain"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::Abstain => "abstain",
        }
    }
}

/// A confidence score, validated to the closed unit interval `[0, 1]`.
///
/// Validation happens at the trust boundary: construction and deserialization both
/// reject `NaN`, infinities, and out-of-range values, so nothing downstream has to
/// re-check. Serializes transparently as an `f64`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Score(f64);

impl Score {
    /// Construct a score, rejecting non-finite or out-of-range values.
    ///
    /// # Errors
    /// Returns [`Error::InvalidScore`] if `v` is not finite or falls outside `[0, 1]`.
    pub fn new(v: f64) -> Result<Self> {
        if v.is_finite() && (0.0..=1.0).contains(&v) {
            Ok(Self(v))
        } else {
            Err(Error::InvalidScore(v))
        }
    }

    /// Construct a score by clamping to `[0, 1]`. A non-finite input is treated as `0.0`.
    #[must_use]
    pub fn clamped(v: f64) -> Self {
        if v.is_finite() {
            Self(v.clamp(0.0, 1.0))
        } else {
            Self(0.0)
        }
    }

    /// The underlying value in `[0, 1]`.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Score {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = f64::deserialize(d)?;
        Score::new(v).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The outcome of running one gate against one attempt (a row in `attempts[].gates`, §9.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateResult {
    /// Stable, versioned gate identity, e.g. `"judge-diff@v2"`.
    pub gate_id: String,
    /// The gate's judgement.
    pub verdict: Verdict,
    /// Confidence in `[0, 1]`. Absent when the gate abstained without a score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<Score>,
    /// Marginal USD spent running this gate (0.0 for deterministic gates like a test run).
    pub cost_usd: f64,
    /// Wall-clock milliseconds the gate took.
    pub ms: u64,
    /// Machine-readable reason, primarily for abstains (e.g. `"provider_error"`, `"timeout"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Opaque pointer to an evidence blob (judge rationale) stored separately with its
    /// own retention clock (§9.2) — never the rationale inline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_ref: Option<String>,
}

/// Well-known abstain reasons (§7.2). Reasons are stored as open strings on
/// [`GateResult::reason`]; these constants keep producers consistent.
pub mod reason {
    /// The provider serving the attempt (or the gate's own model) errored.
    pub const PROVIDER_ERROR: &str = "provider_error";
    /// The gate exceeded its time budget.
    pub const TIMEOUT: &str = "timeout";
    /// The gate process crashed or returned a non-zero exit.
    pub const GATE_CRASH: &str = "gate_crash";
    /// The gate was auto-disabled for exceeding its error budget (§7.2).
    pub const GATE_DISABLED: &str = "gate_disabled";
}

impl GateResult {
    /// Build a deterministic (zero-cost) gate result.
    #[must_use]
    pub fn deterministic(gate_id: impl Into<String>, verdict: Verdict, ms: u64) -> Self {
        Self {
            gate_id: gate_id.into(),
            verdict,
            score: None,
            cost_usd: 0.0,
            ms,
            reason: None,
            evidence_ref: None,
        }
    }

    /// Build an abstain result carrying a machine-readable reason.
    #[must_use]
    pub fn abstain(gate_id: impl Into<String>, reason: impl Into<String>, ms: u64) -> Self {
        Self {
            gate_id: gate_id.into(),
            verdict: Verdict::Abstain,
            score: None,
            cost_usd: 0.0,
            ms,
            reason: Some(reason.into()),
            evidence_ref: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_validates_range_and_finiteness() {
        assert!(Score::new(0.0).is_ok());
        assert!(Score::new(1.0).is_ok());
        assert!(Score::new(0.31).is_ok());
        assert!(Score::new(-0.01).is_err());
        assert!(Score::new(1.01).is_err());
        assert!(Score::new(f64::NAN).is_err());
        assert!(Score::new(f64::INFINITY).is_err());
    }

    #[test]
    fn score_clamps() {
        assert_eq!(Score::clamped(2.0).value(), 1.0);
        assert_eq!(Score::clamped(-5.0).value(), 0.0);
        assert_eq!(Score::clamped(f64::NAN).value(), 0.0);
        assert_eq!(Score::clamped(0.5).value(), 0.5);
    }

    #[test]
    fn score_deserialize_rejects_out_of_range() {
        assert!(serde_json::from_str::<Score>("0.5").is_ok());
        assert!(serde_json::from_str::<Score>("1.5").is_err());
    }

    #[test]
    fn verdict_wire_form_is_lowercase() {
        assert_eq!(serde_json::to_string(&Verdict::Pass).unwrap(), "\"pass\"");
        assert_eq!(
            serde_json::to_string(&Verdict::Abstain).unwrap(),
            "\"abstain\""
        );
        assert_eq!(
            serde_json::from_str::<Verdict>("\"fail\"").unwrap(),
            Verdict::Fail
        );
    }

    #[test]
    fn gate_result_omits_absent_optionals() {
        let g = GateResult::deterministic("cargo-test", Verdict::Pass, 3100);
        let j = serde_json::to_string(&g).unwrap();
        assert!(!j.contains("score"), "absent score should be omitted: {j}");
        assert!(!j.contains("reason"));
        assert!(j.contains("\"verdict\":\"pass\""));
    }
}
