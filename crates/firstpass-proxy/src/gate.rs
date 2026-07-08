//! Runtime gates for enforce mode (Batch 3 minimal set).
//!
//! A gate inspects a [`ModelResponse`] and returns a [`GateResult`] verdict. Batch 3 ships two
//! trivial, deterministic (zero-cost) inline gates so the escalation engine has something real to
//! escalate on; the full pluggable gate framework (subprocess plugins, judges, error budgets) is
//! Batch 4 / M2. Gate ids referenced in config but not yet implemented are skipped with a warning.

use crate::provider::{ModelRequest, ModelResponse};
use firstpass_core::{GateResult, Verdict};

/// A verification gate: judge a model response, cheaply and synchronously.
pub trait Gate: Send + Sync + std::fmt::Debug {
    /// Stable gate id (matches the name used in routing config).
    fn id(&self) -> &str;
    /// Evaluate the response, producing a verdict + evidence.
    fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult;
}

/// Fails an empty (whitespace-only) completion. The cheapest possible sanity gate.
#[derive(Debug, Clone, Copy)]
pub struct NonEmptyGate;

impl Gate for NonEmptyGate {
    fn id(&self) -> &str {
        "non-empty"
    }
    fn evaluate(&self, _req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let verdict = if resp.text.trim().is_empty() {
            Verdict::Fail
        } else {
            Verdict::Pass
        };
        GateResult::deterministic(self.id(), verdict, 0)
    }
}

/// Passes only if the completion parses as JSON. Useful for structured-output routes.
#[derive(Debug, Clone, Copy)]
pub struct JsonValidGate;

impl Gate for JsonValidGate {
    fn id(&self) -> &str {
        "json-valid"
    }
    fn evaluate(&self, _req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let ok = serde_json::from_str::<serde_json::Value>(resp.text.trim()).is_ok();
        let verdict = if ok { Verdict::Pass } else { Verdict::Fail };
        GateResult::deterministic(self.id(), verdict, 0)
    }
}

/// Resolve config gate names into runnable gates. Unknown names are skipped with a warning.
///
// ponytail: only the two Batch-3 inline gates exist; M2 gate names (schema, judge-diff,
// patch-applies, self-consistency, …) resolve to nothing here and become no-ops until Batch 4.
// A no-op gate list means "serve rung 0" — acceptable for the skeleton, loud via the warning.
#[must_use]
pub fn resolve_gates(names: &[String]) -> Vec<Box<dyn Gate>> {
    let mut gates: Vec<Box<dyn Gate>> = Vec::new();
    for name in names {
        match name.as_str() {
            "non-empty" => gates.push(Box::new(NonEmptyGate)),
            "json-valid" => gates.push(Box::new(JsonValidGate)),
            other => tracing::warn!(gate = %other, "unknown gate id — skipped (implemented in M2)"),
        }
    }
    gates
}

/// Aggregate per-gate verdicts into the attempt's overall verdict.
///
/// `Fail` if any gate fails; otherwise `Pass`. An empty gate set passes.
///
// ponytail: `Abstain` is treated as pass here (fail-open). Per-gate fail-open/closed policy and
// gate error budgets are Batch 4 (§7.2); until then an abstaining gate never blocks serving.
#[must_use]
pub fn aggregate(results: &[GateResult]) -> Verdict {
    if results.iter().any(|r| r.verdict == Verdict::Fail) {
        Verdict::Fail
    } else {
        Verdict::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn resp(text: &str) -> ModelResponse {
        ModelResponse {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            text: text.to_owned(),
            in_tokens: 1,
            out_tokens: 1,
            raw: Value::Null,
        }
    }

    fn req() -> ModelRequest {
        ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: None,
            messages: vec![],
            max_tokens: 16,
            tools: Value::Null,
        }
    }

    #[test]
    fn non_empty_gate() {
        assert_eq!(
            NonEmptyGate.evaluate(&req(), &resp("hi")).verdict,
            Verdict::Pass
        );
        assert_eq!(
            NonEmptyGate.evaluate(&req(), &resp("   ")).verdict,
            Verdict::Fail
        );
    }

    #[test]
    fn json_valid_gate() {
        assert_eq!(
            JsonValidGate
                .evaluate(&req(), &resp(r#"{"ok":true}"#))
                .verdict,
            Verdict::Pass
        );
        assert_eq!(
            JsonValidGate.evaluate(&req(), &resp("not json")).verdict,
            Verdict::Fail
        );
    }

    #[test]
    fn resolve_skips_unknown_and_keeps_known() {
        let gates = resolve_gates(&[
            "non-empty".to_owned(),
            "judge-diff".to_owned(), // unknown in Batch 3 -> skipped
            "json-valid".to_owned(),
        ]);
        let ids: Vec<_> = gates.iter().map(|g| g.id()).collect();
        assert_eq!(ids, ["non-empty", "json-valid"]);
    }

    #[test]
    fn aggregate_semantics() {
        let pass = GateResult::deterministic("a", Verdict::Pass, 0);
        let fail = GateResult::deterministic("b", Verdict::Fail, 0);
        let abstain = GateResult::abstain("c", "x", 0);
        assert_eq!(aggregate(&[]), Verdict::Pass); // empty -> pass
        assert_eq!(aggregate(std::slice::from_ref(&pass)), Verdict::Pass);
        assert_eq!(aggregate(&[pass.clone(), fail]), Verdict::Fail); // any fail -> fail
        assert_eq!(aggregate(&[pass, abstain]), Verdict::Pass); // abstain fail-open
    }
}
