//! The gate framework (SPEC §8 — the moat).
//!
//! A gate inspects a candidate [`ModelResponse`] and returns a [`GateResult`] verdict. Gates are
//! **async** because a real gate is I/O: a subprocess plugin (§8.1), an LLM judge, a test run.
//! Pure inline gates (non-empty, json-valid, schema) simply don't await. Gate execution is
//! wrapped by an **error budget** ([`GateHealth`]): a gate that errors too often is auto-disabled
//! with an alarm, so a broken gate can neither silently fail closed (burns money) nor silently
//! fail open (burns trust) (§7.2).

use crate::provider::{ModelRequest, ModelResponse};
use crate::subprocess::SubprocessGate;
use async_trait::async_trait;
use firstpass_core::{GateDef, GateResult, Verdict};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

/// A verification gate. Object-safe + async so subprocess/model gates fit the same contract.
#[async_trait]
pub trait Gate: Send + Sync + std::fmt::Debug {
    /// Stable gate id (matches the name used in routing config).
    fn id(&self) -> &str;
    /// Evaluate the candidate response, producing a verdict + evidence.
    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult;
}

/// Fails an empty (whitespace-only) completion. The cheapest possible sanity gate.
#[derive(Debug, Clone, Copy)]
pub struct NonEmptyGate;

#[async_trait]
impl Gate for NonEmptyGate {
    fn id(&self) -> &str {
        "non-empty"
    }
    async fn evaluate(&self, _req: &ModelRequest, resp: &ModelResponse) -> GateResult {
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

#[async_trait]
impl Gate for JsonValidGate {
    fn id(&self) -> &str {
        "json-valid"
    }
    async fn evaluate(&self, _req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let ok = serde_json::from_str::<serde_json::Value>(resp.text.trim()).is_ok();
        GateResult::deterministic(self.id(), if ok { Verdict::Pass } else { Verdict::Fail }, 0)
    }
}

/// Validates the candidate (parsed as JSON) against a minimal JSON-Schema subset: top-level
/// `type`, `required`, and per-property `type`. Covers tool-call args and extraction tasks.
///
// ponytail: not full JSON Schema draft 2020-12 — just type/required/properties, the 90% that
// structured-output routes need. Swap for the `jsonschema` crate if nested/`$ref` schemas appear.
#[derive(Debug, Clone)]
pub struct SchemaGate {
    schema: serde_json::Value,
}

impl SchemaGate {
    /// Build a schema gate from a JSON Schema value.
    #[must_use]
    pub fn new(schema: serde_json::Value) -> Self {
        Self { schema }
    }

    /// Check `value` against the minimal schema subset; returns the first violation, if any.
    fn violation(&self, value: &serde_json::Value) -> Option<String> {
        use serde_json::Value;
        let type_ok = |v: &Value, ty: &str| match ty {
            "object" => v.is_object(),
            "array" => v.is_array(),
            "string" => v.is_string(),
            "number" => v.is_number(),
            "integer" => v.is_i64() || v.is_u64(),
            "boolean" => v.is_boolean(),
            "null" => v.is_null(),
            _ => true, // unknown type keyword: don't fail on it
        };
        if let Some(ty) = self.schema.get("type").and_then(Value::as_str)
            && !type_ok(value, ty)
        {
            return Some(format!("root is not of type {ty}"));
        }
        if let Some(req) = self.schema.get("required").and_then(Value::as_array) {
            for field in req.iter().filter_map(Value::as_str) {
                if value.get(field).is_none() {
                    return Some(format!("missing required field {field:?}"));
                }
            }
        }
        if let Some(props) = self.schema.get("properties").and_then(Value::as_object) {
            for (name, subschema) in props {
                if let (Some(actual), Some(ty)) = (
                    value.get(name),
                    subschema.get("type").and_then(Value::as_str),
                ) && !type_ok(actual, ty)
                {
                    return Some(format!("property {name:?} is not of type {ty}"));
                }
            }
        }
        None
    }
}

#[async_trait]
impl Gate for SchemaGate {
    fn id(&self) -> &str {
        "schema"
    }
    async fn evaluate(&self, _req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(resp.text.trim()) else {
            let mut r = GateResult::deterministic(self.id(), Verdict::Fail, 0);
            r.reason = Some("candidate is not valid JSON".to_owned());
            return r;
        };
        match self.violation(&value) {
            None => GateResult::deterministic(self.id(), Verdict::Pass, 0),
            Some(reason) => {
                let mut r = GateResult::deterministic(self.id(), Verdict::Fail, 0);
                r.reason = Some(reason);
                r
            }
        }
    }
}

/// Rolling per-gate error budget (§7.2, §8.3.4). Tracks the last `window` outcomes; once the
/// error (abstain) fraction over a full window exceeds `max_error_rate`, the gate is
/// auto-disabled and an alarm is logged. A disabled gate is skipped by the runner (its verdict
/// stops counting) rather than silently failing open or closed.
#[derive(Debug)]
pub struct GateHealth {
    window: usize,
    max_error_rate: f64,
    outcomes: Mutex<VecDeque<bool>>, // true = error (abstain), false = ok
    disabled: std::sync::atomic::AtomicBool,
}

impl GateHealth {
    /// Create a health tracker. `max_error_rate` is the abstain fraction (over a full `window`)
    /// beyond which the gate auto-disables.
    #[must_use]
    pub fn new(window: usize, max_error_rate: f64) -> Self {
        Self {
            window: window.max(1),
            max_error_rate,
            outcomes: Mutex::new(VecDeque::new()),
            disabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Whether the gate is currently enabled (not auto-disabled).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        !self.disabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record one outcome (`errored` = the gate abstained/crashed) and re-evaluate the budget.
    pub fn record(&self, gate_id: &str, errored: bool) {
        // ponytail: a single global lock per gate-health; fine at proxy request rates. Shard by
        // gate if a hot gate's lock ever shows up in a profile.
        let Ok(mut q) = self.outcomes.lock() else {
            return; // poisoned lock: skip accounting rather than panic on the request path
        };
        q.push_back(errored);
        while q.len() > self.window {
            q.pop_front();
        }
        if q.len() == self.window {
            let errors = q.iter().filter(|e| **e).count();
            let rate = errors as f64 / self.window as f64;
            if rate > self.max_error_rate && self.is_enabled() {
                self.disabled
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                tracing::error!(
                    gate = %gate_id,
                    error_rate = rate,
                    "gate exceeded its error budget — auto-disabled (ALARM)"
                );
            }
        }
    }
}

/// Per-gate error budgets for a running proxy (app-level, shared across requests). Lookup by
/// gate id; unknown gates default to enabled with no accounting (a gate the operator didn't
/// register a budget for is simply never auto-disabled).
#[derive(Debug, Default)]
pub struct GateHealthRegistry {
    gates: std::collections::HashMap<String, GateHealth>,
}

impl GateHealthRegistry {
    /// Empty registry — every gate enabled, no accounting.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an error budget for `gate_id` (window size + max abstain fraction).
    #[must_use]
    pub fn with_budget(
        mut self,
        gate_id: impl Into<String>,
        window: usize,
        max_error_rate: f64,
    ) -> Self {
        self.gates
            .insert(gate_id.into(), GateHealth::new(window, max_error_rate));
        self
    }

    /// Whether `gate_id` is currently enabled (unregistered gates are always enabled).
    #[must_use]
    pub fn enabled(&self, gate_id: &str) -> bool {
        self.gates.get(gate_id).is_none_or(GateHealth::is_enabled)
    }

    /// Record one outcome for `gate_id` (`errored` = abstained/crashed). No-op if unregistered.
    pub fn record(&self, gate_id: &str, errored: bool) {
        if let Some(h) = self.gates.get(gate_id) {
            h.record(gate_id, errored);
        }
    }
}

/// Resolve a route's gate ids into runnable gates. Built-in ids (`non-empty`, `json-valid`) map to
/// inline gates; any other id is looked up among the config's `[[gate]]` definitions and built as a
/// [`SubprocessGate`] (SPEC §8.1 — bring your own test / linter / judge, invoked as a command that
/// reads the candidate on stdin). An id that is neither built-in nor defined is skipped with a
/// warning rather than failing the request.
///
// ponytail: a native in-proxy LLM-judge gate (`judge-diff`, `self-consistency`) is the follow-on —
// it needs a live judge model to verify quality + the red-team injection fixtures. Until then a
// judge IS reachable today: point a `[[gate]]` cmd at a script that calls a model.
#[must_use]
pub fn resolve_gates(names: &[String], defs: &[GateDef]) -> Vec<Box<dyn Gate>> {
    let mut gates: Vec<Box<dyn Gate>> = Vec::new();
    for name in names {
        match name.as_str() {
            "non-empty" => gates.push(Box::new(NonEmptyGate)),
            "json-valid" => gates.push(Box::new(JsonValidGate)),
            other => match defs.iter().find(|d| d.id == other) {
                Some(def) => {
                    // `Config::parse` guarantees a non-empty cmd; stay defensive on the request path.
                    let Some((program, args)) = def.cmd.split_first() else {
                        tracing::warn!(gate = %other, "configured gate has empty cmd — skipped");
                        continue;
                    };
                    gates.push(Box::new(SubprocessGate::new(
                        def.id.clone(),
                        program.clone(),
                        args.to_vec(),
                        Duration::from_millis(def.timeout_ms),
                    )));
                }
                None => tracing::warn!(
                    gate = %other,
                    "unknown gate id — not a built-in and not defined in [[gate]]; skipped"
                ),
            },
        }
    }
    gates
}

/// Aggregate per-gate verdicts into the attempt's overall verdict.
///
/// `Fail` if any gate fails; otherwise `Pass`. An empty gate set passes.
///
// ponytail: `Abstain` is treated as pass (fail-open). Per-gate fail-open/closed policy is a
// follow-up; until then an abstaining/disabled gate never blocks serving.
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
    use serde_json::{Value, json};

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

    #[tokio::test]
    async fn non_empty_gate() {
        assert_eq!(
            NonEmptyGate.evaluate(&req(), &resp("hi")).await.verdict,
            Verdict::Pass
        );
        assert_eq!(
            NonEmptyGate.evaluate(&req(), &resp("   ")).await.verdict,
            Verdict::Fail
        );
    }

    #[tokio::test]
    async fn json_valid_gate() {
        assert_eq!(
            JsonValidGate
                .evaluate(&req(), &resp(r#"{"ok":true}"#))
                .await
                .verdict,
            Verdict::Pass
        );
        assert_eq!(
            JsonValidGate.evaluate(&req(), &resp("nope")).await.verdict,
            Verdict::Fail
        );
    }

    #[tokio::test]
    async fn schema_gate_type_and_required() {
        let g = SchemaGate::new(json!({
            "type": "object",
            "required": ["name", "age"],
            "properties": { "name": {"type": "string"}, "age": {"type": "integer"} }
        }));
        assert_eq!(
            g.evaluate(&req(), &resp(r#"{"name":"a","age":3}"#))
                .await
                .verdict,
            Verdict::Pass
        );
        // missing required
        assert_eq!(
            g.evaluate(&req(), &resp(r#"{"name":"a"}"#)).await.verdict,
            Verdict::Fail
        );
        // wrong property type
        assert_eq!(
            g.evaluate(&req(), &resp(r#"{"name":"a","age":"x"}"#))
                .await
                .verdict,
            Verdict::Fail
        );
        // wrong root type
        assert_eq!(g.evaluate(&req(), &resp("[]")).await.verdict, Verdict::Fail);
        // not even JSON
        assert_eq!(
            g.evaluate(&req(), &resp("plain text")).await.verdict,
            Verdict::Fail
        );
    }

    #[test]
    fn resolve_skips_unknown_and_keeps_known() {
        let gates = resolve_gates(
            &[
                "non-empty".to_owned(),
                "judge-diff".to_owned(),
                "json-valid".to_owned(),
            ],
            &[],
        );
        let ids: Vec<_> = gates.iter().map(|g| g.id()).collect();
        assert_eq!(ids, ["non-empty", "json-valid"]);
    }

    #[test]
    fn resolve_builds_configured_subprocess_gate() {
        // A gate id that isn't built-in resolves to a SubprocessGate when defined in `[[gate]]`.
        let defs = vec![GateDef {
            id: "my-tests".to_owned(),
            cmd: vec!["true".to_owned()],
            timeout_ms: 1000,
        }];
        let gates = resolve_gates(&["my-tests".to_owned(), "undefined".to_owned()], &defs);
        let ids: Vec<_> = gates.iter().map(|g| g.id()).collect();
        assert_eq!(
            ids,
            ["my-tests"],
            "configured id resolves; unknown id skipped"
        );
    }

    #[tokio::test]
    async fn configured_subprocess_gate_runs_end_to_end() {
        // A user-defined gate that fails iff the candidate text contains "BAD" — proving the config
        // → SubprocessGate → verdict path works over stdin, no hard-coded gate name.
        let script = r#"c=$(cat); case "$c" in *BAD*) echo '{"verdict":"fail"}';; *) echo '{"verdict":"pass"}';; esac"#;
        let defs = vec![GateDef {
            id: "no-bad".to_owned(),
            cmd: vec!["bash".to_owned(), "-c".to_owned(), script.to_owned()],
            timeout_ms: 5000,
        }];
        let gates = resolve_gates(&["no-bad".to_owned()], &defs);
        assert_eq!(gates.len(), 1);
        let good = gates[0].evaluate(&req(), &resp("all good")).await;
        assert_eq!(good.verdict, Verdict::Pass);
        let bad = gates[0].evaluate(&req(), &resp("this is BAD")).await;
        assert_eq!(bad.verdict, Verdict::Fail);
    }

    #[test]
    fn aggregate_semantics() {
        let pass = GateResult::deterministic("a", Verdict::Pass, 0);
        let fail = GateResult::deterministic("b", Verdict::Fail, 0);
        let abstain = GateResult::abstain("c", "x", 0);
        assert_eq!(aggregate(&[]), Verdict::Pass);
        assert_eq!(aggregate(std::slice::from_ref(&pass)), Verdict::Pass);
        assert_eq!(aggregate(&[pass.clone(), fail]), Verdict::Fail);
        assert_eq!(aggregate(&[pass, abstain]), Verdict::Pass);
    }

    #[test]
    fn error_budget_auto_disables_past_threshold() {
        let h = GateHealth::new(10, 0.5); // disable when >50% of the last 10 error
        // Window fills to exactly 5 errors / 10 = 50% — NOT above threshold, still enabled.
        for _ in 0..5 {
            h.record("g", true);
        }
        for _ in 0..5 {
            h.record("g", false);
        }
        assert!(h.is_enabled(), "50% is at, not above, the budget");
        // Flood errors: the window slides until errors exceed 50% -> auto-disabled.
        for _ in 0..6 {
            h.record("g", true);
        }
        assert!(
            !h.is_enabled(),
            "gate should auto-disable once error rate exceeds budget"
        );
    }

    #[test]
    fn healthy_gate_stays_enabled() {
        let h = GateHealth::new(20, 0.5);
        for _ in 0..100 {
            h.record("g", false);
        }
        assert!(h.is_enabled());
    }
}
