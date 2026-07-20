//! The gate framework (SPEC §8 — the moat).
//!
//! A gate inspects a candidate [`ModelResponse`] and returns a [`GateResult`] verdict. Gates are
//! **async** because a real gate is I/O: a subprocess plugin (§8.1), an LLM judge, a test run.
//! Pure inline gates (non-empty, json-valid, schema) simply don't await. Gate execution is
//! wrapped by an **error budget** ([`GateHealth`]): a gate that errors too often is auto-disabled
//! with an alarm, so a broken gate can neither silently fail closed (burns money) nor silently
//! fail open (burns trust) (§7.2).

use crate::consistency::ConsistencyGate;
use crate::judge::JudgeGate;
use crate::provider::{Auth, ModelRequest, ModelResponse, ProviderRegistry};
use crate::subprocess::SubprocessGate;
use async_trait::async_trait;
use firstpass_core::{AbstainPolicy, GateDef, GateResult, Verdict, cost::PriceTable};
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
    /// Whether an **abstain** from this gate blocks serving like a `Fail` (§7.2,
    /// `on_abstain = "fail_closed"`). Default `false` = fail-open, the historical behavior.
    /// The abstain verdict itself is still recorded honestly on the receipt; only the
    /// aggregation treats it as blocking.
    fn abstain_fails_closed(&self) -> bool {
        false
    }
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
    id: String,
    schema: serde_json::Value,
}

impl SchemaGate {
    /// Build a schema gate from a JSON Schema value, under the id a route references it by.
    #[must_use]
    pub fn new(id: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            schema,
        }
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
        &self.id
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

/// Per-`(tenant, gate)` error budgets for a running proxy (app-level, shared across requests;
/// ADR 0004 §D6). Budgets (window + max abstain fraction) are registered per gate id at startup;
/// the actual rolling-window accounting is a separate [`GateHealth`] per `(tenant, gate)` pair, so
/// one tenant tripping a gate's budget auto-disables it only for that tenant, not globally. With
/// auth off every request carries the tenant id `"default"`, so there is exactly one bucket per
/// gate and behavior is unchanged from the pre-D6 global registry.
///
/// Unregistered gates default to enabled with no accounting (a gate the operator didn't register
/// a budget for is simply never auto-disabled).
#[derive(Debug, Default)]
pub struct GateHealthRegistry {
    budgets: std::collections::HashMap<String, (usize, f64)>,
    state: Mutex<std::collections::HashMap<(String, String), GateHealth>>,
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
        self.budgets
            .insert(gate_id.into(), (window, max_error_rate));
        self
    }

    /// Whether `gate_id` is currently enabled for `tenant` (unregistered gates are always
    /// enabled; a poisoned accounting lock fails open rather than blocking every request).
    #[must_use]
    pub fn enabled(&self, tenant: &str, gate_id: &str) -> bool {
        let Some(&(window, max_error_rate)) = self.budgets.get(gate_id) else {
            return true;
        };
        let Ok(mut state) = self.state.lock() else {
            return true;
        };
        state
            .entry((tenant.to_owned(), gate_id.to_owned()))
            .or_insert_with(|| GateHealth::new(window, max_error_rate))
            .is_enabled()
    }

    /// Record one outcome for `(tenant, gate_id)` (`errored` = abstained/crashed). No-op if the
    /// gate has no registered budget, or if the accounting lock is poisoned.
    pub fn record(&self, tenant: &str, gate_id: &str, errored: bool) {
        let Some(&(window, max_error_rate)) = self.budgets.get(gate_id) else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        state
            .entry((tenant.to_owned(), gate_id.to_owned()))
            .or_insert_with(|| GateHealth::new(window, max_error_rate))
            .record(gate_id, errored);
    }
}

/// Wraps any gate to make its abstains block serving (`on_abstain = "fail_closed"`, §7.2).
/// Pure delegation except [`Gate::abstain_fails_closed`]; the underlying gate's verdicts —
/// including the abstain itself — are recorded unchanged, so the receipt stays honest.
#[derive(Debug)]
struct FailClosed(Box<dyn Gate>);

#[async_trait]
impl Gate for FailClosed {
    fn id(&self) -> &str {
        self.0.id()
    }
    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        self.0.evaluate(req, resp).await
    }
    fn abstain_fails_closed(&self) -> bool {
        true
    }
}

/// Resolve a route's gate ids into runnable gates. Built-in ids (`non-empty`, `json-valid`) map to
/// inline gates; any other id is looked up among the config's `[[gate]]` definitions and built as a
/// [`SubprocessGate`] (`cmd`, SPEC §8.1), a [`JudgeGate`] (`judge`, §8.3), a [`ConsistencyGate`]
/// (`consistency`), or a [`SchemaGate`] (`schema`). Model-backed gates (judge/consistency) need a
/// provider (from `registry`), the caller's credentials (`auth`, BYOK), and `prices` so their
/// sample-call cost lands on the receipt. An id that is neither built-in nor defined — or a
/// model gate whose provider isn't registered — is skipped with a warning rather than failing the
/// request. A def with `on_abstain = "fail_closed"` is wrapped so its abstains block serving.
#[must_use]
pub fn resolve_gates(
    names: &[String],
    defs: &[GateDef],
    registry: &ProviderRegistry,
    auth: &Auth,
    prices: &PriceTable,
) -> Vec<Box<dyn Gate>> {
    let mut gates: Vec<Box<dyn Gate>> = Vec::new();
    let mut push = |gate: Box<dyn Gate>, def: Option<&GateDef>| {
        if def.is_some_and(|d| d.on_abstain == AbstainPolicy::FailClosed) {
            gates.push(Box::new(FailClosed(gate)));
        } else {
            gates.push(gate);
        }
    };
    for name in names {
        match name.as_str() {
            "non-empty" => push(Box::new(NonEmptyGate), None),
            "json-valid" => push(Box::new(JsonValidGate), None),
            other => match defs.iter().find(|d| d.id == other) {
                Some(def) if def.judge.is_some() => {
                    // `Config::parse` guarantees exactly one kind, so this `if let` always binds.
                    if let Some(judge) = def.judge.as_ref() {
                        let provider_id = judge.model.split('/').next().unwrap_or_default();
                        match registry.get(provider_id) {
                            Some(provider) => push(
                                Box::new(JudgeGate::new(
                                    def.id.clone(),
                                    provider,
                                    judge.model.clone(),
                                    auth.clone(),
                                    judge.threshold,
                                    judge.rubric.clone().unwrap_or_default(),
                                    prices.clone(),
                                )),
                                Some(def),
                            ),
                            None => tracing::warn!(
                                gate = %other, provider = %provider_id,
                                "judge gate provider not registered — skipped"
                            ),
                        }
                    }
                }
                Some(def) if def.consistency.is_some() => {
                    // `Config::parse` guarantees exactly one kind, so this `if let` always binds.
                    if let Some(cons) = def.consistency.as_ref() {
                        let provider_id = cons.model.split('/').next().unwrap_or_default();
                        match registry.get(provider_id) {
                            Some(provider) => push(
                                Box::new(ConsistencyGate::new(
                                    def.id.clone(),
                                    provider,
                                    cons.model.clone(),
                                    auth.clone(),
                                    cons.k,
                                    cons.threshold,
                                    prices.clone(),
                                )),
                                Some(def),
                            ),
                            None => tracing::warn!(
                                gate = %other, provider = %provider_id,
                                "consistency gate provider not registered — skipped"
                            ),
                        }
                    }
                }
                Some(def) if def.schema.is_some() => {
                    // `Config::parse` guarantees exactly one kind, so this `if let` always binds.
                    if let Some(schema) = def.schema.as_ref() {
                        push(
                            Box::new(SchemaGate::new(def.id.clone(), schema.clone())),
                            Some(def),
                        );
                    }
                }
                Some(def) => {
                    let Some((program, args)) = def.cmd.split_first() else {
                        tracing::warn!(gate = %other, "configured gate has empty cmd — skipped");
                        continue;
                    };
                    push(
                        Box::new(SubprocessGate::new(
                            def.id.clone(),
                            program.clone(),
                            args.to_vec(),
                            Duration::from_millis(def.timeout_ms),
                        )),
                        Some(def),
                    );
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

/// Aggregate per-gate verdicts into the attempt's overall verdict, honoring each gate's
/// abstain policy (§7.2): `Fail` if any gate fails **or** a `fail_closed` gate abstains;
/// otherwise `Pass`. An empty gate set passes. `fail_closed_ids` is the set of gate ids whose
/// abstains block serving (from [`Gate::abstain_fails_closed`]); a fail-open gate's abstain
/// never blocks (the historical behavior).
#[must_use]
pub fn aggregate_with_policy(
    results: &[GateResult],
    fail_closed_ids: &std::collections::HashSet<&str>,
) -> Verdict {
    let blocking = |r: &GateResult| {
        r.verdict == Verdict::Fail
            || (r.verdict == Verdict::Abstain && fail_closed_ids.contains(r.gate_id.as_str()))
    };
    if results.iter().any(blocking) {
        Verdict::Fail
    } else {
        Verdict::Pass
    }
}

/// Aggregate per-gate verdicts with every gate fail-open — `Fail` iff any gate fails.
#[must_use]
pub fn aggregate(results: &[GateResult]) -> Verdict {
    aggregate_with_policy(results, &std::collections::HashSet::new())
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
            raw: Value::Null,
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
        let g = SchemaGate::new(
            "schema",
            json!({
                "type": "object",
                "required": ["name", "age"],
                "properties": { "name": {"type": "string"}, "age": {"type": "integer"} }
            }),
        );
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

    fn empty_registry() -> ProviderRegistry {
        ProviderRegistry::new("http://localhost", "http://localhost")
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
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
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
            judge: None,
            consistency: None,
            schema: None,
            on_abstain: AbstainPolicy::FailOpen,
        }];
        let gates = resolve_gates(
            &["my-tests".to_owned(), "undefined".to_owned()],
            &defs,
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
        );
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
            judge: None,
            consistency: None,
            schema: None,
            on_abstain: AbstainPolicy::FailOpen,
        }];
        let gates = resolve_gates(
            &["no-bad".to_owned()],
            &defs,
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert_eq!(gates.len(), 1);
        let good = gates[0].evaluate(&req(), &resp("all good")).await;
        assert_eq!(good.verdict, Verdict::Pass);
        let bad = gates[0].evaluate(&req(), &resp("this is BAD")).await;
        assert_eq!(bad.verdict, Verdict::Fail);
    }

    #[test]
    fn resolve_builds_configured_judge_gate() {
        use crate::provider::{MockProvider, Provider};
        use std::collections::HashMap;
        use std::sync::Arc;

        let defs = vec![GateDef {
            id: "quality".to_owned(),
            cmd: vec![],
            timeout_ms: 30_000,
            judge: Some(firstpass_core::JudgeDef {
                model: "anthropic/judge".to_owned(),
                threshold: 0.7,
                rubric: None,
            }),
            consistency: None,
            schema: None,
            on_abstain: AbstainPolicy::FailOpen,
        }];

        // Registry that serves `anthropic` → the judge gate is built.
        let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        map.insert(
            "anthropic".to_owned(),
            Arc::new(MockProvider::new("anthropic", HashMap::new())),
        );
        let gates = resolve_gates(
            &["quality".to_owned()],
            &defs,
            &ProviderRegistry::from_map(map),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert_eq!(
            gates.iter().map(|g| g.id()).collect::<Vec<_>>(),
            ["quality"]
        );

        // Registry without that provider → skipped, not a hard failure.
        let skipped = resolve_gates(
            &["quality".to_owned()],
            &defs,
            &ProviderRegistry::from_map(HashMap::new()),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert!(
            skipped.is_empty(),
            "judge with no registered provider is skipped"
        );
    }

    #[test]
    fn resolve_builds_configured_consistency_gate() {
        use crate::provider::{MockProvider, Provider};
        use std::collections::HashMap;
        use std::sync::Arc;

        let defs = vec![GateDef {
            id: "uncertainty".to_owned(),
            cmd: vec![],
            timeout_ms: 30_000,
            judge: None,
            consistency: Some(firstpass_core::ConsistencyDef {
                model: "anthropic/claude-haiku-4-5".to_owned(),
                k: 3,
                threshold: 0.6,
            }),
            schema: None,
            on_abstain: AbstainPolicy::FailOpen,
        }];

        // Registry that serves `anthropic` → the consistency gate is built.
        let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        map.insert(
            "anthropic".to_owned(),
            Arc::new(MockProvider::new("anthropic", HashMap::new())),
        );
        let gates = resolve_gates(
            &["uncertainty".to_owned()],
            &defs,
            &ProviderRegistry::from_map(map),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert_eq!(
            gates.iter().map(|g| g.id()).collect::<Vec<_>>(),
            ["uncertainty"]
        );

        // Registry without the provider → skipped, not a hard failure.
        let skipped = resolve_gates(
            &["uncertainty".to_owned()],
            &defs,
            &ProviderRegistry::from_map(HashMap::new()),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert!(
            skipped.is_empty(),
            "consistency gate with no registered provider is skipped"
        );
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

    #[test]
    fn gate_health_registry_default_tenant_is_a_single_bucket() {
        // With auth off (single-operator), every request uses the same "default" tenant, so this
        // is exactly the pre-D6 global-registry behavior.
        let registry = GateHealthRegistry::new().with_budget("g", 4, 0.5);
        for _ in 0..4 {
            registry.record("default", "g", true);
        }
        assert!(!registry.enabled("default", "g"));
    }

    #[test]
    fn gate_health_registry_scopes_budget_per_tenant() {
        // ADR 0004 §D6: tenant A tripping the budget must not affect tenant B on the same gate.
        let registry = GateHealthRegistry::new().with_budget("g", 4, 0.5);
        for _ in 0..4 {
            registry.record("tenant-a", "g", true);
        }
        assert!(!registry.enabled("tenant-a", "g"), "A should be disabled");
        assert!(registry.enabled("tenant-b", "g"), "B must be unaffected");
    }

    #[test]
    fn gate_health_registry_unregistered_gate_always_enabled() {
        let registry = GateHealthRegistry::new();
        assert!(registry.enabled("tenant-a", "unknown-gate"));
        registry.record("tenant-a", "unknown-gate", true); // no-op, must not panic
        assert!(registry.enabled("tenant-a", "unknown-gate"));
    }

    #[tokio::test]
    async fn resolve_builds_configured_schema_gate() {
        let defs = vec![GateDef {
            id: "extract-shape".to_owned(),
            cmd: vec![],
            timeout_ms: 30_000,
            judge: None,
            consistency: None,
            schema: Some(json!({"type": "object", "required": ["name"]})),
            on_abstain: firstpass_core::AbstainPolicy::FailOpen,
        }];
        let gates = resolve_gates(
            &["extract-shape".to_owned()],
            &defs,
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert_eq!(
            gates.iter().map(|g| g.id()).collect::<Vec<_>>(),
            ["extract-shape"],
            "a [[gate]] def with `schema` resolves to a runnable SchemaGate"
        );
        let ok = gates[0].evaluate(&req(), &resp(r#"{"name":"a"}"#)).await;
        assert_eq!(ok.verdict, Verdict::Pass);
        let missing = gates[0].evaluate(&req(), &resp(r#"{}"#)).await;
        assert_eq!(missing.verdict, Verdict::Fail);
    }

    #[test]
    fn resolve_wraps_fail_closed_gate() {
        let mk = |on_abstain| {
            vec![GateDef {
                id: "tests".to_owned(),
                cmd: vec!["true".to_owned()],
                timeout_ms: 1000,
                judge: None,
                consistency: None,
                schema: None,
                on_abstain,
            }]
        };
        let open = resolve_gates(
            &["tests".to_owned()],
            &mk(firstpass_core::AbstainPolicy::FailOpen),
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert!(!open[0].abstain_fails_closed(), "default stays fail-open");
        let closed = resolve_gates(
            &["tests".to_owned()],
            &mk(firstpass_core::AbstainPolicy::FailClosed),
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
        );
        assert!(
            closed[0].abstain_fails_closed(),
            "on_abstain = fail_closed wraps the gate"
        );
        assert_eq!(closed[0].id(), "tests", "wrapper preserves the id");
    }

    #[test]
    fn aggregate_with_policy_fail_closed_abstain_blocks() {
        use std::collections::HashSet;
        let pass = GateResult::deterministic("a", Verdict::Pass, 0);
        let abstain = GateResult::abstain("c", "timeout", 0);
        // Fail-open (empty set): abstain never blocks — historical behavior.
        assert_eq!(
            aggregate_with_policy(&[pass.clone(), abstain.clone()], &HashSet::new()),
            Verdict::Pass
        );
        // Fail-closed: the same abstain blocks serving like a Fail.
        let closed: HashSet<&str> = ["c"].into_iter().collect();
        assert_eq!(
            aggregate_with_policy(&[pass, abstain], &closed),
            Verdict::Fail
        );
    }
}
