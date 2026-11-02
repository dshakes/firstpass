//! The gate framework (SPEC §8 — the moat).
//!
//! A gate inspects a candidate [`ModelResponse`] and returns a [`GateResult`] verdict. Gates are
//! **async** because a real gate is I/O: a subprocess plugin (§8.1), an LLM judge, a test run.
//! Pure inline gates (non-empty, json-valid, schema) simply don't await. Gate execution is
//! wrapped by an **error budget** ([`GateHealth`]): a gate that errors too often is auto-disabled
//! with an alarm, so a broken gate can neither silently fail closed (burns money) nor silently
//! fail open (burns trust) (§7.2).

use crate::consistency::ConsistencyGate;
use crate::decision::DecisionGate;
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

/// True when the request's `tools` field is a non-empty array (Anthropic or OpenAI format).
/// Used by inline gates to decide whether tool-call-only responses are valid for this request.
fn request_has_declared_tools(req: &ModelRequest) -> bool {
    req.tools.as_array().is_some_and(|a| !a.is_empty())
}

/// Fails an empty (whitespace-only) completion with no tool calls.
/// Passes when the text is non-empty **or** the response contains at least one tool call AND
/// the request declared tools — an undeclared tool call with empty text is a Fail.
#[derive(Debug, Clone, Copy)]
pub struct NonEmptyGate;

#[async_trait]
impl Gate for NonEmptyGate {
    fn id(&self) -> &str {
        "non-empty"
    }
    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let verdict = if !resp.text.trim().is_empty() {
            Verdict::Pass
        } else if resp.has_tool_calls() && request_has_declared_tools(req) {
            // Tool-call-only turn with declared tools: the tool call IS the non-empty output.
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        GateResult::deterministic(self.id(), verdict, 0)
    }
}

/// Passes only if the completion parses as JSON. Useful for structured-output routes.
///
/// **Passes** (not Abstains) when the response has tool calls AND the request declared tools:
/// there is no text payload to validate (the JSON answer IS the tool call). Abstain would count
/// against GateHealth (auto-disabling the gate above 25% abstain rate) on every agent turn.
///
/// **Note**: preamble text alongside tool calls (tools declared) also passes — the text is not
/// the structured JSON answer in that case. Use `tool-args-valid` to validate tool call args.
#[derive(Debug, Clone, Copy)]
pub struct JsonValidGate;

#[async_trait]
impl Gate for JsonValidGate {
    fn id(&self) -> &str {
        "json-valid"
    }
    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let has_tools = request_has_declared_tools(req);
        // Tool calls present AND tools were declared: the text (preamble or absent) is not the
        // JSON answer — the tool call IS the structured output.
        if resp.has_tool_calls() && has_tools {
            return GateResult::deterministic(self.id(), Verdict::Pass, 0);
        }
        let ok = serde_json::from_str::<serde_json::Value>(resp.text.trim()).is_ok();
        GateResult::deterministic(self.id(), if ok { Verdict::Pass } else { Verdict::Fail }, 0)
    }
}

/// Recursive minimal JSON-Schema validator for the subset tool schemas use.
///
/// Supported keywords: `type` (string or array of strings, incl. `["x","null"]` for nullable),
/// `required`, `properties` (recursive), `items` (recursive, for arrays), `enum`,
/// `additionalProperties: false`. Unknown keywords are silently ignored — don't fail on them.
///
/// Pure function so it is used by both [`SchemaGate`] and [`validate_tool_args_against_tools`].
fn schema_violation(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    path: &str,
) -> Option<String> {
    use serde_json::Value;

    /// True when `v` matches a single JSON Schema type name.
    fn type_matches(v: &Value, ty: &str) -> bool {
        match ty {
            "object" => v.is_object(),
            "array" => v.is_array(),
            "string" => v.is_string(),
            // "integer" is strict: floats with fractional parts are not integers.
            "integer" => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
            "number" => v.is_number(),
            "boolean" => v.is_boolean(),
            "null" => v.is_null(),
            _ => true, // unknown type keyword: don't fail on it
        }
    }

    // `type` keyword — may be a string or an array of strings (e.g. ["string","null"]).
    if let Some(type_kw) = schema.get("type") {
        let ok = match type_kw {
            Value::String(ty) => type_matches(value, ty),
            Value::Array(types) => types
                .iter()
                .filter_map(Value::as_str)
                .any(|ty| type_matches(value, ty)),
            _ => true,
        };
        if !ok {
            return Some(format!("{path} has wrong type"));
        }
    }

    // `enum` keyword — value must be one of the listed literals.
    if let Some(Value::Array(variants)) = schema.get("enum")
        && !variants.contains(value)
    {
        return Some(format!("{path} is not one of the allowed enum values"));
    }

    if let Some(obj) = value.as_object() {
        // `required` keyword — every listed field must be present.
        if let Some(Value::Array(req)) = schema.get("required") {
            for field in req.iter().filter_map(Value::as_str) {
                if !obj.contains_key(field) {
                    return Some(format!("{path}.{field:?} is required but missing"));
                }
            }
        }

        // `additionalProperties: false` — no keys outside `properties`.
        let ap_false = schema.get("additionalProperties").and_then(Value::as_bool) == Some(false);
        let defined_props: std::collections::HashSet<&str> = schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|p| p.keys().map(String::as_str).collect())
            .unwrap_or_default();
        if ap_false {
            for key in obj.keys() {
                if !defined_props.contains(key.as_str()) {
                    return Some(format!(
                        "{path}.{key:?} is an additional property not allowed by the schema"
                    ));
                }
            }
        } else if let Some(extra_schema) =
            schema.get("additionalProperties").filter(|v| v.is_object())
        {
            // `additionalProperties: {schema}` — keys outside `properties` must match it.
            for (key, child) in obj {
                if !defined_props.contains(key.as_str())
                    && let Some(v) = schema_violation(child, extra_schema, &format!("{path}.{key}"))
                {
                    return Some(v);
                }
            }
        }

        // `properties` keyword — recurse into each present property.
        if let Some(props) = schema.get("properties").and_then(Value::as_object) {
            for (name, subschema) in props {
                if let Some(child) = obj.get(name) {
                    let child_path = format!("{path}.{name}");
                    if let Some(v) = schema_violation(child, subschema, &child_path) {
                        return Some(v);
                    }
                }
            }
        }
    }

    // `items` keyword — recurse into each array element.
    if let (Some(arr), Some(items_schema)) = (value.as_array(), schema.get("items")) {
        for (i, elem) in arr.iter().enumerate() {
            let child_path = format!("{path}[{i}]");
            if let Some(v) = schema_violation(elem, items_schema, &child_path) {
                return Some(v);
            }
        }
    }

    None
}

/// Validates the candidate (parsed as JSON) against a minimal JSON-Schema subset.
///
/// Supports: `type` (string or `["x","null"]`), `required`, `properties` (recursive),
/// `items` (recursive), `enum`, `additionalProperties: false`. Unknown keywords ignored.
///
// ponytail: not full JSON Schema draft 2020-12 — no `$ref`, `if/then/else`, `anyOf`, `oneOf`.
// Swap for the `jsonschema` crate if those are needed.
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

    /// Check `value` against the schema; returns the first violation, if any.
    pub(crate) fn violation(&self, value: &serde_json::Value) -> Option<String> {
        schema_violation(value, &self.schema, "root")
    }
}

#[async_trait]
impl Gate for SchemaGate {
    fn id(&self) -> &str {
        &self.id
    }
    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let has_tools = request_has_declared_tools(req);
        // Tool calls with declared tools: text (if any) is preamble, not the JSON answer.
        // Pass without schema-checking it — use tool-args-valid to validate tool call args.
        if resp.has_tool_calls() && has_tools {
            return GateResult::deterministic(self.id(), Verdict::Pass, 0);
        }
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

/// Extract `(tool_name, args_json_string)` pairs from a provider raw response.
/// Handles Anthropic `content[*].type == "tool_use"` (input is a JSON object)
/// and OpenAI `choices[0].message.tool_calls[*].function` (arguments is a JSON string).
///
/// Malformed entries (missing `name` in Anthropic, missing `function` in OpenAI) are recorded
/// with an empty name so `ToolArgsValidGate` fails the hallucinated-tool check rather than
/// silently skipping the entry and treating the call list as empty.
fn extract_response_tool_calls(raw: &serde_json::Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    // Anthropic format
    if let Some(blocks) = raw.get("content").and_then(serde_json::Value::as_array) {
        for b in blocks {
            if b.get("type").and_then(serde_json::Value::as_str) == Some("tool_use") {
                // Missing `name` → empty string triggers hallucinated-tool Fail downstream.
                let name = b
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                // Missing `input` is malformed: an empty string fails the JSON parse, so the call
                // fails the gate instead of passing as fabricated `{}` args.
                let args = b
                    .get("input")
                    .map_or_else(String::new, std::string::ToString::to_string);
                out.push((name, args));
            }
        }
        return out;
    }
    // OpenAI format
    if let Some(calls) = raw
        .pointer("/choices/0/message/tool_calls")
        .and_then(serde_json::Value::as_array)
    {
        for call in calls {
            // Missing `function` → record entry with empty name to trigger Fail, not silent skip.
            let (name, args) = if let Some(func) = call.get("function") {
                (
                    func.get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    // Missing or non-string `arguments` is malformed: an empty string fails the
                    // JSON parse below, so the call fails the gate instead of passing as `{}`.
                    func.get("arguments")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                )
            } else {
                (String::new(), String::new())
            };
            out.push((name, args));
        }
    }
    out
}

/// Validate `tool_args` against a tool's `input_schema` (Anthropic) or `parameters` (OpenAI)
/// using the recursive [`schema_violation`] validator. Returns `None` on success, or the
/// first violation message.
fn validate_tool_args_against_tools(
    name: &str,
    args: &serde_json::Value,
    tools: &serde_json::Value,
) -> Option<String> {
    let arr = tools.as_array()?;
    for tool in arr {
        // Anthropic format: {"name": "...", "input_schema": {...}}
        if let (Some(n), Some(schema)) = (
            tool.get("name").and_then(serde_json::Value::as_str),
            tool.get("input_schema"),
        ) && n == name
        {
            return schema_violation(args, schema, "args");
        }
        // OpenAI format: {"type":"function","function":{"name":"...","parameters":{...}}}
        if let Some(func) = tool.get("function")
            && let (Some(n), Some(schema)) = (
                func.get("name").and_then(serde_json::Value::as_str),
                func.get("parameters"),
            )
            && n == name
        {
            return schema_violation(args, schema, "args");
        }
    }
    None // unknown tool: no schema to validate against — don't fail
}

/// True if `name` appears in the request's `tools` array (Anthropic or OpenAI format).
fn tool_name_is_known(name: &str, tools: &serde_json::Value) -> bool {
    let Some(arr) = tools.as_array() else {
        return false;
    };
    for tool in arr {
        // Anthropic: {"name": "...", "input_schema": {...}}
        if tool.get("name").and_then(serde_json::Value::as_str) == Some(name) {
            return true;
        }
        // OpenAI: {"type": "function", "function": {"name": "...", "parameters": {...}}}
        if tool
            .pointer("/function/name")
            .and_then(serde_json::Value::as_str)
            == Some(name)
        {
            return true;
        }
    }
    false
}

/// Validates every tool call's arguments against the request's tool schema.
///
/// - **Pass**: the response contains no tool calls (not applicable — counts as satisfied,
///   not as an error), OR all tool calls have valid JSON arguments that satisfy their schema.
/// - **Fail**: the response calls a tool that was not declared in the request (hallucinated
///   tool), the request declares no tools but the response contains tool calls, any tool
///   call's arguments are not valid JSON, or any call violates its schema.
///
/// Pass (not Abstain) on no-tool-call turns so GateHealth does not count this as an error
/// and auto-disable the gate on agent traffic where most turns have no tool calls.
///
/// Registered as the built-in gate id `tool-args-valid`. See AGENTS.md gotchas.
#[derive(Debug, Clone, Copy)]
pub struct ToolArgsValidGate;

#[async_trait]
impl Gate for ToolArgsValidGate {
    fn id(&self) -> &str {
        "tool-args-valid"
    }
    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let calls = extract_response_tool_calls(&resp.raw);
        if calls.is_empty() {
            // No tool calls — trivially satisfied. Pass (not Abstain: see doc comment).
            return GateResult::deterministic(self.id(), Verdict::Pass, 0);
        }

        // Response has tool calls. Check that the request actually declared any tools.
        let has_declared_tools = req.tools.as_array().is_some_and(|a| !a.is_empty());
        if !has_declared_tools {
            let mut r = GateResult::deterministic(self.id(), Verdict::Fail, 0);
            r.reason = Some(
                "response contains tool calls but request declared no tools (hallucinated)"
                    .to_owned(),
            );
            return r;
        }

        for (name, args_str) in &calls {
            // Fail on any call to a tool not declared in the request.
            if !tool_name_is_known(name, &req.tools) {
                let mut r = GateResult::deterministic(self.id(), Verdict::Fail, 0);
                r.reason = Some(format!(
                    "tool call {name:?}: tool not declared in request (hallucinated)"
                ));
                return r;
            }
            let args = match serde_json::from_str::<serde_json::Value>(args_str) {
                Ok(v) => v,
                Err(_) => {
                    let mut r = GateResult::deterministic(self.id(), Verdict::Fail, 0);
                    r.reason = Some(format!("tool call {name:?}: arguments is not valid JSON"));
                    return r;
                }
            };
            if let Some(violation) = validate_tool_args_against_tools(name, &args, &req.tools) {
                let mut r = GateResult::deterministic(self.id(), Verdict::Fail, 0);
                r.reason = Some(format!("tool call {name:?}: {violation}"));
                return r;
            }
        }
        GateResult::deterministic(self.id(), Verdict::Pass, 0)
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
/// (`consistency`), a [`SchemaGate`] (`schema`), or a [`DecisionGate`] (`decision`, TypeSafe's Jev
/// — **unmeasured** precision/recall). Model-backed gates (judge/consistency) need a provider
/// (from `registry`), the caller's credentials (`auth`, BYOK), and `prices` so their sample-call
/// cost lands on the receipt; a decision gate needs its own API key, read from `api_key_env` via
/// `http` (the shared client). An id that is neither built-in nor defined — a model gate whose
/// provider isn't registered — or a decision gate whose API key env var is unset — is skipped
/// with a warning rather than failing the request. A def with `on_abstain = "fail_closed"` is
/// wrapped so its abstains block serving.
#[must_use]
pub fn resolve_gates(
    names: &[String],
    defs: &[GateDef],
    registry: &ProviderRegistry,
    auth: &Auth,
    prices: &PriceTable,
    http: &reqwest::Client,
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
            "tool-args-valid" => push(Box::new(ToolArgsValidGate), None),
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
                Some(def) if def.decision.is_some() => {
                    // `Config::parse` guarantees exactly one kind, so this `if let` always binds.
                    if let Some(decision) = def.decision.as_ref() {
                        match std::env::var(&decision.api_key_env) {
                            Ok(api_key) => push(
                                Box::new(DecisionGate::new(
                                    def.id.clone(),
                                    http.clone(),
                                    decision,
                                    api_key,
                                )),
                                Some(def),
                            ),
                            Err(_) => tracing::warn!(
                                gate = %other, env_var = %decision.api_key_env,
                                "decision gate API key env var is unset — skipped"
                            ),
                        }
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
            cache_write_tokens: 0,
            cache_read_tokens: 0,
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
            cache_prefix: false,
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
            &reqwest::Client::new(),
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
            decision: None,
            on_abstain: AbstainPolicy::FailOpen,
        }];
        let gates = resolve_gates(
            &["my-tests".to_owned(), "undefined".to_owned()],
            &defs,
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
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
            decision: None,
            on_abstain: AbstainPolicy::FailOpen,
        }];
        let gates = resolve_gates(
            &["no-bad".to_owned()],
            &defs,
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
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
            decision: None,
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
            &reqwest::Client::new(),
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
            &reqwest::Client::new(),
        );
        assert!(
            skipped.is_empty(),
            "judge with no registered provider is skipped"
        );
    }

    #[test]
    fn resolve_builds_configured_decision_gate() {
        // No unsafe env::set_var (forbidden workspace-wide, and unsound in edition 2024): reuse
        // `PATH`, guaranteed set in any test process, as a stand-in "the key env var is present"
        // — resolve_gates only checks presence, never the value's shape. Mirrors the same trick
        // already used in `provider.rs`'s tests.
        let def = |api_key_env: &str| GateDef {
            id: "verify".to_owned(),
            cmd: vec![],
            timeout_ms: 30_000,
            judge: None,
            consistency: None,
            schema: None,
            decision: Some(firstpass_core::DecisionDef {
                provider: "typesafe".to_owned(),
                model: "jev-latest".to_owned(),
                api_key_env: api_key_env.to_owned(),
                base_url: "https://api.typesafe.ai".to_owned(),
                timeout_ms: 1_000,
                threshold: 0.5,
                instructions: None,
            }),
            on_abstain: AbstainPolicy::FailOpen,
        };

        let gates = resolve_gates(
            &["verify".to_owned()],
            &[def("PATH")],
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
        );
        assert_eq!(
            gates.iter().map(|g| g.id()).collect::<Vec<_>>(),
            ["verify"],
            "decision gate resolves when its API key env var is set"
        );

        let gates = resolve_gates(
            &["verify".to_owned()],
            &[def("FIRSTPASS_TEST_DEFINITELY_UNSET_DECISION_KEY_XYZ")],
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
        );
        assert!(
            gates.is_empty(),
            "decision gate with an unset API key env var is skipped, not a hard failure"
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
            decision: None,
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
            &reqwest::Client::new(),
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
            &reqwest::Client::new(),
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
            decision: None,
            on_abstain: firstpass_core::AbstainPolicy::FailOpen,
        }];
        let gates = resolve_gates(
            &["extract-shape".to_owned()],
            &defs,
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
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
                decision: None,
                on_abstain,
            }]
        };
        let open = resolve_gates(
            &["tests".to_owned()],
            &mk(firstpass_core::AbstainPolicy::FailOpen),
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
        );
        assert!(!open[0].abstain_fails_closed(), "default stays fail-open");
        let closed = resolve_gates(
            &["tests".to_owned()],
            &mk(firstpass_core::AbstainPolicy::FailClosed),
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
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

    // ── Defect 1: gates must handle tool-call-only responses correctly ────────

    fn tool_use_resp() -> ModelResponse {
        ModelResponse {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            text: String::new(), // tool-only: no text
            in_tokens: 10,
            cache_write_tokens: 0,
            cache_read_tokens: 0,
            out_tokens: 5,
            raw: serde_json::json!({
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search",
                    "input": { "q": "hello" }
                }]
            }),
        }
    }

    fn tool_req_with_schema() -> ModelRequest {
        ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: None,
            messages: vec![],
            max_tokens: 64,
            tools: serde_json::json!([{
                "name": "search",
                "input_schema": {
                    "type": "object",
                    "required": ["q"],
                    "properties": { "q": { "type": "string" } }
                }
            }]),
            raw: serde_json::Value::Null,
            cache_prefix: false,
        }
    }

    #[tokio::test]
    async fn non_empty_passes_on_tool_use_when_tools_declared() {
        // A tool-call-only turn with declared tools is not "empty" — it must not escalate.
        // Uses tool_req_with_schema() which has a non-empty tools array.
        assert_eq!(
            NonEmptyGate
                .evaluate(&tool_req_with_schema(), &tool_use_resp())
                .await
                .verdict,
            Verdict::Pass
        );
    }

    #[tokio::test]
    async fn non_empty_fails_on_tool_call_when_no_tools_declared() {
        // Finding 2: tool-call bypass only when request declared tools.
        // req() has tools: Value::Null — so tool call with empty text is treated as empty.
        assert_eq!(
            NonEmptyGate
                .evaluate(&req(), &tool_use_resp())
                .await
                .verdict,
            Verdict::Fail,
            "tool call without declared tools must Fail non-empty (undeclared/hallucinated)"
        );
    }

    #[tokio::test]
    async fn non_empty_still_fails_on_truly_empty_response() {
        assert_eq!(
            NonEmptyGate.evaluate(&req(), &resp("")).await.verdict,
            Verdict::Fail
        );
        assert_eq!(
            NonEmptyGate.evaluate(&req(), &resp("   ")).await.verdict,
            Verdict::Fail
        );
    }

    #[tokio::test]
    async fn json_valid_passes_on_tool_use_when_tools_declared() {
        // json-valid on a tool-only turn (tools declared) has nothing to parse → Pass, not Abstain.
        assert_eq!(
            JsonValidGate
                .evaluate(&tool_req_with_schema(), &tool_use_resp())
                .await
                .verdict,
            Verdict::Pass
        );
    }

    #[tokio::test]
    async fn json_valid_fails_on_tool_call_when_no_tools_declared() {
        // Finding 2: tool-call bypass only when request declared tools.
        // req() has no tools, so the call is treated as a non-JSON text response.
        assert_eq!(
            JsonValidGate
                .evaluate(&req(), &tool_use_resp())
                .await
                .verdict,
            Verdict::Fail,
            "tool call without declared tools must Fail json-valid"
        );
    }

    #[tokio::test]
    async fn json_valid_passes_on_preamble_text_plus_tool_calls() {
        // Finding 4: preamble text alongside tool calls (with declared tools) must Pass.
        // The text is not the JSON answer — tool-args-valid validates the call args instead.
        let resp_with_preamble = ModelResponse {
            text: "I will look that up for you.".to_owned(),
            raw: serde_json::json!({
                "content": [
                    {"type": "text", "text": "I will look that up for you."},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "hello"}}
                ]
            }),
            ..tool_use_resp()
        };
        assert_eq!(
            JsonValidGate
                .evaluate(&tool_req_with_schema(), &resp_with_preamble)
                .await
                .verdict,
            Verdict::Pass,
            "preamble text + tool calls with declared tools must Pass json-valid"
        );
    }

    #[tokio::test]
    async fn json_valid_still_fails_on_non_json_text_response() {
        assert_eq!(
            JsonValidGate
                .evaluate(&req(), &resp("not json"))
                .await
                .verdict,
            Verdict::Fail
        );
    }

    #[tokio::test]
    async fn schema_gate_passes_on_tool_use_when_tools_declared() {
        // Trivially passes (not Abstain) on tool-only turns when tools are declared.
        let g = SchemaGate::new(
            "extract",
            serde_json::json!({"type": "object", "required": ["name"]}),
        );
        assert_eq!(
            g.evaluate(&tool_req_with_schema(), &tool_use_resp())
                .await
                .verdict,
            Verdict::Pass
        );
    }

    #[tokio::test]
    async fn schema_gate_passes_on_preamble_text_plus_tool_calls() {
        // Finding 4: preamble text alongside tool calls (with declared tools) must Pass schema-gate.
        let g = SchemaGate::new(
            "extract",
            serde_json::json!({"type": "object", "required": ["name"]}),
        );
        let resp_with_preamble = ModelResponse {
            text: "Here is the result:".to_owned(),
            raw: serde_json::json!({
                "content": [
                    {"type": "text", "text": "Here is the result:"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "hello"}}
                ]
            }),
            ..tool_use_resp()
        };
        assert_eq!(
            g.evaluate(&tool_req_with_schema(), &resp_with_preamble)
                .await
                .verdict,
            Verdict::Pass,
            "preamble text + tool calls with declared tools must Pass schema-gate"
        );
    }

    #[tokio::test]
    async fn tool_args_valid_passes_with_valid_args() {
        assert_eq!(
            ToolArgsValidGate
                .evaluate(&tool_req_with_schema(), &tool_use_resp())
                .await
                .verdict,
            Verdict::Pass
        );
    }

    #[tokio::test]
    async fn tool_args_valid_passes_when_no_tool_calls() {
        // Text-only response: trivially satisfied (Pass, not Abstain — see doc comment).
        assert_eq!(
            ToolArgsValidGate
                .evaluate(&req(), &resp("hello"))
                .await
                .verdict,
            Verdict::Pass
        );
    }

    #[tokio::test]
    async fn tool_args_valid_fails_on_hallucinated_tool_name() {
        // Response calls "unknown_tool" which is not in the request's tools list.
        let bad_resp = ModelResponse {
            text: String::new(),
            raw: serde_json::json!({
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "unknown_tool",
                    "input": { "x": 1 }
                }]
            }),
            ..tool_use_resp()
        };
        let r = ToolArgsValidGate
            .evaluate(&tool_req_with_schema(), &bad_resp)
            .await;
        assert_eq!(r.verdict, Verdict::Fail);
        assert!(
            r.reason
                .as_deref()
                .is_some_and(|s| s.contains("unknown_tool")),
            "reason must name the hallucinated tool"
        );
    }

    #[tokio::test]
    async fn tool_args_valid_fails_when_no_tools_declared_but_response_has_calls() {
        // Request declared no tools, but response contains tool calls.
        assert_eq!(
            ToolArgsValidGate
                .evaluate(&req(), &tool_use_resp()) // req() has no tools
                .await
                .verdict,
            Verdict::Fail
        );
    }

    #[tokio::test]
    async fn tool_args_valid_fails_on_missing_required_field() {
        let bad_resp = ModelResponse {
            text: String::new(),
            raw: serde_json::json!({
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    "name": "search",
                    "input": {} // missing required "q"
                }]
            }),
            ..tool_use_resp()
        };
        let r = ToolArgsValidGate
            .evaluate(&tool_req_with_schema(), &bad_resp)
            .await;
        assert_eq!(r.verdict, Verdict::Fail);
        assert!(
            r.reason.as_deref().is_some_and(|s| s.contains("search")),
            "reason must name the failing tool"
        );
    }

    #[tokio::test]
    async fn tool_args_valid_resolves_as_builtin() {
        // "tool-args-valid" is a built-in id: resolve_gates must return it.
        let gates = resolve_gates(
            &["tool-args-valid".to_owned()],
            &[],
            &empty_registry(),
            &Auth::default(),
            &PriceTable::default(),
            &reqwest::Client::new(),
        );
        assert_eq!(
            gates.iter().map(|g| g.id()).collect::<Vec<_>>(),
            ["tool-args-valid"]
        );
    }

    // ── Finding 1: recursive schema_violation ──────────────────────────────────────────────────

    /// Nested required field missing → Fail.
    /// Guards: `schema_violation` required-check inside a `properties` child schema.
    #[test]
    fn schema_violation_nested_required_missing() {
        let schema = json!({
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "required": ["street"],
                    "properties": { "street": {"type": "string"} }
                }
            }
        });
        // address.street is required but missing → violation
        assert!(
            schema_violation(&json!({"address": {}}), &schema, "root").is_some(),
            "nested required field must produce a violation"
        );
        // address.street present → no violation
        assert!(
            schema_violation(
                &json!({"address": {"street": "1 Main St"}}),
                &schema,
                "root"
            )
            .is_none(),
            "valid nested object must produce no violation"
        );
    }

    /// Enum violation → Fail.
    /// Guards: `schema_violation` enum-keyword check.
    #[test]
    fn schema_violation_enum_check() {
        let schema = json!({ "enum": ["red", "green", "blue"] });
        assert!(
            schema_violation(&json!("purple"), &schema, "root").is_some(),
            "value outside enum must produce a violation"
        );
        assert!(
            schema_violation(&json!("red"), &schema, "root").is_none(),
            "value inside enum must produce no violation"
        );
    }

    /// items type wrong → Fail.
    /// Guards: `schema_violation` items-keyword recursive check.
    #[test]
    fn schema_violation_items_type_wrong() {
        let schema = json!({
            "type": "array",
            "items": { "type": "integer" }
        });
        // Array containing a string → violation
        assert!(
            schema_violation(&json!([1, 2, "three"]), &schema, "root").is_some(),
            "array item of wrong type must produce a violation"
        );
        // All integers → no violation
        assert!(
            schema_violation(&json!([1, 2, 3]), &schema, "root").is_none(),
            "array of correct type must produce no violation"
        );
    }

    /// additionalProperties: false with extra key → Fail.
    /// Guards: `schema_violation` additionalProperties-false check.
    #[test]
    fn schema_violation_additional_properties_subschema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": {"type": "integer"}
        });
        assert!(
            schema_violation(&serde_json::json!({"a": "x", "b": 1}), &schema, "args").is_none()
        );
        assert!(
            schema_violation(&serde_json::json!({"a": "x", "b": "no"}), &schema, "args").is_some()
        );
    }

    #[test]
    fn anthropic_tool_use_without_input_is_malformed() {
        let raw = serde_json::json!({"content": [{"type": "tool_use", "id": "t1", "name": "f"}]});
        let calls = extract_response_tool_calls(&raw);
        assert_eq!(calls.len(), 1);
        assert!(serde_json::from_str::<serde_json::Value>(&calls[0].1).is_err());
    }

    #[test]
    fn openai_tool_call_without_string_arguments_is_malformed() {
        let raw = serde_json::json!({"choices": [{"message": {"tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f"}},
            {"id": "c2", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
        ]}}]});
        let calls = extract_response_tool_calls(&raw);
        assert_eq!(calls.len(), 2);
        for (_, args) in &calls {
            assert!(
                serde_json::from_str::<serde_json::Value>(args).is_err(),
                "missing/non-string arguments must not parse as an empty object"
            );
        }
    }

    #[test]
    fn schema_violation_additional_properties_false() {
        let schema = json!({
            "type": "object",
            "properties": { "name": {"type": "string"} },
            "additionalProperties": false
        });
        // extra key "age" not in properties → violation
        assert!(
            schema_violation(&json!({"name": "Alice", "age": 30}), &schema, "root").is_some(),
            "extra key with additionalProperties:false must produce a violation"
        );
        // only declared keys → no violation
        assert!(
            schema_violation(&json!({"name": "Alice"}), &schema, "root").is_none(),
            "object with only declared keys must produce no violation"
        );
    }

    /// Full nested valid value → Pass.
    /// Guards: all checks passing silently together.
    #[test]
    fn schema_violation_complex_valid_nested() {
        let schema = json!({
            "type": "object",
            "required": ["id", "tags"],
            "additionalProperties": false,
            "properties": {
                "id": {"type": "integer"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        });
        assert!(
            schema_violation(
                &json!({"id": 1, "tags": ["rust", "agent"]}),
                &schema,
                "root"
            )
            .is_none(),
            "fully valid complex value must produce no violation"
        );
    }

    // ── Finding 5: malformed tool calls produce Fail, not silent Pass ──────────────────────────

    /// OpenAI tool call missing `function` → empty name → ToolArgsValidGate Fail.
    /// Guards: `extract_response_tool_calls` OpenAI missing-function branch (line 302-303).
    #[tokio::test]
    async fn tool_args_valid_fails_on_openai_tool_call_missing_function_key() {
        // An OpenAI-format tool_calls entry without a `function` key is malformed.
        // extract_response_tool_calls must record it as ("", "{}") rather than skipping it,
        // so ToolArgsValidGate can Fail the response (tool "" is not in the request schema).
        let malformed_resp = ModelResponse {
            text: String::new(),
            raw: serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            // Missing "function" key entirely — malformed call
                            {"id": "call_1", "type": "function"}
                        ]
                    }
                }]
            }),
            ..tool_use_resp()
        };
        let r = ToolArgsValidGate
            .evaluate(&tool_req_with_schema(), &malformed_resp)
            .await;
        assert_eq!(
            r.verdict,
            Verdict::Fail,
            "malformed OpenAI tool call (missing function) must Fail, not Pass"
        );
    }

    /// Anthropic tool_use missing `name` → empty name → ToolArgsValidGate Fail.
    /// Guards: `extract_response_tool_calls` Anthropic missing-name branch (line 273-275).
    #[tokio::test]
    async fn tool_args_valid_fails_on_anthropic_tool_use_missing_name() {
        let malformed_resp = ModelResponse {
            text: String::new(),
            raw: serde_json::json!({
                "content": [{
                    "type": "tool_use",
                    "id": "t1",
                    // "name" is absent — malformed
                    "input": {"q": "hello"}
                }]
            }),
            ..tool_use_resp()
        };
        let r = ToolArgsValidGate
            .evaluate(&tool_req_with_schema(), &malformed_resp)
            .await;
        assert_eq!(
            r.verdict,
            Verdict::Fail,
            "Anthropic tool_use missing name must Fail, not Pass"
        );
    }
}
