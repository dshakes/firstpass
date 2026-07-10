//! The enforce-mode escalation engine (SPEC §7.1) — the crown jewel.
//!
//! Cheapest rung first: call the model, gate the output, serve the first output that passes;
//! escalate exactly one rung on gate failure, up to a ladder/budget/`max_rungs` ceiling. A
//! failover-eligible provider error (transport / 5xx) abstains and moves to the next rung — so
//! cross-provider failover falls out of the same loop (§7.2). This is the real-typed, async
//! version of the `Firstpass` policy proven in `firstpass-bench`; the semantics are identical.

use crate::gate::{Gate, GateHealthRegistry, aggregate};
use crate::provider::{Auth, ModelRequest, ModelResponse, ProviderError, ProviderRegistry};
use firstpass_core::verdict::reason;
use firstpass_core::{
    Attempt, Features, FinalOutcome, GENESIS_HASH, GateResult, Mode, ModelRef, PolicyRef,
    PriceTable, RequestInfo, ServedFrom, Trace, Verdict,
};
use jiff::Timestamp;
use std::collections::HashMap;
use std::time::Instant;
use tokio::task::JoinHandle;
use uuid::Uuid;

/// The outcome of an enforce-mode routing decision.
#[derive(Debug)]
pub enum EngineOutcome {
    /// An output was served (from a passing attempt, or the best attempt when the ladder was
    /// exhausted without a pass).
    Served(ModelResponse),
    /// Nothing could be served — every rung errored, or a hard (non-failover) error occurred.
    Failed(String),
}

/// Everything the engine needs for one decision. Borrowed to avoid cloning the ladder/request
/// per call; owned trace-context strings so the resulting [`Trace`] is self-contained.
#[derive(Debug)]
pub struct EnforceCtx<'a> {
    /// Model ladder, cheapest first, as `provider/model` strings.
    pub ladder: &'a [String],
    /// Gates run against each attempt's output (already resolved).
    pub gates: &'a [Box<dyn Gate>],
    /// Per-gate error budgets: a gate over budget is skipped (auto-disabled) this request.
    pub health: &'a GateHealthRegistry,
    /// The base request; its `model` is overwritten per rung.
    pub base_request: &'a ModelRequest,
    /// Provider lookup.
    pub providers: &'a ProviderRegistry,
    /// BYOK credentials for this request.
    pub auth: &'a Auth,
    /// Price table for cost + counterfactual math.
    pub prices: &'a PriceTable,
    /// Per-request USD cap (`None` = uncapped).
    pub budget_per_request_usd: Option<f64>,
    /// Hard ceiling on rungs attempted this request.
    pub max_rungs: u32,
    /// Prefetch depth: fire this many rungs ahead concurrently while gating in ladder order.
    /// `0` = serial (the default): one call at a time, byte-identical to the original engine.
    pub speculation: u32,
    /// Feature vector routed on (recorded in the trace).
    pub features: Features,
    /// Tenant id.
    pub tenant_id: String,
    /// Session id.
    pub session_id: String,
    /// Salted prompt hash (never the raw prompt).
    pub prompt_hash: String,
    /// Wire API label, e.g. `"anthropic.messages"`.
    pub api: String,
    /// Policy identity, e.g. `"static-ladder@v0"`.
    pub policy_id: String,
}

/// Run the enforce-mode ladder and produce both the outcome and its audit trace.
///
/// The trace's `prev_hash` is left as [`GENESIS_HASH`]; the trace-store writer overwrites it with
/// the real chain head when persisting (keeping the single-writer chain invariant).
pub async fn route_enforce(ctx: EnforceCtx<'_>) -> (EngineOutcome, Trace) {
    // Speculation is off by default (serial); the serial path is the original, proven engine, left
    // untouched. Both paths produce the same ladder state; only the tail (serve + trace) is shared.
    let LadderRun {
        attempts,
        spent,
        gate_cost_total,
        best,
        mut served_rung,
        hard_error,
    } = if ctx.speculation == 0 {
        run_serial(&ctx).await
    } else {
        run_speculative(&ctx).await
    };

    // Decide what to serve.
    let (outcome, served_from, served_tokens) = match (served_rung, &best) {
        (Some(_), Some((_, resp))) => (
            EngineOutcome::Served(resp.clone()),
            ServedFrom::Attempt,
            (resp.in_tokens, resp.out_tokens),
        ),
        (None, Some((idx, resp))) => {
            // No pass, but we produced output: serve the best (highest) attempt seen.
            served_rung = Some(*idx);
            (
                EngineOutcome::Served(resp.clone()),
                ServedFrom::BestAttempt,
                (resp.in_tokens, resp.out_tokens),
            )
        }
        (_, None) => {
            let msg = hard_error.unwrap_or_else(|| "all rungs failed".to_owned());
            (EngineOutcome::Failed(msg), ServedFrom::Error, (0, 0))
        }
    };

    // Counterfactual: what the top rung would have cost for the served token counts.
    let top_model = ctx.ladder.last().map(String::as_str).unwrap_or_default();
    let baseline = ctx
        .prices
        .cost_usd(top_model, served_tokens.0, served_tokens.1)
        .unwrap_or(spent);

    let total_latency_ms = attempts.iter().map(|a| a.latency_ms).sum();
    let escalations = attempts.len().saturating_sub(1) as u32;

    let mut trace = Trace {
        trace_id: Uuid::now_v7(),
        prev_hash: GENESIS_HASH.to_owned(),
        tenant_id: ctx.tenant_id,
        session_id: ctx.session_id,
        ts: Timestamp::now(),
        mode: Mode::Enforce,
        policy: PolicyRef {
            id: ctx.policy_id,
            explore: false,
        },
        request: RequestInfo {
            api: ctx.api,
            prompt_hash: ctx.prompt_hash,
            features: ctx.features,
        },
        attempts,
        deferred: vec![],
        final_: FinalOutcome {
            served_rung,
            served_from,
            total_cost_usd: spent,
            gate_cost_usd: gate_cost_total,
            total_latency_ms,
            escalations,
            counterfactual_baseline_usd: baseline,
            savings_usd: 0.0,
        },
    };
    trace.recompute_savings();
    (outcome, trace)
}

/// The ladder state both engine variants produce; the shared tail turns it into a served outcome
/// and audit trace. `spent`/`gate_cost_total` are running USD totals; `best` is the highest attempt
/// that produced gradable output; `served_rung` is `Some` only when a gate actually passed.
struct LadderRun {
    attempts: Vec<Attempt>,
    spent: f64,
    gate_cost_total: f64,
    best: Option<(u32, ModelResponse)>,
    served_rung: Option<u32>,
    hard_error: Option<String>,
}

/// Serial engine: one rung at a time — call, gate, serve the first pass, escalate on fail. This is
/// the original, proven loop; `speculation == 0` routes here unchanged.
async fn run_serial(ctx: &EnforceCtx<'_>) -> LadderRun {
    let mut attempts: Vec<Attempt> = Vec::new();
    let mut spent = 0.0_f64;
    let mut gate_cost_total = 0.0_f64;
    let mut best: Option<(u32, ModelResponse)> = None;
    let mut served_rung: Option<u32> = None;
    let mut hard_error: Option<String> = None;

    let rung_limit = (ctx.max_rungs as usize).min(ctx.ladder.len());
    for (idx, model_str) in ctx.ladder.iter().take(rung_limit).enumerate() {
        let idx = idx as u32;
        let start = Instant::now();

        // Resolve provider from `provider/model`. A missing provider/malformed ref is treated as
        // a failover-eligible abstain: record it and try the next rung rather than hard-failing.
        let provider = match ModelRef::parse(model_str) {
            Ok(m) => ctx.providers.get(&m.provider),
            Err(_) => None,
        };
        let Some(provider) = provider else {
            let ms = elapsed_ms(start);
            attempts.push(abstain_attempt(
                idx,
                model_str,
                "unknown",
                reason::PROVIDER_ERROR,
                ms,
            ));
            continue;
        };

        let mut req = ctx.base_request.clone();
        req.model = model_str.clone();

        match provider.complete(&req, ctx.auth).await {
            Err(err) if err.is_failover_eligible() => {
                // Transport / 5xx: abstain and fail over to the next rung.
                let ms = elapsed_ms(start);
                attempts.push(abstain_attempt(
                    idx,
                    model_str,
                    provider.id(),
                    reason::PROVIDER_ERROR,
                    ms,
                ));
                continue;
            }
            Err(err) => {
                // Hard error (4xx / decode): do not escalate — the request itself is the problem.
                let ms = elapsed_ms(start);
                let (r, msg) = hard_reason(&err);
                attempts.push(abstain_attempt(idx, model_str, provider.id(), r, ms));
                hard_error = Some(msg);
                break;
            }
            Ok(resp) => {
                let ms = elapsed_ms(start);
                let model_cost = ctx
                    .prices
                    .cost_usd(model_str, resp.in_tokens, resp.out_tokens)
                    .unwrap_or(0.0);
                spent += model_cost;

                // Run gates sequentially (they're I/O — subprocess/model), skipping any the
                // error budget has auto-disabled, and feeding each outcome back to the budget.
                let mut gate_results: Vec<GateResult> = Vec::with_capacity(ctx.gates.len());
                for g in ctx.gates {
                    if !ctx.health.enabled(g.id()) {
                        tracing::warn!(gate = %g.id(), "skipping auto-disabled gate");
                        continue;
                    }
                    let r = g.evaluate(&req, &resp).await;
                    ctx.health.record(g.id(), r.verdict == Verdict::Abstain);
                    gate_results.push(r);
                }
                let gc: f64 = gate_results.iter().map(|g| g.cost_usd).sum();
                gate_cost_total += gc;
                spent += gc;

                let verdict = aggregate(&gate_results);
                attempts.push(Attempt {
                    rung: idx,
                    model: model_str.clone(),
                    provider: provider.id().to_owned(),
                    in_tokens: resp.in_tokens,
                    out_tokens: resp.out_tokens,
                    cost_usd: model_cost,
                    latency_ms: ms,
                    gates: gate_results,
                    verdict,
                });
                best = Some((idx, resp));

                if verdict == Verdict::Pass {
                    served_rung = Some(idx);
                    break;
                }
                // Gate failed → escalate, unless the budget is already spent and a next rung exists.
                if let Some(cap) = ctx.budget_per_request_usd
                    && spent >= cap
                    && (idx as usize) + 1 < rung_limit
                {
                    break;
                }
            }
        }
    }

    LadderRun {
        attempts,
        spent,
        gate_cost_total,
        best,
        served_rung,
        hard_error,
    }
}

/// Speculative engine: prefetch up to `speculation` rungs ahead concurrently, but gate strictly in
/// ladder order and serve the first rung whose gate passes. The SERVED result is therefore
/// byte-identical to [`run_serial`] — only latency (prefetched rungs are already in flight) and
/// honest wasted spend (speculative calls that completed but weren't served) differ.
async fn run_speculative(ctx: &EnforceCtx<'_>) -> LadderRun {
    let mut attempts: Vec<Attempt> = Vec::new();
    let mut spent = 0.0_f64;
    let mut gate_cost_total = 0.0_f64;
    let mut best: Option<(u32, ModelResponse)> = None;
    let mut served_rung: Option<u32> = None;
    let mut hard_error: Option<String> = None;

    let rung_limit = (ctx.max_rungs as usize).min(ctx.ladder.len());
    let speculation = ctx.speculation as usize;
    let mut inflight: HashMap<usize, JoinHandle<Result<ModelResponse, ProviderError>>> =
        HashMap::new();

    let mut idx = 0usize;
    // `done` = a rung passed or hard-errored: stop consuming, then cancel/harvest the rest.
    let mut done = false;
    while idx < rung_limit && !done {
        // Fire the window [idx ..= idx+speculation] concurrently. The rung we must gate now (idx)
        // always fires; rungs ahead only while under budget, so speculation can't blow the cap.
        let window_end = (idx + speculation).min(rung_limit - 1);
        for j in idx..=window_end {
            if inflight.contains_key(&j) {
                continue;
            }
            if j > idx
                && let Some(cap) = ctx.budget_per_request_usd
                && spent >= cap
            {
                continue;
            }
            if let Some(handle) = spawn_rung(ctx, j) {
                inflight.insert(j, handle);
            }
        }

        let model_str = &ctx.ladder[idx];
        let provider = match ModelRef::parse(model_str) {
            Ok(m) => ctx.providers.get(&m.provider),
            Err(_) => None,
        };
        let Some(provider) = provider else {
            // Malformed ref / unknown provider: abstain and fail over (no task was spawned).
            attempts.push(abstain_attempt(
                idx as u32,
                model_str,
                "unknown",
                reason::PROVIDER_ERROR,
                0,
            ));
            idx += 1;
            continue;
        };
        // Provider resolved ⇒ we spawned a task for `idx`; await it in strict ladder order.
        let Some(handle) = inflight.remove(&idx) else {
            attempts.push(abstain_attempt(
                idx as u32,
                model_str,
                provider.id(),
                reason::PROVIDER_ERROR,
                0,
            ));
            idx += 1;
            continue;
        };
        let start = Instant::now();
        let joined = handle.await;
        let ms = elapsed_ms(start);

        match joined {
            // Task panicked or was aborted out from under us: treat as a transport abstain.
            Err(_) => {
                attempts.push(abstain_attempt(
                    idx as u32,
                    model_str,
                    provider.id(),
                    reason::PROVIDER_ERROR,
                    ms,
                ));
                idx += 1;
            }
            Ok(Err(err)) if err.is_failover_eligible() => {
                attempts.push(abstain_attempt(
                    idx as u32,
                    model_str,
                    provider.id(),
                    reason::PROVIDER_ERROR,
                    ms,
                ));
                idx += 1;
            }
            Ok(Err(err)) => {
                // Hard error (4xx / decode): do not escalate — the request itself is the problem.
                let (r, msg) = hard_reason(&err);
                attempts.push(abstain_attempt(idx as u32, model_str, provider.id(), r, ms));
                hard_error = Some(msg);
                done = true;
            }
            Ok(Ok(resp)) => {
                let model_cost = ctx
                    .prices
                    .cost_usd(model_str, resp.in_tokens, resp.out_tokens)
                    .unwrap_or(0.0);
                spent += model_cost;

                let mut req = ctx.base_request.clone();
                req.model = model_str.clone();
                let mut gate_results: Vec<GateResult> = Vec::with_capacity(ctx.gates.len());
                for g in ctx.gates {
                    if !ctx.health.enabled(g.id()) {
                        tracing::warn!(gate = %g.id(), "skipping auto-disabled gate");
                        continue;
                    }
                    let r = g.evaluate(&req, &resp).await;
                    ctx.health.record(g.id(), r.verdict == Verdict::Abstain);
                    gate_results.push(r);
                }
                let gc: f64 = gate_results.iter().map(|g| g.cost_usd).sum();
                gate_cost_total += gc;
                spent += gc;

                let verdict = aggregate(&gate_results);
                attempts.push(Attempt {
                    rung: idx as u32,
                    model: model_str.clone(),
                    provider: provider.id().to_owned(),
                    in_tokens: resp.in_tokens,
                    out_tokens: resp.out_tokens,
                    cost_usd: model_cost,
                    latency_ms: ms,
                    gates: gate_results,
                    verdict,
                });
                best = Some((idx as u32, resp));

                if verdict == Verdict::Pass {
                    served_rung = Some(idx as u32);
                    done = true;
                } else if let Some(cap) = ctx.budget_per_request_usd
                    && spent >= cap
                    && idx + 1 < rung_limit
                {
                    done = true;
                }
                idx += 1;
            }
        }
    }

    // Speculative rungs we never gated: those already finished DID bill us (honest waste, recorded
    // in `spent`); those still in flight are cancelled — `abort()` drops the in-flight reqwest.
    // ponytail: harvest is best-effort — a call that finishes between is_finished() and abort() is
    // dropped uncounted; exact wasted-spend under cancellation is unknowable, don't fabricate it.
    for (j, handle) in inflight.drain() {
        if handle.is_finished() {
            if let Ok(Ok(resp)) = handle.await {
                spent += ctx
                    .prices
                    .cost_usd(&ctx.ladder[j], resp.in_tokens, resp.out_tokens)
                    .unwrap_or(0.0);
            }
        } else {
            handle.abort();
        }
    }

    LadderRun {
        attempts,
        spent,
        gate_cost_total,
        best,
        served_rung,
        hard_error,
    }
}

/// Spawn a rung's `complete()` as a concurrent task, or `None` if the model ref is malformed or its
/// provider isn't registered (the consume path records that abstain in ladder order).
fn spawn_rung(
    ctx: &EnforceCtx<'_>,
    j: usize,
) -> Option<JoinHandle<Result<ModelResponse, ProviderError>>> {
    let model_str = ctx.ladder.get(j)?;
    let provider = match ModelRef::parse(model_str) {
        Ok(m) => ctx.providers.get(&m.provider)?,
        Err(_) => return None,
    };
    let mut req = ctx.base_request.clone();
    req.model = model_str.clone();
    let auth = ctx.auth.clone();
    Some(tokio::spawn(
        async move { provider.complete(&req, &auth).await },
    ))
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// An attempt that produced no gradable output (provider error / missing provider).
fn abstain_attempt(rung: u32, model: &str, provider: &str, reason: &str, ms: u64) -> Attempt {
    Attempt {
        rung,
        model: model.to_owned(),
        provider: provider.to_owned(),
        in_tokens: 0,
        out_tokens: 0,
        cost_usd: 0.0,
        latency_ms: ms,
        gates: vec![GateResult::abstain(provider, reason, ms)],
        verdict: Verdict::Abstain,
    }
}

/// Map a hard (non-failover) provider error to an abstain reason + a caller-facing message.
fn hard_reason(err: &ProviderError) -> (&'static str, String) {
    match err {
        ProviderError::Http { status, .. } => {
            (reason::PROVIDER_ERROR, format!("upstream http {status}"))
        }
        ProviderError::Decode(_) => (reason::PROVIDER_ERROR, "upstream decode error".to_owned()),
        ProviderError::Transport(_) => (
            reason::PROVIDER_ERROR,
            "upstream transport error".to_owned(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{JsonValidGate, NonEmptyGate};
    use crate::provider::{MockProvider, Provider};
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    const HAIKU: &str = "anthropic/claude-haiku-4-5";
    const SONNET: &str = "anthropic/claude-sonnet-5";
    const OPUS: &str = "anthropic/claude-opus-4-8";
    const GPT: &str = "openai/gpt-5.5";

    fn resp(model: &str, text: &str) -> ModelResponse {
        ModelResponse {
            model: model.to_owned(),
            text: text.to_owned(),
            in_tokens: 1000,
            out_tokens: 500,
            raw: Value::Null,
        }
    }

    fn base_request() -> ModelRequest {
        ModelRequest {
            model: String::new(),
            system: None,
            messages: vec![crate::provider::ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            }],
            max_tokens: 256,
            tools: Value::Null,
        }
    }

    /// Build a registry where each provider id answers a per-model outcome map.
    fn registry(
        outcomes: Vec<(&str, &str, Result<ModelResponse, ProviderError>)>,
    ) -> ProviderRegistry {
        let mut by_provider: HashMap<
            String,
            HashMap<String, Result<ModelResponse, ProviderError>>,
        > = HashMap::new();
        for (provider, model, out) in outcomes {
            by_provider
                .entry(provider.to_owned())
                .or_default()
                .insert(model.to_owned(), out);
        }
        let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        for (pid, outs) in by_provider {
            map.insert(pid.clone(), Arc::new(MockProvider::new(pid, outs)));
        }
        ProviderRegistry::from_map(map)
    }

    #[allow(clippy::too_many_arguments)]
    fn ctx<'a>(
        ladder: &'a [String],
        gates: &'a [Box<dyn Gate>],
        req: &'a ModelRequest,
        providers: &'a ProviderRegistry,
        auth: &'a Auth,
        prices: &'a PriceTable,
        budget: Option<f64>,
        health: &'a GateHealthRegistry,
    ) -> EnforceCtx<'a> {
        EnforceCtx {
            ladder,
            gates,
            health,
            base_request: req,
            providers,
            auth,
            prices,
            budget_per_request_usd: budget,
            max_rungs: 3,
            speculation: 0,
            features: Features::new(firstpass_core::TaskKind::CodeEdit),
            tenant_id: "acme".into(),
            session_id: "sess-1".into(),
            prompt_hash: "deadbeef".into(),
            api: "anthropic.messages".into(),
            policy_id: "static-ladder@v0".into(),
        }
    }

    #[tokio::test]
    async fn serve_first_pass_no_escalation_with_savings() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned(), OPUS.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let providers = registry(vec![("anthropic", HAIKU, Ok(resp(HAIKU, r#"{"ok":1}"#)))]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;

        match out {
            EngineOutcome::Served(r) => assert_eq!(r.model, HAIKU),
            EngineOutcome::Failed(e) => panic!("expected served, got {e}"),
        }
        assert_eq!(trace.attempts.len(), 1);
        assert_eq!(trace.final_.escalations, 0);
        assert_eq!(trace.final_.served_from, ServedFrom::Attempt);
        assert_eq!(trace.final_.served_rung, Some(0));
        assert!(
            trace.final_.savings_usd > 0.0,
            "top-rung baseline should exceed haiku cost"
        );
    }

    #[tokio::test]
    async fn escalate_on_gate_fail() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        // Haiku returns empty (fails non-empty); Sonnet returns text (passes).
        let providers = registry(vec![
            ("anthropic", HAIKU, Ok(resp(HAIKU, "   "))),
            ("anthropic", SONNET, Ok(resp(SONNET, "real answer"))),
        ]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;

        assert!(matches!(out, EngineOutcome::Served(r) if r.model == SONNET));
        assert_eq!(trace.attempts.len(), 2);
        assert_eq!(trace.attempts[0].verdict, Verdict::Fail);
        assert_eq!(trace.attempts[1].verdict, Verdict::Pass);
        assert_eq!(trace.final_.escalations, 1);
        assert_eq!(trace.final_.served_rung, Some(1));
    }

    #[tokio::test]
    async fn cross_provider_failover_on_transport_error() {
        // Rung 0 is anthropic (transport error), rung 1 is openai (succeeds).
        let ladder = vec![HAIKU.to_owned(), GPT.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let providers = registry(vec![
            (
                "anthropic",
                HAIKU,
                Err(ProviderError::Transport("connection refused".into())),
            ),
            ("openai", GPT, Ok(resp(GPT, "answer from openai"))),
        ]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;

        assert!(matches!(out, EngineOutcome::Served(r) if r.model == GPT));
        assert_eq!(trace.attempts[0].verdict, Verdict::Abstain);
        assert_eq!(
            trace.attempts[0].gates[0].reason.as_deref(),
            Some(reason::PROVIDER_ERROR)
        );
        assert_eq!(trace.attempts[1].verdict, Verdict::Pass);
        assert_eq!(trace.final_.served_rung, Some(1));
    }

    #[tokio::test]
    async fn budget_cap_stops_escalation() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned(), OPUS.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        // All fail the gate (empty), so it would climb — but a tiny budget cuts it short.
        let providers = registry(vec![
            ("anthropic", HAIKU, Ok(resp(HAIKU, ""))),
            ("anthropic", SONNET, Ok(resp(SONNET, ""))),
            ("anthropic", OPUS, Ok(resp(OPUS, ""))),
        ]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (_out, trace) = route_enforce(ctx(
            &ladder,
            &gates,
            &req,
            &providers,
            &auth,
            &prices,
            Some(0.0),
            &health,
        ))
        .await;
        assert!(
            trace.attempts.len() < 3,
            "budget should cut escalation short, got {}",
            trace.attempts.len()
        );
    }

    #[tokio::test]
    async fn all_fail_serves_best_attempt() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(JsonValidGate)]; // demand JSON
        let req = base_request();
        let providers = registry(vec![
            ("anthropic", HAIKU, Ok(resp(HAIKU, "not json"))),
            ("anthropic", SONNET, Ok(resp(SONNET, "still not json"))),
        ]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;

        assert!(
            matches!(out, EngineOutcome::Served(r) if r.model == SONNET),
            "serves highest attempt"
        );
        assert_eq!(trace.final_.served_from, ServedFrom::BestAttempt);
        assert_eq!(trace.final_.served_rung, Some(1));
    }

    #[tokio::test]
    async fn hard_4xx_does_not_escalate() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let providers = registry(vec![
            (
                "anthropic",
                HAIKU,
                Err(ProviderError::Http {
                    status: 400,
                    body: "bad request".into(),
                }),
            ),
            ("anthropic", SONNET, Ok(resp(SONNET, "would have worked"))),
        ]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;

        assert!(
            matches!(out, EngineOutcome::Failed(_)),
            "4xx is a hard error, not failover"
        );
        assert_eq!(
            trace.attempts.len(),
            1,
            "must not escalate past a client error"
        );
        assert_eq!(trace.final_.served_from, ServedFrom::Error);
    }

    #[tokio::test]
    async fn counterfactual_and_savings_math() {
        let ladder = vec![HAIKU.to_owned(), OPUS.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let served = resp(HAIKU, "answer");
        let (in_t, out_t) = (served.in_tokens, served.out_tokens);
        let providers = registry(vec![("anthropic", HAIKU, Ok(served))]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (_out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;

        let expected_baseline = prices.cost_usd(OPUS, in_t, out_t).unwrap();
        assert!((trace.final_.counterfactual_baseline_usd - expected_baseline).abs() < 1e-12);
        let expected_savings = expected_baseline - trace.final_.total_cost_usd;
        assert!((trace.final_.savings_usd - expected_savings).abs() < 1e-12);
        assert!(trace.final_.savings_usd > 0.0);
    }

    #[tokio::test]
    async fn produced_trace_is_chain_verifiable() {
        let ladder = vec![HAIKU.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let providers = registry(vec![("anthropic", HAIKU, Ok(resp(HAIKU, "ok")))]);
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        let health = GateHealthRegistry::new();
        let (_out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;

        // A single trace with the genesis prev_hash must form a valid 1-long chain.
        assert!(firstpass_core::verify_chain(std::slice::from_ref(&trace), GENESIS_HASH).is_ok());
        // And it must round-trip through JSON (wire/audit contract).
        let json = serde_json::to_string(&trace).unwrap();
        let _back: Trace = serde_json::from_str(&json).unwrap();
    }

    #[tokio::test]
    async fn auto_disabled_gate_is_skipped_by_the_engine() {
        // An empty candidate would FAIL the non-empty gate — but once that gate is auto-disabled
        // (over its error budget), the engine skips it, so rung 0 serves with no gate verdict.
        let ladder = vec![HAIKU.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let providers = registry(vec![("anthropic", HAIKU, Ok(resp(HAIKU, "")))]); // empty text
        let (auth, prices) = (Auth::default(), PriceTable::defaults());

        // Drive the "non-empty" budget over threshold so it is disabled before the run.
        let health = GateHealthRegistry::new().with_budget("non-empty", 4, 0.5);
        for _ in 0..4 {
            health.record("non-empty", true);
        }
        assert!(
            !health.enabled("non-empty"),
            "precondition: gate is auto-disabled"
        );

        let (out, trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;
        assert!(matches!(out, EngineOutcome::Served(_)));
        assert_eq!(trace.final_.served_rung, Some(0));
        assert!(
            trace.attempts[0].gates.is_empty(),
            "disabled gate must be skipped, not run"
        );
    }

    /// Like [`registry`], but every model is served by one `anthropic` mock, and its shared call
    /// log is returned so a test can see which rungs `complete()` actually fired.
    fn counted_registry(
        outcomes: Vec<(&str, Result<ModelResponse, ProviderError>)>,
    ) -> (
        ProviderRegistry,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        let mut outs = HashMap::new();
        for (model, out) in outcomes {
            outs.insert(model.to_owned(), out);
        }
        let mock = MockProvider::new("anthropic", outs);
        let log = mock.call_log();
        let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        map.insert("anthropic".to_owned(), Arc::new(mock));
        (ProviderRegistry::from_map(map), log)
    }

    #[tokio::test]
    async fn speculation_prefetches_next_rung_but_serves_identically() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let (auth, prices) = (Auth::default(), PriceTable::defaults());

        // Serial baseline: rung 0 passes → rung 1 is never even called.
        let (providers, log) = counted_registry(vec![
            (HAIKU, Ok(resp(HAIKU, "answer"))),
            (SONNET, Ok(resp(SONNET, "other"))),
        ]);
        let health = GateHealthRegistry::new();
        let (serial_out, serial_trace) = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;
        assert_eq!(
            *log.lock().unwrap(),
            vec![HAIKU.to_owned()],
            "serial must not touch rung 1 when rung 0 passes"
        );

        // Speculative (k=1): rung 1 fires concurrently, but rung 0 still serves.
        let (providers, log) = counted_registry(vec![
            (HAIKU, Ok(resp(HAIKU, "answer"))),
            (SONNET, Ok(resp(SONNET, "other"))),
        ]);
        let health = GateHealthRegistry::new();
        let mut c = ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        );
        c.speculation = 1;
        let (spec_out, spec_trace) = route_enforce(c).await;

        assert!(
            log.lock().unwrap().contains(&SONNET.to_owned()),
            "speculation must fire rung 1 ahead: {:?}",
            *log.lock().unwrap()
        );

        // Served result is byte-identical to serial (same rung, same bytes).
        let (a, b) = match (serial_out, spec_out) {
            (EngineOutcome::Served(a), EngineOutcome::Served(b)) => (a, b),
            _ => panic!("both variants must serve"),
        };
        assert_eq!(
            (a.model, a.text, a.out_tokens),
            (b.model, b.text, b.out_tokens)
        );
        assert_eq!(spec_trace.final_.served_rung, Some(0));
        assert_eq!(spec_trace.attempts.len(), 1, "only rung 0 is gated");
        // Honest waste: the completed speculative rung's cost is recorded in the total.
        assert!(
            spec_trace.final_.total_cost_usd > serial_trace.final_.total_cost_usd,
            "speculative waste must show in total cost: spec={} serial={}",
            spec_trace.final_.total_cost_usd,
            serial_trace.final_.total_cost_usd
        );
    }

    #[tokio::test]
    async fn speculation_preserves_escalation_result() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        // Rung 0 empty (fails non-empty), rung 1 real (passes) — prefetched concurrently.
        let (providers, _log) = counted_registry(vec![
            (HAIKU, Ok(resp(HAIKU, ""))),
            (SONNET, Ok(resp(SONNET, "real answer"))),
        ]);
        let health = GateHealthRegistry::new();
        let mut c = ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        );
        c.speculation = 2; // window wider than the ladder must clamp, not panic
        let (out, trace) = route_enforce(c).await;
        match out {
            EngineOutcome::Served(r) => assert_eq!(r.model, SONNET),
            EngineOutcome::Failed(e) => panic!("expected served, got {e}"),
        }
        assert_eq!(trace.final_.served_rung, Some(1));
        assert_eq!(trace.attempts.len(), 2);
        assert_eq!(trace.final_.escalations, 1);
        assert_eq!(trace.attempts[0].verdict, Verdict::Fail);
        assert_eq!(trace.attempts[1].verdict, Verdict::Pass);
    }

    #[tokio::test]
    async fn speculation_never_fires_past_max_rungs() {
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned(), OPUS.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let (auth, prices) = (Auth::default(), PriceTable::defaults());
        // All fail the gate → best-attempt fallback serves the highest reached rung.
        let (providers, log) = counted_registry(vec![
            (HAIKU, Ok(resp(HAIKU, ""))),
            (SONNET, Ok(resp(SONNET, ""))),
            (OPUS, Ok(resp(OPUS, ""))),
        ]);
        let health = GateHealthRegistry::new();
        let mut c = ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        );
        c.max_rungs = 2;
        c.speculation = 5; // huge window, but the ceiling is 2 rungs
        let (out, trace) = route_enforce(c).await;

        assert!(
            !log.lock().unwrap().contains(&OPUS.to_owned()),
            "must not fire beyond max_rungs: {:?}",
            *log.lock().unwrap()
        );
        assert_eq!(trace.attempts.len(), 2);
        assert_eq!(trace.final_.served_from, ServedFrom::BestAttempt);
        assert_eq!(trace.final_.served_rung, Some(1));
        match out {
            EngineOutcome::Served(r) => assert_eq!(r.model, SONNET),
            EngineOutcome::Failed(e) => panic!("expected best-attempt served, got {e}"),
        }
    }

    #[tokio::test]
    async fn speculation_cuts_wall_clock_vs_serial() {
        // The latency payoff, verified offline: rung 0 fails the gate, so serial pays rung 0 + rung
        // 1 latency *sequentially*; speculation fires both concurrently and finishes in ~one rung's
        // time. Timing-based, but the margin (a full 80ms rung) dwarfs scheduler jitter. This proves
        // the overlap mechanism — absolute live p95 still needs a real-provider run.
        let ladder = vec![HAIKU.to_owned(), SONNET.to_owned()];
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(NonEmptyGate)];
        let req = base_request();
        let (auth, prices) = (Auth::default(), PriceTable::defaults());

        let build = || {
            let mut outs = HashMap::new();
            outs.insert(HAIKU.to_owned(), Ok(resp(HAIKU, ""))); // fails non-empty
            outs.insert(SONNET.to_owned(), Ok(resp(SONNET, "real answer"))); // passes
            let mock = MockProvider::new("anthropic", outs).with_delay(80);
            let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
            map.insert("anthropic".to_owned(), Arc::new(mock));
            ProviderRegistry::from_map(map)
        };

        let providers = build();
        let health = GateHealthRegistry::new();
        let t = std::time::Instant::now();
        let _ = route_enforce(ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        ))
        .await;
        let serial = t.elapsed();

        let providers = build();
        let health = GateHealthRegistry::new();
        let mut c = ctx(
            &ladder, &gates, &req, &providers, &auth, &prices, None, &health,
        );
        c.speculation = 1;
        let t = std::time::Instant::now();
        let _ = route_enforce(c).await;
        let spec = t.elapsed();

        assert!(
            spec < serial * 3 / 4,
            "speculation must overlap rung latencies: serial={serial:?} spec={spec:?}"
        );
    }
}
