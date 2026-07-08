//! The enforce-mode escalation engine (SPEC §7.1) — the crown jewel.
//!
//! Cheapest rung first: call the model, gate the output, serve the first output that passes;
//! escalate exactly one rung on gate failure, up to a ladder/budget/`max_rungs` ceiling. A
//! failover-eligible provider error (transport / 5xx) abstains and moves to the next rung — so
//! cross-provider failover falls out of the same loop (§7.2). This is the real-typed, async
//! version of the `Firstpass` policy proven in `firstpass-bench`; the semantics are identical.

use crate::gate::{Gate, aggregate};
use crate::provider::{Auth, ModelRequest, ModelResponse, ProviderError, ProviderRegistry};
use firstpass_core::verdict::reason;
use firstpass_core::{
    Attempt, Features, FinalOutcome, GENESIS_HASH, GateResult, Mode, ModelRef, PolicyRef,
    PriceTable, RequestInfo, ServedFrom, Trace, Verdict,
};
use jiff::Timestamp;
use std::time::Instant;
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

                let gate_results: Vec<GateResult> =
                    ctx.gates.iter().map(|g| g.evaluate(&req, &resp)).collect();
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

    fn ctx<'a>(
        ladder: &'a [String],
        gates: &'a [Box<dyn Gate>],
        req: &'a ModelRequest,
        providers: &'a ProviderRegistry,
        auth: &'a Auth,
        prices: &'a PriceTable,
        budget: Option<f64>,
    ) -> EnforceCtx<'a> {
        EnforceCtx {
            ladder,
            gates,
            base_request: req,
            providers,
            auth,
            prices,
            budget_per_request_usd: budget,
            max_rungs: 3,
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
        let (out, trace) =
            route_enforce(ctx(&ladder, &gates, &req, &providers, &auth, &prices, None)).await;

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
        let (out, trace) =
            route_enforce(ctx(&ladder, &gates, &req, &providers, &auth, &prices, None)).await;

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
        let (out, trace) =
            route_enforce(ctx(&ladder, &gates, &req, &providers, &auth, &prices, None)).await;

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
        let (_out, trace) = route_enforce(ctx(
            &ladder,
            &gates,
            &req,
            &providers,
            &auth,
            &prices,
            Some(0.0),
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
        let (out, trace) =
            route_enforce(ctx(&ladder, &gates, &req, &providers, &auth, &prices, None)).await;

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
        let (out, trace) =
            route_enforce(ctx(&ladder, &gates, &req, &providers, &auth, &prices, None)).await;

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
        let (_out, trace) =
            route_enforce(ctx(&ladder, &gates, &req, &providers, &auth, &prices, None)).await;

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
        let (_out, trace) =
            route_enforce(ctx(&ladder, &gates, &req, &providers, &auth, &prices, None)).await;

        // A single trace with the genesis prev_hash must form a valid 1-long chain.
        assert!(firstpass_core::verify_chain(std::slice::from_ref(&trace), GENESIS_HASH).is_ok());
        // And it must round-trip through JSON (wire/audit contract).
        let json = serde_json::to_string(&trace).unwrap();
        let _back: Trace = serde_json::from_str(&json).unwrap();
    }
}
