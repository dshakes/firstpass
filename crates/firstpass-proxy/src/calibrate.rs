//! Recalibrate the serving threshold from real deferred feedback (SPEC §10.1, run against live
//! traffic instead of a static benchmark suite) — the "learns your quality bar" loop.
//!
//! Two calibration methods are available:
//! - **conformal** (default): split-conformal with Hoeffding bound — [`calibrate_from_store`].
//! - **ltt**: Learn-then-Test / RCPS with exact-binomial fixed-sequence testing —
//!   [`calibrate_from_store_ltt`].
//!
//! Both enumerate stored traces, pair each trace that has a deferred outcome with the score of
//! the attempt actually served, and hand the pairs to the respective core module. Neither feeds
//! back into the request hot path — that wiring is a deliberate follow-on once an operator has
//! reviewed a report.

use std::path::Path;

use firstpass_core::conformal::{self, ConformalResult};
use firstpass_core::ltt::{self, LttResult};
use firstpass_core::{Attempt, DeferredVerdict, GateResult, Score, Trace, Verdict};

use crate::store::{self, StoreError};

/// The result of calibrating a conformal threshold against real deferred feedback.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    /// Number of `(score, correct)` pairs calibration ran on — one per trace with at least one
    /// deferred verdict recorded.
    pub n_pairs: usize,
    /// The conformal calibration result (threshold, feasibility, calibration risk).
    pub conformal: ConformalResult,
    /// Empirical served-failure rate at `conformal.threshold`, measured on the same pairs used
    /// to calibrate (a sanity check, not a held-out estimate — the proxy doesn't yet split
    /// feedback into separate calibration/test batches).
    pub empirical_served_failure: f64,
    /// How many pairs would be served at the calibrated threshold.
    pub n_served: usize,
}

impl CalibrationReport {
    /// Render the report as human-readable lines for `firstpass calibrate`.
    #[must_use]
    pub fn render(&self) -> String {
        format!(
            "pairs: {n_pairs} ({n_served} served at threshold)\n\
             threshold: {threshold:.4}\n\
             feasible: {feasible}\n\
             target alpha: {alpha:.4} (delta {delta:.4})\n\
             calibration risk: {calib_risk:.4}\n\
             empirical served-failure: {empirical:.4}\n",
            n_pairs = self.n_pairs,
            n_served = self.n_served,
            threshold = self.conformal.threshold,
            feasible = self.conformal.feasible,
            alpha = self.conformal.alpha,
            delta = self.conformal.delta,
            calib_risk = self.conformal.calib_risk,
            empirical = self.empirical_served_failure,
        )
    }
}

/// Calibrate a conformal threshold from `(score, correct)` pairs — a thin wrapper over
/// [`firstpass_core::conformal`] that also reports the empirical served-failure at the chosen
/// threshold.
#[must_use]
pub fn calibrate_pairs(
    pairs: &[(f64, bool)],
    alpha: f64,
    delta: f64,
    min_n: usize,
) -> CalibrationReport {
    let result = conformal::calibrate(pairs, alpha, delta, min_n);
    let (empirical_served_failure, n_served) =
        conformal::served_failure_rate(pairs, result.threshold);
    CalibrationReport {
        n_pairs: pairs.len(),
        conformal: result,
        empirical_served_failure,
        n_served,
    }
}

/// The aggregate score for a set of gate results at a given verdict: the mean of the numeric gate
/// scores, or — when no gate reported a numeric score at all — `1.0` if it passed and `0.0` if it
/// didn't. A bare pass/fail with no score still needs to sit somewhere on the `[0, 1]` axis
/// conformal thresholds against; treating a scoreless pass as maximally confident and a scoreless
/// fail as minimally confident keeps "higher score = more servable" true either way.
///
/// Shared by [`attempt_score`] (calibration, offline) and the router's `serve_threshold` decision
/// (serving, online) so the two agree on what "the score" means.
pub(crate) fn gate_score(gates: &[GateResult], verdict: Verdict) -> f64 {
    let numeric: Vec<f64> = gates
        .iter()
        .filter_map(|g| g.score.map(Score::value))
        .collect();
    if numeric.is_empty() {
        f64::from(verdict == Verdict::Pass)
    } else {
        numeric.iter().sum::<f64>() / numeric.len() as f64
    }
}

/// The aggregate score for a served attempt (see [`gate_score`]).
fn attempt_score(attempt: &Attempt) -> f64 {
    gate_score(&attempt.gates, attempt.verdict)
}

/// Build a `(score, correct)` pair for one trace, if it has deferred feedback and a served
/// attempt. `correct` is whether the MOST RECENT deferred verdict for the trace is `Pass` (later
/// feedback supersedes earlier — e.g. a flaky CI run retried).
fn trace_pair(trace: &Trace, deferred: &[DeferredVerdict]) -> Option<(f64, bool)> {
    let last = deferred.last()?;
    let served_rung = trace.final_.served_rung?;
    let attempt = trace.attempts.iter().find(|a| a.rung == served_rung)?;
    Some((attempt_score(attempt), last.verdict == Verdict::Pass))
}

/// The result of LTT calibration against real deferred feedback.
#[derive(Debug, Clone)]
pub struct LttReport {
    /// Number of `(score, correct)` pairs — one per trace with a deferred verdict.
    pub n_pairs: usize,
    /// The LTT calibration result (threshold, feasibility, empirical risk, diagnostics).
    pub ltt: LttResult,
}

impl LttReport {
    /// Render the report as human-readable lines for `firstpass calibrate --method ltt`.
    /// Format mirrors [`CalibrationReport::render`] with an added verifier ROC note.
    #[must_use]
    pub fn render(&self) -> String {
        let far = match self.ltt.false_accept_rate {
            Some(r) => format!("{r:.4}"),
            None => "N/A (no incorrect items in calibration set)".to_owned(),
        };
        format!(
            "method: ltt\n\
             pairs: {n_pairs} ({n_served} served at threshold)\n\
             threshold: {threshold:.4}\n\
             feasible: {feasible}\n\
             target alpha: {alpha:.4} (delta {delta:.4})\n\
             empirical risk: {risk:.4}\n\
             false-accept rate: {far}  (P(score >= lambda | incorrect); verifier ROC point)\n",
            n_pairs = self.n_pairs,
            n_served = self.ltt.n_served,
            threshold = self.ltt.threshold,
            feasible = self.ltt.feasible,
            alpha = self.ltt.alpha,
            delta = self.ltt.delta,
            risk = self.ltt.empirical_risk,
        )
    }
}

/// Calibrate an LTT threshold from `(score, correct)` pairs — thin wrapper over
/// [`firstpass_core::ltt`].
#[must_use]
pub fn calibrate_pairs_ltt(
    pairs: &[(f64, bool)],
    alpha: f64,
    delta: f64,
    min_n: usize,
) -> LttReport {
    LttReport {
        n_pairs: pairs.len(),
        ltt: ltt::calibrate(pairs, alpha, delta, min_n),
    }
}

/// Calibrate an LTT threshold from every trace in the store that has a deferred outcome.
///
/// Error handling and tenant scoping match [`calibrate_from_store`] exactly.
///
/// # Errors
/// Returns [`StoreError`] if a stored trace's deferred verdicts cannot be read.
pub fn calibrate_from_store_ltt(
    db_path: impl AsRef<Path>,
    tenant: &str,
    alpha: f64,
    delta: f64,
    min_n: usize,
) -> Result<LttReport, StoreError> {
    let traces = store::load_tenant_traces(&db_path, tenant).unwrap_or_default();
    let mut pairs = Vec::with_capacity(traces.len());
    for trace in &traces {
        let deferred = store::load_deferred(&db_path, &trace.trace_id.to_string())?;
        if let Some(pair) = trace_pair(trace, &deferred) {
            pairs.push(pair);
        }
    }
    Ok(calibrate_pairs_ltt(&pairs, alpha, delta, min_n))
}

/// Calibrate a conformal threshold from every trace in the store that has a deferred outcome
/// recorded.
///
/// # Errors
/// Returns [`StoreError`] if a stored trace's deferred verdicts cannot be read. An unreadable or
/// not-yet-initialized store is treated as zero traces (a 0-pair, infeasible report), matching the
/// forgiving behaviour of `firstpass trace` — calibrating before any traffic is a valid state, not
/// an error.
pub fn calibrate_from_store(
    db_path: impl AsRef<Path>,
    tenant: &str,
    alpha: f64,
    delta: f64,
    min_n: usize,
) -> Result<CalibrationReport, StoreError> {
    // Tenant-scoped (ADR 0004 §D3): a tenant only ever calibrates against its own feedback. The
    // per-trace `load_deferred` below is safe unscoped because every `trace` here already belongs
    // to `tenant`.
    let traces = store::load_tenant_traces(&db_path, tenant).unwrap_or_default();
    let mut pairs = Vec::with_capacity(traces.len());
    for trace in &traces {
        let deferred = store::load_deferred(&db_path, &trace.trace_id.to_string())?;
        if let Some(pair) = trace_pair(trace, &deferred) {
            pairs.push(pair);
        }
    }
    Ok(calibrate_pairs(&pairs, alpha, delta, min_n))
}

#[cfg(test)]
mod tests {
    use firstpass_core::{
        Features, FinalOutcome, GENESIS_HASH, GateResult, Mode, PolicyRef, RequestInfo, ServedFrom,
        TaskKind,
    };

    use super::*;
    use crate::store;

    /// A minimal trace serving rung 0 with a single deterministic gate score, mirroring
    /// `store::sample_trace` but with a caller-chosen score.
    fn trace_with_score(score: f64) -> Trace {
        let verdict = if score >= 0.5 {
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        let attempt = Attempt {
            rung: 0,
            model: "claude-haiku-4-5".to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 10,
            out_tokens: 5,
            cost_usd: 0.001,
            latency_ms: 12,
            gates: vec![GateResult {
                gate_id: "gate@v1".to_owned(),
                verdict,
                score: Some(Score::clamped(score)),
                cost_usd: 0.0,
                ms: 10,
                reason: None,
                evidence_ref: None,
            }],
            verdict,
        };
        let mut trace = Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: "tenant-a".to_owned(),
            session_id: "session-1".to_owned(),
            ts: jiff::Timestamp::now(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "test@v0".to_owned(),
                explore: false,
                propensity: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "deadbeef".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![attempt],
            deferred: Vec::new(),
            final_: FinalOutcome {
                served_rung: Some(0),
                served_from: ServedFrom::Attempt,
                total_cost_usd: 0.001,
                gate_cost_usd: 0.0,
                total_latency_ms: 12,
                escalations: 0,
                counterfactual_baseline_usd: 0.001,
                savings_usd: 0.0,
            },
        };
        trace.recompute_savings();
        trace
    }

    #[test]
    fn calibrate_pairs_finds_a_feasible_threshold_on_clean_pairs() {
        // Scores cleanly separate correct (>=0.7) from incorrect (<0.3). alpha=0.2 tolerates
        // some incorrect items being served, so conformal maximizes coverage — not just
        // separation — up to that budget; alpha=0.2 also keeps the Hoeffding slack satisfiable
        // at this sample size (min_n=30 wants a workable n, not the hundreds needed to certify
        // alpha=0.1 at zero observed failures).
        let mut pairs = Vec::new();
        for i in 0..60u32 {
            pairs.push((0.7 + f64::from(i % 10) * 0.01, true));
        }
        for i in 0..60u32 {
            pairs.push((0.2 + f64::from(i % 10) * 0.01, false));
        }
        let report = calibrate_pairs(&pairs, 0.2, 0.1, 30);
        assert!(
            report.conformal.feasible,
            "clean separation must be feasible"
        );
        assert!(
            report.conformal.threshold >= 0.2 && report.conformal.threshold <= 0.79,
            "threshold {} must land inside the observed score range",
            report.conformal.threshold
        );
        assert_eq!(report.n_pairs, 120);
        assert!(
            report.empirical_served_failure <= 0.2 + 1e-9,
            "empirical served-failure {} must respect alpha — the conformal guarantee",
            report.empirical_served_failure
        );
    }

    #[test]
    fn calibrate_pairs_infeasible_below_min_n() {
        let pairs = vec![(0.9, true), (0.9, true), (0.1, false)];
        let report = calibrate_pairs(&pairs, 0.1, 0.1, 30);
        assert!(
            !report.conformal.feasible,
            "too few pairs must be infeasible"
        );
    }

    #[tokio::test]
    async fn calibrate_from_store_pairs_only_traces_with_deferred_feedback() {
        let db_path = std::env::temp_dir().join(format!(
            "firstpass-calibrate-test-{}.db",
            uuid::Uuid::now_v7()
        ));
        let (tx, handle) = store::open(&db_path).unwrap();

        // 40 high-score traces confirmed correct, 40 low-score traces confirmed incorrect, and
        // 5 traces with no deferred verdict at all (must be excluded from calibration).
        let mut correct_ids = Vec::new();
        let mut incorrect_ids = Vec::new();
        for i in 0..40u32 {
            let t = trace_with_score(0.7 + f64::from(i % 10) * 0.01);
            correct_ids.push(t.trace_id.to_string());
            tx.try_send(t).unwrap();
        }
        for i in 0..40u32 {
            let t = trace_with_score(0.2 + f64::from(i % 10) * 0.01);
            incorrect_ids.push(t.trace_id.to_string());
            tx.try_send(t).unwrap();
        }
        for i in 0..5u32 {
            tx.try_send(trace_with_score(0.5 + f64::from(i) * 0.01))
                .unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        for trace_id in &correct_ids {
            let dv = DeferredVerdict {
                gate_id: "outcome".to_owned(),
                verdict: Verdict::Pass,
                score: None,
                reported_at: jiff::Timestamp::now(),
                reporter: "unit-test".to_owned(),
            };
            store::append_deferred(&db_path, trace_id, &dv).unwrap();
        }
        for trace_id in &incorrect_ids {
            let dv = DeferredVerdict {
                gate_id: "outcome".to_owned(),
                verdict: Verdict::Fail,
                score: None,
                reported_at: jiff::Timestamp::now(),
                reporter: "unit-test".to_owned(),
            };
            store::append_deferred(&db_path, trace_id, &dv).unwrap();
        }

        // alpha=0.2 for the same Hoeffding-slack reason as the calibrate_pairs test above.
        let report = calibrate_from_store(&db_path, "tenant-a", 0.2, 0.1, 30).unwrap();
        assert_eq!(
            report.n_pairs, 80,
            "only the 80 traces with deferred feedback pair up"
        );
        assert!(report.conformal.feasible);
        assert!(
            report.empirical_served_failure <= 0.2 + 1e-9,
            "empirical served-failure {} must respect alpha on clean synthetic data",
            report.empirical_served_failure
        );

        // D7 isolation: a different tenant sees none of tenant-a's pairs — calibration is empty.
        let other = calibrate_from_store(&db_path, "tenant-b", 0.2, 0.1, 30).unwrap();
        assert_eq!(
            other.n_pairs, 0,
            "tenant-b must not see tenant-a's feedback"
        );

        let _ = std::fs::remove_file(&db_path);
    }

    // ── LTT wiring tests ─────────────────────────────────────────────────────────────────────

    #[test]
    fn calibrate_pairs_ltt_feasible_on_clean_pairs() {
        // Same synthetic data as the conformal test — clean score separation, alpha=0.2.
        let mut pairs = Vec::new();
        for i in 0..60u32 {
            pairs.push((0.7 + f64::from(i % 10) * 0.01, true));
        }
        for i in 0..60u32 {
            pairs.push((0.2 + f64::from(i % 10) * 0.01, false));
        }
        let report = calibrate_pairs_ltt(&pairs, 0.2, 0.1, 30);
        assert!(
            report.ltt.feasible,
            "clean separation must be feasible with LTT"
        );
        assert!(
            report.ltt.threshold >= 0.2 && report.ltt.threshold <= 0.79,
            "threshold {} must land inside the observed score range",
            report.ltt.threshold
        );
        assert_eq!(report.n_pairs, 120);
        assert!(
            report.ltt.empirical_risk <= 0.2 + 1e-9,
            "empirical risk {} must respect alpha",
            report.ltt.empirical_risk
        );
    }

    #[test]
    fn calibrate_pairs_ltt_infeasible_below_min_n() {
        let pairs = vec![(0.9, true), (0.9, true), (0.1, false)];
        let report = calibrate_pairs_ltt(&pairs, 0.1, 0.05, 30);
        assert!(
            !report.ltt.feasible,
            "too few pairs must be infeasible with LTT"
        );
    }

    #[test]
    fn ltt_report_render_includes_method_and_far() {
        // Smoke-test that render() produces the expected key fields without panicking.
        let mut pairs: Vec<(f64, bool)> = Vec::new();
        for _ in 0..200 {
            pairs.push((0.9, true));
        }
        for _ in 0..5 {
            pairs.push((0.9, false));
        }
        for _ in 0..15 {
            pairs.push((0.2, false));
        }
        let report = calibrate_pairs_ltt(&pairs, 0.10, 0.05, 30);
        let rendered = report.render();
        assert!(
            rendered.contains("method: ltt"),
            "render must tag the method"
        );
        assert!(
            rendered.contains("false-accept rate:"),
            "render must include verifier ROC note"
        );
        assert!(
            rendered.contains("feasible:"),
            "render must include feasibility"
        );
    }
}
