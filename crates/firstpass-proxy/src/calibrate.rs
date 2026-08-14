//! Recalibrate the serving threshold from real deferred feedback (SPEC §10.1, run against live
//! traffic instead of a static benchmark suite) — the "learns your quality bar" loop.
//!
//! Three calibration methods are available:
//! - **conformal** (default): split-conformal with Hoeffding bound — [`calibrate_from_store`].
//! - **ltt**: Learn-then-Test / RCPS with exact-binomial fixed-sequence testing —
//!   [`calibrate_from_store_ltt`].
//! - **eprocess**: anytime-valid risk control — [`calibrate_from_store_eprocess`].
//!
//! All three enumerate stored traces, pair each trace that has a deferred outcome with the score of
//! the attempt actually served, and hand the pairs to the respective core module. None feeds
//! back into the request hot path — that wiring is a deliberate follow-on once an operator has
//! reviewed a report.
//!
//! ## Which one to run, and why there are three
//!
//! The first two are **fixed-sample**: they answer "given this calibration set, what threshold is
//! safe?", and their guarantees are stated over that one set. That is the right question when an
//! operator recalibrates deliberately, reviews the report, and adopts a threshold.
//!
//! It is the wrong question when calibration runs continuously. Re-running a fixed-sample method
//! whenever more feedback lands, and adopting the winner each time, is optional stopping on a
//! growing stream — each individual run is valid while the sequence of adoptions is not. `eprocess`
//! exists for that regime: its bound holds at **every** round, so re-reading it as often as you like
//! costs nothing in validity. It pays for that in conservatism, and
//! [`firstpass_core::eprocess`] documents the trade rather than burying it.

use std::path::Path;

use firstpass_core::conformal::{self, ConformalResult};
use firstpass_core::eprocess;
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

/// Report for `--method eprocess`: anytime-valid risk control.
///
/// Unlike [`CalibrationReport`] and [`LttReport`], which each summarise a **single** fixed-sample
/// decision, this replays the stored feedback as the *stream it actually was* and reports what the
/// controller would certify at the end of it. That difference is the point: the other two answer
/// "given this calibration set, what threshold is safe?", while this answers "having watched this
/// stream round by round, what is certified *now*?" — the question a continuously-recalibrating
/// deployment is really asking.
#[derive(Debug, Clone)]
pub struct EProcessReport {
    /// Pairs replayed.
    pub n_pairs: usize,
    /// Certified threshold and its evidence, or `None` if the stream never certified anything.
    pub certification: Option<firstpass_core::eprocess::Certification>,
    /// Target served-failure rate.
    pub alpha: f64,
    /// Family-wise error budget across the grid and all rounds.
    pub delta: f64,
    /// Evidence required per threshold (`n / delta`, the Bonferroni crossing level).
    pub crossing_level: f64,
    /// Realized served-failure over the replay, as a diagnostic — not the guarantee.
    pub realized_served_failure: f64,
}

impl EProcessReport {
    /// Render for `firstpass calibrate --method eprocess`. Mirrors [`LttReport::render`].
    #[must_use]
    pub fn render(&self) -> String {
        let cert = match &self.certification {
            Some(c) => format!(
                "{:.4}  (e-value {:.1} >= {:.1}, first certified at round {})",
                c.threshold, c.e_value, self.crossing_level, c.certified_at_round
            ),
            // Not an error, and deliberately not a number: an uncertified controller has proven
            // nothing, and printing a threshold anyway is exactly the unproven claim this method
            // exists to refuse.
            None => "NONE — no threshold has earned a guarantee on this stream yet".to_owned(),
        };
        format!(
            "method: eprocess (anytime-valid)\n\
             pairs: {n_pairs}\n\
             certified threshold: {cert}\n\
             target alpha: {alpha:.4} (delta {delta:.4}, family-wise across grid AND rounds)\n\
             realized served-failure: {realized:.4}\n\
             guarantee: holds at EVERY round, no exchangeability assumed (Ville / e-process)\n",
            n_pairs = self.n_pairs,
            alpha = self.alpha,
            delta = self.delta,
            realized = self.realized_served_failure,
        )
    }
}

/// Replay `pairs` through an anytime-valid controller, in order.
///
/// Order matters here and nowhere else in this module: the other two methods sort or sweep, because
/// exchangeability makes order irrelevant to them. This one consumes the sequence as a stream, which
/// is what lets it stay valid when the stream is *not* exchangeable.
#[must_use]
pub fn calibrate_pairs_eprocess(pairs: &[(f64, bool)], alpha: f64, delta: f64) -> EProcessReport {
    // Grid from the observed score support, as LTT does — a candidate per distinct score is enough,
    // and a finer grid only inflates the Bonferroni correction without buying resolution.
    let mut grid: Vec<f64> = pairs.iter().map(|&(s, _)| s).collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    grid.dedup();

    let mut ctrl = eprocess::EProcessRiskControl::new(alpha, delta, eprocess::DEFAULT_BET, &grid);
    for &(score, correct) in pairs {
        ctrl.observe_served(score, correct);
    }
    EProcessReport {
        n_pairs: pairs.len(),
        certification: ctrl.certified_threshold(),
        alpha,
        delta,
        crossing_level: ctrl.crossing_level(),
        realized_served_failure: ctrl.realized_served_failure(),
    }
}

/// Anytime-valid calibration from every trace in the store that has a deferred outcome.
///
/// Error handling and tenant scoping match [`calibrate_from_store`] exactly.
///
/// # Errors
/// Returns [`StoreError`] if a stored trace's deferred verdicts cannot be read.
pub fn calibrate_from_store_eprocess(
    db_path: impl AsRef<Path>,
    tenant: &str,
    alpha: f64,
    delta: f64,
) -> Result<EProcessReport, StoreError> {
    let db_path = db_path.as_ref();
    let traces = store::load_tenant_traces(db_path, tenant).unwrap_or_default();
    let mut pairs = Vec::with_capacity(traces.len());
    for trace in &traces {
        let deferred = store::load_deferred(db_path, &trace.trace_id.to_string())?;
        if let Some(pair) = trace_pair(trace, &deferred) {
            pairs.push(pair);
        }
    }
    Ok(calibrate_pairs_eprocess(&pairs, alpha, delta))
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
            cache_write_tokens: 0,
            cache_read_tokens: 0,
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
                mode_profile: None,
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
                cache_source: None,
            },
            probe: None,
            rollout: None,
            shadow: None,
            route_ix: None,
            predicted_pass: None,
            elastic: None,
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

    /// An uncertified stream must render as NONE, never as a number.
    ///
    /// The failure this guards is a quiet one: printing some default threshold when nothing has been
    /// proven reads exactly like a calibrated result to whoever runs the command. Too few pairs to
    /// certify must look like too few pairs to certify.
    #[test]
    fn eprocess_reports_no_threshold_when_nothing_is_certified() {
        let pairs: Vec<(f64, bool)> = vec![(0.9, true), (0.8, true)];
        let report = calibrate_pairs_eprocess(&pairs, 0.10, 0.05);
        assert!(
            report.certification.is_none(),
            "two pairs cannot certify anything at delta=0.05"
        );
        let rendered = report.render();
        assert!(
            rendered.contains("NONE"),
            "an uncertified stream must say so plainly, got: {rendered}"
        );
        assert!(
            rendered.contains("anytime-valid"),
            "the report must name the guarantee it provides, got: {rendered}"
        );
    }

    /// Enough clean evidence must certify, and the rendered threshold must be the certified one.
    /// The pairing with the test above is the contract: silent when unproven, specific when proven.
    #[test]
    fn eprocess_certifies_and_renders_the_threshold_on_a_clean_stream() {
        let mut pairs: Vec<(f64, bool)> = Vec::new();
        for i in 0..2000 {
            // Deterministic 5% failure rate, well under alpha — no RNG, so this is reproducible.
            pairs.push((0.9, i % 20 != 0));
        }
        let report = calibrate_pairs_eprocess(&pairs, 0.20, 0.05);
        let cert = report
            .certification
            .as_ref()
            .expect("2000 clean pairs must certify");
        assert!(
            cert.e_value >= report.crossing_level,
            "a certified threshold must meet the Bonferroni crossing level: {} < {}",
            cert.e_value,
            report.crossing_level
        );
        assert!(
            report.render().contains(&format!("{:.4}", cert.threshold)),
            "the rendered report must show the certified threshold"
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
