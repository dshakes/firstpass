//! # firstpass-bench — the M0 proof harness
//!
//! A pre-registered, baseline-controlled benchmark that turns "cheapest model that provably
//! passes" from a claim into a measured, reproducible result with confidence intervals, a
//! distribution-free serving guarantee (conformal), and a published kill criterion (SPEC §10).
//!
//! **Simulation-first:** [`sim`] provides deterministic stand-ins behind the [`sim::ModelBackend`]
//! and [`sim::Gate`] traits, so the methodology is fully tested before real-provider numbers
//! exist. Swap those for reqwest-backed impls and the same harness produces real proof.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod conformal;
pub mod live;
pub mod metrics;
pub mod policy;
pub mod report;
pub mod sim;
pub mod stats;

use firstpass_core::PriceTable;
use metrics::evaluate;
use policy::{AlwaysCheap, AlwaysTop, Decision, Firstpass, Policy, PredictiveRouter, RandomRung};
use report::{KillDecision, Report};
use sim::{Gate, ModelBackend, Rung, SimBackend, SimGate, Task, task_suite};
use stats::mean;

/// Knobs for a benchmark run. Defaults model the agent/coding beachhead: a fast cheap tier, a
/// strong top tier, and a deterministic (zero-marginal-cost) gate with realistic error rates.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Number of tasks in the evaluation suite.
    pub n_tasks: usize,
    /// Suite generation seed.
    pub seed: u64,
    /// The model ladder, cheapest first (models must exist in the price table).
    pub ladder: Vec<Rung>,
    /// Backend correctness seed.
    pub backend_seed: u64,
    /// Gate false-positive rate (false accept).
    pub gate_fpr: f64,
    /// Gate false-negative rate (false reject).
    pub gate_fnr: f64,
    /// Marginal USD per gate call (0.0 for a deterministic test/typecheck/schema gate).
    pub gate_cost_usd: f64,
    /// Optional per-request USD budget cap for Firstpass.
    pub budget_usd: Option<f64>,
    /// Predictive-router difficulty-estimation noise (its structural handicap).
    pub predictor_noise: f64,
    /// CI level (`1 - alpha`).
    pub alpha: f64,
    /// Conformal target served-failure rate.
    pub conformal_alpha: f64,
    /// Conformal confidence parameter.
    pub conformal_delta: f64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            n_tasks: 500,
            seed: 20260708,
            ladder: vec![
                Rung::new("anthropic/claude-haiku-4-5", 0.62),
                Rung::new("anthropic/claude-sonnet-5", 0.80),
                Rung::new("anthropic/claude-opus-4-8", 0.93),
            ],
            backend_seed: 20260708,
            gate_fpr: 0.08,
            gate_fnr: 0.10,
            gate_cost_usd: 0.0,
            budget_usd: None,
            predictor_noise: 0.30,
            alpha: 0.05,
            conformal_alpha: 0.10,
            conformal_delta: 0.05,
        }
    }
}

/// Run the full benchmark against the **simulated** backend and produce a [`Report`].
#[must_use]
pub fn run_benchmark(cfg: &BenchConfig) -> Report {
    let suite = task_suite(cfg.n_tasks, cfg.seed);
    let backend = SimBackend::new(cfg.backend_seed);
    let gate = SimGate::new(
        cfg.backend_seed ^ 0xA11CE,
        cfg.gate_fpr,
        cfg.gate_fnr,
        cfg.gate_cost_usd,
    );
    // Disjoint calibration suite for conformal (sim only — free to draw more tasks).
    let calib = task_suite(cfg.n_tasks.max(2000), cfg.seed.wrapping_add(1));
    run_core(cfg, &suite, &calib, &backend, &gate, true)
}

/// Run the full benchmark against **live providers** (Anthropic Messages) over a verifiable task
/// suite, producing the real (non-simulated) [`Report`]. BYOK: `api_key` is used only to call the
/// provider and is never stored or traced.
///
/// A preflight call validates the key/model before the full run so a bad key fails fast instead of
/// burning a whole run. Any hard provider error during the run aborts with `Err` rather than
/// publishing misleading all-fail numbers. Live runs make real API calls (~150 at the default
/// suite size) and cost real tokens.
///
/// # Errors
/// Returns a message if the preflight fails or any live call errors during the run.
pub fn run_benchmark_live(cfg: &BenchConfig, api_key: String) -> Result<Report, String> {
    let backend = live::LiveBackend::new(api_key.clone());
    // Graded verifiable suite; size via FIRSTPASS_BENCH_N (default 200 — the SPEC §10 target).
    let n = std::env::var("FIRSTPASS_BENCH_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(200);
    let suite = live::graded_suite(n);
    backend.preflight(&cfg.ladder[0]).map_err(|e| {
        format!("preflight failed (check ANTHROPIC_API_KEY and ladder model ids): {e}")
    })?;

    // Gate selection: the perfect checker (fast/free, cost/success proof) or the imperfect live
    // judge (FIRSTPASS_GATE=judge) whose real errors earn the conformal guarantee.
    let use_judge = std::env::var("FIRSTPASS_GATE").is_ok_and(|g| g == "judge");
    let (report, backend_errs, judge_errs) = if use_judge {
        let judge_model = std::env::var("FIRSTPASS_JUDGE_MODEL")
            .unwrap_or_else(|_| "anthropic/claude-haiku-4-5".to_string());
        let gate = live::LiveJudgeGate::new(api_key, judge_model);
        let report = run_core(cfg, &suite, &suite, &backend, &gate, false);
        (report, backend.take_errors(), gate.take_errors())
    } else {
        let report = run_core(cfg, &suite, &suite, &backend, &live::LiveGate, false);
        (report, backend.take_errors(), Vec::new())
    };

    // Candidate-model errors corrupt the measurement → refuse to publish. But a judge that
    // soft-fails on a few tasks already ABSTAINED (a valid gate outcome, score-neutral) — that's
    // the gate behaving correctly, not a corrupt run, so we note it and still publish.
    if !backend_errs.is_empty() {
        let mut e = backend_errs;
        e.truncate(20);
        return Err(format!(
            "{} candidate call(s) failed — refusing to publish numbers:\n  {}",
            e.len(),
            e.join("\n  ")
        ));
    }
    if !judge_errs.is_empty() {
        eprintln!(
            "note: {} judge call(s) soft-failed and abstained (a valid gate outcome, not a candidate error)",
            judge_errs.len()
        );
    }
    Ok(report)
}

/// The shared benchmark core: same policies, metrics, conformal calibration, and kill criterion,
/// run against whatever backend/gate/suite it is handed. `simulated` records whether these are
/// simulated or real-provider numbers so the report can label itself honestly.
fn run_core(
    cfg: &BenchConfig,
    suite: &[Task],
    calib: &[Task],
    backend: &dyn ModelBackend,
    gate: &dyn Gate,
    simulated: bool,
) -> Report {
    let prices = PriceTable::defaults();

    let policies: Vec<Box<dyn Policy>> = vec![
        Box::new(AlwaysCheap),
        Box::new(AlwaysTop),
        Box::new(RandomRung { seed: cfg.seed }),
        Box::new(PredictiveRouter {
            seed: cfg.seed,
            noise: cfg.predictor_noise,
        }),
        Box::new(Firstpass {
            budget_usd: cfg.budget_usd,
        }),
    ];

    let mut policy_metrics = Vec::with_capacity(policies.len());
    for pol in &policies {
        let decisions: Vec<Decision> = suite
            .iter()
            .map(|t| pol.decide(t, &cfg.ladder, backend, gate, &prices))
            .collect();
        policy_metrics.push(evaluate(pol.name(), &decisions, cfg.seed, cfg.alpha));
    }

    // Conformal calibration on rung-0 (score, correct) pairs from the calibration suite.
    let rung0 = &cfg.ladder[0];
    let pairs: Vec<(f64, bool)> = calib
        .iter()
        .map(|t| {
            let c = backend.run(t, rung0);
            (gate.judge(t, rung0, &c).score, c.correct)
        })
        .collect();
    let conformal = conformal::calibrate(&pairs, cfg.conformal_alpha, cfg.conformal_delta, 50);

    // Kill criterion (SPEC §10): representative per-rung costs from the suite.
    let rung_cost = |r: &Rung| -> f64 {
        mean(
            &suite
                .iter()
                .map(|t| {
                    let c = backend.run(t, r);
                    prices
                        .cost_usd(&r.model, c.in_tokens, c.out_tokens)
                        .unwrap_or(0.0)
                })
                .collect::<Vec<_>>(),
        )
    };
    let c0 = rung_cost(&cfg.ladder[0]);
    let c1 = if cfg.ladder.len() > 1 {
        rung_cost(&cfg.ladder[1])
    } else {
        c0
    };
    let break_even_p0 = if c1 > 0.0 {
        (c0 + cfg.gate_cost_usd) / c1
    } else {
        1.0
    };

    let fp = policy_metrics.iter().find(|p| p.name == "firstpass");
    let top = policy_metrics.iter().find(|p| p.name == "always-top");
    let measured_p0 = fp.map_or(0.0, |p| p.rung0_clearance);
    let clearance_above = measured_p0 > break_even_p0;
    let cheaper = match (fp, top) {
        (Some(f), Some(t)) => f.cost_per_success.point < t.cost_per_success.point,
        _ => false,
    };
    let proceed = cheaper && clearance_above;
    let rationale = if proceed {
        format!(
            "cheap-tier clears {:.0}% (break-even {:.0}%) and firstpass is cheaper per success — the thesis holds on this data.",
            measured_p0 * 100.0,
            break_even_p0 * 100.0
        )
    } else if !clearance_above {
        "cheap-tier clearance below break-even: cheapest-first cannot pay off here — stop or re-scope the traffic.".to_owned()
    } else {
        "firstpass not cheaper per success than always-top on this data — stop and publish the negative result.".to_owned()
    };

    Report {
        simulated,
        policies: policy_metrics,
        conformal,
        kill: KillDecision {
            break_even_p0,
            measured_p0,
            clearance_above_break_even: clearance_above,
            firstpass_cheaper_per_success: cheaper,
            proceed,
            rationale,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_invariants_hold_on_the_simulation() {
        let r = run_benchmark(&BenchConfig::default());
        let get = |n: &str| r.policy(n).expect("policy present");
        let (cheap, top, pred, fp) = (
            get("always-cheap"),
            get("always-top"),
            get("predictive"),
            get("firstpass"),
        );

        // Quality ordering.
        assert!(
            top.success.point >= cheap.success.point,
            "top should beat cheap on success"
        );
        assert!(
            fp.success.point >= cheap.success.point,
            "firstpass should beat cheap on success"
        );
        // Parity-or-better vs always-top (multiple gated shots).
        assert!(
            fp.success.point >= top.success.point - 0.05,
            "firstpass should reach ~parity with top"
        );

        // The money claim, honestly: cheaper in total AND per successful task.
        assert!(
            fp.total_cost_usd < top.total_cost_usd,
            "firstpass must cost less overall"
        );
        assert!(
            fp.cost_per_success.point < top.cost_per_success.point,
            "firstpass ${:.4}/success must beat always-top ${:.4}/success",
            fp.cost_per_success.point,
            top.cost_per_success.point
        );

        // Verification beats prediction: fewer served failures than a predictive router.
        assert!(
            fp.served_failure_rate < pred.served_failure_rate,
            "firstpass served-failure {:.3} should be below predictive {:.3}",
            fp.served_failure_rate,
            pred.served_failure_rate
        );

        // The gate actually gates.
        assert!(fp.gate_precision.is_some() && fp.gate_recall.is_some());

        // Guarantee is real and the go/no-go fires PROCEED on this healthy data.
        assert!(r.conformal.feasible && r.conformal.threshold.is_finite());
        assert!(
            r.kill.proceed,
            "kill criterion should PROCEED on the default (healthy) sim"
        );
    }

    #[test]
    fn kill_criterion_stops_when_the_cheap_tier_is_hopeless() {
        // A cheap tier that clears almost nothing: cheapest-first cannot pay off -> STOP.
        let mut cfg = BenchConfig::default();
        cfg.ladder[0].strength = 0.05;
        let r = run_benchmark(&cfg);
        assert!(!r.kill.clearance_above_break_even);
        assert!(
            !r.kill.proceed,
            "should STOP when the cheap tier is hopeless"
        );
    }
}
