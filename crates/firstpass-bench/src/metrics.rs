//! Pre-registered metrics, computed per policy over identical traffic (SPEC §10.1).

use crate::policy::Decision;
use crate::stats::{Ci, bootstrap_mean_ci, bootstrap_ratio_ci, mean, quantile};
use firstpass_core::Verdict;
use serde::Serialize;

/// Aggregated metrics for one policy across a task suite.
#[derive(Debug, Clone, Serialize)]
pub struct PolicyMetrics {
    /// Policy name.
    pub name: String,
    /// Tasks evaluated.
    pub n: usize,
    /// Task-success rate (served output truly correct), with CI.
    pub success: Ci,
    /// **Headline:** USD per successful task, with CI. Lower is better at equal success.
    pub cost_per_success: Ci,
    /// Total USD across the suite.
    pub total_cost_usd: f64,
    /// Mean USD per request.
    pub mean_cost_usd: f64,
    /// Fraction of requests that served a truly-incorrect output (the trust cost).
    pub served_failure_rate: f64,
    /// Mean escalations per request.
    pub escalation_rate: f64,
    /// Fraction of requests whose cheapest-rung output was truly correct (tier-0 clearance).
    pub rung0_clearance: f64,
    /// Regret: served-a-failure + needless-escalation events per request.
    pub regret_per_request: f64,
    /// Gate precision `P(correct | pass)` — only for policies that gate.
    pub gate_precision: Option<f64>,
    /// Gate recall `P(pass | correct)` — only for policies that gate.
    pub gate_recall: Option<f64>,
    /// Median end-to-end latency (ms).
    pub p50_latency_ms: f64,
    /// P95 end-to-end latency (ms).
    pub p95_latency_ms: f64,
}

/// Bootstrap iteration count for CIs.
pub const BOOTSTRAP: usize = 1000;

/// Evaluate a policy's decisions into [`PolicyMetrics`].
#[must_use]
pub fn evaluate(name: &str, decisions: &[Decision], seed: u64, alpha: f64) -> PolicyMetrics {
    let n = decisions.len();
    let success01: Vec<f64> = decisions
        .iter()
        .map(|d| f64::from(d.served_correct()))
        .collect();
    let cost: Vec<f64> = decisions.iter().map(Decision::total_cost).collect();
    let latency: Vec<f64> = decisions.iter().map(|d| d.latency_ms() as f64).collect();

    let served_failures = decisions
        .iter()
        .filter(|d| d.served.is_some() && !d.served_correct())
        .count();
    let escalations: f64 = mean(
        &decisions
            .iter()
            .map(|d| f64::from(d.escalations))
            .collect::<Vec<_>>(),
    );
    let rung0_correct = decisions
        .iter()
        .filter(|d| d.attempts.first().is_some_and(|a| a.correct))
        .count();
    let regret: f64 = decisions
        .iter()
        .map(|d| {
            f64::from(
                u32::from(d.served.is_some() && !d.served_correct()) + d.needless_escalations(),
            )
        })
        .sum::<f64>()
        / n.max(1) as f64;

    // Gate precision/recall over all gated attempts (verdict present).
    let (mut tp, mut fp, mut fn_, mut gated) = (0u64, 0u64, 0u64, 0u64);
    for d in decisions {
        for a in &d.attempts {
            if let Some(v) = a.verdict {
                gated += 1;
                match (v, a.correct) {
                    (Verdict::Pass, true) => tp += 1,
                    (Verdict::Pass, false) => fp += 1,
                    (Verdict::Fail, true) => fn_ += 1,
                    _ => {}
                }
            }
        }
    }
    let (gate_precision, gate_recall) = if gated == 0 {
        (None, None)
    } else {
        let prec = if tp + fp > 0 {
            Some(tp as f64 / (tp + fp) as f64)
        } else {
            None
        };
        let rec = if tp + fn_ > 0 {
            Some(tp as f64 / (tp + fn_) as f64)
        } else {
            None
        };
        (prec, rec)
    };

    PolicyMetrics {
        name: name.to_owned(),
        n,
        success: bootstrap_mean_ci(&success01, BOOTSTRAP, seed, alpha),
        cost_per_success: bootstrap_ratio_ci(&cost, &success01, BOOTSTRAP, seed ^ 0x5EED, alpha),
        total_cost_usd: cost.iter().sum(),
        mean_cost_usd: mean(&cost),
        served_failure_rate: served_failures as f64 / n.max(1) as f64,
        escalation_rate: escalations,
        rung0_clearance: rung0_correct as f64 / n.max(1) as f64,
        regret_per_request: regret,
        gate_precision,
        gate_recall,
        p50_latency_ms: quantile(&latency, 0.5),
        p95_latency_ms: quantile(&latency, 0.95),
    }
}
