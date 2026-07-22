//! Elastic-verification validation (ADR 0008 Phase 3, offline) — the money-saver, proven safe
//! *before* it is allowed to change serving.
//!
//! The mechanism: a cheap probe (k samples through the cheap/visible gate) sorts each request
//! into three regimes. Elastic verification then spends the **expensive** gate (LLM judge /
//! hidden tests) *proportional to doubt*:
//! - **confident pass** (probe signal ≥ a calibrated λ) → serve **without** the expensive gate;
//! - **confident fail** (0 of k pass) → escalate immediately;
//! - **ambiguous** → run the expensive gate, as today.
//!
//! The load-bearing question is whether skipping the expensive gate on the confident majority
//! keeps the distribution-free served-failure guarantee. We answer it the honest way: calibrate
//! the skip threshold λ on a **train** split via split-conformal (so served-failure among skips
//! is bounded ≤ α at confidence 1−δ), then measure realized served-failure among skips on a
//! **held-out** split. If the held-out rate stays ≤ α while cost drops, elastic verification is
//! validated; only then is a production serving change even proposed.
//!
//! The item distribution here is parameterized to the committed probe study
//! (`docs/benchmarks/probe-study-mbpp.txt`): ~65% confident-pass at ~99% oracle-correct, ~12%
//! confident-fail at ~0%, ~23% ambiguous. Determinism keeps the validation reproducible; the
//! real per-deployment numbers come from live receipts.

use crate::sim::hash01;
use firstpass_core::conformal::calibrate;

/// Relative costs (arbitrary units; only ratios matter). The expensive gate is what elastic
/// verification skips on the confident majority.
const CHEAP_GATE: f64 = 0.1;
const EXPENSIVE_GATE: f64 = 1.0;
const ESCALATE: f64 = 2.0;

/// One item: the cheap-probe signal (fraction of k samples that pass the visible gate) and the
/// ground-truth oracle outcome of the candidate that would be served.
struct Item {
    signal: f64,
    oracle_correct: bool,
}

/// Generate `n` items matching the probe-study regime distribution (deterministic).
fn generate(n: usize, seed: u64) -> Vec<Item> {
    (0..n)
        .map(|i| {
            let id = i as u64;
            let u = hash01(seed, id, 1);
            if u < 0.65 {
                // confident pass: all k pass visible; oracle-correct ~99%.
                Item {
                    signal: 1.0,
                    oracle_correct: hash01(seed, id, 2) < 0.99,
                }
            } else if u < 0.77 {
                // confident fail: 0 of k pass; oracle-correct ~0%.
                Item {
                    signal: 0.0,
                    oracle_correct: hash01(seed, id, 2) < 0.02,
                }
            } else {
                // ambiguous: partial pass; oracle-correctness tracks the signal.
                let s = 0.2 + 0.6 * hash01(seed, id, 3);
                Item {
                    signal: s,
                    oracle_correct: hash01(seed, id, 2) < s,
                }
            }
        })
        .collect()
}

/// Result of the elastic-verification validation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ElasticResult {
    /// Target served-failure rate and confidence used to calibrate the skip threshold.
    pub alpha: f64,
    pub delta: f64,
    /// Calibrated skip threshold λ (serve-without-expensive-gate iff signal ≥ λ).
    pub skip_threshold: f64,
    /// Whether calibration found a feasible threshold at all.
    pub feasible: bool,
    /// Held-out eval size.
    pub n_eval: usize,
    /// Fraction of eval requests served without the expensive gate.
    pub skipped_frac: f64,
    /// Realized served-failure rate **among the skipped serves**, on held-out data. This is the
    /// number that must stay ≤ `alpha` for the guarantee to survive elastic verification.
    pub skip_served_failure: f64,
    /// Mean expensive-gate cost per request under uniform verification (always run it).
    pub uniform_cost: f64,
    /// Mean cost per request under elastic verification (skip / gate / escalate by regime).
    pub elastic_cost: f64,
    /// `1 − elastic/uniform` — the cost saved.
    pub cost_saved_frac: f64,
    /// The acceptance test: held-out skip served-failure ≤ alpha AND elastic is cheaper.
    pub validated: bool,
}

/// Run the offline elastic-verification validation at target `alpha` / confidence `delta`.
///
/// Calibrates the skip threshold on a train split (split-conformal over `(signal, correct)`
/// pairs restricted to confident-pass-eligible items) and measures cost + held-out served-failure.
#[must_use]
pub fn run_elastic_validation(n: usize, alpha: f64, delta: f64, seed: u64) -> ElasticResult {
    let items = generate(n, seed);
    let split = n / 2;
    let (train, eval) = items.split_at(split);

    // Calibrate λ on train: split-conformal picks the lowest threshold whose served-failure upper
    // bound is ≤ α. Serving on `signal ≥ λ` then carries a distribution-free served-failure ≤ α.
    let pairs: Vec<(f64, bool)> = train
        .iter()
        .map(|it| (it.signal, it.oracle_correct))
        .collect();
    let cal = calibrate(&pairs, alpha, delta, 30);
    let lambda = cal.threshold;

    // Evaluate on held-out data.
    let mut skipped = 0usize;
    let mut skip_fail = 0usize;
    let mut elastic_total = 0.0_f64;
    for it in eval {
        if cal.feasible && it.signal >= lambda {
            // confident pass → serve without the expensive gate.
            skipped += 1;
            if !it.oracle_correct {
                skip_fail += 1;
            }
            elastic_total += CHEAP_GATE;
        } else if it.signal == 0.0 {
            // confident fail → escalate immediately (skip a doomed cheap serve).
            elastic_total += CHEAP_GATE + ESCALATE;
        } else {
            // ambiguous → run the expensive gate, as today.
            elastic_total += CHEAP_GATE + EXPENSIVE_GATE;
        }
    }
    let n_eval = eval.len().max(1) as f64;
    // Uniform verification always runs the expensive gate on the served candidate.
    let uniform_cost = CHEAP_GATE + EXPENSIVE_GATE;
    let elastic_cost = elastic_total / n_eval;
    let skip_served_failure = if skipped == 0 {
        0.0
    } else {
        skip_fail as f64 / skipped as f64
    };
    let cost_saved_frac = if uniform_cost > 0.0 {
        1.0 - elastic_cost / uniform_cost
    } else {
        0.0
    };
    ElasticResult {
        alpha,
        delta,
        skip_threshold: lambda,
        feasible: cal.feasible,
        n_eval: eval.len(),
        skipped_frac: skipped as f64 / n_eval,
        skip_served_failure,
        uniform_cost,
        elastic_cost,
        cost_saved_frac,
        validated: cal.feasible && skip_served_failure <= alpha && elastic_cost < uniform_cost,
    }
}

/// Render the validation as a human-readable verdict.
#[must_use]
pub fn render(r: &ElasticResult) -> String {
    format!(
        "# Elastic verification — offline validation (ADR 0008 Phase 3)\n\n\
         target: served-failure ≤ {:.0}% at {:.0}% confidence\n\
         calibrated skip threshold λ = {:.3} (feasible: {})\n\
         held-out eval: {} requests\n\
         - skipped the expensive gate on {:.1}% of requests (confident pass)\n\
         - served-failure AMONG SKIPS (held-out): {:.2}%  {}\n\
         - cost/request: uniform {:.3} → elastic {:.3}  ({:.1}% cheaper)\n\n\
         VERDICT: {}\n\
         (The skip carries the SAME distribution-free bound as a verified serve — the held-out\n\
         skip served-failure staying ≤ α is the proof. Replicate under drift + beyond MBPP\n\
         before any production serving change; this is the pre-registered Phase 3 gate.)",
        r.alpha * 100.0,
        (1.0 - r.delta) * 100.0,
        r.skip_threshold,
        r.feasible,
        r.n_eval,
        r.skipped_frac * 100.0,
        r.skip_served_failure * 100.0,
        if r.skip_served_failure <= r.alpha {
            "✓ ≤ α"
        } else {
            "✗ EXCEEDS α"
        },
        r.uniform_cost,
        r.elastic_cost,
        r.cost_saved_frac * 100.0,
        if r.validated {
            "VALIDATED — elastic verification saves cost while holding the served-failure bound."
        } else {
            "NOT validated on this data — do NOT enable the skip."
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_saves_cost_and_holds_the_bound() {
        let r = run_elastic_validation(8000, 0.10, 0.05, 3);
        assert!(
            r.feasible,
            "a feasible skip threshold should exist on this distribution"
        );
        assert!(
            r.skip_served_failure <= r.alpha + 1e-9,
            "held-out served-failure among skips must stay ≤ α: {:.3} > {}",
            r.skip_served_failure,
            r.alpha
        );
        assert!(
            r.elastic_cost < r.uniform_cost,
            "elastic must be cheaper than uniform verify"
        );
        assert!(
            r.skipped_frac > 0.4,
            "the confident-pass majority should be skippable"
        );
        assert!(
            r.validated,
            "on the probe-study distribution elastic should validate"
        );
    }

    #[test]
    fn a_stricter_alpha_is_still_honored_or_infeasible() {
        // With a very strict α the calibrated threshold rises (or calibration is infeasible);
        // either way the bound must never be violated among skips.
        let r = run_elastic_validation(8000, 0.02, 0.05, 5);
        assert!(
            r.skip_served_failure <= r.alpha + 1e-9 || !r.feasible,
            "strict α: skips must respect the bound or the skip must not happen"
        );
    }
}
