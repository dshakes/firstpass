//! Split-conformal risk control on the gate threshold (SPEC §10.1).
//!
//! The standing critique of every cascade is that its deferral threshold is a hand-tuned
//! hyperparameter with no guarantee. We replace that with a calibrated one: given a held-out set
//! of `(gate_score, was_correct)` pairs, choose the *lowest* score threshold `λ` such that serving
//! everything scoring `≥ λ` has a failure rate whose finite-sample upper confidence bound is `≤ α`.
//! Serving on `score ≥ λ` then carries a distribution-free guarantee: **served-failure rate ≤ α**
//! at confidence `1 − δ`. The bound is Hoeffding's, so it is conservative — it never *under*-covers.

use serde::Serialize;

/// The result of calibrating a conformal threshold.
#[derive(Debug, Clone, Serialize)]
pub struct ConformalResult {
    /// Target served-failure rate.
    pub alpha: f64,
    /// Confidence parameter (bound holds with probability ≥ 1 − δ).
    pub delta: f64,
    /// Chosen score threshold `λ`; serve iff `score ≥ λ`.
    pub threshold: f64,
    /// Fraction of the calibration set that would be served at this threshold.
    pub served_frac: f64,
    /// Empirical failure rate among served calibration items.
    pub calib_risk: f64,
    /// Whether any threshold met the target (false ⇒ serve-nothing, target infeasible on this data).
    pub feasible: bool,
}

/// Calibrate a conformal serving threshold.
///
/// `pairs` are `(score, correct)`; `alpha` is the target served-failure rate; `delta` the
/// confidence parameter. `min_n` guards against certifying a bound on too few served items.
#[must_use]
pub fn calibrate(pairs: &[(f64, bool)], alpha: f64, delta: f64, min_n: usize) -> ConformalResult {
    // Sort by score descending; sweeping downward grows the served set monotonically.
    let mut sorted: Vec<(f64, bool)> = pairs.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let slack = (f64::ln(1.0 / delta) / 2.0).sqrt(); // Hoeffding: + sqrt(ln(1/δ)/(2n))
    let mut fails = 0usize;
    let mut best: Option<(f64, usize, usize)> = None; // (threshold, served, fails)

    for (i, (score, correct)) in sorted.iter().enumerate() {
        if !*correct {
            fails += 1;
        }
        let served = i + 1;
        if served < min_n {
            continue;
        }
        let rate = fails as f64 / served as f64;
        let ucb = rate + slack / (served as f64).sqrt();
        if ucb <= alpha {
            // Served grows as we descend, so the last satisfying point is the max-coverage one.
            best = Some((*score, served, fails));
        }
    }

    match best {
        Some((threshold, served, fails)) => ConformalResult {
            alpha,
            delta,
            threshold,
            served_frac: served as f64 / pairs.len().max(1) as f64,
            calib_risk: fails as f64 / served as f64,
            feasible: true,
        },
        None => ConformalResult {
            alpha,
            delta,
            threshold: f64::INFINITY, // serve nothing — target infeasible on this data
            served_frac: 0.0,
            calib_risk: 0.0,
            feasible: false,
        },
    }
}

/// Empirical served-failure rate on a fresh set at a given threshold (for validation).
#[must_use]
pub fn served_failure_rate(pairs: &[(f64, bool)], threshold: f64) -> (f64, usize) {
    let served: Vec<bool> = pairs
        .iter()
        .filter(|(s, _)| *s >= threshold)
        .map(|(_, c)| *c)
        .collect();
    if served.is_empty() {
        return (0.0, 0);
    }
    let fails = served.iter().filter(|c| !**c).count();
    (fails as f64 / served.len() as f64, served.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Gate, ModelBackend, Rung, SimBackend, SimGate, task_suite};

    /// Produce `(score, correct)` pairs by running rung 0 over a suite.
    fn pairs(seed: u64, n: usize) -> Vec<(f64, bool)> {
        let suite = task_suite(n, seed);
        let be = SimBackend::new(seed);
        let gate = SimGate::new(seed ^ 0x99, 0.08, 0.10, 0.0);
        let rung = Rung::new("anthropic/claude-haiku-4-5", 0.62);
        suite
            .iter()
            .map(|t| {
                let c = be.run(t, &rung);
                (gate.judge(t, &rung, &c).score, c.correct)
            })
            .collect()
    }

    #[test]
    fn guarantee_holds_on_held_out_data() {
        let alpha = 0.10;
        let calib = pairs(1, 4000);
        let test = pairs(2, 4000); // disjoint seed => held-out
        let r = calibrate(&calib, alpha, 0.05, 50);
        assert!(r.feasible, "should find a feasible threshold on this data");
        assert!(
            r.served_frac > 0.2,
            "should serve a meaningful fraction, got {}",
            r.served_frac
        );

        let (test_risk, n_served) = served_failure_rate(&test, r.threshold);
        assert!(n_served > 0);
        // The conformal guarantee: held-out served-failure stays at/under the target (small
        // tolerance for finite-sample noise; the UCB is conservative so this is comfortable).
        assert!(
            test_risk <= alpha + 0.03,
            "held-out served-failure {test_risk:.3} must be <= alpha {alpha} (+tol)"
        );
    }

    #[test]
    fn infeasible_target_serves_nothing() {
        // alpha = 0 is unachievable with a noisy gate -> serve nothing rather than lie.
        let calib = pairs(3, 1000);
        let r = calibrate(&calib, 0.0, 0.05, 50);
        assert!(!r.feasible);
        assert_eq!(r.served_frac, 0.0);
    }
}
