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

/// Online / adaptive conformal — Gibbs & Candès (2021), *Adaptive Conformal Inference Under
/// Distribution Shift*. [`calibrate`] fixes a threshold ONCE and assumes exchangeability; under real
/// drift (models change, prompts change, the gate's error rate moves) the realized served-failure
/// wanders off target. `AdaptiveConformal` instead tracks the serving threshold **online** from
/// realized outcomes, so the long-run served-failure rate stays at `alpha` as the workload shifts.
///
/// This is the "gate that recalibrates itself from live feedback": every deferred verdict nudges the
/// threshold, so it never drifts too loose (serving junk) or too strict (escalating needlessly). Feed
/// it the deferred-feedback stream and read [`AdaptiveConformal::threshold`] on the router hot path.
#[derive(Debug, Clone)]
pub struct AdaptiveConformal {
    alpha: f64,
    gamma: f64,
    threshold: f64,
    served: u64,
    served_fails: u64,
}

impl AdaptiveConformal {
    /// `alpha` = target served-failure rate; `gamma` = step size (e.g. 0.01–0.05, larger tracks
    /// shift faster but noisier); `init_threshold` = starting `λ` (e.g. a [`calibrate`] result).
    #[must_use]
    pub fn new(alpha: f64, gamma: f64, init_threshold: f64) -> Self {
        Self {
            alpha,
            gamma,
            threshold: init_threshold.clamp(0.0, 1.0),
            served: 0,
            served_fails: 0,
        }
    }

    /// Serve iff `score ≥` the current threshold.
    #[must_use]
    pub fn should_serve(&self, score: f64) -> bool {
        score >= self.threshold
    }

    /// The current serving threshold `λ_t`.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Observe a **served** item's realized correctness (from deferred feedback) and adapt `λ`. The
    /// ACI update raises `λ` when served errors exceed `alpha` (serve more conservatively) and lowers
    /// it when they're below (serve more), so realized served-failure converges to `alpha`.
    pub fn observe_served(&mut self, was_correct: bool) {
        let err = f64::from(!was_correct);
        self.threshold = (self.threshold + self.gamma * (err - self.alpha)).clamp(0.0, 1.0);
        self.served += 1;
        self.served_fails += u64::from(!was_correct);
    }

    /// Realized served-failure rate so far (running diagnostic).
    #[must_use]
    pub fn realized_served_failure(&self) -> f64 {
        if self.served == 0 {
            0.0
        } else {
            self.served_fails as f64 / self.served as f64
        }
    }

    /// Number of served items observed so far.
    #[must_use]
    pub fn served(&self) -> u64 {
        self.served
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ponytail: tiny inline SplitMix64, mirroring firstpass-bench's `sim::hash01` (which this
    // crate must not depend on) — deterministic, dependency-free draws for the synthetic pairs
    // below. Keeps the conformal guarantee test self-contained now that it lives in core.
    fn hash01(seed: u64, a: u64, b: u64) -> f64 {
        let mut s = seed
            .wrapping_mul(0xD1B5_4A32_D192_ED03)
            .wrapping_add(a.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .wrapping_add(b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add(0x1234_5678_9ABC_DEF0);
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Produce `(score, correct)` pairs with a gate score correlated with true correctness plus
    /// noise (correct centers at 0.72, incorrect at 0.30) — the same shape `sim::SimGate` produces
    /// for a real gate, without pulling in the bench simulation crate.
    fn pairs(seed: u64, n: usize) -> Vec<(f64, bool)> {
        (0..n as u64)
            .map(|id| {
                let correct = hash01(seed, id, 1) < 0.7;
                let noise = (hash01(seed ^ 0x00C0_FFEE, id, 2) - 0.5) * 0.4;
                let base = if correct { 0.72 } else { 0.30 };
                ((base + noise).clamp(0.0, 1.0), correct)
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

    #[test]
    fn adaptive_update_moves_threshold_the_right_way() {
        let mut a = AdaptiveConformal::new(0.10, 0.05, 0.5);
        // A served FAILURE raises the threshold (serve more conservatively).
        a.observe_served(false);
        assert!(a.threshold() > 0.5);
        // A served SUCCESS nudges it back down (serve more).
        let mut b = AdaptiveConformal::new(0.10, 0.05, 0.5);
        b.observe_served(true);
        assert!(b.threshold() < 0.5);
    }

    // Generate a `(score, correct)` stream; after the shift, INCORRECT items score high (the gate
    // degrades and starts leaking false-accepts past the old threshold).
    fn shifted(id: u64, shift: bool) -> (f64, bool) {
        let correct = hash01(42, id, 1) < 0.7;
        let noise = (hash01(42 ^ 0xBEEF, id, 2) - 0.5) * 0.3;
        let base = if correct {
            0.78
        } else if shift {
            0.58
        } else {
            0.30
        };
        ((base + noise).clamp(0.0, 1.0), correct)
    }

    #[test]
    fn adaptive_tracks_alpha_under_shift_where_fixed_drifts() {
        let alpha = 0.10;
        let n = 6000u64;

        // Fixed threshold calibrated on the pre-shift regime (what a one-shot `calibrate` gives you).
        let calib: Vec<(f64, bool)> = (0..n).map(|id| shifted(id, false)).collect();
        let fixed = calibrate(&calib, alpha, 0.05, 50).threshold;

        // Run the FIXED threshold and an ADAPTIVE one over the same post-shift stream.
        let mut aci = AdaptiveConformal::new(alpha, 0.03, fixed);
        let (mut fx_served, mut fx_fails) = (0u64, 0u64);
        let (mut ac_served, mut ac_fails) = (0u64, 0u64);
        for id in n..(5 * n) {
            let (score, correct) = shifted(id, true);
            if score >= fixed {
                fx_served += 1;
                fx_fails += u64::from(!correct);
            }
            if aci.should_serve(score) {
                aci.observe_served(correct);
                ac_served += 1;
                ac_fails += u64::from(!correct);
            }
        }
        let fixed_rate = fx_fails as f64 / fx_served.max(1) as f64;
        let aci_rate = ac_fails as f64 / ac_served.max(1) as f64;

        // Under shift the FIXED threshold serves the new false-accepts and drifts above alpha...
        assert!(
            fixed_rate > alpha + 0.05,
            "fixed should drift high under shift, got {fixed_rate:.3}"
        );
        // ...while ADAPTIVE re-converges: strictly better than fixed and near the target.
        assert!(
            aci_rate < fixed_rate,
            "adaptive {aci_rate:.3} should beat fixed {fixed_rate:.3}"
        );
        assert!(
            aci_rate <= alpha + 0.06,
            "adaptive {aci_rate:.3} should track alpha {alpha}"
        );
        assert!(ac_served > 0);
    }
}
