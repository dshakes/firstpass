//! Learn-then-Test (LTT) threshold calibration — distribution-free, finite-sample risk control
//! on the gate serving threshold (SPEC §10.1).
//!
//! ## Method
//! Angelopoulos et al. (2021), "Learn then Test: Calibrating Predictive Algorithms to Achieve
//! Risk Control"; Bates et al. (2021), "Testing for Outliers with Conformal P-values" (RCPS).
//!
//! The guarantee: the returned threshold λ carries a finite-sample risk certificate —
//! served-failure rate ≤ α at family-wise confidence 1 − δ — without assuming exchangeability
//! or a parametric model of the gate's error distribution.
//!
//! ## Algorithm
//! 1. Build candidate thresholds from the observed score grid in descending order (strictest
//!    first). One candidate per distinct observed score suffices — finer grids do not tighten
//!    the guarantee (only the binomial test size δ and sample n do).
//! 2. For each λ compute: the served set (pairs with score ≥ λ), empirical risk
//!    = n_failures / n_served, and the exact-binomial p-value
//!    `P(Bin(n_served, α) ≤ n_failures)` testing H₀: risk(λ) > α.
//!    Reject H₀ (certify λ) when p-value ≤ δ.
//! 3. Walk candidates in the fixed-sequence order. Stop at the first *qualified* candidate
//!    (n_served ≥ min_n) whose test fails — subsequent candidates are never certified, even if
//!    their empirical risk dips below α again. Candidates below min_n are skipped (not in the
//!    test sequence) and do not break the walk.
//! 4. Return the last certified λ (least strict, maximum coverage), or `INFINITY` (serve
//!    nothing) when infeasible.
//!
//! ## Why the fixed-sequence walk controls FWER at δ without Bonferroni
//! A false certification is a false rejection of a true null H₀: risk(λ) > α. In any fixed
//! pre-specified test sequence the family-wise error rate (FWER = P(≥ 1 false rejection))
//! equals the probability that the *first* true H₀ encountered in the walk is falsely rejected,
//! which is at most δ by the individual test level. The stopping rule is the key: once a test
//! fails the sequence ends, so accumulating multiple false rejections in one run is impossible.
//! Theorem 2 of Angelopoulos et al. (2021) formalises this for any fixed order, without a
//! Bonferroni correction factor, provided each individual test is valid at level δ. The
//! one-sided exact-binomial test used here is conservative (it never over-rejects under H₀), so
//! the FWER guarantee holds even though all tests share the same calibration set.

use serde::Serialize;

/// Per-λ diagnostic row emitted by [`calibrate`].
#[derive(Debug, Clone, Serialize)]
pub struct LttDiagnostic {
    /// Candidate threshold value.
    pub lambda: f64,
    /// Number of calibration items with score ≥ λ (served set size).
    pub n_served: usize,
    /// Empirical served-failure rate at this λ (failures / n_served).
    pub empirical_risk: f64,
    /// Exact-binomial p-value for H₀: risk(λ) > α.  Small ⇒ H₀ rejected ⇒ λ certified.
    /// Set to `1.0` for candidates excluded by `min_n` (not in the test sequence).
    pub p_value: f64,
    /// Whether this λ was certified by the fixed-sequence walk.
    pub certified: bool,
}

/// Result of LTT threshold calibration.
#[derive(Debug, Clone, Serialize)]
pub struct LttResult {
    /// Target served-failure rate.
    pub alpha: f64,
    /// Family-wise error rate bound (FWER ≤ δ over the qualified candidate grid).
    pub delta: f64,
    /// Chosen threshold `λ`; serve iff `score ≥ λ`.
    /// `f64::INFINITY` when no threshold is certified (infeasible, serve nothing).
    pub threshold: f64,
    /// Whether any threshold was certified.  `false` ⇒ target infeasible on this data.
    pub feasible: bool,
    /// Empirical served-failure rate at `threshold` (failures / n_served on the calibration set).
    pub empirical_risk: f64,
    /// Served-set size at `threshold`.
    pub n_served: usize,
    /// Gate empirical false-accept rate at `threshold`: fraction of INCORRECT calibration items
    /// with score ≥ λ — the verifier ROC point the threshold is built on.
    /// `None` when the calibration set contains no incorrect items.
    pub false_accept_rate: Option<f64>,
    /// Per-λ diagnostics for every candidate in the grid, strictest to loosest.
    /// Includes candidates below `min_n` (which appear with `certified: false, p_value: 1.0`).
    pub diagnostics: Vec<LttDiagnostic>,
}

/// Calibrate an LTT serving threshold.
///
/// `pairs` are `(score, correct)`; `alpha` is the target served-failure rate; `delta` the
/// family-wise error rate bound; `min_n` guards against certifying on too small a served set.
/// Candidates below `min_n` are excluded from the test sequence but do not break the walk.
///
/// Returns the least-strict certified threshold (maximum coverage), or an infeasible result
/// (`feasible: false`, `threshold: INFINITY`) when no candidate passes.  The `diagnostics`
/// field carries the full per-λ grid for operator inspection.
#[must_use]
pub fn calibrate(pairs: &[(f64, bool)], alpha: f64, delta: f64, min_n: usize) -> LttResult {
    if pairs.is_empty() {
        return mk_infeasible(alpha, delta, Vec::new());
    }

    // Sort by score descending to enable an O(n) sweep as λ decreases.
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Candidate thresholds: one per distinct observed score, descending.
    // ponytail: exact dedup is correct here — scores are calibration values, not computed floats.
    let candidates: Vec<f64> = {
        let mut c: Vec<f64> = sorted.iter().map(|(s, _)| *s).collect();
        c.dedup();
        c
    };

    let n_incorrect = sorted.iter().filter(|(_, c)| !c).count();

    let mut diagnostics: Vec<LttDiagnostic> = Vec::with_capacity(candidates.len());
    // `walk_active`: still in the certified prefix of the fixed-sequence walk.
    let mut walk_active = true;
    // Index into `diagnostics` of the last certified λ (the least-strict certified one).
    let mut best: Option<usize> = None;

    // Running served-set counters (valid because candidates are descending and sorted is sorted).
    let mut ptr = 0usize;
    let mut failures = 0usize;

    for &lambda in &candidates {
        // Advance pointer to include all items with score >= lambda.
        while ptr < sorted.len() && sorted[ptr].0 >= lambda {
            if !sorted[ptr].1 {
                failures += 1;
            }
            ptr += 1;
        }
        let n_served = ptr;
        let empirical_risk = if n_served == 0 {
            0.0
        } else {
            failures as f64 / n_served as f64
        };

        if n_served < min_n {
            // Below minimum size: skip from the test sequence, do not break the walk.
            diagnostics.push(LttDiagnostic {
                lambda,
                n_served,
                empirical_risk,
                p_value: 1.0,
                certified: false,
            });
            continue;
        }

        // Exact-binomial p-value: P(Bin(n_served, alpha) <= failures) testing H0: risk > alpha.
        let p_value = binomial_cdf(n_served, failures, alpha);
        let certified = walk_active && p_value <= delta;

        if walk_active && !certified {
            // First failure in the qualified walk — no subsequent candidate may be certified.
            walk_active = false;
        }

        if certified {
            best = Some(diagnostics.len()); // index of the item about to be pushed
        }

        diagnostics.push(LttDiagnostic {
            lambda,
            n_served,
            empirical_risk,
            p_value,
            certified,
        });
    }

    let Some(best_idx) = best else {
        return mk_infeasible(alpha, delta, diagnostics);
    };

    let d = &diagnostics[best_idx];
    let false_accept_rate = if n_incorrect == 0 {
        None
    } else {
        // Among incorrect items, how many have score >= chosen threshold?
        let fa = sorted.iter().filter(|(s, c)| !c && *s >= d.lambda).count();
        Some(fa as f64 / n_incorrect as f64)
    };

    LttResult {
        alpha,
        delta,
        threshold: d.lambda,
        feasible: true,
        empirical_risk: d.empirical_risk,
        n_served: d.n_served,
        false_accept_rate,
        diagnostics,
    }
}

fn mk_infeasible(alpha: f64, delta: f64, diagnostics: Vec<LttDiagnostic>) -> LttResult {
    LttResult {
        alpha,
        delta,
        threshold: f64::INFINITY,
        feasible: false,
        empirical_risk: 0.0,
        n_served: 0,
        false_accept_rate: None,
        diagnostics,
    }
}

/// Exact binomial CDF: P(X ≤ k) for X ~ Binomial(n, p).
///
/// Computed in log-space to avoid underflow at large n.  The per-term recurrence
/// `log P(X = i) = log P(X = i−1) + log(n−i+1) − log(i) + log(p/(1−p))`
/// is numerically stable for the sample sizes firstpass operates on (up to ~10 000 pairs).
/// A log-sum-exp pass accumulates the k+1 terms without cancellation.
fn binomial_cdf(n: usize, k: usize, p: f64) -> f64 {
    if p <= 0.0 {
        // With p=0, X=0 with certainty, so P(X <= k) = 1 for all k >= 0.
        return 1.0;
    }
    if p >= 1.0 {
        // X = n with certainty.
        return if k >= n { 1.0 } else { 0.0 };
    }
    if k >= n {
        return 1.0;
    }

    let log_p = p.ln();
    let log_q = (1.0 - p).ln();
    let log_ratio = log_p - log_q; // log(p / (1−p))

    // Seed: log P(X = 0) = n · log(1−p)
    let mut log_term = n as f64 * log_q;
    let mut log_terms = Vec::with_capacity(k + 1);
    log_terms.push(log_term);

    for i in 1..=k {
        // Recurrence: log P(X=i) = log P(X=i−1) + log(n−i+1) − log(i) + log(p/(1−p))
        log_term += ((n - i + 1) as f64).ln() - (i as f64).ln() + log_ratio;
        log_terms.push(log_term);
    }

    // log-sum-exp: subtract max before exponentiating to prevent overflow/underflow.
    let max_log = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_log == f64::NEG_INFINITY {
        return 0.0;
    }
    let sum: f64 = log_terms.iter().map(|&l| (l - max_log).exp()).sum();
    (sum * max_log.exp()).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same inline SplitMix64 as conformal.rs — deterministic, dep-free draws for synthetic pairs.
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

    /// Same gate-like score distribution as conformal.rs: correct centers at 0.72, incorrect
    /// at 0.30, with ±0.2 uniform noise — disjoint seeds give independent draws.
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

    // ── 1. Binomial CDF correctness vs hand-computed values ──────────────────────────────────

    #[test]
    fn binomial_cdf_hand_computed() {
        // P(X <= 0 | Bin(10, 0.1)) = 0.9^10 ≈ 0.34868
        let got = binomial_cdf(10, 0, 0.1);
        assert!(
            (got - 0.34868_f64).abs() < 1e-4,
            "Bin(10,0.1) CDF at 0: {got}"
        );

        // P(X <= 1 | Bin(10, 0.1)) = 0.9^10 + 10*0.1*0.9^9 ≈ 0.73610
        let got = binomial_cdf(10, 1, 0.1);
        assert!(
            (got - 0.73610_f64).abs() < 1e-4,
            "Bin(10,0.1) CDF at 1: {got}"
        );

        // P(X <= 2 | Bin(5, 0.3)):
        //   P(0) = 0.7^5            ≈ 0.16807
        //   P(1) = 5*0.3*0.7^4     ≈ 0.36015
        //   P(2) = 10*0.09*0.7^3   ≈ 0.30870
        //   sum                    ≈ 0.83692
        let got = binomial_cdf(5, 2, 0.3);
        assert!(
            (got - 0.83692_f64).abs() < 1e-4,
            "Bin(5,0.3) CDF at 2: {got}"
        );

        // Edge cases.
        assert_eq!(binomial_cdf(10, 10, 0.5), 1.0, "k=n must be 1");
        assert_eq!(binomial_cdf(10, 0, 0.0), 1.0, "p=0 must be 1");
        assert_eq!(binomial_cdf(10, 9, 1.0), 0.0, "k<n and p=1 must be 0");
    }

    // ── 2. LTT guarantee holds on held-out data ──────────────────────────────────────────────

    #[test]
    fn certified_lambda_risk_at_most_alpha_on_held_out() {
        let alpha = 0.10;
        let calib = pairs(1, 4000);
        let test = pairs(2, 4000); // disjoint seed ⇒ held-out

        let r = calibrate(&calib, alpha, 0.05, 50);
        assert!(
            r.feasible,
            "should find a feasible threshold on n=4000 gate-like data"
        );

        let served_n = calib.iter().filter(|(s, _)| *s >= r.threshold).count();
        assert!(
            served_n as f64 / calib.len() as f64 > 0.2,
            "should serve > 20% of calibration set"
        );

        // Measure held-out risk at the certified threshold.
        let held_out_served: Vec<bool> = test
            .iter()
            .filter(|(s, _)| *s >= r.threshold)
            .map(|(_, c)| *c)
            .collect();
        assert!(
            !held_out_served.is_empty(),
            "must serve some held-out items"
        );
        let held_out_risk =
            held_out_served.iter().filter(|c| !**c).count() as f64 / held_out_served.len() as f64;

        // The LTT guarantee: held-out risk ≤ alpha (small tolerance for finite-sample variation;
        // the binomial test is conservative so the margin should be comfortable).
        assert!(
            held_out_risk <= alpha + 0.03,
            "held-out risk {held_out_risk:.3} must be ≤ alpha {alpha} (+tol)"
        );
    }

    // ── 3. Fixed-sequence stops at first failure even if risk dips ────────────────────────────

    #[test]
    fn fixed_sequence_stops_at_first_failure_even_if_risk_dips() {
        // Data layout (scores descending):
        //   λ = 0.95: 100 correct     → risk = 0/100 = 0%    → certified (p-value ≈ 0)
        //   λ = 0.80: +30 incorrect   → risk = 30/130 ≈ 23%  → FAIL → walk stops
        //   λ = 0.60: +200 correct    → risk = 30/330 ≈ 9%   → must NOT certify (walk stopped)
        let alpha = 0.10;
        let mut data: Vec<(f64, bool)> = Vec::new();
        for _ in 0..100 {
            data.push((0.95, true));
        }
        for _ in 0..30 {
            data.push((0.80, false));
        }
        for _ in 0..200 {
            data.push((0.60, true));
        }

        let r = calibrate(&data, alpha, 0.05, 30);
        // The λ = 0.95 row must be certified.
        let d095 = r
            .diagnostics
            .iter()
            .find(|d| (d.lambda - 0.95).abs() < 1e-9)
            .expect("0.95 must appear in diagnostics");
        assert!(d095.certified, "λ=0.95 must be certified");

        // The λ = 0.80 row failed the test and must break the walk.
        let d080 = r
            .diagnostics
            .iter()
            .find(|d| (d.lambda - 0.80).abs() < 1e-9)
            .expect("0.80 must appear in diagnostics");
        assert!(
            !d080.certified,
            "λ=0.80 has risk > alpha; must not be certified"
        );
        assert!(
            d080.p_value > 0.05,
            "λ=0.80 p-value must be large (H0 not rejected)"
        );

        // The λ = 0.60 row has low empirical risk but must NOT be certified (walk stopped).
        let d060 = r
            .diagnostics
            .iter()
            .find(|d| (d.lambda - 0.60).abs() < 1e-9)
            .expect("0.60 must appear in diagnostics");
        assert!(
            !d060.certified,
            "λ=0.60 must not be certified after walk stopped"
        );
        assert!(
            d060.empirical_risk <= alpha,
            "empirical risk at 0.60 dipped to {:.3} but must still be uncertified",
            d060.empirical_risk
        );

        // Chosen threshold is the last certified = 0.95.
        assert!(r.feasible);
        assert!((r.threshold - 0.95).abs() < 1e-9, "threshold must be 0.95");
    }

    // ── 4. Infeasible on tiny n ───────────────────────────────────────────────────────────────

    #[test]
    fn infeasible_on_tiny_n() {
        let tiny = vec![(0.9, true), (0.9, true), (0.1, false)];
        let r = calibrate(&tiny, 0.10, 0.05, 100);
        assert!(!r.feasible, "3 pairs below min_n=100 must be infeasible");
        assert_eq!(r.threshold, f64::INFINITY);
        assert_eq!(r.n_served, 0);
    }

    #[test]
    fn infeasible_target_zero_serves_nothing() {
        // alpha=0 is unachievable with any noisy gate.
        let r = calibrate(&pairs(3, 1000), 0.0, 0.05, 30);
        assert!(!r.feasible);
    }

    // ── 5. false_accept_rate is reported at chosen threshold ─────────────────────────────────

    #[test]
    fn false_accept_rate_reported_correctly() {
        // 200 correct at 0.9, 5 incorrect at 0.9 (false accepts), 15 incorrect at 0.2.
        // At λ=0.9: n_served=205, failures=5, risk≈2.4% < alpha=10% → certified.
        // false_accept_rate = 5/20 = 0.25 (5 of 20 total incorrect have score >= 0.9).
        let mut data: Vec<(f64, bool)> = Vec::new();
        for _ in 0..200 {
            data.push((0.9, true));
        }
        for _ in 0..5 {
            data.push((0.9, false));
        }
        for _ in 0..15 {
            data.push((0.2, false));
        }

        let r = calibrate(&data, 0.10, 0.05, 30);
        assert!(r.feasible, "must certify a threshold on this data");
        let far = r
            .false_accept_rate
            .expect("must report a false_accept_rate");
        assert!(
            (far - 0.25).abs() < 1e-9,
            "false_accept_rate must be 5/20 = 0.25, got {far}"
        );
    }
}
