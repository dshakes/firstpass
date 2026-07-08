//! Reproducible bootstrap confidence intervals. Every reported number carries an interval, and
//! the interval is deterministic (seeded resampling) so the proof is reproducible bit-for-bit.

use crate::sim::Rng;
use serde::Serialize;

/// A point estimate with a `[lo, hi]` confidence interval.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Ci {
    /// Point estimate on the full sample.
    pub point: f64,
    /// Lower confidence bound.
    pub lo: f64,
    /// Upper confidence bound.
    pub hi: f64,
}

/// Arithmetic mean (0.0 for an empty slice).
#[must_use]
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Bootstrap CI for the mean of `values` at level `1 - alpha`, using `b` resamples.
#[must_use]
pub fn bootstrap_mean_ci(values: &[f64], b: usize, seed: u64, alpha: f64) -> Ci {
    let point = mean(values);
    if values.is_empty() {
        return Ci {
            point,
            lo: 0.0,
            hi: 0.0,
        };
    }
    let mut rng = Rng::new(seed);
    let n = values.len();
    let mut stats = Vec::with_capacity(b);
    for _ in 0..b {
        let mut acc = 0.0;
        for _ in 0..n {
            acc += values[rng.below(n)];
        }
        stats.push(acc / n as f64);
    }
    stats.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
    Ci {
        point,
        lo: percentile(&stats, alpha / 2.0),
        hi: percentile(&stats, 1.0 - alpha / 2.0),
    }
}

/// Bootstrap CI for the ratio `sum(num) / sum(den)` — e.g. total-cost / successes for
/// `$/successful-task`. Resamples task indices jointly so numerator and denominator stay paired.
#[must_use]
pub fn bootstrap_ratio_ci(num: &[f64], den: &[f64], b: usize, seed: u64, alpha: f64) -> Ci {
    assert_eq!(num.len(), den.len(), "num/den must be paired per task");
    let ratio = |ns: f64, ds: f64| if ds > 0.0 { ns / ds } else { 0.0 };
    let point = ratio(num.iter().sum(), den.iter().sum());
    if num.is_empty() {
        return Ci {
            point,
            lo: 0.0,
            hi: 0.0,
        };
    }
    let mut rng = Rng::new(seed);
    let n = num.len();
    let mut stats = Vec::with_capacity(b);
    for _ in 0..b {
        let (mut ns, mut ds) = (0.0, 0.0);
        for _ in 0..n {
            let i = rng.below(n);
            ns += num[i];
            ds += den[i];
        }
        stats.push(ratio(ns, ds));
    }
    stats.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
    Ci {
        point,
        lo: percentile(&stats, alpha / 2.0),
        hi: percentile(&stats, 1.0 - alpha / 2.0),
    }
}

/// The `q`-quantile of `values` (0.0 for empty). Used for latency percentiles.
#[must_use]
pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&v, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_brackets_the_point_and_is_deterministic() {
        let v: Vec<f64> = (0..1000).map(|i| (i % 2) as f64).collect(); // mean 0.5
        let a = bootstrap_mean_ci(&v, 500, 123, 0.05);
        let b = bootstrap_mean_ci(&v, 500, 123, 0.05);
        assert_eq!(a.lo, b.lo, "same seed => same CI");
        assert_eq!(a.hi, b.hi);
        assert!((a.point - 0.5).abs() < 1e-9);
        assert!(a.lo <= a.point && a.point <= a.hi);
        assert!(a.hi - a.lo < 0.15, "1000 samples should give a tight CI");
    }

    #[test]
    fn ratio_ci_matches_hand_computed_point() {
        let cost = vec![1.0, 2.0, 3.0, 4.0];
        let success = vec![1.0, 0.0, 1.0, 1.0]; // 3 successes, total cost 10 => 3.333/success
        let ci = bootstrap_ratio_ci(&cost, &success, 200, 1, 0.05);
        assert!((ci.point - (10.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn quantiles() {
        let v: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert!((quantile(&v, 0.5) - 51.0).abs() <= 1.0);
        assert!(quantile(&v, 0.95) >= 95.0);
    }
}
