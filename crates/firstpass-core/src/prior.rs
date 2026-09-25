//! Verified predictive routing: pure mapping from a decision-model's per-rung "least capable
//! tier" choice probabilities to a cumulative gate-pass prior over the ladder.
//!
//! I/O-free by construction (fetching the raw probabilities from the decision-model provider
//! lives in `firstpass-proxy`). This module only does the deterministic math so it is testable
//! in isolation and so the mapping is auditable independent of which provider produced the
//! numbers.
//!
//! # Semantics
//!
//! The upstream model answers a single choice question: "which is the least capable rung that
//! fully and correctly handles this request?" — one option per rung, in ladder order. Its
//! `probabilities` map is therefore `P(rung r is the least capable rung that suffices)`, and the
//! event "rung `s` passes" is "the least capable sufficient rung is `<= s`" (a *more* capable
//! rung always suffices too). So `P(pass | start s) = Σ_{r<=s} P(least-capable-sufficient = r)`
//! — a running cumulative sum, monotone non-decreasing, ending at 1.0 (some rung must suffice).

/// Map raw per-rung "least capable sufficient tier" probabilities (in ladder order) to a
/// cumulative per-rung gate-pass prior, for the start-rung expected-cost argmin.
///
/// Returns `None` when the input can't be turned into a sane probability distribution:
/// - empty
/// - any entry non-finite or negative
/// - the entries sum to `<= 0` (nothing to normalize against)
///
/// Otherwise: normalizes by the sum (so a caller need not pass an already-normalized
/// distribution), takes the running cumulative sum, and clamps each partial sum to `[0, 1]` to
/// absorb floating-point drift. The last element is forced to exactly `1.0` (some rung must
/// suffice).
#[must_use]
pub fn cumulative_pass(probs_in_rung_order: &[f64]) -> Option<Vec<f64>> {
    if probs_in_rung_order.is_empty() {
        return None;
    }
    if probs_in_rung_order
        .iter()
        .any(|p| !p.is_finite() || *p < 0.0)
    {
        return None;
    }
    let sum: f64 = probs_in_rung_order.iter().sum();
    if sum <= 0.0 {
        return None;
    }

    let mut running = 0.0;
    let mut out: Vec<f64> = probs_in_rung_order
        .iter()
        .map(|p| {
            running += p / sum;
            running.clamp(0.0, 1.0)
        })
        .collect();
    if let Some(last) = out.last_mut() {
        *last = 1.0;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty() {
        assert_eq!(cumulative_pass(&[]), None);
    }

    #[test]
    fn rejects_negative() {
        assert_eq!(cumulative_pass(&[0.5, -0.1, 0.6]), None);
    }

    #[test]
    fn rejects_non_finite() {
        assert_eq!(cumulative_pass(&[0.5, f64::NAN, 0.5]), None);
        assert_eq!(cumulative_pass(&[0.5, f64::INFINITY]), None);
    }

    #[test]
    fn rejects_all_zero() {
        assert_eq!(cumulative_pass(&[0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn normalizes_and_cumsums() {
        // Already normalized: 0.2, 0.3, 0.5 -> cumsum 0.2, 0.5, 1.0.
        let out = cumulative_pass(&[0.2, 0.3, 0.5]).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.2).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn normalizes_unnormalized_input() {
        // Sums to 4: 1, 1, 2 -> normalized 0.25, 0.25, 0.5 -> cumsum 0.25, 0.5, 1.0.
        let out = cumulative_pass(&[1.0, 1.0, 2.0]).unwrap();
        assert!((out[0] - 0.25).abs() < 1e-12);
        assert!((out[1] - 0.5).abs() < 1e-12);
        assert_eq!(out[2], 1.0);
    }

    #[test]
    fn is_monotone_non_decreasing() {
        let out = cumulative_pass(&[0.05, 0.15, 0.3, 0.5]).unwrap();
        assert!(out.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn single_rung_is_certain() {
        let out = cumulative_pass(&[0.37]).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn last_is_exactly_one() {
        // Pathological weights that could leave floating-point drift below 1.0 pre-clamp.
        let out = cumulative_pass(&[1e-10, 1e-10, 1e-10]).unwrap();
        assert_eq!(*out.last().unwrap(), 1.0);
    }
}
