//! A seeded noisy oracle prior over per-rung pass probability, and the expected-cost start-rung
//! rule built on it — the ingredients for stress-testing a "Jev-style" pre-generation
//! decision-model prior against Firstpass's gate (see [`crate::prior_sweep`]).
//!
//! # What "true pass probability" means here
//!
//! The simulation's ground truth for whether a rung clears a task is [`SimBackend::clearance`] —
//! the exact probability [`SimBackend::run`](crate::sim::SimBackend) draws its `correct` bit
//! against (sim.rs). That is the closest faithful quantity to "true per-rung pass probability" the
//! simulator exposes, so the noisy oracle prior below perturbs
//! `SimBackend::clearance(rung.strength, task.difficulty)` — never a quantity invented just for
//! this study.
//!
//! # Monotonicity
//!
//! A real decision model's prior is monotone non-decreasing across the ladder by construction: a
//! stronger model is never predicted weaker than a cheaper one it dominates. The noise below is
//! drawn independently per rung and would not have that property on its own, so a running max is
//! applied afterward — matching what a real prior would guarantee, not manufacturing an unrealistic
//! advantage (the noise still perturbs *which* rung looks best; the max only removes an
//! internally-inconsistent prior).

use crate::sim::{Rung, SimBackend, Task, hash01};
use firstpass_core::PriceTable;

/// One standard-normal draw from the bench's existing deterministic `hash01` uniform source
/// (Box-Muller), keyed the same way every other sim draw is: seed plus two coordinates.
fn gaussian01(seed: u64, a: u64, b: u64) -> f64 {
    let u1 = hash01(seed, a, b).max(1e-12);
    let u2 = hash01(seed ^ 0x9E37_79B9_7F4A_7C15, a, b);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Noisy oracle prior: `P(pass)` per ladder rung for `task`, perturbing the sim's true per-rung
/// clearance probability with `N(0, sigma)` noise and clamping to `[0, 1]`, then taking a running
/// max across rungs so the result is monotone non-decreasing (see module doc). `sigma = 0.0`
/// returns the true clearance probabilities unperturbed.
#[must_use]
pub fn noisy_prior(task: &Task, ladder: &[Rung], sigma: f64, seed: u64) -> Vec<f64> {
    let mut running_max = 0.0_f64;
    ladder
        .iter()
        .enumerate()
        .map(|(idx, rung)| {
            let true_p = SimBackend::clearance(rung.strength, task.difficulty);
            let noise = if sigma > 0.0 {
                gaussian01(seed, task.id, idx as u64) * sigma
            } else {
                0.0
            };
            let noisy = (true_p + noise).clamp(0.0, 1.0);
            running_max = running_max.max(noisy);
            running_max
        })
        .collect()
}

/// The expected-cost argmin over candidate start rungs — mirrors `firstpass-proxy`'s
/// `argmin_expected_cost` (`crates/firstpass-proxy/src/bandit.rs:147`) so the bench studies the
/// same start-rung rule; `firstpass-bench` cannot depend on `firstpass-proxy`, so the walk is
/// reimplemented here against the sim's own [`Task`]/[`Rung`]/[`PriceTable`] types.
///
/// Walks the ladder from each candidate start `s`, accumulating `P(reach r) · price(r)` with
/// `P(reach r+1) = P(reach r) · (1 − prior[r])`. Ties prefer the lower start: `<`, not `<=`, so the
/// first (cheapest) rung achieving the minimum wins.
#[must_use]
pub fn argmin_expected_cost(
    task: &Task,
    ladder: &[Rung],
    prices: &PriceTable,
    prior: &[f64],
) -> usize {
    // Same in/out token accounting SimBackend::run uses, so the priced rung matches what would
    // actually be billed if the ladder started there.
    let out_tokens = |r: &Rung| 300 + (r.strength * 500.0) as u64;

    let mut best_s = 0usize;
    let mut best_cost = f64::MAX;
    for s in 0..ladder.len() {
        let mut expected_cost = 0.0_f64;
        let mut p_reach = 1.0_f64;
        for (r, rung) in ladder.iter().enumerate().skip(s) {
            let price = prices
                .cost_usd(&rung.model, task.prompt_tokens, out_tokens(rung))
                .unwrap_or(0.0);
            expected_cost += p_reach * price;
            p_reach *= 1.0 - prior.get(r).copied().unwrap_or(0.0);
            if p_reach < 1e-10 {
                break;
            }
        }
        if expected_cost < best_cost {
            best_cost = expected_cost;
            best_s = s;
        }
    }
    best_s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder() -> Vec<Rung> {
        vec![
            Rung::new("anthropic/claude-haiku-4-5", 0.62),
            Rung::new("anthropic/claude-sonnet-5", 0.80),
            Rung::new("anthropic/claude-opus-4-8", 0.93),
        ]
    }

    fn task(id: u64, difficulty: f64) -> Task {
        Task {
            id,
            difficulty,
            prompt_tokens: 800,
            prompt: None,
            expected: None,
        }
    }

    #[test]
    fn sigma_zero_returns_true_clearance() {
        let t = task(1, 0.7);
        let ladder = ladder();
        let prior = noisy_prior(&t, &ladder, 0.0, 42);
        for (p, rung) in prior.iter().zip(ladder.iter()) {
            assert!(
                (p - SimBackend::clearance(rung.strength, t.difficulty)).abs() < 1e-9,
                "sigma=0 must reproduce the true clearance exactly"
            );
        }
    }

    #[test]
    fn prior_is_monotone_non_decreasing() {
        let ladder = ladder();
        for id in 0..50 {
            // A high sigma is exactly the case that would break monotonicity without the running
            // max — the property under test.
            let prior = noisy_prior(&task(id, hash01(id, id, 3)), &ladder, 0.6, id);
            assert!(
                prior.windows(2).all(|w| w[1] >= w[0]),
                "prior must be monotone non-decreasing, got {prior:?}"
            );
        }
    }

    #[test]
    fn different_seeds_perturb_differently() {
        let t = task(1, 0.7);
        let ladder = ladder();
        let a = noisy_prior(&t, &ladder, 0.3, 1);
        let b = noisy_prior(&t, &ladder, 0.3, 2);
        assert_ne!(a, b, "different seeds must draw different noise");
    }

    #[test]
    fn ties_prefer_the_lower_start() {
        // A single-price ladder (same model at every rung), each rung a CERTAIN pass: every start
        // only ever pays for its own rung (nothing escalates), so every start costs identically
        // `price(rung)`, and with equal prices the lower start must win the tie.
        let ladder = vec![
            Rung::new("anthropic/claude-haiku-4-5", 0.62),
            Rung::new("anthropic/claude-haiku-4-5", 0.62),
        ];
        let prices = PriceTable::defaults();
        let t = task(1, 0.5);
        let prior = vec![1.0, 1.0];
        assert_eq!(argmin_expected_cost(&t, &ladder, &prices, &prior), 0);
    }

    #[test]
    fn an_expensive_near_hopeless_cheap_rung_is_skipped() {
        // A task so hard every rung's clearance clamps to the floor (0.02): with the cheap rung
        // essentially never paying off, the expected-cost rule should skip straight past it rather
        // than pay for it on every request (mirrors costaware.rs's finding).
        let ladder = ladder();
        let prices = PriceTable::defaults();
        let t = task(1, 3.0);
        let prior = noisy_prior(&t, &ladder, 0.0, 1);
        assert!(
            argmin_expected_cost(&t, &ladder, &prices, &prior) > 0,
            "a near-hopeless cheap rung must be skipped, prior={prior:?}"
        );
    }
}
