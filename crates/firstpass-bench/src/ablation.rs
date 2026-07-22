//! Start-rung ablation — does a *learned* start rung actually save money, or is the cascade
//! just safe? (Answers the honest "is the cost-brain smart" question from the roadmap.)
//!
//! Three start-rung policies over the same tasks and ladder:
//! - **start-0** — the naive cascade: always open on the cheapest rung and climb. Pays the
//!   doomed-attempt tax on hard tasks.
//! - **oracle** — start exactly on the cheapest rung that would pass. The unreachable ceiling.
//! - **bandit** — learn, per observable context bucket, the cost-minimizing start rung on a
//!   train split, then apply it on a held-out split. This is what firstpass's start-rung
//!   bandit / predictor is trying to be.
//!
//! The decisive number is **gap-closed** = `(start0_cost − bandit_cost) / (start0_cost −
//! oracle_cost)`: the fraction of the achievable headroom the learned policy captures. It is
//! reported as a function of how well the observable context predicts difficulty (`signal`),
//! because that is the truth the roadmap must be honest about: **a learned start rung is only
//! as smart as context is predictive.** At `signal = 0` (context tells you nothing) even a
//! perfect learner closes ~0% — starting anywhere but rung 0 just gambles. At `signal = 1`
//! (context fully predicts difficulty) it approaches the oracle. Real workloads live in
//! between, which is exactly why `firstpass predictor-eval` measures the real signal on real
//! receipts; this ablation measures the *mechanism's* headroom.

use crate::sim::{SimBackend, hash01};

/// A ladder rung: relative strength (clears harder tasks) and USD price. Prices increase with
/// strength — that is what makes starting too high wasteful and starting too low a tax.
#[derive(Debug, Clone, Copy)]
struct Rung {
    strength: f64,
    price: f64,
}

fn default_ladder() -> Vec<Rung> {
    vec![
        Rung {
            strength: 0.62,
            price: 1.0,
        }, // haiku-class
        Rung {
            strength: 0.80,
            price: 5.0,
        }, // sonnet-class
        Rung {
            strength: 0.95,
            price: 25.0,
        }, // opus-class
    ]
}

/// The cheapest rung that clears `difficulty` (pass = clearance >= 0.5); the top rung if none do.
fn min_pass_rung(ladder: &[Rung], difficulty: f64) -> usize {
    ladder
        .iter()
        .position(|r| SimBackend::clearance(r.strength, difficulty) >= 0.5)
        .unwrap_or(ladder.len() - 1)
}

/// Cost of a cascade that opens on rung `start` for a task whose cheapest-passing rung is
/// `mpr`: if `start > mpr` it passes immediately (overpay `price[start]`); otherwise it climbs
/// `start..=mpr`, paying every rung on the way.
fn cascade_cost(ladder: &[Rung], start: usize, mpr: usize) -> f64 {
    let start = start.min(ladder.len() - 1);
    if start >= mpr {
        ladder[start].price
    } else {
        ladder[start..=mpr].iter().map(|r| r.price).sum()
    }
}

/// One ablation task: an observable context bucket and a hidden difficulty.
struct AblationTask {
    ctx: usize,
    difficulty: f64,
}

/// Generate `n` tasks over `n_ctx` context buckets. `signal` in `[0, 1]` controls how much the
/// (observable) context predicts the (hidden) difficulty: `difficulty = signal·(ctx/(n_ctx−1))
/// + (1−signal)·noise`. `signal = 1` → context fully predicts; `signal = 0` → context useless.
fn generate(n: usize, n_ctx: usize, signal: f64, seed: u64) -> Vec<AblationTask> {
    (0..n)
        .map(|i| {
            let id = i as u64;
            let ctx = (hash01(seed, id, 1) * n_ctx as f64) as usize;
            let ctx = ctx.min(n_ctx - 1);
            let structured = ctx as f64 / (n_ctx.max(2) - 1) as f64;
            let noise = hash01(seed, id, 2);
            let difficulty = (signal * structured + (1.0 - signal) * noise).clamp(0.0, 1.0);
            AblationTask { ctx, difficulty }
        })
        .collect()
}

/// Result of one ablation run at a given `signal`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AblationResult {
    /// How predictive the observable context is of difficulty (0 = none, 1 = perfect).
    pub signal: f64,
    /// Held-out tasks evaluated.
    pub n_eval: usize,
    /// Mean per-task cost, naive start-0 cascade.
    pub start0_cost: f64,
    /// Mean per-task cost, learned per-context start rung (trained on the other split).
    pub bandit_cost: f64,
    /// Mean per-task cost, oracle start rung (cheapest passing).
    pub oracle_cost: f64,
    /// Fraction of the `start0 → oracle` headroom the learned policy captures, in `(-inf, 1]`.
    /// `0` = no better than start-0; `1` = matches the oracle. Can go slightly negative if the
    /// learned start overshoots on a hard-to-predict split.
    pub gap_closed: f64,
}

/// Run the ablation at one `signal` level: split tasks into train/eval, learn the
/// cost-minimizing start rung per context on train, and compare the three policies on eval.
#[must_use]
pub fn run_ablation(n: usize, n_ctx: usize, signal: f64, seed: u64) -> AblationResult {
    let ladder = default_ladder();
    let tasks = generate(n, n_ctx, signal, seed);
    let split = n / 2;
    let (train, eval) = tasks.split_at(split);

    // Learn, per context, the start rung that minimizes mean cascade cost over the train split.
    // (Cost-minimizing, not difficulty-mean — starting a touch low is cheaper than a touch high.)
    let mut best_start = vec![0usize; n_ctx];
    for (ctx, slot) in best_start.iter_mut().enumerate() {
        let members: Vec<usize> = train
            .iter()
            .filter(|t| t.ctx == ctx)
            .map(|t| min_pass_rung(&ladder, t.difficulty))
            .collect();
        if members.is_empty() {
            continue; // unseen context → keep start 0 (cold-start safety, like the real bandit)
        }
        *slot = (0..ladder.len())
            .min_by(|&a, &b| {
                let ca: f64 = members
                    .iter()
                    .map(|&mpr| cascade_cost(&ladder, a, mpr))
                    .sum();
                let cb: f64 = members
                    .iter()
                    .map(|&mpr| cascade_cost(&ladder, b, mpr))
                    .sum();
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
    }

    let mut s0 = 0.0_f64;
    let mut sb = 0.0_f64;
    let mut so = 0.0_f64;
    for t in eval {
        let mpr = min_pass_rung(&ladder, t.difficulty);
        s0 += cascade_cost(&ladder, 0, mpr);
        sb += cascade_cost(&ladder, best_start[t.ctx], mpr);
        so += cascade_cost(&ladder, mpr, mpr);
    }
    let n_eval = eval.len().max(1) as f64;
    let (start0_cost, bandit_cost, oracle_cost) = (s0 / n_eval, sb / n_eval, so / n_eval);
    let headroom = start0_cost - oracle_cost;
    let gap_closed = if headroom.abs() < 1e-12 {
        0.0
    } else {
        (start0_cost - bandit_cost) / headroom
    };
    AblationResult {
        signal,
        n_eval: eval.len(),
        start0_cost,
        bandit_cost,
        oracle_cost,
        gap_closed,
    }
}

/// Run the ablation across a sweep of `signal` levels and render the honest verdict.
#[must_use]
pub fn run_ablation_sweep(n: usize, seed: u64) -> String {
    let n_ctx = 6;
    let mut out = String::from(
        "# Start-rung ablation — is the cost-brain smart, or just safe?\n\n\
         A learned start rung is only as smart as context is predictive of difficulty.\n\
         gap-closed = fraction of the (start-0 → oracle) cost headroom the learned policy captures.\n\n\
         signal  start0   bandit   oracle   gap-closed\n",
    );
    for &signal in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let r = run_ablation(n, n_ctx, signal, seed);
        out.push_str(&format!(
            " {:.2}   {:>6.2}   {:>6.2}   {:>6.2}   {:>6.1}%\n",
            r.signal,
            r.start0_cost,
            r.bandit_cost,
            r.oracle_cost,
            r.gap_closed * 100.0,
        ));
    }
    out.push_str(
        "\nRead: at signal 0 (context useless) even a perfect learner closes ~0% — routing on a\n\
         non-predictive context just gambles. As context predicts difficulty, the learned start\n\
         rung captures more of the headroom. The REAL number for your traffic is whatever\n\
         `firstpass predictor-eval` reports on your receipts; this shows the mechanism's ceiling.",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_signal_closes_most_of_the_gap_useless_signal_closes_little() {
        // signal = 1: context fully predicts difficulty → the learned start rung should capture
        // most of the headroom (well above half).
        let strong = run_ablation(4000, 6, 1.0, 7);
        assert!(
            strong.gap_closed > 0.6,
            "predictive context: learner should close most of the gap, got {:.2}",
            strong.gap_closed
        );
        assert!(
            strong.bandit_cost < strong.start0_cost,
            "learner must beat start-0"
        );
        assert!(
            strong.bandit_cost >= strong.oracle_cost - 1e-9,
            "can't beat the oracle"
        );

        // signal = 0: context is noise → the learner cannot do meaningfully better than start-0.
        let none = run_ablation(4000, 6, 0.0, 7);
        assert!(
            none.gap_closed < 0.35,
            "useless context: learner shouldn't magically close the gap, got {:.2}",
            none.gap_closed
        );
    }

    #[test]
    fn oracle_never_costs_more_than_start0() {
        for &sig in &[0.0, 0.5, 1.0] {
            let r = run_ablation(2000, 6, sig, 11);
            assert!(
                r.oracle_cost <= r.start0_cost + 1e-9,
                "oracle (cheapest passing) can never cost more than climbing from 0"
            );
        }
    }
}
