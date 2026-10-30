//! Certified model selection: an anytime-valid, sequential non-inferiority test between the route's
//! incumbent ladder and a challenger ladder, run on live traffic outcomes (ADR 0014).
//!
//! ## Why this exists
//!
//! On 500 real SWE-bench agent runs per model, no per-request gate available on a finished agent
//! run was precise enough for escalation to beat always-top; what did win was choosing the right
//! model **for the workload** (`docs/benchmarks/agentic-real-runs.md`: −71% cost, quality loss not
//! significant). Choosing that model needs evidence from the route's own traffic, and the only
//! honest evidence for agentic work is the task outcome itself. So the route serves a randomized
//! fraction of sessions from a challenger ladder, and this module decides — at whatever moment the
//! operator looks — whether the challenger is proven no worse than the incumbent by more than a
//! margin, proven worse, or not yet known.
//!
//! ## The test
//!
//! Sessions are randomized to arms independently of everything else, so the `k`-th incumbent
//! outcome and the `k`-th challenger outcome are an unbiased pair. With `D = X_inc − X_ch ∈ [−1, 1]`
//! and margin `m`, two betting e-processes run side by side:
//!
//! ```text
//! certify : H0  E[D] ≥ m     wealth ·= 1 + λ (m − D),   λ ∈ (0, 1/(1−m))
//! halt    : H0  E[D] ≤ m     wealth ·= 1 + λ (D − m),   λ ∈ (0, 1/(1+m))
//! ```
//!
//! Each is a nonnegative supermartingale under its null, so by Ville's inequality it crosses
//! `1/alpha` with probability at most `alpha` **at any time, however often it is read**. That is
//! the property a live experiment needs: the operator (or the halt guard) can check after every
//! session without inflating the error rate, which a fixed-sample test re-run on a growing stream
//! does not allow. Each wealth is an equal-weight mixture over a fixed grid of `λ`, which is itself
//! an e-process and avoids having to know the effect size in advance.
//!
//! Decisions are sticky: once either process crosses, the outcome is final for this experiment.
//! A challenger that is later proven worse after certifying would need a new experiment — the
//! guarantee is about the first crossing, and re-opening it would be optional stopping again.
//!
//! ## What it does not claim
//!
//! - It certifies *outcome* non-inferiority on the traffic it saw. A workload shift after
//!   certification is not covered; a fresh experiment is.
//! - Cost is reported, not tested: a challenger is only worth adopting if it is also cheaper, and
//!   [`Experiment::report`] carries the per-request cost of each arm so that comparison is explicit.
//! - Outcomes must be in `[0, 1]` and one per session. Several outcomes from one session are
//!   correlated, which breaks the pairing argument, so the caller deduplicates by session.
//!
//! Validated on real agent runs before it was written: `scripts/sim-certified-selection.py` replays a
//! live A/B against opus-4.5 on SWE-bench Verified ($0, public data). At `m = 0.05`, `alpha = 0.05`
//! it certifies the three challengers within 1.2pp of opus in 99% of replays (median ≈ 3.6k sessions)
//! and halts gpt-5-mini (−20.6pp) in a median of 223 sessions; at the null boundary it certifies in
//! 4.5% of replays.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Fractions of `λ_max` the mixture bets at. Spread from timid to aggressive so the mixture grows
/// near the best fixed bet for whatever the true effect turns out to be.
const LAMBDA_GRID: [f64; 18] = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90,
];

/// Default non-inferiority margin: the challenger may be at most 5 points worse. Chosen from the
/// replay above — at 3 points the same challengers needed ~9k sessions and certified only 63% of
/// the time within 20k, which is longer than most operators will run an experiment.
pub const DEFAULT_MARGIN: f64 = 0.05;

/// Default error rate for each of the two conclusions.
pub const DEFAULT_ALPHA: f64 = 0.05;

/// Which ladder served a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    /// The route's configured ladder.
    Incumbent,
    /// The ladder under evaluation.
    Challenger,
}

/// Where an experiment stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Not enough evidence either way. Keep running.
    Undecided,
    /// The challenger is no worse than the incumbent by more than the margin, with error at most
    /// `alpha`. It is a candidate to adopt if it is also cheaper — a human decision.
    Certified,
    /// The challenger is worse than the incumbent by more than the margin, with error at most
    /// `alpha`. Stop serving it.
    Halted,
}

/// A grid-mixture betting e-process for `H0: E[sign · (D − m)] ≤ 0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Mixture {
    sign: f64,
    lambdas: Vec<f64>,
    wealth: Vec<f64>,
}

impl Mixture {
    fn new(margin: f64, sign: f64) -> Self {
        // `1 + λ·sign·(D − m)` must stay nonnegative for every `D ∈ [−1, 1]`. The worst case is
        // `D = −1` when betting on `D − m` (sign +1) and `D = 1` when betting on `m − D`.
        let worst = if sign > 0.0 {
            1.0 + margin
        } else {
            1.0 - margin
        };
        Self {
            sign,
            lambdas: LAMBDA_GRID.iter().map(|g| g / worst).collect(),
            wealth: vec![1.0; LAMBDA_GRID.len()],
        }
    }

    fn step(&mut self, d: f64, margin: f64) {
        let x = self.sign * (d - margin);
        for (w, lambda) in self.wealth.iter_mut().zip(&self.lambdas) {
            *w *= 1.0 + lambda * x;
        }
    }

    fn e_value(&self) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the grid has 18 entries; exact in f64"
        )]
        let n = self.wealth.len() as f64;
        self.wealth.iter().sum::<f64>() / n
    }
}

/// Running totals for one arm. Diagnostics and cost — never part of the guarantee.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ArmStats {
    /// Session outcomes observed.
    pub outcomes: u64,
    /// Sum of outcomes (successes, for binary outcomes).
    pub outcome_sum: f64,
    /// Requests served by this arm.
    pub requests: u64,
    /// USD spent serving those requests.
    pub cost_usd: f64,
}

impl ArmStats {
    /// Mean outcome, if any were observed.
    #[must_use]
    pub fn success_rate(&self) -> Option<f64> {
        #[expect(clippy::cast_precision_loss, reason = "counts far below 2^52")]
        (self.outcomes > 0).then(|| self.outcome_sum / self.outcomes as f64)
    }

    /// Mean USD per request, if any were served.
    #[must_use]
    pub fn cost_per_request(&self) -> Option<f64> {
        #[expect(clippy::cast_precision_loss, reason = "counts far below 2^52")]
        (self.requests > 0).then(|| self.cost_usd / self.requests as f64)
    }
}

/// A point-in-time summary of an experiment, safe to serialize into a report or an API response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Current decision.
    pub status: Status,
    /// Non-inferiority margin `m`.
    pub margin: f64,
    /// Error rate for each conclusion.
    pub alpha: f64,
    /// Pairs consumed by the test.
    pub pairs: u64,
    /// Pair at which the decision was reached, if it has been.
    pub decided_at_pair: Option<u64>,
    /// E-value against "challenger is worse by ≥ m". Certifies at `1/alpha`.
    pub e_certify: f64,
    /// E-value against "challenger is worse by ≤ m". Halts at `1/alpha`.
    pub e_halt: f64,
    /// Incumbent arm totals.
    pub incumbent: ArmStats,
    /// Challenger arm totals.
    pub challenger: ArmStats,
}

/// A sequential non-inferiority experiment between two arms. Pure and deterministic: the same
/// outcome stream always yields the same decision, so a report can be re-derived from receipts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    margin: f64,
    alpha: f64,
    certify: Mixture,
    halt: Mixture,
    pending_incumbent: VecDeque<f64>,
    pending_challenger: VecDeque<f64>,
    pairs: u64,
    status: Status,
    decided_at_pair: Option<u64>,
    incumbent: ArmStats,
    challenger: ArmStats,
}

impl Experiment {
    /// A new experiment with no evidence.
    ///
    /// # Errors
    /// If `margin` is not in `(0, 0.5]` or `alpha` is not in `(0, 0.5)`. A zero margin can never
    /// certify and a margin above half is not a quality bar; an `alpha` of a half or more is a coin.
    pub fn new(margin: f64, alpha: f64) -> Result<Self, String> {
        if !margin.is_finite() || margin <= 0.0 || margin > 0.5 {
            return Err(format!("margin must be in (0, 0.5], got {margin}"));
        }
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 0.5 {
            return Err(format!("alpha must be in (0, 0.5), got {alpha}"));
        }
        Ok(Self {
            margin,
            alpha,
            certify: Mixture::new(margin, -1.0),
            halt: Mixture::new(margin, 1.0),
            pending_incumbent: VecDeque::new(),
            pending_challenger: VecDeque::new(),
            pairs: 0,
            status: Status::Undecided,
            decided_at_pair: None,
            incumbent: ArmStats::default(),
            challenger: ArmStats::default(),
        })
    }

    /// Record one session's outcome in `[0, 1]` (1 = the task succeeded).
    ///
    /// Returns `false` and records nothing if `outcome` is not a finite value in `[0, 1]`: an
    /// out-of-range value would break the nonnegativity the guarantee rests on, and clamping it
    /// would silently change what the caller reported.
    pub fn observe_outcome(&mut self, arm: Arm, outcome: f64) -> bool {
        if !outcome.is_finite() || !(0.0..=1.0).contains(&outcome) {
            return false;
        }
        let stats = self.stats_mut(arm);
        stats.outcomes += 1;
        stats.outcome_sum += outcome;
        match arm {
            Arm::Incumbent => self.pending_incumbent.push_back(outcome),
            Arm::Challenger => self.pending_challenger.push_back(outcome),
        }
        while let (Some(&x_inc), Some(&x_ch)) = (
            self.pending_incumbent.front(),
            self.pending_challenger.front(),
        ) {
            self.pending_incumbent.pop_front();
            self.pending_challenger.pop_front();
            self.pairs += 1;
            // Once decided, the totals keep moving for the report but the test is closed.
            if self.status == Status::Undecided {
                let d = x_inc - x_ch;
                self.certify.step(d, self.margin);
                self.halt.step(d, self.margin);
                let level = 1.0 / self.alpha;
                if self.certify.e_value() >= level {
                    self.status = Status::Certified;
                    self.decided_at_pair = Some(self.pairs);
                } else if self.halt.e_value() >= level {
                    self.status = Status::Halted;
                    self.decided_at_pair = Some(self.pairs);
                }
            }
        }
        true
    }

    /// Record the cost of one request served by `arm`. Negative or non-finite costs are ignored.
    pub fn observe_cost(&mut self, arm: Arm, cost_usd: f64) {
        if !cost_usd.is_finite() || cost_usd < 0.0 {
            return;
        }
        let stats = self.stats_mut(arm);
        stats.requests += 1;
        stats.cost_usd += cost_usd;
    }

    /// Current decision.
    #[must_use]
    pub fn status(&self) -> Status {
        self.status
    }

    /// A summary of the experiment as it stands.
    #[must_use]
    pub fn report(&self) -> Report {
        Report {
            status: self.status,
            margin: self.margin,
            alpha: self.alpha,
            pairs: self.pairs,
            decided_at_pair: self.decided_at_pair,
            e_certify: self.certify.e_value(),
            e_halt: self.halt.e_value(),
            incumbent: self.incumbent,
            challenger: self.challenger,
        }
    }

    fn stats_mut(&mut self, arm: Arm) -> &mut ArmStats {
        match arm {
            Arm::Incumbent => &mut self.incumbent,
            Arm::Challenger => &mut self.challenger,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift so the statistical tests are reproducible without a dependency.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            #[expect(clippy::cast_precision_loss, reason = "53-bit mantissa draw")]
            let v = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            v
        }
        fn bern(&mut self, p: f64) -> f64 {
            if self.next_f64() < p { 1.0 } else { 0.0 }
        }
    }

    /// Run one experiment with Bernoulli arms until decided or `cap` pairs.
    fn run(p_inc: f64, p_ch: f64, seed: u64, cap: u64) -> Status {
        let mut rng = Rng(seed);
        let mut exp = Experiment::new(DEFAULT_MARGIN, DEFAULT_ALPHA).expect("valid");
        for _ in 0..cap {
            let a = rng.bern(p_inc);
            let b = rng.bern(p_ch);
            exp.observe_outcome(Arm::Incumbent, a);
            exp.observe_outcome(Arm::Challenger, b);
            if exp.status() != Status::Undecided {
                break;
            }
        }
        exp.status()
    }

    fn rate(p_inc: f64, p_ch: f64, want: Status, reps: u64, cap: u64) -> f64 {
        let hits = (0..reps)
            .filter(|s| run(p_inc, p_ch, 0x9E37_79B9_7F4A_7C15 ^ (s + 1), cap) == want)
            .count();
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let r = hits as f64 / reps as f64;
        r
    }

    #[test]
    fn rejects_incoherent_parameters() {
        assert!(Experiment::new(0.0, 0.05).is_err());
        assert!(Experiment::new(0.6, 0.05).is_err());
        assert!(Experiment::new(f64::NAN, 0.05).is_err());
        assert!(Experiment::new(0.05, 0.0).is_err());
        assert!(Experiment::new(0.05, 0.5).is_err());
        assert!(Experiment::new(0.05, 0.05).is_ok());
    }

    #[test]
    fn out_of_range_outcomes_are_refused_not_clamped() {
        let mut exp = Experiment::new(0.05, 0.05).expect("valid");
        assert!(!exp.observe_outcome(Arm::Incumbent, 1.5));
        assert!(!exp.observe_outcome(Arm::Incumbent, -0.1));
        assert!(!exp.observe_outcome(Arm::Challenger, f64::NAN));
        assert_eq!(exp.report().incumbent.outcomes, 0);
        assert_eq!(exp.report().challenger.outcomes, 0);
    }

    #[test]
    fn pairs_only_form_across_arms_in_arrival_order() {
        let mut exp = Experiment::new(0.05, 0.05).expect("valid");
        for _ in 0..5 {
            exp.observe_outcome(Arm::Incumbent, 1.0);
        }
        assert_eq!(exp.report().pairs, 0, "one arm alone is no evidence");
        exp.observe_outcome(Arm::Challenger, 1.0);
        exp.observe_outcome(Arm::Challenger, 0.0);
        assert_eq!(exp.report().pairs, 2);
    }

    #[test]
    fn equal_arms_certify() {
        // Identical success rates: the challenger is plainly within a 5-point margin.
        assert!(rate(0.76, 0.76, Status::Certified, 100, 20_000) >= 0.95);
    }

    #[test]
    fn a_clearly_worse_challenger_halts_fast_and_never_certifies() {
        let mut halted_quickly = 0;
        for s in 0..100u64 {
            let mut rng = Rng(0xDEAD_BEEF ^ (s + 1));
            let mut exp = Experiment::new(DEFAULT_MARGIN, DEFAULT_ALPHA).expect("valid");
            for _ in 0..2_000 {
                exp.observe_outcome(Arm::Incumbent, rng.bern(0.77));
                exp.observe_outcome(Arm::Challenger, rng.bern(0.56));
                if exp.status() != Status::Undecided {
                    break;
                }
            }
            assert_ne!(exp.status(), Status::Certified, "seed {s}");
            if exp.status() == Status::Halted && exp.report().pairs <= 500 {
                halted_quickly += 1;
            }
        }
        assert!(
            halted_quickly >= 95,
            "halted quickly in {halted_quickly}/100"
        );
    }

    #[test]
    fn false_certification_at_the_null_boundary_is_at_most_alpha() {
        // Challenger exactly `margin` worse: certifying it is the error the test promises to cap.
        // 400 replays to 10k pairs; allow Monte-Carlo slack of ~2 standard errors above alpha.
        let r = rate(0.76, 0.71, Status::Certified, 400, 10_000);
        assert!(r <= DEFAULT_ALPHA + 0.022, "false-certification rate {r}");
    }

    #[test]
    fn false_halt_at_the_null_boundary_is_at_most_alpha() {
        let r = rate(0.76, 0.71, Status::Halted, 400, 10_000);
        assert!(r <= DEFAULT_ALPHA + 0.022, "false-halt rate {r}");
    }

    #[test]
    fn decisions_are_sticky() {
        let mut exp = Experiment::new(0.05, 0.05).expect("valid");
        while exp.status() == Status::Undecided {
            exp.observe_outcome(Arm::Incumbent, 1.0);
            exp.observe_outcome(Arm::Challenger, 0.0);
        }
        assert_eq!(exp.status(), Status::Halted);
        let at = exp.report().decided_at_pair;
        for _ in 0..1_000 {
            exp.observe_outcome(Arm::Incumbent, 0.0);
            exp.observe_outcome(Arm::Challenger, 1.0);
        }
        assert_eq!(exp.status(), Status::Halted);
        assert_eq!(exp.report().decided_at_pair, at);
        assert_eq!(exp.report().challenger.outcomes, exp.report().pairs);
    }

    #[test]
    fn cost_is_tallied_per_arm_and_bad_costs_ignored() {
        let mut exp = Experiment::new(0.05, 0.05).expect("valid");
        exp.observe_cost(Arm::Incumbent, 0.75);
        exp.observe_cost(Arm::Incumbent, 0.25);
        exp.observe_cost(Arm::Challenger, 0.1);
        exp.observe_cost(Arm::Challenger, -1.0);
        exp.observe_cost(Arm::Challenger, f64::INFINITY);
        let r = exp.report();
        assert_eq!(r.incumbent.cost_per_request(), Some(0.5));
        assert_eq!(r.challenger.requests, 1);
        assert_eq!(r.challenger.cost_per_request(), Some(0.1));
    }

    #[test]
    fn wealth_stays_nonnegative_at_the_extremes() {
        for &m in &[0.01, 0.05, 0.5] {
            for &sign in &[1.0, -1.0] {
                let mut mix = Mixture::new(m, sign);
                for &d in &[-1.0, 1.0, -1.0, 1.0] {
                    mix.step(d, m);
                }
                assert!(mix.wealth.iter().all(|w| *w >= 0.0), "m={m} sign={sign}");
            }
        }
    }

    #[test]
    fn the_report_round_trips() {
        let mut exp = Experiment::new(0.05, 0.05).expect("valid");
        exp.observe_outcome(Arm::Incumbent, 1.0);
        exp.observe_outcome(Arm::Challenger, 1.0);
        let r = exp.report();
        let back: Report =
            serde_json::from_str(&serde_json::to_string(&r).expect("ser")).expect("de");
        // serde_json parses floats to within an ulp, not bit-exactly; the decision fields are exact.
        assert_eq!(back.status, r.status);
        assert_eq!(back.pairs, r.pairs);
        assert_eq!(back.incumbent, r.incumbent);
        assert!((back.e_certify - r.e_certify).abs() < 1e-12);
        assert!((back.e_halt - r.e_halt).abs() < 1e-12);
    }
}
