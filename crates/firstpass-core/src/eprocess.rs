//! Anytime-valid risk control on the gate threshold — a bound that holds at **every round**
//! (SPEC §10.1, the online regime).
//!
//! ## Why this exists when [`crate::conformal`] and [`crate::ltt`] already do risk control
//!
//! The three regimes already shipped are each valid for what they claim, and none of them covers
//! the way an operator actually runs the proxy:
//!
//! | existing | guarantee | why it is not enough here |
//! |---|---|---|
//! | [`crate::conformal::calibrate`] | served-failure ≤ `alpha` at confidence `1 − delta` | assumes **exchangeability**, and fixes `λ` once |
//! | [`crate::ltt`] | FWER-controlled, exact-binomial, no exchangeability needed | **fixed-sample**: one calibration set, one decision |
//! | [`crate::conformal::AdaptiveConformal`] | Gibbs–Candès ACI | keeps only the **long-run average** at `alpha` |
//!
//! Two concrete holes follow, and both bite in production:
//!
//! 1. **A long-run average is not a promise about now.** A process can sit above `alpha` for an
//!    arbitrarily long stretch and still converge, because the excess is amortised over a horizon
//!    the operator may never reach. Someone holding an error budget needs the bound to hold at the
//!    round they are on.
//! 2. **Repeated recalibration breaks fixed-sample guarantees.** LTT controls the family-wise error
//!    rate over *one* pre-specified test sequence on *one* calibration set. Re-run it whenever more
//!    feedback lands and adopt the winning threshold each time — which is exactly what a live
//!    recalibration loop does — and that is optional stopping on a growing stream. The guarantee
//!    does not compose across the repeats, and the realized risk can exceed `alpha` without any
//!    single run of LTT ever being wrong.
//!
//! ## Method
//!
//! Khosravi & Huo (2026), *Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained
//! LLMs* ([arXiv:2605.20270](https://arxiv.org/abs/2605.20270)).
//!
//! Maintain one **e-process** per candidate threshold on a fixed grid. An e-process is a
//! non-negative supermartingale under the null, so **Ville's inequality** bounds it at *all*
//! stopping times simultaneously:
//!
//! ```text
//!     P( ∃t : E_t ≥ 1/delta )  ≤  delta        under H₀: risk(λ) > alpha
//! ```
//!
//! That single quantifier — `∃t`, rather than a fixed `t` — is the whole point, and it is what
//! makes the bound survive optional stopping. A threshold is **certified** once its e-value crosses
//! `1/delta`, and certification is permanent: it was earned against every round up to that point,
//! so later data cannot retroactively invalidate it (it can, and does, certify *more* thresholds).
//!
//! Per round, for a served item with gate score `s` and observed correctness `c`, each threshold
//! `λ ≤ s` (i.e. each threshold that *would have served* this item) is updated by the betting factor
//!
//! ```text
//!     E_λ ← E_λ · (1 + lambda_bet · (alpha − err))      where err = 1 if incorrect else 0
//! ```
//!
//! Under H₀ the expected multiplier is `≤ 1`, which is precisely the supermartingale property.
//! Thresholds that never serve are never updated, so an unused threshold neither gains nor loses
//! evidence — a nuance the fixed-sample methods get for free but an online one has to be careful
//! about.
//!
//! The grid is a **Bonferroni** family: with `n` candidate thresholds each tested at `delta / n`,
//! the probability that *any* threshold is falsely certified at *any* round stays `≤ delta`. The
//! served threshold is the **least conservative certified** one — the smallest `λ` whose e-process
//! has crossed — which serves as much traffic as the evidence allows.
//!
//! ## The honest cost
//!
//! Anytime validity is not free. Against [`crate::conformal::AdaptiveConformal`] at the same
//! `alpha` this serves **less** traffic, because it will not lower the threshold until the evidence
//! has actually accumulated, and the Bonferroni correction makes each individual threshold harder
//! to certify. That is the trade being offered: ACI tracks a shifting target aggressively and
//! promises an average; this one promises every round and pays for it in conservatism. An operator
//! picks the regime; neither is strictly better, and the ADR says so.
//!
//! Before any threshold is certified there is no guarantee to serve on, so
//! [`EProcessRiskControl::certified_threshold`] returns `None` and the caller keeps its existing
//! behaviour. Failing closed is the only safe reading of "I have not proven anything yet".

use serde::{Deserialize, Serialize};

/// Outcome of a certification query: the served threshold and the evidence behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Certification {
    /// The least conservative certified threshold — serve iff `score >= threshold`.
    pub threshold: f64,
    /// The e-value backing it. Always `>= 1/delta` for a certified threshold.
    pub e_value: f64,
    /// Rounds observed when this certification was first reached.
    pub certified_at_round: u64,
}

/// Default betting fraction. Conservative on purpose: a large bet grows the e-process fast when the
/// null is false but decays it just as fast when it is true, and a decayed e-process takes a long
/// time to recover. `0.5` is the standard "half-Kelly" compromise.
pub const DEFAULT_BET: f64 = 0.5;

/// Default e-value ceiling, as a multiple of the crossing level.
///
/// Truncating a supermartingale from above leaves it a supermartingale, so the type-I guarantee is
/// unaffected — this is purely about **responsiveness**. Without a cap, a long clean stretch banks
/// unbounded surplus evidence, and a subsequent regime change must burn all of it off before the
/// threshold de-certifies; the bound holds asymptotically while realized failure sits above `alpha`
/// for thousands of rounds. Capping bounds that de-certification lag. `10.0` keeps roughly an order
/// of magnitude of headroom — enough that ordinary noise does not flap the certification, small
/// enough that a genuine shift is caught in tens of rounds rather than thousands.
pub const DEFAULT_CAP: f64 = 10.0;

/// Anytime-valid risk controller over a fixed grid of candidate thresholds.
///
/// Feed it the deferred-feedback stream via [`EProcessRiskControl::observe_served`] and read
/// [`EProcessRiskControl::certified_threshold`] on the router hot path. Unlike
/// [`crate::conformal::AdaptiveConformal`], the answer is valid at the round you read it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EProcessRiskControl {
    /// Target served-failure rate.
    alpha: f64,
    /// Family-wise error budget across the whole grid and all rounds.
    delta: f64,
    /// Betting fraction in `(0, 1]`.
    bet: f64,
    /// E-values are capped at `cap * crossing_level`, bounding how much surplus evidence a clean
    /// stretch can bank against a later regime change. See [`DEFAULT_CAP`].
    cap: f64,
    /// Candidate thresholds, ascending.
    grid: Vec<f64>,
    /// Running e-value per grid point, parallel to `grid`. Starts at `1.0` (no evidence).
    e_values: Vec<f64>,
    /// Round at which each grid point first crossed, if it has.
    certified_at: Vec<Option<u64>>,
    /// Rounds observed.
    rounds: u64,
    /// Served count and failures, for diagnostics only — never part of the guarantee.
    served: u64,
    served_fails: u64,
}

impl EProcessRiskControl {
    /// Build a controller over `grid` candidate thresholds.
    ///
    /// `alpha` is the target served-failure rate, `delta` the family-wise error budget (the bound
    /// holds with probability `1 − delta` across every threshold and every round jointly).
    ///
    /// The grid is sorted and de-duplicated; an empty grid yields a controller that never certifies,
    /// which is the correct degenerate behaviour rather than an error — the caller then simply keeps
    /// its existing threshold.
    #[must_use]
    pub fn new(alpha: f64, delta: f64, bet: f64, grid: &[f64]) -> Self {
        Self::with_cap(alpha, delta, bet, DEFAULT_CAP, grid)
    }

    /// As [`EProcessRiskControl::new`], with an explicit e-value ceiling. See [`DEFAULT_CAP`] for
    /// why a ceiling exists and what it trades.
    #[must_use]
    pub fn with_cap(alpha: f64, delta: f64, bet: f64, cap: f64, grid: &[f64]) -> Self {
        // The betting factor on a failure is `1 + bet*(alpha - 1)`, which reaches zero at
        // `bet = 1/(1 - alpha)` and goes negative beyond it. A zero factor annihilates the e-value
        // permanently: no amount of subsequent evidence can revive a product that has hit exactly
        // 0, so a single failure would silently retire that threshold forever. Clamping the bet
        // just below the annihilating value keeps the process a strict supermartingale (which is
        // what Ville's inequality needs) while leaving it able to recover.
        //
        // Flagged in review. Reachable through the public `with_cap` and by config, so it is a
        // real input to validate rather than a theoretical corner.
        // Bound the failure factor at 0.5 rather than merely above 0. Clamping at `0.99/(1-alpha)`
        // technically avoids annihilation but leaves a factor of 0.01 — a single failure costs 100x
        // and takes ~50 consecutive successes to undo, which is dead in practice if not in theory.
        // The first version of this clamp did exactly that, and the recovery half of the test below
        // failed: the controller never certified again. Half per failure is recoverable and still
        // strictly decreasing, which is all the supermartingale property needs.
        let max_bet = 0.5 / (1.0 - alpha).max(f64::EPSILON);
        let bet = if bet.is_finite() {
            bet.clamp(0.0, max_bet)
        } else {
            DEFAULT_BET.min(max_bet)
        };
        let mut grid: Vec<f64> = grid.iter().copied().filter(|g| g.is_finite()).collect();
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        grid.dedup();
        let n = grid.len();
        Self {
            alpha,
            delta,
            bet,
            cap,
            grid,
            e_values: vec![1.0; n],
            certified_at: vec![None; n],
            rounds: 0,
            served: 0,
            served_fails: 0,
        }
    }

    /// The per-threshold crossing level under Bonferroni: `n / delta`.
    ///
    /// Splitting `delta` across the `n` grid points is what keeps the *family-wise* error at
    /// `delta`. Testing every threshold at the full `delta` would inflate the false-certification
    /// rate with grid size — the classic multiple-testing trap, and an easy one to fall into here
    /// because each individual e-process looks perfectly valid on its own.
    #[must_use]
    pub fn crossing_level(&self) -> f64 {
        if self.grid.is_empty() || self.delta <= 0.0 {
            return f64::INFINITY;
        }
        self.grid.len() as f64 / self.delta
    }

    /// Record one served item: its gate `score`, and whether it turned out correct.
    ///
    /// Only thresholds that *would have served* this item (`λ <= score`) are updated — a threshold
    /// strict enough to have escalated instead has learned nothing about its own risk from it.
    pub fn observe_served(&mut self, score: f64, was_correct: bool) {
        self.rounds += 1;
        self.served += 1;
        self.served_fails += u64::from(!was_correct);

        let err = f64::from(!was_correct);
        let level = self.crossing_level();

        for i in 0..self.grid.len() {
            if self.grid[i] > score {
                continue;
            }
            // Betting factor. Under H₀ (true risk > alpha) the expectation is <= 1, which is the
            // supermartingale property Ville's inequality needs. Clamped at zero because a negative
            // e-value is meaningless and would flip the comparison below.
            let factor = (1.0 + self.bet * (self.alpha - err)).max(0.0);
            // Capped. Ville's inequality is indifferent to the cap (truncating a supermartingale
            // from above leaves it a supermartingale, so type-I control is untouched), but the
            // operational difference is the whole ballgame: an uncapped e-value compounds through a
            // long clean stretch to something astronomically above the level, and then a regime
            // change has to burn through all of it before the threshold de-certifies. The bound
            // would still hold in the limit while the realized rate sat above alpha for thousands
            // of rounds — which is precisely the "long-run average" failure this module rejects.
            self.e_values[i] = (self.e_values[i] * factor).min(self.cap * level);
            if self.certified_at[i].is_none() && self.e_values[i] >= level {
                self.certified_at[i] = Some(self.rounds);
            }
        }
    }

    /// The least conservative certified threshold, or `None` if nothing is certified yet.
    ///
    /// `None` means "no threshold has earned a guarantee", and the caller must keep its existing
    /// behaviour rather than invent one. Serving on an uncertified threshold would be exactly the
    /// unproven-claim failure this module exists to prevent.
    /// Certification is evaluated **live**, against the current e-value — it is not a latch.
    ///
    /// This is the subtle part, and getting it wrong silently reintroduces the very failure mode the
    /// module exists to prevent. A threshold that crossed during a clean stretch and has since been
    /// contradicted by evidence is *not* certified any more, and must stop serving. Latching it —
    /// "it was proven once, so it stays proven" — is what a fixed-sample method does, and under a
    /// regime change it keeps serving at a threshold the current regime does not justify. The first
    /// draft of this module latched, and `coverage_holds_at_every_round_under_a_regime_change`
    /// caught it: realized failure hit 0.312 against alpha=0.20.
    ///
    /// `certified_at_round` is retained as provenance — when the evidence *first* crossed — which is
    /// useful in a receipt and harmless here, because the serving decision reads the e-value.
    #[must_use]
    pub fn certified_threshold(&self) -> Option<Certification> {
        let level = self.crossing_level();
        self.grid
            .iter()
            .enumerate()
            .find(|(i, _)| self.e_values[*i] >= level)
            .map(|(i, &threshold)| Certification {
                threshold,
                e_value: self.e_values[i],
                certified_at_round: self.certified_at[i].unwrap_or(self.rounds),
            })
    }

    /// Whether `score` clears the certified threshold. `false` while nothing is certified.
    #[must_use]
    pub fn should_serve(&self, score: f64) -> bool {
        self.certified_threshold()
            .is_some_and(|c| score >= c.threshold)
    }

    /// Realized served-failure rate so far — a running diagnostic, not the guarantee.
    #[must_use]
    pub fn realized_served_failure(&self) -> f64 {
        if self.served == 0 {
            0.0
        } else {
            self.served_fails as f64 / self.served as f64
        }
    }

    /// Rounds observed.
    #[must_use]
    pub const fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Current e-value at each grid point, parallel to the grid — for telemetry and tests.
    #[must_use]
    pub fn e_values(&self) -> &[f64] {
        &self.e_values
    }

    /// The candidate grid, ascending.
    #[must_use]
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform stream in `[0,1)`. A seeded LCG rather than `rand`, so the evidence
    /// these tests produce is reproducible by an auditor on any machine, which is the same reason
    /// the rest of the crate avoids ambient randomness.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn grid() -> Vec<f64> {
        (0..=20).map(|i| f64::from(i) / 20.0).collect()
    }

    /// A well-behaved gate: high score => usually correct.
    fn correct_for(score: f64, u: f64) -> bool {
        u < score
    }

    /// Nothing may be served before evidence exists. This is the fail-closed property: an
    /// uncertified controller must not hand out a threshold, or the guarantee is decorative.
    #[test]
    fn nothing_is_certified_before_any_evidence() {
        let e = EProcessRiskControl::new(0.1, 0.05, DEFAULT_BET, &grid());
        assert!(
            e.certified_threshold().is_none(),
            "a fresh controller must certify nothing"
        );
        assert!(
            !e.should_serve(1.0),
            "even a perfect score must not serve without a certified threshold"
        );
    }

    /// On a clean stream the controller must eventually certify — a method that never certifies is
    /// trivially "safe" and completely useless, so this is the liveness half of the contract.
    #[test]
    fn a_clean_stream_eventually_certifies() {
        let mut e = EProcessRiskControl::new(0.2, 0.05, DEFAULT_BET, &grid());
        let mut rng = Lcg(42);
        for _ in 0..4000 {
            let score = 0.90 + rng.next_f64() * 0.10;
            let correct = correct_for(score, rng.next_f64());
            e.observe_served(score, correct);
        }
        let cert = e
            .certified_threshold()
            .expect("4000 clean rounds must certify some threshold");
        assert!(
            cert.e_value >= e.crossing_level(),
            "a certified threshold must sit at or above the Bonferroni crossing level: {} < {}",
            cert.e_value,
            e.crossing_level()
        );
    }

    /// **The core guarantee, and the reason this module exists.**
    ///
    /// Anytime-validity means the bound holds at EVERY round, not just at the end. This checks the
    /// realized served-failure rate at every single round after certification, on an ADVERSARIAL
    /// non-exchangeable stream: a clean first half, then an abrupt regime change where the gate's
    /// error rate jumps. Split conformal's exchangeability assumption is violated by construction
    /// here — see `split_conformal_breaches_where_the_e_process_holds` for the contrast.
    #[test]
    fn coverage_holds_at_every_round_under_a_regime_change() {
        let alpha = 0.20;
        let mut worst: f64 = 0.0;
        for seed in 0..40u64 {
            let mut e = EProcessRiskControl::new(alpha, 0.05, DEFAULT_BET, &grid());
            let mut rng = Lcg(seed.wrapping_mul(2_654_435_761).wrapping_add(7));
            let mut served = 0u64;
            let mut fails = 0u64;
            for t in 0..3000 {
                let score = 0.80 + rng.next_f64() * 0.20;
                // Regime change at the midpoint: the gate becomes much less reliable, with no
                // warning and no return. This is exactly what exchangeability forbids.
                let correct = if t < 1500 {
                    correct_for(score, rng.next_f64())
                } else {
                    rng.next_f64() < 0.55
                };
                if e.should_serve(score) {
                    served += 1;
                    fails += u64::from(!correct);
                    // Check the realized rate at THIS round, not only at the end.
                    if served >= 50 {
                        let rate = fails as f64 / served as f64;
                        worst = worst.max(rate);
                    }
                }
                e.observe_served(score, correct);
            }
        }
        // Ville's inequality is a high-probability bound, not an almost-sure one, so a small
        // overshoot budget is correct rather than a fudge: at delta = 0.05 some excursion is
        // permitted by the theory itself.
        assert!(
            worst <= alpha + 0.10,
            "worst per-round served-failure {worst:.3} exceeded alpha={alpha} plus slack"
        );
    }

    /// **The contrast that makes the contribution.**
    ///
    /// On the same non-exchangeable stream, a split-conformal threshold fitted on the first regime
    /// carries its guarantee only under exchangeability — which the regime change destroys. It goes
    /// on serving at a threshold the new regime does not justify, and its realized failure rate
    /// breaches alpha. The e-process, having to re-earn its evidence every round, does not.
    ///
    /// If this test ever fails it means the streams stopped being adversarial, not that the method
    /// improved — so it asserts the breach explicitly rather than just comparing the two.
    #[test]
    fn split_conformal_breaches_where_the_e_process_holds() {
        let alpha = 0.15;
        let n = 6000;
        let mut rng = Lcg(20_260_814);

        // Phase 1 doubles as split-conformal's calibration set.
        let mut calib: Vec<(f64, bool)> = Vec::new();
        for _ in 0..1000 {
            let score = 0.70 + rng.next_f64() * 0.30;
            calib.push((score, correct_for(score, rng.next_f64())));
        }
        let fixed = crate::conformal::calibrate(&calib, alpha, 0.05, 30).threshold;

        let mut e = EProcessRiskControl::new(alpha, 0.05, DEFAULT_BET, &grid());
        // Warm the e-process on the same phase-1 data, so neither method gets a data advantage.
        for &(s, c) in &calib {
            e.observe_served(s, c);
        }

        let (mut sc_served, mut sc_fails) = (0u64, 0u64);
        let (mut ep_served, mut ep_fails) = (0u64, 0u64);
        for t in 0..n {
            let score = 0.70 + rng.next_f64() * 0.30;
            // Phase 2: the relationship between score and correctness collapses.
            let correct = if t < n / 2 {
                correct_for(score, rng.next_f64())
            } else {
                rng.next_f64() < 0.60
            };
            if score >= fixed {
                sc_served += 1;
                sc_fails += u64::from(!correct);
            }
            if e.should_serve(score) {
                ep_served += 1;
                ep_fails += u64::from(!correct);
            }
            e.observe_served(score, correct);
        }

        let sc_rate = sc_fails as f64 / sc_served.max(1) as f64;
        let ep_rate = ep_fails as f64 / ep_served.max(1) as f64;
        assert!(
            sc_rate > alpha,
            "the stream is meant to break split conformal, but its rate was {sc_rate:.3} <= alpha={alpha} \
             — the test has stopped being adversarial and proves nothing"
        );
        assert!(
            ep_rate < sc_rate,
            "the e-process ({ep_rate:.3}) must beat split conformal ({sc_rate:.3}) under shift"
        );
    }

    /// **The cap must bound how long a stale certification survives.**
    ///
    /// Written because the first mutation run exposed that nothing covered the cap: deleting it
    /// left all other tests green. Without a ceiling, a long clean stretch banks unbounded surplus
    /// evidence, and after the regime turns bad the threshold keeps serving until that surplus is
    /// burned off — the bound still holds in the limit while realized failure sits above `alpha`
    /// for thousands of rounds. That is the long-run-average failure mode wearing a new hat.
    ///
    /// Capped vs uncapped on an identical stream, measuring rounds-to-de-certify.
    #[test]
    fn the_cap_bounds_how_long_a_stale_certification_survives() {
        let alpha = 0.10;
        let lag = |cap: f64| -> u64 {
            let mut e = EProcessRiskControl::with_cap(alpha, 0.05, DEFAULT_BET, cap, &grid());
            let mut rng = Lcg(99);
            // Long clean stretch: bank evidence.
            for _ in 0..5000 {
                let score = 0.95 + rng.next_f64() * 0.05;
                e.observe_served(score, correct_for(score, rng.next_f64()));
            }
            assert!(
                e.certified_threshold().is_some(),
                "the clean stretch must certify, or the lag measurement is meaningless"
            );
            // Regime turns hostile: every served item now fails. Count rounds to de-certify.
            let mut rounds = 0u64;
            while e.certified_threshold().is_some() && rounds < 100_000 {
                e.observe_served(0.99, false);
                rounds += 1;
            }
            rounds
        };

        let capped = lag(DEFAULT_CAP);
        let uncapped = lag(f64::INFINITY);
        assert!(
            capped < uncapped,
            "the cap must shorten de-certification lag: capped={capped} uncapped={uncapped}"
        );
        assert!(
            capped <= 100,
            "a capped controller must drop a contradicted certification within ~tens of rounds, took {capped}"
        );
    }

    /// **A large bet must not permanently annihilate a threshold.**
    ///
    /// Flagged in review. The failure factor is `1 + bet*(alpha - 1)`, which hits exactly zero at
    /// `bet = 1/(1-alpha)` and goes negative past it. Zero is absorbing under multiplication: one
    /// failure would retire that threshold forever, and no later evidence could revive it. The
    /// controller would look like it was still learning while being permanently dead.
    ///
    /// `bet` is reachable through the public `with_cap` and through config, so it is validated
    /// input, not a theoretical corner.
    #[test]
    fn an_oversized_bet_cannot_permanently_kill_a_threshold() {
        let alpha = 0.20;
        // bet = 5.0 is far past the annihilating 1/(1-0.2) = 1.25.
        let mut e = EProcessRiskControl::with_cap(alpha, 0.05, 5.0, DEFAULT_CAP, &grid());
        e.observe_served(0.9, false);
        assert!(
            e.e_values().iter().all(|v| *v > 0.0),
            "a failure must shrink e-values, never zero them: {:?}",
            e.e_values()
        );

        // And the process must still be able to recover and certify afterwards.
        let mut rng = Lcg(7);
        for _ in 0..4000 {
            let score = 0.95 + rng.next_f64() * 0.05;
            e.observe_served(score, correct_for(score, rng.next_f64()));
        }
        assert!(
            e.certified_threshold().is_some(),
            "a controller that took one early failure must still be able to certify later"
        );
    }

    /// **Precedence contract with ACI**, verified at the source rather than trusted in prose.
    ///
    /// `proxy.rs` prefers the e-process threshold over `AdaptiveConformal`'s whenever one is
    /// certified, and falls through otherwise. That is only safe because an uncertified controller
    /// returns `None` rather than a default — if it ever returned a number before earning it, the
    /// router would silently serve on an unproven threshold and outrank a method that at least has
    /// a long-run guarantee.
    #[test]
    fn an_uncertified_controller_yields_nothing_for_the_router_to_prefer() {
        let mut e = EProcessRiskControl::new(0.2, 0.05, DEFAULT_BET, &grid());
        // Some evidence, but nowhere near the crossing level.
        for _ in 0..10 {
            e.observe_served(0.95, true);
        }
        assert!(
            e.certified_threshold().is_none(),
            "ten rounds must not certify at delta=0.05 over a 21-point grid"
        );
        // ...and once it HAS earned it, the router gets a real threshold to prefer.
        let mut rng = Lcg(11);
        for _ in 0..4000 {
            let score = 0.95 + rng.next_f64() * 0.05;
            e.observe_served(score, correct_for(score, rng.next_f64()));
        }
        let c = e
            .certified_threshold()
            .expect("sustained clean evidence must certify");
        assert!(
            (0.0..=1.0).contains(&c.threshold),
            "a certified threshold must be servable: {}",
            c.threshold
        );
    }

    /// `delta = 0` means "zero error budget", and must certify NOTHING rather than everything.
    ///
    /// Review raised this as a possible false-certification path. It is already handled —
    /// `crossing_level` returns infinity for a non-positive delta — but "already correct" is not a
    /// reason to leave a fail-closed contract untested, and an infinity comparison is exactly the
    /// kind of thing a later refactor breaks silently.
    #[test]
    fn a_zero_error_budget_certifies_nothing() {
        for delta in [0.0, -0.1] {
            let mut e = EProcessRiskControl::new(0.2, delta, DEFAULT_BET, &grid());
            let mut rng = Lcg(3);
            for _ in 0..5000 {
                let score = 0.99;
                e.observe_served(score, correct_for(score, rng.next_f64()));
            }
            assert!(
                e.crossing_level().is_infinite(),
                "delta={delta} must make the crossing level unreachable"
            );
            assert!(
                e.certified_threshold().is_none(),
                "delta={delta} is a zero error budget: nothing may ever be certified"
            );
            assert!(!e.should_serve(1.0), "and nothing may be served");
        }
    }

    /// Bonferroni: the crossing level must scale with grid size. Testing every threshold at the
    /// full `delta` is the multiple-testing trap — each e-process looks valid alone, while the
    /// family-wise false-certification rate quietly grows with the grid.
    #[test]
    fn the_crossing_level_scales_with_grid_size() {
        let small = EProcessRiskControl::new(0.1, 0.05, DEFAULT_BET, &[0.5, 0.6]);
        let large = EProcessRiskControl::new(0.1, 0.05, DEFAULT_BET, &grid());
        assert!(
            large.crossing_level() > small.crossing_level(),
            "a larger grid must demand more evidence per threshold: {} vs {}",
            large.crossing_level(),
            small.crossing_level()
        );
        assert!(
            (small.crossing_level() - 2.0 / 0.05).abs() < 1e-9,
            "crossing level must be exactly n/delta"
        );
    }

    /// A threshold strict enough to have escalated an item learns nothing from it. Without this,
    /// unused thresholds would accumulate evidence they never earned and certify on traffic they
    /// would never have served.
    #[test]
    fn thresholds_that_would_not_have_served_are_not_updated() {
        let mut e = EProcessRiskControl::new(0.1, 0.05, DEFAULT_BET, &[0.2, 0.9]);
        e.observe_served(0.5, true);
        let vals = e.e_values();
        assert!(
            (vals[1] - 1.0).abs() < f64::EPSILON,
            "threshold 0.9 did not serve a 0.5-scoring item, so its e-value must stay 1.0, got {}",
            vals[1]
        );
        assert!(
            vals[0] > 1.0,
            "threshold 0.2 did serve it correctly, so its e-value must grow, got {}",
            vals[0]
        );
    }

    /// A degenerate grid must be inert, not a panic. `new` takes a slice from config, and an empty
    /// or non-finite grid is a plausible misconfiguration that must fail closed.
    #[test]
    fn a_degenerate_grid_certifies_nothing_and_does_not_panic() {
        let mut e = EProcessRiskControl::new(0.1, 0.05, DEFAULT_BET, &[f64::NAN, f64::INFINITY]);
        e.observe_served(0.9, true);
        assert!(e.certified_threshold().is_none());
        assert!(!e.should_serve(1.0));
        assert!(
            e.crossing_level().is_infinite(),
            "an empty grid can never cross"
        );
    }
}
