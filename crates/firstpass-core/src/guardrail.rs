//! The guardrail: watching the thing that was promised (ADR 0009 D3).
//!
//! Firstpass publishes a distribution-free bound on served failures, calibrates it once, and then
//! trusts it indefinitely. `/metrics` exposes counters and gate error budgets trip individual
//! gates, but nothing watched the *served-failure rate itself* — the quantity the headline
//! guarantee is actually about — or reacted when it degraded.
//!
//! This does. It computes the same Hoeffding upper confidence bound used to earn the published
//! guarantee (see [`crate::conformal`]) over a trailing window of resolved outcomes, and when the
//! bound exceeds the target it demotes the route to observe.
//!
//! Three properties matter more than the feature:
//!
//!   * **It acts on resolved outcomes, not gate verdicts.** A gate verdict is Firstpass's own
//!     opinion. The guarantee is about real downstream results, which arrive via `/v1/feedback`.
//!     Tripping on self-reported verdicts would let the system grade its own homework.
//!   * **A minimum sample size is mandatory.** Without it a route trips on the first two failures
//!     of the day. A bound over 3 samples is noise, and a guardrail that fires on noise is a
//!     guardrail an operator turns off.
//!   * **Demotion is the safe direction.** The failure mode is "you stop saving money", never
//!     "you serve worse answers".

use serde::Deserialize;

/// What a breached guardrail does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuardrailAction {
    /// Revert the route to observe. Traffic is served exactly as it would be without Firstpass.
    #[default]
    Demote,
    /// Emit the event and change nothing — for operators who want a human in the loop.
    Alarm,
}

/// Guardrail settings.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Guardrail {
    /// The served-failure target being defended, in `(0, 1)`.
    pub alpha: f64,
    /// Confidence parameter for the bound, in `(0, 1)`. Matches the `delta` used to earn the
    /// published guarantee, so the guardrail defends the same number that was advertised.
    #[serde(default = "default_delta")]
    pub delta: f64,
    /// Trailing window of resolved outcomes to judge over.
    pub window: usize,
    /// Never act on fewer resolved samples than this.
    pub min_n: usize,
    /// What to do on breach.
    #[serde(default)]
    pub action: GuardrailAction,
}

const fn default_delta() -> f64 {
    0.05
}

impl Guardrail {
    /// Whether this configuration is coherent.
    ///
    /// # Errors
    /// If `alpha`/`delta` are outside `(0, 1)`, the window is zero, or `min_n` exceeds the window
    /// (which would make the guardrail unable to ever act — a silently inert guard is worse than
    /// none, because an operator believes they are protected).
    pub fn validate(&self) -> Result<(), String> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(format!(
                "guardrail.alpha must be within (0, 1), got {}",
                self.alpha
            ));
        }
        if !self.delta.is_finite() || self.delta <= 0.0 || self.delta >= 1.0 {
            return Err(format!(
                "guardrail.delta must be within (0, 1), got {}",
                self.delta
            ));
        }
        if self.window == 0 {
            return Err("guardrail.window must be > 0".to_owned());
        }
        // The slack term alone must leave room under alpha, or a ZERO-failure window breaches
        // and every healthy route is demoted the moment it reaches min_n. Caught by a test that
        // fed 500 perfect outcomes and watched the route get demoted anyway.
        //
        //   bound(rate=0) = sqrt(ln(1/delta) / 2n) < alpha   =>   n > ln(1/delta) / (2*alpha^2)
        let floor = (1.0 / self.delta).ln() / (2.0 * self.alpha * self.alpha);
        #[expect(
            clippy::cast_precision_loss,
            reason = "min_n is operational scale; f64 is exact far beyond any realistic window"
        )]
        let min_n_f = self.min_n as f64;
        if min_n_f <= floor {
            let slack = ((1.0 / self.delta).ln() / (2.0 * min_n_f.max(1.0))).sqrt();
            return Err(format!(
                "guardrail.min_n ({}) is too small for alpha={} delta={}: with that few samples \
                 the confidence slack alone is {slack:.3}, which already exceeds alpha — so even \
                 a window with zero failures would breach and every healthy route would be \
                 demoted. Use min_n > {:.0}.",
                self.min_n,
                self.alpha,
                self.delta,
                floor.ceil()
            ));
        }
        if self.min_n > self.window {
            return Err(format!(
                "guardrail.min_n ({}) exceeds guardrail.window ({}) — the guardrail could never \
                 act, which is worse than having none because it looks like protection",
                self.min_n, self.window
            ));
        }
        Ok(())
    }
}

/// The verdict of one guardrail evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardrailVerdict {
    /// Resolved samples considered.
    pub n: usize,
    /// Observed failure rate over the window.
    pub rate: f64,
    /// Hoeffding upper confidence bound on that rate.
    pub bound: f64,
    /// Whether the bound exceeds the target with enough samples to say so.
    pub breached: bool,
    /// Set when the guardrail declined to judge, and why.
    pub insufficient_samples: bool,
}

/// Evaluate a trailing window of resolved outcomes. `outcomes` is oldest-first; `true` means the
/// served answer was correct.
///
/// Uses the same bound as [`crate::conformal::calibrate`] — `rate + sqrt(ln(1/delta) / 2n)` — so
/// the guardrail defends exactly the quantity the published guarantee describes. A guardrail
/// computing a *different* bound than the advertised one would be defending a number nobody was
/// promised.
#[must_use]
pub fn evaluate(g: &Guardrail, outcomes: &[bool]) -> GuardrailVerdict {
    let window = outcomes.len().min(g.window);
    let recent = &outcomes[outcomes.len() - window..];
    let n = recent.len();
    if n < g.min_n {
        return GuardrailVerdict {
            n,
            rate: 0.0,
            bound: 0.0,
            breached: false,
            insufficient_samples: true,
        };
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "window sizes are operational scale; f64 is exact well past any realistic window"
    )]
    let n_f = n as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "failure counts are bounded by the window"
    )]
    let fails = recent.iter().filter(|ok| !**ok).count() as f64;
    let rate = fails / n_f;
    let slack = (1.0 / g.delta).ln().max(0.0);
    let bound = rate + (slack / (2.0 * n_f)).sqrt();
    GuardrailVerdict {
        n,
        rate,
        bound,
        breached: bound > g.alpha,
        insufficient_samples: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(alpha: f64, window: usize, min_n: usize) -> Guardrail {
        Guardrail {
            alpha,
            delta: 0.05,
            window,
            min_n,
            action: GuardrailAction::Demote,
        }
    }

    /// Healthy traffic must not trip. A guardrail that fires on a good route gets disabled, and a
    /// disabled guardrail protects nothing.
    #[test]
    fn clean_traffic_does_not_breach() {
        let outcomes = vec![true; 500];
        let v = evaluate(&g(0.10, 500, 200), &outcomes);
        assert!(!v.breached, "perfect traffic breached: bound {}", v.bound);
        assert!((v.rate - 0.0).abs() < f64::EPSILON);
    }

    /// Genuinely bad traffic must trip — the whole point.
    #[test]
    fn sustained_failure_breaches() {
        // 30% failures against a 10% target.
        let outcomes: Vec<bool> = (0..500).map(|i| i % 10 >= 3).collect();
        let v = evaluate(&g(0.10, 500, 200), &outcomes);
        assert!(v.breached, "30% failure rate did not breach a 10% target");
        assert!(v.bound > 0.10);
    }

    /// The minimum-sample rule is what separates a guardrail from a noise generator: two early
    /// failures must not demote a route.
    #[test]
    fn a_couple_of_early_failures_cannot_trip_it() {
        let outcomes = vec![false, false, true];
        let v = evaluate(&g(0.10, 500, 200), &outcomes);
        assert!(v.insufficient_samples, "judged on 3 samples");
        assert!(
            !v.breached,
            "tripped on 3 samples — 2 bad calls would demote a route"
        );
    }

    /// Only the trailing window counts: a route that failed badly last week but is healthy now
    /// must not stay demoted on ancient history.
    #[test]
    fn only_the_trailing_window_is_judged() {
        let mut outcomes = vec![false; 400]; // ancient failures
        outcomes.extend(vec![true; 200]); // recent health
        let v = evaluate(&g(0.10, 200, 150), &outcomes);
        assert_eq!(v.n, 200);
        assert!(
            !v.breached,
            "old failures outside the window still tripped it"
        );
    }

    /// A rate sitting exactly at the target must still breach once the confidence slack is added —
    /// otherwise the guardrail lets the bound sit above the promised number indefinitely.
    #[test]
    fn the_bound_not_the_raw_rate_is_what_is_judged() {
        // Exactly 10% failures against a 10% target: the raw rate ties, the bound exceeds.
        let outcomes: Vec<bool> = (0..500).map(|i| i % 10 != 0).collect();
        let v = evaluate(&g(0.10, 500, 200), &outcomes);
        assert!((v.rate - 0.10).abs() < 0.01, "rate was {}", v.rate);
        assert!(
            v.breached,
            "raw rate at target did not breach; bound {} must exceed alpha",
            v.bound
        );
    }

    /// More samples must tighten the bound, or the guardrail would grow more trigger-happy with
    /// evidence rather than less.
    #[test]
    fn the_bound_tightens_as_samples_accumulate() {
        let small: Vec<bool> = (0..200).map(|i| i % 10 != 0).collect();
        let large: Vec<bool> = (0..5000).map(|i| i % 10 != 0).collect();
        let vs = evaluate(&g(0.10, 200, 200), &small);
        let vl = evaluate(&g(0.10, 5000, 200), &large);
        assert!(
            vl.bound < vs.bound,
            "bound did not tighten with evidence: {} samples -> {}, {} samples -> {}",
            vs.n,
            vs.bound,
            vl.n,
            vl.bound
        );
    }

    #[test]
    fn validate_rejects_incoherent_settings() {
        assert!(g(0.0, 100, 10).validate().is_err());
        assert!(g(1.0, 100, 10).validate().is_err());
        assert!(g(0.1, 0, 0).validate().is_err());
        // min_n > window: could never act, but looks like protection.
        assert!(g(0.1, 100, 500).validate().is_err());
        // min_n too small for the bound to ever clear alpha
        assert!(g(0.10, 500, 100).validate().is_err());
        assert!(g(0.1, 500, 400).validate().is_ok());
    }
}
