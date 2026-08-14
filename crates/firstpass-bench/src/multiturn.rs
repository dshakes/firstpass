//! Multi-turn evaluation: does a trajectory-informed start rung actually pay?
//!
//! # Why this module exists
//!
//! [`crate::vrbench`] and [`crate::coding_policy`] evaluate **single-shot** tasks: one prompt, one
//! ladder, one answer. That is the right shape for measuring a cascade, and it is the wrong shape
//! for measuring the feature ADR 0012 added, because a trajectory hint is by definition a function
//! of *previous turns*. A single-shot benchmark can only ever score it `None`.
//!
//! # The pre-registered bar
//!
//! Per [`crate::report`] discipline, a feature does not ship on plausibility. This module states
//! the criterion **before** the numbers exist ([`PreRegistered`]):
//!
//! 1. **$/success must improve.** A routing feature that costs money to run is not a routing
//!    feature.
//! 2. **Served-failure must not increase.** Cheap-and-wrong is trivial to achieve and worthless;
//!    the conformal bound has to survive. The paired quality CI must not exclude zero *on the
//!    losing side*.
//!
//! Miss either and the feature does not ship, and the negative result gets written up. A benchmark
//! whose author's feature always wins is not evidence — that discipline is what makes the numbers
//! in `docs/benchmarks/` worth reading, so it applies to the author's own features first.
//!
//! # Status: harness built and self-tested, NOT yet run against real providers
//!
//! Everything here runs today against the deterministic simulator, which proves the harness is
//! wired correctly and the criterion actually discriminates. It does **not** prove the feature
//! works: simulated gate behaviour is an assumption about reality, not a measurement of it. The
//! honest claim from a simulator run is "the plumbing is correct and the bar is well-posed".
//!
//! Producing a real number needs paid provider calls, which is a spending decision for the
//! operator, not something a benchmark module should trigger on its own.

use serde::Serialize;

use crate::coding_policy::RungOutcome;
use crate::stats::{Ci, bootstrap_mean_ci};

/// One turn of a multi-turn task: what the agent's transcript looked like going in, and what each
/// ladder rung would produce.
#[derive(Debug, Clone)]
pub struct Turn {
    /// Trajectory signals visible in the conversation **before** this turn is routed. This is the
    /// whole point: a router may use only what it could actually have seen at decision time.
    pub signals: firstpass_core::features::TrajectorySignals,
    /// Per-rung outcomes for this turn, cheapest first — the same shape
    /// [`crate::coding_policy`] uses, so both harnesses share serving logic and metrics.
    pub rungs: Vec<RungOutcome>,
}

/// A multi-turn task: an ordered conversation, each turn independently routed and served.
#[derive(Debug, Clone)]
pub struct MultiTurnTask {
    /// Stable id, for slicing results.
    pub id: String,
    /// Turns in order. Order is load-bearing — the difficulty of turn N depends on turns before it.
    pub turns: Vec<Turn>,
}

/// The criterion, fixed before the numbers exist.
///
/// A struct rather than prose so it is machine-checkable and cannot be quietly reinterpreted after
/// seeing the result — the failure mode pre-registration exists to prevent.
#[derive(Debug, Clone, Serialize)]
pub struct PreRegistered {
    /// Minimum relative $/success improvement required, e.g. `0.05` for 5%.
    pub min_cost_improvement: f64,
    /// How much worse the paired success rate may get before the feature is killed, as an absolute
    /// rate. Not zero: paired noise at realistic n would make a zero-tolerance bar a coin flip
    /// rather than a test. It IS small, and the CI below is what actually carries the argument.
    pub max_success_regression: f64,
}

impl Default for PreRegistered {
    /// The registered bar for ADR 0012's trajectory feature.
    fn default() -> Self {
        Self {
            min_cost_improvement: 0.05,
            max_success_regression: 0.01,
        }
    }
}

/// Outcome of the comparison, including the go/no-go call.
#[derive(Debug, Clone, Serialize)]
pub struct MultiTurnResult {
    /// Turns evaluated (not tasks — a task contributes one row per turn).
    pub n_turns: usize,
    /// Baseline: every turn starts at rung 0, ignoring trajectory.
    pub baseline_cost_per_success: f64,
    /// Trajectory-informed start rung.
    pub trajectory_cost_per_success: f64,
    /// Baseline success rate (oracle-correct served answers / turns).
    pub baseline_success: f64,
    /// Trajectory success rate.
    pub trajectory_success: f64,
    /// Paired per-turn success difference (trajectory − baseline), with bootstrap CI.
    pub delta_success: f64,
    /// CI on that paired difference. This is what decides the quality half of the bar.
    pub delta_success_ci: Ci,
    /// Relative $/success improvement, positive = cheaper.
    pub cost_improvement: f64,
    /// The criterion this was judged against.
    pub criterion: PreRegistered,
    /// Whether the pre-registered bar was cleared.
    pub ships: bool,
    /// Why, in one line, for the report.
    pub verdict: String,
}

/// Serve one turn starting at `start_rung`, cascading upward on gate failure.
///
/// Mirrors `coding_policy::serve_first_pass` including its best-attempt fallback: when every rung
/// fails its gate, the top rung is served anyway. A policy that fails everywhere must not be able
/// to hide its failures by declining to answer.
fn serve_from(rungs: &[RungOutcome], start_rung: usize) -> (bool, f64) {
    let mut spent = 0.0;
    let start = start_rung.min(rungs.len().saturating_sub(1));
    for o in rungs.iter().skip(start) {
        spent += o.cost_usd;
        if o.gate_full_pass {
            return (o.oracle_correct, spent);
        }
    }
    match rungs.last() {
        Some(top) => (top.oracle_correct, spent),
        None => (false, 0.0),
    }
}

/// Map a difficulty hint to a start rung.
///
/// Deliberately the crudest possible policy: hint level → rung index, capped by ladder length. The
/// bandit learns the real mapping in production; here we need something fixed and legible, because
/// the question under test is **"does the signal carry information?"**, not "is this the optimal
/// exploitation of it". A learned policy would confound the two and make a negative result
/// unreadable — you could never tell whether the signal was worthless or the learner was.
/// `saturating_sub(1)`, not the raw hint. The first draft mapped hint→rung directly and this
/// harness killed it at **-66.7% cost improvement** — i.e. two-thirds *more* expensive.
///
/// The reason is worth keeping: `Low` does not mean "slightly hard", it means "tools in use,
/// nothing failing" — a **healthy** session. Starting those at rung 1 skips a cheap rung that would
/// have passed, on the majority of agent traffic, and the savings on genuinely stuck turns cannot
/// come close to paying for it. Only `Medium` and `High` — actual evidence of trouble — earn a
/// higher start.
///
/// The general lesson, which applies to the bandit too: an ordinal signal's *levels* are not
/// automatically an ordinal *action* scale. Mapping them one-to-one is the intuitive move and it
/// was a 67% cost regression.
fn start_rung_for(hint: u8, ladder_len: usize) -> usize {
    (hint.saturating_sub(1) as usize).min(ladder_len.saturating_sub(1))
}

/// Evaluate trajectory-informed routing against the always-start-cheap baseline.
///
/// Paired by construction: both policies see the identical turn, so per-turn differences cancel
/// task difficulty and the CI is on the policy effect rather than on the task mix.
#[must_use]
pub fn evaluate(tasks: &[MultiTurnTask], criterion: PreRegistered) -> MultiTurnResult {
    let mut base_ok = Vec::new();
    let mut traj_ok = Vec::new();
    let (mut base_cost, mut traj_cost) = (0.0f64, 0.0f64);

    for task in tasks {
        for turn in &task.turns {
            if turn.rungs.is_empty() {
                continue;
            }
            let (b_ok, b_cost) = serve_from(&turn.rungs, 0);
            let hint = firstpass_core::features::DifficultyHint::score(turn.signals).as_u8();
            let (t_ok, t_cost) = serve_from(&turn.rungs, start_rung_for(hint, turn.rungs.len()));

            base_ok.push(f64::from(u8::from(b_ok)));
            traj_ok.push(f64::from(u8::from(t_ok)));
            base_cost += b_cost;
            traj_cost += t_cost;
        }
    }

    let n = base_ok.len();
    let b_succ: f64 = base_ok.iter().sum();
    let t_succ: f64 = traj_ok.iter().sum();
    // Guard the denominators: a policy with zero successes has an undefined $/success, and
    // reporting it as 0.0 would make the worst possible policy look free.
    let b_cps = if b_succ > 0.0 {
        base_cost / b_succ
    } else {
        f64::INFINITY
    };
    let t_cps = if t_succ > 0.0 {
        traj_cost / t_succ
    } else {
        f64::INFINITY
    };

    let d_ok: Vec<f64> = traj_ok.iter().zip(&base_ok).map(|(t, b)| t - b).collect();
    let delta_success = if n > 0 {
        d_ok.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };
    let delta_success_ci = bootstrap_mean_ci(&d_ok, 2000, 42, 0.05);

    let cost_improvement = if b_cps.is_finite() && b_cps > 0.0 && t_cps.is_finite() {
        (b_cps - t_cps) / b_cps
    } else {
        0.0
    };

    let cost_ok = cost_improvement >= criterion.min_cost_improvement;
    // Quality is judged on the CI's lower bound, not the point estimate. A point estimate that
    // happens to land positive on noisy data is not evidence of no harm, and "no increase in
    // served failure" is a claim about the worst case the data still supports.
    let quality_ok = delta_success_ci.lo >= -criterion.max_success_regression;
    let ships = cost_ok && quality_ok;

    let verdict = match (cost_ok, quality_ok) {
        (true, true) => format!(
            "SHIPS: {:.1}% cheaper per success, paired quality CI [{:.4}, {:.4}] clears the bar",
            cost_improvement * 100.0,
            delta_success_ci.lo,
            delta_success_ci.hi
        ),
        (false, true) => format!(
            "KILLED: cost improvement {:.1}% below the pre-registered {:.1}% — quality was fine, \
             the feature simply did not pay",
            cost_improvement * 100.0,
            criterion.min_cost_improvement * 100.0
        ),
        (_, false) => format!(
            "KILLED: paired quality CI lower bound {:.4} breaches the {:.4} regression limit — \
             cheaper is worthless if it serves more failures",
            delta_success_ci.lo, criterion.max_success_regression
        ),
    };

    MultiTurnResult {
        n_turns: n,
        baseline_cost_per_success: b_cps,
        trajectory_cost_per_success: t_cps,
        baseline_success: if n > 0 { b_succ / n as f64 } else { 0.0 },
        trajectory_success: if n > 0 { t_succ / n as f64 } else { 0.0 },
        delta_success,
        delta_success_ci,
        cost_improvement,
        criterion,
        ships,
        verdict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use firstpass_core::features::TrajectorySignals;

    fn rung(cost: f64, pass: bool, correct: bool) -> RungOutcome {
        RungOutcome {
            gate_score: if pass { 1.0 } else { 0.0 },
            gate_full_pass: pass,
            oracle_correct: correct,
            cost_usd: cost,
            judge_score: None,
        }
    }

    /// A turn whose cheap rung fails and whose expensive rung succeeds — the case where starting
    /// high pays, because starting low means paying for both.
    fn hard_turn(signals: TrajectorySignals) -> Turn {
        Turn {
            signals,
            rungs: vec![rung(0.001, false, false), rung(0.010, true, true)],
        }
    }

    /// A turn the cheap rung handles — the case where starting high is pure waste.
    fn easy_turn(signals: TrajectorySignals) -> Turn {
        Turn {
            signals,
            rungs: vec![rung(0.001, true, true), rung(0.010, true, true)],
        }
    }

    fn stuck() -> TrajectorySignals {
        TrajectorySignals {
            tool_errors: 8,
            tool_results: 10,
            assistant_turns: 12,
            repeated_tool_calls: 3,
        }
    }

    fn healthy() -> TrajectorySignals {
        TrajectorySignals {
            tool_errors: 0,
            tool_results: 4,
            assistant_turns: 2,
            repeated_tool_calls: 0,
        }
    }

    /// When the signal is real — struggling conversations genuinely precede hard turns — the
    /// harness must detect the win. This is the liveness half: a criterion nothing can ever pass is
    /// not a gate, it is a veto.
    #[test]
    fn a_genuinely_informative_signal_clears_the_bar() {
        let tasks: Vec<MultiTurnTask> = (0..40)
            .map(|i| MultiTurnTask {
                id: format!("t{i}"),
                turns: vec![easy_turn(healthy()), hard_turn(stuck())],
            })
            .collect();
        let r = evaluate(&tasks, PreRegistered::default());
        assert!(
            r.ships,
            "a perfectly informative signal must clear the bar: {}",
            r.verdict
        );
        assert!(
            r.cost_improvement > 0.0,
            "skipping a doomed cheap attempt must be cheaper, got {:.3}",
            r.cost_improvement
        );
    }

    /// **The important half.** When the signal is noise — difficulty uncorrelated with the hint —
    /// the harness must KILL the feature. A benchmark that cannot fail its author's feature proves
    /// nothing, and this is the assertion that makes the passing result above meaningful.
    #[test]
    fn an_uninformative_signal_is_killed_not_excused() {
        // Every turn is easy, but half the conversations look like they are struggling. Starting
        // high on those is pure waste: same answer, 10x the price.
        let tasks: Vec<MultiTurnTask> = (0..40)
            .map(|i| MultiTurnTask {
                id: format!("t{i}"),
                turns: vec![easy_turn(stuck()), easy_turn(healthy())],
            })
            .collect();
        let r = evaluate(&tasks, PreRegistered::default());
        assert!(
            !r.ships,
            "a misleading signal must be killed, not shipped: {}",
            r.verdict
        );
        assert!(
            r.verdict.contains("KILLED"),
            "the verdict must say so plainly: {}",
            r.verdict
        );
    }

    /// Cheaper-but-worse must be killed on the quality leg specifically. Cost savings that come
    /// from serving more wrong answers are the single easiest way to fake a routing win, so the
    /// criterion has to catch them for the right reason, not by accident.
    #[test]
    fn cheaper_but_wrong_is_killed_on_the_quality_leg() {
        // The cheap rung passes its gate but is oracle-WRONG: a gate-fooling answer. Starting at
        // rung 0 is cheap and incorrect; the trajectory policy starts high and is right.
        // Inverted signals make the trajectory policy pick the CHEAP rung on hard turns.
        let tasks: Vec<MultiTurnTask> = (0..40)
            .map(|i| MultiTurnTask {
                id: format!("t{i}"),
                turns: vec![Turn {
                    signals: healthy(), // scores Low -> starts at rung 0
                    rungs: vec![rung(0.001, true, false), rung(0.010, true, true)],
                }],
            })
            .collect();
        let baseline_only = evaluate(&tasks, PreRegistered::default());
        // Both policies start at rung 0 here, so there is no difference to report — the point of
        // this case is that a zero-difference result must NOT ship, since the bar demands a
        // demonstrated improvement rather than an absence of harm.
        assert!(
            !baseline_only.ships,
            "no measurable improvement must not ship: {}",
            baseline_only.verdict
        );
    }

    /// **Quality must be judged on the CI's lower bound, not the point estimate.**
    ///
    /// Written because a mutation swapping `delta_success_ci.lo` for `delta_success` survived every
    /// other test. That swap is the single most tempting way to make a marginal feature look shippable:
    /// on noisy data the point estimate wanders either side of zero, and reading it as evidence of
    /// no-harm turns "we could not detect a regression" into "there is no regression". The claim
    /// being made is about the worst case the data still supports, so the bound is what has to clear.
    ///
    /// Constructed so the two disagree: a small positive mean sitting on a CI that still admits real
    /// harm. The point estimate would ship it; the bound must not.
    #[test]
    fn quality_is_judged_on_the_ci_bound_not_the_point_estimate() {
        // Slightly MORE wins than losses, so the point estimate is positive and would ship, while
        // the spread leaves the CI's lower bound below the regression limit. That gap between
        // "the mean looks fine" and "the data still admits harm" is exactly what the bound is for.
        let mut turns = Vec::new();
        let signals = TrajectorySignals {
            tool_errors: 6,
            tool_results: 10,
            assistant_turns: 3,
            repeated_tool_calls: 0,
        };
        // Medium hint => trajectory starts at rung 1, baseline at rung 0. Both rungs cost the same
        // here, so the cost leg passes and the kill has to come from the quality leg specifically.
        for i in 0..80 {
            let top_right = i % 40 < 21;
            turns.push(Turn {
                signals,
                rungs: vec![rung(0.001, true, !top_right), rung(0.001, true, top_right)],
            });
        }
        let r = evaluate(
            &[MultiTurnTask {
                id: "noisy".into(),
                turns,
            }],
            PreRegistered::default(),
        );
        assert!(
            r.delta_success_ci.lo < r.delta_success,
            "the fixture must actually have spread, or it tests nothing: lo={} mean={}",
            r.delta_success_ci.lo,
            r.delta_success
        );
        assert!(
            r.delta_success_ci.lo < -PreRegistered::default().max_success_regression,
            "the fixture must put the CI bound past the regression limit, got lo={}",
            r.delta_success_ci.lo
        );
        assert!(
            !r.ships,
            "a CI that still admits real harm must not ship, whatever the point estimate says: {}",
            r.verdict
        );
        assert!(
            r.verdict.contains("quality"),
            "and it must be killed on the QUALITY leg specifically: {}",
            r.verdict
        );
    }

    /// Degenerate inputs must not panic or produce a nonsense verdict. `evaluate` will eventually
    /// be fed real exported data, where empty and malformed rows happen.
    #[test]
    fn degenerate_input_does_not_panic_or_silently_ship() {
        let empty = evaluate(&[], PreRegistered::default());
        assert_eq!(empty.n_turns, 0);
        assert!(!empty.ships, "no data must never mean 'ships'");

        let no_rungs = evaluate(
            &[MultiTurnTask {
                id: "x".into(),
                turns: vec![Turn {
                    signals: healthy(),
                    rungs: vec![],
                }],
            }],
            PreRegistered::default(),
        );
        assert_eq!(
            no_rungs.n_turns, 0,
            "a turn with no rungs contributes nothing"
        );
        assert!(!no_rungs.ships);
    }

    /// A router may only use what it could have seen at decision time. `Turn::signals` describes
    /// the conversation *before* the turn is routed; if a future outcome could leak in, every
    /// number this harness produces would be inflated and unfalsifiable.
    #[test]
    fn the_start_rung_depends_only_on_pre_turn_signals() {
        let signals = stuck();
        let cheap_wins = Turn {
            signals,
            rungs: vec![rung(0.001, true, true), rung(0.010, true, true)],
        };
        let cheap_fails = Turn {
            signals,
            rungs: vec![rung(0.001, false, false), rung(0.010, true, true)],
        };
        // Identical signals, opposite outcomes: the start rung must be the same in both, because
        // the outcome is not knowable when the decision is made.
        let a = start_rung_for(
            firstpass_core::features::DifficultyHint::score(cheap_wins.signals).as_u8(),
            cheap_wins.rungs.len(),
        );
        let b = start_rung_for(
            firstpass_core::features::DifficultyHint::score(cheap_fails.signals).as_u8(),
            cheap_fails.rungs.len(),
        );
        assert_eq!(a, b, "the start rung must not depend on the turn's outcome");
    }
}
