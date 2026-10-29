//! The competitors' *mechanisms*, replayed over our measured matrix — what each class of router in
//! the market would have served on the same tasks, at the same prices, with the same ground truth.
//!
//! No public leaderboard can score a verification router (RouterBench and RouterArena take one
//! predicted model per query and have no executable oracle), so a head-to-head has to be built
//! here. Calling the vendors live would cost money and measure their prompt classifiers on a
//! workload they were not tuned for; replaying the *decision rule* each one uses is free, exactly
//! reproducible from the committed checkpoints, and isolates the one question that matters: given
//! the same candidates, does deciding before generation, deciding on a judge's opinion, or
//! deciding on the executed tests serve more right answers per dollar?
//!
//! | arm | stands in for | decision rule |
//! |---|---|---|
//! | `oracle-predictive` | Not Diamond, Martian, OpenRouter Auto, Foundry Model Router, Bedrock IPR | the cheapest rung that is *actually correct*, chosen before generation with perfect foresight — an upper bound no predictive router reaches |
//! | `judge-escalate` | NVIDIA NeMo Switchyard's escalation router | serve rung 0 unless an LLM judge doubts it; never runs the tests |
//! | `conformal-cascade` | Conformal Cascade (arXiv 2607.25018) | accept a rung when a confidence score clears a threshold calibrated so accepted-and-wrong ≤ α |
//!
//! Two of these arms read `judge_score`, so they need a judged checkpoint; they are omitted, not
//! faked, on a matrix without one. A judged checkpoint bills the judge into each rung's
//! `cost_usd`, so on it *every* arm pays the judge on every rung it visits — the test-gated
//! first-pass included, though it does not need the judge. That handicaps first-pass, never the
//! competitors, so a first-pass win on that matrix is conservative.

use crate::coding_policy::{RungOutcome, serve_first_pass};
use crate::stats::{Ci, bootstrap_mean_ci, mean};

/// Judge threshold for the Switchyard-style arm. Fixed, not tuned on the data it is scored on —
/// the same value the `first-pass+judge` arm uses, so the two judge arms differ only in whether
/// the tests are also run.
const JUDGE_THRESHOLD: f64 = 0.8;

/// Target risk for the conformal-cascade arm, matching the served-failure α the product uses.
const ALPHA: f64 = 0.10;

/// One arm's result, plus its paired difference against first-pass.
#[derive(Debug, Clone)]
pub struct Arm {
    /// Arm name.
    pub name: &'static str,
    /// Which product or paper this arm's decision rule stands in for.
    pub stands_in_for: &'static str,
    /// Fraction of tasks whose served answer was correct per the hidden oracle.
    pub success: f64,
    /// USD per correct served answer.
    pub usd_per_success: f64,
    /// First-pass minus this arm, per task: positive = first-pass served more correct answers.
    pub fp_delta_success: Ci,
    /// First-pass minus this arm, per task, in USD: negative = first-pass is cheaper.
    pub fp_delta_cost: Ci,
}

/// The head-to-head.
#[derive(Debug, Clone)]
pub struct Study {
    /// Tasks replayed.
    pub n: usize,
    /// First-pass itself, as the row every arm is compared against.
    pub first_pass: (f64, f64),
    /// Competitor arms, in table order.
    pub arms: Vec<Arm>,
    /// Share of the always-top → oracle cost gap first-pass closes, in `[0, 1]` when it lands
    /// between them. Published predictive routers report closing well under half of this gap.
    pub cost_gap_closed: f64,
    /// Per-tier thresholds the conformal arm calibrated, one per cross-fit fold.
    pub conformal_lambdas: [f64; 2],
}

/// A decision rule: task index (for the cross-fit fold) and its row → (served correct, USD paid).
type Rule<'a> = dyn Fn(usize, &[RungOutcome]) -> (bool, f64) + 'a;
type Serve<'a> = Box<Rule<'a>>;

/// Perfect-foresight predictive routing: one call, to the cheapest rung that is correct. When no
/// rung is correct a predictive router still has to pick one; it picks the cheapest, which is the
/// most charitable choice available to it.
fn oracle_predictive(row: &[RungOutcome]) -> (bool, f64) {
    row.iter()
        .find(|o| o.oracle_correct)
        .or(row.first())
        .map_or((false, 0.0), |o| (o.oracle_correct, o.cost_usd))
}

/// Serve the first rung whose judge score clears `threshold`; the last rung always serves. This
/// is both the Switchyard-style arm (escalate on the judge's doubt alone, never run the tests)
/// and conformal-cascade acceptance (the threshold is a calibrated λ). An abstaining judge
/// (`None`) is doubt: neither design accepts what it cannot score.
fn judge_escalate(row: &[RungOutcome], threshold: f64) -> (bool, f64) {
    let mut spent = 0.0;
    for (i, o) in row.iter().enumerate() {
        spent += o.cost_usd;
        let last = i + 1 == row.len();
        if last || o.judge_score.is_some_and(|j| j >= threshold) {
            return (o.oracle_correct, spent);
        }
    }
    (false, spent)
}

/// Smallest threshold whose conformal risk bound on the calibration rows, `(E + 1) / (n + 1)`
/// with `E` = rung-0 answers accepted *and* wrong, is at most `alpha`. Thresholds are scanned
/// from the most permissive up, so the first one that holds is the cheapest valid one. Returns
/// a value above every score (accept nothing) when none holds.
fn calibrate(cal: &[&[RungOutcome]], alpha: f64) -> f64 {
    let mut grid: Vec<f64> = cal
        .iter()
        .filter_map(|r| r.first().and_then(|o| o.judge_score))
        .collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let n = cal.len() as f64;
    for &l in &grid {
        let errors = cal
            .iter()
            .filter(|r| {
                r.first()
                    .is_some_and(|o| o.judge_score.is_some_and(|j| j >= l) && !o.oracle_correct)
            })
            .count() as f64;
        if (errors + 1.0) / (n + 1.0) <= alpha {
            return l;
        }
    }
    f64::INFINITY
}

fn per_task(matrix: &[Vec<RungOutcome>], serve: &Rule<'_>) -> (Vec<f64>, Vec<f64>) {
    matrix
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let (ok, c) = serve(i, r);
            (f64::from(u8::from(ok)), c)
        })
        .unzip()
}

fn usd_per_success(ok: &[f64], cost: &[f64]) -> f64 {
    let s: f64 = ok.iter().sum();
    if s > 0.0 {
        cost.iter().sum::<f64>() / s
    } else {
        f64::INFINITY
    }
}

/// Replay every competitor arm against first-pass. Pure — no I/O, no spend, deterministic.
#[must_use]
pub fn study(matrix: &[Vec<RungOutcome>]) -> Study {
    let judged = matrix
        .iter()
        .any(|r| r.iter().any(|o| o.judge_score.is_some()));

    // Two-fold cross-fit by task parity: each task's threshold is calibrated on the other fold,
    // so no task is scored by a threshold it helped choose. Parity rather than a shuffle keeps
    // the split deterministic without a seed, and MBPP's order carries no difficulty trend.
    let fold = |parity: usize| -> Vec<&[RungOutcome]> {
        matrix
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == parity)
            .map(|(_, r)| r.as_slice())
            .collect()
    };
    let lambdas = [calibrate(&fold(1), ALPHA), calibrate(&fold(0), ALPHA)];

    let fp = |_: usize, r: &[RungOutcome]| {
        let (ok, c, _) = serve_first_pass(r);
        (ok, c)
    };
    let (fp_ok, fp_cost) = per_task(matrix, &fp);

    let mut specs: Vec<(&'static str, &'static str, Serve<'_>)> = vec![(
        "oracle-predictive (upper bound)",
        "Not Diamond · Martian · OpenRouter Auto · Foundry · Bedrock IPR",
        Box::new(|_, r| oracle_predictive(r)),
    )];
    if judged {
        specs.push((
            "judge-escalate",
            "NVIDIA NeMo Switchyard escalation router",
            Box::new(|_, r| judge_escalate(r, JUDGE_THRESHOLD)),
        ));
        specs.push((
            "conformal-cascade (α=0.10)",
            "Conformal Cascade, arXiv 2607.25018",
            Box::new(move |i, r| judge_escalate(r, lambdas[i % 2])),
        ));
    }

    let arms = specs
        .into_iter()
        .map(|(name, stands_in_for, serve)| {
            let (ok, cost) = per_task(matrix, &*serve);
            let d_ok: Vec<f64> = fp_ok.iter().zip(&ok).map(|(a, b)| a - b).collect();
            let d_c: Vec<f64> = fp_cost.iter().zip(&cost).map(|(a, b)| a - b).collect();
            Arm {
                name,
                stands_in_for,
                success: mean(&ok),
                usd_per_success: usd_per_success(&ok, &cost),
                fp_delta_success: bootstrap_mean_ci(&d_ok, 2000, 42, 0.05),
                fp_delta_cost: bootstrap_mean_ci(&d_c, 2000, 42, 0.05),
            }
        })
        .collect();

    let top_cost: f64 = matrix
        .iter()
        .filter_map(|r| r.last())
        .map(|o| o.cost_usd)
        .sum();
    let oracle_cost: f64 = matrix.iter().map(|r| oracle_predictive(r).1).sum();
    let fp_total: f64 = fp_cost.iter().sum();
    let gap = top_cost - oracle_cost;

    Study {
        n: matrix.len(),
        first_pass: (mean(&fp_ok), usd_per_success(&fp_ok, &fp_cost)),
        arms,
        cost_gap_closed: if gap > 0.0 {
            (top_cost - fp_total) / gap
        } else {
            0.0
        },
        conformal_lambdas: lambdas,
    }
}

/// Markdown, in the bench's honest-report style.
#[must_use]
pub fn render(s: &Study) -> String {
    let mut out = format!(
        "## Head-to-head: competitor decision rules on the same matrix\n\n\
         n = {} tasks · every arm sees identical candidates, prices and ground truth\n\n\
         | arm | stands in for | success | $/success | Δ success (fp − arm) | Δ $/task (fp − arm) |\n\
         |---|---|---|---|---|---|\n\
         | **first-pass** | Firstpass (executed-test gate) | {:.4} | ${:.5} | — | — |\n",
        s.n, s.first_pass.0, s.first_pass.1
    );
    for a in &s.arms {
        out.push_str(&format!(
            "| {} | {} | {:.4} | ${:.5} | {:+.3} [{:+.3}, {:+.3}] | {:+.5} [{:+.5}, {:+.5}] |\n",
            a.name,
            a.stands_in_for,
            a.success,
            a.usd_per_success,
            a.fp_delta_success.point,
            a.fp_delta_success.lo,
            a.fp_delta_success.hi,
            a.fp_delta_cost.point,
            a.fp_delta_cost.lo,
            a.fp_delta_cost.hi,
        ));
    }
    out.push('\n');
    // A negative share means first-pass cost MORE than always-top on this ladder; quoting it as a
    // percentage of a gap would read as a (strange) result rather than as the failure it is.
    if s.cost_gap_closed >= 0.0 {
        out.push_str(&format!(
            "First-pass closes {:.1}% of the always-top → perfect-predictive-router cost gap. ",
            s.cost_gap_closed * 100.0
        ));
    } else {
        out.push_str(
            "First-pass costs MORE than always-top on this ladder, so it closes none of the \
             always-top → perfect-predictive-router cost gap. ",
        );
    }
    out.push_str(
        "The oracle row is an upper bound on every predictive router: it knows each answer's \
         correctness before generating it.",
    );
    if s.arms.len() > 1 {
        out.push_str(&format!(
            " Conformal thresholds per fold: {:?}.",
            s.conformal_lambdas
        ));
    }
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(gate: bool, oracle: bool, cost: f64, judge: Option<f64>) -> RungOutcome {
        RungOutcome {
            gate_score: f64::from(u8::from(gate)),
            gate_full_pass: gate,
            oracle_correct: oracle,
            cost_usd: cost,
            judge_score: judge,
        }
    }

    /// The oracle bills exactly one call — the cheapest correct rung — and never an escalation.
    #[test]
    fn oracle_picks_cheapest_correct_rung_with_one_call() {
        let row = [o(false, false, 1.0, None), o(true, true, 5.0, None)];
        assert_eq!(oracle_predictive(&row), (true, 5.0));
        let none = [o(false, false, 1.0, None), o(false, false, 5.0, None)];
        assert_eq!(oracle_predictive(&none), (false, 1.0));
    }

    /// The judge arm must ignore the tests entirely: a confident judge serves a test-failing
    /// wrong answer, and a doubtful judge escalates past a test-passing right one. That
    /// difference from first-pass is the whole point of the arm.
    #[test]
    fn judge_escalate_ignores_tests() {
        let fooled = [
            o(false, false, 1.0, Some(1.0)),
            o(true, true, 5.0, Some(1.0)),
        ];
        assert_eq!(judge_escalate(&fooled, 0.8), (false, 1.0));
        let doubted = [o(true, true, 1.0, Some(0.2)), o(true, true, 5.0, Some(0.2))];
        assert_eq!(judge_escalate(&doubted, 0.8), (true, 6.0));
        // Abstention is doubt for this arm.
        let abstain = [o(true, true, 1.0, None), o(true, false, 5.0, None)];
        assert_eq!(judge_escalate(&abstain, 0.8), (false, 6.0));
    }

    /// Calibration must pick the most permissive threshold whose risk bound holds, and refuse
    /// everything when none does.
    #[test]
    fn calibrate_picks_cheapest_valid_threshold() {
        // 19 confident-right rows and one 0.4-scored wrong row: λ=0.4 gives (1+1)/21 > 0.05,
        // λ=1.0 gives (0+1)/21 ≤ 0.05.
        let mut rows: Vec<Vec<RungOutcome>> = (0..19)
            .map(|_| vec![o(true, true, 1.0, Some(1.0))])
            .collect();
        rows.push(vec![o(true, false, 1.0, Some(0.4))]);
        let cal: Vec<&[RungOutcome]> = rows.iter().map(Vec::as_slice).collect();
        assert!((calibrate(&cal, 0.05) - 1.0).abs() < 1e-12);
        // Every row wrong: no threshold is valid.
        let bad: Vec<Vec<RungOutcome>> = (0..5)
            .map(|_| vec![o(true, false, 1.0, Some(1.0))])
            .collect();
        let cal: Vec<&[RungOutcome]> = bad.iter().map(Vec::as_slice).collect();
        assert!(calibrate(&cal, 0.10).is_infinite());
    }

    /// An unjudged matrix gets only the oracle arm — the judge arms are omitted, not faked.
    #[test]
    fn judge_arms_omitted_without_a_judge() {
        let m = vec![vec![o(true, true, 1.0, None), o(true, true, 5.0, None)]; 4];
        let s = study(&m);
        assert_eq!(s.arms.len(), 1);
        // first-pass serves rung 0 everywhere: cost equals the oracle's, so it closes 100%.
        assert!((s.cost_gap_closed - 1.0).abs() < 1e-12);
    }
}
