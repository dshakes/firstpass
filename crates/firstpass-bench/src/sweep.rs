//! The imperfect-gate study: what happens when the gate is allowed to be wrong.
//!
//! # Why this exists
//!
//! The published MBPP result — 974 tasks, zero regressions — holds because MBPP unit tests are a
//! near-perfect oracle. The measured false-reject rate was 0.0%, so the only path by which
//! escalation can *hurt* (the gate rejects a correct cheap answer and the rung above then gets it
//! wrong) was never entered. That is a property of that workload, not of the design, and a
//! reviewer is right to discount a guarantee demonstrated only where the gate cannot err.
//!
//! The earlier plan for fixing this (`specs/imperfect-gate-benchmark.md`) was to buy a *weak LLM
//! judge* and measure its mistakes. That design predates both the sandbox and the 974-task run,
//! and it costs money to obtain something the existing measurements already contain:
//! [`RungOutcome::gate_score`](crate::coding_policy::RungOutcome) is the **continuous fraction of
//! visible cases passed**, not a boolean. The production policy happens to serve at `τ = 1.0`
//! (`gate_full_pass`), which is the single strictest point on that axis and the one point where a
//! test gate makes almost no false accepts.
//!
//! Sweeping `τ` downward therefore *manufactures a genuinely imperfect gate out of a perfect one*,
//! at no cost and with ground truth attached: at `τ = 0.6` a candidate passing 3 of 5 visible
//! cases is served, and the hidden oracle says whether that was a mistake. Every quantity the
//! judge design was meant to produce — real false-accept and false-reject rates, a non-degenerate
//! score distribution, a feasible conformal threshold — falls out of arithmetic over a matrix
//! already paid for.
//!
//! # Calibration is split, unlike the in-place summary
//!
//! [`crate::coding::summarize`] calibrates the conformal threshold and then measures served-failure
//! on **the same** outcomes. That is in-sample: the threshold has already seen every point it is
//! then scored against, so the reported risk is optimistic by construction. Split conformal
//! requires a held-out set, so this module calibrates `λ` on one half and reports realized
//! served-failure on the disjoint other half. The split is by index parity — deterministic, so the
//! number reproduces without recording a seed.

use crate::coding_policy::RungOutcome;
use firstpass_core::conformal::{self, ConformalResult};

/// Which signal the sweep thresholds on.
///
/// This is not a preference knob. On MBPP the test gate's score is **empirically binary** —
/// measured `{0.0, 1.0}` with nothing between — because a generated function either satisfies all
/// of its asserts or none of them. Partial credit, which the score's type permits, essentially
/// does not occur. A sweep over that signal has exactly two operating points and cannot express a
/// gate that is *slightly* wrong, which is the whole object of study.
///
/// A k-sample judge does produce a graded signal (its score is the fraction of samples agreeing —
/// measured `{0.0, 0.2, 0.4, 0.6, 1.0}` at k=5), so where one ran, the fused score is the only
/// signal on which this study means anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// The deterministic test gate alone. Honest, and on MBPP nearly binary.
    TestOnly,
    /// Tests fused with the judge, per [`crate::coding::combined_from`].
    Combined,
}

impl Signal {
    /// The label used in the rendered report, so a reader always knows which axis was swept.
    const fn label(self) -> &'static str {
        match self {
            Self::TestOnly => "test gate only (binary on MBPP)",
            Self::Combined => "tests fused with a k-sample judge",
        }
    }

    /// Score one outcome under this signal.
    fn score(self, o: &RungOutcome) -> f64 {
        match self {
            Self::TestOnly => o.gate_score,
            Self::Combined => {
                crate::coding::combined_from(o.gate_score, o.gate_full_pass, o.judge_score)
            }
        }
    }
}

/// Pick the signal that actually carries gradation: the fused score where a judge ran, the test
/// score otherwise. Chosen from the data rather than configured, so a run cannot silently sweep an
/// axis that has no intermediate values on it.
#[must_use]
pub fn signal_for(matrix: &[Vec<RungOutcome>]) -> Signal {
    let judged = matrix
        .iter()
        .any(|r| r.iter().any(|o| o.judge_score.is_some()));
    if judged {
        Signal::Combined
    } else {
        Signal::TestOnly
    }
}

/// One row of the sweep: what serving at threshold `tau` does to quality, cost, and gate error.
#[derive(Debug, Clone)]
pub struct SweepRow {
    /// Serve the first rung whose `gate_score >= tau`.
    pub tau: f64,
    /// Fraction of tasks served from the cheapest rung.
    pub served_from_cheap: f64,
    /// Fraction of tasks that paid for more than one rung.
    pub escalation_rate: f64,
    /// Among cheap answers the gate **accepted**, the fraction the oracle says were wrong. This is
    /// the error that reaches the caller.
    pub false_accept: f64,
    /// Among cheap answers the gate **rejected**, the fraction the oracle says were right. This is
    /// the error that costs money and opens the only path to a regression.
    pub false_reject: f64,
    /// Fraction of tasks whose served answer was correct.
    pub success_rate: f64,
    /// Fraction of served answers that were wrong.
    pub served_failure: f64,
    /// Total USD across the suite, including rungs tried and rejected.
    pub total_cost_usd: f64,
    /// USD per successfully-solved task.
    pub usd_per_success: f64,
}

/// The whole study.
#[derive(Debug, Clone)]
pub struct GateSweep {
    /// Tasks in the matrix.
    pub n: usize,
    /// One row per threshold, strictest first.
    pub rows: Vec<SweepRow>,
    /// How well `gate_score` discriminates oracle-correct from oracle-wrong at the cheap rung.
    /// 0.5 is a coin flip; a gate below ~0.65 is not carrying usable signal whatever its accuracy.
    pub auc: f64,
    /// Target served-failure rate the conformal threshold was calibrated for.
    pub alpha: f64,
    /// Confidence parameter.
    pub delta: f64,
    /// Threshold calibrated on the calibration split.
    pub conformal: ConformalResult,
    /// Realized served-failure on the **held-out** split at that threshold — the number that can
    /// embarrass the bound, which is the point of holding it out.
    pub validation_served_failure: f64,
    /// How many held-out items were actually served at the threshold.
    pub validation_served_n: usize,
    /// Which signal was swept, chosen from the data by [`signal_for`].
    pub signal: Signal,
    /// Distinct threshold values the signal actually took. A count of 2 means the signal is binary
    /// on this workload and the sweep has no interior to explore — reported rather than hidden,
    /// because a two-row sweep otherwise looks like a completed study.
    pub distinct_levels: usize,
}

/// Serve the first rung scoring at least `tau`, paying for every rung tried.
///
/// Returns `(served_correct, spent, rungs_paid)`. When no rung clears the bar the top rung is
/// served anyway — refusing to answer is not on the table for a proxy, and counting it as unserved
/// would let a policy hide its failures by declining them.
fn serve_at(row: &[RungOutcome], tau: f64, sig: Signal) -> (bool, f64, usize) {
    let mut spent = 0.0;
    for (i, o) in row.iter().enumerate() {
        spent += o.cost_usd;
        if sig.score(o) >= tau {
            return (o.oracle_correct, spent, i + 1);
        }
    }
    (
        row.last().is_some_and(|o| o.oracle_correct),
        spent,
        row.len(),
    )
}

/// Thresholds to sweep. The distinct observed scores are the only points where behaviour changes,
/// so sweeping them is both exhaustive and free of an arbitrary grid — plus 1.0, which is the
/// production policy and must always appear for comparison.
fn thresholds(matrix: &[Vec<RungOutcome>], sig: Signal) -> Vec<f64> {
    let mut ts: Vec<f64> = matrix
        .iter()
        .filter_map(|r| r.first())
        .map(|o| sig.score(o))
        .collect();
    ts.push(1.0);
    ts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    ts
}

/// Run the sweep over a measured matrix. Pure — no I/O, no spend, deterministic.
///
/// `alpha` is the target served-failure rate, `delta` the confidence parameter, `min_n` the
/// smallest served set a bound may be certified on.
#[must_use]
pub fn sweep(matrix: &[Vec<RungOutcome>], alpha: f64, delta: f64, min_n: usize) -> GateSweep {
    let n = matrix.len();
    let mut rows = Vec::new();
    let sig = signal_for(matrix);

    for tau in thresholds(matrix, sig) {
        let mut served_cheap = 0usize;
        let mut escalated = 0usize;
        let mut correct = 0usize;
        let mut spent_total = 0.0;
        // Gate confusion is measured at the cheap rung, where the gate's decision is actually made.
        let (mut acc, mut acc_wrong, mut rej, mut rej_right) = (0usize, 0usize, 0usize, 0usize);

        for row in matrix {
            let (ok, spent, rungs) = serve_at(row, tau, sig);
            if rungs == 1 {
                served_cheap += 1;
            } else {
                escalated += 1;
            }
            if ok {
                correct += 1;
            }
            spent_total += spent;

            if let Some(c) = row.first() {
                if sig.score(c) >= tau {
                    acc += 1;
                    if !c.oracle_correct {
                        acc_wrong += 1;
                    }
                } else {
                    rej += 1;
                    if c.oracle_correct {
                        rej_right += 1;
                    }
                }
            }
        }

        let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };
        rows.push(SweepRow {
            tau,
            served_from_cheap: ratio(served_cheap, n),
            escalation_rate: ratio(escalated, n),
            false_accept: ratio(acc_wrong, acc),
            false_reject: ratio(rej_right, rej),
            success_rate: ratio(correct, n),
            served_failure: ratio(n - correct, n),
            total_cost_usd: spent_total,
            usd_per_success: if correct == 0 {
                f64::INFINITY
            } else {
                spent_total / correct as f64
            },
        });
    }

    // Split conformal on the cheap rung's continuous score. Parity split: deterministic, balanced,
    // and reproducible without a recorded seed.
    let pairs: Vec<(f64, bool)> = matrix
        .iter()
        .filter_map(|r| r.first())
        .map(|o| (sig.score(o), o.oracle_correct))
        .collect();
    let calib: Vec<(f64, bool)> = pairs.iter().step_by(2).copied().collect();
    let valid: Vec<(f64, bool)> = pairs.iter().skip(1).step_by(2).copied().collect();

    let conformal = conformal::calibrate(&calib, alpha, delta, min_n);
    let (validation_served_failure, validation_served_n) =
        conformal::served_failure_rate(&valid, conformal.threshold);

    GateSweep {
        n,
        distinct_levels: rows.len(),
        rows,
        auc: crate::coding::auc(&pairs),
        alpha,
        delta,
        conformal,
        validation_served_failure,
        validation_served_n,
        signal: sig,
    }
}

/// Render the sweep as Markdown.
#[must_use]
pub fn render(s: &GateSweep) -> String {
    let mut out = String::new();
    out.push_str("## Imperfect-gate sweep — serving below full-pass\n\n");
    out.push_str(&format!(
        "n = {} tasks · signal = {} · distinct levels = {} · AUC (score → oracle-correct) = {:.3}\n\n",
        s.n,
        s.signal.label(),
        s.distinct_levels,
        s.auc
    ));
    if s.distinct_levels <= 2 {
        out.push_str(
            "> **This sweep is degenerate.** The signal took only two values, so there is no \
             interior threshold to study and no graded gate error to measure. On MBPP the test \
             gate is all-or-nothing; run with a k-sample judge \
             (`FIRSTPASS_CODING_JUDGE_SAMPLES=5`) to obtain a graded signal.\n\n",
        );
    }
    out.push_str(
        "| τ | served cheap | escalated | false-accept | false-reject | success | served-fail | $/success |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for r in &s.rows {
        out.push_str(&format!(
            "| {:.2} | {:.0}% | {:.0}% | {:.3} | {:.3} | {:.4} | {:.4} | ${:.5} |\n",
            r.tau,
            r.served_from_cheap * 100.0,
            r.escalation_rate * 100.0,
            r.false_accept,
            r.false_reject,
            r.success_rate,
            r.served_failure,
            r.usd_per_success
        ));
    }

    out.push_str(&format!(
        "\n### Split-conformal bound (α = {:.2}, δ = {:.2})\n\n",
        s.alpha, s.delta
    ));
    if s.conformal.feasible {
        out.push_str(&format!(
            "Calibrated λ = {:.4} on the calibration half (risk {:.4}, serving {:.0}% of it).\n\
             Held-out realized served-failure at that λ: **{:.4}** over {} served items.\n",
            s.conformal.threshold,
            s.conformal.calib_risk,
            s.conformal.served_frac * 100.0,
            s.validation_served_failure,
            s.validation_served_n
        ));
        if s.validation_served_failure <= s.alpha {
            out.push_str("The held-out rate is within α — the bound survived data it never saw.\n");
        } else {
            out.push_str(
                "**The held-out rate exceeds α.** Reported as measured: an in-sample threshold \
                 that does not hold out of sample is exactly what a split is for.\n",
            );
        }
    } else {
        out.push_str(
            "No threshold met the target on the calibration half — the bound is **infeasible** on \
             this gate, which is the honest reading when the score cannot separate a safe prefix.\n",
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(score: f64, oracle: bool, cost: f64) -> RungOutcome {
        RungOutcome {
            gate_score: score,
            gate_full_pass: (score - 1.0).abs() < f64::EPSILON,
            oracle_correct: oracle,
            cost_usd: cost,
            judge_score: None,
        }
    }

    /// Same, with a k-sample judge's graded opinion attached.
    fn oj(score: f64, oracle: bool, cost: f64, judge: f64) -> RungOutcome {
        RungOutcome {
            judge_score: Some(judge),
            ..o(score, oracle, cost)
        }
    }

    /// The premise of the whole module: at τ = 1.0 a test gate makes no false accepts, and
    /// lowering τ creates them. If that were not so there would be no imperfect gate to study.
    #[test]
    fn lowering_tau_manufactures_false_accepts() {
        // Partial-credit answers that are actually wrong: invisible at τ=1.0, served below it.
        let matrix = vec![
            vec![o(1.0, true, 0.01), o(1.0, true, 0.10)],
            vec![o(0.6, false, 0.01), o(1.0, true, 0.10)],
            vec![o(0.6, false, 0.01), o(1.0, true, 0.10)],
        ];
        let s = sweep(&matrix, 0.10, 0.05, 1);
        let strict = s
            .rows
            .iter()
            .find(|r| (r.tau - 1.0).abs() < 1e-9)
            .expect("τ=1.0 row");
        let loose = s
            .rows
            .iter()
            .find(|r| (r.tau - 0.6).abs() < 1e-9)
            .expect("τ=0.6 row");
        assert!(
            (strict.false_accept - 0.0).abs() < 1e-9,
            "a full-pass gate accepts nothing wrong here, got {}",
            strict.false_accept
        );
        assert!(
            loose.false_accept > 0.0,
            "serving partial credit must admit real false accepts, got {}",
            loose.false_accept
        );
    }

    /// The error that matters for regressions: rejecting an answer that was already right. It must
    /// be counted at the cheap rung and must rise as the bar rises.
    #[test]
    fn false_rejects_are_counted_when_the_bar_is_above_a_correct_answer() {
        // Correct answers that only score 0.8 — rejected at τ=1.0, accepted at τ=0.8.
        let matrix = vec![
            vec![o(0.8, true, 0.01), o(1.0, true, 0.10)],
            vec![o(0.8, true, 0.01), o(1.0, true, 0.10)],
        ];
        let s = sweep(&matrix, 0.10, 0.05, 1);
        let strict = s.rows.iter().find(|r| r.tau >= 1.0).expect("τ=1.0");
        assert!(
            (strict.false_reject - 1.0).abs() < 1e-9,
            "both correct answers were rejected, got {}",
            strict.false_reject
        );
        let loose = s
            .rows
            .iter()
            .find(|r| (r.tau - 0.8).abs() < 1e-9)
            .expect("τ=0.8");
        assert!(
            (loose.false_reject - 0.0).abs() < 1e-9,
            "at the lower bar nothing correct is rejected, got {}",
            loose.false_reject
        );
    }

    /// Escalation must be billed for the rung it rejected. A sweep that under-bills would make a
    /// stricter threshold look free.
    #[test]
    fn escalation_bills_every_rung_tried() {
        let matrix = vec![vec![o(0.5, false, 0.02), o(1.0, true, 0.20)]];
        let s = sweep(&matrix, 0.10, 0.05, 1);
        let strict = s.rows.iter().find(|r| r.tau >= 1.0).expect("τ=1.0");
        assert!(
            (strict.total_cost_usd - 0.22).abs() < 1e-9,
            "expected $0.22 (both rungs), got {}",
            strict.total_cost_usd
        );
    }

    /// The finding that forced this module to take a signal at all, pinned as a test so a future
    /// change cannot quietly reintroduce a two-point "sweep" and present it as a study.
    ///
    /// Measured on real MBPP data: `gate_score` took only `{0.0, 1.0}` across 40 rung outcomes.
    /// Generated code satisfies all of a task's asserts or none of them, so the fraction the score
    /// type permits does not occur in practice.
    #[test]
    fn a_binary_test_gate_is_reported_as_degenerate_rather_than_swept() {
        let matrix: Vec<Vec<RungOutcome>> = (0..10)
            .map(|i| {
                vec![
                    o(f64::from(u8::from(i % 3 != 0)), i % 3 != 0, 0.01),
                    o(1.0, true, 0.10),
                ]
            })
            .collect();
        let s = sweep(&matrix, 0.10, 0.05, 1);
        assert_eq!(s.signal, Signal::TestOnly, "no judge ran");
        assert_eq!(
            s.distinct_levels, 2,
            "an all-or-nothing gate has exactly two operating points"
        );
        assert!(
            render(&s).contains("degenerate"),
            "a two-point sweep must say so instead of rendering as a completed study"
        );
    }

    /// With a k-sample judge the fused score becomes graded, which is what makes the sweep mean
    /// anything. Measured at k=5 the judge produced {0.0, 0.2, 0.4, 0.6, 1.0}.
    #[test]
    fn a_graded_judge_gives_the_sweep_an_interior() {
        let matrix: Vec<Vec<RungOutcome>> = [0.0, 0.2, 0.4, 0.6, 1.0]
            .iter()
            .enumerate()
            .map(|(i, &j)| vec![oj(1.0, i >= 3, 0.01, j), o(1.0, true, 0.10)])
            .collect();
        let s = sweep(&matrix, 0.10, 0.05, 1);
        assert_eq!(s.signal, Signal::Combined, "a judge ran, so fuse it");
        assert!(
            s.distinct_levels > 2,
            "a graded judge must open an interior, got {} levels",
            s.distinct_levels
        );
        assert!(
            !render(&s).contains("degenerate"),
            "a graded sweep must not be flagged degenerate"
        );
    }

    /// A gate whose score is unrelated to correctness must not read as informative. This is the
    /// guard against reporting a confident bound on noise.
    #[test]
    fn a_gate_with_no_signal_reports_auc_near_a_coin_flip() {
        // Scores alternate independently of the oracle outcome.
        let matrix = vec![
            vec![o(1.0, true, 0.01), o(1.0, true, 0.10)],
            vec![o(1.0, false, 0.01), o(1.0, true, 0.10)],
            vec![o(0.5, true, 0.01), o(1.0, true, 0.10)],
            vec![o(0.5, false, 0.01), o(1.0, true, 0.10)],
        ];
        let s = sweep(&matrix, 0.10, 0.05, 1);
        assert!(
            (s.auc - 0.5).abs() < 1e-9,
            "uninformative scores must give AUC 0.5, got {}",
            s.auc
        );
    }

    /// Calibration and validation must not share items. If they did, the "held-out" rate would be
    /// the in-sample rate wearing a different name — the exact weakness this module exists to fix.
    #[test]
    fn the_conformal_split_is_disjoint() {
        // Odd n so the halves differ in size and an off-by-one would show.
        let matrix: Vec<Vec<RungOutcome>> = (0..9)
            .map(|i| {
                let score = f64::from(i) / 8.0;
                vec![o(score, i % 2 == 0, 0.01), o(1.0, true, 0.10)]
            })
            .collect();
        let pairs: Vec<(f64, bool)> = matrix
            .iter()
            .filter_map(|r| r.first())
            .map(|o| (o.gate_score, o.oracle_correct))
            .collect();
        let calib: Vec<(f64, bool)> = pairs.iter().step_by(2).copied().collect();
        let valid: Vec<(f64, bool)> = pairs.iter().skip(1).step_by(2).copied().collect();
        assert_eq!(
            calib.len() + valid.len(),
            pairs.len(),
            "the split must cover"
        );
        for c in &calib {
            assert!(
                !valid
                    .iter()
                    .any(|v| (v.0 - c.0).abs() < 1e-12 && v.1 == c.1),
                "an item appears in both halves: {c:?}"
            );
        }
        // And the study runs on it without claiming a bound it cannot support.
        let s = sweep(&matrix, 0.10, 0.05, 100);
        assert!(
            !s.conformal.feasible,
            "min_n=100 on 5 calibration items must refuse to certify"
        );
    }
}
