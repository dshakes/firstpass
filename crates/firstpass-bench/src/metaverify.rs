//! A meta-verifier: deciding when to trust the gate, after AutoMix (arXiv:2310.12963, NeurIPS 2024).
//!
//! # The problem it solves
//!
//! `first-pass` treats the gate as authoritative: pass ⇒ serve, fail ⇒ escalate. That is correct
//! exactly when the gate is right. On MBPP unit tests it essentially always is, which is why the
//! published result shows zero regressions — but it is not a property that survives contact with
//! gates that can err (a judge, a partial test suite, a schema that admits wrong-but-well-formed
//! output). [`crate::sweep`] shows what happens as the gate degrades: false accepts reach the
//! caller, false rejects buy an escalation that was never needed.
//!
//! AutoMix's contribution is to stop treating verification as ground truth and start treating it
//! as **evidence**. Its meta-verifier observes the verifier's own signal, knows that signal is
//! noisy, and decides under that uncertainty. Critically, AutoMix implements the decision layer
//! **without an LLM** — precisely so that verifier error does not compound into the thing meant to
//! correct it. This module keeps that property: the decision rule is arithmetic over calibrated
//! frequencies, with no model call anywhere in it.
//!
//! # The rule
//!
//! Bucket the evidence, estimate `P(cheap answer is actually correct | bucket)` on a calibration
//! split, and escalate exactly when serving is the worse bet:
//!
//! ```text
//! escalate  ⟺  P(correct | evidence) < p*
//! ```
//!
//! `p*` is not hand-tuned. It falls out of the ladder's own measured prices: escalating buys at
//! most `(1 − P)` additional expected successes for a known additional cost, so the threshold that
//! makes the trade break even is a function of the cost ratio between the rungs. A cheap ladder
//! escalates readily; an expensive ceiling makes the gate justify itself.
//!
//! # Why the split matters
//!
//! The frequencies are fit on one half of the matrix and the policy is scored on the other. Fitting
//! and scoring on the same tasks would let the rule memorise which specific tasks the cheap model
//! failed — reporting near-oracle performance that no live traffic would reproduce. This is the
//! same maker ≠ checker discipline the repo applies elsewhere.

use crate::coding_policy::RungOutcome;

/// Number of buckets the gate score is discretised into. Four is deliberate: enough to separate
/// "clean pass", "mostly passing", "mostly failing", and "clean fail", few enough that each bucket
/// still holds a usable count at n ≈ 1000, which is the scale these matrices are measured at.
const SCORE_BUCKETS: usize = 4;

/// A fitted meta-verifier: what the evidence was worth, measured rather than assumed.
#[derive(Debug, Clone)]
pub struct MetaVerifier {
    /// `P(oracle_correct | bucket)` per evidence bucket, from the calibration split.
    /// `None` where the calibration split held no examples of that bucket.
    posterior: Vec<Option<f64>>,
    /// The break-even probability below which escalating is the better bet.
    p_star: f64,
    /// Fallback when a bucket was never observed: the calibration split's base rate. Used rather
    /// than a guess, so an unseen bucket behaves like an average task instead of a confident one.
    base_rate: f64,
}

/// Which bucket a piece of evidence falls in.
///
/// The judge, when present, doubles the space: a gate score means something different when a second
/// opinion agrees with it than when it does not. That disagreement region is exactly where a plain
/// threshold policy is wrong and where AutoMix's meta-verification earns its keep, so it must be
/// addressable by the rule rather than averaged away.
fn bucket(o: &RungOutcome) -> usize {
    let s = o.gate_score.clamp(0.0, 1.0);
    // `min` guards the s == 1.0 edge, which would otherwise index one past the last bucket.
    let b = ((s * SCORE_BUCKETS as f64) as usize).min(SCORE_BUCKETS - 1);
    match o.judge_score {
        // Judge agrees with the tests (both high or both low) vs disagrees.
        Some(j) => {
            let judge_high = j >= 0.5;
            let gate_high = s >= 0.5;
            if judge_high == gate_high {
                b
            } else {
                b + SCORE_BUCKETS
            }
        }
        None => b,
    }
}

/// Total bucket count: score buckets, doubled for the judge agree/disagree split.
const BUCKETS: usize = SCORE_BUCKETS * 2;

impl MetaVerifier {
    /// Fit on a calibration slice of the matrix.
    ///
    /// `cheap_cost` and `top_cost` are the ladder's measured mean prices; they set `p*`. When the
    /// ladder is degenerate (the top rung is not dearer) escalation is free, so `p*` saturates at 1
    /// and the rule escalates whenever there is any doubt at all.
    #[must_use]
    pub fn fit(calibration: &[Vec<RungOutcome>], cheap_cost: f64, top_cost: f64) -> Self {
        let mut hit = [0usize; BUCKETS];
        let mut seen = [0usize; BUCKETS];
        let mut correct = 0usize;
        for row in calibration {
            if let Some(c) = row.first() {
                let b = bucket(c);
                seen[b] += 1;
                if c.oracle_correct {
                    hit[b] += 1;
                    correct += 1;
                }
            }
        }
        let posterior = (0..BUCKETS)
            .map(|b| (seen[b] > 0).then(|| hit[b] as f64 / seen[b] as f64))
            .collect();
        let base_rate = if calibration.is_empty() {
            0.0
        } else {
            correct as f64 / calibration.len() as f64
        };

        // Break-even: escalating costs `top_cost` extra and can recover at most the probability
        // mass that the cheap answer is wrong. Expressed as a ratio so it is scale-free — the same
        // rule holds whether the ladder is measured in cents or dollars.
        let ratio = if top_cost > 0.0 {
            (cheap_cost / top_cost).clamp(0.0, 1.0)
        } else {
            1.0
        };
        // The threshold IS the cost ratio, and the derivation is worth writing out because an
        // earlier version of this line had it inverted:
        //
        //     serve cheap  ⟺  c₀ + (1−p)·c₁ < c₁  ⟺  c₀ < p·c₁  ⟺  p > c₀/c₁ = ratio
        //
        // so `serve iff confidence ≥ ratio`, and escalation happens below it. A DEAR ceiling
        // (ratio → 0) therefore escalates rarely — it demands near-certainty that the cheap answer
        // is wrong before paying — while a cheap ceiling (ratio → 1) escalates on mild doubt.
        // Writing `1.0 - ratio` inverts exactly this and makes an expensive ladder escalate almost
        // every request.
        let p_star = ratio;

        Self {
            posterior,
            p_star,
            base_rate,
        }
    }

    /// `P(correct)` this verifier assigns to one measured outcome.
    #[must_use]
    pub fn confidence(&self, o: &RungOutcome) -> f64 {
        self.posterior
            .get(bucket(o))
            .copied()
            .flatten()
            .unwrap_or(self.base_rate)
    }

    /// The break-even probability this verifier escalates below.
    #[must_use]
    pub fn p_star(&self) -> f64 {
        self.p_star
    }

    /// Serve the first rung whose evidence clears the break-even bar, paying for every rung tried.
    ///
    /// Returns `(served_correct, spent, rungs_paid)`, matching the other policies so it can be
    /// dropped into the same replay and paired comparison.
    #[must_use]
    pub fn serve(&self, row: &[RungOutcome]) -> (bool, f64, usize) {
        let mut spent = 0.0;
        for (i, o) in row.iter().enumerate() {
            spent += o.cost_usd;
            if self.confidence(o) >= self.p_star {
                return (o.oracle_correct, spent, i + 1);
            }
        }
        (
            row.last().is_some_and(|o| o.oracle_correct),
            spent,
            row.len(),
        )
    }
}

/// Mean cost at one rung across the matrix.
fn mean_cost(matrix: &[Vec<RungOutcome>], idx: usize) -> f64 {
    let xs: Vec<f64> = matrix
        .iter()
        .filter_map(|r| r.get(idx))
        .map(|o| o.cost_usd)
        .collect();
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// What the meta-verifier achieved on held-out tasks, against plain first-pass on the same tasks.
#[derive(Debug, Clone)]
pub struct MetaStudy {
    /// Tasks the rule was fitted on.
    pub calibration_n: usize,
    /// Tasks it was scored on — disjoint from the above.
    pub validation_n: usize,
    /// Break-even probability derived from the ladder's prices.
    pub p_star: f64,
    /// Meta-verified policy: success rate on held-out tasks.
    pub meta_success: f64,
    /// Meta-verified policy: total USD on held-out tasks.
    pub meta_cost_usd: f64,
    /// Plain first-pass on the same held-out tasks.
    pub first_pass_success: f64,
    /// Plain first-pass cost on the same held-out tasks.
    pub first_pass_cost_usd: f64,
    /// Tasks the meta-verifier got right that first-pass got wrong.
    pub wins: usize,
    /// Tasks it got wrong that first-pass got right — the number that decides whether to ship it.
    pub losses: usize,
}

/// Fit on half the matrix, score on the other half, and compare against plain first-pass.
///
/// The split is by index parity: deterministic, so the result reproduces without a recorded seed.
#[must_use]
pub fn study(matrix: &[Vec<RungOutcome>]) -> MetaStudy {
    let a: Vec<Vec<RungOutcome>> = matrix.iter().step_by(2).cloned().collect();
    let b: Vec<Vec<RungOutcome>> = matrix.iter().skip(1).step_by(2).cloned().collect();

    let top_idx = matrix.first().map_or(0, |r| r.len().saturating_sub(1));
    let (c0, c1) = (mean_cost(matrix, 0), mean_cost(matrix, top_idx));

    // Cross-fitted, for the same reason `costaware::study` is: a single fixed split scores only
    // half the matrix, and on real data the halves are not equivalent — the expensive escalating
    // tasks concentrated in one of them, which made the other look cheap and manufactured a
    // saving that was an artifact of the split. Each fold is scored by a verifier fitted on the
    // other, so every task is scored exactly once and still out-of-sample.
    let fit_for_b = MetaVerifier::fit(&a, c0, c1);
    let fit_for_a = MetaVerifier::fit(&b, c0, c1);
    let scored: Vec<(&Vec<RungOutcome>, &MetaVerifier)> = b
        .iter()
        .map(|r| (r, &fit_for_b))
        .chain(a.iter().map(|r| (r, &fit_for_a)))
        .collect();
    let calibration_n = a.len();

    let (mut ms, mut mc, mut fs, mut fc) = (0usize, 0.0, 0usize, 0.0);
    let (mut wins, mut losses) = (0usize, 0usize);
    for (row, mv) in &scored {
        let (m_ok, m_cost, _) = mv.serve(row);
        let (f_ok, f_cost, _) = first_pass(row);
        if m_ok {
            ms += 1;
        }
        if f_ok {
            fs += 1;
        }
        mc += m_cost;
        fc += f_cost;
        match (m_ok, f_ok) {
            (true, false) => wins += 1,
            (false, true) => losses += 1,
            _ => {}
        }
    }
    let n = scored.len().max(1) as f64;
    MetaStudy {
        calibration_n,
        validation_n: scored.len(),
        p_star: fit_for_b.p_star(),
        meta_success: ms as f64 / n,
        meta_cost_usd: mc,
        first_pass_success: fs as f64 / n,
        first_pass_cost_usd: fc,
        wins,
        losses,
    }
}

/// Plain first-pass, duplicated here so the comparison is against the exact policy shape this
/// module replaces, evaluated on identical rows.
fn first_pass(row: &[RungOutcome]) -> (bool, f64, usize) {
    let mut spent = 0.0;
    for (i, o) in row.iter().enumerate() {
        spent += o.cost_usd;
        if o.gate_full_pass {
            return (o.oracle_correct, spent, i + 1);
        }
    }
    (
        row.last().is_some_and(|o| o.oracle_correct),
        spent,
        row.len(),
    )
}

/// Render the comparison as Markdown.
#[must_use]
pub fn render(s: &MetaStudy) -> String {
    let mut out = String::new();
    out.push_str("## Meta-verifier (AutoMix, arXiv:2310.12963) vs plain first-pass\n\n");
    out.push_str(&format!(
        "Fitted on {} tasks, scored on {} held-out tasks. Break-even p* = {:.3}.\n\n",
        s.calibration_n, s.validation_n, s.p_star
    ));
    out.push_str("| policy | success | total $ |\n|---|---|---|\n");
    out.push_str(&format!(
        "| first-pass | {:.4} | ${:.4} |\n",
        s.first_pass_success, s.first_pass_cost_usd
    ));
    out.push_str(&format!(
        "| meta-verified | {:.4} | ${:.4} |\n",
        s.meta_success, s.meta_cost_usd
    ));
    out.push_str(&format!(
        "\nHead to head on held-out tasks: **{} won, {} lost**.\n",
        s.wins, s.losses
    ));
    if s.meta_success <= s.first_pass_success && s.meta_cost_usd >= s.first_pass_cost_usd {
        out.push_str(
            "\nOn this matrix the meta-verifier does **not** beat treating the gate as \
             authoritative — neither better nor cheaper. That is the expected result where the \
             gate is a near-perfect oracle (unit tests), and it is reported rather than buried: \
             the rule is worth shipping only on gates that actually err.\n",
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

    fn oj(score: f64, oracle: bool, cost: f64, judge: f64) -> RungOutcome {
        RungOutcome {
            judge_score: Some(judge),
            ..o(score, oracle, cost)
        }
    }

    /// The whole point: where the gate says "pass" but that pass is historically unreliable AND
    /// the next rung is affordable, the meta-verifier escalates anyway. A policy that treats the
    /// gate as authoritative cannot do this.
    ///
    /// The ladder here is deliberately shallow (2x). `p*` is a purely **cost-minimising**
    /// threshold — it weighs the price of escalating against the probability it buys a correct
    /// answer, and encodes no independent preference for quality. So on a 100x ladder the same
    /// 20%-reliable pass is correctly SERVED: $0.05 per success against $1.01 for escalating.
    /// An earlier version of this test asserted escalation on exactly that 100x ladder, which was
    /// only satisfiable because `p*` was inverted at the time.
    #[test]
    fn it_escalates_an_unreliable_pass_when_the_next_rung_is_affordable() {
        // Calibration: full-pass cheap answers are wrong 4 times out of 5 in this bucket.
        let calib: Vec<Vec<RungOutcome>> = (0..5)
            .map(|i| vec![o(1.0, i == 0, 0.01), o(1.0, true, 0.02)])
            .collect();
        let mv = MetaVerifier::fit(&calib, 0.01, 0.02);
        // p* = 0.01/0.02 = 0.5, and this bucket's measured reliability is 0.2 — below it.
        assert!((mv.p_star() - 0.5).abs() < 1e-9);
        let passing_but_untrustworthy = vec![o(1.0, false, 0.01), o(1.0, true, 0.02)];
        let (ok, _, rungs) = mv.serve(&passing_but_untrustworthy);
        assert_eq!(
            rungs, 2,
            "a 20%-reliable pass must not be served when the next rung costs only 2x"
        );
        assert!(ok, "escalating rescued the answer");
    }

    /// The converse, and the guard against a rule that just escalates everything: where the gate's
    /// passes are historically sound, it serves the cheap rung and spends nothing extra.
    #[test]
    fn it_serves_the_cheap_rung_when_the_gate_has_earned_it() {
        let calib: Vec<Vec<RungOutcome>> = (0..10)
            .map(|_| vec![o(1.0, true, 0.01), o(1.0, true, 1.00)])
            .collect();
        let mv = MetaVerifier::fit(&calib, 0.01, 1.00);
        let (_, spent, rungs) = mv.serve(&[o(1.0, true, 0.01), o(1.0, true, 1.00)]);
        assert_eq!(rungs, 1, "a reliable pass must be served");
        assert!(
            (spent - 0.01).abs() < 1e-9,
            "and must not pay for the top rung"
        );
    }

    /// The judge's role is to split a bucket, not to be averaged into it. Two candidates with the
    /// same gate score but opposite judge opinions must be distinguishable by the rule.
    #[test]
    fn agreement_and_disagreement_land_in_different_buckets() {
        let agree = oj(1.0, true, 0.01, 0.9);
        let disagree = oj(1.0, true, 0.01, 0.1);
        assert_ne!(
            bucket(&agree),
            bucket(&disagree),
            "a judge that contradicts the tests must not be collapsed into agreement"
        );
    }

    /// A bucket the calibration split never saw must fall back to the base rate rather than to a
    /// confident number, or an unseen region of the score space would be served on no evidence.
    #[test]
    fn an_unseen_bucket_falls_back_to_the_base_rate() {
        let calib = vec![vec![o(1.0, true, 0.01), o(1.0, true, 1.00)]];
        let mv = MetaVerifier::fit(&calib, 0.01, 1.00);
        let never_seen = o(0.1, false, 0.01);
        assert!(
            (mv.confidence(&never_seen) - mv.base_rate).abs() < 1e-9,
            "unseen evidence must not get a confident posterior"
        );
    }

    /// Cross-fitting scores every task exactly once, each by a verifier that never saw it — so
    /// the scored set is the whole matrix while remaining out-of-sample. A single fixed split
    /// scores only half, and on real data the halves are not equivalent (see `costaware::study`).
    #[test]
    fn cross_fitting_scores_every_task_out_of_sample() {
        let matrix: Vec<Vec<RungOutcome>> = (0..11)
            .map(|i| vec![o(1.0, i % 3 == 0, 0.01), o(1.0, true, 1.00)])
            .collect();
        let s = study(&matrix);
        assert_eq!(
            s.validation_n,
            matrix.len(),
            "every task must be scored, not just one half"
        );
        assert!(s.calibration_n > 0 && s.calibration_n < matrix.len());
    }

    /// A dearer ceiling must make the rule MORE reluctant to escalate.
    ///
    /// Escalation happens below `p*`, so "more reluctant" means a *lower* threshold. An earlier
    /// version of this test asserted the opposite and so certified an inverted `p*` — the failure
    /// mode where a test encodes the same mistake as the code it guards. The numeric expectations
    /// below are pinned to the derivation (`p* = c₀/c₁`) rather than to whatever the code returns.
    #[test]
    fn a_dearer_ceiling_makes_escalation_rarer() {
        let calib = vec![vec![o(1.0, true, 0.01), o(1.0, true, 1.00)]];
        let cheap_ceiling = MetaVerifier::fit(&calib, 0.01, 0.02).p_star();
        let dear_ceiling = MetaVerifier::fit(&calib, 0.01, 10.0).p_star();

        assert!(
            (cheap_ceiling - 0.5).abs() < 1e-9,
            "c₀/c₁ = 0.01/0.02 = 0.5, got {cheap_ceiling}"
        );
        assert!(
            (dear_ceiling - 0.001).abs() < 1e-9,
            "c₀/c₁ = 0.01/10.0 = 0.001, got {dear_ceiling}"
        );
        assert!(
            dear_ceiling < cheap_ceiling,
            "escalation happens below p*, so a dear ceiling ({dear_ceiling:.4}) must sit BELOW a \
             cheap one ({cheap_ceiling:.4}) — it should escalate less, not more"
        );
    }

    /// The behavioural consequence, asserted on `serve` rather than on the threshold, so the rule
    /// cannot be inverted again without this failing.
    #[test]
    fn an_expensive_ceiling_does_not_escalate_a_merely_uncertain_answer() {
        // Calibration: full-pass answers are right 70% of the time in this bucket.
        let calib: Vec<Vec<RungOutcome>> = (0..10)
            .map(|i| vec![o(1.0, i < 7, 0.01), o(1.0, true, 10.0)])
            .collect();

        // Dear ceiling (ratio 0.001): 70% confidence is far above it — serve, do not spend.
        let dear = MetaVerifier::fit(&calib, 0.01, 10.0);
        let (_, spent, rungs) = dear.serve(&[o(1.0, true, 0.01), o(1.0, true, 10.0)]);
        assert_eq!(
            rungs, 1,
            "a 70%-reliable answer must not buy a 1000x ceiling"
        );
        assert!((spent - 0.01).abs() < 1e-9);

        // Cheap ceiling (ratio 0.9): 70% is below it — escalating is worth it.
        let cheap = MetaVerifier::fit(&calib, 0.009, 0.01);
        let (_, _, rungs2) = cheap.serve(&[o(1.0, true, 0.009), o(1.0, true, 0.01)]);
        assert_eq!(
            rungs2, 2,
            "when the ceiling is nearly free, mild doubt should escalate"
        );
    }
}
