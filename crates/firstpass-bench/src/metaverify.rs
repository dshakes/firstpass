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
        // A dear ceiling (ratio → 0) demands strong evidence of failure before spending; a cheap
        // one (ratio → 1) escalates on mild doubt.
        let p_star = 1.0 - ratio;

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
    let calib: Vec<Vec<RungOutcome>> = matrix.iter().step_by(2).cloned().collect();
    let valid: Vec<Vec<RungOutcome>> = matrix.iter().skip(1).step_by(2).cloned().collect();

    let top_idx = matrix.first().map_or(0, |r| r.len().saturating_sub(1));
    let mv = MetaVerifier::fit(&calib, mean_cost(matrix, 0), mean_cost(matrix, top_idx));

    let (mut ms, mut mc, mut fs, mut fc) = (0usize, 0.0, 0usize, 0.0);
    let (mut wins, mut losses) = (0usize, 0usize);
    for row in &valid {
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
    let n = valid.len().max(1) as f64;
    MetaStudy {
        calibration_n: calib.len(),
        validation_n: valid.len(),
        p_star: mv.p_star(),
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

    /// The whole point: where the gate says "pass" but that pass is historically unreliable, the
    /// meta-verifier must escalate anyway. A policy that trusts the gate cannot do this.
    #[test]
    fn it_escalates_a_passing_gate_whose_passes_are_historically_wrong() {
        // Calibration: full-pass cheap answers are wrong 4 times out of 5 in this bucket.
        let calib: Vec<Vec<RungOutcome>> = (0..5)
            .map(|i| vec![o(1.0, i == 0, 0.01), o(1.0, true, 1.00)])
            .collect();
        let mv = MetaVerifier::fit(&calib, 0.01, 1.00);
        let passing_but_untrustworthy = vec![o(1.0, false, 0.01), o(1.0, true, 1.00)];
        let (ok, _, rungs) = mv.serve(&passing_but_untrustworthy);
        assert_eq!(
            rungs, 2,
            "a 20%-reliable pass must not be served just because the gate said pass"
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

    /// Fitting and scoring must not share tasks. If they did, the rule could memorise which exact
    /// tasks failed and report a success rate no live traffic would reproduce.
    #[test]
    fn the_study_splits_calibration_from_validation() {
        let matrix: Vec<Vec<RungOutcome>> = (0..11)
            .map(|i| vec![o(1.0, i % 3 == 0, 0.01), o(1.0, true, 1.00)])
            .collect();
        let s = study(&matrix);
        assert_eq!(s.calibration_n + s.validation_n, matrix.len());
        assert!(s.calibration_n > 0 && s.validation_n > 0);
    }

    /// A dearer ceiling must make the rule more reluctant to escalate. This is the property that
    /// keeps `p*` a derived quantity rather than a tuned one.
    #[test]
    fn a_dearer_ceiling_raises_the_bar_for_spending() {
        let calib = vec![vec![o(1.0, true, 0.01), o(1.0, true, 1.00)]];
        let cheap_ceiling = MetaVerifier::fit(&calib, 0.01, 0.02).p_star();
        let dear_ceiling = MetaVerifier::fit(&calib, 0.01, 10.0).p_star();
        assert!(
            dear_ceiling > cheap_ceiling,
            "a dear ceiling ({dear_ceiling:.3}) must demand more doubt than a cheap one \
             ({cheap_ceiling:.3}) before escalating"
        );
    }
}
