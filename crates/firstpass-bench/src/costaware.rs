//! Cost-aware start-rung selection: spend the cheap attempt only where it is likely to pay.
//!
//! # The defect this addresses
//!
//! A cascade pays `c₀` on **every** request. That is defensible when the cheap attempt is nearly
//! free, and indefensible when it is not — and measurement says it is not, because the cost of the
//! cheap attempt is *correlated with the probability that it fails*.
//!
//! Measured on 114 MBPP tasks (haiku → sonnet):
//!
//! | | cheap rung | top rung |
//! |---|---|---|
//! | tasks whose gate **passed** | $0.00869 | — |
//! | tasks that **escalated** | $0.01881 (2.2×) | $0.02703 (2.1× the average) |
//!
//! Hard tasks are long tasks; long tasks are expensive; long tasks are the ones that escalate. So
//! escalation never pays the *average* price for the top rung — here it paid 2.06× average. The
//! usual cascade cost model, `c₀ + e·c̄₁`, is optimistic by exactly that multiplier, which is why
//! first-pass measured $0.01456/task where the naive model predicted $0.01236 and always-top costs
//! $0.01312. The cascade lost on a ladder its own cost model said it would win.
//!
//! This is a plausible partial explanation for RouterBench's headline finding that cascading
//! routers did not beat its parameter-free Zero Router: a cost model that ignores the correlation
//! between difficulty and length overstates every cascade's savings.
//!
//! # The rule
//!
//! Treat the starting rung as a decision rather than a constant. With `p = P(rung 0 clears the
//! gate)`:
//!
//! ```text
//! start cheap:  E[cost] = c₀ + (1 − p)·c₁
//! start at top: E[cost] = c₁
//! ⇒ start cheap  ⟺  c₀ < p · c₁
//! ```
//!
//! Closed-form, per request, and every term is observable *before* generating: `c₀` and `c₁` are
//! predictable from prompt length, and `p` is what a gate-pass predictor already estimates. It
//! degrades gracefully — with `p → 1` it is exactly first-pass; with `c₀ → 0` it is always
//! first-pass, which is why the rule has been invisible in settings where the draft model is free.
//!
//! # Honesty of the evaluation
//!
//! Two variants are reported, and the gap between them is the point:
//!
//! - **oracle-p** uses the true outcome as `p`. It is *cheating* and exists only as an upper bound
//!   on what any predictor could buy. A gain that is small even here is not worth pursuing.
//! - **learned-p** buckets by cheap-rung cost, estimates the pass rate per bucket on a calibration
//!   split, and is scored on the disjoint other half. This is the number that means something.
//!
//! Quality is never traded away silently: the rule only ever moves *where a request starts*. A
//! request that starts at the top serves the top rung's answer, which is what always-top would
//! have served anyway.

use crate::coding_policy::RungOutcome;

/// Buckets the cheap-rung cost is discretised into for the learned pass-rate estimate. Quartiles:
/// enough resolution to separate short from long requests, coarse enough that each bucket keeps a
/// usable count at the n these matrices are measured at.
const COST_BUCKETS: usize = 4;

/// A fitted estimate of `P(rung 0 clears the gate | cheap-rung cost)`.
#[derive(Debug, Clone)]
pub struct PassPredictor {
    /// Upper cost edge of each bucket (last is infinity).
    edges: Vec<f64>,
    /// Empirical pass rate per bucket, from the calibration split.
    rate: Vec<Option<f64>>,
    /// Calibration-split base rate, used where a bucket held no examples.
    base: f64,
}

impl PassPredictor {
    /// Fit on a calibration slice. Deterministic: quartile edges come from the sorted costs.
    #[must_use]
    pub fn fit(calibration: &[Vec<RungOutcome>]) -> Self {
        let mut costs: Vec<f64> = calibration
            .iter()
            .filter_map(|r| r.first())
            .map(|o| o.cost_usd)
            .collect();
        costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let edges: Vec<f64> = (1..COST_BUCKETS)
            .map(|i| {
                let idx = i * costs.len() / COST_BUCKETS;
                costs.get(idx).copied().unwrap_or(f64::INFINITY)
            })
            .chain(std::iter::once(f64::INFINITY))
            .collect();

        let mut hit = [0usize; COST_BUCKETS];
        let mut seen = [0usize; COST_BUCKETS];
        let mut passed = 0usize;
        for row in calibration {
            if let Some(c) = row.first() {
                let b = Self::bucket_of(&edges, c.cost_usd);
                seen[b] += 1;
                if c.gate_full_pass {
                    hit[b] += 1;
                    passed += 1;
                }
            }
        }
        Self {
            rate: (0..COST_BUCKETS)
                .map(|b| (seen[b] > 0).then(|| hit[b] as f64 / seen[b] as f64))
                .collect(),
            edges,
            base: if calibration.is_empty() {
                0.0
            } else {
                passed as f64 / calibration.len() as f64
            },
        }
    }

    fn bucket_of(edges: &[f64], cost: f64) -> usize {
        edges
            .iter()
            .position(|e| cost <= *e)
            .unwrap_or(COST_BUCKETS - 1)
            .min(COST_BUCKETS - 1)
    }

    /// `P(gate passes at rung 0)` for a request whose cheap attempt would cost `cost`.
    #[must_use]
    pub fn p(&self, cost: f64) -> f64 {
        let b = Self::bucket_of(&self.edges, cost);
        self.rate.get(b).copied().flatten().unwrap_or(self.base)
    }
}

/// Serve under the cost-aware rule. `p` is the caller's estimate of `P(rung 0 clears the gate)`.
///
/// Returns `(served_correct, spent, rungs_paid)`, matching the other policies so it drops into the
/// same comparison.
#[must_use]
pub fn serve(row: &[RungOutcome], p: f64) -> (bool, f64, usize) {
    let (Some(c0), Some(c1)) = (row.first(), row.last()) else {
        return (false, 0.0, 0);
    };
    // The decision. Skipping the cheap attempt is only ever right when its cost exceeds what it
    // buys in expectation.
    if row.len() > 1 && c0.cost_usd >= p * c1.cost_usd {
        return (c1.oracle_correct, c1.cost_usd, 1);
    }
    // Otherwise: ordinary first-pass from rung 0.
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

/// One policy's totals on the held-out split.
#[derive(Debug, Clone)]
pub struct Arm {
    /// Policy name.
    pub name: &'static str,
    /// Fraction of held-out tasks answered correctly.
    pub success: f64,
    /// Mean USD per task.
    pub cost_per_task: f64,
    /// USD per successfully-solved task.
    pub usd_per_success: f64,
    /// Fraction of tasks where the cheap attempt was skipped entirely.
    pub skipped_cheap: f64,
}

/// The comparison.
#[derive(Debug, Clone)]
pub struct CostAwareStudy {
    /// Tasks the predictor was fitted on.
    pub calibration_n: usize,
    /// Held-out tasks every arm below is scored on.
    pub validation_n: usize,
    /// Measured adverse-selection multiplier: mean top-rung cost among escalating tasks, over the
    /// mean top-rung cost across all tasks. 1.0 would mean escalation is cost-neutral.
    pub adverse_selection: f64,
    /// Arms, in table order.
    pub arms: Vec<Arm>,
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// One scored task: the measured row, paired with the `p` predicted for it by a model that never
/// saw it.
type Scored = (Vec<RungOutcome>, f64);

/// Plain first-pass: always open on rung 0, gate, escalate on failure, pay for every rung tried.
/// Written out rather than expressed as a special case of [`serve`], so the baseline cannot drift
/// into the policy it is supposed to be measuring.
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

/// Score one policy over every cross-fitted task.
fn arm(
    name: &'static str,
    valid: &[Scored],
    serve_fn: impl Fn(&[RungOutcome], f64) -> (bool, f64, usize, bool),
) -> Arm {
    let (mut ok, mut cost, mut skipped) = (0usize, Vec::new(), 0usize);
    for (row, p) in valid {
        let (correct, spent, _, skip) = serve_fn(row, *p);
        if correct {
            ok += 1;
        }
        if skip {
            skipped += 1;
        }
        cost.push(spent);
    }
    let n = valid.len().max(1) as f64;
    let per_task = mean(&cost);
    Arm {
        name,
        success: ok as f64 / n,
        cost_per_task: per_task,
        usd_per_success: if ok == 0 {
            f64::INFINITY
        } else {
            cost.iter().sum::<f64>() / ok as f64
        },
        skipped_cheap: skipped as f64 / n,
    }
}

/// Fit on half the matrix and score every arm on the other half — **both ways round**, averaging
/// the two.
///
/// A single fixed split is not safe here, and this was caught by measurement rather than by
/// reasoning. On the first 121-task matrix the escalating (and therefore expensive) tasks landed
/// disproportionately in one half: `always-top` cost the same in both ($0.01316 vs $0.01312 on the
/// full matrix, since it never escalates) while `first-pass` differed by 30% ($0.01106 vs
/// $0.01456). Reporting the easy half alone would have manufactured an 18% saving out of nothing.
///
/// Cross-fitting removes that asymmetry: every task is scored exactly once, by a predictor that
/// never saw it. The estimate uses the whole matrix while remaining honestly out-of-sample.
#[must_use]
pub fn study(matrix: &[Vec<RungOutcome>]) -> CostAwareStudy {
    let a: Vec<Vec<RungOutcome>> = matrix.iter().step_by(2).cloned().collect();
    let b: Vec<Vec<RungOutcome>> = matrix.iter().skip(1).step_by(2).cloned().collect();

    // Each fold is scored by the predictor fitted on the *other* fold. Attaching `p` to the row
    // here — rather than consulting a predictor later — is what guarantees no arm can accidentally
    // score a task with a model that trained on it.
    let pred_for_b = PassPredictor::fit(&a);
    let pred_for_a = PassPredictor::fit(&b);
    let valid: Vec<Scored> = b
        .iter()
        .map(|r| {
            let p = r.first().map_or(1.0, |o| pred_for_b.p(o.cost_usd));
            (r.clone(), p)
        })
        .chain(a.iter().map(|r| {
            let p = r.first().map_or(1.0, |o| pred_for_a.p(o.cost_usd));
            (r.clone(), p)
        }))
        .collect();
    let calib_n = a.len();

    // The multiplier that breaks the naive cost model, measured rather than assumed.
    let top_all: Vec<f64> = matrix
        .iter()
        .filter_map(|r| r.last())
        .map(|o| o.cost_usd)
        .collect();
    let top_esc: Vec<f64> = matrix
        .iter()
        .filter(|r| r.first().is_some_and(|o| !o.gate_full_pass))
        .filter_map(|r| r.last())
        .map(|o| o.cost_usd)
        .collect();
    let base = mean(&top_all);
    let adverse_selection = if base > 0.0 {
        mean(&top_esc) / base
    } else {
        1.0
    };

    /// Whether the rule skips the cheap rung for this row at this `p` — the decision itself.
    fn skips(row: &[RungOutcome], p: f64) -> bool {
        row.first()
            .zip(row.last())
            .is_some_and(|(a, b)| row.len() > 1 && a.cost_usd >= p * b.cost_usd)
    }

    let arms = vec![
        arm("always-top", &valid, |row, _p| {
            row.last().map_or((false, 0.0, 0, false), |o| {
                (o.oracle_correct, o.cost_usd, 1, true)
            })
        }),
        arm("first-pass", &valid, |row, _p| {
            // NOT `serve(row, 1.0)`. At p = 1 the rule still skips the cheap rung whenever
            // c₀ >= c₁, which on this data is not rare — adverse selection means some cheap
            // attempts really do cost more than the top rung. Routing the baseline through the
            // rule under test would compare the rule against a slightly cost-aware version of
            // itself and understate what plain first-pass actually costs.
            let (c, s, r) = first_pass(row);
            (c, s, r, false)
        }),
        arm("cost-aware (learned p)", &valid, |row, p| {
            let (c, s, r) = serve(row, p);
            (c, s, r, skips(row, p))
        }),
        arm("cost-aware (ORACLE p — cheats)", &valid, |row, _p| {
            // Upper bound only: uses the true outcome as the prediction.
            let p = row
                .first()
                .map_or(1.0, |o| f64::from(u8::from(o.gate_full_pass)));
            let (c, s, r) = serve(row, p);
            (c, s, r, skips(row, p))
        }),
    ];

    CostAwareStudy {
        calibration_n: calib_n,
        validation_n: valid.len(),
        adverse_selection,
        arms,
    }
}

/// Render as Markdown.
#[must_use]
pub fn render(s: &CostAwareStudy) -> String {
    let mut out = String::new();
    out.push_str("## Cost-aware start rung — `start cheap ⟺ c₀ < p·c₁`\n\n");
    out.push_str(&format!(
        "Fitted on {} tasks, scored on {} held-out. Adverse-selection multiplier **{:.2}×** \
         (mean top-rung cost among escalating tasks ÷ mean across all tasks; 1.00 = cost-neutral \
         escalation, which is what the usual cascade cost model assumes).\n\n",
        s.calibration_n, s.validation_n, s.adverse_selection
    ));
    out.push_str(
        "| policy | success | $/task | $/success | skipped cheap |\n|---|---|---|---|---|\n",
    );
    for a in &s.arms {
        out.push_str(&format!(
            "| {} | {:.4} | ${:.5} | ${:.5} | {:.0}% |\n",
            a.name,
            a.success,
            a.cost_per_task,
            a.usd_per_success,
            a.skipped_cheap * 100.0
        ));
    }

    let top = s.arms.iter().find(|a| a.name == "always-top");
    let learned = s.arms.iter().find(|a| a.name.contains("learned"));
    if let (Some(t), Some(l)) = (top, learned) {
        let delta = (t.cost_per_task - l.cost_per_task) / t.cost_per_task.max(f64::EPSILON);
        out.push_str(&format!(
            "\n**Learned rule vs always-top: {:+.1}% cost/task at {:+.4} success.**\n",
            -delta * 100.0,
            l.success - t.success
        ));
        if delta <= 0.0 {
            out.push_str(
                "The rule does not beat always-top on this matrix. Reported as measured — a \
                 start-rung rule cannot rescue a ladder whose gradient is too shallow to pay for \
                 verification at all.\n",
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(gate: bool, oracle: bool, cost: f64) -> RungOutcome {
        RungOutcome {
            gate_score: f64::from(u8::from(gate)),
            gate_full_pass: gate,
            oracle_correct: oracle,
            cost_usd: cost,
            judge_score: None,
        }
    }

    /// The rule's whole purpose: when the cheap attempt costs more than it can buy, skip it and
    /// pay for the top rung once instead of paying for both.
    #[test]
    fn an_expensive_hopeless_cheap_attempt_is_skipped() {
        // c0 = 0.09, c1 = 0.10, p = 0.1 ⇒ 0.09 >= 0.1*0.10 ⇒ skip.
        let row = vec![o(false, false, 0.09), o(true, true, 0.10)];
        let (correct, spent, rungs) = serve(&row, 0.1);
        assert!(correct, "the top rung answers it");
        assert!(
            (spent - 0.10).abs() < 1e-9,
            "must pay for the top rung ONLY, not both; got {spent}"
        );
        assert_eq!(rungs, 1);
    }

    /// The converse: a cheap attempt that is likely to work is still taken, so the rule never
    /// degenerates into always-top.
    #[test]
    fn a_promising_cheap_attempt_is_still_taken() {
        // c0 = 0.01, c1 = 0.10, p = 0.9 ⇒ 0.01 < 0.09 ⇒ start cheap.
        let row = vec![o(true, true, 0.01), o(true, true, 0.10)];
        let (_, spent, rungs) = serve(&row, 0.9);
        assert_eq!(rungs, 1, "served from the cheap rung");
        assert!((spent - 0.01).abs() < 1e-9, "and paid only the cheap price");
    }

    /// With `p = 1` the rule matches first-pass **only while the cheap rung is actually cheaper**.
    #[test]
    fn p_equals_one_matches_first_pass_when_the_cheap_rung_is_cheaper() {
        let row = vec![o(false, false, 0.01), o(true, true, 0.10)];
        let (correct, spent, rungs) = serve(&row, 1.0);
        assert_eq!(rungs, 2, "escalates like first-pass");
        assert!(
            (spent - 0.11).abs() < 1e-9,
            "and bills both rungs like first-pass"
        );
        assert!(correct);
    }

    /// ...and diverges from it when the cheap attempt costs at least as much as the top rung,
    /// which adverse selection makes a real case rather than a corner one.
    ///
    /// This is why the study's `first-pass` arm calls [`first_pass`] directly. Using
    /// `serve(row, 1.0)` as the baseline silently made it cost-aware, so the rule was being
    /// compared against a weakened copy of itself — measured, not hypothesised: it moved the
    /// reported baseline from $0.01227 to a materially different figure on real data.
    #[test]
    fn the_baseline_is_not_the_rule_in_disguise() {
        // A long, hard request: the cheap attempt costs MORE than the top rung and still fails.
        let row = vec![o(false, false, 0.12), o(true, true, 0.10)];

        let (_, rule_spent, rule_rungs) = serve(&row, 1.0);
        assert_eq!(rule_rungs, 1, "the rule skips a cheap rung that costs more");
        assert!((rule_spent - 0.10).abs() < 1e-9);

        let (_, fp_spent, fp_rungs) = first_pass(&row);
        assert_eq!(fp_rungs, 2, "true first-pass always opens on rung 0");
        assert!(
            (fp_spent - 0.22).abs() < 1e-9,
            "and pays for both rungs: expected $0.22, got ${fp_spent}"
        );
        assert!(
            fp_spent > rule_spent,
            "the baseline must be the more expensive, unmodified policy"
        );
    }

    /// The predictor must actually learn that expensive cheap-attempts fail, or the rule has no
    /// signal to act on. This is the assumption the whole module rests on.
    #[test]
    fn the_predictor_learns_that_costly_cheap_attempts_fail() {
        // Cheap+passing vs expensive+failing, the shape measured on MBPP.
        let calib: Vec<Vec<RungOutcome>> = (0..40)
            .map(|i| {
                if i % 2 == 0 {
                    vec![o(true, true, 0.005), o(true, true, 0.02)]
                } else {
                    vec![o(false, false, 0.030), o(true, true, 0.05)]
                }
            })
            .collect();
        let pred = PassPredictor::fit(&calib);
        assert!(
            pred.p(0.005) > pred.p(0.030),
            "a cheap request must be predicted likelier to pass than an expensive one: {} vs {}",
            pred.p(0.005),
            pred.p(0.030)
        );
    }

    /// Quality must never be traded away silently: skipping the cheap rung serves the top rung's
    /// answer, which is exactly what always-top would have served.
    #[test]
    fn skipping_the_cheap_rung_serves_what_always_top_would_have() {
        let row = vec![o(false, false, 0.09), o(false, true, 0.10)];
        let (correct, _, _) = serve(&row, 0.0);
        assert!(
            correct,
            "the served answer is the top rung's, gate verdict notwithstanding"
        );
    }

    /// Cross-fitting scores **every** task, each by a predictor that never saw it — so the scored
    /// set is the whole matrix while still being out-of-sample. A single fixed split would score
    /// only half, which on this data meant scoring only the *easy* half (see [`study`]).
    #[test]
    fn cross_fitting_scores_every_task_exactly_once() {
        let matrix: Vec<Vec<RungOutcome>> = (0..21)
            .map(|i| vec![o(i % 3 != 0, i % 3 != 0, 0.01), o(true, true, 0.10)])
            .collect();
        let s = study(&matrix);
        assert_eq!(
            s.validation_n,
            matrix.len(),
            "every task must be scored, not just one half"
        );
        assert!(s.calibration_n > 0 && s.calibration_n < matrix.len());
        assert!(
            s.arms.iter().any(|a| a.name == "first-pass"),
            "first-pass must be present as the thing to beat"
        );
    }

    /// The bug cross-fitting exists to prevent, pinned. A matrix whose expensive escalating tasks
    /// sit entirely on even indices makes the odd half look cheap; scoring only that half would
    /// report a saving that is an artifact of the split. Under cross-fitting both arms see every
    /// task, so `first-pass` costs the same as computing it directly over the whole matrix.
    #[test]
    fn a_lopsided_split_cannot_manufacture_a_saving() {
        // Even indices: expensive and gate-failing. Odd: cheap and passing.
        let matrix: Vec<Vec<RungOutcome>> = (0..20)
            .map(|i| {
                if i % 2 == 0 {
                    vec![o(false, false, 0.05), o(true, true, 0.10)]
                } else {
                    vec![o(true, true, 0.01), o(true, true, 0.10)]
                }
            })
            .collect();
        let s = study(&matrix);
        let fp = s
            .arms
            .iter()
            .find(|a| a.name == "first-pass")
            .expect("first-pass arm");
        // Ground truth over the WHOLE matrix: half cost 0.05+0.10, half cost 0.01.
        let expected = (10.0 * 0.15 + 10.0 * 0.01) / 20.0;
        assert!(
            (fp.cost_per_task - expected).abs() < 1e-9,
            "first-pass must be scored over every task (expected ${expected:.5}, got ${:.5}) — \
             a half-matrix score here would read ${:.5}",
            fp.cost_per_task,
            0.01
        );
    }
}
