//! Policy study on real coding tasks — does the first-pass logic actually beat the alternatives?
//!
//! `coding.rs` answers a different question. It measures how good a *gate* is (false-accept,
//! false-reject, the conformal bound) for one solver. That is the scientific novelty, but it is
//! not the product claim. The product claim is about the **routing policy**: serve the cheapest
//! rung that provably passes, escalate only when it does not. Until now that claim was only ever
//! compared against baselines on generated arithmetic, where the gate is perfect and the tasks are
//! self-checking — the easiest possible case for it.
//!
//! This runs the comparison on real problems, against the baselines that matter:
//!
//! | policy | what it does | what it is here to show |
//! |---|---|---|
//! | always-cheap | serve rung 0, never verify | the floor: cheapest possible, worst quality |
//! | always-top | serve the top rung, never verify | the ceiling everyone pays for by default |
//! | first-pass | rung 0, gate it, escalate on failure | cheaper than the ceiling at the ceiling's quality — or it does not work |
//!
//! **Measure once, evaluate every policy.** Each task is solved once per rung and scored against
//! the visible gate and the hidden oracle; every policy is then replayed offline over that same
//! matrix of measurements. So the cost is `tasks × rungs` model calls no matter how many policies
//! are compared, and — more importantly — every policy is judged on **identical** outcomes, which
//! removes sampling noise from the comparison entirely. Re-solving per policy would both cost more
//! and make the difference between policies partly an artifact of which samples each one drew.

use crate::coding::{CandidateSolver, CodingTask, TaskOutcome, evaluate_task};
use crate::sandbox::{Limits, Sandbox};
use crate::stats::{Ci, bootstrap_mean_ci, bootstrap_ratio_ci};
use firstpass_core::cost::PriceTable;

/// One rung of the ladder: the ladder id used for pricing, and the solver that speaks for it.
pub struct Rung<'a> {
    /// Price-table key, e.g. `"anthropic/claude-haiku-4-5"`.
    pub model: String,
    /// The solver that produces candidates for this rung.
    pub solver: &'a dyn CandidateSolver,
}

// Hand-written rather than derived: a solver is a trait object with no `Debug` bound, and adding
// one to `CandidateSolver` would push a formatting requirement onto every implementor to satisfy a
// lint. The model id is the only part worth printing anyway.
impl std::fmt::Debug for Rung<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rung").field("model", &self.model).finish()
    }
}

/// What one task cost and produced at one rung.
#[derive(Debug, Clone)]
pub struct RungOutcome {
    /// Fraction of visible (gate) cases passed, in `[0, 1]`.
    pub gate_score: f64,
    /// Every visible case passed — the condition on which a gate serves.
    pub gate_full_pass: bool,
    /// Every hidden oracle case passed — ground truth.
    pub oracle_correct: bool,
    /// USD for this call, from the shared price table.
    pub cost_usd: f64,
}

/// One policy's measured result over the suite.
#[derive(Debug, Clone)]
pub struct PolicyResult {
    /// Policy name.
    pub name: &'static str,
    /// Fraction of tasks where the SERVED answer was actually correct per the hidden oracle.
    pub success_rate: f64,
    /// Bootstrap CI on the success rate.
    pub success_ci: Ci,
    /// Total USD spent across the suite, including rungs that were tried and rejected.
    pub total_cost_usd: f64,
    /// USD per successfully-solved task — the number the product claim lives or dies on.
    pub usd_per_success: f64,
    /// Bootstrap CI on `$/success` (ratio of two means, so a ratio bootstrap).
    pub usd_per_success_ci: Ci,
    /// Fraction of served answers that were wrong — what a caller actually experiences.
    pub served_failure_rate: f64,
    /// Fraction of tasks where this policy paid for more than one rung. Zero for the fixed
    /// policies; for first-pass it is the diagnosis when the numbers disappoint — escalations
    /// that never convert a failure into a success are pure cost.
    pub escalation_rate: f64,
    /// Of those escalations, the fraction that actually turned a wrong answer into a right one.
    /// This is what the gate is *for*; if it is near zero the gate is not earning its keep on
    /// this workload, whatever the headline $/success says.
    pub escalation_conversion: f64,
}

/// The whole study.
#[derive(Debug, Clone)]
pub struct PolicyStudy {
    /// Isolation tier the sandbox ran under.
    pub runtime_tier: String,
    /// Tasks measured.
    pub n: usize,
    /// Ladder model ids, cheapest first.
    pub ladder: Vec<String>,
    /// Per-policy results, in table order.
    pub policies: Vec<PolicyResult>,
}

/// Solve every task at every rung and score both suites. This is the only part that spends money.
///
/// # Errors
/// Any solver or environment failure, verbatim — a partial matrix would silently bias every
/// policy computed from it, so nothing is published from a broken run.
pub fn measure(
    tasks: &[CodingTask],
    rungs: &[Rung<'_>],
    sb: &dyn Sandbox,
    prices: &PriceTable,
    limits: &Limits,
) -> Result<Vec<Vec<RungOutcome>>, String> {
    let mut matrix = Vec::with_capacity(tasks.len());
    for task in tasks {
        let mut row = Vec::with_capacity(rungs.len());
        for rung in rungs {
            let o: TaskOutcome = evaluate_task(sb, rung.solver, task, limits)?;
            let cost = prices
                .cost_usd(&rung.model, o.in_tokens, o.out_tokens)
                .map_err(|e| {
                    format!(
                        "rung {:?} has no price, so $/success would be computed from a fabricated \
                         zero: {e}",
                        rung.model
                    )
                })?;
            row.push(RungOutcome {
                gate_score: o.gate_score,
                gate_full_pass: o.gate_full_pass,
                oracle_correct: o.oracle_correct,
                cost_usd: cost,
            });
        }
        matrix.push(row);
    }
    Ok(matrix)
}

/// Replay every policy over an already-measured matrix. Pure — no I/O, no spend, deterministic.
#[must_use]
pub fn replay(matrix: &[Vec<RungOutcome>], ladder: &[String], runtime_tier: &str) -> PolicyStudy {
    let top = ladder.len().saturating_sub(1);
    let policies = vec![
        eval_policy("always-cheap", matrix, |row| serve_fixed(row, 0)),
        eval_policy("always-top", matrix, |row| serve_fixed(row, top)),
        eval_policy("first-pass", matrix, serve_first_pass),
    ];
    PolicyStudy {
        runtime_tier: runtime_tier.to_owned(),
        n: matrix.len(),
        ladder: ladder.to_vec(),
        policies,
    }
}

/// `(served_correct, cost_paid, rungs_paid)` for serving one fixed rung, with no verification.
/// One rung is called and one rung is billed — starting at the top is not the same as escalating
/// to it, and conflating the two made `always-top` report "escalated 100%".
fn serve_fixed(row: &[RungOutcome], idx: usize) -> (bool, f64, usize) {
    row.get(idx)
        .map_or((false, 0.0, 0), |o| (o.oracle_correct, o.cost_usd, 1))
}

/// The first-pass policy: try the cheapest rung, run the gate, and escalate only when the gate
/// says no. Cost accumulates across every rung tried — the rejected attempt is still billed, which
/// is exactly the honesty the comparison needs.
fn serve_first_pass(row: &[RungOutcome]) -> (bool, f64, usize) {
    let mut spent = 0.0;
    for (i, o) in row.iter().enumerate() {
        spent += o.cost_usd;
        if o.gate_full_pass {
            return (o.oracle_correct, spent, i + 1);
        }
    }
    // Every rung failed its gate. Best-attempt fallback: the top rung is served anyway, because
    // refusing to answer is not on the table for a proxy. It is counted as served, so a policy
    // that fails everywhere cannot hide its failures by declining to serve them.
    (
        row.last().is_some_and(|o| o.oracle_correct),
        spent,
        row.len(),
    )
}

fn eval_policy(
    name: &'static str,
    matrix: &[Vec<RungOutcome>],
    serve: impl Fn(&[RungOutcome]) -> (bool, f64, usize),
) -> PolicyResult {
    let (mut successes, mut costs) = (Vec::new(), Vec::new());
    let (mut escalations, mut converted) = (0usize, 0usize);
    for row in matrix {
        let (ok, cost, rungs_paid) = serve(row);
        successes.push(f64::from(u8::from(ok)));
        costs.push(cost);
        // An escalation is paying for MORE THAN ONE rung on a task — not "paying more than the
        // cheap rung costs", which is also true of any policy that simply starts higher up.
        // Whether it CONVERTED is the only thing that justifies the extra spend.
        if rungs_paid > 1 {
            escalations += 1;
            if ok && row.first().is_some_and(|c| !c.oracle_correct) {
                converted += 1;
            }
        }
    }
    let n = successes.len().max(1) as f64;
    let success_rate = successes.iter().sum::<f64>() / n;
    let total_cost_usd = costs.iter().sum::<f64>();
    let n_success = successes.iter().sum::<f64>();
    PolicyResult {
        name,
        success_rate,
        success_ci: bootstrap_mean_ci(&successes, 2000, 42, 0.05),
        total_cost_usd,
        // Infinite rather than a divide-by-zero: a policy that solves nothing has no cost per
        // success, and reporting 0.0 there would make it look like the cheapest option.
        usd_per_success: if n_success > 0.0 {
            total_cost_usd / n_success
        } else {
            f64::INFINITY
        },
        usd_per_success_ci: bootstrap_ratio_ci(&costs, &successes, 2000, 42, 0.05),
        served_failure_rate: 1.0 - success_rate,
        escalation_rate: escalations as f64 / n,
        escalation_conversion: if escalations > 0 {
            converted as f64 / escalations as f64
        } else {
            0.0
        },
    }
}

impl PolicyStudy {
    /// Markdown table plus the verdict, in the honest-report style the rest of the bench uses.
    #[must_use]
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("## Routing policy on real coding tasks\n\n");
        s.push_str(&format!(
            "n = {} tasks · ladder = {} · sandbox tier = {}\n\n",
            self.n,
            self.ladder.join(" → "),
            self.runtime_tier
        ));
        s.push_str(
            "| policy | success | 95% CI | total $ | $/success | served-failure | escalated | converted |\n",
        );
        s.push_str("|---|---|---|---|---|---|---|---|\n");
        for p in &self.policies {
            s.push_str(&format!(
                "| {} | {:.2} | [{:.2}, {:.2}] | ${:.4} | ${:.4} | {:.2} | {:.0}% | {:.0}% |\n",
                p.name,
                p.success_rate,
                p.success_ci.lo,
                p.success_ci.hi,
                p.total_cost_usd,
                p.usd_per_success,
                p.served_failure_rate,
                p.escalation_rate * 100.0,
                p.escalation_conversion * 100.0
            ));
        }
        s.push('\n');
        s.push_str(&self.verdict());
        s
    }

    /// The claim, stated so it can be wrong — and so it can be *found* wrong by this data.
    ///
    /// The first version asked only "cheaper than always-top, and is the CI hi at least its
    /// success rate?". At pilot n the CI hi reaches 1.00, so that second clause was true no
    /// matter what happened: it stamped PROCEED on a run where the gate converted nothing. An
    /// assertion that cannot fail is worse than none, so the test is now the one that matters:
    ///
    /// 1. **vs always-cheap** — if gating does not buy more successes than simply serving the
    ///    cheap rung, the gate is pure overhead whatever the ceiling comparison says. This clause
    ///    catches a gate that never fires, and it runs first because it is the cheapest way for
    ///    the whole idea to be wrong.
    /// 2. **vs always-top** — cheaper per success, without giving up quality beyond noise.
    ///
    /// Overlapping intervals at small n mean INCONCLUSIVE, not PROCEED. A pilot decides whether
    /// the full run is worth paying for; it does not get to declare the result early.
    #[must_use]
    pub fn verdict(&self) -> String {
        let (Some(fp), Some(top), Some(cheap)) = (
            self.find("first-pass"),
            self.find("always-top"),
            self.find("always-cheap"),
        ) else {
            return "VERDICT: incomplete — missing a policy to compare.\n".to_owned();
        };
        let saving = if top.usd_per_success.is_finite() && top.usd_per_success > 0.0 {
            (1.0 - fp.usd_per_success / top.usd_per_success) * 100.0
        } else {
            0.0
        };

        // Clause 1: did the gate earn its cost at all?
        if fp.success_rate <= cheap.success_rate && fp.total_cost_usd > cheap.total_cost_usd {
            return format!(
                "VERDICT: **STOP — the gate did not earn its cost.** first-pass matched \
                 always-cheap on quality ({:.2} vs {:.2}) while spending more (${:.4} vs ${:.4}). \
                 It escalated on {:.0}% of tasks and {:.0}% of those escalations turned a wrong \
                 answer into a right one. On this workload the visible tests are not catching what \
                 the cheap rung gets wrong, so gating buys nothing — the lever is a better gate or \
                 harder tasks, not more traffic.\n",
                fp.success_rate,
                cheap.success_rate,
                fp.total_cost_usd,
                cheap.total_cost_usd,
                fp.escalation_rate * 100.0,
                fp.escalation_conversion * 100.0,
            );
        }

        // Clause 2: against the ceiling everyone pays for by default.
        if fp.usd_per_success >= top.usd_per_success {
            return format!(
                "VERDICT: **STOP** — first-pass did not cost less per success (${:.4} vs ${:.4}). \
                 Escalation overhead exceeded what the cheap rung saved.\n",
                fp.usd_per_success, top.usd_per_success
            );
        }
        if fp.success_ci.hi < top.success_ci.lo {
            return format!(
                "VERDICT: **STOP** — {saving:.0}% cheaper but measurably worse ({:.2} vs {:.2}, \
                 intervals do not overlap). Cheaper at lower quality is not the claim.\n",
                fp.success_rate, top.success_rate
            );
        }
        if fp.success_rate < top.success_rate {
            return format!(
                "VERDICT: **INCONCLUSIVE** — first-pass is {saving:.0}% cheaper per success but \
                 scored below always-top ({:.2} vs {:.2}) with overlapping intervals at n={}. \
                 Neither a win nor a loss; size the run so the intervals separate before claiming \
                 either.\n",
                fp.success_rate, top.success_rate, self.n
            );
        }
        format!(
            "VERDICT: **PROCEED** — first-pass matched or beat always-top on quality ({:.2} vs \
             {:.2}) at {saving:.0}% lower $/success, and beat always-cheap ({:.2} vs {:.2}) so the \
             gate paid for itself.\n",
            fp.success_rate, top.success_rate, fp.success_rate, cheap.success_rate
        )
    }

    fn find(&self, name: &str) -> Option<&PolicyResult> {
        self.policies.iter().find(|p| p.name == name)
    }
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
        }
    }

    fn ladder() -> Vec<String> {
        vec!["cheap".to_owned(), "top".to_owned()]
    }

    /// The shape the product claims: the cheap rung is right most of the time and the gate catches
    /// it when it is not, so quality matches the top rung at a fraction of the cost.
    #[test]
    fn first_pass_beats_always_top_when_the_gate_catches_the_cheap_rung() {
        // 3 tasks the cheap rung gets right; 1 it gets wrong AND the gate catches.
        let matrix = vec![
            vec![o(true, true, 0.01), o(true, true, 0.10)],
            vec![o(true, true, 0.01), o(true, true, 0.10)],
            vec![o(true, true, 0.01), o(true, true, 0.10)],
            vec![o(false, false, 0.01), o(true, true, 0.10)],
        ];
        let study = replay(&matrix, &ladder(), "test");
        let fp = study.find("first-pass").unwrap();
        let top = study.find("always-top").unwrap();
        let cheap = study.find("always-cheap").unwrap();

        assert!(
            (fp.success_rate - 1.0).abs() < 1e-9,
            "gate caught the bad one"
        );
        assert!((top.success_rate - 1.0).abs() < 1e-9);
        assert!(
            (cheap.success_rate - 0.75).abs() < 1e-9,
            "cheap alone is wrong once"
        );
        // 3 cheap-only + 1 escalation (cheap + top) = 0.03 + 0.11
        assert!((fp.total_cost_usd - 0.14).abs() < 1e-9);
        assert!(fp.usd_per_success < top.usd_per_success);
        assert!(study.verdict().contains("PROCEED"), "{}", study.verdict());
    }

    /// The failure the verdict must be willing to report: a gate that waves through a wrong cheap
    /// answer is cheap AND worse, and that is not the product.
    #[test]
    fn a_leaky_gate_is_reported_as_stop_not_as_a_saving() {
        // The cheap rung is wrong but the gate passes it — a false accept, every time.
        let matrix = vec![
            vec![o(true, false, 0.01), o(true, true, 0.10)],
            vec![o(true, false, 0.01), o(true, true, 0.10)],
        ];
        let study = replay(&matrix, &ladder(), "test");
        let fp = study.find("first-pass").unwrap();
        assert!((fp.success_rate - 0.0).abs() < 1e-9);
        assert!(
            fp.usd_per_success.is_infinite(),
            "no successes ⇒ not $0.00/success"
        );
        let v = study.verdict();
        assert!(
            v.contains("STOP"),
            "a leaky gate must not read as a win: {v}"
        );
    }

    /// The exact run that exposed the old verdict as a rubber stamp, frozen as a test.
    ///
    /// Real numbers from the first BigCodeBench pilot (n=24, haiku→sonnet): first-pass tied
    /// always-cheap on quality and cost MORE, because the visible tests almost never caught what
    /// haiku got wrong. The previous logic called that PROCEED — its quality clause was
    /// `success_ci.hi >= top.success_rate`, and at n=24 the interval reaches 1.00, so the clause
    /// was true regardless of the data. This asserts the verdict can now say no.
    #[test]
    fn a_gate_that_converts_nothing_is_reported_as_stop_not_as_a_saving() {
        let mut matrix = Vec::new();
        // 18: cheap is right and the gate agrees — served cheap, nothing escalates.
        for _ in 0..18 {
            matrix.push(vec![o(true, true, 0.0028), o(true, true, 0.0054)]);
        }
        // 3: cheap is right but the gate FAILS it — a false reject. Escalates, pays for the top
        // rung, and gets back an answer that is correct exactly as the cheap one already was.
        // This is where the extra money went in the real pilot: motion without conversion.
        for _ in 0..3 {
            matrix.push(vec![o(false, true, 0.0028), o(true, true, 0.0054)]);
        }
        // 3: cheap is WRONG and the gate waves it through — a false accept. No escalation, so
        // the failure is served and costs nothing extra to get wrong.
        for _ in 0..3 {
            matrix.push(vec![o(true, false, 0.0028), o(true, true, 0.0054)]);
        }
        let study = replay(&matrix, &ladder(), "runc");
        let fp = study.find("first-pass").unwrap();
        let cheap = study.find("always-cheap").unwrap();
        assert!(
            (fp.success_rate - cheap.success_rate).abs() < 1e-9,
            "the gate changed nothing, so quality must equal always-cheap"
        );
        let v = study.verdict();
        assert!(
            v.contains("STOP") && v.contains("did not earn its cost"),
            "a gate that converts nothing must not read as a win: {v}"
        );
    }

    /// Starting at the top rung is not escalating to it. The first version counted an escalation
    /// as "paid more than rung 0 costs", which is trivially true of `always-top` — so the pilot
    /// table claimed it escalated on 100% of tasks, which is meaningless for a policy that only
    /// ever makes one call.
    #[test]
    fn a_policy_that_starts_at_the_top_has_not_escalated() {
        let matrix = vec![
            vec![o(false, false, 0.01), o(true, true, 0.10)],
            vec![o(true, true, 0.01), o(true, true, 0.10)],
        ];
        let study = replay(&matrix, &ladder(), "test");
        for name in ["always-cheap", "always-top"] {
            let p = study.find(name).unwrap();
            assert!(
                (p.escalation_rate - 0.0).abs() < 1e-9,
                "{name} makes exactly one call per task, so it cannot escalate; got {}",
                p.escalation_rate
            );
        }
        // first-pass escalated on the one task whose gate failed.
        let fp = study.find("first-pass").unwrap();
        assert!(
            (fp.escalation_rate - 0.5).abs() < 1e-9,
            "got {}",
            fp.escalation_rate
        );
    }

    /// Escalation is billed for every rung it tried, not just the one it served. Hiding the
    /// rejected attempt would flatter first-pass in exactly the comparison it is meant to lose
    /// when it deserves to.
    #[test]
    fn a_rejected_rung_is_still_paid_for() {
        let matrix = vec![vec![o(false, false, 0.05), o(true, true, 0.10)]];
        let study = replay(&matrix, &ladder(), "test");
        let fp = study.find("first-pass").unwrap();
        assert!(
            (fp.total_cost_usd - 0.15).abs() < 1e-9,
            "cost must include the failed cheap attempt, got {}",
            fp.total_cost_usd
        );
    }
}
