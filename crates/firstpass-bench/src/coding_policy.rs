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

use crate::coding::{
    CandidateSolver, CodingTask, Judge, TaskOutcome, evaluate_task, evaluate_task_judged,
};
use crate::sandbox::{Limits, Sandbox};
use crate::stats::{Ci, bootstrap_mean_ci, bootstrap_ratio_ci};
use firstpass_core::cost::PriceTable;
use std::collections::HashMap;

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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RungOutcome {
    /// Fraction of visible (gate) cases passed, in `[0, 1]`.
    pub gate_score: f64,
    /// Every visible case passed — the condition on which a gate serves.
    pub gate_full_pass: bool,
    /// Every hidden oracle case passed — ground truth.
    pub oracle_correct: bool,
    /// USD for this call, from the shared price table.
    pub cost_usd: f64,
    /// A second, independent opinion on the same candidate in `[0, 1]` — present only when the
    /// study was run with a judge. `None` also covers an abstention, which must defer to the test
    /// gate rather than be read as a low score.
    pub judge_score: Option<f64>,
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
    /// Whether every model call this policy made is represented in `total_cost_usd`.
    ///
    /// False for the judged arm: `Judge::score` returns an opinion but not its token usage, so a
    /// judge's calls are currently invisible to the price table. Reporting a $/success that omits
    /// them would make adding a judge look free, which is the same fabricated-cost bug the
    /// unpriced-rung validator exists to prevent — so the figure is withheld rather than shown
    /// wrong. Metering judge tokens (the proxy already does this as `gate_cost_usd`) is the fix.
    pub cost_metered: bool,
}

/// A **paired** comparison of first-pass against one baseline.
///
/// Every policy here is replayed over the identical task matrix, so the two arms are not
/// independent samples — they are the same tasks decided two ways. Comparing their marginal
/// confidence intervals therefore asks the wrong question and answers it with far less power than
/// the data holds: shared task difficulty inflates both intervals, and they overlap even when the
/// per-task difference is consistent. Bootstrapping the DIFFERENCE cancels that shared variance.
///
/// The 974-task MBPP run is the case in point. Marginal intervals [0.91, 0.94] and [0.93, 0.96]
/// overlap, so the verdict read INCONCLUSIVE — while the per-task record showed first-pass and
/// always-top disagreeing on only a handful of tasks, in a consistent direction.
#[derive(Debug, Clone)]
pub struct Paired {
    /// Baseline being compared against.
    pub vs: &'static str,
    /// Mean per-task success difference (first-pass minus baseline). Negative = worse.
    pub delta_success: f64,
    /// Bootstrap CI on that difference. Excludes 0 ⇒ a real difference, not noise.
    pub delta_success_ci: Ci,
    /// Mean per-task cost difference in USD. Negative = cheaper.
    pub delta_cost_usd: f64,
    /// Bootstrap CI on the cost difference.
    pub delta_cost_ci: Ci,
    /// Tasks first-pass got right that the baseline got wrong.
    pub wins: usize,
    /// Tasks first-pass got wrong that the baseline got right — the ones a quality claim rests on.
    pub losses: usize,
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
    /// Paired comparisons of first-pass against each baseline — the correct test for this design.
    pub paired: Vec<Paired>,
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
    measure_judged(tasks, rungs, sb, prices, limits, None)
}

/// As [`measure`], but also asks a judge for a second opinion on every candidate.
///
/// The first pilot's whole finding was that a visible-test prefix does not separate the cheap
/// rung's good answers from its bad ones — it passed both. A judge is a signal the tests do not
/// have, so this is the experiment that decides whether first-pass loses on *this workload* or on
/// *this gate*. Costs one extra (cheap) call per candidate.
///
/// # Errors
/// Any solver, judge, or environment failure, verbatim.
pub fn measure_judged(
    tasks: &[CodingTask],
    rungs: &[Rung<'_>],
    sb: &dyn Sandbox,
    prices: &PriceTable,
    limits: &Limits,
    judge: Option<&dyn Judge>,
) -> Result<Vec<Vec<RungOutcome>>, String> {
    measure_resumable(tasks, rungs, sb, prices, limits, judge, None)
}

/// As [`measure_judged`], but checkpoints each finished task to `checkpoint` and resumes from it.
///
/// A 974-task run is ~2000 paid calls over hours, and it has now been killed twice — once by a
/// starved response, once by a transient network drop — each time discarding every call already
/// bought. Retries alone cannot fix that: the failure that matters is the one that outlives the
/// retry budget. So the completed work goes to disk as it happens, and a re-run skips what is
/// already measured. Nothing about the result changes; only whether an interruption costs the
/// whole run or just the tail of it.
///
/// The checkpoint is keyed by task id, so it is safe to resume with a different `--limit` or a
/// reordered file; tasks not present are simply measured.
///
/// # Errors
/// Any solver, judge, or environment failure, verbatim — but the checkpoint keeps whatever
/// completed before it, so the retry resumes rather than restarts.
pub fn measure_resumable(
    tasks: &[CodingTask],
    rungs: &[Rung<'_>],
    sb: &dyn Sandbox,
    prices: &PriceTable,
    limits: &Limits,
    judge: Option<&dyn Judge>,
    checkpoint: Option<&std::path::Path>,
) -> Result<Vec<Vec<RungOutcome>>, String> {
    let mut done = load_checkpoint(checkpoint);
    if !done.is_empty() {
        eprintln!("resuming: {} tasks already measured", done.len());
    }
    let mut matrix = Vec::with_capacity(tasks.len());
    for task in tasks {
        if let Some(row) = done.remove(&task.id) {
            matrix.push(row);
            continue;
        }
        let mut row = Vec::with_capacity(rungs.len());
        for rung in rungs {
            let o: TaskOutcome = match judge {
                Some(j) => evaluate_task_judged(sb, rung.solver, j, task, limits)?,
                None => evaluate_task(sb, rung.solver, task, limits)?,
            };
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
                judge_score: o.judge_score,
            });
        }
        append_checkpoint(checkpoint, &task.id, &row);
        matrix.push(row);
    }
    Ok(matrix)
}

/// Read a checkpoint into `task_id -> row`. A malformed or absent file yields nothing and the run
/// simply measures everything: a checkpoint is an optimisation, never a source of truth, so it
/// must not be able to fail a run or — worse — inject a row nobody measured.
fn load_checkpoint(path: Option<&std::path::Path>) -> HashMap<String, Vec<RungOutcome>> {
    let Some(p) = path else {
        return HashMap::new();
    };
    let Ok(text) = std::fs::read_to_string(p) else {
        return HashMap::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str::<CheckpointRow>(l).ok())
        .map(|r| (r.task_id, r.rungs))
        .collect()
}

/// Append one finished task. Best-effort by the same reasoning: a failure to checkpoint must not
/// discard a measurement that was already paid for.
fn append_checkpoint(path: Option<&std::path::Path>, task_id: &str, row: &[RungOutcome]) {
    let Some(p) = path else { return };
    let line = serde_json::to_string(&CheckpointRow {
        task_id: task_id.to_owned(),
        rungs: row.to_vec(),
    });
    if let Ok(line) = line {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CheckpointRow {
    task_id: String,
    rungs: Vec<RungOutcome>,
}

/// Replay every policy over an already-measured matrix. Pure — no I/O, no spend, deterministic.
#[must_use]
pub fn replay(matrix: &[Vec<RungOutcome>], ladder: &[String], runtime_tier: &str) -> PolicyStudy {
    let top = ladder.len().saturating_sub(1);
    let mut policies = vec![
        eval_policy("always-cheap", matrix, |row| serve_fixed(row, 0)),
        eval_policy("always-top", matrix, |row| serve_fixed(row, top)),
        eval_policy("first-pass", matrix, serve_first_pass),
    ];
    // Only meaningful if a judge actually scored something; otherwise the row would duplicate
    // first-pass and imply an experiment that was never run.
    if matrix
        .iter()
        .any(|row| row.iter().any(|o| o.judge_score.is_some()))
    {
        let mut judged = eval_policy("first-pass+judge", matrix, |row| {
            serve_first_pass_judged(row, JUDGE_THRESHOLD)
        });
        judged.cost_metered = false;
        policies.push(judged);
    }
    let paired_cmps = vec![
        paired("always-top", matrix, |row| serve_fixed(row, top)),
        paired("always-cheap", matrix, |row| serve_fixed(row, 0)),
    ];
    PolicyStudy {
        runtime_tier: runtime_tier.to_owned(),
        n: matrix.len(),
        ladder: ladder.to_vec(),
        policies,
        paired: paired_cmps,
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

/// How confident the judge must be for its opinion to let a candidate through. Deliberately a
/// constant rather than a tuned knob: tuning it on the same 24 tasks it is evaluated on would be
/// fitting the threshold to the answer.
const JUDGE_THRESHOLD: f64 = 0.8;

/// First-pass with a second opinion: serve the cheap rung only when the tests pass **and** the
/// judge is confident. The pilot showed the tests alone pass the cheap rung's wrong answers, so
/// this is the arm that tests whether a signal the tests do not have changes the verdict.
///
/// A judge that abstains (`None`) defers to the tests rather than blocking — an abstention is an
/// absence of evidence, and treating it as a veto would escalate everything the judge found hard.
fn serve_first_pass_judged(row: &[RungOutcome], threshold: f64) -> (bool, f64, usize) {
    let mut spent = 0.0;
    for (i, o) in row.iter().enumerate() {
        spent += o.cost_usd;
        let judge_ok = o.judge_score.is_none_or(|j| j >= threshold);
        if o.gate_full_pass && judge_ok {
            return (o.oracle_correct, spent, i + 1);
        }
    }
    (
        row.last().is_some_and(|o| o.oracle_correct),
        spent,
        row.len(),
    )
}

/// Per-task `(success, cost)` for one policy — the raw series both the marginal summary and the
/// paired comparison are computed from, so they can never disagree about what was measured.
fn series(
    matrix: &[Vec<RungOutcome>],
    serve: impl Fn(&[RungOutcome]) -> (bool, f64, usize),
) -> (Vec<f64>, Vec<f64>) {
    let mut ok = Vec::with_capacity(matrix.len());
    let mut cost = Vec::with_capacity(matrix.len());
    for row in matrix {
        let (s, c, _) = serve(row);
        ok.push(f64::from(u8::from(s)));
        cost.push(c);
    }
    (ok, cost)
}

/// Bootstrap the per-task difference between first-pass and one baseline.
fn paired(
    vs: &'static str,
    matrix: &[Vec<RungOutcome>],
    baseline: impl Fn(&[RungOutcome]) -> (bool, f64, usize),
) -> Paired {
    let (fp_ok, fp_cost) = series(matrix, serve_first_pass);
    let (bl_ok, bl_cost) = series(matrix, baseline);
    let d_ok: Vec<f64> = fp_ok.iter().zip(&bl_ok).map(|(a, b)| a - b).collect();
    let d_cost: Vec<f64> = fp_cost.iter().zip(&bl_cost).map(|(a, b)| a - b).collect();
    Paired {
        vs,
        delta_success: crate::stats::mean(&d_ok),
        delta_success_ci: bootstrap_mean_ci(&d_ok, 2000, 42, 0.05),
        delta_cost_usd: crate::stats::mean(&d_cost),
        delta_cost_ci: bootstrap_mean_ci(&d_cost, 2000, 42, 0.05),
        wins: d_ok.iter().filter(|d| **d > 0.0).count(),
        losses: d_ok.iter().filter(|d| **d < 0.0).count(),
    }
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
        cost_metered: true,
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
                "| {} | {:.2} | [{:.2}, {:.2}] | {} | {} | {:.2} | {:.0}% | {:.0}% |\n",
                p.name,
                p.success_rate,
                p.success_ci.lo,
                p.success_ci.hi,
                if p.cost_metered {
                    format!("${:.4}", p.total_cost_usd)
                } else {
                    "—".to_owned()
                },
                if p.cost_metered {
                    format!("${:.4}", p.usd_per_success)
                } else {
                    "—".to_owned()
                },
                p.served_failure_rate,
                p.escalation_rate * 100.0,
                p.escalation_conversion * 100.0
            ));
        }
        if let Some(t) = self.paired.iter().find(|p| p.vs == "always-top") {
            s.push_str(
                "\nPaired against always-top (same tasks, so the per-task difference is the \
                 test with the power this design actually has):\n\n",
            );
            s.push_str("| vs | Δ success | 95% CI | Δ $/task | 95% CI | fp wins | fp loses |\n");
            s.push_str("|---|---|---|---|---|---|---|\n");
            for p in &self.paired {
                s.push_str(&format!(
                    "| {} | {:+.3} | [{:+.3}, {:+.3}] | {:+.5} | [{:+.5}, {:+.5}] | {} | {} |\n",
                    p.vs,
                    p.delta_success,
                    p.delta_success_ci.lo,
                    p.delta_success_ci.hi,
                    p.delta_cost_usd,
                    p.delta_cost_ci.lo,
                    p.delta_cost_ci.hi,
                    p.wins,
                    p.losses
                ));
            }
            let _ = t;
        }
        if self.policies.iter().any(|p| !p.cost_metered) {
            s.push_str(
                "\n\u{2014} cost withheld: the judge's own model calls are not yet metered, so any \
                 $/success for that arm would omit them. Its quality and conversion columns are \
                 measured normally; those are what the arm exists to answer.\n",
            );
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
        // Quality, judged on the PAIRED difference. Marginal intervals overlap on this design
        // even when every task agrees, because both arms carry the same task-difficulty variance;
        // the per-task difference is where the signal is. If the interval on that difference
        // contains 0, the two are not distinguishable — that is a genuine result, not a shrug.
        let Some(pt) = self.paired.iter().find(|p| p.vs == "always-top") else {
            return "VERDICT: incomplete — no paired comparison against always-top.\n".to_owned();
        };
        if pt.delta_success_ci.hi < 0.0 {
            return format!(
                "VERDICT: **TRADEOFF, MEASURED** — first-pass costs {saving:.0}% less per success \
                 and is {:.1} points worse in quality (paired Δ {:+.3}, 95% CI [{:+.3}, {:+.3}], \
                 so the gap is real and not noise). It loses {} tasks to always-top and wins {}. \
                 Whether that trade is worth taking is a product decision, not a benchmark one — \
                 but it is now a stated price rather than an unmeasured hope.\n",
                -pt.delta_success * 100.0,
                pt.delta_success,
                pt.delta_success_ci.lo,
                pt.delta_success_ci.hi,
                pt.losses,
                pt.wins,
            );
        }
        format!(
            "VERDICT: **PROCEED** — first-pass is not distinguishable from always-top on quality \
             (paired Δ {:+.3}, 95% CI [{:+.3}, {:+.3}] contains 0) at {saving:.0}% lower \
             $/success, and beats always-cheap ({:.2} vs {:.2}) so the gate paid for itself.\n",
            pt.delta_success,
            pt.delta_success_ci.lo,
            pt.delta_success_ci.hi,
            fp.success_rate,
            cheap.success_rate
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
            judge_score: None,
        }
    }

    /// Same, with a judge opinion attached.
    fn oj(gate: bool, oracle: bool, cost: f64, judge: f64) -> RungOutcome {
        RungOutcome {
            judge_score: Some(judge),
            ..o(gate, oracle, cost)
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

    /// The judged arm exists to catch exactly what the test gate misses: a wrong answer that
    /// passes the visible tests. The judge only helps if it can veto that; and it must NOT veto
    /// on an abstention, or every task it finds hard gets escalated for no evidence.
    #[test]
    fn a_judge_vetoes_a_test_passing_wrong_answer_but_an_abstention_defers() {
        // Cheap rung is wrong and the tests wave it through — the pilot's actual failure mode.
        // The judge is unconvinced (0.2), so the judged arm escalates and gets it right.
        let caught = vec![vec![oj(true, false, 0.01, 0.2), oj(true, true, 0.10, 0.95)]];
        let study = replay(&caught, &ladder(), "test");
        let plain = study.find("first-pass").unwrap();
        let judged = study.find("first-pass+judge").expect("judged arm present");
        assert!(
            (plain.success_rate - 0.0).abs() < 1e-9,
            "tests alone serve the wrong answer"
        );
        assert!(
            (judged.success_rate - 1.0).abs() < 1e-9,
            "the judge's veto should have escalated to a correct answer"
        );

        // An abstaining judge must defer to the tests, not block: same outcome as plain
        // first-pass, and no escalation bought on the strength of a shrug.
        let abstained = vec![vec![
            RungOutcome {
                judge_score: None,
                ..o(true, true, 0.01)
            },
            oj(true, true, 0.10, 0.95),
        ]];
        let study = replay(&abstained, &ladder(), "test");
        let judged = study.find("first-pass+judge").expect("judged arm present");
        assert!(
            (judged.escalation_rate - 0.0).abs() < 1e-9,
            "abstention is not a veto"
        );
    }

    /// The paired test must find a difference that overlapping marginal intervals hide. That is
    /// the entire reason it exists: both arms are the SAME tasks decided two ways, so their
    /// marginal intervals both carry task-difficulty variance and overlap even when every
    /// disagreement points the same way. On the 974-task MBPP run that read INCONCLUSIVE while
    /// the per-task record was consistent.
    #[test]
    fn pairing_resolves_what_overlapping_marginal_intervals_cannot() {
        // BOTH arms must be imperfect or this demonstrates nothing: a baseline that never fails
        // has zero variance, so its marginal interval is a point and cannot overlap anything.
        // Real data is not like that, and neither is this fixture.
        let mut matrix = Vec::new();
        for i in 0..200 {
            let spread = f64::from(i % 20) * 0.01; // per-task cost variance, identical in both arms
            if i < 10 {
                // Gate wrongly PASSES a bad cheap answer; the top rung would have been right.
                // These 10 tasks are the entire quality difference, and they all point one way.
                matrix.push(vec![
                    o(true, false, 0.01 + spread),
                    o(true, true, 0.10 + spread),
                ]);
            } else if i < 30 {
                // Both rungs wrong: adds variance to BOTH marginals, contributes 0 to the paired
                // difference. This is what makes the marginal intervals overlap.
                matrix.push(vec![
                    o(true, false, 0.01 + spread),
                    o(true, false, 0.10 + spread),
                ]);
            } else {
                matrix.push(vec![
                    o(true, true, 0.01 + spread),
                    o(true, true, 0.10 + spread),
                ]);
            }
        }
        let study = replay(&matrix, &ladder(), "test");
        let fp = study.find("first-pass").unwrap();
        let top = study.find("always-top").unwrap();
        let pt = study.paired.iter().find(|p| p.vs == "always-top").unwrap();

        // Every disagreement points one way: 10 losses, no wins.
        assert_eq!((pt.wins, pt.losses), (0, 10));
        // Both arms genuinely imperfect, so both marginals carry real width.
        assert!(fp.success_rate < 0.9 && top.success_rate < 0.95);
        // The paired interval excludes 0 — a real difference…
        assert!(
            pt.delta_success_ci.hi < 0.0,
            "paired CI must exclude 0: [{:+.4}, {:+.4}]",
            pt.delta_success_ci.lo,
            pt.delta_success_ci.hi
        );
        // …and it is far tighter than the gap between the two marginal intervals suggests.
        let marginal_overlap = fp.success_ci.hi >= top.success_ci.lo;
        assert!(
            marginal_overlap,
            "this fixture is only interesting while the marginal intervals overlap"
        );
        assert!(
            study.verdict().contains("TRADEOFF, MEASURED"),
            "a consistent, real quality gap must be stated as a measured trade: {}",
            study.verdict()
        );
    }

    /// A checkpoint must resume real work and never invent it. Two failure modes matter and they
    /// pull in opposite directions: losing finished tasks wastes money, and *trusting* a bad file
    /// would put rows into a published result that nobody measured. So a valid entry is reused
    /// verbatim, and anything unreadable is ignored rather than allowed to fail — or fake — a run.
    #[test]
    fn a_checkpoint_resumes_real_work_and_ignores_a_corrupt_file() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("fp-ckpt-{}.jsonl", std::process::id()));

        let row = vec![o(true, true, 0.01), o(true, true, 0.10)];
        append_checkpoint(Some(&p), "mbpp-1", &row);
        append_checkpoint(Some(&p), "mbpp-2", &row);
        let loaded = load_checkpoint(Some(&p));
        assert_eq!(loaded.len(), 2);
        let back = &loaded["mbpp-1"];
        assert_eq!(back.len(), 2);
        assert!(
            (back[0].cost_usd - 0.01).abs() < 1e-12 && back[0].oracle_correct,
            "a resumed row must be the measurement, not a default"
        );

        // A truncated final line — exactly what an interrupted write leaves behind — must cost
        // only that row, not the whole file.
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        writeln!(f, "{{\"task_id\": \"mbpp-3\", \"rungs\": [{{\"gate_sc").unwrap();
        drop(f);
        assert_eq!(
            load_checkpoint(Some(&p)).len(),
            2,
            "a torn line is skipped; the intact rows survive"
        );

        std::fs::remove_file(&p).ok();
        assert!(
            load_checkpoint(Some(&p)).is_empty() && load_checkpoint(None).is_empty(),
            "a missing checkpoint just means measure everything"
        );
    }

    /// The judged arm must not publish a $/success while the judge's own calls are unmetered.
    /// Showing one would make adding a judge look free — the same fabricated-cost failure the
    /// unpriced-rung validator exists to stop, just moved into the benchmark.
    #[test]
    fn the_judged_arm_withholds_a_cost_it_cannot_account_for() {
        let matrix = vec![vec![oj(true, false, 0.01, 0.2), oj(true, true, 0.10, 0.95)]];
        let study = replay(&matrix, &ladder(), "test");
        let judged = study.find("first-pass+judge").expect("judged arm present");
        assert!(!judged.cost_metered, "judge tokens are not counted yet");
        for p in study
            .policies
            .iter()
            .filter(|p| p.name != "first-pass+judge")
        {
            assert!(p.cost_metered, "{} must report a real cost", p.name);
        }
        let out = study.render();
        assert!(
            out.contains("cost withheld"),
            "the table must say why a cost is missing: {out}"
        );
    }

    /// The judged arm must not appear at all when no judge ran — a row identical to first-pass
    /// would imply an experiment nobody paid for.
    #[test]
    fn the_judged_arm_is_absent_without_a_judge() {
        let study = replay(
            &[vec![o(true, true, 0.01), o(true, true, 0.10)]],
            &ladder(),
            "t",
        );
        assert!(study.find("first-pass+judge").is_none());
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
