//! VRBench: the harness a **third-party** router runs against ([`specs/vrbench-v1.md`]).
//!
//! # Why this module is separate from the rest of the crate
//!
//! Everything else in `firstpass-bench` measures Firstpass. This measures *any* router, including
//! ones that are not cascades and are not written in Rust, by defining a data contract rather than
//! a code interface: a router reads a task file and writes a submission file. That is deliberate —
//! a Rust trait would exclude every router in the field, which are overwhelmingly Python.
//!
//! # The two things existing benchmarks cannot express
//!
//! **An executable oracle separate from the gate.** Tasks carry two disjoint check sets. The
//! router may read the `visible` set — that is its gate, the thing a real operator would write.
//! The `hidden` set never reaches it and is run by the harness after the router has committed.
//! Without that split, gate *error* is unmeasurable: nothing distinguishes an answer the gate
//! wrongly passed from one it rightly passed.
//!
//! **A cost the router itself reports.** A cascade that escalates has paid for the rejected
//! attempt as well as the served one. RouterArena's format derives cost from the single model
//! named per query, so a cascade cannot state what it actually spent. Here the router reports
//! `cost_usd` plus a per-attempt breakdown, and [`Submission::validate`] enforces that they agree —
//! which is what makes under-reporting detectable by a third party rather than a matter of trust.

use serde::{Deserialize, Serialize};

/// Format major version. A harness refuses data it does not understand rather than guessing.
pub const VRBENCH_VERSION: u32 = 1;

/// How a check set is executed. A harness that meets an unknown kind must fail, never skip: a
/// silently skipped task is a silently inflated score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CheckKind {
    /// Python asserts run under pytest.
    Pytest,
    /// JavaScript tests run under jest.
    Jest,
    /// Rust tests run under `cargo test`.
    CargoTest,
    /// Output validated against a JSON Schema.
    JsonSchema,
    /// Output compared verbatim to an expected string.
    ExactMatch,
}

/// One check set — either the router-visible gate or the harness-only oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckSet {
    /// Executor for these cases.
    pub kind: CheckKind,
    /// The cases themselves (asserts, schema, or expected output).
    pub cases: Vec<String>,
    /// Symbol the cases call, where the domain has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entrypoint: Option<String>,
}

/// One benchmark task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Format version.
    pub vrbench_version: u32,
    /// Stable, globally unique, namespaced by source dataset.
    pub id: String,
    /// e.g. `code/python`.
    pub domain: String,
    /// Verbatim prompt given to the model.
    pub prompt: String,
    /// The gate. The router MAY read this.
    pub visible: CheckSet,
    /// The oracle. The router MUST NOT read this; the harness runs it after the router commits.
    pub hidden: CheckSet,
    /// Free-form, used only for slicing results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<String>,
}

/// What the router's gate said about one attempt. Recorded so gate error is measurable against the
/// oracle; a router with no gate reports `Ungated`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateVerdict {
    /// The gate accepted this attempt.
    Pass,
    /// The gate rejected it.
    Fail,
    /// The gate could not form an opinion (runner missing, timeout).
    Abstain,
    /// This router does not gate — a single-shot or pre-judgment router.
    Ungated,
}

/// One model call the router made and paid for, **including** calls whose output it discarded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    /// Priced model id, e.g. `anthropic/claude-haiku-4-5`.
    pub model: String,
    /// USD for this call.
    pub cost_usd: f64,
    /// What the router's own gate said about it.
    pub gate_verdict: GateVerdict,
}

/// One line of a submission: what the router served for one task, and what it cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submission {
    /// Format version.
    pub vrbench_version: u32,
    /// Task this answers.
    pub id: String,
    /// The served output, which the harness runs the hidden set against.
    pub answer: String,
    /// Total USD, including discarded attempts.
    pub cost_usd: f64,
    /// Every attempt, in order. Non-empty even for single-shot routers.
    pub attempts: Vec<Attempt>,
    /// Wall-clock for the whole routing decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

/// Tolerance when checking that `cost_usd` equals the attempt sum. Wide enough for float
/// accumulation over a handful of attempts, far tighter than the cost of one dropped call.
const COST_EPSILON: f64 = 1e-9;

impl Submission {
    /// Enforce the submission rules from the spec.
    ///
    /// The arithmetic check is the load-bearing one. A cascade that pays for three generations and
    /// reports one is not approximating — it is under-reporting on exactly the queries it found
    /// hard, which is where its cost advantage would otherwise evaporate. Because the total and the
    /// breakdown are both required, any third party can recompute the sum and catch it. That is the
    /// property RouterArena's format cannot offer at any level of good faith, since it has nowhere
    /// to record a discarded attempt at all.
    ///
    /// # Errors
    /// Unknown version, empty attempts, negative cost, or a total that disagrees with the sum.
    pub fn validate(&self) -> Result<(), String> {
        if self.vrbench_version != VRBENCH_VERSION {
            return Err(format!(
                "{}: vrbench_version {} not supported (this harness implements {})",
                self.id, self.vrbench_version, VRBENCH_VERSION
            ));
        }
        // Rule 4: even a single-shot router reports one attempt, so escalation rate is derived
        // from the record rather than self-declared.
        if self.attempts.is_empty() {
            return Err(format!(
                "{}: attempts must be non-empty — a single-shot router reports one attempt",
                self.id
            ));
        }
        // `!is_finite()` rather than `< 0.0`: NaN fails EVERY comparison, so a NaN cost passes a
        // negativity test AND passes the sum check below (|NaN| > eps is false), landing in the
        // report as a silently poisoned total. Infinity is rejected for the same reason.
        if !self.cost_usd.is_finite() || self.attempts.iter().any(|a| !a.cost_usd.is_finite()) {
            return Err(format!(
                "{}: cost must be finite (NaN and infinity are rejected — NaN would pass every \
                 comparison below and poison the aggregate)",
                self.id
            ));
        }
        if self.cost_usd < 0.0 || self.attempts.iter().any(|a| a.cost_usd < 0.0) {
            return Err(format!("{}: negative cost", self.id));
        }
        // Rule 3: the total and the breakdown must agree. Preferring one silently would make the
        // other decorative.
        let summed: f64 = self.attempts.iter().map(|a| a.cost_usd).sum();
        if (summed - self.cost_usd).abs() > COST_EPSILON {
            return Err(format!(
                "{}: cost_usd {:.9} disagrees with the sum of attempts {:.9} — every attempt, \
                 including discarded ones, must be counted",
                self.id, self.cost_usd, summed
            ));
        }
        Ok(())
    }

    /// Whether the router paid for more than one model call on this task.
    #[must_use]
    pub fn escalated(&self) -> bool {
        self.attempts.len() > 1
    }

    /// The gate's verdict on the attempt that was actually served — the last one.
    #[must_use]
    pub fn served_verdict(&self) -> GateVerdict {
        self.attempts
            .last()
            .map_or(GateVerdict::Ungated, |a| a.gate_verdict)
    }
}

/// Per-task outcome after the harness has run the hidden set.
#[derive(Debug, Clone)]
pub struct Scored {
    /// Task id.
    pub id: String,
    /// Whether the served answer passed the full hidden (oracle) set.
    pub oracle_pass: bool,
    /// What the router's own gate said about the served attempt.
    pub gate: GateVerdict,
    /// Total USD the router reported.
    pub cost_usd: f64,
    /// Whether more than one attempt was paid for.
    pub escalated: bool,
}

/// The aggregate report (spec §5).
#[derive(Debug, Clone)]
pub struct Report {
    /// Tasks scored.
    pub n: usize,
    /// Fraction whose served answer passed the oracle.
    pub success: f64,
    /// Total USD over successful tasks.
    pub usd_per_success: f64,
    /// Fraction of served answers that failed the oracle.
    pub served_failure: f64,
    /// Of answers the router's gate passed, the fraction the oracle failed.
    pub gate_false_accept: f64,
    /// Of answers the router's gate failed, the fraction the oracle passed.
    pub gate_false_reject: f64,
    /// Fraction of tasks paying for more than one attempt.
    pub escalation_rate: f64,
    /// Total USD across all tasks.
    pub total_cost_usd: f64,
    /// RouterBench's AIQ, when baselines were supplied (spec §5).
    ///
    /// `None` is the honest answer without them, and the report says so rather than printing a
    /// number computed from a hull the router itself defines. AIQ is measured against the
    /// **Zero Router** — the probabilistic mix of the raw models — so it needs each model's
    /// solo (cost, quality) point. A single router's submission contains its own path and nothing
    /// about what any model would have scored alone, so AIQ is not derivable from it.
    pub aiq: Option<AiqBlock>,
}

/// AIQ and the bar it is measured against.
#[derive(Debug, Clone)]
pub struct AiqBlock {
    /// AIQ of the Zero Router — the raw models' own hull.
    pub zero_router: f64,
    /// AIQ with the router's point added.
    pub with_router: f64,
    /// Shared cost domain both were integrated over.
    pub domain: (f64, f64),
}

impl AiqBlock {
    /// Lift over the Zero Router. Positive means the router beat the bar RouterBench reports no
    /// learned router significantly clearing.
    #[must_use]
    pub fn lift(&self) -> f64 {
        self.with_router - self.zero_router
    }
}

/// One baseline: what a single model scored alone, across the same task set.
///
/// Supplied as its own single-shot submission per model, so a baseline is produced by exactly the
/// same path as a router result and cannot be asserted rather than measured.
#[derive(Debug, Clone)]
pub struct Baseline {
    /// Model id.
    pub model: String,
    /// Mean USD per task.
    pub cost_per_task: f64,
    /// Fraction of tasks it solved alone.
    pub quality: f64,
}

/// Compute AIQ for a router against model baselines, exactly as RouterBench defines it.
///
/// Reuses [`crate::routerbench`] rather than reimplementing the hull, so VRBench's AIQ and the one
/// this repo reports elsewhere cannot drift apart.
#[must_use]
pub fn aiq_against(
    baselines: &[Baseline],
    router_cost: f64,
    router_quality: f64,
) -> Option<AiqBlock> {
    // Two models are the minimum for an interpolation to exist at all; with fewer there is no
    // Zero Router and therefore no bar.
    if baselines.len() < 2 {
        return None;
    }
    let pts: Vec<(f64, f64)> = baselines
        .iter()
        .map(|b| (b.cost_per_task, b.quality))
        .collect();
    let mut with = pts.clone();
    with.push((router_cost, router_quality));

    let all: Vec<f64> = with.iter().map(|p| p.0).collect();
    let c_min = all.iter().copied().fold(f64::INFINITY, f64::min);
    let c_max = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !(c_min.is_finite() && c_max.is_finite()) || c_max <= c_min {
        return None;
    }
    Some(AiqBlock {
        zero_router: crate::routerbench::aiq(
            &crate::routerbench::non_decreasing_hull(&pts),
            c_min,
            c_max,
        ),
        with_router: crate::routerbench::aiq(
            &crate::routerbench::non_decreasing_hull(&with),
            c_min,
            c_max,
        ),
        domain: (c_min, c_max),
    })
}

/// Aggregate scored outcomes into the reported metrics.
///
/// The two gate-error rates have no counterpart in any existing router benchmark, and exist only
/// because the visible/hidden split makes them observable.
#[must_use]
pub fn report(scored: &[Scored]) -> Report {
    let n = scored.len();
    let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f64 / b as f64 };

    let ok = scored.iter().filter(|s| s.oracle_pass).count();
    let total: f64 = scored.iter().map(|s| s.cost_usd).sum();

    // Gate error is measured only over attempts the gate actually judged. An ungated or abstaining
    // router contributes to neither rate — averaging it in as though it were a pass would credit a
    // router for a decision it never made.
    let passed: Vec<&Scored> = scored
        .iter()
        .filter(|s| s.gate == GateVerdict::Pass)
        .collect();
    let failed: Vec<&Scored> = scored
        .iter()
        .filter(|s| s.gate == GateVerdict::Fail)
        .collect();

    Report {
        n,
        success: ratio(ok, n),
        usd_per_success: if ok == 0 {
            f64::INFINITY
        } else {
            total / ok as f64
        },
        served_failure: ratio(n - ok, n),
        gate_false_accept: ratio(
            passed.iter().filter(|s| !s.oracle_pass).count(),
            passed.len(),
        ),
        gate_false_reject: ratio(
            failed.iter().filter(|s| s.oracle_pass).count(),
            failed.len(),
        ),
        escalation_rate: ratio(scored.iter().filter(|s| s.escalated).count(), n),
        total_cost_usd: total,
        aiq: None,
    }
}

/// [`report`], plus AIQ measured against model baselines (spec §5).
#[must_use]
pub fn report_with_baselines(scored: &[Scored], baselines: &[Baseline]) -> Report {
    let mut r = report(scored);
    let n = scored.len().max(1) as f64;
    r.aiq = aiq_against(baselines, r.total_cost_usd / n, r.success);
    r
}

/// Parse a submission file, rejecting the whole file if any line is invalid.
///
/// All-or-nothing on purpose: scoring the valid subset of a submission would quietly change the
/// task set a result is reported over, which is the difference between a benchmark and a claim.
///
/// # Errors
/// Any malformed line, or any line failing [`Submission::validate`].
pub fn parse_submissions(text: &str) -> Result<Vec<Submission>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let s: Submission =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        s.validate()?;
        out.push(s);
    }
    if out.is_empty() {
        return Err("submission is empty".to_owned());
    }
    Ok(out)
}

/// Render the report as Markdown.
#[must_use]
pub fn render(r: &Report) -> String {
    let mut s = String::new();
    s.push_str(&format!("## VRBench v{VRBENCH_VERSION} results\n\n"));
    s.push_str(&format!("n = {} tasks\n\n", r.n));
    s.push_str("| metric | value |\n|---|---|\n");
    s.push_str(&format!("| success | {:.4} |\n", r.success));
    s.push_str(&format!("| $/success | ${:.5} |\n", r.usd_per_success));
    s.push_str(&format!("| total $ | ${:.4} |\n", r.total_cost_usd));
    s.push_str(&format!("| served-failure | {:.4} |\n", r.served_failure));
    s.push_str(&format!(
        "| gate false-accept | {:.4} |\n",
        r.gate_false_accept
    ));
    s.push_str(&format!(
        "| gate false-reject | {:.4} |\n",
        r.gate_false_reject
    ));
    s.push_str(&format!(
        "| escalation rate | {:.0}% |\n",
        r.escalation_rate * 100.0
    ));
    match &r.aiq {
        Some(a) => {
            s.push_str(&format!(
                "\n| curve | AIQ |\n|---|---|\n| Zero Router (models only) | {:.4} |\n\
                 | + this router | {:.4} |\n\n**AIQ lift: {:+.4}** over the cost domain \
                 ${:.5}..${:.5}.\n",
                a.zero_router,
                a.with_router,
                a.lift(),
                a.domain.0,
                a.domain.1
            ));
        }
        None => s.push_str(
            "\nAIQ: **not computed** — it is measured against the Zero Router, the probabilistic \
             mix of the raw models, so it needs each model's solo (cost, quality) point. Supply \
             single-model baseline submissions to obtain it. Reporting a number without them \
             would score the router against a hull it defined itself.\n",
        ),
    }
    s.push_str(
        "\nThe two gate-error rates exist only because tasks carry a router-visible gate set and a \
         harness-only oracle set. A benchmark with one check set can measure neither.\n",
    );
    s
}

// -- scoring: run the HIDDEN set against what the router served ---------------------------------

use crate::sandbox::{ExecOutcome, ExecUnit, Limits, Sandbox};

/// Run a task's **hidden** set against a served answer.
///
/// This is the step the router never participates in. It happens after the router has committed,
/// in a network-free sandbox, because the answer is model-generated code from an untrusted source
/// — a harness that executes it on the host is a remote-code-execution vector wearing an
/// evaluation costume.
///
/// Returns whether the answer passed **every** hidden case. Partial credit is deliberately not
/// reported here: the oracle is ground truth, and a partially-correct program is a wrong answer.
///
/// # Errors
/// Sandbox failure — never a candidate failure, which is simply a `false`. The distinction matters:
/// scoring an infrastructure fault as a failed task would silently understate every router.
pub fn score_hidden(
    sb: &dyn Sandbox,
    task: &Task,
    answer: &str,
    limits: &Limits,
) -> Result<bool, String> {
    if task.hidden.kind != CheckKind::Pytest {
        return Err(format!(
            "{}: harness implements only the `pytest` check kind so far, task needs `{:?}` — \
             refusing rather than skipping, because a skipped task is an inflated score",
            task.id, task.hidden.kind
        ));
    }
    let unit = ExecUnit {
        files: vec![
            ("solution.py".to_owned(), answer.to_owned()),
            (
                "fp_runner.py".to_owned(),
                crate::coding::build_runner(&task.hidden.cases),
            ),
        ],
        command: "python3 fp_runner.py".to_owned(),
    };
    match sb.run(&unit, limits) {
        Ok(ExecOutcome::Completed { stdout, .. }) => {
            let (passed, total) =
                crate::coding::parse_score(&stdout).unwrap_or((0, task.hidden.cases.len()));
            Ok(total > 0 && passed == total)
        }
        // A timeout or a crash is the candidate's failure, not the harness's.
        Ok(_) => Ok(false),
        Err(e) => Err(format!("{}: sandbox failed: {e}", task.id)),
    }
}

/// Score a whole submission against the task set.
///
/// Every submitted task must exist in the task file and every task must be answered: a submission
/// covering a subset would silently change the population a published number refers to.
///
/// # Errors
/// Coverage mismatch, or any sandbox failure.
pub fn score_all(
    sb: &dyn Sandbox,
    tasks: &[Task],
    subs: &[Submission],
    limits: &Limits,
) -> Result<Vec<Scored>, String> {
    use std::collections::HashMap;
    let by_id: HashMap<&str, &Task> = tasks.iter().map(|t| (t.id.as_str(), t)).collect();

    // Length equality is not coverage: a submission with one id twice and one task missing has
    // the right count and the wrong population. Check the id SET.
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for s in subs {
        if !seen.insert(s.id.as_str()) {
            return Err(format!("submission answers `{}` more than once", s.id));
        }
    }
    let missing: Vec<&str> = tasks
        .iter()
        .map(|t| t.id.as_str())
        .filter(|id| !seen.contains(id))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "submission is missing {} of {} tasks (e.g. {}) — score the whole set or none, \
             otherwise the reported population is not the benchmark",
            missing.len(),
            tasks.len(),
            missing
                .iter()
                .take(3)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut out = Vec::with_capacity(subs.len());
    for s in subs {
        let task = by_id
            .get(s.id.as_str())
            .ok_or_else(|| format!("submission names unknown task `{}`", s.id))?;
        out.push(Scored {
            id: s.id.clone(),
            oracle_pass: score_hidden(sb, task, &s.answer, limits)?,
            gate: s.served_verdict(),
            cost_usd: s.cost_usd,
            escalated: s.escalated(),
        });
    }
    Ok(out)
}

/// Parse a task file, refusing any line the harness cannot faithfully score.
///
/// # Errors
/// Malformed JSON, an unsupported format major, or an empty file.
pub fn parse_tasks(text: &str) -> Result<Vec<Task>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let t: Task = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        if t.vrbench_version != VRBENCH_VERSION {
            return Err(format!(
                "line {}: vrbench_version {} not supported (this harness implements {})",
                i + 1,
                t.vrbench_version,
                VRBENCH_VERSION
            ));
        }
        out.push(t);
    }
    if out.is_empty() {
        return Err("task file is empty".to_owned());
    }
    Ok(out)
}

// -- reference router: produces a submission others can copy ------------------------------------

use crate::coding::{CandidateSolver, CodingTask};
use firstpass_core::PriceTable;

/// Run a verified cascade over VRBench tasks and produce a [`Submission`] per task.
///
/// This is the reference participant, and it exists for two reasons. It gives VRBench a worked
/// example an implementer can copy — the spec describes a data contract, and a contract with no
/// reference implementation is a document people interpret differently. And it is how Firstpass
/// itself enters its own benchmark, on identical footing to anyone else: same task file, same
/// submission format, same cost arithmetic that [`Submission::validate`] checks.
///
/// **The gate runs the task's `visible` cases only.** The `hidden` set is never read here — it is
/// the harness's, and reading it would be the cheat the whole split exists to prevent. That is
/// enforced structurally: this function is handed the visible cases and builds its gate from them,
/// so there is no path by which the oracle could leak into the routing decision.
///
/// A ladder of one model produces a single-shot, ungated submission — which is exactly what an AIQ
/// **baseline** is, so baselines are generated by this same code path rather than asserted.
///
/// # Errors
/// Any solver or sandbox failure, verbatim. A failed call aborts rather than being recorded as a
/// wrong answer, because scoring an infrastructure fault as a model failure understates the router.
pub fn run_reference_router(
    tasks: &[Task],
    solvers: &[(String, &dyn CandidateSolver)],
    sb: &dyn Sandbox,
    prices: &PriceTable,
    limits: &Limits,
) -> Result<Vec<Submission>, String> {
    if solvers.is_empty() {
        return Err("need at least one rung".to_owned());
    }
    let mut out = Vec::with_capacity(tasks.len());

    for task in tasks {
        let mut attempts = Vec::new();
        let mut answer = String::new();
        let single = solvers.len() == 1;

        for (i, (model, solver)) in solvers.iter().enumerate() {
            // The gate sees only `visible`. `task.hidden` is deliberately not in scope here.
            // The router is ENTITLED to the visible cases — that is what "the gate is the
            // operator's own suite" means — so it puts them in the model's prompt. Sending the
            // bare spec instead leaves the model to invent a function name, the gate then fails
            // every candidate on a naming mismatch rather than on correctness, and the cascade
            // escalates 100% of the time. Measured: exactly that, on the first 3-task run.
            //
            // This mirrors how `dataset::load_coding_dataset` builds its prompt, deliberately: a
            // reference participant should not be handicapped relative to the harness's own
            // loader, or the benchmark measures prompt construction rather than routing.
            let model_prompt = format!(
                "{}\n\nYour solution must pass these tests:\n{}\n\nWrite the solution to \
                 `solution.py` defining the required function(s).",
                task.prompt,
                task.visible.cases.join("\n")
            );
            let ct = CodingTask {
                id: task.id.clone(),
                prompt: model_prompt,
                entrypoint: "solution.py".to_owned(),
                visible_cases: task.visible.cases.clone(),
                hidden_cases: Vec::new(),
                unit_test: None,
            };
            let sol = solver
                .solve(&ct)
                .map_err(|e| format!("{}: solver failed on {model}: {e}", task.id))?;
            let cost = prices
                .get(model)
                .map_or(0.0, |p| p.cost(sol.in_tokens, sol.out_tokens));
            answer = sol.code.clone();

            // A single-rung ladder is a baseline: it does not gate, and says so rather than
            // reporting a verdict it never formed.
            let verdict = if single {
                GateVerdict::Ungated
            } else {
                match gate_on_visible(sb, task, &sol.code, limits) {
                    Ok(true) => GateVerdict::Pass,
                    Ok(false) => GateVerdict::Fail,
                    Err(_) => GateVerdict::Abstain,
                }
            };
            attempts.push(Attempt {
                model: model.clone(),
                cost_usd: cost,
                gate_verdict: verdict,
            });
            // Serve on a pass; an abstain escalates rather than being read as approval, matching
            // the fail-closed default a wrong answer deserves.
            if verdict != GateVerdict::Fail && verdict != GateVerdict::Abstain {
                break;
            }
            if i + 1 == solvers.len() {
                break; // top rung: serve it regardless, there is nothing above.
            }
        }

        let total: f64 = attempts.iter().map(|a| a.cost_usd).sum();
        let s = Submission {
            vrbench_version: VRBENCH_VERSION,
            id: task.id.clone(),
            answer,
            cost_usd: total,
            attempts,
            latency_ms: None,
        };
        // Validate as we go: a router that cannot produce a valid submission should fail here,
        // not at scoring time after a whole run has been paid for.
        s.validate()?;
        out.push(s);
    }
    Ok(out)
}

/// Run a task's **visible** cases — the gate the router is entitled to see.
fn gate_on_visible(
    sb: &dyn Sandbox,
    task: &Task,
    code: &str,
    limits: &Limits,
) -> Result<bool, String> {
    let unit = ExecUnit {
        files: vec![
            ("solution.py".to_owned(), code.to_owned()),
            (
                "fp_runner.py".to_owned(),
                crate::coding::build_runner(&task.visible.cases),
            ),
        ],
        command: "python3 fp_runner.py".to_owned(),
    };
    match sb.run(&unit, limits) {
        Ok(ExecOutcome::Completed { stdout, .. }) => {
            let (p, t) =
                crate::coding::parse_score(&stdout).unwrap_or((0, task.visible.cases.len()));
            Ok(t > 0 && p == t)
        }
        Ok(_) => Ok(false),
        Err(e) => Err(format!("{}: gate sandbox failed: {e}", task.id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn att(model: &str, cost: f64, v: GateVerdict) -> Attempt {
        Attempt {
            model: model.to_owned(),
            cost_usd: cost,
            gate_verdict: v,
        }
    }

    fn task(id: &str) -> Task {
        Task {
            vrbench_version: VRBENCH_VERSION,
            id: id.to_owned(),
            domain: "code/python".to_owned(),
            prompt: "p".to_owned(),
            visible: CheckSet {
                kind: CheckKind::Pytest,
                cases: vec!["f() == 1".into()],
                entrypoint: None,
            },
            hidden: CheckSet {
                kind: CheckKind::Pytest,
                cases: vec!["f() == 1".into()],
                entrypoint: None,
            },
            difficulty: None,
        }
    }

    fn sub(id: &str, cost: f64, attempts: Vec<Attempt>) -> Submission {
        Submission {
            vrbench_version: VRBENCH_VERSION,
            id: id.to_owned(),
            answer: "def f(): pass".to_owned(),
            cost_usd: cost,
            attempts,
            latency_ms: None,
        }
    }

    /// The rule the whole format exists for: a cascade cannot report only the serving model's
    /// price. Under-reporting is what would let a cascade claim a cost advantage it did not earn,
    /// and it is exactly what RouterArena's format cannot even detect.
    #[test]
    fn a_cascade_cannot_hide_the_cost_of_a_discarded_attempt() {
        let honest = sub(
            "t1",
            0.11,
            vec![
                att("cheap", 0.01, GateVerdict::Fail),
                att("dear", 0.10, GateVerdict::Pass),
            ],
        );
        assert!(honest.validate().is_ok());

        // Same two calls, but the total names only the served one.
        let understated = sub(
            "t1",
            0.10,
            vec![
                att("cheap", 0.01, GateVerdict::Fail),
                att("dear", 0.10, GateVerdict::Pass),
            ],
        );
        let err = understated
            .validate()
            .expect_err("under-reported cost must be rejected");
        assert!(err.contains("disagrees"), "got {err}");
    }

    /// Rule 4: a single-shot router still reports one attempt, so escalation rate is derived from
    /// the record rather than taken on the router's word.
    #[test]
    fn a_single_shot_router_still_reports_its_one_attempt() {
        let empty = sub("t1", 0.0, vec![]);
        assert!(empty.validate().is_err(), "empty attempts must be rejected");

        let one = sub("t1", 0.05, vec![att("m", 0.05, GateVerdict::Ungated)]);
        assert!(one.validate().is_ok());
        assert!(!one.escalated(), "one attempt is not an escalation");
    }

    /// A file is scored whole or not at all: silently dropping bad lines would change the task set
    /// a published number refers to.
    #[test]
    fn one_bad_line_rejects_the_whole_submission() {
        let good = serde_json::to_string(&sub("t1", 0.05, vec![att("m", 0.05, GateVerdict::Pass)]))
            .unwrap();
        let bad = serde_json::to_string(&sub("t2", 0.99, vec![att("m", 0.05, GateVerdict::Pass)]))
            .unwrap();
        assert!(parse_submissions(&format!("{good}\n{good}")).is_ok());
        let err = parse_submissions(&format!("{good}\n{bad}")).expect_err("must reject the file");
        assert!(
            err.contains("t2"),
            "error must name the offending task: {err}"
        );
    }

    /// Gate error is only defined over attempts the gate actually judged. An ungated router must
    /// not be credited with a perfect gate it never ran.
    #[test]
    fn an_ungated_router_contributes_to_neither_gate_error_rate() {
        let scored = vec![
            Scored {
                id: "a".into(),
                oracle_pass: false,
                gate: GateVerdict::Ungated,
                cost_usd: 0.01,
                escalated: false,
            },
            Scored {
                id: "b".into(),
                oracle_pass: true,
                gate: GateVerdict::Ungated,
                cost_usd: 0.01,
                escalated: false,
            },
        ];
        let r = report(&scored);
        assert!((r.gate_false_accept - 0.0).abs() < 1e-9);
        assert!((r.gate_false_reject - 0.0).abs() < 1e-9);
        assert!((r.success - 0.5).abs() < 1e-9, "success is still measured");
    }

    /// The measurement the visible/hidden split exists to make possible.
    #[test]
    fn gate_error_is_measured_against_the_oracle() {
        let scored = vec![
            // gate passed it, oracle failed it — a wrong answer served.
            Scored {
                id: "a".into(),
                oracle_pass: false,
                gate: GateVerdict::Pass,
                cost_usd: 0.01,
                escalated: false,
            },
            Scored {
                id: "b".into(),
                oracle_pass: true,
                gate: GateVerdict::Pass,
                cost_usd: 0.01,
                escalated: false,
            },
            // gate failed it, oracle would have passed — a needless escalation.
            Scored {
                id: "c".into(),
                oracle_pass: true,
                gate: GateVerdict::Fail,
                cost_usd: 0.11,
                escalated: true,
            },
        ];
        let r = report(&scored);
        assert!(
            (r.gate_false_accept - 0.5).abs() < 1e-9,
            "1 of 2 passes was wrong"
        );
        assert!(
            (r.gate_false_reject - 1.0).abs() < 1e-9,
            "the only rejection was correct"
        );
        assert!((r.escalation_rate - 1.0 / 3.0).abs() < 1e-9);
    }

    /// NaN fails every comparison, so a NaN cost passes both the negativity test and the sum
    /// check and lands in the aggregate as a silent poison. Caught by the PR reviewer.
    #[test]
    fn a_non_finite_cost_is_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let s = sub("t1", bad, vec![att("m", bad, GateVerdict::Pass)]);
            let err = s
                .validate()
                .expect_err("non-finite cost must be rejected, got acceptance for {bad}");
            assert!(err.contains("finite"), "got {err}");
        }
        // Sanity: the same shape with a finite cost is fine.
        assert!(
            sub("t1", 0.05, vec![att("m", 0.05, GateVerdict::Pass)])
                .validate()
                .is_ok()
        );
    }

    /// Coverage is about the SET of ids, not the count. A submission answering one task twice
    /// while omitting another has the right length and the wrong population.
    #[test]
    fn a_duplicate_id_cannot_stand_in_for_a_missing_task() {
        let tasks = vec![task("a"), task("b")];
        let subs = vec![
            sub("a", 0.01, vec![att("m", 0.01, GateVerdict::Pass)]),
            sub("a", 0.01, vec![att("m", 0.01, GateVerdict::Pass)]),
        ];
        // Length matches (2 == 2) but `b` is unanswered.
        // Panicking stub: coverage is validated BEFORE any task is executed, so this must never
        // be called. If it is, the ordering regressed and money/time would be spent on a
        // submission that was already invalid.
        struct NeverRuns;
        impl Sandbox for NeverRuns {
            fn runtime(&self) -> &str {
                "never"
            }
            fn run(
                &self,
                _u: &ExecUnit,
                _l: &Limits,
            ) -> Result<ExecOutcome, crate::sandbox::SandboxError> {
                panic!("coverage must be checked before anything is executed")
            }
        }
        let sb = NeverRuns;
        let err = score_all(&sb, &tasks, &subs, &Limits::default())
            .expect_err("duplicate id with a missing task must be rejected");
        assert!(
            err.contains("more than once") || err.contains("missing"),
            "got {err}"
        );
    }

    /// Spec §5 requires AIQ. It is measured against the Zero Router, so it needs the models' solo
    /// points — which a single router's submission cannot contain. Absent baselines the report
    /// must say so rather than invent a hull.
    #[test]
    fn aiq_is_withheld_without_baselines_and_computed_with_them() {
        let scored = vec![Scored {
            id: "a".into(),
            oracle_pass: true,
            gate: GateVerdict::Pass,
            cost_usd: 0.003,
            escalated: false,
        }];
        let plain = report(&scored);
        assert!(plain.aiq.is_none(), "no baselines ⇒ no AIQ");
        assert!(
            render(&plain).contains("not computed"),
            "the report must say why AIQ is absent"
        );

        let baselines = vec![
            Baseline {
                model: "cheap".into(),
                cost_per_task: 0.001,
                quality: 0.78,
            },
            Baseline {
                model: "top".into(),
                cost_per_task: 0.007,
                quality: 0.94,
            },
        ];
        let with = report_with_baselines(&scored, &baselines);
        let a = with.aiq.clone().expect("baselines ⇒ AIQ");
        assert!(a.zero_router > 0.0 && a.with_router > 0.0);
        assert!(render(&with).contains("AIQ lift"));
    }

    /// One baseline is not an interpolation, so there is no Zero Router and no bar.
    #[test]
    fn a_single_baseline_yields_no_bar() {
        let one = vec![Baseline {
            model: "only".into(),
            cost_per_task: 0.001,
            quality: 0.8,
        }];
        assert!(aiq_against(&one, 0.002, 0.9).is_none());
    }

    /// Version skew must fail loudly rather than be interpreted optimistically.
    #[test]
    fn an_unknown_format_version_is_refused() {
        let mut s = sub("t1", 0.05, vec![att("m", 0.05, GateVerdict::Pass)]);
        s.vrbench_version = 99;
        assert!(s.validate().is_err());
    }
}
