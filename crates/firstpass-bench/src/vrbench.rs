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
    }
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
    s.push_str(
        "\nThe two gate-error rates exist only because tasks carry a router-visible gate set and a \
         harness-only oracle set. A benchmark with one check set can measure neither.\n",
    );
    s
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

    /// Version skew must fail loudly rather than be interpreted optimistically.
    #[test]
    fn an_unknown_format_version_is_refused() {
        let mut s = sub("t1", 0.05, vec![att("m", 0.05, GateVerdict::Pass)]);
        s.vrbench_version = 99;
        assert!(s.validate().is_err());
    }
}
