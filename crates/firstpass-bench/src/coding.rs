//! Coding-with-tests benchmark (Batch 3b) — the domain where the best practical gate is STILL
//! imperfect, so the conformal served-failure bound is *meaningful* (unlike self-checking
//! arithmetic, which has a zero-error gate and degenerate conformal — proven in Batch 1).
//!
//! A candidate model writes code for a task. The **visible** tests are the gate (real coverage gaps
//! → genuine false-accept / false-reject); the **hidden** oracle tests are ground truth. Every
//! candidate runs in the fail-closed sandbox (ADR 0002 — untrusted, model-generated code). Conformal
//! calibrates on `(gate_pass, oracle_correct)` pairs, where the gate's error is now real.
//!
//! Cost note: unlike the arithmetic harness, this measures each task **once** (one candidate call,
//! two sandbox runs) rather than re-running per policy — candidate calls and sandbox execs are
//! expensive, so we record the (gate, oracle) pair directly and derive the numbers from it.
//!
//! Gate-score note: this first cut uses a **binary** gate (visible suite passes or not), so its
//! conformal score is `{0, 1}`. Conformal can then only choose "serve every gate-pass" or "serve
//! none" — it cannot exclude individual false-accepts, so it needs enough samples for the Hoeffding
//! bound to absorb the empirical failure rate. A **continuous** gate score (fraction of visible
//! test cases passed, or a judge score) would let conformal actually tune the threshold; that is the
//! natural next refinement.

use crate::conformal::{self, ConformalResult};
use crate::sandbox::{ExecUnit, Limits, Sandbox};
use std::collections::HashMap;

/// A coding task: prompt, the file the candidate writes, and two test suites.
#[derive(Debug, Clone)]
pub struct CodingTask {
    /// Stable id, e.g. `"is_prime"`.
    pub id: String,
    /// Natural-language spec shown to the candidate.
    pub prompt: String,
    /// File the candidate's code is written to, e.g. `"solution.py"`.
    pub entrypoint: String,
    /// The **gate**: a Python script that imports the solution and asserts (exit 0 = pass). Has
    /// deliberate coverage gaps, so passing it does not guarantee correctness.
    pub visible_tests: String,
    /// The **oracle** (ground truth): a thorough Python script; exit 0 = actually correct.
    pub hidden_tests: String,
}

/// Candidate code plus its token cost.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The candidate's code for [`CodingTask::entrypoint`].
    pub code: String,
    /// Input tokens billed (0 for the mock solver).
    pub in_tokens: u64,
    /// Output tokens billed (0 for the mock solver).
    pub out_tokens: u64,
}

/// Produces candidate code for a task. [`MockSolver`] (offline, deterministic) or [`LiveSolver`]
/// (Anthropic, opt-in, costs tokens).
pub trait CandidateSolver {
    /// Solve one task.
    ///
    /// # Errors
    /// A solver failure aborts the whole run — we refuse to publish partial numbers (mirrors the
    /// arithmetic live path).
    fn solve(&self, task: &CodingTask) -> Result<Solution, String>;
}

/// One task's measured pair: did the gate pass, and was it actually correct?
#[derive(Debug, Clone)]
pub struct TaskOutcome {
    /// Task id.
    pub id: String,
    /// Visible (gate) tests passed.
    pub gate_pass: bool,
    /// Hidden (oracle) tests passed — ground truth.
    pub oracle_correct: bool,
    /// Input tokens billed.
    pub in_tokens: u64,
    /// Output tokens billed.
    pub out_tokens: u64,
}

/// Aggregate result of a coding-with-tests run.
#[derive(Debug, Clone)]
pub struct CodingReport {
    /// Isolation tier the sandbox ran under (e.g. `"gvisor"`, `"runc"`).
    pub runtime_tier: String,
    /// Number of tasks.
    pub n: usize,
    /// Fraction of tasks the candidate got actually correct (per the oracle).
    pub oracle_pass_rate: f64,
    /// Gate **false-accept** rate: `P(gate passes | oracle incorrect)` — the coverage-gap leakage
    /// that makes conformal meaningful. Zero for a perfect gate (arithmetic); non-zero here.
    pub gate_false_accept_rate: f64,
    /// Gate **false-reject** rate: `P(gate fails | oracle correct)`.
    pub gate_false_reject_rate: f64,
    /// Served-failure if you serve whenever the gate passes, with no conformal threshold:
    /// `P(incorrect | gate passes)`. What conformal is there to bound.
    pub served_failure_rate: f64,
    /// Conformal threshold + its calibrated served-failure bound (needs `n >= min_n` to be
    /// `feasible`; a small demo suite will be infeasible — that's honest, size up for a real run).
    pub conformal: ConformalResult,
    /// Per-task detail.
    pub outcomes: Vec<TaskOutcome>,
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// Run one test suite against candidate code in the sandbox; `true` iff it exits 0. A sandbox error
/// (or non-zero exit) is a fail — never a panic, never a host fallback.
fn suite_passes(
    sb: &dyn Sandbox,
    task: &CodingTask,
    code: &str,
    tests: &str,
    limits: &Limits,
) -> bool {
    let unit = ExecUnit {
        files: vec![
            (task.entrypoint.clone(), code.to_owned()),
            ("fp_tests.py".to_owned(), tests.to_owned()),
        ],
        command: "python3 fp_tests.py".to_owned(),
    };
    sb.run(&unit, limits)
        .map(|o| o.is_success())
        .unwrap_or(false)
}

/// Evaluate one task: solve, then run the visible (gate) and hidden (oracle) suites in the sandbox.
///
/// # Errors
/// Propagates a solver failure so the caller can refuse to publish.
pub fn evaluate_task(
    sb: &dyn Sandbox,
    solver: &dyn CandidateSolver,
    task: &CodingTask,
    limits: &Limits,
) -> Result<TaskOutcome, String> {
    let sol = solver.solve(task)?;
    let gate_pass = suite_passes(sb, task, &sol.code, &task.visible_tests, limits);
    let oracle_correct = suite_passes(sb, task, &sol.code, &task.hidden_tests, limits);
    Ok(TaskOutcome {
        id: task.id.clone(),
        gate_pass,
        oracle_correct,
        in_tokens: sol.in_tokens,
        out_tokens: sol.out_tokens,
    })
}

/// Run the full coding-with-tests benchmark. Aborts (`Err`) on any solver failure — a corrupted
/// candidate would corrupt the measurement, so we refuse to publish partial numbers.
///
/// # Errors
/// The first solver failure, verbatim.
pub fn run_coding_benchmark(
    tasks: &[CodingTask],
    solver: &dyn CandidateSolver,
    sb: &dyn Sandbox,
    alpha: f64,
    delta: f64,
    min_n: usize,
) -> Result<CodingReport, String> {
    let limits = Limits::default();
    let mut outcomes = Vec::with_capacity(tasks.len());
    for t in tasks {
        outcomes.push(evaluate_task(sb, solver, t, &limits)?);
    }
    Ok(summarize(
        sb.runtime().to_owned(),
        outcomes,
        alpha,
        delta,
        min_n,
    ))
}

/// Derive the gate error, served-failure, and conformal bound from measured `(gate, oracle)` pairs.
/// Pure — no sandbox, no model — so the metric math is unit-testable without Docker.
#[must_use]
pub fn summarize(
    runtime_tier: String,
    outcomes: Vec<TaskOutcome>,
    alpha: f64,
    delta: f64,
    min_n: usize,
) -> CodingReport {
    let n = outcomes.len();
    let incorrect = outcomes.iter().filter(|o| !o.oracle_correct).count();
    let correct = n - incorrect;
    let false_accept = outcomes
        .iter()
        .filter(|o| o.gate_pass && !o.oracle_correct)
        .count();
    let false_reject = outcomes
        .iter()
        .filter(|o| !o.gate_pass && o.oracle_correct)
        .count();
    let served = outcomes.iter().filter(|o| o.gate_pass).count();

    // Conformal on binary gate scores: serving-when-gate-passes; the bound is on P(incorrect|served).
    let pairs: Vec<(f64, bool)> = outcomes
        .iter()
        .map(|o| (f64::from(u8::from(o.gate_pass)), o.oracle_correct))
        .collect();
    let conformal = conformal::calibrate(&pairs, alpha, delta, min_n);

    CodingReport {
        runtime_tier,
        n,
        oracle_pass_rate: ratio(correct, n),
        gate_false_accept_rate: ratio(false_accept, incorrect),
        gate_false_reject_rate: ratio(false_reject, correct),
        served_failure_rate: ratio(false_accept, served),
        conformal,
        outcomes,
    }
}

// ---- solvers ----------------------------------------------------------------------------------

/// Deterministic offline solver: returns a canned solution per task id. For tests and dry runs.
#[derive(Debug, Clone)]
pub struct MockSolver {
    by_id: HashMap<String, String>,
}

impl MockSolver {
    /// Build from `(task_id, code)` pairs.
    #[must_use]
    pub fn new(solutions: Vec<(&str, &str)>) -> Self {
        Self {
            by_id: solutions
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect(),
        }
    }
}

impl CandidateSolver for MockSolver {
    fn solve(&self, task: &CodingTask) -> Result<Solution, String> {
        self.by_id
            .get(&task.id)
            .map(|code| Solution {
                code: code.clone(),
                in_tokens: 0,
                out_tokens: 0,
            })
            .ok_or_else(|| format!("no mock solution for task {:?}", task.id))
    }
}

/// Live solver: asks Anthropic to write the file. Opt-in (costs tokens); needs `ANTHROPIC_API_KEY`.
#[derive(Debug, Clone)]
pub struct LiveSolver {
    client: reqwest::blocking::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl LiveSolver {
    /// Build for a given candidate `model` (e.g. `"claude-haiku-4-5"`).
    #[must_use]
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: "https://api.anthropic.com".to_owned(),
            api_key,
            model,
        }
    }
}

impl CandidateSolver for LiveSolver {
    fn solve(&self, task: &CodingTask) -> Result<Solution, String> {
        let system = "You are a Python coding assistant. Output ONLY the full contents of the \
                      requested file — no explanation, no markdown code fences.";
        let (text, in_tokens, out_tokens) = crate::live::anthropic_call(
            &self.client,
            &self.base_url,
            &self.api_key,
            &self.model,
            Some(system),
            &task.prompt,
            1024,
        )?;
        Ok(Solution {
            code: strip_fences(&text),
            in_tokens,
            out_tokens,
        })
    }
}

/// Strip a leading ```` ```lang ```` / trailing ```` ``` ```` markdown fence if the model added one.
fn strip_fences(s: &str) -> String {
    let t = s.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t.to_owned();
    };
    // Drop the rest of the opening fence line (an optional language tag), then a trailing fence.
    let body = rest.split_once('\n').map_or("", |(_, b)| b);
    body.trim_end()
        .strip_suffix("```")
        .unwrap_or(body)
        .trim_end()
        .to_owned()
}

// ---- embedded demo suite ----------------------------------------------------------------------

/// A small coding suite with **deliberate visible-test coverage gaps**, so a real candidate can pass
/// the gate while failing the oracle (a genuine false-accept). Enough to exercise and verify the
/// pipeline end-to-end; a real conformal bound needs a much larger live run (size `n` accordingly).
#[must_use]
pub fn coding_suite() -> Vec<CodingTask> {
    fn task(id: &str, prompt: &str, visible: &str, hidden: &str) -> CodingTask {
        CodingTask {
            id: id.to_owned(),
            prompt: prompt.to_owned(),
            entrypoint: "solution.py".to_owned(),
            visible_tests: visible.to_owned(),
            hidden_tests: hidden.to_owned(),
        }
    }
    vec![
        task(
            "is_prime",
            "Write solution.py defining `is_prime(n: int) -> bool` returning True iff n is prime.",
            "from solution import is_prime\nassert is_prime(2) and is_prime(3) and is_prime(13)\nassert not is_prime(4) and not is_prime(9)\nprint('ok')\n",
            // gap in visible: n < 2 is never tested.
            "from solution import is_prime\nassert not is_prime(0) and not is_prime(1) and not is_prime(-7)\nassert is_prime(2) and is_prime(7919)\nprint('ok')\n",
        ),
        task(
            "count_vowels",
            "Write solution.py defining `count_vowels(s: str) -> int` counting vowels (a,e,i,o,u), case-insensitive.",
            "from solution import count_vowels\nassert count_vowels('hello') == 2\nassert count_vowels('xyz') == 0\nprint('ok')\n",
            // gap in visible: uppercase vowels never tested.
            "from solution import count_vowels\nassert count_vowels('AEIOU') == 5\nassert count_vowels('Hello World') == 3\nprint('ok')\n",
        ),
        task(
            "reverse_string",
            "Write solution.py defining `reverse_string(s: str) -> str` returning s reversed.",
            "from solution import reverse_string\nassert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\nprint('ok')\n",
            "from solution import reverse_string\nassert reverse_string('racecar') == 'racecar'\nassert reverse_string('ab cd') == 'dc ba'\nprint('ok')\n",
        ),
        task(
            "list_sum",
            "Write solution.py defining `list_sum(xs: list[int]) -> int` returning the sum of xs.",
            "from solution import list_sum\nassert list_sum([1,2,3]) == 6\nassert list_sum([]) == 0\nprint('ok')\n",
            "from solution import list_sum\nassert list_sum([-1,-2,-3]) == -6\nassert list_sum([100]) == 100\nprint('ok')\n",
        ),
        task(
            "gcd",
            "Write solution.py defining `gcd(a: int, b: int) -> int` returning the greatest common divisor.",
            "from solution import gcd\nassert gcd(12, 8) == 4\nassert gcd(7, 1) == 1\nprint('ok')\n",
            "from solution import gcd\nassert gcd(0, 5) == 5\nassert gcd(270, 192) == 6\nprint('ok')\n",
        ),
        task(
            "factorial",
            "Write solution.py defining `factorial(n: int) -> int` returning n! (with factorial(0)==1).",
            "from solution import factorial\nassert factorial(0) == 1\nassert factorial(5) == 120\nprint('ok')\n",
            "from solution import factorial\nassert factorial(1) == 1\nassert factorial(6) == 720\nprint('ok')\n",
        ),
    ]
}

/// Canned solutions matching [`coding_suite`], with a **known** mix: two false-accepts (pass the
/// gappy gate, fail the oracle), one true-reject (fails both), three correct. Used by the mock run
/// and the real-sandbox pipeline test.
#[must_use]
pub fn mock_solutions() -> MockSolver {
    MockSolver::new(vec![
        // FALSE ACCEPT: is_prime that returns True for n < 2 (untested by the visible gate).
        (
            "is_prime",
            "def is_prime(n):\n    d = 2\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 1\n    return True\n",
        ),
        // FALSE ACCEPT: count_vowels that ignores uppercase (untested by the visible gate).
        (
            "count_vowels",
            "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
        ),
        // CORRECT.
        (
            "reverse_string",
            "def reverse_string(s):\n    return s[::-1]\n",
        ),
        // CORRECT.
        ("list_sum", "def list_sum(xs):\n    return sum(xs)\n"),
        // CORRECT.
        (
            "gcd",
            "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        ),
        // TRUE REJECT: broken factorial — the gate correctly catches it (fails visible too).
        ("factorial", "def factorial(n):\n    return 0\n"),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str, gate: bool, oracle: bool) -> TaskOutcome {
        TaskOutcome {
            id: id.to_owned(),
            gate_pass: gate,
            oracle_correct: oracle,
            in_tokens: 0,
            out_tokens: 0,
        }
    }

    /// The known mix from `mock_solutions`: 3 correct, 2 false-accept, 1 true-reject.
    #[test]
    fn summarize_computes_gate_error_and_served_failure() {
        let outs = vec![
            outcome("is_prime", true, false),      // false accept
            outcome("count_vowels", true, false),  // false accept
            outcome("factorial", false, false),    // true reject
            outcome("reverse_string", true, true), // correct
            outcome("list_sum", true, true),       // correct
            outcome("gcd", true, true),            // correct
        ];
        let r = summarize("fake".to_owned(), outs, 0.1, 0.05, 50);
        assert_eq!(r.n, 6);
        assert!((r.oracle_pass_rate - 0.5).abs() < 1e-9);
        // false accepts among the 3 incorrect = 2/3.
        assert!((r.gate_false_accept_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((r.gate_false_reject_rate - 0.0).abs() < 1e-9);
        // served = 5 (all gate-passes); 2 of them are wrong → 0.4.
        assert!((r.served_failure_rate - 0.4).abs() < 1e-9);
    }

    /// The whole point of coding-with-tests: a gate with REAL false-accepts, unlike arithmetic.
    #[test]
    fn gate_has_nonzero_false_accept_unlike_arithmetic() {
        let outs = vec![outcome("a", true, false), outcome("b", true, true)];
        let r = summarize("fake".to_owned(), outs, 0.1, 0.05, 1);
        assert!(
            r.gate_false_accept_rate > 0.0,
            "coding gate must have genuine error to make conformal meaningful"
        );
    }

    /// With enough clean samples, conformal earns a bounded served-failure — the guarantee
    /// arithmetic could not (its gate had zero error, nothing to bound). NOTE: a binary tests-pass
    /// gate gives conformal only two thresholds (serve-all vs serve-none), so it cannot *exclude*
    /// individual false-accepts — it needs enough `n` for the Hoeffding bound to absorb the
    /// empirical rate under `alpha`. A continuous gate score (fraction of visible tests passed, or a
    /// judge score) would let conformal actually tune — the natural upgrade (see module docs).
    #[test]
    fn conformal_bounds_served_failure_on_coding_pairs() {
        let mut outs = vec![outcome("bad1", true, false), outcome("bad2", true, false)];
        for i in 0..400 {
            outs.push(outcome(&format!("ok{i}"), true, true)); // 402 served, 2 wrong ≈ 0.5%
        }
        for i in 0..20 {
            outs.push(outcome(&format!("rej{i}"), false, false)); // gate-rejects, score 0
        }
        let r = summarize("fake".to_owned(), outs, 0.1, 0.05, 50);
        assert!(r.served_failure_rate < 0.1);
        assert!(
            r.conformal.feasible,
            "402 served with 2 fails should clear the Hoeffding bound at alpha=0.10"
        );
    }

    /// A missing solution aborts the run — we never publish partial numbers.
    #[test]
    fn run_aborts_on_solver_error() {
        struct NeverSandbox;
        impl Sandbox for NeverSandbox {
            fn runtime(&self) -> &str {
                "never"
            }
            fn run(
                &self,
                _: &ExecUnit,
                _: &Limits,
            ) -> Result<crate::sandbox::ExecOutcome, crate::sandbox::SandboxError> {
                panic!("must not run: solver error should short-circuit before the sandbox");
            }
        }
        let tasks = vec![coding_suite().remove(0)];
        let solver = MockSolver::new(vec![]); // no solution for the task
        let err = run_coding_benchmark(&tasks, &solver, &NeverSandbox, 0.1, 0.05, 50).unwrap_err();
        assert!(err.contains("no mock solution"), "{err}");
    }

    /// The wiring end-to-end with a fake sandbox: gate/oracle verdicts flow into the report.
    #[test]
    fn run_wires_sandbox_verdicts_into_report() {
        // Fake: visible suite passes, hidden suite fails → a false accept.
        struct GapSandbox;
        impl Sandbox for GapSandbox {
            fn runtime(&self) -> &str {
                "gap"
            }
            fn run(
                &self,
                u: &ExecUnit,
                _: &Limits,
            ) -> Result<crate::sandbox::ExecOutcome, crate::sandbox::SandboxError> {
                let tests = &u.files.iter().find(|(p, _)| p == "fp_tests.py").unwrap().1;
                let code = if tests.contains("7919") { 1 } else { 0 }; // hidden suite (has 7919) fails
                Ok(crate::sandbox::ExecOutcome::Completed {
                    exit_code: code,
                    stdout: String::new(),
                    stderr: String::new(),
                })
            }
        }
        let tasks = vec![coding_suite().remove(0)]; // is_prime
        let solver = mock_solutions();
        let r = run_coding_benchmark(&tasks, &solver, &GapSandbox, 0.1, 0.05, 1).unwrap();
        assert_eq!(r.runtime_tier, "gap");
        assert_eq!(r.outcomes.len(), 1);
        assert!(r.outcomes[0].gate_pass && !r.outcomes[0].oracle_correct);
    }

    #[test]
    fn strip_fences_handles_plain_and_fenced() {
        assert_eq!(strip_fences("def f():\n    pass"), "def f():\n    pass");
        assert_eq!(
            strip_fences("```python\ndef f():\n    pass\n```"),
            "def f():\n    pass"
        );
        assert_eq!(strip_fences("```\nx = 1\n```"), "x = 1");
    }

    #[test]
    fn mock_solutions_cover_the_suite() {
        let solver = mock_solutions();
        for t in coding_suite() {
            assert!(solver.solve(&t).is_ok(), "missing mock for {}", t.id);
        }
    }

    // ---- real Docker pipeline (opt-in; needs a daemon; NO model spend — uses MockSolver) --------
    //   cargo test -p firstpass-bench --lib coding::tests::real_ -- --ignored --nocapture

    #[test]
    #[ignore = "requires a running container daemon"]
    fn real_pipeline_detects_a_true_false_accept() {
        use crate::sandbox::establish_sandbox;
        let sb = establish_sandbox("python:3.12-alpine").expect("sandbox");
        let suite = coding_suite();
        let solver = mock_solutions();
        let limits = Limits::default();

        // is_prime: buggy mock passes the gappy visible gate but fails the oracle — a REAL false
        // accept, proven by running real Python in the sandbox.
        let is_prime = suite.iter().find(|t| t.id == "is_prime").unwrap();
        let o = evaluate_task(sb.as_ref(), &solver, is_prime, &limits).expect("eval");
        assert!(
            o.gate_pass,
            "buggy is_prime should pass the gappy visible gate"
        );
        assert!(!o.oracle_correct, "buggy is_prime should fail the oracle");

        // reverse_string: correct mock passes both.
        let rev = suite.iter().find(|t| t.id == "reverse_string").unwrap();
        let o = evaluate_task(sb.as_ref(), &solver, rev, &limits).expect("eval");
        assert!(
            o.gate_pass && o.oracle_correct,
            "correct solution should pass both"
        );
    }
}
