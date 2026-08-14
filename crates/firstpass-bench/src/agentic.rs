//! Agentic multi-turn MBPP: produce **real** trajectories, not synthetic ones.
//!
//! # Why this exists
//!
//! [`crate::multiturn`] scores a trajectory-informed router against a baseline, but every dataset in
//! this repo is single-shot: one prompt, one answer, no conversation. A single-shot task cannot
//! exercise a feature whose entire input is *what happened on previous turns* — it scores
//! `DifficultyHint::None` by construction.
//!
//! This module closes that gap the only honest way: run a genuine agentic loop. The model writes
//! code, the sandbox runs the visible tests, and **real failures come back as real `tool_result`
//! errors** in the next turn's message array. The trajectory signals are then *observed* from a
//! conversation that actually happened, rather than hand-assembled to match what the scorer expects.
//!
//! That distinction is the whole value. A synthetic trajectory tests that the arithmetic works; a
//! real one tests whether the signal exists in production traffic at all. Only the second can
//! honestly clear a pre-registered bar.
//!
//! # What it costs, and the checkpoint
//!
//! Every turn is a paid model call, so a run is worth real money and a crash partway through must
//! not destroy it. Each finished task is appended to a checkpoint immediately and a resumed run
//! skips what it already bought — the same discipline (and the same hard-won reason) as
//! [`crate::coding_policy`]'s checkpoint, which exists because a transient provider error at task
//! 898 of 974 once destroyed every call before it.
//!
//! # The honesty constraint on the loop
//!
//! The model may see the **visible** tests — that is the gate an operator would write, and hiding it
//! would model a router nobody runs. It never sees the **hidden** oracle. Turn `n`'s routing
//! decision may only use signals from turns `< n`, which the runner enforces structurally by
//! computing signals before the turn is served rather than after.

use serde::{Deserialize, Serialize};

use crate::coding::{CodingTask, suite_score};
use crate::coding_policy::RungOutcome;
use crate::multiturn::{MultiTurnTask, Turn};
use crate::sandbox::{Limits, Sandbox};
use firstpass_core::features::TrajectorySignals;

/// A ladder rung: a priced model to call.
#[derive(Debug, Clone)]
pub struct AgenticRung {
    /// Priced model id, e.g. `anthropic/claude-haiku-4-5`.
    pub model: String,
    /// Bare model name for the provider call.
    pub bare: String,
    /// USD per input token.
    pub in_price: f64,
    /// USD per output token.
    pub out_price: f64,
}

/// One turn as it actually happened, persisted so a run can be re-studied offline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedTurn {
    /// Trajectory signals visible **before** this turn was routed.
    pub tool_errors: u32,
    pub tool_results: u32,
    pub assistant_turns: u32,
    pub repeated_tool_calls: u32,
    /// Per-rung outcome for this turn, cheapest first.
    pub rungs: Vec<RecordedRung>,
}

/// What one rung produced on one turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedRung {
    pub model: String,
    pub gate_pass: bool,
    pub oracle_correct: bool,
    pub cost_usd: f64,
    /// Fraction of visible tests passed — the continuous gate score.
    pub gate_score: f64,
}

/// One task's full recorded trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedTask {
    pub id: String,
    pub turns: Vec<RecordedTurn>,
}

impl RecordedTask {
    /// Convert to the shape [`crate::multiturn::evaluate`] scores.
    #[must_use]
    pub fn to_multiturn(&self) -> MultiTurnTask {
        MultiTurnTask {
            id: self.id.clone(),
            turns: self
                .turns
                .iter()
                .map(|t| Turn {
                    signals: TrajectorySignals {
                        tool_errors: t.tool_errors,
                        tool_results: t.tool_results,
                        assistant_turns: t.assistant_turns,
                        repeated_tool_calls: t.repeated_tool_calls,
                    },
                    rungs: t
                        .rungs
                        .iter()
                        .map(|r| RungOutcome {
                            gate_score: r.gate_score,
                            gate_full_pass: r.gate_pass,
                            oracle_correct: r.oracle_correct,
                            cost_usd: r.cost_usd,
                            judge_score: None,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

/// Signals read off a conversation that actually happened.
///
/// Mirrors the proxy's extraction semantics deliberately: depth over the whole conversation, errors
/// and repetition over a recent window. If the bench measured different signals than the product
/// computes, the benchmark would be scoring a feature nobody ships — the exact class of mismatch
/// that made `DifficultyHint::High` unreachable in production while its unit test stayed green.
fn signals_from(history: &[TurnRecord]) -> TrajectorySignals {
    const WINDOW: usize = 6;
    let recent = &history[history.len().saturating_sub(WINDOW)..];
    let mut repeats = 0u32;
    let mut seen: Vec<&str> = Vec::new();
    for t in recent {
        if seen.contains(&t.code.as_str()) {
            repeats += 1;
        } else {
            seen.push(&t.code);
        }
    }
    TrajectorySignals {
        tool_errors: u32::try_from(recent.iter().filter(|t| !t.passed).count()).unwrap_or(u32::MAX),
        tool_results: u32::try_from(recent.len()).unwrap_or(u32::MAX),
        // Depth is a whole-conversation property (ADR 0012), not a windowed one.
        assistant_turns: u32::try_from(history.len()).unwrap_or(u32::MAX),
        repeated_tool_calls: repeats,
    }
}

/// One completed attempt in the conversation.
struct TurnRecord {
    code: String,
    passed: bool,
    feedback: String,
}

/// Build the Anthropic message array for the next turn from the conversation so far.
///
/// Prior attempts appear as assistant turns and their test results as user turns, so the model sees
/// what it already tried and why it failed. This is what makes the trajectory real: the failures in
/// the transcript are the sandbox's actual output, not a description of failure.
fn build_messages(task: &CodingTask, history: &[TurnRecord]) -> serde_json::Value {
    let mut msgs = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "Write a Python function `{}` that satisfies:\n\n{}\n\nIt must pass these tests:\n{}\n\n\
             Output ONLY the function definition and any imports it needs. No prose, no fences.",
            task.entrypoint,
            task.prompt,
            task.visible_cases.join("\n"),
        ),
    })];
    for t in history {
        msgs.push(serde_json::json!({ "role": "assistant", "content": t.code }));
        msgs.push(serde_json::json!({
            "role": "user",
            "content": format!(
                "Error: the tests failed.\n{}\nFix the function and output the corrected \
                 definition only.",
                t.feedback
            ),
        }));
    }
    serde_json::Value::Array(msgs)
}

/// Strip markdown fences a model adds despite instructions. Cheap and forgiving: a fenced answer is
/// a formatting slip, and scoring it as a code failure would fabricate gate error.
fn strip_fences(text: &str) -> String {
    let t = text.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t.to_owned();
    };
    let body = rest.strip_prefix("python").unwrap_or(rest);
    body.trim_start_matches('\n')
        .rsplit_once("```")
        .map_or(body, |(before, _)| before)
        .to_owned()
}

/// How the runner reaches a model: `(bare_model, messages) -> (text, in_tokens, out_tokens)`.
///
/// A callback rather than a concrete client so the loop is testable without a network, and so the
/// same loop can drive any provider.
pub type ModelCall<'a> =
    dyn Fn(&str, &serde_json::Value, u32) -> Result<(String, u64, u64), String> + 'a;

/// Invoke the model callback at a specific token budget.
fn call_with_budget(
    call: &ModelCall<'_>,
    bare: &str,
    msgs: &serde_json::Value,
    budget: u32,
) -> Result<(String, u64, u64), String> {
    call(bare, msgs, budget)
}

/// Outcome of running one agentic task.
#[derive(Debug)]
pub struct TaskRun {
    pub recorded: RecordedTask,
    pub spent_usd: f64,
}

/// Run one task as a multi-turn agentic conversation across the ladder.
///
/// For each turn, every rung is called with the same conversation prefix, so the comparison between
/// rungs is paired: the difference is the model, not the context it saw.
///
/// # Errors
/// Any provider or sandbox failure, verbatim. A failed call aborts the task rather than being
/// scored as a wrong answer — recording infrastructure faults as model failures would understate
/// every policy equally and quietly.
pub fn run_task(
    task: &CodingTask,
    rungs: &[AgenticRung],
    sb: &dyn Sandbox,
    limits: &Limits,
    max_turns: usize,
    call: &ModelCall<'_>,
) -> Result<TaskRun, String> {
    let mut history: Vec<TurnRecord> = Vec::new();
    let mut turns: Vec<RecordedTurn> = Vec::new();
    let mut spent = 0.0f64;

    for _turn in 0..max_turns {
        // Signals are computed BEFORE the turn is served. A router may only use what it could have
        // seen at decision time; computing them after would leak the outcome into the decision.
        let sig = signals_from(&history);
        let msgs = build_messages(task, &history);

        let mut recorded_rungs = Vec::new();
        let mut cheapest_pass: Option<TurnRecord> = None;

        for rung in rungs {
            // An empty response earns more budget, not an abort — the same policy `LiveSolver`
            // already uses, and for the same hard-won reason: treating it as fatal once killed a
            // run at task 31 of 974 and threw away every call before it. This runner rediscovered
            // that at task 5. Out of budgets, the candidate is scored as a failure, which is what
            // it is: a model that produced no answer did not produce a correct one. Anything else
            // (auth, 4xx, decode) still aborts, because more room cannot fix it.
            let mut last = String::new();
            let mut got: Option<(String, u64, u64)> = None;
            for budget in [2048u32, 4096, 8192] {
                match call_with_budget(call, &rung.bare, &msgs, budget) {
                    Ok(v) => {
                        got = Some(v);
                        break;
                    }
                    Err(e) if e.contains("no text content") => last = e,
                    Err(e) => return Err(e),
                }
            }
            let (text, in_tok, out_tok) = match got {
                Some(v) => v,
                None => {
                    eprintln!(
                        "WARNING: {} produced no text at any budget ({last}) on task {} — scoring                          it as a failed candidate. Many of these in one run mean the ladder is                          broken, not the models.",
                        rung.model, task.id
                    );
                    (String::new(), 0, 0)
                }
            };
            let cost = in_tok as f64 * rung.in_price + out_tok as f64 * rung.out_price;
            spent += cost;
            let code = strip_fences(&text);

            let (vis_pass, vis_total) = suite_score(sb, task, &code, &task.visible_cases, limits)?;
            let (hid_pass, hid_total) = suite_score(sb, task, &code, &task.hidden_cases, limits)?;
            let gate_pass = vis_total > 0 && vis_pass == vis_total;
            let gate_score = if vis_total == 0 {
                0.0
            } else {
                vis_pass as f64 / vis_total as f64
            };

            recorded_rungs.push(RecordedRung {
                model: rung.model.clone(),
                gate_pass,
                // The ORACLE decides correctness, never the gate. Conflating them is what makes a
                // benchmark unable to measure its own gate's error.
                oracle_correct: hid_total > 0 && hid_pass == hid_total,
                cost_usd: cost,
                gate_score,
            });

            if cheapest_pass.is_none() {
                cheapest_pass = Some(TurnRecord {
                    code: code.clone(),
                    passed: gate_pass,
                    feedback: format!("{vis_pass} of {vis_total} visible tests passed."),
                });
            }
        }

        turns.push(RecordedTurn {
            tool_errors: sig.tool_errors,
            tool_results: sig.tool_results,
            assistant_turns: sig.assistant_turns,
            repeated_tool_calls: sig.repeated_tool_calls,
            rungs: recorded_rungs,
        });

        // The conversation continues only while the CHEAP rung is still failing — that is what an
        // agent retry loop actually looks like, and it is what generates the failing tool results
        // the trajectory signal is meant to read.
        match cheapest_pass {
            Some(t) if !t.passed => history.push(t),
            _ => break,
        }
    }

    Ok(TaskRun {
        recorded: RecordedTask {
            id: task.id.clone(),
            turns,
        },
        spent_usd: spent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(code: &str, passed: bool) -> TurnRecord {
        TurnRecord {
            code: code.to_owned(),
            passed,
            feedback: String::new(),
        }
    }

    /// An empty history is no signal — the first turn of any conversation, and the state a
    /// single-shot request is permanently in.
    #[test]
    fn the_first_turn_carries_no_signal() {
        let s = signals_from(&[]);
        assert_eq!(s.tool_results, 0);
        assert_eq!(s.assistant_turns, 0);
        assert_eq!(
            firstpass_core::features::DifficultyHint::score(s),
            firstpass_core::features::DifficultyHint::None
        );
    }

    /// Depth counts the whole conversation while errors count only the recent window — the same
    /// split the proxy uses (ADR 0012). If the bench windowed depth too, `High` would be
    /// unreachable here exactly as it was in production, and the benchmark would silently be
    /// measuring a three-level feature.
    #[test]
    fn depth_is_whole_conversation_and_errors_are_windowed() {
        let hist: Vec<TurnRecord> = (0..20).map(|i| rec(&format!("code{i}"), i < 14)).collect();
        let s = signals_from(&hist);
        assert_eq!(
            s.assistant_turns, 20,
            "depth must see the whole conversation"
        );
        assert!(
            s.tool_results <= 6,
            "errors must be windowed, got {} results",
            s.tool_results
        );
        assert_eq!(
            firstpass_core::features::DifficultyHint::score(s),
            firstpass_core::features::DifficultyHint::High,
            "a long, recently-all-failing conversation must reach the top bucket"
        );
    }

    /// Repeated identical code is the signal an error count alone cannot see: a model re-emitting
    /// the same wrong answer is stuck in a way that a fresh wrong answer is not.
    #[test]
    fn identical_repeated_attempts_are_counted() {
        let hist = vec![rec("same", false), rec("same", false), rec("same", false)];
        assert_eq!(signals_from(&hist).repeated_tool_calls, 2);
        let varied = vec![rec("a", false), rec("b", false), rec("c", false)];
        assert_eq!(signals_from(&varied).repeated_tool_calls, 0);
    }

    /// Failures must appear in the transcript as real error text, or the trajectory the model sees
    /// is not the one the signals describe.
    #[test]
    fn prior_failures_reach_the_next_turns_messages() {
        let task = CodingTask {
            id: "t".into(),
            prompt: "add two numbers".into(),
            entrypoint: "add".into(),
            visible_cases: vec!["assert add(1,2)==3".into()],
            hidden_cases: vec!["assert add(2,2)==4".into()],
            unit_test: None,
        };
        let hist = vec![rec("def add(a,b): return 0", false)];
        let msgs = build_messages(&task, &hist);
        let arr = msgs.as_array().expect("messages must be an array");
        assert_eq!(arr.len(), 3, "prompt + failed attempt + its result");
        assert_eq!(arr[1]["role"], "assistant");
        assert_eq!(arr[2]["role"], "user");
        assert!(
            arr[2]["content"]
                .as_str()
                .unwrap_or_default()
                .starts_with("Error:"),
            "the failure must read as an error, which is what the extractor keys on"
        );
        // And the hidden oracle must never leak into what the model sees.
        let whole = msgs.to_string();
        assert!(
            !whole.contains("add(2,2)==4"),
            "hidden cases must never appear in the conversation"
        );
    }

    /// **An empty model response must not abort a paid run.**
    ///
    /// This killed the first 974-task attempt at task 5, throwing away everything bought before it.
    /// `LiveSolver` already had the right policy — escalate the token budget, then score the
    /// candidate as failed — documented with the note that it once killed a run at task 31 of 974.
    /// I did not carry it into this runner and rediscovered the same failure from the other end.
    ///
    /// A model that returns nothing did not return a correct answer, so scoring it as a failed
    /// candidate is accurate, not lenient. Aborting instead converts one bad response into the loss
    /// of every call before it.
    #[test]
    fn an_empty_response_escalates_budget_then_scores_a_failure_rather_than_aborting() {
        use std::cell::RefCell;
        let budgets_seen = RefCell::new(Vec::new());
        // Always empty: forces every budget to be tried, then the degrade path.
        let call = |_bare: &str, _m: &serde_json::Value, budget: u32| {
            budgets_seen.borrow_mut().push(budget);
            Err("no text content in response".to_owned())
        };

        let task = CodingTask {
            id: "t".into(),
            prompt: "p".into(),
            entrypoint: "f".into(),
            visible_cases: vec!["assert f()==1".into()],
            hidden_cases: vec!["assert f()==1".into()],
            unit_test: None,
        };
        let rungs = vec![AgenticRung {
            model: "m".into(),
            bare: "m".into(),
            in_price: 0.0,
            out_price: 0.0,
        }];
        let out = run_task(&task, &rungs, &NeverRuns, &Limits::default(), 1, &call);
        let run = out.expect("an empty response must NOT abort the run");
        assert!(
            budgets_seen.borrow().len() >= 2,
            "the budget must escalate before giving up, saw {:?}",
            budgets_seen.borrow()
        );
        let r = &run.recorded.turns[0].rungs[0];
        assert!(
            !r.gate_pass && !r.oracle_correct,
            "no answer is not a correct answer"
        );
    }

    /// A hard error (auth, 4xx) must still abort: more tokens cannot fix a bad key, and grinding
    /// through 974 tasks against a broken credential would spend nothing useful and report a
    /// catastrophic-looking result that is really a config problem.
    #[test]
    fn a_hard_error_still_aborts() {
        let call = |_b: &str, _m: &serde_json::Value, _bud: u32| {
            Err("HTTP 401: invalid x-api-key".to_owned())
        };
        let task = CodingTask {
            id: "t".into(),
            prompt: "p".into(),
            entrypoint: "f".into(),
            visible_cases: vec!["assert f()==1".into()],
            hidden_cases: vec!["assert f()==1".into()],
            unit_test: None,
        };
        let rungs = vec![AgenticRung {
            model: "m".into(),
            bare: "m".into(),
            in_price: 0.0,
            out_price: 0.0,
        }];
        let err = run_task(&task, &rungs, &NeverRuns, &Limits::default(), 1, &call)
            .expect_err("an auth failure must abort");
        assert!(err.contains("401"), "the cause must survive: {err}");
    }

    /// Sandbox stub: these tests never reach execution (the model never produces code).
    struct NeverRuns;
    impl Sandbox for NeverRuns {
        fn runtime(&self) -> &str {
            "never"
        }
        fn run(
            &self,
            _u: &crate::sandbox::ExecUnit,
            _l: &Limits,
        ) -> Result<crate::sandbox::ExecOutcome, crate::sandbox::SandboxError> {
            Ok(crate::sandbox::ExecOutcome::Completed {
                exit_code: 1,
                stdout: String::new(),
                stderr: String::new(),
            })
        }
    }

    /// Models fence code despite instructions. Scoring a formatting slip as a code failure would
    /// fabricate gate error, which is the one thing a gate-error benchmark must not do.
    #[test]
    fn fenced_code_is_unwrapped() {
        assert_eq!(
            strip_fences("```python\ndef f(): pass\n```"),
            "def f(): pass\n"
        );
        assert_eq!(strip_fences("```\ndef f(): pass\n```"), "def f(): pass\n");
        assert_eq!(strip_fences("def f(): pass"), "def f(): pass");
    }
}
