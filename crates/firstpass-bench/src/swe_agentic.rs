//! Agentic SWE-bench: the workload trajectory routing was actually built for.
//!
//! # Why MBPP could not answer the question
//!
//! [`crate::agentic`] runs MBPP as a multi-turn loop and it works, but the measurement it produces
//! cannot decide anything. Measured over ~300 tasks: **~15% go multi-turn, ~20% of turns carry any
//! signal, and the maximum conversation depth observed is 2**. `DifficultyHint::deep` needs ≥ 8, so
//! `High` is unreachable by construction, and `Low` never occurs because the loop only continues
//! when the cheap rung fails. Two of four levels never fire. An MBPP task is one function; a model
//! that needs eight attempts at one function does not exist in that dataset.
//!
//! A benchmark where 80% of turns are decision-identical between the policies cannot produce an
//! informative paired CI — it will be narrow because the policies *agree*, not because the effect
//! is well-measured.
//!
//! # Why SWE-bench can
//!
//! A SWE-bench instance is a repository at a commit. Resolving one means reading a problem
//! statement, editing real files, running a real test suite, reading real failures, and trying
//! again. Eight turns is ordinary rather than pathological, so the full hint range is reachable and
//! the trajectory signal has something to be a signal *about*.
//!
//! # What this module adds
//!
//! [`crate::swebench`] already evaluates a **finished patch** against `FAIL_TO_PASS` /
//! `PASS_TO_PASS` in a fail-closed container (ADR 0010). It has no model call and no loop — it
//! scores work someone else did. This adds the agentic half: propose a patch, evaluate it, feed the
//! real failure back, retry. Every signal is observed from a conversation that happened.
//!
//! # Cost discipline
//!
//! Turns here are far more expensive than MBPP's: bigger prompts, bigger outputs, and a container
//! per evaluation. The driver enforces a hard budget before each instance and checkpoints every
//! finished one, because losing a paid run to a transient error is a mistake this repo has already
//! made twice.

use serde::{Deserialize, Serialize};

use crate::agentic::{ModelCall, RecordedRung, RecordedTask, RecordedTurn};
use crate::swebench::{SweInstance, SweLimits, evaluate};
use firstpass_core::features::TrajectorySignals;

/// A priced ladder rung.
#[derive(Debug, Clone)]
pub struct SweRung {
    /// Priced model id, e.g. `anthropic/claude-haiku-4-5`.
    pub model: String,
    /// Bare model name for the provider call.
    pub bare: String,
    /// USD per input token.
    pub in_price: f64,
    /// USD per output token.
    pub out_price: f64,
}

/// One attempt at an instance, as it happened.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Attempt {
    patch: String,
    resolved: bool,
    /// Did the patch even apply? A patch that does not apply is a different kind of failure from
    /// one that applies and fails tests, and the model needs to be told which it was.
    applied: bool,
    feedback: String,
}

/// Signals read off the attempt history.
///
/// Identical semantics to [`crate::agentic`] and to the proxy: depth over the whole conversation,
/// errors and repetition over a recent window. Three implementations must agree or the benchmark
/// measures something the product does not compute — the mismatch that made `High` unreachable in
/// production while its unit test stayed green.
fn signals_from(history: &[Attempt]) -> TrajectorySignals {
    const WINDOW: usize = 6;
    let recent = &history[history.len().saturating_sub(WINDOW)..];
    let mut repeats = 0u32;
    let mut seen: Vec<&str> = Vec::new();
    for a in recent {
        if seen.contains(&a.patch.as_str()) {
            repeats += 1;
        } else {
            seen.push(&a.patch);
        }
    }
    TrajectorySignals {
        tool_errors: u32::try_from(recent.iter().filter(|a| !a.resolved).count())
            .unwrap_or(u32::MAX),
        tool_results: u32::try_from(recent.len()).unwrap_or(u32::MAX),
        assistant_turns: u32::try_from(history.len()).unwrap_or(u32::MAX),
        repeated_tool_calls: repeats,
    }
}

/// Build the conversation for the next attempt.
///
/// Prior patches appear as assistant turns and their real test output as user turns. The model sees
/// what it tried and exactly how it failed, which is what makes the trajectory genuine rather than
/// a description of one.
fn build_messages(instance: &SweInstance, history: &[Attempt]) -> serde_json::Value {
    let mut msgs = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "Repository: {}\nCommit: {}\n\n## Problem\n\n{}\n\n\
             Produce a unified diff that fixes this. Output ONLY the diff, starting with \
             `diff --git` or `---`. No prose, no fences. Paths must be relative to the repository \
             root.",
            instance.repo, instance.base_commit, instance.problem_statement,
        ),
    })];
    for a in history {
        msgs.push(serde_json::json!({ "role": "assistant", "content": a.patch }));
        msgs.push(serde_json::json!({
            "role": "user",
            "content": format!("Error: that patch did not resolve the issue.\n{}\n\nTry again.", a.feedback),
        }));
    }
    serde_json::Value::Array(msgs)
}

/// Strip fences a model adds despite instructions. A fenced diff is a formatting slip; scoring it as
/// a failed patch would fabricate gate error, which a gate-error benchmark must never do.
fn strip_fences(text: &str) -> String {
    let t = text.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t.to_owned();
    };
    let body = rest.strip_prefix("diff").unwrap_or(rest);
    body.trim_start_matches('\n')
        .rsplit_once("```")
        .map_or(body, |(before, _)| before)
        .to_owned()
}

/// Outcome of running one instance.
#[derive(Debug)]
pub struct InstanceRun {
    /// Recorded turns, in the shape [`crate::multiturn::evaluate`] scores.
    pub recorded: RecordedTask,
    /// USD spent on model calls for this instance.
    pub spent_usd: f64,
}

/// Run one SWE-bench instance as a multi-turn agentic conversation across the ladder.
///
/// # Errors
/// Provider or evaluator failures, verbatim. An empty model response escalates the token budget and
/// then scores a failed candidate rather than aborting — the policy `LiveSolver` learned the hard
/// way and [`crate::agentic`] relearned. Anything else (auth, 4xx) aborts, because more tokens
/// cannot fix it.
pub fn run_instance(
    instance: &SweInstance,
    rungs: &[SweRung],
    limits: &SweLimits,
    max_turns: usize,
    call: &ModelCall<'_>,
) -> Result<InstanceRun, String> {
    let mut history: Vec<Attempt> = Vec::new();
    let mut turns: Vec<RecordedTurn> = Vec::new();
    let mut spent = 0.0f64;

    for _turn in 0..max_turns {
        // Computed BEFORE the turn is served: a routing decision may not see its own outcome.
        let sig = signals_from(&history);
        let msgs = build_messages(instance, &history);

        let mut recorded_rungs = Vec::new();
        let mut cheapest: Option<Attempt> = None;

        for rung in rungs {
            let mut last = String::new();
            let mut got: Option<(String, u64, u64)> = None;
            for budget in [4096u32, 8192] {
                match call(&rung.bare, &msgs, budget) {
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
                        "WARNING: {} produced no patch at any budget ({last}) on {} — scoring as a \
                         failed candidate.",
                        rung.model, instance.instance_id
                    );
                    (String::new(), 0, 0)
                }
            };
            spent += in_tok as f64 * rung.in_price + out_tok as f64 * rung.out_price;
            let cost = in_tok as f64 * rung.in_price + out_tok as f64 * rung.out_price;
            let patch = strip_fences(&text);

            let outcome = evaluate(instance, &patch, limits)?;
            // `resolved` is the ORACLE: every FAIL_TO_PASS now passes and every PASS_TO_PASS still
            // does. The gate is the weaker, cheaper check the router is allowed to see — whether
            // the patch applied and the targeted tests moved.
            let gate_pass = outcome.patch_applied && outcome.f2p.0 == outcome.f2p.1;
            let gate_score = if outcome.f2p.1 == 0 {
                0.0
            } else {
                outcome.f2p.0 as f64 / outcome.f2p.1 as f64
            };

            recorded_rungs.push(RecordedRung {
                model: rung.model.clone(),
                gate_pass,
                oracle_correct: outcome.resolved,
                cost_usd: cost,
                gate_score,
            });

            if cheapest.is_none() {
                cheapest = Some(Attempt {
                    patch: patch.clone(),
                    resolved: outcome.resolved,
                    applied: outcome.patch_applied,
                    feedback: if outcome.patch_applied {
                        format!(
                            "The patch applied. {} of {} target tests pass; {} of {} regression \
                             tests still pass.",
                            outcome.f2p.0, outcome.f2p.1, outcome.p2p.0, outcome.p2p.1
                        )
                    } else {
                        "The patch did not apply. Check the paths and context lines.".to_owned()
                    },
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

        match cheapest {
            Some(a) if !a.resolved => history.push(a),
            _ => break,
        }
    }

    Ok(InstanceRun {
        recorded: RecordedTask {
            id: instance.instance_id.clone(),
            turns,
        },
        spent_usd: spent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn att(patch: &str, resolved: bool) -> Attempt {
        Attempt {
            patch: patch.to_owned(),
            resolved,
            applied: true,
            feedback: String::new(),
        }
    }

    /// **The reason this module exists.** MBPP capped conversation depth at 2, so `High` was
    /// unreachable and two of four hint levels never fired. A repository task can genuinely run
    /// long, and the signal must reach the top bucket when it does — otherwise SWE-bench would
    /// inherit exactly the limitation it was chosen to escape.
    #[test]
    fn a_long_failing_repair_session_reaches_the_top_bucket() {
        let hist: Vec<Attempt> = (0..10).map(|i| att(&format!("patch{i}"), false)).collect();
        let s = signals_from(&hist);
        assert_eq!(s.assistant_turns, 10, "depth spans the whole session");
        assert_eq!(
            firstpass_core::features::DifficultyHint::score(s),
            firstpass_core::features::DifficultyHint::High,
            "ten failed repair attempts must be the top difficulty bucket"
        );
    }

    /// Depth is whole-session; errors are windowed. All three implementations (proxy, MBPP agentic,
    /// this) must agree, or the benchmark scores a feature the product does not compute.
    #[test]
    fn depth_is_whole_session_and_errors_are_windowed() {
        let mut hist: Vec<Attempt> = (0..12).map(|i| att(&format!("old{i}"), false)).collect();
        hist.extend((0..6).map(|i| att(&format!("new{i}"), true)));
        let s = signals_from(&hist);
        assert_eq!(s.assistant_turns, 18, "every attempt counts toward depth");
        assert_eq!(
            s.tool_errors, 0,
            "failures outside the window must not count: a session that recovered is not failing"
        );
    }

    /// A repeated identical patch is being stuck in a way a fresh wrong patch is not.
    #[test]
    fn repeated_identical_patches_are_detected() {
        let hist = vec![att("same", false), att("same", false), att("same", false)];
        assert_eq!(signals_from(&hist).repeated_tool_calls, 2);
        let varied = vec![att("a", false), att("b", false)];
        assert_eq!(signals_from(&varied).repeated_tool_calls, 0);
    }

    /// The model must be told WHICH failure it hit. "Did not apply" and "applied but tests failed"
    /// call for completely different next attempts, and collapsing them wastes a paid turn.
    #[test]
    fn failure_feedback_distinguishes_apply_from_test_failure() {
        let inst = SweInstance {
            instance_id: "x__y-1".into(),
            repo: "x/y".into(),
            base_commit: "abc".into(),
            problem_statement: "boom".into(),
            test_patch: String::new(),
            fail_to_pass: vec!["t::a".into()],
            pass_to_pass: vec![],
            image: "img".into(),
        };
        let did_not_apply = Attempt {
            patch: "p".into(),
            resolved: false,
            applied: false,
            feedback: "The patch did not apply. Check the paths and context lines.".into(),
        };
        let msgs = build_messages(&inst, std::slice::from_ref(&did_not_apply));
        let text = msgs.to_string();
        assert!(
            text.contains("did not apply"),
            "apply failures must say so: {text}"
        );
        assert!(
            text.contains("Error:"),
            "the failure must read as an error, which is what the extractor keys on"
        );
    }

    /// Fenced diffs are a formatting slip, not a failed patch.
    #[test]
    fn fenced_diffs_are_unwrapped() {
        assert_eq!(strip_fences("```diff\n--- a\n+++ b\n```"), "--- a\n+++ b\n");
        assert_eq!(strip_fences("--- a\n+++ b"), "--- a\n+++ b");
    }
}
