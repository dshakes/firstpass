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
fn build_messages(
    instance: &SweInstance,
    history: &[Attempt],
    findings: &str,
) -> serde_json::Value {
    let mut msgs = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "Repository: {}\nCommit: {}\n\n## Problem\n\n{}\n\n## Code you inspected\n\n{}\n\n\
             Produce a unified diff that fixes this. Output ONLY the diff, starting with \
             `diff --git` or `---`. No prose, no fences. Paths must be relative to the repository \
             root, and context lines must match the code above exactly.",
            instance.repo,
            instance.base_commit,
            instance.problem_statement,
            if findings.is_empty() { "(no exploration performed)" } else { findings },
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
/// How many read-only exploration steps the model may take before writing a patch.
///
/// Each step is a paid model call plus a container run, so this is a real cost knob. Four is enough
/// to `ls` the package, `grep` the symbol, and `read` the function twice — the minimum sequence a
/// human would perform — without letting a confused model wander an entire repository at $0.02 a
/// look.
const MAX_EXPLORE_STEPS: usize = 4;

/// Let the model read the repository before it patches it.
///
/// Returns a transcript of `command -> output` pairs to prepend to the patch prompt. Exploration is
/// **best-effort**: a malformed command or a container hiccup is reported back to the model as text
/// and the loop continues, because a failed `ls` is not a reason to abandon a paid instance.
/// The exploration prompt. Pure, so it can be asserted on without a network call — the first
/// version of this text passed its arguments positionally and swapped two of them, producing a
/// prompt that read perfectly while showing the model its own empty transcript as "the failing
/// test". A prompt bug that is still grammatical is invisible in review and invisible at runtime;
/// the only thing that catches it is an assertion on the built string.
fn explore_prompt(instance: &SweInstance, transcript: &str) -> String {
    format!(
        "Repository: {repo}\n\n## Problem\n\n{problem}\n\n## What you have found so far\n\n{found}\n\n\
         You may inspect the repository with ONE command per turn:\n\
           ls <dir>                    list a directory\n\
           grep <pattern> <dir>        search file contents (fixed string, shows line numbers)\n\
           read <file> <start> <end>   read a line range\n\
           test <pytest-node-id>       run a test and read its real output\n\n\
         The failing test for this issue is:\n  {failing_test}\n\n\
         Run it first — the traceback tells you which file and line to look at, which is \
         faster than guessing from the problem statement.\n\n\
         Output ONLY the command, or the single word DONE when you have seen enough to write \
         the patch. No prose.",
        repo = instance.repo,
        problem = instance.problem_statement,
        found = if transcript.is_empty() {
            "(nothing yet)"
        } else {
            transcript
        },
        failing_test = instance
            .fail_to_pass
            .first()
            .map_or("(none listed)", String::as_str),
    )
}

fn explore_phase(
    instance: &SweInstance,
    limits: &SweLimits,
    bare_model: &str,
    call: &ModelCall<'_>,
    spent: &mut f64,
    in_price: f64,
    out_price: f64,
) -> String {
    let mut transcript = String::new();
    for _ in 0..MAX_EXPLORE_STEPS {
        let prompt = explore_prompt(instance, &transcript);
        let msgs = serde_json::json!([{ "role": "user", "content": prompt }]);
        let Ok((text, in_tok, out_tok)) = call(bare_model, &msgs, 512) else {
            break;
        };
        *spent += in_tok as f64 * in_price + out_tok as f64 * out_price;
        let line = text.trim().lines().next().unwrap_or("").trim().to_owned();
        if line.is_empty() || line.eq_ignore_ascii_case("done") {
            break;
        }
        let result = match crate::swe_explore::parse_cmd(&line) {
            Ok(cmd) => crate::swe_explore::run_explore(instance, &cmd, limits)
                .unwrap_or_else(|e| format!("(command failed: {e})")),
            Err(e) => format!("(invalid command: {e})"),
        };
        transcript.push_str(&format!("$ {line}\n{result}\n\n"));
    }
    transcript
}

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

    // Read the code before patching it. Done ONCE per instance on the cheapest rung, not per turn
    // and not per rung: the repository does not change between attempts, so re-exploring would pay
    // for the same answer repeatedly. Using the cheap model keeps a phase that is pure input-
    // gathering from costing frontier-model rates.
    //
    // This is what the first SWE-bench run lacked, and why it resolved 1 issue in 352 calls: the
    // solver was writing diffs for files it had never opened.
    let before_explore = spent;
    let findings = rungs.first().map_or_else(String::new, |r| {
        explore_phase(
            instance,
            limits,
            &r.bare,
            call,
            &mut spent,
            r.in_price,
            r.out_price,
        )
    });

    // Snapshot the exploration cost HERE, immediately after the phase that incurred it.
    //
    // The first version computed `spent - explore_start` inside the turn loop, AFTER the rungs had
    // run — so it captured exploration plus that turn's model calls and added them a second time.
    // The recorded cost for the trajectory arm was inflated, not merely misattributed. Flagged in
    // review; the fix for a missing cost introduced a double-counted one.
    let explore_cost = spent - before_explore;
    let mut explore_cost_recorded = false;

    for _turn in 0..max_turns {
        // Computed BEFORE the turn is served: a routing decision may not see its own outcome.
        let sig = signals_from(&history);
        let msgs = build_messages(instance, &history, &findings);

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

        // Exploration is paid ONCE, before the first turn, and must appear in the recorded cost or
        // every offline replay understates the policy that paid for it. It is attributed to the
        // first turn's cheapest rung, which is exactly who spent it.
        //
        // Measured on the 30-instance run: $3.36 recorded against $3.57 actually spent. A 6% gap,
        // landing entirely on the trajectory policy since the baseline never explores — so it
        // flattered the arm under test in the published number. Flagged in review.
        if !explore_cost_recorded && let Some(first) = recorded_rungs.first_mut() {
            first.cost_usd += explore_cost;
            explore_cost_recorded = true;
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
        let msgs = build_messages(&inst, std::slice::from_ref(&did_not_apply), "");
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

    /// **Recorded cost must equal what was actually spent — no gap, no double-count.**
    ///
    /// This assertion has now caught the accounting twice in opposite directions. First the
    /// exploration cost was MISSING from the checkpoint ($3.36 recorded vs $3.57 spent on the
    /// 30-instance run, understating the arm that paid it). The fix for that then DOUBLE-COUNTED
    /// the first turn's model calls, because the delta was read after the rung loop rather than
    /// immediately after exploration.
    ///
    /// Both errors land entirely on the trajectory arm, since the baseline never explores — one
    /// flattering it, one penalising it. Neither is acceptable in a number used to accept or reject
    /// a feature, so the invariant is asserted rather than reasoned about.
    /// Needs Docker: `run_instance` evaluates every candidate patch in a real container. Ignored by
    /// default so CI without a daemon stays green, matching `sandbox::tests::real_*`. Run with:
    ///   cargo test -p firstpass-bench --lib recorded_cost -- --ignored
    ///
    /// It passed locally and failed on the macOS runner, which has no Docker — the test was
    /// portable only by accident of my machine.
    #[test]
    #[ignore = "needs a running Docker daemon"]
    fn recorded_cost_equals_actual_spend() {
        // Every call bills exactly 1000 in + 1000 out at unit prices, so each is $0.002.
        const PER_CALL: f64 = 0.002;
        let call =
            |_b: &str, _m: &serde_json::Value, _t: u32| Ok(("DONE".to_owned(), 1000u64, 1000u64));
        let task = SweInstance {
            instance_id: "t".into(),
            repo: "r".into(),
            base_commit: "c".into(),
            problem_statement: "p".into(),
            test_patch: String::new(),
            fail_to_pass: vec!["t::a".into()],
            pass_to_pass: vec![],
            image: "img".into(),
        };
        let rungs = vec![SweRung {
            model: "m".into(),
            bare: "m".into(),
            in_price: 1e-6,
            out_price: 1e-6,
        }];
        let run = run_instance(&task, &rungs, &SweLimits::default(), 1, &call)
            .expect("the loop must complete");

        let recorded: f64 = run
            .recorded
            .turns
            .iter()
            .flat_map(|t| t.rungs.iter().map(|r| r.cost_usd))
            .sum();
        assert!(
            (recorded - run.spent_usd).abs() < 1e-9,
            "recorded ${recorded:.6} must equal spent ${:.6} — a gap understates the policy that \
             paid, a surplus penalises it, and both land on the trajectory arm",
            run.spent_usd
        );
        // Sanity: the fixture must actually have spent something, or the equality is vacuous.
        assert!(
            run.spent_usd >= PER_CALL,
            "the fixture must incur real cost, got ${:.6}",
            run.spent_usd
        );
    }

    /// **What the model read must reach the prompt it writes the patch from.**
    ///
    /// This is the wiring-bug class that has bitten this feature repeatedly: a signal that is
    /// extracted, recorded, and then silently dropped before the only consumer that matters. The
    /// exploration phase is pure cost if its findings do not appear in the patch prompt — and the
    /// symptom would be indistinguishable from "the model is bad at patching".
    #[test]
    fn exploration_findings_reach_the_patch_prompt() {
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
        let findings = "$ read src/mod.py 10 20\n10: def separability_matrix(x):\n";
        let msgs = build_messages(&inst, &[], findings);
        let text = msgs.to_string();
        assert!(
            text.contains("separability_matrix"),
            "the code the model read must appear in the patch prompt: {text}"
        );
        assert!(
            text.contains("Code you inspected"),
            "and it must be labelled so the model knows what it is looking at"
        );

        // With no exploration, the prompt says so plainly rather than showing an empty section —
        // an empty heading reads as "there was nothing to find", which is a different claim.
        let empty = build_messages(&inst, &[], "").to_string();
        assert!(empty.contains("no exploration performed"), "{empty}");
    }

    /// Fenced diffs are a formatting slip, not a failed patch.
    #[test]
    fn fenced_diffs_are_unwrapped() {
        assert_eq!(strip_fences("```diff\n--- a\n+++ b\n```"), "--- a\n+++ b\n");
        assert_eq!(strip_fences("--- a\n+++ b"), "--- a\n+++ b");
    }

    fn swe_instance_fixture() -> SweInstance {
        SweInstance {
            instance_id: "astropy__astropy-12907".to_owned(),
            repo: "astropy/astropy".to_owned(),
            base_commit: "deadbeef".to_owned(),
            problem_statement: "Nested CompoundModel gives wrong separability.".to_owned(),
            test_patch: String::new(),
            fail_to_pass: vec![],
            pass_to_pass: vec![],
            image: "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest".to_owned(),
        }
    }

    /// Asserts each value lands in ITS OWN SECTION, not merely that it appears somewhere in the
    /// prompt. The swapped version contained every substring too — "does the test id appear?"
    /// passes on a prompt that is telling the model the opposite of what it means to.
    #[test]
    fn the_failing_test_and_the_transcript_do_not_swap_sections() {
        let mut inst = swe_instance_fixture();
        inst.fail_to_pass = vec!["tests/test_wcs.py::test_sip".to_owned()];
        let transcript = "$ ls astropy\nwcs/ io/ table/";

        let p = explore_prompt(&inst, transcript);
        let found_at = p
            .find("## What you have found so far")
            .expect("found section");
        let test_at = p
            .find("The failing test for this issue is:")
            .expect("test section");

        // The transcript belongs to the "found so far" section, i.e. after that header and
        // before the failing-test header.
        let t_at = p.find(transcript).expect("transcript present");
        assert!(
            t_at > found_at && t_at < test_at,
            "transcript escaped its section (found@{found_at} t@{t_at} test@{test_at})"
        );
        // ...and the test id belongs after the failing-test header, not in the transcript slot.
        let id_at = p
            .find("tests/test_wcs.py::test_sip")
            .expect("test id present");
        assert!(
            id_at > test_at,
            "failing-test id landed in the transcript slot (test@{test_at} id@{id_at})"
        );
    }

    /// The empty-transcript path is the one every instance takes on its first step, and it is the
    /// path the swap corrupted worst: an empty string rendered as the failing test.
    #[test]
    fn an_empty_transcript_reads_as_nothing_yet_and_leaves_the_test_id_intact() {
        let mut inst = swe_instance_fixture();
        inst.fail_to_pass = vec!["tests/test_a.py::test_b".to_owned()];
        let p = explore_prompt(&inst, "");
        let nothing_at = p.find("(nothing yet)").expect("placeholder present");
        let test_at = p
            .find("The failing test for this issue is:")
            .expect("test section");
        assert!(nothing_at < test_at, "placeholder landed in the wrong slot");
        assert!(
            p.find("tests/test_a.py::test_b").expect("id present") > test_at,
            "test id landed in the transcript slot"
        );
    }
}
