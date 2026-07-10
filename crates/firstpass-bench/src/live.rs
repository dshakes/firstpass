//! Live provider backend — the real-provider seam the simulation was built behind.
//!
//! `--live` swaps [`SimBackend`](crate::sim::SimBackend)/[`SimGate`](crate::sim::SimGate) for
//! [`LiveBackend`] (Anthropic Messages API over blocking HTTP) and [`LiveGate`] (a deterministic
//! answer-checker over a verifiable task suite). The same policies, metrics, bootstrap CIs,
//! conformal calibration, and kill criterion then run on **real** token usage and **real** per-rung
//! clearance rates — turning the simulated methodology into actual proof.
//!
//! LIVE-UNVERIFIED: the HTTP request/auth path compiles against Anthropic's documented Messages
//! wire shape but is exercised only when you run `--live` with a real key — that run is the proof.
//! The fragile parts (response parsing, answer checking) are unit-tested offline against canned
//! bodies, and any hard call error aborts the run, so a bad key surfaces as a clear error, never as
//! silently wrong numbers.

use std::cell::RefCell;
use std::time::Instant;

use firstpass_core::Verdict;

use crate::sim::{Completion, Gate, GateJudgement, ModelBackend, Rung, Task};

/// Live Anthropic Messages backend. BYOK: the key is used only to call the provider.
#[derive(Debug)]
pub struct LiveBackend {
    client: reqwest::blocking::Client,
    api_key: String,
    base_url: String,
    /// Hard call errors recorded during a run; a non-empty list aborts before publishing numbers.
    errors: RefCell<Vec<String>>,
}

impl LiveBackend {
    /// Construct a backend from a provider API key. Honors `ANTHROPIC_BASE_URL` for proxies/tests.
    #[must_use]
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            api_key,
            base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            errors: RefCell::new(Vec::new()),
        }
    }

    /// One cheap call to validate key + model before the full run, so a bad setup fails fast.
    ///
    /// # Errors
    /// Returns the provider/transport error if the call does not succeed.
    pub fn preflight(&self, rung: &Rung) -> Result<(), String> {
        self.call(model_id(&rung.model), "Reply with the word: ok")
            .map(|_| ())
    }

    /// Drain and return the recorded call errors.
    pub fn take_errors(&self) -> Vec<String> {
        std::mem::take(&mut self.errors.borrow_mut())
    }

    /// POST one message, return `(text, input_tokens, output_tokens)`.
    fn call(&self, model: &str, prompt: &str) -> Result<(String, u64, u64), String> {
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 1024,
            "messages": [{ "role": "user", "content": prompt }],
        });
        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("request failed: {e}"))?;
        let status = resp.status();
        let text = resp
            .text()
            .map_err(|e| format!("reading body failed: {e}"))?;
        if !status.is_success() {
            return Err(format!(
                "HTTP {status}: {}",
                text.chars().take(300).collect::<String>()
            ));
        }
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("response was not JSON: {e}"))?;
        parse_anthropic(&v)
    }
}

impl ModelBackend for LiveBackend {
    fn run(&self, task: &Task, rung: &Rung) -> Completion {
        let prompt = task.prompt.as_deref().unwrap_or_default();
        let expected = task.expected.as_deref().unwrap_or_default();
        let start = Instant::now();
        match self.call(model_id(&rung.model), prompt) {
            Ok((text, in_tokens, out_tokens)) => Completion {
                in_tokens,
                out_tokens,
                correct: check_answer(&text, expected),
                latency_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            },
            Err(e) => {
                self.errors
                    .borrow_mut()
                    .push(format!("{} on task {}: {e}", rung.model, task.id));
                // A recorded error aborts the run; these placeholder numbers are never published.
                Completion {
                    in_tokens: task.prompt_tokens,
                    out_tokens: 0,
                    correct: false,
                    latency_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                }
            }
        }
    }
}

/// The gate for a live run over a verifiable suite: the checker (already applied in the backend) is
/// ground truth here, so the gate is deterministic — pass iff the checked answer was correct. This
/// is a *perfect* gate for THIS suite; the imperfect-gate study stays in the sim, and a real
/// (imperfect) judge gate arrives with the gate framework, not here.
#[derive(Debug)]
pub struct LiveGate;

impl Gate for LiveGate {
    fn judge(&self, _task: &Task, _rung: &Rung, completion: &Completion) -> GateJudgement {
        GateJudgement {
            verdict: if completion.correct {
                Verdict::Pass
            } else {
                Verdict::Fail
            },
            score: if completion.correct { 1.0 } else { 0.0 },
            cost_usd: 0.0,
            ms: 0,
        }
    }
}

/// Strip the `provider/` prefix from a ladder model to the provider's own model id.
fn model_id(ladder_model: &str) -> &str {
    ladder_model
        .strip_prefix("anthropic/")
        .unwrap_or(ladder_model)
}

/// Parse an Anthropic Messages response into `(text, input_tokens, output_tokens)`.
///
/// # Errors
/// Returns a message if the response is missing text content or usage counts.
pub fn parse_anthropic(v: &serde_json::Value) -> Result<(String, u64, u64), String> {
    let text: String = v
        .get("content")
        .and_then(|c| c.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(serde_json::Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .filter(|s| !s.is_empty())
        .ok_or("no text content in response")?;
    let usage = v.get("usage").ok_or("no usage in response")?;
    let in_tokens = usage
        .get("input_tokens")
        .and_then(serde_json::Value::as_u64)
        .ok_or("no usage.input_tokens")?;
    let out_tokens = usage
        .get("output_tokens")
        .and_then(serde_json::Value::as_u64)
        .ok_or("no usage.output_tokens")?;
    Ok((text, in_tokens, out_tokens))
}

/// Ground-truth check. For an integer answer (the arithmetic suite) it compares the **last integer**
/// in the output exactly, so `"12"` never spuriously matches inside `"1234"`. For a short factual
/// answer it falls back to a normalized substring match.
#[must_use]
pub fn check_answer(output: &str, expected: &str) -> bool {
    if let Ok(exp) = expected.trim().parse::<i64>() {
        return last_integer(output) == Some(exp);
    }
    fn norm(s: &str) -> String {
        s.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    let (out, exp) = (norm(output), norm(expected));
    !exp.is_empty() && out.contains(&exp)
}

/// The last integer appearing in `s` (commas stripped, leading `-` honored) — the model's final
/// answer when it's told to "reply with only the final integer".
fn last_integer(s: &str) -> Option<i64> {
    let cleaned = s.replace(',', "");
    let mut last = None;
    let mut cur = String::new();
    for c in cleaned.chars() {
        if c.is_ascii_digit() || (c == '-' && cur.is_empty()) {
            cur.push(c);
        } else {
            if let Ok(n) = cur.parse::<i64>() {
                last = Some(n);
            }
            cur.clear();
        }
    }
    if let Ok(n) = cur.parse::<i64>() {
        last = Some(n);
    }
    last
}

/// A ≥200-scale suite of **verifiable, graded** arithmetic tasks with a real difficulty gradient
/// (fully-parenthesized left-to-right expressions of escalating length/magnitude). Deterministic:
/// task `i` is a pure function of `i`, so a run is reproducible without a stored dataset.
///
/// This scales the **cost / success / escalation** proof to statistical power. It does *not*
/// exercise an imperfect gate — arithmetic is self-checking, so a judge can just recompute it; a
/// meaningful conformal guarantee needs a coding-with-tests benchmark, which is a separate effort.
#[must_use]
pub fn graded_suite(n: usize) -> Vec<Task> {
    (0..n)
        .map(|i| {
            let (prompt, answer) = arithmetic_task(i as u64);
            Task::verifiable(i as u64, prompt, answer.to_string())
        })
        .collect()
}

/// Build one deterministic arithmetic task from `seed`: `(prompt, exact_answer)`. Difficulty tiers
/// by `seed % 3` (2/3/4 steps, growing operands) create the gradient that makes routing matter.
fn arithmetic_task(seed: u64) -> (String, i64) {
    // A small deterministic LCG so operands are a pure function of the seed (no RNG state, no dates).
    let mut state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mut next = |m: u64| -> u64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % m
    };

    let (steps, max) = match seed % 3 {
        0 => (2u32, 20i64), // easy
        1 => (3, 99),       // medium
        _ => (4, 999),      // hard
    };

    #[allow(clippy::cast_possible_wrap)]
    let mut val = (next(max as u64) as i64) + 1;
    let mut expr = format!("{val}");
    for _ in 0..steps {
        #[allow(clippy::cast_possible_wrap)]
        match next(3) {
            0 => {
                let o = (next(max as u64) as i64) + 1;
                expr = format!("({expr} + {o})");
                val += o;
            }
            1 => {
                let o = (next(max as u64) as i64) + 1;
                expr = format!("({expr} - {o})");
                val -= o;
            }
            // Keep × operands small so magnitudes stay in a range that differentiates models
            // (rather than exploding past what any model can do).
            _ => {
                let o = (next(9) as i64) + 2;
                expr = format!("({expr} × {o})");
                val *= o;
            }
        }
    }
    (
        format!("Compute the exact value of {expr}. Reply with only the final integer."),
        val,
    )
}

/// A small embedded suite of verifiable tasks (unambiguous known answers) to prove the live pipeline
/// end-to-end.
///
/// ponytail: 15 tasks prove the plumbing; a publishable M0 (SPEC §10) wants ≥200 curated verifiable
/// tasks — load those from a dataset file when scaling past this proof-of-pipeline.
#[must_use]
pub fn live_suite() -> Vec<Task> {
    const TASKS: &[(&str, &str)] = &[
        (
            "What is 17 multiplied by 23? Reply with only the number.",
            "391",
        ),
        (
            "What is the capital city of Australia? Reply with only the city name.",
            "Canberra",
        ),
        (
            "What is the chemical symbol for gold? Reply with only the symbol.",
            "Au",
        ),
        (
            "What is the square root of 144? Reply with only the number.",
            "12",
        ),
        (
            "Who wrote the play 'Romeo and Juliet'? Reply with only the author's full name.",
            "William Shakespeare",
        ),
        (
            "What is the largest planet in our solar system? Reply with only the planet name.",
            "Jupiter",
        ),
        ("What is 100 minus 37? Reply with only the number.", "63"),
        (
            "In what year did Apollo 11 first land humans on the Moon? Reply with only the year.",
            "1969",
        ),
        (
            "What is the chemical formula for water? Reply with only the formula.",
            "H2O",
        ),
        (
            "What is the capital of Japan? Reply with only the city name.",
            "Tokyo",
        ),
        ("What is 8 factorial? Reply with only the number.", "40320"),
        (
            "Which language has the most native speakers worldwide? Reply with only the language name.",
            "Mandarin",
        ),
        (
            "What planet is known as the Red Planet? Reply with only the planet name.",
            "Mars",
        ),
        (
            "What gas do plants primarily absorb for photosynthesis? Reply with only the gas name.",
            "carbon dioxide",
        ),
        (
            "What is the tallest mountain above sea level on Earth? Reply with only the name.",
            "Everest",
        ),
    ];
    TASKS
        .iter()
        .enumerate()
        .map(|(i, (prompt, answer))| Task::verifiable(i as u64, *prompt, *answer))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_anthropic_response() {
        let body = serde_json::json!({
            "content": [{ "type": "text", "text": "The answer is 391." }],
            "usage": { "input_tokens": 20, "output_tokens": 7 }
        });
        let (text, inp, out) = parse_anthropic(&body).unwrap();
        assert_eq!(text, "The answer is 391.");
        assert_eq!((inp, out), (20, 7));
    }

    #[test]
    fn parse_errors_when_usage_is_missing() {
        let body = serde_json::json!({ "content": [{ "type": "text", "text": "hi" }] });
        assert!(parse_anthropic(&body).is_err());
    }

    #[test]
    fn parse_errors_when_text_is_empty() {
        let body = serde_json::json!({
            "content": [],
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        });
        assert!(parse_anthropic(&body).is_err());
    }

    #[test]
    fn checks_answers_normalized_and_lenient() {
        assert!(check_answer("The answer is 391.", "391"));
        assert!(check_answer("H2O", "h2o"));
        assert!(check_answer("It is Canberra.", "canberra"));
        assert!(!check_answer("The capital is Sydney.", "Canberra"));
        assert!(!check_answer("anything", "")); // empty expected never matches
    }

    #[test]
    fn checks_numeric_answers_exactly() {
        assert!(check_answer("The final answer is 391.", "391"));
        assert!(!check_answer("3910", "391")); // no spurious substring match
        assert!(check_answer("result: -6", "-6"));
        assert!(check_answer("1,274", "1274")); // commas stripped
        assert!(!check_answer("12", "1234"));
    }

    /// Re-evaluate a fully-left-nested expression by folding its numbers/ops left-to-right.
    fn eval_left(expr: &str) -> i64 {
        let mut nums = Vec::new();
        let mut ops = Vec::new();
        let mut cur = String::new();
        for c in expr.chars() {
            if c.is_ascii_digit() {
                cur.push(c);
            } else {
                if !cur.is_empty() {
                    nums.push(cur.parse::<i64>().unwrap());
                    cur.clear();
                }
                if c == '+' || c == '-' || c == '×' {
                    ops.push(c);
                    // Stop once we leave the expression (the trailing "Reply..." has no ops).
                }
            }
        }
        if !cur.is_empty() {
            nums.push(cur.parse::<i64>().unwrap());
        }
        // Only fold the operands the ops actually pair with (ignore any trailing prose numbers).
        let mut acc = nums[0];
        for (i, op) in ops.iter().enumerate() {
            let n = nums[i + 1];
            acc = match op {
                '+' => acc + n,
                '-' => acc - n,
                _ => acc * n,
            };
        }
        acc
    }

    #[test]
    fn graded_tasks_are_deterministic_and_correct() {
        for seed in 0..60u64 {
            let (p1, a1) = arithmetic_task(seed);
            let (p2, a2) = arithmetic_task(seed);
            assert_eq!((p1.clone(), a1), (p2, a2), "deterministic for seed {seed}");
            let expr = p1
                .trim_start_matches("Compute the exact value of ")
                .split(". Reply")
                .next()
                .unwrap();
            assert_eq!(
                eval_left(expr),
                a1,
                "answer matches expression for seed {seed}"
            );
        }
    }

    #[test]
    fn graded_suite_scales_and_is_verifiable() {
        let suite = graded_suite(200);
        assert_eq!(suite.len(), 200);
        assert!(
            suite
                .iter()
                .all(|t| t.prompt.is_some() && t.expected.is_some())
        );
    }

    #[test]
    fn model_id_strips_provider_prefix() {
        assert_eq!(model_id("anthropic/claude-haiku-4-5"), "claude-haiku-4-5");
        assert_eq!(model_id("claude-opus-4-8"), "claude-opus-4-8");
    }

    #[test]
    fn live_suite_tasks_are_all_verifiable() {
        let suite = live_suite();
        assert!(suite.len() >= 12);
        assert!(
            suite
                .iter()
                .all(|t| t.prompt.is_some() && t.expected.is_some())
        );
    }
}
