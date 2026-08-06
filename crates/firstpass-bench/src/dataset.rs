//! Dataset loaders for the coding-with-tests benchmark (`coding.rs`): turn MBPP/HumanEval JSONL
//! into `CodingTask`s so the gate error and conformal bound can be measured on real, external
//! problems instead of the hand-authored/generated suites.
//!
//! MBPP's `test_list` is a list of Python `assert` *statements*; the coding runner needs bare
//! boolean *expressions* (it does `eval(case)`, not `exec`). `convert_assert` strips the leading
//! `assert` and any trailing `, "message"`, tracking bracket depth and string literals so it
//! never splits inside a call's arguments (e.g. `similar_elements((3,4,5),(4,5,7))==(4,5)`).
//!
//! `visible_cases` is a strict prefix of `hidden_cases` (the first half, rounded up) — a real
//! coverage gap the candidate can exploit, same shape as the gappy suites in `coding.rs`.
//!
//! **BigCodeBench** is the reason `CodingTask::unit_test` exists. Its tasks call real libraries
//! (numpy, pandas, requests…) and ship a whole `unittest.TestCase` per task instead of a list of
//! assertions, so they cannot be flattened into `eval`-able expressions without losing exactly
//! what makes them hard: setup, mocking, and tolerance-based comparisons. Those are also where a
//! test suite's genuine coverage gaps live, which is why this dataset is the one that can produce
//! a non-degenerate served-failure bound — every suite tried before it had a near-perfect gate,
//! and a gate with no error leaves conformal nothing to control.

use crate::coding::CodingTask;

/// Load MBPP-style JSONL: one `{"task_id", "text", "code", "test_list": [...]}` object per line.
///
/// # Errors
/// The path can't be read, a line isn't valid JSON, a required field is missing/mistyped, or an
/// `assert` in `test_list` doesn't parse into an expression.
pub fn load_mbpp_jsonl(path: &str) -> Result<Vec<CodingTask>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| parse_mbpp_line(line).map_err(|e| format!("{path}:{}: {e}", i + 1)))
        .collect()
}

fn parse_mbpp_line(line: &str) -> Result<CodingTask, String> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let task_id = v.get("task_id").ok_or("missing task_id")?;
    let task_id = match task_id {
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        _ => return Err("task_id must be a number or string".to_owned()),
    };
    let text = v
        .get("text")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing text")?;
    let test_list = v
        .get("test_list")
        .and_then(serde_json::Value::as_array)
        .ok_or("missing test_list")?;
    if test_list.is_empty() {
        return Err("test_list is empty".to_owned());
    }
    let hidden_cases: Vec<String> = test_list
        .iter()
        .map(|t| {
            let s = t.as_str().ok_or("test_list entry is not a string")?;
            convert_assert(s)
        })
        .collect::<Result<_, String>>()?;
    // Strict prefix subset: first half (rounded up) is visible, the rest is oracle-only —
    // a real coverage gap a candidate can pass on visible and still fail on hidden.
    let n_visible = hidden_cases.len().div_ceil(2);
    let visible_cases = hidden_cases[..n_visible].to_vec();
    // Show the candidate the VISIBLE tests (the gate) so it knows the exact function
    // name/signature to implement — MBPP's `text` alone doesn't specify it, and the
    // asserts call a specific name. The candidate is meant to see its gate; the HIDDEN
    // tests stay held out as the oracle, preserving the real coverage gap.
    let visible_src: Vec<&str> = test_list[..n_visible]
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    let prompt = format!(
        "{text}\n\nYour solution must pass these tests:\n{}\n\nWrite the solution to `solution.py` defining the required function(s).",
        visible_src.join("\n")
    );
    Ok(CodingTask {
        id: format!("mbpp-{task_id}"),
        prompt,
        entrypoint: "solution.py".to_owned(),
        visible_cases,
        hidden_cases,
        unit_test: None,
    })
}

/// Load HumanEval-style JSONL: one `{"task_id", "prompt", "entry_point", "test": "def check..."}`
/// object per line.
///
/// # Errors
/// The path can't be read, a line isn't valid JSON, a required field is missing/mistyped, or the
/// `test` field has no single-line `assert` statements.
///
// ponytail: only extracts single-line `assert ...` statements out of `def check(candidate):` and
// rewrites `candidate(` -> `<entry_point>(`. HumanEval's `test` field occasionally spans an
// assert across multiple lines (parenthesized continuations) — those are silently skipped rather
// than mis-parsed. Good enough for the common shape; revisit with a real Python tokenizer if a
// live run drops too many cases.
pub fn load_humaneval_jsonl(path: &str) -> Result<Vec<CodingTask>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| parse_humaneval_line(line).map_err(|e| format!("{path}:{}: {e}", i + 1)))
        .collect()
}

fn parse_humaneval_line(line: &str) -> Result<CodingTask, String> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let task_id = v
        .get("task_id")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing task_id")?;
    let prompt_text = v
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing prompt")?;
    let entry_point = v
        .get("entry_point")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing entry_point")?;
    let test = v
        .get("test")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing test")?;

    let hidden_cases: Vec<String> = test
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("assert "))
        .map(convert_assert)
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .map(|expr| expr.replace("candidate(", &format!("{entry_point}(")))
        .collect();
    if hidden_cases.is_empty() {
        return Err("no single-line assert cases found in test".to_owned());
    }
    let n_visible = hidden_cases.len().div_ceil(2);
    let visible_cases = hidden_cases[..n_visible].to_vec();
    let prompt =
        format!("{prompt_text}\n\nWrite the solution to `solution.py` defining `{entry_point}`.");
    Ok(CodingTask {
        id: format!("humaneval-{task_id}"),
        prompt,
        entrypoint: "solution.py".to_owned(),
        visible_cases,
        hidden_cases,
        unit_test: None,
    })
}

/// Convert a Python `assert <expr>[, "message"]` statement into the bare boolean expression the
/// coding runner `eval()`s. Only strips a trailing message at a top-level comma — one outside all
/// `()`/`[]`/`{}` nesting and outside any string literal — so call arguments like
/// `f((1,2),(3,4))` are never split.
fn convert_assert(stmt: &str) -> Result<String, String> {
    let stmt = stmt.trim();
    let rest = stmt
        .strip_prefix("assert")
        .ok_or_else(|| format!("not an assert statement: {stmt}"))?
        .trim_start();
    if rest.is_empty() {
        return Err(format!("empty assert expression: {stmt}"));
    }
    let expr = match top_level_comma(rest) {
        Some(idx) => rest[..idx].trim_end(),
        None => rest,
    };
    if expr.is_empty() {
        return Err(format!("empty assert expression: {stmt}"));
    }
    Ok(expr.to_owned())
}

/// Byte index of the first comma at bracket depth 0, outside any string literal. `None` if there
/// is no such comma (i.e. no trailing message to strip).
fn top_level_comma(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string: Option<char> = None;
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if let Some(q) = in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == q {
                in_string = None;
            }
            continue;
        }
        match c {
            '\'' | '"' => in_string = Some(c),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Load a coding dataset, detecting which of the three supported shapes it is from its first
/// record. Nothing about a `.jsonl` path says which benchmark it holds, and picking the wrong
/// loader does not fail loudly — it produces subtly wrong cases — so the shape decides.
///
/// # Errors
/// The file can't be read, is empty, has an unrecognisable first record, or fails the chosen
/// loader. The error names which shape was detected so a mis-detection is obvious.
pub fn load_coding_dataset(path: &str) -> Result<Vec<CodingTask>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    let first = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| format!("{path} is empty"))?;
    let v: serde_json::Value =
        serde_json::from_str(first).map_err(|e| format!("{path}:1: invalid JSON: {e}"))?;

    if v.get("test_list").is_some() {
        return load_mbpp_jsonl(path);
    }
    match v.get("test").and_then(serde_json::Value::as_str) {
        Some(t) if t.contains("unittest") => load_bigcodebench_jsonl(path),
        Some(t) if t.contains("def check(") => load_humaneval_jsonl(path),
        _ => Err(format!(
            "{path}: unrecognised dataset shape — expected MBPP (`test_list`), BigCodeBench \
             (`test` holding a unittest.TestCase), or HumanEval (`test` holding `def check(`)"
        )),
    }
}

/// Load BigCodeBench-style JSONL: one
/// `{"task_id", "instruct_prompt"|"complete_prompt", "test", "entry_point", "libs"}` per line.
///
/// The `test` field is a `unittest.TestCase` source. Its `test_*` methods are split the same way
/// every other loader here splits cases: the first half (rounded up) is **visible** — the gate the
/// candidate is shown and scored on — and the whole set is the **hidden oracle**. Because these
/// are real suites, the gap between them is a real coverage gap rather than a synthetic one.
///
/// The candidate is shown the visible method sources so it knows the exact contract (BigCodeBench
/// prompts name a function but the suite pins the behaviour). The held-out half never appears.
///
/// # Errors
/// The path can't be read, a line isn't valid JSON, a required field is missing or mistyped, or a
/// task's `test` source declares no `test_*` methods (nothing to score, so it would silently
/// contribute a zero to every rate).
pub fn load_bigcodebench_jsonl(path: &str) -> Result<Vec<CodingTask>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            parse_bigcodebench_line(line).map_err(|e| format!("{path}:{}: {e}", i + 1))
        })
        .collect()
}

fn parse_bigcodebench_line(line: &str) -> Result<CodingTask, String> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let task_id = v
        .get("task_id")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing task_id")?;
    // `instruct_prompt` is the natural-language form; `complete_prompt` is the signature+docstring
    // completion form. Prefer instruct, fall back, so both released splits load.
    let prompt_text = v
        .get("instruct_prompt")
        .or_else(|| v.get("complete_prompt"))
        .and_then(serde_json::Value::as_str)
        .ok_or("missing instruct_prompt/complete_prompt")?;
    let entry_point = v
        .get("entry_point")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("task_func");
    let test_src = v
        .get("test")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing test")?;

    let hidden_cases = test_method_names(test_src);
    if hidden_cases.is_empty() {
        return Err(format!(
            "task {task_id:?} declares no `test_*` methods, so it cannot be scored"
        ));
    }
    let n_visible = hidden_cases.len().div_ceil(2);
    let visible_cases = hidden_cases[..n_visible].to_vec();

    let prompt = format!(
        "{prompt_text}\n\nYour solution must pass these tests:\n{}\n\nWrite the complete solution defining `{entry_point}`. Output only Python code.",
        visible_method_sources(test_src, &visible_cases)
    );
    Ok(CodingTask {
        id: format!("bigcodebench-{task_id}"),
        prompt,
        entrypoint: "solution.py".to_owned(),
        visible_cases,
        hidden_cases,
        unit_test: Some(test_src.to_owned()),
    })
}

/// Names of every `def test_*(...)` in a `unittest` source, in declaration order.
///
/// Deliberately textual rather than a Python parse: the harness never executes this source on the
/// host — it only selects names to run *inside* the sandbox — so reading it is a string operation,
/// not an evaluation. Declaration order is what makes the visible/hidden split reproducible.
fn test_method_names(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|l| {
            let t = l.trim_start();
            let rest = t.strip_prefix("def ")?;
            let name = rest.split('(').next()?.trim();
            (name.starts_with("test") && !name.is_empty()).then(|| name.to_owned())
        })
        .collect()
}

/// The source lines of just the visible test methods, for showing the candidate its own gate.
/// A method runs from its `def` line to the next line at the same or lower indentation.
fn visible_method_sources(src: &str, visible: &[String]) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("def ") else {
            continue;
        };
        let Some(name) = rest.split('(').next().map(str::trim) else {
            continue;
        };
        if !visible.iter().any(|v| v == name) {
            continue;
        }
        let indent = line.len() - trimmed.len();
        out.push(*line);
        for next in &lines[i + 1..] {
            let next_indent = next.len() - next.trim_start().len();
            if !next.trim().is_empty() && next_indent <= indent {
                break;
            }
            out.push(next);
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three real MBPP problems (task_id/text/test_list verbatim from the public MBPP set),
    /// covering: a plain expression, a nested-paren expression, and one with a trailing message.
    const MBPP_FIXTURE: &str = r#"{"task_id": 1, "text": "Write a function to find the similar elements from the given two tuple lists.", "code": "def similar_elements(test_tup1, test_tup2): return tuple(set(test_tup1) & set(test_tup2))", "test_list": ["assert similar_elements((3,4,5,6),(5,7,4,10)) == (4, 5)", "assert similar_elements((1,2,3,4),(5,4,3,7)) == (3, 4)"]}
{"task_id": 8, "text": "Write a function to find the maximum difference between available pairs in the given tuple list.", "code": "def max_difference(test_list): return max(abs(a - b) for a, b in test_list)", "test_list": ["assert max_difference([(3, 5), (1, 7), (10, 3), (1, 2)]) == 7", "assert max_difference([(4, 6), (2, 17), (9, 13), (11, 12)]) == 15", "assert max_difference([(12, 35), (21, 27), (13, 23), (41, 22)]) == 23"]}
{"task_id": 15, "text": "Write a python function to remove first and last occurrence of a given character from the string.", "code": "def remove_Occ(s,ch): return s", "test_list": ["assert remove_Occ(\"hello\",\"l\") == \"heo\", \"first and last l removed\"", "assert remove_Occ(\"abcda\",\"a\") == \"bcd\""]}
"#;

    #[test]
    fn parses_mbpp_fixture_into_coding_tasks() {
        let tmp = std::env::temp_dir().join(format!("mbpp-fixture-{}.jsonl", std::process::id()));
        std::fs::write(&tmp, MBPP_FIXTURE).expect("write fixture");
        let tasks = load_mbpp_jsonl(tmp.to_str().expect("utf8 path")).expect("parse fixture");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].id, "mbpp-1");
        assert_eq!(tasks[1].id, "mbpp-8");
        assert_eq!(tasks[2].id, "mbpp-15");
        for t in &tasks {
            assert_eq!(t.entrypoint, "solution.py");
            assert!(t.prompt.contains("solution.py"));
        }
    }

    #[test]
    fn visible_cases_are_a_strict_prefix_subset_of_hidden() {
        let tmp =
            std::env::temp_dir().join(format!("mbpp-fixture-prefix-{}.jsonl", std::process::id()));
        std::fs::write(&tmp, MBPP_FIXTURE).expect("write fixture");
        let tasks = load_mbpp_jsonl(tmp.to_str().expect("utf8 path")).expect("parse fixture");
        std::fs::remove_file(&tmp).ok();

        // task 8 has 3 hidden cases -> ceil(3/2) = 2 visible, a strict subset.
        let t8 = &tasks[1];
        assert_eq!(t8.hidden_cases.len(), 3);
        assert_eq!(t8.visible_cases.len(), 2);
        assert_eq!(t8.visible_cases, t8.hidden_cases[..2]);
        assert!(t8.visible_cases.len() < t8.hidden_cases.len());
    }

    #[test]
    fn converts_plain_assert_to_expression() {
        let expr = convert_assert("assert similar_elements((3,4,5,6),(5,7,4,10)) == (4, 5)")
            .expect("convert");
        assert_eq!(expr, "similar_elements((3,4,5,6),(5,7,4,10)) == (4, 5)");
    }

    #[test]
    fn strips_trailing_message_without_splitting_nested_parens() {
        let expr = convert_assert(
            r#"assert remove_Occ("hello","l") == "heo", "first and last l removed""#,
        )
        .expect("convert");
        assert_eq!(expr, r#"remove_Occ("hello","l") == "heo""#);
    }

    #[test]
    fn does_not_split_on_commas_inside_call_arguments() {
        // No top-level comma at all (both commas are nested inside tuple literals) -> unchanged.
        let expr =
            convert_assert("assert similar_elements((3,4,5),(4,5,7))==(4,5)").expect("convert");
        assert_eq!(expr, "similar_elements((3,4,5),(4,5,7))==(4,5)");
    }

    #[test]
    fn rejects_non_assert_lines() {
        assert!(convert_assert("print('not an assert')").is_err());
    }

    #[test]
    fn malformed_line_is_an_error() {
        let tmp = std::env::temp_dir().join(format!("mbpp-malformed-{}.jsonl", std::process::id()));
        std::fs::write(&tmp, "{ not json\n").expect("write fixture");
        let result = load_mbpp_jsonl(tmp.to_str().expect("utf8 path"));
        std::fs::remove_file(&tmp).ok();
        assert!(result.is_err());
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(load_mbpp_jsonl("/nonexistent/path/does/not/exist.jsonl").is_err());
    }

    /// A minimal HumanEval-shaped fixture: single-line asserts inside `def check(candidate):`.
    const HUMANEVAL_FIXTURE: &str = r#"{"task_id": "HumanEval/0", "prompt": "def has_close_elements(numbers, threshold):\n", "entry_point": "has_close_elements", "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"}
"#;

    /// A BigCodeBench-shaped fixture: a real `unittest.TestCase` with four methods, one of which
    /// uses a numeric tolerance — the kind of assertion that cannot survive being flattened into
    /// an `eval`-able boolean expression.
    const BCB_FIXTURE: &str = r#"{"task_id": "BigCodeBench/7", "instruct_prompt": "Compute the mean of a column.", "entry_point": "task_func", "libs": ["numpy"], "test": "import unittest\nimport numpy as np\n\nclass TestCases(unittest.TestCase):\n    def setUp(self):\n        self.rows = [1, 2, 3, 4]\n\n    def test_basic(self):\n        self.assertEqual(task_func([2, 4]), 3)\n\n    def test_single(self):\n        self.assertEqual(task_func([5]), 5)\n\n    def test_tolerance(self):\n        self.assertAlmostEqual(task_func([1, 2]), 1.5, places=6)\n\n    def test_empty(self):\n        with self.assertRaises(ValueError):\n            task_func([])\n"}
"#;

    #[test]
    fn bigcodebench_splits_test_methods_into_a_gate_and_a_held_out_oracle() {
        let tmp = std::env::temp_dir().join(format!("bcb-fixture-{}.jsonl", std::process::id()));
        std::fs::write(&tmp, BCB_FIXTURE).expect("write fixture");
        let tasks = load_bigcodebench_jsonl(tmp.to_str().expect("utf8 path")).expect("parse");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(tasks.len(), 1);
        let t = &tasks[0];
        assert_eq!(t.id, "bigcodebench-BigCodeBench/7");
        // `setUp` is not a case; the four `test_*` methods are.
        assert_eq!(
            t.hidden_cases,
            ["test_basic", "test_single", "test_tolerance", "test_empty"]
        );
        // Visible is a STRICT PREFIX of hidden — the coverage gap the candidate can fall into.
        assert_eq!(t.visible_cases, ["test_basic", "test_single"]);
        assert_eq!(t.hidden_cases[..2], t.visible_cases[..]);

        // The suite source rides along so the sandbox can run the real assertions.
        let suite = t.unit_test.as_deref().expect("unit_test carried");
        assert!(suite.contains("assertAlmostEqual"));

        // The candidate sees its gate and ONLY its gate: showing a held-out method would close
        // the very coverage gap the measurement depends on.
        assert!(t.prompt.contains("def test_basic"));
        assert!(t.prompt.contains("assertEqual(task_func([2, 4]), 3)"));
        assert!(
            !t.prompt.contains("test_tolerance") && !t.prompt.contains("test_empty"),
            "held-out oracle methods leaked into the prompt: {}",
            t.prompt
        );
    }

    /// Detection must route each shape to its own loader. Picking the wrong one does not error —
    /// it silently yields wrong cases — so this is the check that keeps a run honest.
    #[test]
    fn dataset_shape_is_detected_not_assumed() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        for (name, body, want_prefix) in [
            ("mbpp", MBPP_FIXTURE, "mbpp-"),
            ("bcb", BCB_FIXTURE, "bigcodebench-"),
            ("he", HUMANEVAL_FIXTURE, "humaneval-"),
        ] {
            let p = dir.join(format!("detect-{name}-{pid}.jsonl"));
            std::fs::write(&p, body).expect("write fixture");
            let tasks = load_coding_dataset(p.to_str().expect("utf8 path"))
                .unwrap_or_else(|e| panic!("{name} must load: {e}"));
            std::fs::remove_file(&p).ok();
            assert!(
                tasks[0].id.starts_with(want_prefix),
                "{name} routed to the wrong loader: got id {:?}",
                tasks[0].id
            );
            // Only BigCodeBench carries a unittest suite.
            assert_eq!(tasks[0].unit_test.is_some(), name == "bcb");
        }

        let p = dir.join(format!("detect-junk-{pid}.jsonl"));
        std::fs::write(&p, "{\"task_id\": 1}\n").expect("write fixture");
        let err = load_coding_dataset(p.to_str().expect("utf8 path"))
            .expect_err("an unrecognisable shape must not be guessed at");
        std::fs::remove_file(&p).ok();
        assert!(err.contains("unrecognised dataset shape"), "got: {err}");
    }

    #[test]
    fn bigcodebench_rejects_a_task_with_nothing_to_score() {
        let line = r#"{"task_id": "X/1", "instruct_prompt": "p", "test": "import unittest\nclass T(unittest.TestCase):\n    def helper(self):\n        pass\n"}"#;
        let tmp = std::env::temp_dir().join(format!("bcb-empty-{}.jsonl", std::process::id()));
        std::fs::write(&tmp, line).expect("write fixture");
        let err = load_bigcodebench_jsonl(tmp.to_str().expect("utf8 path"))
            .expect_err("a task with no test_* methods must not load");
        std::fs::remove_file(&tmp).ok();
        assert!(err.contains("no `test_*` methods"), "got: {err}");
    }

    #[test]
    fn parses_humaneval_fixture_and_rewrites_candidate() {
        let tmp =
            std::env::temp_dir().join(format!("humaneval-fixture-{}.jsonl", std::process::id()));
        std::fs::write(&tmp, HUMANEVAL_FIXTURE).expect("write fixture");
        let tasks = load_humaneval_jsonl(tmp.to_str().expect("utf8 path")).expect("parse fixture");
        std::fs::remove_file(&tmp).ok();

        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "humaneval-HumanEval/0");
        assert_eq!(tasks[0].hidden_cases.len(), 2);
        assert!(tasks[0].hidden_cases[0].starts_with("has_close_elements("));
        assert_eq!(tasks[0].visible_cases.len(), 1);
    }
}

// ── Structured extraction ───────────────────────────────────────────────────────────────────────
//
// The sharpest objection to every number this project publishes is that they all come from coding
// tasks with executable tests — the one domain where a gate is easy. Extraction answers it in the
// same currency: the output is machine-checkable, so a real gate exists, but nothing is executed
// and the task is not programming.
//
// Modelled as a `CodingTask` rather than a parallel harness so the entire policy machinery —
// measure-once-replay-many, paired bootstrap, the gate/oracle split — is reused unchanged. A second
// harness would be free to drift on exactly the statistics that make the claim.

/// Load a structured-extraction dataset as [`CodingTask`]s.
///
/// One JSON object per line:
///
/// ```json
/// {"id": "inv-1",
///  "text": "Invoice from Acme Corp dated 2026-03-04, total $1,240.50",
///  "required": ["vendor", "date", "total"],
///  "expected": {"vendor": "Acme Corp", "date": "2026-03-04", "total": "1240.50"}}
/// ```
///
/// The **gate** (`visible_cases`) checks only that the answer is well-formed JSON carrying the
/// required keys. The **oracle** (`hidden_cases`) additionally checks every value against ground
/// truth. That gap is the point and mirrors the coding case exactly: a model can emit perfectly
/// shaped JSON with the wrong vendor in it, pass the gate, and still be wrong — which is what makes
/// a served-failure bound meaningful rather than circular.
///
/// # Errors
/// Unreadable file, malformed line, or a record missing `text`/`expected`.
pub fn load_extraction_jsonl(path: &str) -> Result<Vec<CodingTask>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| parse_extraction_line(line).map_err(|e| format!("{path}:{}: {e}", i + 1)))
        .collect()
}

fn parse_extraction_line(line: &str) -> Result<CodingTask, String> {
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let id = v
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `id`")?
        .to_owned();
    let text = v
        .get("text")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `text`")?;
    let expected = v
        .get("expected")
        .and_then(serde_json::Value::as_object)
        .ok_or("missing `expected` object")?;
    // Absent `required` defaults to every expected key: the weakest honest gate is "the shape is
    // there", and defaulting to nothing would make the gate vacuous and the comparison meaningless.
    let required: Vec<String> = match v.get("required").and_then(serde_json::Value::as_array) {
        Some(a) => a
            .iter()
            .filter_map(|k| k.as_str().map(str::to_owned))
            .collect(),
        None => expected.keys().cloned().collect(),
    };

    let schema_hint = required
        .iter()
        .map(|k| format!("  \"{k}\": <string>"))
        .collect::<Vec<_>>()
        .join(",\n");
    let prompt = format!(
        "Extract the following fields from the text and reply with ONLY a JSON object — no prose, \
         no code fence.\n\nFields:\n{{\n{schema_hint}\n}}\n\nText:\n{text}\n"
    );

    Ok(CodingTask {
        id,
        prompt,
        // The model's raw reply is written here verbatim; the generated suite reads it as TEXT and
        // parses it, so nothing is imported and no code is executed.
        entrypoint: "solution.py".to_owned(),
        visible_cases: vec!["test_parses".to_owned(), "test_required_fields".to_owned()],
        hidden_cases: {
            let mut h = vec!["test_parses".to_owned(), "test_required_fields".to_owned()];
            h.extend((0..expected.len()).map(|i| format!("test_field_{i}")));
            h
        },
        unit_test: Some(extraction_suite(&required, expected)),
    })
}

/// Generate the Python suite for one extraction task.
///
/// Values are embedded with `serde_json::to_string`, never interpolated raw: an expected value
/// containing a quote or a backslash would otherwise produce a syntactically broken suite, and a
/// suite that fails to import scores as a task the model got wrong. That would quietly bias the
/// measurement against the model on exactly the records with awkward data.
fn extraction_suite(
    required: &[String],
    expected: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let mut s = String::from(
        "import json, unittest, pathlib\n\n\
         def _load():\n\
         \x20   raw = pathlib.Path('solution.py').read_text().strip()\n\
         \x20   # Models fence JSON even when told not to; unwrap rather than score it wrong for\n\
         \x20   # a formatting habit the gate is not trying to measure.\n\
         \x20   if raw.startswith('```'):\n\
         \x20       raw = raw.split('```')[1]\n\
         \x20       if raw.startswith('json'):\n\
         \x20           raw = raw[4:]\n\
         \x20   return json.loads(raw)\n\n\
         class T(unittest.TestCase):\n\
         \x20   def test_parses(self):\n\
         \x20       self.assertIsInstance(_load(), dict)\n\n\
         \x20   def test_required_fields(self):\n\
         \x20       d = _load()\n",
    );
    for k in required {
        s.push_str(&format!(
            "        self.assertIn({}, d)\n",
            serde_json::to_string(k).unwrap_or_else(|_| "\"\"".to_owned())
        ));
    }
    for (i, (k, want)) in expected.iter().enumerate() {
        let kq = serde_json::to_string(k).unwrap_or_else(|_| "\"\"".to_owned());
        let wq = serde_json::to_string(want).unwrap_or_else(|_| "null".to_owned());
        s.push_str(&format!(
            "\n    def test_field_{i}(self):\n\
             \x20       d = _load()\n\
             \x20       self.assertEqual(str(d.get({kq})).strip(), str(json.loads({})).strip())\n",
            serde_json::to_string(&wq).unwrap_or_else(|_| "\"null\"".to_owned())
        ));
    }
    s
}

#[cfg(test)]
mod extraction_tests {
    use super::*;

    fn line() -> &'static str {
        r#"{"id":"inv-1","text":"Invoice from Acme Corp dated 2026-03-04","required":["vendor"],"expected":{"vendor":"Acme Corp","date":"2026-03-04"}}"#
    }

    #[test]
    fn the_gate_is_strictly_weaker_than_the_oracle() {
        // The entire design rests on this gap. If the gate checked values too, every gate pass
        // would be a correct answer by construction, the served-failure bound would be trivially
        // zero, and the benchmark would prove nothing.
        let t = parse_extraction_line(line()).expect("parses");

        assert_eq!(t.visible_cases, ["test_parses", "test_required_fields"]);
        assert!(
            t.hidden_cases.len() > t.visible_cases.len(),
            "oracle must check more than the gate: {:?}",
            t.hidden_cases
        );
        for v in &t.visible_cases {
            assert!(t.hidden_cases.contains(v), "oracle must subsume the gate");
        }
        // One value check per expected field.
        assert!(t.hidden_cases.contains(&"test_field_0".to_owned()));
        assert!(t.hidden_cases.contains(&"test_field_1".to_owned()));
    }

    #[test]
    fn the_prompt_carries_the_text_and_asks_for_bare_json() {
        let t = parse_extraction_line(line()).expect("parses");
        assert!(
            t.prompt.contains("Acme Corp dated 2026-03-04"),
            "{}",
            t.prompt
        );
        assert!(t.prompt.contains("ONLY a JSON object"), "{}", t.prompt);
        assert!(t.prompt.contains("\"vendor\""), "{}", t.prompt);
    }

    #[test]
    fn absent_required_defaults_to_every_expected_key() {
        // Defaulting to an EMPTY required list would make the gate vacuous — it would pass any
        // parseable JSON, so "gate passed" would carry no information at all.
        let t = parse_extraction_line(r#"{"id":"x","text":"t","expected":{"a":"1","b":"2"}}"#)
            .expect("parses");
        let suite = t.unit_test.expect("suite");
        assert!(suite.contains("self.assertIn(\"a\", d)"), "{suite}");
        assert!(suite.contains("self.assertIn(\"b\", d)"), "{suite}");
    }

    #[test]
    fn awkward_values_do_not_break_the_generated_suite() {
        // A quote or backslash interpolated raw would produce a suite that fails to IMPORT, which
        // scores as the model getting it wrong — biasing the measurement against the model on
        // exactly the records with difficult data.
        let t = parse_extraction_line(
            r#"{"id":"q","text":"t","expected":{"name":"O'Brien \"Bob\" \\ Co"}}"#,
        )
        .expect("parses");
        let suite = t.unit_test.expect("suite");
        // Balanced quoting is the property that matters; assert the raw text did not leak in.
        assert!(
            !suite.contains("O'Brien \"Bob\""),
            "raw value leaked unescaped:\n{suite}"
        );
        assert!(suite.contains("test_field_0"), "{suite}");
    }

    #[test]
    fn a_record_without_ground_truth_is_rejected() {
        // No oracle means no correctness signal — such a record would silently count as a pass.
        assert!(parse_extraction_line(r#"{"id":"x","text":"t"}"#).is_err());
        assert!(parse_extraction_line(r#"{"id":"x","expected":{"a":"1"}}"#).is_err());
    }
}
