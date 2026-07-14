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
