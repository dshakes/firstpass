//! Pre-registered replay of an OpenJev decision-model prior over real MBPP matrices
//! (`specs/openjev-prior-replay-ab.md`). Two independent stages, matching the pre-registration:
//!
//! - [`run_fetch_priors`]: I/O. Calls `POST {base_url}/v1/systemone` once per task, sequentially,
//!   and writes one [`PriorRecord`] per line — resumable, fails open per task (never panics, never
//!   blocks the rest of the run on one bad call).
//! - [`run_replay_prior`]: pure. Joins a matrix against its saved priors and scores the arms the
//!   spec pre-registers, deterministically, with bootstrap CIs from `stats.rs`.
//!
//! Splitting fetch from scoring is what makes the scoring reproducible by a reviewer with no
//! server and no key: the priors file is the only thing `run_replay_prior` reads.

use std::collections::HashMap;

use serde_json::Value;

use crate::coding_policy::RungOutcome;
use crate::costaware::{self, PassPredictor};
use crate::stats::{
    self, Ci, bootstrap_mean_ci, bootstrap_paired_ratio_diff_ci, bootstrap_ratio_ci,
};

// ---------------------------------------------------------------------------------------------
// Stage 1: fetch priors (I/O)
// ---------------------------------------------------------------------------------------------

/// The single choice question asked of the decision model.
const QUESTION_NAME: &str = "tier";

/// Per-call timeout. Local MLX serving is slow; this is generous on purpose.
const FETCH_TIMEOUT_SECS: u64 = 120;

/// One fetched (or failed) prior, one JSONL line.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PriorRecord {
    pub task_id: String,
    pub ladder: Vec<String>,
    /// `P(least-capable-sufficient = r)` per rung, in ladder order — `None` when the call or the
    /// parse failed (`raw_ok = false`). Never invented: a failed fetch has no prior at all, so the
    /// scoring stage falls back to `first-pass` for that task rather than guessing.
    pub probs: Option<Vec<f64>>,
    pub raw_ok: bool,
    pub latency_ms: u64,
}

/// A row from a replay matrix (`{task_id, ladder, rungs}`), keeping `task_id` — which
/// `coding_policy::load_matrix` deliberately discards because policy replay never needs it, but
/// prior-fetch and prior-join both need it to line a saved prior up with the right task.
#[derive(Debug, Clone, serde::Deserialize)]
struct MatrixRow {
    task_id: String,
    #[serde(default)]
    ladder: Vec<String>,
    rungs: Vec<RungOutcome>,
}

/// Load a replay matrix, keeping `task_id`, and return it alongside the single ladder it was
/// measured on.
///
/// # Errors
/// Unreadable file, a malformed row, an empty file, a row whose ladder disagrees with the file's
/// first row, or a row whose rung count disagrees with the ladder.
fn load_matrix_with_ids(path: &str) -> Result<(Vec<MatrixRow>, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: MatrixRow = serde_json::from_str(line)
            .map_err(|e| format!("{path}:{}: malformed matrix row: {e}", i + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{path} has no rows"));
    }
    let ladder = rows[0].ladder.clone();
    if let Some(bad) = rows.iter().find(|r| r.ladder != ladder) {
        return Err(format!(
            "{path}: task {} has ladder {:?}, but the file's first row has {ladder:?} — a file \
             may hold only one ladder",
            bad.task_id, bad.ladder
        ));
    }
    if let Some(bad) = rows.iter().find(|r| r.rungs.len() != ladder.len()) {
        return Err(format!(
            "{path}: task {} has {} rungs but the ladder has {}",
            bad.task_id,
            bad.rungs.len(),
            ladder.len()
        ));
    }
    Ok((rows, ladder))
}

/// One MBPP task's prompt text plus a single example test, as fetched from the canonical dataset.
struct MbppExample {
    text: String,
    example_test: String,
}

/// Load canonical MBPP JSONL (`{task_id, text, test_list}`) keyed by its integer `task_id`.
/// Deliberately separate from `dataset::load_mbpp_jsonl`: that loader converts each `assert` into
/// a bare boolean expression for `eval`, but the prior only ever needs the raw prompt text and one
/// raw example assertion to show the decision model, never an executable case.
///
/// # Errors
/// Unreadable file, invalid JSON, or a row missing `task_id`/`text`/a non-empty `test_list`.
fn load_mbpp_examples(path: &str) -> Result<HashMap<u64, MbppExample>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut out = HashMap::new();
    for (i, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)
            .map_err(|e| format!("{path}:{}: invalid JSON: {e}", i + 1))?;
        let task_id = v
            .get("task_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{path}:{}: missing/non-integer task_id", i + 1))?;
        let text = v
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{path}:{}: missing text", i + 1))?
            .to_owned();
        let example_test = v
            .get("test_list")
            .and_then(Value::as_array)
            .and_then(|l| l.first())
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{path}:{}: missing/empty test_list", i + 1))?
            .to_owned();
        out.insert(task_id, MbppExample { text, example_test });
    }
    Ok(out)
}

/// `"mbpp-974"` -> `974`.
fn mbpp_numeric_id(task_id: &str) -> Option<u64> {
    task_id.strip_prefix("mbpp-").and_then(|s| s.parse().ok())
}

/// Build the `/v1/systemone` request: one `choice` question, criteria labelled `r0..rN` in
/// ladder order. `r0`/the last rung get the spec's exact phrasing; any rung in between (not
/// exercised by the pre-registered 2-rung ladders, but not assumed away either) gets a plain
/// fallback.
fn build_request(ladder: &[String], task_text: &str, example_test: &str) -> Value {
    let last = ladder.len().saturating_sub(1);
    let mut criteria = serde_json::Map::new();
    for (i, m) in ladder.iter().enumerate() {
        let desc = if i == 0 {
            format!("{m} (the smaller, cheaper model) fully solves this")
        } else if i == last {
            format!("{m} (the frontier model) is needed to solve this")
        } else {
            format!("{m} is needed to solve this")
        };
        criteria.insert(format!("r{i}"), Value::String(desc));
    }
    serde_json::json!({
        "model": "jev-latest",
        "state": { "task": task_text, "example_test": example_test },
        "questions": {
            QUESTION_NAME: {
                "type": "choice",
                "instructions": "Which is the least capable model tier that will fully and \
                    correctly solve this programming task?",
                "criteria": criteria,
            }
        }
    })
}

/// Extract the `tier` question's `probabilities`, in ladder order (`r0..rN`; a missing option
/// maps to `0.0`). Accepts either `{"answers":{"tier":{...}}}` or `{"tier":{...}}` at top level —
/// mirrors `firstpass_proxy::prior::extract_probabilities`, whose response-shape uncertainty this
/// is deliberately consistent with. Anything else, or a missing/non-object `probabilities`,
/// yields `None`.
fn extract_probabilities(json: &Value, n_rungs: usize) -> Option<Vec<f64>> {
    let answer = json
        .get("answers")
        .and_then(|a| a.get(QUESTION_NAME))
        .or_else(|| json.get(QUESTION_NAME))?;
    let probabilities = answer.get("probabilities")?.as_object()?;
    Some(
        (0..n_rungs)
            .map(|i| {
                probabilities
                    .get(&format!("r{i}"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
            })
            .collect(),
    )
}

/// One blocking call. Never panics: transport error, non-2xx, undecodable body, or an unexpected
/// shape all yield `(false, None, elapsed_ms)` — the same fail-open discipline as the proxy's live
/// prior client, just synchronous.
fn fetch_one(
    client: &reqwest::blocking::Client,
    base_url: &str,
    ladder: &[String],
    task_text: &str,
    example_test: &str,
) -> (bool, Option<Vec<f64>>, u64) {
    let body = build_request(ladder, task_text, example_test);
    let url = format!("{}/v1/systemone", base_url.trim_end_matches('/'));
    let start = std::time::Instant::now();
    let elapsed_ms = || start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;

    // Hosted Jev needs a bearer key; a local keyless server (OpenJev) ignores it. The key is read
    // from the environment only and is never written to the priors file or logged.
    let req = client.post(&url).json(&body);
    let req = match std::env::var("TYPESAFE_API_KEY") {
        Ok(key) if !key.is_empty() => req.bearer_auth(key),
        _ => req,
    };
    let resp = match req.send() {
        Ok(r) => r,
        Err(_) => return (false, None, elapsed_ms()),
    };
    if !resp.status().is_success() {
        return (false, None, elapsed_ms());
    }
    let json: Value = match resp.json() {
        Ok(j) => j,
        Err(_) => return (false, None, elapsed_ms()),
    };
    let probs = extract_probabilities(&json, ladder.len());
    let ok = probs.is_some();
    (ok, probs, elapsed_ms())
}

fn append_prior(path: &str, rec: &PriorRecord) {
    let Ok(line) = serde_json::to_string(rec) else {
        return;
    };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Fetch priors for every task in `matrix_path`, sequentially, resuming from whatever is already
/// in `out_path`. Writes one line per task **as it completes** (not buffered), so an interrupted
/// run loses at most the task in flight.
///
/// # Errors
/// The matrix or MBPP file can't be read/parsed, or the HTTP client can't be built. A single
/// task's fetch failing is not an error here — it is recorded as `raw_ok: false` and counted.
pub fn run_fetch_priors(
    matrix_path: &str,
    mbpp_path: &str,
    base_url: &str,
    out_path: &str,
) -> Result<String, String> {
    let (rows, ladder) = load_matrix_with_ids(matrix_path)?;
    let mbpp = load_mbpp_examples(mbpp_path)?;

    let resumed = std::fs::read_to_string(out_path)
        .map(|t| resumable_ids(&t))
        .unwrap_or_default();
    if !resumed.is_empty() {
        eprintln!(
            "resuming: {} priors already recorded in {out_path}",
            resumed.len()
        );
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .build()
        .map_err(|e| format!("cannot build HTTP client: {e}"))?;

    let (mut ok_count, mut fail_count, mut missing_prompt) = (0usize, 0usize, 0usize);
    for (i, row) in rows.iter().enumerate() {
        if resumed.contains(&row.task_id) {
            continue;
        }
        let Some(id) = mbpp_numeric_id(&row.task_id) else {
            eprintln!("skipping {}: task_id is not \"mbpp-<N>\"", row.task_id);
            missing_prompt += 1;
            continue;
        };
        let Some(example) = mbpp.get(&id) else {
            eprintln!("skipping {}: no MBPP prompt for task_id {id}", row.task_id);
            missing_prompt += 1;
            continue;
        };
        let (raw_ok, probs, latency_ms) = fetch_one(
            &client,
            base_url,
            &ladder,
            &example.text,
            &example.example_test,
        );
        if raw_ok {
            ok_count += 1;
        } else {
            fail_count += 1;
        }
        append_prior(
            out_path,
            &PriorRecord {
                task_id: row.task_id.clone(),
                ladder: ladder.clone(),
                probs,
                raw_ok,
                latency_ms,
            },
        );
        eprintln!(
            "[{}/{}] {} raw_ok={raw_ok} latency_ms={latency_ms}",
            i + 1,
            rows.len(),
            row.task_id
        );
    }
    Ok(format!(
        "fetch-priors done: {ok_count} ok, {fail_count} failed, {missing_prompt} had no MBPP \
         prompt, {} resumed from {out_path}",
        resumed.len()
    ))
}

// ---------------------------------------------------------------------------------------------
// Stage 2: replay (pure)
// ---------------------------------------------------------------------------------------------

const BOOT_B: usize = 2000;
const BOOT_SEED: u64 = 42;
const ALPHA: f64 = 0.05;
/// Above this fraction of tasks sharing one argmin start rung, the prior is reported as carrying
/// no per-query signal (spec's degeneracy guard).
const DEGENERATE_THRESHOLD: f64 = 0.95;

/// One task, joined with its (possibly missing) prior and its cross-fitted ex-ante prices.
struct TaskFit {
    row: Vec<RungOutcome>,
    /// Cumulative `P(pass | start = r)` from the decision model, or `None` when no usable prior
    /// exists for this task (failed fetch, malformed probs, or a length mismatch).
    prior: Option<Vec<f64>>,
    /// Cross-fitted mean cost per rung — known before generation, never the task's own cost.
    price: Vec<f64>,
    /// `costaware`'s learned pass-rate estimate for this task's rung-0 cost, cross-fitted the same
    /// way `costaware::study` fits it.
    p_learned: f64,
}

fn mean_cost_per_rung(fold: &[Vec<RungOutcome>], n_rungs: usize) -> Vec<f64> {
    (0..n_rungs)
        .map(|r| {
            let costs: Vec<f64> = fold
                .iter()
                .filter_map(|row| row.get(r))
                .map(|o| o.cost_usd)
                .collect();
            stats::mean(&costs)
        })
        .collect()
}

/// Join matrix rows with their fetched priors and attach cross-fitted prices/pass-rate estimates.
/// 2-fold cross-fitting (even/odd indices), same discipline as `costaware::study`: every task is
/// scored by statistics fitted on the *other* fold, so the whole matrix is used without any task
/// pricing or predicting itself.
fn build_task_fits(
    rows: &[MatrixRow],
    priors: &HashMap<String, PriorRecord>,
    ladder_len: usize,
) -> Vec<TaskFit> {
    let rows_a: Vec<Vec<RungOutcome>> = rows.iter().step_by(2).map(|r| r.rungs.clone()).collect();
    let rows_b: Vec<Vec<RungOutcome>> = rows
        .iter()
        .skip(1)
        .step_by(2)
        .map(|r| r.rungs.clone())
        .collect();

    let pred_for_a = PassPredictor::fit(&rows_b);
    let pred_for_b = PassPredictor::fit(&rows_a);
    let price_for_a = mean_cost_per_rung(&rows_b, ladder_len);
    let price_for_b = mean_cost_per_rung(&rows_a, ladder_len);

    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let in_a = i % 2 == 0;
            let (pred, price) = if in_a {
                (&pred_for_a, &price_for_a)
            } else {
                (&pred_for_b, &price_for_b)
            };
            let p_learned = r.rungs.first().map_or(1.0, |o| pred.p(o.cost_usd));
            let prior = priors.get(&r.task_id).and_then(|p| {
                if !p.raw_ok {
                    return None;
                }
                let probs = p.probs.as_ref()?;
                if probs.len() != ladder_len {
                    return None;
                }
                firstpass_core::cumulative_pass(probs)
            });
            TaskFit {
                row: r.rungs.clone(),
                prior,
                price: price.clone(),
                p_learned,
            }
        })
        .collect()
}

/// The expected-cost argmin start rung under `prior` (`prior[r] = P(pass | start = r)`) and
/// ex-ante `price` per rung. Mirrors `firstpass_proxy::bandit::argmin_expected_cost`'s decision
/// exactly (same walk, same tie rule), just against a flat per-rung price instead of a priced
/// `PriceTable` call — that IS the cross-fitted price this module computes.
///
/// # Panics
/// If `prior.len() != price.len()` (a caller bug — build_task_fits always keeps them in lockstep).
fn argmin_expected_cost_prior(prior: &[f64], price: &[f64]) -> usize {
    assert_eq!(prior.len(), price.len());
    let mut best_s = 0usize;
    let mut best_cost = f64::MAX;
    for s in 0..price.len() {
        let mut expected_cost = 0.0_f64;
        let mut p_reach = 1.0_f64;
        for r in s..price.len() {
            expected_cost += p_reach * price[r];
            p_reach *= 1.0 - prior[r];
            if p_reach < 1e-10 {
                break;
            }
        }
        if expected_cost < best_cost {
            best_cost = expected_cost;
            best_s = s;
        }
    }
    best_s
}

/// `(served_correct, cost_paid, rungs_paid, had_prior)` — every arm below has this shape so one
/// `eval_arm` can score all of them. `had_prior = false` marks a task that fell back to
/// unconditional first-pass because no usable prior existed for it (counted as `fallback_rate`).
type Served = (bool, f64, usize, bool);

fn serve_first_pass(t: &TaskFit) -> Served {
    let (ok, cost, rungs) = costaware::first_pass_from(&t.row, 0);
    (ok, cost, rungs, true)
}

/// A task with no usable prior falls back to unconditional first-pass, marked `had_prior = false`
/// so [`eval_arm`] counts it in `fallback_rate` rather than crediting the prior for a decision it
/// never made.
fn fallback_to_first_pass(t: &TaskFit) -> Served {
    let (ok, cost, rungs) = costaware::first_pass_from(&t.row, 0);
    (ok, cost, rungs, false)
}

fn serve_prior(t: &TaskFit) -> Served {
    match &t.prior {
        Some(p) => {
            let start = argmin_expected_cost_prior(p, &t.price);
            let (ok, cost, rungs) = costaware::first_pass_from(&t.row, start);
            (ok, cost, rungs, true)
        }
        None => fallback_to_first_pass(t),
    }
}

fn serve_prior_unverified(t: &TaskFit) -> Served {
    match &t.prior {
        Some(p) => {
            let start = argmin_expected_cost_prior(p, &t.price);
            t.row.get(start).map_or((false, 0.0, 0, true), |o| {
                (o.oracle_correct, o.cost_usd, 1, true)
            })
        }
        None => fallback_to_first_pass(t),
    }
}

fn serve_always_cheap(t: &TaskFit) -> Served {
    t.row.first().map_or((false, 0.0, 0, true), |o| {
        (o.oracle_correct, o.cost_usd, 1, true)
    })
}

fn serve_always_top(t: &TaskFit) -> Served {
    t.row.last().map_or((false, 0.0, 0, true), |o| {
        (o.oracle_correct, o.cost_usd, 1, true)
    })
}

fn serve_cost_aware_learned(t: &TaskFit) -> Served {
    let (ok, cost, rungs) = costaware::serve(&t.row, t.p_learned);
    (ok, cost, rungs, true)
}

fn serve_cost_aware_oracle(t: &TaskFit) -> Served {
    let p = t
        .row
        .first()
        .map_or(1.0, |o| f64::from(u8::from(o.gate_full_pass)));
    let (ok, cost, rungs) = costaware::serve(&t.row, p);
    (ok, cost, rungs, true)
}

/// One arm's measured result, with bootstrap CIs on every ratio/rate the spec asks for.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArmResult {
    pub name: &'static str,
    pub n: usize,
    pub success_rate: f64,
    pub success_ci: Ci,
    pub usd_per_success: f64,
    pub usd_per_success_ci: Ci,
    pub served_failure_rate: f64,
    pub served_failure_ci: Ci,
    pub escalation_rate: f64,
    /// Fraction of tasks this arm served via the first-pass fallback because no usable prior
    /// existed. Always `0.0` for arms that don't consult a prior.
    pub fallback_rate: f64,
}

/// Per-task success/cost, kept alongside the aggregated [`ArmResult`] so a caller (pooling, the
/// paired diff) can recombine them without re-serving.
struct ArmSeries {
    result: ArmResult,
    success: Vec<f64>,
    cost: Vec<f64>,
}

fn eval_arm<T>(name: &'static str, tasks: &[T], serve: impl Fn(&T) -> Served) -> ArmSeries {
    let mut success = Vec::with_capacity(tasks.len());
    let mut cost = Vec::with_capacity(tasks.len());
    let (mut escalations, mut fallbacks) = (0usize, 0usize);
    for t in tasks {
        let (ok, spent, rungs, had_prior) = serve(t);
        success.push(f64::from(u8::from(ok)));
        cost.push(spent);
        if rungs > 1 {
            escalations += 1;
        }
        if !had_prior {
            fallbacks += 1;
        }
    }
    let n = tasks.len().max(1) as f64;
    let success_rate = stats::mean(&success);
    let failure: Vec<f64> = success.iter().map(|s| 1.0 - s).collect();
    let result = ArmResult {
        name,
        n: tasks.len(),
        success_rate,
        success_ci: bootstrap_mean_ci(&success, BOOT_B, BOOT_SEED, ALPHA),
        usd_per_success: {
            let total: f64 = cost.iter().sum();
            let ns: f64 = success.iter().sum();
            if ns > 0.0 { total / ns } else { f64::INFINITY }
        },
        usd_per_success_ci: bootstrap_ratio_ci(&cost, &success, BOOT_B, BOOT_SEED, ALPHA),
        served_failure_rate: 1.0 - success_rate,
        served_failure_ci: bootstrap_mean_ci(&failure, BOOT_B, BOOT_SEED, ALPHA),
        escalation_rate: escalations as f64 / n,
        fallback_rate: fallbacks as f64 / n,
    };
    ArmSeries {
        result,
        success,
        cost,
    }
}

/// Every argmin start rung the prior chose, for tasks that actually had a usable prior. Empty
/// means no task in this set had a prior at all — reported as degenerate (see [`degeneracy`]).
fn argmin_starts(tasks: &[TaskFit]) -> Vec<usize> {
    tasks
        .iter()
        .filter_map(|t| {
            t.prior
                .as_ref()
                .map(|p| argmin_expected_cost_prior(p, &t.price))
        })
        .collect()
}

/// `(is_degenerate, mode_rung, mode_fraction)`. An empty `starts` (no task ever had a usable
/// prior) is degenerate by definition — there is no per-query signal to have collapsed.
fn degeneracy(starts: &[usize]) -> (bool, usize, f64) {
    if starts.is_empty() {
        return (true, 0, 1.0);
    }
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &s in starts {
        *counts.entry(s).or_insert(0) += 1;
    }
    let (mode_rung, mode_count) = counts.into_iter().max_by_key(|&(_, c)| c).unwrap_or((0, 0));
    let frac = mode_count as f64 / starts.len() as f64;
    (frac > DEGENERATE_THRESHOLD, mode_rung, frac)
}

/// The spec's kill criterion, evaluated on one set of (pooled) arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Verdict {
    Proceed,
    Stop,
    /// Counts as STOP; reported separately because the reason is structural (no per-query
    /// signal), not "the prior didn't help".
    Degenerate,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Verdict::Proceed => "PROCEED",
            Verdict::Stop => "PRIOR=STOP",
            Verdict::Degenerate => "DEGENERATE (counts as PRIOR=STOP)",
        })
    }
}

/// One ladder's (or the pooled) scored arms.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LadderResult {
    pub label: String,
    pub ladder: Vec<String>,
    pub n: usize,
    pub arms: Vec<ArmResult>,
}

/// The whole replay: per-ladder results, the pooled result, the pooled paired diff, the
/// degeneracy guard, and the verdict the spec's kill criterion produces on the pooled result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReplayPriorStudy {
    pub per_ladder: Vec<LadderResult>,
    pub pooled: LadderResult,
    /// Bootstrap CI on `prior − first-pass` for `$/success`, pooled across every ladder.
    pub pooled_usd_per_success_diff_ci: Ci,
    pub degenerate: bool,
    pub degenerate_mode_rung: usize,
    pub degenerate_fraction: f64,
    pub verdict: Verdict,
}

const ARM_NAMES: [&str; 7] = [
    "first-pass",
    "prior",
    "prior-unverified",
    "always-cheap",
    "always-top",
    "cost-aware (learned p, HINDSIGHT)",
    "cost-aware (ORACLE p — cheats)",
];

fn eval_all_arms(tasks: &[TaskFit]) -> Vec<ArmSeries> {
    vec![
        eval_arm(ARM_NAMES[0], tasks, serve_first_pass),
        eval_arm(ARM_NAMES[1], tasks, serve_prior),
        eval_arm(ARM_NAMES[2], tasks, serve_prior_unverified),
        eval_arm(ARM_NAMES[3], tasks, serve_always_cheap),
        eval_arm(ARM_NAMES[4], tasks, serve_always_top),
        eval_arm(ARM_NAMES[5], tasks, serve_cost_aware_learned),
        eval_arm(ARM_NAMES[6], tasks, serve_cost_aware_oracle),
    ]
}

fn ladder_result(label: &str, ladder: &[String], series: &[ArmSeries]) -> LadderResult {
    LadderResult {
        label: label.to_owned(),
        ladder: ladder.to_vec(),
        n: series.first().map_or(0, |a| a.result.n),
        arms: series.iter().map(|a| a.result.clone()).collect(),
    }
}

/// Task ids a resumed fetch may skip: only the ones that already have a *successful* prior. A
/// failed record (`raw_ok: false`) is retried, and the retry is appended; [`load_priors`] keeps the
/// last record per task, so the retry wins. Skipping failures instead made a keyless first run
/// poison the file: a rerun with a valid key would skip every task.
fn resumable_ids(text: &str) -> std::collections::HashSet<String> {
    text.lines()
        .filter_map(|l| serde_json::from_str::<PriorRecord>(l).ok())
        .filter(|r| r.raw_ok)
        .map(|r| r.task_id)
        .collect()
}

/// Load a priors JSONL file into a `task_id -> PriorRecord` map.
///
/// # Errors
/// Unreadable file. A malformed line is skipped (not fatal): `run_fetch_priors` may have been
/// interrupted mid-write, and a partial last line must not sink the whole replay.
fn load_priors(path: &str) -> Result<HashMap<String, PriorRecord>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<PriorRecord>(l).ok())
        .map(|r| (r.task_id.clone(), r))
        .collect())
}

/// Replay every `(matrix, priors)` pair, score the spec's arms per ladder and pooled, and produce
/// the pre-registered verdict on the pooled result. Pure once the files are read — deterministic,
/// no spend, no network.
///
/// # Errors
/// Any matrix or priors file can't be read/parsed, or `pairs` is empty.
pub fn run_replay_prior(pairs: &[(String, String)]) -> Result<ReplayPriorStudy, String> {
    if pairs.is_empty() {
        return Err("--replay-prior needs at least one <matrix> <priors> pair".to_owned());
    }

    let mut per_ladder = Vec::with_capacity(pairs.len());
    let mut pooled_tasks: Vec<TaskFit> = Vec::new();

    for (matrix_path, priors_path) in pairs {
        let (rows, ladder) = load_matrix_with_ids(matrix_path)?;
        let priors = load_priors(priors_path)?;
        let tasks = build_task_fits(&rows, &priors, ladder.len());
        let series = eval_all_arms(&tasks);
        per_ladder.push(ladder_result(matrix_path, &ladder, &series));
        pooled_tasks.extend(tasks);
    }

    let pooled_series = eval_all_arms(&pooled_tasks);
    let pooled = ladder_result("POOLED", &[], &pooled_series);

    let fp = &pooled_series[0];
    let prior = &pooled_series[1];
    let diff_ci = bootstrap_paired_ratio_diff_ci(
        &prior.cost,
        &prior.success,
        &fp.cost,
        &fp.success,
        BOOT_B,
        BOOT_SEED,
        ALPHA,
    );

    let starts = argmin_starts(&pooled_tasks);
    let (degenerate, degenerate_mode_rung, degenerate_fraction) = degeneracy(&starts);

    let cost_lower_excludes_zero = diff_ci.hi < 0.0;
    let failure_not_worse =
        prior.result.served_failure_rate <= fp.result.served_failure_rate + 0.01;
    let verdict = if degenerate {
        Verdict::Degenerate
    } else if cost_lower_excludes_zero && failure_not_worse {
        Verdict::Proceed
    } else {
        Verdict::Stop
    };

    Ok(ReplayPriorStudy {
        per_ladder,
        pooled,
        pooled_usd_per_success_diff_ci: diff_ci,
        degenerate,
        degenerate_mode_rung,
        degenerate_fraction,
        verdict,
    })
}

/// Markdown render.
#[must_use]
pub fn render(s: &ReplayPriorStudy) -> String {
    let mut out = String::new();
    out.push_str("## OpenJev decision-model prior — replayed on real MBPP matrices\n\n");
    out.push_str(
        "Pre-registration: `specs/openjev-prior-replay-ab.md`. `first-pass`/`prior` are the two \
         arms the kill criterion compares; the rest are context.\n\n",
    );

    let render_table = |lr: &LadderResult| -> String {
        let mut t = String::new();
        t.push_str(&format!("### {} (n = {})\n\n", lr.label, lr.n));
        t.push_str(
            "| arm | success | 95% CI | $/success | 95% CI | served-failure | escalated | fallback (no prior) |\n\
             |---|---|---|---|---|---|---|---|\n",
        );
        for a in &lr.arms {
            t.push_str(&format!(
                "| {} | {:.4} | [{:.4}, {:.4}] | ${:.5} | [${:.5}, ${:.5}] | {:.4} | {:.0}% | {:.0}% |\n",
                a.name,
                a.success_rate,
                a.success_ci.lo,
                a.success_ci.hi,
                a.usd_per_success,
                a.usd_per_success_ci.lo,
                a.usd_per_success_ci.hi,
                a.served_failure_rate,
                a.escalation_rate * 100.0,
                a.fallback_rate * 100.0,
            ));
        }
        t
    };

    for lr in &s.per_ladder {
        out.push_str(&render_table(lr));
        out.push('\n');
    }
    out.push_str(&render_table(&s.pooled));

    out.push_str(&format!(
        "\n**Pooled `$/success` (prior − first-pass): {:+.5} [{:+.5}, {:+.5}]**\n",
        s.pooled_usd_per_success_diff_ci.point,
        s.pooled_usd_per_success_diff_ci.lo,
        s.pooled_usd_per_success_diff_ci.hi,
    ));
    out.push_str(&format!(
        "\nDegeneracy guard: mode start rung {} carries {:.1}% of prior-covered decisions ({}).\n",
        s.degenerate_mode_rung,
        s.degenerate_fraction * 100.0,
        if s.degenerate {
            "DEGENERATE — over the 95% threshold"
        } else {
            "below the 95% threshold"
        }
    ));
    out.push_str(&format!("\n**Verdict: {}**\n", s.verdict));
    out
}

// ---------------------------------------------------------------------------------------------
// Study A: prior + learned blend (`specs/prior-blend-and-decision-gate.md`)
// ---------------------------------------------------------------------------------------------

/// `[escalation.prior] strength`'s pre-registered default — the pseudo-count weight the prior
/// carries in the blend, same constant `PriorConfig::default_prior_strength` ships.
const BLEND_STRENGTH: f64 = 10.0;

/// Quartile buckets of the ex-ante MBPP prompt character length, same resolution as
/// `costaware::PassPredictor`'s cost buckets.
const EXANTE_BUCKETS: usize = 4;

/// A fitted `P(rung 0 clears the gate | prompt char length)`, keeping the raw per-bucket counts
/// (`hit`, `seen`) rather than folding them into a rate up front — `prior+learned` blends against
/// the counts themselves, not their ratio.
struct ExAntePredictor {
    /// Upper char-length edge of each bucket (last is infinity).
    edges: Vec<f64>,
    hit: Vec<usize>,
    seen: Vec<usize>,
    /// Calibration-split base rate, used where a bucket held no examples.
    base: f64,
}

impl ExAntePredictor {
    /// Fit on `(prompt_chars, rung_0_gate_full_pass)` pairs from a calibration fold.
    /// Deterministic: quartile edges come from the sorted lengths.
    fn fit(calib: &[(f64, bool)]) -> Self {
        let mut lens: Vec<f64> = calib.iter().map(|(l, _)| *l).collect();
        lens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let edges: Vec<f64> = (1..EXANTE_BUCKETS)
            .map(|i| {
                let idx = i * lens.len() / EXANTE_BUCKETS;
                lens.get(idx).copied().unwrap_or(f64::INFINITY)
            })
            .chain(std::iter::once(f64::INFINITY))
            .collect();

        let mut hit = vec![0usize; EXANTE_BUCKETS];
        let mut seen = vec![0usize; EXANTE_BUCKETS];
        let mut passed = 0usize;
        for (len, pass) in calib {
            let b = Self::bucket_of(&edges, *len);
            seen[b] += 1;
            if *pass {
                hit[b] += 1;
                passed += 1;
            }
        }
        Self {
            edges,
            hit,
            seen,
            base: if calib.is_empty() {
                0.0
            } else {
                passed as f64 / calib.len() as f64
            },
        }
    }

    fn bucket_of(edges: &[f64], len: f64) -> usize {
        edges
            .iter()
            .position(|e| len <= *e)
            .unwrap_or(EXANTE_BUCKETS - 1)
            .min(EXANTE_BUCKETS - 1)
    }

    /// `P(rung 0 passes)` for a prompt this long — bucket rate, base-rate fallback for an empty
    /// bucket.
    fn rate(&self, len: f64) -> f64 {
        let b = Self::bucket_of(&self.edges, len);
        if self.seen[b] > 0 {
            self.hit[b] as f64 / self.seen[b] as f64
        } else {
            self.base
        }
    }

    /// `(passes_b, seen_b)` for the bucket this length falls in — the raw counts `prior+learned`
    /// blends against, not the rate.
    fn counts(&self, len: f64) -> (usize, usize) {
        let b = Self::bucket_of(&self.edges, len);
        (self.hit[b], self.seen[b])
    }
}

/// The blend formula: `(s·prior_r0 + passes_b) / (s + seen_b)`. At `seen_b = 0` this is exactly
/// `prior_r0` (no traffic statistics to blend in yet); as `seen_b` grows past `s`, it converges to
/// `passes_b / seen_b`, the bucket's own rate.
fn blended_r0(prior_r0: f64, passes_b: usize, seen_b: usize) -> f64 {
    (BLEND_STRENGTH * prior_r0 + passes_b as f64) / (BLEND_STRENGTH + seen_b as f64)
}

/// Build a `[p0, 1.0, 1.0, ...]` per-rung pass vector from a single rung-0 pass probability,
/// mirroring how `prior`'s cumulative vector is always forced to `1.0` at the last rung (some
/// rung must suffice). No ex-ante or blended signal is defined for rungs between 0 and the last —
/// every ladder this spec measures has exactly two rungs, so this is the 2-rung case the spec
/// pre-registers (`P(pass r1) = 1`), generalised by treating any further rung the same way.
fn r0_only_pass_vector(r0: f64, n_rungs: usize) -> Vec<f64> {
    let mut v = vec![1.0; n_rungs.max(1)];
    v[0] = r0.clamp(0.0, 1.0);
    v
}

/// One task for Study A: the same row/prior/price [`TaskFit`] carries, plus the ex-ante prompt
/// length feature and what the calibration-fold ex-ante predictor says about it. `None` means no
/// MBPP prompt matched this task's id — the ex-ante and blended arms then fall back to first-pass,
/// same discipline as a missing prior.
struct BlendTaskFit {
    row: Vec<RungOutcome>,
    prior: Option<Vec<f64>>,
    price: Vec<f64>,
    /// `costaware`'s hindsight pass-rate estimate — kept only for the `learned-p (hindsight)`
    /// reference arm and the leak-size report, never for a decision an ex-ante arm makes.
    p_hindsight: f64,
    ex_ante_r0: Option<f64>,
    /// `(passes_b, seen_b)` for this task's ex-ante bucket. `None` when no MBPP prompt matched —
    /// distinct from `Some((0, 0))`, which means the bucket matched but was empty on this fold.
    ex_ante_counts: Option<(usize, usize)>,
}

/// Join matrix rows with priors and the ex-ante prompt-length feature, attaching cross-fitted
/// prices, the hindsight pass-rate estimate, and the ex-ante rate/counts — all fit the same
/// 2-fold way `build_task_fits` fits the prior study, so no task ever prices, predicts, or
/// buckets itself.
fn build_blend_task_fits(
    rows: &[MatrixRow],
    priors: &HashMap<String, PriorRecord>,
    prompt_chars: &HashMap<String, f64>,
    ladder_len: usize,
) -> Vec<BlendTaskFit> {
    let fold_a: Vec<&MatrixRow> = rows.iter().step_by(2).collect();
    let fold_b: Vec<&MatrixRow> = rows.iter().skip(1).step_by(2).collect();
    let rungs_a: Vec<Vec<RungOutcome>> = fold_a.iter().map(|r| r.rungs.clone()).collect();
    let rungs_b: Vec<Vec<RungOutcome>> = fold_b.iter().map(|r| r.rungs.clone()).collect();

    let pred_for_a = PassPredictor::fit(&rungs_b);
    let pred_for_b = PassPredictor::fit(&rungs_a);
    let price_for_a = mean_cost_per_rung(&rungs_b, ladder_len);
    let price_for_b = mean_cost_per_rung(&rungs_a, ladder_len);

    let exante_calib = |fold: &[&MatrixRow]| -> Vec<(f64, bool)> {
        fold.iter()
            .filter_map(|r| {
                let len = *prompt_chars.get(&r.task_id)?;
                let pass = r.rungs.first().is_some_and(|o| o.gate_full_pass);
                Some((len, pass))
            })
            .collect()
    };
    let exante_for_a = ExAntePredictor::fit(&exante_calib(&fold_b));
    let exante_for_b = ExAntePredictor::fit(&exante_calib(&fold_a));

    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let in_a = i % 2 == 0;
            let (pred, price, exante) = if in_a {
                (&pred_for_a, &price_for_a, &exante_for_a)
            } else {
                (&pred_for_b, &price_for_b, &exante_for_b)
            };
            let p_hindsight = r.rungs.first().map_or(1.0, |o| pred.p(o.cost_usd));
            let prior = priors.get(&r.task_id).and_then(|p| {
                if !p.raw_ok {
                    return None;
                }
                let probs = p.probs.as_ref()?;
                if probs.len() != ladder_len {
                    return None;
                }
                firstpass_core::cumulative_pass(probs)
            });
            let (ex_ante_r0, ex_ante_counts) = match prompt_chars.get(&r.task_id) {
                Some(&len) => (Some(exante.rate(len)), Some(exante.counts(len))),
                None => (None, None),
            };
            BlendTaskFit {
                row: r.rungs.clone(),
                prior,
                price: price.clone(),
                p_hindsight,
                ex_ante_r0,
                ex_ante_counts,
            }
        })
        .collect()
}

/// `task_id -> MBPP prompt character length`, the ex-ante feature — known before generation,
/// never the task's own realized cost.
///
/// # Errors
/// The MBPP file can't be read/parsed.
fn build_prompt_char_map(
    mbpp_path: &str,
    rows: &[MatrixRow],
) -> Result<HashMap<String, f64>, String> {
    let mbpp = load_mbpp_examples(mbpp_path)?;
    Ok(rows
        .iter()
        .filter_map(|r| {
            let id = mbpp_numeric_id(&r.task_id)?;
            let ex = mbpp.get(&id)?;
            Some((r.task_id.clone(), ex.text.chars().count() as f64))
        })
        .collect())
}

fn serve_first_pass_blend(t: &BlendTaskFit) -> Served {
    let (ok, cost, rungs) = costaware::first_pass_from(&t.row, 0);
    (ok, cost, rungs, true)
}

fn fallback_to_first_pass_blend(t: &BlendTaskFit) -> Served {
    let (ok, cost, rungs) = costaware::first_pass_from(&t.row, 0);
    (ok, cost, rungs, false)
}

fn serve_prior_blend(t: &BlendTaskFit) -> Served {
    match &t.prior {
        Some(p) => {
            let start = argmin_expected_cost_prior(p, &t.price);
            let (ok, cost, rungs) = costaware::first_pass_from(&t.row, start);
            (ok, cost, rungs, true)
        }
        None => fallback_to_first_pass_blend(t),
    }
}

/// `learned-p (ex-ante)`: decides purely from the calibration-fold bucket rate for this task's
/// prompt length and the cross-fitted price — never the task's own `cost_usd`.
fn serve_learned_exante(t: &BlendTaskFit) -> Served {
    match t.ex_ante_r0 {
        Some(r0) => {
            let v = r0_only_pass_vector(r0, t.price.len());
            let start = argmin_expected_cost_prior(&v, &t.price);
            let (ok, cost, rungs) = costaware::first_pass_from(&t.row, start);
            (ok, cost, rungs, true)
        }
        None => fallback_to_first_pass_blend(t),
    }
}

/// `prior+learned`: the posterior mean blend of the OpenJev prior and the ex-ante traffic
/// statistics, decided the same expected-cost-argmin way as every other arm. Falls back to
/// first-pass only when no prior exists at all; a missing ex-ante bucket degrades gracefully to
/// the prior alone (`passes_b = seen_b = 0` leaves `blended_r0` unchanged).
fn serve_prior_plus_learned(t: &BlendTaskFit) -> Served {
    match &t.prior {
        Some(p) => {
            let (passes_b, seen_b) = t.ex_ante_counts.unwrap_or((0, 0));
            let r0 = blended_r0(p[0], passes_b, seen_b);
            let v = r0_only_pass_vector(r0, t.price.len());
            let start = argmin_expected_cost_prior(&v, &t.price);
            let (ok, cost, rungs) = costaware::first_pass_from(&t.row, start);
            (ok, cost, rungs, true)
        }
        None => fallback_to_first_pass_blend(t),
    }
}

fn serve_always_top_blend(t: &BlendTaskFit) -> Served {
    t.row.last().map_or((false, 0.0, 0, true), |o| {
        (o.oracle_correct, o.cost_usd, 1, true)
    })
}

/// `learned-p (hindsight)`: reference only — `costaware::PassPredictor` buckets by the task's own
/// realized rung-0 cost, which exists only after generation. Never the comparison target.
fn serve_learned_hindsight(t: &BlendTaskFit) -> Served {
    let (ok, cost, rungs) = costaware::serve(&t.row, t.p_hindsight);
    (ok, cost, rungs, true)
}

const BLEND_ARM_NAMES: [&str; 6] = [
    "first-pass",
    "prior",
    "learned-p (ex-ante)",
    "prior+learned",
    "learned-p (hindsight)",
    "always-top",
];

fn eval_all_blend_arms(tasks: &[BlendTaskFit]) -> Vec<ArmSeries> {
    vec![
        eval_arm(BLEND_ARM_NAMES[0], tasks, serve_first_pass_blend),
        eval_arm(BLEND_ARM_NAMES[1], tasks, serve_prior_blend),
        eval_arm(BLEND_ARM_NAMES[2], tasks, serve_learned_exante),
        eval_arm(BLEND_ARM_NAMES[3], tasks, serve_prior_plus_learned),
        eval_arm(BLEND_ARM_NAMES[4], tasks, serve_learned_hindsight),
        eval_arm(BLEND_ARM_NAMES[5], tasks, serve_always_top_blend),
    ]
}

/// The spec's Study A pooled verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum BlendVerdict {
    Helps,
    Neutral,
}

impl std::fmt::Display for BlendVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BlendVerdict::Helps => "BLEND-HELPS",
            BlendVerdict::Neutral => "BLEND-NEUTRAL",
        })
    }
}

/// The whole Study A replay: per-ladder and pooled arms, the pooled paired diff (`prior+learned` −
/// `prior`), the reported (not gated) leak size, and the verdict.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReplayBlendStudy {
    pub per_ladder: Vec<LadderResult>,
    pub pooled: LadderResult,
    /// Bootstrap CI on `prior+learned − prior` for `$/success`, pooled across every ladder.
    pub pooled_usd_per_success_diff_ci: Ci,
    /// `learned-p (hindsight)` minus `learned-p (ex-ante)`, pooled `$/success` — the size of the
    /// hindsight leak the ex-ante arm closes. Reported, not gated.
    pub leak_usd_per_success: f64,
    pub verdict: BlendVerdict,
}

/// Replay every `(matrix, priors)` pair against the shared `mbpp_path`, score Study A's arms per
/// ladder and pooled, and produce the pre-registered pooled verdict. Pure once the files are read.
///
/// # Errors
/// The MBPP file, any matrix, or any priors file can't be read/parsed, or `pairs` is empty.
pub fn run_replay_blend(
    mbpp_path: &str,
    pairs: &[(String, String)],
) -> Result<ReplayBlendStudy, String> {
    if pairs.is_empty() {
        return Err("--replay-blend needs at least one <matrix> <priors> pair".to_owned());
    }

    let mut per_ladder = Vec::with_capacity(pairs.len());
    let mut pooled_tasks: Vec<BlendTaskFit> = Vec::new();

    for (matrix_path, priors_path) in pairs {
        let (rows, ladder) = load_matrix_with_ids(matrix_path)?;
        let priors = load_priors(priors_path)?;
        let prompt_chars = build_prompt_char_map(mbpp_path, &rows)?;
        let tasks = build_blend_task_fits(&rows, &priors, &prompt_chars, ladder.len());
        let series = eval_all_blend_arms(&tasks);
        per_ladder.push(ladder_result(matrix_path, &ladder, &series));
        pooled_tasks.extend(tasks);
    }

    let pooled_series = eval_all_blend_arms(&pooled_tasks);
    let pooled = ladder_result("POOLED", &[], &pooled_series);

    let prior = &pooled_series[1];
    let blend = &pooled_series[3];
    let exante = &pooled_series[2];
    let hindsight = &pooled_series[4];

    let diff_ci = bootstrap_paired_ratio_diff_ci(
        &blend.cost,
        &blend.success,
        &prior.cost,
        &prior.success,
        BOOT_B,
        BOOT_SEED,
        ALPHA,
    );
    let leak_usd_per_success = hindsight.result.usd_per_success - exante.result.usd_per_success;

    let cost_lower_excludes_zero = diff_ci.hi < 0.0;
    let failure_not_worse =
        blend.result.served_failure_rate <= prior.result.served_failure_rate + 0.01;
    let verdict = if cost_lower_excludes_zero && failure_not_worse {
        BlendVerdict::Helps
    } else {
        BlendVerdict::Neutral
    };

    Ok(ReplayBlendStudy {
        per_ladder,
        pooled,
        pooled_usd_per_success_diff_ci: diff_ci,
        leak_usd_per_success,
        verdict,
    })
}

/// Markdown render for Study A.
#[must_use]
pub fn render_blend(s: &ReplayBlendStudy) -> String {
    let mut out = String::new();
    out.push_str("## Prior + learned blend — Study A (real MBPP matrices)\n\n");
    out.push_str(
        "Pre-registration: `specs/prior-blend-and-decision-gate.md`. `prior`/`prior+learned` are \
         the two arms the verdict compares; the rest are context. `learned-p (hindsight)` is \
         reference only — it uses the task's own realized cost and must never be the comparison \
         target.\n\n",
    );

    let render_table = |lr: &LadderResult| -> String {
        let mut t = String::new();
        t.push_str(&format!("### {} (n = {})\n\n", lr.label, lr.n));
        t.push_str(
            "| arm | success | 95% CI | $/success | 95% CI | served-failure | escalated | fallback (no prior) |\n\
             |---|---|---|---|---|---|---|---|\n",
        );
        for a in &lr.arms {
            t.push_str(&format!(
                "| {} | {:.4} | [{:.4}, {:.4}] | ${:.5} | [${:.5}, ${:.5}] | {:.4} | {:.0}% | {:.0}% |\n",
                a.name,
                a.success_rate,
                a.success_ci.lo,
                a.success_ci.hi,
                a.usd_per_success,
                a.usd_per_success_ci.lo,
                a.usd_per_success_ci.hi,
                a.served_failure_rate,
                a.escalation_rate * 100.0,
                a.fallback_rate * 100.0,
            ));
        }
        t
    };

    for lr in &s.per_ladder {
        out.push_str(&render_table(lr));
        out.push('\n');
    }
    out.push_str(&render_table(&s.pooled));

    out.push_str(&format!(
        "\n**Pooled `$/success` (prior+learned − prior): {:+.5} [{:+.5}, {:+.5}]**\n",
        s.pooled_usd_per_success_diff_ci.point,
        s.pooled_usd_per_success_diff_ci.lo,
        s.pooled_usd_per_success_diff_ci.hi,
    ));
    out.push_str(&format!(
        "\nHindsight leak (learned-p (hindsight) − learned-p (ex-ante), pooled `$/success`): \
         {:+.5}. Reported, not gated.\n",
        s.leak_usd_per_success,
    ));
    out.push_str(&format!("\n**Verdict: {}**\n", s.verdict));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failed record must not be skipped on resume, or a keyless first run poisons the file.
    #[test]
    fn resume_skips_only_successful_priors_and_retries_win() {
        let text = concat!(
            r#"{"task_id":"mbpp-1","ladder":["a","b"],"probs":[0.9,0.1],"raw_ok":true,"latency_ms":1}"#,
            "\n",
            r#"{"task_id":"mbpp-2","ladder":["a","b"],"probs":null,"raw_ok":false,"latency_ms":1}"#,
            "\n",
        );
        let ids = resumable_ids(text);
        assert!(ids.contains("mbpp-1"));
        assert!(!ids.contains("mbpp-2"), "failed fetch must be retried");

        let dir = std::env::temp_dir().join(format!("fp-resume-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("p.jsonl");
        let retried = format!(
            "{text}{}\n",
            r#"{"task_id":"mbpp-2","ladder":["a","b"],"probs":[0.2,0.8],"raw_ok":true,"latency_ms":1}"#
        );
        std::fs::write(&path, retried).expect("write");
        let loaded = load_priors(path.to_str().expect("utf8")).expect("load");
        assert!(loaded["mbpp-2"].raw_ok, "the later successful retry wins");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn rung(gate: bool, oracle: bool, cost: f64) -> RungOutcome {
        RungOutcome {
            gate_score: f64::from(u8::from(gate)),
            gate_full_pass: gate,
            oracle_correct: oracle,
            cost_usd: cost,
            judge_score: None,
        }
    }

    // ---- cumulative mapping via core -------------------------------------------------

    #[test]
    fn cumulative_pass_feeds_the_argmin_as_expected() {
        // A confident-easy distribution should push the argmin to rung 0 regardless of price.
        let prior = firstpass_core::cumulative_pass(&[0.9, 0.1]).unwrap();
        assert_eq!(argmin_expected_cost_prior(&prior, &[0.01, 0.10]), 0);
    }

    // ---- argmin picks rung 1 when the prior says r0 is hopeless, at realistic prices --

    #[test]
    fn argmin_skips_a_hopeless_cheap_rung_at_realistic_prices() {
        // r0 almost never suffices; r1 almost always does. c0 is not free, so paying it AND
        // then paying c1 anyway is worse than starting at c1.
        let prior = firstpass_core::cumulative_pass(&[0.02, 0.98]).unwrap();
        let price = vec![0.006, 0.030]; // realistic haiku/sonnet-scale MBPP prices
        assert_eq!(argmin_expected_cost_prior(&prior, &price), 1);
    }

    #[test]
    fn argmin_ties_prefer_the_lower_rung() {
        // E[0] = c0 + (1-p0)*c1; an exact tie with E[1] = c1 needs c0 == p0*c1. The lower rung
        // must win because `<` (not `<=`) drives the update in `argmin_expected_cost_prior`.
        let prior = vec![0.5, 1.0];
        let price = vec![0.025, 0.05]; // c0 = 0.025 = 0.5 * 0.05 = p0 * c1 -> exact tie
        assert_eq!(argmin_expected_cost_prior(&prior, &price), 0);
    }

    fn two_rung_row(g0: bool, o0: bool, c0: f64, g1: bool, o1: bool, c1: f64) -> Vec<RungOutcome> {
        vec![rung(g0, o0, c0), rung(g1, o1, c1)]
    }

    fn task(row: Vec<RungOutcome>, prior: Option<Vec<f64>>, price: Vec<f64>) -> TaskFit {
        TaskFit {
            row,
            prior,
            price: price.clone(),
            p_learned: 0.5,
        }
    }

    // ---- degenerate guard fires ---------------------------------------------------------

    #[test]
    fn degeneracy_guard_fires_above_95_percent() {
        // 96/100 tasks pick rung 1, 4 pick rung 0 -> 96% > 95% threshold.
        let starts: Vec<usize> = (0..100).map(|i| usize::from(i >= 4)).collect();
        let (deg, mode, frac) = degeneracy(&starts);
        assert!(deg);
        assert_eq!(mode, 1);
        assert!((frac - 0.96).abs() < 1e-9);
    }

    #[test]
    fn degeneracy_guard_does_not_fire_below_threshold() {
        let starts: Vec<usize> = (0..100).map(|i| usize::from(i >= 10)).collect(); // 90/10
        let (deg, _, frac) = degeneracy(&starts);
        assert!(!deg);
        assert!((frac - 0.90).abs() < 1e-9);
    }

    #[test]
    fn no_prior_at_all_is_degenerate() {
        let (deg, _, frac) = degeneracy(&[]);
        assert!(deg);
        assert!((frac - 1.0).abs() < 1e-9);
    }

    // ---- kill verdict PROCEED/STOP on constructed fixtures -------------------------------

    /// A prior that correctly identifies which tasks need the top rung, priced realistically,
    /// must PROCEED: cheaper $/success than first-pass, no worse served-failure.
    #[test]
    fn kill_verdict_proceeds_when_the_prior_earns_its_keep() {
        let mut tasks = Vec::new();
        // 40 "hard" tasks: rung 0 wastes money and fails; rung 1 passes. A correct prior starts
        // these at rung 1, skipping the wasted rung-0 spend that first-pass always pays.
        for _ in 0..40 {
            tasks.push(task(
                two_rung_row(false, false, 0.02, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.02, 0.98]),
                vec![0.02, 0.05],
            ));
        }
        // 60 "easy" tasks: rung 0 passes cheaply. A correct prior starts these at rung 0, same as
        // first-pass, so there is no quality cost.
        for _ in 0..60 {
            tasks.push(task(
                two_rung_row(true, true, 0.01, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.98, 0.02]),
                vec![0.02, 0.05],
            ));
        }
        let series = eval_all_arms(&tasks);
        let fp = &series[0];
        let prior = &series[1];
        let diff_ci = bootstrap_paired_ratio_diff_ci(
            &prior.cost,
            &prior.success,
            &fp.cost,
            &fp.success,
            BOOT_B,
            BOOT_SEED,
            ALPHA,
        );
        let starts = argmin_starts(&tasks);
        let (degenerate, _, _) = degeneracy(&starts);
        let cost_lower = diff_ci.hi < 0.0;
        let failure_ok = prior.result.served_failure_rate <= fp.result.served_failure_rate + 0.01;
        // Sanity: the prior must actually be cheaper AND not degenerate for this fixture to be a
        // meaningful PROCEED test — if either flips the fixture stopped testing what it claims to.
        assert!(cost_lower, "prior must be cheaper: diff CI {diff_ci:?}");
        assert!(failure_ok);
        assert!(
            !degenerate,
            "60/40 split must not trip the degeneracy guard"
        );
        let verdict = if degenerate {
            Verdict::Degenerate
        } else if cost_lower && failure_ok {
            Verdict::Proceed
        } else {
            Verdict::Stop
        };
        assert_eq!(verdict, Verdict::Proceed);
    }

    /// A prior with no signal (same distribution regardless of the task's true difficulty) must
    /// STOP: it cannot beat first-pass's $/success once tested against the same hard/easy mix.
    #[test]
    fn kill_verdict_stops_when_the_prior_has_no_signal() {
        let mut tasks = Vec::new();
        // Same hard/easy mix as above, but the prior is UNINFORMATIVE — identical for every task
        // — so it cannot tell hard from easy and buys nothing over first-pass.
        for _ in 0..40 {
            tasks.push(task(
                two_rung_row(false, false, 0.02, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
            ));
        }
        for _ in 0..60 {
            tasks.push(task(
                two_rung_row(true, true, 0.01, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
            ));
        }
        let series = eval_all_arms(&tasks);
        let fp = &series[0];
        let prior = &series[1];
        let diff_ci = bootstrap_paired_ratio_diff_ci(
            &prior.cost,
            &prior.success,
            &fp.cost,
            &fp.success,
            BOOT_B,
            BOOT_SEED,
            ALPHA,
        );
        let cost_lower = diff_ci.hi < 0.0;
        let verdict = if cost_lower {
            Verdict::Proceed
        } else {
            Verdict::Stop
        };
        assert_eq!(verdict, Verdict::Stop);
    }

    /// **Mutation test**: flip the sign convention (as if a future edit swapped `prior - fp` for
    /// `fp - prior`) and confirm the PROCEED fixture above would then report STOP — i.e. the
    /// assertion actually depends on getting the subtraction order right, not on a tautology.
    #[test]
    fn kill_verdict_test_is_sensitive_to_the_diff_sign() {
        let mut tasks = Vec::new();
        for _ in 0..40 {
            tasks.push(task(
                two_rung_row(false, false, 0.02, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.02, 0.98]),
                vec![0.02, 0.05],
            ));
        }
        for _ in 0..60 {
            tasks.push(task(
                two_rung_row(true, true, 0.01, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.98, 0.02]),
                vec![0.02, 0.05],
            ));
        }
        let series = eval_all_arms(&tasks);
        let fp = &series[0];
        let prior = &series[1];
        // Mutated: fp - prior instead of prior - fp.
        let mutated_ci = bootstrap_paired_ratio_diff_ci(
            &fp.cost,
            &fp.success,
            &prior.cost,
            &prior.success,
            BOOT_B,
            BOOT_SEED,
            ALPHA,
        );
        // Under the correct sign this fixture is `cost_lower = hi < 0`; under the mutated
        // (flipped) sign the interval must NOT satisfy the same test, proving the test would have
        // caught the mutation.
        assert!(
            mutated_ci.hi >= 0.0,
            "mutated sign convention must not still read as PROCEED"
        );
    }

    // ---- fetch parsing on both response shapes --------------------------------------------

    #[test]
    fn extract_probabilities_nested_under_answers() {
        let json = serde_json::json!({
            "answers": { "tier": { "probabilities": { "r0": 0.3, "r1": 0.7 } } }
        });
        assert_eq!(extract_probabilities(&json, 2), Some(vec![0.3, 0.7]));
    }

    #[test]
    fn extract_probabilities_flat_top_level() {
        let json = serde_json::json!({
            "tier": { "probabilities": { "r0": 0.6, "r1": 0.4 } }
        });
        assert_eq!(extract_probabilities(&json, 2), Some(vec![0.6, 0.4]));
    }

    #[test]
    fn extract_probabilities_missing_option_defaults_to_zero() {
        let json = serde_json::json!({ "tier": { "probabilities": { "r0": 1.0 } } });
        assert_eq!(extract_probabilities(&json, 2), Some(vec![1.0, 0.0]));
    }

    #[test]
    fn extract_probabilities_rejects_unexpected_shape() {
        assert_eq!(extract_probabilities(&serde_json::json!({}), 2), None);
        assert_eq!(
            extract_probabilities(&serde_json::json!({"tier": {}}), 2),
            None
        );
        assert_eq!(
            extract_probabilities(&serde_json::json!({"tier": {"probabilities": "nope"}}), 2),
            None
        );
    }

    #[test]
    fn build_request_shape_matches_the_spec() {
        let ladder = vec!["haiku".to_owned(), "sonnet".to_owned()];
        let body = build_request(&ladder, "do the thing", "assert f(1) == 2");
        assert_eq!(body["model"], "jev-latest");
        assert_eq!(body["state"]["task"], "do the thing");
        assert_eq!(body["state"]["example_test"], "assert f(1) == 2");
        assert_eq!(body["questions"]["tier"]["type"], "choice");
        assert_eq!(
            body["questions"]["tier"]["criteria"]["r0"],
            "haiku (the smaller, cheaper model) fully solves this"
        );
        assert_eq!(
            body["questions"]["tier"]["criteria"]["r1"],
            "sonnet (the frontier model) is needed to solve this"
        );
    }

    // ---- matrix/mbpp id plumbing --------------------------------------------------------

    #[test]
    fn mbpp_numeric_id_parses_the_prefix() {
        assert_eq!(mbpp_numeric_id("mbpp-974"), Some(974));
        assert_eq!(mbpp_numeric_id("mbpp-1"), Some(1));
        assert_eq!(mbpp_numeric_id("not-mbpp"), None);
        assert_eq!(mbpp_numeric_id("mbpp-abc"), None);
    }

    // ---- prior fallback: a task with no usable prior must fall back to first-pass --------

    #[test]
    fn missing_prior_falls_back_to_first_pass_and_is_counted() {
        let t = task(
            two_rung_row(false, false, 0.02, true, true, 0.05),
            None,
            vec![0.02, 0.05],
        );
        let served = serve_prior(&t);
        let fp = serve_first_pass(&t);
        assert_eq!((served.0, served.1, served.2), (fp.0, fp.1, fp.2));
        assert!(!served.3, "must be counted as a fallback");
    }

    #[test]
    fn missing_prior_unverified_also_falls_back() {
        let t = task(
            two_rung_row(false, false, 0.02, true, true, 0.05),
            None,
            vec![0.02, 0.05],
        );
        let served = serve_prior_unverified(&t);
        assert!(!served.3);
    }

    // ---- build_task_fits joins by task_id correctly ----------------------------------------

    #[test]
    fn build_task_fits_uses_the_prior_only_when_lengths_match() {
        let rows = vec![
            MatrixRow {
                task_id: "mbpp-1".to_owned(),
                ladder: vec!["a".to_owned(), "b".to_owned()],
                rungs: two_rung_row(false, false, 0.01, true, true, 0.05),
            },
            MatrixRow {
                task_id: "mbpp-2".to_owned(),
                ladder: vec!["a".to_owned(), "b".to_owned()],
                rungs: two_rung_row(true, true, 0.01, true, true, 0.05),
            },
        ];
        let mut priors = HashMap::new();
        priors.insert(
            "mbpp-1".to_owned(),
            PriorRecord {
                task_id: "mbpp-1".to_owned(),
                ladder: vec!["a".to_owned(), "b".to_owned()],
                probs: Some(vec![0.1, 0.9]),
                raw_ok: true,
                latency_ms: 5,
            },
        );
        // mbpp-2 has a raw_ok:false record -> no prior should be attached.
        priors.insert(
            "mbpp-2".to_owned(),
            PriorRecord {
                task_id: "mbpp-2".to_owned(),
                ladder: vec!["a".to_owned(), "b".to_owned()],
                probs: None,
                raw_ok: false,
                latency_ms: 120_000,
            },
        );
        let fits = build_task_fits(&rows, &priors, 2);
        assert!(fits[0].prior.is_some());
        assert!(fits[1].prior.is_none());
    }

    // =========================================================================================
    // Study A: prior + learned blend
    // =========================================================================================

    fn blend_task(
        row: Vec<RungOutcome>,
        prior: Option<Vec<f64>>,
        price: Vec<f64>,
        ex_ante_r0: Option<f64>,
        ex_ante_counts: Option<(usize, usize)>,
    ) -> BlendTaskFit {
        BlendTaskFit {
            row,
            prior,
            price,
            p_hindsight: 0.5,
            ex_ante_r0,
            ex_ante_counts,
        }
    }

    // ---- blend formula ----------------------------------------------------------------------

    #[test]
    fn blend_at_zero_seen_equals_the_prior() {
        assert!((blended_r0(0.7, 0, 0) - 0.7).abs() < 1e-12);
        assert!((blended_r0(0.05, 0, 0) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn blend_at_large_seen_approaches_the_bucket_rate() {
        // Bucket rate 0.8 (800/1000), prior deliberately far away (0.1) — with seen_b >> s the
        // prior's pseudo-count weight is swamped by the observed traffic.
        let blended = blended_r0(0.1, 800, 1000);
        assert!(
            (blended - 0.8).abs() < 0.01,
            "blend {blended} should be within 1pp of the bucket rate 0.8"
        );
    }

    // ---- ex-ante arms never read the task's own realized cost --------------------------------

    /// Two tasks share everything the ex-ante decision is allowed to see (`ex_ante_r0`, `price`)
    /// but differ wildly in their own realized rung-0 `cost_usd`. If `serve_learned_exante` ever
    /// started reading the task's own cost instead of the ex-ante price, an outlier `cost_usd`
    /// like the second task's would flip the argmin and this test would fail.
    #[test]
    fn exante_decision_is_blind_to_the_scored_tasks_own_cost() {
        let cheap_row = two_rung_row(true, true, 0.001, true, true, 0.05);
        let mut expensive_row = cheap_row.clone();
        expensive_row[0].cost_usd = 50.0; // wildly different realized cost, same everything else

        let cheap = blend_task(cheap_row, None, vec![0.02, 0.05], Some(0.5), None);
        let expensive = blend_task(expensive_row, None, vec![0.02, 0.05], Some(0.5), None);

        let start_cheap = argmin_expected_cost_prior(&r0_only_pass_vector(0.5, 2), &cheap.price);
        let start_expensive =
            argmin_expected_cost_prior(&r0_only_pass_vector(0.5, 2), &expensive.price);
        assert_eq!(
            start_cheap, start_expensive,
            "the ex-ante decision must depend only on ex_ante_r0 and price, never on cost_usd"
        );

        // Same at the served-decision level: rungs paid may legitimately differ (gating still
        // runs on the real row), but the escalation decision (start rung) must not.
        let served_cheap = serve_learned_exante(&cheap);
        let served_expensive = serve_learned_exante(&expensive);
        assert_eq!(
            served_cheap.2, served_expensive.2,
            "same rungs paid from the same start"
        );
    }

    /// `prior+learned` must be equally blind: the blended r0 comes from `prior[0]` and the
    /// calibration-fold bucket counts, never from `t.row`.
    #[test]
    fn blend_decision_is_blind_to_the_scored_tasks_own_cost() {
        let prior = firstpass_core::cumulative_pass(&[0.5, 0.5]);
        let cheap_row = two_rung_row(true, true, 0.001, true, true, 0.05);
        let mut expensive_row = cheap_row.clone();
        expensive_row[0].cost_usd = 50.0;

        let cheap = blend_task(
            cheap_row,
            prior.clone(),
            vec![0.02, 0.05],
            Some(0.5),
            Some((40, 100)),
        );
        let expensive = blend_task(
            expensive_row,
            prior,
            vec![0.02, 0.05],
            Some(0.5),
            Some((40, 100)),
        );
        let served_cheap = serve_prior_plus_learned(&cheap);
        let served_expensive = serve_prior_plus_learned(&expensive);
        assert_eq!(
            served_cheap.2, served_expensive.2,
            "the blended decision must depend only on prior[0] and the ex-ante counts, never on \
             the scored task's own cost_usd"
        );
    }

    // ---- verdict fixtures: HELPS and NEUTRAL --------------------------------------------------

    /// A blend that correctly starts hard tasks at the top rung (because both the prior and the
    /// ex-ante bucket agree it is hopeless) and easy tasks cheap must report BLEND-HELPS: cheaper
    /// `$/success` than the prior alone, with no worse served-failure.
    #[test]
    fn blend_verdict_helps_when_the_blend_earns_its_keep() {
        let mut tasks = Vec::new();
        // 40 "hard" tasks: the prior alone is uninformative (0.5/0.5), but the ex-ante bucket
        // (fit on real traffic) knows this bucket almost never passes at rung 0 — so only the
        // blend, not the prior alone, correctly skips the wasted cheap attempt.
        for _ in 0..40 {
            tasks.push(blend_task(
                two_rung_row(false, false, 0.02, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
                Some(0.02),
                Some((2, 100)),
            ));
        }
        // 60 "easy" tasks: cheap rung passes; both prior and blend agree, no quality cost.
        for _ in 0..60 {
            tasks.push(blend_task(
                two_rung_row(true, true, 0.01, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
                Some(0.98),
                Some((98, 100)),
            ));
        }
        let series = eval_all_blend_arms(&tasks);
        let prior = &series[1];
        let blend = &series[3];
        let diff_ci = bootstrap_paired_ratio_diff_ci(
            &blend.cost,
            &blend.success,
            &prior.cost,
            &prior.success,
            BOOT_B,
            BOOT_SEED,
            ALPHA,
        );
        let cost_lower = diff_ci.hi < 0.0;
        let failure_ok =
            blend.result.served_failure_rate <= prior.result.served_failure_rate + 0.01;
        assert!(cost_lower, "blend must be cheaper: diff CI {diff_ci:?}");
        assert!(failure_ok);
        let verdict = if cost_lower && failure_ok {
            BlendVerdict::Helps
        } else {
            BlendVerdict::Neutral
        };
        assert_eq!(verdict, BlendVerdict::Helps);
    }

    /// A blend fed a bucket signal identical for hard and easy tasks (no traffic signal at all)
    /// on top of an already-uninformative prior cannot beat the prior's own $/success — reports
    /// BLEND-NEUTRAL.
    #[test]
    fn blend_verdict_neutral_when_the_blend_has_no_signal() {
        let mut tasks = Vec::new();
        for _ in 0..40 {
            tasks.push(blend_task(
                two_rung_row(false, false, 0.02, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
                Some(0.5),
                Some((50, 100)),
            ));
        }
        for _ in 0..60 {
            tasks.push(blend_task(
                two_rung_row(true, true, 0.01, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
                Some(0.5),
                Some((50, 100)),
            ));
        }
        let series = eval_all_blend_arms(&tasks);
        let prior = &series[1];
        let blend = &series[3];
        let diff_ci = bootstrap_paired_ratio_diff_ci(
            &blend.cost,
            &blend.success,
            &prior.cost,
            &prior.success,
            BOOT_B,
            BOOT_SEED,
            ALPHA,
        );
        let cost_lower = diff_ci.hi < 0.0;
        let verdict = if cost_lower {
            BlendVerdict::Helps
        } else {
            BlendVerdict::Neutral
        };
        assert_eq!(verdict, BlendVerdict::Neutral);
    }

    /// **Mutation test**: flip the sign convention (`prior - blend` instead of `blend - prior`)
    /// and confirm the HELPS fixture above would then read as NEUTRAL — proving the verdict test
    /// depends on the subtraction order, not on a tautology.
    #[test]
    fn blend_verdict_test_is_sensitive_to_the_diff_sign() {
        let mut tasks = Vec::new();
        for _ in 0..40 {
            tasks.push(blend_task(
                two_rung_row(false, false, 0.02, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
                Some(0.02),
                Some((2, 100)),
            ));
        }
        for _ in 0..60 {
            tasks.push(blend_task(
                two_rung_row(true, true, 0.01, true, true, 0.05),
                firstpass_core::cumulative_pass(&[0.5, 0.5]),
                vec![0.02, 0.05],
                Some(0.98),
                Some((98, 100)),
            ));
        }
        let series = eval_all_blend_arms(&tasks);
        let prior = &series[1];
        let blend = &series[3];
        // Mutated: prior - blend instead of blend - prior.
        let mutated_ci = bootstrap_paired_ratio_diff_ci(
            &prior.cost,
            &prior.success,
            &blend.cost,
            &blend.success,
            BOOT_B,
            BOOT_SEED,
            ALPHA,
        );
        assert!(
            mutated_ci.hi >= 0.0,
            "mutated sign convention must not still read as BLEND-HELPS"
        );
    }

    // ---- missing prior / missing ex-ante data fall back gracefully ---------------------------

    #[test]
    fn blend_falls_back_to_first_pass_when_no_prior_exists() {
        let t = blend_task(
            two_rung_row(false, false, 0.02, true, true, 0.05),
            None,
            vec![0.02, 0.05],
            Some(0.5),
            Some((10, 20)),
        );
        let served = serve_prior_plus_learned(&t);
        let fp = serve_first_pass_blend(&t);
        assert_eq!((served.0, served.1, served.2), (fp.0, fp.1, fp.2));
        assert!(!served.3, "must be counted as a fallback");
    }

    #[test]
    fn blend_degrades_to_the_prior_alone_when_no_exante_bucket_matched() {
        let prior = firstpass_core::cumulative_pass(&[0.02, 0.98]);
        let with_counts = blend_task(
            two_rung_row(false, false, 0.02, true, true, 0.05),
            prior.clone(),
            vec![0.02, 0.05],
            None,
            Some((0, 0)),
        );
        let without_counts = blend_task(
            two_rung_row(false, false, 0.02, true, true, 0.05),
            prior,
            vec![0.02, 0.05],
            None,
            None,
        );
        let a = serve_prior_plus_learned(&with_counts);
        let b = serve_prior_plus_learned(&without_counts);
        assert_eq!((a.0, a.1, a.2), (b.0, b.1, b.2));
    }

    #[test]
    fn exante_falls_back_to_first_pass_when_no_prompt_matched() {
        let t = blend_task(
            two_rung_row(false, false, 0.02, true, true, 0.05),
            None,
            vec![0.02, 0.05],
            None,
            None,
        );
        let served = serve_learned_exante(&t);
        assert!(!served.3);
    }

    // ---- prompt char map join ------------------------------------------------------------

    #[test]
    fn build_prompt_char_map_keys_by_task_id_and_skips_unmatched() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "jev_replay_blend_test_{}.jsonl",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "{\"task_id\": 1, \"text\": \"abc\", \"test_list\": [\"assert f(1)==1\"]}\n\
             {\"task_id\": 2, \"text\": \"abcdefghij\", \"test_list\": [\"assert f(2)==2\"]}\n",
        )
        .unwrap();
        let rows = vec![
            MatrixRow {
                task_id: "mbpp-1".to_owned(),
                ladder: vec!["a".to_owned(), "b".to_owned()],
                rungs: two_rung_row(true, true, 0.01, true, true, 0.05),
            },
            MatrixRow {
                task_id: "mbpp-2".to_owned(),
                ladder: vec!["a".to_owned(), "b".to_owned()],
                rungs: two_rung_row(true, true, 0.01, true, true, 0.05),
            },
            MatrixRow {
                task_id: "mbpp-999".to_owned(), // no matching prompt
                ladder: vec!["a".to_owned(), "b".to_owned()],
                rungs: two_rung_row(true, true, 0.01, true, true, 0.05),
            },
        ];
        let map = build_prompt_char_map(path.to_str().unwrap(), &rows).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(map.get("mbpp-1"), Some(&3.0));
        assert_eq!(map.get("mbpp-2"), Some(&10.0));
        assert_eq!(map.get("mbpp-999"), None);
    }
}
