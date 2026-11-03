//! Certified model selection (ADR 0014) replayed on recorded coding-agent sessions, at $0.
//!
//! # What this answers
//!
//! The audited finding (#239) is that on real agent runs, workload-level model selection beats a
//! per-request cascade. ADR 0014 turns selection into a sequential non-inferiority experiment.
//! This module replays that experiment on sessions where **every candidate model's outcome on the
//! same task is recorded**, using the production code end to end:
//!
//! - [`firstpass_core::Experiment`] — the betting e-processes, pairing and stopping rule;
//! - [`firstpass_core::Challenger::arm`] — the salted-hash session-to-arm assignment of live mode;
//! - [`recommend`] — the adopt rule of `firstpass experiment` (mirrored, see its doc).
//!
//! Nothing statistical is reimplemented here: this file only decides which recorded outcome a
//! session would have been served, and in which order sessions arrive.
//!
//! # Two modes, both replayed
//!
//! - **paired** — both arms' outcomes on each session feed `observe_pair`. On recorded whole-run
//!   data each model ran its own full agent session on the task, so this pairs *task outcomes*.
//!   The serving path's paired mode re-runs one request, not a whole agent session, and is
//!   restricted to single-turn routes (ADR 0014 stage 2). So this mode answers "what would paired
//!   task-outcome evidence buy", which a live run can only get by running both agents per task.
//! - **live** — what the serving path supports for agents today: sessions are split by
//!   `Challenger::arm`, each arm reports its own outcome through `observe_outcome`, and the k-th
//!   incumbent outcome is FIFO-paired with the k-th challenger outcome.
//!
//! # The selection policy (pre-registered)
//!
//! The incumbent is the model with the highest recorded success rate (ties: cheaper). Every other
//! model is a candidate, tried **one at a time, cheapest first**, because the serving path runs one
//! challenger per route. Each experiment runs at `alpha / k` (`k` candidates, Bonferroni) so the
//! chance of adopting *any* challenger that is worse by more than the margin stays at most `alpha`.
//! On `Halted` or "certified but not cheaper", the operator moves on to the next candidate (a
//! fresh experiment); on "certified and cheaper" the challenger is adopted for every later session.
//! While an experiment runs in paired mode the incumbent is served and the challenger's run is
//! paid as shadow cost. In live mode each session is served by its arm.
//!
//! Every replay draws a fresh random session order (and, in live mode, a fresh assignment salt).
//! A policy's per-session success and cost are averaged over replays, then seeded bootstrap CIs
//! are taken over sessions ([`crate::stats`]). Order randomness is reported separately as the
//! distribution of final decisions.
//!
//! # Input: what a recorded (or future paid) run must contain
//!
//! One JSON object per line, one line per session (task), every session carrying a run for every
//! model:
//!
//! ```json
//! {"id":"django__django-11099","runs":{
//!   "anthropic/claude-opus-4-5":{"success":true,"cost_usd":0.71,"gate_signal":41},
//!   "openrouter/minimax-m2.5":{"success":true,"cost_usd":0.06,"gate_signal":55}}}
//! ```
//!
//! - `success` — the **task outcome** of that model's own whole session (hidden tests pass), never
//!   a gate verdict: certifying on gate verdicts repeats ADR 0014 finding 1.
//! - `cost_usd` — the whole session's spend for that model.
//! - `gate_signal` (optional) — what a live router could see on the finished run, lower = more
//!   confident (agent step count, or 0/1 for a gate pass/fail). Only the cascade baseline uses it.
//!
//! Two recorded sources are adapted to this shape: the public SWE-bench `per_instance_details.json`
//! files ([`from_swebench_experiments`]) and Firstpass's own multi-turn checkpoints
//! ([`from_checkpoint`], first turn only — later turns share one transcript, so only turn 1 is a
//! clean pair).

use std::collections::BTreeMap;

use firstpass_core::{Arm, Challenger, Experiment, ExperimentReport, ExperimentStatus};
use serde::{Deserialize, Serialize};

use crate::sim::Rng;
use crate::stats::{
    Ci, bootstrap_mean_ci, bootstrap_paired_ratio_diff_ci, bootstrap_ratio_ci, mean,
};

/// One model's recorded run of one session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRun {
    /// Task outcome of the whole session.
    pub success: bool,
    /// Whole-session spend in USD.
    pub cost_usd: f64,
    /// Live-observable confidence signal on the finished run; lower = more confident.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_signal: Option<f64>,
}

/// One task, with every candidate model's run of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    /// Stable task id.
    pub id: String,
    /// Model id → that model's run.
    pub runs: BTreeMap<String, ModelRun>,
}

/// Parse the JSONL format in the module doc.
///
/// # Errors
/// On a malformed line, or see [`check`].
pub fn parse_jsonl(text: &str) -> Result<Vec<Session>, String> {
    let sessions = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect::<Result<Vec<Session>, _>>()?;
    check(sessions)
}

/// Adapt public SWE-bench `per_instance_details.json` files, one `(model, file text)` per system.
/// Keeps only instances every system ran. `gate_signal` is the agent's step count (`api_calls`).
///
/// # Errors
/// On malformed JSON or a missing field.
pub fn from_swebench_experiments(files: &[(String, String)]) -> Result<Vec<Session>, String> {
    #[derive(Deserialize)]
    struct Row {
        resolved: bool,
        cost: f64,
        api_calls: f64,
    }
    let mut by_id: BTreeMap<String, BTreeMap<String, ModelRun>> = BTreeMap::new();
    for (model, text) in files {
        let rows: BTreeMap<String, Row> =
            serde_json::from_str(text).map_err(|e| format!("{model}: {e}"))?;
        for (id, r) in rows {
            by_id.entry(id).or_default().insert(
                model.clone(),
                ModelRun {
                    success: r.resolved,
                    cost_usd: r.cost,
                    gate_signal: Some(r.api_calls),
                },
            );
        }
    }
    check(
        by_id
            .into_iter()
            .filter(|(_, runs)| runs.len() == files.len())
            .map(|(id, runs)| Session { id, runs })
            .collect(),
    )
}

/// Adapt a Firstpass multi-turn checkpoint ([`crate::agentic::RecordedTask`] lines), **turn 1
/// only**: every rung saw the same prompt there. `success` is the hidden-test oracle, `gate_signal`
/// is 0 for a visible-gate pass and 1 for a fail. Duplicate ids keep the first record.
///
/// # Errors
/// On a malformed line, or see [`check`].
pub fn from_checkpoint(text: &str) -> Result<Vec<Session>, String> {
    let mut out: BTreeMap<String, Session> = BTreeMap::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let task: crate::agentic::RecordedTask =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        let Some(first) = task.turns.first() else {
            continue;
        };
        let runs = first
            .rungs
            .iter()
            .map(|r| {
                let run = ModelRun {
                    success: r.oracle_correct,
                    cost_usd: r.cost_usd,
                    gate_signal: Some(if r.gate_pass { 0.0 } else { 1.0 }),
                };
                (r.model.clone(), run)
            })
            .collect();
        out.entry(task.id.clone())
            .or_insert(Session { id: task.id, runs });
    }
    check(out.into_values().collect())
}

/// Sort by id and require a complete pairing: every session has a run for the same ≥ 2 models,
/// with finite nonnegative costs. A partial session would silently unpair the comparison.
///
/// # Errors
/// If any of the above fails.
pub fn check(mut sessions: Vec<Session>) -> Result<Vec<Session>, String> {
    sessions.sort_by(|a, b| a.id.cmp(&b.id));
    let first = sessions.first().ok_or("no sessions")?;
    let models: Vec<&String> = first.runs.keys().collect();
    if models.len() < 2 {
        return Err(format!(
            "only {} model per session: a paired comparison needs at least 2",
            models.len()
        ));
    }
    for s in &sessions {
        if s.runs.keys().collect::<Vec<_>>() != models {
            return Err(format!(
                "session {} does not have runs for exactly {models:?}",
                s.id
            ));
        }
        if let Some((m, _)) = s
            .runs
            .iter()
            .find(|(_, r)| !r.cost_usd.is_finite() || r.cost_usd < 0.0)
        {
            return Err(format!(
                "session {} model {m}: cost must be finite and >= 0",
                s.id
            ));
        }
    }
    Ok(sessions)
}

/// The `firstpass experiment` adopt rule: certified **and** cheaper per request.
///
/// ponytail: mirrors `firstpass_proxy::experiment::recommend` (5 lines) rather than pulling the
/// whole proxy (axum, tokio) into the bench; move it into `firstpass-core` if a third caller appears.
#[must_use]
pub fn recommend(r: &ExperimentReport) -> Decision {
    match r.status {
        ExperimentStatus::Undecided => Decision::KeepRunning,
        ExperimentStatus::Halted => Decision::Halted,
        ExperimentStatus::Certified => match (
            r.challenger.cost_per_request(),
            r.incumbent.cost_per_request(),
        ) {
            (Some(c), Some(i)) if c < i => Decision::Adopt,
            _ => Decision::CertifiedNoSaving,
        },
    }
}

/// What an experiment's state tells the operator to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Certified and cheaper.
    Adopt,
    /// Certified, not cheaper.
    CertifiedNoSaving,
    /// Proven worse by more than the margin.
    Halted,
    /// Not enough evidence.
    KeepRunning,
}

/// How evidence is collected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Both arms' outcomes per session (`observe_pair`).
    Paired,
    /// Sessions split by `Challenger::arm` at `percent` (`observe_outcome`).
    Live {
        /// Percent of sessions on the challenger arm.
        percent: f64,
    },
}

/// One experiment within one replay.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Trial {
    /// Challenger model.
    pub challenger: String,
    /// Final decision (`KeepRunning` if the sessions ran out first).
    pub decision: Decision,
    /// Pairs the test consumed when it decided (or at the end).
    pub pairs: u64,
}

/// One replayed order.
#[derive(Debug, Clone)]
struct Replay {
    /// Per session (indexed like the input): served success and cost incl. shadow spend.
    success: Vec<f64>,
    cost: Vec<f64>,
    trials: Vec<Trial>,
    adopted: Option<String>,
}

/// Run the sequential selection policy over `order`. `candidates` are tried in the given order.
#[expect(
    clippy::too_many_arguments,
    reason = "private helper; each argument is one knob of a single replay"
)]
fn replay_once(
    sessions: &[Session],
    incumbent: &str,
    candidates: &[&str],
    mode: Mode,
    margin: f64,
    alpha: f64,
    order: &[usize],
    salt: &str,
) -> Result<Replay, String> {
    let n = sessions.len();
    let mut r = Replay {
        success: vec![0.0; n],
        cost: vec![0.0; n],
        trials: Vec::new(),
        adopted: None,
    };
    let mut next = candidates.iter();
    let mut current: Option<(&str, Experiment, Challenger)> = None;
    let start = |c: &str| -> Result<(Experiment, Challenger), String> {
        let percent = match mode {
            Mode::Paired => 0.0,
            Mode::Live { percent } => percent,
        };
        let mode_tag = if matches!(mode, Mode::Paired) {
            "paired"
        } else {
            "live"
        };
        let cfg: Challenger = serde_json::from_value(serde_json::json!({
            "ladder": [c], "percent": percent, "margin": margin, "alpha": alpha,
            "mode": mode_tag, "max_usd_per_day": 1.0,
        }))
        .map_err(|e| e.to_string())?;
        cfg.validate()?;
        Ok((Experiment::new(margin, alpha)?, cfg))
    };
    if let Some(c) = next.next() {
        let (e, cfg) = start(c)?;
        current = Some((c, e, cfg));
    }
    for &i in order {
        let s = &sessions[i];
        let run = |m: &str| -> Result<&ModelRun, String> {
            s.runs
                .get(m)
                .ok_or_else(|| format!("session {} lacks model {m}", s.id))
        };
        let x = |m: &ModelRun| if m.success { 1.0 } else { 0.0 };
        let inc = run(incumbent)?;
        let Some((ch_name, exp, cfg)) = current.as_mut() else {
            let served = run(r.adopted.as_deref().unwrap_or(incumbent))?;
            r.success[i] = x(served);
            r.cost[i] = served.cost_usd;
            continue;
        };
        let ch = run(ch_name)?;
        match mode {
            Mode::Paired => {
                exp.observe_pair(x(inc), x(ch));
                exp.observe_cost(Arm::Incumbent, inc.cost_usd);
                exp.observe_cost(Arm::Challenger, ch.cost_usd);
                r.success[i] = x(inc);
                r.cost[i] = inc.cost_usd + ch.cost_usd;
            }
            Mode::Live { .. } => {
                let arm = cfg.arm(salt, &s.id);
                let served = if arm == Arm::Challenger { ch } else { inc };
                exp.observe_outcome(arm, x(served));
                exp.observe_cost(arm, served.cost_usd);
                r.success[i] = x(served);
                r.cost[i] = served.cost_usd;
            }
        }
        let rep = exp.report();
        let decision = recommend(&rep);
        if decision != Decision::KeepRunning {
            r.trials.push(Trial {
                challenger: (*ch_name).to_owned(),
                decision,
                pairs: rep.pairs,
            });
            if decision == Decision::Adopt {
                r.adopted = Some((*ch_name).to_owned());
                current = None;
            } else if let Some(c) = next.next() {
                let (e, cfg) = start(c)?;
                current = Some((c, e, cfg));
            } else {
                current = None;
            }
        }
    }
    if let Some((c, exp, _)) = current {
        r.trials.push(Trial {
            challenger: c.to_owned(),
            decision: Decision::KeepRunning,
            pairs: exp.report().pairs,
        });
    }
    Ok(r)
}

/// A seeded random permutation of `0..n` (Fisher–Yates).
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        v.swap(i, rng.below(i + 1));
    }
    v
}

/// The criterion, fixed before any number exists.
#[derive(Debug, Clone, Serialize)]
pub struct PreRegistered {
    /// Required relative `$/success` reduction vs always-top, with the paired CI excluding zero.
    pub min_cost_reduction: f64,
    /// Non-inferiority: the lower CI bound of `success(policy) − success(top)` must be ≥ −margin.
    pub margin: f64,
    /// Error rate of the whole selection (split over candidates) and level of every CI.
    pub alpha: f64,
    /// Random orders replayed.
    pub replays: usize,
    /// Bootstrap resamples.
    pub bootstrap: usize,
    /// Seed for orders, salts and bootstrap.
    pub seed: u64,
}

impl Default for PreRegistered {
    fn default() -> Self {
        Self {
            min_cost_reduction: 0.10,
            margin: firstpass_core::select::DEFAULT_MARGIN,
            alpha: firstpass_core::select::DEFAULT_ALPHA,
            replays: 200,
            bootstrap: 2000,
            seed: 20_261_103,
        }
    }
}

/// One policy's per-session results, summarized.
#[derive(Debug, Clone, Serialize)]
pub struct PolicyRow {
    /// Policy name.
    pub name: String,
    /// Success rate.
    pub success: Ci,
    /// Mean USD per session (incl. shadow spend where paid).
    pub cost_per_session: f64,
    /// USD per successful session.
    pub cost_per_success: Ci,
    /// `success − success(always-top)`, paired per session.
    pub success_vs_top: Ci,
    /// `$/success − $/success(always-top)`, paired per session.
    pub cost_per_success_vs_top: Ci,
}

/// Distribution of what the selection ended on, across replays.
#[derive(Debug, Clone, Serialize)]
pub struct SelectionSummary {
    /// Mode replayed.
    pub mode: Mode,
    /// Final adopted model (or `"<incumbent>"`) → number of replays.
    pub adopted: BTreeMap<String, usize>,
    /// Median pairs consumed (summed over trials) before adopting, over adopting replays.
    pub median_pairs_to_adopt: Option<u64>,
    /// Per challenger: decision → count of replays that reached it.
    pub trials: BTreeMap<String, BTreeMap<String, usize>>,
    /// Kill criterion on this mode.
    pub verdict: Verdict,
    /// The policy row.
    pub policy: PolicyRow,
}

/// PROCEED / STOP with the reason.
#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    /// Both bars met.
    pub proceed: bool,
    /// Why.
    pub rationale: String,
}

/// The whole replay result.
#[derive(Debug, Clone, Serialize)]
pub struct ReplayReport {
    /// Always "replay": these are recorded outcomes, never a live run.
    pub kind: &'static str,
    /// Where the outcomes came from.
    pub source: String,
    /// Sessions.
    pub n: usize,
    /// The criterion.
    pub preregistered: PreRegistered,
    /// Incumbent (always-top).
    pub incumbent: String,
    /// Candidates in try order, with alpha per experiment.
    pub candidates: Vec<String>,
    /// Alpha per experiment (`alpha / k`).
    pub alpha_each: f64,
    /// Baselines: always-<model> and cross-fit cascades.
    pub baselines: Vec<PolicyRow>,
    /// The selection policy, per mode.
    pub selection: Vec<SelectionSummary>,
    /// Diagnostic: each candidate as the only challenger, at the full `alpha`. Not
    /// multiplicity-corrected — reading the best row as the result is post-hoc selection.
    pub single: Vec<SingleChallenger>,
}

/// One candidate alone against the incumbent, over the replayed orders.
#[derive(Debug, Clone, Serialize)]
pub struct SingleChallenger {
    /// Challenger model.
    pub challenger: String,
    /// Mode replayed.
    pub mode: Mode,
    /// Decision → replays.
    pub decisions: BTreeMap<String, usize>,
    /// Median pairs at the decision, over replays that decided.
    pub median_pairs_to_decide: Option<u64>,
}

fn row(
    name: String,
    success: &[f64],
    cost: &[f64],
    top: (&[f64], &[f64]),
    p: &PreRegistered,
) -> PolicyRow {
    let diff: Vec<f64> = success.iter().zip(top.0).map(|(a, b)| a - b).collect();
    PolicyRow {
        name,
        success: bootstrap_mean_ci(success, p.bootstrap, p.seed, p.alpha),
        cost_per_session: mean(cost),
        cost_per_success: bootstrap_ratio_ci(cost, success, p.bootstrap, p.seed, p.alpha),
        success_vs_top: bootstrap_mean_ci(&diff, p.bootstrap, p.seed, p.alpha),
        cost_per_success_vs_top: bootstrap_paired_ratio_diff_ci(
            cost,
            success,
            top.1,
            top.0,
            p.bootstrap,
            p.seed,
            p.alpha,
        ),
    }
}

/// The kill criterion applied to a policy row against always-top. `adopt_share` is the fraction of
/// replays that ended on a certified challenger: a saving that comes only from serving an
/// unproven challenger *during* the experiment is transient, so it cannot carry the verdict.
#[must_use]
pub fn verdict(
    policy: &PolicyRow,
    top: &PolicyRow,
    adopt_share: f64,
    p: &PreRegistered,
) -> Verdict {
    let reduction = if top.cost_per_success.point > 0.0 {
        1.0 - policy.cost_per_success.point / top.cost_per_success.point
    } else {
        0.0
    };
    let cheaper = reduction >= p.min_cost_reduction && policy.cost_per_success_vs_top.hi < 0.0;
    let non_inferior = policy.success_vs_top.lo >= -p.margin;
    let adopted = adopt_share > 0.5;
    let proceed = cheaper && non_inferior && adopted;
    let rationale = format!(
        "certified adoption in {:.0}% of orders (bar > 50%): {}; $/success {:+.1}% vs always-top (bar: ≤ −{:.0}% with CI hi < 0; CI [{:+.4}, {:+.4}]): {}; \
         success − top CI lo {:+.3} (bar ≥ −{:.2}): {}",
        adopt_share * 100.0,
        if adopted { "met" } else { "MISSED" },
        -reduction * 100.0,
        p.min_cost_reduction * 100.0,
        policy.cost_per_success_vs_top.lo,
        policy.cost_per_success_vs_top.hi,
        if cheaper { "met" } else { "MISSED" },
        policy.success_vs_top.lo,
        p.margin,
        if non_inferior { "met" } else { "MISSED" },
    );
    Verdict { proceed, rationale }
}

/// Two-rung cascade `cheap → top` on whole runs: accept the cheap run iff `gate_signal ≤ tau`,
/// else rerun with `top`. `tau` is fitted on one half (alternate sessions by sorted id) and scored
/// on the other, so no session is scored by a threshold it helped fit. Objective, fixed in advance:
/// the cheapest `tau` whose training success is ≥ top's; if none, the most successful.
fn cascade(sessions: &[Session], cheap: &str, top: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    let get = |s: &Session, m: &str| s.runs.get(m).cloned();
    let rows: Vec<(f64, f64, f64, f64, f64)> = sessions
        .iter()
        .map(|s| {
            let (c, t) = (get(s, cheap)?, get(s, top)?);
            let x = |r: &ModelRun| if r.success { 1.0 } else { 0.0 };
            Some((c.gate_signal?, x(&c), c.cost_usd, x(&t), t.cost_usd))
        })
        .collect::<Option<_>>()?;
    let serve = |r: &(f64, f64, f64, f64, f64), tau: f64| {
        if r.0 <= tau {
            (r.1, r.2)
        } else {
            (r.3, r.2 + r.4)
        }
    };
    let (mut success, mut cost) = (vec![0.0; rows.len()], vec![0.0; rows.len()]);
    for fold in 0..2 {
        let train: Vec<_> = rows
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 != fold)
            .collect();
        let target: f64 = train.iter().map(|(_, r)| r.3).sum();
        let mut taus: Vec<f64> = train.iter().map(|(_, r)| r.0).collect();
        taus.push(f64::NEG_INFINITY);
        let score = |tau: f64| {
            train.iter().fold((0.0, 0.0), |(s, c), (_, r)| {
                let (a, b) = serve(r, tau);
                (s + a, c + b)
            })
        };
        let best = taus
            .iter()
            .map(|&t| (t, score(t)))
            .min_by(|(_, (sa, ca)), (_, (sb, cb))| {
                // Reaching the target beats not reaching it; then cheaper, then more successful.
                let key = |s: f64, c: f64| (s < target, if s >= target { c } else { -s });
                key(*sa, *ca)
                    .partial_cmp(&key(*sb, *cb))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(f64::NEG_INFINITY, |(t, _)| t);
        for (i, r) in rows.iter().enumerate().filter(|(i, _)| i % 2 == fold) {
            (success[i], cost[i]) = serve(r, best);
        }
    }
    Some((success, cost))
}

/// Run every baseline and the selection policy in both modes.
///
/// # Errors
/// If a challenger config is refused by the production validator, or the data is unpaired.
pub fn run(
    sessions: &[Session],
    source: &str,
    incumbent: Option<&str>,
    live_percent: f64,
    p: &PreRegistered,
) -> Result<ReplayReport, String> {
    let sessions = check(sessions.to_vec())?;
    let models: Vec<String> = sessions[0].runs.keys().cloned().collect();
    let col = |m: &str| -> (Vec<f64>, Vec<f64>) {
        sessions
            .iter()
            .map(|s| {
                let r = &s.runs[m];
                (if r.success { 1.0 } else { 0.0 }, r.cost_usd)
            })
            .unzip()
    };
    let rate = |m: &String| {
        let (s, c) = col(m);
        (mean(&s), mean(&c))
    };
    let incumbent = match incumbent {
        Some(m) if models.iter().any(|x| x == m) => m.to_owned(),
        Some(m) => return Err(format!("incumbent {m} not in {models:?}")),
        None => models
            .iter()
            .max_by(|a, b| {
                let ((sa, ca), (sb, cb)) = (rate(a), rate(b));
                (sa, -ca)
                    .partial_cmp(&(sb, -cb))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or("no models")?,
    };
    let mut candidates: Vec<String> = models
        .iter()
        .filter(|m| **m != incumbent)
        .cloned()
        .collect();
    candidates.sort_by(|a, b| {
        rate(a)
            .1
            .partial_cmp(&rate(b).1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    #[expect(clippy::cast_precision_loss, reason = "a handful of candidates")]
    let alpha_each = p.alpha / candidates.len() as f64;

    let (top_s, top_c) = col(&incumbent);
    let top = (top_s.as_slice(), top_c.as_slice());
    let mut baselines = Vec::new();
    for m in std::iter::once(&incumbent).chain(&candidates) {
        let (s, c) = col(m);
        baselines.push(row(format!("always {m}"), &s, &c, top, p));
    }
    for m in &candidates {
        if let Some((s, c)) = cascade(&sessions, m, &incumbent) {
            baselines.push(row(
                format!("cascade {m} → {incumbent} (cross-fit gate)"),
                &s,
                &c,
                top,
                p,
            ));
        }
    }

    let cand: Vec<&str> = candidates.iter().map(String::as_str).collect();
    let modes = [
        Mode::Paired,
        Mode::Live {
            percent: live_percent,
        },
    ];
    let mut selection = Vec::new();
    let mut single = Vec::new();
    for mode in modes {
        let m = replay_many(&sessions, &incumbent, &cand, mode, alpha_each, p)?;
        let name = match mode {
            Mode::Paired => "selection, paired (both agents run per task)".to_owned(),
            Mode::Live { percent } => format!("selection, live A/B ({percent}% challenger)"),
        };
        let policy = row(name, &m.success, &m.cost, top, p);
        #[expect(clippy::cast_precision_loss, reason = "replay counts are small")]
        let adopt_share = m.adopted_pairs.len() as f64 / p.replays as f64;
        selection.push(SelectionSummary {
            mode,
            verdict: verdict(&policy, &baselines[0], adopt_share, p),
            policy,
            adopted: m.adopted,
            median_pairs_to_adopt: median(m.adopted_pairs),
            trials: m.trials,
        });
        for c in &cand {
            let one = replay_many(&sessions, &incumbent, &[c], mode, p.alpha, p)?;
            single.push(SingleChallenger {
                challenger: (*c).to_owned(),
                mode,
                decisions: one.trials.into_values().next().unwrap_or_default(),
                median_pairs_to_decide: median(one.decided_pairs),
            });
        }
    }
    Ok(ReplayReport {
        kind: "replay",
        source: source.to_owned(),
        n: sessions.len(),
        preregistered: p.clone(),
        incumbent,
        candidates,
        alpha_each,
        baselines,
        selection,
        single,
    })
}

/// Policy outcomes accumulated over replays.
struct Many {
    success: Vec<f64>,
    cost: Vec<f64>,
    adopted: BTreeMap<String, usize>,
    /// Per adopting replay: pairs summed over its trials.
    adopted_pairs: Vec<u64>,
    /// Per decided trial: pairs at the decision.
    decided_pairs: Vec<u64>,
    trials: BTreeMap<String, BTreeMap<String, usize>>,
}

fn replay_many(
    sessions: &[Session],
    incumbent: &str,
    cand: &[&str],
    mode: Mode,
    alpha: f64,
    p: &PreRegistered,
) -> Result<Many, String> {
    let n = sessions.len();
    let mut m = Many {
        success: vec![0.0; n],
        cost: vec![0.0; n],
        adopted: BTreeMap::new(),
        adopted_pairs: Vec::new(),
        decided_pairs: Vec::new(),
        trials: BTreeMap::new(),
    };
    for rep in 0..p.replays {
        let seed = p.seed ^ (rep as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let order = permutation(n, seed);
        let salt = format!("paired-replay:{seed}");
        let r = replay_once(
            sessions, incumbent, cand, mode, p.margin, alpha, &order, &salt,
        )?;
        for i in 0..n {
            m.success[i] += r.success[i];
            m.cost[i] += r.cost[i];
        }
        *m.adopted
            .entry(r.adopted.clone().unwrap_or_else(|| "<incumbent>".into()))
            .or_default() += 1;
        if r.adopted.is_some() {
            m.adopted_pairs.push(r.trials.iter().map(|t| t.pairs).sum());
        }
        for t in r.trials {
            if t.decision != Decision::KeepRunning {
                m.decided_pairs.push(t.pairs);
            }
            let d = serde_json::to_value(t.decision).map_err(|e| e.to_string())?;
            *m.trials
                .entry(t.challenger)
                .or_default()
                .entry(d.as_str().unwrap_or("?").to_owned())
                .or_default() += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "replay counts are small")]
    let k = p.replays as f64;
    m.success.iter_mut().for_each(|v| *v /= k);
    m.cost.iter_mut().for_each(|v| *v /= k);
    Ok(m)
}

fn median(mut v: Vec<u64>) -> Option<u64> {
    v.sort_unstable();
    v.get(v.len() / 2).copied()
}

impl ReplayReport {
    /// Markdown rendering.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        use std::fmt::Write as _;
        let mut o = String::new();
        let p = &self.preregistered;
        let _ = writeln!(
            o,
            "# Certified selection (ADR 0014) — REPLAY of recorded outcomes, NOT a live run\n"
        );
        let _ = writeln!(
            o,
            "data source: {}  ·  sessions: {}  ·  spend: $0\n",
            self.source, self.n
        );
        let _ = writeln!(
            o,
            "## Pre-registered\n\n- PROCEED iff a challenger is certified-and-adopted in > 50% of orders, AND selection's $/success is ≥ {:.0}% below always-top with the paired \
             {:.0}% CI of the difference below 0, AND the CI lower bound of success − always-top ≥ −{:.2}.\n\
             - incumbent (always-top): {} (highest recorded success; ties → cheaper)\n\
             - candidates, cheapest first, one experiment at a time at alpha/k = {:.4}: {}\n\
             - {} random orders, {} bootstrap resamples, seed {}\n",
            p.min_cost_reduction * 100.0,
            (1.0 - p.alpha) * 100.0,
            p.margin,
            self.incumbent,
            self.alpha_each,
            self.candidates.join(", "),
            p.replays,
            p.bootstrap,
            p.seed,
        );
        let _ = writeln!(
            o,
            "## Policies\n\n| policy | success [CI] | $/session | $/success [CI] | Δsuccess vs top [CI] | Δ$/success vs top [CI] |\n|---|---|---|---|---|---|"
        );
        let ci = |c: &Ci, d: usize| format!("{:.d$} [{:.d$}, {:.d$}]", c.point, c.lo, c.hi);
        for r in self
            .baselines
            .iter()
            .chain(self.selection.iter().map(|s| &s.policy))
        {
            let _ = writeln!(
                o,
                "| {} | {} | {:.4} | {} | {} | {} |",
                r.name,
                ci(&r.success, 3),
                r.cost_per_session,
                ci(&r.cost_per_success, 4),
                ci(&r.success_vs_top, 3),
                ci(&r.cost_per_success_vs_top, 4),
            );
        }
        for s in &self.selection {
            let _ = writeln!(o, "\n## {}\n", s.policy.name);
            let _ = writeln!(
                o,
                "- final model over {} orders: {:?}",
                p.replays, s.adopted
            );
            if let Some(m) = s.median_pairs_to_adopt {
                let _ = writeln!(o, "- median pairs consumed before adopting: {m}");
            }
            for (c, d) in &s.trials {
                let _ = writeln!(o, "- {c}: {d:?}");
            }
            let _ = writeln!(
                o,
                "- **{}** — {}",
                if s.verdict.proceed { "PROCEED" } else { "STOP" },
                s.verdict.rationale
            );
        }
        let _ = writeln!(
            o,
            "\n## Diagnostic: each candidate alone at alpha = {} (NOT multiplicity-corrected)\n\n\
             | challenger | mode | decisions over {} orders | median pairs to decide |\n|---|---|---|---|",
            p.alpha, p.replays
        );
        for d in &self.single {
            let mode = match d.mode {
                Mode::Paired => "paired".to_owned(),
                Mode::Live { percent } => format!("live {percent}%"),
            };
            let med = d
                .median_pairs_to_decide
                .map_or("—".to_owned(), |v| v.to_string());
            let _ = writeln!(
                o,
                "| {} | {mode} | {:?} | {med} |",
                d.challenger, d.decisions
            );
        }
        let _ = writeln!(
            o,
            "\nCIs are over sessions with each session's served outcome averaged over the replayed orders; \
             order-to-order variation is the decision distribution above. Recorded outcomes are one run per \
             model per task, so a model's run-to-run variance is not in these intervals."
        );
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic sessions: `top` succeeds at `p_top`, `good` is as good and 10× cheaper, `bad` is
    /// 20 points worse and cheaper still. Deterministic.
    fn synthetic(n: usize) -> Vec<Session> {
        let mut rng = Rng::new(7);
        (0..n)
            .map(|i| {
                let u = rng.unit();
                let mk = |success: bool, cost_usd: f64, sig: f64| ModelRun {
                    success,
                    cost_usd,
                    gate_signal: Some(sig),
                };
                let mut runs = BTreeMap::new();
                runs.insert("top".into(), mk(u < 0.8, 1.0, 0.0));
                runs.insert(
                    "good".into(),
                    mk(
                        u < 0.8 || rng.unit() < 0.02,
                        0.1,
                        f64::from(u32::from(u >= 0.8)),
                    ),
                );
                runs.insert("bad".into(), mk(u < 0.6, 0.05, 1.0));
                Session {
                    id: format!("t{i:04}"),
                    runs,
                }
            })
            .collect()
    }

    fn fast() -> PreRegistered {
        PreRegistered {
            replays: 20,
            bootstrap: 300,
            ..PreRegistered::default()
        }
    }

    #[test]
    fn paired_selection_halts_the_bad_challenger_then_adopts_the_good_one() {
        let s = synthetic(1500);
        let r = run(&s, "synthetic", None, 50.0, &fast()).unwrap();
        assert_eq!(
            r.incumbent, "good",
            "good edges top on success (ties broken by cost)"
        );
        // Make `top` the incumbent explicitly: the interesting case.
        let r2 = run(&s, "synthetic", Some("top"), 50.0, &fast()).unwrap();
        assert_eq!(r2.candidates, vec!["bad", "good"], "cheapest first");
        let paired = &r2.selection[0];
        assert_eq!(
            paired.trials["bad"].get("halted"),
            Some(&20),
            "{:?}",
            paired.trials
        );
        assert_eq!(
            paired.adopted.get("good"),
            Some(&20),
            "{:?}",
            paired.adopted
        );
        assert!(paired.verdict.proceed, "{}", paired.verdict.rationale);
    }

    #[test]
    fn a_worse_only_field_never_adopts_and_the_verdict_stops() {
        let s: Vec<Session> = synthetic(800)
            .into_iter()
            .map(|mut s| {
                s.runs.remove("good");
                s
            })
            .collect();
        let r = run(&s, "synthetic", Some("top"), 50.0, &fast()).unwrap();
        for sel in &r.selection {
            assert_eq!(
                sel.adopted.get("<incumbent>"),
                Some(&20),
                "{:?}",
                sel.adopted
            );
            assert!(!sel.verdict.proceed);
        }
    }

    #[test]
    fn paired_charges_shadow_cost_until_decided() {
        let s = synthetic(300);
        let order: Vec<usize> = (0..s.len()).collect();
        let r = replay_once(&s, "top", &["good"], Mode::Paired, 0.05, 0.05, &order, "x").unwrap();
        // Before any decision both runs are paid.
        assert!((r.cost[0] - 1.1).abs() < 1e-12);
    }

    #[test]
    fn live_mode_uses_production_arm_assignment() {
        let s = synthetic(400);
        let order: Vec<usize> = (0..s.len()).collect();
        let r = replay_once(
            &s,
            "top",
            &["good"],
            Mode::Live { percent: 50.0 },
            0.05,
            0.05,
            &order,
            "salt",
        )
        .unwrap();
        let cfg: Challenger =
            serde_json::from_value(serde_json::json!({"ladder":["good"],"percent":50.0})).unwrap();
        // Undecided at the start, so session 0 is served by its hashed arm.
        let want = if cfg.arm("salt", &s[0].id) == Arm::Challenger {
            0.1
        } else {
            1.0
        };
        assert!((r.cost[0] - want).abs() < 1e-12);
    }

    #[test]
    fn cascade_gate_is_cross_fit_and_reaches_top() {
        let s = synthetic(1000);
        let (succ, cost) = cascade(&s, "good", "top").unwrap();
        // good's signal is 0 exactly where it is right by construction, so the fitted gate escalates
        // only on signal 1: success ≥ top, cost well below top + good.
        assert!(
            mean(&succ)
                >= mean(
                    &s.iter()
                        .map(|x| f64::from(u8::from(x.runs["top"].success)))
                        .collect::<Vec<_>>()
                )
        );
        assert!(mean(&cost) < 0.5);
    }

    #[test]
    fn unpaired_or_single_model_data_is_refused() {
        let mut s = synthetic(10);
        s[3].runs.remove("bad");
        assert!(check(s).unwrap_err().contains("does not have runs"));
        let one: Vec<Session> = synthetic(5)
            .into_iter()
            .map(|mut s| {
                s.runs.retain(|k, _| k == "top");
                s
            })
            .collect();
        assert!(check(one).unwrap_err().contains("needs at least 2"));
    }

    #[test]
    fn adapters_parse_both_recorded_shapes() {
        let swe = vec![
            ("a".to_owned(), r#"{"i1":{"resolved":true,"cost":0.5,"api_calls":10},"i2":{"resolved":false,"cost":0.4,"api_calls":30}}"#.to_owned()),
            ("b".to_owned(), r#"{"i1":{"resolved":false,"cost":0.1,"api_calls":12}}"#.to_owned()),
        ];
        let s = from_swebench_experiments(&swe).unwrap();
        assert_eq!(s.len(), 1, "i2 lacks model b and is dropped");
        assert_eq!(s[0].runs["a"].gate_signal, Some(10.0));
        let ck = r#"{"id":"x","turns":[{"tool_errors":0,"tool_results":0,"assistant_turns":0,"repeated_tool_calls":0,"rungs":[{"model":"h","gate_pass":false,"oracle_correct":true,"cost_usd":0.1,"gate_score":0.0},{"model":"s","gate_pass":true,"oracle_correct":true,"cost_usd":0.2,"gate_score":1.0}]}]}"#;
        let c = from_checkpoint(ck).unwrap();
        assert!(c[0].runs["h"].success && c[0].runs["h"].gate_signal == Some(1.0));
        let j = parse_jsonl(r#"{"id":"q","runs":{"a":{"success":true,"cost_usd":1.0},"b":{"success":false,"cost_usd":0.1}}}"#).unwrap();
        assert_eq!(j.len(), 1);
    }

    #[test]
    fn report_is_deterministic() {
        let s = synthetic(300);
        let a = serde_json::to_string(&run(&s, "x", Some("top"), 50.0, &fast()).unwrap()).unwrap();
        let b = serde_json::to_string(&run(&s, "x", Some("top"), 50.0, &fast()).unwrap()).unwrap();
        assert_eq!(a, b);
    }
}
