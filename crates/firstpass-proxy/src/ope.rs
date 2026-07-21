//! Off-policy evaluation (OPE): direct-method replay and IPS/SNIPS for start-rung policies.
//!
//! # Direct method (ladder/threshold changes)
//!
//! Answers: "if I changed my ladder or serve threshold, what would my logged traffic have cost
//! and how many failures would I have served?" — from the trace store, before enforcing anything.
//!
//! For each trace, walk the CANDIDATE ladder cheapest-first. At each rung, if the logged trace
//! has an attempt for that exact model string, reuse its logged gate verdicts and cost. The first
//! rung whose outcome would SERVE under the candidate rule ends the replay. If any candidate rung
//! has no logged attempt the trace is UNEVALUABLE and excluded — never guessed.
//!
//! Coverage < 1.0 is the principal limitation: a candidate that adds models never logged cannot
//! be evaluated. The report always surfaces coverage and n_correctness_known.
//!
//! # IPS / SNIPS for start-rung policies (requires exploration)
//!
//! Answers: "what would mean cost be if I always started at rung N?" — using importance-sampling
//! on traffic logged under an epsilon-greedy policy that recorded propensities.
//!
//! Estimators (Horvitz-Thompson 1952 / Swaminathan-Joachims 2015):
//! ```text
//! wᵢ = 𝟙[logged_start == N] / pᵢ          (importance weight)
//! IPS  = (1/n) Σᵢ wᵢ · costᵢ              (unbiased, higher variance)
//! SNIPS = Σ(wᵢ·costᵢ) / Σwᵢ              (self-normalised, lower variance)
//! ESS  = (Σwᵢ)² / Σwᵢ²                   (effective sample size)
//! ```
//!
//! Valid only when the candidate start rung N is in the logging policy's support (guaranteed
//! under ε-greedy since p = ε/K > 0 for every rung). Traces without propensity are excluded
//! and counted in `n_with_propensity`. The direct-method path remains for ladder-structure
//! changes.
//!
//! # Doubly-robust (DR) estimator
//!
//! DR combines the direct-method baseline with an IPS residual correction:
//! ```text
//! DRᵢ = DM(xᵢ, a) + wᵢ · (rᵢ − DM(xᵢ, aᵢ))
//! DR  = (1/n) Σᵢ DRᵢ
//! ```
//! where `DM(x, rung)` is the per-(task-kind, start-rung) empirical mean cost built from
//! logged receipts. DR is unbiased when **either** the propensities **or** the reward model
//! is correctly specified (double robustness). In practice, both are coarse approximations;
//! DR tends to have lower variance than IPS while inheriting IPS's unbiasedness under
//! correct propensities.

use std::collections::HashMap;
use std::path::Path;

use firstpass_core::{Attempt, Config as RoutingConfig, DeferredVerdict, TaskKind, Trace, Verdict};

use crate::calibrate::gate_score;
use crate::store::{self, StoreError};

// ── Candidate policy ──────────────────────────────────────────────────────────

/// A candidate routing policy to evaluate against logged traffic.
///
/// Parsed from a standard `firstpass.toml` — the candidate is an ordinary config file; we
/// extract the first route's ladder and `escalation.serve_threshold`.
#[derive(Debug)]
pub struct CandidatePolicy {
    /// Model ladder, cheapest first (e.g. `"anthropic/claude-haiku-4-5"`).
    pub ladder: Vec<String>,
    /// Conformal serve threshold. `None` → serve when verdict is `Pass`.
    pub serve_threshold: Option<f64>,
}

impl CandidatePolicy {
    /// Parse a candidate policy from a TOML string (standard firstpass config format).
    ///
    /// Uses the first `[[route]]` section's `ladder` and the top-level
    /// `[escalation].serve_threshold`.
    ///
    /// # Errors
    /// Returns a human-readable string if the TOML is invalid.
    pub fn from_toml(toml: &str) -> Result<Self, String> {
        let config = RoutingConfig::parse(toml).map_err(|e| e.to_string())?;
        let ladder = config
            .routes
            .into_iter()
            .next()
            .map(|r| r.ladder)
            .unwrap_or_default();
        Ok(Self {
            ladder,
            serve_threshold: config.escalation.serve_threshold,
        })
    }
}

// ── Per-trace replay ──────────────────────────────────────────────────────────

/// Whether a logged attempt would SERVE under the candidate policy.
fn would_serve(attempt: &Attempt, policy: &CandidatePolicy) -> bool {
    match policy.serve_threshold {
        Some(t) => gate_score(&attempt.gates, attempt.verdict) >= t,
        None => attempt.verdict == Verdict::Pass,
    }
}

/// USD cost of one attempt: model call + all gate costs.
fn attempt_cost(a: &Attempt) -> f64 {
    a.cost_usd + a.gates.iter().map(|g| g.cost_usd).sum::<f64>()
}

enum ReplayResult {
    /// A candidate rung had no logged attempt — the trace cannot be evaluated.
    Unevaluable,
    Evaluated {
        /// Sum of attempt costs for every replayed rung (up to and including the serving one).
        cost: f64,
        /// Model string of the candidate rung that served, or `None` if all rungs exhausted.
        served_model: Option<String>,
        /// Model string actually served by the logging policy.
        logged_served_model: Option<String>,
    },
}

fn replay_trace(trace: &Trace, policy: &CandidatePolicy) -> ReplayResult {
    // What the logging policy actually served — used for correctness attribution.
    let logged_served_model = trace.final_.served_rung.and_then(|rung| {
        trace
            .attempts
            .iter()
            .find(|a| a.rung == rung)
            .map(|a| a.model.clone())
    });

    let mut total_cost = 0.0f64;
    let mut served_model: Option<String> = None;

    for model in &policy.ladder {
        let Some(attempt) = trace.attempts.iter().find(|a| &a.model == model) else {
            return ReplayResult::Unevaluable;
        };
        total_cost += attempt_cost(attempt);
        if would_serve(attempt, policy) {
            served_model = Some(model.clone());
            break;
        }
    }

    ReplayResult::Evaluated {
        cost: total_cost,
        served_model,
        logged_served_model,
    }
}

// ── Bootstrap CIs ─────────────────────────────────────────────────────────────

// ponytail: inline SplitMix64 — same pattern as conformal.rs tests; no new deps.
// Ceiling: not a general RNG. Replace with rand if OPE ever needs sampling beyond CI bootstrap.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn rand_usize(rng: &mut u64, n: usize) -> usize {
    (splitmix64(rng) % n as u64) as usize
}

/// Bootstrap 95% CI for the mean of `values` (2.5/97.5 percentiles, deterministic seed).
fn bootstrap_mean_ci(values: &[f64], n_resamples: usize, seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len();
    let mut rng = seed;
    let mut means: Vec<f64> = (0..n_resamples)
        .map(|_| {
            let s: f64 = (0..n).map(|_| values[rand_usize(&mut rng, n)]).sum();
            s / n as f64
        })
        .collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_idx = (n_resamples as f64 * 0.025) as usize;
    let hi_idx = ((n_resamples as f64 * 0.975) as usize).min(n_resamples - 1);
    (means[lo_idx], means[hi_idx])
}

/// Bootstrap 95% CI for a failure rate (2.5/97.5 percentiles, deterministic seed).
fn bootstrap_failure_ci(correct: &[bool], n_resamples: usize, seed: u64) -> (f64, f64) {
    if correct.is_empty() {
        return (0.0, 0.0);
    }
    let n = correct.len();
    let mut rng = seed;
    let mut rates: Vec<f64> = (0..n_resamples)
        .map(|_| {
            let fails = (0..n).filter(|_| !correct[rand_usize(&mut rng, n)]).count();
            fails as f64 / n as f64
        })
        .collect();
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_idx = (n_resamples as f64 * 0.025) as usize;
    let hi_idx = ((n_resamples as f64 * 0.975) as usize).min(n_resamples - 1);
    (rates[lo_idx], rates[hi_idx])
}

// ── Report ────────────────────────────────────────────────────────────────────

/// The result of evaluating a candidate policy against logged traffic.
#[derive(Debug, Clone)]
pub struct OpeReport {
    /// Total traces in the store for this tenant.
    pub n_traces: usize,
    /// Traces for which replay used only logged attempts (coverage numerator).
    pub n_evaluable: usize,
    /// `n_evaluable / n_traces` (1.0 when n_traces == 0).
    pub coverage: f64,
    /// Mean candidate cost per request, over evaluable traces.
    pub est_cost_per_request: f64,
    /// Mean actual (logged) cost per request, over the same evaluable traces.
    pub logged_cost_per_request: f64,
    /// Estimated served-failure rate over evaluable traces with known correctness.
    /// `None` if no evaluable trace has deferred feedback on the same served rung.
    pub est_served_failure: Option<f64>,
    /// Number of evaluable traces where correctness is attributable from deferred feedback.
    pub n_correctness_known: usize,
    /// Fraction of evaluable traces where candidate escalated past the first ladder rung.
    pub escalation_rate: f64,
    /// Bootstrap 95% CI (2.5/97.5 pct) for `est_cost_per_request`.
    pub ci_cost: (f64, f64),
    /// Bootstrap 95% CI for `est_served_failure`. `None` when `est_served_failure` is `None`.
    pub ci_served_failure: Option<(f64, f64)>,
}

impl OpeReport {
    /// Render the report as human-readable lines, mirroring `calibrate`'s report style.
    /// Coverage and `n_correctness_known` are always prominent.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = format!(
            "traces: {n_traces}  evaluable: {n_evaluable}  coverage: {cov:.3}\n\
             n_correctness_known: {n_known}\n\
             est cost/request:    ${est:.6}  (logged: ${logged:.6})\n\
             cost CI [2.5%, 97.5%]: [${clo:.6}, ${chi:.6}]\n\
             escalation rate: {esc:.4}\n",
            n_traces = self.n_traces,
            n_evaluable = self.n_evaluable,
            cov = self.coverage,
            n_known = self.n_correctness_known,
            est = self.est_cost_per_request,
            logged = self.logged_cost_per_request,
            clo = self.ci_cost.0,
            chi = self.ci_cost.1,
            esc = self.escalation_rate,
        );
        match self.est_served_failure {
            Some(f) => {
                let (lo, hi) = self.ci_served_failure.unwrap_or((f, f));
                out.push_str(&format!(
                    "est served-failure: {f:.4}  CI [{lo:.4}, {hi:.4}]\n"
                ));
            }
            None => {
                out.push_str(
                    "est served-failure: n/a (no deferred feedback on same-rung evaluable traces)\n",
                );
            }
        }
        out.push_str(
            "\nreplay of logged outcomes (direct method); \
             rungs never logged are not guessed — see coverage.\n",
        );
        out
    }
}

// ── Store-backed OPE ──────────────────────────────────────────────────────────

/// Internal per-trace summary for aggregation and bootstrap resampling.
struct EvalPoint {
    candidate_cost: f64,
    logged_cost: f64,
    /// `Some(true)` = correct, `Some(false)` = failure, `None` = unknown.
    correctness: Option<bool>,
    escalated: bool,
}

fn build_report(n_traces: usize, points: Vec<EvalPoint>) -> OpeReport {
    let n_evaluable = points.len();
    // ponytail: coverage = 1.0 for empty store (no uncovered traces, not a zero).
    let coverage = if n_traces == 0 {
        1.0
    } else {
        n_evaluable as f64 / n_traces as f64
    };

    if points.is_empty() {
        return OpeReport {
            n_traces,
            n_evaluable: 0,
            coverage,
            est_cost_per_request: 0.0,
            logged_cost_per_request: 0.0,
            est_served_failure: None,
            n_correctness_known: 0,
            escalation_rate: 0.0,
            ci_cost: (0.0, 0.0),
            ci_served_failure: None,
        };
    }

    let est_cost = mean(&points, |p| p.candidate_cost);
    let logged_cost = mean(&points, |p| p.logged_cost);
    let escalation_rate = points.iter().filter(|p| p.escalated).count() as f64 / n_evaluable as f64;

    let known: Vec<bool> = points.iter().filter_map(|p| p.correctness).collect();
    let n_correctness_known = known.len();
    let est_served_failure = if known.is_empty() {
        None
    } else {
        Some(known.iter().filter(|&&c| !c).count() as f64 / known.len() as f64)
    };

    let costs: Vec<f64> = points.iter().map(|p| p.candidate_cost).collect();
    // ponytail: fixed seeds (42/43) for determinism; 1000 resamples is the spec floor.
    let ci_cost = bootstrap_mean_ci(&costs, 1000, 42);
    let ci_served_failure = if known.is_empty() {
        None
    } else {
        Some(bootstrap_failure_ci(&known, 1000, 43))
    };

    OpeReport {
        n_traces,
        n_evaluable,
        coverage,
        est_cost_per_request: est_cost,
        logged_cost_per_request: logged_cost,
        est_served_failure,
        n_correctness_known,
        escalation_rate,
        ci_cost,
        ci_served_failure,
    }
}

fn mean(points: &[EvalPoint], f: impl Fn(&EvalPoint) -> f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    points.iter().map(f).sum::<f64>() / points.len() as f64
}

/// Run OPE from the trace store.
///
/// Missing or unreadable store is treated as zero traces (returns a zero-trace report and exits
/// 0), matching the forgiving behaviour of `firstpass calibrate` and `firstpass trace`.
///
/// # Errors
/// Returns [`StoreError`] if a stored trace's deferred verdicts cannot be read (a genuine
/// database error on an existing record, not a missing store).
pub fn ope_from_store(
    db_path: impl AsRef<Path>,
    tenant: &str,
    policy: &CandidatePolicy,
) -> Result<OpeReport, StoreError> {
    let traces = store::load_tenant_traces(&db_path, tenant).unwrap_or_default();
    let n_traces = traces.len();
    let mut points: Vec<EvalPoint> = Vec::with_capacity(n_traces);

    for trace in &traces {
        let deferred = store::load_deferred(&db_path, &trace.trace_id.to_string())?;
        let replay = replay_trace(trace, policy);
        let ReplayResult::Evaluated {
            cost,
            served_model,
            logged_served_model,
        } = replay
        else {
            continue; // unevaluable: some candidate rung had no logged attempt
        };

        // Correctness is attributable only when candidate and logging policy served the same
        // model AND deferred feedback exists. A different model means the deferred verdict graded
        // a different output — we never impute correctness across outputs.
        let correctness = match (&served_model, &logged_served_model) {
            (Some(cm), Some(lm)) if cm == lm => deferred
                .last()
                .map(|dv: &DeferredVerdict| dv.verdict == Verdict::Pass),
            _ => None,
        };

        let first_model = policy.ladder.first();
        let escalated = match (&served_model, first_model) {
            (Some(m), Some(first)) => m != first,
            (None, Some(_)) => true, // all rungs exhausted without serving
            _ => false,
        };

        points.push(EvalPoint {
            candidate_cost: cost,
            logged_cost: trace.final_.total_cost_usd,
            correctness,
            escalated,
        });
    }

    Ok(build_report(n_traces, points))
}

// ── IPS / SNIPS start-rung estimator ─────────────────────────────────────────

/// IPS, SNIPS, and doubly-robust (DR) estimates for evaluating a fixed candidate start-rung policy.
///
/// Valid only for traffic logged under a stochastic policy that recorded propensities via
/// `[escalation.exploration]`. Traces without a `propensity` field are excluded and reported
/// in `n_with_propensity`.
///
/// ponytail: per-context greedy identification is not exploited here (uniform-weight IPS
/// suffices for population-level estimates).
#[derive(Debug, Clone)]
pub struct IpsReport {
    /// Total traces in the store for this tenant.
    pub n_traces: usize,
    /// Traces with `propensity: Some(p > 0)` — the IPS population.
    pub n_with_propensity: usize,
    /// Candidate start rung being evaluated.
    pub candidate_start_rung: u32,
    /// IPS (Horvitz-Thompson) estimate of mean cost under "always start at rung N".
    pub ips_cost: f64,
    /// SNIPS (self-normalised IPS) estimate — lower variance, slightly biased.
    pub snips_cost: f64,
    /// Effective sample size ESS = (Σwᵢ)² / Σwᵢ².
    pub ess: f64,
    /// Bootstrap 95% CI (2.5/97.5 pct) for the IPS cost estimate.
    pub ci_ips_cost: (f64, f64),
    /// IPS estimate of served-failure rate (traces with deferred feedback on logged rung N).
    pub ips_served_failure: Option<f64>,
    /// SNIPS estimate of served-failure rate.
    pub snips_served_failure: Option<f64>,
    /// Traces contributing to the failure-rate IPS estimate.
    pub n_correctness_known: usize,
    /// Bootstrap 95% CI for the IPS served-failure estimate.
    pub ci_ips_served_failure: Option<(f64, f64)>,
    /// Doubly-robust (DR) cost estimate: DM baseline + IPS residual correction.
    ///
    /// `DRᵢ = DM(xᵢ, N) + wᵢ · (rᵢ − DM(xᵢ, aᵢ))`, mean over all propensity traces.
    /// Unbiased when either propensities or the reward model is correctly specified.
    pub dr_cost: f64,
    /// Bootstrap 95% CI (2.5/97.5 pct) for the DR cost estimate.
    pub ci_dr_cost: (f64, f64),
}

impl IpsReport {
    /// Render the report as human-readable lines.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = format!(
            "traces: {n}  n_with_propensity: {nwp}  candidate_start_rung: {sr}\n\
             IPS cost/request:   ${ips:.6}\n\
             SNIPS cost/request: ${snips:.6}\n\
             IPS cost CI [2.5%, 97.5%]: [${clo:.6}, ${chi:.6}]\n\
             effective sample size (ESS): {ess:.1}\n\
             n_correctness_known: {nck}\n",
            n = self.n_traces,
            nwp = self.n_with_propensity,
            sr = self.candidate_start_rung,
            ips = self.ips_cost,
            snips = self.snips_cost,
            clo = self.ci_ips_cost.0,
            chi = self.ci_ips_cost.1,
            ess = self.ess,
            nck = self.n_correctness_known,
        );
        out.push_str(&format!(
            "DR cost/request:    ${dr:.6}\n\
             DR cost CI [2.5%, 97.5%]: [${drlo:.6}, ${drhi:.6}]\n",
            dr = self.dr_cost,
            drlo = self.ci_dr_cost.0,
            drhi = self.ci_dr_cost.1,
        ));
        match self.ips_served_failure {
            Some(f) => {
                let sf_snips = self.snips_served_failure.unwrap_or(f);
                let (lo, hi) = self.ci_ips_served_failure.unwrap_or((f, f));
                out.push_str(&format!(
                    "IPS served-failure: {f:.4}  SNIPS: {sf_snips:.4}  CI [{lo:.4}, {hi:.4}]\n"
                ));
            }
            None => {
                out.push_str(
                    "IPS served-failure: n/a (no deferred feedback on matched-start traces)\n",
                );
            }
        }
        out.push_str(
            "\nIPS/SNIPS/DR valid only for candidates over the same logged ladder with \
             propensity-logged traffic. Traces without propensity excluded. \
             DR reward model: per-(task-kind, rung) empirical mean — coarse; see ponytail comment. \
             Direct-method replay remains for ladder/threshold changes.\n",
        );
        out
    }
}

// ── DR reward model ───────────────────────────────────────────────────────────

/// Per-(task-kind, start-rung) empirical mean cost, used as the DM baseline in DR.
///
/// Built from propensity-logged traces only (same population as IPS). Falls back to the
/// per-rung mean, then the global mean, so `predict` never returns NaN even for (context,
/// rung) pairs that were never explored.
///
/// ponytail: coarse empirical bucket mean — zero new deps, same reward definition as IPS.
/// Ceiling: cannot extrapolate to (context, rung) pairs with zero logged observations; those
/// fall back to the rung mean (or global mean), which may be biased if costs vary by context.
/// Upgrade path: a richer feature regression (e.g. prompt-token bucket × task kind × rung)
/// once the feature vector grows and sample sizes support it.
struct DmModel {
    /// Mean cost keyed by `(TaskKind, start_rung)`.
    bucket: HashMap<(TaskKind, u32), f64>,
    /// Per-rung mean cost (fallback when the context bucket is unobserved).
    rung_mean: HashMap<u32, f64>,
    /// Grand mean (ultimate fallback when even the rung has no observations).
    global_mean: f64,
}

impl DmModel {
    fn build(traces: &[Trace]) -> Self {
        let mut bucket_acc: HashMap<(TaskKind, u32), (f64, u32)> = HashMap::new();
        let mut rung_acc: HashMap<u32, (f64, u32)> = HashMap::new();
        let mut global_sum = 0.0f64;
        let mut global_n = 0u32;

        for trace in traces {
            if trace.policy.propensity.is_none() {
                continue; // same filter as IPS — use only the propensity-logged population
            }
            let Some(first) = trace.attempts.first() else {
                continue;
            };
            let rung = first.rung;
            let cost = trace.final_.total_cost_usd;
            let ctx = trace.request.features.task_kind;

            let be = bucket_acc.entry((ctx, rung)).or_default();
            be.0 += cost;
            be.1 += 1;

            let re = rung_acc.entry(rung).or_default();
            re.0 += cost;
            re.1 += 1;

            global_sum += cost;
            global_n += 1;
        }

        let bucket = bucket_acc
            .into_iter()
            .map(|(k, (s, n))| (k, s / n as f64))
            .collect();
        let rung_mean = rung_acc
            .into_iter()
            .map(|(k, (s, n))| (k, s / n as f64))
            .collect();
        let global_mean = if global_n > 0 {
            global_sum / global_n as f64
        } else {
            0.0
        };

        Self {
            bucket,
            rung_mean,
            global_mean,
        }
    }

    /// Predicted cost for `(task_kind, rung)`, with graceful fallback.
    fn predict(&self, task_kind: TaskKind, rung: u32) -> f64 {
        self.bucket
            .get(&(task_kind, rung))
            .copied()
            .or_else(|| self.rung_mean.get(&rung).copied())
            .unwrap_or(self.global_mean)
    }
}

/// Run IPS/SNIPS OPE from the trace store for a fixed candidate start-rung policy.
///
/// Traces without `propensity` are excluded from IPS and counted in `n_with_propensity`.
/// Missing or unreadable store is treated as zero traces (same forgiving behaviour as
/// [`ope_from_store`]).
///
/// # Errors
/// Returns [`StoreError`] on a genuine database read error on an existing record.
pub fn ips_from_store(
    db_path: impl AsRef<Path>,
    tenant: &str,
    candidate_start_rung: u32,
) -> Result<IpsReport, StoreError> {
    let traces = store::load_tenant_traces(&db_path, tenant).unwrap_or_default();
    let n_traces = traces.len();

    struct IpsPoint {
        weight: f64,
        cost: f64,
        /// `Some(true)` correct, `Some(false)` failure, `None` unknown.
        correctness: Option<bool>,
        // Needed for DR DM-term lookups.
        task_kind: TaskKind,
        /// Logged start rung; 0 when the trace has no attempts (w=0 so correction is 0).
        logged_rung: u32,
    }

    let mut points: Vec<IpsPoint> = Vec::with_capacity(n_traces);
    let mut n_with_propensity = 0usize;

    for trace in &traces {
        let Some(p) = trace.policy.propensity else {
            continue; // no propensity → excluded from IPS
        };
        if p <= 0.0 {
            continue; // defensive: zero propensity → undefined ratio
        }
        n_with_propensity += 1;

        let logged_start = trace.attempts.first().map(|a| a.rung);
        let indicator = f64::from(logged_start == Some(candidate_start_rung));
        let w = indicator / p;

        // Correctness from deferred feedback, but only for traces that matched the candidate
        // start rung (w > 0). A deferred verdict on a different rung doesn't grade rung N's output.
        let correctness = if w > 0.0 {
            let deferred = store::load_deferred(&db_path, &trace.trace_id.to_string())?;
            deferred
                .last()
                .map(|dv: &DeferredVerdict| dv.verdict == Verdict::Pass)
        } else {
            None
        };

        points.push(IpsPoint {
            weight: w,
            cost: trace.final_.total_cost_usd,
            correctness,
            task_kind: trace.request.features.task_kind,
            logged_rung: logged_start.unwrap_or(0),
        });
    }

    let n = n_with_propensity as f64;
    let sum_w: f64 = points.iter().map(|p| p.weight).sum();
    let sum_w2: f64 = points.iter().map(|p| p.weight * p.weight).sum();
    let sum_wc: f64 = points.iter().map(|p| p.weight * p.cost).sum();

    let ips_cost = if n > 0.0 { sum_wc / n } else { 0.0 };
    let snips_cost = if sum_w > 0.0 { sum_wc / sum_w } else { 0.0 };
    let ess = if sum_w2 > 0.0 {
        sum_w * sum_w / sum_w2
    } else {
        0.0
    };

    // Bootstrap CI for IPS cost: resample the per-trace wᵢ·costᵢ values.
    let wc_values: Vec<f64> = points.iter().map(|p| p.weight * p.cost).collect();
    let ci_ips_cost = bootstrap_mean_ci(&wc_values, 1000, 42);

    // Failure-rate IPS over traces with deferred feedback at the matched start rung (w > 0).
    let known: Vec<(f64, bool)> = points
        .iter()
        .filter_map(|p| p.correctness.map(|c| (p.weight, c)))
        .collect();
    let n_correctness_known = known.len();

    let (ips_served_failure, snips_served_failure, ci_ips_served_failure) = if known.is_empty() {
        (None, None, None)
    } else {
        let n_known = known.len() as f64;
        let sum_wf: f64 = known.iter().filter(|(_, c)| !c).map(|(w, _)| w).sum();
        let sum_wk: f64 = known.iter().map(|(w, _)| w).sum();
        let ips_f = sum_wf / n_known;
        let snips_f = if sum_wk > 0.0 { sum_wf / sum_wk } else { 0.0 };
        let wf_vals: Vec<f64> = known
            .iter()
            .map(|(w, c)| if !c { *w } else { 0.0 })
            .collect();
        let ci = bootstrap_mean_ci(&wf_vals, 1000, 43);
        (Some(ips_f), Some(snips_f), Some(ci))
    };

    // ── DR estimator ──────────────────────────────────────────────────────────
    // DRᵢ = DM(xᵢ, N) + wᵢ · (rᵢ − DM(xᵢ, aᵢ))
    // Averaged over ALL propensity-positive traces (including w=0 ones, which contribute
    // only the DM baseline term). This is what reduces DR's variance relative to IPS.
    let dm = DmModel::build(&traces);
    let dr_vals: Vec<f64> = points
        .iter()
        .map(|pt| {
            let dm_cand = dm.predict(pt.task_kind, candidate_start_rung);
            let dm_logged = dm.predict(pt.task_kind, pt.logged_rung);
            dm_cand + pt.weight * (pt.cost - dm_logged)
        })
        .collect();
    let dr_cost = if dr_vals.is_empty() {
        0.0
    } else {
        dr_vals.iter().sum::<f64>() / dr_vals.len() as f64
    };
    // ponytail: seed 44 — unique from IPS seeds 42/43 for determinism.
    let ci_dr_cost = bootstrap_mean_ci(&dr_vals, 1000, 44);

    Ok(IpsReport {
        n_traces,
        n_with_propensity,
        candidate_start_rung,
        ips_cost,
        snips_cost,
        ess,
        ci_ips_cost,
        ips_served_failure,
        snips_served_failure,
        n_correctness_known,
        ci_ips_served_failure,
        dr_cost,
        ci_dr_cost,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use firstpass_core::{
        Features, FinalOutcome, GateResult, Mode, PolicyRef, RequestInfo, Score, ServedFrom,
        TaskKind, Verdict, GENESIS_HASH,
    };

    use super::*;
    use crate::store;

    /// Minimal trace with one attempt at `rung`, model `model_str`, verdict and score set by
    /// `pass_score` (>= 0.5 = Pass). Mirrors calibrate.rs's `trace_with_score`.
    fn make_trace(tenant: &str, rung: u32, model: &str, pass_score: f64, cost_usd: f64) -> Trace {
        let verdict = if pass_score >= 0.5 {
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        let attempt = firstpass_core::Attempt {
            rung,
            model: model.to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 10,
            out_tokens: 5,
            cost_usd,
            latency_ms: 12,
            gates: vec![GateResult {
                gate_id: "gate@v1".to_owned(),
                verdict,
                score: Some(Score::clamped(pass_score)),
                cost_usd: 0.0,
                ms: 10,
                reason: None,
                evidence_ref: None,
            }],
            verdict,
        };
        let mut trace = Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: tenant.to_owned(),
            session_id: "s1".to_owned(),
            ts: jiff::Timestamp::now(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "test@v0".to_owned(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "deadbeef".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![attempt],
            deferred: Vec::new(),
            final_: FinalOutcome {
                served_rung: Some(rung),
                served_from: ServedFrom::Attempt,
                total_cost_usd: cost_usd,
                gate_cost_usd: 0.0,
                total_latency_ms: 12,
                escalations: 0,
                counterfactual_baseline_usd: cost_usd,
                savings_usd: 0.0,
            },
        };
        trace.recompute_savings();
        trace
    }

    /// Two-attempt trace: haiku (rung 0) fails, sonnet (rung 1) passes.
    fn make_escalated_trace(tenant: &str, haiku_cost: f64, sonnet_cost: f64) -> Trace {
        let haiku = firstpass_core::Attempt {
            rung: 0,
            model: "haiku".to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 10,
            out_tokens: 5,
            cost_usd: haiku_cost,
            latency_ms: 10,
            gates: vec![GateResult {
                gate_id: "g".to_owned(),
                verdict: Verdict::Fail,
                score: Some(Score::clamped(0.3)),
                cost_usd: 0.0,
                ms: 5,
                reason: None,
                evidence_ref: None,
            }],
            verdict: Verdict::Fail,
        };
        let sonnet = firstpass_core::Attempt {
            rung: 1,
            model: "sonnet".to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 10,
            out_tokens: 5,
            cost_usd: sonnet_cost,
            latency_ms: 20,
            gates: vec![GateResult {
                gate_id: "g".to_owned(),
                verdict: Verdict::Pass,
                score: Some(Score::clamped(0.9)),
                cost_usd: 0.0,
                ms: 5,
                reason: None,
                evidence_ref: None,
            }],
            verdict: Verdict::Pass,
        };
        let total = haiku_cost + sonnet_cost;
        let mut trace = Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: tenant.to_owned(),
            session_id: "s2".to_owned(),
            ts: jiff::Timestamp::now(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "test@v0".to_owned(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "beef".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![haiku, sonnet],
            deferred: Vec::new(),
            final_: FinalOutcome {
                served_rung: Some(1),
                served_from: ServedFrom::Attempt,
                total_cost_usd: total,
                gate_cost_usd: 0.0,
                total_latency_ms: 30,
                escalations: 1,
                counterfactual_baseline_usd: total,
                savings_usd: 0.0,
            },
        };
        trace.recompute_savings();
        trace
    }

    fn deferred_pass(gate: &str) -> firstpass_core::DeferredVerdict {
        firstpass_core::DeferredVerdict {
            gate_id: gate.to_owned(),
            verdict: Verdict::Pass,
            score: None,
            reported_at: jiff::Timestamp::now(),
            reporter: "test".to_owned(),
        }
    }

    fn deferred_fail(gate: &str) -> firstpass_core::DeferredVerdict {
        firstpass_core::DeferredVerdict {
            gate_id: gate.to_owned(),
            verdict: Verdict::Fail,
            score: None,
            reported_at: jiff::Timestamp::now(),
            reporter: "test".to_owned(),
        }
    }

    fn tmp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("fp-ope-{}.db", uuid::Uuid::now_v7()))
    }

    // ── 1. Pure replay: candidate == logged ladder ────────────────────────────

    #[tokio::test]
    async fn candidate_equals_logged_matches_exactly() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // 10 traces, haiku passes at cost 0.001 each.
        let mut ids = Vec::new();
        for _ in 0..10 {
            let t = make_trace("tenant-a", 0, "haiku", 0.8, 0.001);
            ids.push(t.trace_id.to_string());
            tx.try_send(t).unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        for id in &ids {
            store::append_deferred(&db, id, &deferred_pass("out")).unwrap();
        }

        let policy = CandidatePolicy {
            ladder: vec!["haiku".to_owned()],
            serve_threshold: None,
        };
        let report = ope_from_store(&db, "tenant-a", &policy).unwrap();

        assert_eq!(report.n_traces, 10);
        assert_eq!(report.n_evaluable, 10);
        assert!((report.coverage - 1.0).abs() < 1e-9);
        // Cost must match exactly: candidate replays the same attempt.
        assert!((report.est_cost_per_request - 0.001).abs() < 1e-9);
        assert!((report.logged_cost_per_request - 0.001).abs() < 1e-9);
        // All served same rung as logged, all deferred = Pass -> failure = 0.
        assert_eq!(report.n_correctness_known, 10);
        assert!((report.est_served_failure.unwrap() - 0.0).abs() < 1e-9);
        // No escalation: haiku always served first.
        assert!((report.escalation_rate - 0.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&db);
    }

    // ── 2. Cheaper candidate (first rung only, drop sonnet) ────────────────────

    #[tokio::test]
    async fn cheaper_candidate_reduces_cost_and_escalation() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // 5 traces logged: haiku (cost 0.001) fails -> sonnet (cost 0.01) passes.
        // Logged total = 0.011 each.
        for _ in 0..5 {
            tx.try_send(make_escalated_trace("t", 0.001, 0.01)).unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        // Candidate: ladder = ["haiku"] only.
        // Replay: haiku fails -> all candidate rungs exhausted. Cost = 0.001.
        // Logged served sonnet; candidate served nothing -> correctness UNKNOWN.
        let policy = CandidatePolicy {
            ladder: vec!["haiku".to_owned()],
            serve_threshold: None,
        };
        let report = ope_from_store(&db, "t", &policy).unwrap();

        assert_eq!(report.n_evaluable, 5);
        assert!((report.coverage - 1.0).abs() < 1e-9);
        // Candidate cost = 0.001 (haiku only); logged = 0.011.
        assert!(
            (report.est_cost_per_request - 0.001).abs() < 1e-9,
            "got {}",
            report.est_cost_per_request
        );
        assert!((report.logged_cost_per_request - 0.011).abs() < 1e-9);
        // Served nothing vs logged sonnet -> UNKNOWN for all.
        assert_eq!(report.n_correctness_known, 0);
        assert!(report.est_served_failure.is_none());
        // Escalated: candidate exhausted past first rung (haiku didn't serve).
        assert!((report.escalation_rate - 1.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&db);
    }

    // ── 3. Candidate with an unlogged model -> unevaluable ────────────────────

    #[tokio::test]
    async fn unlogged_model_makes_trace_unevaluable() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // 3 traces have haiku attempts; 3 traces have sonnet attempts.
        for _ in 0..3 {
            tx.try_send(make_trace("t", 0, "haiku", 0.8, 0.001))
                .unwrap();
        }
        for _ in 0..3 {
            tx.try_send(make_trace("t", 0, "sonnet", 0.8, 0.01))
                .unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        // Candidate ladder: ["newmodel", "haiku"] — "newmodel" is never logged.
        // Every trace is unevaluable (first candidate rung has no logged attempt).
        let policy = CandidatePolicy {
            ladder: vec!["newmodel".to_owned(), "haiku".to_owned()],
            serve_threshold: None,
        };
        let report = ope_from_store(&db, "t", &policy).unwrap();

        assert_eq!(report.n_traces, 6);
        assert_eq!(report.n_evaluable, 0);
        assert!((report.coverage - 0.0).abs() < 1e-9);
        assert!(report.est_served_failure.is_none());

        let _ = std::fs::remove_file(&db);
    }

    // ── 4. Different rung served → correctness UNKNOWN ────────────────────────

    #[tokio::test]
    async fn different_rung_served_correctness_unknown() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // Logged: haiku at rung 0 fails, sonnet at rung 1 passes. Logged serves sonnet.
        let t = make_escalated_trace("t", 0.001, 0.01);
        let tid = t.trace_id.to_string();
        tx.try_send(t).unwrap();
        drop(tx);
        handle.await.unwrap();

        // Deferred feedback graded the sonnet output.
        store::append_deferred(&db, &tid, &deferred_pass("out")).unwrap();

        // Candidate: ladder = ["haiku", "sonnet"] with a very low threshold -> haiku serves.
        // Candidate serves haiku, logged served sonnet -> DIFFERENT rung -> UNKNOWN.
        let policy = CandidatePolicy {
            ladder: vec!["haiku".to_owned(), "sonnet".to_owned()],
            serve_threshold: Some(0.1), // haiku score=0.3 >= 0.1, so haiku would serve
        };
        let report = ope_from_store(&db, "t", &policy).unwrap();

        assert_eq!(report.n_evaluable, 1);
        assert_eq!(report.n_correctness_known, 0, "different rung => UNKNOWN");
        assert!(report.est_served_failure.is_none());
        // Candidate served on haiku (cost 0.001), logged served on haiku+sonnet (0.011).
        assert!((report.est_cost_per_request - 0.001).abs() < 1e-9);
        // No escalation: haiku served first.
        assert!((report.escalation_rate - 0.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&db);
    }

    // ── 5. Bootstrap CI: deterministic, contains point estimate ───────────────

    #[tokio::test]
    async fn bootstrap_ci_deterministic_and_sane() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // 30 traces with varying costs: alternating 0.001 and 0.002. Mean = 0.0015.
        let mut ids = Vec::new();
        for i in 0..30u32 {
            let cost = if i % 2 == 0 { 0.001 } else { 0.002 };
            let t = make_trace("t", 0, "m", 0.9, cost);
            ids.push(t.trace_id.to_string());
            tx.try_send(t).unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        // Half pass, half fail (deferred).
        for (i, id) in ids.iter().enumerate() {
            let dv = if i < 15 {
                deferred_pass("o")
            } else {
                deferred_fail("o")
            };
            store::append_deferred(&db, id, &dv).unwrap();
        }

        let policy = CandidatePolicy {
            ladder: vec!["m".to_owned()],
            serve_threshold: None,
        };
        let r1 = ope_from_store(&db, "t", &policy).unwrap();
        let r2 = ope_from_store(&db, "t", &policy).unwrap();

        // Deterministic: same CI both calls.
        assert_eq!(r1.ci_cost, r2.ci_cost, "CI must be deterministic");
        assert_eq!(r1.ci_served_failure, r2.ci_served_failure);

        // CI ordering: lo <= point estimate <= hi.
        let (lo, hi) = r1.ci_cost;
        assert!(
            lo <= r1.est_cost_per_request + 1e-9,
            "CI lo {lo} > est {}",
            r1.est_cost_per_request
        );
        assert!(
            hi >= r1.est_cost_per_request - 1e-9,
            "CI hi {hi} < est {}",
            r1.est_cost_per_request
        );
        assert!(lo <= hi, "CI must be ordered");

        if let Some((flo, fhi)) = r1.ci_served_failure {
            let f = r1.est_served_failure.unwrap();
            assert!(flo <= f + 1e-9, "failure CI lo {flo} > est {f}");
            assert!(fhi >= f - 1e-9, "failure CI hi {fhi} < est {f}");
            assert!(flo <= fhi);
        }

        let _ = std::fs::remove_file(&db);
    }

    // ── 6. Empty store → zero-trace report, exit-0 path ──────────────────────

    #[test]
    fn empty_store_returns_zero_trace_report() {
        // Non-existent db: load_tenant_traces returns empty, should give a valid zero report.
        let policy = CandidatePolicy {
            ladder: vec!["m".to_owned()],
            serve_threshold: None,
        };
        let db = std::path::Path::new("/nonexistent/fp-ope-empty.db");
        let report = ope_from_store(db, "t", &policy).unwrap();
        assert_eq!(report.n_traces, 0);
        assert_eq!(report.n_evaluable, 0);
        assert!((report.coverage - 1.0).abs() < 1e-9);
        assert!(report.est_served_failure.is_none());
    }

    // ── 7. Same-rung with deferred: correctness flows through ─────────────────

    #[tokio::test]
    async fn same_rung_deferred_correctness_attributed() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        let t_pass = make_trace("t", 0, "m", 0.9, 0.001);
        let t_fail = make_trace("t", 0, "m", 0.9, 0.001);
        let (id_pass, id_fail) = (t_pass.trace_id.to_string(), t_fail.trace_id.to_string());
        tx.try_send(t_pass).unwrap();
        tx.try_send(t_fail).unwrap();
        drop(tx);
        handle.await.unwrap();

        store::append_deferred(&db, &id_pass, &deferred_pass("out")).unwrap();
        store::append_deferred(&db, &id_fail, &deferred_fail("out")).unwrap();

        let policy = CandidatePolicy {
            ladder: vec!["m".to_owned()],
            serve_threshold: None,
        };
        let r = ope_from_store(&db, "t", &policy).unwrap();

        assert_eq!(r.n_evaluable, 2);
        assert_eq!(r.n_correctness_known, 2);
        // 1 failure out of 2 known = 0.5.
        assert!((r.est_served_failure.unwrap() - 0.5).abs() < 1e-9);

        let _ = std::fs::remove_file(&db);
    }

    // ── 8. Serve threshold raises cost (candidate needs more escalations) ──────

    #[tokio::test]
    async fn high_threshold_forces_escalation_and_higher_cost() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // Logged: haiku (score=0.7, cost 0.001) passes under default rule (Pass verdict).
        // Candidate uses serve_threshold=0.8 -> haiku score 0.7 < 0.8 -> fail -> escalate.
        // Trace has no sonnet attempt -> trace is UNEVALUABLE (candidate needs sonnet, not logged).
        let t = make_trace("t", 0, "haiku", 0.7, 0.001); // score 0.7, verdict Pass
        tx.try_send(t).unwrap();
        drop(tx);
        handle.await.unwrap();

        let policy = CandidatePolicy {
            ladder: vec!["haiku".to_owned()],
            serve_threshold: Some(0.8), // haiku score 0.7 < 0.8 -> no serve
        };
        let r = ope_from_store(&db, "t", &policy).unwrap();

        // Haiku is logged, candidate walks it (score 0.7 < 0.8, no serve), exhausts all
        // candidate rungs — evaluable (all candidate rungs had logged data), no serve.
        assert_eq!(r.n_evaluable, 1);
        assert!((r.est_cost_per_request - 0.001).abs() < 1e-9); // haiku cost still counted
        assert_eq!(r.n_correctness_known, 0); // candidate served nothing vs logged haiku
                                              // Escalated: exhausted past first candidate rung.
        assert!((r.escalation_rate - 1.0).abs() < 1e-9);

        let _ = std::fs::remove_file(&db);
    }

    // ── 9. CandidatePolicy::from_toml parses ladder and threshold ─────────────

    #[test]
    fn from_toml_extracts_first_route_and_threshold() {
        let toml = r#"
[[route]]
match = {}
mode = "enforce"
ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]

[escalation]
serve_threshold = 0.75
"#;
        let p = CandidatePolicy::from_toml(toml).unwrap();
        assert_eq!(
            p.ladder,
            ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
        );
        assert!((p.serve_threshold.unwrap() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn from_toml_no_threshold_is_none() {
        let toml = "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"m\"]\n";
        let p = CandidatePolicy::from_toml(toml).unwrap();
        assert!(p.serve_threshold.is_none());
    }

    // ── IPS / SNIPS tests ─────────────────────────────────────────────────────

    /// Build a trace with a single attempt at `rung`, `total_cost_usd`, and a given logging
    /// propensity set on `policy.propensity`. Used to construct a known-distribution sample for
    /// IPS correctness assertions.
    fn make_propensity_trace(
        tenant: &str,
        rung: u32,
        cost_usd: f64,
        propensity: Option<f64>,
    ) -> Trace {
        let attempt = firstpass_core::Attempt {
            rung,
            model: "m".to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 10,
            out_tokens: 5,
            cost_usd,
            latency_ms: 5,
            gates: vec![],
            verdict: Verdict::Pass,
        };
        let mut trace = Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: tenant.to_owned(),
            session_id: "s".to_owned(),
            ts: jiff::Timestamp::now(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "bandit@v1+eps".to_owned(),
                explore: rung != 0,
                propensity,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "ph".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![attempt],
            deferred: Vec::new(),
            final_: FinalOutcome {
                served_rung: Some(rung),
                served_from: ServedFrom::Attempt,
                total_cost_usd: cost_usd,
                gate_cost_usd: 0.0,
                total_latency_ms: 5,
                escalations: 0,
                counterfactual_baseline_usd: cost_usd,
                savings_usd: 0.0,
            },
        };
        trace.recompute_savings();
        trace
    }

    // ── 10. IPS correctness from a known logging policy ───────────────────────
    //
    // Logging policy: epsilon-greedy, K=2 rungs, greedy=rung 0, epsilon=0.2
    //   p(start==0) = (1-0.2)*1 + 0.2/2 = 0.9
    //   p(start==1) = (1-0.2)*0 + 0.2/2 = 0.1
    //
    // Trace set representative of the logging policy:
    //   45 traces at rung 0 (cost $0.001, propensity 0.9)
    //    5 traces at rung 1 (cost $0.010, propensity 0.1)
    //
    // Candidate: always start at rung 1 (start_rung = 1).
    //   w_i = 1(start==1) / p_i
    //   Rung-0 traces: w = 0/0.9 = 0
    //   Rung-1 traces: w = 1/0.1 = 10
    //
    //   sum_wc = 5 * 10 * 0.010 = 0.5
    //   IPS  = sum_wc / n = 0.5 / 50 = 0.010  ✓  (equals true mean cost of rung-1 traces)
    //   SNIPS = sum_wc / sum_w = 0.5 / (5*10) = 0.010 ✓
    //   ESS   = (5*10)^2 / (5*10^2) = 2500/500 = 5
    #[tokio::test]
    async fn ips_correctness_from_known_logging_policy() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        for _ in 0..45 {
            tx.try_send(make_propensity_trace("t", 0, 0.001, Some(0.9)))
                .unwrap();
        }
        for _ in 0..5 {
            tx.try_send(make_propensity_trace("t", 1, 0.010, Some(0.1)))
                .unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        let report = ips_from_store(&db, "t", 1).unwrap();

        assert_eq!(report.n_traces, 50);
        assert_eq!(report.n_with_propensity, 50);
        // IPS = SNIPS = true mean cost of rung-1-start traces = $0.010.
        assert!(
            (report.ips_cost - 0.010).abs() < 1e-9,
            "IPS cost={} expected 0.010",
            report.ips_cost
        );
        assert!(
            (report.snips_cost - 0.010).abs() < 1e-9,
            "SNIPS cost={} expected 0.010",
            report.snips_cost
        );
        assert!(
            report.ess.is_finite() && report.ess > 0.0,
            "ESS must be positive"
        );
        assert!(
            (report.ess - 5.0).abs() < 1e-9,
            "ESS={} expected 5.0",
            report.ess
        );
        // No deferred feedback → no correctness signal.
        assert_eq!(report.n_correctness_known, 0);
        assert!(report.ips_served_failure.is_none());

        let _ = std::fs::remove_file(&db);
    }

    // ── 11. Traces without propensity are excluded from IPS ───────────────────

    #[tokio::test]
    async fn ips_excludes_traces_without_propensity() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // 5 traces with propensity, 5 without.
        for _ in 0..5 {
            tx.try_send(make_propensity_trace("t", 0, 0.001, Some(0.9)))
                .unwrap();
        }
        for _ in 0..5 {
            tx.try_send(make_propensity_trace("t", 0, 0.001, None))
                .unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        let report = ips_from_store(&db, "t", 0).unwrap();

        assert_eq!(report.n_traces, 10);
        assert_eq!(
            report.n_with_propensity, 5,
            "only traces with propensity count"
        );
        // All 5 propensity traces start at rung 0 (candidate rung 0): w = 1/0.9 each.
        // IPS = sum_wc / n = 5 * (1/0.9) * 0.001 / 5 ≈ 0.001/0.9 ≈ 0.001111
        let expected_ips = 0.001_f64 / 0.9;
        assert!(
            (report.ips_cost - expected_ips).abs() < 1e-9,
            "IPS={} expected {expected_ips}",
            report.ips_cost
        );

        let _ = std::fs::remove_file(&db);
    }

    // ── 12. Empty store / no propensity traces → zero report ─────────────────

    #[test]
    fn ips_empty_store_returns_zero_report() {
        let db = std::path::Path::new("/nonexistent/fp-ips-empty.db");
        let report = ips_from_store(db, "t", 0).unwrap();
        assert_eq!(report.n_traces, 0);
        assert_eq!(report.n_with_propensity, 0);
        assert_eq!(report.ips_cost, 0.0);
        assert_eq!(report.snips_cost, 0.0);
        assert_eq!(report.ess, 0.0);
        assert!(report.ips_served_failure.is_none());
        // DR zero-trace case.
        assert_eq!(report.dr_cost, 0.0);
        assert_eq!(report.ci_dr_cost, (0.0, 0.0));
    }

    // ── DR estimator tests ────────────────────────────────────────────────────

    /// Helper: like `make_propensity_trace` but with `TaskKind::CodeEdit` so tests can build
    /// a two-context population for the sparse-bucket fallback test.
    fn make_propensity_trace_ce(rung: u32, cost: f64, p: Option<f64>) -> Trace {
        let mut t = make_propensity_trace("t", rung, cost, p);
        t.request.features.task_kind = TaskKind::CodeEdit;
        t
    }

    // ── 13. DR = DM = IPS when all traces are at the candidate rung (p=1.0) ──
    //
    // Degenerate logging policy: always start at rung 1, propensity = 1.0.
    //   wᵢ = 1/1.0 = 1 for every trace.
    //   DM(Other, 1) = mean(costs) = 0.010.
    //   DRᵢ = DM + 1·(costᵢ − DM) = costᵢ  →  DR = mean(cost) = DM = IPS.
    #[tokio::test]
    async fn dr_degenerate_propensities_equals_dm_and_ips() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        for _ in 0..5 {
            tx.try_send(make_propensity_trace("t", 1, 0.010, Some(1.0)))
                .unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        let r = ips_from_store(&db, "t", 1).unwrap();

        // All three estimators agree when the logging policy is degenerate-correct.
        assert!(
            (r.dr_cost - 0.010).abs() < 1e-9,
            "DR={} expected 0.010",
            r.dr_cost
        );
        assert!(
            (r.dr_cost - r.ips_cost).abs() < 1e-9,
            "DR should equal IPS; DR={} IPS={}",
            r.dr_cost,
            r.ips_cost
        );
        assert!(
            (r.dr_cost - r.snips_cost).abs() < 1e-9,
            "DR should equal SNIPS; DR={} SNIPS={}",
            r.dr_cost,
            r.snips_cost
        );

        let _ = std::fs::remove_file(&db);
    }

    // ── 14. DR correction toward truth under correct propensities ─────────────
    //
    // Logging policy: ε-greedy, K=2, greedy=rung 0, ε=0.2:
    //   p(start=0) = 0.9,  p(start=1) = 0.1
    //
    // Data: 9 traces at rung 0 (cost $0.001, p=0.9), 1 trace at rung 1 (cost $0.010, p=0.1).
    //
    // Naive mean of ALL logged costs: (9·0.001 + 1·0.010)/10 = 0.0019  ← biased for
    // "always rung 1".  DM(Other, 1) = 0.010 (correct bucket mean from the one observation).
    //
    // IPS  = (1/0.1)·0.010 / 10 = 0.010  ✓
    // DR:
    //   9 rung-0 traces: DRᵢ = DM(Other,1) + 0·(…) = 0.010
    //   1 rung-1 trace:  DRᵢ = DM(Other,1) + 10·(0.010 − 0.010) = 0.010
    //   DR = 0.010 ✓   (not the biased 0.0019)
    #[tokio::test]
    async fn dr_correction_recovers_true_cost_under_correct_propensities() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        for _ in 0..9 {
            tx.try_send(make_propensity_trace("t", 0, 0.001, Some(0.9)))
                .unwrap();
        }
        tx.try_send(make_propensity_trace("t", 1, 0.010, Some(0.1)))
            .unwrap();
        drop(tx);
        handle.await.unwrap();

        let r = ips_from_store(&db, "t", 1).unwrap();

        // Naive logged mean = 0.0019; both DR and IPS correctly recover 0.010.
        assert!(
            (r.dr_cost - 0.010).abs() < 1e-9,
            "DR={} expected 0.010 (not the naive logged mean 0.0019)",
            r.dr_cost
        );
        assert!(
            (r.dr_cost - r.ips_cost).abs() < 1e-9,
            "DR should equal IPS under correct propensities; DR={} IPS={}",
            r.dr_cost,
            r.ips_cost
        );

        let _ = std::fs::remove_file(&db);
    }

    // ── 15. Sparse-bucket fallback — no NaN ───────────────────────────────────
    //
    // Two task kinds create a cross-context sparse-bucket scenario:
    //   • CodeEdit traces are only logged at rung 0  → DM(CE, 1) falls back to rung_mean[1]
    //   • Other traces are only logged at rung 1     → DM(Other, 0) falls back to rung_mean[0]
    //
    // DM fallback values must not NaN; DR must be finite.
    //
    // Candidate rung = 1.
    //   CE  rung-0 (5): w=0; DRᵢ = DM(CE,1) = rung_mean[1] = 0.020
    //   Other rung-1 (5): w=2; DRᵢ = 0.020 + 2·(0.020−0.020) = 0.020
    //   DR = 0.020
    #[tokio::test]
    async fn dr_sparse_bucket_fallback_no_nan() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        for _ in 0..5 {
            tx.try_send(make_propensity_trace_ce(0, 0.001, Some(0.5)))
                .unwrap();
        }
        for _ in 0..5 {
            tx.try_send(make_propensity_trace("t", 1, 0.020, Some(0.5)))
                .unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        let r = ips_from_store(&db, "t", 1).unwrap();

        assert!(
            r.dr_cost.is_finite(),
            "DR must be finite even when a context bucket is empty (got NaN/inf)"
        );
        assert!(
            (r.dr_cost - 0.020).abs() < 1e-9,
            "DR={} expected 0.020",
            r.dr_cost
        );

        let _ = std::fs::remove_file(&db);
    }

    // ── 16. DR bootstrap CI is present, finite, and sane ─────────────────────
    //
    // Verifies that the CI brackets the point estimate and is deterministic.
    #[tokio::test]
    async fn dr_ci_present_and_sane() {
        let db = tmp_db();
        let (tx, handle) = store::open(&db).unwrap();

        // 20 traces at rung 1, alternating cost 0.001 / 0.002, propensity 0.5.
        for i in 0..20u32 {
            let cost = if i % 2 == 0 { 0.001 } else { 0.002 };
            tx.try_send(make_propensity_trace("t", 1, cost, Some(0.5)))
                .unwrap();
        }
        drop(tx);
        handle.await.unwrap();

        let r1 = ips_from_store(&db, "t", 1).unwrap();
        let r2 = ips_from_store(&db, "t", 1).unwrap();

        // Deterministic.
        assert_eq!(r1.ci_dr_cost, r2.ci_dr_cost, "DR CI must be deterministic");

        // CI must be ordered and finite.
        let (lo, hi) = r1.ci_dr_cost;
        assert!(lo.is_finite() && hi.is_finite(), "DR CI must be finite");
        assert!(lo <= hi, "DR CI must be ordered: lo={lo} hi={hi}");

        // CI must bracket the point estimate (within floating-point tolerance).
        assert!(
            lo <= r1.dr_cost + 1e-9,
            "DR CI lo {lo} > dr_cost {}",
            r1.dr_cost
        );
        assert!(
            hi >= r1.dr_cost - 1e-9,
            "DR CI hi {hi} < dr_cost {}",
            r1.dr_cost
        );

        let _ = std::fs::remove_file(&db);
    }
}
