//! σ-sweep of the noisy-prior policies (**SIMULATION**): does a noisy pre-generation
//! decision-model prior — like TypeSafe's Jev, which predicts which model tier can handle a query
//! — improve Firstpass when its prediction is fused with the gate, instead of served unverified?
//!
//! This only exists in the deterministic simulation: the noisy prior is built from a sim-only
//! ground truth ([`SimBackend::clearance`](crate::sim::SimBackend::clearance)), which a live task
//! has no equivalent of.
//!
//! # Exploratory (post-hoc, NOT pre-registered): adverse-selection workload
//!
//! The default suite above ties: at σ=0 `firstpass+prior` never beats `firstpass`, because on that
//! suite the expected-cost start rule ([`argmin_expected_cost`]) never has a reason to skip rung 0.
//! Two things hold it back, and both are the default suite's construction, not the rule's:
//!
//! 1. Rung 0's clearance (`strength − 0.45·difficulty`, [`SimBackend::clearance`]) never drops much
//!    below ~0.10 on a `difficulty ~ U(0,1)` suite, so `argmin_expected_cost` is never offered a
//!    task hopeless enough to justify skipping the cheap rung.
//! 2. `Task::prompt_tokens` is drawn independent of `difficulty` ([`task_suite`]), so the model has
//!    no reason to think a hard task costs more than an easy one — unlike `costaware.rs`'s
//!    real-MBPP finding that escalating (hard) tasks pay **2.16×** the cheap-rung cost of tasks
//!    that pass there ($0.01881 vs $0.00869 — see [`ADVERSE_SELECTION_TOKEN_MULTIPLIER`]).
//!
//! [`adverse_selection_suite`] and [`run_adverse_selection_scenario`] build a second workload —
//! same base tasks and seed as the pre-registered suite — that repairs both gaps in one documented,
//! modest way (chosen *before* looking at the result, per costaware.rs's measured multiplier and
//! this crate's own floor-clearance test fixture, not tuned to produce a win):
//!
//! - a **hard tail**, one task in [`HARD_TAIL_EVERY`], pushed past rung 0's clearance floor by
//!   [`HARD_TAIL_DIFFICULTY_OFFSET`];
//! - those same hard-tail tasks' prompt and completion tokens scaled by
//!   [`ADVERSE_SELECTION_TOKEN_MULTIPLIER`], the multiplier `costaware.rs` measured, not invented.
//!
//! **This scenario is exploratory and cannot override the pre-registered kill criterion above.**
//! It exists to check the mechanism can act at all on a workload shaped like the one real
//! measurement (`costaware.rs`) found adversely selected — nothing here is pre-registered, and its
//! own kill-criterion-shaped row (rendered for legibility, not as a gate) is informational only.

use crate::metrics::{PolicyMetrics, evaluate};
use crate::policy::{
    AlwaysCheap, AlwaysTop, Decision, Firstpass, FirstpassPrior, Policy, PredictivePrior,
    PredictiveRouter, RandomRung,
};
use crate::sim::{Completion, Gate, ModelBackend, Rung, Task, task_suite};
use firstpass_core::PriceTable;
use serde::Serialize;
use std::fmt::Write as _;

/// The noise levels swept — pre-registered before the numbers were read.
pub const SIGMA_SWEEP: [f64; 4] = [0.0, 0.1, 0.2, 0.4];

/// σ at or below which the kill criterion is checked.
const KILL_SIGMA_CEILING: f64 = 0.2;

/// One σ's worth of policy metrics.
#[derive(Debug, Clone, Serialize)]
pub struct PriorSigmaRow {
    /// Noise std-dev applied to the oracle prior.
    pub sigma: f64,
    /// Every policy's metrics at this σ (baselines are σ-invariant but repeated here for a
    /// legible side-by-side table).
    pub policies: Vec<PolicyMetrics>,
}

impl PriorSigmaRow {
    /// Find a policy's metrics by name within this row.
    #[must_use]
    pub fn policy(&self, name: &str) -> Option<&PolicyMetrics> {
        self.policies.iter().find(|p| p.name == name)
    }
}

/// Pre-registered go/no-go: "firstpass+prior must have lower `$/success` than firstpass at
/// σ ≤ 0.2 with served-failure not higher; otherwise PRIOR=STOP."
#[derive(Debug, Clone, Serialize)]
pub struct PriorKillDecision {
    /// Every σ ≤ [`KILL_SIGMA_CEILING`] checked, in sweep order.
    pub sigmas_checked: Vec<f64>,
    /// Whether every checked σ satisfied both legs of the criterion.
    pub proceed: bool,
    /// Human-readable rationale, naming the first failing σ if any.
    pub rationale: String,
}

/// Full σ-sweep report.
#[derive(Debug, Clone, Serialize)]
pub struct PriorSweepReport {
    /// Always true — see module doc.
    pub simulated: bool,
    /// One row per swept σ.
    pub rows: Vec<PriorSigmaRow>,
    /// The pre-registered verdict.
    pub kill: PriorKillDecision,
}

/// Run every policy (baselines plus the two prior-fused arms) at every swept σ over the same
/// suite/ladder/backend/gate/prices every other policy in the report is scored on.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn run_sweep(
    suite: &[Task],
    ladder: &[Rung],
    backend: &dyn ModelBackend,
    gate: &dyn Gate,
    prices: &PriceTable,
    seed: u64,
    alpha: f64,
    predictor_noise: f64,
    budget_usd: Option<f64>,
) -> PriorSweepReport {
    let rows = sweep_rows(
        suite,
        ladder,
        backend,
        gate,
        prices,
        seed,
        alpha,
        predictor_noise,
        budget_usd,
    );
    let kill = kill_criterion(&rows);
    PriorSweepReport {
        simulated: true,
        rows,
        kill,
    }
}

/// The per-σ row-building loop shared by [`run_sweep`] (pre-registered) and
/// [`run_adverse_selection_scenario`] (exploratory) — same policies, same σ grid, different
/// suite/backend.
#[allow(clippy::too_many_arguments)]
fn sweep_rows(
    suite: &[Task],
    ladder: &[Rung],
    backend: &dyn ModelBackend,
    gate: &dyn Gate,
    prices: &PriceTable,
    seed: u64,
    alpha: f64,
    predictor_noise: f64,
    budget_usd: Option<f64>,
) -> Vec<PriorSigmaRow> {
    let mut rows = Vec::with_capacity(SIGMA_SWEEP.len());
    for &sigma in &SIGMA_SWEEP {
        let policies: Vec<Box<dyn Policy>> = vec![
            Box::new(AlwaysCheap),
            Box::new(AlwaysTop),
            Box::new(RandomRung { seed }),
            Box::new(PredictiveRouter {
                seed,
                noise: predictor_noise,
            }),
            Box::new(Firstpass { budget_usd }),
            Box::new(PredictivePrior { seed, sigma }),
            Box::new(FirstpassPrior {
                seed,
                sigma,
                budget_usd,
            }),
        ];
        let mut policy_metrics = Vec::with_capacity(policies.len());
        for pol in &policies {
            let decisions: Vec<Decision> = suite
                .iter()
                .map(|t| pol.decide(t, ladder, backend, gate, prices))
                .collect();
            policy_metrics.push(evaluate(pol.name(), &decisions, seed, alpha));
        }
        rows.push(PriorSigmaRow {
            sigma,
            policies: policy_metrics,
        });
    }
    rows
}

/// One task in this many is pushed into the hard tail of
/// [`adverse_selection_suite`] — a documented, modest 10%.
const HARD_TAIL_EVERY: u64 = 10;

/// Difficulty added to hard-tail tasks. [`SimBackend::clearance`](crate::sim::SimBackend::clearance)
/// is `(strength − 0.45·difficulty).clamp(0.02, 0.99)`; adding 2.0 drives `strength − 0.45·(d+2.0)`
/// negative (hence clamped to the 0.02 floor) for every rung strength up to 0.9 — i.e. every
/// realistic rung-0 model, not just this suite's. Matches the floor-inducing fixture already used
/// by `prior.rs`'s own test (`difficulty = 3.0`), not a new number invented for this scenario.
const HARD_TAIL_DIFFICULTY_OFFSET: f64 = 2.0;

/// Measured adverse-selection multiplier from `costaware.rs`'s 114-task MBPP measurement: tasks
/// that escalate pay $0.01881 for the cheap rung, vs $0.00869 for tasks whose gate passed there —
/// chosen from that measurement, before this scenario's numbers were read, not fit to produce one.
pub const ADVERSE_SELECTION_TOKEN_MULTIPLIER: f64 = 0.01881 / 0.00869;

/// Build the exploratory adverse-selection workload: the same base tasks
/// [`task_suite`](crate::sim::task_suite)`(n, seed)` produces, with every
/// [`HARD_TAIL_EVERY`]th task pushed into a hard tail — its difficulty raised by
/// [`HARD_TAIL_DIFFICULTY_OFFSET`] (past rung 0's clearance floor) and its prompt tokens scaled by
/// [`ADVERSE_SELECTION_TOKEN_MULTIPLIER`] (completion tokens are scaled the same way by
/// [`AdverseSelectionBackend`], since [`Task`] carries no output-token field of its own). See the
/// module doc for why these two changes, and why they are not pre-registered.
#[must_use]
fn adverse_selection_suite(n: usize, seed: u64) -> Vec<Task> {
    let mut suite = task_suite(n, seed);
    for t in &mut suite {
        if t.id % HARD_TAIL_EVERY == 0 {
            t.difficulty += HARD_TAIL_DIFFICULTY_OFFSET;
            t.prompt_tokens = (t.prompt_tokens as f64 * ADVERSE_SELECTION_TOKEN_MULTIPLIER) as u64;
        }
    }
    suite
}

/// Wraps a [`ModelBackend`] and scales completion tokens by
/// [`ADVERSE_SELECTION_TOKEN_MULTIPLIER`] for hard-tail tasks (identified the same way
/// [`adverse_selection_suite`] built them: `difficulty >= HARD_TAIL_DIFFICULTY_OFFSET`) — the
/// simulator otherwise sizes completions purely off rung strength
/// ([`SimBackend::run`](crate::sim::SimBackend)), giving output length no channel to depend on task
/// difficulty at all.
struct AdverseSelectionBackend<'a> {
    inner: &'a dyn ModelBackend,
}

impl ModelBackend for AdverseSelectionBackend<'_> {
    fn run(&self, task: &Task, rung: &Rung) -> Completion {
        let mut c = self.inner.run(task, rung);
        if task.difficulty >= HARD_TAIL_DIFFICULTY_OFFSET {
            c.out_tokens = (c.out_tokens as f64 * ADVERSE_SELECTION_TOKEN_MULTIPLIER) as u64;
        }
        c
    }
}

/// Run the σ-sweep against the exploratory adverse-selection workload (module doc). Same policies,
/// σ grid, ladder, gate, prices, and seed as [`run_sweep`]; only the suite and the completion-token
/// accounting differ. The returned report's `kill` field is the same criterion computed for
/// legibility — it is **not** a gate; see [`render_adverse_selection`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn run_adverse_selection_scenario(
    n_tasks: usize,
    ladder: &[Rung],
    backend: &dyn ModelBackend,
    gate: &dyn Gate,
    prices: &PriceTable,
    seed: u64,
    alpha: f64,
    predictor_noise: f64,
    budget_usd: Option<f64>,
) -> PriorSweepReport {
    let suite = adverse_selection_suite(n_tasks, seed);
    let wrapped = AdverseSelectionBackend { inner: backend };
    let rows = sweep_rows(
        &suite,
        ladder,
        &wrapped,
        gate,
        prices,
        seed,
        alpha,
        predictor_noise,
        budget_usd,
    );
    let kill = kill_criterion(&rows);
    PriorSweepReport {
        simulated: true,
        rows,
        kill,
    }
}

fn kill_criterion(rows: &[PriorSigmaRow]) -> PriorKillDecision {
    let mut sigmas_checked = Vec::new();
    let mut proceed = true;
    let mut rationale = String::new();
    for row in rows.iter().filter(|r| r.sigma <= KILL_SIGMA_CEILING) {
        sigmas_checked.push(row.sigma);
        let Some(fp) = row.policy("firstpass") else {
            proceed = false;
            rationale = "missing firstpass arm — cannot evaluate the criterion".to_owned();
            break;
        };
        let Some(fpp) = row.policy("firstpass+prior") else {
            proceed = false;
            rationale = "missing firstpass+prior arm — cannot evaluate the criterion".to_owned();
            break;
        };
        let cheaper = fpp.cost_per_success.point < fp.cost_per_success.point;
        let not_worse_failure = fpp.served_failure_rate <= fp.served_failure_rate;
        if !(cheaper && not_worse_failure) {
            proceed = false;
            rationale = format!(
                "at σ={:.2}: firstpass+prior ${:.4}/success vs firstpass ${:.4}/success \
                 (cheaper={cheaper}); served-failure {:.3} vs {:.3} (not-higher={not_worse_failure}) \
                 — criterion fails",
                row.sigma,
                fpp.cost_per_success.point,
                fp.cost_per_success.point,
                fpp.served_failure_rate,
                fp.served_failure_rate,
            );
            break;
        }
    }
    if proceed {
        rationale = format!(
            "firstpass+prior beat firstpass on $/success with served-failure not higher at every \
             σ ≤ {KILL_SIGMA_CEILING:.2} ({sigmas_checked:?})"
        );
    }
    PriorKillDecision {
        sigmas_checked,
        proceed,
        rationale,
    }
}

/// Render as Markdown.
#[must_use]
pub fn render(r: &PriorSweepReport) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n## Noisy decision-model prior — σ sweep (**SIMULATION**)\n"
    );
    let _ = writeln!(
        s,
        "Does a noisy pre-generation prior (a Jev-style decision model that predicts which tier \
         can handle a query) help once it only ever picks the *start* rung, versus a router that \
         serves that tier's output unverified? `predictive-prior` never gates (the Jev-router \
         stand-in); `firstpass+prior` starts from the identical prior but the gate decides what \
         ships — a failed rung is never served.\n"
    );
    for row in &r.rows {
        let _ = writeln!(s, "### σ = {:.2}\n", row.sigma);
        let _ = writeln!(
            s,
            "| policy | success | $/success | mean $ | served-fail | escal |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|");
        for p in &row.policies {
            let _ = writeln!(
                s,
                "| {} | {:.3} [{:.3}, {:.3}] | {:.4} [{:.4}, {:.4}] | {:.4} | {:.3} | {:.2} |",
                p.name,
                p.success.point,
                p.success.lo,
                p.success.hi,
                p.cost_per_success.point,
                p.cost_per_success.lo,
                p.cost_per_success.hi,
                p.mean_cost_usd,
                p.served_failure_rate,
                p.escalation_rate,
            );
        }
        s.push('\n');
    }
    let _ = writeln!(s, "### Kill criterion (pre-registered)\n");
    let _ = writeln!(
        s,
        "firstpass+prior must have lower $/success than firstpass at σ ≤ {KILL_SIGMA_CEILING:.2} \
         with served-failure not higher; otherwise PRIOR=STOP.\n"
    );
    let _ = writeln!(
        s,
        "**Decision: PRIOR={}** — {}",
        if r.kill.proceed { "PROCEED" } else { "STOP" },
        r.kill.rationale
    );
    s
}

/// The five arms this exploratory table shows, in display order — a subset of [`sweep_rows`]'s
/// full policy list (drops `random`/`predictive`, which the pre-registered table already covers).
const ADVERSE_SELECTION_TABLE_POLICIES: [&str; 5] = [
    "firstpass",
    "predictive-prior",
    "firstpass+prior",
    "always-cheap",
    "always-top",
];

/// Render the exploratory adverse-selection scenario as Markdown. Clearly separate from
/// [`render`]: a different heading, its own table, and an explicit statement that its
/// kill-criterion-shaped row is informational and cannot override the pre-registered verdict
/// above.
#[must_use]
pub fn render_adverse_selection(r: &PriorSweepReport) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n## Exploratory (post-hoc, NOT pre-registered): adverse-selection workload\n"
    );
    let _ = writeln!(
        s,
        "Same tasks and seed as the σ sweep above, but every {HARD_TAIL_EVERY}th task is pushed \
         into a hard tail: difficulty raised by {HARD_TAIL_DIFFICULTY_OFFSET:.1} (past rung 0's \
         clearance floor) and its prompt/completion tokens scaled \
         {ADVERSE_SELECTION_TOKEN_MULTIPLIER:.3}× — the multiplier `costaware.rs` measured on real \
         MBPP escalations ($0.01881 vs $0.00869), not invented for this scenario. This did not \
         inform, and cannot override, the pre-registered kill criterion above.\n"
    );
    for row in &r.rows {
        let _ = writeln!(s, "### σ = {:.2}\n", row.sigma);
        let _ = writeln!(
            s,
            "| policy | success | $/success | mean $ | served-fail | escal |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|");
        for name in ADVERSE_SELECTION_TABLE_POLICIES {
            let Some(p) = row.policy(name) else {
                continue;
            };
            let _ = writeln!(
                s,
                "| {} | {:.3} [{:.3}, {:.3}] | {:.4} [{:.4}, {:.4}] | {:.4} | {:.3} | {:.2} |",
                p.name,
                p.success.point,
                p.success.lo,
                p.success.hi,
                p.cost_per_success.point,
                p.cost_per_success.lo,
                p.cost_per_success.hi,
                p.mean_cost_usd,
                p.served_failure_rate,
                p.escalation_rate,
            );
        }
        s.push('\n');
    }
    let _ = writeln!(
        s,
        "**Informational reading only (NOT a gate):** firstpass+prior vs firstpass by the same \
         rule as the pre-registered criterion — {}. This cannot promote PRIOR to PROCEED nor \
         demote it to STOP; the decision of record is the pre-registered verdict above.",
        if r.kill.proceed {
            "cheaper per success with served-failure not higher"
        } else {
            "did not clear that bar"
        }
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::{argmin_expected_cost, noisy_prior};
    use crate::sim::{SimBackend, SimGate, task_suite};

    fn ladder() -> Vec<Rung> {
        vec![
            Rung::new("anthropic/claude-haiku-4-5", 0.62),
            Rung::new("anthropic/claude-sonnet-5", 0.80),
            Rung::new("anthropic/claude-opus-4-8", 0.93),
        ]
    }

    #[test]
    fn sweep_covers_every_sigma_and_arm() {
        let suite = task_suite(200, 7);
        let ladder = ladder();
        let be = SimBackend::new(7);
        let gate = SimGate::new(7, 0.08, 0.10, 0.0);
        let prices = PriceTable::defaults();
        let r = run_sweep(&suite, &ladder, &be, &gate, &prices, 7, 0.05, 0.30, None);
        assert_eq!(r.rows.len(), SIGMA_SWEEP.len());
        for row in &r.rows {
            assert!(row.policy("predictive-prior").is_some());
            assert!(row.policy("firstpass+prior").is_some());
            assert!(row.policy("firstpass").is_some());
        }
        assert!(!r.kill.sigmas_checked.is_empty());
    }

    #[test]
    fn kill_criterion_fires_stop_when_prior_arm_is_worse() {
        // A row where firstpass+prior costs strictly MORE per success than firstpass must STOP.
        let make = |name: &'static str, cost_per_success: f64, served_failure_rate: f64| {
            let mut m = evaluate(name, &[], 1, 0.05);
            m.name = name.to_owned();
            m.cost_per_success.point = cost_per_success;
            m.served_failure_rate = served_failure_rate;
            m
        };
        let rows = vec![PriorSigmaRow {
            sigma: 0.0,
            policies: vec![
                make("firstpass", 0.01, 0.02),
                make("firstpass+prior", 0.02, 0.02), // more expensive -> must fail
            ],
        }];
        let kill = kill_criterion(&rows);
        assert!(!kill.proceed, "a worse prior arm must yield PRIOR=STOP");
    }

    #[test]
    fn kill_criterion_proceeds_when_prior_arm_wins_at_every_checked_sigma() {
        let make = |name: &'static str, cost_per_success: f64, served_failure_rate: f64| {
            let mut m = evaluate(name, &[], 1, 0.05);
            m.name = name.to_owned();
            m.cost_per_success.point = cost_per_success;
            m.served_failure_rate = served_failure_rate;
            m
        };
        let rows: Vec<PriorSigmaRow> = [0.0, 0.1, 0.2, 0.4]
            .into_iter()
            .map(|sigma| PriorSigmaRow {
                sigma,
                policies: vec![
                    make("firstpass", 0.02, 0.02),
                    make("firstpass+prior", 0.01, 0.02),
                ],
            })
            .collect();
        let kill = kill_criterion(&rows);
        assert!(kill.proceed, "a strictly better prior arm should PROCEED");
        // Only sigmas <= 0.2 are checked, per the pre-registered ceiling.
        assert_eq!(kill.sigmas_checked, vec![0.0, 0.1, 0.2]);
    }

    #[test]
    fn adverse_selection_suite_only_scales_the_hard_tail() {
        let base = task_suite(50, 7);
        let scaled = adverse_selection_suite(50, 7);
        for (b, s) in base.iter().zip(scaled.iter()) {
            if b.id % HARD_TAIL_EVERY == 0 {
                assert!((s.difficulty - (b.difficulty + HARD_TAIL_DIFFICULTY_OFFSET)).abs() < 1e-9);
                assert_eq!(
                    s.prompt_tokens,
                    (b.prompt_tokens as f64 * ADVERSE_SELECTION_TOKEN_MULTIPLIER) as u64
                );
            } else {
                assert!((s.difficulty - b.difficulty).abs() < 1e-9);
                assert_eq!(s.prompt_tokens, b.prompt_tokens);
            }
        }
    }

    #[test]
    fn adverse_selection_scenario_at_sigma_zero_skips_rung_zero_for_a_hopeless_task() {
        // Proves the mechanism can act on this workload: at least one hard-tail task's noisy prior
        // (unperturbed at sigma=0, i.e. the true clearance) makes the expected-cost rule skip the
        // cheap rung entirely.
        let suite = adverse_selection_suite(200, 7);
        let ladder = ladder();
        let prices = PriceTable::defaults();
        let skipped = suite
            .iter()
            .filter(|t| t.difficulty >= HARD_TAIL_DIFFICULTY_OFFSET)
            .any(|t| {
                let prior = noisy_prior(t, &ladder, 0.0, 7);
                argmin_expected_cost(t, &ladder, &prices, &prior) > 0
            });
        assert!(
            skipped,
            "expected at least one hard-tail task to skip rung 0 at sigma=0"
        );
    }

    #[test]
    fn adverse_selection_scenario_is_deterministic() {
        let ladder = ladder();
        let be = SimBackend::new(7);
        let gate = SimGate::new(7, 0.08, 0.10, 0.0);
        let prices = PriceTable::defaults();
        let run = || {
            run_adverse_selection_scenario(200, &ladder, &be, &gate, &prices, 7, 0.05, 0.30, None)
        };
        let a = serde_json::to_string(&run()).expect("json");
        let b = serde_json::to_string(&run()).expect("json");
        assert_eq!(a, b, "same seed must produce byte-identical scenario JSON");
    }

    #[test]
    fn adverse_selection_scenario_covers_every_sigma_and_the_five_table_arms() {
        let ladder = ladder();
        let be = SimBackend::new(7);
        let gate = SimGate::new(7, 0.08, 0.10, 0.0);
        let prices = PriceTable::defaults();
        let r =
            run_adverse_selection_scenario(200, &ladder, &be, &gate, &prices, 7, 0.05, 0.30, None);
        assert_eq!(r.rows.len(), SIGMA_SWEEP.len());
        for row in &r.rows {
            for name in ADVERSE_SELECTION_TABLE_POLICIES {
                assert!(row.policy(name).is_some(), "missing arm {name} at σ");
            }
        }
        // Rendering must not panic and must carry the "exploratory" framing.
        let md = render_adverse_selection(&r);
        assert!(md.contains("Exploratory (post-hoc, NOT pre-registered)"));
        assert!(md.contains("cannot override"));
    }
}
