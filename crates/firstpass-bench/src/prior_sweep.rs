//! σ-sweep of the noisy-prior policies (**SIMULATION**): does a noisy pre-generation
//! decision-model prior — like TypeSafe's Jev, which predicts which model tier can handle a query
//! — improve Firstpass when its prediction is fused with the gate, instead of served unverified?
//!
//! This only exists in the deterministic simulation: the noisy prior is built from a sim-only
//! ground truth ([`SimBackend::clearance`](crate::sim::SimBackend::clearance)), which a live task
//! has no equivalent of.

use crate::metrics::{PolicyMetrics, evaluate};
use crate::policy::{
    AlwaysCheap, AlwaysTop, Decision, Firstpass, FirstpassPrior, Policy, PredictivePrior,
    PredictiveRouter, RandomRung,
};
use crate::sim::{Gate, ModelBackend, Rung, Task};
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
