//! Routing policies under test — Firstpass and the baselines it is proven against.
//!
//! Every policy sees the same tasks, ladder, backend, gate, and price table. The point of the
//! harness is that a policy decides **without** reading ground-truth correctness; only the gate
//! (imperfectly) and the final metrics (with hindsight) get to see it.

use crate::sim::{Gate, ModelBackend, Rung, Task, hash01};
use firstpass_core::{PriceTable, Verdict};

/// One model call plus its gate judgement within a decision.
#[derive(Debug, Clone)]
pub struct Attempt {
    /// Ladder rung index.
    pub rung_idx: usize,
    /// Model called.
    pub model: String,
    /// Model-call cost.
    pub model_cost_usd: f64,
    /// Gate cost (0 if the policy didn't gate).
    pub gate_cost_usd: f64,
    /// Gate verdict (`None` if the policy didn't gate this attempt).
    pub verdict: Option<Verdict>,
    /// Ground-truth correctness (recorded for hindsight metrics, never read by the policy).
    pub correct: bool,
    /// Latency of this attempt (model + gate).
    pub latency_ms: u64,
}

/// The full record of how a policy handled one task.
#[derive(Debug, Clone)]
pub struct Decision {
    /// Attempts in the order made.
    pub attempts: Vec<Attempt>,
    /// Index into `attempts` of the one served (`None` only if nothing was produced).
    pub served: Option<usize>,
    /// Number of escalations taken (`attempts - 1`, floored at 0).
    pub escalations: u32,
}

impl Decision {
    /// Total USD spent (all model calls + all gates).
    #[must_use]
    pub fn total_cost(&self) -> f64 {
        self.attempts
            .iter()
            .map(|a| a.model_cost_usd + a.gate_cost_usd)
            .sum()
    }

    /// Gate-only USD.
    #[must_use]
    pub fn gate_cost(&self) -> f64 {
        self.attempts.iter().map(|a| a.gate_cost_usd).sum()
    }

    /// Total latency actually experienced (attempts are sequential).
    #[must_use]
    pub fn latency_ms(&self) -> u64 {
        self.attempts.iter().map(|a| a.latency_ms).sum()
    }

    /// Whether the served output was truly correct (false if nothing served).
    #[must_use]
    pub fn served_correct(&self) -> bool {
        self.served.is_some_and(|i| self.attempts[i].correct)
    }

    /// Needless escalations: a strictly-earlier attempt was actually correct but we escalated
    /// past it (money wasted because the gate false-rejected a good answer).
    #[must_use]
    pub fn needless_escalations(&self) -> u32 {
        match self.served {
            Some(s) => self.attempts[..s].iter().filter(|a| a.correct).count() as u32,
            None => 0,
        }
    }
}

/// A routing policy.
pub trait Policy {
    /// Short identifier for the report.
    fn name(&self) -> &'static str;
    /// Decide how to serve `task`.
    fn decide(
        &self,
        task: &Task,
        ladder: &[Rung],
        backend: &dyn ModelBackend,
        gate: &dyn Gate,
        prices: &PriceTable,
    ) -> Decision;
}

/// Run a single rung, gate it, and return the attempt (ground-truth cost via the price table).
fn run_attempt(
    task: &Task,
    rung_idx: usize,
    rung: &Rung,
    backend: &dyn ModelBackend,
    gate: Option<&dyn Gate>,
    prices: &PriceTable,
) -> Attempt {
    let c = backend.run(task, rung);
    // Unknown models fall back to zero cost rather than panicking — a benchmark must not crash on
    // a mis-specified ladder; the report surfaces a zero-cost model as an obvious anomaly instead.
    let model_cost_usd = prices
        .cost_usd(&rung.model, c.in_tokens, c.out_tokens)
        .unwrap_or(0.0);
    let (verdict, gate_cost_usd, gate_ms) = match gate {
        Some(g) => {
            let j = g.judge(task, rung, &c);
            (Some(j.verdict), j.cost_usd, j.ms)
        }
        None => (None, 0.0, 0),
    };
    Attempt {
        rung_idx,
        model: rung.model.clone(),
        model_cost_usd,
        gate_cost_usd,
        verdict,
        correct: c.correct,
        latency_ms: c.latency_ms + gate_ms,
    }
}

/// Serve only the cheapest rung, ungated. Cheapest possible, lowest quality — a lower bound.
#[derive(Debug, Clone, Copy)]
pub struct AlwaysCheap;
impl Policy for AlwaysCheap {
    fn name(&self) -> &'static str {
        "always-cheap"
    }
    fn decide(
        &self,
        task: &Task,
        ladder: &[Rung],
        be: &dyn ModelBackend,
        _g: &dyn Gate,
        p: &PriceTable,
    ) -> Decision {
        let a = run_attempt(task, 0, &ladder[0], be, None, p);
        Decision {
            attempts: vec![a],
            served: Some(0),
            escalations: 0,
        }
    }
}

/// Serve only the top rung, ungated. Highest quality per single shot, highest cost — the
/// counterfactual baseline Firstpass must beat on `$/successful-task`.
#[derive(Debug, Clone, Copy)]
pub struct AlwaysTop;
impl Policy for AlwaysTop {
    fn name(&self) -> &'static str {
        "always-top"
    }
    fn decide(
        &self,
        task: &Task,
        ladder: &[Rung],
        be: &dyn ModelBackend,
        _g: &dyn Gate,
        p: &PriceTable,
    ) -> Decision {
        let top = ladder.len() - 1;
        let a = run_attempt(task, top, &ladder[top], be, None, p);
        Decision {
            attempts: vec![a],
            served: Some(0),
            escalations: 0,
        }
    }
}

/// Pick a rung uniformly at random, ungated. A trivial control.
#[derive(Debug, Clone, Copy)]
pub struct RandomRung {
    /// Seed for reproducibility.
    pub seed: u64,
}
impl Policy for RandomRung {
    fn name(&self) -> &'static str {
        "random"
    }
    fn decide(
        &self,
        task: &Task,
        ladder: &[Rung],
        be: &dyn ModelBackend,
        _g: &dyn Gate,
        p: &PriceTable,
    ) -> Decision {
        let idx = (hash01(self.seed, task.id, 7) * ladder.len() as f64) as usize % ladder.len();
        let a = run_attempt(task, idx, &ladder[idx], be, None, p);
        Decision {
            attempts: vec![a],
            served: Some(0),
            escalations: 0,
        }
    }
}

/// A **predictive router** stand-in: estimate difficulty (noisily), pick the rung it predicts will
/// clear, and serve it — **without verifying the output**. This is the incumbent class Firstpass
/// competes with; its structural weakness is that a wrong prediction is served, undetected.
#[derive(Debug, Clone, Copy)]
pub struct PredictiveRouter {
    /// Seed for the prediction noise.
    pub seed: u64,
    /// Std-dev of the router's difficulty-estimation error.
    pub noise: f64,
}
impl Policy for PredictiveRouter {
    fn name(&self) -> &'static str {
        "predictive"
    }
    fn decide(
        &self,
        task: &Task,
        ladder: &[Rung],
        be: &dyn ModelBackend,
        _g: &dyn Gate,
        p: &PriceTable,
    ) -> Decision {
        // Noisy difficulty estimate in [0,1].
        let err = (hash01(self.seed, task.id, 11) - 0.5) * 2.0 * self.noise;
        let est = (task.difficulty + err).clamp(0.0, 1.0);
        // Map estimated difficulty to a rung: harder => higher rung.
        let idx = ((est * ladder.len() as f64) as usize).min(ladder.len() - 1);
        let a = run_attempt(task, idx, &ladder[idx], be, None, p);
        Decision {
            attempts: vec![a],
            served: Some(0),
            escalations: 0,
        }
    }
}

/// **Firstpass:** cheapest rung first, gate the output, escalate one rung only on gate failure,
/// serve the first output the gate passes. If the ladder (or budget) is exhausted without a pass,
/// serve the best attempt seen (here: the highest rung tried).
#[derive(Debug, Clone, Copy)]
pub struct Firstpass {
    /// Optional per-request USD cap; escalation stops once the next attempt would exceed it.
    pub budget_usd: Option<f64>,
}
impl Policy for Firstpass {
    fn name(&self) -> &'static str {
        "firstpass"
    }
    fn decide(
        &self,
        task: &Task,
        ladder: &[Rung],
        be: &dyn ModelBackend,
        gate: &dyn Gate,
        p: &PriceTable,
    ) -> Decision {
        let mut attempts = Vec::with_capacity(ladder.len());
        let mut spent = 0.0;
        let mut served = None;
        for (idx, rung) in ladder.iter().enumerate() {
            let a = run_attempt(task, idx, rung, be, Some(gate), p);
            spent += a.model_cost_usd + a.gate_cost_usd;
            let passed = a.verdict == Some(Verdict::Pass);
            attempts.push(a);
            if passed {
                served = Some(idx);
                break;
            }
            // Budget guard: stop escalating if the next attempt would blow the cap.
            if let Some(cap) = self.budget_usd
                && idx + 1 < ladder.len()
                && spent >= cap
            {
                break;
            }
        }
        // No pass: serve the best (highest) attempt we made.
        let served = served.or_else(|| attempts.len().checked_sub(1));
        let escalations = attempts.len().saturating_sub(1) as u32;
        Decision {
            attempts,
            served,
            escalations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{SimBackend, SimGate};

    fn ladder() -> Vec<Rung> {
        vec![
            Rung::new("anthropic/claude-haiku-4-5", 0.62),
            Rung::new("anthropic/claude-sonnet-5", 0.80),
            Rung::new("anthropic/claude-opus-4-8", 0.93),
        ]
    }

    #[test]
    fn always_cheap_uses_one_cheap_attempt() {
        let t = &crate::sim::task_suite(1, 1)[0];
        let d = AlwaysCheap.decide(
            t,
            &ladder(),
            &SimBackend::new(1),
            &SimGate::new(1, 0.0, 0.0, 0.0),
            &PriceTable::defaults(),
        );
        assert_eq!(d.attempts.len(), 1);
        assert_eq!(d.attempts[0].rung_idx, 0);
        assert_eq!(d.gate_cost(), 0.0);
    }

    #[test]
    fn firstpass_stops_at_first_pass_and_escalates_on_fail() {
        let ladder = ladder();
        let prices = PriceTable::defaults();
        let be = SimBackend::new(3);
        // A gate that always passes -> firstpass must serve rung 0, no escalation.
        let perfect_pass = SimGate::new(3, 1.0, 0.0, 0.0); // fpr=1 => everything passes
        let t = &crate::sim::task_suite(50, 3)[0];
        let d = Firstpass { budget_usd: None }.decide(t, &ladder, &be, &perfect_pass, &prices);
        assert_eq!(d.attempts.len(), 1);
        assert_eq!(d.served, Some(0));

        // A gate that always fails -> firstpass climbs the whole ladder, serves the top.
        let always_fail = SimGate::new(3, 0.0, 1.0, 0.0); // fnr=1 => everything fails
        let d2 = Firstpass { budget_usd: None }.decide(t, &ladder, &be, &always_fail, &prices);
        assert_eq!(d2.attempts.len(), 3);
        assert_eq!(d2.served, Some(2));
        assert_eq!(d2.escalations, 2);
    }

    #[test]
    fn firstpass_budget_caps_escalation() {
        let ladder = ladder();
        let always_fail = SimGate::new(3, 0.0, 1.0, 0.0);
        let t = &crate::sim::task_suite(10, 3)[0];
        // Tiny budget: after the first (cheap) attempt, spent already exceeds cap -> stop.
        let d = Firstpass {
            budget_usd: Some(0.0),
        }
        .decide(
            t,
            &ladder,
            &SimBackend::new(3),
            &always_fail,
            &PriceTable::defaults(),
        );
        assert!(
            d.attempts.len() < 3,
            "budget should cut escalation short, got {}",
            d.attempts.len()
        );
    }
}
