//! Assembling and rendering the proof report — honestly, including where Firstpass loses and a
//! pre-registered kill criterion (SPEC §10).

use crate::metrics::PolicyMetrics;
use crate::stats::Ci;
use firstpass_core::conformal::ConformalResult;
use serde::Serialize;
use std::fmt::Write as _;

/// The pre-registered go / no-go decision.
#[derive(Debug, Clone, Serialize)]
pub struct KillDecision {
    /// Break-even cheap-tier clearance `(c0 + g0) / c1` below which cheapest-first cannot pay off.
    pub break_even_p0: f64,
    /// Measured cheap-tier clearance.
    pub measured_p0: f64,
    /// Whether measured clearance beats break-even.
    pub clearance_above_break_even: bool,
    /// Whether Firstpass's `$/successful-task` beats always-top's.
    pub firstpass_cheaper_per_success: bool,
    /// Overall: proceed to build the product, or stop and publish the negative result.
    pub proceed: bool,
    /// Human-readable rationale.
    pub rationale: String,
}

/// The full proof report.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Whether the numbers are simulated (M0 harness) or from real providers.
    pub simulated: bool,
    /// Per-policy metrics.
    pub policies: Vec<PolicyMetrics>,
    /// Conformal serving guarantee.
    pub conformal: ConformalResult,
    /// Go / no-go.
    pub kill: KillDecision,
}

fn ci(c: Ci) -> String {
    format!("{:.3} [{:.3}, {:.3}]", c.point, c.lo, c.hi)
}

impl Report {
    /// Find a policy's metrics by name.
    #[must_use]
    pub fn policy(&self, name: &str) -> Option<&PolicyMetrics> {
        self.policies.iter().find(|p| p.name == name)
    }

    /// Serialize to pretty JSON.
    ///
    /// # Errors
    /// Returns [`serde_json::Error`] if serialization fails (it should not for this type).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render a human- and agent-readable Markdown report.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Firstpass — M0 proof report\n");
        if self.simulated {
            let _ = writeln!(
                s,
                "> **SIMULATION.** These numbers come from the deterministic `firstpass-bench` \
                 model, not real providers. They validate the *methodology*; swap in the \
                 reqwest backend + a real task suite for real proof.\n"
            );
        }

        // Metrics table.
        let _ = writeln!(
            s,
            "| policy | success | $/success | mean $ | served-fail | escal | regret | gate P/R | p95 ms |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|");
        for p in &self.policies {
            let pr = match (p.gate_precision, p.gate_recall) {
                (Some(pr), Some(re)) => format!("{pr:.2}/{re:.2}"),
                _ => "—".to_owned(),
            };
            let _ = writeln!(
                s,
                "| {} | {} | {} | {:.4} | {:.3} | {:.2} | {:.3} | {} | {:.0} |",
                p.name,
                ci(p.success),
                ci(p.cost_per_success),
                p.mean_cost_usd,
                p.served_failure_rate,
                p.escalation_rate,
                p.regret_per_request,
                pr,
                p.p95_latency_ms,
            );
        }

        // Conformal guarantee.
        let c = &self.conformal;
        let _ = writeln!(s, "\n## Conformal serving guarantee");
        if c.feasible {
            let _ = writeln!(
                s,
                "Serve iff gate score ≥ **{:.3}** ⇒ served-failure rate ≤ **{:.0}%** at {:.0}% \
                 confidence (calibration risk {:.1}%, serves {:.0}% of traffic at that bar).",
                c.threshold,
                c.alpha * 100.0,
                (1.0 - c.delta) * 100.0,
                c.calib_risk * 100.0,
                c.served_frac * 100.0,
            );
        } else {
            let _ = writeln!(
                s,
                "Target α={:.0}% infeasible on this gate — the honest output is *serve nothing* rather \
                 than a false guarantee. A better gate is required.",
                c.alpha * 100.0
            );
        }

        // Honest head-to-head.
        let _ = writeln!(
            s,
            "\n## Firstpass vs always-top (the money claim, honestly)"
        );
        if let (Some(fp), Some(top)) = (self.policy("firstpass"), self.policy("always-top")) {
            let savings = if top.cost_per_success.point > 0.0 {
                1.0 - fp.cost_per_success.point / top.cost_per_success.point
            } else {
                0.0
            };
            let _ = writeln!(
                s,
                "- **$/successful-task:** firstpass {:.4} vs always-top {:.4} → **{:.0}% cheaper** at proven quality.",
                fp.cost_per_success.point,
                top.cost_per_success.point,
                savings * 100.0
            );
            let _ = writeln!(
                s,
                "- **Success rate:** firstpass {:.3} vs always-top {:.3} (parity-or-better; firstpass gets \
                 multiple gated shots).",
                fp.success.point, top.success.point
            );
            let _ = writeln!(
                s,
                "- **Where firstpass loses:** latency in *enforce* mode — p50 {:.0}ms / p95 {:.0}ms vs \
                 always-cheap p95 {:.0}ms. This is the full-escalation worst case with a judge-latency gate. \
                 In **observe** mode (default) added latency is **0** — serve first, gate async. Each rung is \
                 **one** model call; a deterministic gate (tests/typecheck/schema) adds no LLM call, and in an \
                 agent loop those tests already run. Enforce is scoped to subagent/batch/CI, not the hot path.",
                fp.p50_latency_ms,
                fp.p95_latency_ms,
                self.policy("always-cheap")
                    .map_or(0.0, |c| c.p95_latency_ms)
            );
        }
        if let (Some(fp), Some(pred)) = (self.policy("firstpass"), self.policy("predictive")) {
            let _ = writeln!(
                s,
                "- **vs a predictive router:** served-failure {:.3} (firstpass) vs {:.3} (predictive) — \
                 verification catches what prediction serves blind.",
                fp.served_failure_rate, pred.served_failure_rate
            );
        }

        // Kill criterion.
        let k = &self.kill;
        let _ = writeln!(s, "\n## Kill criterion (pre-registered)");
        let _ = writeln!(
            s,
            "- cheap-tier clearance {:.3} vs break-even {:.3} → **{}**",
            k.measured_p0,
            k.break_even_p0,
            if k.clearance_above_break_even {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = writeln!(
            s,
            "- firstpass cheaper per success than always-top → **{}**",
            if k.firstpass_cheaper_per_success {
                "PASS"
            } else {
                "FAIL"
            }
        );
        let _ = writeln!(
            s,
            "\n**Decision: {}** — {}",
            if k.proceed { "PROCEED" } else { "STOP" },
            k.rationale
        );
        s
    }
}
