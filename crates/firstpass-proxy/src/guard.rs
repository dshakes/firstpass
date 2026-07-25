//! Guardrail state: outcome windows, demotion, and cooldown (ADR 0009 D3).
//!
//! [`firstpass_core::guardrail`] decides *whether* a window has breached. This holds the state
//! that decision runs against: a trailing window of resolved outcomes per (tenant, route), and
//! whether that route is currently demoted.
//!
//! Demotion is deliberately sticky. A guardrail that re-promotes the moment its window recovers
//! oscillates, and every oscillation is a visible behaviour change for real users — models
//! switching under them for reasons nobody can explain. So a demoted route stays demoted for a
//! cooldown, and re-promotion is a separate, conservative decision rather than an automatic
//! consequence of one good window.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Mutex;

use firstpass_core::guardrail::{Guardrail, GuardrailAction, GuardrailVerdict, evaluate};

/// What happened when an outcome was recorded.
#[derive(Debug, Clone, PartialEq)]
pub enum Reaction {
    /// Nothing to do.
    None,
    /// The route just breached and has been demoted to observe.
    Demoted(GuardrailVerdict),
    /// The route just breached and the operator asked only to be told.
    Alarmed(GuardrailVerdict),
}

#[derive(Debug, Default)]
struct RouteState {
    outcomes: VecDeque<bool>,
    /// Unix seconds until which this route stays demoted.
    demoted_until: Option<i64>,
}

/// Per-(tenant, route) guardrail state.
#[derive(Debug, Default)]
pub struct GuardrailRegistry {
    inner: Mutex<HashMap<(String, usize), RouteState>>,
}

impl GuardrailRegistry {
    /// A fresh registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether this route is currently demoted and must take the observe path.
    ///
    /// Fails **closed toward safety**: if the lock is poisoned we report "not demoted" rather than
    /// demoting everything, because a panic in bookkeeping should not silently switch a whole
    /// deployment's routing. The breach will be re-detected on the next resolved outcome.
    #[must_use]
    pub fn is_demoted(&self, tenant: &str, route_ix: usize, now_secs: i64) -> bool {
        self.inner.lock().is_ok_and(|g| {
            g.get(&(tenant.to_owned(), route_ix))
                .and_then(|s| s.demoted_until)
                .is_some_and(|until| now_secs < until)
        })
    }

    /// Record one resolved outcome and react if the window has breached.
    ///
    /// `was_correct` must come from a real downstream result (the feedback API), not from a gate
    /// verdict — otherwise the system grades its own homework.
    pub fn record(
        &self,
        tenant: &str,
        route_ix: usize,
        cfg: &Guardrail,
        was_correct: bool,
        now_secs: i64,
        cooldown_secs: i64,
    ) -> Reaction {
        let Ok(mut g) = self.inner.lock() else {
            return Reaction::None;
        };
        let st = g.entry((tenant.to_owned(), route_ix)).or_default();
        st.outcomes.push_back(was_correct);
        while st.outcomes.len() > cfg.window {
            st.outcomes.pop_front();
        }
        // Already demoted: keep accumulating evidence, but do not re-fire. Re-promotion is a
        // separate decision, not a side effect of the cooldown lapsing mid-window.
        if st.demoted_until.is_some_and(|u| now_secs < u) {
            return Reaction::None;
        }
        let window: Vec<bool> = st.outcomes.iter().copied().collect();
        let verdict = evaluate(cfg, &window);
        if !verdict.breached {
            return Reaction::None;
        }
        match cfg.action {
            GuardrailAction::Demote => {
                st.demoted_until = Some(now_secs + cooldown_secs);
                // Clear the window on demotion. Keeping it would leave the route re-breaching
                // instantly on the first outcome after cooldown, using evidence from a regime that
                // is no longer in effect.
                st.outcomes.clear();
                Reaction::Demoted(verdict)
            }
            GuardrailAction::Alarm => Reaction::Alarmed(verdict),
        }
    }

    /// Clear a demotion — the manual reset an operator reaches for after fixing the cause.
    pub fn reset(&self, tenant: &str, route_ix: usize) {
        if let Ok(mut g) = self.inner.lock()
            && let Some(st) = g.get_mut(&(tenant.to_owned(), route_ix))
        {
            st.demoted_until = None;
            st.outcomes.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(action: GuardrailAction) -> Guardrail {
        Guardrail {
            alpha: 0.10,
            delta: 0.05,
            window: 400,
            min_n: 200,
            action,
        }
    }

    fn feed(
        r: &GuardrailRegistry,
        c: &Guardrail,
        oks: impl Iterator<Item = bool>,
    ) -> Vec<Reaction> {
        oks.map(|ok| r.record("t", 0, c, ok, 1_000, 3_600))
            .filter(|x| !matches!(x, Reaction::None))
            .collect()
    }

    /// A healthy route is never demoted, and stays servable.
    #[test]
    fn healthy_traffic_is_never_demoted() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Demote);
        let fired = feed(&r, &c, std::iter::repeat_n(true, 500));
        assert!(fired.is_empty(), "healthy route fired: {fired:?}");
        assert!(!r.is_demoted("t", 0, 1_000));
    }

    /// A genuinely failing route is demoted — and demotion is what makes the failure mode
    /// "stops saving money" rather than "keeps serving bad answers".
    #[test]
    fn sustained_failure_demotes_the_route() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Demote);
        let fired = feed(&r, &c, (0..600).map(|i| i % 10 >= 3));
        assert!(
            matches!(fired.first(), Some(Reaction::Demoted(_))),
            "failing route was not demoted: {fired:?}"
        );
        assert!(r.is_demoted("t", 0, 1_000), "route not marked demoted");
    }

    /// Under `alarm` the route must keep serving — an operator who asked to be told must not have
    /// their traffic rerouted behind their back.
    #[test]
    fn alarm_action_reports_without_changing_routing() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Alarm);
        let fired = feed(&r, &c, (0..600).map(|i| i % 10 >= 3));
        assert!(
            matches!(fired.first(), Some(Reaction::Alarmed(_))),
            "alarm action did not alarm: {fired:?}"
        );
        assert!(
            !r.is_demoted("t", 0, 1_000),
            "alarm action rerouted traffic — it must only report"
        );
    }

    /// Demotion must be sticky for the cooldown. Flapping is a visible behaviour change for users
    /// every time it happens.
    #[test]
    fn demotion_is_sticky_for_the_cooldown_then_lapses() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Demote);
        for i in 0..600 {
            r.record("t", 0, &c, i % 10 >= 3, 1_000, 3_600);
        }
        assert!(r.is_demoted("t", 0, 1_000), "not demoted at t=1000");
        assert!(
            r.is_demoted("t", 0, 4_500),
            "lapsed before the cooldown expired"
        );
        assert!(
            !r.is_demoted("t", 0, 4_601),
            "still demoted after the cooldown"
        );
    }

    /// A demoted route must not re-fire on every subsequent outcome — one breach is one event.
    #[test]
    fn a_demoted_route_does_not_re_fire() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Demote);
        let fired = feed(&r, &c, (0..600).map(|i| i % 10 >= 3));
        assert_eq!(fired.len(), 1, "breach fired more than once: {fired:?}");
    }

    /// One tenant's failures must not demote another's route.
    #[test]
    fn demotion_is_scoped_per_tenant_and_route() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Demote);
        for i in 0..600 {
            r.record("noisy", 0, &c, i % 10 >= 3, 1_000, 3_600);
        }
        assert!(r.is_demoted("noisy", 0, 1_000));
        assert!(
            !r.is_demoted("quiet", 0, 1_000),
            "another tenant was demoted"
        );
        assert!(
            !r.is_demoted("noisy", 1, 1_000),
            "another route was demoted"
        );
    }

    /// The manual reset is what an operator uses after fixing the cause.
    #[test]
    fn reset_clears_a_demotion() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Demote);
        for i in 0..600 {
            r.record("t", 0, &c, i % 10 >= 3, 1_000, 3_600);
        }
        assert!(r.is_demoted("t", 0, 1_000));
        r.reset("t", 0);
        assert!(
            !r.is_demoted("t", 0, 1_000),
            "reset did not clear the demotion"
        );
    }

    /// The reason route attribution exists: a failing route must not hide behind a healthy
    /// sibling. Before outcomes carried their route, every tenant's feedback pooled onto route 0,
    /// so a config with one bad route and one good one could stay green while the bad one
    /// degraded — the guard sitting quiet exactly when it was needed.
    #[test]
    fn a_failing_route_is_demoted_without_dragging_down_a_healthy_sibling() {
        let r = GuardrailRegistry::new();
        let c = cfg(GuardrailAction::Demote);
        // Route 0 fails hard; route 1 is clean. Interleaved, as real traffic would arrive.
        for i in 0..600 {
            r.record("t", 0, &c, i % 10 >= 3, 1_000, 3_600);
            r.record("t", 1, &c, true, 1_000, 3_600);
        }
        assert!(
            r.is_demoted("t", 0, 1_000),
            "the failing route was not demoted"
        );
        assert!(
            !r.is_demoted("t", 1, 1_000),
            "a healthy route was demoted by its sibling's failures — attribution is not working"
        );
    }

    /// The mirror, and the concrete reason this fix matters: pooled onto one bucket, a healthy
    /// sibling's successes can dilute a failing route below the target so the guardrail never
    /// fires.
    ///
    /// The arithmetic, spelled out because the effect depends on it (delta=0.05, alpha=0.10).
    /// Note the guardrail evaluates on EVERY outcome, so what matters is the bound at `min_n`,
    /// where the slack is widest — not at the full window:
    ///   min_n=800 -> slack 0.043
    ///   attributed: rate 0.10 + 0.043 = 0.143  -> breaches
    ///   pooled:     rate 0.05 + 0.043 = 0.093  -> does NOT breach
    ///
    /// Two earlier attempts at this test were wrong and both taught something. A 30% rate halves
    /// to 15% and still breaches — dilution only masks when it pulls the bound under alpha. And
    /// min_n=400 gives slack 0.061, so even the diluted 5% breached at the moment judging began:
    /// staying under needs n > ln(1/delta) / (2*(alpha-rate)^2), which is 600 here.
    #[test]
    fn a_healthy_sibling_can_mask_a_failing_route_when_outcomes_are_pooled() {
        let wide = Guardrail {
            alpha: 0.10,
            delta: 0.05,
            window: 4000,
            min_n: 800,
            action: GuardrailAction::Demote,
        };
        let pooled = GuardrailRegistry::new();
        let attributed = GuardrailRegistry::new();
        for i in 0..2000 {
            let failing = i % 10 != 0; // 10% failures on the bad route
            // Pre-fix behaviour: both streams land on route 0.
            pooled.record("t", 0, &wide, failing, 1_000, 3_600);
            pooled.record("t", 0, &wide, true, 1_000, 3_600);
            // With attribution: each stream to the route that produced it.
            attributed.record("t", 0, &wide, failing, 1_000, 3_600);
            attributed.record("t", 1, &wide, true, 1_000, 3_600);
        }
        assert!(
            attributed.is_demoted("t", 0, 1_000),
            "attributed: the failing route must be caught"
        );
        assert!(
            !attributed.is_demoted("t", 1, 1_000),
            "attributed: the healthy sibling must be left alone"
        );
        assert!(
            !pooled.is_demoted("t", 0, 1_000),
            "pooling was expected to mask this failure — that masking is exactly what route \
             attribution fixes, so if this fires the demonstration needs new numbers"
        );
    }
}
