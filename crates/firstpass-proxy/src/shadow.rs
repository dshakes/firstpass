//! Shadow-spend ledger (ADR 0009 D2).
//!
//! Shadow scoring makes **real model calls** to answer "what would Firstpass have served, and what
//! would it have cost?" for traffic that is only being observed. That is the number an operator
//! needs before flipping to enforce, and it is not free.
//!
//! So the ceiling is not advisory. This ledger is checked before every shadow evaluation and
//! debited after it, and when a day's budget is exhausted shadow work stops and *says so* — a
//! measurement that silently degrades is worse than one that never ran, because the operator
//! keeps trusting a projection that has quietly stopped tracking their traffic.
//!
//! Scoped per (tenant, route) so one tenant's shadow spend cannot consume another's budget, which
//! matters the moment this runs anywhere multi-tenant (ADR 0004).

use std::collections::HashMap;
use std::sync::Mutex;

/// UTC day number, used as the reset boundary. A calendar day is the unit an operator budgets in.
fn day_of(ts: jiff::Timestamp) -> i64 {
    ts.as_second().div_euclid(86_400)
}

/// Why a shadow evaluation did not run. Recorded rather than swallowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// This request was not in the configured sample.
    NotSampled,
    /// The day's shadow budget is spent.
    BudgetExhausted,
}

/// Per-(tenant, route) shadow spend for the current UTC day.
#[derive(Debug, Default)]
pub struct ShadowLedger {
    inner: Mutex<HashMap<(String, usize), (i64, f64)>>, // key -> (day, spent_usd)
}

impl ShadowLedger {
    /// A fresh ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a shadow evaluation may start, given the day's ceiling.
    ///
    /// Checked *before* the call rather than after, because the point of a ceiling is to not spend
    /// the money — discovering the overrun afterwards defeats it. The trade is that a single
    /// evaluation can overshoot by its own cost; with per-call costs in fractions of a cent
    /// against a dollar-scale daily cap that is immaterial, and erring toward one extra cheap call
    /// is better than erring toward silently skipping measurements.
    pub fn may_spend(
        &self,
        tenant: &str,
        route_ix: usize,
        max_usd_per_day: f64,
        now: jiff::Timestamp,
    ) -> bool {
        if max_usd_per_day <= 0.0 {
            return false;
        }
        let today = day_of(now);
        let mut g = match self.inner.lock() {
            Ok(g) => g,
            // A poisoned lock must not take shadow down with it, and must not be treated as
            // "budget available" either — refusing to spend is the safe direction.
            Err(_) => return false,
        };
        let e = g
            .entry((tenant.to_owned(), route_ix))
            .or_insert((today, 0.0));
        if e.0 != today {
            *e = (today, 0.0); // new UTC day, fresh budget
        }
        e.1 < max_usd_per_day
    }

    /// Record what a completed shadow evaluation cost.
    pub fn debit(&self, tenant: &str, route_ix: usize, usd: f64, now: jiff::Timestamp) {
        if !usd.is_finite() || usd <= 0.0 {
            return;
        }
        let today = day_of(now);
        if let Ok(mut g) = self.inner.lock() {
            let e = g
                .entry((tenant.to_owned(), route_ix))
                .or_insert((today, 0.0));
            if e.0 != today {
                *e = (today, 0.0);
            }
            e.1 += usd;
        }
    }

    /// Spend so far today, for reporting.
    #[must_use]
    pub fn spent_today(&self, tenant: &str, route_ix: usize, now: jiff::Timestamp) -> f64 {
        let today = day_of(now);
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.get(&(tenant.to_owned(), route_ix)).copied())
            .filter(|(d, _)| *d == today)
            .map_or(0.0, |(_, spent)| spent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(secs: i64) -> jiff::Timestamp {
        jiff::Timestamp::from_second(secs).unwrap()
    }

    #[test]
    fn spending_stops_at_the_ceiling() {
        let l = ShadowLedger::new();
        let now = ts(1_000_000);
        assert!(
            l.may_spend("t", 0, 1.00, now),
            "fresh budget must allow the first call"
        );
        l.debit("t", 0, 0.60, now);
        assert!(
            l.may_spend("t", 0, 1.00, now),
            "0.60 of 1.00 spent — still under"
        );
        l.debit("t", 0, 0.50, now);
        assert!(
            !l.may_spend("t", 0, 1.00, now),
            "1.10 of 1.00 spent — must refuse, or the ceiling is decoration"
        );
    }

    /// The budget is per UTC day; a new day must restore it, or shadow silently dies after the
    /// first busy day and the operator's projection stops tracking without any signal.
    #[test]
    fn budget_resets_on_a_new_utc_day() {
        let l = ShadowLedger::new();
        let day1 = ts(86_400 * 100);
        l.debit("t", 0, 5.0, day1);
        assert!(!l.may_spend("t", 0, 1.0, day1));
        let day2 = ts(86_400 * 101);
        assert!(
            l.may_spend("t", 0, 1.0, day2),
            "new UTC day must restore the budget"
        );
        assert!((l.spent_today("t", 0, day2) - 0.0).abs() < f64::EPSILON);
    }

    /// One tenant must not be able to exhaust another's shadow budget.
    #[test]
    fn budgets_are_isolated_per_tenant_and_route() {
        let l = ShadowLedger::new();
        let now = ts(1_000_000);
        l.debit("noisy", 0, 99.0, now);
        assert!(!l.may_spend("noisy", 0, 1.0, now));
        assert!(
            l.may_spend("quiet", 0, 1.0, now),
            "another tenant's budget was consumed"
        );
        assert!(
            l.may_spend("noisy", 1, 1.0, now),
            "another route's budget was consumed"
        );
    }

    /// A zero or negative ceiling means "never", not "unlimited" — the failure direction of a
    /// misconfigured budget must be to spend nothing.
    #[test]
    fn non_positive_ceiling_never_spends() {
        let l = ShadowLedger::new();
        let now = ts(1_000_000);
        assert!(!l.may_spend("t", 0, 0.0, now));
        assert!(!l.may_spend("t", 0, -5.0, now));
    }

    #[test]
    fn nonsense_debits_are_ignored() {
        let l = ShadowLedger::new();
        let now = ts(1_000_000);
        for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            l.debit("t", 0, bad, now);
        }
        assert!((l.spent_today("t", 0, now) - 0.0).abs() < f64::EPSILON);
        assert!(l.may_spend("t", 0, 1.0, now));
    }
}
