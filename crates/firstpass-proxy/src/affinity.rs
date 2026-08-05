//! Session promotion (SPEC §8.4): start a continuing session on the rung it already needed.
//!
//! `[escalation.session_promotion]` has been parseable — and exported from `firstpass_core` — since
//! before this module existed, but nothing ever read it. A config that parses and does nothing is
//! the same class of defect as a receipt that records `$0.00`: the file states one thing and the
//! system does another, with no error to notice. This makes it real.
//!
//! ## What it buys
//!
//! The escalation tax, measured at 18% of first-pass spend on MBPP: when turn 1 of a conversation
//! escalates haiku → sonnet, turn 2 pays for the doomed haiku call all over again, and so does
//! turn 3. A session that has demonstrated it needs the stronger rung should start there.
//!
//! ## What it must not become
//!
//! A ratchet. Promotion on its own is one-way — a session that escalates once would stay expensive
//! for its whole life, including the trivial turns at the end ("thanks, that worked"). So every
//! `probe_every` turns the promotion is deliberately ignored and the request starts one rung lower.
//! If that cheaper rung passes its gate, the promotion is released. That probe is the only way a
//! promotion ever comes down, which is why it is not optional.
//!
//! ## What it cannot do
//!
//! Lower quality. Like the bandit, this chooses only where the ladder **starts**; the gate still
//! verifies whatever is served, and escalation still runs from wherever it started. A wrong
//! promotion costs money, never correctness — and the probe bounds even that.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use firstpass_core::config::SessionPromotion;

/// Per-session state. Small and `Copy`-cheap; there is one of these per live session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pin {
    /// Rung this session is promoted to, once `failures` has crossed the threshold.
    rung: u32,
    /// Gate failures observed inside the window.
    failures: u32,
    /// Turns served since the last downward probe.
    since_probe: u32,
    /// Last touch, for both TTL expiry and eviction order.
    seen: Instant,
}

/// What the router should do for this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Start at rung 0, as if promotion were off.
    Cold,
    /// Start at this rung — the session has earned the promotion.
    Promoted(u32),
    /// Start one rung below the promotion, to test whether it is still needed.
    Probe(u32),
}

impl Decision {
    /// The rung to actually start on.
    #[must_use]
    pub const fn start_rung(self) -> u32 {
        match self {
            Self::Cold => 0,
            Self::Promoted(r) | Self::Probe(r) => r,
        }
    }

    /// Label for the audit trace, so a receipt says *why* it did not start at rung 0.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Promoted(_) => "session-promoted",
            Self::Probe(_) => "session-probe",
        }
    }
}

/// Decide from a pin alone. Pure — every rule below is testable without a clock or a map.
fn decide(pin: Option<&Pin>, cfg: &SessionPromotion) -> Decision {
    let Some(pin) = pin else {
        return Decision::Cold;
    };
    if pin.failures < cfg.after_failures || pin.rung == 0 {
        return Decision::Cold;
    }
    // Time to check whether the promotion is still earned. `probe_every` of 0 would divide by
    // zero and also means "never probe", which is the ratchet this must not have — treat it as 1.
    if pin.since_probe >= cfg.probe_every.max(1) {
        return Decision::Probe(pin.rung.saturating_sub(1));
    }
    Decision::Promoted(pin.rung)
}

/// Tracks which sessions have earned a higher starting rung.
///
/// Keyed by `(tenant, session)`: a session id is only unique inside a tenant, and a promotion must
/// never cross that boundary — one tenant's traffic pattern is not a fact about another's.
#[derive(Debug)]
pub struct SessionPromoter {
    cfg: SessionPromotion,
    ttl: Duration,
    // ponytail: one mutex over the whole map. Each critical section is a hash lookup and a few
    // integer writes, so contention is far below what a router's provider I/O costs. Shard by
    // session hash if this ever shows up in a profile.
    pins: Mutex<HashMap<(String, String), Pin>>,
}

impl SessionPromoter {
    /// Build from config. `window` doubles as the idle TTL.
    ///
    /// # Errors
    /// [`firstpass_core::Error::BadDuration`] when `window` is not `<int><unit>`.
    pub fn new(cfg: SessionPromotion) -> firstpass_core::Result<Self> {
        let ttl = cfg.window_duration()?;
        Ok(Self {
            cfg,
            ttl,
            pins: Mutex::new(HashMap::new()),
        })
    }

    /// Where this request should start.
    #[must_use]
    pub fn decide(&self, tenant: &str, session: &str, now: Instant) -> Decision {
        let key = (tenant.to_owned(), session.to_owned());
        let Ok(pins) = self.pins.lock() else {
            // A poisoned lock must not take routing down; falling back to cold is always safe,
            // because cold is exactly the behaviour with promotion disabled.
            return Decision::Cold;
        };
        match pins.get(&key) {
            // Expired pins are treated as absent here and reaped on the next write, so a read
            // never has to take a write lock.
            Some(p) if now.duration_since(p.seen) < self.ttl => decide(Some(p), &self.cfg),
            _ => Decision::Cold,
        }
    }

    /// Record how a request ended: which rung actually served it, and whether the ladder had to
    /// climb to get there.
    pub fn record(
        &self,
        tenant: &str,
        session: &str,
        served_rung: u32,
        escalated: bool,
        now: Instant,
    ) {
        let key = (tenant.to_owned(), session.to_owned());
        let Ok(mut pins) = self.pins.lock() else {
            return;
        };
        self.evict_if_needed(&mut pins, now);

        let expired = pins
            .get(&key)
            .is_some_and(|p| now.duration_since(p.seen) >= self.ttl);
        if expired {
            pins.remove(&key);
        }

        let entry = pins.entry(key).or_insert(Pin {
            rung: 0,
            failures: 0,
            since_probe: 0,
            seen: now,
        });
        entry.seen = now;

        if escalated {
            // The session needed a higher rung than it started on: that is the evidence promotion
            // is for. Promote to where it actually landed.
            entry.failures = entry.failures.saturating_add(1);
            entry.rung = entry.rung.max(served_rung);
            entry.since_probe = 0;
        } else if served_rung < entry.rung {
            // A probe (or any request) served below the promotion without escalating — the
            // promotion is no longer earned. Release it to what actually worked. This is the path
            // that stops promotion being one-way.
            entry.rung = served_rung;
            entry.failures = 0;
            entry.since_probe = 0;
        } else {
            entry.since_probe = entry.since_probe.saturating_add(1);
        }
    }

    /// Drop expired entries, then oldest-first until under the cap.
    fn evict_if_needed(&self, pins: &mut HashMap<(String, String), Pin>, now: Instant) {
        if pins.len() < self.cfg.max_sessions {
            return;
        }
        pins.retain(|_, p| now.duration_since(p.seen) < self.ttl);
        // ponytail: full sort only on the rare over-cap write, not per request. A heap would be
        // asymptotically better and materially more code for a path that runs approximately never.
        if pins.len() >= self.cfg.max_sessions {
            let mut by_age: Vec<((String, String), Instant)> =
                pins.iter().map(|(k, p)| (k.clone(), p.seen)).collect();
            by_age.sort_by_key(|(_, seen)| *seen);
            for (k, _) in by_age.into_iter().take(self.cfg.max_sessions.max(2) / 2) {
                pins.remove(&k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SessionPromotion {
        SessionPromotion {
            after_failures: 1,
            window: "30m".to_owned(),
            probe_every: 3,
            max_sessions: 100,
        }
    }

    fn promoter() -> SessionPromoter {
        SessionPromoter::new(cfg()).expect("30m parses")
    }

    #[test]
    fn an_unknown_session_starts_cold() {
        let p = promoter();
        assert_eq!(p.decide("t", "never-seen", Instant::now()), Decision::Cold);
    }

    #[test]
    fn a_session_that_escalated_starts_where_it_landed() {
        // The escalation tax in one test: turn 1 climbs 0 → 1, so turn 2 must not pay for rung 0
        // again.
        let p = promoter();
        let now = Instant::now();
        p.record("t", "s", 1, true, now);
        assert_eq!(p.decide("t", "s", now), Decision::Promoted(1));
        assert_eq!(p.decide("t", "s", now).start_rung(), 1);
    }

    #[test]
    fn one_escalation_is_not_enough_when_the_threshold_is_higher() {
        let mut c = cfg();
        c.after_failures = 3;
        let p = SessionPromoter::new(c).expect("parses");
        let now = Instant::now();
        p.record("t", "s", 1, true, now);
        p.record("t", "s", 1, true, now);
        assert_eq!(p.decide("t", "s", now), Decision::Cold, "2 < 3");
        p.record("t", "s", 1, true, now);
        assert_eq!(p.decide("t", "s", now), Decision::Promoted(1));
    }

    #[test]
    fn promotion_is_not_a_ratchet_it_probes_downward() {
        // Without this the feature is a cost regression on any long conversation that gets easier.
        let p = promoter();
        let now = Instant::now();
        p.record("t", "s", 2, true, now);
        assert_eq!(p.decide("t", "s", now), Decision::Promoted(2));

        // Serve at the promoted rung without escalating, probe_every(3) times.
        for _ in 0..3 {
            p.record("t", "s", 2, false, now);
        }
        assert_eq!(
            p.decide("t", "s", now),
            Decision::Probe(1),
            "must periodically test one rung lower"
        );
    }

    #[test]
    fn a_successful_probe_releases_the_promotion() {
        let p = promoter();
        let now = Instant::now();
        p.record("t", "s", 2, true, now);
        for _ in 0..3 {
            p.record("t", "s", 2, false, now);
        }
        assert_eq!(p.decide("t", "s", now), Decision::Probe(1));

        // The probe served at rung 1 with no escalation — rung 2 is no longer earned.
        p.record("t", "s", 1, false, now);
        assert_eq!(
            p.decide("t", "s", now),
            Decision::Cold,
            "released to rung 1"
        );
    }

    #[test]
    fn a_failed_probe_restores_the_promotion() {
        let p = promoter();
        let now = Instant::now();
        p.record("t", "s", 2, true, now);
        for _ in 0..3 {
            p.record("t", "s", 2, false, now);
        }
        assert_eq!(p.decide("t", "s", now), Decision::Probe(1));

        // Probe started at 1, still had to climb to 2 → promotion re-earned.
        p.record("t", "s", 2, true, now);
        assert_eq!(p.decide("t", "s", now), Decision::Promoted(2));
    }

    #[test]
    fn a_promotion_never_crosses_a_tenant_boundary() {
        // One tenant's traffic pattern is not a fact about another's, and a session id is only
        // unique within a tenant.
        let p = promoter();
        let now = Instant::now();
        p.record("tenant-a", "shared-id", 2, true, now);
        assert_eq!(
            p.decide("tenant-a", "shared-id", now),
            Decision::Promoted(2)
        );
        assert_eq!(p.decide("tenant-b", "shared-id", now), Decision::Cold);
    }

    #[test]
    fn an_idle_session_forgets_its_promotion() {
        let mut c = cfg();
        c.window = "1s".to_owned();
        let p = SessionPromoter::new(c).expect("parses");
        let t0 = Instant::now();
        p.record("t", "s", 2, true, t0);
        assert_eq!(p.decide("t", "s", t0), Decision::Promoted(2));

        let later = t0 + Duration::from_secs(2);
        assert_eq!(
            p.decide("t", "s", later),
            Decision::Cold,
            "a reused session id must not inherit a stale rung"
        );
    }

    #[test]
    fn the_map_cannot_grow_without_bound() {
        let mut c = cfg();
        c.max_sessions = 10;
        let p = SessionPromoter::new(c).expect("parses");
        let now = Instant::now();
        for i in 0..100 {
            p.record("t", &format!("s{i}"), 1, true, now);
        }
        let len = p.pins.lock().expect("not poisoned").len();
        assert!(len <= 10, "expected <= 10 tracked sessions, got {len}");
    }

    #[test]
    fn probe_every_zero_is_rejected_at_parse_not_silently_coerced() {
        // A permanent pin is a cost regression an operator would only discover in a bill, so it
        // fails loudly at startup instead. `.max(1)` in `decide` is defence in depth for a config
        // built in code rather than parsed.
        let err = firstpass_core::config::Config::parse(
            r#"
            [escalation.session_promotion]
            after_failures = 2
            window = "30m"
            probe_every = 0

            [[route]]
            mode = "enforce"
            ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
            "#,
        )
        .expect_err("probe_every = 0 must not parse");
        let msg = err.to_string();
        assert!(msg.contains("probe_every"), "unhelpful message: {msg}");

        // Defence in depth: constructed directly, 0 behaves as 1 rather than dividing or pinning.
        let mut c = cfg();
        c.probe_every = 0;
        let pin = Pin {
            rung: 2,
            failures: 5,
            since_probe: 1,
            seen: Instant::now(),
        };
        assert_eq!(decide(Some(&pin), &c), Decision::Probe(1));
    }

    #[test]
    fn a_valid_promotion_block_parses_with_defaults() {
        let cfg = firstpass_core::config::Config::parse(
            r#"
            [escalation.session_promotion]
            after_failures = 2
            window = "30m"

            [[route]]
            mode = "enforce"
            ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
            "#,
        )
        .expect("valid promotion config parses");
        let p = cfg
            .escalation
            .session_promotion
            .expect("promotion block present");
        assert_eq!(p.after_failures, 2);
        assert_eq!(p.probe_every, 5, "default must not be a ratchet");
        assert_eq!(p.max_sessions, 10_000);
    }

    #[test]
    fn the_decision_carries_its_reason_into_the_receipt() {
        assert_eq!(Decision::Cold.reason(), "cold");
        assert_eq!(Decision::Promoted(2).reason(), "session-promoted");
        assert_eq!(Decision::Probe(1).reason(), "session-probe");
    }
}
