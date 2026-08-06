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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pin {
    /// Rung this session is promoted to, once `failures` has crossed the threshold.
    pub rung: u32,
    /// Gate failures observed inside the window.
    pub failures: u32,
    /// Turns served since the last downward probe.
    pub since_probe: u32,
    /// Last touch, for both TTL expiry and eviction order.
    ///
    /// Skipped on the wire and re-stamped on read: an `Instant` is meaningless in another process,
    /// and expiry in the shared store is Redis's own TTL rather than a comparison two replicas'
    /// monotonic clocks would disagree about.
    #[serde(skip, default = "Instant::now")]
    pub seen: Instant,
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

/// Where promotion state lives.
///
/// Same seam as the verified cache, for the same reason: in-process state fragments behind a load
/// balancer. A session that escalated on replica A starts cold on replica B, so the escalation tax
/// this feature exists to remove is paid again on roughly (N-1)/N of turns.
///
/// **Concurrent turns of one session race.** `record` is a read-modify-write, and two replicas
/// handling turns simultaneously can lose one increment — last write wins. That is deliberate
/// rather than overlooked: the cost is a promotion arriving one turn late, which is one extra
/// cheap-rung call. The gate still verifies whatever is served, so the race cannot affect
/// correctness, and a Lua script or WATCH loop to remove it would add real complexity to an
/// optimisation.
#[async_trait::async_trait]
pub trait PromotionStore: Send + Sync + std::fmt::Debug {
    /// Read a session's pin, if one is live.
    async fn load(&self, key: &str, now: Instant) -> Option<Pin>;
    /// Write a session's pin.
    async fn store(&self, key: &str, pin: Pin, now: Instant);
}

/// Key for a session's promotion. Tenant first, and not optional: a session id is only unique
/// inside a tenant, and one tenant's traffic pattern is not a fact about another's.
#[must_use]
pub fn promotion_key(tenant: &str, session: &str) -> String {
    format!("{tenant}|{session}")
}

/// Tracks which sessions have earned a higher starting rung.
///
/// Keyed by `(tenant, session)`: a session id is only unique inside a tenant, and a promotion must
/// never cross that boundary — one tenant's traffic pattern is not a fact about another's.
#[derive(Debug)]
pub struct SessionPromoter {
    cfg: SessionPromotion,
    ttl: Duration,
    /// Shared backing, when configured. `None` uses `pins` below — the in-process map — so a
    /// single-instance deployment is byte-identical to before and pays for no network hop.
    shared: Option<std::sync::Arc<dyn PromotionStore>>,
    // ponytail: one mutex over the whole map. Each critical section is a hash lookup and a few
    // integer writes, so contention is far below what a router's provider I/O costs. Shard by
    // session hash if this ever shows up in a profile.
    pins: Mutex<HashMap<String, Pin>>,
}

#[async_trait::async_trait]
impl PromotionStore for SessionPromoter {
    // Delegates to the existing map rather than reimplementing it: the in-process path is the one
    // already covered by 12 tests, and a parallel async copy would be free to drift on the probe
    // and ratchet rules that keep a promotion from becoming permanent.
    async fn load(&self, key: &str, now: Instant) -> Option<Pin> {
        let Ok(pins) = self.pins.lock() else {
            return None;
        };
        pins.get(key)
            .filter(|p| now.duration_since(p.seen) < self.ttl)
            .copied()
    }

    async fn store(&self, key: &str, pin: Pin, now: Instant) {
        let Ok(mut pins) = self.pins.lock() else {
            return;
        };
        if pins.len() >= self.cfg.max_sessions {
            pins.retain(|_, p| now.duration_since(p.seen) < self.ttl);
            if pins.len() >= self.cfg.max_sessions {
                let mut by_age: Vec<(String, Instant)> =
                    pins.iter().map(|(k, p)| (k.clone(), p.seen)).collect();
                by_age.sort_by_key(|(_, t)| *t);
                for (k, _) in by_age.into_iter().take(self.cfg.max_sessions.max(2) / 2) {
                    pins.remove(&k);
                }
            }
        }
        pins.insert(key.to_owned(), pin);
    }
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
            shared: None,
            pins: Mutex::new(HashMap::new()),
        })
    }

    /// Back this promoter with a shared store, so a session keeps its rung across replicas.
    #[must_use]
    pub fn with_shared(mut self, store: std::sync::Arc<dyn PromotionStore>) -> Self {
        self.shared = Some(store);
        self
    }

    /// Read a pin from wherever this promoter's state lives.
    async fn pin_for(&self, key: &str, now: Instant) -> Option<Pin> {
        match self.shared.as_ref() {
            Some(s) => s.load(key, now).await,
            None => PromotionStore::load(self, key, now).await,
        }
    }

    /// Write a pin to wherever this promoter's state lives.
    async fn put_pin(&self, key: &str, pin: Pin, now: Instant) {
        match self.shared.as_ref() {
            Some(s) => s.store(key, pin, now).await,
            None => PromotionStore::store(self, key, pin, now).await,
        }
    }

    /// Where this request should start.
    pub async fn decide_async(&self, tenant: &str, session: &str, now: Instant) -> Decision {
        let key = promotion_key(tenant, session);
        decide(self.pin_for(&key, now).await.as_ref(), &self.cfg)
    }

    /// Record how a request ended, against wherever this promoter's state lives.
    ///
    /// Read-modify-write, so two replicas handling concurrent turns of one session can lose an
    /// increment — last write wins. The cost is a promotion arriving one turn late, which is one
    /// extra cheap-rung call; the gate still verifies whatever is served, so it cannot affect
    /// correctness.
    pub async fn record_async(
        &self,
        tenant: &str,
        session: &str,
        served_rung: u32,
        escalated: bool,
        now: Instant,
    ) {
        let key = promotion_key(tenant, session);
        let mut pin = self.pin_for(&key, now).await.unwrap_or(Pin {
            rung: 0,
            failures: 0,
            since_probe: 0,
            seen: now,
        });
        pin.seen = now;
        if escalated {
            pin.failures = pin.failures.saturating_add(1);
            pin.rung = pin.rung.max(served_rung);
            pin.since_probe = 0;
        } else if served_rung < pin.rung {
            pin.rung = served_rung;
            pin.failures = 0;
            pin.since_probe = 0;
        } else {
            pin.since_probe = pin.since_probe.saturating_add(1);
        }
        self.put_pin(&key, pin, now).await;
    }

    /// Where this request should start.
    #[must_use]
    pub fn decide(&self, tenant: &str, session: &str, now: Instant) -> Decision {
        let key = promotion_key(tenant, session);
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
        let key = promotion_key(tenant, session);
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
    fn evict_if_needed(&self, pins: &mut HashMap<String, Pin>, now: Instant) {
        if pins.len() < self.cfg.max_sessions {
            return;
        }
        pins.retain(|_, p| now.duration_since(p.seen) < self.ttl);
        // ponytail: full sort only on the rare over-cap write, not per request. A heap would be
        // asymptotically better and materially more code for a path that runs approximately never.
        if pins.len() >= self.cfg.max_sessions {
            let mut by_age: Vec<(String, Instant)> =
                pins.iter().map(|(k, p)| (k.clone(), p.seen)).collect();
            by_age.sort_by_key(|(_, seen)| *seen);
            for (k, _) in by_age.into_iter().take(self.cfg.max_sessions.max(2) / 2) {
                pins.remove(&k);
            }
        }
    }
}

/// A Redis-backed promotion store, so a session keeps its rung across replicas.
///
/// Without it, promotion fragments behind a load balancer: a session that escalated on replica A
/// starts cold on replica B, so the escalation tax this feature exists to remove is paid again on
/// roughly (N-1)/N of turns — the feature is configured, reports nothing wrong, and does a fraction
/// of its job.
///
/// One key per session, `fp:sp:<tenant>|<session>`, with a Redis TTL equal to the configured
/// `window`. Expiry is the server's job for the same reason it is in the verified cache: two
/// replicas comparing monotonic clocks would disagree about what is stale.
///
/// `max_sessions` does not apply here — a Redis-backed store is bounded by its TTL and the
/// server's own eviction policy, and imposing a per-replica cap on a shared store would have each
/// replica evicting the others' sessions.
#[cfg(feature = "redis-cache")]
pub struct RedisPromotionStore {
    client: redis::aio::ConnectionManager,
    ttl_secs: u64,
}

#[cfg(feature = "redis-cache")]
impl std::fmt::Debug for RedisPromotionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Same reasoning as the cache's store: never print a URL that may carry credentials.
        f.debug_struct("RedisPromotionStore")
            .field("ttl_secs", &self.ttl_secs)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "redis-cache")]
impl RedisPromotionStore {
    /// Connect to `url`, proving the connection with a PING.
    ///
    /// # Errors
    /// A malformed URL, or a server that does not answer within the timeout. `ConnectionManager`
    /// connects lazily, so without the PING this would report success against a dead server and
    /// the proxy would run with promotion silently fragmented — configured, and not working.
    pub async fn connect(url: &str, ttl_secs: u64) -> Result<Self, String> {
        let safe = crate::verified_cache::redact_redis_url(url);
        let client = redis::Client::open(url).map_err(|e| format!("redis url {safe}: {e}"))?;
        let mgr = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            redis::aio::ConnectionManager::new(client),
        )
        .await
        .map_err(|_| format!("redis at {safe} did not answer within 5s"))?
        .map_err(|e| format!("redis connect {safe}: {e}"))?;

        let mut probe = mgr.clone();
        let pong: Result<String, _> = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            redis::cmd("PING").query_async(&mut probe),
        )
        .await
        .map_err(|_| {
            format!(
                "redis at {safe} did not answer PING within 5s. Refusing to start rather than run \
                 with session promotion silently fragmented across replicas."
            )
        })?;
        pong.map_err(|e| format!("redis at {safe} rejected PING: {e}"))?;

        Ok(Self {
            client: mgr,
            ttl_secs,
        })
    }
}

#[cfg(feature = "redis-cache")]
#[async_trait::async_trait]
impl PromotionStore for RedisPromotionStore {
    async fn load(&self, key: &str, _now: Instant) -> Option<Pin> {
        let mut c = self.client.clone();
        let raw: Option<String> = redis::cmd("GET")
            .arg(format!("fp:sp:{key}"))
            .query_async(&mut c)
            .await
            .ok()?;
        // A malformed value is a miss, not an error: the session simply starts cold, which is
        // exactly the behaviour with promotion off. Failing a request over it would turn an
        // optimisation into an outage.
        serde_json::from_str(&raw?).ok()
    }

    async fn store(&self, key: &str, pin: Pin, _now: Instant) {
        let Ok(body) = serde_json::to_string(&pin) else {
            return;
        };
        let mut c = self.client.clone();
        let _: Result<(), _> = redis::cmd("SETEX")
            .arg(format!("fp:sp:{key}"))
            .arg(self.ttl_secs)
            .arg(body)
            .query_async::<()>(&mut c)
            .await;
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
            redis_url: None,
        }
    }

    fn promoter() -> SessionPromoter {
        SessionPromoter::new(cfg()).expect("30m parses")
    }

    #[tokio::test]
    async fn the_async_path_matches_the_sync_one() {
        // Both APIs must agree, or a deployment gets different routing depending on which one the
        // request path happens to call — a difference no test of either alone would surface.
        let p = promoter();
        let now = Instant::now();
        p.record_async("t", "s", 1, true, now).await;

        assert_eq!(p.decide_async("t", "s", now).await, Decision::Promoted(1));
        assert_eq!(p.decide("t", "s", now), Decision::Promoted(1));
    }

    #[tokio::test]
    async fn a_pin_survives_serde_without_its_local_clock() {
        // Redis carries pins between processes, where an Instant is meaningless. The routing state
        // — rung, failures, probe counter — is what must cross; `seen` is re-stamped, and expiry
        // there is Redis's TTL rather than two machines comparing monotonic clocks.
        let pin = Pin {
            rung: 2,
            failures: 3,
            since_probe: 1,
            seen: Instant::now(),
        };
        let wire = serde_json::to_string(&pin).expect("serializes");
        assert!(
            !wire.contains("seen"),
            "a local Instant must not go on the wire: {wire}"
        );

        let back: Pin = serde_json::from_str(&wire).expect("deserializes");
        assert_eq!(back.rung, 2);
        assert_eq!(back.failures, 3);
        assert_eq!(back.since_probe, 1);
    }

    /// The shared store, against a real Redis.
    ///
    /// Requires `FIRSTPASS_TEST_REDIS`. **Fails without it rather than skipping** — a suite that
    /// passes quietly when its subject is unavailable reports green having tested nothing.
    #[cfg(feature = "redis-cache")]
    #[tokio::test]
    async fn promotion_survives_across_replicas() {
        let url = std::env::var("FIRSTPASS_TEST_REDIS").unwrap_or_else(|_| {
            panic!("set FIRSTPASS_TEST_REDIS (e.g. redis://127.0.0.1:6379/15) to run this test")
        });
        let store = std::sync::Arc::new(
            RedisPromotionStore::connect(&url, 300)
                .await
                .expect("connect"),
        );
        let session = format!("sess-{}", uuid::Uuid::now_v7());
        let now = Instant::now();

        // Replica A sees the escalation.
        let a = SessionPromoter::new(cfg())
            .expect("cfg")
            .with_shared(store.clone());
        a.record_async("t", &session, 2, true, now).await;

        // Replica B — a different process in production — must see the promotion.
        let b = SessionPromoter::new(cfg())
            .expect("cfg")
            .with_shared(store.clone());
        assert_eq!(
            b.decide_async("t", &session, now).await,
            Decision::Promoted(2),
            "a session that escalated on one replica must not start cold on the next"
        );

        // And the tenant boundary still holds through the shared store.
        assert_eq!(
            b.decide_async("other-tenant", &session, now).await,
            Decision::Cold,
            "promotion must not cross tenants through the shared store"
        );
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
