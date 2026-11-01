//! Live monitor for certified model selection (ADR 0014).
//!
//! A route with a `[route.challenger]` serves a randomized fraction of sessions from a second
//! ladder. This module owns the evidence: one [`Experiment`] per `(tenant, route)`, fed with the
//! cost of every served experiment request and the first task outcome of every session. Its two
//! jobs are the halt guard (the serving path asks [`ExperimentMonitor::status`] and stops using a
//! challenger that has been proven worse) and the report an operator reads before adopting one.
//!
//! ## What counts as an outcome
//!
//! Feedback with a `pass` or `fail` verdict counts, **whatever its `gate_id`**; `abstain` is
//! ignored, and only the first counted outcome per session is used. Nothing here can tell a CI
//! result from a human rating, so operators should post exactly one thing per session: the task
//! outcome the experiment is meant to compare. Posting several different signals under different
//! gate ids makes "first one wins" arbitrary.
//!
//! ## Configuration identity
//!
//! Every receipt carries a fingerprint of the experiment configuration (both ladders, margin,
//! alpha). Receipts whose fingerprint differs from the current config's are ignored and counted
//! (`ignored_stale`), so changing either ladder or the margin/alpha starts a fresh experiment
//! instead of pooling evidence about a different challenger.
//!
//! Tenants never share an experiment (ADR 0004): the key is `(tenant, route_ix)` and nothing here
//! reads across tenants.
//!
//! The same state is re-derivable from receipts. [`experiments_from_traces`] replays the stored
//! traces and deferred verdicts through the very same monitor, so a restart (or the
//! `firstpass experiment` CLI) lands on the decision the live process would have reached.
//!
//! ## Shared state across replicas (opt-in)
//!
//! When [`ExperimentMonitor`] is built with a [`ExperimentLogBackend`] (set via
//! [`ExperimentMonitor::with_log`]), every outcome (live mode) or pair (paired mode) is appended
//! to a shared log keyed by `(tenant, route_ix, fingerprint)`. Each replica re-reads the log on
//! every feedback event and on a periodic timer (≤ 5 s), rebuilding its local [`Experiment`] from
//! the canonical arrival order. Because the e-process is deterministic, all replicas holding the
//! same log reach the same halt/certify decision. Without a backend the behaviour is unchanged.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use firstpass_core::{
    Arm, Config, Experiment, ExperimentReport, ExperimentStatus, Trace, Verdict, select,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Upper bound on distinct sessions remembered per experiment, as 8-byte hashes (~8 MB per
/// million). Past it new sessions' outcomes are ignored rather than evicting old entries: evicting
/// would let a session count twice, which breaks the pairing argument, while ignoring only slows
/// the experiment down. A million sessions is far beyond what the test needs to decide.
const MAX_TRACKED_SESSIONS: usize = 1_000_000;

// ── Shared log (optional Redis / in-memory backend) ───────────────────────────────────────────

/// A record appended to the shared experiment log.
///
/// Live mode uses `Outcome`; paired mode uses `Pair`. The list for a given
/// `(tenant, route_ix, fingerprint)` contains only one kind (the mode is fixed per config).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LogRecord {
    /// One session's outcome from live mode.
    Outcome {
        arm: Arm,
        /// Gate verdict in [0, 1].
        outcome: f64,
        /// SHA-256 prefix of the session id (for in-log dedup on replay, not primary dedup).
        session_hash: u64,
        /// USD cost of this request.
        cost_usd: f64,
    },
    /// One paired shadow result.
    Pair {
        x_inc: f64,
        x_ch: f64,
        /// Cost of the incumbent request.
        incumbent_cost_usd: f64,
        /// Cost of the challenger off-path evaluation.
        challenger_cost_usd: f64,
        /// Whether the challenger call ended in an error with no output.
        /// `#[serde(default)]` so records written by older replicas deserialize to `false`.
        #[serde(default)]
        ch_error: bool,
        /// Number of challenger attempts cut off at the output-token limit.
        /// `#[serde(default)]` so records written by older replicas deserialize to `0`.
        #[serde(default)]
        ch_truncated: u64,
        /// Unix seconds when this pair was recorded. `0` for records written before this field
        /// was added (treated as "long ago" by `ShadowLedger::seed`, i.e., not debited today).
        #[serde(default)]
        ts_unix: i64,
    },
}

/// A shared log that lets several replicas converge on the same [`Experiment`] state.
///
/// Implementors: [`InMemoryExperimentLog`] (tests), and `RedisExperimentLog` (behind the
/// `redis-cache` feature).
///
/// The log for a given experiment is identified by `(tenant, route_ix, fingerprint)`. Multiple
/// replicas can append concurrently; the authoritative ordering is the list's arrival order (for
/// Redis: RPUSH index; for the in-memory implementation: insertion position). A halt decision
/// derived from the list at position `N` holds forever — re-reading the list can only add more
/// evidence, never retract it.
#[async_trait::async_trait]
pub trait ExperimentLogBackend: Send + Sync + fmt::Debug {
    /// Append `record` to the list. Returns the new list length (1-based).
    async fn append(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        record: LogRecord,
    ) -> u64;

    /// Return records starting at `offset` (0-based). Empty if the list has ≤ `offset` entries.
    async fn load_since(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        offset: u64,
    ) -> Vec<LogRecord>;

    /// Atomically claim a session for this experiment (live mode session dedup across replicas).
    ///
    /// Returns `true` if the session was newly claimed (this replica should count the outcome),
    /// `false` if another replica already claimed it. Claimants are permanent: once claimed the
    /// result is final regardless of which replica appended it.
    async fn claim_session(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        session_id: &str,
    ) -> bool;
}

/// In-process log for tests (and the no-Redis single-replica path, where it is never used).
///
/// Two `ExperimentMonitor` instances that hold `Arc`s to the same `InMemoryExperimentLog` will
/// converge to the same `Experiment` state after syncing, exactly as two remote replicas sharing
/// a Redis-backed log would.
#[derive(Debug, Default)]
pub struct InMemoryExperimentLog {
    lists: Mutex<HashMap<(String, usize, String), Vec<LogRecord>>>,
    sessions: Mutex<HashSet<(String, usize, String, String)>>,
}

impl InMemoryExperimentLog {
    /// A fresh empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl ExperimentLogBackend for InMemoryExperimentLog {
    async fn append(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        record: LogRecord,
    ) -> u64 {
        let key = (tenant.to_owned(), route_ix, fingerprint.to_owned());
        let mut lists = self.lists.lock().unwrap_or_else(PoisonError::into_inner);
        let list = lists.entry(key).or_default();
        list.push(record);
        list.len() as u64
    }

    async fn load_since(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        offset: u64,
    ) -> Vec<LogRecord> {
        let key = (tenant.to_owned(), route_ix, fingerprint.to_owned());
        let lists = self.lists.lock().unwrap_or_else(PoisonError::into_inner);
        match lists.get(&key) {
            None => vec![],
            Some(list) => {
                let start = offset as usize;
                if start >= list.len() {
                    vec![]
                } else {
                    list[start..].to_vec()
                }
            }
        }
    }

    async fn claim_session(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        session_id: &str,
    ) -> bool {
        let key = (
            tenant.to_owned(),
            route_ix,
            fingerprint.to_owned(),
            session_id.to_owned(),
        );
        let mut sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        sessions.insert(key)
    }
}

// ── Redis backend (behind `redis-cache` feature) ─────────────────────────────────────────────

#[cfg(feature = "redis-cache")]
pub use redis_log::RedisExperimentLog;

#[cfg(feature = "redis-cache")]
mod redis_log {
    use super::{ExperimentLogBackend, LogRecord};
    use std::fmt;

    // Use tokio::sync::Mutex so the guard is Send and can cross await points.
    use tokio::sync::Mutex;

    const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    /// Strips credentials from a Redis URL so it can appear in log lines safely.
    fn redact(url: &str) -> String {
        let Some((scheme, rest)) = url.split_once("://") else {
            return "<malformed redis url>".to_owned();
        };
        let host = match rest.rfind('@') {
            Some(i) => &rest[i + 1..],
            None => rest,
        };
        format!("{scheme}://{host}")
    }

    /// Redis-backed experiment log. Each experiment lives in a list at
    /// `firstpass:exp:{tenant}:{route_ix}:{fingerprint}` and sessions are claimed via
    /// `SET NX firstpass:exp:sess:{tenant}:{route_ix}:{fingerprint}:{session_id}`.
    pub struct RedisExperimentLog {
        /// `tokio::sync::Mutex` so the guard is `Send` and can be held across `.await`.
        conn: Mutex<redis::aio::ConnectionManager>,
    }

    impl fmt::Debug for RedisExperimentLog {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("RedisExperimentLog").finish_non_exhaustive()
        }
    }

    impl RedisExperimentLog {
        /// Connect to `url`, timing out after 5 s.
        ///
        /// # Errors
        /// A malformed URL, an unreachable server, or a timeout. Surfaced at startup so an operator
        /// who configured Redis is told immediately rather than discovering it via a missing halt.
        pub async fn connect(url: &str) -> Result<Self, String> {
            let safe = redact(url);
            let client = redis::Client::open(url).map_err(|e| format!("redis url {safe}: {e}"))?;
            let conn =
                tokio::time::timeout(CONNECT_TIMEOUT, redis::aio::ConnectionManager::new(client))
                    .await
                    .map_err(|_| format!("redis url {safe}: connect timed out"))?
                    .map_err(|e| format!("redis url {safe}: {e}"))?;
            Ok(Self {
                conn: Mutex::new(conn),
            })
        }

        fn list_key(tenant: &str, route_ix: usize, fingerprint: &str) -> String {
            format!("firstpass:exp:{tenant}:{route_ix}:{fingerprint}")
        }

        fn session_key(
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            session_id: &str,
        ) -> String {
            format!("firstpass:exp:sess:{tenant}:{route_ix}:{fingerprint}:{session_id}")
        }
    }

    #[async_trait::async_trait]
    impl ExperimentLogBackend for RedisExperimentLog {
        async fn append(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            record: LogRecord,
        ) -> u64 {
            let key = Self::list_key(tenant, route_ix, fingerprint);
            let Ok(json) = serde_json::to_string(&record) else {
                return 0;
            };
            let mut conn = self.conn.lock().await;
            use redis::AsyncCommands as _;
            match conn.rpush::<_, _, u64>(&key, json).await {
                Ok(n) => n,
                Err(e) => {
                    tracing::warn!(key, %e, "experiment log: redis RPUSH failed");
                    0
                }
            }
        }

        async fn load_since(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            offset: u64,
        ) -> Vec<LogRecord> {
            let key = Self::list_key(tenant, route_ix, fingerprint);
            let mut conn = self.conn.lock().await;
            use redis::AsyncCommands as _;
            let raw: Vec<String> = match conn.lrange(&key, offset as isize, -1).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(key, %e, "experiment log: redis LRANGE failed");
                    return vec![];
                }
            };
            raw.iter()
                .filter_map(|s| match serde_json::from_str(s) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        tracing::warn!(%e, "experiment log: skipping malformed log record");
                        None
                    }
                })
                .collect()
        }

        async fn claim_session(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            session_id: &str,
        ) -> bool {
            let key = Self::session_key(tenant, route_ix, fingerprint, session_id);
            let mut conn = self.conn.lock().await;
            use redis::AsyncCommands as _;
            // SET key "1" NX EX 86400 — expires after 1 day so the set doesn't grow unboundedly.
            match conn
                .set_options::<_, _, ()>(
                    &key,
                    "1",
                    redis::SetOptions::default()
                        .with_expiration(redis::SetExpiry::EX(86_400))
                        .conditional_set(redis::ExistenceCheck::NX),
                )
                .await
            {
                Ok(()) => true,  // NX succeeded: this replica claimed the session
                Err(_) => false, // key already exists: another replica claimed it
            }
        }
    }
}

/// One experiment plus the sessions that already contributed an outcome.
#[derive(Debug)]
struct Entry {
    experiment: Experiment,
    /// Receipts seen from a previous experiment configuration and therefore not counted.
    ignored_stale: u64,
    /// Errors and cut-off answers per arm, for the operator to judge what the pairs left out.
    health: Health,
    /// Hashes of sessions whose outcome was already counted. A 64-bit collision merely drops one
    /// legitimate outcome, which costs power and never validity.
    sessions: HashSet<u64>,
    saturated_logged: bool,
    /// How many records have been consumed from the shared log so far (incremental LRANGE).
    remote_log_offset: u64,
}

fn session_hash(session_id: &str) -> u64 {
    // SHA-256, not `DefaultHasher`: std reserves the right to change its algorithm between
    // releases, and the dedup set rebuilt from receipts after an upgrade must agree with the one
    // the live path built before it.
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(session_id.as_bytes());
    u64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}

/// Gauge value for [`ExperimentStatus`].
fn status_code(s: ExperimentStatus) -> f64 {
    match s {
        ExperimentStatus::Undecided => 0.0,
        ExperimentStatus::Certified => 1.0,
        ExperimentStatus::Halted => 2.0,
    }
}

#[derive(Debug)]
struct Params {
    margin: f64,
    alpha: f64,
    fingerprint: String,
}

/// Per-`(tenant, route)` certified-selection state. Cheap to share behind an `Arc`.
#[derive(Debug, Default)]
pub struct ExperimentMonitor {
    /// `(margin, alpha)` per route index that has a challenger.
    params: HashMap<usize, Params>,
    /// Set when the trace store could not be read at startup: the history needed to know whether a
    /// challenger was already halted is missing, so no challenger is served (fail closed).
    unavailable: bool,
    /// tenant -> route -> entry. Nested so a lookup by `&str` never allocates on the request path.
    entries: Mutex<HashMap<String, HashMap<usize, Entry>>>,
    /// Optional shared log that synchronises evidence across replicas (Redis or in-memory).
    log: Option<Arc<dyn ExperimentLogBackend>>,
    /// Optional paired-challenger spend ledger. When present, `apply_log_record` seeds cluster
    /// spend into it so the daily cap is cluster-wide rather than per-process.
    paired_ledger: Option<Arc<crate::shadow::ShadowLedger>>,
}

impl ExperimentMonitor {
    /// A monitor for every route in `config` that has a challenger. Routes without one are ignored
    /// by every method, so a deployment with no experiments pays nothing.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let params = config
            .routes
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                r.challenger.as_ref().map(|c| {
                    (
                        i,
                        Params {
                            margin: c.margin,
                            alpha: c.alpha,
                            fingerprint: c.fingerprint(&r.ladder),
                        },
                    )
                })
            })
            .collect();
        Self {
            params,
            unavailable: false,
            entries: Mutex::default(),
            log: None,
            paired_ledger: None,
        }
    }

    /// A monitor that refuses to serve any challenger. Used when the history could not be read.
    #[must_use]
    pub fn unavailable(config: &Config) -> Self {
        Self {
            unavailable: true,
            ..Self::from_config(config)
        }
    }

    /// Attach a shared log backend. Every `observe_outcome`/`observe_pair` call will append to
    /// the backend, and `sync_from_backend` will replay newly-arrived records from it. Returns
    /// `self` for method-chaining.
    #[must_use]
    pub fn with_log(mut self, log: Arc<dyn ExperimentLogBackend>) -> Self {
        self.log = Some(log);
        self
    }

    /// Attach a paired-challenger spend ledger. When set, `apply_log_record` seeds cluster spend
    /// into it so `may_spend` reflects all replicas' spend, not just this process's. Returns
    /// `self` for method-chaining. Only effective when a log backend is also attached.
    #[must_use]
    pub fn with_paired_ledger(mut self, ledger: Arc<crate::shadow::ShadowLedger>) -> Self {
        self.paired_ledger = Some(ledger);
        self
    }

    /// Whether a shared log backend is currently attached.
    #[must_use]
    pub fn has_log(&self) -> bool {
        self.log.is_some()
    }

    /// The fingerprint receipts for `route_ix` must carry to count, if it has a challenger.
    #[must_use]
    pub fn fingerprint(&self, route_ix: usize) -> Option<&str> {
        self.params.get(&route_ix).map(|p| p.fingerprint.as_str())
    }

    /// Whether the serving path must NOT use the challenger for `(tenant, route_ix)`: the
    /// experiment is halted, or its history was unreadable at startup.
    #[must_use]
    pub fn challenger_blocked(&self, tenant: &str, route_ix: usize) -> bool {
        self.unavailable || self.status(tenant, route_ix) == ExperimentStatus::Halted
    }

    /// Whether any route runs an experiment.
    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.params.is_empty()
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, HashMap<usize, Entry>>> {
        // A poisoned lock only means another thread panicked mid-update of plain counters; the
        // state is still usable, and the halt guard must not turn into a request-path panic.
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn entry<'a>(
        &self,
        map: &'a mut HashMap<String, HashMap<usize, Entry>>,
        tenant: &str,
        route_ix: usize,
    ) -> Option<&'a mut Entry> {
        let p = self.params.get(&route_ix)?;
        let (margin, alpha) = (p.margin, p.alpha);
        if !map.contains_key(tenant) {
            map.insert(tenant.to_owned(), HashMap::new());
        }
        let routes = map.get_mut(tenant)?;
        if let std::collections::hash_map::Entry::Vacant(v) = routes.entry(route_ix) {
            match Experiment::new(margin, alpha) {
                Ok(experiment) => {
                    v.insert(Entry {
                        experiment,
                        ignored_stale: 0,
                        health: Health::default(),
                        sessions: HashSet::new(),
                        saturated_logged: false,
                        remote_log_offset: 0,
                    });
                }
                Err(e) => {
                    // Unreachable for a parsed config; fail open (no experiment) rather than panic.
                    tracing::warn!(route = route_ix, %e, "experiment parameters rejected");
                    return None;
                }
            }
        }
        routes.get_mut(&route_ix)
    }

    /// Where the experiment for `(tenant, route_ix)` stands. [`ExperimentStatus::Undecided`] when
    /// nothing has been observed or the route has no challenger. The serving path stops using the
    /// challenger when this is [`ExperimentStatus::Halted`].
    #[must_use]
    pub fn status(&self, tenant: &str, route_ix: usize) -> ExperimentStatus {
        self.lock()
            .get(tenant)
            .and_then(|r| r.get(&route_ix))
            .map_or(ExperimentStatus::Undecided, |e| e.experiment.status())
    }

    /// Record the cost of one served request of `arm`. Call with `trace.final_.total_cost_usd`.
    ///
    /// `fingerprint` is the receipt's; a mismatch with the current config counts the receipt in
    /// `ignored_stale` and records nothing.
    pub fn observe_cost(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        arm: Arm,
        usd: f64,
    ) {
        let current = self.fingerprint(route_ix) == Some(fingerprint);
        let mut map = self.lock();
        if let Some(e) = self.entry(&mut map, tenant, route_ix) {
            if current {
                e.experiment.observe_cost(arm, usd);
            } else {
                e.ignored_stale += 1;
            }
        }
    }

    /// Tally one request of `arm` that errored and/or produced answers cut off at the output-token
    /// limit (`truncated` counts attempts). Stale fingerprints are ignored: `observe_cost` already
    /// counts the receipt once.
    pub fn observe_health(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        arm: Arm,
        errored: bool,
        truncated: u64,
    ) {
        if (!errored && truncated == 0) || self.fingerprint(route_ix) != Some(fingerprint) {
            return;
        }
        let mut map = self.lock();
        if let Some(e) = self.entry(&mut map, tenant, route_ix) {
            let h = match arm {
                Arm::Incumbent => &mut e.health.incumbent,
                Arm::Challenger => &mut e.health.challenger,
            };
            h.errors += u64::from(errored);
            h.truncated += truncated;
        }
    }

    /// Record a session's task outcome. Returns whether it was counted.
    ///
    /// Only the **first** clear outcome per session counts; later ones are ignored. The test pairs
    /// the k-th incumbent outcome with the k-th challenger outcome and treats sessions as
    /// independent draws, so several correlated outcomes from one long agent run would both
    /// overweight that session and break the independence the error bound rests on.
    ///
    /// `Pass` is 1.0, `Fail` is 0.0, `Abstain` carries no information and is ignored.
    pub fn observe_outcome(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        arm: Arm,
        session_id: &str,
        verdict: Verdict,
    ) -> bool {
        // A stale receipt is counted once, by `observe_cost`; here it is just not evidence.
        if self.fingerprint(route_ix) != Some(fingerprint) {
            return false;
        }
        let outcome = match verdict {
            Verdict::Pass => 1.0,
            Verdict::Fail => 0.0,
            Verdict::Abstain => return false,
        };
        let mut map = self.lock();
        let Some(e) = self.entry(&mut map, tenant, route_ix) else {
            return false;
        };
        let key = session_hash(session_id);
        if e.sessions.contains(&key) {
            return false;
        }
        if e.sessions.len() >= MAX_TRACKED_SESSIONS {
            if !e.saturated_logged {
                e.saturated_logged = true;
                tracing::warn!(
                    tenant,
                    route = route_ix,
                    "experiment session ledger is full; new sessions' outcomes are ignored"
                );
            }
            return false;
        }
        let before = e.experiment.status();
        if !e.experiment.observe_outcome(arm, outcome) {
            return false;
        }
        e.sessions.insert(key);
        let after = e.experiment.status();
        if before != after {
            match after {
                ExperimentStatus::Halted => tracing::error!(
                    tenant,
                    route = route_ix,
                    "EXPERIMENT HALTED: the challenger ladder is worse than the incumbent by more \
                     than the margin — it is no longer served"
                ),
                ExperimentStatus::Certified => tracing::info!(
                    tenant,
                    route = route_ix,
                    "experiment certified: the challenger is within the margin of the incumbent; \
                     adopting it is a human decision (`firstpass experiment`)"
                ),
                ExperimentStatus::Undecided => {}
            }
            Self::publish(&map, route_ix);
        }
        true
    }

    /// Export the route's status as `firstpass_experiment_status{route}` (0 undecided, 1 certified,
    /// 2 halted). The label is the route index — bounded by the config, never by traffic — and
    /// tenants are deliberately not a label (unbounded in a hosted deployment), so the value is the
    /// worst status across tenants: a halt anywhere is visible.
    fn publish(map: &HashMap<String, HashMap<usize, Entry>>, route_ix: usize) {
        let worst = map
            .values()
            .filter_map(|r| r.get(&route_ix))
            .map(|e| status_code(e.experiment.status()))
            .fold(0.0, f64::max);
        metrics::gauge!("firstpass_experiment_status", "route" => route_ix.to_string()).set(worst);
    }

    /// A snapshot of every experiment, ordered by `(tenant, route_ix)`.
    #[must_use]
    pub fn reports(&self) -> Vec<Snapshot> {
        let map = self.lock();
        let mut out: Vec<_> = map
            .iter()
            .flat_map(|(t, routes)| {
                routes.iter().map(move |(ix, e)| Snapshot {
                    tenant: t.clone(),
                    route_ix: *ix,
                    report: e.experiment.report(),
                    ignored_stale: e.ignored_stale,
                    health: e.health,
                })
            })
            .collect();
        out.sort_by(|a, b| (&a.tenant, a.route_ix).cmp(&(&b.tenant, b.route_ix)));
        out
    }

    // ── Shared-log async methods ──────────────────────────────────────────────────────────────

    /// Apply a single [`LogRecord`] from the shared backend to local in-memory state.
    ///
    /// All records arriving through `sync_from_backend` already carry the correct fingerprint
    /// (they were keyed by it). This check is defense-in-depth. Returns `true` if the experiment
    /// status changed so the caller can call `Self::publish`.
    fn apply_log_record(
        &self,
        map: &mut HashMap<String, HashMap<usize, Entry>>,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        record: &LogRecord,
    ) -> bool {
        if self.fingerprint(route_ix) != Some(fingerprint) {
            if let Some(e) = self.entry(map, tenant, route_ix) {
                e.ignored_stale += 1;
            }
            return false;
        }
        match record {
            LogRecord::Outcome {
                arm,
                outcome,
                session_hash: hash,
                ..
            } => {
                let Some(e) = self.entry(map, tenant, route_ix) else {
                    return false;
                };
                if e.sessions.contains(hash) {
                    return false; // idempotent re-apply on overlapping LRANGE
                }
                if e.sessions.len() >= MAX_TRACKED_SESSIONS {
                    return false;
                }
                let before = e.experiment.status();
                if !e.experiment.observe_outcome(*arm, *outcome) {
                    return false;
                }
                e.sessions.insert(*hash);
                before != e.experiment.status()
            }
            LogRecord::Pair {
                x_inc,
                x_ch,
                challenger_cost_usd,
                ch_error,
                ch_truncated,
                ts_unix,
                ..
            } => {
                let Some(e) = self.entry(map, tenant, route_ix) else {
                    return false;
                };
                let before = e.experiment.status();
                if e.experiment.observe_pair(*x_inc, *x_ch) {
                    // The incumbent's cost is already counted on the serving path; only the
                    // off-path challenger call is new spend.
                    e.experiment
                        .observe_cost(Arm::Challenger, *challenger_cost_usd);
                }
                // Tally challenger health. The live path skips `observe_health` when a log is
                // attached, so this is the single place health is counted per replica view.
                if *ch_error || *ch_truncated > 0 {
                    e.health.challenger.errors += u64::from(*ch_error);
                    e.health.challenger.truncated += *ch_truncated;
                }
                // Seed the paired-challenger spend ledger for cluster-wide cap enforcement.
                // `seed` uses the pair's original timestamp and only debits today's spend,
                // so records from previous UTC days never affect today's cap.
                // The live path skips `paired_ledger.debit` when a log is attached, so this
                // is the single debit point per replica view (no double-counting).
                if let Some(ledger) = &self.paired_ledger {
                    let now = jiff::Timestamp::now();
                    let pair_ts = jiff::Timestamp::from_second(*ts_unix)
                        .unwrap_or(jiff::Timestamp::UNIX_EPOCH);
                    ledger.seed(tenant, route_ix, *challenger_cost_usd, pair_ts, now);
                }
                before != e.experiment.status()
            }
        }
    }

    /// Read records appended to the shared log since the last sync and apply them locally.
    ///
    /// Each call is cheap when there are no new records: one LRANGE that returns empty.
    /// Call this after every feedback event (within the feedback handler, not on the hot
    /// serve path) and from a background timer so all replicas stay ≤ 5 s stale.
    ///
    /// No-op when no log backend is configured.
    pub async fn sync_from_backend(&self, tenant: &str, route_ix: usize, fingerprint: &str) {
        let Some(log) = &self.log else {
            return;
        };
        let offset = {
            let mut map = self.lock();
            self.entry(&mut map, tenant, route_ix)
                .map_or(0, |e| e.remote_log_offset)
        };
        let records = log.load_since(tenant, route_ix, fingerprint, offset).await;
        if records.is_empty() {
            return;
        }
        let mut map = self.lock();
        for (i, rec) in records.iter().enumerate() {
            let changed = self.apply_log_record(&mut map, tenant, route_ix, fingerprint, rec);
            // Update offset regardless of whether the record was applied (stale records
            // still advance the cursor; we must not re-read them on the next sync).
            if let Some(e) = map.get_mut(tenant).and_then(|r| r.get_mut(&route_ix)) {
                e.remote_log_offset = offset + i as u64 + 1;
            }
            if changed {
                let after = map
                    .get(tenant)
                    .and_then(|r| r.get(&route_ix))
                    .map(|e| e.experiment.status());
                match after {
                    Some(ExperimentStatus::Halted) => tracing::error!(
                        tenant,
                        route = route_ix,
                        "EXPERIMENT HALTED (shared log): challenger worse than incumbent"
                    ),
                    Some(ExperimentStatus::Certified) => tracing::info!(
                        tenant,
                        route = route_ix,
                        "experiment certified (shared log)"
                    ),
                    _ => {}
                }
                Self::publish(&map, route_ix);
            }
        }
    }

    /// Record an outcome to the shared log (if configured), then sync from the log.
    ///
    /// In single-replica mode (no log), falls back to the synchronous [`Self::observe_outcome`].
    /// `cost_usd` is stored in the log so other replicas can tally costs without needing the
    /// full trace. Returns `true` when the outcome was counted.
    #[allow(clippy::too_many_arguments)]
    pub async fn record_outcome(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        arm: Arm,
        session_id: &str,
        verdict: Verdict,
        cost_usd: f64,
    ) -> bool {
        // Filter Abstain before claim_session: an abstain carries no evidence, and burning the
        // session slot would prevent a later non-Abstain verdict for the same session from being
        // counted across replicas.
        let outcome = match verdict {
            Verdict::Pass => 1.0,
            Verdict::Fail => 0.0,
            Verdict::Abstain => return false,
        };
        let Some(log) = self.log.as_ref().cloned() else {
            return self.observe_outcome(tenant, route_ix, fingerprint, arm, session_id, verdict);
        };
        // Cross-replica session dedup: only one replica may claim a given session.
        if !log
            .claim_session(tenant, route_ix, fingerprint, session_id)
            .await
        {
            return false;
        }
        log.append(
            tenant,
            route_ix,
            fingerprint,
            LogRecord::Outcome {
                arm,
                outcome,
                session_hash: session_hash(session_id),
                cost_usd,
            },
        )
        .await;
        // Pull in any records other replicas appended since last sync, plus the one we
        // just appended (so local state is always up-to-date after an append).
        self.sync_from_backend(tenant, route_ix, fingerprint).await;
        true
    }

    /// Record a paired shadow result to the shared log (if configured), then sync.
    ///
    /// In single-replica mode (no log), applies the pair directly and emits the status log.
    /// Returns `true` if `x_inc` / `x_ch` are valid (finite, in `[0, 1]`) and the pair was
    /// accepted by the underlying [`Experiment`].
    ///
    /// `ch_error`, `ch_truncated`, and `ts_unix` are forwarded into [`LogRecord::Pair`] so other
    /// replicas can rebuild challenger health and seed the paired-spend ledger. In the no-log path
    /// they are ignored (the caller handles health and ledger directly).
    #[allow(clippy::too_many_arguments)]
    pub async fn record_pair(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        x_inc: f64,
        x_ch: f64,
        incumbent_cost_usd: f64,
        challenger_cost_usd: f64,
        ch_error: bool,
        ch_truncated: u64,
        ts_unix: i64,
    ) -> bool {
        if self.fingerprint(route_ix) != Some(fingerprint) {
            return false;
        }
        let Some(log) = self.log.as_ref().cloned() else {
            // Single-replica path: apply directly and publish.
            let mut map = self.lock();
            let Some(e) = self.entry(&mut map, tenant, route_ix) else {
                return false;
            };
            let before = e.experiment.status();
            let ok = e.experiment.observe_pair(x_inc, x_ch);
            if ok {
                // Without this the challenger's $/request reads n/a, and a certified, cheaper
                // paired challenger could never be recommended for adoption.
                e.experiment
                    .observe_cost(Arm::Challenger, challenger_cost_usd);
            }
            if ok && before != e.experiment.status() {
                match e.experiment.status() {
                    ExperimentStatus::Halted => tracing::error!(
                        tenant,
                        route = route_ix,
                        "EXPERIMENT HALTED: paired challenger worse than incumbent"
                    ),
                    ExperimentStatus::Certified => tracing::info!(
                        tenant,
                        route = route_ix,
                        "experiment certified (paired): challenger within margin"
                    ),
                    ExperimentStatus::Undecided => {}
                }
                Self::publish(&map, route_ix);
            }
            return ok;
        };
        // Validate before appending: an invalid record would permanently pollute the log.
        if !x_inc.is_finite()
            || !(0.0..=1.0).contains(&x_inc)
            || !x_ch.is_finite()
            || !(0.0..=1.0).contains(&x_ch)
        {
            return false;
        }
        let appended = log
            .append(
                tenant,
                route_ix,
                fingerprint,
                LogRecord::Pair {
                    x_inc,
                    x_ch,
                    incumbent_cost_usd,
                    challenger_cost_usd,
                    ch_error,
                    ch_truncated,
                    ts_unix,
                },
            )
            .await;
        if appended == 0 {
            // The shared log did not take the record (e.g. Redis RPUSH failed). Its evidence is
            // withheld, which only delays a decision. Its spend and health must still count here:
            // the live path skipped them because the log was meant to deliver them, and leaving
            // them out would switch the daily cap off exactly while the log is unreachable.
            if let Some(ledger) = &self.paired_ledger {
                ledger.debit(
                    tenant,
                    route_ix,
                    challenger_cost_usd,
                    jiff::Timestamp::now(),
                );
            }
            self.observe_health(
                tenant,
                route_ix,
                fingerprint,
                Arm::Challenger,
                ch_error,
                ch_truncated,
            );
            return false;
        }
        self.sync_from_backend(tenant, route_ix, fingerprint).await;
        true
    }

    /// Spawn a background task that calls `sync_from_backend` for every known experiment at
    /// most every `interval` (spec: ≤ 5 s). The task exits silently when the `Arc` becomes
    /// the last reference. No-op when no log backend is configured.
    pub fn start_sync_task(self: &Arc<Self>, interval: std::time::Duration) {
        if self.log.is_none() {
            return;
        }
        let monitor = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                // Collect (tenant, route_ix, fingerprint) tuples while holding the lock,
                // then drop the lock before each async sync call.
                let keys: Vec<(String, usize, String)> = {
                    let map = monitor.lock();
                    let mut v = Vec::new();
                    for (tenant, routes) in map.iter() {
                        for &route_ix in routes.keys() {
                            if let Some(p) = monitor.params.get(&route_ix) {
                                v.push((tenant.clone(), route_ix, p.fingerprint.clone()));
                            }
                        }
                    }
                    v
                };
                for (tenant, route_ix, fingerprint) in keys {
                    monitor
                        .sync_from_backend(&tenant, route_ix, &fingerprint)
                        .await;
                }
            }
        });
    }
}

/// One experiment as the monitor holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// Tenant the experiment belongs to.
    pub tenant: String,
    /// Index of the route in the config.
    pub route_ix: usize,
    /// The test's state and arm totals.
    pub report: ExperimentReport,
    /// Receipts from a previous experiment configuration that were ignored.
    pub ignored_stale: u64,
    /// Errors and cut-off answers per arm.
    pub health: Health,
}

/// Requests an arm could not answer cleanly. Neither kind is random: errors (timeouts above all)
/// and output-limit cut-offs cluster on long, hard requests. An incumbent error drops its pair, so
/// many of them mean the pass rates describe an easier workload than the route serves; a
/// challenger error counts as a failed pair. Truncated answers are graded as they are, so a high
/// count says the route's `max_tokens` may be deciding the comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ArmHealth {
    /// Requests that ended in an error, with no output to grade.
    pub errors: u64,
    /// Attempts whose answer stopped at the output-token limit.
    pub truncated: u64,
}

/// [`ArmHealth`] for both arms. For a **paired** challenger the counts are persisted in
/// `experiment_pairs` and in [`LogRecord::Pair`], so they survive restarts and are shared across
/// replicas when a shared log backend is attached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Health {
    /// The incumbent arm (rebuilt from receipts at startup).
    pub incumbent: ArmHealth,
    /// The challenger arm.
    pub challenger: ArmHealth,
}

/// Number of attempts in `t` that stopped at the output-token limit.
#[must_use]
pub fn truncated_attempts(t: &Trace) -> u64 {
    t.attempts.iter().filter(|a| a.truncated).count() as u64
}

/// Whether a trace may be used for rung-indexed or threshold analyses (`ope`, `calibrate`).
/// Challenger-arm traces are excluded: their rungs index a different ladder and their gate scores
/// relate to correctness differently, so pooling them would silently mix two models' behaviour.
#[must_use]
pub fn is_incumbent_trace(t: &Trace) -> bool {
    t.experiment
        .as_ref()
        .is_none_or(|e| e.arm != Arm::Challenger)
}

/// Rebuild a monitor from receipts.
///
/// `traces` are in chain order; `deferred` are `(trace_id, verdict)` in the order the feedback
/// arrived (the store's insertion order); `pairs` are paired-mode records in insertion order.
/// Costs are tallied per trace and outcomes are replayed in arrival order with the same
/// first-outcome-per-session rule as the live path. For sequential feedback the result is exactly
/// what the live monitor held. Under concurrent feedback the store's insertion order and the
/// monitor's lock order can differ, so the pairing — and with it `decided_at_pair` and the
/// e-values — may differ slightly from the live run. The replay is then the authoritative record
/// (it is what an auditor can re-derive), and both remain valid e-processes because arm
/// assignment is randomized independently of arrival order. Traces without an experiment record
/// (other routes, other eras) are skipped; deferred verdicts for unknown traces are ignored.
#[must_use]
pub fn monitor_from_traces(
    traces: &[Trace],
    deferred: &[(Uuid, Verdict)],
    pairs: &[crate::store::ExperimentPairRow],
    config: &Config,
) -> ExperimentMonitor {
    let monitor = ExperimentMonitor::from_config(config);
    if !monitor.is_active() {
        return monitor;
    }
    let mut by_id: HashMap<Uuid, &Trace> = HashMap::new();
    for t in traces {
        let Some(rec) = &t.experiment else { continue };
        // Matches the feedback path, which attributes a trace with no recorded route to route 0.
        let route_ix = t.route_ix.map_or(0, |r| r as usize);
        monitor.observe_cost(
            &t.tenant_id,
            route_ix,
            &rec.fingerprint,
            rec.arm,
            t.final_.total_cost_usd,
        );
        monitor.observe_health(
            &t.tenant_id,
            route_ix,
            &rec.fingerprint,
            rec.arm,
            t.final_.served_from == firstpass_core::ServedFrom::Error,
            truncated_attempts(t),
        );
        by_id.insert(t.trace_id, t);
    }
    for (id, verdict) in deferred {
        let Some(t) = by_id.get(id) else { continue };
        let Some(rec) = &t.experiment else { continue };
        let route_ix = t.route_ix.map_or(0, |r| r as usize);
        monitor.observe_outcome(
            &t.tenant_id,
            route_ix,
            &rec.fingerprint,
            rec.arm,
            &t.session_id,
            *verdict,
        );
    }
    // Replay paired-mode evidence in insertion order. Each pair bypasses FIFO queuing
    // (both outcomes arrived simultaneously in the off-path task), which is why they land
    // in `experiment_pairs` rather than the deferred-verdict table.
    for row in pairs {
        let mut map = monitor.lock();
        if let Some(e) = monitor.entry(&mut map, &row.tenant, row.route_ix) {
            if monitor.fingerprint(row.route_ix) == Some(row.fingerprint.as_str()) {
                if e.experiment.observe_pair(row.x_inc, row.x_ch) {
                    e.experiment
                        .observe_cost(Arm::Challenger, row.challenger_cost_usd);
                }
                // Rebuild challenger health from the persisted fields.
                if row.ch_error || row.ch_truncated > 0 {
                    e.health.challenger.errors += u64::from(row.ch_error);
                    e.health.challenger.truncated += row.ch_truncated;
                }
            } else {
                e.ignored_stale += 1;
            }
        }
    }
    monitor
}

/// The monitor to serve with, rebuilt from the trace store at startup.
///
/// Only touches the store when some route has a challenger. **Fails closed on the challenger**: if
/// the store cannot be read, the proxy still starts but serves the incumbent only
/// ([`ExperimentMonitor::unavailable`]), because a challenger already proven worse must never be
/// served again just because its history could not be read.
#[must_use]
pub fn build_monitor(db_path: &str, config: &Config) -> ExperimentMonitor {
    build_monitor_with(db_path, config, false)
}

/// [`build_monitor`], told whether a shared experiment log will be attached.
///
/// With a shared log, the log is the cluster-wide record of every paired evaluation, and the
/// sync task reads it from the start on boot. Replaying the local `experiment_pairs` table as
/// well would apply each pair twice (pairs, unlike live outcomes, carry no session to dedup on),
/// doubling the e-values and the challenger's spend. So pairs then only register their
/// experiment, and the evidence itself comes from the log on the first sync.
#[must_use]
pub fn build_monitor_with(db_path: &str, config: &Config, shared_log: bool) -> ExperimentMonitor {
    let empty = ExperimentMonitor::from_config(config);
    if !empty.is_active() {
        return empty;
    }
    let loaded = crate::store::load_all_traces(db_path)
        .and_then(|t| crate::store::load_all_deferred(db_path).map(|d| (t, d)))
        .and_then(|(t, d)| crate::store::load_all_experiment_pairs(db_path).map(|p| (t, d, p)));
    match loaded {
        Ok((traces, deferred, pairs)) => {
            let replayed: &[crate::store::ExperimentPairRow] =
                if shared_log { &[] } else { &pairs };
            let m = monitor_from_traces(&traces, &deferred, replayed, config);
            if shared_log {
                let mut map = m.lock();
                for row in &pairs {
                    let _ = m.entry(&mut map, &row.tenant, row.route_ix);
                }
            }
            for s in m.reports() {
                tracing::info!(
                    tenant = %s.tenant,
                    route = s.route_ix,
                    status = ?s.report.status,
                    pairs = s.report.pairs,
                    ignored_stale = s.ignored_stale,
                    "experiment restored from receipts"
                );
                ExperimentMonitor::publish(&m.lock(), s.route_ix);
            }
            m
        }
        Err(e) => {
            tracing::error!(
                %e,
                "could not read the trace store, so it is unknown whether a challenger was already \
                 halted: challenger arms are NOT served (incumbent only) until restart with a \
                 readable store"
            );
            ExperimentMonitor::unavailable(config)
        }
    }
}

/// Replay receipts into per-`(tenant, route)` snapshots. See [`monitor_from_traces`].
#[must_use]
pub fn experiments_from_traces(
    traces: &[Trace],
    deferred: &[(Uuid, Verdict)],
    pairs: &[crate::store::ExperimentPairRow],
    config: &Config,
) -> Vec<Snapshot> {
    monitor_from_traces(traces, deferred, pairs, config).reports()
}

/// What an operator should do with an experiment's current state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    /// Certified, and the challenger costs less per request: adopt it (by editing the config).
    Adopt,
    /// Certified, but the challenger is not measurably cheaper: nothing to gain.
    CertifiedNoSaving,
    /// Proven worse by more than the margin; no longer served.
    ChallengerStopped,
    /// Not enough evidence yet.
    KeepRunning,
}

/// The recommendation for a report. Cost is compared as dollars per served request on each arm;
/// missing cost data on either arm counts as "no saving" because adoption must not rest on a
/// number that was never measured.
#[must_use]
pub fn recommend(r: &ExperimentReport) -> Recommendation {
    match r.status {
        select::Status::Undecided => Recommendation::KeepRunning,
        select::Status::Halted => Recommendation::ChallengerStopped,
        select::Status::Certified => {
            match (
                r.challenger.cost_per_request(),
                r.incumbent.cost_per_request(),
            ) {
                (Some(c), Some(i)) if c < i => Recommendation::Adopt,
                _ => Recommendation::CertifiedNoSaving,
            }
        }
    }
}

/// One experiment as the CLI presents it.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentView {
    /// Tenant the experiment belongs to.
    pub tenant: String,
    /// Index of the route in the config.
    pub route_ix: usize,
    /// The test's state and arm totals.
    pub report: ExperimentReport,
    /// Receipts from a previous experiment configuration that were ignored.
    pub ignored_stale: u64,
    /// Errors and cut-off answers per arm.
    pub health: Health,
    /// What to do about it.
    pub recommendation: Recommendation,
    /// The challenger ladder under test, so `adopt` is actionable without re-reading the config.
    pub challenger_ladder: Vec<String>,
}

/// Pair each report with its recommendation and the challenger ladder from `config`.
#[must_use]
pub fn views(reports: Vec<Snapshot>, config: &Config) -> Vec<ExperimentView> {
    reports
        .into_iter()
        .map(|s| ExperimentView {
            recommendation: recommend(&s.report),
            ignored_stale: s.ignored_stale,
            health: s.health,
            challenger_ladder: config
                .routes
                .get(s.route_ix)
                .and_then(|r| r.challenger.as_ref())
                .map(|c| c.ladder.clone())
                .unwrap_or_default(),
            tenant: s.tenant,
            route_ix: s.route_ix,
            report: s.report,
        })
        .collect()
}

fn fmt_opt(v: Option<f64>, f: impl Fn(f64) -> String) -> String {
    v.map_or_else(|| "n/a".to_owned(), f)
}

/// Human-readable report, one block per `(tenant, route)`.
#[must_use]
pub fn render(views: &[ExperimentView]) -> String {
    use std::fmt::Write as _;
    if views.is_empty() {
        return "no experiment data yet — the config has no [route.challenger], or no experiment \
                traffic has been recorded\n"
            .to_owned();
    }
    let mut out = String::new();
    for v in views {
        let r = &v.report;
        let _ = writeln!(
            out,
            "tenant {} · route {} · status: {:?}",
            v.tenant, v.route_ix, r.status
        );
        let _ = writeln!(
            out,
            "  pairs: {} (margin {}, alpha {})",
            r.pairs, r.margin, r.alpha
        );
        for (name, a) in [("incumbent ", &r.incumbent), ("challenger", &r.challenger)] {
            let _ = writeln!(
                out,
                "  {name}: success {} over {} sessions · ${} per request over {} requests",
                fmt_opt(a.success_rate(), |s| format!("{:.1}%", s * 100.0)),
                a.outcomes,
                fmt_opt(a.cost_per_request(), |c| format!("{c:.6}")),
                a.requests,
            );
        }
        let _ = writeln!(
            out,
            "  e-value to certify: {:.3} · e-value to halt: {:.3} (decide at {:.1})",
            r.e_certify,
            r.e_halt,
            1.0 / r.alpha
        );
        if v.ignored_stale > 0 {
            let _ = writeln!(
                out,
                "  {} receipts from a previous experiment configuration ignored",
                v.ignored_stale
            );
        }
        for (name, h) in [
            ("incumbent", &v.health.incumbent),
            ("challenger", &v.health.challenger),
        ] {
            if h.errors > 0 || h.truncated > 0 {
                let _ = writeln!(
                    out,
                    "  {name}: {} errors, {} answers cut off at max_tokens (neither is random: \
                     check them before trusting the rates)",
                    h.errors, h.truncated
                );
            }
        }
        match v.recommendation {
            Recommendation::Adopt => {
                let ladder = v
                    .challenger_ladder
                    .iter()
                    .map(|m| format!("{m:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(
                    out,
                    "  ADOPT: set this route's ladder to [{ladder}]\n\n    ladder = [{ladder}]\n"
                );
            }
            Recommendation::CertifiedNoSaving => {
                let _ = writeln!(out, "  certified, but no saving — keep incumbent");
            }
            Recommendation::ChallengerStopped => {
                let _ = writeln!(out, "  challenger stopped: worse by more than the margin");
            }
            Recommendation::KeepRunning => {
                let _ = writeln!(out, "  keep running");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use firstpass_core::trace::ExperimentRecord;
    use firstpass_core::{
        Features, FinalOutcome, GENESIS_HASH, Mode, PolicyRef, RequestInfo, ServedFrom, TaskKind,
    };

    use super::*;

    const CFG: &str = "[[route]]\nmatch = {}\nmode = \"enforce\"\n\
        ladder = [\"anthropic/claude-sonnet-5\"]\n\
        [route.challenger]\nladder = [\"anthropic/claude-haiku-4-5\"]\npercent = 50.0\n";

    fn config() -> Config {
        Config::parse(CFG).expect("test config parses")
    }

    /// The fingerprint the current test config stamps on its receipts.
    fn fp() -> String {
        let c = config();
        c.routes[0]
            .challenger
            .as_ref()
            .expect("challenger")
            .fingerprint(&c.routes[0].ladder)
    }

    fn trace(tenant: &str, session: &str, arm: Arm, cost: f64) -> Trace {
        Trace {
            trace_id: Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: tenant.to_owned(),
            session_id: session.to_owned(),
            ts: jiff::Timestamp::now(),
            mode: Mode::Enforce,
            policy: PolicyRef {
                id: "static-ladder@v0".to_owned(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "h".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![],
            deferred: vec![],
            final_: FinalOutcome {
                served_rung: Some(0),
                served_from: ServedFrom::Attempt,
                total_cost_usd: cost,
                gate_cost_usd: 0.0,
                total_latency_ms: 1,
                escalations: 0,
                counterfactual_baseline_usd: cost,
                savings_usd: 0.0,
                cache_source: None,
            },
            probe: None,
            rollout: None,
            shadow: None,
            route_ix: Some(0),
            predicted_pass: None,
            decision_prior: None,
            elastic: None,
            risk: None,
            experiment: Some(ExperimentRecord {
                arm,
                percent: 50.0,
                bucket: 0,
                fingerprint: fp(),
            }),
        }
    }

    /// Errors and cut-off answers are rebuilt from receipts, per arm, and receipts from an older
    /// experiment configuration contribute nothing.
    #[test]
    fn arm_health_is_rebuilt_from_receipts() {
        let attempt = |truncated| firstpass_core::trace::Attempt {
            rung: 0,
            model: "anthropic/claude-sonnet-5".to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 1,
            cache_write_tokens: 0,
            cache_read_tokens: 0,
            out_tokens: 1,
            cost_usd: 0.0,
            latency_ms: 1,
            gates: vec![],
            verdict: Verdict::Pass,
            truncated,
        };
        let mut errored = trace("t", "a", Arm::Incumbent, 0.01);
        errored.final_.served_from = ServedFrom::Error;
        let mut cut = trace("t", "b", Arm::Incumbent, 0.01);
        cut.attempts = vec![attempt(true), attempt(true), attempt(false)];
        let mut ch_cut = trace("t", "c", Arm::Challenger, 0.01);
        ch_cut.attempts = vec![attempt(true)];
        let mut stale = trace("t", "d", Arm::Incumbent, 0.01);
        stale.final_.served_from = ServedFrom::Error;
        if let Some(rec) = stale.experiment.as_mut() {
            rec.fingerprint = "an-older-config".to_owned();
        }
        let snaps = experiments_from_traces(&[errored, cut, ch_cut, stale], &[], &[], &config());
        let h = snaps[0].health;
        assert_eq!(
            h.incumbent,
            ArmHealth {
                errors: 1,
                truncated: 2
            },
            "{h:?}"
        );
        assert_eq!(
            h.challenger,
            ArmHealth {
                errors: 0,
                truncated: 1
            },
            "{h:?}"
        );
        assert_eq!(snaps[0].ignored_stale, 1);
    }

    #[test]
    fn first_outcome_per_session_counts_and_abstain_is_ignored() {
        let m = ExperimentMonitor::from_config(&config());
        assert!(!m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s1", Verdict::Abstain));
        assert!(m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s1", Verdict::Pass));
        // Later outcomes from the same session are dropped, whatever they say.
        assert!(!m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s1", Verdict::Fail));
        assert!(m.observe_outcome("t", 0, &fp(), Arm::Incumbent, "s2", Verdict::Fail));
        let r = &m.reports()[0].report;
        assert_eq!(r.incumbent.outcomes, 2);
        assert!((r.incumbent.outcome_sum - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn routes_without_a_challenger_and_unknown_tenants_are_inert() {
        let m = ExperimentMonitor::from_config(&config());
        assert!(!m.observe_outcome("t", 7, &fp(), Arm::Incumbent, "s", Verdict::Pass));
        assert_eq!(m.status("nobody", 0), ExperimentStatus::Undecided);
        assert!(m.reports().is_empty());
        let off = ExperimentMonitor::default();
        assert!(!off.is_active());
        off.observe_cost("t", 0, &fp(), Arm::Incumbent, 1.0);
        assert!(off.reports().is_empty());
    }

    #[test]
    fn tenants_never_share_an_experiment() {
        let m = ExperimentMonitor::from_config(&config());
        for i in 0..400 {
            m.observe_outcome(
                "a",
                0,
                &fp(),
                Arm::Incumbent,
                &format!("i{i}"),
                Verdict::Pass,
            );
            m.observe_outcome(
                "a",
                0,
                &fp(),
                Arm::Challenger,
                &format!("c{i}"),
                Verdict::Fail,
            );
        }
        assert_eq!(m.status("a", 0), ExperimentStatus::Halted);
        assert_eq!(m.status("b", 0), ExperimentStatus::Undecided);
    }

    #[test]
    fn a_clearly_worse_challenger_halts() {
        let m = ExperimentMonitor::from_config(&config());
        let mut at = None;
        for i in 0..500 {
            m.observe_outcome(
                "t",
                0,
                &fp(),
                Arm::Incumbent,
                &format!("i{i}"),
                Verdict::Pass,
            );
            m.observe_outcome(
                "t",
                0,
                &fp(),
                Arm::Challenger,
                &format!("c{i}"),
                Verdict::Fail,
            );
            if at.is_none() && m.status("t", 0) == ExperimentStatus::Halted {
                at = Some(i);
            }
        }
        assert!(at.is_some(), "never halted");
        assert_eq!(m.status("t", 0), ExperimentStatus::Halted);
    }

    #[test]
    fn replay_matches_live_with_session_dedup_and_halt() {
        let cfg = config();
        let mut traces = Vec::new();
        let mut deferred = Vec::new();
        for i in 0..400 {
            let inc = trace("t", &format!("inc{i}"), Arm::Incumbent, 0.10);
            let chl = trace("t", &format!("chl{i}"), Arm::Challenger, 0.02);
            deferred.push((inc.trace_id, Verdict::Pass));
            deferred.push((chl.trace_id, Verdict::Fail));
            // A second trace in the same session whose outcome contradicts the first: ignored.
            let again = trace("t", &format!("chl{i}"), Arm::Challenger, 0.02);
            deferred.push((again.trace_id, Verdict::Pass));
            traces.extend([inc, chl, again]);
        }
        let reports = experiments_from_traces(&traces, &deferred, &[], &cfg);
        assert_eq!(reports.len(), 1);
        let snap = &reports[0];
        let r = &snap.report;
        assert_eq!((snap.tenant.as_str(), snap.route_ix), ("t", 0));
        assert_eq!(snap.ignored_stale, 0);
        assert_eq!(r.status, select::Status::Halted);
        // Dedup: one counted outcome per session, and the counted one is the first (Fail = 0).
        assert_eq!(r.challenger.outcomes, 400);
        assert!(r.challenger.outcome_sum.abs() < f64::EPSILON);
        // Cost is per request, not per session: the extra trace in each session still cost money.
        assert_eq!(r.challenger.requests, 800);
        assert!((r.challenger.cost_per_request().unwrap_or(0.0) - 0.02).abs() < 1e-12);
        assert!((r.incumbent.cost_per_request().unwrap_or(0.0) - 0.10).abs() < 1e-12);
        assert_eq!(recommend(r), Recommendation::ChallengerStopped);

        // The live monitor, fed the same events, reaches the same decision.
        let live = ExperimentMonitor::from_config(&cfg);
        for t in &traces {
            let rec = t.experiment.as_ref().expect("recorded");
            live.observe_cost(
                &t.tenant_id,
                0,
                &rec.fingerprint,
                rec.arm,
                t.final_.total_cost_usd,
            );
        }
        for (id, v) in &deferred {
            let t = traces.iter().find(|t| t.trace_id == *id).expect("trace");
            let rec = t.experiment.as_ref().expect("recorded");
            live.observe_outcome("t", 0, &rec.fingerprint, rec.arm, &t.session_id, *v);
        }
        assert_eq!(live.reports()[0], *snap);
        // Replaying the same receipts under the same config is exactly reproducible.
        assert_eq!(
            experiments_from_traces(&traces, &deferred, &[], &cfg),
            reports
        );
    }

    #[test]
    fn replay_ignores_unrecorded_traces_and_unknown_deferred() {
        let cfg = config();
        let mut plain = trace("t", "s", Arm::Incumbent, 0.1);
        plain.experiment = None;
        let deferred = [
            (plain.trace_id, Verdict::Pass),
            (Uuid::now_v7(), Verdict::Pass),
        ];
        assert!(experiments_from_traces(&[plain], &deferred, &[], &cfg).is_empty());
        // A config with no challenger rebuilds nothing at all.
        let bare = Config::parse(
            "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-sonnet-5\"]\n",
        )
        .expect("parses");
        let t = trace("t", "s", Arm::Incumbent, 0.1);
        assert!(experiments_from_traces(&[t], &[], &[], &bare).is_empty());
    }

    fn view_with(status: select::Status, inc_cost: f64, ch_cost: f64) -> ExperimentView {
        let mut exp = Experiment::new(0.05, 0.05).expect("valid");
        exp.observe_cost(Arm::Incumbent, inc_cost);
        exp.observe_cost(Arm::Challenger, ch_cost);
        let mut report = exp.report();
        report.status = status;
        ExperimentView {
            tenant: "t".to_owned(),
            route_ix: 0,
            recommendation: recommend(&report),
            ignored_stale: 0,
            health: Health::default(),
            report,
            challenger_ladder: vec!["anthropic/claude-haiku-4-5".to_owned()],
        }
    }

    #[test]
    fn recommendation_and_rendering_cover_every_state() {
        let adopt = view_with(select::Status::Certified, 0.10, 0.02);
        assert_eq!(adopt.recommendation, Recommendation::Adopt);
        let text = render(&[adopt]);
        assert!(
            text.contains("ADOPT: set this route's ladder to [\"anthropic/claude-haiku-4-5\"]"),
            "{text}"
        );
        assert!(text.contains("ladder = [\"anthropic/claude-haiku-4-5\"]"));

        let dear = view_with(select::Status::Certified, 0.02, 0.10);
        assert_eq!(dear.recommendation, Recommendation::CertifiedNoSaving);
        assert!(render(&[dear]).contains("certified, but no saving — keep incumbent"));

        let halted = view_with(select::Status::Halted, 0.10, 0.02);
        assert!(render(&[halted]).contains("challenger stopped: worse by more than the margin"));

        let open = view_with(select::Status::Undecided, 0.10, 0.02);
        assert_eq!(open.recommendation, Recommendation::KeepRunning);
        assert!(render(&[open]).contains("keep running"));
        assert!(render(&[]).contains("no experiment data"));
    }

    #[test]
    fn certified_without_cost_data_is_not_adopted() {
        let exp = Experiment::new(0.05, 0.05).expect("valid");
        let mut report = exp.report();
        report.status = select::Status::Certified;
        assert_eq!(recommend(&report), Recommendation::CertifiedNoSaving);
    }

    /// Changing the challenger ladder (or margin/alpha) starts a fresh experiment: receipts from
    /// the old configuration are ignored and counted, never pooled.
    #[test]
    fn changing_the_challenger_ladder_resets_the_experiment() {
        let old = config();
        let mut traces = Vec::new();
        let mut deferred = Vec::new();
        for i in 0..300 {
            let inc = trace("t", &format!("inc{i}"), Arm::Incumbent, 0.10);
            let chl = trace("t", &format!("chl{i}"), Arm::Challenger, 0.02);
            deferred.push((inc.trace_id, Verdict::Pass));
            deferred.push((chl.trace_id, Verdict::Fail));
            traces.extend([inc, chl]);
        }
        let before = experiments_from_traces(&traces, &deferred, &[], &old);
        assert_eq!(before[0].report.status, select::Status::Halted);
        assert_eq!(before[0].ignored_stale, 0);

        let new =
            Config::parse(&CFG.replace("claude-haiku-4-5", "claude-opus-4-8")).expect("parses");
        let after = experiments_from_traces(&traces, &deferred, &[], &new);
        assert_eq!(after[0].report.status, select::Status::Undecided);
        assert_eq!(after[0].report.pairs, 0);
        assert_eq!(after[0].report.challenger.requests, 0);
        assert_eq!(after[0].ignored_stale, 600);
        let text = render(&views(after, &new));
        assert!(
            text.contains("600 receipts from a previous experiment configuration ignored"),
            "{text}"
        );

        // margin and alpha are part of the identity too.
        let margin = Config::parse(&format!("{CFG}margin = 0.04\n")).expect("parses");
        assert_eq!(
            experiments_from_traces(&traces, &deferred, &[], &margin)[0].ignored_stale,
            600
        );
        // percent is not: ramping the split keeps the evidence.
        let ramped = Config::parse(&CFG.replace("50.0", "30.0")).expect("parses");
        assert_eq!(
            experiments_from_traces(&traces, &deferred, &[], &ramped)[0].report,
            before[0].report
        );
    }

    #[test]
    fn challenger_traces_are_not_incumbent_traces() {
        assert!(is_incumbent_trace(&trace("t", "s", Arm::Incumbent, 0.1)));
        assert!(!is_incumbent_trace(&trace("t", "s", Arm::Challenger, 0.1)));
        let mut plain = trace("t", "s", Arm::Challenger, 0.1);
        plain.experiment = None;
        assert!(is_incumbent_trace(&plain));
    }

    /// An unreadable store must never re-open a halted challenger: the monitor is built blocked.
    #[tokio::test]
    async fn unreadable_store_fails_closed_on_the_challenger() {
        let cfg = config();
        let ok = ExperimentMonitor::from_config(&cfg);
        assert!(!ok.challenger_blocked("t", 0));
        let blocked = ExperimentMonitor::unavailable(&cfg);
        assert!(blocked.challenger_blocked("t", 0));
        assert!(blocked.challenger_blocked("any-other-tenant", 0));

        // A path that is a directory cannot be opened as a database.
        let dir = std::env::temp_dir();
        let m = build_monitor(&dir.to_string_lossy(), &cfg);
        assert!(
            m.challenger_blocked("t", 0),
            "unreadable store served a challenger"
        );

        // A readable (fresh) store is not blocked.
        let db = dir.join(format!("firstpass-exp-build-{}.db", Uuid::now_v7()));
        let (tx, handle) = crate::store::open(&db).expect("open");
        drop(tx);
        handle.await.expect("writer");
        let m = build_monitor(&db.to_string_lossy(), &cfg);
        assert!(!m.challenger_blocked("t", 0));
        let _ = std::fs::remove_file(&db);

        // No challenger configured: nothing to block, store never touched.
        let bare = Config::parse(
            "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-sonnet-5\"]\n",
        )
        .expect("parses");
        assert!(!build_monitor(&dir.to_string_lossy(), &bare).challenger_blocked("t", 0));
    }

    /// A paired evaluation's challenger spend must reach the report on every path that applies a
    /// pair: direct (single replica), synced from the shared log, and rebuilt from the store at
    /// startup. Without it the challenger's $/request is n/a and a certified, cheaper paired
    /// challenger can never be recommended for adoption.
    #[tokio::test]
    async fn paired_challenger_cost_reaches_the_report_on_every_path() {
        let cfg = config();
        let fp_str = fp();
        let challenger = |m: &ExperimentMonitor| {
            let r = m.reports();
            assert_eq!(r.len(), 1);
            let c = r[0].report.challenger;
            (c.requests, c.cost_usd, r[0].report.pairs)
        };
        let close = |(n, usd, pairs): (u64, f64, u64)| {
            assert_eq!((n, pairs), (3, 3));
            assert!((usd - 0.03).abs() < 1e-12, "{usd}");
        };

        let direct = ExperimentMonitor::from_config(&cfg);
        for _ in 0..3 {
            assert!(
                direct
                    .record_pair("t", 0, &fp_str, 1.0, 1.0, 0.10, 0.01, false, 0, 0)
                    .await
            );
        }
        close(challenger(&direct));

        let shared = Arc::new(InMemoryExperimentLog::new());
        let a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        let b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        for _ in 0..3 {
            a.record_pair("t", 0, &fp_str, 1.0, 1.0, 0.10, 0.01, false, 0, 0)
                .await;
        }
        b.sync_from_backend("t", 0, &fp_str).await;
        close(challenger(&a));
        close(challenger(&b));

        let db =
            std::env::temp_dir().join(format!("firstpass-paircost-{}.db", uuid::Uuid::now_v7()));
        let (tx, handle) = crate::store::open(&db).unwrap();
        drop(tx);
        handle.await.unwrap();
        for _ in 0..3 {
            crate::store::append_experiment_pair(
                &db,
                &crate::store::ExperimentPairRow {
                    tenant: "t".to_owned(),
                    route_ix: 0,
                    fingerprint: fp_str.clone(),
                    trace_id: uuid::Uuid::now_v7(),
                    x_inc: 1.0,
                    x_ch: 1.0,
                    challenger_cost_usd: 0.01,
                    ch_error: false,
                    ch_truncated: 0,
                    ts: jiff::Timestamp::now(),
                },
            )
            .unwrap();
        }
        close(challenger(&build_monitor(&db.to_string_lossy(), &cfg)));

        // Restart with a shared log: the same three pairs are in the store AND the log. They
        // must count once, not twice (pairs carry no session to dedup on).
        let restarted = build_monitor_with(&db.to_string_lossy(), &cfg, true)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        restarted.sync_from_backend("t", 0, &fp_str).await;
        close(challenger(&restarted));
        let _ = std::fs::remove_file(&db);
    }

    // ── Shared-log tests (InMemoryExperimentLog) ─────────────────────────────────────────────

    /// Two monitors sharing one log converge to the same state after syncing.
    ///
    /// Mutation test: if `sync_from_backend` does not apply records, replica_b stays Undecided
    /// even after 400 pairs of Pass/Fail evidence, so the final `Halted` assertion fails.
    #[tokio::test]
    async fn two_replicas_sharing_one_log_converge_after_sync() {
        let shared = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);

        // Feed all evidence through replica A (it appends to the shared log).
        for i in 0..400 {
            replica_a
                .record_outcome(
                    "t",
                    0,
                    &fp_str,
                    Arm::Incumbent,
                    &format!("i{i}"),
                    Verdict::Pass,
                    0.10,
                )
                .await;
            replica_a
                .record_outcome(
                    "t",
                    0,
                    &fp_str,
                    Arm::Challenger,
                    &format!("c{i}"),
                    Verdict::Fail,
                    0.02,
                )
                .await;
        }

        // Replica B has not received any feedback directly: still Undecided.
        // (Mutation check: disabling sync_from_backend makes the next assertion the failing one.)
        assert_eq!(
            replica_b.status("t", 0),
            ExperimentStatus::Undecided,
            "replica_b should not have any evidence before syncing"
        );

        // After one sync pass, replica B reads the shared log and reaches the same decision.
        replica_b.sync_from_backend("t", 0, &fp_str).await;
        assert_eq!(
            replica_b.status("t", 0),
            ExperimentStatus::Halted,
            "replica_b must halt after syncing from shared log"
        );
        assert!(
            replica_b.challenger_blocked("t", 0),
            "halted experiment must block the challenger on the synced replica"
        );

        // Both replicas agree on pair counts and e-values.
        let reps_a = replica_a.reports();
        let reps_b = replica_b.reports();
        assert_eq!(reps_a.len(), 1);
        assert_eq!(reps_b.len(), 1);
        assert_eq!(reps_a[0].report.status, reps_b[0].report.status);
        assert_eq!(reps_a[0].report.pairs, reps_b[0].report.pairs);
    }

    /// A halt observed via one replica blocks the challenger on the other.
    ///
    /// Specifically: replica A drives the experiment to Halted; replica B has not synced yet,
    /// so it does NOT block. After one sync, replica B also blocks.
    ///
    /// Mutation test: if `challenger_blocked` does not consult status (always returns false),
    /// the first `assert!(!replica_b.challenger_blocked(...))` passes trivially, but the
    /// second one (`assert!(... after sync)`) fails — verifying the block gate actually changed.
    #[tokio::test]
    async fn halt_on_one_replica_blocks_challenger_on_other_after_sync() {
        let shared = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);

        // B has not synced yet — it does not know about the halt.
        assert!(
            !replica_b.challenger_blocked("t", 0),
            "should be unblocked before any evidence"
        );

        // Drive A to halted.
        let mut halted = false;
        for i in 0..500 {
            replica_a
                .record_outcome(
                    "t",
                    0,
                    &fp_str,
                    Arm::Incumbent,
                    &format!("i{i}"),
                    Verdict::Pass,
                    0.10,
                )
                .await;
            replica_a
                .record_outcome(
                    "t",
                    0,
                    &fp_str,
                    Arm::Challenger,
                    &format!("c{i}"),
                    Verdict::Fail,
                    0.02,
                )
                .await;
            if replica_a.status("t", 0) == ExperimentStatus::Halted {
                halted = true;
                break;
            }
        }
        assert!(halted, "experiment never halted on replica A");

        // B still does not know.
        assert!(
            !replica_b.challenger_blocked("t", 0),
            "replica_b should not block before syncing"
        );

        // After sync, B learns of the halt and blocks.
        replica_b.sync_from_backend("t", 0, &fp_str).await;
        assert!(
            replica_b.challenger_blocked("t", 0),
            "replica_b must block after syncing the halt from the shared log"
        );
    }

    /// Cross-replica session dedup: `claim_session` prevents the same session from being
    /// counted twice even when two replicas receive the same feedback.
    ///
    /// Mutation test: if `claim_session` always returns `true`, both replicas count the
    /// same session, the pair count doubles, and the equality assertion fails.
    #[tokio::test]
    async fn cross_replica_session_dedup_prevents_double_counting() {
        let shared = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);

        // Both replicas independently receive the same feedback for the same session.
        replica_a
            .record_outcome(
                "t",
                0,
                &fp_str,
                Arm::Incumbent,
                "session-1",
                Verdict::Pass,
                0.10,
            )
            .await;
        replica_b
            .record_outcome(
                "t",
                0,
                &fp_str,
                Arm::Incumbent,
                "session-1",
                Verdict::Pass,
                0.10,
            )
            .await;

        // Sync both so they agree on the log.
        replica_a.sync_from_backend("t", 0, &fp_str).await;
        replica_b.sync_from_backend("t", 0, &fp_str).await;

        // Only one outcome should have been counted (not two).
        let reps_a = replica_a.reports();
        let reps_b = replica_b.reports();
        assert_eq!(
            reps_a[0].report.incumbent.outcomes, 1,
            "session should be counted exactly once"
        );
        assert_eq!(
            reps_b[0].report.incumbent.outcomes, 1,
            "both replicas should agree on outcome count"
        );
    }

    /// Abstain must not burn the session slot: if a session's first feedback is Abstain,
    /// a subsequent Pass/Fail for the same session must still be accepted.
    ///
    /// Mutation test: if Abstain is filtered AFTER `claim_session` (the pre-fix behaviour),
    /// Abstain claims the slot and the subsequent Pass returns `false`; the `== 1` assertion
    /// below fails.
    #[tokio::test]
    async fn abstain_does_not_burn_session_slot_before_claim() {
        let shared = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();

        let monitor = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);

        // First feedback for this session is Abstain — must not claim the session slot.
        let counted = monitor
            .record_outcome(
                "t",
                0,
                &fp_str,
                Arm::Incumbent,
                "session-abstain",
                Verdict::Abstain,
                0.10,
            )
            .await;
        assert!(!counted, "Abstain must not be counted");

        // Second feedback for the SAME session: Pass — must be accepted.
        let counted = monitor
            .record_outcome(
                "t",
                0,
                &fp_str,
                Arm::Incumbent,
                "session-abstain",
                Verdict::Pass,
                0.10,
            )
            .await;
        assert!(counted, "Pass after Abstain must be counted");

        let reps = monitor.reports();
        assert_eq!(
            reps[0].report.incumbent.outcomes, 1,
            "exactly one outcome must have been counted"
        );
    }

    // ── Redis-backed shared-log tests ────────────────────────────────────────────────────────

    /// Connect to the test Redis, wipe all keys whose name contains `unique_tenant`, and return
    /// the open connection for subsequent cleanup calls.
    #[cfg(feature = "redis-cache")]
    async fn redis_cleanup(url: &str, unique_tenant: &str) {
        let client = redis::Client::open(url).expect("redis client");
        let mut conn = redis::aio::ConnectionManager::new(client)
            .await
            .expect("redis connection for cleanup");
        use redis::AsyncCommands as _;
        // SCAN for all experiment keys that carry this tenant in their name.
        let pattern = format!("firstpass:exp*{unique_tenant}*");
        let keys: Vec<String> = {
            let mut out = Vec::new();
            let mut cursor: u64 = 0;
            loop {
                let (next, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                    .arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(100u64)
                    .query_async(&mut conn)
                    .await
                    .expect("SCAN");
                out.extend(batch);
                cursor = next;
                if cursor == 0 {
                    break;
                }
            }
            out
        };
        if !keys.is_empty() {
            let _: () = conn.del(keys).await.expect("DEL cleanup");
        }
    }

    /// Two `RedisExperimentLog`-backed monitors on the same key converge to identical reports
    /// after sync.
    ///
    /// Requires `FIRSTPASS_TEST_REDIS` (e.g. `redis://127.0.0.1:56411/15`). **Absent, this
    /// panics rather than skipping**: a green suite that silently tested nothing is worse than a
    /// failing one.
    ///
    /// Mutation test: disabling `sync_from_backend` keeps replica_b `Undecided` and the
    /// `Halted` assertion below fails.
    #[cfg(feature = "redis-cache")]
    #[tokio::test]
    async fn redis_two_replicas_converge_to_identical_reports_after_sync() {
        let url = std::env::var("FIRSTPASS_TEST_REDIS").unwrap_or_else(|_| {
            panic!(
                "set FIRSTPASS_TEST_REDIS (e.g. redis://127.0.0.1:56411/15) to run the \
                 redis-cache tests. Not skipped on purpose: a green suite that tested nothing \
                 is worse than a red one."
            )
        });
        // Unique tenant prevents key collisions across concurrent or repeated test runs.
        let unique_tenant = format!("t-conv-{}", Uuid::now_v7());
        let log_a = Arc::new(
            RedisExperimentLog::connect(&url)
                .await
                .expect("RedisExperimentLog::connect"),
        );
        let log_b = Arc::new(
            RedisExperimentLog::connect(&url)
                .await
                .expect("RedisExperimentLog::connect"),
        );
        let cfg = config();
        let fp_str = fp();

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log_a) as Arc<dyn ExperimentLogBackend>);
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log_b) as Arc<dyn ExperimentLogBackend>);

        // All evidence through replica A.
        for i in 0..400 {
            replica_a
                .record_outcome(
                    &unique_tenant,
                    0,
                    &fp_str,
                    Arm::Incumbent,
                    &format!("i{i}"),
                    Verdict::Pass,
                    0.10,
                )
                .await;
            replica_a
                .record_outcome(
                    &unique_tenant,
                    0,
                    &fp_str,
                    Arm::Challenger,
                    &format!("c{i}"),
                    Verdict::Fail,
                    0.02,
                )
                .await;
        }

        // B has not synced: still Undecided locally.
        assert_eq!(
            replica_b.status(&unique_tenant, 0),
            ExperimentStatus::Undecided,
            "replica_b must be Undecided before syncing"
        );

        // One sync makes B read the full log and reach the same decision.
        replica_b
            .sync_from_backend(&unique_tenant, 0, &fp_str)
            .await;
        assert_eq!(
            replica_b.status(&unique_tenant, 0),
            ExperimentStatus::Halted,
            "replica_b must be Halted after syncing from Redis log"
        );

        let reps_a = replica_a.reports();
        let reps_b = replica_b.reports();
        assert_eq!(reps_a.len(), 1);
        assert_eq!(reps_b.len(), 1);
        assert_eq!(
            reps_a[0].report.status, reps_b[0].report.status,
            "both replicas must agree on status"
        );
        assert_eq!(
            reps_a[0].report.pairs, reps_b[0].report.pairs,
            "both replicas must agree on pair counts"
        );

        redis_cleanup(&url, &unique_tenant).await;
    }

    /// `claim_session` prevents the same session from being double-counted across replicas when
    /// both receive the same feedback.
    ///
    /// Requires `FIRSTPASS_TEST_REDIS`. Panics when unset (same rationale as above).
    ///
    /// Mutation test: if `claim_session` always returns `true`, both replicas append the same
    /// session and the outcome count doubles; the `== 1` assertion fails.
    #[cfg(feature = "redis-cache")]
    #[tokio::test]
    async fn redis_cross_replica_session_dedup_via_claim_session_holds() {
        let url = std::env::var("FIRSTPASS_TEST_REDIS").unwrap_or_else(|_| {
            panic!(
                "set FIRSTPASS_TEST_REDIS (e.g. redis://127.0.0.1:56411/15) to run the \
                 redis-cache tests. Not skipped on purpose: a green suite that tested nothing \
                 is worse than a red one."
            )
        });
        let unique_tenant = format!("t-dedup-{}", Uuid::now_v7());
        let log = Arc::new(
            RedisExperimentLog::connect(&url)
                .await
                .expect("RedisExperimentLog::connect"),
        );
        let cfg = config();
        let fp_str = fp();

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);

        // Both replicas receive the same session feedback.
        replica_a
            .record_outcome(
                &unique_tenant,
                0,
                &fp_str,
                Arm::Incumbent,
                "session-x",
                Verdict::Pass,
                0.10,
            )
            .await;
        replica_b
            .record_outcome(
                &unique_tenant,
                0,
                &fp_str,
                Arm::Incumbent,
                "session-x",
                Verdict::Pass,
                0.10,
            )
            .await;

        // Sync both.
        replica_a
            .sync_from_backend(&unique_tenant, 0, &fp_str)
            .await;
        replica_b
            .sync_from_backend(&unique_tenant, 0, &fp_str)
            .await;

        let reps_a = replica_a.reports();
        let reps_b = replica_b.reports();
        assert_eq!(
            reps_a[0].report.incumbent.outcomes, 1,
            "session must be counted exactly once on replica A"
        );
        assert_eq!(
            reps_b[0].report.incumbent.outcomes, 1,
            "session must be counted exactly once on replica B"
        );

        redis_cleanup(&url, &unique_tenant).await;
    }

    /// A halt driven on replica A is propagated to replica B after one sync, blocking the
    /// challenger there too.
    ///
    /// Requires `FIRSTPASS_TEST_REDIS`. Panics when unset.
    ///
    /// Mutation test: removing the `sync_from_backend` call means B never learns of the halt,
    /// and the `challenger_blocked` assertion at the end fails.
    #[cfg(feature = "redis-cache")]
    #[tokio::test]
    async fn redis_halt_on_one_replica_blocks_challenger_on_other() {
        let url = std::env::var("FIRSTPASS_TEST_REDIS").unwrap_or_else(|_| {
            panic!(
                "set FIRSTPASS_TEST_REDIS (e.g. redis://127.0.0.1:56411/15) to run the \
                 redis-cache tests. Not skipped on purpose: a green suite that tested nothing \
                 is worse than a red one."
            )
        });
        let unique_tenant = format!("t-halt-{}", Uuid::now_v7());
        let log = Arc::new(
            RedisExperimentLog::connect(&url)
                .await
                .expect("RedisExperimentLog::connect"),
        );
        let cfg = config();
        let fp_str = fp();

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);

        // B starts unblocked.
        assert!(
            !replica_b.challenger_blocked(&unique_tenant, 0),
            "replica_b must be unblocked before any evidence"
        );

        // Drive A to Halted.
        let mut halted = false;
        for i in 0..500 {
            replica_a
                .record_outcome(
                    &unique_tenant,
                    0,
                    &fp_str,
                    Arm::Incumbent,
                    &format!("i{i}"),
                    Verdict::Pass,
                    0.10,
                )
                .await;
            replica_a
                .record_outcome(
                    &unique_tenant,
                    0,
                    &fp_str,
                    Arm::Challenger,
                    &format!("c{i}"),
                    Verdict::Fail,
                    0.02,
                )
                .await;
            if replica_a.status(&unique_tenant, 0) == ExperimentStatus::Halted {
                halted = true;
                break;
            }
        }
        assert!(halted, "experiment never halted on replica A");

        // B still does not know: it hasn't synced.
        assert!(
            !replica_b.challenger_blocked(&unique_tenant, 0),
            "replica_b must not block before syncing"
        );

        // After one sync, B learns the halt and blocks.
        replica_b
            .sync_from_backend(&unique_tenant, 0, &fp_str)
            .await;
        assert!(
            replica_b.challenger_blocked(&unique_tenant, 0),
            "replica_b must block the challenger after syncing the halt from Redis"
        );

        redis_cleanup(&url, &unique_tenant).await;
    }

    /// Keys are isolated per tenant: evidence written under `tenant_a` is invisible to a monitor
    /// reading under `tenant_b`.
    ///
    /// Requires `FIRSTPASS_TEST_REDIS`. Panics when unset.
    ///
    /// Mutation test: if the key omits the tenant and both monitors share one list, B would
    /// gain outcomes after sync and the `Undecided` assertion fails.
    #[cfg(feature = "redis-cache")]
    #[tokio::test]
    async fn redis_keys_are_isolated_per_tenant() {
        let url = std::env::var("FIRSTPASS_TEST_REDIS").unwrap_or_else(|_| {
            panic!(
                "set FIRSTPASS_TEST_REDIS (e.g. redis://127.0.0.1:56411/15) to run the \
                 redis-cache tests. Not skipped on purpose: a green suite that tested nothing \
                 is worse than a red one."
            )
        });
        let run_id = Uuid::now_v7();
        let tenant_a = format!("t-iso-a-{run_id}");
        let tenant_b = format!("t-iso-b-{run_id}");
        let log = Arc::new(
            RedisExperimentLog::connect(&url)
                .await
                .expect("RedisExperimentLog::connect"),
        );
        let cfg = config();
        let fp_str = fp();

        let monitor_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);
        let monitor_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);

        // Feed evidence only under tenant_a.
        for i in 0..5 {
            monitor_a
                .record_outcome(
                    &tenant_a,
                    0,
                    &fp_str,
                    Arm::Incumbent,
                    &format!("s{i}"),
                    Verdict::Pass,
                    0.10,
                )
                .await;
        }

        // Prove storage-level isolation: load_since for tenant_b returns nothing.
        let records = log.load_since(&tenant_b, 0, &fp_str, 0).await;
        assert!(
            records.is_empty(),
            "backend must return no records for tenant_b"
        );

        // monitor_b reads under tenant_b via status() (does not create an entry) and sees
        // Undecided, confirming no cross-tenant contamination.
        assert_eq!(
            monitor_b.status(&tenant_b, 0),
            ExperimentStatus::Undecided,
            "evidence written under tenant_a must not appear under tenant_b"
        );

        // monitor_a still has its own evidence.
        let reps = monitor_a.reports();
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].report.incumbent.outcomes, 5);

        // Clean up both tenants' keys.
        redis_cleanup(&url, &tenant_a).await;
        redis_cleanup(&url, &tenant_b).await;
    }

    // ── Challenger health persistence and cluster-wide spend cap ─────────────────────────────

    /// An old-format `LogRecord::Pair` JSON (without `ch_error`, `ch_truncated`, `ts_unix`)
    /// deserializes without error, with all three fields defaulting to `false`/`0`/`0`.
    ///
    /// Mutation test: removing `#[serde(default)]` from any of the three fields makes this
    /// deserialize fail and the `unwrap` panics — the assertion never runs.
    #[test]
    fn old_log_record_pair_deserializes_without_new_fields() {
        let json = r#"{"kind":"pair","x_inc":1.0,"x_ch":0.5,"incumbent_cost_usd":0.1,"challenger_cost_usd":0.01}"#;
        let record: LogRecord = serde_json::from_str(json).unwrap();
        match record {
            LogRecord::Pair {
                ch_error,
                ch_truncated,
                ts_unix,
                x_inc,
                x_ch,
                ..
            } => {
                assert!(!ch_error, "ch_error defaults to false");
                assert_eq!(ch_truncated, 0, "ch_truncated defaults to 0");
                assert_eq!(ts_unix, 0, "ts_unix defaults to 0");
                assert!((x_inc - 1.0).abs() < f64::EPSILON);
                assert!((x_ch - 0.5).abs() < f64::EPSILON);
            }
            _ => panic!("expected Pair variant"),
        }
    }

    /// Challenger health (`ch_error`, `ch_truncated`) persists to `experiment_pairs` via
    /// `ExperimentPairRow` and is rebuilt by `monitor_from_traces` at startup (no shared log).
    ///
    /// Mutation test: if `monitor_from_traces` does not tally `ch_error`/`ch_truncated` from
    /// pair rows, the challenger health stays at 0 and these assertions fail.
    #[tokio::test]
    async fn pair_health_persists_and_rebuilds_from_store() {
        let db =
            std::env::temp_dir().join(format!("firstpass-health-persist-{}.db", Uuid::now_v7()));
        let (tx, handle) = crate::store::open(&db).unwrap();
        drop(tx);
        handle.await.unwrap();
        let cfg = config();
        let fp_str = fp();
        // Write a pair with ch_error=true, ch_truncated=2.
        crate::store::append_experiment_pair(
            &db,
            &crate::store::ExperimentPairRow {
                tenant: "t".to_owned(),
                route_ix: 0,
                fingerprint: fp_str.clone(),
                trace_id: Uuid::now_v7(),
                x_inc: 1.0,
                x_ch: 1.0,
                challenger_cost_usd: 0.01,
                ch_error: true,
                ch_truncated: 2,
                ts: jiff::Timestamp::now(),
            },
        )
        .unwrap();
        // Rebuild without a shared log: pairs are replayed from SQLite.
        let m = build_monitor(&db.to_string_lossy(), &cfg);
        let reps = m.reports();
        assert_eq!(reps.len(), 1, "one experiment entry");
        assert_eq!(reps[0].health.challenger.errors, 1, "ch_error rebuilt");
        assert_eq!(
            reps[0].health.challenger.truncated, 2,
            "ch_truncated rebuilt"
        );
        let _ = std::fs::remove_file(&db);
    }

    /// Using a shared `InMemoryExperimentLog`, challenger health is counted exactly once per
    /// replica view (not double-counted on the originating replica), and replica B's paired
    /// ledger reflects replica A's spend after one sync, reaching the cluster cap.
    ///
    /// Mutation tests (one assertion per invariant):
    /// (a) If `apply_log_record` does not tally health → A's health.errors == 0 → assertion fails.
    /// (b) If `apply_log_record` does not seed the ledger → A's spent_today == 0.0 → fails.
    /// (c) If live `observe_health` were also called (double-count) → A's errors == 2 → fails.
    /// (d) If B's sync does not read the ledger → B's spent_today == 0.0 → fails.
    /// (e) If cluster cap not enforced → `may_spend` returns true when over cap → fails.
    /// A shared log that is down: every append fails (returns 0, as the Redis backend does on a
    /// failed RPUSH) and nothing can be read back.
    #[derive(Debug)]
    struct DownLog;

    #[async_trait::async_trait]
    impl ExperimentLogBackend for DownLog {
        async fn append(&self, _: &str, _: usize, _: &str, _: LogRecord) -> u64 {
            0
        }
        async fn load_since(&self, _: &str, _: usize, _: &str, _: u64) -> Vec<LogRecord> {
            vec![]
        }
        async fn claim_session(&self, _: &str, _: usize, _: &str, _: &str) -> bool {
            false
        }
    }

    /// With the shared log unreachable, a paired challenger's spend still counts against the cap
    /// and its health is still tallied: the live path defers both to the log, so a failed append
    /// must not silently drop them.
    #[tokio::test]
    async fn paired_spend_counts_against_the_cap_when_the_shared_log_is_down() {
        let ledger = Arc::new(crate::shadow::ShadowLedger::new());
        let m = ExperimentMonitor::from_config(&config())
            .with_log(Arc::new(DownLog) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger));
        let now = jiff::Timestamp::now();
        let accepted = m
            .record_pair("t", 0, &fp(), 1.0, 0.0, 0.1, 0.7, true, 2, now.as_second())
            .await;
        assert!(!accepted, "nothing reached the log");
        assert!(
            (ledger.spent_today("t", 0, now) - 0.7).abs() < 1e-9,
            "spend debited locally"
        );
        assert!(
            !ledger.may_spend("t", 0, 0.5, now),
            "cap enforced during the outage"
        );
        let h = m.reports()[0].health.challenger;
        assert_eq!((h.errors, h.truncated), (1, 2), "{h:?}");
    }

    #[tokio::test]
    async fn two_replicas_health_and_spend_count_once() {
        let shared = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();
        let now_ts = jiff::Timestamp::now();
        let now_secs = now_ts.as_second();

        let ledger_a = Arc::new(crate::shadow::ShadowLedger::new());
        let ledger_b = Arc::new(crate::shadow::ShadowLedger::new());

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger_a));
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger_b));

        assert!(
            replica_a.has_log(),
            "log must be attached for this test to be meaningful"
        );

        // Replica A records a pair: ch_error=true, ch_truncated=1, cost=0.8.
        // The live path (spawn_paired_challenger) would skip observe_health and debit when
        // has_log() is true; here we test record_pair directly (same code path via apply_log_record).
        let accepted = replica_a
            .record_pair("t", 0, &fp_str, 1.0, 0.5, 0.1, 0.8, true, 1, now_secs)
            .await;
        assert!(accepted, "pair must be accepted");

        // (a) & (c): A's health comes from apply_log_record, counted exactly once.
        let reps_a = replica_a.reports();
        assert_eq!(
            reps_a[0].health.challenger.errors, 1,
            "A: challenger error counted once, not zero or doubled"
        );
        assert_eq!(
            reps_a[0].health.challenger.truncated, 1,
            "A: challenger truncated counted once"
        );

        // (b): A's ledger is seeded by apply_log_record (not live debit, which is skipped).
        let a_spent = ledger_a.spent_today("t", 0, now_ts);
        assert!(
            (a_spent - 0.8).abs() < 1e-12,
            "A: ledger = 0.8 (not doubled or zero); got {a_spent}"
        );

        // B has not synced yet: no health, no ledger.
        assert_eq!(
            ledger_b.spent_today("t", 0, now_ts),
            0.0,
            "B: no spend before sync"
        );

        // Sync B from the shared log.
        replica_b.sync_from_backend("t", 0, &fp_str).await;

        // B's health matches A's.
        let reps_b = replica_b.reports();
        assert_eq!(
            reps_b[0].health.challenger.errors, 1,
            "B: challenger error visible after sync"
        );

        // (d): B's ledger includes A's spend.
        let b_spent = ledger_b.spent_today("t", 0, now_ts);
        assert!(
            (b_spent - 0.8).abs() < 1e-12,
            "B: ledger reflects A's spend after sync; got {b_spent}"
        );

        // (e): Add another pair to push cluster total over 1.0 cap.
        replica_a
            .record_pair("t", 0, &fp_str, 1.0, 0.5, 0.1, 0.3, false, 0, now_secs)
            .await;
        let a_total = ledger_a.spent_today("t", 0, now_ts);
        assert!(
            (a_total - 1.1).abs() < 1e-12,
            "A: cluster total = 1.1; got {a_total}"
        );
        assert!(
            !ledger_a.may_spend("t", 0, 1.0, now_ts),
            "A: cap exhausted (1.1 > 1.0)"
        );

        // B syncs and also sees the cap exhausted.
        replica_b.sync_from_backend("t", 0, &fp_str).await;
        assert!(
            !ledger_b.may_spend("t", 0, 1.0, now_ts),
            "B: cap exhausted from cluster spend"
        );
    }
}
