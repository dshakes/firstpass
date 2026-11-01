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

/// After a cold-start `sync_from_backend` failure, `ensure_synced` returns `false` immediately
/// for this long without retrying the backend. Prevents a Redis outage from turning N
/// concurrent first-requests per `(tenant, route)` into N × backend-timeout of user-visible
/// latency: only the first caller pays the wait; subsequent callers within this window
/// fast-fail. The cooldown resets once the window expires and the next retry succeeds.
const SYNC_FAILURE_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(5);

/// `ensure_synced` returns `true` (state is fresh) only while the last successful sync was within
/// this bound. 30 s = 6 × the 5 s background-sync interval: a single missed tick does not
/// trigger a re-sync; two consecutive failures do. Once the bound is exceeded, the next
/// `ensure_synced` call re-attempts a sync (subject to the failure cooldown above).
const SYNC_STALENESS_BOUND: std::time::Duration = std::time::Duration::from_secs(30);

/// Hard ceiling on the time `ensure_synced` may block waiting for the backend. A hung Redis
/// connection (e.g., half-open TCP) would otherwise stall live request processing for minutes.
/// 500 ms is generous (one LRANGE of a small list) but short enough to keep tail latency
/// acceptable. On timeout, `ensure_synced` fails closed (same as any other backend error).
const SYNC_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

// ── Shared log (optional Redis / in-memory backend) ───────────────────────────────────────────

/// An error returned by [`ExperimentLogBackend`] operations.
///
/// Backend failures are not fatal for the request path: evidence is withheld from other replicas
/// (delaying a decision), but the local replica keeps serving. Callers use this to decide whether
/// to serve the challenger arm (fail closed when the backend is unreachable).
#[derive(Debug, thiserror::Error)]
pub enum LogError {
    /// The backend (e.g. Redis) returned an error.
    #[error("experiment log backend error: {0}")]
    Backend(String),
}

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
        /// The per-process UUID of the replica that wrote this record. Used to skip double-tallying
        /// health and spend on the originating replica (which already tallied them locally when the
        /// challenger call completed). Records from older replicas (missing this field) deserialize
        /// to `Uuid::nil()`, which never matches any live replica's UUID and so are treated as remote.
        #[serde(default)]
        replica_id: Uuid,
        /// The challenger trace-id, used for cross-source dedup: prevents a pair from being
        /// counted twice when an append times out after Redis applied it (append returns Err but
        /// the record is in the log). `#[serde(default)]` so records written by older replicas
        /// deserialize to `Uuid::nil()`, which is never deduplicated.
        #[serde(default)]
        trace_id: Uuid,
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
    /// Append `record` to the list. Returns the new list length (1-based), or an error if the
    /// backend is unreachable. A failure means the pair/outcome will not reach other replicas,
    /// but local accounting (health, ledger) was already performed by the caller.
    async fn append(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        record: LogRecord,
    ) -> Result<u64, LogError>;

    /// Return records starting at `offset` (0-based). Each element is `Some(record)` for a valid
    /// entry or `None` for a malformed one (failed to deserialize). The list length equals the
    /// number of positions consumed: a `None` at index `i` still advances `offset` by 1. Empty
    /// if the list has ≤ `offset` entries. Returns an error if the backend is unreachable; callers
    /// use this to distinguish "truly empty" from "unreachable" and fail closed.
    async fn load_since(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        offset: u64,
    ) -> Result<Vec<Option<LogRecord>>, LogError>;

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
    ) -> Result<u64, LogError> {
        let key = (tenant.to_owned(), route_ix, fingerprint.to_owned());
        let mut lists = self.lists.lock().unwrap_or_else(PoisonError::into_inner);
        let list = lists.entry(key).or_default();
        list.push(record);
        Ok(list.len() as u64)
    }

    async fn load_since(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        offset: u64,
    ) -> Result<Vec<Option<LogRecord>>, LogError> {
        let key = (tenant.to_owned(), route_ix, fingerprint.to_owned());
        let lists = self.lists.lock().unwrap_or_else(PoisonError::into_inner);
        match lists.get(&key) {
            None => Ok(vec![]),
            Some(list) => {
                let start = offset as usize;
                if start >= list.len() {
                    Ok(vec![])
                } else {
                    // In-memory records are always valid (never malformed).
                    Ok(list[start..].iter().map(|r| Some(r.clone())).collect())
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
        ) -> Result<u64, super::LogError> {
            let key = Self::list_key(tenant, route_ix, fingerprint);
            let json = serde_json::to_string(&record)
                .map_err(|e| super::LogError::Backend(format!("serialize: {e}")))?;
            let mut conn = self.conn.lock().await;
            use redis::AsyncCommands as _;
            conn.rpush::<_, _, u64>(&key, json)
                .await
                .map_err(|e| super::LogError::Backend(format!("RPUSH {key}: {e}")))
        }

        async fn load_since(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            offset: u64,
        ) -> Result<Vec<Option<super::LogRecord>>, super::LogError> {
            let key = Self::list_key(tenant, route_ix, fingerprint);
            let mut conn = self.conn.lock().await;
            use redis::AsyncCommands as _;
            // Saturating cast: a list with > isize::MAX entries is unreachable in practice
            // (Redis is bounded by memory long before that), but the cast must not wrap to a
            // negative index (which LRANGE would interpret as "from the end").
            let start = isize::try_from(offset).unwrap_or(isize::MAX);
            let raw: Vec<String> = conn
                .lrange(&key, start, -1)
                .await
                .map_err(|e| super::LogError::Backend(format!("LRANGE {key}: {e}")))?;
            // Preserve positions: None for malformed entries so sync_from_backend advances
            // remote_log_offset over them and does not re-read them on the next sync.
            Ok(raw
                .iter()
                .map(|s| match serde_json::from_str(s) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        tracing::warn!(
                            %e,
                            "experiment log: malformed record; position consumed, evidence skipped"
                        );
                        None
                    }
                })
                .collect())
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
    /// Challenger trace-ids whose evidence (e-process pair) has already been applied.
    ///
    /// Prevents a pair from being counted twice when an append times out after Redis applied
    /// the record (the append returns `Err` but the record is in the log). The apply-once
    /// rule covers three paths:
    /// - `apply_pair_locally` called by `spawn_paired_challenger` after a SQLite write
    ///   (inserts the id so a later log-sync sees it as already applied).
    /// - SQLite replay at startup (`monitor_from_traces` inserts ids for `NotInLog` rows).
    /// - Log sync in `apply_log_record` (skips evidence if id already present).
    ///
    /// A nil UUID (`Uuid::nil()`) is never inserted and never checked (old records without
    /// a trace_id are always applied; their dedup depends on the e-process being ordered and
    /// idempotent by offset, not by id).
    ///
    /// Capped at `MAX_TRACKED_SESSIONS` with the same saturation policy: once full, new
    /// pairs are skipped rather than evicting old ids (evicting would let a pair count twice).
    applied_pair_ids: HashSet<Uuid>,
    /// How many records have been consumed from the shared log so far (incremental LRANGE).
    remote_log_offset: u64,
    /// Whether this entry has completed at least one full `sync_from_backend` pass since the
    /// process started. Used by `ensure_synced` to gate the cold-start pull: a fresh replica must
    /// see cluster state (halt decisions, spend) before it checks caps, so the very first request
    /// per `(tenant, route_ix)` per process triggers one synchronous sync. Subsequent requests see
    /// `synced_once = true` and skip the extra round-trip.
    synced_once: bool,
    /// Whether a `tracing::warn!` has already been emitted for a failed cold-start sync on this
    /// entry. Guards against per-request log spam when the backend is persistently unreachable.
    sync_warn_logged: bool,
    /// Tokio instant of the last failed `sync_from_backend` cold-start attempt. Used by
    /// `ensure_synced` to enforce the failure cooldown: callers that arrive within
    /// `SYNC_FAILURE_COOLDOWN` of the last failure return `false` immediately without
    /// re-issuing a backend read, so a Redis outage cannot turn N concurrent requests into
    /// N × backend-timeout of user-visible latency.
    sync_failed_at: Option<tokio::time::Instant>,
    /// Tokio instant of the last *successful* `sync_from_backend`. Used by `ensure_synced` to
    /// enforce the staleness bound: state is considered fresh only while this is within
    /// `SYNC_STALENESS_BOUND`. Beyond it, a new sync is attempted; callers that arrive during a
    /// Redis outage will keep failing closed until connectivity is restored.
    last_synced_at: Option<tokio::time::Instant>,
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

/// Type alias to keep the `sync_gates` field type below clippy's complexity threshold.
type SyncGates = Mutex<HashMap<(String, usize), Arc<tokio::sync::Mutex<()>>>>;

/// Per-`(tenant, route)` certified-selection state. Cheap to share behind an `Arc`.
#[derive(Debug)]
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
    /// Random UUID for this process instance. Stamped on every `LogRecord::Pair` this replica
    /// writes so that `apply_log_record` can skip double-tallying health and ledger for own
    /// records (which were already tallied locally when the challenger call completed).
    replica_id: Uuid,
    /// Per-`(tenant, route_ix)` gate for coalescing concurrent cold-start syncs.
    ///
    /// All concurrent first-request callers for a fresh entry block on the same
    /// `tokio::sync::Mutex`. Only one calls the backend (LRANGE); subsequent waiters re-check
    /// `synced_once` and return immediately once the winner sets it. The outer `Mutex` protects
    /// the map; the inner async `Mutex` serialises the cold-start burst without holding the outer
    /// lock across an await.
    sync_gates: SyncGates,
}

impl Default for ExperimentMonitor {
    /// An inert monitor with no experiments configured: every lookup returns `false`/unknown, no
    /// log is attached, and no ledger is attached.  Used in demo/test paths that construct an
    /// `AppState` without a real `Config` (e.g. `Arc::default()`).
    fn default() -> Self {
        Self {
            params: HashMap::new(),
            unavailable: false,
            entries: Mutex::new(HashMap::new()),
            log: None,
            paired_ledger: None,
            replica_id: Uuid::now_v7(),
            sync_gates: Mutex::new(HashMap::new()),
        }
    }
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
            replica_id: Uuid::now_v7(),
            sync_gates: Mutex::default(),
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

    /// Perform one `sync_from_backend` the first time a `(tenant, route)` entry is used in this
    /// process. Returns `true` when the entry is synced (either it already was, or the sync just
    /// succeeded). Returns `false` when the log backend is unreachable; callers should fail closed
    /// for this request (serve incumbent / skip evaluation).
    ///
    /// **Why this is needed.** `build_paired_ledger` starts empty when `shared_log=true`, and
    /// `start_sync_task` only syncs entries already in the map. A brand-new replica (or a
    /// `(tenant, route)` pair not yet seen on this instance) would check `challenger_blocked` and
    /// `may_spend` against nothing, silently bypassing a cluster-level halt decision or the daily
    /// spend cap. One synchronous sync before the first check closes that window.
    ///
    /// **Burst coalescing.** A per-entry `tokio::sync::Mutex` gate serialises concurrent
    /// first-request callers so exactly one LRANGE is issued per burst. Subsequent waiters
    /// re-check `synced_once` after acquiring the gate and return immediately. `synced_once` is
    /// set only on success; on failure it stays false (so the next burst retries), but a warning
    /// is logged only once per entry to avoid spam during a persistent outage.
    ///
    /// Only the first request per `(tenant, route_ix)` per process pays one backend LRANGE.
    pub async fn ensure_synced(&self, tenant: &str, route_ix: usize, fingerprint: &str) -> bool {
        if self.log.is_none() {
            return true; // no log = always "synced" (local is authoritative)
        }

        // ── Quick path: inspect current state without acquiring the async gate ──────────────
        // Done first so the common steady-state case never touches sync_gates.
        {
            let mut map = self.lock();
            match self.entry(&mut map, tenant, route_ix) {
                None => return true, // no challenger on this route
                Some(e) => {
                    // State is fresh: last successful sync was within the staleness bound.
                    // 30 s = 6 × the 5 s background interval, so one missed tick is fine.
                    if e.last_synced_at
                        .is_some_and(|t| t.elapsed() <= SYNC_STALENESS_BOUND)
                    {
                        return true;
                    }
                    // Cooldown: a recent failure means Redis is likely still down.
                    // Return false *before* touching sync_gates to avoid lock contention
                    // during an outage (Issue 6a: fast-fail before acquiring the gate).
                    if e.sync_failed_at
                        .is_some_and(|t| t.elapsed() < SYNC_FAILURE_COOLDOWN)
                    {
                        return false;
                    }
                    // Otherwise: state is stale (or never synced) and not in cooldown.
                    // Fall through to the gated sync path below.
                }
            }
        }

        // Get or create the per-entry coalescing gate. We clone the Arc while holding the
        // outer (sync) lock, then drop the lock before awaiting the async gate.
        let gate = {
            let mut gates = self
                .sync_gates
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            gates
                .entry((tenant.to_owned(), route_ix))
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };

        // All concurrent callers block here on the same gate.
        // Only one calls the backend; the others re-check after the winner finishes.
        let _guard = gate.lock().await;

        // Re-check after acquiring the gate: a concurrent caller may have synced while we
        // waited, or the failure cooldown may now be active.
        {
            let map = self.lock();
            if let Some(entry) = map.get(tenant).and_then(|r| r.get(&route_ix)) {
                if entry
                    .last_synced_at
                    .is_some_and(|t| t.elapsed() <= SYNC_STALENESS_BOUND)
                {
                    return true;
                }
                if entry
                    .sync_failed_at
                    .is_some_and(|t| t.elapsed() < SYNC_FAILURE_COOLDOWN)
                {
                    return false;
                }
            }
        }

        // Perform the sync with a hard timeout so a hung Redis cannot stall live requests.
        // On timeout we fail closed (same as any backend error) and activate the cooldown.
        let sync_result = tokio::time::timeout(
            SYNC_TIMEOUT,
            self.sync_from_backend(tenant, route_ix, fingerprint),
        )
        .await;

        match sync_result {
            Ok(Ok(())) => {
                // sync_from_backend sets last_synced_at and synced_once on success.
                // Once synced, remove the gate so subsequent callers take the quick path
                // instead of re-acquiring an Arc that will never be needed again (Issue 6b).
                let mut gates = self
                    .sync_gates
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                gates.remove(&(tenant.to_owned(), route_ix));
                true
            }
            Ok(Err(e)) => {
                let mut map = self.lock();
                if let Some(entry) = self.entry(&mut map, tenant, route_ix) {
                    entry.sync_failed_at = Some(tokio::time::Instant::now());
                    if !entry.sync_warn_logged {
                        entry.sync_warn_logged = true;
                        tracing::warn!(
                            tenant,
                            route = route_ix,
                            error = %e,
                            "shared log unreachable; halt guard and spend cap may use stale \
                             state — serving incumbent only for this request"
                        );
                    }
                }
                false
            }
            Err(_elapsed) => {
                // Backend timed out: fail closed + activate cooldown.
                let mut map = self.lock();
                if let Some(entry) = self.entry(&mut map, tenant, route_ix) {
                    entry.sync_failed_at = Some(tokio::time::Instant::now());
                    if !entry.sync_warn_logged {
                        entry.sync_warn_logged = true;
                        tracing::warn!(
                            tenant,
                            route = route_ix,
                            "shared log sync timed out (>{} ms); serving incumbent only",
                            SYNC_TIMEOUT.as_millis()
                        );
                    }
                }
                false
            }
        }
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
                        applied_pair_ids: HashSet::new(),
                        remote_log_offset: 0,
                        synced_once: false,
                        sync_warn_logged: false,
                        sync_failed_at: None,
                        last_synced_at: None,
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
                incumbent_cost_usd: _,
                challenger_cost_usd,
                ch_error,
                ch_truncated,
                ts_unix,
                replica_id: record_replica_id,
                trace_id: record_trace_id,
            } => {
                let Some(e) = self.entry(map, tenant, route_ix) else {
                    return false;
                };
                // Dedup by trace_id: a pair whose append timed out after Redis applied it
                // may have already been counted locally (in record_pair's failure path or
                // replayed from SQLite at startup). Skip evidence if the id is already known.
                // Nil UUID (old records without trace_id) bypasses dedup — always applied.
                let already_applied =
                    !record_trace_id.is_nil() && e.applied_pair_ids.contains(record_trace_id);
                let before = e.experiment.status();
                if !already_applied {
                    if e.applied_pair_ids.len() >= MAX_TRACKED_SESSIONS {
                        // Saturated: skip evidence rather than evicting (evicting lets a pair
                        // count twice, breaking the pairing argument).
                        if !e.saturated_logged {
                            e.saturated_logged = true;
                            tracing::warn!(
                                tenant,
                                route = route_ix,
                                "experiment pair-id set is full; new pairs' evidence is skipped"
                            );
                        }
                    } else {
                        // Evidence (e-process wealth) is applied for ALL non-deduped records —
                        // own and remote — because in shared-log mode the log is the canonical
                        // source of evidence for all replicas.
                        if e.experiment.observe_pair(*x_inc, *x_ch) {
                            // The incumbent's cost is already counted on the serving path; only
                            // the off-path challenger call is new spend.
                            e.experiment
                                .observe_cost(Arm::Challenger, *challenger_cost_usd);
                        }
                        if !record_trace_id.is_nil() {
                            e.applied_pair_ids.insert(*record_trace_id);
                        }
                    }
                }
                // Health and spend for remote records are nested inside `!already_applied`:
                // if `already_applied` is true, the originating replica counted them locally
                // before restart (via SQLite replay in `monitor_from_traces` which rebuilds
                // health from persisted ch_error/ch_truncated and spend via build_paired_ledger).
                // Applying them again here would double-count health and spend.
                //
                // Skipping own records (is_own) prevents double-counting on the live path:
                // health and spend were tallied in `spawn_paired_challenger` before append.
                let is_own = *record_replica_id == self.replica_id;
                if !already_applied && !is_own {
                    if *ch_error || *ch_truncated > 0 {
                        e.health.challenger.errors += u64::from(*ch_error);
                        e.health.challenger.truncated += *ch_truncated;
                    }
                    if let Some(ledger) = &self.paired_ledger {
                        let now = jiff::Timestamp::now();
                        let pair_ts = jiff::Timestamp::from_second(*ts_unix)
                            .unwrap_or(jiff::Timestamp::UNIX_EPOCH);
                        ledger.seed(tenant, route_ix, *challenger_cost_usd, pair_ts, now);
                    }
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
    /// **Concurrent-call safety.** `record_pair` and `start_sync_task` both call this
    /// concurrently on the same `(tenant, route_ix)`. Both may snapshot the same
    /// `remote_log_offset` before either updates it, load the same slice from the backend,
    /// and then race to apply it. The fix is two-part:
    /// 1. After re-acquiring the lock, skip any record whose absolute index
    ///    `offset + i` is already covered by the current `remote_log_offset` (a concurrent
    ///    call advanced it while we were awaiting `load_since`).
    /// 2. Only ever advance `remote_log_offset` — never rewind it — so a slower
    ///    concurrent call that loaded a shorter slice cannot wind the cursor back.
    ///
    /// No-op (returns `Ok`) when no log backend is configured.
    /// Returns `Err` if the backend is unreachable; callers should treat a failure as
    /// "state unknown" and fail closed (do not serve the challenger for this request).
    pub async fn sync_from_backend(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
    ) -> Result<(), LogError> {
        let Some(log) = &self.log else {
            return Ok(());
        };
        // Snapshot offset before dropping the lock to await I/O.
        let offset = {
            let mut map = self.lock();
            self.entry(&mut map, tenant, route_ix)
                .map_or(0, |e| e.remote_log_offset)
        };
        // Lock is dropped here. Another sync may advance `remote_log_offset` while we await.
        let records = log
            .load_since(tenant, route_ix, fingerprint, offset)
            .await?;
        if records.is_empty() {
            // Successful read (log reachable), just nothing new. Mark synced so a background
            // sync spares the first explicit request from a cold-start LRANGE.
            let mut map = self.lock();
            if let Some(e) = map.get_mut(tenant).and_then(|r| r.get_mut(&route_ix)) {
                e.synced_once = true;
                e.last_synced_at = Some(tokio::time::Instant::now());
            }
            return Ok(());
        }
        let mut map = self.lock();
        for (i, opt_rec) in records.iter().enumerate() {
            let abs_ix = offset + i as u64;
            // Skip records already applied by a concurrent sync that ran ahead of us.
            let current_offset = map
                .get(tenant)
                .and_then(|r| r.get(&route_ix))
                .map_or(0, |e| e.remote_log_offset);
            if abs_ix < current_offset {
                continue;
            }
            // Malformed entries (None): advance the offset but apply nothing.
            let changed = if let Some(rec) = opt_rec {
                self.apply_log_record(&mut map, tenant, route_ix, fingerprint, rec)
            } else {
                false
            };
            // Monotonic advance: never rewind the offset even if a concurrent call loaded
            // a shorter prefix.  Update regardless of whether the record was applied (a
            // stale-fingerprint record still must not be re-read on the next sync).
            if let Some(e) = map.get_mut(tenant).and_then(|r| r.get_mut(&route_ix)) {
                e.remote_log_offset = e.remote_log_offset.max(abs_ix + 1);
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
        // Mark synced: a successful background sync spares the first explicit request from
        // a cold-start LRANGE and marks the state as fresh for the staleness bound.
        if let Some(e) = map.get_mut(tenant).and_then(|r| r.get_mut(&route_ix)) {
            e.synced_once = true;
            e.last_synced_at = Some(tokio::time::Instant::now());
        }
        Ok(())
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
        if let Err(e) = log
            .append(
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
            .await
        {
            tracing::warn!(
                tenant,
                route = route_ix,
                error = %e,
                "outcome log append failed; evidence will not reach other replicas"
            );
            // Fall back to local state so at least this replica counts the outcome.
            return self.observe_outcome(tenant, route_ix, fingerprint, arm, session_id, verdict);
        }
        // Pull in any records other replicas appended since last sync, plus the one we
        // just appended (so local state is always up-to-date after an append).
        // A sync failure is non-fatal: we'll catch up on the next background sync.
        let _ = self.sync_from_backend(tenant, route_ix, fingerprint).await;
        true
    }

    /// Apply a paired shadow result directly to local in-memory state.
    ///
    /// **Call only after the pair has been durably recorded** (written to SQLite as `NotInLog`).
    /// This ordering guarantee is what `spawn_paired_challenger` enforces: the SQLite row is the
    /// only durable record for local-only pairs, and the evidence must not exist in memory without
    /// it — otherwise a crash between the evidence apply and the write lets a halted challenger
    /// run again on restart (it has evidence but no persisted state to replay the halt from).
    ///
    /// Deduplicates by `trace_id` so a pair is never counted twice even if this method is called
    /// from both the local path and the log-sync path. `Uuid::nil()` bypasses dedup.
    ///
    /// Returns `true` if the pair was accepted by the underlying [`Experiment`].
    #[allow(clippy::too_many_arguments)]
    pub fn apply_pair_locally(
        &self,
        tenant: &str,
        route_ix: usize,
        fingerprint: &str,
        x_inc: f64,
        x_ch: f64,
        challenger_cost_usd: f64,
        trace_id: Uuid,
    ) -> bool {
        if self.fingerprint(route_ix) != Some(fingerprint) {
            return false;
        }
        let mut map = self.lock();
        let Some(e) = self.entry(&mut map, tenant, route_ix) else {
            return false;
        };
        // Cross-source dedup: skip if this trace_id was already counted (e.g., via a concurrent
        // log sync that saw the same record, or a previous local apply call).
        let already_applied = !trace_id.is_nil() && e.applied_pair_ids.contains(&trace_id);
        if already_applied {
            return false;
        }
        if e.applied_pair_ids.len() >= MAX_TRACKED_SESSIONS {
            if !e.saturated_logged {
                e.saturated_logged = true;
                tracing::warn!(
                    tenant,
                    route = route_ix,
                    "experiment pair-id set is full; new pairs' evidence is skipped"
                );
            }
            return false;
        }
        let before = e.experiment.status();
        let ok = e.experiment.observe_pair(x_inc, x_ch);
        if ok {
            // Without this the challenger's $/request reads n/a, so a certified cheaper
            // challenger could never be recommended for adoption.
            e.experiment
                .observe_cost(Arm::Challenger, challenger_cost_usd);
        }
        if !trace_id.is_nil() {
            e.applied_pair_ids.insert(trace_id);
        }
        if ok && before != e.experiment.status() {
            match e.experiment.status() {
                ExperimentStatus::Halted => tracing::error!(
                    tenant,
                    route = route_ix,
                    "EXPERIMENT HALTED: paired challenger worse than incumbent (local evidence)"
                ),
                ExperimentStatus::Certified => tracing::info!(
                    tenant,
                    route = route_ix,
                    "experiment certified (paired, local evidence): challenger within margin"
                ),
                ExperimentStatus::Undecided => {}
            }
            Self::publish(&map, route_ix);
        }
        ok
    }

    /// Try to append a paired shadow result to the shared log (if configured), then sync.
    ///
    /// **Durability contract for `spawn_paired_challenger`:**
    ///
    /// - Returns `true` → the record is in the shared log (the durable record). Evidence is
    ///   applied via `sync_from_backend` inside this call. Caller writes the SQLite row as
    ///   `InLog` after this returns (the row is just a startup-rebuild cache; it can follow
    ///   the log write safely).
    /// - Returns `false` → NOT in the log (no log configured, or the append failed). Evidence
    ///   is NOT applied here. The caller must write the SQLite row as `NotInLog` **first**,
    ///   then call [`Self::apply_pair_locally`] only on success. This ordering ensures the
    ///   durable record exists before evidence is in memory: a crash between the two would
    ///   leave no evidence, which is the safe direction (a missed pair only slows the
    ///   experiment; evidence without a durable record could mislead a halt).
    ///
    /// `ch_error`, `ch_truncated`, and `ts_unix` are forwarded into [`LogRecord::Pair`] so other
    /// replicas can rebuild challenger health and seed the paired-spend ledger.
    ///
    /// `trace_id` is used for cross-source dedup: if Redis applied the record but the append
    /// returned `Err` (timeout), the next sync sees the same trace_id and skips evidence it
    /// already counted locally. Pass `Uuid::nil()` in tests not exercising dedup.
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
        trace_id: Uuid,
    ) -> bool {
        if self.fingerprint(route_ix) != Some(fingerprint) {
            return false;
        }
        let Some(log) = self.log.as_ref().cloned() else {
            // No shared log: this replica is the only one; evidence is NOT applied here.
            // The caller (spawn_paired_challenger) writes SQLite first, then calls
            // apply_pair_locally so the durable record precedes the in-memory evidence.
            return false;
        };
        // Validate before appending: an invalid record would permanently pollute the log.
        if !x_inc.is_finite()
            || !(0.0..=1.0).contains(&x_inc)
            || !x_ch.is_finite()
            || !(0.0..=1.0).contains(&x_ch)
        {
            return false;
        }
        // Health and ledger were already tallied locally in `spawn_paired_challenger` (always,
        // regardless of log state). The caller's local debit is the single source of truth;
        // `apply_log_record` will skip re-tallying for own records (matched by replica_id).
        // An append failure means the evidence will not reach other replicas, which only delays
        // a decision — it does not affect local accounting.
        if let Err(e) = log
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
                    replica_id: self.replica_id,
                    trace_id,
                },
            )
            .await
        {
            tracing::warn!(
                tenant,
                route = route_ix,
                error = %e,
                "paired log append failed; evidence withheld until SQLite write succeeds \
                 (local spend and health were already tallied)"
            );
            // NOT in the log: caller writes SQLite (NotInLog) then calls apply_pair_locally.
            return false;
        }
        // Sync to apply own record immediately (so this replica sees its own evidence) and
        // to pick up any remote records. A sync failure is non-fatal.
        let _ = self.sync_from_backend(tenant, route_ix, fingerprint).await;
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
                    if let Err(e) = monitor
                        .sync_from_backend(&tenant, route_ix, &fingerprint)
                        .await
                    {
                        // Record the failure time so `ensure_synced` sees the backend as
                        // unavailable and transitions to stale → fail-closed once the
                        // staleness bound is exceeded. Without this update the staleness
                        // bound would tick down against `last_synced_at` from the last
                        // success but the cooldown would never activate, causing every
                        // `ensure_synced` caller to re-attempt the failing backend.
                        let mut map = monitor.lock();
                        if let Some(e_entry) =
                            map.get_mut(&tenant).and_then(|r| r.get_mut(&route_ix))
                        {
                            e_entry.sync_failed_at = Some(tokio::time::Instant::now());
                        }
                        tracing::warn!(
                            tenant,
                            route = route_ix,
                            error = %e,
                            "background experiment sync failed"
                        );
                    }
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
    //
    // After applying evidence, register the trace_id in applied_pair_ids so that when the
    // same pair later arrives from the shared log (e.g. an append that timed out but Redis
    // applied it) sync_from_backend skips it and does not double-count.
    for row in pairs {
        let mut map = monitor.lock();
        if let Some(e) = monitor.entry(&mut map, &row.tenant, row.route_ix) {
            if monitor.fingerprint(row.route_ix) == Some(row.fingerprint.as_str()) {
                if e.experiment.observe_pair(row.x_inc, row.x_ch) {
                    e.experiment
                        .observe_cost(Arm::Challenger, row.challenger_cost_usd);
                }
                // Register trace_id for cross-source dedup (prevents double-counting if the
                // same pair arrives from the shared log on the first sync_from_backend).
                if !row.trace_id.is_nil() {
                    e.applied_pair_ids.insert(row.trace_id);
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
            // With a shared log, the restart rules for evidence are (see InLogStatus docs):
            //   NotInLog → replay evidence locally (written during an outage; not in Redis).
            //   InLog    → do NOT replay evidence; the log delivers it on the first sync.
            //   Unknown  → do NOT replay evidence; we cannot know if it's in the log,
            //              so we conservatively withhold it (delays a decision — safe).
            // Without a shared log: all pairs replay normally (original behaviour).
            let offline_pairs: Vec<crate::store::ExperimentPairRow>;
            let replayed: &[crate::store::ExperimentPairRow] = if shared_log {
                offline_pairs = pairs
                    .iter()
                    .filter(|r| r.in_log == crate::store::InLogStatus::NotInLog)
                    .cloned()
                    .collect();
                &offline_pairs
            } else {
                &pairs
            };
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
            upstream_provider: String::new(),
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

        // Direct (no-log) path: record_pair returns false and does not apply evidence;
        // the caller uses apply_pair_locally after a successful SQLite write.
        let direct = ExperimentMonitor::from_config(&cfg);
        for _ in 0..3 {
            let ok = direct
                .record_pair(
                    "t",
                    0,
                    &fp_str,
                    1.0,
                    1.0,
                    0.10,
                    0.01,
                    false,
                    0,
                    0,
                    Uuid::nil(),
                )
                .await;
            assert!(
                !ok,
                "no log → record_pair returns false (evidence via apply_pair_locally)"
            );
            direct.apply_pair_locally("t", 0, &fp_str, 1.0, 1.0, 0.01, Uuid::nil());
        }
        close(challenger(&direct));

        let shared = Arc::new(InMemoryExperimentLog::new());
        let a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        let b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        for _ in 0..3 {
            a.record_pair(
                "t",
                0,
                &fp_str,
                1.0,
                1.0,
                0.10,
                0.01,
                false,
                0,
                0,
                Uuid::nil(),
            )
            .await;
        }
        let _ = b.sync_from_backend("t", 0, &fp_str).await;
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
                    in_log: crate::store::InLogStatus::InLog,
                },
            )
            .unwrap();
        }
        close(challenger(&build_monitor(&db.to_string_lossy(), &cfg)));

        // Restart with a shared log: the same three pairs are in the store AND the log. They
        // must count once, not twice (pairs carry no session to dedup on).
        let restarted = build_monitor_with(&db.to_string_lossy(), &cfg, true)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        let _ = restarted.sync_from_backend("t", 0, &fp_str).await;
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
        let _ = replica_b.sync_from_backend("t", 0, &fp_str).await;
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
        let _ = replica_b.sync_from_backend("t", 0, &fp_str).await;
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
        let _ = replica_a.sync_from_backend("t", 0, &fp_str).await;
        let _ = replica_b.sync_from_backend("t", 0, &fp_str).await;

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
        let _ = replica_b
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
        let _ = replica_a
            .sync_from_backend(&unique_tenant, 0, &fp_str)
            .await;
        let _ = replica_b
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
        let _ = replica_b
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
        let records = log
            .load_since(&tenant_b, 0, &fp_str, 0)
            .await
            .expect("load_since must not fail on a reachable Redis");
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
                replica_id,
                ..
            } => {
                assert!(!ch_error, "ch_error defaults to false");
                assert_eq!(ch_truncated, 0, "ch_truncated defaults to 0");
                assert_eq!(ts_unix, 0, "ts_unix defaults to 0");
                assert!((x_inc - 1.0).abs() < f64::EPSILON);
                assert!((x_ch - 0.5).abs() < f64::EPSILON);
                // Old records have no replica_id field → default nil UUID, treated as "remote"
                // by apply_log_record (health+ledger tallied for them, not skipped).
                assert_eq!(
                    replica_id,
                    uuid::Uuid::nil(),
                    "missing replica_id defaults to nil"
                );
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
                in_log: crate::store::InLogStatus::InLog,
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
    /// A log that is fully down: every append and load_since returns an error.
    #[derive(Debug)]
    struct DownLog;

    #[async_trait::async_trait]
    impl ExperimentLogBackend for DownLog {
        async fn append(&self, _: &str, _: usize, _: &str, _: LogRecord) -> Result<u64, LogError> {
            Err(LogError::Backend("simulated outage".to_owned()))
        }
        async fn load_since(
            &self,
            _: &str,
            _: usize,
            _: &str,
            _: u64,
        ) -> Result<Vec<Option<LogRecord>>, LogError> {
            Err(LogError::Backend("simulated outage".to_owned()))
        }
        async fn claim_session(&self, _: &str, _: usize, _: &str, _: &str) -> bool {
            false
        }
    }

    /// A log where append succeeds but load_since always fails (simulates a write-ok/read-fail
    /// scenario, e.g. Redis replica lag or a partial outage).
    #[derive(Debug)]
    struct AppendOnlyLog(InMemoryExperimentLog);

    #[async_trait::async_trait]
    impl ExperimentLogBackend for AppendOnlyLog {
        async fn append(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            record: LogRecord,
        ) -> Result<u64, LogError> {
            self.0.append(tenant, route_ix, fingerprint, record).await
        }
        async fn load_since(
            &self,
            _: &str,
            _: usize,
            _: &str,
            _: u64,
        ) -> Result<Vec<Option<LogRecord>>, LogError> {
            Err(LogError::Backend("simulated read failure".to_owned()))
        }
        async fn claim_session(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            session_id: &str,
        ) -> bool {
            self.0
                .claim_session(tenant, route_ix, fingerprint, session_id)
                .await
        }
    }

    /// With the shared log fully down, a paired challenger's spend is still counted against the cap
    /// and health is still tallied because `spawn_paired_challenger` debits/tallies locally before
    /// calling `record_pair`. The test simulates that local accounting path directly.
    #[tokio::test]
    async fn paired_spend_counts_against_the_cap_when_the_shared_log_is_down() {
        let ledger = Arc::new(crate::shadow::ShadowLedger::new());
        let m = ExperimentMonitor::from_config(&config())
            .with_log(Arc::new(DownLog) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger));
        let now = jiff::Timestamp::now();

        // Simulate what spawn_paired_challenger does: ALWAYS debit and tally health locally,
        // then call record_pair. The debit/health happen before record_pair regardless of
        // whether the log append succeeds.
        ledger.debit("t", 0, 0.7, now);
        m.observe_health("t", 0, &fp(), Arm::Challenger, true, 2);

        let accepted = m
            .record_pair(
                "t",
                0,
                &fp(),
                1.0,
                0.0,
                0.1,
                0.7,
                true,
                2,
                now.as_second(),
                Uuid::nil(),
            )
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

    /// When append succeeds but load_since fails, local spend (debited by spawn_paired_challenger)
    /// still enforces the cap. The pair record is in the backend (other replicas can read it on
    /// their next sync), but this replica cannot read it back immediately.
    #[tokio::test]
    async fn paired_spend_enforced_when_append_succeeds_but_read_fails() {
        let ledger = Arc::new(crate::shadow::ShadowLedger::new());
        let m = ExperimentMonitor::from_config(&config())
            .with_log(Arc::new(AppendOnlyLog(InMemoryExperimentLog::new()))
                as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger));
        let now = jiff::Timestamp::now();

        // Simulate spawn_paired_challenger: local debit + health first.
        ledger.debit("t", 0, 0.6, now);
        m.observe_health("t", 0, &fp(), Arm::Challenger, false, 1);

        // record_pair: append succeeds, but sync (load_since) fails.
        let accepted = m
            .record_pair(
                "t",
                0,
                &fp(),
                1.0,
                1.0,
                0.1,
                0.6,
                false,
                1,
                now.as_second(),
                Uuid::nil(),
            )
            .await;
        // The pair was appended to the log (returns false only on append failure).
        // With AppendOnlyLog, append succeeds but sync returns Err, so record_pair still
        // returns false (sync failure propagates the same as before... actually no).
        // record_pair returns true when append succeeds, regardless of sync outcome.
        // Actually: record_pair ignores the sync error (let _ = sync...) and returns true.
        assert!(accepted, "pair was appended to log");

        // Local spend (from the explicit debit above) enforces the cap.
        assert!(
            (ledger.spent_today("t", 0, now) - 0.6).abs() < 1e-9,
            "local spend = 0.6"
        );
        assert!(
            !ledger.may_spend("t", 0, 0.5, now),
            "cap enforced by local debit even when read-back fails"
        );
        let h = m.reports()[0].health.challenger;
        assert_eq!((h.errors, h.truncated), (0, 1), "{h:?}");
    }

    /// Two replicas sharing a log count each pair's health and spend exactly once.
    ///
    /// `spawn_paired_challenger` ALWAYS debits the local ledger and tallies health, then calls
    /// `record_pair`. `apply_log_record` skips health and ledger for records bearing the replica's
    /// own `replica_id` (preventing double-count on the originating node) but applies them for
    /// remote records (different `replica_id`).
    ///
    /// This test simulates that contract directly:
    ///   - A does explicit `debit` + `observe_health` (what `spawn_paired_challenger` does), then
    ///     calls `record_pair`.  A's counts stay at the values set by those local calls.
    ///   - B calls `sync_from_backend`.  B's counts come from `apply_log_record` (remote record),
    ///     matching A's locally-set values exactly.
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

        // Simulate what spawn_paired_challenger does: ALWAYS debit and tally health locally,
        // regardless of whether the shared log is up.  record_pair only propagates evidence to
        // other replicas — it does not re-tally on the originating replica.
        ledger_a.debit("t", 0, 0.8, now_ts);
        replica_a.observe_health("t", 0, &fp_str, Arm::Challenger, true, 1);

        let accepted = replica_a
            .record_pair(
                "t",
                0,
                &fp_str,
                1.0,
                0.5,
                0.1,
                0.8,
                true,
                1,
                now_secs,
                Uuid::nil(),
            )
            .await;
        assert!(accepted, "pair must be accepted");

        // A's health comes from the explicit observe_health above, counted exactly once.
        // apply_log_record skips own records (same replica_id), so no double-count.
        let reps_a = replica_a.reports();
        assert_eq!(
            reps_a[0].health.challenger.errors, 1,
            "A: challenger error counted once, not zero or doubled"
        );
        assert_eq!(
            reps_a[0].health.challenger.truncated, 1,
            "A: challenger truncated counted once"
        );

        // A's ledger comes from the explicit debit above.
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

        // Sync B from the shared log: B sees A's record as "remote" (different replica_id),
        // so apply_log_record tallies health and seeds the ledger for B.
        let _ = replica_b.sync_from_backend("t", 0, &fp_str).await;

        let reps_b = replica_b.reports();
        assert_eq!(
            reps_b[0].health.challenger.errors, 1,
            "B: challenger error visible after sync"
        );
        let b_spent = ledger_b.spent_today("t", 0, now_ts);
        assert!(
            (b_spent - 0.8).abs() < 1e-12,
            "B: ledger reflects A's spend after sync; got {b_spent}"
        );

        // Add a second pair (via A's local path then record_pair) to push cluster total over cap.
        ledger_a.debit("t", 0, 0.3, now_ts);
        replica_a.observe_health("t", 0, &fp_str, Arm::Challenger, false, 0);
        replica_a
            .record_pair(
                "t",
                0,
                &fp_str,
                1.0,
                0.5,
                0.1,
                0.3,
                false,
                0,
                now_secs,
                Uuid::nil(),
            )
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
        let _ = replica_b.sync_from_backend("t", 0, &fp_str).await;
        assert!(
            !ledger_b.may_spend("t", 0, 1.0, now_ts),
            "B: cap exhausted from cluster spend"
        );
    }

    // ── Concurrent sync idempotency (Issue 1 fix) ────────────────────────────────────────────

    /// A log backend that yields after returning records from `load_since`, allowing the tokio
    /// scheduler to run other tasks between the data-load and the re-acquire of the monitor lock.
    /// This deterministically surfaces the "two concurrent syncs, same slice" race.
    #[derive(Debug)]
    struct YieldingLog(InMemoryExperimentLog);

    #[async_trait::async_trait]
    impl ExperimentLogBackend for YieldingLog {
        async fn append(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            record: LogRecord,
        ) -> Result<u64, LogError> {
            self.0.append(tenant, route_ix, fingerprint, record).await
        }

        async fn load_since(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            offset: u64,
        ) -> Result<Vec<Option<LogRecord>>, LogError> {
            let records = self
                .0
                .load_since(tenant, route_ix, fingerprint, offset)
                .await?;
            // Yield control so a concurrently-polled task can enter between load and re-lock.
            tokio::task::yield_now().await;
            Ok(records)
        }

        async fn claim_session(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            session_id: &str,
        ) -> bool {
            self.0
                .claim_session(tenant, route_ix, fingerprint, session_id)
                .await
        }
    }

    /// Two concurrent `sync_from_backend` calls on the same monitor, both loading the same 3
    /// records from offset=0, must result in exactly 3 pairs / 3 errors / 3 truncated — never 6.
    ///
    /// The `YieldingLog` above forces actual task interleaving: both syncs snapshot `offset=0`,
    /// call `load_since` (which yields), and then race to apply the same slice. Without the
    /// `abs_ix < current_offset` guard and the monotonic-advance, the second sync would re-apply
    /// all 3 records and produce 6 pairs (doubling the e-process wealth — a correctness violation).
    ///
    /// Mutation test: removing the `if abs_ix < current_offset { continue; }` guard makes the
    /// second sync apply records 0–2 again. With `ch_error=true` per record, errors go from 3 to
    /// 6 and the `assert_eq!(3)` below fails.
    #[tokio::test]
    async fn concurrent_sync_does_not_double_count_pairs() {
        let shared = Arc::new(YieldingLog(InMemoryExperimentLog::new()));
        let cfg = config();
        let fp_str = fp();

        // Append 3 pairs directly (bypassing the monitor, so offset stays at 0 on the monitor).
        for _ in 0..3 {
            let _ = shared
                .append(
                    "t",
                    0,
                    &fp_str,
                    LogRecord::Pair {
                        x_inc: 1.0,
                        x_ch: 1.0,
                        incumbent_cost_usd: 0.10,
                        challenger_cost_usd: 0.05,
                        ch_error: true,
                        ch_truncated: 1,
                        ts_unix: 0,
                        // nil UUID: simulates records from an older replica; treated as remote
                        // by apply_log_record (health+ledger are tallied here, not locally).
                        replica_id: uuid::Uuid::nil(),
                        // nil: no per-pair dedup for this test.
                        trace_id: uuid::Uuid::nil(),
                    },
                )
                .await;
        }

        let monitor = Arc::new(
            ExperimentMonitor::from_config(&cfg)
                .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>),
        );

        // Both tasks snapshot `offset=0`, call load_since (which yields to let the other task
        // also snapshot offset=0 and call load_since), then race to apply the same 3 records.
        let m1 = Arc::clone(&monitor);
        let m2 = Arc::clone(&monitor);
        let fp1 = fp_str.clone();
        let fp2 = fp_str.clone();
        tokio::join!(
            async move {
                let _ = m1.sync_from_backend("t", 0, &fp1).await;
            },
            async move {
                let _ = m2.sync_from_backend("t", 0, &fp2).await;
            },
        );

        let snaps = monitor.reports();
        assert_eq!(snaps.len(), 1);
        let snap = &snaps[0];
        assert_eq!(
            snap.report.pairs, 3,
            "pairs must be 3, not double-counted: got {}",
            snap.report.pairs
        );
        assert_eq!(
            snap.health.challenger.errors, 3,
            "errors must be 3, not double-counted: got {}",
            snap.health.challenger.errors
        );
        assert_eq!(
            snap.health.challenger.truncated, 3,
            "truncated must be 3, not double-counted: got {}",
            snap.health.challenger.truncated
        );
    }

    // ── ensure_synced cold-start guard (Issue 2 fix) ─────────────────────────────────────────

    /// A replica whose map is empty (cold start) but whose shared log already shows today's spend
    /// at the cap must refuse a new evaluation after `ensure_synced`.
    ///
    /// Mutation test: removing `ensure_synced` or making it a no-op means B starts with an empty
    /// ledger, `may_spend` returns true even though the cap is already exhausted, and the
    /// `assert!(!may_spend_after_sync)` assertion fails.
    #[tokio::test]
    async fn ensure_synced_cold_start_respects_cluster_cap() {
        let shared = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();
        let now_ts = jiff::Timestamp::now();
        let cap = 0.5_f64;

        let ledger_a = Arc::new(crate::shadow::ShadowLedger::new());
        // Replica A already spent the full cap.
        // Simulate spawn_paired_challenger: debit locally first, then record to the shared log.
        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger_a));
        ledger_a.debit("t", 0, cap, now_ts);
        replica_a
            .record_pair(
                "t",
                0,
                &fp_str,
                1.0,
                1.0,
                0.10,
                cap,
                false,
                0,
                now_ts.as_second(),
                Uuid::nil(),
            )
            .await;
        assert!(
            !ledger_a.may_spend("t", 0, cap, now_ts),
            "A: cap exhausted after recording one pair"
        );

        // Replica B starts with an empty map — it has never seen this (tenant, route).
        let ledger_b = Arc::new(crate::shadow::ShadowLedger::new());
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger_b));

        // Before ensure_synced: B's ledger is empty → `may_spend` would incorrectly allow spend.
        assert!(
            ledger_b.may_spend("t", 0, cap, now_ts),
            "B: before ensure_synced, empty ledger allows spend (expected)"
        );

        // After ensure_synced: B syncs from the log and seeds its ledger from A's pair.
        // ensure_synced returns true on success; the caller (proxy.rs) uses this to fail-closed.
        let synced = replica_b.ensure_synced("t", 0, &fp_str).await;
        assert!(
            synced,
            "ensure_synced must return true when the log is healthy"
        );
        let may_spend_after_sync = ledger_b.may_spend("t", 0, cap, now_ts);
        assert!(
            !may_spend_after_sync,
            "B: after ensure_synced, cluster spend at cap — must refuse"
        );
    }

    /// A replica whose log already shows a halted experiment must block the challenger after
    /// `ensure_synced`, even though it has never seen the experiment locally.
    ///
    /// Mutation test: removing `ensure_synced` leaves B with `Undecided` (empty map) after the
    /// check. `challenger_blocked` returns `false` and the second `assert!` fails.
    /// Two requests arriving together on a cold replica both see the cluster halt by the time
    /// `ensure_synced` returns: the second must not slip through while the first is still syncing.
    #[tokio::test]
    async fn ensure_synced_burst_waits_for_synced_state() {
        let shared = Arc::new(YieldingLog(InMemoryExperimentLog::new()));
        let cfg = config();
        let fp_str = fp();
        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
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
                break;
            }
        }
        assert_eq!(replica_a.status("t", 0), ExperimentStatus::Halted);

        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);
        let (first, second) = tokio::join!(
            async {
                replica_b.ensure_synced("t", 0, &fp_str).await;
                replica_b.challenger_blocked("t", 0)
            },
            async {
                replica_b.ensure_synced("t", 0, &fp_str).await;
                replica_b.challenger_blocked("t", 0)
            },
        );
        assert!(
            first && second,
            "both burst requests see the halt: {first} {second}"
        );
    }

    #[tokio::test]
    async fn ensure_synced_cold_start_respects_cluster_halt() {
        let shared = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();

        let replica_a = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);

        // Drive experiment to Halted on replica A.
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

        // Replica B: cold start — empty map.
        let replica_b = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>);

        // Before ensure_synced: B does not know about the halt.
        assert!(
            !replica_b.challenger_blocked("t", 0),
            "B: before ensure_synced, halt unknown (expected)"
        );

        // After ensure_synced: B reads the log and sees the Halted status.
        let synced = replica_b.ensure_synced("t", 0, &fp_str).await;
        assert!(synced, "ensure_synced must succeed with healthy log");
        assert!(
            replica_b.challenger_blocked("t", 0),
            "B: after ensure_synced, halt must block the challenger"
        );
    }

    // ── Issue 1: restart with shared log replays offline pairs ───────────────────────────────

    /// On restart with a shared log, `in_log=false` rows (written during a Redis outage) are
    /// replayed locally into the monitor. `in_log=true` rows are NOT replayed locally — they
    /// come from the log on the first `sync_from_backend` and must not be double-counted.
    #[tokio::test]
    async fn restart_with_shared_log_replays_offline_pairs_and_skips_log_pairs() {
        let db =
            std::env::temp_dir().join(format!("firstpass-restart-{}.db", uuid::Uuid::now_v7()));
        // Initialize the schema (migrate) before calling append_experiment_pair directly.
        let (tx, handle) = crate::store::open(&db).unwrap();
        drop(tx);
        handle.await.unwrap();
        let cfg = config();
        let fp_str = fp();
        let now_ts = jiff::Timestamp::now();

        // NotInLog: written during a Redis outage; must be replayed on restart.
        crate::store::append_experiment_pair(
            &db,
            &crate::store::ExperimentPairRow {
                tenant: "t".to_owned(),
                route_ix: 0,
                fingerprint: fp_str.clone(),
                trace_id: uuid::Uuid::now_v7(),
                x_inc: 1.0,
                x_ch: 0.0,
                challenger_cost_usd: 0.5,
                ch_error: false,
                ch_truncated: 0,
                ts: now_ts,
                in_log: crate::store::InLogStatus::NotInLog,
            },
        )
        .unwrap();

        // InLog: already in Redis; must NOT be replayed locally (comes from sync).
        crate::store::append_experiment_pair(
            &db,
            &crate::store::ExperimentPairRow {
                tenant: "t".to_owned(),
                route_ix: 0,
                fingerprint: fp_str.clone(),
                trace_id: uuid::Uuid::now_v7(),
                x_inc: 1.0,
                x_ch: 0.0,
                challenger_cost_usd: 0.3,
                ch_error: false,
                ch_truncated: 0,
                ts: now_ts,
                in_log: crate::store::InLogStatus::InLog,
            },
        )
        .unwrap();

        // Restart with shared_log=true: only the offline row is replayed locally.
        let m = build_monitor_with(&db.to_string_lossy(), &cfg, true);
        let reps = m.reports();
        assert_eq!(
            reps[0].report.pairs, 1,
            "only in_log=false row replayed; got {}",
            reps[0].report.pairs
        );

        // Restart with shared_log=false: both rows are replayed (original behaviour).
        let m2 = build_monitor_with(&db.to_string_lossy(), &cfg, false);
        let reps2 = m2.reports();
        assert_eq!(
            reps2[0].report.pairs, 2,
            "shared_log=false replays all rows; got {}",
            reps2[0].report.pairs
        );

        let _ = std::fs::remove_file(&db);
    }

    // ── Issue 2: ensure_synced failure cooldown ───────────────────────────────────────────────

    /// A log backend whose `load_since` always fails and counts how many times it is called.
    #[derive(Debug)]
    struct CountingFailLog {
        load_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ExperimentLogBackend for CountingFailLog {
        async fn append(&self, _: &str, _: usize, _: &str, _: LogRecord) -> Result<u64, LogError> {
            Err(LogError::Backend("down".to_owned()))
        }
        async fn load_since(
            &self,
            _: &str,
            _: usize,
            _: &str,
            _: u64,
        ) -> Result<Vec<Option<LogRecord>>, LogError> {
            self.load_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(LogError::Backend("down".to_owned()))
        }
        async fn claim_session(&self, _: &str, _: usize, _: &str, _: &str) -> bool {
            false
        }
    }

    /// A burst of concurrent `ensure_synced` calls when the backend always fails must issue
    /// exactly one backend read: the coalescing gate serialises callers, the first one fails
    /// and sets the cooldown, and subsequent callers return false immediately without retrying.
    #[tokio::test]
    async fn ensure_synced_failure_burst_issues_exactly_one_read() {
        let load_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let m = Arc::new(ExperimentMonitor::from_config(&config()).with_log(Arc::new(
            CountingFailLog {
                load_count: Arc::clone(&load_count),
            },
        )
            as Arc<dyn ExperimentLogBackend>));
        let fp_str = fp();

        // Concurrent burst of 10 callers on the same (tenant, route).
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let m = Arc::clone(&m);
                let fp_str = fp_str.clone();
                tokio::spawn(async move { m.ensure_synced("t", 0, &fp_str).await })
            })
            .collect();
        for h in handles {
            let result = h.await.unwrap();
            assert!(!result, "every caller gets false on backend failure");
        }

        assert_eq!(
            load_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "exactly one backend read issued during a failure burst"
        );
    }

    /// A second call within the cooldown window issues no additional backend read.
    #[tokio::test]
    async fn ensure_synced_cooldown_skips_retry_within_window() {
        let load_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let m = ExperimentMonitor::from_config(&config())
            .with_log(Arc::new(CountingFailLog {
                load_count: Arc::clone(&load_count),
            }) as Arc<dyn ExperimentLogBackend>);
        let fp_str = fp();

        // First call: fails, sets cooldown.
        assert!(!m.ensure_synced("t", 0, &fp_str).await);
        assert_eq!(load_count.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Second call within cooldown: returns false immediately, no backend read.
        assert!(!m.ensure_synced("t", 0, &fp_str).await);
        assert_eq!(
            load_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "no additional read within the cooldown window"
        );
    }

    /// After the cooldown expires a retry is attempted (exactly one more read).
    #[tokio::test]
    async fn ensure_synced_cooldown_allows_retry_after_expiry() {
        tokio::time::pause();
        let load_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let m = ExperimentMonitor::from_config(&config())
            .with_log(Arc::new(CountingFailLog {
                load_count: Arc::clone(&load_count),
            }) as Arc<dyn ExperimentLogBackend>);
        let fp_str = fp();

        // First call: fails, sets cooldown.
        assert!(!m.ensure_synced("t", 0, &fp_str).await);
        assert_eq!(load_count.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Advance past the cooldown.
        tokio::time::advance(SYNC_FAILURE_COOLDOWN + std::time::Duration::from_millis(1)).await;

        // Next call: cooldown expired, issues one more read.
        assert!(!m.ensure_synced("t", 0, &fp_str).await);
        assert_eq!(
            load_count.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "one more read after the cooldown expires"
        );
    }

    // ── Durability ordering: SQLite precedes local evidence (Issue 1) ────────────────────────

    /// `record_pair` returns false when the log is down and does NOT apply evidence.
    /// The caller (`spawn_paired_challenger`) writes SQLite first, then calls
    /// `apply_pair_locally`. Here we verify the contract: `record_pair` returns false and
    /// the experiment is still Undecided; only `apply_pair_locally` advances the state.
    #[tokio::test]
    async fn record_pair_does_not_apply_evidence_on_append_failure_caller_does() {
        let m = ExperimentMonitor::from_config(&config())
            .with_log(Arc::new(DownLog) as Arc<dyn ExperimentLogBackend>);
        let fp_str = fp();

        let mut pairs_applied = 0u32;
        for _ in 0..500 {
            if m.challenger_blocked("t", 0) {
                break;
            }
            // record_pair returns false and does NOT apply evidence.
            let accepted = m
                .record_pair(
                    "t",
                    0,
                    &fp_str,
                    1.0,
                    0.0,
                    0.1,
                    0.01,
                    false,
                    0,
                    0,
                    Uuid::nil(),
                )
                .await;
            assert!(!accepted, "append failing log returns false");
            // Caller writes SQLite (not modelled here), then applies evidence.
            m.apply_pair_locally("t", 0, &fp_str, 1.0, 0.0, 0.01, Uuid::nil());
            pairs_applied += 1;
        }

        // Evidence was applied locally via apply_pair_locally: enough losing pairs halted.
        assert!(
            m.challenger_blocked("t", 0),
            "experiment must halt locally after {pairs_applied} pairs via apply_pair_locally"
        );
        assert!(
            m.reports()[0].report.pairs > 0,
            "pairs counted locally via apply_pair_locally"
        );
    }

    /// `record_pair` in no-log mode returns false and leaves the experiment Undecided.
    /// Only `apply_pair_locally` advances state. Mutation check: removing `apply_pair_locally`
    /// keeps the experiment Undecided and the assertion fires.
    #[tokio::test]
    async fn record_pair_no_log_does_not_apply_evidence() {
        let m = ExperimentMonitor::from_config(&config()); // no log
        let fp_str = fp();

        // Call record_pair many times — should never advance evidence without apply_pair_locally.
        for _ in 0..10 {
            let ok = m
                .record_pair(
                    "t",
                    0,
                    &fp_str,
                    1.0,
                    0.0,
                    0.1,
                    0.01,
                    false,
                    0,
                    0,
                    Uuid::nil(),
                )
                .await;
            assert!(!ok, "no log → record_pair returns false");
        }
        // No entry was created: record_pair in no-log mode returns immediately.
        assert_eq!(
            m.reports().len(),
            0,
            "record_pair in no-log mode must not create an entry or advance evidence"
        );

        // Now apply via apply_pair_locally — entry created and evidence advances.
        m.apply_pair_locally("t", 0, &fp_str, 1.0, 0.0, 0.01, Uuid::nil());
        assert_eq!(
            m.reports()[0].report.pairs,
            1,
            "apply_pair_locally counted the pair"
        );
    }

    // ── Issue 1b: Unknown tri-state for legacy rows ───────────────────────────────────────────

    /// Rows migrated from a schema without `in_log` get `InLogStatus::Unknown` (DEFAULT 2).
    /// With shared_log=true, Unknown rows contribute spend (cap is stricter — safe) but NOT
    /// evidence (withholding only delays a decision — safe).
    #[tokio::test]
    async fn unknown_rows_contribute_spend_but_not_evidence_with_shared_log() {
        use crate::store::InLogStatus;
        let db =
            std::env::temp_dir().join(format!("firstpass-unknown-{}.db", uuid::Uuid::now_v7()));
        let (tx, handle) = crate::store::open(&db).unwrap();
        drop(tx);
        handle.await.unwrap();
        let cfg = config();
        let fp_str = fp();
        let now_ts = jiff::Timestamp::now();

        // One Unknown row (legacy) and one NotInLog row (outage).
        for (in_log, x_ch) in [
            (InLogStatus::Unknown, 0.8_f64),
            (InLogStatus::NotInLog, 0.6),
        ] {
            crate::store::append_experiment_pair(
                &db,
                &crate::store::ExperimentPairRow {
                    tenant: "t".to_owned(),
                    route_ix: 0,
                    fingerprint: fp_str.clone(),
                    trace_id: uuid::Uuid::now_v7(),
                    x_inc: 1.0,
                    x_ch,
                    challenger_cost_usd: 0.1,
                    ch_error: false,
                    ch_truncated: 0,
                    ts: now_ts,
                    in_log,
                },
            )
            .unwrap();
        }

        // shared_log=true: only NotInLog row contributes evidence (1 pair); Unknown withheld.
        let m = build_monitor_with(&db.to_string_lossy(), &cfg, true);
        let reps = m.reports();
        assert_eq!(
            reps[0].report.pairs, 1,
            "shared_log=true: only NotInLog evidence replayed; got {}",
            reps[0].report.pairs
        );

        // shared_log=false: both rows contribute evidence (2 pairs).
        let m2 = build_monitor_with(&db.to_string_lossy(), &cfg, false);
        let reps2 = m2.reports();
        assert_eq!(
            reps2[0].report.pairs, 2,
            "shared_log=false: all rows replayed; got {}",
            reps2[0].report.pairs
        );

        let _ = std::fs::remove_file(&db);
    }

    // ── Issue 2: load_since preserves positions of malformed entries ─────────────────────────

    /// A log backend that injects a None (malformed) entry between two valid ones.
    /// Position-preserving: the returned vec has length 3 (valid, None, valid).
    #[derive(Debug)]
    struct MalformedMiddleLog {
        inner: InMemoryExperimentLog,
    }

    #[async_trait::async_trait]
    impl ExperimentLogBackend for MalformedMiddleLog {
        async fn append(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            record: LogRecord,
        ) -> Result<u64, LogError> {
            self.inner
                .append(tenant, route_ix, fingerprint, record)
                .await
        }

        async fn load_since(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            offset: u64,
        ) -> Result<Vec<Option<LogRecord>>, LogError> {
            // Call inner (which returns Vec<Option<LogRecord>> with all Some).
            let mut records = self
                .inner
                .load_since(tenant, route_ix, fingerprint, offset)
                .await?;
            // Inject None at index 1 (if the list has at least 2 elements) to simulate a
            // malformed entry. This preserves list length so offset arithmetic stays correct.
            if records.len() >= 2 {
                records[1] = None;
            }
            Ok(records)
        }

        async fn claim_session(&self, _: &str, _: usize, _: &str, _: &str) -> bool {
            false
        }
    }

    /// A log with a malformed entry at position 1 must:
    /// 1. Apply position 0 and position 2 exactly once each (None at position 1 is skipped).
    /// 2. After two consecutive syncs, `remote_log_offset` == 3 (all positions consumed).
    /// 3. A third sync reads nothing new (offset already at 3).
    #[tokio::test]
    async fn malformed_log_entry_advances_offset_and_valid_entries_applied_once() {
        let log = Arc::new(MalformedMiddleLog {
            inner: InMemoryExperimentLog::new(),
        });
        let cfg = config();
        let fp_str = fp();

        // Append 3 pairs directly to the inner log (records 0, 1, 2).
        for _ in 0..3 {
            let _ = log
                .inner
                .append(
                    "t",
                    0,
                    &fp_str,
                    LogRecord::Pair {
                        x_inc: 1.0,
                        x_ch: 1.0,
                        incumbent_cost_usd: 0.1,
                        challenger_cost_usd: 0.05,
                        ch_error: false,
                        ch_truncated: 0,
                        ts_unix: 0,
                        replica_id: uuid::Uuid::nil(),
                        trace_id: uuid::Uuid::nil(),
                    },
                )
                .await;
        }
        // The MalformedMiddleLog replaces position 1 with None.

        let m = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);

        // First sync: applies positions 0 and 2 (valid), skips position 1 (None).
        let _ = m.sync_from_backend("t", 0, &fp_str).await;
        let pairs_after_first = m.reports()[0].report.pairs;
        // 2 valid entries applied (positions 0 and 2); position 1 consumed but not counted.
        assert_eq!(
            pairs_after_first, 2,
            "first sync: 2 valid entries applied, got {pairs_after_first}"
        );

        // Second sync: offset is already at 3, nothing new.
        let _ = m.sync_from_backend("t", 0, &fp_str).await;
        let pairs_after_second = m.reports()[0].report.pairs;
        assert_eq!(
            pairs_after_second, 2,
            "second sync: no re-application, pairs still 2, got {pairs_after_second}"
        );

        // Verify offset equals list length: offset must be 3 (all 3 positions consumed).
        let offset = {
            let map = m.lock();
            map.get("t")
                .and_then(|r| r.get(&0))
                .map_or(0, |e| e.remote_log_offset)
        };
        assert_eq!(
            offset, 3,
            "offset must equal list length after consuming all positions"
        );
    }

    // ── Issue 3: trace_id dedup prevents double-counting on append-timeout ───────────────────

    /// A backend that appends successfully (the record IS in the log) but always returns Err
    /// from `append` (simulates an append that timed out after Redis applied it).
    #[derive(Debug)]
    struct PushThenErrorLog {
        inner: InMemoryExperimentLog,
    }

    #[async_trait::async_trait]
    impl ExperimentLogBackend for PushThenErrorLog {
        async fn append(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            record: LogRecord,
        ) -> Result<u64, LogError> {
            // Write the record to the inner log (it IS there).
            let _ = self
                .inner
                .append(tenant, route_ix, fingerprint, record)
                .await;
            // But report failure to the caller (timeout simulation).
            Err(LogError::Backend("timeout after write".to_owned()))
        }

        async fn load_since(
            &self,
            tenant: &str,
            route_ix: usize,
            fingerprint: &str,
            offset: u64,
        ) -> Result<Vec<Option<LogRecord>>, LogError> {
            self.inner
                .load_since(tenant, route_ix, fingerprint, offset)
                .await
        }

        async fn claim_session(&self, _: &str, _: usize, _: &str, _: &str) -> bool {
            false
        }
    }

    /// When `append` times out after Redis applied it, the evidence must count exactly once:
    /// - Caller writes SQLite (NotInLog) then calls `apply_pair_locally`.
    /// - `sync_from_backend` sees the same trace_id in the log and skips the evidence dedup.
    ///
    /// Also covers the restart path (Issue 3): health and ledger are only applied once even
    /// when the record was seen at startup from SQLite and later from the log.
    #[tokio::test]
    async fn trace_id_dedup_prevents_double_count_on_append_timeout() {
        let log = Arc::new(PushThenErrorLog {
            inner: InMemoryExperimentLog::new(),
        });
        let cfg = config();
        let fp_str = fp();
        let tid = Uuid::now_v7();
        let now_ts = jiff::Timestamp::now();

        let ledger = Arc::new(crate::shadow::ShadowLedger::new());
        let m = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger));

        // record_pair: append fails (Err) but the record IS in the log.
        // With the new durability ordering, record_pair returns false and does NOT
        // apply evidence — the caller (spawn_paired_challenger) does it after SQLite write.
        let appended = m
            .record_pair(
                "t",
                0,
                &fp_str,
                1.0,
                1.0,
                0.1,
                0.05,
                false,
                1,
                now_ts.as_second(),
                tid,
            )
            .await;
        assert!(!appended, "append returned Err → record_pair returns false");
        // record_pair returns false and does NOT create an Entry or apply evidence.
        // The entry is created lazily by apply_pair_locally or observe_* calls.
        assert_eq!(
            m.reports().len(),
            0,
            "no entry created yet by record_pair alone"
        );

        // Caller simulates the SQLite write then calls apply_pair_locally.
        // Also simulates spawn_paired_challenger's local health+ledger tally.
        ledger.debit("t", 0, 0.05, now_ts);
        m.observe_health("t", 0, &fp_str, Arm::Challenger, false, 1);
        m.apply_pair_locally("t", 0, &fp_str, 1.0, 1.0, 0.05, tid);
        assert_eq!(
            m.reports()[0].report.pairs,
            1,
            "evidence applied via apply_pair_locally"
        );

        // Sync: the record is in the log (PushThenErrorLog wrote it). Without dedup,
        // apply_log_record would count evidence + health + ledger again.  With dedup:
        // - evidence is skipped (trace_id already in applied_pair_ids)
        // - health and ledger are also skipped (Issue 3: nested under !already_applied)
        let _ = m.sync_from_backend("t", 0, &fp_str).await;
        assert_eq!(
            m.reports()[0].report.pairs,
            1,
            "after sync: pairs still 1 (trace_id dedup prevented double-count)"
        );
        let h = m.reports()[0].health.challenger;
        assert_eq!(
            (h.errors, h.truncated),
            (0, 1),
            "health counted exactly once (not re-applied by apply_log_record)"
        );
        assert!(
            (ledger.spent_today("t", 0, now_ts) - 0.05).abs() < 1e-9,
            "ledger spend counted exactly once (not re-seeded by apply_log_record)"
        );
    }

    // ── Issue 2: Staleness bound ──────────────────────────────────────────────────────────────

    /// `ensure_synced` returns true while the last sync was within `SYNC_STALENESS_BOUND`,
    /// and returns false (and attempts a re-sync) once the bound is exceeded.
    ///
    /// Uses `tokio::time::pause` to advance time without real I/O.
    #[tokio::test]
    async fn ensure_synced_stale_after_bound_exceeded() {
        tokio::time::pause();
        let shared = Arc::new(InMemoryExperimentLog::new());
        let m = Arc::new(
            ExperimentMonitor::from_config(&config())
                .with_log(Arc::clone(&shared) as Arc<dyn ExperimentLogBackend>),
        );
        let fp_str = fp();

        // First sync succeeds: state is fresh.
        assert!(
            m.ensure_synced("t", 0, &fp_str).await,
            "first sync must succeed"
        );

        // Advance time within the staleness bound (half-way).
        tokio::time::advance(SYNC_STALENESS_BOUND / 2).await;
        assert!(
            m.ensure_synced("t", 0, &fp_str).await,
            "still within staleness bound → true"
        );

        // Advance past the staleness bound.
        tokio::time::advance(SYNC_STALENESS_BOUND).await;

        // Now state is stale.  The log is reachable (InMemory) so the re-sync succeeds.
        assert!(
            m.ensure_synced("t", 0, &fp_str).await,
            "stale but log reachable → sync succeeds → true"
        );

        // Swap to a failing log to simulate Redis going down.
        // Replace the log: build a new monitor with the failing log but same entry state.
        let m_down = Arc::new(ExperimentMonitor::from_config(&config()).with_log(Arc::new(
            CountingFailLog {
                load_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            },
        )
            as Arc<dyn ExperimentLogBackend>));
        // No prior sync → first ensure_synced attempt fails (backend down).
        assert!(
            !m_down.ensure_synced("t", 0, &fp_str).await,
            "log down on first sync → false"
        );
        // Cooldown active: second call returns false without re-attempting.
        assert!(
            !m_down.ensure_synced("t", 0, &fp_str).await,
            "cooldown active → false immediately"
        );
    }

    /// After a restart, a pair replayed from the local store (its append had timed out after
    /// Redis applied it) and then seen again in the log under the old replica id counts health
    /// and spend once: the restarted process has a new replica id, so only the trace_id dedup
    /// stops the log copy from being tallied a second time.
    #[tokio::test]
    async fn restart_counts_a_timed_out_pair_once_across_store_and_log() {
        let db =
            std::env::temp_dir().join(format!("firstpass-restart-dedup-{}.db", Uuid::now_v7()));
        crate::store::ensure_migrated(&db).unwrap();
        let log = Arc::new(InMemoryExperimentLog::new());
        let cfg = config();
        let fp_str = fp();
        let tid = Uuid::now_v7();
        let now = jiff::Timestamp::now();
        // The record IS in the log, written by the previous process (a different replica id).
        let before_restart = ExperimentMonitor::from_config(&cfg)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);
        log.append(
            "t",
            0,
            &fp_str,
            LogRecord::Pair {
                x_inc: 1.0,
                x_ch: 1.0,
                incumbent_cost_usd: 0.1,
                challenger_cost_usd: 0.05,
                ch_error: false,
                ch_truncated: 1,
                ts_unix: now.as_second(),
                replica_id: before_restart.replica_id,
                trace_id: tid,
            },
        )
        .await
        .unwrap();
        // ...and in the local store as NotInLog, because the append reported a timeout.
        crate::store::append_experiment_pair(
            &db,
            &crate::store::ExperimentPairRow {
                tenant: "t".to_owned(),
                route_ix: 0,
                fingerprint: fp_str.clone(),
                trace_id: tid,
                x_inc: 1.0,
                x_ch: 1.0,
                challenger_cost_usd: 0.05,
                ch_error: false,
                ch_truncated: 1,
                ts: now,
                in_log: crate::store::InLogStatus::NotInLog,
            },
        )
        .unwrap();

        // Restart: rebuild from the store with a shared log, as run.rs does.
        let ledger = Arc::new(crate::shadow::ShadowLedger::new());
        ledger.debit("t", 0, 0.05, now); // what build_paired_ledger seeds from the NotInLog row
        let restarted = build_monitor_with(&db.to_string_lossy(), &cfg, true)
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>)
            .with_paired_ledger(Arc::clone(&ledger));
        assert_ne!(restarted.replica_id, before_restart.replica_id);
        restarted.sync_from_backend("t", 0, &fp_str).await.unwrap();

        let r = &restarted.reports()[0];
        assert_eq!(r.report.pairs, 1, "evidence once");
        assert_eq!(r.health.challenger.truncated, 1, "health once");
        assert!(
            (ledger.spent_today("t", 0, now) - 0.05).abs() < 1e-9,
            "spend once: {}",
            ledger.spent_today("t", 0, now)
        );
        let _ = std::fs::remove_file(&db);
    }

    /// A log that can be switched to failing mid-test, counting reads.
    #[derive(Debug, Default)]
    struct SwitchableLog {
        inner: InMemoryExperimentLog,
        failing: std::sync::atomic::AtomicBool,
        reads: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl ExperimentLogBackend for SwitchableLog {
        async fn append(
            &self,
            t: &str,
            r: usize,
            f: &str,
            rec: LogRecord,
        ) -> Result<u64, LogError> {
            self.inner.append(t, r, f, rec).await
        }
        async fn load_since(
            &self,
            t: &str,
            r: usize,
            f: &str,
            o: u64,
        ) -> Result<Vec<Option<LogRecord>>, LogError> {
            self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.failing.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(LogError::Backend("down".to_owned()));
            }
            self.inner.load_since(t, r, f, o).await
        }
        async fn claim_session(&self, t: &str, r: usize, f: &str, s: &str) -> bool {
            self.inner.claim_session(t, r, f, s).await
        }
    }

    /// On ONE monitor: synced state stays trusted only within the staleness bound. Once the log
    /// goes down, a recent sync still counts (no read), but past the bound the stale state is
    /// not trusted and `ensure_synced` fails closed.
    #[tokio::test]
    async fn synced_state_expires_after_the_staleness_bound_when_the_log_is_down() {
        tokio::time::pause();
        let log = Arc::new(SwitchableLog::default());
        let m = ExperimentMonitor::from_config(&config())
            .with_log(Arc::clone(&log) as Arc<dyn ExperimentLogBackend>);
        let fp_str = fp();
        assert!(
            m.ensure_synced("t", 0, &fp_str).await,
            "first sync succeeds"
        );
        log.failing.store(true, std::sync::atomic::Ordering::SeqCst);
        let reads = log.reads.load(std::sync::atomic::Ordering::SeqCst);
        tokio::time::advance(SYNC_STALENESS_BOUND / 2).await;
        assert!(m.ensure_synced("t", 0, &fp_str).await, "fresh: trusted");
        assert_eq!(
            log.reads.load(std::sync::atomic::Ordering::SeqCst),
            reads,
            "fresh state needs no read"
        );
        tokio::time::advance(SYNC_STALENESS_BOUND).await;
        assert!(
            !m.ensure_synced("t", 0, &fp_str).await,
            "stale and the log is down: fail closed"
        );
    }

    // ── Issue 4: sync timeout ────────────────────────────────────────────────────────────────

    /// A `load_since` that never completes (pending forever) must not stall `ensure_synced`.
    /// With `tokio::time::pause`, the timeout fires instantly after advancing time.
    #[tokio::test]
    async fn ensure_synced_times_out_on_hung_backend() {
        tokio::time::pause();

        /// A backend whose `load_since` never completes (simulates a half-open TCP connection).
        #[derive(Debug)]
        struct HangingLog;

        #[async_trait::async_trait]
        impl ExperimentLogBackend for HangingLog {
            async fn append(
                &self,
                _: &str,
                _: usize,
                _: &str,
                _: LogRecord,
            ) -> Result<u64, LogError> {
                Err(LogError::Backend("down".to_owned()))
            }
            async fn load_since(
                &self,
                _: &str,
                _: usize,
                _: &str,
                _: u64,
            ) -> Result<Vec<Option<LogRecord>>, LogError> {
                // Pend forever: never resolves.
                std::future::pending::<Result<Vec<Option<LogRecord>>, LogError>>().await
            }
            async fn claim_session(&self, _: &str, _: usize, _: &str, _: &str) -> bool {
                false
            }
        }

        let m = Arc::new(
            ExperimentMonitor::from_config(&config())
                .with_log(Arc::new(HangingLog) as Arc<dyn ExperimentLogBackend>),
        );
        let fp_str = fp();

        // ensure_synced wraps the sync in a timeout; the hanging load_since fires the timeout.
        // We advance time past SYNC_TIMEOUT so the timeout elapses.
        let result = tokio::select! {
            r = m.ensure_synced("t", 0, &fp_str) => r,
            _ = tokio::time::sleep(SYNC_TIMEOUT + std::time::Duration::from_millis(1)) => {
                // Advance the mock clock past the timeout.
                tokio::time::advance(SYNC_TIMEOUT + std::time::Duration::from_millis(10)).await;
                m.ensure_synced("t", 0, &fp_str).await
            }
        };
        assert!(
            !result,
            "ensure_synced must return false when load_since hangs (timeout fires)"
        );
    }
}
