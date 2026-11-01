//! Tamper-evident trace storage: a background SQLite writer fed by a bounded channel.
//!
//! The hot request path never blocks on disk: it `try_send`s a [`Trace`] down a bounded channel
//! and returns immediately (dropping the trace if the writer has fallen far enough behind to fill
//! the buffer — bounded memory over a guaranteed write). A single background task owns the SQLite
//! connection, assigns each trace's `prev_hash` from the current chain head, computes its `hash`,
//! and appends it (SPEC §9: the hash chain is only meaningful if every writer agrees on chain
//! order, which a single writer task guarantees for free).
//!
//! In [`crate::config::ReceiptsMode::Durable`] mode the channel-full path spills traces to
//! `<db_path>.spill.jsonl` (one JSON line per trace, synced to disk) instead of dropping them.
//! The writer drains the spill file at startup and whenever the channel empties, inserting spilled
//! traces BEFORE new channel arrivals so the hash chain stays append-only and valid.

use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use firstpass_core::{DeferredVerdict, GENESIS_HASH, Score, Trace, Verdict};
use rusqlite::{Connection, OptionalExtension as _};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::config::ReceiptsMode;

/// Open a connection with WAL + a busy timeout, so the background writer and short-lived
/// feedback/read connections can share the file without "database is locked" errors.
fn connect(db_path: impl AsRef<Path>) -> Result<Connection, StoreError> {
    let conn = Connection::open(db_path.as_ref())?;
    conn.busy_timeout(Duration::from_secs(5))?;
    Ok(conn)
}

/// Errors from the trace store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The SQLite database could not be opened, migrated, or queried.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A trace could not be hashed or (de)serialized.
    #[error("trace error: {0}")]
    Trace(#[from] firstpass_core::Error),
    /// A stored row was not valid trace JSON.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// An I/O error from the spill file.
    #[error("spill I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A shared, mutex-guarded handle to the spill file used in durable mode.
///
/// Multiple async tasks may hit a full channel simultaneously; the `Mutex` serialises their
/// appends without data loss.
///
/// ponytail: global Mutex over the file — upgrade to a dedicated spill-writer task if append
/// throughput under sustained backpressure becomes the bottleneck.
pub type SpillHandle = Arc<std::sync::Mutex<std::fs::File>>;

/// Derive the spill-file path from the main DB path: `<db_path>.spill.jsonl`.
pub(crate) fn spill_path(db_path: &Path) -> std::path::PathBuf {
    let mut s = db_path.as_os_str().to_owned();
    s.push(".spill.jsonl");
    std::path::PathBuf::from(s)
}

/// Serialize `trace` as one JSON line and append it to the spill file, flushing to disk.
///
/// Called from the hot async path only when the writer channel is full (durable mode). This
/// blocks the calling tokio task on the disk write — the deliberate tradeoff of durable mode;
/// it only fires under sustained backpressure.
///
/// # Errors
/// Returns [`StoreError::Json`] on serialization failure, [`StoreError::Io`] on disk failure.
pub fn append_to_spill(handle: &SpillHandle, trace: &Trace) -> Result<(), StoreError> {
    let line = serde_json::to_string(trace)?;
    // Recover from a poisoned mutex rather than failing: a previous writer's panic doesn't
    // corrupt the file, so we can safely continue appending.
    let mut guard = handle.lock().unwrap_or_else(|e| e.into_inner());
    writeln!(guard, "{line}")?;
    // sync_data flushes the write to stable storage — the whole point of durable mode.
    guard.sync_data()?;
    Ok(())
}

/// Sending half of the trace channel; cheap to clone, safe to share across request handlers.
/// Sending is fire-and-forget via `try_send` — the hot path never awaits the writer, and a bounded
/// buffer means a stalled writer sheds load (drops traces) instead of growing memory without limit.
pub type TraceSender = mpsc::Sender<Trace>;

/// Trace buffer depth. Deep enough to absorb normal write bursts; bounded so a wedged writer (disk
/// stall) can't OOM the process — excess traces are dropped with a warning, not queued forever.
pub const TRACE_CHANNEL_CAP: usize = 8192;

/// Open (creating if needed) the SQLite trace database in best-effort mode, migrate its schema,
/// and spawn the background writer task.
///
/// Returns a [`TraceSender`] for the hot path and the writer's [`JoinHandle`]. The writer
/// exits cleanly once every clone of the sender is dropped.
///
/// This is the backward-compatible convenience wrapper. Use [`open_with_receipts`] when you need
/// durable (never-drop) mode.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] if the database cannot be opened or migrated.
pub fn open(db_path: impl AsRef<Path>) -> Result<(TraceSender, JoinHandle<()>), StoreError> {
    let (tx, _spill, handle) = open_with_receipts(db_path, ReceiptsMode::BestEffort)?;
    Ok((tx, handle))
}

/// Open the SQLite trace database with the given receipts mode, migrate its schema, and spawn
/// the background writer task.
///
/// Returns a [`TraceSender`], an optional [`SpillHandle`] (present only in durable mode, for use
/// by [`append_to_spill`] on channel-full), and the writer's [`JoinHandle`].
///
/// In [`ReceiptsMode::Durable`] mode the writer drains any existing spill file at startup and
/// whenever the channel empties, inserting spilled traces BEFORE new channel arrivals so the hash
/// chain stays append-only and valid across crashes and restarts.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] if the database cannot be opened or migrated, or
/// [`StoreError::Io`] if the spill file cannot be created in durable mode.
pub fn open_with_receipts(
    db_path: impl AsRef<Path>,
    receipts_mode: ReceiptsMode,
) -> Result<(TraceSender, Option<SpillHandle>, JoinHandle<()>), StoreError> {
    let db_path = db_path.as_ref();
    let conn = connect(db_path)?;
    migrate(&conn)?;

    let spill = if receipts_mode == ReceiptsMode::Durable {
        let path = spill_path(db_path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Some(Arc::new(std::sync::Mutex::new(file)))
    } else {
        None
    };

    let (tx, rx) = mpsc::channel::<Trace>(TRACE_CHANNEL_CAP);
    let spill_writer = spill.clone();
    let handle = tokio::task::spawn_blocking(move || writer_loop(conn, rx, spill_writer));
    Ok((tx, spill, handle))
}

fn migrate(conn: &Connection) -> Result<(), StoreError> {
    // WAL: lets the background writer and short-lived feedback/read connections share the file
    // concurrently. `journal_mode` is persisted in the file header once set by any connection.
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS traces (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL,
            tenant TEXT NOT NULL,
            session TEXT NOT NULL,
            body TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS traces_session_idx ON traces(session);
        -- Tenant-scoped reads (ADR 0004 §D3) filter on `tenant`; index it with `seq` so a
        -- per-tenant scan stays ordered and cheap. Existing rows keep their `tenant` value.
        CREATE INDEX IF NOT EXISTS traces_tenant_seq_idx ON traces(tenant, seq);
        -- Deferred verdicts live in their OWN table, keyed by trace_id. They are NEVER folded
        -- into the sealed, hashed `traces.body`, so a late outcome can't alter a past record and
        -- the tamper-evident chain stays valid. They are merged onto a trace only on read.
        CREATE TABLE IF NOT EXISTS deferred_verdicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            gate_id TEXT NOT NULL,
            verdict TEXT NOT NULL,
            score REAL,
            reported_at TEXT NOT NULL,
            reporter TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS deferred_trace_idx ON deferred_verdicts(trace_id);
        -- Paired-shadow experiment pairs (ADR 0014 Stage 2): one row per paired evaluation.
        -- Gate verdicts from the incumbent (x_inc) and challenger (x_ch) are stored here so the
        -- experiment can be rebuilt deterministically at startup or CLI replay: arrival order is
        -- the row id, and replaying in that order yields the same e-process trajectory.
        CREATE TABLE IF NOT EXISTS experiment_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant TEXT NOT NULL,
            route_ix INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            x_inc REAL NOT NULL,
            x_ch REAL NOT NULL,
            challenger_cost_usd REAL NOT NULL,
            ch_error INTEGER NOT NULL DEFAULT 0,
            ch_truncated INTEGER NOT NULL DEFAULT 0,
            ts TEXT NOT NULL,
            in_log INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS exp_pairs_tenant_idx ON experiment_pairs(tenant, route_ix, fingerprint);",
    )?;
    // Idempotent migrations: each column is checked and added independently.
    // PRAGMA table_info returns one row per column; column name is at index 1.
    // SQLite does not support "ALTER TABLE … ADD COLUMN IF NOT EXISTS", so we check first.
    let has_column = |name: &str| -> Result<bool, rusqlite::Error> {
        let mut stmt = conn.prepare("PRAGMA table_info(experiment_pairs)")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let col: String = row.get(1)?;
            if col == name {
                return Ok(true);
            }
        }
        Ok(false)
    };
    if !has_column("ch_error")? {
        conn.execute_batch(
            "ALTER TABLE experiment_pairs ADD COLUMN ch_error INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !has_column("ch_truncated")? {
        conn.execute_batch(
            "ALTER TABLE experiment_pairs ADD COLUMN ch_truncated INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    // `in_log = 1` for old rows: assume they were already pushed to the shared log (safe
    // default — they will not be double-replayed on startup with shared_log=true).
    if !has_column("in_log")? {
        conn.execute_batch(
            "ALTER TABLE experiment_pairs ADD COLUMN in_log INTEGER NOT NULL DEFAULT 1;",
        )?;
    }
    Ok(())
}

/// Assign `prev_hash`, compute `hash`, and insert one trace. Updates `head` on success.
/// Logs and skips on hash or insert failure — one bad record must not wedge the pipeline.
fn write_trace(conn: &Connection, trace: &mut Trace, head: &mut String) {
    trace.prev_hash = head.clone();
    let hash = match trace.hash() {
        Ok(h) => h,
        Err(err) => {
            tracing::error!(%err, trace_id = %trace.trace_id, "trace writer: hash failed, dropping");
            return;
        }
    };
    if let Err(err) = insert(conn, trace, &hash) {
        tracing::error!(%err, trace_id = %trace.trace_id, "trace writer: insert failed, dropping");
        return;
    }
    *head = hash;
}

/// Drain the spill file into the store, inserting traces in file order BEFORE any new channel
/// arrivals. Truncates the file after a successful drain. Errors on individual lines are logged
/// and skipped so one bad spill line can't stall the recovery.
///
/// ponytail: reads the entire file into memory; the spill file is bounded by the channel size
/// and trace sizes, so this is fine in practice. Stream-parse if traces grow very large.
fn drain_spill(conn: &Connection, spill: &SpillHandle, head: &mut String) {
    // Hold the lock for the entire drain: prevents concurrent spill writers from appending
    // new lines mid-drain, which would corrupt ordering.
    let mut guard = spill.lock().unwrap_or_else(|e| e.into_inner());

    // Seek to start before reading.
    if let Err(e) = guard.seek(SeekFrom::Start(0)) {
        tracing::error!(%e, "drain_spill: seek failed");
        return;
    }

    // Collect all non-empty lines while holding the lock (reader borrows from guard).
    let lines: Vec<String> = {
        let reader = BufReader::new(&*guard);
        reader
            .lines()
            .filter_map(|l| match l {
                Ok(s) if !s.trim().is_empty() => Some(s),
                Ok(_) => None,
                Err(e) => {
                    tracing::error!(%e, "drain_spill: read error");
                    None
                }
            })
            .collect()
    };

    if lines.is_empty() {
        return;
    }

    let n = lines.len();
    for line in &lines {
        let mut trace: Trace = match serde_json::from_str(line) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!(%e, "drain_spill: bad JSON line, skipping");
                continue;
            }
        };
        write_trace(conn, &mut trace, head);
    }
    tracing::info!(drained = n, "spill file drained into trace store");

    // Truncate: all lines have been processed (failures are skipped and logged above).
    if let Err(e) = guard.seek(SeekFrom::Start(0)) {
        tracing::error!(%e, "drain_spill: post-drain seek failed");
        return;
    }
    if let Err(e) = guard.set_len(0) {
        tracing::error!(%e, "drain_spill: truncate failed — spill file may replay on next boot");
    }
}

/// The writer's main loop: runs on a blocking-pool thread for the lifetime of the store.
/// Never panics on a bad trace — logs and drops it, so one malformed record can't wedge the
/// whole audit pipeline.
///
/// In durable mode (`spill` is `Some`): drains the spill file at startup and whenever the
/// channel empties, so spilled traces always land BEFORE later channel traces in the chain.
fn writer_loop(conn: Connection, mut rx: mpsc::Receiver<Trace>, spill: Option<SpillHandle>) {
    let mut head = match current_head(&conn) {
        Ok(h) => h,
        Err(err) => {
            tracing::error!(%err, "trace writer: failed to load chain head, stopping");
            return;
        }
    };

    // Startup drain: recover any traces spilled during a previous run's backpressure.
    if let Some(ref s) = spill {
        drain_spill(&conn, s, &mut head);
    }

    loop {
        let Some(mut trace) = rx.blocking_recv() else {
            break;
        };
        write_trace(&conn, &mut trace, &mut head);

        // Fast-drain: pull any immediately available channel items, then drain spill when
        // the channel is momentarily empty so spilled traces land before the next wave.
        loop {
            match rx.try_recv() {
                Ok(mut t) => write_trace(&conn, &mut t, &mut head),
                Err(mpsc::error::TryRecvError::Empty) => {
                    if let Some(ref s) = spill {
                        drain_spill(&conn, s, &mut head);
                    }
                    break;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }
    }

    // Channel closed: one final spill drain to capture any last-moment spills.
    if let Some(ref s) = spill {
        drain_spill(&conn, s, &mut head);
    }
}

fn current_head(conn: &Connection) -> Result<String, StoreError> {
    let mut stmt = conn.prepare("SELECT hash FROM traces ORDER BY seq DESC LIMIT 1")?;
    let mut rows = stmt.query([])?;
    match rows.next()? {
        Some(row) => Ok(row.get(0)?),
        None => Ok(GENESIS_HASH.to_owned()),
    }
}

fn insert(conn: &Connection, trace: &Trace, hash: &str) -> Result<(), StoreError> {
    let body = serde_json::to_string(trace)?;
    conn.execute(
        "INSERT INTO traces (trace_id, ts, prev_hash, hash, tenant, session, body)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            trace.trace_id.to_string(),
            trace.ts.to_string(),
            trace.prev_hash,
            hash,
            trace.tenant_id,
            trace.session_id,
            body,
        ],
    )?;
    Ok(())
}

/// Load every trace from the database in insertion (chain) order — used by tests and
/// operators to verify the hash chain with [`firstpass_core::verify_chain`].
///
/// **Operator-wide** read: every trace across ALL tenants, in `seq` order.
///
/// This deliberately crosses tenant boundaries and must stay reserved for operator-scoped work
/// where a global view is intrinsic — namely verifying the single hash-chain, which spans every
/// tenant's traces in one sequence (ADR 0004 §D3). For tenant-facing reads use
/// [`load_tenant_traces`].
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error, or [`StoreError::Json`] if a stored
/// row is not valid trace JSON.
pub fn load_all_traces(db_path: impl AsRef<Path>) -> Result<Vec<Trace>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let mut stmt = conn.prepare("SELECT body FROM traces ORDER BY seq ASC")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut traces = Vec::new();
    for row in rows {
        traces.push(serde_json::from_str(&row?)?);
    }
    Ok(traces)
}

/// **Tenant-scoped** read: only traces owned by `tenant`, in `seq` order (ADR 0004 §D3). Tenant A
/// can never see tenant B's traces through this path.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error, or [`StoreError::Json`] if a stored
/// row is not valid trace JSON.
pub fn load_tenant_traces(
    db_path: impl AsRef<Path>,
    tenant: &str,
) -> Result<Vec<Trace>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let mut stmt = conn.prepare("SELECT body FROM traces WHERE tenant = ?1 ORDER BY seq ASC")?;
    let rows = stmt.query_map([tenant], |row| row.get::<_, String>(0))?;
    let mut traces = Vec::new();
    for row in rows {
        traces.push(serde_json::from_str(&row?)?);
    }
    Ok(traces)
}

/// **Tenant-scoped** page: the tenant's `limit` most recent traces, newest first. Reads only the
/// rows it returns (the `(tenant, seq)` index serves the order), so a panel polling a long-lived
/// store pays for one page, not for the whole history.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error, or [`StoreError::Json`] if a stored
/// row is not valid trace JSON.
pub fn load_recent_tenant_traces(
    db_path: impl AsRef<Path>,
    tenant: &str,
    limit: usize,
) -> Result<Vec<Trace>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let mut stmt =
        conn.prepare("SELECT body FROM traces WHERE tenant = ?1 ORDER BY seq DESC LIMIT ?2")?;
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let rows = stmt.query_map(rusqlite::params![tenant, limit], |row| {
        row.get::<_, String>(0)
    })?;
    let mut traces = Vec::new();
    for row in rows {
        traces.push(serde_json::from_str(&row?)?);
    }
    Ok(traces)
}

/// All-time totals over a tenant's receipts, for the panel's overview tiles.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReceiptsSummary {
    /// Number of receipts.
    pub requests: u64,
    /// Sum of `final.total_cost_usd`.
    pub spent_usd: f64,
    /// Sum of `final.counterfactual_baseline_usd`.
    pub baseline_usd: f64,
    /// Receipts served at rung 0.
    pub first_pass: u64,
    /// Receipts that climbed at least one rung.
    pub escalated: u64,
    /// Gate verdicts across every attempt: pass, fail, abstain.
    pub gates: [u64; 3],
    /// Attempts whose answer stopped at the output-token limit.
    pub truncated: u64,
}

impl ReceiptsSummary {
    fn merged(&self, d: &Self) -> Self {
        Self {
            requests: self.requests + d.requests,
            spent_usd: self.spent_usd + d.spent_usd,
            baseline_usd: self.baseline_usd + d.baseline_usd,
            first_pass: self.first_pass + d.first_pass,
            escalated: self.escalated + d.escalated,
            gates: [
                self.gates[0] + d.gates[0],
                self.gates[1] + d.gates[1],
                self.gates[2] + d.gates[2],
            ],
            truncated: self.truncated + d.truncated,
        }
    }
}

/// Per-(store, tenant) running totals and the last `seq` folded into them.
type SummaryCache = std::collections::HashMap<(std::path::PathBuf, String), (i64, ReceiptsSummary)>;
static SUMMARY_CACHE: std::sync::LazyLock<std::sync::Mutex<SummaryCache>> =
    std::sync::LazyLock::new(Default::default);

/// **Tenant-scoped** aggregate over every stored receipt, computed inside SQLite and kept
/// incrementally.
///
/// The panel polls this every two seconds, so it must not re-read the whole history each time.
/// Receipts are append-only (the hash chain forbids rewriting one), so the totals up to a `seq`
/// never change: each call folds in only the rows after the last `seq` it saw. A store whose
/// newest `seq` is below the cached one was replaced, and is recounted from zero.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error.
pub fn tenant_receipts_summary(
    db_path: impl AsRef<Path>,
    tenant: &str,
) -> Result<ReceiptsSummary, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let key = (db_path.as_ref().to_path_buf(), tenant.to_owned());
    let hi: i64 = conn.query_row(
        "SELECT coalesce(max(seq), 0) FROM traces WHERE tenant = ?1",
        [tenant],
        |r| r.get(0),
    )?;
    let cached = SUMMARY_CACHE
        .lock()
        .ok()
        .and_then(|c| c.get(&key).cloned())
        .filter(|(seen, _)| *seen <= hi);
    let (after, base) = cached.unwrap_or_default();
    if after == hi {
        return Ok(base);
    }
    let total = base.merged(&summary_between(&conn, tenant, after, hi)?);
    if let Ok(mut c) = SUMMARY_CACHE.lock() {
        // Bounded: dropping the cache only costs one full recount per tenant, never correctness.
        if c.len() >= 4096 {
            c.clear();
        }
        c.insert(key, (hi, total.clone()));
    }
    Ok(total)
}

/// Totals over the tenant's rows with `after < seq <= hi`. Both queries share the same upper
/// bound, so a write landing between them cannot be counted in one and not the other.
fn summary_between(
    conn: &Connection,
    tenant: &str,
    after: i64,
    hi: i64,
) -> Result<ReceiptsSummary, StoreError> {
    let mut s = conn.query_row(
        "SELECT count(*),
                total(json_extract(body, '$.final.total_cost_usd')),
                total(json_extract(body, '$.final.counterfactual_baseline_usd')),
                coalesce(sum(json_extract(body, '$.final.served_rung') = 0), 0),
                coalesce(sum(json_extract(body, '$.final.escalations') > 0), 0)
         FROM traces WHERE tenant = ?1 AND seq > ?2 AND seq <= ?3",
        rusqlite::params![tenant, after, hi],
        |r| {
            Ok(ReceiptsSummary {
                requests: u64::try_from(r.get::<_, i64>(0)?).unwrap_or(0),
                spent_usd: r.get(1)?,
                baseline_usd: r.get(2)?,
                first_pass: u64::try_from(r.get::<_, i64>(3)?).unwrap_or(0),
                escalated: u64::try_from(r.get::<_, i64>(4)?).unwrap_or(0),
                gates: [0; 3],
                truncated: 0,
            })
        },
    )?;
    let truncated: i64 = conn.query_row(
        "SELECT count(*) FROM traces t, json_each(t.body, '$.attempts') a
         WHERE t.tenant = ?1 AND t.seq > ?2 AND t.seq <= ?3
           AND json_extract(a.value, '$.truncated') = 1",
        rusqlite::params![tenant, after, hi],
        |r| r.get(0),
    )?;
    s.truncated = u64::try_from(truncated).unwrap_or(0);
    let mut stmt = conn.prepare(
        "SELECT json_extract(g.value, '$.verdict'), count(*)
         FROM traces t, json_each(t.body, '$.attempts') a, json_each(a.value, '$.gates') g
         WHERE t.tenant = ?1 AND t.seq > ?2 AND t.seq <= ?3 GROUP BY 1",
    )?;
    let rows = stmt.query_map(rusqlite::params![tenant, after, hi], |r| {
        Ok((r.get::<_, Option<String>>(0)?, r.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (verdict, n) = row?;
        let slot = match verdict.as_deref() {
            Some("pass") => 0,
            Some("fail") => 1,
            Some("abstain") => 2,
            _ => continue,
        };
        s.gates[slot] += u64::try_from(n).unwrap_or(0);
    }
    Ok(s)
}

/// Whether a trace with `trace_id` exists **and is owned by `tenant`** — used to reject feedback
/// for unknown traces and, crucially, to deny cross-tenant feedback (ADR 0004 §D3/§D4). A trace
/// owned by another tenant is indistinguishable from a non-existent one here, so the caller can
/// return a `404` with no existence oracle.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error.
pub fn trace_exists(
    db_path: impl AsRef<Path>,
    tenant: &str,
    trace_id: &str,
) -> Result<bool, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let n: i64 = conn.query_row(
        "SELECT COUNT(1) FROM traces WHERE tenant = ?1 AND trace_id = ?2",
        [tenant, trace_id],
        |row| row.get(0),
    )?;
    Ok(n > 0)
}

/// The route index recorded on `trace_id`, scoped to `tenant`.
///
/// `Ok(None)` means the trace is not visible to this tenant (or does not exist) — the same
/// ownership check [`trace_exists`] makes, so this cannot be used to probe another tenant's
/// traces. `Ok(Some(None))` means the trace exists but predates route recording.
///
/// # Errors
/// On a store read failure.
/// The decision a trace's answer was originally proven by, if it was itself a cache replay.
///
/// A cache hit is recorded under a NEW trace id, with `cache_source` naming the decision whose gate
/// passed. So feedback arriving on a replay must be applied to that source, not to the replay's own
/// id — otherwise a Fail reported against a replayed answer retracts nothing and the disproven
/// entry keeps serving until TTL.
///
/// Returns `Ok(None)` for an unknown trace, `Ok(Some(None))` for one that ran the ladder itself.
///
/// # Errors
/// [`StoreError`] when the database cannot be opened or queried.
pub fn trace_cache_source(
    db_path: impl AsRef<Path>,
    tenant: &str,
    trace_id: &str,
) -> Result<Option<Option<uuid::Uuid>>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let body: Option<String> = conn
        .query_row(
            "SELECT body FROM traces WHERE tenant = ?1 AND trace_id = ?2",
            [tenant, trace_id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(body.map(|b: String| {
        serde_json::from_str::<firstpass_core::Trace>(&b)
            .ok()
            .and_then(|t| t.final_.cache_source)
    }))
}

pub fn trace_route(
    db_path: impl AsRef<Path>,
    tenant: &str,
    trace_id: &str,
) -> Result<Option<Option<u32>>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let body: Option<String> = conn
        .query_row(
            "SELECT body FROM traces WHERE tenant = ?1 AND trace_id = ?2",
            [tenant, trace_id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(body.map(|b: String| {
        serde_json::from_str::<firstpass_core::Trace>(&b)
            .ok()
            .and_then(|t| t.route_ix)
    }))
}

/// Append a deferred verdict for `trace_id` (a downstream outcome or async gate result). This
/// writes ONLY to the `deferred_verdicts` table; the sealed trace and its hash are untouched, so
/// the audit chain remains verifiable (SPEC §8.3.4 — the outcome-feedback loop).
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error.
pub fn append_deferred(
    db_path: impl AsRef<Path>,
    trace_id: &str,
    v: &DeferredVerdict,
) -> Result<(), StoreError> {
    let conn = connect(db_path.as_ref())?;
    conn.execute(
        "INSERT INTO deferred_verdicts (trace_id, gate_id, verdict, score, reported_at, reporter)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            trace_id,
            v.gate_id,
            v.verdict.as_str(),
            v.score.map(Score::value),
            v.reported_at.to_string(),
            v.reporter,
        ],
    )?;
    Ok(())
}

/// Load the deferred verdicts recorded for `trace_id`, oldest first. Malformed stored rows are
/// skipped (logged), never fatal — a corrupt late outcome must not break reading the trace.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error.
pub fn load_deferred(
    db_path: impl AsRef<Path>,
    trace_id: &str,
) -> Result<Vec<DeferredVerdict>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let mut stmt = conn.prepare(
        "SELECT gate_id, verdict, score, reported_at, reporter
         FROM deferred_verdicts WHERE trace_id = ?1 ORDER BY id ASC",
    )?;
    let rows = stmt.query_map([trace_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<f64>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;

    let mut out = Vec::new();
    for row in rows {
        let (gate_id, verdict_s, score, reported_s, reporter) = row?;
        let verdict = match verdict_s.as_str() {
            "pass" => Verdict::Pass,
            "fail" => Verdict::Fail,
            "abstain" => Verdict::Abstain,
            other => {
                tracing::warn!(verdict = %other, %trace_id, "skipping deferred row with bad verdict");
                continue;
            }
        };
        let Ok(reported_at) = reported_s.parse::<jiff::Timestamp>() else {
            tracing::warn!(%trace_id, "skipping deferred row with bad timestamp");
            continue;
        };
        out.push(DeferredVerdict {
            gate_id,
            verdict,
            score: score.and_then(|s| Score::new(s).ok()),
            reported_at,
            reporter,
        });
    }
    Ok(out)
}

/// Every deferred verdict in the store as `(trace_id, verdict)`, in the order the feedback
/// arrived. **Operator-wide**: rows are not tenant-scoped here, so callers must join them to
/// tenant-scoped traces (an id that matches no loaded trace is simply ignored). Rows with an
/// unparseable id or verdict are skipped, as in [`load_deferred`].
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error.
pub fn load_all_deferred(
    db_path: impl AsRef<Path>,
) -> Result<Vec<(uuid::Uuid, Verdict)>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let mut stmt =
        conn.prepare("SELECT trace_id, verdict FROM deferred_verdicts ORDER BY id ASC")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (id, verdict) = row?;
        let verdict = match verdict.as_str() {
            "pass" => Verdict::Pass,
            "fail" => Verdict::Fail,
            "abstain" => Verdict::Abstain,
            _ => continue,
        };
        if let Ok(id) = id.parse::<uuid::Uuid>() {
            out.push((id, verdict));
        }
    }
    Ok(out)
}

/// What the feedback path needs to attribute an outcome to a certified-selection experiment:
/// the trace's session, its route, and its experiment record. `Ok(None)` when the trace is not
/// visible to `tenant`, or carries no experiment record. Tenant-scoped like [`trace_route`].
///
/// # Errors
/// On a store read failure.
pub fn trace_experiment(
    db_path: impl AsRef<Path>,
    tenant: &str,
    trace_id: &str,
) -> Result<Option<(String, usize, firstpass_core::trace::ExperimentRecord, f64)>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let body: Option<String> = conn
        .query_row(
            "SELECT body FROM traces WHERE tenant = ?1 AND trace_id = ?2",
            [tenant, trace_id],
            |row| row.get(0),
        )
        .optional()?;
    let Some(body) = body else { return Ok(None) };
    let trace: Trace = serde_json::from_str(&body)?;
    let cost_usd = trace.final_.total_cost_usd;
    Ok(trace.experiment.map(|rec| {
        (
            trace.session_id,
            // Same fallback as the guardrail: a trace with no recorded route is route 0.
            trace.route_ix.map_or(0, |r| r as usize),
            rec,
            cost_usd,
        )
    }))
}

/// Load a single trace by id **scoped to `tenant`**, with its deferred verdicts merged into
/// `deferred` — the **view** for display/inspection (ADR 0004 §D3). A trace owned by another
/// tenant returns `None`, exactly like a missing one, so an inspecting agent can never read across
/// tenants. This is deliberately separate from [`load_all_traces`]: merging deferred verdicts
/// changes the record, so a merged trace must NOT be fed to `verify_chain` (chain verification
/// always runs on the sealed bodies from [`load_all_traces`]).
///
/// # Errors
/// Returns [`StoreError::Sqlite`] / [`StoreError::Json`] on database or decode errors.
pub fn load_trace_view(
    db_path: impl AsRef<Path>,
    tenant: &str,
    trace_id: &str,
) -> Result<Option<Trace>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let body: Option<String> = conn
        .query_row(
            "SELECT body FROM traces WHERE tenant = ?1 AND trace_id = ?2",
            [tenant, trace_id],
            |row| row.get(0),
        )
        .ok();
    let Some(body) = body else { return Ok(None) };
    let mut trace: Trace = serde_json::from_str(&body)?;
    trace.deferred = load_deferred(db_path, trace_id)?;
    Ok(Some(trace))
}

/// One row from the `experiment_pairs` table.
///
/// Used to rebuild paired-mode experiments at startup and in the CLI, so the e-process trajectory
/// is identical whether derived live or replayed. Arrival order = row `id` (insertion order).
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentPairRow {
    /// Tenant that owns this pair.
    pub tenant: String,
    /// Route index the challenger belongs to.
    pub route_ix: usize,
    /// Fingerprint of the experiment config (incumbent + challenger ladders, margin, alpha).
    pub fingerprint: String,
    /// Trace id of the served (incumbent) request that triggered this pair.
    pub trace_id: uuid::Uuid,
    /// Incumbent gate verdict: 1.0 = gate passed, 0.0 = gate failed / best-attempt.
    pub x_inc: f64,
    /// Challenger gate verdict: 1.0 = gate passed, 0.0 = gate failed.
    pub x_ch: f64,
    /// USD spent by the challenger on this pair.
    pub challenger_cost_usd: f64,
    /// Whether the challenger call ended in an error with no output.
    pub ch_error: bool,
    /// Number of challenger attempts whose answer was cut off at the output-token limit.
    pub ch_truncated: u64,
    /// When the pair was recorded.
    pub ts: jiff::Timestamp,
    /// Whether this pair was successfully appended to the shared Redis log at record time.
    ///
    /// `false` for pairs written to SQLite during a Redis outage that were never pushed to
    /// the log. On restart with a shared log configured, these rows are replayed locally
    /// (evidence + cost) so spend caps and halt decisions survive an outage + restart cycle.
    /// Older rows default to `true` (assumed already in the log) via the schema default.
    pub in_log: bool,
}

/// Append one experiment pair to the `experiment_pairs` table.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database write failure.
pub fn append_experiment_pair(
    db_path: impl AsRef<std::path::Path>,
    row: &ExperimentPairRow,
) -> Result<(), StoreError> {
    let conn = connect(db_path.as_ref())?;
    conn.execute(
        "INSERT INTO experiment_pairs \
         (tenant, route_ix, fingerprint, trace_id, x_inc, x_ch, challenger_cost_usd, \
          ch_error, ch_truncated, ts, in_log) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            row.tenant,
            row.route_ix as i64,
            row.fingerprint,
            row.trace_id.to_string(),
            row.x_inc,
            row.x_ch,
            row.challenger_cost_usd,
            i64::from(row.ch_error),
            row.ch_truncated as i64,
            row.ts.to_string(),
            i64::from(row.in_log),
        ],
    )?;
    Ok(())
}

/// Load every experiment pair in insertion order. Used at startup (rebuild) and CLI replay.
///
/// Rows with an unparseable `trace_id` or `ts` are skipped with a warning rather than failing the
/// read — a single corrupt row must not prevent an otherwise healthy experiment from being restored.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database read failure.
pub fn load_all_experiment_pairs(
    db_path: impl AsRef<std::path::Path>,
) -> Result<Vec<ExperimentPairRow>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let mut stmt = conn.prepare(
        "SELECT tenant, route_ix, fingerprint, trace_id, x_inc, x_ch, challenger_cost_usd, \
                ch_error, ch_truncated, ts, in_log \
         FROM experiment_pairs ORDER BY id ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, f64>(4)?,
            row.get::<_, f64>(5)?,
            row.get::<_, f64>(6)?,
            row.get::<_, i64>(7)?,
            row.get::<_, i64>(8)?,
            row.get::<_, String>(9)?,
            row.get::<_, i64>(10)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (
            tenant,
            route_ix,
            fingerprint,
            trace_id_s,
            x_inc,
            x_ch,
            ch_cost,
            ch_error_i,
            ch_truncated_i,
            ts_s,
            in_log_i,
        ) = row?;
        let Ok(trace_id) = trace_id_s.parse::<uuid::Uuid>() else {
            tracing::warn!(
                %trace_id_s,
                "experiment_pairs: skipping row with unparseable trace_id"
            );
            continue;
        };
        let Ok(ts) = ts_s.parse::<jiff::Timestamp>() else {
            tracing::warn!(
                %trace_id_s,
                "experiment_pairs: skipping row with unparseable ts"
            );
            continue;
        };
        out.push(ExperimentPairRow {
            tenant,
            route_ix: route_ix as usize,
            fingerprint,
            trace_id,
            x_inc,
            x_ch,
            challenger_cost_usd: ch_cost,
            ch_error: ch_error_i != 0,
            ch_truncated: u64::try_from(ch_truncated_i).unwrap_or(0),
            ts,
            in_log: in_log_i != 0,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use firstpass_core::{
        Attempt, Features, FinalOutcome, GENESIS_HASH, PolicyRef, RequestInfo, ServedFrom,
        TaskKind, Verdict, verify_chain,
    };

    use super::*;

    fn sample_trace(tenant: &str, session: &str) -> Trace {
        let attempt = Attempt {
            rung: 0,
            model: "claude-haiku-4-5".to_owned(),
            provider: "anthropic".to_owned(),
            in_tokens: 10,
            cache_write_tokens: 0,
            cache_read_tokens: 0,
            out_tokens: 5,
            cost_usd: 0.001,
            latency_ms: 12,
            gates: vec![],
            verdict: Verdict::Pass,
            truncated: false,
        };
        let mut trace = Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: GENESIS_HASH.to_owned(),
            tenant_id: tenant.to_owned(),
            session_id: session.to_owned(),
            ts: jiff::Timestamp::now(),
            mode: firstpass_core::Mode::Observe,
            policy: PolicyRef {
                id: "observe-passthrough@v0".to_owned(),
                explore: false,
                propensity: None,
                mode_profile: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "deadbeef".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![attempt],
            deferred: Vec::new(),
            final_: FinalOutcome {
                served_rung: Some(0),
                served_from: ServedFrom::Attempt,
                total_cost_usd: 0.001,
                gate_cost_usd: 0.0,
                total_latency_ms: 12,
                escalations: 0,
                counterfactual_baseline_usd: 0.001,
                savings_usd: 0.0,
                cache_source: None,
            },
            probe: None,
            rollout: None,
            shadow: None,
            route_ix: None,
            predicted_pass: None,
            decision_prior: None,
            elastic: None,
            risk: None,
            experiment: None,
        };
        trace.recompute_savings();
        trace
    }

    /// The SQL aggregate must equal a plain Rust fold over the same receipts, including traces
    /// with no served rung, escalations, mixed gate verdicts, and another tenant's rows mixed in.
    /// The page must be the newest `limit` rows of this tenant only, newest first.
    #[tokio::test]
    async fn summary_aggregate_matches_a_rust_fold_and_page_is_newest_first() {
        use firstpass_core::GateResult;
        let db_path =
            std::env::temp_dir().join(format!("firstpass-store-sum-{}.db", uuid::Uuid::now_v7()));
        let gate = |v: Verdict| GateResult {
            gate_id: "tests".to_owned(),
            verdict: v,
            score: None,
            cost_usd: 0.0,
            ms: 1,
            reason: None,
            evidence_ref: None,
        };
        let mut mine = Vec::new();
        for i in 0..23u32 {
            let mut t = sample_trace("tenant-a", &format!("s{i}"));
            t.final_.total_cost_usd = 0.01 * f64::from(i + 1);
            t.final_.counterfactual_baseline_usd = 0.05;
            t.final_.escalations = i % 3;
            t.final_.served_rung = match i % 5 {
                0 => None,
                k => Some(k % 3), // rung 0, 1 and 2 in unequal counts
            };
            t.attempts[0].gates = (0..i % 4)
                .map(|k| {
                    gate([Verdict::Pass, Verdict::Fail, Verdict::Abstain][(k + i) as usize % 3])
                })
                .collect();
            t.attempts[0].truncated = i % 3 == 0;
            if i % 2 == 0 {
                let mut second = t.attempts[0].clone();
                second.rung = 1;
                second.gates = vec![gate(Verdict::Pass)];
                second.truncated = i % 4 == 0;
                t.attempts.push(second);
            }
            mine.push(t);
        }
        let (tx, handle) = open(&db_path).unwrap();
        for (i, t) in mine.iter().enumerate() {
            tx.send(t.clone()).await.unwrap();
            if i % 4 == 0 {
                let mut other = sample_trace("tenant-b", "x");
                other.final_.total_cost_usd = 100.0;
                other.attempts[0].gates = vec![gate(Verdict::Fail)];
                other.attempts[0].truncated = true;
                tx.send(other).await.unwrap();
            }
        }
        drop(tx);
        handle.await.unwrap();

        let mut want = ReceiptsSummary {
            requests: mine.len() as u64,
            ..Default::default()
        };
        for t in &mine {
            want.spent_usd += t.final_.total_cost_usd;
            want.baseline_usd += t.final_.counterfactual_baseline_usd;
            want.first_pass += u64::from(t.final_.served_rung == Some(0));
            want.escalated += u64::from(t.final_.escalations > 0);
            want.truncated += t.attempts.iter().filter(|a| a.truncated).count() as u64;
            for g in t.attempts.iter().flat_map(|a| &a.gates) {
                want.gates[match g.verdict {
                    Verdict::Pass => 0,
                    Verdict::Fail => 1,
                    Verdict::Abstain => 2,
                }] += 1;
            }
        }
        let got = tenant_receipts_summary(&db_path, "tenant-a").unwrap();
        assert_eq!(got.requests, want.requests);
        assert_eq!(got.first_pass, want.first_pass);
        assert_eq!(got.escalated, want.escalated);
        assert_eq!(got.gates, want.gates);
        assert_eq!(
            got.truncated, want.truncated,
            "only this tenant's cut-off attempts"
        );
        assert!(want.truncated > 0);
        assert!(want.first_pass > 0 && want.escalated > 0 && want.gates.iter().all(|&n| n > 0));
        assert!(
            (got.spent_usd - want.spent_usd).abs() < 1e-9,
            "{got:?} {want:?}"
        );
        assert!((got.baseline_usd - want.baseline_usd).abs() < 1e-9);
        assert_eq!(
            tenant_receipts_summary(&db_path, "nobody").unwrap(),
            ReceiptsSummary::default()
        );

        let page = load_recent_tenant_traces(&db_path, "tenant-a", 5).unwrap();
        let ids: Vec<_> = page.iter().map(|t| t.trace_id).collect();
        let newest: Vec<_> = mine.iter().rev().take(5).map(|t| t.trace_id).collect();
        assert_eq!(ids, newest);

        // Incremental: rows appended after a read are folded in exactly once.
        let (tx, handle) = open(&db_path).unwrap();
        for _ in 0..4 {
            let mut t = sample_trace("tenant-a", "late");
            t.final_.served_rung = Some(0);
            t.attempts[0].gates = vec![gate(Verdict::Abstain)];
            tx.send(t).await.unwrap();
        }
        drop(tx);
        handle.await.unwrap();
        let again = tenant_receipts_summary(&db_path, "tenant-a").unwrap();
        assert_eq!(again.requests, want.requests + 4);
        assert_eq!(again.first_pass, want.first_pass + 4);
        assert_eq!(
            again.gates,
            [want.gates[0], want.gates[1], want.gates[2] + 4]
        );
        assert_eq!(
            tenant_receipts_summary(&db_path, "tenant-a").unwrap(),
            again
        );

        // A store replaced at the same path (newest seq below the cached one) is recounted.
        let _ = std::fs::remove_file(&db_path);
        let (tx, handle) = open(&db_path).unwrap();
        tx.send(sample_trace("tenant-a", "fresh")).await.unwrap();
        drop(tx);
        handle.await.unwrap();
        assert_eq!(
            tenant_receipts_summary(&db_path, "tenant-a")
                .unwrap()
                .requests,
            1
        );
        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn writer_assigns_prev_hash_and_forms_a_valid_chain() {
        let db_path =
            std::env::temp_dir().join(format!("firstpass-store-test-{}.db", uuid::Uuid::now_v7()));
        let (tx, handle) = open(&db_path).unwrap();

        tx.try_send(sample_trace("tenant-a", "session-1")).unwrap();
        tx.try_send(sample_trace("tenant-a", "session-1")).unwrap();
        drop(tx);
        handle.await.unwrap();

        let traces = load_all_traces(&db_path).unwrap();
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].prev_hash, GENESIS_HASH);
        assert_eq!(traces[1].prev_hash, traces[0].hash().unwrap());
        verify_chain(&traces, GENESIS_HASH).unwrap();

        let _ = std::fs::remove_file(&db_path);
    }

    /// D7 cross-tenant isolation, at the store layer: with rows for tenants A and B, every
    /// tenant-scoped read for A returns only A's data, and vice-versa. The operator-wide
    /// [`load_all_traces`] still sees both (for chain verification).
    #[tokio::test]
    async fn tenant_scoped_reads_never_cross_the_boundary() {
        let db_path =
            std::env::temp_dir().join(format!("firstpass-isolation-{}.db", uuid::Uuid::now_v7()));
        let (tx, handle) = open(&db_path).unwrap();

        // Two traces for A, one for B.
        let a0 = sample_trace("tenant-a", "sa-0");
        let a1 = sample_trace("tenant-a", "sa-1");
        let b0 = sample_trace("tenant-b", "sb-0");
        let (a0_id, a1_id, b0_id) = (
            a0.trace_id.to_string(),
            a1.trace_id.to_string(),
            b0.trace_id.to_string(),
        );
        tx.try_send(a0).unwrap();
        tx.try_send(b0).unwrap();
        tx.try_send(a1).unwrap();
        drop(tx);
        handle.await.unwrap();

        // Scoped list: A sees exactly its two, B sees exactly its one.
        let a_traces = load_tenant_traces(&db_path, "tenant-a").unwrap();
        assert_eq!(a_traces.len(), 2, "A must see only A's traces");
        assert!(a_traces.iter().all(|t| t.tenant_id == "tenant-a"));
        let b_traces = load_tenant_traces(&db_path, "tenant-b").unwrap();
        assert_eq!(b_traces.len(), 1, "B must see only B's trace");
        assert!(b_traces.iter().all(|t| t.tenant_id == "tenant-b"));

        // A can prove its own trace exists but cannot see B's, and vice-versa.
        assert!(trace_exists(&db_path, "tenant-a", &a0_id).unwrap());
        assert!(!trace_exists(&db_path, "tenant-a", &b0_id).unwrap());
        assert!(trace_exists(&db_path, "tenant-b", &b0_id).unwrap());
        assert!(!trace_exists(&db_path, "tenant-b", &a1_id).unwrap());

        // The view is likewise scoped: cross-tenant reads are indistinguishable from a miss.
        assert!(
            load_trace_view(&db_path, "tenant-a", &a0_id)
                .unwrap()
                .is_some()
        );
        assert!(
            load_trace_view(&db_path, "tenant-a", &b0_id)
                .unwrap()
                .is_none()
        );
        assert!(
            load_trace_view(&db_path, "tenant-b", &a0_id)
                .unwrap()
                .is_none()
        );

        // A tenant that owns nothing sees nothing.
        assert!(load_tenant_traces(&db_path, "ghost").unwrap().is_empty());

        // Operator-wide read still spans both, and the global chain stays valid.
        let all = load_all_traces(&db_path).unwrap();
        assert_eq!(all.len(), 3);
        verify_chain(&all, GENESIS_HASH).unwrap();

        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn deferred_verdicts_attach_on_read_without_breaking_the_chain() {
        let db_path = std::env::temp_dir().join(format!(
            "firstpass-deferred-test-{}.db",
            uuid::Uuid::now_v7()
        ));
        let (tx, handle) = open(&db_path).unwrap();
        let t0 = sample_trace("acme", "run-1");
        let t1 = sample_trace("acme", "run-1");
        let (id0, id1) = (t0.trace_id.to_string(), t1.trace_id.to_string());
        tx.try_send(t0).unwrap();
        tx.try_send(t1).unwrap();
        drop(tx);
        handle.await.unwrap();

        // A downstream outcome arrives for the first trace (e.g. "tests passed an hour later").
        let dv = DeferredVerdict {
            gate_id: "tests".to_owned(),
            verdict: Verdict::Pass,
            score: Some(Score::new(1.0).unwrap()),
            reported_at: jiff::Timestamp::now(),
            reporter: "ci".to_owned(),
        };
        append_deferred(&db_path, &id0, &dv).unwrap();
        assert!(trace_exists(&db_path, "acme", &id0).unwrap());
        assert!(!trace_exists(&db_path, "acme", "no-such-trace").unwrap());
        // Cross-tenant: the same real trace id is invisible to a different tenant.
        assert!(!trace_exists(&db_path, "other-tenant", &id0).unwrap());

        // The view surfaces the deferred verdict...
        let view = load_trace_view(&db_path, "acme", &id0).unwrap().unwrap();
        assert_eq!(view.deferred.len(), 1);
        assert_eq!(view.deferred[0].gate_id, "tests");
        assert_eq!(view.deferred[0].verdict, Verdict::Pass);
        // ...the second trace has none.
        assert!(
            load_trace_view(&db_path, "acme", &id1)
                .unwrap()
                .unwrap()
                .deferred
                .is_empty()
        );
        // Cross-tenant: another tenant cannot read this trace's view at all.
        assert!(
            load_trace_view(&db_path, "other-tenant", &id0)
                .unwrap()
                .is_none()
        );

        // THE INVARIANT: the sealed bodies are untouched, so the chain still verifies. A late
        // outcome can never alter a past decision's hash.
        let traces = load_all_traces(&db_path).unwrap();
        assert!(
            traces.iter().all(|t| t.deferred.is_empty()),
            "sealed records stay deferred-free"
        );
        verify_chain(&traces, GENESIS_HASH).unwrap();

        let _ = std::fs::remove_file(&db_path);
    }

    // ── Durable-receipts tests ────────────────────────────────────────────────

    /// Test (2): `append_to_spill` writes valid JSON lines to the spill file.
    #[test]
    fn durable_append_to_spill_writes_valid_json_line() {
        let db_path =
            std::env::temp_dir().join(format!("firstpass-spill-write-{}.db", uuid::Uuid::now_v7()));
        let spill_p = spill_path(&db_path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&spill_p)
            .unwrap();
        let handle = Arc::new(std::sync::Mutex::new(file));

        let t = sample_trace("t", "s");
        append_to_spill(&handle, &t).unwrap();

        let content = std::fs::read_to_string(&spill_p).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1, "one line per spilled trace");
        let parsed: Trace = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.trace_id, t.trace_id);

        let _ = std::fs::remove_file(&spill_p);
    }

    /// Test (3): on boot with a pre-populated spill file, the writer drains it and
    /// `verify_chain` passes over the full store.
    #[tokio::test]
    async fn drain_on_boot_restores_spilled_traces_and_chain_is_valid() {
        let db_path =
            std::env::temp_dir().join(format!("firstpass-drain-boot-{}.db", uuid::Uuid::now_v7()));
        let spill_p = spill_path(&db_path);

        // Pre-populate spill file with 2 traces (simulating backpressure from a previous run).
        let t1 = sample_trace("t", "s1");
        let t2 = sample_trace("t", "s2");
        {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .read(true)
                .open(&spill_p)
                .unwrap();
            let h = Arc::new(std::sync::Mutex::new(file));
            append_to_spill(&h, &t1).unwrap();
            append_to_spill(&h, &t2).unwrap();
        }

        // Open in durable mode — startup drain fires before any channel traffic.
        let (tx, _spill, handle) = open_with_receipts(&db_path, ReceiptsMode::Durable).unwrap();
        drop(tx); // close the channel immediately
        handle.await.unwrap();

        let traces = load_all_traces(&db_path).unwrap();
        assert_eq!(traces.len(), 2, "both spilled traces recovered");
        verify_chain(&traces, GENESIS_HASH).unwrap();

        // Spill file must be empty after a successful drain.
        let remaining = std::fs::read_to_string(&spill_p).unwrap();
        assert!(
            remaining.trim().is_empty(),
            "spill file must be empty after drain"
        );

        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(&spill_p);
    }

    /// Test (4): spilled traces land before later channel traces in the store, preserving the
    /// hash chain's append-only ordering.
    #[tokio::test]
    async fn spilled_traces_land_before_later_channel_traces() {
        let db_path =
            std::env::temp_dir().join(format!("firstpass-spill-order-{}.db", uuid::Uuid::now_v7()));
        let spill_p = spill_path(&db_path);

        // Pre-populate the spill file (as if these traces were spilled under backpressure).
        let spilled = sample_trace("t", "spilled");
        {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .read(true)
                .open(&spill_p)
                .unwrap();
            let h = Arc::new(std::sync::Mutex::new(file));
            append_to_spill(&h, &spilled).unwrap();
        }

        // Open in durable mode: startup drain inserts the spilled trace first.
        let (tx, _spill, handle) = open_with_receipts(&db_path, ReceiptsMode::Durable).unwrap();
        // Send a new channel trace after the store is open (arrives after startup drain).
        let channel_trace = sample_trace("t", "channel");
        tx.try_send(channel_trace.clone()).unwrap();
        drop(tx);
        handle.await.unwrap();

        let traces = load_all_traces(&db_path).unwrap();
        assert_eq!(traces.len(), 2, "spilled + channel trace");
        // Spilled trace must come first (lower seq).
        assert_eq!(
            traces[0].trace_id, spilled.trace_id,
            "spilled trace must precede channel trace"
        );
        assert_eq!(
            traces[1].trace_id, channel_trace.trace_id,
            "channel trace must follow spilled trace"
        );
        verify_chain(&traces, GENESIS_HASH).unwrap();

        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(&spill_p);
    }

    /// An existing `experiment_pairs` table without `ch_error`/`ch_truncated` columns gets
    /// those columns added by `migrate()`, and old rows load with defaults of `false`/`0`.
    ///
    /// Mutation test: if the `ALTER TABLE` branch is removed from `migrate`, the load panics on
    /// a missing column or returns a wrong index — either way this assertion fails.
    #[tokio::test]
    async fn experiment_pairs_migration_adds_columns_to_old_schema() {
        let db_path = std::env::temp_dir().join(format!(
            "firstpass-pairs-migrate-{}.db",
            uuid::Uuid::now_v7()
        ));
        // Create an old-schema DB (no ch_error / ch_truncated).
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE experiment_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant TEXT NOT NULL,
                    route_ix INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    x_inc REAL NOT NULL,
                    x_ch REAL NOT NULL,
                    challenger_cost_usd REAL NOT NULL,
                    ts TEXT NOT NULL
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO experiment_pairs \
                 (tenant, route_ix, fingerprint, trace_id, x_inc, x_ch, challenger_cost_usd, ts) \
                 VALUES ('t', 0, 'fp', '00000000-0000-0000-0000-000000000001', 0.5, 0.5, 0.01, \
                         '2026-01-01T00:00:00Z')",
                [],
            )
            .unwrap();
        }
        // Open via `open()` so `migrate()` runs and adds the missing columns, then close.
        let (tx, handle) = open(&db_path).unwrap();
        drop(tx);
        handle.await.unwrap();
        // Now load: old row must have default values for the new columns.
        let rows = load_all_experiment_pairs(&db_path).unwrap();
        assert_eq!(rows.len(), 1, "old row must be readable after migration");
        assert!(!rows[0].ch_error, "old row defaults ch_error = false");
        assert_eq!(rows[0].ch_truncated, 0, "old row defaults ch_truncated = 0");
        let _ = std::fs::remove_file(&db_path);
    }
}
