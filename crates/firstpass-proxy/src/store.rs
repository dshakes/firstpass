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
use rusqlite::Connection;
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
        CREATE INDEX IF NOT EXISTS deferred_trace_idx ON deferred_verdicts(trace_id);",
    )?;
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
            out_tokens: 5,
            cost_usd: 0.001,
            latency_ms: 12,
            gates: vec![],
            verdict: Verdict::Pass,
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
            },
            probe: None,
        };
        trace.recompute_savings();
        trace
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
}
