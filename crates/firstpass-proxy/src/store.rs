//! Tamper-evident trace storage: a background SQLite writer fed by an unbounded channel.
//!
//! The hot request path never blocks on disk: it sends a [`Trace`] down an unbounded channel
//! and returns immediately. A single background task owns the SQLite connection, assigns
//! each trace's `prev_hash` from the current chain head, computes its `hash`, and appends it
//! (SPEC §9: the hash chain is only meaningful if every writer agrees on chain order, which a
//! single writer task guarantees for free).

use std::path::Path;
use std::time::Duration;

use firstpass_core::{DeferredVerdict, GENESIS_HASH, Score, Trace, Verdict};
use rusqlite::Connection;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

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
}

/// Sending half of the trace channel; cheap to clone, safe to share across request handlers.
/// Sending is fire-and-forget — the hot path never awaits the writer.
pub type TraceSender = mpsc::UnboundedSender<Trace>;

/// Open (creating if needed) the SQLite trace database, migrate its schema, and spawn the
/// background writer task.
///
/// Returns a [`TraceSender`] for the hot path and the writer's [`JoinHandle`]. The writer
/// exits cleanly once every clone of the sender is dropped.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] if the database cannot be opened or migrated.
pub fn open(db_path: impl AsRef<Path>) -> Result<(TraceSender, JoinHandle<()>), StoreError> {
    let conn = connect(db_path.as_ref())?;
    migrate(&conn)?;

    let (tx, rx) = mpsc::unbounded_channel::<Trace>();
    let handle = tokio::task::spawn_blocking(move || writer_loop(conn, rx));
    Ok((tx, handle))
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

/// The writer's main loop: runs on a blocking-pool thread for the lifetime of the store.
/// Never panics on a bad trace — logs and drops it, so one malformed record can't wedge the
/// whole audit pipeline.
fn writer_loop(conn: Connection, mut rx: mpsc::UnboundedReceiver<Trace>) {
    let mut head = match current_head(&conn) {
        Ok(head) => head,
        Err(err) => {
            tracing::error!(%err, "trace writer: failed to load chain head, stopping");
            return;
        }
    };

    while let Some(mut trace) = rx.blocking_recv() {
        trace.prev_hash = head.clone();
        let hash = match trace.hash() {
            Ok(hash) => hash,
            Err(err) => {
                tracing::error!(%err, trace_id = %trace.trace_id, "trace writer: failed to hash trace, dropping");
                continue;
            }
        };
        if let Err(err) = insert(&conn, &trace, &hash) {
            tracing::error!(%err, trace_id = %trace.trace_id, "trace writer: failed to persist trace, dropping");
            continue;
        }
        head = hash;
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

/// Whether a trace with `trace_id` exists — used to reject feedback for unknown traces.
///
/// # Errors
/// Returns [`StoreError::Sqlite`] on a database error.
pub fn trace_exists(db_path: impl AsRef<Path>, trace_id: &str) -> Result<bool, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let n: i64 = conn.query_row(
        "SELECT COUNT(1) FROM traces WHERE trace_id = ?1",
        [trace_id],
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

/// Load a single trace by id with its deferred verdicts merged into `deferred` — the **view**
/// for display/inspection. This is deliberately separate from [`load_all_traces`]: merging
/// deferred verdicts changes the record, so a merged trace must NOT be fed to `verify_chain`
/// (chain verification always runs on the sealed bodies from [`load_all_traces`]).
///
/// # Errors
/// Returns [`StoreError::Sqlite`] / [`StoreError::Json`] on database or decode errors.
pub fn load_trace_view(
    db_path: impl AsRef<Path>,
    trace_id: &str,
) -> Result<Option<Trace>, StoreError> {
    let conn = connect(db_path.as_ref())?;
    let body: Option<String> = conn
        .query_row(
            "SELECT body FROM traces WHERE trace_id = ?1",
            [trace_id],
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
        };
        trace.recompute_savings();
        trace
    }

    #[tokio::test]
    async fn writer_assigns_prev_hash_and_forms_a_valid_chain() {
        let db_path =
            std::env::temp_dir().join(format!("firstpass-store-test-{}.db", uuid::Uuid::now_v7()));
        let (tx, handle) = open(&db_path).unwrap();

        tx.send(sample_trace("tenant-a", "session-1")).unwrap();
        tx.send(sample_trace("tenant-a", "session-1")).unwrap();
        drop(tx);
        handle.await.unwrap();

        let traces = load_all_traces(&db_path).unwrap();
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].prev_hash, GENESIS_HASH);
        assert_eq!(traces[1].prev_hash, traces[0].hash().unwrap());
        verify_chain(&traces, GENESIS_HASH).unwrap();

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
        tx.send(t0).unwrap();
        tx.send(t1).unwrap();
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
        assert!(trace_exists(&db_path, &id0).unwrap());
        assert!(!trace_exists(&db_path, "no-such-trace").unwrap());

        // The view surfaces the deferred verdict...
        let view = load_trace_view(&db_path, &id0).unwrap().unwrap();
        assert_eq!(view.deferred.len(), 1);
        assert_eq!(view.deferred[0].gate_id, "tests");
        assert_eq!(view.deferred[0].verdict, Verdict::Pass);
        // ...the second trace has none.
        assert!(
            load_trace_view(&db_path, &id1)
                .unwrap()
                .unwrap()
                .deferred
                .is_empty()
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
}
