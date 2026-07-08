//! Tamper-evident trace storage: a background SQLite writer fed by an unbounded channel.
//!
//! The hot request path never blocks on disk: it sends a [`Trace`] down an unbounded channel
//! and returns immediately. A single background task owns the SQLite connection, assigns
//! each trace's `prev_hash` from the current chain head, computes its `hash`, and appends it
//! (SPEC §9: the hash chain is only meaningful if every writer agrees on chain order, which a
//! single writer task guarantees for free).

use std::path::Path;
use std::time::Duration;

use firstpass_core::{GENESIS_HASH, Trace};
use rusqlite::Connection;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

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
    let conn = Connection::open(db_path.as_ref())?;
    conn.busy_timeout(Duration::from_secs(5))?;
    migrate(&conn)?;

    let (tx, rx) = mpsc::unbounded_channel::<Trace>();
    let handle = tokio::task::spawn_blocking(move || writer_loop(conn, rx));
    Ok((tx, handle))
}

fn migrate(conn: &Connection) -> Result<(), StoreError> {
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
        CREATE INDEX IF NOT EXISTS traces_session_idx ON traces(session);",
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
    let conn = Connection::open(db_path.as_ref())?;
    conn.busy_timeout(Duration::from_secs(5))?;
    let mut stmt = conn.prepare("SELECT body FROM traces ORDER BY seq ASC")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut traces = Vec::new();
    for row in rows {
        traces.push(serde_json::from_str(&row?)?);
    }
    Ok(traces)
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
}
