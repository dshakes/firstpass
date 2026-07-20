//! Minimal MCP (Model Context Protocol) stdio server (SPEC §0.2/§7.4): lets an agent inspect and
//! correct its own routing — read the audit traces, and submit feedback — over JSON-RPC 2.0.
//!
//! Hand-rolled over `serde_json` (newline-delimited JSON-RPC on stdin/stdout). The surface is three
//! methods — `initialize`, `tools/list`, `tools/call` — so a dependency-free handler we can
//! unit-test by value beats pulling in an async MCP framework. [`handle_rpc`] is the pure core;
//! [`serve_stdio`] is the thin transport loop around it.

use firstpass_core::{DeferredVerdict, Score, Verdict};
use serde_json::{Value, json};

use crate::store;

/// MCP protocol revision this server implements.
const PROTOCOL_VERSION: &str = "2024-11-05";

/// The tools this server exposes, with their JSON-Schema input contracts (agent-first: an agent
/// discovers them at runtime via `tools/list`).
fn tool_schemas() -> Value {
    json!([
        {
            "name": "list_traces",
            "description": "List the most recent Firstpass audit traces (routing decisions + receipts).",
            "inputSchema": {
                "type": "object",
                "properties": { "limit": { "type": "integer", "description": "Max traces to return (default 20)." } }
            }
        },
        {
            "name": "get_trace",
            "description": "Fetch a single audit trace by id, with any deferred verdicts merged in.",
            "inputSchema": {
                "type": "object",
                "properties": { "trace_id": { "type": "string" } },
                "required": ["trace_id"]
            }
        },
        {
            "name": "get_savings",
            "description": "Aggregate spend vs the always-top counterfactual from this deployment's own receipts — measured savings, not a marketing number.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "get_evals",
            "description": "Per-gate verdict rates, escalation count, and serve-by-rung distribution computed from receipts — the live eval suite.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "submit_feedback",
            "description": "Attach a downstream outcome (deferred verdict) to a past decision, closing the feedback loop.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "trace_id": { "type": "string" },
                    "gate_id":  { "type": "string" },
                    "verdict":  { "type": "string", "enum": ["pass", "fail", "abstain"] },
                    "score":    { "type": "number" },
                    "reporter": { "type": "string" }
                },
                "required": ["trace_id", "gate_id", "verdict", "reporter"]
            }
        }
    ])
}

/// Handle one JSON-RPC message. Returns the response for a request, or `None` for a notification
/// (a message with no `id`, which must not be answered). Tool calls touch the trace store at
/// `db_path`; everything else is pure.
#[must_use]
pub fn handle_rpc(req: &Value, db_path: &str, tenant: &str) -> Option<Value> {
    let is_notification = req.get("id").is_none();
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();

    match method {
        "initialize" => Some(ok(
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "firstpass", "version": env!("CARGO_PKG_VERSION") }
            }),
        )),
        "tools/list" => Some(ok(id, json!({ "tools": tool_schemas() }))),
        "tools/call" => Some(handle_tool_call(id, req.get("params"), db_path, tenant)),
        "ping" => Some(ok(id, json!({}))),
        // Notifications (e.g. `notifications/initialized`) and anything else without an id: silent.
        _ if is_notification => None,
        other => Some(err(id, -32601, &format!("method not found: {other}"))),
    }
}

/// Dispatch a `tools/call`. Tool-level failures come back as an `isError` result (MCP convention),
/// not a protocol error, so the agent can read the reason as content.
fn handle_tool_call(id: Value, params: Option<&Value>, db_path: &str, tenant: &str) -> Value {
    let name = params
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let args = params
        .and_then(|p| p.get("arguments"))
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = match name {
        "list_traces" => tool_list_traces(&args, db_path, tenant),
        "get_savings" => tool_get_savings(db_path, tenant),
        "get_evals" => tool_get_evals(db_path, tenant),
        "get_trace" => tool_get_trace(&args, db_path, tenant),
        "submit_feedback" => tool_submit_feedback(&args, db_path, tenant),
        other => Err(format!("unknown tool: {other}")),
    };

    match result {
        Ok(text) => ok(id, json!({ "content": [{ "type": "text", "text": text }] })),
        Err(msg) => ok(
            id,
            json!({ "content": [{ "type": "text", "text": msg }], "isError": true }),
        ),
    }
}

fn tool_list_traces(args: &Value, db_path: &str, tenant: &str) -> Result<String, String> {
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(20, |n| n as usize);
    // Mirror `firstpass trace`: an absent/unreadable store (no traffic yet) is "no traces", not an
    // error — unlike get_trace/submit_feedback, which name a specific trace and should fail loudly.
    // Tenant-scoped (ADR 0004 §D3): the reader only ever sees its own tenant's traces.
    let all = store::load_tenant_traces(db_path, tenant).unwrap_or_default();
    let start = all.len().saturating_sub(limit);
    let recent = &all[start..];
    serde_json::to_string(recent).map_err(|e| format!("encode error: {e}"))
}

fn tool_get_savings(db_path: &str, tenant: &str) -> Result<String, String> {
    // Absent/unreadable store mirrors list_traces: a zero summary, not an error.
    let traces = store::load_tenant_traces(db_path, tenant).unwrap_or_default();
    serde_json::to_string(&crate::cli::summarize_savings(&traces))
        .map_err(|e| format!("encode error: {e}"))
}

fn tool_get_evals(db_path: &str, tenant: &str) -> Result<String, String> {
    let traces = store::load_tenant_traces(db_path, tenant).unwrap_or_default();
    serde_json::to_string(&crate::cli::summarize_evals(&traces))
        .map_err(|e| format!("encode error: {e}"))
}

fn tool_get_trace(args: &Value, db_path: &str, tenant: &str) -> Result<String, String> {
    let trace_id = args
        .get("trace_id")
        .and_then(Value::as_str)
        .ok_or("missing `trace_id`")?;
    match store::load_trace_view(db_path, tenant, trace_id)
        .map_err(|e| format!("store error: {e}"))?
    {
        Some(trace) => serde_json::to_string(&trace).map_err(|e| format!("encode error: {e}")),
        None => Err(format!("unknown trace_id {trace_id:?}")),
    }
}

fn tool_submit_feedback(args: &Value, db_path: &str, tenant: &str) -> Result<String, String> {
    let trace_id = args
        .get("trace_id")
        .and_then(Value::as_str)
        .ok_or("missing `trace_id`")?;
    let gate_id = args
        .get("gate_id")
        .and_then(Value::as_str)
        .ok_or("missing `gate_id`")?;
    let reporter = args
        .get("reporter")
        .and_then(Value::as_str)
        .ok_or("missing `reporter`")?;
    let verdict = match args.get("verdict").and_then(Value::as_str) {
        Some("pass") => Verdict::Pass,
        Some("fail") => Verdict::Fail,
        Some("abstain") => Verdict::Abstain,
        other => return Err(format!("invalid verdict {other:?}")),
    };
    let score = match args.get("score") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let s = v.as_f64().ok_or("score must be a number")?;
            Some(Score::new(s).map_err(|_| format!("score {s} out of range [0,1]"))?)
        }
    };

    // Tenant-scoped existence check (ADR 0004 §D3/§D4): a trace owned by another tenant reads as
    // unknown, so an agent cannot attach feedback across the tenant boundary.
    if !store::trace_exists(db_path, tenant, trace_id).map_err(|e| format!("store error: {e}"))? {
        return Err(format!("unknown trace_id {trace_id:?}"));
    }
    let dv = DeferredVerdict {
        gate_id: gate_id.to_owned(),
        verdict,
        score,
        reported_at: jiff::Timestamp::now(),
        reporter: reporter.to_owned(),
    };
    store::append_deferred(db_path, trace_id, &dv).map_err(|e| format!("store error: {e}"))?;
    Ok(json!({ "status": "recorded", "trace_id": trace_id }).to_string())
}

fn ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn err(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Serve MCP over stdio: read newline-delimited JSON-RPC from stdin, write responses to stdout.
/// Blocks until stdin closes. Synchronous by design — one request in flight at a time.
///
/// All store reads/writes are scoped to `tenant` (ADR 0004 §D3): the reader is bound to a single
/// tenant for its whole session and never reads or writes across the boundary. Single-operator
/// runs pass the default tenant, so behavior is unchanged.
///
/// # Errors
/// Returns an [`std::io::Error`] if reading stdin or writing stdout fails.
pub fn serve_stdio(db_path: &str, tenant: &str) -> std::io::Result<()> {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(req) => handle_rpc(&req, db_path, tenant),
            Err(_) => Some(err(Value::Null, -32700, "parse error")),
        };
        if let Some(resp) = response {
            writeln!(stdout, "{resp}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_db() -> String {
        std::env::temp_dir()
            .join(format!("firstpass-mcp-{}.db", uuid::Uuid::now_v7()))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn initialize_announces_protocol_and_server() {
        let req = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" });
        let resp = handle_rpc(&req, "unused.db", "default").unwrap();
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(resp["result"]["serverInfo"]["name"], "firstpass");
        assert_eq!(resp["id"], 1);
    }

    #[test]
    fn tools_list_exposes_the_three_tools() {
        let req = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
        let resp = handle_rpc(&req, "unused.db", "default").unwrap();
        let names: Vec<&str> = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(names, ["list_traces", "get_trace", "submit_feedback"]);
    }

    #[test]
    fn notifications_get_no_response() {
        let req = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(handle_rpc(&req, "unused.db", "default").is_none());
    }

    #[test]
    fn unknown_method_errors_for_requests_but_not_notifications() {
        let request = json!({ "jsonrpc": "2.0", "id": 9, "method": "does/not/exist" });
        let resp = handle_rpc(&request, "unused.db", "default").unwrap();
        assert_eq!(resp["error"]["code"], -32601);

        let notification = json!({ "jsonrpc": "2.0", "method": "does/not/exist" });
        assert!(handle_rpc(&notification, "unused.db", "default").is_none());
    }

    #[test]
    fn list_traces_on_empty_store_is_empty_not_error() {
        let db = tmp_db();
        let req = json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                          "params": { "name": "list_traces", "arguments": {} } });
        let resp = handle_rpc(&req, &db, "default").unwrap();
        assert!(
            resp["result"]["isError"].is_null(),
            "empty store is not an error"
        );
        assert_eq!(resp["result"]["content"][0]["text"], "[]");
    }

    #[test]
    fn get_trace_unknown_is_tool_error() {
        let db = tmp_db();
        // Touch the store so the file exists but has no such trace.
        let _ = store::load_all_traces(&db);
        let req = json!({ "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                          "params": { "name": "get_trace", "arguments": { "trace_id": "nope" } } });
        let resp = handle_rpc(&req, &db, "default").unwrap();
        assert_eq!(resp["result"]["isError"], true);
    }

    #[test]
    fn submit_feedback_validates_and_rejects_unknown_trace() {
        let db = tmp_db();
        let _ = store::load_all_traces(&db);

        // Bad verdict → tool error.
        let bad = json!({ "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": { "name": "submit_feedback", "arguments": {
                "trace_id": "t", "gate_id": "g", "verdict": "banana", "reporter": "ci" } } });
        assert_eq!(
            handle_rpc(&bad, &db, "default").unwrap()["result"]["isError"],
            true
        );

        // Valid shape but unknown trace → tool error.
        let unknown = json!({ "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": { "name": "submit_feedback", "arguments": {
                "trace_id": "missing", "gate_id": "g", "verdict": "pass", "reporter": "ci" } } });
        assert_eq!(
            handle_rpc(&unknown, &db, "default").unwrap()["result"]["isError"],
            true
        );
    }

    #[test]
    fn unknown_tool_is_tool_error() {
        let req = json!({ "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                          "params": { "name": "no_such_tool", "arguments": {} } });
        let resp = handle_rpc(&req, "unused.db", "default").unwrap();
        assert_eq!(resp["result"]["isError"], true);
    }

    #[test]
    fn savings_and_evals_tools_are_listed_and_zero_state_on_fresh_store() {
        let db = tmp_db();
        let db_str = db.as_str();

        let listed = handle_rpc(
            &json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }),
            db_str,
            "default",
        )
        .unwrap();
        let names: Vec<&str> = listed["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert!(names.contains(&"get_savings") && names.contains(&"get_evals"));

        for tool in ["get_savings", "get_evals"] {
            let out = handle_rpc(
                &json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                         "params": { "name": tool, "arguments": {} } }),
                db_str,
                "default",
            )
            .unwrap();
            let text = out["result"]["content"][0]["text"].as_str().unwrap();
            let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
            assert!(
                parsed.is_object(),
                "{tool} must return a JSON summary even on a fresh store"
            );
        }
    }
}
