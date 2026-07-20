//! CLI surfaces for the `firstpass` binary (SPEC §7.3/§7.4): `doctor` validates a setup before
//! you route real traffic through it, and `trace` reads recent audit records from the store. Both
//! are kept here (not in the binary) so the judgment is unit-tested.

use crate::config::ProxyConfig;
use firstpass_core::{Mode, Trace};

/// A single doctor check outcome.
#[derive(Debug, PartialEq, Eq)]
pub enum CheckStatus {
    /// Healthy.
    Ok,
    /// Works, but worth knowing (e.g. no key in env — observe still runs).
    Warn,
    /// Broken; `firstpass doctor` exits non-zero.
    Fail,
}

/// One line of the doctor report.
#[derive(Debug)]
pub struct Check {
    /// Short check name.
    pub name: String,
    /// Outcome.
    pub status: CheckStatus,
    /// Human-readable detail.
    pub detail: String,
}

impl Check {
    fn new(name: impl Into<String>, status: CheckStatus, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status,
            detail: detail.into(),
        }
    }
}

/// The result of `firstpass doctor`.
#[derive(Debug)]
pub struct DoctorReport {
    /// One entry per check, in report order.
    pub checks: Vec<Check>,
}

impl DoctorReport {
    /// Healthy iff no check failed (warnings are fine).
    #[must_use]
    pub fn healthy(&self) -> bool {
        self.checks.iter().all(|c| c.status != CheckStatus::Fail)
    }

    /// Render the report as human-readable lines (a rendering of the structured checks).
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        for c in &self.checks {
            let mark = match c.status {
                CheckStatus::Ok => "✓",
                CheckStatus::Warn => "!",
                CheckStatus::Fail => "✗",
            };
            out.push_str(&format!("{mark} {}: {}\n", c.name, c.detail));
        }
        out.push_str(if self.healthy() {
            "\nhealthy — ready to route.\n"
        } else {
            "\nnot healthy — fix the ✗ items above.\n"
        });
        out
    }
}

/// Validate a loaded config against the environment: routing sanity, provider key presence, and
/// that every configured gate's command actually exists. `env` looks up environment variables
/// (injected so this is testable).
#[must_use]
pub fn doctor(config: &ProxyConfig, env: impl Fn(&str) -> Option<String>) -> DoctorReport {
    let mut checks = Vec::new();

    // Config parsed (we hold a ProxyConfig), report the shape.
    let route_count = config.routing.as_ref().map_or(0, |c| c.routes.len());
    checks.push(Check::new(
        "config",
        CheckStatus::Ok,
        format!("default mode {:?}, {route_count} route(s)", config.mode),
    ));

    // Enforce is only meaningful with an enforce route to activate the engine.
    let enforce_routes = config.routing.as_ref().map_or(0, |c| {
        c.routes.iter().filter(|r| r.mode == Mode::Enforce).count()
    });
    if config.mode == Mode::Enforce && enforce_routes == 0 {
        checks.push(Check::new(
            "routing",
            CheckStatus::Warn,
            "mode is enforce but no enforce route is defined — all traffic will observe",
        ));
    } else {
        checks.push(Check::new(
            "routing",
            CheckStatus::Ok,
            format!("{enforce_routes} enforce route(s)"),
        ));
    }

    // A provider key in the environment. Observe uses the caller's key, so absence is a warning.
    if env("ANTHROPIC_API_KEY").is_some_and(|k| !k.is_empty()) {
        checks.push(Check::new("anthropic-key", CheckStatus::Ok, "present"));
    } else {
        checks.push(Check::new(
            "anthropic-key",
            CheckStatus::Warn,
            "ANTHROPIC_API_KEY not set — observe uses the caller's key; enforce needs one reachable",
        ));
    }

    // Every configured gate command must resolve, or that gate silently abstains at runtime.
    let path = env("PATH");
    let gate_defs = config.routing.as_ref().map_or(&[][..], |c| &c.gate_defs);
    for def in gate_defs {
        match def.cmd.first() {
            Some(program) if command_on_path(program, path.as_deref()) => checks.push(Check::new(
                format!("gate:{}", def.id),
                CheckStatus::Ok,
                format!("`{program}` found"),
            )),
            Some(program) => checks.push(Check::new(
                format!("gate:{}", def.id),
                CheckStatus::Fail,
                format!("`{program}` not found on PATH"),
            )),
            None => checks.push(Check::new(
                format!("gate:{}", def.id),
                CheckStatus::Fail,
                "empty command",
            )),
        }
    }

    // The trace store must be writable, or we'd trade the audit trail (or availability) for it.
    if can_write_db(&config.db_path) {
        checks.push(Check::new(
            "trace-store",
            CheckStatus::Ok,
            format!("{} is writable", config.db_path),
        ));
    } else {
        checks.push(Check::new(
            "trace-store",
            CheckStatus::Fail,
            format!("cannot write near {}", config.db_path),
        ));
    }

    DoctorReport { checks }
}

/// Whether `program` is runnable: an explicit path (contains a separator) that exists, or a bare
/// name found in one of `PATH`'s directories.
#[must_use]
pub fn command_on_path(program: &str, path_var: Option<&str>) -> bool {
    if program.contains('/') || program.contains('\\') {
        return std::path::Path::new(program).is_file();
    }
    let Some(path) = path_var else { return false };
    std::env::split_paths(path).any(|dir| dir.join(program).is_file())
}

/// Probe whether the trace DB's directory is writable, without creating the DB itself.
fn can_write_db(db_path: &str) -> bool {
    let path = std::path::Path::new(db_path);
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map_or_else(
            || std::path::PathBuf::from("."),
            std::path::Path::to_path_buf,
        );
    let probe = dir.join(format!(".firstpass-doctor-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Per-gate and per-rung evaluation summary computed from receipts — the operator's live
/// eval suite: how each gate is verdict-ing, how often routing escalates, where serves land.
#[derive(Debug, serde::Serialize)]
pub struct EvalsSummary {
    /// Traces aggregated (enforce mode only — observe records no gate decisions on-path).
    pub enforce_traces: usize,
    /// Total escalations across those traces.
    pub escalations: u64,
    /// gate_id → (pass, fail, abstain) counts across every attempt.
    pub gates: std::collections::BTreeMap<String, (u64, u64, u64)>,
    /// rung index → times an attempt at that rung was the served one.
    pub served_by_rung: std::collections::BTreeMap<u32, u64>,
}

/// Aggregate gate verdicts + routing behavior over `traces` (enforce only).
#[must_use]
pub fn summarize_evals(traces: &[Trace]) -> EvalsSummary {
    let mut s = EvalsSummary {
        enforce_traces: 0,
        escalations: 0,
        gates: std::collections::BTreeMap::new(),
        served_by_rung: std::collections::BTreeMap::new(),
    };
    for t in traces {
        if t.mode != Mode::Enforce {
            continue;
        }
        s.enforce_traces += 1;
        s.escalations += u64::from(t.final_.escalations);
        if let Some(rung) = t.final_.served_rung {
            *s.served_by_rung.entry(rung).or_insert(0) += 1;
        }
        for attempt in &t.attempts {
            for g in &attempt.gates {
                let e = s.gates.entry(g.gate_id.clone()).or_insert((0, 0, 0));
                match g.verdict {
                    firstpass_core::Verdict::Pass => e.0 += 1,
                    firstpass_core::Verdict::Fail => e.1 += 1,
                    firstpass_core::Verdict::Abstain => e.2 += 1,
                }
            }
        }
    }
    s
}

/// Render an [`EvalsSummary`] for humans (`--json` callers serialize the struct instead).
#[must_use]
pub fn format_evals(s: &EvalsSummary) -> String {
    if s.enforce_traces == 0 {
        return "no enforce traces yet — route some traffic in enforce mode first".to_owned();
    }
    let mut out = format!(
        "enforce traces: {} · escalations: {}\n",
        s.enforce_traces, s.escalations
    );
    out.push_str("gates (pass / fail / abstain):\n");
    for (id, (p, f, a)) in &s.gates {
        out.push_str(&format!("  {id:<24} {p:>6} / {f:>6} / {a:>6}\n"));
    }
    out.push_str("served by rung:\n");
    for (rung, n) in &s.served_by_rung {
        out.push_str(&format!("  rung {rung:<2} {n:>6}\n"));
    }
    out.trim_end().to_owned()
}

/// Outcome of an independent receipt-chain verification — the compliance artifact.
#[derive(Debug, serde::Serialize)]
pub struct VerifyReport {
    /// Receipts checked.
    pub receipts: usize,
    /// `true` iff the hash chain re-derives cleanly from genesis (tamper-evident proof).
    pub valid: bool,
    /// On failure, the 0-based index of the first broken link and why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub broken_at: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Independently re-derive the hash chain over `traces` from genesis — the same computation an
/// external auditor runs, with no trust in the proxy or the database. A single altered or
/// reordered receipt breaks the chain at its index.
#[must_use]
pub fn verify_receipts(traces: &[Trace]) -> VerifyReport {
    match firstpass_core::verify_chain(traces, firstpass_core::GENESIS_HASH) {
        Ok(()) => VerifyReport {
            receipts: traces.len(),
            valid: true,
            broken_at: None,
            detail: None,
        },
        Err(firstpass_core::Error::ChainBroken { index, detail }) => VerifyReport {
            receipts: traces.len(),
            valid: false,
            broken_at: Some(index),
            detail: Some(detail),
        },
        Err(e) => VerifyReport {
            receipts: traces.len(),
            valid: false,
            broken_at: None,
            detail: Some(e.to_string()),
        },
    }
}

/// Parse a JSONL receipt export (one [`Trace`] per line) back into traces, for offline
/// verification. Blank lines are skipped; a malformed line fails loudly with its line number.
///
/// # Errors
/// Returns the 1-based line number and parse error of the first unparseable line.
pub fn parse_receipt_jsonl(text: &str) -> Result<Vec<Trace>, String> {
    let mut traces = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let t: Trace = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        traces.push(t);
    }
    Ok(traces)
}

/// Render `traces` as a JSONL receipt export — one sealed receipt per line, in chain order.
/// This is the artifact an operator hands an auditor; the deferred-verdict side table is never
/// included (it is not part of the hashed body).
#[must_use]
pub fn export_receipts_jsonl(traces: &[Trace]) -> String {
    let mut out = String::new();
    for t in traces {
        if let Ok(line) = serde_json::to_string(t) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out
}

/// Aggregated spend/savings over a set of traces — the number the operator screenshots.
/// Pure so it's unit-testable; `firstpass savings` feeds it the trace store.
#[derive(Debug, serde::Serialize)]
pub struct SavingsSummary {
    /// Traces aggregated.
    pub traces: usize,
    /// Enforce-mode traces (the ones where routing actually decided).
    pub enforce_traces: usize,
    /// USD actually spent (model + gate calls), summed over all traces.
    pub spent_usd: f64,
    /// USD spent on gates alone (the price of proof).
    pub gate_usd: f64,
    /// What always calling the top rung would have cost (§9.1 counterfactual), summed.
    pub baseline_usd: f64,
    /// `baseline - spent` — the savings the receipts prove.
    pub savings_usd: f64,
    /// Savings as a fraction of baseline, `0.0` when there is no baseline.
    pub savings_pct: f64,
}

/// Aggregate spend vs the always-top counterfactual over `traces`.
#[must_use]
pub fn summarize_savings(traces: &[Trace]) -> SavingsSummary {
    let mut s = SavingsSummary {
        traces: traces.len(),
        enforce_traces: 0,
        spent_usd: 0.0,
        gate_usd: 0.0,
        baseline_usd: 0.0,
        savings_usd: 0.0,
        savings_pct: 0.0,
    };
    for t in traces {
        if t.mode == Mode::Enforce {
            s.enforce_traces += 1;
        }
        s.spent_usd += t.final_.total_cost_usd;
        s.gate_usd += t.final_.gate_cost_usd;
        s.baseline_usd += t.final_.counterfactual_baseline_usd;
    }
    s.savings_usd = s.baseline_usd - s.spent_usd;
    if s.baseline_usd > 0.0 {
        s.savings_pct = s.savings_usd / s.baseline_usd;
    }
    s
}

/// Render a [`SavingsSummary`] for humans (`--json` callers serialize the struct instead).
#[must_use]
pub fn format_savings(s: &SavingsSummary) -> String {
    if s.traces == 0 {
        return "no traces recorded yet — route some traffic through the proxy first".to_owned();
    }
    format!(
        "traces: {} ({} enforce)\nspent:    ${:.4}  (gates ${:.4})\nbaseline: ${:.4}  (always top rung)\nsavings:  ${:.4}  ({:.1}%)",
        s.traces,
        s.enforce_traces,
        s.spent_usd,
        s.gate_usd,
        s.baseline_usd,
        s.savings_usd,
        s.savings_pct * 100.0
    )
}

/// Render the most recent `limit` traces as JSON lines — machine-first (SPEC §0.2): each line is a
/// full [`Trace`], newest last.
#[must_use]
pub fn format_traces(traces: &[Trace], limit: usize) -> String {
    let mut lines: Vec<String> = traces
        .iter()
        .rev()
        .take(limit)
        .filter_map(|t| serde_json::to_string(t).ok())
        .collect();
    lines.reverse();
    if lines.is_empty() {
        "no traces recorded yet".to_owned()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with(toml: Option<&str>, db_path: &str) -> ProxyConfig {
        ProxyConfig::from_lookup(|k| match k {
            "FIRSTPASS_MODE" if toml.is_some() => Some("enforce".to_owned()),
            "FIRSTPASS_CONFIG_TOML" => toml.map(str::to_owned),
            "FIRSTPASS_DB" => Some(db_path.to_owned()),
            _ => None,
        })
        .unwrap()
    }

    #[test]
    fn command_on_path_finds_real_and_rejects_fake() {
        let path = std::env::var("PATH").ok();
        assert!(command_on_path("sh", path.as_deref()), "sh is on PATH");
        assert!(!command_on_path(
            "firstpass-definitely-not-a-real-binary",
            path.as_deref()
        ));
        assert!(!command_on_path("/nonexistent/abs/path", None));
    }

    #[test]
    fn doctor_flags_a_missing_gate_binary() {
        let toml = "[[route]]\nmatch = {}\nmode = \"enforce\"\nladder = [\"anthropic/claude-haiku-4-5\"]\ngates = [\"good\", \"bad\"]\n\
                    [[gate]]\nid = \"good\"\ncmd = [\"sh\"]\n\
                    [[gate]]\nid = \"bad\"\ncmd = [\"firstpass-nope-not-real\"]\n";
        let db = std::env::temp_dir().join("firstpass-doctor-test.db");
        let config = config_with(Some(toml), db.to_str().unwrap());

        // Real PATH so `sh` resolves; no ANTHROPIC_API_KEY -> a warning, not a failure.
        let report = doctor(&config, |k| match k {
            "PATH" => std::env::var("PATH").ok(),
            _ => None,
        });

        assert!(
            !report.healthy(),
            "a missing gate binary must fail the report"
        );
        let bad = report.checks.iter().find(|c| c.name == "gate:bad").unwrap();
        assert_eq!(bad.status, CheckStatus::Fail);
        let good = report
            .checks
            .iter()
            .find(|c| c.name == "gate:good")
            .unwrap();
        assert_eq!(good.status, CheckStatus::Ok);
        let key = report
            .checks
            .iter()
            .find(|c| c.name == "anthropic-key")
            .unwrap();
        assert_eq!(key.status, CheckStatus::Warn);
    }

    #[test]
    fn doctor_is_healthy_for_a_plain_observe_setup() {
        let db = std::env::temp_dir().join("firstpass-doctor-ok.db");
        let config = config_with(None, db.to_str().unwrap());
        let report = doctor(&config, |k| {
            (k == "ANTHROPIC_API_KEY").then(|| "sk-test".to_owned())
        });
        assert!(report.healthy(), "{}", report.render());
    }

    #[test]
    fn format_traces_handles_empty() {
        assert_eq!(format_traces(&[], 10), "no traces recorded yet");
    }

    #[test]
    fn savings_summary_empty_and_nonempty() {
        let empty = summarize_savings(&[]);
        assert_eq!(empty.traces, 0);
        assert!(format_savings(&empty).contains("no traces"));
    }

    #[test]
    fn evals_summary_empty_zero_state() {
        let s = summarize_evals(&[]);
        assert_eq!(s.enforce_traces, 0);
        assert!(format_evals(&s).contains("no enforce traces"));
    }

    #[test]
    fn verify_and_export_round_trip_detects_tampering() {
        use firstpass_core::{
            Attempt, Features, FinalOutcome, GENESIS_HASH, PolicyRef, RequestInfo, ServedFrom,
            TaskKind, Verdict,
        };

        // Build a valid 3-link chain: each prev_hash = the previous record's hash.
        let mk = |prev: &str, session: &str| Trace {
            trace_id: uuid::Uuid::now_v7(),
            prev_hash: prev.to_owned(),
            tenant_id: "default".to_owned(),
            session_id: session.to_owned(),
            ts: jiff::Timestamp::UNIX_EPOCH,
            mode: Mode::Observe,
            policy: PolicyRef {
                id: "observe-passthrough@v0".to_owned(),
                explore: false,
                propensity: None,
            },
            request: RequestInfo {
                api: "anthropic.messages".to_owned(),
                prompt_hash: "deadbeef".to_owned(),
                features: Features::new(TaskKind::Other),
            },
            attempts: vec![Attempt {
                rung: 0,
                model: "anthropic/claude-haiku-4-5".to_owned(),
                provider: "anthropic".to_owned(),
                in_tokens: 10,
                out_tokens: 5,
                cost_usd: 0.001,
                latency_ms: 12,
                gates: vec![],
                verdict: Verdict::Pass,
            }],
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
        let t0 = mk(GENESIS_HASH, "s0");
        let t1 = mk(&t0.hash().unwrap(), "s1");
        let t2 = mk(&t1.hash().unwrap(), "s2");
        let chain = vec![t0, t1, t2];

        // Clean chain verifies.
        let report = verify_receipts(&chain);
        assert!(report.valid, "intact chain must verify: {report:?}");
        assert_eq!(report.receipts, 3);

        // Export → parse round-trips and still verifies (the auditor's offline path).
        let jsonl = export_receipts_jsonl(&chain);
        assert_eq!(jsonl.lines().count(), 3);
        let reparsed = parse_receipt_jsonl(&jsonl).expect("export must re-parse");
        assert!(
            verify_receipts(&reparsed).valid,
            "round-tripped chain must verify"
        );

        // Tamper with a middle receipt's body → verification catches it at that index.
        let mut tampered = reparsed;
        tampered[1].final_.total_cost_usd = 999.0; // alter a sealed field
        let bad = verify_receipts(&tampered);
        assert!(!bad.valid, "a mutated receipt must break the chain");
        assert_eq!(
            bad.broken_at,
            Some(2),
            "the break surfaces at the next link"
        );

        // Reordering is also caught.
        let mut reordered = parse_receipt_jsonl(&jsonl).unwrap();
        reordered.swap(0, 1);
        assert!(
            !verify_receipts(&reordered).valid,
            "reordering breaks the chain"
        );
    }

    #[test]
    fn parse_receipt_jsonl_reports_bad_line() {
        let err = parse_receipt_jsonl("not json\n").unwrap_err();
        assert!(err.starts_with("line 1:"), "must name the bad line: {err}");
        assert!(
            parse_receipt_jsonl("\n\n").unwrap().is_empty(),
            "blank lines skipped"
        );
    }
}
