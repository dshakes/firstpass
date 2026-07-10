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
}
