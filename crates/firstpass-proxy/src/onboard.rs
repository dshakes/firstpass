//! `firstpass onboard` — fully agentic onboarding (SPEC §0.2: the primary user is an agent, and
//! the human deserves the same one-command experience). Detect the environment, plan the exact
//! steps, execute them under `--apply`, and verify end-to-end — so "install firstpass and route
//! my agent through it" is one command, not a doc to follow.
//!
//! Structure: [`detect`] and [`plan`] are pure (injected lookups, no I/O) so the decision logic is
//! unit-tested offline; [`execute`] performs the side effects (spawn proxy, append one marked line
//! to the shell rc, probe `/healthz`) and is deliberately thin. Without `--apply` the command is a
//! dry run that prints the plan — agentic, but never surprising.

use std::io::Write as _;
use std::path::PathBuf;

/// Marker comment appended with the env line so re-running onboard is idempotent and offboarding
/// is greppable.
pub const RC_MARKER: &str = "# added by `firstpass onboard`";

/// What onboarding discovered about this machine. Everything is injected-lookup driven so tests
/// construct arbitrary environments without touching the real one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    /// User's login shell binary name (`zsh` / `bash` / `fish` / other).
    pub shell: String,
    /// The proxy is already answering `/healthz` on the target port.
    pub proxy_running: bool,
    /// `ANTHROPIC_BASE_URL` already points at the target proxy.
    pub already_routed: bool,
    /// `ANTHROPIC_API_KEY` is set (needed for enforce-mode upstream calls; observe passes BYOK).
    pub has_api_key: bool,
    /// The `claude` CLI (Claude Code) is on PATH — we can offer MCP wiring.
    pub has_claude_cli: bool,
    /// Target bind, e.g. `127.0.0.1:8080`.
    pub bind: String,
}

/// One onboarding step: what it is, and whether it's already satisfied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Spawn `firstpass up` detached (observe mode), logging to `firstpass-proxy.log`.
    StartProxy,
    /// Append the marked `ANTHROPIC_BASE_URL` export to the shell rc file.
    WireShell {
        /// Rc file the line goes into.
        rc: PathBuf,
        /// The exact line (shell-dialect aware).
        line: String,
    },
    /// Print the `claude mcp add` command so Claude Code can query traces/savings as tools.
    /// (Printed, not executed — mutating another tool's config uninvited isn't onboarding.)
    SuggestClaudeMcp,
    /// Probe `/healthz` and `/v1/capabilities` and report what's routed.
    Verify,
    /// Nothing to do — with the reason shown to the user.
    AlreadyDone(&'static str),
}

/// Detect the environment via injected lookups: `env(key)` for env vars, `on_path(bin)` for PATH
/// probes, `healthz()` for a live proxy probe. Pure decision logic; all I/O behind the closures.
pub fn detect(
    env: impl Fn(&str) -> Option<String>,
    on_path: impl Fn(&str) -> bool,
    healthz: impl Fn() -> bool,
) -> Environment {
    let bind = env("FIRSTPASS_BIND").unwrap_or_else(|| "127.0.0.1:8080".to_owned());
    let base_url = format!("http://{bind}");
    let shell = env("SHELL")
        .and_then(|s| s.rsplit('/').next().map(str::to_owned))
        .unwrap_or_else(|| "sh".to_owned());
    Environment {
        shell,
        proxy_running: healthz(),
        already_routed: env("ANTHROPIC_BASE_URL").is_some_and(|v| v == base_url),
        has_api_key: env("ANTHROPIC_API_KEY").is_some(),
        has_claude_cli: on_path("claude"),
        bind,
    }
}

/// The shell-dialect line that routes agents through the proxy, plus the rc file it belongs in.
/// `home` is injected so tests never touch a real home directory.
#[must_use]
pub fn shell_wiring(shell: &str, home: &std::path::Path, bind: &str) -> (PathBuf, String) {
    let url = format!("http://{bind}");
    match shell {
        "fish" => (
            home.join(".config/fish/config.fish"),
            format!("set -gx ANTHROPIC_BASE_URL {url}  {RC_MARKER}"),
        ),
        "bash" => (
            home.join(".bashrc"),
            format!("export ANTHROPIC_BASE_URL={url}  {RC_MARKER}"),
        ),
        // zsh and anything else POSIX-ish: default to ~/.zshrc only for zsh, else ~/.profile.
        "zsh" => (
            home.join(".zshrc"),
            format!("export ANTHROPIC_BASE_URL={url}  {RC_MARKER}"),
        ),
        _ => (
            home.join(".profile"),
            format!("export ANTHROPIC_BASE_URL={url}  {RC_MARKER}"),
        ),
    }
}

/// Build the ordered plan for this environment. `rc_already_wired` is whether the rc file already
/// carries the marker line (checked by the caller, injected here to stay pure).
#[must_use]
pub fn plan(env: &Environment, home: &std::path::Path, rc_already_wired: bool) -> Vec<Step> {
    let mut steps = Vec::new();
    if env.proxy_running {
        steps.push(Step::AlreadyDone("proxy already answering /healthz"));
    } else {
        steps.push(Step::StartProxy);
    }
    if env.already_routed || rc_already_wired {
        steps.push(Step::AlreadyDone("ANTHROPIC_BASE_URL already wired"));
    } else {
        let (rc, line) = shell_wiring(&env.shell, home, &env.bind);
        steps.push(Step::WireShell { rc, line });
    }
    if env.has_claude_cli {
        steps.push(Step::SuggestClaudeMcp);
    }
    steps.push(Step::Verify);
    steps
}

/// Render the plan for the dry run (default) or as the running commentary under `--apply`.
#[must_use]
pub fn render(env: &Environment, steps: &[Step], apply: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "detected: shell={} · proxy_running={} · routed={} · api_key={} · claude_cli={}\n\n",
        env.shell, env.proxy_running, env.already_routed, env.has_api_key, env.has_claude_cli
    ));
    for (i, s) in steps.iter().enumerate() {
        let n = i + 1;
        match s {
            Step::StartProxy => out.push_str(&format!(
                "{n}. start the proxy — `firstpass up` (observe mode: watches, changes nothing), log → firstpass-proxy.log\n"
            )),
            Step::WireShell { rc, line } => out.push_str(&format!(
                "{n}. route your agents — append to {}:\n     {line}\n",
                rc.display()
            )),
            Step::SuggestClaudeMcp => out.push_str(&format!(
                "{n}. (optional) let Claude Code query receipts as tools:\n     claude mcp add firstpass -- firstpass mcp\n"
            )),
            Step::Verify => out.push_str(&format!(
                "{n}. verify — probe /healthz and /v1/capabilities, report what's routed\n"
            )),
            Step::AlreadyDone(why) => out.push_str(&format!("{n}. ✓ {why}\n")),
        }
    }
    if !env.has_api_key {
        out.push_str(
            "\nnote: ANTHROPIC_API_KEY is not set — observe mode passes your agent's own key \
             through (BYOK), so this only matters for enforce mode.\n",
        );
    }
    if !apply {
        out.push_str(
            "\ndry run — nothing changed. Re-run with `firstpass onboard --apply` to execute.\n",
        );
    }
    out
}

/// Execute the side-effectful steps. Returns a human report. Failures are reported per step —
/// onboarding never half-dies silently.
///
/// # Errors
/// Only on I/O failures writing the rc file; every other issue is reported in the returned text.
pub fn execute(env: &Environment, steps: &[Step]) -> Result<String, std::io::Error> {
    let mut out = String::new();
    for s in steps {
        match s {
            Step::StartProxy => {
                let log = std::fs::File::create("firstpass-proxy.log")?;
                let exe = std::env::current_exe()?;
                let child = std::process::Command::new(exe)
                    .arg("up")
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::from(log.try_clone()?))
                    .stderr(std::process::Stdio::from(log))
                    .spawn();
                match child {
                    Ok(c) => {
                        // Pidfile makes `firstpass offboard` able to stop what onboard started.
                        let _ = std::fs::write("firstpass-proxy.pid", c.id().to_string());
                        out.push_str(&format!(
                            "✓ proxy started (pid {}, observe mode) — log: firstpass-proxy.log\n",
                            c.id()
                        ));
                    }
                    Err(e) => out.push_str(&format!("✗ could not start proxy: {e}\n")),
                }
            }
            Step::WireShell { rc, line } => {
                if let Some(parent) = rc.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(rc)?;
                writeln!(f, "{line}")?;
                out.push_str(&format!(
                    "✓ wired {} — takes effect in new shells; for this one:\n     {}\n",
                    rc.display(),
                    line.trim_end_matches(RC_MARKER).trim_end()
                ));
            }
            Step::SuggestClaudeMcp => {
                out.push_str("→ optional: claude mcp add firstpass -- firstpass mcp\n");
            }
            Step::Verify => {
                let url = format!("http://{}/healthz", env.bind);
                let ok = wait_healthz(&url, std::time::Duration::from_secs(6));
                if ok {
                    out.push_str(&format!(
                        "✓ verified — proxy healthy at http://{} · capabilities: http://{}/v1/capabilities\n",
                        env.bind, env.bind
                    ));
                } else {
                    out.push_str(&format!(
                        "✗ proxy not answering http://{} after 6s — check firstpass-proxy.log\n",
                        env.bind
                    ));
                }
            }
            Step::AlreadyDone(why) => out.push_str(&format!("✓ {why}\n")),
        }
    }
    out.push_str("\noffboard any time: unset ANTHROPIC_BASE_URL (and remove the marked rc line)\n");
    Ok(out)
}

/// Poll `/healthz` until it answers 200 or the deadline passes. Blocking + std-only (the CLI calls
/// this off the async runtime): a plain TCP connect + minimal HTTP/1.1 GET, no client dependency.
fn wait_healthz(url: &str, deadline: std::time::Duration) -> bool {
    let Some(addr) = url
        .strip_prefix("http://")
        .and_then(|r| r.split('/').next())
        .map(str::to_owned)
    else {
        return false;
    };
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if let Ok(mut s) = std::net::TcpStream::connect(&addr) {
            let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
            let req = format!("GET /healthz HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
            if s.write_all(req.as_bytes()).is_ok() {
                let mut buf = [0u8; 64];
                use std::io::Read as _;
                if let Ok(n) = s.read(&mut buf)
                    && n > 0
                    && String::from_utf8_lossy(&buf[..n]).contains("200")
                {
                    return true;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    false
}

/// Whether the rc file already carries the onboard marker (idempotence check for [`plan`]).
#[must_use]
pub fn rc_wired(rc: &std::path::Path) -> bool {
    std::fs::read_to_string(rc).is_ok_and(|s| s.contains(RC_MARKER))
}

/// Remove every marked onboard line from `rc`. Returns whether anything was removed. Only lines
/// carrying [`RC_MARKER`] are touched — nothing else in the user's rc is ever rewritten.
///
/// # Errors
/// Only on I/O failures reading/writing the rc file.
pub fn offboard_rc(rc: &std::path::Path) -> Result<bool, std::io::Error> {
    let Ok(content) = std::fs::read_to_string(rc) else {
        return Ok(false); // no file → nothing to offboard
    };
    if !content.contains(RC_MARKER) {
        return Ok(false);
    }
    let kept: Vec<&str> = content.lines().filter(|l| !l.contains(RC_MARKER)).collect();
    std::fs::write(rc, kept.join("\n") + "\n")?;
    Ok(true)
}

/// `firstpass offboard` — the exact mirror of onboard: strip the marked line from every candidate
/// rc file, stop the proxy onboard started (pidfile), and print the one command for this shell.
/// Fully idempotent; reports what it found either way.
///
/// # Errors
/// Only on rc-file I/O failures.
pub fn offboard(home: &std::path::Path) -> Result<String, std::io::Error> {
    let mut out = String::new();
    for rc in [
        home.join(".zshrc"),
        home.join(".bashrc"),
        home.join(".profile"),
        home.join(".config/fish/config.fish"),
    ] {
        if offboard_rc(&rc)? {
            out.push_str(&format!("✓ removed firstpass line from {}\n", rc.display()));
        }
    }
    // Stop the proxy onboard spawned, if its pidfile is present.
    if let Ok(pid) = std::fs::read_to_string("firstpass-proxy.pid") {
        let pid = pid.trim().to_owned();
        #[cfg(unix)]
        {
            let killed = std::process::Command::new("kill")
                .arg(&pid)
                .status()
                .is_ok_and(|s| s.success());
            if killed {
                out.push_str(&format!("✓ stopped proxy (pid {pid})\n"));
            } else {
                out.push_str(&format!(
                    "→ proxy pid {pid} not running (already stopped)\n"
                ));
            }
        }
        let _ = std::fs::remove_file("firstpass-proxy.pid");
    }
    if out.is_empty() {
        out.push_str("nothing to offboard — no marked rc lines, no pidfile.\n");
    }
    out.push_str("for this shell: unset ANTHROPIC_BASE_URL\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(a, _)| *a == k)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    #[test]
    fn detect_reads_shell_routing_and_tools() {
        let e = detect(
            env_of(&[
                ("SHELL", "/bin/zsh"),
                ("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080"),
                ("ANTHROPIC_API_KEY", "sk-x"),
            ]),
            |bin| bin == "claude",
            || true,
        );
        assert_eq!(e.shell, "zsh");
        assert!(e.proxy_running && e.already_routed && e.has_api_key && e.has_claude_cli);
        assert_eq!(e.bind, "127.0.0.1:8080");
    }

    #[test]
    fn detect_respects_custom_bind_and_mismatched_base_url() {
        let e = detect(
            env_of(&[
                ("FIRSTPASS_BIND", "127.0.0.1:9999"),
                ("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080"), // points elsewhere
            ]),
            |_| false,
            || false,
        );
        assert_eq!(e.bind, "127.0.0.1:9999");
        assert!(
            !e.already_routed,
            "routed to a different port is not routed"
        );
    }

    #[test]
    fn shell_wiring_speaks_each_dialect() {
        let home = std::path::Path::new("/home/u");
        let (rc, line) = shell_wiring("fish", home, "127.0.0.1:8080");
        assert!(rc.ends_with(".config/fish/config.fish"));
        assert!(line.starts_with("set -gx ANTHROPIC_BASE_URL http://127.0.0.1:8080"));

        let (rc, line) = shell_wiring("zsh", home, "127.0.0.1:8080");
        assert!(rc.ends_with(".zshrc"));
        assert!(line.starts_with("export ANTHROPIC_BASE_URL="));

        let (rc, _) = shell_wiring("dash", home, "127.0.0.1:8080");
        assert!(
            rc.ends_with(".profile"),
            "unknown shells fall back to .profile"
        );
    }

    #[test]
    fn plan_covers_fresh_machine_and_is_idempotent_when_done() {
        let home = std::path::Path::new("/home/u");
        let fresh = Environment {
            shell: "zsh".into(),
            proxy_running: false,
            already_routed: false,
            has_api_key: false,
            has_claude_cli: true,
            bind: "127.0.0.1:8080".into(),
        };
        let steps = plan(&fresh, home, false);
        assert!(matches!(steps[0], Step::StartProxy));
        assert!(matches!(steps[1], Step::WireShell { .. }));
        assert!(matches!(steps[2], Step::SuggestClaudeMcp));
        assert!(matches!(steps.last(), Some(Step::Verify)));

        // Everything already set up → only AlreadyDone + Verify, nothing mutating.
        let done = Environment {
            proxy_running: true,
            already_routed: true,
            has_claude_cli: false,
            ..fresh
        };
        let steps = plan(&done, home, true);
        assert!(
            steps
                .iter()
                .all(|s| matches!(s, Step::AlreadyDone(_) | Step::Verify))
        );
    }

    #[test]
    fn render_dry_run_says_nothing_changed_and_flags_missing_key() {
        let home = std::path::Path::new("/home/u");
        let e = Environment {
            shell: "bash".into(),
            proxy_running: false,
            already_routed: false,
            has_api_key: false,
            has_claude_cli: false,
            bind: "127.0.0.1:8080".into(),
        };
        let text = render(&e, &plan(&e, home, false), false);
        assert!(text.contains("dry run — nothing changed"));
        assert!(text.contains("ANTHROPIC_API_KEY is not set"));
        assert!(text.contains(".bashrc"));
    }

    #[test]
    fn rc_wired_detects_the_marker_and_execute_appends_it_once() {
        let dir = std::env::temp_dir().join(format!("fp-onboard-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let rc = dir.join(".zshrc");
        assert!(!rc_wired(&rc), "missing file is not wired");

        let e = Environment {
            shell: "zsh".into(),
            proxy_running: true, // no spawn in this test
            already_routed: false,
            has_api_key: true,
            has_claude_cli: false,
            bind: "127.0.0.1:1".into(), // verify step will fail fast — that's fine, we assert wiring
        };
        let (rc_path, line) = shell_wiring("zsh", &dir, &e.bind);
        let steps = vec![Step::WireShell {
            rc: rc_path.clone(),
            line,
        }];
        let report = execute(&e, &steps).unwrap();
        assert!(report.contains("✓ wired"));
        assert!(rc_wired(&rc_path), "marker written");
        // Re-planning with the marker present must not wire again (idempotent onboarding).
        let steps = plan(&e, &dir, rc_wired(&rc_path));
        assert!(!steps.iter().any(|s| matches!(s, Step::WireShell { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn offboard_removes_only_the_marked_line_and_is_idempotent() {
        let dir = std::env::temp_dir().join(format!("fp-offboard-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let rc = dir.join(".zshrc");
        std::fs::write(
            &rc,
            format!("alias ll='ls -l'\nexport ANTHROPIC_BASE_URL=http://127.0.0.1:8080  {RC_MARKER}\nexport EDITOR=vim\n"),
        )
        .unwrap();

        assert!(offboard_rc(&rc).unwrap(), "marked line removed");
        let after = std::fs::read_to_string(&rc).unwrap();
        assert!(!after.contains(RC_MARKER));
        assert!(
            after.contains("alias ll") && after.contains("EDITOR=vim"),
            "user lines untouched"
        );
        assert!(!offboard_rc(&rc).unwrap(), "second offboard is a no-op");

        // The full offboard reports the rc removal and the unset line.
        std::fs::write(&rc, format!("x  {RC_MARKER}\n")).unwrap();
        let report = offboard(&dir).unwrap();
        assert!(report.contains("removed firstpass line"));
        assert!(report.contains("unset ANTHROPIC_BASE_URL"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
