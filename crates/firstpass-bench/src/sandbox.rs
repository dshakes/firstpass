//! Fail-closed code-execution sandbox for the coding-with-tests benchmark (ADR 0002).
//!
//! The benchmark runs **model-generated candidate code** to measure real gate error — that code is
//! untrusted (it can `rm`, exfiltrate, fork-bomb, read secrets). This module runs it under the
//! strongest OCI runtime available (gVisor `runsc` if present, else `runc` with a warning; a
//! Firecracker microVM impl slots behind [`Sandbox`] as the ideal tier), with **no network, no host
//! filesystem, resource + wall-clock caps, non-root, and cap-drop**.
//!
//! The cardinal rule (ADR 0002 §D3) is **fail closed**: [`establish_sandbox`] returns `Err` unless a
//! real sandbox is present *and proven isolating* by [`verify_isolating`]. Callers MUST abort on
//! `Err` — candidate code is never run on the host. Nothing here executes a candidate until that
//! proof passes.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// One unit of untrusted work: files to materialize in the workdir, plus the command to run there.
#[derive(Debug, Clone)]
pub struct ExecUnit {
    /// `(relative_path, contents)` written under `/work` inside the sandbox.
    pub files: Vec<(String, String)>,
    /// Shell command run in `/work` (e.g. `python -m pytest -q`).
    pub command: String,
}

impl ExecUnit {
    /// A unit that just runs `command` with no files.
    #[must_use]
    pub fn cmd(command: impl Into<String>) -> Self {
        Self {
            files: Vec::new(),
            command: command.into(),
        }
    }
}

/// Resource + time ceilings for one execution.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// CPU cores (fractional allowed).
    pub cpus: f32,
    /// Memory cap in MiB (swap is disabled by pinning swap == memory).
    pub mem_mb: u64,
    /// Max process count (kills fork bombs).
    pub pids: u32,
    /// Wall-clock timeout in milliseconds.
    pub wall_ms: u64,
    /// Workdir (tmpfs) size cap in MiB.
    pub workdir_mb: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            cpus: 1.0,
            mem_mb: 512,
            pids: 128,
            wall_ms: 10_000,
            workdir_mb: 64,
        }
    }
}

/// What a run produced. `TimedOut`/`Killed` are gate **fails**, never dropped samples (ADR 0002 §D5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecOutcome {
    /// The command ran to completion with this exit code and captured output.
    Completed {
        /// Process exit code (`0` = success; non-zero = e.g. failing tests).
        exit_code: i32,
        /// Captured stdout.
        stdout: String,
        /// Captured stderr.
        stderr: String,
    },
    /// Killed by the wall-clock timeout (a hung candidate).
    TimedOut,
    /// Killed by the runtime (OOM, pids cap, or forced removal), with a short reason.
    Killed(String),
}

impl ExecOutcome {
    /// Did the command exit cleanly with code 0? (The coding gate's pass condition.)
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, ExecOutcome::Completed { exit_code: 0, .. })
    }
}

/// Why a sandbox could not be used. **`Unavailable` and `Setup` both mean the caller ABORTS** — it
/// must never fall back to running candidate code on the host (ADR 0002 §D3).
#[derive(Debug)]
pub enum SandboxError {
    /// No container engine/runtime present. Abort — do not host-exec.
    Unavailable(String),
    /// A sandbox exists but could not be set up or failed its isolation proof. Abort.
    Setup(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::Unavailable(m) => write!(f, "sandbox unavailable: {m}"),
            SandboxError::Setup(m) => write!(f, "sandbox setup failed: {m}"),
        }
    }
}
impl std::error::Error for SandboxError {}

/// A backend that runs an [`ExecUnit`] in isolation. The real impl is [`ContainerSandbox`]; tests use
/// an in-process fake. A future Firecracker microVM impl slots in here unchanged (ADR 0002 §D1).
pub trait Sandbox {
    /// Isolation tier for the report, e.g. `"gvisor"` or `"runc"`.
    fn runtime(&self) -> &str;
    /// Run one unit. Returns `Err` only when the sandbox itself failed (not when the *candidate*
    /// exits non-zero — that is `Ok(Completed { exit_code != 0 })`).
    fn run(&self, unit: &ExecUnit, limits: &Limits) -> Result<ExecOutcome, SandboxError>;
}

// ---- isolation proof (ADR 0002 §D3) -----------------------------------------------------------

// Distinct probe programs. A probe that REACHES the network or host FS means the sandbox is broken.
// `verify_isolating` asserts each escape probe fails and the workdir positive-control succeeds.
const WORK_PROBE: &str = "python3 -c \"open('/work/fp_ok','w').write('ok')\"";
const NET_PROBE: &str = "python3 -c \"import socket; socket.create_connection(('1.1.1.1',53),2)\""; // exit 0 == BREACH
const FS_PROBE: &str = "python3 -c \"open('/fp_escape','w').write('x')\""; // exit 0 == BREACH

/// Prove a sandbox actually isolates before any candidate runs: the workdir is writable (so we can
/// run at all), the network is unreachable, and the host filesystem is read-only. Any breach →
/// `Err` → the caller aborts. This is the safety gate; it runs against the real sandbox at startup.
///
/// # Errors
/// [`SandboxError::Setup`] if the workdir is unusable or either escape probe succeeds (a breach).
pub fn verify_isolating<S: Sandbox + ?Sized>(sb: &S) -> Result<(), SandboxError> {
    let limits = Limits {
        wall_ms: 8_000,
        ..Limits::default()
    };

    // Positive control: we can write to the workdir. If not, the sandbox is unusable.
    match sb.run(&ExecUnit::cmd(WORK_PROBE), &limits)? {
        o if o.is_success() => {}
        other => {
            return Err(SandboxError::Setup(format!(
                "workdir not writable, sandbox unusable: {other:?}"
            )));
        }
    }
    // Escape probes: reaching the network or host FS is a breach. Anything but a clean exit 0 is the
    // isolation working (blocked, timed out, or killed all mean "did not escape").
    if sb.run(&ExecUnit::cmd(NET_PROBE), &limits)?.is_success() {
        return Err(SandboxError::Setup(
            "ISOLATION BREACH: network reachable from sandbox (expected --network none)".to_owned(),
        ));
    }
    if sb.run(&ExecUnit::cmd(FS_PROBE), &limits)?.is_success() {
        return Err(SandboxError::Setup(
            "ISOLATION BREACH: host filesystem writable from sandbox (expected read-only)"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Establish a sandbox for the benchmark, **fail-closed**: detect the strongest runtime, then PROVE
/// it isolates. Returns `Err` if no sandbox is available or it fails its isolation proof — the
/// caller must abort rather than run candidate code on the host (ADR 0002 §D3).
///
/// # Errors
/// [`SandboxError`] when no engine/runtime is present or the isolation proof fails.
pub fn establish_sandbox(image: &str) -> Result<Box<dyn Sandbox>, SandboxError> {
    let sb = ContainerSandbox::detect(image)?;
    verify_isolating(&sb)?;
    Ok(Box::new(sb))
}

// ---- real container sandbox -------------------------------------------------------------------

/// Runs each unit in a fresh, throwaway container: `--network none`, read-only rootfs, a `tmpfs`
/// workdir, cpu/mem/pids caps, non-root, `cap-drop ALL`, `--rm`. Files are streamed in over stdin
/// (base64) so **no host path is ever mounted**. The strongest available runtime is chosen.
#[derive(Debug, Clone)]
pub struct ContainerSandbox {
    engine: String,  // "docker" | "podman"
    runtime: String, // "runsc" (gVisor, ideal) | "runc" (warned fallback)
    image: String,
}

/// Monotonic suffix so concurrent/repeated runs get unique container names without a uuid dep.
static SEQ: AtomicU64 = AtomicU64::new(0);

impl ContainerSandbox {
    /// Detect a container engine and the strongest runtime present. `Unavailable` if none — abort.
    ///
    /// # Errors
    /// [`SandboxError::Unavailable`] if neither `docker` nor `podman` has a working daemon.
    pub fn detect(image: &str) -> Result<Self, SandboxError> {
        let engine = ["docker", "podman"]
            .into_iter()
            .find(|e| capture(e, &["info"]).is_some())
            .ok_or_else(|| {
                SandboxError::Unavailable("no working docker or podman daemon".to_owned())
            })?;

        // Ideal isolation first: use gVisor's `runsc` if the engine has it registered; otherwise
        // fall back to the default runtime with a loud warning (ADR 0002 §D1 — never silent).
        let runtime = if capture(engine, &["info", "-f", "{{json .Runtimes}}"])
            .is_some_and(|s| s.contains("runsc"))
        {
            "runsc".to_owned()
        } else {
            eprintln!(
                "WARNING: gVisor (runsc) not found; falling back to the default OCI runtime \
                 (shared-kernel, weaker isolation). Install gVisor for microVM-grade isolation."
            );
            "runc".to_owned()
        };

        Ok(Self {
            engine: engine.to_owned(),
            runtime,
            image: image.to_owned(),
        })
    }

    /// The shell script (run via `sh -s` on stdin) that materializes files without a host mount and
    /// runs the command under an inner `timeout` (a race-free primary; the host watchdog backstops).
    fn build_script(unit: &ExecUnit, wall_s: u64) -> String {
        let mut s = String::from("set -e\n");
        for (path, contents) in &unit.files {
            let b64 = b64encode(contents.as_bytes());
            // Shell-quote the path so a task path containing a quote can't break the setup script's
            // quoting (blast radius is container-confined, but keep it robust — matches `command`).
            let dst = shell_quote(&format!("/work/{path}"));
            s.push_str(&format!(
                "mkdir -p \"$(dirname {dst})\"\nprintf %s '{b64}' | base64 -d > {dst}\n"
            ));
        }
        s.push_str("cd /work\n");
        // `timeout` exits 124 on expiry; -k sends KILL after a grace period.
        s.push_str(&format!(
            "exec timeout -k 2 {wall_s} sh -c {}\n",
            shell_quote(&unit.command)
        ));
        s
    }
}

impl Sandbox for ContainerSandbox {
    fn runtime(&self) -> &str {
        &self.runtime
    }

    fn run(&self, unit: &ExecUnit, limits: &Limits) -> Result<ExecOutcome, SandboxError> {
        let name = format!(
            "fp-sbx-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let wall_s = limits.wall_ms.div_ceil(1000).max(1);
        let script = Self::build_script(unit, wall_s);

        let mut cmd = Command::new(&self.engine);
        cmd.args(["run", "--rm", "-i", "--name", &name])
            .args(["--runtime", &self.runtime])
            .args(["--network", "none"])
            .arg("--read-only")
            .args([
                "--tmpfs",
                &format!("/work:size={}m,mode=1777", limits.workdir_mb),
            ])
            .args(["--workdir", "/work"])
            .args(["--memory", &format!("{}m", limits.mem_mb)])
            .args(["--memory-swap", &format!("{}m", limits.mem_mb)]) // == memory ⇒ no swap
            .args(["--cpus", &format!("{}", limits.cpus)])
            .args(["--pids-limit", &format!("{}", limits.pids)])
            .args(["--cap-drop", "ALL"])
            .args(["--security-opt", "no-new-privileges"])
            .args(["--user", "65534:65534"]) // nobody:nogroup
            .arg(&self.image)
            .args(["sh", "-s"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| SandboxError::Setup(format!("spawn {} failed: {e}", self.engine)))?;
        // Kill the container if we bail before wait completes (best-effort hygiene / kill_on_drop).
        let guard = KillGuard::new(&self.engine, &name);

        child
            .stdin
            .take()
            .ok_or_else(|| SandboxError::Setup("child stdin not piped".to_owned()))?
            .write_all(script.as_bytes())
            .map_err(|e| SandboxError::Setup(format!("write script: {e}")))?;

        // Host-side watchdog: force-kill the container if it outlives wall_ms + grace, in case the
        // inner `timeout` or the runtime itself wedges. Race-free vs normal exit via `done`.
        let done = Arc::new(AtomicBool::new(false));
        let fired = Arc::new(AtomicBool::new(false));
        let watchdog = {
            let (engine, name) = (self.engine.clone(), name.clone());
            let (done, fired) = (Arc::clone(&done), Arc::clone(&fired));
            let deadline = Duration::from_millis(limits.wall_ms) + Duration::from_secs(3);
            thread::spawn(move || {
                let start = Instant::now();
                while start.elapsed() < deadline {
                    if done.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                if !done.load(Ordering::Relaxed) {
                    fired.store(true, Ordering::Relaxed);
                    let _ = Command::new(&engine)
                        .args(["kill", "--signal", "KILL", &name])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                }
            })
        };

        let output = child
            .wait_with_output()
            .map_err(|e| SandboxError::Setup(format!("wait failed: {e}")))?;
        done.store(true, Ordering::Relaxed);
        let _ = watchdog.join();
        guard.disarm();

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let code = output.status.code();

        // Timed out: the host watchdog fired, or the inner `timeout` reported expiry (124, or
        // 128+SIGTERM/SIGKILL = 143/137 when -k escalated).
        if fired.load(Ordering::Relaxed) || matches!(code, Some(124 | 137 | 143)) {
            return Ok(ExecOutcome::TimedOut);
        }
        match code {
            Some(c) => Ok(ExecOutcome::Completed {
                exit_code: c,
                stdout,
                stderr,
            }),
            // No exit code ⇒ terminated by signal we didn't classify (e.g. OOM kill).
            None => Ok(ExecOutcome::Killed(format!(
                "container terminated by signal; stderr: {}",
                stderr.trim()
            ))),
        }
    }
}

/// Force-removes a container by name on drop unless disarmed — so an early return never leaves a
/// candidate running.
struct KillGuard {
    engine: String,
    name: String,
    armed: bool,
}
impl KillGuard {
    fn new(engine: &str, name: &str) -> Self {
        Self {
            engine: engine.to_owned(),
            name: name.to_owned(),
            armed: true,
        }
    }
    fn disarm(mut self) {
        self.armed = false;
    }
}
impl Drop for KillGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = Command::new(&self.engine)
                .args(["rm", "-f", &self.name])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

/// Run a command and capture stdout iff it exits 0; `None` on failure/non-zero (used for probing).
fn capture(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Single-quote a string for POSIX `sh` (wrap in `'…'`, escaping embedded quotes as `'\''`).
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Minimal standard base64 encoder (host side; the container decodes with `base64 -d`). Avoids a
/// dependency for the one thing we need — streaming candidate files in without a host mount.
fn b64encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An in-process fake that answers each probe from a closure — lets us test the isolation proof
    /// and outcome plumbing deterministically, with no Docker.
    struct FakeSandbox {
        answer: Box<dyn Fn(&ExecUnit) -> ExecOutcome + Send + Sync>,
    }
    impl FakeSandbox {
        fn new(f: impl Fn(&ExecUnit) -> ExecOutcome + Send + Sync + 'static) -> Self {
            Self {
                answer: Box::new(f),
            }
        }
    }
    impl Sandbox for FakeSandbox {
        fn runtime(&self) -> &str {
            "fake"
        }
        fn run(&self, unit: &ExecUnit, _l: &Limits) -> Result<ExecOutcome, SandboxError> {
            Ok((self.answer)(unit))
        }
    }

    fn ok0() -> ExecOutcome {
        ExecOutcome::Completed {
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
        }
    }
    fn blocked() -> ExecOutcome {
        ExecOutcome::Completed {
            exit_code: 1,
            stdout: String::new(),
            stderr: "denied".into(),
        }
    }

    /// A well-behaved sandbox: workdir writable, net + host-fs blocked → isolation proof passes.
    #[test]
    fn verify_isolating_passes_when_probes_blocked() {
        let sb = FakeSandbox::new(|u| {
            if u.command == WORK_PROBE {
                ok0()
            } else {
                blocked() // net + fs probes are denied
            }
        });
        assert!(verify_isolating(&sb).is_ok());
    }

    /// The net probe reaching the network (exit 0) is a breach → proof must FAIL closed.
    #[test]
    fn verify_isolating_fails_on_network_breach() {
        let sb = FakeSandbox::new(|u| {
            if u.command == FS_PROBE {
                blocked()
            } else {
                ok0()
            } // work ok, NET reachable
        });
        let err = verify_isolating(&sb).unwrap_err();
        assert!(format!("{err}").contains("network reachable"), "{err}");
    }

    /// Writing to the host FS (exit 0) is a breach → proof must FAIL closed.
    #[test]
    fn verify_isolating_fails_on_host_fs_breach() {
        let sb = FakeSandbox::new(|u| {
            if u.command == NET_PROBE {
                blocked()
            } else {
                ok0()
            } // work ok, net blocked, FS writable
        });
        let err = verify_isolating(&sb).unwrap_err();
        assert!(
            format!("{err}").contains("host filesystem writable"),
            "{err}"
        );
    }

    /// If the workdir positive-control fails, the sandbox is unusable → proof fails (don't run code).
    #[test]
    fn verify_isolating_fails_when_workdir_unusable() {
        let sb = FakeSandbox::new(|_| blocked()); // even the workdir write "fails"
        let err = verify_isolating(&sb).unwrap_err();
        assert!(format!("{err}").contains("unusable"), "{err}");
    }

    /// Timeout/kill on an escape probe still counts as "did not escape" (proof passes).
    #[test]
    fn timeout_on_escape_probe_is_not_a_breach() {
        let sb = FakeSandbox::new(|u| {
            if u.command == WORK_PROBE {
                ok0()
            } else {
                ExecOutcome::TimedOut
            }
        });
        assert!(verify_isolating(&sb).is_ok());
    }

    #[test]
    fn base64_matches_reference() {
        assert_eq!(b64encode(b""), "");
        assert_eq!(b64encode(b"f"), "Zg==");
        assert_eq!(b64encode(b"fo"), "Zm8=");
        assert_eq!(b64encode(b"foo"), "Zm9v");
        assert_eq!(b64encode(b"foob"), "Zm9vYg==");
        assert_eq!(b64encode(b"hello world"), "aGVsbG8gd29ybGQ=");
        // round-trips through the shell decoder we rely on
        assert_eq!(
            b64encode("print('héllo')\n".as_bytes()),
            "cHJpbnQoJ2jDqWxsbycpCg=="
        );
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn build_script_writes_files_and_wraps_in_timeout() {
        let unit = ExecUnit {
            files: vec![("m.py".into(), "print(1)".into())],
            command: "python3 m.py".into(),
        };
        let s = ContainerSandbox::build_script(&unit, 5);
        assert!(s.contains("base64 -d > '/work/m.py'"));
        assert!(s.contains("timeout -k 2 5 sh -c 'python3 m.py'"));
    }

    #[test]
    fn is_success_only_for_clean_exit() {
        assert!(ok0().is_success());
        assert!(!blocked().is_success());
        assert!(!ExecOutcome::TimedOut.is_success());
        assert!(!ExecOutcome::Killed("oom".into()).is_success());
    }

    // ---- real Docker (opt-in; needs a running daemon) ----------------------------------------
    // Ignored by default so CI without Docker stays green. Run locally with:
    //   cargo test -p firstpass-bench --lib sandbox::tests::real_ -- --ignored --nocapture

    #[test]
    #[ignore = "requires a running container daemon"]
    fn real_sandbox_establishes_and_isolates() {
        // Fail-closed proof against a REAL runtime: establish must succeed AND prove isolation.
        let sb = establish_sandbox("python:3.12-alpine").expect("establish + isolation proof");
        eprintln!("runtime tier: {}", sb.runtime());
    }

    #[test]
    #[ignore = "requires a running container daemon"]
    fn real_sandbox_runs_candidate_and_blocks_network() {
        let sb = ContainerSandbox::detect("python:3.12-alpine").expect("detect");
        // A benign candidate: write a file, run it, exit 0.
        let unit = ExecUnit {
            files: vec![("solution.py".into(), "print('ok')\n".into())],
            command: "python3 solution.py".into(),
        };
        let out = sb.run(&unit, &Limits::default()).expect("run");
        assert!(out.is_success(), "benign candidate should pass: {out:?}");
        // Network is blocked (exit 0 here would be a breach).
        let net = sb
            .run(&ExecUnit::cmd(NET_PROBE), &Limits::default())
            .expect("run");
        assert!(!net.is_success(), "network must be unreachable: {net:?}");
    }
}
