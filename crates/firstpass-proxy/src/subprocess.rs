//! Subprocess gate plugin contract (SPEC §8.1) — the language-agnostic moat mechanism.
//!
//! A gate can be *any* executable: a 10-line Python script, a compiled binary, `bash -c '…'`.
//! The proxy speaks to it over a tiny, stable contract:
//! - **stdin** (JSON): `{ "gate_id", "candidate", "request": { model, system, messages } }`.
//!   The model output is passed as **data on stdin, never as a command-line argument**, so a
//!   malicious candidate can't be interpreted as flags or shell (§8.3.2 injection resistance).
//! - **stdout** (JSON): `{ "verdict": "pass|fail|abstain", "score"?: 0.0-1.0, "reason"?, "evidence"? }`.
//! - **exit ≠ 0** → gate error → `abstain` (reason `gate_crash`, stderr captured).
//! - **timeout** → the child is killed → `abstain` (reason `timeout`).
//!
//! A gate never gets the API keys or anything beyond the candidate + request metadata it needs.

use crate::gate::Gate;
use crate::provider::{ModelRequest, ModelResponse};
use async_trait::async_trait;
use firstpass_core::verdict::reason;
use firstpass_core::{GateResult, Score, Verdict};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// A gate backed by an external process.
#[derive(Debug, Clone)]
pub struct SubprocessGate {
    id: String,
    program: String,
    args: Vec<String>,
    timeout: Duration,
}

impl SubprocessGate {
    /// Build a subprocess gate. `program`/`args` are operator-configured (trusted); the
    /// untrusted candidate only ever travels on stdin.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        program: impl Into<String>,
        args: Vec<String>,
        timeout: Duration,
    ) -> Self {
        Self {
            id: id.into(),
            program: program.into(),
            args,
            timeout,
        }
    }
}

/// What the proxy writes to the gate's stdin.
#[derive(Serialize)]
struct GateInput<'a> {
    gate_id: &'a str,
    candidate: &'a str,
    request: GateRequestView<'a>,
}

/// The request metadata a gate may need — never keys, never more than this.
#[derive(Serialize)]
struct GateRequestView<'a> {
    model: &'a str,
    system: Option<&'a str>,
    messages: &'a [crate::provider::ChatMessage],
}

/// What the proxy reads from the gate's stdout.
#[derive(Deserialize)]
struct GateOutput {
    verdict: String,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    evidence: Option<String>,
}

#[async_trait]
impl Gate for SubprocessGate {
    fn id(&self) -> &str {
        &self.id
    }

    async fn evaluate(&self, req: &ModelRequest, resp: &ModelResponse) -> GateResult {
        let start = Instant::now();
        let input = GateInput {
            gate_id: &self.id,
            candidate: &resp.text,
            request: GateRequestView {
                model: &req.model,
                system: req.system.as_deref(),
                messages: &req.messages,
            },
        };
        let payload = match serde_json::to_vec(&input) {
            Ok(p) => p,
            Err(e) => {
                return self.abstain(reason::GATE_CRASH, &format!("serialize input: {e}"), start);
            }
        };

        match self.run(&payload).await {
            Ok(out) => self.parse_output(&out, start),
            Err(GateRunError::Timeout) => self.abstain(reason::TIMEOUT, "gate timed out", start),
            Err(GateRunError::Spawn(e)) => {
                self.abstain(reason::GATE_CRASH, &format!("spawn: {e}"), start)
            }
            Err(GateRunError::NonZero { code, stderr }) => self.abstain(
                reason::GATE_CRASH,
                &format!("exit {code}: {}", truncate(&stderr, 200)),
                start,
            ),
        }
    }
}

/// Internal run errors, mapped to abstain reasons by the caller.
enum GateRunError {
    Timeout,
    Spawn(std::io::Error),
    NonZero { code: i32, stderr: String },
}

impl SubprocessGate {
    /// Spawn the child, write `payload` to stdin, enforce the timeout, and return stdout bytes.
    async fn run(&self, payload: &[u8]) -> Result<Vec<u8>, GateRunError> {
        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(GateRunError::Spawn)?;

        if let Some(mut stdin) = child.stdin.take() {
            // Ignore a write error here: the child may have exited early; wait_with_output below
            // surfaces the real failure (non-zero exit / empty stdout).
            let _ = stdin.write_all(payload).await;
            let _ = stdin.shutdown().await;
        }

        let output = match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Err(GateRunError::Spawn(e)),
            Err(_elapsed) => return Err(GateRunError::Timeout), // kill_on_drop reaps the child
        };

        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(GateRunError::NonZero {
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })
        }
    }

    /// Parse the gate's stdout JSON into a [`GateResult`], defaulting sanely on malformed output.
    fn parse_output(&self, stdout: &[u8], start: Instant) -> GateResult {
        let ms = elapsed_ms(start);
        let out: GateOutput = match serde_json::from_slice(stdout) {
            Ok(o) => o,
            Err(e) => {
                return self.abstain(reason::GATE_CRASH, &format!("bad stdout json: {e}"), start);
            }
        };
        let verdict = match out.verdict.as_str() {
            "pass" => Verdict::Pass,
            "fail" => Verdict::Fail,
            "abstain" => Verdict::Abstain,
            other => {
                return self.abstain(
                    reason::GATE_CRASH,
                    &format!("unknown verdict {other:?}"),
                    start,
                );
            }
        };
        GateResult {
            gate_id: self.id.clone(),
            verdict,
            score: out.score.and_then(|s| Score::new(s).ok()),
            cost_usd: 0.0,
            ms,
            reason: out.reason,
            evidence_ref: out.evidence,
        }
    }

    fn abstain(&self, reason: &str, detail: &str, start: Instant) -> GateResult {
        tracing::warn!(gate = %self.id, %reason, %detail, "subprocess gate abstained");
        let mut r = GateResult::abstain(&self.id, reason, elapsed_ms(start));
        r.evidence_ref = Some(detail.to_owned());
        r
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_owned()
    } else {
        format!("{}…", &s[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn req() -> ModelRequest {
        ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: None,
            messages: vec![],
            max_tokens: 16,
            tools: Value::Null,
        }
    }
    fn resp(text: &str) -> ModelResponse {
        ModelResponse {
            model: "m".to_owned(),
            text: text.to_owned(),
            in_tokens: 1,
            out_tokens: 1,
            raw: Value::Null,
        }
    }

    /// A gate that always passes (ignores input).
    fn echo_pass() -> SubprocessGate {
        SubprocessGate::new(
            "always-pass",
            "sh",
            vec![
                "-c".into(),
                r#"echo '{"verdict":"pass","score":1.0}'"#.into(),
            ],
            Duration::from_secs(5),
        )
    }

    #[tokio::test]
    async fn passing_subprocess_gate() {
        let r = echo_pass().evaluate(&req(), &resp("x")).await;
        assert_eq!(r.verdict, Verdict::Pass);
        assert_eq!(r.score.map(firstpass_core::Score::value), Some(1.0));
    }

    #[tokio::test]
    async fn gate_reads_candidate_from_stdin_as_data() {
        // The gate echoes back a fail iff it can read the candidate from stdin JSON — proving the
        // candidate travels as stdin data (jq extracts .candidate), not as an argument.
        let g = SubprocessGate::new(
            "reads-stdin",
            "sh",
            vec![
                "-c".into(),
                // If .candidate == "SECRET", fail; else pass. Reads stdin, never argv.
                r#"c=$(cat); case "$c" in *SECRET*) echo '{"verdict":"fail"}';; *) echo '{"verdict":"pass"}';; esac"#.into(),
            ],
            Duration::from_secs(5),
        );
        assert_eq!(
            g.evaluate(&req(), &resp("SECRET")).await.verdict,
            Verdict::Fail
        );
        assert_eq!(
            g.evaluate(&req(), &resp("benign")).await.verdict,
            Verdict::Pass
        );
    }

    #[tokio::test]
    async fn nonzero_exit_becomes_abstain() {
        let g = SubprocessGate::new(
            "crasher",
            "sh",
            vec!["-c".into(), "echo oops >&2; exit 3".into()],
            Duration::from_secs(5),
        );
        let r = g.evaluate(&req(), &resp("x")).await;
        assert_eq!(r.verdict, Verdict::Abstain);
        assert_eq!(r.reason.as_deref(), Some(reason::GATE_CRASH));
    }

    #[tokio::test]
    async fn timeout_becomes_abstain() {
        let g = SubprocessGate::new(
            "slow",
            "sh",
            vec!["-c".into(), "sleep 10".into()],
            Duration::from_millis(150),
        );
        let r = g.evaluate(&req(), &resp("x")).await;
        assert_eq!(r.verdict, Verdict::Abstain);
        assert_eq!(r.reason.as_deref(), Some(reason::TIMEOUT));
    }

    #[tokio::test]
    async fn malformed_stdout_becomes_abstain() {
        let g = SubprocessGate::new(
            "garbage",
            "sh",
            vec!["-c".into(), "echo not-json".into()],
            Duration::from_secs(5),
        );
        assert_eq!(
            g.evaluate(&req(), &resp("x")).await.verdict,
            Verdict::Abstain
        );
    }

    #[tokio::test]
    async fn missing_program_becomes_abstain() {
        let g = SubprocessGate::new(
            "nope",
            "this-binary-does-not-exist-firstpass",
            vec![],
            Duration::from_secs(5),
        );
        assert_eq!(
            g.evaluate(&req(), &resp("x")).await.verdict,
            Verdict::Abstain
        );
    }
}
