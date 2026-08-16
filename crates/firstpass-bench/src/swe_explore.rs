//! Repository exploration for the SWE-bench agent loop: let the model **read the code**.
//!
//! # Why this exists
//!
//! The first agentic SWE-bench run resolved **1 issue in 352 model calls**
//! (`docs/benchmarks/swebench-22-agentic.txt`). The cause was not the models and not the router: the
//! solver received a prose problem statement and had to emit a unified diff for a file it had never
//! seen. It was guessing at line numbers and context lines in a repository it could not open.
//!
//! A real SWE-bench agent explores first — lists the tree, greps for the symbol, reads the function
//! it is about to change — and only then writes a patch. This module provides exactly that, and
//! nothing more.
//!
//! # The security position, which is the hard part
//!
//! [`crate::swebench::evaluate`] runs a **fail-closed** container: `--read-only`, `--network none`,
//! `--cap-drop ALL`, `no-new-privileges`, tmpfs work dir, pids/memory/cpu ceilings (ADR 0002/0010).
//! Giving a model "file access" must not weaken any of that, and the obvious implementation —
//! letting the model name a shell command — throws all of it away, because the model chooses what
//! runs.
//!
//! So the model does **not** get a shell. It gets three verbs with typed arguments
//! ([`ExploreCmd`]), each executed by *our* code inside the same fail-closed container:
//!
//! - `ls <dir>` — list a directory
//! - `grep <pattern> <dir>` — search file contents
//! - `read <file> [start] [end]` — read a line range
//!
//! Everything is **read-only by construction**: no verb writes, and the container is still
//! `--read-only` with no network. The model cannot install a package, reach the internet, or mutate
//! the repo. Path traversal is rejected before a command is built (see [`ExploreCmd::to_shell`]),
//! so `read ../../etc/passwd` never becomes a container argument.
//!
//! # Why this is not "just running the agent's tool calls"
//!
//! The distinction is the whole safety argument: a tool-calling agent decides *what command runs*,
//! and the sandbox is then the only thing standing between it and the host. Here the model decides
//! *which of three read-only questions to ask*, and the command is assembled by this module. The
//! blast radius of a compromised or adversarial model is bounded by the verb set, not by the
//! sandbox alone — defence in depth, with the sandbox still underneath.

use std::process::{Command, Stdio};

use crate::swebench::{SweInstance, SweLimits};

/// One read-only question the model may ask about the repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExploreCmd {
    /// List a directory, relative to the repo root.
    Ls { dir: String },
    /// Search file contents under a directory.
    Grep { pattern: String, dir: String },
    /// Read a line range of a file (1-based, inclusive).
    Read {
        file: String,
        start: usize,
        end: usize,
    },
}

/// Cap on returned output. A `grep` across a large repo can produce megabytes, which would blow the
/// next turn's context window and cost more than the exploration is worth.
const MAX_OUTPUT_BYTES: usize = 8_000;

/// Widest line range a single `read` may request, so one call cannot pull a whole file.
const MAX_READ_LINES: usize = 400;

/// Reject anything that could escape the repository root.
///
/// Checked **before** a command is constructed, so a hostile path never reaches the container at
/// all. Absolute paths and `..` are refused outright rather than normalised: normalising invites
/// the classic bugs (`....//`, encoded separators), and no legitimate exploration of a repo needs
/// either form.
fn safe_path(p: &str) -> Result<(), String> {
    if p.starts_with('/') {
        return Err(format!("absolute paths are not allowed: {p}"));
    }
    if p.split('/').any(|seg| seg == "..") {
        return Err(format!("path traversal is not allowed: {p}"));
    }
    if p.contains('\0') {
        return Err("path contains a NUL byte".to_owned());
    }
    Ok(())
}

/// Single-quote a value for `sh -c`, so no argument can break out into shell syntax.
///
/// Quoting alone is **not sufficient**, which is why every command template also carries a `--`
/// terminator (and `grep` an explicit `-e`). A path like `--version` or `-rf` is perfectly safe as
/// *shell syntax* and still gets parsed as an *option* by the tool receiving it — a second
/// injection surface one layer down from the shell. Flagged in review.
///
/// The model supplies grep patterns, which are the one place arbitrary text reaches a command line.
/// POSIX single quotes suppress every metacharacter; the only escape needed is for the quote itself.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

impl ExploreCmd {
    /// Render to a shell command run inside the container.
    ///
    /// # Errors
    /// Any unsafe path, or a read range that is inverted or too wide.
    pub fn to_shell(&self) -> Result<String, String> {
        match self {
            Self::Ls { dir } => {
                safe_path(dir)?;
                // `-p` marks directories with a trailing slash so the model can navigate without a
                // second call to find out what is a directory.
                Ok(format!("ls -p -- {} 2>&1 | head -200", shell_quote(dir)))
            }
            Self::Grep { pattern, dir } => {
                safe_path(dir)?;
                if pattern.is_empty() {
                    return Err("empty grep pattern".to_owned());
                }
                // `safe_path` rejects NUL in paths; the pattern needs the same check. A NUL cannot
                // survive an exec argument, so it truncates silently and the model searches for
                // something other than what it asked. Flagged in review.
                if pattern.contains('\0') {
                    return Err("grep pattern contains a NUL byte".to_owned());
                }
                // `-F` (fixed strings): the model is looking for symbol names, and a stray `(` in a
                // regex would either error or match unexpectedly. `-n` gives line numbers, which is
                // what makes the follow-up `read` precise.
                Ok(format!(
                    "grep -rnF -e {} -- {} 2>&1 | head -100",
                    shell_quote(pattern),
                    shell_quote(dir)
                ))
            }
            Self::Read { file, start, end } => {
                safe_path(file)?;
                if end < start {
                    return Err(format!("inverted range {start}..{end}"));
                }
                if end - start >= MAX_READ_LINES {
                    return Err(format!(
                        "range {start}..{end} exceeds the {MAX_READ_LINES}-line limit"
                    ));
                }
                // A width check alone does not catch a SATURATED range: `read f 18446744073709551615`
                // saturates to start == end, width 0, which passes. No file has a line
                // 18446744073709551615, so an absurd start is refused on its own terms. Caught by
                // the test written for the overflow fix — the fix removed the panic and left the
                // nonsense.
                const MAX_LINE: usize = 10_000_000;
                if *start > MAX_LINE {
                    return Err(format!("start line {start} is beyond any real file"));
                }
                // `awk` with the REAL line number, not `sed | cat -n`.
                //
                // The previous form printed a constant `{start}:` prefix on every line, so `read f
                // 100 200` labelled all 101 lines "100:" — actively misleading for the one job this
                // verb has, which is giving the model accurate line references to cite in a patch.
                // Caught in review; the unit tests only asserted the generated shell string, never
                // ran it, so a wrong-but-well-formed command passed.
                //
                // `./` prefix rather than `--`: it makes an option-looking filename an operand for
                // any tool (BSD awk and sed both reject `--`, and this must also run outside the
                // container for local checks), and `safe_path` has already refused absolute paths
                // and traversal.
                Ok(format!(
                    "awk 'NR>={start} && NR<={end} {{ printf \"%d:%s\\n\", NR, $0 }}' {} 2>&1",
                    shell_quote(&format!("./{file}"))
                ))
            }
        }
    }
}

/// Parse one model-emitted exploration command.
///
/// Deliberately a tiny hand-written parser over three verbs rather than JSON tool-calling: the model
/// is being asked for one line, and a malformed line must degrade to a clear error the model can
/// read and retry, not to a schema violation.
///
/// # Errors
/// Unknown verb, missing arguments, or unparseable line numbers.
pub fn parse_cmd(line: &str) -> Result<ExploreCmd, String> {
    let line = line.trim();
    let (verb, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    let rest = rest.trim();
    match verb {
        "ls" => Ok(ExploreCmd::Ls {
            dir: if rest.is_empty() {
                ".".to_owned()
            } else {
                rest.to_owned()
            },
        }),
        "grep" => {
            // `grep <pattern> <dir>` — the LAST whitespace-separated token is the directory, so a
            // pattern may itself contain spaces.
            let (pattern, dir) = rest
                .rsplit_once(char::is_whitespace)
                .ok_or_else(|| "grep needs a pattern and a directory".to_owned())?;
            Ok(ExploreCmd::Grep {
                pattern: pattern.trim().to_owned(),
                dir: dir.trim().to_owned(),
            })
        }
        "read" => {
            let mut parts = rest.split_whitespace();
            let file = parts
                .next()
                .ok_or_else(|| "read needs a file".to_owned())?
                .to_owned();
            let start: usize = parts
                .next()
                .unwrap_or("1")
                .parse()
                .map_err(|_| "read start line must be a number".to_owned())?;
            let end: usize = parts
                .next()
                // `saturating_add`: `start` is parsed from model output, so `read f 18446744073709551615`
                // would overflow and panic in debug. The range check below then rejects the
                // saturated value, which is the correct outcome — a nonsense range is refused, not
                // crashed on. Flagged in review.
                .map_or(Ok(start.saturating_add(120)), str::parse)
                .map_err(|_| "read end line must be a number".to_owned())?;
            Ok(ExploreCmd::Read {
                file,
                start: start.max(1),
                end,
            })
        }
        other => Err(format!(
            "unknown command {other:?} — use `ls <dir>`, `grep <pattern> <dir>`, or \
             `read <file> <start> <end>`"
        )),
    }
}

/// Which verb a command is, for exit-code interpretation.
enum CmdKind {
    Ls,
    Grep,
    Read,
}

/// Classify a command. Used only to decide whether a non-zero exit is meaningful: `grep` returning
/// 1 is "no matches", while the same code from `ls` or `awk` is a real failure.
const fn self_kind(cmd: &ExploreCmd) -> CmdKind {
    match cmd {
        ExploreCmd::Ls { .. } => CmdKind::Ls,
        ExploreCmd::Grep { .. } => CmdKind::Grep,
        ExploreCmd::Read { .. } => CmdKind::Read,
    }
}

/// Run one exploration command inside the instance's container.
///
/// Reuses the **same fail-closed flags** as [`crate::swebench::evaluate`]: read-only root, no
/// network, all capabilities dropped, no-new-privileges, and the same memory/cpu/pids ceilings.
/// Nothing here relaxes the sandbox — the container is identical, only the command differs, and the
/// command is one of three read-only verbs assembled by us.
///
/// # Errors
/// Docker failures, verbatim. A command that merely finds nothing is a successful run with empty
/// output, not an error — "no matches" is information the model needs.
pub fn run_explore(
    instance: &SweInstance,
    cmd: &ExploreCmd,
    limits: &SweLimits,
) -> Result<String, String> {
    let script = cmd.to_shell()?;
    let out = Command::new("docker")
        .args(["run", "--rm"])
        .args(["--platform", "linux/amd64"])
        .args(["--network", "none"])
        .arg("--read-only")
        .args(["--tmpfs", "/tmp:rw,size=64m"])
        .args(["--memory", &format!("{}m", limits.mem_mb)])
        .args(["--cpus", &format!("{}", limits.cpus)])
        .args(["--pids-limit", "256"])
        .args(["--cap-drop", "ALL"])
        .args(["--security-opt", "no-new-privileges"])
        .args(["-w", "/testbed"])
        .arg(&instance.image)
        .args(["sh", "-c", &script])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("docker run failed for {}: {e}", instance.instance_id))?;

    // A non-zero exit is an INFRASTRUCTURE failure, and it must never be reported to the model as
    // "(no output)". That phrasing means "the path does not exist" — a false fact the model will
    // then reason from, wasting paid turns chasing a file it was wrongly told is absent. Docker
    // writes the real cause to stderr, so that is what gets surfaced. Flagged in review.
    // Exit 1 from `grep` means "no matches" — a legitimate ANSWER, and the most common one, since a
    // model's first guess at a symbol name often misses. Review flagged that the `status.success()`
    // check would turn that into a container failure and abort the instance.
    //
    // MEASURED, and the concern does not reproduce: every template ends in `| head -N`, and a shell
    // pipeline reports the LAST command's status, so a fruitless grep exits 0 via head. Verified
    // directly — piped 0, unpiped 1. The guard below is kept explicit anyway, because that
    // protection is an accident of the pipeline: anyone removing `| head` would silently
    // reintroduce the bug, and this makes the dependency visible instead of load-bearing-by-luck.
    //
    // Only exit 1 is forgiven, and only for grep. Exit 2 (grep's own error), 125-127 (docker could
    // not start), and 137 (OOM kill) still fail loudly.
    let grep_no_match = matches!(self_kind(cmd), CmdKind::Grep) && out.status.code() == Some(1);
    if !out.status.success() && !grep_no_match {
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim();
        return Err(format!(
            "exploration container failed for {} (exit {:?}): {}",
            instance.instance_id,
            out.status.code(),
            if err.is_empty() { "(no stderr)" } else { err }
        ));
    }

    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    if text.trim().is_empty() {
        // An empty result is a real answer ("no matches", "empty directory") and the model must be
        // told so explicitly, or it will assume the tool broke and waste a turn retrying.
        text = "(no output — no matches, or the path does not exist)".to_owned();
    }
    if text.len() > MAX_OUTPUT_BYTES {
        // `String::truncate` PANICS if the index is not a char boundary, and source files are full
        // of non-ASCII — a comment with an em dash or an accented name straddling byte 8000 would
        // abort the whole benchmark run mid-instance. Walk back to the nearest boundary instead.
        //
        // Same defect class as the `get(..64)` byte-slicing bug in the OpenAI error detector caught
        // earlier: reasoning about a UTF-8 string in bytes. I fixed that one and then wrote this
        // one. Flagged in review.
        let mut cut = MAX_OUTPUT_BYTES;
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
        text.push_str("\n… (truncated)");
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The security boundary.** A model-supplied path must never escape the repository, and it is
    /// rejected before a command is built rather than filtered afterwards.
    #[test]
    fn path_traversal_and_absolute_paths_are_refused() {
        for bad in [
            "../etc/passwd",
            "a/../../b",
            "/etc/passwd",
            "/",
            "foo/../../../root",
        ] {
            assert!(
                ExploreCmd::Ls { dir: bad.into() }.to_shell().is_err(),
                "must refuse {bad:?}"
            );
            assert!(
                ExploreCmd::Read {
                    file: bad.into(),
                    start: 1,
                    end: 10
                }
                .to_shell()
                .is_err(),
                "must refuse reading {bad:?}"
            );
        }
        // ...while ordinary repository paths are fine.
        assert!(
            ExploreCmd::Ls {
                dir: "astropy/modeling".into()
            }
            .to_shell()
            .is_ok()
        );
    }

    /// Shell metacharacters in a model-supplied grep pattern must not become shell syntax. This is
    /// the one place arbitrary model text reaches a command line, so it is the one place a quoting
    /// bug turns "read the code" into "run whatever the model wants".
    #[test]
    fn shell_metacharacters_cannot_escape_the_quoting() {
        let nasty = ExploreCmd::Grep {
            pattern: "'; rm -rf / #".into(),
            dir: "src".into(),
        }
        .to_shell()
        .expect("a hostile pattern is still a legal search string");
        // The dangerous sequence must appear only INSIDE quotes, never as bare syntax.
        assert!(
            !nasty.contains("; rm -rf / #\n") && nasty.contains(r"'\''"),
            "the quote must be escaped rather than closing the string: {nasty}"
        );
        // And the classic backtick / $() forms stay inert inside single quotes.
        let subst = ExploreCmd::Grep {
            pattern: "$(whoami)`id`".into(),
            dir: "src".into(),
        }
        .to_shell()
        .unwrap();
        assert!(subst.starts_with("grep -rnF -e '$(whoami)`id`'"), "{subst}");
    }

    /// **Truncating output at a byte offset must not panic on UTF-8.**
    ///
    /// `String::truncate` panics unless the index is a char boundary, and source files are full of
    /// non-ASCII: one em dash or accented identifier straddling byte 8000 would abort the whole
    /// benchmark mid-instance, losing every paid instance after it.
    ///
    /// Same defect class as the `get(..64)` byte-slicing bug in the OpenAI error detector — reasoning
    /// about a UTF-8 string in bytes. I fixed that one this session and then wrote this one, which
    /// is why the test asserts the boundary case directly rather than trusting the next reader to
    /// notice.
    #[test]
    fn oversized_output_truncates_on_a_char_boundary() {
        // Construct a string whose byte 8000 lands mid-character.
        let mut text = "x".repeat(MAX_OUTPUT_BYTES - 1);
        text.push('\u{00e9}');
        text.push_str(&"y".repeat(100));
        assert!(
            !text.is_char_boundary(MAX_OUTPUT_BYTES),
            "the fixture must actually straddle the cut, or it proves nothing"
        );

        // Mirror the truncation logic exactly.
        let mut cut = MAX_OUTPUT_BYTES;
        while cut > 0 && !text.is_char_boundary(cut) {
            cut -= 1;
        }
        let mut out = text.clone();
        out.truncate(cut); // must not panic
        assert!(out.len() < MAX_OUTPUT_BYTES);
        assert!(out.is_char_boundary(out.len()));
    }

    /// **`grep` exit 1 means "no matches", not "the container failed".**
    ///
    /// This regression was introduced by the previous commit's fix for silent container failures:
    /// tightening the error path to check `status.success()` swept up grep's most common SUCCESS
    /// case. A model's first guess at a symbol name often misses, so aborting the instance on a
    /// search miss would be routine. Flagged in review.
    #[test]
    fn a_grep_with_no_matches_is_an_answer_not_a_failure() {
        let dir = std::env::temp_dir().join("fp-swe-explore-grep");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "hello\n").unwrap();

        let script = ExploreCmd::Grep {
            pattern: "definitely_not_present_xyzzy".into(),
            dir: ".".into(),
        }
        .to_shell()
        .unwrap();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .current_dir(&dir)
            .output()
            .expect("sh runs");
        // Exit 0, not 1: the `| head -N` pipeline reports HEAD's status, so a fruitless grep is
        // already benign before the guard in `run_explore` sees it. Asserting the true value rather
        // than the one the fix assumed — the reviewer's concern was real in principle and does not
        // reproduce here, and the reason is worth pinning down so a later change to the pipeline
        // does not silently reintroduce it.
        assert_eq!(
            out.status.code(),
            Some(0),
            "a piped grep reports head's status, masking grep's exit 1"
        );
        assert!(
            out.stdout.is_empty(),
            "no matches must produce no output, which renders as '(no output)'"
        );
    }

    /// A NUL byte in a grep pattern truncates the argument at exec, so the model would silently
    /// search for a prefix of what it asked for. Paths were already checked; patterns were not.
    #[test]
    fn a_nul_byte_in_a_grep_pattern_is_refused() {
        assert!(
            ExploreCmd::Grep {
                pattern: "foo\0bar".into(),
                dir: "src".into()
            }
            .to_shell()
            .is_err()
        );
    }

    /// **The generated command must be EXECUTED, not just inspected.**
    ///
    /// Every other test here asserts the shell string. That is how a wrong-but-well-formed command
    /// shipped: `sed -n '100,200p' | cat -n | sed 's/^/100:/'` looks plausible and labels all 101
    /// lines "100:", destroying the line references this verb exists to provide. Reviewer caught it;
    /// no string assertion could have.
    ///
    /// Runs `sh` on the host against a temp file — no Docker, so it works in CI — because the
    /// question is whether the command text is *correct*, not whether the container works.
    #[test]
    fn the_read_command_actually_emits_real_line_numbers() {
        use std::io::Write as _;
        let dir = std::env::temp_dir().join("fp-swe-explore-read");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 1..=10 {
            writeln!(f, "line{i}").unwrap();
        }
        drop(f);

        let script = ExploreCmd::Read {
            file: "lines.txt".into(),
            start: 3,
            end: 5,
        }
        .to_shell()
        .unwrap();

        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .current_dir(&dir)
            .output()
            .expect("sh must run");
        let text = String::from_utf8_lossy(&out.stdout);

        assert!(
            text.contains("3:line3") && text.contains("4:line4") && text.contains("5:line5"),
            "each line must carry its OWN number, got:\n{text}\nfrom: {script}"
        );
        assert!(
            !text.contains("3:line4"),
            "a constant prefix on every line is the bug this test exists for:\n{text}"
        );
        assert!(
            !text.contains("line2") && !text.contains("line6"),
            "the range must be respected:\n{text}"
        );
    }

    /// **Quoting is not enough: a path that looks like a CLI OPTION must not be parsed as one.**
    ///
    /// `--version` or `-rf` is perfectly safe as shell syntax and still gets consumed as an option
    /// by the tool receiving it — an injection surface one layer below the shell. Every template
    /// carries a `--` terminator (and `grep` an explicit `-e`) so a model-supplied value is always
    /// an operand. Flagged in review.
    #[test]
    fn option_like_arguments_are_treated_as_operands() {
        let ls = ExploreCmd::Ls {
            dir: "--version".into(),
        }
        .to_shell()
        .unwrap();
        assert!(
            ls.contains("-- '--version'"),
            "ls must terminate options: {ls}"
        );

        let grep = ExploreCmd::Grep {
            pattern: "-rf".into(),
            dir: "-x".into(),
        }
        .to_shell()
        .unwrap();
        assert!(
            grep.contains("-e '-rf'") && grep.contains("-- '-x'"),
            "grep must take the pattern via -e and terminate options before the path: {grep}"
        );

        let read = ExploreCmd::Read {
            file: "-n".into(),
            start: 1,
            end: 5,
        }
        .to_shell()
        .unwrap();
        // `read` uses a `./` prefix rather than `--`: it makes an option-looking filename an
        // operand for any tool, including the BSD awk/sed that reject `--` outright.
        assert!(
            read.contains("'./-n'"),
            "an option-looking filename must be made an operand: {read}"
        );
    }

    /// A read must be bounded. Without a cap, one call could pull an entire file into the next
    /// turn's prompt — expensive, and it drowns the signal the model needs.
    #[test]
    fn reads_are_bounded_and_ranges_validated() {
        assert!(
            ExploreCmd::Read {
                file: "f.py".into(),
                start: 1,
                end: 10_000
            }
            .to_shell()
            .is_err(),
            "an unbounded read must be refused"
        );
        assert!(
            ExploreCmd::Read {
                file: "f.py".into(),
                start: 500,
                end: 100
            }
            .to_shell()
            .is_err(),
            "an inverted range must be refused"
        );
        assert!(
            ExploreCmd::Read {
                file: "f.py".into(),
                start: 100,
                end: 200
            }
            .to_shell()
            .is_ok()
        );
    }

    /// The parser must accept what a model actually writes, and reject the rest with a message the
    /// model can act on. A cryptic parse failure costs a paid turn.
    #[test]
    fn the_parser_handles_realistic_model_output() {
        assert_eq!(
            parse_cmd("ls astropy/modeling").unwrap(),
            ExploreCmd::Ls {
                dir: "astropy/modeling".into()
            }
        );
        assert_eq!(
            parse_cmd("ls").unwrap(),
            ExploreCmd::Ls { dir: ".".into() },
            "a bare `ls` means the repo root"
        );
        // A pattern containing spaces still parses: the LAST token is the directory.
        assert_eq!(
            parse_cmd("grep def separability_matrix astropy").unwrap(),
            ExploreCmd::Grep {
                pattern: "def separability_matrix".into(),
                dir: "astropy".into()
            }
        );
        assert_eq!(
            parse_cmd("read astropy/modeling/separable.py 100 200").unwrap(),
            ExploreCmd::Read {
                file: "astropy/modeling/separable.py".into(),
                start: 100,
                end: 200
            }
        );
        // Unknown verbs fail with guidance rather than silently doing something.
        let err = parse_cmd("cat /etc/passwd").unwrap_err();
        assert!(err.contains("unknown command"), "{err}");
        assert!(
            err.contains("ls"),
            "the error must list the legal verbs: {err}"
        );
    }

    /// **A model-supplied line number must not be able to panic the harness.**
    ///
    /// `read <file> <huge>` overflowed `start + 120` and panicked in debug builds. Every argument
    /// here is parsed from model output, so "no sensible model would send that" is not a bound —
    /// a confused model, or a hostile one, sends exactly that. Flagged in review.
    #[test]
    fn an_absurd_line_number_is_refused_rather_than_panicking() {
        let cmd = parse_cmd(&format!("read f.py {}", usize::MAX))
            .expect("parsing a huge start line must not panic");
        // Saturation makes the range absurd, and the range check then refuses it — refused, not
        // crashed, which is the whole point.
        assert!(
            cmd.to_shell().is_err(),
            "a saturated range must be rejected by the width check"
        );
        // The explicit two-argument form must be equally safe.
        let cmd2 = parse_cmd(&format!("read f.py 1 {}", usize::MAX)).expect("must not panic");
        assert!(cmd2.to_shell().is_err());
    }

    /// There is no verb that writes, executes arbitrary code, or reaches the network. This asserts
    /// the *shape* of the interface: if someone later adds a write verb, this test is where the
    /// safety argument gets revisited rather than quietly lost.
    #[test]
    fn the_verb_set_is_read_only() {
        for cmd in [
            ExploreCmd::Ls { dir: "src".into() },
            ExploreCmd::Grep {
                pattern: "x".into(),
                dir: "src".into(),
            },
            ExploreCmd::Read {
                file: "f.py".into(),
                start: 1,
                end: 10,
            },
        ] {
            let sh = cmd.to_shell().unwrap();
            for forbidden in ["rm ", "mv ", "cp ", "curl", "wget", "pip", "chmod", "tee "] {
                assert!(
                    !sh.contains(forbidden),
                    "read-only verb {cmd:?} produced {forbidden:?} in: {sh}"
                );
            }
            // File redirection specifically, not every `>`: `2>&1` is stderr capture and is fine,
            // while `> file` or `>> file` would be a write. Checking for a bare `>` failed here on
            // exactly that distinction — worth encoding precisely rather than loosening.
            // Strip constructs that legitimately contain `>`: stderr capture, and awk's `>=`
            // comparison. What must not survive is redirection INTO a file. Sharpened twice now —
            // first for `2>&1`, then for `>=` — because "contains a greater-than sign" is a proxy
            // for "writes" and the proxy keeps catching syntax that is not a write.
            let writes = sh.replace("2>&1", "").replace(">=", "");
            assert!(
                !writes.contains('>'),
                "read-only verb {cmd:?} redirects to a file: {sh}"
            );
        }
    }
}
