//! Validator for the Codex `apply_patch` freeform-tool grammar.
//!
//! A port of the *acceptance* behaviour of Codex's own parser (`codex-rs/apply-patch`:
//! `parser.rs` boundary checks + `streaming_parser.rs` state machine, read 2026-11-04), including
//! its lenient mode (a `<<EOF` heredoc wrapper is stripped) — a patch Codex accepts must never
//! fail here. It checks structure only; it builds no hunks and never touches the filesystem.
//! One deliberate stricter point: a patch with zero hunks parses in Codex but is rejected at
//! apply time ("No files were modified."), so it fails here too (the old envelope regex agreed).

const BEGIN: &str = "*** Begin Patch";
const END: &str = "*** End Patch";
const ADD: &str = "*** Add File: ";
const DELETE: &str = "*** Delete File: ";
const UPDATE: &str = "*** Update File: ";
const MOVE_TO: &str = "*** Move to: ";
const EOF: &str = "*** End of File";
const ENV_ID: &str = "*** Environment ID:";
const CTX: &str = "@@ ";
const CTX_EMPTY: &str = "@@";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    NotStarted,
    Started,
    Add,
    Delete,
    Update,
    Ended,
}

/// Update-hunk bookkeeping: just enough to reproduce Codex's accept/reject decisions.
#[derive(Default)]
struct Update {
    chunks: usize,
    /// Last chunk has at least one `+`/`-`/context line.
    last_has_lines: bool,
    last_eof: bool,
    moved: bool,
}

/// `Ok(())` when Codex's lenient parser would accept `patch` with at least one hunk; otherwise a
/// short reason (with 1-based line number) suitable for a gate verdict.
pub(crate) fn validate(patch: &str) -> Result<(), String> {
    let all: Vec<&str> = patch.trim().lines().collect();
    let lines = boundaries(&all)?;
    let mut mode = Mode::NotStarted;
    let (mut hunks, mut env_seen) = (0usize, false);
    let mut up = Update::default();
    let last = lines.len().saturating_sub(1);
    for (i, raw) in lines.iter().enumerate() {
        let n = i + 1;
        // Codex re-splits on '\n' and strips one more '\r'.
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        let trimmed = line.trim();
        let err = |m: String| -> Result<(), String> { Err(format!("line {n}: {m}")) };
        match mode {
            Mode::NotStarted => {
                if trimmed != BEGIN {
                    return err("first line must be '*** Begin Patch'".into());
                }
                mode = Mode::Started;
                continue;
            }
            Mode::Ended => {
                if trimmed.is_empty() {
                    continue;
                }
                return err("content after '*** End Patch'".into());
            }
            _ => {}
        }
        // Headers / End Patch: update hunks compare the trim_end form, other modes the fully
        // trimmed one; the unterminated final line is always matched trimmed (Codex `finish`).
        let head = if mode == Mode::Update && !(i == last && trimmed == END) {
            line.trim_end()
        } else {
            trimmed
        };
        if mode == Mode::Started
            && let Some(id) = head.strip_prefix(ENV_ID)
        {
            if env_seen || id.trim().is_empty() {
                return err("environment id repeated or empty".into());
            }
            env_seen = true;
            continue;
        }
        let next = if head == END {
            Some(Mode::Ended)
        } else if head.starts_with(ADD) {
            Some(Mode::Add)
        } else if head.starts_with(DELETE) {
            Some(Mode::Delete)
        } else if head.starts_with(UPDATE) {
            Some(Mode::Update)
        } else {
            None
        };
        if let Some(next) = next {
            if mode == Mode::Update {
                if up.chunks == 0 {
                    return err("update hunk is empty".into());
                }
                if !up.last_has_lines {
                    return err("update hunk does not contain any lines".into());
                }
            }
            if next != Mode::Ended {
                hunks += 1;
            }
            if next == Mode::Update {
                up = Update::default();
            }
            mode = next;
            continue;
        }
        match mode {
            Mode::Add if !line.starts_with('+') => {
                return err(format!(
                    "Add File line must start with '+', got '{trimmed}'"
                ));
            }
            Mode::Update => {
                if let Err(m) = update_line(&mut up, line, head) {
                    return err(m);
                }
            }
            Mode::Started | Mode::Delete => {
                return err(format!("'{trimmed}' is not a valid hunk header"));
            }
            _ => {}
        }
    }
    if mode != Mode::Ended {
        return Err("last line must be '*** End Patch'".into());
    }
    if hunks == 0 {
        return Err("patch has no file operations".into());
    }
    Ok(())
}

/// Strict `Begin`/`End` check, or the `<<EOF` heredoc-wrapped form (Codex lenient mode).
fn boundaries<'a>(lines: &'a [&'a str]) -> Result<&'a [&'a str], String> {
    fn strict<'a>(l: &'a [&'a str]) -> Result<&'a [&'a str], String> {
        match (l.first(), l.last()) {
            (Some(f), Some(e)) if f.trim() == BEGIN && e.trim() == END => Ok(l),
            (Some(f), _) if f.trim() != BEGIN => Err("first line must be '*** Begin Patch'".into()),
            _ => Err("last line must be '*** End Patch'".into()),
        }
    }
    strict(lines).or_else(|e| match lines {
        [first, .., last]
            if matches!(*first, "<<EOF" | "<<'EOF'" | "<<\"EOF\"")
                && last.ends_with("EOF")
                && lines.len() >= 4 =>
        {
            strict(&lines[1..lines.len() - 1])
        }
        _ => Err(e),
    })
}

/// One non-header line inside an `*** Update File` hunk. `t` is the `trim_end` form.
fn update_line(up: &mut Update, line: &str, t: &str) -> Result<(), String> {
    let bad = || format!("unexpected line in update hunk: '{line}' (need ' ', '+' or '-' prefix)");
    if up.last_eof {
        if t.is_empty() {
            return Ok(());
        }
        if t != CTX_EMPTY && !t.starts_with(CTX) {
            return Err(format!(
                "expected '@@' after '*** End of File', got '{line}'"
            ));
        }
    }
    if up.chunks == 0 && !up.moved && t.starts_with(MOVE_TO) {
        up.moved = true;
        return Ok(());
    }
    let is_ctx = t == CTX_EMPTY || t.starts_with(CTX);
    let empty_chunk = up.chunks > 0 && !up.last_has_lines;
    if is_ctx && empty_chunk {
        return Err(bad());
    }
    if is_ctx {
        (up.chunks, up.last_has_lines, up.last_eof) = (up.chunks + 1, false, false);
        return Ok(());
    }
    if t == EOF {
        if up.chunks == 0 || empty_chunk {
            return Err("update hunk does not contain any lines".into());
        }
        up.last_eof = true;
        return Ok(());
    }
    if line.is_empty() || line.starts_with([' ', '+', '-']) {
        if up.chunks == 0 {
            up.chunks = 1;
        }
        up.last_has_lines = true;
        return Ok(());
    }
    if up.chunks > 0 && up.last_has_lines {
        return Err(format!("expected '@@' context marker, got '{line}'"));
    }
    Err(bad())
}

#[cfg(test)]
mod tests {
    use super::validate;

    fn ok(p: &str) {
        assert_eq!(validate(p), Ok(()), "should pass: {p:?}");
    }
    fn bad(p: &str) {
        assert!(validate(p).is_err(), "should fail: {p:?}");
    }

    /// Positive cases are lifted from Codex's own parser tests (`parser.rs`, `streaming_parser.rs`).
    #[test]
    fn codex_valid_patches_pass() {
        ok("*** Begin Patch\n*** Add File: foo\n+hi\n *** End Patch");
        ok(
            "*** Begin Patch\n*** Add File: path/add.py\n+abc\n+def\n*** Delete File: path/delete.py\n*** Update File: path/update.py\n*** Move to: path/update2.py\n@@ def f():\n-    pass\n+    return 123\n*** End Patch",
        );
        ok(
            "*** Begin Patch\n*** Update File: file.py\n@@\n+line\n*** Add File: other.py\n+content\n*** End Patch",
        );
        // No leading @@ on the first chunk.
        ok("*** Begin Patch\n*** Update File: file2.py\n import foo\n+bar\n*** End Patch");
        // End of File, followed by a blank line.
        ok(
            "*** Begin Patch\n*** Update File: file.txt\n@@\n+quux\n*** End of File\n\n*** End Patch",
        );
        // Two chunks, blank line (= empty context), CRLF endings.
        ok(
            "*** Begin Patch\r\n*** Update File: a.rs\r\n@@ fn a()\r\n-x\r\n\r\n+y\r\n@@\r\n z\r\n-q\r\n*** End Patch\r\n",
        );
        // Environment id; absolute paths; empty `+` line in an Add.
        ok(
            "*** Begin Patch\n*** Environment ID: dev\n*** Add File: /tmp/a\n+\n+x\n*** End Patch\n",
        );
    }

    /// Codex lenient mode: heredoc wrapper (quoted or not).
    #[test]
    fn heredoc_wrapped_patch_passes() {
        let body = "*** Begin Patch\n*** Update File: README.md\n@@\n-a\n+b\n*** End Patch";
        ok(&format!("<<'EOF'\n{body}\nEOF\n"));
        ok(&format!("<<EOF\n{body}\nEOF"));
        ok(&format!("<<\"EOF\"\n{body}\nEOF"));
        bad(&format!("<<'EOT'\n{body}\nEOT"));
    }

    #[test]
    fn structurally_broken_patches_fail() {
        let wrap = |b: &str| format!("*** Begin Patch\n{b}\n*** End Patch");
        // Hunk line with no prefix, in a valid envelope (the old regex accepted this).
        bad(&wrap("*** Update File: a.rs\n@@\n-x\n+y\nbare line"));
        bad(&wrap("*** Update File: a.rs\nbare line"));
        // Add File with non-+ lines.
        bad(&wrap("*** Add File: a.rs\nfn main() {}"));
        bad(&wrap("*** Add File: a.rs\n+ok\n-nope"));
        // Missing path.
        bad(&wrap("*** Add File:\n+x"));
        bad(&wrap("*** Delete File:"));
        // Empty update hunk, empty chunk, junk after delete, junk header, junk after EOF marker.
        bad(&wrap("*** Update File: a.rs"));
        bad(&wrap("*** Update File: a.rs\n@@"));
        bad(&wrap("*** Update File: a.rs\n@@\n@@\n+x"));
        bad(&wrap("*** Delete File: a.rs\n+x"));
        bad(&wrap("*** Frobnicate: a.rs"));
        bad(&wrap(
            "*** Update File: a.rs\n@@\n+x\n*** End of File\njunk",
        ));
        bad("*** Begin Patch\n*** Delete File: a\n*** End Patch\ntrailing");
        // Envelope: unified diff, missing end, empty patch (Codex rejects at apply time).
        bad("--- a/a.rs\n+++ b/a.rs\n@@\n-x\n+y\n");
        bad("*** Begin Patch\n*** Delete File: a");
        bad("*** Begin Patch\n*** End Patch");
        bad("");
    }
}
