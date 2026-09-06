//! SWE-bench evaluation (ADR 0010) — repository-scale tasks, run without weakening the sandbox.
//!
//! A SWE-bench instance is a **repository at a commit**, not a function. Resolving one means
//! applying the benchmark's `test_patch`, applying a candidate patch, and checking that every test
//! in `FAIL_TO_PASS` now passes while every test in `PASS_TO_PASS` still does.
//!
//! # Why this does not reuse `ContainerSandbox`
//!
//! Two things differ, and both are structural rather than cosmetic. Each instance needs its **own**
//! image (the repo and its built dependencies are baked in), whereas `ContainerSandbox` is
//! constructed around one image for a whole run. And the code under test arrives *in* the image at
//! `/testbed` rather than being streamed in as files. Reusing the type would mean bending it into
//! a per-run image and a second delivery mechanism — so this module runs the container directly
//! and keeps ADR 0002's D2 invariants **verbatim**:
//!
//! - `--network none`, at eval time (images are pulled beforehand, by a step that runs no model output)
//! - `--read-only` rootfs — see below for how a repo gets patched anyway
//! - no host bind-mounts; the repo comes from the image, inputs arrive on stdin as a tar
//! - `--rm`, `--cap-drop ALL`, `--security-opt no-new-privileges`, cpu/mem/pids caps, wall-clock kill
//!
//! # The read-only rootfs is kept, not traded away
//!
//! ADR 0010's first draft assumed patching a repo requires a writable rootfs and proposed a weaker
//! sandbox tier for it. That was never tested, and it is wrong. The repo is copied out of the image
//! into the **tmpfs workdir**, which is already writable and already discarded with the container,
//! and patched there.
//!
//! One subtlety decides whether this measures anything at all: these images install the project
//! **editable, pointing at `/testbed`**. So a patched copy at `/work/repo` can be silently ignored
//! and the run would score the *unpatched* code — passing tests that prove nothing. `PYTHONPATH`
//! puts the copy first, and this was verified directly on `astropy__astropy-12907`: a marker
//! appended to the copy is visible to the interpreter, and `/testbed` is confirmed unwritable in
//! the same breath.
//!
//! # The control that makes a result trustworthy
//!
//! `PASS_TO_PASS` is run **before** the candidate patch as well as after. If those tests do not
//! already pass on the base commit, the environment is broken and the instance is excluded and
//! counted — never scored as a model failure. This is the same rule as `FP_MISSING` in
//! `coding.rs`: an environment fault that is scored as a wrong answer manufactures error out of
//! nothing, and a rate computed over an unstated subset is worse than no rate at all.

use std::process::{Command, Stdio};

/// One SWE-bench instance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SweInstance {
    /// e.g. `"astropy__astropy-12907"`.
    pub instance_id: String,
    /// e.g. `"astropy/astropy"`.
    pub repo: String,
    /// Commit the image is built at.
    pub base_commit: String,
    /// The issue text a candidate is asked to fix.
    pub problem_statement: String,
    /// Diff adding the tests that encode the bug. Always applied; never shown to the candidate.
    pub test_patch: String,
    /// Tests that must go from failing to passing.
    pub fail_to_pass: Vec<String>,
    /// Tests that must not regress.
    pub pass_to_pass: Vec<String>,
    /// Published eval image for this instance.
    pub image: String,
}

/// Which tolerance level (if any) applied the candidate patch, read off the eval script's markers.
///
/// Pure and separate so the classification is testable without a container. Two distinctions here
/// are load-bearing and both were got wrong first time:
/// - `no-patch` (the model emitted nothing) is NOT `rejected` (git refused a real diff). One is a
///   budget/format failure, the other is diff brittleness; collapsing them rebuilds the exact
///   conflation the taxonomy exists to break.
/// - the fallbacks are checked most-permissive-first, because the script's markers are prefixed
///   ("applied fuzz" contains "applied") and a naive check would report every fallback as `clean`.
#[must_use]
pub fn classify_apply(stdout: &str) -> String {
    if stdout.contains("FP_PATCH empty") {
        "no-patch"
    } else if stdout.contains("FP_PATCH applied fuzz") {
        "fuzz"
    } else if stdout.contains("FP_PATCH applied c1") {
        "c1"
    } else if stdout.contains("FP_PATCH applied recount") {
        "recount"
    } else if stdout.contains("FP_PATCH applied") {
        "clean"
    } else {
        "rejected"
    }
    .to_owned()
}

/// Split `PASS_TO_PASS` into the part a GATE may see and the part reserved for the oracle.
///
/// The 50-instance run measured the gate at **3/82 = 3.7% precision**: 79 patches fixed the named
/// failing test and broke something else, and the gate — which only checked the named test —
/// waved every one of them through. A cascade router whose verifier is wrong 96% of the time
/// stops early on garbage, confidently. That is the product's core mechanism failing, not a
/// benchmark artifact.
///
/// The obvious fix is to make the gate check regressions too, and the obvious fix is a trap: a
/// gate that runs the FULL `PASS_TO_PASS` list becomes byte-identical to the oracle, and a
/// benchmark whose gate equals its oracle can no longer measure its own gate's error. That is the
/// one line this repo does not cross.
///
/// So the gate sees a realistic SUBSET: the regression tests living in the same files as the
/// failing tests — what a developer or agent actually runs next to a fix. The oracle keeps the
/// entire list, including everything the gate never saw. The gate can still be wrong, and the
/// benchmark can still catch it being wrong.
///
/// Deterministic: no sampling, no ambient randomness, so an auditor reproduces the split exactly.
#[must_use]
pub fn split_p2p_for_gate(
    fail_to_pass: &[String],
    pass_to_pass: &[String],
) -> (Vec<String>, Vec<String>) {
    let file_of = |t: &str| t.split("::").next().unwrap_or(t).to_owned();
    let touched: std::collections::BTreeSet<String> =
        fail_to_pass.iter().map(|t| file_of(t)).collect();
    // Siblings FIRST, then a deterministic slice of everything else.
    //
    // Siblings alone were measured at 0/256 precision on django: 256 patches passed their target
    // tests and their same-file regression tests, and every one broke something further away.
    // Django's reverse dependencies span the codebase, so "tests in the file I changed" is not a
    // regression check there -- it only looked like one on astropy, where behaviour is local.
    //
    // Every SPREAD_DENOM-th non-sibling test joins the gate, so coverage reaches modules the
    // change might break without the gate becoming the oracle: the rest stays reserved, which is
    // what keeps the gate's own error measurable.
    const SPREAD_DENOM: usize = 3;
    let (mut gate, mut rest): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    let mut far = 0usize;
    for t in pass_to_pass {
        if touched.contains(&file_of(t)) {
            gate.push(t.clone());
        } else {
            if far.is_multiple_of(SPREAD_DENOM) {
                gate.push(t.clone());
            } else {
                rest.push(t.clone());
            }
            far += 1;
        }
    }
    // A fix in a file with no sibling regression tests would leave the gate blind again, so fall
    // back to a deterministic slice rather than to nothing.
    if gate.is_empty() {
        let take = rest.len().min(10);
        gate = rest.drain(..take).collect();
    }
    (gate, rest)
}

/// What one instance produced.
#[derive(Debug, Clone)]
pub struct SweOutcome {
    /// Instance id.
    pub instance_id: String,
    /// `FAIL_TO_PASS` all pass **and** `PASS_TO_PASS` all still pass — the official bar.
    pub resolved: bool,
    /// The pre-flight control: `PASS_TO_PASS` passed on the base commit before any candidate patch.
    /// False means the environment is broken and the instance must be excluded, not scored.
    pub control_ok: bool,
    /// The candidate patch did not apply. A real outcome (a model producing an unusable diff),
    /// distinct from an environment fault.
    pub patch_applied: bool,
    /// Which tolerance level the patch needed: `clean`, `recount`, `c1`, `fuzz`, or `rejected`.
    pub apply_method: String,
    /// `(passed, total)` for `FAIL_TO_PASS` after the candidate patch.
    pub f2p: (usize, usize),
    /// `(passed, total)` for `PASS_TO_PASS` after the candidate patch — the ORACLE's full list.
    pub p2p: (usize, usize),
    /// `(passed, total)` for the regression tests a GATE is allowed to see: the `PASS_TO_PASS`
    /// entries sharing a file with the failing tests. A strict subset of [`Self::p2p`], so the
    /// gate stays weaker than the oracle and its error remains measurable.
    pub p2p_gate: (usize, usize),
}

/// Resource ceilings for one instance. The workdir must hold a copy of the repository.
#[derive(Debug, Clone, Copy)]
pub struct SweLimits {
    /// tmpfs size for `/work` in MiB. The repo is copied here, so it must exceed the repo size —
    /// a repo that does not fit is an abort, never a silent truncation.
    pub workdir_mb: u64,
    /// Memory cap in MiB.
    pub mem_mb: u64,
    /// CPU cores.
    pub cpus: f32,
    /// Wall-clock ceiling in seconds for the whole instance.
    pub wall_s: u64,
}

impl Default for SweLimits {
    fn default() -> Self {
        Self {
            workdir_mb: 4096,
            mem_mb: 8192,
            cpus: 2.0,
            wall_s: 1800,
        }
    }
}

/// Load SWE-bench instances from JSONL (see `scripts/fetch-coding-dataset.py --dataset swebench`).
///
/// # Errors
/// The path can't be read, a line isn't valid JSON, or a required field is missing.
pub fn load_swebench_jsonl(path: &str) -> Result<Vec<SweInstance>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str::<SweInstance>(line)
                .map_err(|e| format!("{path}:{}: {e}", i + 1))
                .and_then(|inst| {
                    if inst.fail_to_pass.is_empty() {
                        Err(format!(
                            "{path}:{}: {:?} has no FAIL_TO_PASS tests, so resolving it is \
                             unfalsifiable",
                            i + 1,
                            inst.instance_id
                        ))
                    } else {
                        Ok(inst)
                    }
                })
        })
        .collect()
}

/// The in-container script. Kept as one shell program so the whole instance is a single
/// container lifetime — no state survives it, and the wall-clock kill covers everything.
///
/// Emits `FP_*` markers rather than relying on exit codes, because pytest's exit code cannot
/// distinguish "the tests failed" (a real result) from "the environment is broken" (not one).
fn eval_script() -> String {
    r#"set -u
mkdir -p /work/in && tar -xf - -C /work/in
# -a would also preserve ownership; some images (matplotlib) carry a vendored build tree
# owned by an alien uid, and chown fails for a non-root container user. The copy itself
# succeeds, so treating that exit code as a dead environment silently drops instances.
cp -dR --preserve=mode,timestamps,links /testbed /work/repo 2>/work/cp.err || { echo "FP_ENV copy-failed: $(tr '\n' ' ' < /work/cp.err | cut -c1-200)"; exit 0; }
cd /work/repo
. /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate testbed 2>/dev/null
# The copy must win over the editable install that points at /testbed, or the run scores
# unpatched code and every number it produces is meaningless.
export PYTHONPATH=/work/repo
run() { python -m pytest -q -p no:cacheprovider --no-header $(tr '\n' ' ' < "$1") 2>&1 | tail -3; }
# Verbose variant: emits one line per test so the GATE subset can be scored from the SAME
# invocation as the oracle. Splitting PASS_TO_PASS into two pytest runs looked equivalent and was
# not -- the 6 oracle-only tests hit `TypeError` during COLLECTION when imported without their
# siblings, so a gold patch that resolves scored 13/19 and the oracle called it a failure. Same
# tests, same container, different invocation, opposite verdict. Partition the RESULTS, never the
# run. Filtered to result lines plus the summary so a suite of hundreds does not flood stdout.
runv() { python -m pytest -q -p no:cacheprovider --no-header -v --tb=no $(tr '\n' ' ' < "$1") 2>&1 \
  | grep -E "::.+ (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)|[0-9]+ (passed|failed|error)" | tail -800; }

git apply /work/in/test.patch 2>/work/tp.err || { echo "FP_ENV test-patch-failed: $(tr '\n' ' ' < /work/tp.err | cut -c1-200)"; exit 0; }

# CONTROL: PASS_TO_PASS must already pass on the base commit. If not, this environment cannot
# measure anything and the instance is excluded rather than blamed on the model.
echo "FP_CONTROL_BEGIN"; run /work/in/p2p.txt; echo "FP_CONTROL_END"

if [ -s /work/in/model.patch ]; then
  # Tolerance cascade, strictest first. A 10-instance diagnostic found 95% of model patches
  # (57/60) rejected by plain `git apply` -- the reasoning was never reached, because the diff
  # was thrown away before any test ran. The patches are structurally well-formed unified diffs;
  # what they get wrong is exact hunk line numbers and context, which is precisely what these
  # flags forgive. Each step is strictly more permissive than the last, and the FIRST that
  # succeeds wins, so a patch that applies cleanly is unaffected.
  #
  #   --recount    : trust the hunk BODY, recompute the @@ counts the model miscounted
  #   -C1          : require 1 line of matching context instead of 3
  #   patch --fuzz : GNU patch, ignore whitespace, allow hunks to slide
  #
  # The `patch` step is guarded by --dry-run first, and this is not belt-and-braces. `git apply`
  # is ATOMIC -- it applies every hunk or none -- but GNU `patch` is not: on partial success it
  # writes the hunks it liked, drops .rej files, and exits non-zero. Without the dry run, a
  # "rejected" verdict could leave the repo HALF PATCHED, and the F2P/P2P runs that follow would
  # score a tree that is neither the base commit nor the model's patch. Silent result
  # contamination, reported as a clean rejection. Caught in review.
  #
  # This forgives WHERE a change goes, never WHAT it changes: a hunk still has to match real
  # code, and FAIL_TO_PASS/PASS_TO_PASS remain the oracle. A patch that applies in the wrong
  # place fails the tests, so tolerance cannot manufacture a resolution.
  if git apply /work/in/model.patch 2>/dev/null; then echo "FP_PATCH applied"
  elif git apply --recount /work/in/model.patch 2>/dev/null; then echo "FP_PATCH applied recount"
  elif git apply --recount -C1 /work/in/model.patch 2>/dev/null; then echo "FP_PATCH applied c1"
  elif patch -p1 -l -f --fuzz=3 --dry-run -i /work/in/model.patch >/dev/null 2>&1 \
       && patch -p1 -l -f --fuzz=3 -i /work/in/model.patch >/dev/null 2>&1; then echo "FP_PATCH applied fuzz"
  else echo "FP_PATCH rejected"; fi
else
  echo "FP_PATCH empty"
fi

echo "FP_F2P_BEGIN"; run /work/in/f2p.txt; echo "FP_F2P_END"
echo "FP_P2P_BEGIN"; runv /work/in/p2p.txt; echo "FP_P2P_END"
"#
    .to_owned()
}

/// Parse `N passed`, `N failed`, `N error` out of a pytest summary line into `(passed, total)`.
///
/// Deliberately tolerant: pytest's summary wording shifts between versions and plugins, and a
/// harness that mis-parses one project's output would silently report zeros for it.
#[must_use]
pub fn parse_pytest(section: &str) -> (usize, usize) {
    let (mut passed, mut other) = (0usize, 0usize);
    for line in section.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        for w in toks.windows(2) {
            let Ok(n) = w[0].trim_start_matches('=').trim().parse::<usize>() else {
                continue;
            };
            match w[1].trim_end_matches(',') {
                "passed" => passed = passed.max(n),
                "failed" | "error" | "errors" | "xfailed" => other = other.max(n),
                _ => {}
            }
        }
    }
    (passed, passed + other)
}

/// Score the gate-visible subset from the per-test lines of the full `PASS_TO_PASS` run.
///
/// The oracle keeps reading the pytest SUMMARY, so `resolved` means exactly what it meant in
/// every earlier benchmark file. The gate is derived from the same output, never a second run:
/// re-invoking pytest on a subset changes collection and therefore changes results.
///
/// Returns `(passed, scored)` over gate-visible tests only. SKIPPED is excluded from both, which
/// matches [`parse_pytest`] — a skipped test is not evidence either way.
#[must_use]
pub fn score_gate_subset(section: &str, gate_ids: &[String]) -> (usize, usize) {
    let want: std::collections::BTreeSet<&str> = gate_ids.iter().map(String::as_str).collect();
    let (mut passed, mut scored) = (0usize, 0usize);
    for line in section.lines() {
        let mut it = line.split_whitespace();
        let (Some(id), Some(verdict)) = (it.next(), it.next()) else {
            continue;
        };
        if !want.contains(id) {
            continue;
        }
        match verdict {
            "PASSED" | "XPASS" => {
                passed += 1;
                scored += 1;
            }
            "FAILED" | "ERROR" | "XFAIL" => scored += 1,
            _ => {}
        }
    }
    (passed, scored)
}

/// Pull the text between two markers.
fn section<'a>(out: &'a str, begin: &str, end: &str) -> &'a str {
    out.split_once(begin)
        .and_then(|(_, rest)| rest.split_once(end))
        .map_or("", |(inner, _)| inner)
}

/// Evaluate one instance against `model_patch` (empty = measure the base state).
///
/// # Errors
/// The `FP_ENV <what>` marker, if the container reported one.
///
/// The marker carries the underlying `cp`/`git apply` stderr after a colon, because a bare
/// "copy-failed" is unactionable: two matplotlib instances failed this way on 2026-09-04 and the
/// reason had already been discarded to `/dev/null`, so the fault could not be told apart from a
/// disk-space problem, a tmpfs mount failure, or a permissions one without re-running by hand.
pub fn parse_env_fault(stdout: &str) -> Option<String> {
    let (_, rest) = stdout.split_once("FP_ENV ")?;
    Some(rest.lines().next().unwrap_or("unknown").trim().to_owned())
}

/// Docker could not run the instance, or the container reported an environment fault (`FP_ENV`).
/// Both mean the instance cannot be scored — the caller excludes and counts it rather than
/// recording a model failure that never happened.
pub fn evaluate(
    instance: &SweInstance,
    model_patch: &str,
    limits: &SweLimits,
) -> Result<SweOutcome, String> {
    let tar = build_input_tar(instance, model_patch)?;

    let mut cmd = Command::new("docker");
    cmd.args(["run", "--rm", "-i"])
        // Published images are x86_64 only; on other hosts this runs under emulation.
        .args(["--platform", "linux/amd64"])
        // ADR 0002 D2, unchanged.
        .args(["--network", "none"])
        .arg("--read-only")
        .args([
            "--tmpfs",
            &format!("/work:rw,exec,size={}m", limits.workdir_mb),
        ])
        .args(["--tmpfs", "/tmp:rw,exec,size=256m"])
        .args(["--memory", &format!("{}m", limits.mem_mb)])
        .args(["--cpus", &format!("{}", limits.cpus)])
        .args(["--pids-limit", "512"])
        .args(["--cap-drop", "ALL"])
        .args(["--security-opt", "no-new-privileges"])
        .arg(&instance.image)
        .args(["sh", "-c", &eval_script()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("docker run failed for {}: {e}", instance.instance_id))?;
    {
        use std::io::Write;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "docker stdin unavailable".to_owned())?;
        stdin
            .write_all(&tar)
            .map_err(|e| format!("writing inputs to {}: {e}", instance.instance_id))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("docker wait failed for {}: {e}", instance.instance_id))?;
    let stdout = String::from_utf8_lossy(&out.stdout);

    if let Some(what) = parse_env_fault(&stdout) {
        return Err(format!(
            "{}: environment fault ({what}) — excluded rather than scored, because an environment \
             that cannot run the tests says nothing about the model",
            instance.instance_id
        ));
    }

    let control = parse_pytest(section(&stdout, "FP_CONTROL_BEGIN", "FP_CONTROL_END"));
    // Raw eval stdout, for when a verdict needs explaining rather than trusting. Off by default.
    if std::env::var("FIRSTPASS_SWE_DUMP_EVAL").is_ok() {
        eprintln!("=== EVAL STDOUT ===\n{stdout}\n=== END ===");
    }
    let p2p_section = section(&stdout, "FP_P2P_BEGIN", "FP_P2P_END");
    let (gate_ids, _) = split_p2p_for_gate(&instance.fail_to_pass, &instance.pass_to_pass);
    let p2p_gate = score_gate_subset(p2p_section, &gate_ids);
    let f2p = parse_pytest(section(&stdout, "FP_F2P_BEGIN", "FP_F2P_END"));
    let p2p = parse_pytest(p2p_section);
    let patch_applied = stdout.contains("FP_PATCH applied");
    // WHICH tolerance level was needed. Recorded because "we made apply more permissive and the
    // score went up" is exactly the claim a reviewer should distrust: if resolutions only appear
    // under the fuzziest fallback, the tolerance is doing suspicious work. Plain applies and
    // --recount applies are unremarkable; a resolution that needs `patch --fuzz` deserves a look.
    let apply_method = classify_apply(&stdout);
    let control_ok = control.1 > 0 && control.0 == control.1;

    Ok(SweOutcome {
        instance_id: instance.instance_id.clone(),
        // The official bar: every FAIL_TO_PASS passes and nothing in PASS_TO_PASS regressed.
        resolved: control_ok && f2p.1 > 0 && f2p.0 == f2p.1 && p2p.1 > 0 && p2p.0 == p2p.1,
        control_ok,
        patch_applied,
        apply_method,
        f2p,
        p2p,
        p2p_gate,
    })
}

/// Build the uncompressed tar delivered on stdin: the two patch files and the two test lists.
/// A tar rather than base64-per-file because a `test_patch` can be large and this keeps one
/// delivery mechanism for every input. PASS_TO_PASS ships three ways: the full list (control),
/// the gate-visible subset, and the oracle-only remainder.
fn build_input_tar(instance: &SweInstance, model_patch: &str) -> Result<Vec<u8>, String> {
    let files: [(&str, String); 4] = [
        ("test.patch", instance.test_patch.clone()),
        ("model.patch", model_patch.to_owned()),
        ("f2p.txt", instance.fail_to_pass.join("\n")),
        ("p2p.txt", instance.pass_to_pass.join("\n")),
    ];
    let mut out = Vec::new();
    for (name, body) in &files {
        out.extend_from_slice(&tar_header(name, body.len())?);
        out.extend_from_slice(body.as_bytes());
        // Records are 512-byte aligned.
        let pad = (512 - body.len() % 512) % 512;
        out.extend(std::iter::repeat_n(0u8, pad));
    }
    // Two zero blocks terminate the archive.
    out.extend(std::iter::repeat_n(0u8, 1024));
    Ok(out)
}

/// A minimal ustar header. Hand-rolled to avoid a tar dependency for four in-memory files.
fn tar_header(name: &str, size: usize) -> Result<[u8; 512], String> {
    if name.len() >= 100 {
        return Err(format!("tar name too long: {name}"));
    }
    let mut h = [0u8; 512];
    h[..name.len()].copy_from_slice(name.as_bytes());
    // mode, uid, gid
    h[100..107].copy_from_slice(b"0000644");
    h[108..115].copy_from_slice(b"0000000");
    h[116..123].copy_from_slice(b"0000000");
    let sz = format!("{size:011o}");
    h[124..135].copy_from_slice(sz.as_bytes());
    h[136..147].copy_from_slice(b"00000000000");
    h[156] = b'0'; // regular file
    h[257..262].copy_from_slice(b"ustar");
    h[263..265].copy_from_slice(b"00");
    // Checksum is computed with the checksum field itself read as spaces.
    h[148..156].copy_from_slice(b"        ");
    let sum: u32 = h.iter().map(|&b| u32::from(b)).sum();
    let cs = format!("{sum:06o}\0 ");
    h[148..156].copy_from_slice(cs.as_bytes());
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pytest_summaries_parse_across_wordings() {
        assert_eq!(parse_pytest("2 passed in 0.12s"), (2, 2));
        assert_eq!(parse_pytest("2 failed in 0.24s"), (0, 2));
        assert_eq!(parse_pytest("13 passed in 0.13s"), (13, 13));
        assert_eq!(parse_pytest("1 failed, 12 passed in 1.02s"), (12, 13));
        assert_eq!(parse_pytest("=== 3 passed, 1 error in 2s ==="), (3, 4));
        // Nothing recognisable ⇒ (0, 0), which the caller reads as "no result", never as success.
        assert_eq!(parse_pytest("collected 0 items"), (0, 0));
    }

    /// `resolved` is a conjunction and every clause has to bite. A run that scores an instance
    /// resolved on a broken environment, or on a partial FAIL_TO_PASS, or while regressing
    /// PASS_TO_PASS, is reporting a number that did not happen.
    #[test]
    fn resolved_requires_the_control_the_fix_and_no_regression() {
        let mk = |control: (usize, usize), f2p: (usize, usize), p2p: (usize, usize)| {
            let control_ok = control.1 > 0 && control.0 == control.1;
            control_ok && f2p.1 > 0 && f2p.0 == f2p.1 && p2p.1 > 0 && p2p.0 == p2p.1
        };
        assert!(mk((13, 13), (2, 2), (13, 13)), "the happy path resolves");
        assert!(
            !mk((11, 13), (2, 2), (13, 13)),
            "broken control cannot resolve"
        );
        assert!(
            !mk((13, 13), (1, 2), (13, 13)),
            "a partial fix is not a fix"
        );
        assert!(!mk((13, 13), (2, 2), (12, 13)), "a regression is not a fix");
        assert!(
            !mk((13, 13), (0, 0), (13, 13)),
            "no F2P result is not a pass"
        );
    }

    /// The tar is hand-rolled, so its checksum has to be right or the container silently gets no
    /// inputs — which would look like a model that never fixes anything.
    #[test]
    fn the_input_tar_is_a_valid_archive() {
        let inst = SweInstance {
            instance_id: "x__y-1".to_owned(),
            repo: "x/y".to_owned(),
            base_commit: "abc".to_owned(),
            problem_statement: "p".to_owned(),
            test_patch: "diff --git a/t b/t\n".to_owned(),
            fail_to_pass: vec!["t::a".to_owned()],
            pass_to_pass: vec!["t::b".to_owned(), "t::c".to_owned()],
            image: "img".to_owned(),
        };
        let tar = build_input_tar(&inst, "diff --git a/m b/m\n").expect("tar");
        assert_eq!(tar.len() % 512, 0, "tar must be 512-byte aligned");
        assert!(tar.ends_with(&[0u8; 1024]), "archive must be terminated");

        // Verify the checksum the way tar does: sum all bytes with the checksum field blanked.
        let mut h = [0u8; 512];
        h.copy_from_slice(&tar[..512]);
        let stored = std::str::from_utf8(&h[148..154])
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim(), 8).ok())
            .expect("checksum parses");
        h[148..156].copy_from_slice(b"        ");
        let actual: u32 = h.iter().map(|&b| u32::from(b)).sum();
        assert_eq!(stored, actual, "ustar checksum must verify");
        assert!(
            tar.starts_with(b"test.patch\0"),
            "first member is the test patch"
        );
    }

    #[test]
    fn an_instance_with_no_fail_to_pass_is_rejected() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("swe-bad-{}.jsonl", std::process::id()));
        let row = serde_json::json!({
            "instance_id": "a__b-1", "repo": "a/b", "base_commit": "c",
            "problem_statement": "p", "test_patch": "d",
            "fail_to_pass": [], "pass_to_pass": ["t"], "image": "i"
        });
        std::fs::write(&p, format!("{row}\n")).expect("write");
        let err = load_swebench_jsonl(p.to_str().expect("utf8")).expect_err("must reject");
        std::fs::remove_file(&p).ok();
        assert!(err.contains("unfalsifiable"), "{err}");
    }
    /// End-to-end oracle check: the REAL gold patch must resolve, an empty patch must not.
    ///
    /// This test was silently dead. It read its dataset from a hardcoded `/tmp/swe-3.jsonl` and
    /// its patch from `/tmp/gold.patch`, both leftovers from a machine state that no longer
    /// exists, so it failed on `.expect()` before asserting anything -- and being `#[ignore]`d,
    /// CI never ran it and nobody noticed. A test that looks like coverage and provides none is
    /// worse than no test.
    ///
    /// Now self-contained: the dataset ships in `docs/benchmarks/` and the gold patch is a
    /// committed fixture. It is the regression guard for the gate/oracle split -- if partitioning
    /// PASS_TO_PASS ever loses or double-counts a test, a known-correct patch stops resolving
    /// and this fails.
    #[test]
    #[ignore = "requires the published SWE-bench image (~1.1GB) and a container daemon"]
    fn real_gold_patch_resolves_and_an_empty_patch_does_not() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root");
        let dataset = root.join("docs/benchmarks/swebench-50-testexec.dataset.jsonl");
        let instances =
            load_swebench_jsonl(&dataset.to_string_lossy()).expect("committed dataset loads");
        let inst = instances
            .iter()
            .find(|i| i.instance_id == "astropy__astropy-12907")
            .expect("astropy instance present");
        let gold = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gold_astropy-12907.patch"),
        )
        .expect("committed gold fixture");

        let fixed = evaluate(inst, &gold, &SweLimits::default()).expect("gold eval runs");
        assert!(
            fixed.control_ok,
            "environment must be sane before blaming a patch"
        );
        assert!(fixed.patch_applied, "gold patch must apply");
        assert!(
            fixed.resolved,
            "the GOLD patch must resolve; if this breaks, the oracle is wrong, not the model \
             (f2p {:?} p2p {:?} gate {:?})",
            fixed.f2p, fixed.p2p, fixed.p2p_gate
        );
        // The split must not lose tests: the gate subset plus the remainder is the whole suite.
        assert_eq!(
            fixed.p2p.1,
            inst.pass_to_pass.len(),
            "oracle must still run every PASS_TO_PASS test after the gate/oracle split"
        );
        assert!(
            fixed.p2p_gate.1 < fixed.p2p.1,
            "the gate must see FEWER tests than the oracle (gate {} of {})",
            fixed.p2p_gate.1,
            fixed.p2p.1
        );

        let empty = evaluate(inst, "", &SweLimits::default()).expect("empty eval runs");
        assert!(!empty.resolved, "an empty patch must never resolve");
    }

    /// The cascade forgives WHERE a change lands, never WHAT it does. This is the guard against
    /// the obvious objection: "you loosened `git apply` until the score improved."
    ///
    /// A patch that applies under fuzz but breaks the code must still fail the oracle, because
    /// `resolved` requires every FAIL_TO_PASS to pass AND every PASS_TO_PASS to still pass.
    /// Tolerance changes the denominator of "patches evaluated", never the numerator of
    /// "patches correct".
    #[test]
    #[ignore = "requires the published SWE-bench image and a container daemon"]
    fn a_fuzzily_applied_but_wrong_patch_still_fails_the_oracle() {
        let path = std::env::var("FIRSTPASS_SWE_DATASET").unwrap_or_default();
        let Ok(instances) = load_swebench_jsonl(&path) else {
            eprintln!("set FIRSTPASS_SWE_DATASET to run this");
            return;
        };
        let Some(inst) = instances
            .iter()
            .find(|i| i.instance_id == "astropy__astropy-12907")
        else {
            return;
        };
        // Deliberately sloppy line numbers (so strict apply refuses and the cascade engages)
        // wrapped around a change that is real but WRONG.
        let wrong = "--- a/astropy/modeling/separable.py\n\
                     +++ b/astropy/modeling/separable.py\n\
                     @@ -1,3 +1,4 @@\n\
                      # Licensed under a 3-clause BSD style license - see LICENSE.rst\n\
                     +BROKEN_SENTINEL = 1 / 0\n\
                      \n";
        let out = evaluate(inst, wrong, &SweLimits::default()).expect("eval runs");
        // NOT VACUOUS: the patch must actually have been APPLIED, or this proves nothing about
        // tolerance -- a rejected patch fails the oracle trivially and would let the test pass
        // while testing nothing.
        assert!(
            out.patch_applied,
            "the cascade must have applied this patch for the test to mean anything (method: {})",
            out.apply_method
        );
        assert_ne!(
            out.apply_method, "clean",
            "sloppy line numbers should have needed a fallback, not applied cleanly"
        );
        assert!(
            !out.resolved,
            "a patch that injects a ZeroDivisionError must never resolve, however it applied \
             (method: {})",
            out.apply_method
        );
    }

    /// Every bucket, against the REAL classifier. The first version of this test re-implemented
    /// the branching inline, so it would have passed even if the shipped code were broken.
    #[test]
    fn every_apply_outcome_maps_to_its_own_bucket() {
        for (marker, want) in [
            ("FP_PATCH empty", "no-patch"),
            ("FP_PATCH rejected", "rejected"),
            ("FP_PATCH applied", "clean"),
            ("FP_PATCH applied recount", "recount"),
            ("FP_PATCH applied c1", "c1"),
            ("FP_PATCH applied fuzz", "fuzz"),
        ] {
            let out = format!("FP_CONTROL_BEGIN\nFP_CONTROL_END\n{marker}\nFP_F2P_BEGIN");
            assert_eq!(
                classify_apply(&out),
                want,
                "marker {marker:?} misclassified"
            );
        }
        // The ordering trap: "applied fuzz" CONTAINS "applied", so a most-specific-last check
        // would call every fallback `clean` and hide exactly the case worth auditing.
        assert_eq!(classify_apply("FP_PATCH applied fuzz"), "fuzz");
    }

    /// The gate must stay STRICTLY WEAKER than the oracle. If the split ever hands the gate every
    /// PASS_TO_PASS test, gate_pass becomes oracle_correct by construction, every gate-error
    /// measurement in this repo silently reads 100% precision, and the benchmark stops being able
    /// to catch its own verifier being wrong. That is the failure this test exists to prevent.
    #[test]
    fn the_gate_never_sees_the_whole_oracle_suite() {
        let f2p = vec!["tests/test_wcs.py::test_sip".to_owned()];
        let p2p = vec![
            "tests/test_wcs.py::test_a".to_owned(),   // same file -> gate
            "tests/test_wcs.py::test_b".to_owned(),   // same file -> gate
            "tests/test_io.py::test_c".to_owned(),    // elsewhere -> oracle only
            "tests/test_table.py::test_d".to_owned(), // elsewhere -> oracle only
        ];
        let (gate, rest) = split_p2p_for_gate(&f2p, &p2p);
        // Both siblings, plus reach BEYOND the changed file -- siblings alone measured 0/256
        // precision on django, whose regression surface is not file-local.
        assert!(
            gate.contains(&"tests/test_wcs.py::test_a".to_owned())
                && gate.contains(&"tests/test_wcs.py::test_b".to_owned()),
            "gate must take the siblings: {gate:?}"
        );
        assert!(
            gate.iter().any(|t| !t.starts_with("tests/test_wcs.py")),
            "gate must also reach beyond the changed file: {gate:?}"
        );
        assert!(
            gate.len() < p2p.len(),
            "gate must be a STRICT subset of the oracle suite: {gate:?}"
        );
        assert!(
            !rest.is_empty(),
            "the oracle MUST retain tests the gate cannot see"
        );
        assert_eq!(
            gate.len() + rest.len(),
            p2p.len(),
            "the split must lose nothing"
        );
        for t in &gate {
            assert!(!rest.contains(t), "buckets must be disjoint");
        }
    }

    /// A fix in a file with no sibling regression tests must not leave the gate blind again —
    /// that would silently restore the 3.7%-precision gate for exactly the instances where the
    /// change is most isolated.
    #[test]
    fn a_file_with_no_sibling_tests_still_gets_a_gate_subset() {
        let f2p = vec!["tests/test_lonely.py::test_x".to_owned()];
        let p2p: Vec<String> = (0..25)
            .map(|i| format!("tests/test_other.py::t{i}"))
            .collect();
        let (gate, rest) = split_p2p_for_gate(&f2p, &p2p);
        assert!(
            !gate.is_empty(),
            "fallback must give the gate something to check"
        );
        assert!(
            !rest.is_empty(),
            "and must still reserve tests for the oracle"
        );
        assert_eq!(
            gate.len() + rest.len(),
            p2p.len(),
            "the split must lose nothing"
        );
    }

    /// Deterministic: an auditor re-running the split gets byte-identical buckets.
    #[test]
    fn the_split_is_deterministic() {
        let f2p = vec!["a/t.py::x".to_owned()];
        let p2p: Vec<String> = (0..40).map(|i| format!("b/u.py::t{i}")).collect();
        let first = split_p2p_for_gate(&f2p, &p2p);
        for _ in 0..5 {
            assert_eq!(split_p2p_for_gate(&f2p, &p2p), first, "split must not vary");
        }
    }

    /// Scoring the gate from the shared run's per-test lines, including the case that caused the
    /// bug: a test the gate cannot see must not affect the gate's verdict.
    #[test]
    fn the_gate_scores_only_its_own_tests() {
        let out = "\
tests/test_wcs.py::test_a PASSED\n\
tests/test_wcs.py::test_b FAILED\n\
tests/test_io.py::test_c FAILED\n\
tests/test_io.py::test_d PASSED\n\
tests/test_wcs.py::test_e SKIPPED\n\
2 passed, 2 failed in 1.0s\n";
        let gate = vec![
            "tests/test_wcs.py::test_a".to_owned(),
            "tests/test_wcs.py::test_b".to_owned(),
            "tests/test_wcs.py::test_e".to_owned(),
        ];
        // 1 of 2 scored: test_e is SKIPPED and counts for neither, and the two test_io failures
        // belong to the oracle alone -- if they leaked in, the gate would inherit the oracle's
        // strictness and stop being independently measurable.
        assert_eq!(score_gate_subset(out, &gate), (1, 2));
    }

    /// The django failure, in miniature. A fix in one module with regression tests spread across
    /// the codebase: the sibling-only gate saw 2 of 30 tests and certified 256 django patches
    /// with zero correct. The gate must now reach modules the change could break.
    #[test]
    fn the_gate_reaches_beyond_the_changed_module() {
        let f2p = vec!["tests/forms/test_fields.py::test_x".to_owned()];
        let mut p2p = vec![
            "tests/forms/test_fields.py::sib1".to_owned(),
            "tests/forms/test_fields.py::sib2".to_owned(),
        ];
        // 28 regression tests living far from the change, as django's do.
        p2p.extend((0..28).map(|i| format!("tests/model_fields/test_m{i}.py::t")));

        let (gate, rest) = split_p2p_for_gate(&f2p, &p2p);
        let far_in_gate = gate.iter().filter(|t| t.contains("model_fields")).count();
        assert!(
            far_in_gate >= 5,
            "gate must cover distant modules, saw {far_in_gate} of 28: {gate:?}"
        );
        // ...and must still be strictly weaker than the oracle, or its error stops being
        // measurable.
        assert!(
            !rest.is_empty(),
            "oracle must keep tests the gate cannot see"
        );
        assert!(gate.len() < p2p.len(), "gate must remain a strict subset");
        assert_eq!(
            gate.len() + rest.len(),
            p2p.len(),
            "split must lose nothing"
        );
    }
}

#[cfg(test)]
mod env_fault_tests {
    use super::parse_env_fault;

    #[test]
    fn carries_the_underlying_reason_not_just_the_bucket() {
        let out = "some pytest noise\nFP_ENV copy-failed: cp: cannot create directory: No space left on device\nmore\n";
        let what = parse_env_fault(out).expect("marker present");
        // The bucket alone was what made the 2026-09-04 matplotlib faults undiagnosable.
        assert!(what.contains("No space left on device"), "{what}");
    }

    #[test]
    fn absent_marker_is_none_so_a_clean_run_is_never_a_fault() {
        assert_eq!(parse_env_fault("FP_CONTROL_BEGIN\n1 passed\n"), None);
    }
}
