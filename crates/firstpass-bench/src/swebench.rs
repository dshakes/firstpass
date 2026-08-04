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
    /// `(passed, total)` for `FAIL_TO_PASS` after the candidate patch.
    pub f2p: (usize, usize),
    /// `(passed, total)` for `PASS_TO_PASS` after the candidate patch.
    pub p2p: (usize, usize),
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
cp -a /testbed /work/repo 2>/dev/null || { echo "FP_ENV copy-failed"; exit 0; }
cd /work/repo
. /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate testbed 2>/dev/null
# The copy must win over the editable install that points at /testbed, or the run scores
# unpatched code and every number it produces is meaningless.
export PYTHONPATH=/work/repo
run() { python -m pytest -q -p no:cacheprovider --no-header $(tr '\n' ' ' < "$1") 2>&1 | tail -3; }

git apply /work/in/test.patch 2>/dev/null || { echo "FP_ENV test-patch-failed"; exit 0; }

# CONTROL: PASS_TO_PASS must already pass on the base commit. If not, this environment cannot
# measure anything and the instance is excluded rather than blamed on the model.
echo "FP_CONTROL_BEGIN"; run /work/in/p2p.txt; echo "FP_CONTROL_END"

if [ -s /work/in/model.patch ]; then
  git apply /work/in/model.patch 2>/dev/null && echo "FP_PATCH applied" || echo "FP_PATCH rejected"
else
  echo "FP_PATCH empty"
fi

echo "FP_F2P_BEGIN"; run /work/in/f2p.txt; echo "FP_F2P_END"
echo "FP_P2P_BEGIN"; run /work/in/p2p.txt; echo "FP_P2P_END"
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

/// Pull the text between two markers.
fn section<'a>(out: &'a str, begin: &str, end: &str) -> &'a str {
    out.split_once(begin)
        .and_then(|(_, rest)| rest.split_once(end))
        .map_or("", |(inner, _)| inner)
}

/// Evaluate one instance against `model_patch` (empty = measure the base state).
///
/// # Errors
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

    if let Some(rest) = stdout.split_once("FP_ENV ") {
        let what = rest.1.lines().next().unwrap_or("unknown").trim();
        return Err(format!(
            "{}: environment fault ({what}) — excluded rather than scored, because an environment \
             that cannot run the tests says nothing about the model",
            instance.instance_id
        ));
    }

    let control = parse_pytest(section(&stdout, "FP_CONTROL_BEGIN", "FP_CONTROL_END"));
    let f2p = parse_pytest(section(&stdout, "FP_F2P_BEGIN", "FP_F2P_END"));
    let p2p = parse_pytest(section(&stdout, "FP_P2P_BEGIN", "FP_P2P_END"));
    let patch_applied = stdout.contains("FP_PATCH applied");
    let control_ok = control.1 > 0 && control.0 == control.1;

    Ok(SweOutcome {
        instance_id: instance.instance_id.clone(),
        // The official bar: every FAIL_TO_PASS passes and nothing in PASS_TO_PASS regressed.
        resolved: control_ok && f2p.1 > 0 && f2p.0 == f2p.1 && p2p.1 > 0 && p2p.0 == p2p.1,
        control_ok,
        patch_applied,
        f2p,
        p2p,
    })
}

/// Build the uncompressed tar delivered on stdin: the two patch files and the two test lists.
/// A tar rather than base64-per-file because a `test_patch` can be large and this keeps one
/// delivery mechanism for all four inputs.
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

    /// The real thing, against a real published image. Reproduces by code the hand-run that
    /// validated ADR 0010: gold patch resolves, empty patch does not, and the control holds in
    /// both. Ignored by default — it needs ~1.1GB of image.
    ///
    ///   docker pull --platform linux/amd64 swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
    ///   cargo test -p firstpass-bench --lib swebench::tests::real_ -- --ignored --nocapture
    #[test]
    #[ignore = "requires the published SWE-bench image (~1.1GB) and a container daemon"]
    fn real_gold_patch_resolves_and_an_empty_patch_does_not() {
        let path = std::env::var("FIRSTPASS_SWE_DATASET")
            .unwrap_or_else(|_| "/tmp/swe-3.jsonl".to_owned());
        let instances = load_swebench_jsonl(&path).expect("dataset");
        let inst = instances
            .iter()
            .find(|i| i.instance_id == "astropy__astropy-12907")
            .expect("astropy instance present");
        let gold = std::fs::read_to_string("/tmp/gold.patch").expect("gold patch");
        let limits = SweLimits::default();

        // No patch: the control must hold and FAIL_TO_PASS must still fail. If this "resolves",
        // the harness is measuring unpatched code — the exact trap the editable install sets.
        let base = evaluate(inst, "", &limits).expect("base run");
        assert!(base.control_ok, "PASS_TO_PASS must pass on the base commit");
        assert!(!base.resolved, "the bug must be present before the fix");
        assert!(
            base.f2p.0 < base.f2p.1,
            "F2P should fail at base: {:?}",
            base.f2p
        );

        // Gold patch: resolves.
        let fixed = evaluate(inst, &gold, &limits).expect("gold run");
        assert!(fixed.patch_applied, "gold patch must apply");
        assert!(
            fixed.resolved,
            "gold must resolve the instance; f2p={:?} p2p={:?}",
            fixed.f2p, fixed.p2p
        );
    }
}
