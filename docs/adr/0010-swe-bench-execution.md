# ADR 0010 — Running SWE-bench without quietly dismantling the sandbox

- Status: Proposed — design only; no code written against it yet
- Date: 2026-08-03
- Related: ADR 0002 (bench code-execution sandbox, **D2 isolation invariants**),
  `crates/firstpass-bench/src/sandbox.rs`, `crates/firstpass-bench/src/coding.rs`,
  `crates/firstpass-bench/src/dataset.rs` (BigCodeBench loader)

## Context

BigCodeBench now runs on the existing harness: a task is self-contained Python,
the candidate writes one file, and a `unittest` suite scores it inside the
fail-closed sandbox. SWE-bench is the other benchmark worth quoting, and it does
not fit that shape at all.

A SWE-bench instance is a **repository at a commit**, not a function. Evaluating
one means: check out `repo@base_commit`, apply the model's patch, apply the
benchmark's `test_patch`, run the named tests, and check that every test in
`FAIL_TO_PASS` now passes while every test in `PASS_TO_PASS` still does. The
official harness does this inside a per-instance image that already contains the
repository and its installed dependencies.

That collides head-on with ADR 0002 **D2**, whose invariants are labelled
"never simplified away":

| D2 invariant | What SWE-bench needs |
|---|---|
| Rootfs read-only, tmpfs `/work`, no host mounts | A **writable** repo tree — `git apply`, `pip install -e .`, `.pyc` writes, test artifacts |
| `--network none` | Nothing at eval time *if* the image is pre-provisioned; a lot if it is not |
| Files streamed in via stdin, never mounted | The repo is already **in the image**, at a path (`/testbed`) the runner must `cd` into |

The tempting move is to add a `writable: bool` to `Limits` and move on. That is
the failure this ADR exists to prevent: it would relax an invariant the whole
benchmark's credibility rests on, in a struct field, with no record of why —
and the sandbox exists specifically because we run **untrusted, model-generated
code**. A SWE-bench patch is exactly that, and it arrives with a whole
repository's build system attached.

The second, quieter risk is measurement validity. This repo has already learned
that an environment failure scored as a test failure fabricates gate error out of
nothing (`FP_ENVERR`, BigCodeBench). SWE-bench multiplies that: a missing
dependency, an unbuilt extension, or a wrong Python version makes a *correct*
patch look wrong, and 500 instances of that produce a confident, meaningless
number.

## Decision

### D1 — A separate `Sandbox` implementation, not a flag on the existing one

Add `RepoSandbox` alongside `ContainerSandbox`, behind the same `Sandbox` seam.
It is a different threat model with a different, explicitly weaker ceiling, and
it should be a different type so that no existing caller can silently acquire it.
`ContainerSandbox`'s invariants stay exactly as ADR 0002 wrote them.

`RepoSandbox` runs a per-instance image with a **writable container filesystem**
and keeps every other control:

- `--network none` at eval time (see D2)
- no host bind-mounts, ever — the repo comes from the image, not from the host
- `--rm`, `kill_on_drop`, wall-clock watchdog, cpu/mem/pids caps
- `--cap-drop ALL`, `--security-opt no-new-privileges`
- non-root where the image permits it (many SWE-bench images assume root; when
  it does not permit it, that is recorded in the run report rather than waived
  silently)

The honest statement of the ceiling: **container-grade isolation with a writable
rootfs.** That is weaker than ADR 0002's dev-time tier. It is acceptable only for
a single-operator, dev-time benchmark run, and it must never become the sandbox
that a hosted plane runs customer gates in (ADR 0001 §D3 remains untouched).
`RepoSandbox::runtime()` reports the weakened tier so it appears in the report.

### D2 — Network stays off at eval time; provisioning is a separate, explicit step

Dependency installation is the only thing that genuinely wants a network, and it
must not happen in the same step as running untrusted code. Images are pulled and
provisioned **before** the run, by digest, in a step that has a network and runs
no model output. Eval then runs `--network none`, exactly like every other
sandbox in this repo.

If an instance's image is not present, the run **aborts** naming the instance. It
does not fall back to "install it now, with the patch already applied".

### D3 — An unusable environment is an abort, not a failed test

Inherited directly from the BigCodeBench `FP_ENVERR` rule and generalised:

- `PASS_TO_PASS` is a **pre-flight control**, not only a post-condition. Before
  the model patch is applied, those tests must already pass on the base commit.
  If they do not, the environment is broken and the instance is excluded, loudly,
  with the count reported.
- Import errors, collection errors, and build failures are classified as
  environment outcomes, not as the patch being wrong.
- Excluded instances are **reported as a count in the result**, never dropped
  silently. A number computed over an unstated subset is the thing this ADR is
  most concerned with preventing.

### D4 — What SWE-bench actually measures here

SWE-bench scores an *agent*, and Firstpass is not an agent — it is what an agent
routes through. So the comparison it can honestly make is the same one the rest
of the bench makes, held at fixed capability:

- fix the scaffold (one patch-generating loop), vary only the **routing policy**
  underneath it: always-cheap, always-top, and the Firstpass ladder with gates;
- report resolve-rate and **$/resolved-instance**, with bootstrap CIs, plus the
  served-failure rate the gate let through;
- the gate here is real and imperfect in the way the conformal bound needs — a
  patch can pass the visible tests it was given and still fail `PASS_TO_PASS`.

We do **not** publish a bare "Firstpass scores X% on SWE-bench". That number
would be a property of the scaffold, not of the router, and claiming it would be
the kind of borrowed credit this project's positioning explicitly refuses.

### D5 — Subset first, and say it

Start on SWE-bench **Verified** (500 human-validated instances), and run a
pre-registered subset before committing to a full sweep — the BigCodeBench pilot
already demonstrated that a small pilot catches a degenerate setup before the
expensive run. Every published figure names the split, the instance count, the
scaffold, and the model ladder. A subset is honest; an unlabelled subset is not.

## Consequences

- ADR 0002's invariants remain literally true for `ContainerSandbox`; the weaker
  tier is a named, separate type whose report says so.
- A run needs real operator resources: per-instance images (gigabytes), a
  provisioning step with a network, and per-instance model spend. None of that
  can be hidden inside the benchmark.
- `Limits` grows no `writable` flag, so no existing caller can drift into the
  weaker mode by passing a bool.
- If we later want kernel-grade isolation for this tier, it slots in behind the
  same `Sandbox` seam (gVisor/microVM), exactly as ADR 0002 planned.

## Status of the work

Design only. Nothing here is implemented. The BigCodeBench path shipped first on
purpose: it needs no new isolation posture, and it is the one that can produce a
non-degenerate conformal bound, which is the open scientific gap.
