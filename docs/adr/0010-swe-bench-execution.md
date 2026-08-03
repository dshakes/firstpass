# ADR 0010 — Running SWE-bench without quietly dismantling the sandbox

- Status: **Accepted** (design; implementation deferred behind BigCodeBench results)
- Date: 2026-08-03, revised same day — see *"D1, corrected"*: the premise that SWE-bench
  requires a writable rootfs turned out to be false when tested, so no isolation
  invariant is relaxed after all.
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

### D1, corrected — seed the tmpfs workdir from the image; change no invariant

The first draft of this ADR proposed a `RepoSandbox` with a **writable rootfs**,
on the reasoning that patching and building a repository cannot happen on a
read-only filesystem. That reasoning was never tested. When tested, it is wrong.

The repository does not have to be modified *where the image put it*. It can be
copied into the existing `tmpfs` workdir — which is already writable, already
size-capped, and already discarded with the container — and patched, built, and
tested there. Verified directly:

```
docker run --rm --network none --read-only --tmpfs /work:rw,size=256m \
           --user 65534:65534 --cap-drop ALL python:3.12-alpine sh -c '
  cp -r "$SRC" /work/repo
  echo "# patch applied" >> /work/repo/__init__.py   # COPY+PATCH: ok
  python3 -c "import sys; sys.path.insert(0,\"/work\"); import repo"  # IMPORT FROM COPY: ok
  touch "$SRC/evil"                                   # ROOTFS STILL READ-ONLY: ok
'
```

So the decision is the opposite of the one first drafted: **no new sandbox type,
no weaker tier, and every ADR 0002 D2 invariant holds unchanged.** What
`ContainerSandbox` gains is one capability — seeding the workdir from a path
inside the image before the command runs (`seed_from: Option<String>`) — which is
strictly less power than it already has, since the workdir is writable either way.

Two limits worth stating rather than discovering later:

- **The workdir is RAM.** `Limits::workdir_mb` must cover the repository; large
  instances need it raised, and a repo that does not fit is an abort, not a
  silent truncation.
- **Import precedence must favour the copy.** An image with the package installed
  editable against the original path could shadow the patched copy, and the
  failure mode is a run that silently scores the *unpatched* code. The
  `PASS_TO_PASS` preflight in D3 is what catches that: it is run after the copy,
  so a mis-wired import shows up as a broken control before any result is
  attributed to a model.

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

- **ADR 0002 D2 is untouched.** No writable rootfs, no second threat model, no
  weaker tier to keep out of the hosted plane later. The thing this ADR was
  opened to authorise turned out not to need authorising.
- `Limits` grows no `writable` flag, so no caller can drift into a weaker mode by
  passing a bool.
- A run still needs real operator resources: per-instance images (gigabytes), a
  provisioning step with a network, and per-instance model spend. None of that is
  hidden inside the benchmark.
- The workdir is RAM, so instance size becomes a resource limit to set rather
  than an isolation property to trade away.

## What the BigCodeBench pilot said first

Before spending anything on SWE-bench, `--coding-policy` ran the same question on
24 stdlib-only BigCodeBench tasks (haiku → sonnet, 48 live calls, cents). The
answer was **not** the one the roadmap assumed:

| policy | success | $/success |
|---|---|---|
| always-cheap | 0.88 | $0.0032 |
| always-top | 0.92 | $0.0059 |
| first-pass | 0.88 | $0.0041 |

First-pass tied always-cheap on quality and cost **more** than it. The gate
escalated, and those escalations converted nothing — it was spending money to
re-derive answers the cheap rung had already gotten right, while waving through
the ones it got wrong. Against always-top it is cheaper but scored lower, with
overlapping intervals at this n.

That is a finding about the *gate*, not about the ladder: on this workload a
visible-test prefix does not separate haiku's good answers from its bad ones. So
the obvious follow-up ran too — the same 24 tasks with a judge as a second
opinion, serving the cheap rung only when the tests pass **and** the judge is
confident:

| policy | success | total $ (see note) | escalated | converted |
|---|---|---|---|---|
| always-cheap | 0.83 | $0.0614 | 0% | — |
| always-top | 0.92 | $0.1282 | 0% | — |
| first-pass | 0.83 | $0.0786 | 8% | 0% |
| first-pass+judge | 0.88 | $0.1151 | 33% | 12% |

The judge works as a *signal*: quality moved 0.83 → 0.88, escalation rose 8% →
33%, and 12% of those escalations converted a wrong answer into a right one. It
sees what the tests cannot.

It does not work as *economics*. That $0.1151 omits the judge's own 48 calls,
which are not yet metered. At haiku's published rate those add roughly $0.07,
which would put the arm near **$0.19 against always-top's $0.128** — about 45%
more expensive, at lower quality. That is an estimate rather than a measurement,
which is exactly why the harness now withholds a $/success it cannot account for.

**The likely root cause is the workload, not the design.** haiku already solves
0.83 of these tasks and sonnet 0.92 — nine points of headroom for any amount of
verification to recover. Cascade economics need a real capability gap between
rungs; when the cheap model is nearly as good as the expensive one, no gate can
pay for itself, because there is almost nothing to catch. The stdlib-only subset
selected for cheap infrastructure is also, unavoidably, the easier end of
BigCodeBench.

So the next experiment is neither a better gate nor SWE-bench: it is **harder
tasks, or a wider ladder**. The full BigCodeBench (third-party libs, needs a
dependency-carrying image), or a genuinely weak rung 0. If the gap between rungs
stays this narrow, the honest conclusion is that this workload does not need a
router — and that is worth knowing for a fifth of a dollar.

## Status of the work

Design accepted; implementation deliberately **not** started, and the pilot above
is now the reason rather than the sandbox question. SWE-bench scores an agent
scaffold; what we need answered is narrower — *is the routing policy better?* —
and `--coding-policy` answers that for cents by holding the scaffold fixed and
varying only the policy. Paying twenty times as much for a benchmark whose name
is more famous, while the cheap version of the same question is still returning
"the gate did not earn its cost", would be buying recognition instead of
evidence.

The one thing this ADR changes today: it removes "we would have to weaken the
sandbox" from the list of reasons not to.
