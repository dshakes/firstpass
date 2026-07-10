# ADR 0002 — Bench code-execution sandbox (coding-with-tests benchmark)

- **Status:** Proposed (2026-07-10)
- **Blocks:** Batch 3b — the coding-with-tests benchmark (`firstpass-bench`). No candidate code is
  executed until the sandbox in this ADR is built and its fail-closed preflight is verified.
- **Relates:** ADR 0001 §D3 (hosted-plane *customer-gate* sandbox — a **different, harder** boundary,
  see *Relationship to D3* below); SPEC §10 (proof harness), §12 (trust model).

## Context

The conformal served-failure guarantee is the one differentiator Firstpass has **not** been able to
earn on real traffic. Proven definitively (3 live experiments, see memory / build log): arithmetic
tasks are *self-checking* — the best practical gate is a deterministic checker with **zero** error,
so there is no false-pass/false-fail rate to bound, and conformal degenerates. To make the guarantee
*meaningful* we need a domain where the best practical gate is **still imperfect**. That domain is
**coding-with-tests**: a candidate model writes code; a **visible** test suite is the gate (real
suites have coverage gaps → genuine fpr/fnr); a **hidden oracle** suite is ground truth. Conformal
calibrates on `(gate_pass, oracle_correct)` with *real* error → the bound becomes real.

Running that benchmark means **executing code the model wrote**. That is untrusted code — the model
can emit `os.system("rm -rf ~")`, open a socket and exfiltrate, fork-bomb, read `~/.aws`, or loop
forever — by accident or because a task/prompt was adversarial. Today's `SubprocessGate` spawns a
bare host process with no isolation. That is **acceptable for a trusted operator-authored gate; it is
not acceptable for model-generated candidate code.** Per our rules, executing untrusted code is a new
trust boundary and gets designed before it is built. This ADR is that design.

## Threat model (scope it honestly — do not over- or under-build)

- **Who runs it:** the Firstpass operator, on their own dev machine or CI, at benchmark time. Single
  trust domain. **Not** multi-tenant, **not** on the request hot path, **not** hosted infra.
- **What's untrusted:** the *candidate program* and the *task fixtures* if sourced externally
  (HumanEval/MBPP are community datasets). The test harness and oracle we author are trusted.
- **Threats in scope:** (1) destructive host FS access (delete/modify operator files); (2) network
  exfiltration or callback; (3) resource exhaustion (CPU spin, memory bomb, fork bomb, disk fill);
  (4) reading host secrets (env, `~/.ssh`, `~/.aws`, the operator's own `ANTHROPIC_API_KEY`);
  (5) escaping the workdir to touch the repo/toolchain.
- **Explicitly out of scope (documented, not silently ignored):** kernel-level container escape by a
  determined attacker, and cross-tenant isolation. Those are the hosted-plane D3 problem. A dev-time
  benchmark accepts container-grade (not microVM-grade) isolation as its ceiling — see *D4* and
  *Relationship to D3*. **This ceiling is a deliberate decision, revisited if the harness ever runs
  untrusted third-party code on shared infra.**

## Decisions

### D1 — Mechanism: strongest-available OCI runtime (microVM / gVisor ideal), auto-selected, never `runc`-by-default-silently

The bar is *running untrusted, model-generated code* — the same problem E2B, Modal, Fly, and Daytona
solve for AI code execution, and they all converge on **microVM-class isolation**, not shared-kernel
containers. So the ideal is not "a Docker container"; it is **hardware-virtualized or
kernel-intercepting isolation**, with plain containers as an explicitly-weaker fallback.

Tiers, strongest first (all behind the D4 `Sandbox` seam, so the runtime is a swap, not a rewrite):
- **(ideal) microVM — Firecracker / Cloud Hypervisor.** Hardware-virtualized, separate guest kernel;
  the industry standard for untrusted AI-generated code (Lambda, E2B, Modal). Strongest isolation.
  Heavier ops (KVM, a guest rootfs/kernel image, vsock I/O) — a dedicated `Sandbox` impl, built when
  the benchmark runs on Linux+KVM infra. **The target ceiling.**
- **(strong, drop-in) gVisor — `runsc`.** A userspace guest kernel that intercepts syscalls; near-
  microVM isolation with **zero** image changes (it's just an OCI `--runtime`), and it runs CPython +
  arbitrary pip packages. **The tier we implement and ship first** — ideal-grade isolation that is
  actually buildable and verifiable today.
- **(fallback, loud) plain OCI `runc` (Docker/Podman)** — namespace isolation, shared host kernel.
  Used **only** when no stronger runtime is present, and then **only with a printed warning** naming
  the weaker boundary. Never the silent default.
- **(rejected) bare subprocess + `ulimit`** — resource caps are not a sandbox (still `rm`s, reads
  secrets, opens sockets). **(rejected for this benchmark) WASI/wasmtime** — great for *pure-compute*
  gates, but CPython + arbitrary test-suite packages under WASI is infeasible today; kept as the D3
  path for pure-compute gates.

**Decision: implement one OCI runner that auto-selects the strongest available runtime** — probe for
`runsc` (gVisor) first and use it; else fall back to `runc` **with a warning**; and leave the
`Sandbox` trait ready for a Firecracker microVM impl as the ideal tier. Same flags across runtimes
(`--network none`, no host mounts, ro rootfs + tmpfs workdir, cpu/mem/pids caps, non-root, cap-drop).
The chosen runtime is recorded in the trace/report, so a published number always states its isolation
tier. This delivers ideal-grade isolation where the infra supports it, degrades **visibly** (never
silently) where it doesn't, and makes the microVM upgrade a new impl behind an unchanged seam.

### D2 — Non-negotiable isolation invariants (never simplified away)

Every candidate execution MUST:
1. **No network.** `--network none`. Not "restricted" — none.
2. **No host filesystem.** No bind mounts of host paths. Code + tests are written into a `tmpfs`
   workdir *inside* the container (or copied in via `docker cp` to an ephemeral container), never a
   mounted host directory. Rootfs read-only.
3. **Resource caps.** CPU (`--cpus`), memory (`--memory`, `--memory-swap` equal to disable swap),
   process count (`--pids-limit`, kills fork bombs), and workdir size (tmpfs `size=`).
4. **Wall-clock timeout with kill.** A hard timeout; on expiry the container is force-removed. Maps
   to a distinct `TimedOut` outcome (a hung candidate is a gate signal, not a crash).
5. **Ephemeral + `kill_on_drop`.** `--rm`; the Rust handle force-removes the container if the future
   is dropped (mirrors the existing `SubprocessGate` / speculation `kill_on_drop` discipline).
6. **Non-root, no added capabilities.** `--user` a non-root uid, `--cap-drop ALL`, `--security-opt
   no-new-privileges`. No secrets in the container env (the operator's provider keys never enter it).

### D3 — Fail **closed**: no sandbox ⇒ no execution, ever

If the container runtime is missing, the image is unavailable, or the sandbox preflight fails, the
benchmark **aborts with a clear error**. It **must never** fall back to running candidate code on the
host. This is the single most important rule in this ADR: a silent fallback turns a proof harness into
a way to run model-authored `rm -rf` on the operator's machine. A `sandbox_selfcheck()` runs once at
startup (e.g. execute a candidate that tries to open a socket and touch `/etc` → assert it is denied),
and the benchmark refuses to proceed if the selfcheck does not demonstrate isolation.

### D4 — Interface: a small `Sandbox` seam, one outcome type

```
trait Sandbox {
    /// Run `program` against `tests` under `limits`; never returns host-side effects.
    async fn run(&self, unit: &ExecUnit, limits: &Limits) -> Result<ExecOutcome, SandboxError>;
}

struct Limits { cpus: f32, mem_mb: u64, pids: u32, wall_ms: u64, workdir_mb: u64 }

enum ExecOutcome {
    Completed { exit_code: i32, stdout: String, stderr: String }, // gate reads pass/fail from this
    TimedOut,                                                      // distinct: a hung candidate
    Killed(String),                                               // OOM / pids / runtime-killed
}

enum SandboxError { Unavailable(String), Setup(String) }          // Unavailable ⇒ ABORT (D3)
```

The trait is the seam (mirrors bench's existing backend/gate traits): the real impl is the container
runner; an offline `#[cfg(test)]` fake lets us unit-test the harness/gate wiring without Docker in CI.
The **coding gate** is then just "run the visible tests in the sandbox, pass iff exit 0"; the **oracle
ground truth** is "run the hidden tests in the sandbox, correct iff exit 0". Same mechanism, two suites.

### D5 — Determinism & honesty of the resulting numbers

- Pin the base image by digest; pin task datasets by checksum. A benchmark whose environment drifts
  can't back a published fpr/fnr.
- `TimedOut`/`Killed` are **gate fails**, not dropped samples — silently discarding non-terminating
  candidates would bias the error estimate. They count.
- Report the sandbox config (image digest, limits) alongside the numbers, like the live-run caveats.

## Relationship to ADR 0001 §D3 (why this is a *separate*, lighter ADR)

D3 governs the **hosted control plane** running **customer** gate code across **tenants** — it demands
microVM/gVisor isolation and external review, and it is Phase 3, gated behind a proven data plane.
**This ADR is not that.** It is a **dev-time, single-operator** harness that runs **our own
model-generated** candidate code to produce a research number. Shared DNA — *no network, capped,
ephemeral, kill_on_drop, fail-closed* — so the container tier here is a faithful stepping stone toward
D3's general tier, and the `Sandbox` trait is where a future Firecracker/gVisor impl slots in. But the
**stakes and isolation ceiling differ** (container-grade here vs kernel-grade there), and conflating
them would either over-build the benchmark or under-protect the hosted plane. They stay distinct.

## Invariants — must never regress (blockers for 3b)

1. Candidate code runs **only** inside the sandbox — grep-able: there is exactly one execution path,
   and it is the `Sandbox` impl. No `Command::new` on candidate code anywhere else.
2. `--network none` and no host bind-mounts on every run — asserted by `sandbox_selfcheck()`.
3. Runtime unavailable ⇒ **abort**, never host-exec (D3). A test forces `Unavailable` and asserts the
   harness errors out rather than falling back.
4. `TimedOut`/`Killed` are counted as gate fails, never dropped.

## Phasing

- **3a (this ADR → then build):** the `Sandbox` trait, the container runner, `sandbox_selfcheck()`,
  the offline fake, and the fail-closed preflight. Ships behind a `--sandbox` / cfg so nothing runs
  candidate code until it exists.
- **3b:** the coding-with-tests benchmark on top — task loader (HumanEval/MBPP or `claude -p`-
  generated, checksum-pinned), visible-tests gate, hidden-oracle ground truth, wired into the existing
  `firstpass-bench` conformal path (reuse `conformal.rs`; needs enough clear served samples — α=10%
  wants ~149 served with 0 fails — so size `n` accordingly).

## Consequences

- **Positive:** unblocks the *one* proof Firstpass can't yet show — a conformal bound with real gate
  error — behind a security-first, fail-closed design; the `Sandbox` seam doubles as the on-ramp to
  D3's hosted gate sandbox.
- **Cost:** benchmark now requires a container runtime in the dev/CI environment (documented; the
  harness fails closed and says so if it's missing). Slower per-task than in-process arithmetic.
- **Risk:** container-grade isolation is the accepted ceiling for a dev-time harness; if this code is
  ever repurposed to run untrusted *third-party* code on shared infra, that decision must be reopened
  (it becomes a D3 problem, not a 0002 problem).
