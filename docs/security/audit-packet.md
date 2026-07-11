# External security review — audit packet

This is the packet to hand a non-author reviewer (external firm or a qualified internal reviewer who
did not write the code) so the review is targeted, not a cold read. It is the "audit-ready" artifact
referenced by [ADR 0003 §D3](../adr/0003-production-ga-readiness.md) and the GA gate.

> **Why external:** the author cannot self-certify a trust boundary. The internal audit (below)
> found and fixed real issues, but a GA stamp requires a review by someone who did not write the
> code. This packet exists to make that review efficient.

## Scope

- **In scope:** the request/trust-boundary surfaces — the proxy request plane, BYOK secret handling,
  the code-execution sandbox, the gate plugins, the trace store, and (highest priority) the
  multi-tenant isolation + auth code as it lands.
- **Out of scope:** third-party dependency internals (covered by `cargo audit`/`cargo deny` in CI),
  and the SOC2 organizational process (covered separately by `docs/compliance/soc2-controls.md`).

## Priority review targets (ranked)

1. **Multi-tenant isolation + authentication** — `tenant_auth` + the tenant-scoped store + the
   feedback ownership check (ADR 0004 §D1–D4). This is the newest and most security-critical code
   and ships **experimental / pre-review, default-off**. Verify the invariant: *tenant A can never
   read, influence, or spend on behalf of tenant B.* Scrutinize the Argon2 verify path, the auth
   middleware's failure semantics (opaque, no existence oracle), and every store query's tenant
   predicate. The cross-tenant isolation tests are the claim; try to break them.
2. **BYOK secret handling** — `provider.rs` `Auth` (redacted `Debug`), the header-only key flow, and
   confirm no key reaches logs / errors / URLs / the trace store. (Internal grep found no leak path.)
3. **Code-execution sandbox** — `firstpass-bench/src/sandbox.rs` (ADR 0002). The fail-closed
   invariant (no host-exec fallback), `--network none`, cap-drop, no host mounts, the base64-stdin
   file path, and the `runc`-vs-gVisor tier warning.
4. **Gate plugins** — the judge (candidate-as-data, prompt-injection resistance) and the subprocess
   gate (stdin, never argv).
5. **Trace store** — parameterized SQL, the tamper-evident hash chain, the bounded-channel
   load-shedding, and (for hosted) the tenant scoping in §1.
6. **Request-plane resilience** — HTTP timeouts (streaming-safe nuance), body cap, concurrency
   load-shed, error opacity.

## Reference material (read in this order)

- [`docs/threat-model.md`](../threat-model.md) — STRIDE across the trust boundaries; assets, entry
  points, mitigations, residual risk.
- [ADR 0001](../adr/0001-hosted-ga-architecture.md) — hosted GA architecture (the trust model).
- [ADR 0002](../adr/0002-bench-code-execution-sandbox.md) — the sandbox design + fail-closed rule.
- [ADR 0003](../adr/0003-production-ga-readiness.md) — GA readiness + the tenancy gap map.
- [ADR 0004](../adr/0004-hosted-multitenant-plane.md) — the multi-tenant plane design.
- [`SECURITY.md`](../../SECURITY.md) — disclosure policy + supported versions.

## Internal audit summary (2026-07-10)

An internal read-only audit reviewed the request/trust boundaries. **Fixed:** outbound HTTP timeouts
(DoS), bounded trace channel (OOM), error-response opacity, sandbox path quoting, explicit body cap.
**Confirmed solid:** parameterized SQL, redacted BYOK keys (no leak path), fail-closed sandbox,
candidate-as-data judge, stdin-not-argv subprocess gate. **Mapped for the hosted plane** (fixed by
ADR 0004, not yet enabled): unauthenticated handlers, feedback IDOR, non-tenant-scoped store, shared
gate-health budget. Full detail in ADR 0003 §D2.

## How to build, run, and test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                      # full suite
cargo test -p firstpass-bench --lib sandbox::tests::real_ -- --ignored   # real Docker sandbox isolation proof
cargo audit                                 # RustSec advisories (also in CI: .github/workflows/audit.yml)
firstpass up                                # run the proxy (see README for env)
```

## What a reviewer should return

Ranked findings (severity, file:line, concrete exploit, minimal fix), an explicit judgment on the
tenant-isolation invariant, and a sign-off statement (or the blockers preventing one). That sign-off
is the human GA gate this packet exists to obtain.
