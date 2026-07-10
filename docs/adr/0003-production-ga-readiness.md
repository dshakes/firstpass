# ADR 0003 — Production / GA readiness: honest gap, tiers, and exit criteria

- **Status:** Proposed (2026-07-10)
- **Relates:** ADR 0001 (hosted GA architecture — the multi-tenant control plane), ADR 0002 (bench
  sandbox). This ADR is the umbrella: what "production-grade / GA" *means* for Firstpass, what is
  genuinely done, and the concrete gate to call it GA — so "make it GA" is a reviewable plan, not a
  stamp.

## Why this ADR exists (the honesty this project owes)

Firstpass has reached a strong, *measured* M0: the core thesis is proven on real Anthropic (n=200),
both headline caveats are addressed (parallel speculative escalation for latency; the coding-with-
tests sandbox + continuous gate score + feedback calibration for the conformal guarantee), and the
data plane (observe + enforce proxy, typed gates, tamper-evident trace chain, MCP/CLI surfaces) is
built and tested. **That is not the same as GA.** GA is a claim about *operational and security
posture under real load and adversarial conditions*, and parts of it are irreducibly not a code-only
deliverable:

- **A security audit** must be performed by someone who did not write the code (ideally external).
- **A soak/dogfood period** must actually elapse under real traffic (SPEC §F: 30 days).
- **Compliance (e.g. SOC2)** is an organizational + auditor process spanning months, not a source
  change. No amount of Rust makes a repo "SOC2 compliant."
- **The hosted multi-tenant plane** crosses trust boundaries (cross-tenant isolation, per-tenant key
  custody) that ADR 0001 §D3 explicitly gates behind design + external review. It must not be
  blind-coded.

So this ADR **refuses to fake a GA stamp** and instead pins the exit criteria and splits the work
into what is safe to build now vs. what is gated.

## D1 — Two deployment tiers; GA the small surface first

- **Tier 1 — self-host (single trust domain).** The operator runs the proxy against their own
  provider key, for their own traffic. This is what exists today. Its attack surface is small: a
  local/edge HTTP service, BYOK, an on-disk SQLite trace store. **This tier is the near-term GA
  target** — it can meet a real GA bar with the hardening in D3, no new trust boundaries.
- **Tier 2 — hosted multi-tenant.** Firstpass runs the plane for many tenants. This adds the hard
  boundaries (tenant isolation, key custody/KMS, per-tenant quotas, authz, gate sandboxing for
  tenant-supplied gate code). **This is a distinct, later GA**, governed by ADR 0001 + the design in
  D2, and it does not ship without external security review.

**Decision:** GA **Tier 1 first**. Do not let Tier 2's complexity block a shippable, honest Tier-1
GA. Market/position accordingly (self-host GA now; hosted in preview).

## D2 — Hosted-plane security design (Tier 2) — DESIGN ONLY, gated

Recorded so the boundary is explicit; **not implemented in this pass.** Each item is a trust boundary
that gets its own review before code:

- **Tenant isolation:** every store row, trace, gate-health counter, and budget keyed by tenant id;
  no query path without a tenant predicate. A tenant must be unable to read another's traces or
  influence another's gate-health/auto-disable.
- **Key custody:** BYOK keys never at rest in plaintext. Per-tenant envelope encryption via a KMS
  (the operator's cloud KMS); keys decrypted in-memory per request, never logged/traced (the
  existing `Auth` `Debug` redaction extends to storage). This is the ADR 0001 §D (KMS) work.
- **Gate sandboxing:** tenant-supplied gate code (subprocess/judge) is untrusted and runs under the
  ADR 0002 sandbox tier appropriate to the hosted plane (gVisor/microVM, not the bare
  `SubprocessGate`) — ADR 0001 §D3, external review required.
- **Per-tenant quotas + authz:** authenticated tenants, rate/spend limits per tenant, admin authz on
  config changes.

**Concrete gaps the internal audit (2026-07-10) confirmed are safe today (single-operator) but
become blockers in Tier 2 — recorded so none is forgotten:**

- **No authn on any handler.** `/v1/messages`, `/v1/feedback`, `/v1/capabilities` are all
  unauthenticated (`app()` has no auth layer). Correct for a local operator; required before hosted.
- **Feedback IDOR** (`proxy.rs` `feedback`): the handler checks `trace_exists(trace_id)` then
  `append_deferred` with **no ownership/tenant check** — any caller who guesses a `trace_id` could
  attach a verdict to another tenant's decision (feedback/label poisoning). Hosted needs unguessable
  trace ids **and** a tenant-ownership check before append.
- **`tenant_id` is a static env value** (`FIRSTPASS_TENANT`, default `"default"`), stamped onto
  traces — not derived from caller identity. Fine single-tenant; must come from authenticated
  identity in hosted.
- **Trace store is not tenant-scoped:** `load_all_traces`, `get_trace`, `trace_exists`, the MCP
  reader, and `calibrate_from_store` read across all rows. Hosted needs a tenant predicate on every
  query + row-level enforcement.
- **Shared env-fallback provider key:** BYOK is per-request, but the env-var fallback is a single
  shared credential — must be forbidden in hosted so tenant A never spends tenant B's key.
- **`gate_health` error budget is shared mutable state** across all requests — in hosted, one
  tenant tripping a gate's budget would auto-disable it for everyone; needs per-tenant scoping.

## D3 — The GA gate (Tier 1) — concrete exit criteria

Firstpass Tier 1 is **not** GA until all of these are true and evidenced:

1. **Security audit** by a non-author, findings triaged, criticals/highs fixed. (This pass runs an
   internal audit + fixes; an external pass is still owed before the stamp.)
2. **Request-plane hardening** landed and tested: explicit request body-size cap; server + upstream
   timeouts; a concurrency/overload guard; no secret ever logged/traced/returned in an error;
   structured, non-leaky error responses. *(This pass — see D4.)*
3. **Supply chain:** `cargo-deny` (advisories + licenses) green in CI *(already in CI)*; a documented
   dependency-update cadence; `cargo-audit` on a schedule.
4. **Observability:** request/trace/tenant ids on every path; metrics for latency (enforce p50/p95),
   escalation rate, gate error-budget trips, upstream failures; a health endpoint that reflects real
   dependency health, not just liveness.
5. **Resilience:** the trace-store writer degrades gracefully (already: hot path non-blocking); a
   documented backup/restore for the SQLite store; defined behavior when the store is unavailable
   (serve continues, traces best-effort — already the observe posture, verify for enforce).
6. **Performance budget** met and measured: enforce p95 under a stated ceiling on the default ladder;
   observe adds ≈0 on-path latency (already designed); sandbox throughput characterized.
7. **Runbook + SLOs:** documented failure modes, on-call runbook, error budget/SLO, and a rollback
   path (the cargo-dist release + a pinned prior version).
8. **Release cut exercised:** the `release.yml` (cargo-dist) actually run once end-to-end to produce
   signed artifacts + hashes (currently unexercised — SPEC §7.3/§E).
9. **Soak:** a real dogfood period under live traffic with no Sev-1 (SPEC §F).

## D4 — What THIS hardening pass lands (safe, verifiable, Tier-1)

Scoped to the existing single-tenant plane, no new trust boundaries:

- Internal **security audit** (read-only) → fix the real findings.
- **HTTP hardening:** explicit configurable request body-size limit; upstream (reqwest) request
  timeouts so a hung provider can't pin a handler; a request/concurrency guard; confirm error
  responses never leak internals or secrets.
- **Secret-logging audit:** grep + review that no key reaches logs/traces/errors/URLs; keep the
  `Auth` redaction invariant covered by a test.
- Optionally, land the **calibrated-threshold serving** wiring (from the feedback loop) behind config,
  defaulting to today's behavior — completing the last open feature without changing defaults.

Everything here is gated by the same bar as the rest of the repo: `fmt` clean, `clippy -D warnings`,
`cargo test --workspace`, and — for request-plane changes — driven/observed, not assumed.

## Consequences

- **Honest:** GA becomes a checklist with evidence, not a marketing adjective. Tier-1 GA is close and
  reachable; Tier-2 (hosted) is correctly held behind design + external review.
- **Cost:** the hardening pass is real work but bounded; the external audit, soak, release-cut, and
  compliance are process items that need calendar time and non-author humans — they cannot be
  collapsed into a code session, and this ADR says so plainly.
- **Risk if ignored:** stamping "GA/secured" without D3 evidence is how a proxy that holds provider
  keys and runs model-generated code ends up in production under-protected. The gate exists to
  prevent exactly that.
