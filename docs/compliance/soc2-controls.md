# SOC 2 readiness map

**This is a readiness map, not a certification.** Firstpass has not undergone
a SOC 2 audit and makes no SOC 2 compliance claim. This document exists so an
operator (or a prospective auditor) can see, control by control, what
mechanism in the codebase would satisfy a given Trust Service Criterion today
— and, just as importantly, which controls are outright **gaps** because the
hosted plane they depend on doesn't exist yet. See
[`docs/threat-model.md`](../threat-model.md) for the underlying risk analysis
and [ADR 0001](../adr/0001-hosted-ga-architecture.md)/[ADR 0003](../adr/0003-production-ga-readiness.md)
for the gap tracking these controls reference.

Scope: the criteria below are the ones a self-hosted, single-operator proxy
can plausibly speak to (Security, Availability, Confidentiality). Processing
Integrity and Privacy are out of scope for this map — Firstpass does not
process regulated personal data as part of its own function.

## Security (Common Criteria)

| Criterion | Implementing mechanism | Status |
| --- | --- | --- |
| Audit logging of system activity | Tamper-evident hash-chain trace store — every routing/gate/escalation decision is a chained SQLite row (`crates/firstpass-proxy/src/store.rs`); `firstpass trace` re-derives and verifies the chain | **Satisfied** (self-hosted scope) |
| Change management | PR review (branch protection on `main`) + CI gates: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace`, `cargo deny` (license + advisory), and (new) `cargo audit` — see `.github/workflows/ci.yml` and `.github/workflows/audit.yml` | **Satisfied** |
| Secrets management | BYOK: provider keys travel per-request (`x-api-key`/`Authorization` header) or local env var, forwarded to the upstream provider, never persisted to the trace store, never passed to gate subprocesses, redacted from error responses (`crates/firstpass-proxy/src/error.rs`) | **Satisfied** for the self-hosted, per-request model. No secrets-manager/KMS integration — not needed until there's a hosted plane holding keys at rest |
| Encryption in transit | TLS to model providers is handled by the provider's HTTPS endpoint; TLS in front of Firstpass itself (agent → proxy) is the **operator's ingress** to configure — Firstpass does not terminate TLS itself | **Operator-dependent** — document as a deployment requirement, not a Firstpass-internal control |
| Encryption at rest | None. The SQLite trace store is plaintext on disk; there is no KMS-backed envelope encryption | **GAP** — [ADR 0001](../adr/0001-hosted-ga-architecture.md) proposes cloud + KMS-backed key/trace encryption for a hosted plane; **not implemented**. Self-hosted operators must supply disk-level encryption themselves |
| Vulnerability management | `cargo audit` (RustSec advisories) + `cargo deny check advisories`, weekly and on every PR — `.github/workflows/audit.yml` | **Satisfied** (new — this change) |
| Access control | Single-operator process; no authn/authz layer at the proxy (see threat model §1) | **GAP** — anyone who can reach the listening port with a valid provider key can route through it. Acceptable in the documented single-operator/self-hosted deployment; a hosted multi-tenant plane would need real authn (ADR 0001) |
| Monitoring / alerting | Prometheus `GET /metrics` exporter (enforce latency, escalations, served/upstream-failure/dropped-trace counters); structured `tracing` logs on every warn/error path; `GET /healthz` liveness probe (`crates/firstpass-proxy/src/{metrics,proxy}.rs`) | **Partial** — metrics, logs, and a liveness probe now ship; **alerting rules are not wired** (the operator scrapes `/metrics` and sets thresholds in their own monitoring stack). Deeper readiness (health that reflects dependency state, per-tenant metrics) remains per [ADR 0003](../adr/0003-production-ga-readiness.md) |
| Incident response | [`docs/runbooks/soak.md`](../runbooks/soak.md) covers the observe-mode soak and rollback path; a dedicated incident-response runbook (on-call rotation, sev classification) is listed in ADR 0003 as a GA exit criterion, not yet written | **Partial** — soak/rollback documented; formal IR runbook is a gap |

## Availability

| Criterion | Implementing mechanism | Status |
| --- | --- | --- |
| Resilience to a single upstream failure | Cross-provider failover on a provider 5xx (SPEC §7.2; `crates/firstpass-proxy/src/provider.rs`) | **Satisfied** |
| Backpressure / load shedding | Bounded trace channel — a stalled writer drops traces with a warning instead of blocking requests or OOMing (`store.rs`, `proxy.rs:66-69`) | **Satisfied** |
| Request-level timeouts | Provider HTTP client has connect (10s) and total (120s) timeouts (`provider.rs:405-406`); gate subprocesses are killed on timeout (`subprocess.rs`) | **Satisfied** |
| Capacity / concurrency limits | Request body cap (`DefaultBodyLimit::max`, `proxy.rs:77,87`) | **Partial** — body size is capped; there is no request-concurrency or rate limiter yet |
| Documented recovery procedure | [`docs/runbooks/soak.md`](../runbooks/soak.md) (rollback), [`docs/runbooks/release.md`](../runbooks/release.md) (pin to a prior version) | **Satisfied** |
| SLO / error budget | Not formally defined — no published availability target | **GAP** — tracked in ADR 0003 as a GA exit criterion |

## Confidentiality

| Criterion | Implementing mechanism | Status |
| --- | --- | --- |
| Credential confidentiality | BYOK keys never logged, never traced, never handed to a gate subprocess (see Security row above and `docs/threat-model.md` §1) | **Satisfied** |
| Error-response confidentiality | Errors returned to the caller are opaque; diagnostic detail stays server-side in `tracing` logs (`error.rs:101`) | **Satisfied** |
| Data-at-rest confidentiality | Trace store (which contains full request/response content) is unencrypted on disk | **GAP** — same as encryption-at-rest above; operator-supplied disk encryption is the only mitigation today |
| Data minimization to third parties (gates) | Gate subprocesses receive only `{gate_id, candidate, request}` on stdin — no API keys, no unrelated trace history (`subprocess.rs:6-11`) | **Satisfied** |

## Explicit gap summary

The following are **not** satisfied and should not be represented as such to
a customer or auditor until they change:

1. **Encryption at rest** for the trace store (needs the KMS-backed hosted
   plane in ADR 0001).
2. **Authentication/authorization** at the proxy boundary (currently: whoever
   can reach the port and holds a valid provider key can route through it).
3. **Alerting** — a Prometheus `/metrics` exporter now ships (plus structured
   logs and `/healthz`), but alerting rules/thresholds are the operator's to
   wire in their monitoring stack; none are shipped.
4. **Formal incident-response runbook and SLO/error budget.**
5. **External security audit / SOC 2 audit itself** — has not happened.
6. **Hosted multi-tenant plane** — does not exist; every control in this
   document is scoped to a single-operator, self-hosted deployment.
