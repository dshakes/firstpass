# ADR 0001 — Hosted / commercial GA architecture

- **Status:** Proposed (2026-07-09)
- **Blocks:** the hosted control plane build (roadmap batches G/H). Implementation must not start
  until the *Decisions needed from the business* below are made and Phase 0 (self-host GA) is met.
- **Supersedes/relates:** SPEC §12 (security & trust model), §7.3/§7.4 (distribution, pluggability).

## Context

Firstpass today is a **single-tenant, self-hosted** proxy. Provider keys (BYOK) flow through from the
caller's own request headers; traces live in local SQLite; there is exactly one trust domain — the
operator's. Milestones M0–M2 are built and PR #20 adds live-proof machinery, streaming, config-wired
gates, and the `firstpass` CLI + MCP surface.

The stated goal is **hosted / commercial GA**: Firstpass runs the proxy *for* customers. That single
change — we now hold other people's traffic, provider keys, and audit trails — turns SPEC §12 from
aspiration into load-bearing requirement and introduces trust boundaries that do not exist in the
self-host product. This ADR exists because those are **new trust boundaries**, and per our
engineering rules a new trust boundary is designed in an ADR before it is coded, never blind-built.

**Hard prerequisite (Phase 0):** self-host **dogfood GA** (SPEC §16 M3) — 30 days continuous,
verified savings > 0 at quality parity, false-pass < 2%. We do not host an unproven data plane. That
gate is not yet met (it needs the live-provider proof numbers, which need real API keys).

## The three new trust boundaries

1. **Tenant ↔ tenant** — one customer's data, policies, and learning must never leak to another.
2. **Customer keys at rest** — we now store BYOK provider keys, not just forward a header.
3. **Customer gate code** — a subprocess/judge gate is *untrusted code we execute on our infra*.

Each is a "never widen to make something work" boundary. The rest of this ADR is how we hold them.

## Decisions

### D1 — Tenancy: one control plane, per-tenant data isolation enforced in code
Options: (a) pod-per-tenant (strong isolation, high cost/ops); (b) shared process, per-tenant
row/namespace isolation with `tenant_id` on every record and query-level enforcement; (c) tiered
hybrid.
**Recommend (b), with (a)/(c) available for enterprise.** Every stored object — traces, hash chains,
routing policies, bandit posteriors — is keyed by `tenant_id` and access is enforced at the data
layer (not just filtered in app code). **No cross-tenant learning without explicit opt-in.** The
hash chain is per-tenant so each tenant can independently re-derive and audit its own trail.

### D2 — Provider keys: BYOK, envelope-encrypted via KMS, plaintext only in memory
- **Never at rest in plaintext.** Envelope encryption: a per-tenant data key wrapped by a KMS
  customer master key; the provider key is decrypted **in memory, per request**, used, and dropped.
  Never logged, never written to a trace (SPEC §9.3, §12.1).
- A **KMS-agnostic trait** (`encrypt`/`decrypt`/`rotate`) with one cloud implementation to start.
- **Self-host is unchanged** — keys stay in the customer's own environment; this applies only to the
  hosted plane.
- *Decision needed:* which cloud/KMS first.

### D3 — Gate sandboxing: customer gates are untrusted code (the hardest boundary)
Today's `SubprocessGate` spawns a host process — **acceptable self-host, forbidden for tenant gates
in the hosted plane.** Tiers:
- **Deterministic/pure-compute gates → WASI (wasmtime):** no syscalls, no network, deterministic,
  cheap. Covers schema/format/regex-style judges.
- **General gates (run a test suite, call a model) → microVM (Firecracker) or gVisor:** **no network
  by default**, CPU/memory/time capped, ephemeral, `kill_on_drop`.
**Recommend:** wasmtime tier first (smaller blast radius, ships sooner), then the microVM tier.
This tier is security-critical enough to get **its own ADR + external review** before implementation.
- *Decision needed:* microVM tech for the general tier (Firecracker vs gVisor) — infra-dependent.

### D4 — Tenant authn/authz
Tenants authenticate to the hosted proxy with a **Firstpass-issued token**, distinct from any
provider key. The BYOK provider key is attached to the tenant and handled per D2. RBAC per §12.

### D5 — Compliance & data handling (batch H)
- **Zero-retention default;** traces persisted only under an explicit per-tenant retention policy.
- **Supply chain:** signed releases + SBOM + provenance. Partly wired (cargo-deny in CI, cosign in
  `release.yml`); extend with an SBOM attestation on releases.
- **SOC 2 Type I within 12 months of hosted GA;** legal review before the first external tenant.
- *Decision needed:* auditor, timeline, and legal-review owner.

## Isolation invariants — must never regress (release blockers)
1. Each tenant's hash chain is independently re-derivable by that tenant.
2. No lock-in at the data plane (self-host offboards with one env var).
3. **No query ever returns another tenant's data** — proven by an adversarial tenant-isolation suite.
4. **A customer gate cannot reach the network or another tenant** — proven by a red-team fixture suite.
5. A release that regresses the isolation suite **or** the prompt-injection suite (§12.3) does not ship.

## Phasing
- **Phase 0 (prerequisite):** self-host dogfood GA (F). *Not started — needs live-provider proof first.*
- **Phase 1:** control plane + tenancy (D1) + tenant authn (D4); the tenant-isolation test suite.
- **Phase 2:** BYOK envelope encryption (D2) — KMS integration.
- **Phase 3:** gate sandboxing (D3) — wasmtime tier, then microVM tier (own ADR + review).
- **Phase 4:** compliance (D5) — SOC 2 path, legal, external design partners.

## Decisions needed from the business (block Phase 1+)
1. Cloud + KMS provider (D2).
2. microVM sandbox tech for the general gate tier (D3).
3. SOC 2 auditor + timeline; legal-review owner (D5).
4. Hosting model — single-region first? A BYOC / on-prem option for enterprise?

## Consequences
- **Positive:** a build-ready, security-first hosted design; the sacred boundaries are named and made
  testable *before* any code.
- **Cost:** Phases 1–4 are a multi-quarter, security-critical build, deliberately gated behind a
  proven self-host data plane. Hosting an unproven plane is explicitly out of scope.
- **Risk concentrated in D3** (gate sandboxing); it does not proceed on a rushed timeline or without
  external review.
