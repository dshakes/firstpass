# ADR 0004 — Hosted multi-tenant plane: authn, tenant isolation, key custody

- **Status:** Proposed (2026-07-10)
- **Relates:** ADR 0001 (hosted GA architecture — the umbrella), ADR 0002 (sandbox, §D3 gate
  sandboxing), ADR 0003 (GA readiness — chose Tier-1 self-host GA first; §D2 mapped the tenancy
  gaps). This ADR is the **Tier-2 design**: the trust-boundary work that turns the single-operator
  data plane into a hosted multi-tenant service.
- **Gate:** the isolation and key-custody boundaries here ship **only after external security
  review** (ADR 0001 §D3). This ADR is the reviewable design; implementation lands incrementally,
  each piece behind tests + review. It is not blind-coded.

## Context

Today Firstpass is single-operator: no handler is authenticated, `tenant_id` is a static env value
stamped onto traces, the trace store and gate-health budget are global, and BYOK keys are
per-request (never stored). That is correct and safe for one operator. To host it for many tenants,
every one of those becomes a cross-tenant boundary. The internal audit (2026-07-10, recorded in ADR
0003 §D2) enumerated the exact gaps; this ADR designs the fix for each.

**Non-negotiable invariant for the whole plane:** *tenant A can never read, influence, or spend on
behalf of tenant B.* Every decision below serves that invariant; anything that can't be shown to
uphold it does not ship.

## Decisions

### D1 — Authentication: per-tenant API keys, hashed at rest

- Each tenant gets one or more API keys. The **plaintext key is shown once at creation**; only a
  **hash** is stored — Argon2id (`argon2` crate) with a per-key salt. Verification is constant-time
  (`subtle`), never a plain `==`.
- An auth middleware (axum `from_fn`/extractor) reads the key from an `Authorization: Bearer <key>`
  (or `x-firstpass-key`) header, looks up the tenant, and injects a `TenantId` into request
  extensions. **Every** business handler (`/v1/messages`, `/v1/feedback`, `/v1/capabilities`)
  requires it; `/healthz` and `/metrics` are operator-scoped (separate bind or operator auth), never
  tenant-facing.
- Missing/invalid key → `401`, opaque body (no "no such tenant" oracle). Failed lookups are rate-
  limited to blunt key-guessing.
- BYOK is unchanged for the *provider* key (the tenant still supplies their Anthropic/OpenAI key per
  request); the Firstpass API key authenticates *the tenant to Firstpass*. The two are distinct.

### D2 — Tenant identity propagation

`TenantId` flows from the auth extractor → `EnforceCtx` → the `Trace` (replacing the static
`FIRSTPASS_TENANT`). No code path may stamp a tenant from anything other than the authenticated
identity. The observe passthrough and enforce engine both carry it; the trace writer persists it.

### D3 — Tenant-scoped trace store

- Every `traces`/`deferred_verdicts` row already has a `tenant` column; the fix is **enforcement**:
  every read (`load_all_traces`, `get_trace`, `trace_exists`, the MCP reader, `calibrate_from_store`)
  and the feedback path take a `TenantId` and add a `WHERE tenant = ?` predicate. There is no code
  path that reads across tenants except an explicit operator/admin tool.
- Add a `(tenant, seq)` index. The hash chain stays **global** (single writer, chain integrity is
  cross-tenant by construction) — a tenant reads only its own rows, but the chain that proves
  tamper-evidence spans all of them; a tenant is shown its slice, verification runs on the whole.
- **Calibration becomes per-tenant** (`calibrate_from_store` scopes to the caller's tenant), so one
  tenant's feedback never moves another's serving threshold.

### D4 — Feedback IDOR fix

Two layers (defense in depth):
- **Ownership check (primary):** `POST /v1/feedback` verifies the target trace's `tenant` equals the
  caller's `TenantId` before `append_deferred`. D3 makes this a one-line predicate. A caller can only
  attach outcomes to *its own* decisions — closes the label-poisoning vector.
- **Unguessable references:** the client-facing trace reference is not a bare time-ordered UUIDv7
  (partially guessable). Either expose a separate random 128-bit token mapped to the trace, or append
  a random component. The ownership check is the real control; this removes the guessing surface.
- Feedback is per-tenant rate-limited (D6).

### D5 — Per-tenant key custody (only for stored keys)

- **Prefer not storing provider keys at all** — pure per-request BYOK stays the default; nothing to
  encrypt. Storage exists *only* for features that need a key without the tenant present (async/
  deferred re-grading, scheduled calibration).
- When a key must be stored, it is **envelope-encrypted**: a per-tenant Data Encryption Key (DEK)
  encrypts the key with **AES-256-GCM** (`aes-gcm`, RustCrypto — vetted); the DEK is wrapped by a Key
  Encryption Key (KEK) held in a **KMS**. Behind a `KeyCustody` trait:
  - `LocalKeyCustody` — real AES-256-GCM with a KEK loaded from a mounted secret/file (for self-host
    and dev). **Real crypto, not a stub.**
  - `AwsKmsKeyCustody` / `GcpKmsKeyCustody` / `VaultKeyCustody` — the KEK never leaves the KMS;
    `encrypt`/`decrypt` call the KMS. Pluggable, added per deployment target.
- Plaintext keys are decrypted **in memory, per request**, never logged (the existing `Auth` `Debug`
  redaction already holds) and never written to the trace store (traces already exclude auth
  headers). At-rest encryption is the SOC2 confidentiality control (ADR 0003).

### D6 — Per-tenant quotas, rate limits, and gate-health scoping

- **Rate limit + spend cap per tenant** (token bucket keyed by `TenantId`), so one noisy tenant
  can't starve others or exhaust the shared trace channel (which the D-hardening bounded, but a per-
  tenant cap is the real fairness control).
- **Gate-health error budget becomes per-`(tenant, gate)`** — today it's global, so one tenant
  tripping a gate's budget would auto-disable it for everyone. Key the `GateHealthRegistry` by tenant.
- Admin/config changes require operator (not tenant) authz.

### D7 — Isolation testing + the review gate

- **Cross-tenant isolation tests are mandatory** and part of the merge gate for this plane: tenant A
  cannot read B's traces (`get_trace`/`list`/`calibrate`), cannot attach feedback to B's trace
  (D4), cannot trip B's gate budget (D6), cannot see B's config. These are property-style tests, run
  in CI.
- **External review** of the auth, isolation, and key-custody code before it ships (ADR 0001 §D3).
  The internal author does not self-certify a tenant boundary.

## Phasing (each a reviewed PR; order matters)

1. **D1 + D2** — auth middleware + tenant propagation (identity is the prerequisite for everything).
2. **D3 + D4** — tenant-scoped store + the feedback ownership check (the core isolation + the audit's
   IDOR finding). Ships with D7 isolation tests.
3. **D6** — per-tenant rate limits + gate-health scoping.
4. **D5** — key custody (`LocalKeyCustody` first with real AES-256-GCM + tests; KMS adapters per
   target). Only needed once a stored-key feature exists.
5. **External review** → then enable the hosted plane.

## New dependencies (all vetted, real)

`argon2` (key hashing), `subtle` (constant-time compare), `aes-gcm` (RustCrypto AEAD), a KMS SDK per
target (`aws-sdk-kms` etc.). No hand-rolled crypto.

## Consequences

- **Positive:** a designed, reviewable path to a hosted multi-tenant GA with the cross-tenant
  invariant enforced in code and tested, plus the SOC2 confidentiality/access controls it implies.
- **Cost:** real work across auth, store, and crypto; a hard dependency on a KMS for the hosted
  target; and an external-review gate that is a human step, not a merge.
- **Risk if rushed:** auth and tenant isolation are exactly the code where a subtle bug is a breach.
  This ADR exists so that code is written against a reviewed design and lands behind isolation tests
  + external review — never blind-coded to hit a deadline.
