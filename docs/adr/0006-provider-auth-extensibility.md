# ADR 0006 — Provider dialects and bespoke-auth backends (Gemini, Bedrock, Vertex)

- Status: Accepted — Gemini, Bedrock, and Vertex all implemented (offline-tested;
  Bedrock/Vertex are LIVE-UNVERIFIED — not yet exercised against real AWS/GCP)
- Date: 2026-07-15
- Related: `[[provider]]` config (multi-provider routing), ADR 0001 (hosted plane)

## Context

Firstpass now routes to any provider via a `[[provider]]` entry with a `dialect`
(`anthropic` | `openai`) and an `api_key_env`. That covers Anthropic, OpenAI, and
every OpenAI-compatible host (Groq, Together, Fireworks, DeepSeek, Mistral, xAI,
OpenRouter, Azure, local Ollama / vLLM). Two gaps remain among the *major*
providers, each blocked on something the current model doesn't express:

1. **Google Gemini** — a third **wire dialect** (`contents`/`parts`,
   `system_instruction`, `generateContent`), API-key auth via the
   `x-goog-api-key` header. No new auth *scheme*, just a new translation.
2. **AWS Bedrock** — the request body for Claude-on-Bedrock is Anthropic-shaped,
   but auth is **AWS SigV4 request signing** (access key / secret / session token,
   region-scoped), and the URL is `/model/{id}/invoke`. A new *auth scheme*.
3. **Google Vertex AI** — Claude/Gemini-on-Vertex, but auth is a **GCP OAuth2
   bearer token** minted from a service account (RS256-signed JWT → token
   exchange, cached and refreshed). A new *auth scheme*.

The through-line: **dialect** (how the body/response are shaped) and **auth
scheme** (how the request is credentialed) are independent axes. Today both are
implicitly "api-key in a header." Gemini adds a dialect; Bedrock and Vertex add
auth schemes.

## Decision

Keep dialect and auth as separate, additive axes on `[[provider]]`.

- **Dialect** stays an enum. Add `gemini`. Each dialect is a `Provider` adapter
  that translates `ModelRequest`/`ModelResponse` to/from one wire API. **Done for
  Gemini** — a pure translation with offline round-trip tests
  (`gemini_request_body`, `gemini_parse_response`); routes text today, with
  Gemini's `functionCall`/`inlineData` tool/multimodal shapes as the same
  follow-on the OpenAI adapter carries.
- **Auth scheme** becomes an explicit, optional dimension (default: the current
  api-key-in-header behavior). Bedrock and Vertex are new schemes that wrap an
  existing dialect's body:
  - **Bedrock** = `anthropic` body + SigV4 signing + `/model/{id}/invoke` URL.
  - **Vertex** = `anthropic` or `gemini` body + GCP OAuth bearer + Vertex URL.

## The dependency decision (resolved — Bedrock/Vertex are now coded)

SigV4 signing and GCP service-account OAuth are **security-sensitive auth
primitives**. Hand-rolling either — canonical-request construction and HMAC
signing-key derivation for SigV4, or RS256 JWT minting and token exchange for GCP
— is exactly the class of code where a subtle bug becomes an auth vulnerability.
The responsible implementations use maintained crates, now added to the
workspace:

- Bedrock: `aws-sigv4` 1.4 + `aws-credential-types` 1.2 (pinned below their
  latest release — newer point releases bump the declared MSRV past this
  workspace's `rust-version`; see `Cargo.toml` for the exact pins), credentials
  from the standard `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/
  `AWS_SESSION_TOKEN` env vars.
- Vertex: `gcp_auth` 0.12 (service-account → cached access token).

This pulled the AWS Smithy / GCP auth stacks into the dependency tree — a
**supply-chain and `cargo-deny` decision** (new licenses + advisory surface,
larger binary) that the operator has now signed off on. `cargo deny check`
passes for the new crates (Apache-2.0/MIT, no advisories); the only new
findings are `bans.multiple-versions` warnings (transitive `http`/`sha2`/
`digest` version splits from the AWS/GCP stacks pulling in their own copies) —
non-blocking under this repo's `multiple-versions = "warn"` policy.

## Consequences / Invariants

- **I1 — Additive, default unchanged.** Adding `gemini` and the auth-scheme axis
  changes no existing provider's behavior; `[[provider]]` entries without the new
  fields route exactly as before.
- **I2 — No secret in a URL.** Every auth scheme credentials the request via a
  header (Gemini: `x-goog-api-key`; Bedrock: `Authorization` SigV4; Vertex:
  `Authorization: Bearer`), never a query string — consistent with the proxy's
  no-secrets-in-URLs rule.
- **I3 — Auth crypto is delegated, not hand-rolled.** SigV4 and GCP OAuth are
  implemented with maintained crates, never bespoke signing code.

## Phases

- **P1 — Gemini dialect. ✅ Done.** `Dialect::Gemini` + `GeminiProvider`; offline
  translation tests; example config entry; docs.
- **P2 — Bedrock (SigV4). ✅ Done.** `AuthScheme::AwsSigv4` + `BedrockProvider`:
  Anthropic body (`anthropic_messages_body`, model in the URL not the body) + SigV4
  signing via `aws-sigv4`/`aws-credential-types` (`sign_bedrock`) +
  `/model/{id}/invoke`; creds from `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/
  `AWS_SESSION_TOKEN`. Offline tests assert the signed request carries an
  `AWS4-HMAC-SHA256` `authorization` header and the right `host` header, given dummy
  credentials — signature validity against real AWS is LIVE-UNVERIFIED.
- **P3 — Vertex (GCP OAuth). ✅ Done.** `AuthScheme::GcpOauth` + `VertexProvider`:
  Anthropic body + cached OAuth bearer minted by `gcp_auth` (`cloud-platform` scope)
  + Vertex `rawPredict` URL. Offline tests cover URL construction and the
  missing-region/missing-project error paths; the live token exchange and Vertex
  endpoint are LIVE-UNVERIFIED.

## Risks

- **Auth vulnerability from hand-rolled signing** — mitigated by I3 (delegate to
  maintained crates; that is why P2/P3 waited on the dependency decision rather
  than shipping bespoke signing).
- **Dependency/supply-chain growth** — the AWS/GCP auth stacks are non-trivial;
  now accepted (see the dependency decision above). Still open: neither adapter
  has been exercised against a real AWS/GCP endpoint (LIVE-UNVERIFIED) — verify
  against real credentials before relying on either in production.
