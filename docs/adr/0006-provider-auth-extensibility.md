# ADR 0006 — Provider dialects and bespoke-auth backends (Gemini, Bedrock, Vertex)

- Status: Accepted — Gemini implemented; Bedrock + Vertex designed, gated on a dependency decision
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

## The dependency decision (why Bedrock/Vertex are not yet coded)

SigV4 signing and GCP service-account OAuth are **security-sensitive auth
primitives**. Hand-rolling either — canonical-request construction and HMAC
signing-key derivation for SigV4, or RS256 JWT minting and token exchange for GCP
— is exactly the class of code where a subtle bug becomes an auth vulnerability.
The responsible implementations use maintained crates:

- Bedrock: `aws-sigv4` + `aws-credential-types` (or the `aws-sdk-bedrockruntime`
  client), credentials from the standard env vars / provider chain.
- Vertex: `gcp_auth` (service-account → cached access token).

Adding these pulls the AWS Smithy / GCP auth stacks into the dependency tree,
which is a **supply-chain and `cargo-deny` decision** (new licenses + advisory
surface, larger binary) that belongs to the operator, not to a silent import.
So this ADR ships the **design and the seam**; wiring Bedrock/Vertex is a
one-adapter change once the dependency footprint is signed off.

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
- **P2 — Bedrock (SigV4). Designed; gated on the `aws-sigv4` dependency sign-off.**
  Anthropic body + SigV4 auth + `/model/{id}/invoke`; creds from env; offline test
  that a signed request carries an `Authorization` SigV4 header and the right host.
- **P3 — Vertex (GCP OAuth). Designed; gated on the `gcp_auth` dependency sign-off.**
  Anthropic/Gemini body + cached OAuth bearer + Vertex URL.

## Risks

- **Auth vulnerability from hand-rolled signing** — mitigated by I3 (delegate to
  maintained crates; that is the reason P2/P3 are gated, not rushed).
- **Dependency/supply-chain growth** — the AWS/GCP auth stacks are non-trivial;
  the operator weighs that against needing Bedrock/Vertex before it lands.
