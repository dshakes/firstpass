# Security Policy

Firstpass is a proxy that sits between your agent and model providers, holding
BYOK provider credentials in-flight and writing an audit trace of every
decision. We take vulnerabilities in that position seriously. See
[`docs/threat-model.md`](docs/threat-model.md) for the full trust-boundary
analysis this policy is scoped against.

## Supported versions

Firstpass is pre-1.0 (workspace version `0.1.0`, see [`Cargo.toml`](Cargo.toml)).
There is no published binary release yet — see
[`docs/runbooks/release.md`](docs/runbooks/release.md). Until the first tagged
release, **only `main`** is supported; security fixes land there and there is
no back-port matrix.

| Version | Supported |
| --- | --- |
| `main` (source build) | :white_check_mark: |
| Tagged releases | none published yet |

## Reporting a vulnerability

**Please do not open a public GitHub issue for a suspected vulnerability.**
Public issues are for functional bugs; a vulnerability report filed there is
visible to everyone before a fix ships.

Report privately to: **`<security-contact>`**
(operator: replace this placeholder with a monitored security inbox or GitHub
Security Advisory link before this policy is treated as live).

If the repository has GitHub's private vulnerability reporting enabled, you
may instead use **Security → Report a vulnerability** on the repo, which
opens a private advisory thread with the maintainers directly.

Please include:

- The affected component (proxy, gate runner, trace store, bench harness,
  sandbox) and, if known, the file/line (`crates/.../src/....rs:NN`).
- A minimal reproduction — request/response pair, config, or PoC script.
- The trust boundary crossed (see the table in
  [`docs/threat-model.md`](docs/threat-model.md)) and the impact you assess
  (credential exposure, trace tampering, sandbox escape, gate bypass, etc).
- Whether you've verified it against `main` or a specific commit SHA.

### What's in scope

- The `firstpass-proxy` crate: routing, gate execution (subprocess/JSON-schema/
  judge), escalation, cross-provider failover, the trace store, the feedback
  API, and the MCP/CLI surfaces.
- The `firstpass-bench` crate's code-execution sandbox (ADR 0002).
- The `firstpass-core` crate (shared types, conformal calibration).
- CI/release tooling in `.github/workflows/` to the extent it could compromise
  a build artifact or leak a secret.

### What's out of scope

- Findings that require an operator to have already misconfigured a trust
  boundary described in `docs/threat-model.md` as the operator's
  responsibility (e.g. running the proxy with an operator-supplied gate
  command the operator does not trust; disabling the sandbox fail-closed
  check on purpose).
- The hosted/multi-tenant plane — it does not exist yet (see ADR 0001; this is
  a single-operator project today).
- Denial of service via raw traffic volume against a self-hosted deployment
  you control (that's a deployment/infra concern, not a Firstpass defect).
- Vulnerabilities in upstream dependencies without a demonstrated exploit path
  through Firstpass — report those upstream and to us via `cargo audit`/
  `cargo deny` advisory tracking (see `.github/workflows/audit.yml`) instead.

## Response window

This is a small, single-maintainer project — response times are best-effort,
not contractual:

- **Acknowledgment:** within 5 business days.
- **Triage / severity assessment:** within 10 business days of acknowledgment.
- **Fix or mitigation:** timeline communicated after triage, prioritized by
  severity. Critical issues (credential exfiltration, trace-chain forgery,
  sandbox escape) are prioritized above all other work.

We'll keep you updated as the report moves through triage and ask that you
give us a reasonable window to ship a fix before any public disclosure.

## Safe harbor

If you make a good-faith effort to comply with this policy while researching
and reporting a vulnerability — including avoiding privacy violations, data
destruction, and service disruption to anyone other than your own test
instance — we will not pursue legal action against you for that research, and
we'll consider your activity authorized for the purpose of any applicable
computer-fraud or anti-hacking laws. This safe harbor does not extend to
testing against infrastructure or data you don't own or have explicit
permission to test.

## Credit

With your permission, we'll credit you by name (or handle) in the fix's
commit/changelog once it ships. Let us know your preference when you report.
