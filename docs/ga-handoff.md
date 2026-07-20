# GA handoff — what only a human can close

Everything in Phases 0–3 that is **code** has shipped and is on `main` at v0.2.0, gate-green
(365 tests, clippy `-D warnings`, fmt clean). What remains between here and a GA stamp is, by
[ADR 0003](adr/0003-ga-readiness.md), deliberately **not** a code task — it needs a person,
a clock, a credential, or an external party. This is that list.

## Credentials / secrets (unblocks already-built automation)

| Item | What's built | What you do |
|---|---|---|
| Provider live badges | `provider-smoke.yml` runs a real request through the proxy per provider and asserts served + receipt. Anthropic is proven. | Add `OPENAI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` as repo secrets. Each green run flips that provider from "implemented" to "wire-verified". |
| crates.io publish | `crates-io.yml` is wired and gated on a token. | Add `CARGO_REGISTRY_TOKEN`; then `cargo install firstpass-proxy` (registry form) works and the README's cargo row loses its caveat. |
| npm publish | `npm-publish.yml` exists. | Add `NPM_TOKEN` if you want the npm channel; otherwise leave the row off. |
| Homebrew tap | Formula name is `firstpass-proxy`. | Create/point the `dshakes/homebrew-tap` repo so `brew install dshakes/tap/firstpass-proxy` resolves. |

## Time-bound (a clock has to run)

- **30-day self-host soak** (SPEC §M3, [runbook](runbooks/soak.md)): run the proxy on real
  agent traffic for 30 continuous days with the false-pass SLO alarm armed. This is the GA
  gate that no amount of code replaces. The observability for it is committed
  ([Grafana dashboard](observability/grafana-firstpass.json)); durable receipts
  (`FIRSTPASS_RECEIPTS=durable`) mean the audit chain survives the whole window.
- **Live bandit A/B**: promote `algorithm = "thompson"` to default only after a real
  thompson-vs-ucb1 run on your workload shows the win (ADR 0007). Same for flipping the
  `speculation_band` on by default.
- **Judged-gate MBPP artifact**: the base bound is committed; the LLM-judge variant needs one
  more live run (`FIRSTPASS_CODING_JUDGE=claude-sonnet-5 … --coding-live`) to land its
  artifact next to `mbpp-live-base.txt`. Cost ~$10, ~1h.

## External parties (someone outside the repo)

- **External security audit** of the multi-tenant plane — tenant auth (Argon2id) and key
  custody (AES-256-GCM) are implemented and passed two internal adversarial reviews, but
  ADR 0004 §D7 makes the hosted plane contingent on an *external* review. Do not enable the
  hosted plane for real tenants until this clears.
- **SOC 2** process against the committed control docs (`docs/compliance/soc2-controls.md`).

## Explicitly out of scope for this cut (named, not forgotten)

- Postgres store backend (durable spill closed the data-loss gap; multi-node is a separate
  effort behind the same `store` trait seam).
- Cross-dialect *structured* translation beyond Anthropic↔OpenAI (Gemini tool/image shapes).
- The hosted control plane itself (ADR 0004) — gated on the external audit above.

The one-line status for a reader: **the product is code-complete for self-host through
Phase 3; GA is now audit + soak + credentials, which are people and clocks, not commits.**
