# Changelog

## [0.3.0]

**Breaking.** Two config changes can stop an existing deployment from starting. Both are cases
where the previous behaviour was silently producing a wrong number, so failing at startup with a
message is the intended upgrade path — read the two fixes below before upgrading.

### Breaking: an unpriced ladder rung is now rejected at parse

`cost_usd()` returns an error for a model with no price entry, and every call site was swallowing
it with `.unwrap_or(0.0)`. The built-in table knows only the seven first-party models, so on the
entire bring-your-own-provider path — Groq, DeepSeek, OpenRouter, a local Ollama rung — this had
two silent consequences:

- the **receipt recorded `cost_usd: 0.0`** for calls that really cost money. In a product whose
  claim is a tamper-evident audit record, that is a fabricated figure, not a missing one.
- **`[budget]` caps could never trip**, because they are fed by that same total. The budget
  guardrail was inert for exactly the operators who configured a provider by hand.

If your ladder names a model outside the built-in table, the proxy now refuses to start and prints
the block to paste:

```toml
[[price]]
model = "groq/llama-3.3-70b-versatile"
input_per_mtok = 0.59
output_per_mtok = 0.79
```

A model you host yourself really is free — declare it as an explicit `0.0`, so the `$0.00` in the
receipt is a stated fact rather than an inferred one.

### Breaking: two Google models were removed because they never existed

`google/gemini-3.1-flash` and `google/gemini-3.1-pro` were in the price table and in the ladder
`firstpass onboard` generated, and neither is a real model — `generateContent` answers
`models/gemini-3.1-flash is not found for API version v1beta`. Every Google user's first request
failed. Replaced with GA ids at Google's published rates:

| id | input / 1M | output / 1M |
| --- | --- | --- |
| `google/gemini-3.5-flash-lite` | $0.30 | $2.50 |
| `google/gemini-3.6-flash` | $1.50 | $7.50 |
| `google/gemini-2.5-pro` | $1.25 | $10.00 |

The generated Google ladder is now Flash-Lite → Flash. It was Flash → Pro, which at $1.50/$7.50
against $1.25/$10.00 is an ~8% price gradient — escalation bought nothing.

### Added

- **`firstpass kiosk`** — one press, no questions: finds the provider key already in your
  environment, writes a config, puts one real request through the ladder, prints the receipt. With
  no key it runs the keyless demo, so the button always produces a working receipt.
- **`firstpass handshake`** — the same for a caller with no terminal: one JSON document of keys
  found, the config that would run, and a **keyless** end-to-end self-test. Reports; never writes.
  Also an MCP tool.
- **`GET /panel`** — live receipts served by the binary itself, no CDN. Backed by
  `GET /v1/receipts`, which stays inside the authed group and is tenant-scoped.
- **Benchmarks**: `--coding-policy` compares routing policies on identical measurements;
  BigCodeBench and SWE-bench harnesses; `scripts/fetch-coding-dataset.py`.

### Fixed

- Provider HTTP errors log the upstream body server-side. `"upstream http 404"` alone hid the
  Gemini bug above for weeks; the body named the exact model that did not exist.
- `provider-smoke` jobs that skip for a missing secret no longer read as verified — each job
  writes its real state into the run summary.
- Judge model calls are metered and priced, so a judged policy can report a real `$/success`
  instead of one that omits the calls that produced it.

### Measured

On 974 real MBPP coding tasks (committed artifacts under `docs/benchmarks/`): gating recovered
**+15.2 points** of quality over serving the cheap model alone (paired, 95% CI [+12.9, +17.5]) and
made **zero of those 974 tasks worse**, at 12–57% lower cost per success **depending on the
ladder** — 12% against a `sonnet` ceiling, 57% against an `opus` one.

## [0.2.7]

Restored a dual-arch `latest` image: `latest` now tracks release tags rather than every push to
`main`, so it is built for amd64 and arm64.
