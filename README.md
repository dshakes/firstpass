# Switchyard

**Routes every LLM request to the cheapest model that provably clears your quality gate.**
Verification-gated escalation with a full audit trail — proof over prediction.

> A switchyard is where railcars are sorted and routed — deterministically, by switches
> and gravity, one hump at a time. Same idea, for tokens.

Status: **founding spec stage.** Read [SPEC.md](SPEC.md).

## The idea in four lines

1. Send the request to the cheapest plausible model tier.
2. Run a real gate on the output — tests, typecheck, schema, fresh-context judge.
3. Escalate one rung only on gate failure. Serve the first output that passes.
4. Log every decision as an auditable trace; gate verdicts train the routing policy.

Everyone else routes by *predicting* quality. Switchyard routes by *verifying* it —
and hands you the receipt.

## Roadmap (see SPEC.md §16)

- **M0** — tier-clearance benchmark on real agentic coding tasks (go/no-go gate)
- **M1** — Rust proxy MVP: Anthropic + OpenAI endpoints, static ladder, trace store
- **M2** — gate framework: plugin contract, reference gates, enforce mode, feedback API
- **M3** — dogfood GA on real agent traffic
- **M4/M5** — learned routing (contextual bandit): shadow → live
- **M6** — OSS launch
