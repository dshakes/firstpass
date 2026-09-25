# Pre-registration: find a verifier stronger than OpenJev for the second gate

Written 2026-09-25, **before any new verifier was scored.** This builds on
`specs/prior-blend-and-decision-gate.md` Study B, where OpenJev `noul` got catch 0.216 and collateral 0.103 →
NOT-RECOMMENDED.

## Data

The same 974 served MBPP answers (`~/vrb-cascade.jsonl`), with the same cached hidden-test oracle labels (111 wrong).
Every candidate has already passed the existing gate, so **catch rate means wrong answers that gate let through.**

## Held-out protocol (guards against choosing the winner on the scoring data)

- A seeded, deterministic 50/50 split by `sha256(id)`. The two halves are **dev** and **held-out**.
- **On dev:** for each verifier, pick τ to maximize catch rate subject to collateral ≤ 0.05. Then pick the verifier with
  the highest dev catch rate.
- **On held-out:** score only that one verifier at its dev τ. That is the verdict. The other verifiers' held-out numbers
  are reported for context and cannot change it.

## Verifiers

Each one is local and needs no API key. Candidate code is always passed as data, never as instructions.

- **V0, baseline:** OpenJev `noul` with default options, the same as Study B.
- **V1:** OpenJev `noul` with `think: 1024` and `samples: 4`. The model reasons before reading the answer, and the answer
  is averaged over reads.
- **V2, local LLM judge:** the proxy's `JudgeGate` prompt, verbatim, sent to a local coder model through an
  OpenAI-compatible server (mlx-lm). The model is `Qwen3-Coder-30B-A3B-Instruct` (MLX 4-bit), or the nearest available
  coder model at least that strong, named in the report. Its score is parsed exactly as `judge.rs` parses it. A V2 win is
  deployable today through the existing `judge` gate plus a `[[provider]]` block.
- **V3, generated tests (CodeT-style):** the same local coder writes 5 assert-based tests **from the task text only,
  without seeing the candidate**. The candidate runs against them in the fail-closed sandbox. Score = fraction passed.

## Verdict (held-out, point estimates, 95% CIs reported)

**VALUE-ADD** requires catch rate ≥ 0.30 and collateral ≤ 0.05; otherwise **NOT-RECOMMENDED**. If the held-out half has
fewer than 20 wrong answers, the verdict is **UNDERPOWERED**. The pre-registration also records wall-clock latency per
verification, the model used and memory use.
