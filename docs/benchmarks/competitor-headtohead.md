# Head-to-head vs competitor routing mechanisms — MBPP-974, offline replay

date: 2026-10-28 · spend: $0 (replay of committed checkpoints) · commit: 6acfde5+competitors.rs
command: `firstpass-bench --replay <checkpoint.jsonl>` (section "Head-to-head")

Each competitor is represented by its **decision rule**, replayed over the same measured candidates,
prices and hidden-oracle ground truth — not by calling the vendor. That isolates the routing decision
from each vendor's prompt classifier. It also means the predictive row is a *perfect-foresight upper bound*:
no shipped predictive router knows correctness before generating.

Caveats: (1) MBPP is code, where an executable gate exists — the case most favourable to Firstpass.
The judge arms are the relevant comparison where tests do not exist. (2) The k5 checkpoint bills five
haiku judge calls into every rung visit, for every arm including test-only first-pass, which handicaps
first-pass rather than the competitors. That is why first-pass costs more than always-top on that ladder.
(3) Switchyard's escalation router judges multi-turn session progress; single-shot MBPP exercises only
its per-turn judge decision.

### `mbpp-974-opus.checkpoint.jsonl`

n = 974 tasks · every arm sees identical candidates, prices and ground truth

| arm | stands in for | success | $/success | Δ success (fp − arm) | Δ $/task (fp − arm) |
|---|---|---|---|---|---|
| **first-pass** | Firstpass (executed-test gate) | 0.9333 | $0.00330 | — | — |
| oracle-predictive (upper bound) | Not Diamond · Martian · OpenRouter Auto · Foundry · Bedrock IPR | 0.9497 | $0.00250 | -0.016 [-0.025, -0.009] | +0.00070 [+0.00051, +0.00093] |

First-pass closes 86.3% of the always-top → perfect-predictive-router cost gap. The oracle row is an upper bound on every predictive router: it knows each answer's correctness before generating it.


### `mbpp-974-sonnet-k5.checkpoint.jsonl`

n = 974 tasks · every arm sees identical candidates, prices and ground truth

| arm | stands in for | success | $/success | Δ success (fp − arm) | Δ $/task (fp − arm) |
|---|---|---|---|---|---|
| **first-pass** | Firstpass (executed-test gate) | 0.9230 | $0.01697 | — | — |
| oracle-predictive (upper bound) | Not Diamond · Martian · OpenRouter Auto · Foundry · Bedrock IPR | 0.9456 | $0.01289 | -0.023 [-0.032, -0.013] | +0.00347 [+0.00280, +0.00416] |
| judge-escalate | NVIDIA NeMo Switchyard escalation router | 0.8901 | $0.01970 | +0.033 [+0.018, +0.048] | -0.00187 [-0.00256, -0.00117] |
| conformal-cascade (α=0.10) | Conformal Cascade, arXiv 2607.25018 | 0.8717 | $0.01763 | +0.051 [+0.036, +0.068] | +0.00029 [-0.00033, +0.00091] |

First-pass costs MORE than always-top on this ladder, so it closes none of the always-top → perfect-predictive-router cost gap. The oracle row is an upper bound on every predictive router: it knows each answer's correctness before generating it. Conformal thresholds per fold: [0.2, 0.4].

