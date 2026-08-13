# Related work — what Firstpass borrows, and what is actually new

Firstpass is **verified cascade routing**. Neither half of that is novel: cascades with
verification have been published since 2023, and conformal risk control is a standard tool.
This page states plainly what came first, so the parts that *are* new can be judged on their
own.

The reference map for the field is
[Awesome-Routing-LLMs](https://github.com/MilkThink-Lab/Awesome-Routing-LLMs), which sorts
~130 methods by *when* the routing decision happens. Firstpass sits in its smallest family —
**Verification Routing → Self-Assessment** — alongside AutoMix and CP-Router. The large
majority of the field is *pre-judgment* routing: decide before the output exists.

## The nearest prior art

| Work | What it does | What Firstpass adds |
|---|---|---|
| [FrugalGPT](https://arxiv.org/abs/2305.05176) (2023) | Cheapest-first cascade; a learned scorer on the answer decides whether to escalate. | The audit receipt; a gate the operator writes (their tests, their schema) rather than a learned scorer. |
| [AutoMix](https://arxiv.org/abs/2310.12963) (NeurIPS 2024) | Few-shot **self**-verification of the small model's answer, plus a POMDP **meta-verifier** that treats verification itself as noisy. | Gates are external and deterministic where the domain allows it (unit tests, schema) — not the model grading itself. Firstpass has **no** meta-verifier; see the open caveat below. |
| [CP-Router](https://arxiv.org/abs/2505.19970) (AAAI 2025) | Conformal prediction on LLM output probabilities builds a prediction set *before* committing; small set → cheap model, large set → reasoning model. | Different quantity under the same tool: Firstpass's split-conformal bound is on the **served-failure rate after the gate**, not on pre-decision answer-set size. |
| [RouteLLM](https://arxiv.org/abs/2406.18665), [Hybrid LLM](https://arxiv.org/abs/2404.14618) | Preference/quality-trained classifier picks the model from the **prompt**. | Nothing is guessed from the prompt; the decision reads the real answer. Adding a model is a config line, not a retrain. |
| [Arch-Router](https://arxiv.org/abs/2506.16655), [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) | Production routers matching queries to models by preference/semantics, pre-generation. | The closest *products*, not the closest methods. Same distinction: they select, Firstpass verifies. |

**The honest summary:** if you strip out the receipts and the guarantee, Firstpass's control
flow is FrugalGPT's, and AutoMix got to noisy-verification-aware escalation first. We would
rather write that here than have a reviewer find it.

## What is not in the literature

Three properties do not appear anywhere in that survey, in any of the four families:

1. **A tamper-evident audit of the routing decision.** The survey's Safety Analysis category
   is entirely about *attacking* routers (Rerouting LLM Routers, R2A, RerouteGuard). No entry
   emits a hash-chained, independently re-derivable receipt of which model ran, which gate
   fired, what the verdict was, and what it cost.
2. **Zero-retrain model onboarding.** Every predictive router retrains its policy to add a
   model. A Firstpass ladder rung is one config block.
3. **Threshold calibration from live downstream outcomes** (`/v1/feedback` + adaptive
   conformal). Research routers are static after training.

## Open caveat: our gates are assumed sound, and AutoMix's are not

Firstpass's headline result — 974 MBPP tasks, zero regressions — holds because MBPP unit
tests are a near-perfect oracle: the measured false-reject rate was 0.0%, so the one path by
which escalation can hurt (gate rejects a correct cheap answer, next rung gets it wrong) was
never entered. That is a property of *that workload*, not of the design.

AutoMix's POMDP meta-verifier is the published treatment of exactly this: it models
verification as unreliable and decides under that uncertainty, using a non-LLM decision layer
so verifier errors don't compound. If our imperfect-gate benchmark
([`specs/imperfect-gate-benchmark.md`](../specs/imperfect-gate-benchmark.md)) shows a
material false-reject rate on noisier gates, that is the prior art to adopt.

## Not yet comparable

Firstpass does not appear on RouterBench, RouterEval, RouterArena, or RouterXBench. Our
evidence is deep on one domain (MBPP, SWE-bench, real test gates, committed artifacts) and
absent everywhere else, so no ranking against the field is currently defensible — including
a favorable one.
