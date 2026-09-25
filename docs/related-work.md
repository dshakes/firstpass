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

## Decision-model routers (Jev) and where Firstpass sits

A new entrant since the survey above was written: **"System One" decision models** — TypeSafe
AI's **Jev** (launched 2026-09-15; one closed-form question in, a probability distribution out;
$0.042/M input tokens, $0 output; no published accuracy benchmarks; answers are not deterministic)
— let a router ask "which tier handles this?" for a fraction of a cent instead of training a
classifier. `jev-router` and `prismhq/jev-router` use it to pick a tier and **serve that tier's
output unverified**. That is a faster way to do the same thing RouteLLM, Not Diamond, Martian, and
Sakana Fugu already do: decide before the output exists. It does not change *which family* the
decision belongs to, and per the survey's own taxonomy that family is the large majority of the
field — Firstpass and its two nearest neighbors (AutoMix, CP-Router) remain the minority that
decides after.

| System | Decides before / after generation | Verifies the output | Cold start without training | Per-query granularity | Tamper-evident audit | Served-failure guarantee |
|---|---|---|---|---|---|---|
| Jev-based routers (`jev-router`) | before | no — serves the predicted tier | yes (pretrained decision model, no local training) | yes | not reported | no |
| RouteLLM | before | no | no — classifier is trained; a new model needs retraining | yes | not reported | no |
| Not Diamond | before | not reported | not reported | not reported | not reported | not reported |
| Martian | before | not reported | not reported | not reported | not reported | not reported |
| Sakana Fugu | before | not reported | not reported | not reported | not reported | not reported |
| FrugalGPT | after | yes — learned scorer on the answer (soft, not a hard check) | no — scorer is trained | yes | not reported | not reported |
| AutoMix | after | yes — self-verification + POMDP meta-verifier | not reported | yes | not reported | not reported |
| CP-Router | before | no — builds a prediction set, doesn't check the answer | not reported | yes | not reported | no — coverage guarantee is on prediction-set size, not served output |
| Firstpass (today) | after | yes — operator-written gate (tests/schema/judge) | yes — zero-retrain; but start-rung prediction needs a warm bandit (`min_observations`) per context | yes (gate); no (start-rung, until warm) | yes — hash-chained receipt | yes — split-conformal bound |
| Firstpass + decision prior (ADR 0013) | both — prior before, gate after | yes — unchanged | yes — the prior removes the bandit's own cold-start wait | yes — including on the very first query in a context | yes — prior vector recorded as `decision_prior` | yes — unchanged |

**The thesis:** predictive routers pick the start; verification decides what is served. Those are
different jobs, and the fusion in ADR 0013 takes the cheap start a decision model is good at
*without giving up the gate that decides what ships* — Jev's own model family is exactly what
these routers already skip verifying. A pre-registered simulation of this fusion found PRIOR=STOP;
a second pre-registration replayed the same mechanism on 2,418 real MBPP outcomes using **OpenJev**
(Apache-2.0, run locally — not TypeSafe's hosted Jev) as the prior source and found **PROCEED**:
pooled $/success $0.01126 → $0.01075 (−4.6%, CI excludes 0), served-failure held. The win is
ladder-dependent and says nothing about hosted Jev's own accuracy — full numbers in
[`docs/benchmarks/openjev-prior-replay.md`](benchmarks/openjev-prior-replay.md), addendum in
[ADR 0013](adr/0013-verified-predictive-routing.md).

A third pre-registration tested blending traffic-learned pass rates into the prior: pooled
`prior+learned` $0.01094 vs `prior` $0.01075, paired diff +0.00019 [+0.00004, +0.00036] (excludes
0, worse). **Verdict: BLEND-NEUTRAL** — the prior alone stays the recommendation. That run also
traced an earlier "cost-aware learned-p" claim to a hindsight leak (it decided on each task's own
post-generation cost); that claim, including any earlier "~22% cheaper than first-pass" figure, is
withdrawn — full numbers in [`docs/benchmarks/prior-blend-replay.md`](benchmarks/prior-blend-replay.md),
correction addendum in [ADR 0013](adr/0013-verified-predictive-routing.md).

A fourth pre-registration measured the `decision` gate itself as a second gate (not the prior) on
974 served MBPP answers (111 oracle-wrong) using **local OpenJev** as the verifier: catch rate
0.2162 [0.1441, 0.2973], collateral 0.1031 [0.0834, 0.1228], AUC 0.6310 [0.5711, 0.6886] — below the
pre-registered bar (catch ≥0.30 and collateral ≤0.05). **Verdict: NOT-RECOMMENDED** at τ=0.5; this
measures OpenJev, not hosted Jev, which remains unmeasured — full numbers in
[`docs/benchmarks/decision-gate-study.md`](benchmarks/decision-gate-study.md), addendum in
[ADR 0013](adr/0013-verified-predictive-routing.md).

## Not yet comparable

Firstpass does not appear on RouterBench, RouterEval, RouterArena, or RouterXBench. Our
evidence is deep on one domain (MBPP, SWE-bench, real test gates, committed artifacts) and
absent everywhere else, so no ranking against the field is currently defensible — including
a favorable one.
