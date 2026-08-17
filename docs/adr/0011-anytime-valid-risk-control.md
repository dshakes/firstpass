# ADR 0011 — Anytime-valid risk control: a bound that holds at every round, not on average

Status: **accepted — LIVE on the serving hot path since v0.8.0** · 2026-08-14, revised 2026-08-16

Supersedes nothing. Adds a fourth calibration regime alongside `conformal`, `ltt`, and ACI.
Related: [ADR 0008](0008-elastic-verification.md) (conformal guarantee over verify/skip),
`crates/firstpass-core/src/eprocess.rs`, `crates/firstpass-proxy/src/calibrate.rs`, SPEC §10.1.

## Context

Firstpass ships three risk-control regimes, and each is valid for what it claims:

| module | guarantee | assumption / scope |
|---|---|---|
| `conformal::calibrate` | served-failure ≤ α at confidence 1−δ (Hoeffding) | **exchangeability**; λ fixed once |
| `ltt` | FWER-controlled, exact-binomial, distribution-free | **fixed-sample**: one calibration set, one decision |
| `conformal::AdaptiveConformal` | Gibbs–Candès ACI | its own doc: the ***long-run*** rate converges to α |

The gap is not "no guarantee". It is narrower, and it only becomes visible when you ask what the
proxy actually does with these. `calibrate.rs` exists to recalibrate **from live deferred feedback**
— that is the product feature, "learns your quality bar". In that regime:

1. **A long-run average is not a promise about now.** A process can sit above α for an arbitrarily
   long stretch and still converge, because the excess is amortised over a horizon the operator may
   never reach. An operator holding an error budget needs the bound to hold at the round they are
   on, not in expectation over a future they will not see.
2. **Repeated recalibration does not compose.** LTT controls the family-wise error rate over *one*
   pre-specified test sequence on *one* calibration set. Re-run it whenever more feedback lands and
   adopt the winning threshold each time, and that is optional stopping on a growing stream. Every
   individual run is valid; the sequence of adoptions is not. Nothing in the code was wrong — the
   guarantee simply never covered the loop it was being used inside.

Honest framing for the paper and the pitch: **the per-batch guarantees are sound; the per-round
guarantee under continuous recalibration did not exist.**

## Decision

Implement **Conformal Selective Acting** (Khosravi & Huo 2026,
[arXiv:2605.20270](https://arxiv.org/abs/2605.20270)) in `firstpass-core/src/eprocess.rs`:

- One **e-process per candidate threshold** on a **Bonferroni** grid (`n` thresholds each at level
  `δ/n`, so the crossing level is `n/δ`).
- Per served item with score `s` and correctness `c`, every threshold `λ ≤ s` — i.e. every threshold
  that *would have served this item* — is updated by the betting factor
  `E_λ ← E_λ · (1 + bet · (α − err))`. Under H₀ the expected multiplier is ≤ 1, which is the
  supermartingale property.
- **Ville's inequality** then bounds the process at *all* stopping times simultaneously, so the
  guarantee survives optional stopping. Re-reading the certified threshold as often as you like
  costs nothing in validity — which is exactly what a live recalibration loop needs.
- The served threshold is the **least conservative certified** one.

Two design points the tests forced, both of which the first draft got wrong:

**Certification is live, not latched.** The first implementation recorded "this threshold crossed"
permanently. Under a regime change it then kept serving at a threshold the new regime did not
justify: realized failure **0.312 against α = 0.20**. That is the fixed-sample failure mode
reappearing *inside* the online method — the subtlest possible version of the bug this ADR exists to
fix. Certification is now evaluated against the current e-value on every read.

**E-values are capped** (`DEFAULT_CAP = 10 × crossing level`). Truncating a supermartingale from
above leaves it a supermartingale, so type-I control is untouched; the cap is about
*responsiveness*. Uncapped, a long clean stretch banks unbounded surplus evidence that a later shift
must burn off before de-certifying — the bound holds asymptotically while realized failure sits
above α for thousands of rounds. Which is the long-run-average failure mode wearing a new hat.

Exposed as `firstpass calibrate --method eprocess`, so it is reachable rather than dead code.

## Consequences

**The cost, stated plainly.** Anytime validity is not free. At the same α this serves **less**
traffic than ACI: it will not lower the threshold until evidence has accumulated, and Bonferroni
makes each threshold individually harder to certify. That is the trade — ACI tracks aggressively and
promises an average; this promises every round and pays in conservatism. Neither is strictly better.
The operator picks the regime, and the module documents the trade rather than burying it.

**Fails closed.** Before anything is certified, `certified_threshold()` returns `None` and the caller
keeps its existing behaviour. Serving on an uncertified threshold would be precisely the
unproven-claim failure the module exists to prevent, so "I have not proven anything yet" must never
render as a number.

**The grid must be quantised, and that is not a detail.** The crossing level is `n/δ`, so evidence
required per threshold grows *linearly* in the number of distinct candidate scores. Deriving the
grid from the observed score support — which is what LTT does, for free, because its exact-binomial
test does not pay per candidate — means a judge emitting full-precision floats over 10k traces
produces ~10k candidates and a crossing level of **200,000**. Nothing would ever certify, and the
report would say "no threshold certified", which is indistinguishable from honest insufficient data.
A silent power collapse that reads as a normal result is worse than a crash. Scores are quantised to
2dp, bounding the grid at 101 points regardless of store size; resolution finer than 0.01 on a gate
score is not information anyone acts on. Caught in review, after this ADR's own text warned about
Bonferroni cost and the implementation then took the unbounded path anyway.

**On the hot path since v0.8.0.** Threshold precedence is **e-process > ACI > fixed config**: the
e-process wins whenever it has certified something, because it is the only one of the three whose
bound holds at the round it is read. It **fails closed by omission** — before anything is certified
`certified_threshold()` returns `None` and the caller keeps its existing behaviour, so enabling it
cannot change serving until the evidence exists. `/v1/feedback` closes the loop, using the score of
the attempt actually served (read back from the stored trace, never the optional payload field —
only thresholds that *would have served* an item may be updated, which is the condition Ville's
inequality needs).

**One controller per process, not per tenant.** `AppState.eprocess` is a single
`Arc<Mutex<EProcessRiskControl>>`, so deferred feedback from every tenant updates the same
e-processes. Raised in review, and the mechanism is real: the guarantee is over *one* stream, and
pooling tenants with different quality distributions means the certified threshold reflects a
mixture nobody is actually served from. A tenant with a strict gate drags the shared threshold up;
one with a loose gate drags it down.

Two things bound the severity. It is **pre-existing, not introduced here** —
`AdaptiveConformal` has had the identical shape since it was wired into live serving, so this
describes the calibration layer generally rather than the e-process specifically. And multi-tenancy
is itself **experimental and default-off** (`tenant_auth.rs`, ADR 0004 §D7: not yet independently
reviewed, not to be relied on as a hard isolation boundary). A single-tenant deployment — which is
every deployment today — has one distribution and is unaffected.

The fix, when tenancy graduates, is to key both controllers by tenant, at the cost of slower
certification per tenant: the Bonferroni crossing level does not shrink because the traffic is
split. Recorded here rather than patched now, because doing it properly is a change to the tenancy
model and not to this ADR's mechanism.

**Verified, not field-proven.** It has never certified a threshold against production traffic; that
requires accumulated deferred feedback. `firstpass_eprocess_rounds_total` is the signal that the loop
is turning. The paper states this limitation explicitly rather than leaving it to be discovered.

## Evidence

The tests are the argument, and they run offline, deterministically, for $0 (seeded LCG, no ambient
randomness — an auditor reproduces them exactly):

- `coverage_holds_at_every_round_under_a_regime_change` — 40 seeded streams with an abrupt
  mid-stream regime change; asserts the realized rate at **every** round after certification, not
  just at the end.
- `split_conformal_breaches_where_the_e_process_holds` — the contrast that makes the contribution.
  On the same stream, split conformal's realized rate **exceeds α** while the e-process does not.
  It asserts the breach explicitly, so if the stream ever stops being adversarial the test fails
  rather than quietly proving nothing.
- `the_cap_bounds_how_long_a_stale_certification_survives` — capped vs uncapped de-certification lag.
- Plus fail-closed, Bonferroni scaling, and non-serving-threshold isolation.

**Every assertion was mutation-tested** in a throwaway worktree: re-latching certification, removing
the Bonferroni correction, updating thresholds that would not have served, and removing the cap each
fail exactly their own test and no other. The cap mutation initially **survived** — nothing covered
it — so the test above was written and the mutation re-run. That is the process working; an
uncovered constant is indistinguishable from a decorative one.
