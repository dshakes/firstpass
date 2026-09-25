# Pre-registration: prior + learned blend, and the `decision` gate's error rates

Written 2026-09-25, **before either measurement ran.** Everything below is fixed from this point.

## Finding that shaped this spec (already verified in code)

`costaware::serve` (`costaware.rs:145`) makes its decision from the task's own **realized** `c0.cost_usd` and
`c1.cost_usd`. `PassPredictor` buckets tasks by the realized cheap-rung cost. Both include output tokens that exist only
**after** generation, and hard tasks produce longer outputs. So the existing "cost-aware (learned p)" arm uses hindsight,
and its $/success is optimistic. It stays in reports as `learned-p (hindsight)`. It must never be the comparison target.

## Study A: does learned traffic statistics blended into the prior beat the prior alone?

- **Data:** the same three matrices and committed OpenJev priors as `specs/openjev-prior-replay-ab.md`, with the same
  2-fold cross-fitting and the same ex-ante price (the cross-fitted mean cost per rung).
- **Ex-ante feature:** the MBPP prompt's character length (`text` field), which is known before generation. Buckets are
  the quartiles of the calibration fold.
- **Arms:**
  1. `first-pass`
  2. `prior` (as before)
  3. `learned-p (ex-ante)`: the bucketed rung-0 gate-pass rate from the calibration fold
  4. **`prior+learned`**: the proxy's semantics. The posterior mean is
     `(s·prior_r0 + passes_b)/(s + seen_b)` with `s = 10` (the `[escalation.prior] strength` default), and `passes_b`
     and `seen_b` come from the calibration-fold bucket.
  5. `learned-p (hindsight)`: for reference only
  6. `always-top`
- Every ex-ante arm decides with the shared expected-cost argmin.
- **Verdict (pooled):**
  - **BLEND-HELPS** if `prior+learned` has lower $/success than `prior`, with the paired bootstrap CI excluding 0 and
    served-failure no more than 1 pp higher.
  - **Otherwise BLEND-NEUTRAL**: the prior alone is the recommendation.
  - The leak size (`learned-p (hindsight)` minus `learned-p (ex-ante)`) is reported, not gated.

## Study B: the `decision` gate's error rates (OpenJev as the verifier)

- **Candidates:** the 974 served MBPP answers in `~/vrb-cascade.jsonl`. They were already served, so they already passed
  the existing gate. **Labels** come from VRBench's hidden-test oracle, run in the fail-closed sandbox.
- **Verifier:** the proxy's `DecisionGate` request shape (a `noul` question, candidate as data, threshold 0.5), sent to
  local OpenJev at `http://127.0.0.1:8080`. This measures OpenJev, not hosted Jev.
- **Metrics:** each has a bootstrap 95% CI.
  - **catch rate** = P(reject | oracle-wrong): wrong answers the existing gate let through that this gate would stop.
  - **collateral** = P(reject | oracle-right): needless escalations it would add.
  - AUC of the `noul` score against the oracle.
  - Abstain rate.
- **Verdict:**
  - **VALUE-ADD** as a second gate if catch rate ≥ 0.30 and collateral ≤ 0.05, both judged on point estimates, with a
    CI reported.
  - **Otherwise NOT-RECOMMENDED** at τ=0.5.
  - A τ sweep is reported as exploratory and cannot change the verdict.
- **Degeneracy guard:** if fewer than 20 oracle-wrong answers exist, report **UNDERPOWERED** instead of a verdict.
