## Prior + learned blend — Study A (real MBPP matrices)

Pre-registration: `specs/prior-blend-and-decision-gate.md`. `prior`/`prior+learned` are the two arms the verdict compares; the rest are context. `learned-p (hindsight)` is reference only — it uses the task's own realized cost and must never be the comparison target.

### ~/fp-mbpp-974-sonnet-k5.jsonl (n = 974)

| arm | success | 95% CI | $/success | 95% CI | served-failure | escalated | fallback (no prior) |
|---|---|---|---|---|---|---|---|
| first-pass | 0.9230 | [0.9066, 0.9394] | $0.01697 | [$0.01545, $0.01851] | 0.0770 | 18% | 0% |
| prior | 0.9240 | [0.9066, 0.9405] | $0.01590 | [$0.01457, $0.01726] | 0.0760 | 13% | 0% |
| learned-p (ex-ante) | 0.9261 | [0.9086, 0.9425] | $0.01660 | [$0.01521, $0.01806] | 0.0739 | 15% | 0% |
| prior+learned | 0.9261 | [0.9086, 0.9425] | $0.01620 | [$0.01489, $0.01757] | 0.0739 | 13% | 0% |
| learned-p (hindsight) | 0.9271 | [0.9107, 0.9425] | $0.01348 | [$0.01235, $0.01469] | 0.0729 | 9% | 0% |
| always-top | 0.9333 | [0.9179, 0.9487] | $0.01500 | [$0.01401, $0.01606] | 0.0667 | 0% | 0% |

### ~/fp-mbpp-974-opus-k5.jsonl (n = 470)

| arm | success | 95% CI | $/success | 95% CI | served-failure | escalated | fallback (no prior) |
|---|---|---|---|---|---|---|---|
| first-pass | 0.9149 | [0.8894, 0.9404] | $0.02003 | [$0.01773, $0.02246] | 0.0851 | 24% | 0% |
| prior | 0.9170 | [0.8936, 0.9426] | $0.01955 | [$0.01728, $0.02193] | 0.0830 | 21% | 0% |
| learned-p (ex-ante) | 0.9149 | [0.8894, 0.9404] | $0.01988 | [$0.01758, $0.02227] | 0.0851 | 20% | 0% |
| prior+learned | 0.9149 | [0.8894, 0.9404] | $0.01993 | [$0.01761, $0.02236] | 0.0851 | 23% | 0% |
| learned-p (hindsight) | 0.9170 | [0.8915, 0.9404] | $0.01653 | [$0.01469, $0.01829] | 0.0830 | 10% | 0% |
| always-top | 0.9298 | [0.9064, 0.9532] | $0.02050 | [$0.01897, $0.02212] | 0.0702 | 0% | 0% |

### ~/fp-mbpp-974-openai.jsonl (n = 974)

| arm | success | 95% CI | $/success | 95% CI | served-failure | escalated | fallback (no prior) |
|---|---|---|---|---|---|---|---|
| first-pass | 0.9230 | [0.9066, 0.9384] | $0.00136 | [$0.00112, $0.00162] | 0.0770 | 13% | 0% |
| prior | 0.9230 | [0.9066, 0.9384] | $0.00136 | [$0.00112, $0.00162] | 0.0770 | 13% | 0% |
| learned-p (ex-ante) | 0.9230 | [0.9066, 0.9384] | $0.00136 | [$0.00112, $0.00162] | 0.0770 | 13% | 0% |
| prior+learned | 0.9230 | [0.9066, 0.9384] | $0.00136 | [$0.00112, $0.00162] | 0.0770 | 13% | 0% |
| learned-p (hindsight) | 0.9230 | [0.9066, 0.9384] | $0.00136 | [$0.00112, $0.00162] | 0.0770 | 13% | 0% |
| always-top | 0.9292 | [0.9127, 0.9456] | $0.00424 | [$0.00396, $0.00452] | 0.0708 | 0% | 0% |

### POOLED (n = 2418)

| arm | success | 95% CI | $/success | 95% CI | served-failure | escalated | fallback (no prior) |
|---|---|---|---|---|---|---|---|
| first-pass | 0.9214 | [0.9103, 0.9318] | $0.01126 | [$0.01048, $0.01213] | 0.0786 | 17% | 0% |
| prior | 0.9222 | [0.9107, 0.9326] | $0.01075 | [$0.01005, $0.01155] | 0.0778 | 14% | 0% |
| learned-p (ex-ante) | 0.9227 | [0.9111, 0.9326] | $0.01109 | [$0.01036, $0.01190] | 0.0773 | 15% | 0% |
| prior+learned | 0.9227 | [0.9111, 0.9326] | $0.01094 | [$0.01021, $0.01177] | 0.0773 | 15% | 0% |
| learned-p (hindsight) | 0.9235 | [0.9123, 0.9338] | $0.00919 | [$0.00857, $0.00989] | 0.0765 | 11% | 0% |
| always-top | 0.9309 | [0.9206, 0.9409] | $0.01175 | [$0.01119, $0.01237] | 0.0691 | 0% | 0% |

**Pooled `$/success` (prior+learned − prior): +0.00019 [+0.00004, +0.00036]**

Hindsight leak (learned-p (hindsight) − learned-p (ex-ante), pooled `$/success`): -0.00190. Reported, not gated.

**Verdict: BLEND-NEUTRAL**

