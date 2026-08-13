# Awesome-Routing-LLMs listing entry

The PR to open against
[MilkThink-Lab/Awesome-Routing-LLMs](https://github.com/MilkThink-Lab/Awesome-Routing-LLMs)
**after** the paper is on arXiv. The list's rows are `title | venue | tags | code`, and ~128 of its
~130 entries are papers — a code-only submission is very unlikely to be accepted, which is why the
paper is the gating step rather than an optional extra.

## Placement

**Verification Routing → Self-Assessment**, alongside AutoMix, CP-Router, DiSRouter, and
"Confident or Seek Stronger". That is the correct cell: the decision is made after generation from
the model's own output, not from the prompt.

Do **not** propose it under Pre-judgment Routing. The bandit start-rung is a pre-judgment component,
but it selects where to *start*, not what to serve, and filing it there would misrepresent the
mechanism to exactly the readers most able to notice.

## Proposed row

| Title | Venue | Tags | Code |
|---|---|---|---|
| Verified Cascade Routing: Serving the Cheapest Model That Provably Passes a Gate, With an Auditable Record of Why | arXiv `<ID>` | Code, Safety, Agent | [firstpass](https://github.com/dshakes/firstpass) |

Tag rationale, matching how the list uses them: **Code** (evaluated on MBPP with executable test
gates), **Safety** (the served-failure bound and the tamper-evident audit record), **Agent** (the
proxy sits in agent serving paths). Drop any tag the maintainers' own usage does not support rather
than arguing for it.

## PR description (draft)

> Adds *Verified Cascade Routing* under Verification Routing → Self-Assessment.
>
> The paper studies post-generation routing: serve the cheapest model whose output passes an
> operator-supplied gate, escalate one rung on failure. It relates the approach explicitly to
> AutoMix and CP-Router, which are the nearest prior art in this cell, and does not claim the
> cascade mechanism as novel.
>
> Its contributions are a split-conformal bound on the served-failure rate, a hash-chained audit
> record of routing decisions that an external auditor can re-derive, and evaluation under
> RouterBench's own AIQ metric against the Zero Router bar.
>
> It also reports a negative result relevant to this list's Benchmark & Evaluation section:
> RouterBench cannot score verification routers, because it stores one response per model with no
> executable oracle, and its MBPP prompts are paraphrases that cannot be re-scored against canonical
> tests. Happy to add that as a note if useful.

## Before opening it

- The arXiv ID must exist. Replace `<ID>`.
- Read the list's `CONTRIBUTING` if present and match its row formatting exactly — a
  formatting-mismatched PR is the cheapest possible reason to be rejected.
- Check the entry does not already exist under another name.
