# paper/ — the arXiv submission

`main.tex` is the paper. Build it with [tectonic](https://tectonic-typesetting.github.io/)
(no local TeX install needed):

```
cd paper && tectonic main.tex
```

## The draft guard

Every empirical number in the paper is a `\result{...}` placeholder that renders in red as
**[TODO: ...]**, and the document prints a `DRAFT — N unfilled results. Not submittable.` banner
while any remain. The build also emits a machine-readable count:

```
tectonic --print main.tex 2>&1 | grep FIRSTPASS_UNFILLED_RESULTS
```

**Do not submit while that count is non-zero**, and do not replace a placeholder with a number you
have not read out of a committed artifact under `docs/benchmarks/`. The guard exists because a
half-filled paper and a finished one otherwise look identical.

## Filling the results

Every table comes from one measured matrix, replayed offline. Producing that matrix is the only
step that costs money:

```
# 1. Fetch the dataset (free).
scripts/fetch-coding-dataset.py --dataset mbpp --out mbpp-974.jsonl --limit 974

# 2. Measure the matrix (PAID — ~3,900 calls, ~$9 on the haiku→sonnet ladder).
#    Checkpoints per task, so an interruption costs only the tail; re-run to resume.
ANTHROPIC_API_KEY=... \
FIRSTPASS_CODING_DATASET=mbpp-974.jsonl \
FIRSTPASS_CODING_JUDGE=anthropic/claude-haiku-4-5 \
FIRSTPASS_CODING_CHECKPOINT=mbpp-sonnet.jsonl \
cargo run -p firstpass-bench -- --coding-policy

# 3. Every table in the paper, from that checkpoint (FREE — no key, no sandbox, no spend).
cargo run -p firstpass-bench -- --replay mbpp-sonnet.jsonl
```

Step 3 prints, in order:

| paper section | study | source |
|---|---|---|
| §6.1 Does routing beat one model | policy comparison + paired CIs | `coding_policy::replay` |
| §6.2 Under RouterBench's metric | AIQ, NDCH, Zero Router bar | `routerbench::evaluate` |
| §6.3 When the gate is wrong | τ sweep, split-conformal λ | `sweep::sweep` |
| §6.4 Does a meta-verifier help | AutoMix-style rule, held-out | `metaverify::study` |

Commit the checkpoint and the rendered report under `docs/benchmarks/` with a self-labeling header
before quoting any number from them, matching the convention in `mbpp-live-base.txt`.

## The ladder arm

The paper reports savings against more than one ceiling because the saving depends on it. Re-run
step 2 with `FIRSTPASS_CODING_LADDER=anthropic/claude-haiku-4-5,anthropic/claude-opus-4-8` into a
second checkpoint (~$31), and replay both. One checkpoint file can hold several ladders; `--replay`
refuses to average them and asks you to name one via `FIRSTPASS_CODING_LADDER`.

## After it is filled

1. `tectonic --print main.tex 2>&1 | grep FIRSTPASS_UNFILLED_RESULTS` must report `=0`.
2. Submit to arXiv (cs.LG or cs.SE).
3. Open the PR against [Awesome-Routing-LLMs](https://github.com/MilkThink-Lab/Awesome-Routing-LLMs)
   adding the entry under **Verification Routing → Self-Assessment** — see `listing-entry.md`.
