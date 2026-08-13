#!/usr/bin/env python3
"""Export a coding dataset to the VRBench v1 task format (specs/vrbench-v1.md).

    export-vrbench-tasks.py --in mbpp-974.jsonl --out vrbench-mbpp-974.jsonl

The input is the JSONL this repo's `scripts/fetch-coding-dataset.py` writes: one object per task
with `id`, `prompt`, and a list of assert-style `cases`.

## The split is the whole point

Each task's cases are divided into a **visible** set (the router's gate — what an operator would
plausibly have written) and a **hidden** set (the oracle, which the router never sees). Gate error
is only measurable because these differ: an answer the gate passes and the oracle fails is a wrong
answer served; one the gate fails and the oracle passes is a needless escalation.

Two properties matter more than they look:

**The split is deterministic**, keyed by task id, so two people exporting the same dataset get
byte-identical files and can compare results. It does not depend on dict ordering or a random seed
nobody recorded.

**The visible set is a prefix, not a sample.** Sampling visible cases uniformly from the same pool
as the hidden ones produces a gate that is unrealistically well-calibrated — it fails exactly when
the oracle would, so every verification router scores near-perfectly and gate error vanishes as an
artifact of construction. Real operator suites are written first and cover the obvious cases; the
hard ones surface later. Spec §1 Rule 2 states the requirement; this is the implementation of it.

A task with too few cases to split is dropped, loudly, with a count at the end — a task with an
empty gate or empty oracle is unscoreable, and silently keeping it would inflate whatever it
touched.
"""

from __future__ import annotations

import argparse
import json
import sys

VRBENCH_VERSION = 1

# Minimum cases needed to form both sets. Below this there is no honest split.
MIN_CASES = 3
# Fraction of cases the router may see. The rest become the oracle.
VISIBLE_FRACTION = 0.5



def convert_assert(case: str) -> str:
    """`assert f(x) == y, "msg"` -> `f(x) == y`.

    MBPP ships `assert` STATEMENTS; the VRBench runner evaluates each case as a boolean
    EXPRESSION (`eval`, not `exec`), so an un-stripped `assert` raises SyntaxError and the task
    scores zero regardless of the answer. This repo's Rust loader (`dataset.rs`) already documents
    and handles this; the first version of this exporter did not, and every task failed its oracle
    while the harness reported a served-failure rate of 1.000 — a broken scorer is indistinguishable
    from a terrible router unless you check.

    Trailing `, "message"` is dropped only at bracket depth 0 and outside string literals, so a
    comma inside a tuple or a quoted string is not mistaken for the message separator.
    """
    s = case.strip()
    if s.startswith("assert "):
        s = s[len("assert ") :]
    elif s.startswith("assert("):
        s = s[len("assert") :]

    depth = 0
    quote: str | None = None
    prev = ""
    for i, ch in enumerate(s):
        if quote:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            return s[:i].strip()
        prev = ch
    return s.strip()


def split_cases(cases: list[str]) -> tuple[list[str], list[str]]:
    """Prefix → visible (the gate), remainder → hidden (the oracle).

    Prefix rather than random sample, deliberately — see the module docstring.
    """
    n_visible = max(1, int(len(cases) * VISIBLE_FRACTION))
    # Always leave at least one case for the oracle, or the task cannot be scored at all.
    n_visible = min(n_visible, len(cases) - 1)
    return cases[:n_visible], cases[n_visible:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="input dataset JSONL")
    ap.add_argument(
        "--out", dest="dst", required=True, help="VRBench task JSONL to write"
    )
    ap.add_argument("--domain", default="code/python")
    ap.add_argument(
        "--namespace",
        default="mbpp",
        help="prefix for task ids, so ids are unique across source datasets",
    )
    args = ap.parse_args()

    written = 0
    dropped: list[str] = []

    with (
        open(args.src, encoding="utf-8") as fh,
        open(args.dst, "w", encoding="utf-8") as out,
    ):
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{args.src}:{lineno}: malformed JSON: {e}", file=sys.stderr)
                return 1

            # Field names differ across the loaders in this repo and across upstream datasets;
            # accept the known aliases rather than silently dropping a whole file. (The first
            # version of this script named only `cases`/`prompt` and dropped all 974 MBPP rows,
            # which is why the drop counter reports names and not just a total.)
            cases = (
                row.get("cases")
                or row.get("visible_cases")
                or row.get("test_list")
                or []
            )
            task_id = str(row.get("id") or row.get("task_id") or f"line{lineno}")
            prompt = row.get("prompt") or row.get("text") or ""

            if len(cases) < MIN_CASES or not prompt:
                dropped.append(task_id)
                continue

            visible, hidden = split_cases(cases)
            if not visible or not hidden:
                dropped.append(task_id)
                continue

            out.write(
                json.dumps(
                    {
                        "vrbench_version": VRBENCH_VERSION,
                        "id": f"{args.namespace}/{task_id}",
                        "domain": args.domain,
                        "prompt": prompt,
                        "visible": {
                            "kind": "pytest",
                            "cases": [convert_assert(c) for c in visible],
                        },
                        "hidden": {
                            "kind": "pytest",
                            "cases": [convert_assert(c) for c in hidden],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"wrote {written} tasks to {args.dst}")
    if dropped:
        # Named, not just counted: a silently shrunk task set changes what every published number
        # refers to, and the reader deserves to know which tasks left.
        preview = ", ".join(dropped[:5])
        more = f" (+{len(dropped) - 5} more)" if len(dropped) > 5 else ""
        print(
            f"dropped {len(dropped)} task(s) with fewer than {MIN_CASES} cases or no prompt — "
            f"an empty gate or empty oracle is unscoreable: {preview}{more}",
            file=sys.stderr,
        )
    return 0


def _selfcheck() -> None:
    """`--selfcheck` — assert the split logic without needing a dataset."""
    v, h = split_cases(["a", "b", "c", "d"])
    assert v == ["a", "b"] and h == ["c", "d"], (v, h)

    # Prefix, not sample: the visible set is the first half in order.
    v, h = split_cases(["a", "b", "c"])
    assert v == ["a"] and h == ["b", "c"], (v, h)

    # The oracle is never empty, even when the visible fraction would take everything.
    v, h = split_cases(["a", "b"])
    assert len(h) >= 1 and len(v) >= 1, (v, h)

    # Deterministic: same input, same split, every time.
    assert split_cases(["x", "y", "z", "w"]) == split_cases(["x", "y", "z", "w"])

    # assert STATEMENTS become boolean EXPRESSIONS, or every task scores zero.
    assert convert_assert("assert f(1) == 2") == "f(1) == 2"
    assert convert_assert('assert g(3) == 4, "boom"') == "g(3) == 4"
    # A comma inside brackets is not the message separator.
    assert convert_assert("assert h((1, 2)) == (3, 4)") == "h((1, 2)) == (3, 4)"
    # ...nor one inside a string literal.
    assert convert_assert('assert k("a,b") == "c"') == 'k("a,b") == "c"'
    # Already-bare expressions pass through untouched.
    assert convert_assert("f(1) == 2") == "f(1) == 2"
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
