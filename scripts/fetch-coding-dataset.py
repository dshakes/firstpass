#!/usr/bin/env python3
"""Fetch a coding-benchmark subset as JSONL for `firstpass-bench --coding-live` / `--coding-policy`.

Two datasets, because the two answer different questions. **MBPP** (`--dataset mbpp`) is the
workload the published guarantee artifact was measured on, so it is the one to use when a new
number has to be comparable with what is already claimed. **BigCodeBench**
(`--dataset bigcodebench`) is harder and library-heavy — see the note below on why only part of
it is taken.

Why a BigCodeBench subset: BigCodeBench tasks import real third-party libraries, and the
benchmark sandbox runs with `--network none` (ADR 0002), so anything not already in the image
cannot be installed at run time. The usual answer is a multi-gigabyte image carrying every
dependency. The cheap answer is this: each task declares its imports in `libs`, so selecting the
tasks whose libs are **entirely standard library** gives real BigCodeBench problems that run on a
plain `python:3.x` image with no custom build at all.

That is a subset, and a subset is only honest when it is labelled. The output filename and the
manifest written beside it both record the split, the filter, and the count — quote those
alongside any number produced from this file.

MBPP needs no such filter: its tasks are plain standard-library Python by construction. Rows are
taken as a **stated prefix in dataset order**, not sampled — a prefix is reproducible without
recording a seed, and the manifest says so plainly so nobody mistakes it for a random draw.

Usage:
    scripts/fetch-coding-dataset.py --dataset mbpp --out mbpp-200.jsonl --limit 200
    scripts/fetch-coding-dataset.py --dataset bigcodebench --out bcb-stdlib.jsonl --limit 40
    FIRSTPASS_CODING_DATASET=mbpp-200.jsonl cargo run -p firstpass-bench -- --coding-policy
"""

import argparse
import ast
import json
import sys
import urllib.parse
import urllib.request

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the datasets-server per-request cap

# `google-research/mbpp` is the canonical id but the datasets-server reports it renamed and
# refuses it; this mirror serves the same 974 rows and the same field names.
SOURCES = {
    "bigcodebench": {"dataset": "bigcode/bigcodebench", "config": "default", "split": "v0.1.4"},
    "mbpp": {"dataset": "Muennighoff/mbpp", "config": "full", "split": "test"},
    "swebench": {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "config": "default",
        "split": "test",
    },
}


def swebench_image(instance_id: str) -> str:
    """The published eval image for an instance.

    The tag mangles the id: `astropy__astropy-12907` becomes
    `sweb.eval.x86_64.astropy_1776_astropy-12907`. `1776` is just the separator upstream chose
    for `__`; getting it wrong yields a 404 at pull time rather than anything diagnosable, so it
    is derived here once rather than hand-written per instance.
    """
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_")



def fetch_page(src: dict, split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "dataset": src["dataset"],
            "config": src["config"],
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_URL}?{q}", timeout=60) as r:
        return [row["row"] for row in json.load(r)["rows"]]


def write_out(path: str, kept: list[dict], src: dict, split: str, filt: str, scanned: int) -> None:
    """Write the JSONL and a manifest beside it.

    Rows are a **prefix in dataset order**, never a sample: a prefix reproduces without recording
    a seed. The manifest says which dataset, split, filter and count produced the file, because a
    subset is only honest when whatever quotes it can name what was left out.
    """
    with open(path, "w") as f:
        for t in kept:
            f.write(json.dumps(t) + "\n")
    with open(path + ".manifest.json", "w") as f:
        json.dump(
            {
                "dataset": src["dataset"],
                "config": src["config"],
                "split": split,
                "selection": "first N rows in dataset order (a stated prefix, not a random sample)",
                "filter": filt,
                "scanned": scanned,
                "kept": len(kept),
                "task_ids": [t.get("task_id", t.get("instance_id")) for t in kept],
            },
            f,
            indent=2,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SOURCES), default="bigcodebench")
    ap.add_argument("--split", default=None, help="override the source's default split")
    ap.add_argument("--out", default="bcb-stdlib.jsonl")
    ap.add_argument(
        "--limit", type=int, default=40, help="how many matching tasks to keep"
    )
    ap.add_argument(
        "--scan", type=int, default=400, help="how many rows to scan for matches"
    )
    ap.add_argument(
        "--allow-third-party",
        action="store_true",
        help="keep every task regardless of libs (needs a sandbox image carrying them)",
    )
    args = ap.parse_args()
    src = SOURCES[args.dataset]
    split = args.split or src["split"]

    # `sys.stdlib_module_names` is the interpreter's own answer, so this tracks the Python the
    # sandbox runs rather than a list that quietly goes stale.
    stdlib = set(sys.stdlib_module_names)

    # MBPP is plain stdlib Python by construction, and its rows carry `test_list` rather than a
    # unittest class, so the lib filter neither applies nor is needed.
    if args.dataset == "swebench":
        kept, scanned = [], 0
        while scanned < args.limit:
            rows = fetch_page(src, split, scanned, min(PAGE, args.limit - scanned))
            if not rows:
                break
            scanned += len(rows)
            for r in rows:
                # FAIL_TO_PASS / PASS_TO_PASS arrive as JSON *strings*, not lists.
                f2p = r["FAIL_TO_PASS"]
                p2p = r["PASS_TO_PASS"]
                kept.append(
                    {
                        "instance_id": r["instance_id"],
                        "repo": r["repo"],
                        "base_commit": r["base_commit"],
                        "problem_statement": r["problem_statement"],
                        "test_patch": r["test_patch"],
                        "fail_to_pass": json.loads(f2p) if isinstance(f2p, str) else f2p,
                        "pass_to_pass": json.loads(p2p) if isinstance(p2p, str) else p2p,
                        "image": swebench_image(r["instance_id"]),
                        "difficulty": r.get("difficulty", ""),
                    }
                )
        write_out(args.out, kept, src, split, "none (SWE-bench Verified, human-validated)", scanned)
        print(f"wrote {len(kept)} SWE-bench instances to {args.out}")
        print("each needs its own multi-GB eval image; pull them BEFORE the run (no network at eval)")
        return 0 if kept else 1

    if args.dataset == "mbpp":
        kept, scanned = [], 0
        while scanned < args.limit:
            rows = fetch_page(src, split, scanned, min(PAGE, args.limit - scanned))
            if not rows:
                break
            scanned += len(rows)
            kept.extend(
                {
                    "task_id": r["task_id"],
                    "text": r["text"],
                    "code": r.get("code", ""),
                    "test_list": r["test_list"],
                }
                for r in rows
            )
        write_out(args.out, kept, src, split, "none (MBPP is stdlib by construction)", scanned)
        print(f"wrote {len(kept)} MBPP tasks to {args.out}")
        return 0 if kept else 1

    kept, scanned, skipped_libs = [], 0, {}
    while scanned < args.scan and len(kept) < args.limit:
        rows = fetch_page(src, split, scanned, min(PAGE, args.scan - scanned))
        if not rows:
            break
        scanned += len(rows)
        for row in rows:
            libs = row.get("libs") or []
            if isinstance(libs, str):
                # Shipped as a Python list *repr* (`"['random', 'itertools']"`), not JSON, so
                # `json.loads` rejects it on the single quotes. `ast.literal_eval` reads that
                # shape and, unlike `eval`, cannot execute anything from the dataset.
                libs = ast.literal_eval(libs) if libs.strip() else []
            outside = sorted(set(libs) - stdlib)
            if outside and not args.allow_third_party:
                for lib in outside:
                    skipped_libs[lib] = skipped_libs.get(lib, 0) + 1
                continue
            kept.append(
                {
                    "task_id": row["task_id"],
                    "instruct_prompt": row.get("instruct_prompt")
                    or row.get("complete_prompt"),
                    "entry_point": row.get("entry_point", "task_func"),
                    "test": row["test"],
                    "libs": libs,
                }
            )
            if len(kept) >= args.limit:
                break

    write_out(
        args.out,
        kept,
        src,
        split,
        "third-party libs allowed" if args.allow_third_party else "stdlib-only libs",
        scanned,
    )
    print(f"wrote {len(kept)} tasks to {args.out} (scanned {scanned} of split {split})")
    if skipped_libs:
        top = sorted(skipped_libs.items(), key=lambda kv: -kv[1])[:8]
        print("skipped for third-party libs: " + ", ".join(f"{k}×{v}" for k, v in top))
    print(
        f"manifest: {args.out}.manifest.json — quote its split/filter/count with any result"
    )
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
