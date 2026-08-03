#!/usr/bin/env python3
"""Fetch a BigCodeBench subset as JSONL for `firstpass-bench --coding-live`.

Why a subset, and why this one: BigCodeBench tasks import real third-party libraries, and the
benchmark sandbox runs with `--network none` (ADR 0002), so anything not already in the image
cannot be installed at run time. The usual answer is a multi-gigabyte image carrying every
dependency. The cheap answer is this: each task declares its imports in `libs`, so selecting the
tasks whose libs are **entirely standard library** gives real BigCodeBench problems that run on a
plain `python:3.x` image with no custom build at all.

That is a subset, and a subset is only honest when it is labelled. The output filename and the
manifest written beside it both record the split, the filter, and the count — quote those
alongside any number produced from this file.

Usage:
    scripts/fetch-bigcodebench.py --out bcb-stdlib.jsonl --limit 40
    FIRSTPASS_CODING_DATASET=bcb-stdlib.jsonl cargo run -p firstpass-bench -- --coding-live
"""

import argparse
import ast
import json
import sys
import urllib.parse
import urllib.request

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "bigcode/bigcodebench"
PAGE = 100  # the datasets-server per-request cap


def fetch_page(split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_URL}?{q}", timeout=60) as r:
        return [row["row"] for row in json.load(r)["rows"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v0.1.4")
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

    # `sys.stdlib_module_names` is the interpreter's own answer, so this tracks the Python the
    # sandbox runs rather than a list that quietly goes stale.
    stdlib = set(sys.stdlib_module_names)

    kept, scanned, skipped_libs = [], 0, {}
    while scanned < args.scan and len(kept) < args.limit:
        rows = fetch_page(args.split, scanned, min(PAGE, args.scan - scanned))
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

    with open(args.out, "w") as f:
        for t in kept:
            f.write(json.dumps(t) + "\n")

    manifest = {
        "dataset": DATASET,
        "split": args.split,
        "filter": "third-party libs allowed"
        if args.allow_third_party
        else "stdlib-only libs",
        "scanned": scanned,
        "kept": len(kept),
        "task_ids": [t["task_id"] for t in kept],
    }
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"wrote {len(kept)} tasks to {args.out} (scanned {scanned} of split {args.split})"
    )
    if skipped_libs:
        top = sorted(skipped_libs.items(), key=lambda kv: -kv[1])[:8]
        print("skipped for third-party libs: " + ", ".join(f"{k}×{v}" for k, v in top))
    print(
        f"manifest: {args.out}.manifest.json — quote its split/filter/count with any result"
    )
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
