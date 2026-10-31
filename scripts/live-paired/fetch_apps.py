#!/usr/bin/env python3
"""Fetch the APPS test split and write the seeded 300-problem sample the live run used.

Usage: fetch_apps.py OUT_DIR   (writes OUT_DIR/apps300.jsonl and OUT_DIR/tests.json)

Keeps interview/competition problems with stdin/stdout tests (no starter code, no fn_name),
statements under 6000 chars, and at most 20 test cases each. The live run used the first 150.
"""

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rows = []
for off in range(0, 5000, 100):
    url = (
        "https://datasets-server.huggingface.co/rows?dataset=codeparrot%2Fapps"
        f"&config=all&split=test&offset={off}&length=100"
    )
    for attempt in range(4):
        try:
            page = json.load(urllib.request.urlopen(url, timeout=60))
            break
        except OSError:
            time.sleep(3)
    else:
        sys.exit(f"could not fetch offset {off}")
    for r in page["rows"]:
        x = r["row"]
        if (
            x["difficulty"] not in ("interview", "competition")
            or x["starter_code"].strip()
        ):
            continue
        try:
            io = json.loads(x["input_output"])
        except ValueError:
            continue
        if (
            "fn_name" in io
            or not io.get("inputs")
            or not all(isinstance(i, str) for i in io["inputs"])
        ):
            continue
        if len(x["question"]) > 6000:
            continue
        rows.append(
            {
                "task_id": x["problem_id"],
                "difficulty": x["difficulty"],
                "question": x["question"],
                "inputs": io["inputs"][:20],
                "outputs": io["outputs"][:20],
            }
        )
print("eligible", len(rows))
random.seed(20261031)
sample = random.sample(rows, 300)
(out / "tests.json").write_text(
    json.dumps(
        {
            str(r["task_id"]): {"inputs": r["inputs"], "outputs": r["outputs"]}
            for r in sample
        }
    )
)
with open(out / "apps300.jsonl", "w") as f:
    for r in sample:
        f.write(
            json.dumps(
                {
                    "task_id": r["task_id"],
                    "difficulty": r["difficulty"],
                    "question": r["question"],
                }
            )
            + "\n"
        )
