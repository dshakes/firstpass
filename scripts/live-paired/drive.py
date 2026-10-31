#!/usr/bin/env python3
"""Drive MBPP or APPS tasks through a live proxy, stopping at a spend cap.

Usage: FP_BASE=http://127.0.0.1:8080 FP_CAP_USD=5 [FP_MAX_TOKENS=4096] drive.py tasks.jsonl out.jsonl
Resumable: task ids already in out.jsonl are skipped. Spend is read back from the proxy's own
/v1/receipts and /v1/experiments, so the cap covers the off-path challenger too.
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ["FP_BASE"]
CAP_USD = float(os.environ["FP_CAP_USD"])  # hard ceiling; all runs together stay under the $30 approval
rows = [json.loads(l) for l in open(sys.argv[1])]
out = open(sys.argv[2], "a")
done = {json.loads(l)["task_id"] for l in open(sys.argv[2])} if out.tell() else set()


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=30))


def spend():
    inc = get("/v1/receipts?limit=1")["summary"]["spent_usd"]
    ch = sum(
        (c.get("paired") or {}).get("spent_today_usd", 0)
        for c in get("/v1/experiments")["configured"]
    )
    return inc, ch


def ask(r):
    if "question" in r:  # APPS: stdin/stdout, tests hidden from the model
        prompt = (f"<task id={r['task_id']}>\n{r['question']}\n\nRead input from stdin and write the "
                  "answer to stdout. Reply with only the Python 3 program in one ```python block.")
    else:  # MBPP: the task's asserts are part of the prompt, as in the standard setup
        prompt = (f"{r['text']}\nYour code must pass these tests:\n<tests>\n"
                  + "\n".join(r["test_setup_code"].splitlines() + r["test_list"])
                  + "\n</tests>\nReply with only the Python code in one ```python block.")
    body = json.dumps(
        {
            "model": "claude-sonnet-5",
            "max_tokens": int(os.environ.get("FP_MAX_TOKENS", "1024")),
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/messages",
        body,
        {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-firstpass-session": f"mbpp-{r['task_id']}",
        },
    )
    t = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=900)
        return {
            "task_id": r["task_id"],
            "status": resp.status,
            "ms": int((time.time() - t) * 1000),
            "firstpass": {
                k: v
                for k, v in resp.headers.items()
                if k.lower().startswith("x-firstpass")
            },
        }
    except Exception as e:  # recorded, never retried silently
        return {"task_id": r["task_id"], "error": str(e)[:300]}


todo = [r for r in rows if r["task_id"] not in done]
with ThreadPoolExecutor(4) as pool:
    for i in range(0, len(todo), 20):
        inc, ch = spend()
        exp = get("/v1/experiments")
        print(
            json.dumps(
                {
                    "sent": len(done) + i,
                    "incumbent_usd": round(inc, 4),
                    "challenger_usd": round(ch, 4),
                    "experiments": exp,
                }
            ),
            flush=True,
        )
        if inc + ch >= CAP_USD:
            print("STOP: spend cap", flush=True)
            break
        for res in pool.map(ask, todo[i : i + 20]):
            out.write(json.dumps(res) + "\n")
            out.flush()
time.sleep(30)  # let off-path challenger calls finish
inc, ch = spend()
print(
    json.dumps(
        {
            "final": True,
            "incumbent_usd": inc,
            "challenger_usd": ch,
            "experiments": get("/v1/experiments"),
            "receipts": get("/v1/receipts?limit=1")["summary"],
        }
    ),
    flush=True,
)
