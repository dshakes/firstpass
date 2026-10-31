#!/usr/bin/env python3
"""APPS gate: run the candidate on each stdin case in a no-network container, compare stdout."""

import json
import re
import subprocess
import sys

TESTS = json.load(open(sys.argv[1]))
HARNESS = r"""
import json, subprocess, sys
job = json.load(sys.stdin)
outs = []
for i in job["inputs"]:
    try:
        r = subprocess.run([sys.executable, "-c", job["code"]], input=i, capture_output=True, text=True, timeout=4)
        outs.append(r.stdout if r.returncode == 0 else None)
    except subprocess.TimeoutExpired:
        outs.append(None)
print(json.dumps(outs))
"""

inp = json.load(sys.stdin)
text = "\n".join(
    m["content"]
    if isinstance(m["content"], str)
    else "".join(p.get("text", "") for p in m["content"])
    for m in inp["request"]["messages"]
    if m["role"] == "user"
)
m = re.search(r"<task id=(\d+)>", text)
if not m or m.group(1) not in TESTS:
    print(json.dumps({"verdict": "abstain", "reason": "unknown task"}))
    sys.exit(0)
t = TESTS[m.group(1)]
blocks = re.findall(r"```(?:python|py)?\n(.*?)```", inp["candidate"], re.S)
if not blocks:  # an unclosed fence (e.g. truncated output): take what follows it
    blocks = re.findall(r"```(?:python|py)?\n(.*)", inp["candidate"], re.S)
code = max(blocks, key=len) if blocks else inp["candidate"]
def norm(s):
    return [ln.split() for ln in s.strip().splitlines() if ln.strip()]


try:
    r = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--memory",
            "512m",
            "--pids-limit",
            "64",
            "--cap-drop",
            "ALL",
            "--read-only",
            "--user",
            "65534",
            "python:3.12-alpine",
            "python",
            "-c",
            HARNESS,
        ],
        input=json.dumps({"code": code, "inputs": t["inputs"]}),
        capture_output=True,
        text=True,
        timeout=150,
    )
    got = json.loads(r.stdout)
except Exception as e:
    print(
        json.dumps(
            {"verdict": "fail", "score": 0.0, "reason": f"harness: {type(e).__name__}"}
        )
    )
    sys.exit(0)
ok = sum(g is not None and norm(g) == norm(w) for g, w in zip(got, t["outputs"]))
n = len(t["outputs"])
print(
    json.dumps(
        {
            "verdict": "pass" if ok == n else "fail",
            "score": ok / n,
            "reason": f"{ok}/{n} cases",
        }
    )
)
