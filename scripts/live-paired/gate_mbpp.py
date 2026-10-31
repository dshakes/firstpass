#!/usr/bin/env python3
"""MBPP test gate: run the task's asserts against the candidate in a no-network container."""

import json
import re
import subprocess
import sys

inp = json.load(sys.stdin)
prompt = "\n".join(
    m["content"]
    if isinstance(m["content"], str)
    else "".join(p.get("text", "") for p in m["content"])
    for m in inp["request"]["messages"]
    if m["role"] == "user"
)
tests = re.search(r"<tests>\n(.*?)\n</tests>", prompt, re.S)
if not tests:
    print(json.dumps({"verdict": "abstain", "reason": "no tests in prompt"}))
    sys.exit(0)
cand = inp["candidate"]
blocks = re.findall(r"```(?:python|py)?\n(.*?)```", cand, re.S)
if not blocks:  # an unclosed fence (e.g. truncated output): take what follows it
    blocks = re.findall(r"```(?:python|py)?\n(.*)", cand, re.S)
code = max(blocks, key=len) if blocks else cand
src = code + "\n\n" + tests.group(1) + "\nprint('FP_OK')\n"
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
            "256m",
            "--pids-limit",
            "64",
            "--cap-drop",
            "ALL",
            "--read-only",
            "--user",
            "65534",
            "python:3.12-alpine",
            "timeout",
            "10",
            "python",
            "-",
        ],
        input=src,
        capture_output=True,
        text=True,
        timeout=30,
    )
    ok = r.returncode == 0 and "FP_OK" in r.stdout
    print(
        json.dumps(
            {
                "verdict": "pass" if ok else "fail",
                "score": 1.0 if ok else 0.0,
                "reason": "tests passed" if ok else "tests failed",
                "evidence": (r.stderr or "")[-300:],
            }
        )
    )
except subprocess.TimeoutExpired:
    print(json.dumps({"verdict": "fail", "score": 0.0, "reason": "timeout"}))
