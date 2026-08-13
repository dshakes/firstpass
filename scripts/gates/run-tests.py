#!/usr/bin/env python3
"""Turn any test command into a Firstpass gate.

Firstpass gates the *actual output* before serving it, so a gate has to run your real check.
The contract is small but easy to get subtly wrong by hand, and getting it wrong is expensive:
a gate that prints nothing abstains on every request, and a route configured `fail_open` then
serves everything unverified. This wrapper implements the contract once.

    stdin  (JSON):  {"gate_id": ..., "candidate": "<model output>", "request": {...}}
    stdout (JSON):  {"verdict": "pass"|"fail"|"abstain", "score": 0.0-1.0, "reason": ...}

Usage — the command after `--` is your real test command:

    run-tests.py --write solution.py -- pytest -q
    run-tests.py --write solution.js -- npx jest --silent
    run-tests.py --write src/lib.rs  -- cargo test --quiet

In firstpass.toml:

    [[gate]]
    id  = "unit-tests"
    cmd = ["python3", "scripts/gates/run-tests.py", "--write", "solution.py", "--", "pytest", "-q"]

## The score is graded, and that matters

Where the runner reports counts ("5 passed, 2 failed"), the score is the pass *fraction* rather
than a bare 1.0/0.0. That is not cosmetic. Firstpass calibrates its serve threshold with split
conformal over the gate score, and a strictly binary score gives the calibration exactly two
operating points — measured on MBPP, where a generated function passes all of a task's asserts or
none, the score was `{0.0, 1.0}` with nothing between and no threshold could separate a safe
prefix. A graded score is what makes the guarantee reachable.

## Abstain, never guess

Infrastructure failures (runner missing, timeout, unparseable output) return `abstain`, never a
fabricated pass or fail. Set `on_abstain = "fail_closed"` on the gate if silence must block
serving.

## Running untrusted code

This executes model-generated code with your permissions. By default it runs in a fresh temporary
directory, which limits accidents but is **not** a security boundary. For untrusted traffic pass
`--docker IMAGE` to run the command inside a container with no network:

    run-tests.py --docker python:3.12-alpine --write solution.py -- pytest -q
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Emitted when the gate cannot form an opinion. Kept in one place so every early return is honest.
ABSTAIN = "abstain"


def emit(verdict: str, score: float | None = None, reason: str | None = None) -> None:
    """Write the verdict and exit. Always valid JSON on stdout, nothing else ever."""
    out: dict[str, object] = {"verdict": verdict}
    if score is not None:
        out["score"] = max(0.0, min(1.0, score))
    if reason:
        out["reason"] = reason
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()
    sys.exit(0)


def extract_code(candidate: str, filename: str) -> str:
    """Pull runnable source out of a model response.

    Models wrap code in Markdown fences and narrate around it, so the raw candidate is usually not
    a valid source file. Prefer a fence whose language tag matches the target file's extension,
    then any fence, then the whole text (a model that replied with bare code is still valid).
    """
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    aliases = {
        "py": ["python", "py"],
        "js": ["javascript", "js", "node"],
        "ts": ["typescript", "ts"],
        "rs": ["rust", "rs"],
        "go": ["go"],
        "java": ["java"],
        "rb": ["ruby", "rb"],
    }
    want = aliases.get(ext, [ext] if ext else [])

    fences = re.findall(r"```([A-Za-z0-9_+-]*)\n(.*?)```", candidate, re.DOTALL)
    for tag, body in fences:
        if tag.lower() in want:
            return body
    if fences:
        return fences[0][1]
    return candidate


# Runner output patterns, most specific first. Each yields (passed, failed).
_COUNTS = [
    # pytest: "5 passed, 2 failed" / "3 failed, 1 passed"
    (r"(\d+)\s+passed", r"(\d+)\s+failed"),
    # cargo test: "test result: ok. 12 passed; 0 failed"
    (r"(\d+)\s+passed;", r"(\d+)\s+failed"),
    # jest: "Tests:  2 failed, 5 passed, 7 total"
    (r"(\d+)\s+passed", r"(\d+)\s+failed"),
]


# Distinguishing "the runner never ran" from "the runner ran and the code failed".
#
# This is the whole safety property, and it cuts BOTH ways.
#
# Miss it in one direction and a broken runner reports every candidate as a test failure:
# `python3 -m pytest` with pytest uninstalled exits non-zero through a *present* interpreter, so
# there is no FileNotFoundError to catch. Every request then escalates and the top rung is billed
# forever while the gate looks healthy.
#
# Miss it in the other direction and it is worse. A candidate whose own code does `import requests`
# produces "ModuleNotFoundError: No module named 'requests'" — a genuine FAILURE. Matching that as
# "runner unavailable" abstains instead, and under `on_abstain = "fail_open"` the broken candidate
# is SERVED. A gate that serves broken code is worse than no gate.
#
# So the module named in the error is compared against the runner's own tokens. Only the runner
# going missing is infrastructure; anything the candidate imports is the candidate's problem.


def _runner_tokens(cmd: list[str]) -> set[str]:
    """Names that identify the runner itself, e.g. {"python3", "pytest"} for `python3 -m pytest`.

    Flags and their values are skipped so `-q` or `--silent` cannot be mistaken for a package.
    """
    toks = set()
    for part in cmd:
        if part.startswith("-"):
            continue
        base = os.path.basename(part)
        toks.add(base.lower())
        # npx jest → also match a bare "jest"; python3 -m pytest → "pytest"
        toks.add(base.split(".")[0].lower())
    return toks


def runner_unavailable(output: str, returncode: int, cmd: list[str]) -> str | None:
    """Reason the runner could not run, or None if it genuinely ran and reported a result."""
    if returncode == 127:  # POSIX: the shell could not find the command at all
        return "runner exited 127 (command not found)"

    low = output.lower()
    tokens = _runner_tokens(cmd)

    # "No module named 'X'" / "Cannot find module 'X'" — infrastructure only if X IS the runner.
    for pat in (r"no module named '?([\w.\-]+)'?", r"cannot find module '?([\w.\-/]+)'?"):
        for m in re.finditer(pat, low):
            missing = m.group(1).strip("'\"").split(".")[0].split("/")[-1]
            if missing in tokens:
                return f"runner unavailable: {missing!r} is not installed"

    # "sh: jest: command not found" — only when the named command is the runner.
    for m in re.finditer(r"([\w.\-]+): command not found", low):
        if m.group(1) in tokens:
            return f"runner unavailable: {m.group(1)!r} not found"
    if "is not recognized as an internal or external command" in low:
        return "runner unavailable: not recognised (Windows)"
    if "executable file not found" in low:
        return "runner unavailable: executable not found"

    return None


def parse_score(output: str, ok: bool) -> float:
    """Fraction of tests that passed, or a bare 1.0/0.0 when counts are not reported.

    The fraction is what gives conformal calibration something continuous to threshold on.
    """
    for pass_re, fail_re in _COUNTS:
        p = re.search(pass_re, output)
        f = re.search(fail_re, output)
        if p or f:
            passed = int(p.group(1)) if p else 0
            failed = int(f.group(1)) if f else 0
            total = passed + failed
            if total > 0:
                return passed / total
    return 1.0 if ok else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--write",
        required=True,
        help="filename to write the candidate code to, relative to the run directory",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds before the runner is killed and the gate abstains (default 60)",
    )
    ap.add_argument(
        "--docker",
        metavar="IMAGE",
        help="run the command inside this container image with no network",
    )
    ap.add_argument(
        "--workdir",
        metavar="DIR",
        help=(
            "run inside DIR instead of a fresh temp dir. Needed with --docker when the daemon "
            "cannot see the host's temp path (Docker-in-Docker, or a remote/Desktop daemon): the "
            "bind mount then silently resolves to an empty directory and every candidate abstains. "
            "Point this at a path the daemon shares."
        ),
    )
    ap.add_argument(
        "--copy",
        action="append",
        default=[],
        help="path to copy into the run directory first (e.g. your test file). Repeatable.",
    )
    ap.add_argument("command", nargs=argparse.REMAINDER, help="-- your test command")
    args = ap.parse_args()

    cmd = [c for c in args.command if c != "--"]
    if not cmd:
        emit(ABSTAIN, reason="no test command given after --")

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError) as e:
        emit(ABSTAIN, reason=f"unreadable gate payload: {e}")

    candidate = payload.get("candidate")
    if not isinstance(candidate, str) or not candidate.strip():
        # An empty candidate is a real failure, not an infrastructure problem: there is nothing
        # to serve. `non-empty` would catch it too, but a gate must not depend on its neighbours.
        emit("fail", score=0.0, reason="empty candidate")

    code = extract_code(candidate, args.write)

    # An explicit --workdir is used as-is (the caller owns it); otherwise a temp dir that is
    # cleaned up on exit.
    ctx = (
        contextlib.nullcontext(args.workdir)
        if args.workdir
        else tempfile.TemporaryDirectory(prefix="firstpass-gate-")
    )
    with ctx as workdir:
        os.makedirs(workdir, exist_ok=True)
        target = os.path.join(workdir, args.write)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(code)

        for src in args.copy:
            if not os.path.exists(src):
                emit(ABSTAIN, reason=f"--copy path not found: {src}")
            dst = os.path.join(workdir, os.path.basename(src))
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        if args.docker:
            run = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{workdir}:/work",
                "-w",
                "/work",
                args.docker,
                *cmd,
            ]
        else:
            run = cmd

        try:
            proc = subprocess.run(
                run,
                cwd=None if args.docker else workdir,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except FileNotFoundError:
            emit(ABSTAIN, reason=f"runner not found: {run[0]}")
        except subprocess.TimeoutExpired:
            emit(ABSTAIN, reason=f"runner exceeded {args.timeout}s")
        except OSError as e:
            emit(ABSTAIN, reason=f"runner failed to start: {e}")

        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        if not ok:
            why = runner_unavailable(output, proc.returncode, cmd)
            if why:
                emit(ABSTAIN, reason=why)
        score = parse_score(output, ok)
        # Exit status decides the verdict; the parsed fraction only grades it. A runner that exits
        # non-zero has failed even if it printed encouraging counts.
        emit(
            "pass" if ok else "fail", score=score, reason=None if ok else "tests failed"
        )


def _selfcheck() -> None:
    """`run-tests.py --selfcheck` — assert the pure logic, no runner needed."""
    # Fence extraction prefers the matching language.
    c = "here you go\n```python\ndef f():\n    return 1\n```\nhope that helps"
    assert "def f()" in extract_code(c, "solution.py"), "must pull code out of a fence"
    assert "hope that helps" not in extract_code(c, "solution.py"), (
        "must drop the prose"
    )

    # A bare-code response is still usable.
    assert extract_code("def g(): pass", "s.py").strip() == "def g(): pass"

    # The language tag is honoured when several fences are present.
    two = "```js\nvar a=1;\n```\n```python\nb=2\n```"
    assert "b=2" in extract_code(two, "s.py"), "must pick the python fence"
    assert "var a=1" in extract_code(two, "s.js"), "must pick the js fence"

    # Graded scores: the whole reason conformal has something to threshold on.
    assert abs(parse_score("5 passed, 5 failed", False) - 0.5) < 1e-9
    assert abs(parse_score("8 passed, 2 failed", False) - 0.8) < 1e-9
    assert parse_score("test result: ok. 12 passed; 0 failed", True) == 1.0
    # No counts reported: fall back to the exit status rather than inventing a fraction.
    assert parse_score("all good", True) == 1.0
    assert parse_score("boom", False) == 0.0

    # A broken runner must abstain, never fail. Found by running this for real: `python3 -m pytest`
    # with pytest uninstalled exits non-zero through a present interpreter, so there is no
    # FileNotFoundError, and naive exit-code handling reported a healthy candidate as "fail".
    pytest_cmd = ["python3", "-m", "pytest", "-q"]
    jest_cmd = ["npx", "jest", "--silent"]

    # The runner itself missing IS infrastructure — abstain.
    assert runner_unavailable("/usr/bin/python3: No module named pytest", 1, pytest_cmd)
    assert runner_unavailable("", 127, pytest_cmd)
    assert runner_unavailable("sh: jest: command not found", 127, jest_cmd)
    assert runner_unavailable("Error: Cannot find module 'jest'", 1, jest_cmd)

    # A genuine test failure is NOT a broken runner.
    assert runner_unavailable("2 failed, 3 passed", 1, pytest_cmd) is None
    assert runner_unavailable("assert 5 == 4\nE  AssertionError", 1, pytest_cmd) is None
    assert runner_unavailable("", 0, pytest_cmd) is None

    # THE ONE THAT MATTERS: the candidate's own missing import is a FAILURE, not an abstain.
    # Abstaining here would, under on_abstain = "fail_open", serve code that cannot even import.
    assert (
        runner_unavailable(
            "ModuleNotFoundError: No module named 'requests'", 1, pytest_cmd
        )
        is None
    ), "a candidate's missing import must fail, never abstain"
    assert (
        runner_unavailable("ModuleNotFoundError: No module named 'mathh'", 1, pytest_cmd)
        is None
    ), "a candidate's import typo must fail, never abstain"
    assert (
        runner_unavailable("Error: Cannot find module './helper'", 1, jest_cmd) is None
    ), "a candidate's missing local require must fail, never abstain"
    # A flag must never be mistaken for a package name.
    assert runner_unavailable("No module named 'q'", 1, pytest_cmd) is None
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
