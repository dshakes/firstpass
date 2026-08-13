# Gate wrappers

A Firstpass gate decides whether a model's output is good enough to serve. It is a subprocess with
a small contract:

```
stdin  (JSON):  {"gate_id": "...", "candidate": "<the model's reply>", "request": {...}}
stdout (JSON):  {"verdict": "pass" | "fail" | "abstain", "score"?: 0.0-1.0, "reason"?: "..."}
```

**A bare `pytest` is not a gate.** It does not read the candidate from stdin and does not print a
verdict, so Firstpass records an abstain on every request — and a route with `on_abstain =
"fail_open"` will then serve everything unverified while looking healthy. That trap is why
`run-tests.py` exists.

## `run-tests.py` — wrap any test command

```
run-tests.py --write <file> [--copy PATH]... [--docker IMAGE] [--timeout S] -- <your command>
```

It extracts the code from the model's reply (Markdown fences included), writes it to `--write`
inside a fresh temporary directory, copies in anything you name with `--copy`, runs your command,
and reports the verdict.

### Recipes

**pytest**
```toml
[[gate]]
id  = "unit-tests"
cmd = ["python3", "scripts/gates/run-tests.py",
       "--write", "solution.py", "--copy", "tests/",
       "--", "python3", "-m", "pytest", "-q"]
```

**jest**
```toml
[[gate]]
id  = "unit-tests"
cmd = ["python3", "scripts/gates/run-tests.py",
       "--write", "solution.js", "--copy", "__tests__/",
       "--", "npx", "jest", "--silent"]
```

**cargo test**
```toml
[[gate]]
id  = "unit-tests"
cmd = ["python3", "scripts/gates/run-tests.py",
       "--write", "src/lib.rs", "--copy", "Cargo.toml", "--copy", "tests/",
       "--", "cargo", "test", "--quiet"]
```

## The score is graded, and it matters

Where your runner reports counts, the score is the **pass fraction**, not a bare 1.0/0.0:

```
$ echo '{"candidate":"```python\ndef add(a,b): return a+b if a else 99\n```"}' | run-tests.py ...
{"verdict": "fail", "score": 0.6666666666666666, "reason": "tests failed"}
```

Firstpass calibrates its serve threshold with split conformal over that score. A strictly binary
score gives the calibration only two operating points — measured on MBPP, where a generated
function passes all of a task's asserts or none, the observed score set was `{0.0, 1.0}` and no
threshold could separate a safe prefix. The pass fraction is what makes the bound reachable.

## Abstain is not failure

The wrapper returns `abstain` when it could not form an opinion — runner missing, timeout,
unreadable payload — and never a fabricated pass or fail.

This is load-bearing. `python3 -m pytest` with pytest uninstalled exits non-zero through a
*present* interpreter, so there is no "command not found" to catch; naive exit-code handling
reports a perfectly good candidate as a test failure. A gate broken that way fails every
candidate, escalates every request, and bills your top rung indefinitely while looking like it is
working. The wrapper detects these signatures and abstains instead.

Decide what an abstain should mean for your traffic:

```toml
on_abstain = "fail_closed"   # silence blocks serving — use when a wrong answer is expensive
on_abstain = "fail_open"     # silence serves — use when availability matters more
```

## Abstain vs fail: the distinction that decides correctness

The wrapper compares the module named in an error against the **runner's own tokens**, and this
cuts both ways:

| output | verdict | why |
|---|---|---|
| `No module named 'pytest'` (runner is pytest) | `abstain` | infrastructure — the gate could not run |
| `No module named 'requests'` (candidate's import) | **`fail`** | the candidate is broken |
| `2 failed, 3 passed` | `fail` | the runner ran and reported |

The second row is the one that matters. Treating a candidate's own missing import as "runner
unavailable" would abstain, and under `on_abstain = "fail_open"` the proxy would then **serve code
that cannot even import**. A gate that serves broken code is worse than no gate at all.

## Running untrusted code

The gate executes model-generated code with your permissions. The temp directory limits accidents;
it is **not** a security boundary. For untrusted traffic, run it in a container with no network:

```
--docker python:3.12-alpine
```

**With a daemon that cannot see your temp path** — Docker-in-Docker, a remote daemon, or Docker
Desktop with `/tmp` outside the shared paths — a bind mount of `/tmp/firstpass-gate-XXXX` resolves
to an *empty* directory inside the container. The code is invisible, every run fails, and the gate
abstains on everything. Point `--workdir` at a directory the daemon shares:

```
--docker python:3.12-alpine --workdir /shared/gate-run
```

## Checking your gate

```
python3 scripts/gates/run-tests.py --selfcheck      # asserts the pure logic, no runner needed

echo '{"gate_id":"t","candidate":"```python\ndef add(a,b): return a+b\n```","request":{}}' \
  | python3 scripts/gates/run-tests.py --write solution.py --copy tests/ -- python3 -m pytest -q
```

If that prints `{"verdict": "pass", "score": 1.0}`, the gate is wired correctly. `firstpass doctor`
also checks that every configured gate command exists.
