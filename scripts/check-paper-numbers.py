#!/usr/bin/env python3
"""Fail if any number in paper/main.tex does not appear in a committed artifact.

Written after two fabrications slipped into the paper in one sitting: an AIQ table and a
meta-verifier table were filled from memory of a different run rather than read out of the file,
and both looked plausible enough to survive review. The \\result{} placeholder guard catches an
*unfilled* number; nothing caught a *wrong* one.

Run it in CI, or before submitting anywhere:

    python3 scripts/check-paper-numbers.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "main.tex"
ARTIFACTS = sorted((ROOT / "docs" / "benchmarks").glob("*.txt"))

# Numbers that legitimately are not measurements: arXiv ids, years, section/version numbers,
# and the pre-registered constants (alpha, delta, confidence level).
ALLOW = {
    "0.10", "0.05", "0.95", "1.0", "0.5", "2021", "2023", "2024", "2025",
    "2108.07732", "2305.05176", "2310.12963", "2403.12031", "2404.14618",
    "2406.18665", "2505.19970", "0.78", "0.94", "0.93",
}


def main() -> int:
    tex = PAPER.read_text(encoding="utf-8")
    # Strip comments — commented-out numbers are not claims. Split on an UNESCAPED `%` only:
    # `\%` is a literal percent sign and appears in nearly every table cell ("59\%"), so a naive
    # split truncated each row at its first percentage and silently exempted every number after it
    # from verification. The check then reported all-clear while skipping most of the paper.
    tex = "\n".join(re.split(r"(?<!\\)%", l)[0] for l in tex.splitlines())
    corpus = "\n".join(a.read_text(encoding="utf-8") for a in ARTIFACTS)

    # Any decimal with 3+ significant places is a measurement worth checking. Two-place numbers
    # are too collision-prone to be useful evidence either way.
    claims = sorted(set(re.findall(r"\d+\.\d{3,}", tex)))
    missing = [c for c in claims if c not in ALLOW and c not in corpus]

    print(f"paper: {PAPER.relative_to(ROOT)}")
    print(f"artifacts: {len(ARTIFACTS)} file(s) under docs/benchmarks/")
    print(f"checked {len(claims)} numeric claims")
    if missing:
        print("\nFAIL — these appear in the paper but in NO committed artifact:")
        for m in missing:
            print(f"  {m}")
        print("\nEither cite the artifact that contains them, or remove the claim.")
        return 1
    print("OK — every numeric claim traces to a committed artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
