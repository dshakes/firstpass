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
PAPER_FILES = sorted((ROOT / "paper").glob("*.tex"))
ARTIFACTS = sorted((ROOT / "docs" / "benchmarks").glob("*.txt"))

# Numbers that legitimately are not measurements: arXiv ids, years, section/version numbers,
# and the pre-registered constants (alpha, delta, confidence level).
# Values that are legitimately not measurements. Each needs a reason, because the whole point of
# this file is that an unexplained number is a bug.
#   0.0082  — a plot axis limit, chosen to frame the data
ALLOW = {
    "0.0082",
    "0.10", "0.05", "0.95", "1.0", "0.5", "2021", "2023", "2024", "2025",
    "2108.07732", "2305.05176", "2310.12963", "2403.12031", "2404.14618",
    "2406.18665", "2505.19970", "2510.00202", "0.78", "0.94", "0.93",
}


def main() -> int:
    # EVERY .tex in paper/, not just main.tex: figure coordinates live in figures.tex and are
    # claims exactly like table cells are. Checking only main.tex would exempt every data point in
    # every plot -- the same blind spot as the escaped-percent bug documented above.
    tex = "\n".join(f.read_text(encoding="utf-8") for f in PAPER_FILES)
    # Strip comments — commented-out numbers are not claims. Split on an UNESCAPED `%` only:
    # `\%` is a literal percent sign and appears in nearly every table cell ("59\%"), so a naive
    # split truncated each row at its first percentage and silently exempted every number after it
    # from verification. The check then reported all-clear while skipping most of the paper.
    tex = "\n".join(re.split(r"(?<!\\)%", l)[0] for l in tex.splitlines())
    # Exact numeric tokens, not substrings. `0.902 in corpus` is True whenever the artifact
    # contains 0.9025 — so a fabricated number passes whenever some real measurement happens to
    # extend it. Flagged by review; the checker was weaker than its output suggested.
    corpus_text = "\n".join(a.read_text(encoding="utf-8") for a in ARTIFACTS)
    corpus = set(re.findall(r"\d+\.\d+", corpus_text))

    # Strip plot STYLING before looking for claims. Axis bounds, opacities, mark sizes and
    # number-format precisions are presentation choices, not measurements, and they change
    # whenever a figure is re-framed. Whitelisting them by value instead would grow without
    # limit and — much worse — could silently exempt a real fabricated number that happened to
    # collide with a bound.
    #
    # Data coordinates are deliberately NOT stripped: everything inside `coordinates {...}` is a
    # claim and must trace to an artifact.
    styling = re.compile(
        r"\b(?:x|y)(?:min|max)\s*=\s*-?[\d.]+"
        r"|\b(?:width|height|precision|opacity|mark size|line width|xshift|yshift)"
        r"\s*=\s*-?[\d.]+\w*"
        r"|fill opacity\s*=\s*[\d.]+"
        # `axis cs:X,Y` places an ANNOTATION (a node or a rule) in data space. It is layout, not
        # a measurement. Plot data lives in `coordinates {...}` and is deliberately left alone.
        r"|axis cs:\s*-?[\d.]+\s*,\s*-?[\d.]+"
    )
    tex = styling.sub(" ", tex)

    # Any decimal with 3+ significant places is a measurement worth checking. Two-place numbers
    # are too collision-prone to be useful evidence either way.
    claims = sorted(set(re.findall(r"\d+\.\d{3,}", tex)))
    missing = [c for c in claims if c not in ALLOW and c not in corpus]

    print("paper: " + ", ".join(f.name for f in PAPER_FILES))
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
