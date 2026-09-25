## OpenJev `decision` gate — error rates on real MBPP (Study B)

Pre-registration: `specs/prior-blend-and-decision-gate.md`. Candidates already passed the existing (test) gate; labels are VRBench's hidden-test oracle, run in the fail-closed sandbox.

n = 974 (wrong = 111, right = 863); 0 missing an MBPP task, 0 missing a decision score.

| metric | point | 95% CI |
|---|---|---|
| catch rate (τ=0.5) | 0.2162 | [0.1441, 0.2973] |
| collateral (τ=0.5) | 0.1031 | [0.0834, 0.1228] |
| AUC | 0.6310 | [0.5711, 0.6886] |
| abstain rate | 0.0000 | [0.0000, 0.0000] |

**Verdict: NOT-RECOMMENDED**

Exploratory τ sweep (point estimates only; cannot change the verdict above):

| τ | catch rate | collateral |
|---|---|---|
| 0.1 | 0.1261 | 0.0672 |
| 0.3 | 0.1892 | 0.0950 |
| 0.5 | 0.2162 | 0.1031 |
| 0.7 | 0.2432 | 0.1194 |
| 0.9 | 0.3063 | 0.1553 |


---

**Provenance.**

```
cargo run --release -p firstpass-bench -- --decision-study \
  ~/vrb-cascade.jsonl scratchpad/mbpp.jsonl http://127.0.0.1:8080 \
  scratchpad/vrb-labels.jsonl scratchpad/decision-scores.jsonl
```

- Candidates: `~/vrb-cascade.jsonl` (974 rows, sha256 `32c30182a45e6b7b9dd0e43804de91b640835e5766be63a3885d0d0c3c2d1890`).
- Oracle labels: 974/974 scored in the fail-closed sandbox (runtime tier `runc` — gVisor not installed), 863 oracle-pass / 111 oracle-wrong, 0 sandbox faults.
- Decision scores: 974/974 calls to local OpenJev at `http://127.0.0.1:8080`, 0 abstains (`raw_ok=true` on every call).
- Harness commit: `284fd37af3a60ea4dd29e22e6985cde2b6a3e127`.
- This measures **local OpenJev**, not hosted Jev (spec's caveat).
