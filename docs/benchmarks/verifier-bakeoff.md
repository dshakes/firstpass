## Verifier bake-off — a verifier stronger than OpenJev `noul`? (`specs/verifier-bakeoff.md`)

Held-out split: 488 dev / 486 held-out (seeded `sha256(id)` parity). Coder model: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` via mlx-lm.

### Dev table (τ: maximize catch rate s.t. collateral ≤ 0.05; ties → higher τ)

| verifier | τ | n | n_wrong | n_right | catch rate | collateral |
|---|---|---|---|---|---|---|
| V0 | 0.0623 | 488 | 58 | 430 | 0.1379 | 0.0488 |
| V1 | 8.39e-6 | 488 | 58 | 430 | 0.1207 | 0.0488 |
| V2 | 0 | 488 | 58 | 430 | 0.0000 | 0.0000 |
| V3 | 0.0800 | 488 | 58 | 430 | 0.2069 | 0.0488 |

**Selected on dev: V3**

### Held-out (all four for context; only the selected verifier's row is the verdict)

| verifier | τ (from dev) | n | n_wrong | n_right | catch rate [CI] | collateral [CI] | AUC [CI] | abstain [CI] | row verdict |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0623 | 486 | 53 | 433 | 0.0943 [0.0189, 0.1887] | 0.0600 [0.0393, 0.0831] | 0.6268 [0.5466, 0.7054] | 0.0000 [0.0000, 0.0000] | NOT-RECOMMENDED |
| V1 | 8.39e-6 | 486 | 53 | 433 | 0.0755 [0.0189, 0.1509] | 0.0624 [0.0393, 0.0855] | 0.5976 [0.5113, 0.6818] | 0.0000 [0.0000, 0.0000] | NOT-RECOMMENDED |
| V2 | 0 | 486 | 53 | 433 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.5917 [0.5083, 0.6698] | 0.0000 [0.0000, 0.0000] | NOT-RECOMMENDED |
| V3 | 0.0800 | 486 | 53 | 433 | 0.0943 [0.0189, 0.1887] | 0.0670 [0.0439, 0.0924] | 0.6646 [0.5908, 0.7334] | 0.0000 [0.0000, 0.0000] | NOT-RECOMMENDED |

**Verdict (selected verifier `V3`, held-out): NOT-RECOMMENDED**

### Latency (all cached calls, ms; V3 = test-writer + sandbox exec)

| verifier | n | p50 | p95 |
|---|---|---|---|
| V0 | 974 | 767 | 1314 |
| V1 | 974 | 6841 | 22157 |
| V2 | 974 | 670 | 981 |
| V3 | 974 | 2870 | 7154 |

