# The Firstpass served-failure SLO

Firstpass is the only router that can put a **served-failure rate** in a contract and back it
with math and a live instrument. This document states the guarantee precisely enough to write
into an SLA, and points at the mechanism that holds it.

## The guarantee

> Of the requests Firstpass **serves** (i.e. an output cleared the gate at or above the serve
> threshold), no more than **α** are wrong, at confidence **1 − δ**, distribution-free.

- **α** (target served-failure rate) and **δ** (confidence) are yours to set. The published
  benchmark uses α = 0.10, δ = 0.05 and earns it empirically at 5.5% calibrated risk on 974
  real MBPP tasks ([artifact](../benchmarks/mbpp-live-base.txt)).
- **Distribution-free** means no assumption about your traffic's shape — the bound comes from
  conformal / Learn-then-Test finite-sample statistics, not a model of your workload.
- "Wrong" is defined by *your* gate. The guarantee is about the gate's served-failure rate;
  choosing a gate that reflects your quality bar is the operator's job (tests are strongest).

## How it is set, and how it holds under drift

1. **Calibrate** the serve threshold from your own labeled receipts:
   `firstpass calibrate --method ltt` gives the distribution-free, finite-sample
   (risk-controlling) threshold and reports the gate's observed false-accept rate at that
   point — the verifier ROC the bound rests on.
2. **Enforce** at that threshold (`serve_threshold` in config).
3. **Track it live**: `[escalation.adaptive]` runs adaptive conformal inference against the
   realized-served-failure gauge, nudging the threshold as traffic drifts so the served-failure
   rate stays at α instead of silently decaying. Two Prometheus gauges make it observable:
   `firstpass_serve_threshold` and `firstpass_realized_served_failure`.

## Making it contractual

- **Instrument the SLO**: alert when `firstpass_realized_served_failure` exceeds α over your
  measurement window (the committed [Grafana dashboard](../observability/grafana-firstpass.json)
  colors this panel red at 0.10).
- **Prove compliance after the fact**: the served-failure rate is computed from the same sealed,
  hash-chained receipts an auditor can independently verify (`firstpass export` +
  `firstpass verify`, see [receipt-audit.md](receipt-audit.md)). The number in the SLA and the
  number in the audit log are the same number, and neither is takeable on trust.

## Honest limits

- The bound is only as meaningful as the gate. A gate that rubber-stamps everything yields a
  vacuous guarantee; `calibrate --method ltt` surfaces the false-accept rate precisely so this
  cannot hide.
- Calibration needs enough labeled outcomes to be feasible; below that, `calibrate` reports
  infeasible rather than inventing a threshold.
- An imperfect verifier over-optimizes if you sample without bound — sample counts are capped,
  by design (see the roadmap's verifier-imperfection rails).
