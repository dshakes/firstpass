# Independently verifying Firstpass receipts

Every routing decision Firstpass makes is sealed into a hash-chained receipt: each receipt
stores the SHA-256 of the previous one, so the whole log is append-only and tamper-evident.
The point of this document is that you do **not** have to trust the running proxy, the
database, or Firstpass itself to check that — the verification is reproducible by anyone with
the exported log and the `firstpass` binary (or any SHA-256 implementation).

## The auditor workflow

1. **Operator exports the sealed log:**

   ```bash
   firstpass export --out receipts.jsonl
   ```

   This writes one receipt per line, in chain order. It contains only the hashed bodies —
   the deferred-verdict side table (downstream outcomes reported after the fact) is never
   part of the chain and is not included.

2. **Auditor verifies it, offline, on their own machine:**

   ```bash
   firstpass verify --file receipts.jsonl
   # OK: 12043 receipts, hash chain intact from genesis
   ```

   `verify` re-derives every link from the genesis hash forward. No proxy runs, no database
   is opened, no network call is made. A clean chain exits `0`; a broken one prints the index
   of the first bad link and exits `1`, so it drops straight into a CI or compliance gate:

   ```bash
   firstpass verify --file receipts.jsonl --json
   ```

## What verification catches

- **Any altered field** in a sealed receipt — the receipt's own hash changes, so the *next*
  receipt's `prev_hash` no longer matches, and the chain breaks at that index.
- **Reordering** — the links only line up in the original sequence.
- **Deletion of a middle receipt** — same: the surrounding links stop matching.

What it deliberately does not cover: the deferred-verdict table (by design not chained), and
truncation of the *tail* of the log (a shortened-but-consistent prefix still verifies — pair
export with your own retention/offsite policy if tail-truncation is in your threat model).

## Why this matters

Predictive and black-box routers can show you a dashboard number; they cannot hand a regulator
a log that a third party re-derives from first principles. This is the concrete substance
behind "tamper-evident audit trail" — the EU-AI-Act-style logging obligation satisfied by a
file and a one-line command, not a promise.

The chain algorithm is `crates/firstpass-core/src/hashchain.rs` (`verify_chain`) — small,
dependency-light, and independently re-implementable from the field definitions in
[`SPEC.md`](../../SPEC.md) §9.1.
