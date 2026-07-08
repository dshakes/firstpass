//! Tamper-evident audit hash chain (SPEC §9).
//!
//! Every trace record carries the hash of the previous record (`prev_hash`), forming an
//! append-only chain: altering any past record changes its hash and breaks every link after
//! it. The chain is **re-derivable by an external auditor** from the stored records alone —
//! so hashing must not depend on struct field order or on which serde features are compiled
//! in. We achieve that by canonicalizing to sorted-key, whitespace-free JSON before hashing.
//!
//! A record hashes over its *entire* content **including** its own `prev_hash`, but **not**
//! any field holding its own hash (that would be circular) — hence trace records store
//! `prev_hash` only, and their own hash is derived on demand via [`record_hash`].

use crate::error::{Error, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The `prev_hash` of the first record in a chain: 64 hex zeros (SHA-256 width).
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Hex-encoded SHA-256 of arbitrary bytes.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Canonical JSON encoding of a value: object keys sorted lexicographically at every depth,
/// no insignificant whitespace.
///
/// Determinism notes: numbers are emitted by `serde_json` (shortest round-trip via `ryu`),
/// which is stable for a given `f64`. Inputs in this crate are finite and non-negative, so
/// signed-zero / non-finite edge cases do not arise.
///
/// # Errors
/// Returns [`Error::Json`] if the value cannot be represented as JSON.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<String> {
    let v = serde_json::to_value(value)?;
    Ok(serde_json::to_string(&canonicalize(v))?)
}

/// Recursively rebuild a JSON value with object keys in sorted order. This is explicit
/// (rather than relying on `serde_json`'s default `BTreeMap`-backed map) so canonicalization
/// holds even if the `preserve_order` feature is enabled anywhere in the build graph.
fn canonicalize(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            let sorted: BTreeMap<String, Value> = map
                .into_iter()
                .map(|(k, val)| (k, canonicalize(val)))
                .collect();
            let mut out = serde_json::Map::new();
            for (k, val) in sorted {
                out.insert(k, val);
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

/// The hash of a record: `SHA-256` over its canonical JSON, hex-encoded.
///
/// # Errors
/// Returns [`Error::Json`] if the record cannot be serialized.
pub fn record_hash<T: Serialize>(record: &T) -> Result<String> {
    Ok(sha256_hex(canonical_json(record)?.as_bytes()))
}

/// A record that participates in the hash chain by exposing the previous record's hash.
pub trait Chained {
    /// The hash of the preceding record (or [`GENESIS_HASH`] for the first).
    fn prev_hash(&self) -> &str;
}

/// Verify that `records` form an unbroken chain starting from `genesis`.
///
/// For each record, its `prev_hash` must equal the hash of the record before it (or
/// `genesis` for the first). Any mismatch — a tampered payload or a re-linked record —
/// surfaces as [`Error::ChainBroken`] pointing at the first bad index.
///
/// # Errors
/// Returns [`Error::ChainBroken`] on the first broken link, or [`Error::Json`] if a record
/// cannot be serialized for hashing.
pub fn verify_chain<T: Serialize + Chained>(records: &[T], genesis: &str) -> Result<()> {
    let mut expected_prev = genesis.to_owned();
    for (i, rec) in records.iter().enumerate() {
        if rec.prev_hash() != expected_prev {
            return Err(Error::ChainBroken {
                index: i,
                detail: format!(
                    "prev_hash link mismatch: expected {expected_prev}, found {}",
                    rec.prev_hash()
                ),
            });
        }
        expected_prev = record_hash(rec)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_at_every_depth() {
        let v = json!({ "b": 1, "a": { "d": 4, "c": 3 }, "arr": [ { "y": 2, "x": 1 } ] });
        assert_eq!(
            canonical_json(&v).unwrap(),
            r#"{"a":{"c":3,"d":4},"arr":[{"x":1,"y":2}],"b":1}"#
        );
    }

    #[test]
    fn canonical_json_is_field_order_independent() {
        // Same logical content, different insertion order -> identical canonical form -> identical hash.
        let a = json!({ "x": 1, "y": 2 });
        let b = json!({ "y": 2, "x": 1 });
        assert_eq!(record_hash(&a).unwrap(), record_hash(&b).unwrap());
    }

    #[test]
    fn record_hash_is_deterministic_and_sensitive() {
        let a = json!({ "n": 1 });
        assert_eq!(record_hash(&a).unwrap(), record_hash(&a).unwrap());
        assert_ne!(
            record_hash(&a).unwrap(),
            record_hash(&json!({ "n": 2 })).unwrap()
        );
    }

    #[derive(Serialize)]
    struct Rec {
        prev_hash: String,
        n: u64,
    }
    impl Chained for Rec {
        fn prev_hash(&self) -> &str {
            &self.prev_hash
        }
    }

    fn build_chain(payloads: &[u64]) -> Vec<Rec> {
        let mut prev = GENESIS_HASH.to_owned();
        let mut out = Vec::new();
        for &n in payloads {
            let rec = Rec {
                prev_hash: prev.clone(),
                n,
            };
            prev = record_hash(&rec).unwrap();
            out.push(rec);
        }
        out
    }

    #[test]
    fn valid_chain_verifies() {
        let chain = build_chain(&[10, 20, 30]);
        assert!(verify_chain(&chain, GENESIS_HASH).is_ok());
    }

    #[test]
    fn tampering_payload_breaks_the_next_link() {
        let mut chain = build_chain(&[10, 20, 30]);
        chain[1].n = 999; // alter middle record's content
        match verify_chain(&chain, GENESIS_HASH) {
            Err(Error::ChainBroken { index, .. }) => assert_eq!(index, 2),
            other => panic!("expected ChainBroken at 2, got {other:?}"),
        }
    }

    #[test]
    fn relinking_a_prev_hash_is_detected() {
        let mut chain = build_chain(&[10, 20, 30]);
        chain[1].prev_hash = GENESIS_HASH.to_owned(); // forge the link itself
        match verify_chain(&chain, GENESIS_HASH) {
            Err(Error::ChainBroken { index, .. }) => assert_eq!(index, 1),
            other => panic!("expected ChainBroken at 1, got {other:?}"),
        }
    }
}
