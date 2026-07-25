//! Percentage rollout: deciding which slice of matched traffic actually enforces (ADR 0009 D1).
//!
//! The whole mechanism is one pure function, and the reason it is a *function of a stable key*
//! rather than a random draw is the point of the module. Drawing per request would:
//!
//!   * flip a multi-turn conversation between enforced and observed mid-thread, so a user sees
//!     the model change for no reason they can perceive; and, worse,
//!   * make the served population unstable. The distribution-free bound Firstpass publishes is
//!     computed over *the population that was served* — if membership is re-drawn every request,
//!     that population is not a stable sample of anything and the bound describes a cohort that
//!     never existed.
//!
//! So assignment is `hash(salt || key_kind || key_value)`: deterministic for a fixed salt,
//! identical across restarts and across replicas sharing config, needing no coordination, and
//! leaking nothing — the same salted-hash discipline [`crate::features::repo_fingerprint`] uses.

use serde::Deserialize;

/// Buckets per rollout space. 10 000 gives two decimal places of percent, which is finer than
/// any operator needs and keeps the modulo bias negligible against a 32-bit draw.
pub const BUCKETS: u32 = 10_000;

/// Which stable identity a rollout holds constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RolloutKey {
    /// The `x-firstpass-session` header — a whole agent run moves together.
    ///
    /// The default, because a conversation is the unit a user experiences, so it is the unit
    /// that should be held constant. Falls back to [`RolloutKey::Request`] when absent.
    #[default]
    Session,
    /// The request's own identity. For stateless single-shot traffic with no session.
    Request,
    /// The tenant — for hosted deployments ramping tenant by tenant.
    Tenant,
}

impl RolloutKey {
    /// Domain-separator tag mixed into the hash, so the same value under two different key kinds
    /// lands in unrelated buckets (a tenant id that happens to equal a session id must not
    /// inherit its assignment).
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Request => "request",
            Self::Tenant => "tenant",
        }
    }
}

/// Rollout settings for a route. Absent means "all matched traffic", i.e. today's behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rollout {
    /// Percent of matched traffic that actually enforces, in `[0, 100]`. The remainder takes the
    /// ordinary observe path. Rollout only ever *subtracts* from what `mode` already allows.
    pub percent: f64,
    /// Which identity is held constant.
    #[serde(default)]
    pub key: RolloutKey,
}

impl Rollout {
    /// Whether this rollout is configured coherently.
    ///
    /// # Errors
    /// If `percent` is not finite or falls outside `[0, 100]`.
    pub fn validate(&self) -> Result<(), String> {
        if !self.percent.is_finite() || !(0.0..=100.0).contains(&self.percent) {
            return Err(format!(
                "route.rollout.percent must be finite and within [0, 100], got {}",
                self.percent
            ));
        }
        Ok(())
    }
}

/// The bucket `key_value` falls in, within `0..`[`BUCKETS`].
///
/// Deterministic for a fixed `salt`. The salt is the per-deployment secret already used for
/// feature fingerprints, so bucket assignment cannot be reconstructed — or gamed — by a caller
/// who can choose their own session id.
#[must_use]
pub fn bucket_of(salt: &str, key: RolloutKey, key_value: &str) -> u32 {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(salt.as_bytes());
    h.update([0u8]); // separator: salt||tag must not collide with a different split
    h.update(key.tag().as_bytes());
    h.update([0u8]);
    h.update(key_value.as_bytes());
    let d = h.finalize();
    u32::from_be_bytes([d[0], d[1], d[2], d[3]]) % BUCKETS
}

/// Whether a bucket is inside a `percent` rollout.
///
/// Strictly `<` so that `percent = 0` enforces nothing and `percent = 100` enforces everything —
/// both endpoints exact, which matters because "ramp to 0" is how an operator backs out.
#[must_use]
pub fn in_rollout(bucket: u32, percent: f64) -> bool {
    if percent <= 0.0 {
        return false;
    }
    if percent >= 100.0 {
        return true;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "percent is validated finite in [0,100]; the product is within u32"
    )]
    let cutoff = (percent / 100.0 * f64::from(BUCKETS)) as u32;
    bucket < cutoff
}

/// A stable identity for one request, for use as bucketing input when there is no session to key
/// on. Hashes the request bytes, so the same request always lands in the same arm.
///
/// The digest is truncated and the input is never retained — this exists so rollout can bucket
/// stateless traffic without Firstpass holding on to prompt content.
#[must_use]
pub fn request_identity(body: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(body);
    hex::encode(&h.finalize()[..8])
}

/// The decision recorded on the trace, so an auditor can re-derive which arm a request was in.
///
/// Without this a mixed-population receipt log cannot be split back into the enforced and
/// observed arms, and every savings or failure figure computed over it silently blends two
/// different regimes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutDecision {
    /// Configured percent at decision time.
    pub percent: f64,
    /// Which identity was held constant.
    pub key: RolloutKey,
    /// The bucket this request landed in.
    pub bucket: u32,
    /// Whether it enforced.
    pub enforced: bool,
}

/// Decide the arm for one request.
#[must_use]
pub fn decide(salt: &str, rollout: &Rollout, key_value: &str) -> RolloutDecision {
    let bucket = bucket_of(salt, rollout.key, key_value);
    RolloutDecision {
        percent: rollout.percent,
        key: rollout.key,
        bucket,
        enforced: in_rollout(bucket, rollout.percent),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole design rests on: the same key always lands in the same arm, so a
    /// conversation cannot flip mid-thread and the served population stays a stable sample.
    #[test]
    fn assignment_is_stable_for_a_key() {
        let r = Rollout {
            percent: 5.0,
            key: RolloutKey::Session,
        };
        let first = decide("salt-a", &r, "session-123");
        for _ in 0..1000 {
            assert_eq!(decide("salt-a", &r, "session-123"), first);
        }
    }

    /// Different salts must reshuffle, or two deployments would enforce on exactly the same
    /// sessions and a caller who learned one deployment's arm would know the other's.
    #[test]
    fn salt_reshuffles_assignment() {
        let differing = (0..200)
            .filter(|i| {
                let k = format!("session-{i}");
                bucket_of("salt-a", RolloutKey::Session, &k)
                    != bucket_of("salt-b", RolloutKey::Session, &k)
            })
            .count();
        assert!(
            differing > 190,
            "salt barely changed assignment: {differing}/200"
        );
    }

    /// The same value under two key kinds must not inherit one assignment.
    #[test]
    fn key_kind_is_domain_separated() {
        let v = "same-identifier";
        let s = bucket_of("salt", RolloutKey::Session, v);
        let t = bucket_of("salt", RolloutKey::Tenant, v);
        let q = bucket_of("salt", RolloutKey::Request, v);
        assert!(s != t || t != q, "key kinds collapsed to one bucket");
    }

    /// A skewed hash would silently ramp to the wrong percentage — the operator would ask for 10%
    /// and get 3%, and only notice through a savings number that never moved.
    #[test]
    fn buckets_are_roughly_uniform() {
        for pct in [1.0, 5.0, 10.0, 50.0] {
            let r = Rollout {
                percent: pct,
                key: RolloutKey::Session,
            };
            let n = 20_000;
            let hits = (0..n)
                .filter(|i| decide("salt", &r, &format!("s{i}")).enforced)
                .count();
            let observed = hits as f64 / f64::from(n) * 100.0;
            assert!(
                (observed - pct).abs() < 1.0,
                "{pct}% rollout produced {observed:.2}% of {n} keys"
            );
        }
    }

    /// Both endpoints must be exact: 0 is how an operator backs out, 100 is full enforcement.
    #[test]
    fn endpoints_are_exact() {
        for i in 0..500 {
            let k = format!("s{i}");
            assert!(
                !decide(
                    "salt",
                    &Rollout {
                        percent: 0.0,
                        key: RolloutKey::Session
                    },
                    &k
                )
                .enforced
            );
            assert!(
                decide(
                    "salt",
                    &Rollout {
                        percent: 100.0,
                        key: RolloutKey::Session
                    },
                    &k
                )
                .enforced
            );
        }
    }

    /// Ramping up must never eject a key that was already enforcing — an operator going 5% → 25%
    /// expects to add traffic, not to reshuffle who is in the experiment.
    #[test]
    fn ramping_up_is_monotonic() {
        let keys: Vec<String> = (0..2000).map(|i| format!("s{i}")).collect();
        let arm = |pct: f64| -> Vec<bool> {
            keys.iter()
                .map(|k| {
                    decide(
                        "salt",
                        &Rollout {
                            percent: pct,
                            key: RolloutKey::Session,
                        },
                        k,
                    )
                    .enforced
                })
                .collect()
        };
        let (small, large) = (arm(5.0), arm(25.0));
        for (i, (s, l)) in small.iter().zip(&large).enumerate() {
            assert!(!*s || *l, "key {i} enforced at 5% but not at 25%");
        }
    }

    #[test]
    fn validate_rejects_impossible_percentages() {
        for bad in [-0.1, 100.1, f64::NAN, f64::INFINITY] {
            assert!(
                Rollout {
                    percent: bad,
                    key: RolloutKey::Session
                }
                .validate()
                .is_err()
            );
        }
        for good in [0.0, 0.01, 50.0, 100.0] {
            assert!(
                Rollout {
                    percent: good,
                    key: RolloutKey::Session
                }
                .validate()
                .is_ok()
            );
        }
    }
}
