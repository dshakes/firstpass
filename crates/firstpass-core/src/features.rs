//! The request feature vector (SPEC §9.2) — deterministic, versioned, privacy-preserving.
//!
//! Features are the input to routing policy. They are extracted deterministically from a
//! request so that the same request always produces the same vector (a precondition for a
//! re-derivable audit trail), and they are **privacy-preserving by construction**: no raw
//! prompt text, only coarse buckets and salted hashes. The vector is versioned
//! ([`FEATURE_VERSION`]); a change to how any feature is computed bumps the version so old
//! traces remain interpretable.

use serde::{Deserialize, Serialize};

/// Version of the feature-extraction contract. Bump on any change to how a feature is
/// computed. Recorded per trace as `features@vN`.
pub const FEATURE_VERSION: u32 = 1;

/// Coarse task classification. `Other` is the safe default when classification is uncertain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    /// Editing or writing code (the primary M0/M1 target).
    CodeEdit,
    /// Generating or repairing tests.
    TestGen,
    /// Read-only investigation / search / navigation.
    Explore,
    /// Reviewing or critiquing existing work.
    Review,
    /// Structured extraction / classification.
    Extract,
    /// Free-form conversation.
    Chat,
    /// Anything not confidently classified (the safe default).
    #[default]
    Other,
}

/// The per-request feature vector (§9.2).
///
/// Rolling per-bucket statistics (e.g. `prior_rung_clearance`) are intentionally **not**
/// here — they live in the trace store, not the deterministic per-request contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Features {
    /// Feature-extraction contract version (`features@vN`).
    pub version: u32,
    /// Coarse task classification.
    pub task_kind: TaskKind,
    /// Programming language, when known (lowercased identifier, e.g. `"rust"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Calling agent identity, when known (e.g. `"compass"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// Calling subagent identity, when known (e.g. `"test-runner"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent: Option<String>,
    /// Coarse bucket of the prompt's token count (see [`token_bucket`]) — never the raw count.
    pub prompt_token_bucket: u32,
    /// Number of tools/functions offered in the request.
    pub tool_count: u32,
    /// Whether the request carried image content.
    pub has_images: bool,
    /// Salted, truncated hash of the repository identity (see [`repo_fingerprint`]) — never a path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_fingerprint: Option<String>,
    /// How many attempts have already failed in this session (drives session promotion, §8.4).
    pub session_failure_count: u32,
    /// Hour-of-day bucket in UTC, `0..=23` (see [`hour_bucket`]).
    pub hour_bucket: u8,
}

impl Features {
    /// A minimal vector stamped with the current [`FEATURE_VERSION`] and the given task kind.
    #[must_use]
    pub fn new(task_kind: TaskKind) -> Self {
        Self {
            version: FEATURE_VERSION,
            task_kind,
            language: None,
            agent: None,
            subagent: None,
            prompt_token_bucket: 0,
            tool_count: 0,
            has_images: false,
            repo_fingerprint: None,
            session_failure_count: 0,
            hour_bucket: 0,
        }
    }
}

/// Bucket a token count into a coarse, privacy-preserving band: `floor(log2(n))`, with
/// `0` and `1` both mapping to bucket `0`.
///
/// Monotonic non-decreasing in `n`, so ordering is preserved while the exact count is not
/// recoverable. Deterministic — the same `n` always yields the same bucket.
#[must_use]
pub fn token_bucket(n: u64) -> u32 {
    if n < 2 {
        0
    } else {
        // 63 - leading_zeros == floor(log2(n)) for n >= 1.
        63 - n.leading_zeros()
    }
}

/// Salted, truncated fingerprint of a repository identity: the first 16 hex chars of
/// `SHA-256(salt || 0x00 || repo)`.
///
/// The salt is a per-deployment secret; without it the fingerprint is not reversible to the
/// repo identity, and cross-deployment correlation is prevented. Deterministic for a fixed salt.
#[must_use]
pub fn repo_fingerprint(salt: &str, repo: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(salt.as_bytes());
    h.update([0u8]); // domain separator so salt||repo can't collide with a different split
    h.update(repo.as_bytes());
    let digest = h.finalize();
    hex::encode(&digest[..8]) // 8 bytes -> 16 hex chars
}

/// Hour-of-day bucket in UTC (`0..=23`) for a timestamp — a routing feature (traffic varies
/// by hour) that leaks nothing finer than the hour.
#[must_use]
pub fn hour_bucket(ts: jiff::Timestamp) -> u8 {
    // Convert to a UTC civil time; hour is 0..=23.
    ts.to_zoned(jiff::tz::TimeZone::UTC).hour() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_bucket_is_monotonic_and_coarse() {
        assert_eq!(token_bucket(0), 0);
        assert_eq!(token_bucket(1), 0);
        assert_eq!(token_bucket(2), 1);
        assert_eq!(token_bucket(3), 1);
        assert_eq!(token_bucket(4), 2);
        assert_eq!(token_bucket(1024), 10);
        // monotonic non-decreasing
        let mut last = 0;
        for n in 0..5000u64 {
            let b = token_bucket(n);
            assert!(b >= last);
            last = b;
        }
    }

    #[test]
    fn repo_fingerprint_is_deterministic_salted_and_truncated() {
        let a = repo_fingerprint("salt1", "github.com/acme/api");
        assert_eq!(a, repo_fingerprint("salt1", "github.com/acme/api")); // deterministic
        assert_ne!(a, repo_fingerprint("salt2", "github.com/acme/api")); // salt-sensitive
        assert_ne!(a, repo_fingerprint("salt1", "github.com/acme/web")); // repo-sensitive
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        // domain separator: "a"+"bc" must not collide with "ab"+"c"
        assert_ne!(repo_fingerprint("a", "bc"), repo_fingerprint("ab", "c"));
    }

    #[test]
    fn hour_bucket_in_range() {
        // 2026-07-08T15:04:05Z -> hour 15 UTC
        let ts: jiff::Timestamp = "2026-07-08T15:04:05Z".parse().unwrap();
        assert_eq!(hour_bucket(ts), 15);
        let midnight: jiff::Timestamp = "2026-01-01T00:00:00Z".parse().unwrap();
        assert_eq!(hour_bucket(midnight), 0);
    }

    #[test]
    fn features_default_version_stamped() {
        let f = Features::new(TaskKind::CodeEdit);
        assert_eq!(f.version, FEATURE_VERSION);
        assert_eq!(f.task_kind, TaskKind::CodeEdit);
    }
}
