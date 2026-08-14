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
/// Feature-extraction contract version.
///
/// **v2** added [`Features::difficulty_hint`] (trajectory signals read from the agent's own
/// conversation). The bump is required even though the field is `#[serde(default)]` and old traces
/// still deserialize: the version says *how a vector was computed*, and a v1 trace genuinely had no
/// hint available. Leaving it at 1 would let a policy fitted on v2 traffic be replayed against v1
/// traces as though the missing hints were real zeroes — silently mixing "no signal" with "not
/// measured", which is the class of error the version field exists to prevent.
pub const FEATURE_VERSION: u32 = 2;

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

/// Raw counts read off an agent conversation, before they are collapsed into a
/// [`DifficultyHint`].
///
/// Deliberately counts, not content: how many tool results failed, not what they said. A router
/// needs to know a session is going badly; it does not need the stack trace, and [`Features`] is
/// not a place raw prompt text may ever land.
///
/// Extraction lives in the proxy (it parses HTTP bodies, which is I/O-shaped); the *scoring* lives
/// here so it stays deterministic, versioned, and testable without a server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TrajectorySignals {
    /// Tool results in the recent window that reported an error.
    pub tool_errors: u32,
    /// Tool results in the recent window, total. Zero means "no tool activity to judge".
    pub tool_results: u32,
    /// Assistant turns in the conversation — a proxy for how long this has been going on.
    pub assistant_turns: u32,
    /// Repeated identical tool invocations: the agent trying the same thing again.
    pub repeated_tool_calls: u32,
}

/// How hard an agent's own conversation looks. Ordinal, four levels, deliberately coarse.
///
/// Four levels rather than a continuous score, because this feeds
/// `ContextBucket` and every extra distinction multiplies the bandit's state space — the arms then
/// share less traffic and learn slower. A signal too fine to learn from is worse than a coarse one:
/// it looks more informative and performs worse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum DifficultyHint {
    /// No usable signal: a single-shot request, a non-agent client, or an unparseable body.
    #[default]
    None = 0,
    /// Some tool activity, nothing going wrong.
    Low = 1,
    /// Errors or repetition appearing.
    Medium = 2,
    /// Sustained failure: the agent is stuck.
    High = 3,
}

impl DifficultyHint {
    /// Score signals into a hint.
    ///
    /// The thresholds below are a **prior, not a finding**. They are a reasonable starting shape,
    /// and the bandit is what actually learns whether a given bucket deserves a higher start rung —
    /// so the job here is to separate genuinely different situations into different buckets, not to
    /// be right about which is harder. Getting the cut points slightly wrong costs a little
    /// learning rate; collapsing distinct situations into one bucket costs the signal entirely.
    ///
    /// No tool activity yields [`DifficultyHint::None`] rather than [`DifficultyHint::Low`]: a
    /// request with no tools has told us nothing, and "nothing" must not be confused with "fine".
    #[must_use]
    pub fn score(s: TrajectorySignals) -> Self {
        if s.tool_results == 0 && s.repeated_tool_calls == 0 {
            return Self::None;
        }
        // Integer-ratio comparisons rather than float division: this feeds a hash bucket and a
        // versioned audit field, so it must be bit-identical everywhere, and `a/b > 0.5` is not
        // something to trust across platforms when `a*2 > b` says it exactly.
        let e = s.tool_errors;
        let n = s.tool_results;
        // `e > n / 2`, not `e * 2 > n`: the multiplication overflows for `e > u32::MAX/2`, which
        // panics in debug builds. Unreachable through the proxy (its window caps the counts), but
        // this is public API in an I/O-free crate — anyone may call it with any values, and
        // "unreachable through the one caller I was thinking about" is not a bound.
        //
        // Integer division truncates, so `e > n/2` is very slightly stricter than a true majority
        // at odd `n` (n=5 needs e>2, i.e. 3 of 5 — which IS the majority). Same semantics, no
        // overflow. Flagged in review.
        let half_failing = n > 0 && e > n / 2;
        let any_failing = e > 0;
        let stuck = s.repeated_tool_calls >= 2;
        let deep = s.assistant_turns >= 8;

        match (half_failing || stuck, any_failing, deep) {
            // Most calls failing, or the same call retried repeatedly. Being deep on top of that is
            // the clearest "stuck" signature there is.
            (true, _, true) => Self::High,
            (true, _, false) => Self::Medium,
            // Some errors, and the conversation has been running a while.
            (false, true, true) => Self::Medium,
            (false, true, false) => Self::Low,
            // Tools in use, nothing failing.
            (false, false, _) => Self::Low,
        }
    }

    /// The wire value stored in [`Features::difficulty_hint`].
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Whether a difficulty hint carries no signal, and so should be omitted from the wire form.
///
/// Used by [`Features::difficulty_hint`]'s `skip_serializing_if` to keep v1 traces byte-identical
/// under re-serialization — see that field's note on the hash chain.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's skip_serializing_if requires a &T predicate"
)]
fn is_no_signal(hint: &u8) -> bool {
    *hint == DifficultyHint::None as u8
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
    /// How hard the *agent's own conversation* looks, `0..=3` — see [`DifficultyHint`].
    ///
    /// Distinct from [`Features::session_failure_count`], and the distinction is the whole point.
    /// That field counts **our gate's** failures: endogenous evidence, available only after we have
    /// already paid for a failed attempt. This one reads **the agent's** failures out of the
    /// transcript it sent us — failing tool calls, repeated identical actions, a long unproductive
    /// exchange — which is evidence we get *before* spending anything, and which is present on turns
    /// where our own gate would have passed.
    ///
    /// `0` means "no signal", which is what a single-shot request, a non-agent client, and an
    /// unparseable body all produce. It is a routing *hint* and never a serving decision: it may
    /// choose which rung to start on, never what is fit to serve. Only a gate decides that.
    ///
    /// **`skip_serializing_if` is load-bearing for the hash chain, not a size optimisation.** A v1
    /// trace's JSON has no such field; serde defaults it to `0` on read, and if re-serialization
    /// emitted `"difficulty_hint":0` the canonical JSON would differ from the bytes that were
    /// originally hashed. Every historical receipt would then re-derive to a different hash and
    /// `firstpass verify` would report `TAMPERED` across the whole store — breaking invariant #1
    /// ("the hash chain is re-derivable") on upgrade, with no tampering having occurred.
    ///
    /// Omitting the default is exactly right rather than merely convenient: a v1 trace *had* no
    /// hint, and a receipt should say what was measured, not what a later version would have
    /// defaulted to. A v2 request carrying a real hint serialises it normally.
    #[serde(default, skip_serializing_if = "is_no_signal")]
    pub difficulty_hint: u8,
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
            difficulty_hint: DifficultyHint::None as u8,
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

    /// A request with no tool activity must score `None`, not `Low`.
    ///
    /// "I have no evidence" and "I have evidence that things are fine" are different states, and
    /// conflating them would make every single-shot request look like a healthy agent session —
    /// polluting the bandit's healthiest bucket with traffic that carries no trajectory signal
    /// whatsoever.
    #[test]
    fn absent_tool_activity_is_no_signal_not_a_good_signal() {
        assert_eq!(
            DifficultyHint::score(TrajectorySignals::default()),
            DifficultyHint::None
        );
        // Turn depth alone is not evidence of difficulty: a long, smooth conversation is still
        // smooth, and charging it a higher start rung would be a pure cost regression.
        assert_eq!(
            DifficultyHint::score(TrajectorySignals {
                assistant_turns: 50,
                ..Default::default()
            }),
            DifficultyHint::None
        );
    }

    /// Clean tool use scores `Low`; a majority-failing window scores higher. This is the ordering
    /// the whole feature rests on — if these two land in the same bucket, there is no signal.
    #[test]
    fn a_failing_window_outranks_a_clean_one() {
        let clean = DifficultyHint::score(TrajectorySignals {
            tool_results: 6,
            tool_errors: 0,
            assistant_turns: 4,
            repeated_tool_calls: 0,
        });
        let failing = DifficultyHint::score(TrajectorySignals {
            tool_results: 6,
            tool_errors: 5,
            assistant_turns: 4,
            repeated_tool_calls: 0,
        });
        assert_eq!(clean, DifficultyHint::Low);
        assert!(
            failing > clean,
            "5-of-6 failing must outrank 0-of-6: {failing:?} vs {clean:?}"
        );
    }

    /// Repetition alone is a stuck signal, even with no reported errors.
    ///
    /// This is the case an error-count-only heuristic misses entirely: an agent re-running the same
    /// command that "succeeds" each time while making no progress reports zero errors and is
    /// obviously stuck. Switchyard's stage router reads exactly this, and it is why the feature
    /// counts repeats separately rather than folding them into an error rate.
    #[test]
    fn repetition_is_a_stuck_signal_without_any_errors() {
        let hint = DifficultyHint::score(TrajectorySignals {
            tool_results: 4,
            tool_errors: 0,
            assistant_turns: 5,
            repeated_tool_calls: 3,
        });
        assert!(
            hint >= DifficultyHint::Medium,
            "3 repeated calls with no errors must still read as difficulty, got {hint:?}"
        );
    }

    /// Sustained failure deep into a conversation is the top bucket, and the depth matters: the
    /// same failure rate early is recoverable, late it is a pattern.
    #[test]
    fn sustained_failure_deep_in_a_session_is_the_top_bucket() {
        let deep = DifficultyHint::score(TrajectorySignals {
            tool_results: 10,
            tool_errors: 8,
            assistant_turns: 12,
            repeated_tool_calls: 2,
        });
        let shallow = DifficultyHint::score(TrajectorySignals {
            tool_results: 10,
            tool_errors: 8,
            assistant_turns: 2,
            repeated_tool_calls: 0,
        });
        assert_eq!(deep, DifficultyHint::High);
        assert!(
            shallow < deep,
            "the same failure rate early must not score as high as late: {shallow:?} vs {deep:?}"
        );
    }

    /// Extreme counts must not panic. `score` is public API in an I/O-free crate, so it is callable
    /// with any values — the proxy's window is one caller's bound, not the function's. `e * 2`
    /// overflowed for `e > u32::MAX/2` and panicked in debug. Flagged in review.
    #[test]
    fn extreme_counts_do_not_overflow() {
        let h = DifficultyHint::score(TrajectorySignals {
            tool_errors: u32::MAX,
            tool_results: u32::MAX,
            assistant_turns: u32::MAX,
            repeated_tool_calls: u32::MAX,
        });
        assert_eq!(h, DifficultyHint::High, "all-failing at any scale is High");

        // And the majority semantics are unchanged at ordinary sizes: 3-of-5 is a majority, 2-of-5
        // is not. Integer division must not have shifted the cut point.
        assert!(
            DifficultyHint::score(TrajectorySignals {
                tool_errors: 3,
                tool_results: 5,
                ..Default::default()
            }) >= DifficultyHint::Medium
        );
        assert_eq!(
            DifficultyHint::score(TrajectorySignals {
                tool_errors: 2,
                tool_results: 5,
                ..Default::default()
            }),
            DifficultyHint::Low,
            "2 of 5 is not a majority and must stay Low"
        );
    }

    /// The hint is bounded. It indexes a bandit bucket, so an unbounded value would silently
    /// fragment the state space; and it is a versioned audit field, where an out-of-range value is
    /// a contract violation rather than a curiosity.
    #[test]
    fn the_hint_is_always_within_its_declared_range() {
        for tool_results in 0..12u32 {
            for tool_errors in 0..=tool_results {
                for assistant_turns in [0u32, 1, 7, 8, 40] {
                    for repeated_tool_calls in 0..4u32 {
                        let h = DifficultyHint::score(TrajectorySignals {
                            tool_errors,
                            tool_results,
                            assistant_turns,
                            repeated_tool_calls,
                        });
                        assert!(h.as_u8() <= DifficultyHint::High.as_u8());
                    }
                }
            }
        }
    }

    /// A v1 trace, written before this field existed, must still deserialize — and must land on
    /// `None` rather than any other level. Old receipts stay readable; that is a hard requirement
    /// of a tamper-evident log, since an auditor cannot re-derive what will not parse.
    #[test]
    fn a_v1_trace_without_the_hint_still_deserializes() {
        let v1 = r#"{"version":1,"task_kind":"other","prompt_token_bucket":3,
                     "tool_count":0,"has_images":false,"session_failure_count":0,"hour_bucket":9}"#;
        let f: Features = serde_json::from_str(v1).expect("a v1 trace must still parse");
        assert_eq!(
            f.version, 1,
            "the stored version must be preserved, not rewritten"
        );
        assert_eq!(
            f.difficulty_hint,
            DifficultyHint::None as u8,
            "a missing hint must read as no-signal"
        );
    }

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

#[cfg(test)]
mod v1_chain_regression {
    use super::*;

    /// A v1 trace's hash must be re-derivable after v2 adds a field.
    ///
    /// Reviewer flagged this as a blocking regression, and the mechanism is exactly right in shape:
    /// v1 JSON has no `difficulty_hint`, serde defaults it to 0 on read, and if re-serialization
    /// emits `"difficulty_hint":0` the canonical JSON differs from what was hashed — every historical
    /// receipt reads as TAMPERED. That is invariant #1 in CLAUDE.md ("the hash chain is
    /// re-derivable"), so it gets a test rather than an argument.
    #[test]
    fn a_v1_feature_vector_round_trips_to_its_original_canonical_json() {
        let v1 = r#"{"version":1,"task_kind":"other","prompt_token_bucket":3,"tool_count":0,"has_images":false,"session_failure_count":0,"hour_bucket":9}"#;
        let f: Features = serde_json::from_str(v1).expect("v1 must parse");
        let reserialized = serde_json::to_string(&f).expect("must serialize");
        assert!(
            !reserialized.contains("difficulty_hint"),
            "re-serializing a v1 vector must NOT emit the v2 field, or every historical receipt \
             re-derives to a different hash and reads as TAMPERED: {reserialized}"
        );
    }

    /// **The chain-level proof**, not just the serde-level one.
    ///
    /// The serde test above catches the field appearing; this catches what that would COST. A v1
    /// receipt's hash was computed over v1 canonical JSON. If today's binary re-derives a different
    /// hash from the same stored record, `firstpass verify` reports TAMPERED across the entire
    /// history — the auditor sees fraud where there was only an upgrade, which is the single worst
    /// failure this system can have.
    #[test]
    fn a_v1_receipt_hash_is_still_re_derivable_by_todays_binary() {
        let v1 = r#"{"version":1,"task_kind":"other","prompt_token_bucket":3,"tool_count":0,"has_images":false,"session_failure_count":0,"hour_bucket":9}"#;
        // What the v1 binary would have hashed: the canonical form of the v1 document itself.
        let original: serde_json::Value = serde_json::from_str(v1).expect("v1 must parse");
        let hash_then = crate::hashchain::record_hash(&original).expect("must hash");

        // What today's binary derives after a round trip through the v2 struct.
        let f: Features = serde_json::from_str(v1).expect("v1 must parse into v2 struct");
        let hash_now = crate::hashchain::record_hash(&f).expect("must hash");

        assert_eq!(
            hash_then, hash_now,
            "a v1 record must re-derive to its original hash; a mismatch is reported to the \
             operator as TAMPERED, with no tampering having occurred"
        );
    }

    /// ...and a v2 vector that actually carries a hint must still emit it, or the field is
    /// unauditable — present in routing, absent from the receipt that is supposed to explain it.
    #[test]
    fn a_v2_vector_with_a_real_hint_still_serializes_it() {
        let mut f = Features::new(TaskKind::CodeEdit);
        f.difficulty_hint = DifficultyHint::High.as_u8();
        let s = serde_json::to_string(&f).expect("must serialize");
        assert!(
            s.contains("\"difficulty_hint\":3"),
            "a non-default hint must appear in the receipt: {s}"
        );
    }
}
