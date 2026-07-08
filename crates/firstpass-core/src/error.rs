//! Error types for the domain contract.

/// Errors produced while constructing or validating domain values.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A [`crate::Score`] was outside the closed unit interval `[0, 1]`.
    #[error("score {0} out of range [0,1]")]
    InvalidScore(f64),

    /// A model reference was not of the form `provider/model`.
    #[error("invalid model reference {0:?} (expected `provider/model`)")]
    BadModelRef(String),

    /// A duration string was not of the form `<int><unit>` (`s`/`m`/`h`/`d`).
    #[error("invalid duration {0:?} (expected e.g. `30m`, `2h`)")]
    BadDuration(String),

    /// Pricing was requested for a model with no entry in the price table.
    #[error("unknown model {0:?} — add it to the price table")]
    UnknownModel(String),

    /// The audit hash chain did not verify.
    #[error("hash chain broken at index {index}: {detail}")]
    ChainBroken {
        /// Zero-based position of the first record that failed verification.
        index: usize,
        /// Human-readable reason (link mismatch or self-hash mismatch).
        detail: String,
    },

    /// TOML config failed to parse.
    #[error("config parse error: {0}")]
    Config(#[from] toml::de::Error),

    /// JSON (de)serialization failed — e.g. during canonicalization.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias for fallible domain operations.
pub type Result<T> = std::result::Result<T, Error>;
