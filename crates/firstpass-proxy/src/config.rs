//! Proxy configuration, loaded from the environment (SPEC §7.4: agent-first, zero-config
//! by default — every knob has a sane local default and can be overridden by env var).

use std::env;

use firstpass_core::{Mode, PriceTable};

/// The built-in prompt salt used when `FIRSTPASS_PROMPT_SALT` is unset. Fine for local
/// development; an operator should set their own before handling real traffic, since a
/// shared default salt makes `prompt_hash` comparable/guessable across installs.
const DEFAULT_PROMPT_SALT: &str = "firstpass-dev-salt";

/// Runtime configuration for the proxy.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Address to bind the HTTP server to, e.g. `127.0.0.1:8080`.
    pub bind: String,
    /// Base URL of the upstream Anthropic-compatible API.
    pub upstream_anthropic: String,
    /// Path to the SQLite trace database.
    pub db_path: String,
    /// Tenant identifier stamped on every trace.
    pub tenant_id: String,
    /// Salt mixed into the prompt hash so raw prompt text never touches storage.
    pub prompt_salt: String,
    /// Routing mode. Only [`Mode::Observe`] is implemented — gating is M2.
    pub mode: Mode,
    /// Model pricing table used to cost each attempt.
    pub prices: PriceTable,
}

/// Errors that prevent the proxy from starting.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// `FIRSTPASS_MODE` named a mode this build doesn't serve yet.
    #[error(
        "FIRSTPASS_MODE={0:?} is not implemented yet (gating ships in M2); \
         set FIRSTPASS_MODE=observe or leave it unset"
    )]
    UnsupportedMode(String),
}

impl ProxyConfig {
    /// Load configuration from environment variables, falling back to local defaults.
    ///
    /// # Errors
    /// Returns [`ConfigError::UnsupportedMode`] if `FIRSTPASS_MODE` names anything other
    /// than `observe`.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(|key| env::var(key).ok())
    }

    /// Load configuration from an arbitrary key lookup — the seam `from_env` uses, exposed
    /// so tests can exercise defaulting/validation without touching real process env vars.
    ///
    /// # Errors
    /// Returns [`ConfigError::UnsupportedMode`] if `FIRSTPASS_MODE` names anything other
    /// than `observe`.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let bind = lookup("FIRSTPASS_BIND").unwrap_or_else(|| "127.0.0.1:8080".to_owned());
        let upstream_anthropic = lookup("FIRSTPASS_UPSTREAM_ANTHROPIC")
            .unwrap_or_else(|| "https://api.anthropic.com".to_owned());
        let db_path = lookup("FIRSTPASS_DB").unwrap_or_else(|| "firstpass.db".to_owned());
        let tenant_id = lookup("FIRSTPASS_TENANT").unwrap_or_else(|| "default".to_owned());
        let prompt_salt = lookup("FIRSTPASS_PROMPT_SALT").unwrap_or_else(|| {
            tracing::warn!(
                "FIRSTPASS_PROMPT_SALT is unset — using the built-in dev default; \
                 set a real secret before handling production traffic"
            );
            DEFAULT_PROMPT_SALT.to_owned()
        });
        let mode_str = lookup("FIRSTPASS_MODE").unwrap_or_else(|| "observe".to_owned());
        let mode = match mode_str.as_str() {
            "observe" => Mode::Observe,
            other => return Err(ConfigError::UnsupportedMode(other.to_owned())),
        };

        Ok(Self {
            bind,
            upstream_anthropic,
            db_path,
            tenant_id,
            prompt_salt,
            mode,
            prices: PriceTable::defaults(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane_when_unset() {
        let cfg = ProxyConfig::from_lookup(|_| None).unwrap();
        assert_eq!(cfg.bind, "127.0.0.1:8080");
        assert_eq!(cfg.upstream_anthropic, "https://api.anthropic.com");
        assert_eq!(cfg.db_path, "firstpass.db");
        assert_eq!(cfg.tenant_id, "default");
        assert_eq!(cfg.prompt_salt, DEFAULT_PROMPT_SALT);
        assert_eq!(cfg.mode, Mode::Observe);
    }

    #[test]
    fn overrides_are_applied() {
        let cfg = ProxyConfig::from_lookup(|key| match key {
            "FIRSTPASS_BIND" => Some("0.0.0.0:9090".to_owned()),
            "FIRSTPASS_TENANT" => Some("acme".to_owned()),
            _ => None,
        })
        .unwrap();
        assert_eq!(cfg.bind, "0.0.0.0:9090");
        assert_eq!(cfg.tenant_id, "acme");
    }

    #[test]
    fn enforce_mode_is_rejected() {
        let result =
            ProxyConfig::from_lookup(|key| (key == "FIRSTPASS_MODE").then(|| "enforce".to_owned()));
        assert!(matches!(result, Err(ConfigError::UnsupportedMode(m)) if m == "enforce"));
    }
}
