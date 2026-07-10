//! Proxy configuration, loaded from the environment (SPEC §7.4: agent-first, zero-config
//! by default — every knob has a sane local default and can be overridden by env var).

use std::env;

use firstpass_core::{Config as RoutingConfig, Mode, PriceTable};

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
    /// Base URL of the upstream OpenAI-compatible API.
    pub upstream_openai: String,
    /// Path to the SQLite trace database.
    pub db_path: String,
    /// Tenant identifier stamped on every trace.
    pub tenant_id: String,
    /// Salt mixed into the prompt hash so raw prompt text never touches storage.
    pub prompt_salt: String,
    /// Default mode when no routing config matches (`observe` unless overridden).
    pub mode: Mode,
    /// Optional declarative routing config (§8.4). When present, its routes decide the mode,
    /// ladder, and gates per request; enforce routes activate the escalation engine.
    pub routing: Option<RoutingConfig>,
    /// Model pricing table used to cost each attempt.
    pub prices: PriceTable,
    /// Max in-flight requests before new ones queue behind the concurrency limiter
    /// (`FIRSTPASS_MAX_CONCURRENCY`, default 512). A load-shed valve, not a timeout — it never
    /// severs an in-flight SSE stream.
    pub max_concurrency: usize,
}

/// Default for [`ProxyConfig::max_concurrency`] when `FIRSTPASS_MAX_CONCURRENCY` is unset.
const DEFAULT_MAX_CONCURRENCY: usize = 512;

/// Errors that prevent the proxy from starting.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// `FIRSTPASS_MODE` named an unknown mode (valid: `observe`, `enforce`).
    #[error(
        "FIRSTPASS_MODE={0:?} is not a known mode; set `observe` or `enforce`, or leave it unset"
    )]
    UnsupportedMode(String),

    /// The routing config file could not be read or parsed.
    #[error("routing config error: {0}")]
    Config(String),
}

impl ProxyConfig {
    /// Load configuration from environment variables, falling back to local defaults.
    ///
    /// # Errors
    /// [`ConfigError::UnsupportedMode`] for an unknown `FIRSTPASS_MODE`; [`ConfigError::Config`]
    /// if the `FIRSTPASS_CONFIG` routing file cannot be read or parsed.
    pub fn from_env() -> Result<Self, ConfigError> {
        // Read the routing config file (if any) here, then hand its *content* to the pure
        // `from_lookup` seam via the synthetic `FIRSTPASS_CONFIG_TOML` key — keeping file I/O
        // out of the unit-testable path.
        let routing_toml = match env::var("FIRSTPASS_CONFIG").ok() {
            Some(path) => Some(
                std::fs::read_to_string(&path)
                    .map_err(|e| ConfigError::Config(format!("reading {path}: {e}")))?,
            ),
            None => None,
        };
        Self::from_lookup(|key| match key {
            "FIRSTPASS_CONFIG_TOML" => routing_toml.clone(),
            other => env::var(other).ok(),
        })
    }

    /// Load configuration from an arbitrary key lookup — the seam `from_env` uses, exposed
    /// so tests can exercise defaulting/validation without touching real process env vars.
    /// The routing config is supplied inline via the `FIRSTPASS_CONFIG_TOML` key.
    ///
    /// # Errors
    /// [`ConfigError::UnsupportedMode`] for an unknown `FIRSTPASS_MODE`; [`ConfigError::Config`]
    /// if the inline routing TOML fails to parse.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let bind = lookup("FIRSTPASS_BIND").unwrap_or_else(|| "127.0.0.1:8080".to_owned());
        let upstream_anthropic = lookup("FIRSTPASS_UPSTREAM_ANTHROPIC")
            .unwrap_or_else(|| "https://api.anthropic.com".to_owned());
        let upstream_openai = lookup("FIRSTPASS_UPSTREAM_OPENAI")
            .unwrap_or_else(|| "https://api.openai.com".to_owned());
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
            "enforce" => Mode::Enforce,
            other => return Err(ConfigError::UnsupportedMode(other.to_owned())),
        };
        let routing = match lookup("FIRSTPASS_CONFIG_TOML") {
            Some(toml) => {
                Some(RoutingConfig::parse(&toml).map_err(|e| ConfigError::Config(e.to_string()))?)
            }
            None => None,
        };
        let max_concurrency = match lookup("FIRSTPASS_MAX_CONCURRENCY") {
            Some(s) => s.parse().map_err(|e| {
                ConfigError::Config(format!("FIRSTPASS_MAX_CONCURRENCY={s:?}: {e}"))
            })?,
            None => DEFAULT_MAX_CONCURRENCY,
        };

        Ok(Self {
            bind,
            upstream_anthropic,
            upstream_openai,
            db_path,
            tenant_id,
            prompt_salt,
            mode,
            routing,
            prices: PriceTable::defaults(),
            max_concurrency,
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
        assert_eq!(cfg.max_concurrency, DEFAULT_MAX_CONCURRENCY);
    }

    #[test]
    fn max_concurrency_is_parsed_from_env() {
        let cfg = ProxyConfig::from_lookup(|key| {
            (key == "FIRSTPASS_MAX_CONCURRENCY").then(|| "64".to_owned())
        })
        .unwrap();
        assert_eq!(cfg.max_concurrency, 64);
    }

    #[test]
    fn bad_max_concurrency_is_an_error() {
        let result = ProxyConfig::from_lookup(|key| {
            (key == "FIRSTPASS_MAX_CONCURRENCY").then(|| "not-a-number".to_owned())
        });
        assert!(matches!(result, Err(ConfigError::Config(_))));
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
    fn enforce_mode_is_accepted() {
        let cfg =
            ProxyConfig::from_lookup(|key| (key == "FIRSTPASS_MODE").then(|| "enforce".to_owned()))
                .unwrap();
        assert_eq!(cfg.mode, Mode::Enforce);
    }

    #[test]
    fn unknown_mode_is_rejected() {
        let result =
            ProxyConfig::from_lookup(|key| (key == "FIRSTPASS_MODE").then(|| "banana".to_owned()));
        assert!(matches!(result, Err(ConfigError::UnsupportedMode(m)) if m == "banana"));
    }

    #[test]
    fn routing_config_parses_inline() {
        let toml = r#"
[[route]]
match = { task_kind = "code_edit" }
mode = "enforce"
ladder = ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]
gates = ["non-empty"]
"#;
        let cfg = ProxyConfig::from_lookup(|key| match key {
            "FIRSTPASS_CONFIG_TOML" => Some(toml.to_owned()),
            _ => None,
        })
        .unwrap();
        let routing = cfg.routing.expect("routing config present");
        assert_eq!(routing.routes.len(), 1);
        assert_eq!(routing.routes[0].mode, Mode::Enforce);
    }

    #[test]
    fn bad_routing_config_is_an_error() {
        let result = ProxyConfig::from_lookup(|key| match key {
            "FIRSTPASS_CONFIG_TOML" => Some("this is not valid = = toml".to_owned()),
            _ => None,
        });
        assert!(matches!(result, Err(ConfigError::Config(_))));
    }
}
