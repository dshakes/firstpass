//! Proxy configuration, loaded from the environment (SPEC §7.4: agent-first, zero-config
//! by default — every knob has a sane local default and can be overridden by env var).

use std::env;
use std::num::NonZeroU32;

use firstpass_core::{Config as RoutingConfig, Mode, PriceTable};

/// The built-in prompt salt used when `FIRSTPASS_PROMPT_SALT` is unset. Fine for local
/// development; an operator should set their own before handling real traffic, since a
/// shared default salt makes `prompt_hash` comparable/guessable across installs.
const DEFAULT_PROMPT_SALT: &str = "firstpass-dev-salt";

/// Controls what happens to audit traces when the background writer channel is full.
///
/// - [`BestEffort`](ReceiptsMode::BestEffort) — `try_send` drops the trace and increments
///   `firstpass_traces_dropped_total`; bounded memory, zero added latency. The hash chain over
///   persisted traces stays valid; a dropped trace is simply absent.
/// - [`Durable`](ReceiptsMode::Durable) — on `TrySendError::Full` the trace is serialised as a
///   JSON line and appended (with `sync_data`) to `<db_path>.spill.jsonl`. The background writer
///   drains the spill file at startup and whenever the channel empties, inserting spilled traces
///   BEFORE new channel arrivals so the hash chain remains append-only and valid.
///   The blocking fs append on the hot path is the DELIBERATE tradeoff of durable mode —
///   it only fires under sustained backpressure; switch to `best_effort` if disk latency
///   under load is unacceptable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReceiptsMode {
    /// Drop traces when the writer channel is full (default — bounded memory, zero latency impact).
    #[default]
    BestEffort,
    /// Spill traces to disk when the writer channel is full; drain and recover on the next boot
    /// or whenever the channel empties.
    Durable,
}

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
    /// Tenant identifier stamped on every trace when `require_auth` is off (single-operator
    /// default). When `require_auth` is on, the per-request tenant comes from the verified API
    /// key instead, and this is only the fallback for operator-scoped tooling.
    pub tenant_id: String,
    /// Multi-tenant auth gate (ADR 0004 §D1). Default `false` — single-operator behavior is
    /// byte-identical: no auth checks, tenant is the static [`Self::tenant_id`]. Set via
    /// `FIRSTPASS_REQUIRE_AUTH`. **Experimental / pre-external-review (ADR 0004 §D7).**
    pub require_auth: bool,
    /// Per-tenant Argon2id API-key hashes (ADR 0004 §D1). Empty unless configured via
    /// `FIRSTPASS_TENANT_KEYS` (path to a JSON `{ tenant_id: hash }`) or `FIRSTPASS_TENANT_KEYS_JSON`
    /// (inline JSON). Only consulted when `require_auth` is on.
    ///
    /// The stored `hash` is the Argon2id hash of a tenant's **secret** (make one with
    /// `TenantKeys::hash_key`). A tenant then authenticates with the key `<tenant_id>.<secret>` —
    /// the tenant id names which hash to verify, so a request does exactly one Argon2 check. Tenant
    /// ids must not contain a `.` (the first `.` splits id from secret).
    pub tenant_keys: crate::tenant_auth::TenantKeys,
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
    /// Per-tenant request rate limit, requests/sec (ADR 0004 §D6). `None` (the default) means
    /// unlimited — single-operator and existing deployments are unaffected. Set via
    /// `FIRSTPASS_TENANT_RATE_PER_SEC`; only enforced when configured.
    pub tenant_rate_per_sec: Option<NonZeroU32>,
    /// Whether to guarantee audit-trace durability under backpressure (`FIRSTPASS_RECEIPTS`).
    /// Default [`ReceiptsMode::BestEffort`] is byte-identical to the previous behavior.
    pub receipts_mode: ReceiptsMode,
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

    /// `FIRSTPASS_RECEIPTS` named an unknown mode (valid: `best_effort`, `durable`).
    #[error("FIRSTPASS_RECEIPTS={0:?} is not valid; set `best_effort` (default) or `durable`")]
    UnsupportedReceiptsMode(String),

    /// The routing config file could not be read or parsed.
    #[error("routing config error: {0}")]
    Config(String),

    /// The multi-tenant auth config (`FIRSTPASS_TENANT_KEYS`) is invalid (ADR 0004 §D1). The
    /// message never echoes hash material — only the structural problem.
    #[error("tenant auth config error: {0}")]
    Auth(#[from] crate::tenant_auth::AuthConfigError),
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
        // Read the tenant-keys file (if any) here and hand its *content* to `from_lookup` via the
        // synthetic `FIRSTPASS_TENANT_KEYS_JSON` key, keeping file I/O out of the testable path.
        // An inline `FIRSTPASS_TENANT_KEYS_JSON` in the real env takes precedence if both are set.
        let tenant_keys_json = match env::var("FIRSTPASS_TENANT_KEYS").ok() {
            Some(path) => Some(
                std::fs::read_to_string(&path)
                    .map_err(|e| ConfigError::Config(format!("reading {path}: {e}")))?,
            ),
            None => None,
        };
        Self::from_lookup(|key| match key {
            "FIRSTPASS_CONFIG_TOML" => routing_toml.clone(),
            "FIRSTPASS_TENANT_KEYS_JSON" => env::var(key).ok().or_else(|| tenant_keys_json.clone()),
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

        // Multi-tenant auth (ADR 0004 §D1) — default OFF. Anything other than an explicit truthy
        // value leaves single-operator behavior byte-identical.
        let require_auth = lookup("FIRSTPASS_REQUIRE_AUTH")
            .map(|s| {
                matches!(
                    s.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        let tenant_keys = match lookup("FIRSTPASS_TENANT_KEYS_JSON") {
            Some(json) => crate::tenant_auth::TenantKeys::from_json(&json)?,
            None => crate::tenant_auth::TenantKeys::default(),
        };
        if require_auth && tenant_keys.is_empty() {
            tracing::warn!(
                "FIRSTPASS_REQUIRE_AUTH is on but no tenant keys are configured — every request \
                 will be rejected with 401 until FIRSTPASS_TENANT_KEYS is set"
            );
        }

        // Per-tenant rate limiting (ADR 0004 §D6) — default OFF (unlimited).
        let tenant_rate_per_sec = match lookup("FIRSTPASS_TENANT_RATE_PER_SEC") {
            Some(s) => Some(s.trim().parse::<NonZeroU32>().map_err(|e| {
                ConfigError::Config(format!("FIRSTPASS_TENANT_RATE_PER_SEC={s:?}: {e}"))
            })?),
            None => None,
        };

        // Receipts durability mode — default BestEffort (byte-identical to pre-Phase-3 behavior).
        let receipts_mode = match lookup("FIRSTPASS_RECEIPTS").as_deref() {
            None | Some("best_effort") => ReceiptsMode::BestEffort,
            Some("durable") => ReceiptsMode::Durable,
            Some(other) => {
                return Err(ConfigError::UnsupportedReceiptsMode(other.to_owned()));
            }
        };

        Ok(Self {
            bind,
            upstream_anthropic,
            upstream_openai,
            db_path,
            tenant_id,
            require_auth,
            tenant_keys,
            prompt_salt,
            mode,
            prices: {
                // Operator [[price]] overrides pin THIS deployment's real prices onto the
                // built-in defaults — savings math tracks the contract, not a stale list.
                let mut prices = PriceTable::defaults();
                if let Some(cfg) = routing.as_ref() {
                    for p in &cfg.price_defs {
                        prices = prices.with_override(
                            p.model.clone(),
                            firstpass_core::ModelPrice {
                                input_per_mtok: p.input_per_mtok,
                                output_per_mtok: p.output_per_mtok,
                            },
                        );
                    }
                }
                prices
            },
            routing,
            max_concurrency,
            tenant_rate_per_sec,
            receipts_mode,
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

    #[test]
    fn receipts_mode_defaults_to_best_effort() {
        let cfg = ProxyConfig::from_lookup(|_| None).unwrap();
        assert_eq!(cfg.receipts_mode, ReceiptsMode::BestEffort);
    }

    #[test]
    fn receipts_mode_durable_is_accepted() {
        let cfg =
            ProxyConfig::from_lookup(|k| (k == "FIRSTPASS_RECEIPTS").then(|| "durable".to_owned()))
                .unwrap();
        assert_eq!(cfg.receipts_mode, ReceiptsMode::Durable);
    }

    #[test]
    fn receipts_mode_best_effort_explicit_is_accepted() {
        let cfg = ProxyConfig::from_lookup(|k| {
            (k == "FIRSTPASS_RECEIPTS").then(|| "best_effort".to_owned())
        })
        .unwrap();
        assert_eq!(cfg.receipts_mode, ReceiptsMode::BestEffort);
    }

    #[test]
    fn receipts_mode_unknown_is_rejected() {
        let result = ProxyConfig::from_lookup(|k| {
            (k == "FIRSTPASS_RECEIPTS").then(|| "never_drop".to_owned())
        });
        assert!(
            matches!(result, Err(ConfigError::UnsupportedReceiptsMode(m)) if m == "never_drop")
        );
    }

    #[test]
    fn price_overrides_reach_the_price_table() {
        let toml = "[[route]]\nmatch = {}\nmode = \"observe\"\nladder = [\"anthropic/claude-haiku-4-5\"]\n\n[[price]]\nmodel = \"anthropic/claude-haiku-4-5\"\ninput_per_mtok = 2.0\noutput_per_mtok = 10.0\n";
        let cfg = ProxyConfig::from_lookup(|k| match k {
            "FIRSTPASS_CONFIG_TOML" => Some(toml.to_owned()),
            _ => None,
        })
        .unwrap();
        // 1000 in + 1000 out at 2.0/10.0 per Mtok = 0.002 + 0.010.
        let cost = cfg
            .prices
            .cost_usd("anthropic/claude-haiku-4-5", 1000, 1000)
            .unwrap();
        assert!((cost - 0.012).abs() < 1e-9, "override must win: {cost}");
    }
}
