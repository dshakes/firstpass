//! Multi-tenant API-key authentication (ADR 0004 §D1) — the hosted plane's front door.
//!
//! **EXPERIMENTAL / pre-external-review (ADR 0004 §D7).** This is the *foundation* for tenant
//! isolation; it has NOT yet had the independent, non-author security review the ADR requires.
//! Do not rely on it as a hard isolation boundary for real, mutually-distrusting tenants until
//! that review lands. It is default-OFF (`FIRSTPASS_REQUIRE_AUTH=false`): with auth disabled the
//! single-operator path is unchanged and this module only stamps the static default tenant.
//!
//! Keys are verified against **Argon2id** password hashes via the crate's own constant-time
//! [`PasswordVerifier`] path (ADR 0004 §D1) — we never roll our own comparison. A missing or
//! invalid key returns `401` with an opaque body, so there is no "unknown tenant" oracle.

use std::collections::HashMap;

use argon2::Argon2;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::error::ProxyError;
use crate::proxy::AppState;

/// Prefix for the `Authorization: Bearer <key>` header.
const BEARER_PREFIX: &str = "Bearer ";
/// Alternative header carrying the raw tenant API key.
const KEY_HEADER: &str = "x-firstpass-key";

/// A resolved tenant identity, injected into request extensions by [`auth_middleware`].
///
/// This is the *only* source of a request's tenant: it comes either from a verified API key
/// (auth on) or the static default (auth off) — never from anything in the request body.
#[derive(Clone, Debug)]
pub struct TenantId(pub String);

/// Argon2id-hashed per-tenant API keys, keyed by tenant id.
///
/// A presented key has the form `<tenant_id>.<secret>`: the (non-secret) tenant id names which
/// hash to check, so verification is O(1) — exactly one Argon2 verify — rather than one per tenant.
///
/// `Debug` deliberately redacts every hash — they are password-equivalent secrets — revealing
/// only the configured tenant ids so a `ProxyConfig` dump can never leak credential material.
#[derive(Clone, Default)]
pub struct TenantKeys {
    /// `tenant_id` -> PHC-encoded Argon2id hash of that tenant's secret.
    hashes: HashMap<String, String>,
    /// A valid Argon2 hash of a throwaway secret, verified against on the unknown-tenant path so an
    /// invalid key takes the same time whether or not the named tenant exists — no existence oracle.
    decoy: String,
}

impl std::fmt::Debug for TenantKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantKeys")
            .field("tenants", &self.hashes.keys().collect::<Vec<_>>())
            .field("hashes", &"<redacted>")
            .finish()
    }
}

/// Errors building [`TenantKeys`] or hashing a key.
#[derive(Debug, thiserror::Error)]
pub enum AuthConfigError {
    /// The tenant-keys JSON did not parse into `{ tenant_id: hash }`.
    #[error("tenant keys JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    /// A configured hash is not a valid Argon2 PHC string (fail closed at startup, not per-request).
    #[error("tenant {tenant:?} has an invalid Argon2 hash")]
    BadHash {
        /// The offending tenant id (never the hash itself).
        tenant: String,
    },
    /// The Argon2 hasher failed while provisioning a new key.
    #[error("argon2 hashing failed")]
    Hash,
}

impl TenantKeys {
    /// Parse from a JSON object `{ "<tenant_id>": "<argon2id-phc-hash>", ... }`.
    ///
    /// Every hash is validated up front so a malformed roster fails at startup rather than
    /// silently rejecting every request at runtime.
    ///
    /// # Errors
    /// [`AuthConfigError::Json`] if the JSON is malformed; [`AuthConfigError::BadHash`] if any
    /// value is not a valid Argon2 PHC hash string.
    pub fn from_json(s: &str) -> Result<Self, AuthConfigError> {
        let hashes: HashMap<String, String> = serde_json::from_str(s)?;
        for (tenant, hash) in &hashes {
            PasswordHash::new(hash).map_err(|_| AuthConfigError::BadHash {
                tenant: tenant.clone(),
            })?;
        }
        // Decoy hash for the unknown-tenant path (equalizes verify time). Computed once at startup.
        let decoy = Self::hash_key("firstpass-unknown-tenant-decoy")?;
        Ok(Self { hashes, decoy })
    }

    /// Whether any tenant keys are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Verify a presented `<tenant_id>.<secret>` key, returning the owning tenant on a match. The
    /// tenant id (before the first `.`, not itself secret) names which hash to check, so this does
    /// **exactly one** Argon2 verify — O(1), not O(tenants) — using the crate's constant-time
    /// [`PasswordVerifier`], never a plain `==`.
    ///
    /// An unknown tenant id (or a malformed key) still performs one Argon2 verify against a decoy
    /// hash, so an invalid key takes the same time whether or not the named tenant exists — this
    /// closes both the CPU-amplification DoS and the tenant-existence timing oracle. Tenant ids must
    /// not contain `.` (they are simple identifiers; the first `.` splits id from secret).
    #[must_use]
    pub fn verify(&self, presented_key: &str) -> Option<TenantId> {
        let argon = Argon2::default();
        let (tenant_id, secret) = presented_key.split_once('.').unwrap_or(("", presented_key));
        if let Some(hash) = self.hashes.get(tenant_id) {
            if let Ok(parsed) = PasswordHash::new(hash)
                && argon.verify_password(secret.as_bytes(), &parsed).is_ok()
            {
                return Some(TenantId(tenant_id.to_owned()));
            }
            return None;
        }
        // Unknown tenant: burn one verify against the decoy to equalize timing, then fail.
        if let Ok(decoy) = PasswordHash::new(&self.decoy) {
            let _ = argon.verify_password(secret.as_bytes(), &decoy);
        }
        None
    }

    /// Hash a fresh secret into a PHC-encoded Argon2id string for provisioning a tenant key.
    /// Operators store the returned hash; the plaintext key is shown to the tenant once and is
    /// never persisted here.
    ///
    /// # Errors
    /// [`AuthConfigError::Hash`] if the Argon2 hasher fails.
    pub fn hash_key(secret: &str) -> Result<String, AuthConfigError> {
        let salt = SaltString::generate(&mut argon2::password_hash::rand_core::OsRng);
        Argon2::default()
            .hash_password(secret.as_bytes(), &salt)
            .map(|h| h.to_string())
            .map_err(|_| AuthConfigError::Hash)
    }
}

/// Axum middleware: resolve the tenant for a request and inject a [`TenantId`] into extensions.
///
/// - `require_auth = false` (default): no auth check; injects the static default tenant so every
///   downstream handler reads a uniform identity. Single-operator behavior is unchanged.
/// - `require_auth = true`: reads the key from `Authorization: Bearer <key>` (or `x-firstpass-key`),
///   verifies it against the configured tenant hashes, and injects the resolved tenant. A missing
///   or invalid key returns `401` with an opaque body — no "unknown tenant" oracle.
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    if !state.config.require_auth {
        req.extensions_mut()
            .insert(TenantId(state.config.tenant_id.clone()));
        return next.run(req).await;
    }
    match extract_key(req.headers()).and_then(|k| state.config.tenant_keys.verify(&k)) {
        Some(tenant) => {
            req.extensions_mut().insert(tenant);
            next.run(req).await
        }
        None => ProxyError::Unauthorized.into_response(),
    }
}

/// Read the tenant API key from `Authorization: Bearer <key>` or the `x-firstpass-key` header.
fn extract_key(headers: &HeaderMap) -> Option<String> {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix(BEARER_PREFIX))
        .map(str::trim)
        .filter(|s| !s.is_empty());
    if let Some(key) = bearer {
        return Some(key.to_owned());
    }
    headers
        .get(KEY_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argon2_hash_round_trips_and_rejects_wrong_key() {
        let hash = TenantKeys::hash_key("s3cret-key").unwrap();
        let json = format!("{{\"acme\": {hash:?}}}");
        let keys = TenantKeys::from_json(&json).unwrap();

        // Correct `<tenant>.<secret>` key resolves to its tenant.
        assert_eq!(
            keys.verify("acme.s3cret-key").map(|t| t.0).as_deref(),
            Some("acme")
        );
        // Right tenant, wrong secret → nothing.
        assert!(keys.verify("acme.wrong-key").is_none());
        // Unknown tenant → nothing (and it still burns a decoy verify — no existence oracle).
        assert!(keys.verify("ghost.s3cret-key").is_none());
        // Malformed (no `.`) and empty keys never match.
        assert!(keys.verify("s3cret-key").is_none());
        assert!(keys.verify("").is_none());
    }

    #[test]
    fn two_tenants_resolve_to_their_own_identity() {
        let a = TenantKeys::hash_key("key-a").unwrap();
        let b = TenantKeys::hash_key("key-b").unwrap();
        let json = format!("{{\"tenant-a\": {a:?}, \"tenant-b\": {b:?}}}");
        let keys = TenantKeys::from_json(&json).unwrap();

        assert_eq!(
            keys.verify("tenant-a.key-a").map(|t| t.0).as_deref(),
            Some("tenant-a")
        );
        assert_eq!(
            keys.verify("tenant-b.key-b").map(|t| t.0).as_deref(),
            Some("tenant-b")
        );
        // Tenant A's id with tenant B's secret is rejected — you can't cross tenants.
        assert!(keys.verify("tenant-a.key-b").is_none());
        // A secret for no configured tenant is rejected.
        assert!(keys.verify("tenant-c.key-c").is_none());
    }

    #[test]
    fn invalid_hash_in_config_is_rejected_at_build_time() {
        let json = r#"{"acme": "not-a-real-argon2-hash"}"#;
        assert!(matches!(
            TenantKeys::from_json(json),
            Err(AuthConfigError::BadHash { tenant }) if tenant == "acme"
        ));
    }

    #[test]
    fn debug_redacts_hashes() {
        let hash = TenantKeys::hash_key("top-secret").unwrap();
        let json = format!("{{\"acme\": {hash:?}}}");
        let keys = TenantKeys::from_json(&json).unwrap();
        let dbg = format!("{keys:?}");
        assert!(dbg.contains("acme"), "tenant id is fine to show");
        assert!(dbg.contains("<redacted>"), "hashes must be redacted");
        assert!(
            !dbg.contains("$argon2"),
            "no hash material may leak in Debug"
        );
    }

    #[test]
    fn extract_key_reads_bearer_and_custom_header() {
        let mut h = HeaderMap::new();
        h.insert(header::AUTHORIZATION, "Bearer abc123".parse().unwrap());
        assert_eq!(extract_key(&h).as_deref(), Some("abc123"));

        let mut h2 = HeaderMap::new();
        h2.insert(KEY_HEADER, "xyz789".parse().unwrap());
        assert_eq!(extract_key(&h2).as_deref(), Some("xyz789"));

        // No key header at all.
        assert!(extract_key(&HeaderMap::new()).is_none());
        // Empty bearer token is not a key.
        let mut h3 = HeaderMap::new();
        h3.insert(header::AUTHORIZATION, "Bearer ".parse().unwrap());
        assert!(extract_key(&h3).is_none());
    }
}
