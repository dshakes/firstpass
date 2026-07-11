//! Per-tenant envelope key custody (ADR 0004 §D5).
//!
//! **EXPERIMENTAL / pre-external-review (ADR 0004 §D7).** This is a self-contained crypto
//! building block, deliberately **not wired to any request flow or stored-key feature yet**.
//! The default hosted plane is pure per-request BYOK — nothing tenant-supplied is stored — so
//! there is no consumer for stored ciphertext today. This module exists so that when a
//! stored-key feature lands, the vetted primitive is already here and already reviewed. It has
//! not yet had the external crypto review ADR 0004 §D7 requires; do not treat it as blessed.
//!
//! # What it does
//!
//! [`KeyCustody`] is the seam ADR 0004 §D5 names: encrypt/decrypt blobs under a per-tenant
//! envelope so a cloud-KMS implementation can slot in later behind the same trait. The local
//! tier — [`LocalKeyCustody`] — is implemented here with real AES-256-GCM (the vetted
//! [`aes_gcm`] / RustCrypto crate), a single 32-byte key-encryption key (KEK) from config, and
//! the tenant id bound as AEAD associated data so a blob sealed for tenant A cannot be opened
//! under tenant B.
//!
//! # Construction (the exact bytes on disk)
//!
//! - **Cipher:** AES-256-GCM (`Aes256Gcm`), 256-bit KEK, 128-bit auth tag.
//! - **Nonce:** fresh random 96 bits per [`encrypt`](KeyCustody::encrypt) call, drawn from the
//!   OS ambient CSPRNG (`getrandom`'s `SysRng`). A nonce is **never** reused with the KEK; a
//!   96-bit random nonce per message is the standard GCM construction.
//! - **AAD:** the tenant id (`tenant.as_bytes()`) is passed as associated data on both encrypt
//!   and decrypt. It is authenticated but not encrypted; a mismatch fails the auth tag.
//! - **Blob layout:** `nonce (12 bytes) ‖ ciphertext ‖ tag (16 bytes)`. Self-describing: the
//!   nonce is a fixed-width prefix, and `ciphertext ‖ tag` is exactly what `aes-gcm` returns.
//!   The blob is safe to store at rest; it reveals only the plaintext length.
//!
//! # KEK loading
//!
//! [`LocalKeyCustody::from_env`] reads, in order of precedence:
//! - `FIRSTPASS_KEK` — the KEK **hex-encoded** (64 lowercase/uppercase hex chars = 32 bytes).
//!   Surrounding whitespace is trimmed. Hex (not base64) is used for parity with the rest of
//!   the crate, which already depends on `hex` (e.g. audit-trace digests).
//! - `FIRSTPASS_KEK_FILE` — path to a file whose contents are exactly 32 **raw** bytes.
//!
//! A missing, short, over-long, or malformed KEK is a hard error ([`KeyCustodyError`]); the
//! loader is **fail-closed** and never falls back to a zero/default key.

// `Nonce<A>` here is the AeadCore-parameterized alias (`aes_gcm::aead::Nonce`), i.e. a nonce
// sized for the given AEAD — NOT `aes_gcm::Nonce<NonceSize>`, which is parameterized by the
// nonce width directly.
use aes_gcm::aead::{Aead, Generate, Nonce, Payload};
use aes_gcm::{Aes256Gcm, KeyInit};

/// Size of the KEK in bytes (AES-256 → 32 bytes).
const KEK_LEN: usize = 32;
/// AES-GCM nonce size in bytes (96 bits — the standard GCM nonce width).
const NONCE_LEN: usize = 12;

/// Encrypt/decrypt opaque blobs under a per-tenant envelope.
///
/// This is the trust seam ADR 0004 §D5 names. The local tier ([`LocalKeyCustody`]) is
/// implemented here; a cloud-KMS tier can implement the same trait later without touching
/// callers. `tenant` is authenticated (bound as AEAD associated data), so a blob is only
/// decryptable under the same tenant it was encrypted for.
pub trait KeyCustody {
    /// Seal `plaintext` for `tenant`, returning an opaque `nonce ‖ ciphertext ‖ tag` blob that
    /// is safe to store at rest.
    ///
    /// # Errors
    /// [`KeyCustodyError::Encrypt`] if the RNG or the AEAD layer fails. The error is opaque and
    /// carries no key material.
    fn encrypt(&self, tenant: &str, plaintext: &[u8]) -> Result<Vec<u8>, KeyCustodyError>;

    /// Open a blob previously produced by [`encrypt`](KeyCustody::encrypt) for the **same**
    /// `tenant`.
    ///
    /// # Errors
    /// [`KeyCustodyError::Decrypt`] on any failure — tamper, wrong KEK, wrong tenant, or a
    /// truncated blob. The failure mode is **indistinguishable** on purpose (no oracle): all of
    /// them surface as the same opaque error, and never a partial/garbage plaintext.
    fn decrypt(&self, tenant: &str, blob: &[u8]) -> Result<Vec<u8>, KeyCustodyError>;
}

/// Errors from key custody. No variant ever carries key material or plaintext.
#[derive(Debug, thiserror::Error)]
pub enum KeyCustodyError {
    /// No KEK is configured (`FIRSTPASS_KEK` / `FIRSTPASS_KEK_FILE` both unset).
    #[error("no KEK configured: set FIRSTPASS_KEK (hex-encoded 32 bytes) or FIRSTPASS_KEK_FILE")]
    MissingKek,

    /// A KEK was supplied but is unusable (wrong length, bad hex, unreadable file). The reason
    /// describes the shape of the failure only — never the key bytes.
    #[error("invalid KEK: {0}")]
    BadKek(String),

    /// Encryption failed (RNG or AEAD layer). Opaque by design.
    #[error("encryption failed")]
    Encrypt,

    /// Decryption/authentication failed — tamper, wrong KEK, wrong tenant, or truncated blob.
    /// Opaque by design: callers cannot tell which, so the type is not a decryption oracle.
    #[error("decryption failed")]
    Decrypt,
}

/// Local-tier [`KeyCustody`]: real AES-256-GCM under a single process-held KEK.
///
/// The KEK is baked into an [`Aes256Gcm`] key schedule at construction and the raw bytes are
/// not retained. [`std::fmt::Debug`] is hand-written to redact it regardless.
pub struct LocalKeyCustody {
    cipher: Aes256Gcm,
}

impl LocalKeyCustody {
    /// Build a custody instance from a raw 32-byte KEK.
    ///
    /// Infallible: a `[u8; KEK_LEN]` is always a valid AES-256 key. The key schedule is derived
    /// once here, so per-message [`encrypt`](KeyCustody::encrypt)/[`decrypt`](KeyCustody::decrypt)
    /// calls do not re-expand it.
    #[must_use]
    pub fn new(kek: [u8; KEK_LEN]) -> Self {
        // `kek.into()` produces the `Key<Aes256Gcm>` (a 32-byte `Array`); a `[u8; 32]` is always
        // the right length, so `new` (not `new_from_slice`) is the correct infallible path.
        Self {
            cipher: Aes256Gcm::new(&kek.into()),
        }
    }

    /// Load the KEK from the environment. Precedence: `FIRSTPASS_KEK` (hex), then
    /// `FIRSTPASS_KEK_FILE` (32 raw bytes).
    ///
    /// # Errors
    /// - [`KeyCustodyError::MissingKek`] if neither is set.
    /// - [`KeyCustodyError::BadKek`] if the value is malformed, the wrong length, or the file is
    ///   unreadable.
    pub fn from_env() -> Result<Self, KeyCustodyError> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// [`from_env`](Self::from_env) with an injectable variable lookup — the testable seam
    /// (mirrors [`crate::config::ProxyConfig::from_lookup`]), so KEK-loading tests need no
    /// process-global env mutation.
    ///
    /// # Errors
    /// Same as [`from_env`](Self::from_env).
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, KeyCustodyError> {
        if let Some(hex_kek) = lookup("FIRSTPASS_KEK") {
            let bytes = hex::decode(hex_kek.trim()).map_err(|e| {
                KeyCustodyError::BadKek(format!("FIRSTPASS_KEK is not valid hex: {e}"))
            })?;
            return Self::from_bytes(&bytes);
        }
        if let Some(path) = lookup("FIRSTPASS_KEK_FILE") {
            let bytes = std::fs::read(&path).map_err(|e| {
                KeyCustodyError::BadKek(format!("cannot read FIRSTPASS_KEK_FILE ({path}): {e}"))
            })?;
            return Self::from_bytes(&bytes);
        }
        Err(KeyCustodyError::MissingKek)
    }

    /// Validate an exact-length KEK and build the cipher. Rejects any length other than
    /// [`KEK_LEN`]; reports only the length, never the bytes.
    fn from_bytes(bytes: &[u8]) -> Result<Self, KeyCustodyError> {
        let kek: [u8; KEK_LEN] = bytes.try_into().map_err(|_| {
            KeyCustodyError::BadKek(format!(
                "KEK must be exactly {KEK_LEN} bytes, got {}",
                bytes.len()
            ))
        })?;
        Ok(Self::new(kek))
    }
}

impl std::fmt::Debug for LocalKeyCustody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalKeyCustody")
            .field("cipher", &"<redacted AES-256-GCM key>")
            .finish()
    }
}

impl KeyCustody for LocalKeyCustody {
    fn encrypt(&self, tenant: &str, plaintext: &[u8]) -> Result<Vec<u8>, KeyCustodyError> {
        // Fresh random 96-bit nonce per call from the OS CSPRNG. Fail closed if the RNG faults:
        // a non-random nonce could collide and break GCM's security, so we never proceed.
        let nonce = Nonce::<Aes256Gcm>::try_generate().map_err(|_| KeyCustodyError::Encrypt)?;

        // Bind the tenant as AAD: authenticated, not encrypted. A wrong tenant on decrypt fails
        // the auth tag.
        let ciphertext = self
            .cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: plaintext,
                    aad: tenant.as_bytes(),
                },
            )
            .map_err(|_| KeyCustodyError::Encrypt)?;

        let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        blob.extend_from_slice(&nonce);
        blob.extend_from_slice(&ciphertext);
        Ok(blob)
    }

    fn decrypt(&self, tenant: &str, blob: &[u8]) -> Result<Vec<u8>, KeyCustodyError> {
        // A well-formed blob is nonce ‖ (ciphertext ‖ 16-byte tag). Anything shorter than the
        // nonce plus a tag cannot be authentic — treat as an ordinary decryption failure (no
        // distinct "too short" oracle).
        if blob.len() < NONCE_LEN {
            return Err(KeyCustodyError::Decrypt);
        }
        let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
        let nonce =
            <&Nonce<Aes256Gcm>>::try_from(nonce_bytes).map_err(|_| KeyCustodyError::Decrypt)?;

        self.cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad: tenant.as_bytes(),
                },
            )
            .map_err(|_| KeyCustodyError::Decrypt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEK1: [u8; KEK_LEN] = [0x11; KEK_LEN];
    const KEK2: [u8; KEK_LEN] = [0x22; KEK_LEN];

    #[test]
    fn round_trip_various_messages() {
        let custody = LocalKeyCustody::new(KEK1);
        let messages: &[&[u8]] = &[
            b"",                     // empty
            b"x",                    // single byte
            b"hello tenant secrets", // typical
            &[0xABu8; 64 * 1024],    // large (64 KiB)
        ];
        for msg in messages {
            let blob = custody.encrypt("acme", msg).expect("encrypt");
            let out = custody.decrypt("acme", &blob).expect("decrypt");
            assert_eq!(&out, msg, "round-trip must recover the exact plaintext");
        }
    }

    #[test]
    fn blob_layout_is_nonce_then_ciphertext_and_tag() {
        let custody = LocalKeyCustody::new(KEK1);
        let msg = b"layout check";
        let blob = custody.encrypt("acme", msg).expect("encrypt");
        // nonce (12) + ciphertext (== plaintext len for a stream cipher) + tag (16).
        assert_eq!(blob.len(), NONCE_LEN + msg.len() + 16);
    }

    #[test]
    fn tenant_binding_blocks_cross_tenant_decrypt() {
        // The per-tenant isolation guarantee: a blob sealed for tenant-a must NOT open under
        // tenant-b, because the tenant is bound as AEAD associated data.
        let custody = LocalKeyCustody::new(KEK1);
        let blob = custody.encrypt("tenant-a", b"a's data").expect("encrypt");
        let result = custody.decrypt("tenant-b", &blob);
        assert!(
            matches!(result, Err(KeyCustodyError::Decrypt)),
            "cross-tenant decrypt must fail the auth tag"
        );
    }

    #[test]
    fn tamper_of_any_region_is_detected() {
        let custody = LocalKeyCustody::new(KEK1);
        let msg = b"integrity matters";
        let blob = custody.encrypt("acme", msg).expect("encrypt");
        // Flip one byte in each region: nonce, ciphertext, and tag.
        for idx in [0usize, NONCE_LEN + 1, blob.len() - 1] {
            let mut tampered = blob.clone();
            tampered[idx] ^= 0x01;
            let result = custody.decrypt("acme", &tampered);
            assert!(
                matches!(result, Err(KeyCustodyError::Decrypt)),
                "flipping byte {idx} must be detected"
            );
        }
    }

    #[test]
    fn wrong_kek_cannot_decrypt() {
        let sealer = LocalKeyCustody::new(KEK1);
        let opener = LocalKeyCustody::new(KEK2);
        let blob = sealer.encrypt("acme", b"secret").expect("encrypt");
        let result = opener.decrypt("acme", &blob);
        assert!(matches!(result, Err(KeyCustodyError::Decrypt)));
    }

    #[test]
    fn nonce_is_fresh_per_call() {
        // Two encryptions of the same plaintext under the same key/tenant must differ, proving a
        // fresh nonce each call (nonce reuse would be catastrophic for GCM).
        let custody = LocalKeyCustody::new(KEK1);
        let a = custody.encrypt("acme", b"same message").expect("encrypt");
        let b = custody.encrypt("acme", b"same message").expect("encrypt");
        assert_ne!(a, b, "distinct nonces must yield distinct blobs");
        assert_ne!(&a[..NONCE_LEN], &b[..NONCE_LEN], "nonces must differ");
    }

    #[test]
    fn from_env_hex_key_loads_and_works() {
        let hex_kek = hex::encode(KEK1);
        let custody = LocalKeyCustody::from_lookup(|k| match k {
            "FIRSTPASS_KEK" => Some(hex_kek.clone()),
            _ => None,
        })
        .expect("valid hex KEK must load");
        let blob = custody.encrypt("acme", b"env-loaded").expect("encrypt");
        assert_eq!(
            custody.decrypt("acme", &blob).expect("decrypt"),
            b"env-loaded"
        );
    }

    #[test]
    fn from_env_trims_surrounding_whitespace() {
        let hex_kek = format!("  {}\n", hex::encode(KEK1));
        let custody =
            LocalKeyCustody::from_lookup(|k| (k == "FIRSTPASS_KEK").then(|| hex_kek.clone()));
        assert!(custody.is_ok(), "trimmed hex KEK must load");
    }

    #[test]
    fn from_env_missing_is_missing_kek() {
        let result = LocalKeyCustody::from_lookup(|_| None);
        assert!(matches!(result, Err(KeyCustodyError::MissingKek)));
    }

    #[test]
    fn from_env_short_key_is_bad_kek() {
        let short = hex::encode([0x11u8; 16]); // 16 bytes, not 32
        let result =
            LocalKeyCustody::from_lookup(|k| (k == "FIRSTPASS_KEK").then(|| short.clone()));
        assert!(matches!(result, Err(KeyCustodyError::BadKek(_))));
    }

    #[test]
    fn from_env_malformed_hex_is_bad_kek() {
        let result =
            LocalKeyCustody::from_lookup(|k| (k == "FIRSTPASS_KEK").then(|| "nothex!!".to_owned()));
        assert!(matches!(result, Err(KeyCustodyError::BadKek(_))));
    }

    #[test]
    fn from_env_file_with_32_raw_bytes_loads() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("firstpass-kek-{}.bin", std::process::id()));
        std::fs::write(&path, KEK1).expect("write temp KEK file");
        let path_str = path.to_string_lossy().into_owned();
        let custody =
            LocalKeyCustody::from_lookup(|k| (k == "FIRSTPASS_KEK_FILE").then(|| path_str.clone()));
        let _ = std::fs::remove_file(&path);
        let custody = custody.expect("32-byte key file must load");
        let blob = custody.encrypt("acme", b"file-loaded").expect("encrypt");
        assert_eq!(
            custody.decrypt("acme", &blob).expect("decrypt"),
            b"file-loaded"
        );
    }

    #[test]
    fn from_env_file_wrong_length_is_bad_kek() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("firstpass-kek-short-{}.bin", std::process::id()));
        std::fs::write(&path, [0x11u8; 10]).expect("write temp KEK file");
        let path_str = path.to_string_lossy().into_owned();
        let result =
            LocalKeyCustody::from_lookup(|k| (k == "FIRSTPASS_KEK_FILE").then(|| path_str.clone()));
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(KeyCustodyError::BadKek(_))));
    }

    #[test]
    fn from_env_unreadable_file_is_bad_kek() {
        let result = LocalKeyCustody::from_lookup(|k| {
            (k == "FIRSTPASS_KEK_FILE").then(|| "/no/such/firstpass/kek".to_owned())
        });
        assert!(matches!(result, Err(KeyCustodyError::BadKek(_))));
    }

    #[test]
    fn debug_redacts_the_kek() {
        let custody = LocalKeyCustody::new(KEK1);
        let rendered = format!("{custody:?}");
        assert!(
            rendered.contains("redacted"),
            "Debug must mark the key redacted"
        );
        // The raw KEK (all 0x11) must never appear, in any common encoding.
        assert!(
            !rendered.contains("1111111111"),
            "no raw key bytes in Debug"
        );
        assert!(
            !rendered.contains(&hex::encode(KEK1)),
            "no hex key in Debug"
        );
    }
}
