//! `cargo run -p firstpass-proxy --example demo`
//!
//! Kept as a thin shim so the from-source invocation in older docs keeps working. The demo itself
//! lives in `firstpass_proxy::demo` so it ships inside the `firstpass` binary too — which is what
//! makes `uvx firstpass demo` (no Rust toolchain) possible.

#[tokio::main]
async fn main() {
    if let Err(e) = firstpass_proxy::demo::run().await {
        eprintln!("demo failed: {e}");
        std::process::exit(1);
    }
}
