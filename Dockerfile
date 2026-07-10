# Multi-stage build → a small, dependency-free runtime image.
# Build:  docker build -t firstpass .
# Run:    docker run -p 8080:8080 -e FIRSTPASS_BIND=0.0.0.0:8080 firstpass

FROM rust:1.93-slim AS builder
WORKDIR /build
# Cache dependencies: copy manifests first, then sources.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
# Both binaries: the server (`firstpass-proxy`, the default entrypoint) and the unified CLI
# (`firstpass up | doctor | trace | mcp`).
RUN cargo build --release --bin firstpass-proxy --bin firstpass

# Distroless runtime: no shell, no package manager, minimal attack surface.
FROM gcr.io/distroless/cc-debian12 AS runtime
COPY --from=builder /build/target/release/firstpass-proxy /usr/local/bin/firstpass-proxy
COPY --from=builder /build/target/release/firstpass /usr/local/bin/firstpass
# Bind on all interfaces inside the container by default.
ENV FIRSTPASS_BIND=0.0.0.0:8080
EXPOSE 8080
# Server by default; use `--entrypoint firstpass` for doctor / trace / mcp.
ENTRYPOINT ["/usr/local/bin/firstpass-proxy"]
