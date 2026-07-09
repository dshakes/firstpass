# Multi-stage build → a small, dependency-free runtime image.
# Build:  docker build -t firstpass .
# Run:    docker run -p 8080:8080 -e FIRSTPASS_BIND=0.0.0.0:8080 firstpass

FROM rust:1.93-slim AS builder
WORKDIR /build
# Cache dependencies: copy manifests first, then sources.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --release --bin firstpass-proxy

# Distroless runtime: no shell, no package manager, minimal attack surface.
FROM gcr.io/distroless/cc-debian12 AS runtime
COPY --from=builder /build/target/release/firstpass-proxy /usr/local/bin/firstpass-proxy
# Bind on all interfaces inside the container by default.
ENV FIRSTPASS_BIND=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/firstpass-proxy"]
