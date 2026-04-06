# Build stage
FROM rust:1.83-slim-bookworm AS builder

WORKDIR /build

# Install dependencies
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Cache dependencies: copy manifests first
COPY Cargo.toml Cargo.lock ./
COPY crates/rpx-core/Cargo.toml crates/rpx-core/Cargo.toml
COPY crates/rpx-cli/Cargo.toml crates/rpx-cli/Cargo.toml

# Create dummy sources for dependency caching
RUN mkdir -p crates/rpx-core/src crates/rpx-cli/src && \
    echo "fn main() {}" > crates/rpx-cli/src/main.rs && \
    echo "" > crates/rpx-core/src/lib.rs && \
    cargo build --release --bin rpx 2>/dev/null || true && \
    rm -rf crates/rpx-core/src crates/rpx-cli/src

# Copy real source and build
COPY crates/ crates/
COPY catalog/ catalog/
RUN cargo build --release --bin rpx

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

RUN useradd -r -m -s /bin/false rpx
RUN mkdir -p /home/rpx/.rpx
COPY --from=builder /build/target/release/rpx /usr/local/bin/rpx
COPY --from=builder /build/catalog/ /etc/rpx/catalog/
COPY deploy/cloud-run/rpx.yaml /etc/rpx/rpx.yaml

USER rpx
EXPOSE 8080

ENTRYPOINT ["rpx", "serve", "-c", "/etc/rpx/rpx.yaml"]
