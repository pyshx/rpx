# Build stage
FROM rust:1.83-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
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
