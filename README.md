# rpx

Multi-model inference orchestration platform. Deploy ML models to serverless GPUs and expose them as a single OpenAI-compatible API.

```
rpx deploy Qwen/Qwen2.5-7B-Instruct    # one model, one command
rpx serve -c rpx.yaml                   # multi-model gateway
```

## What rpx does

rpx is the missing orchestration layer between API gateways (LiteLLM, Portkey) and inference engines (vLLM, SGLang). It handles:

- **GPU provisioning** across cloud providers (RunPod first, extensible)
- **Model lifecycle** — hot (always on), warm (scale to zero), cold (deploy on demand)
- **Auto GPU selection** — model params → VRAM estimate → cheapest GPU
- **OpenAI-compatible API** — drop-in replacement for any OpenAI SDK
- **Multi-model routing** — one gateway, many models
- **Auth + rate limiting + spend tracking** per API key

## Quick start

```bash
# Install (from source)
cargo install --path crates/rpx-cli

# Store your RunPod API key
rpx login

# Deploy a single model
rpx deploy Qwen/Qwen2.5-7B-Instruct

# Or run a multi-model gateway
rpx serve -c rpx.yaml
```

## Single-model deploy

```bash
# Auto-selects GPU, backend, and scaling
rpx deploy meta-llama/Llama-3.1-8B-Instruct

# With options
rpx deploy meta-llama/Llama-3.1-70B-Instruct \
  --backend vllm \
  --gpu a100-80gb \
  --min-workers 1 \
  --max-workers 5

# Dry run — shows plan without creating anything
rpx deploy Qwen/Qwen2.5-1.5B-Instruct --dry-run
```

## Multi-model gateway

Write an `rpx.yaml`:

```yaml
gateway:
  port: 4000

provider:
  name: runpod

models:
  - id: meta-llama/Llama-3.1-8B-Instruct
    alias: llama-8b
    tier: hot           # always running
    scaling:
      min_workers: 1
      max_workers: 5

  - id: Qwen/Qwen2.5-72B-Instruct
    alias: qwen-72b
    tier: warm          # scale to zero when idle
    gpu: a100-80gb
    scaling:
      min_workers: 0
      max_workers: 3
      idle_timeout: 600

  - id: mistralai/Mistral-7B-Instruct-v0.3
    alias: mistral-7b
    tier: cold           # deploy on first request

api_keys:
  - key: sk-my-app
    name: my-app
    rate_limit_rpm: 120
    budget_usd: 100.0
```

Start the gateway:

```bash
rpx serve
```

Use with any OpenAI client:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-my-app" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Model tiers

| Tier | Behavior | Cost |
|------|----------|------|
| **hot** | Always running, min_workers ≥ 1. Deployed at gateway startup. | Highest (always paying) |
| **warm** | Endpoint exists but can scale to 0 workers. RunPod handles scaling. Evicted after `eviction_timeout`. | Medium (pay when used) |
| **cold** | No endpoint until first request. Request held up to 120s while deploying. | Lowest (pay nothing until used) |

## Management endpoints

```bash
# Fleet status
curl http://localhost:4000/v1/rpx/fleet

# Spend report
curl http://localhost:4000/v1/rpx/spend

# List available models
curl http://localhost:4000/v1/models

# Health check
curl http://localhost:4000/health
```

## CLI commands

| Command | Description |
|---------|-------------|
| `rpx login [provider]` | Store provider API key |
| `rpx deploy <model>` | Deploy a single model |
| `rpx serve [-c config]` | Run multi-model gateway |
| `rpx status <endpoint>` | Show endpoint status |
| `rpx list` | List all endpoints |
| `rpx destroy <endpoint>` | Delete an endpoint |
| `rpx proxy <endpoint>` | Local OpenAI proxy for a single endpoint |

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full design.

```
Client (OpenAI SDK)
    │
    ▼
┌─ rpx serve ────────────────────────┐
│  Gateway (auth, route, stream)     │
│  Model Manager (deploy, lifecycle) │
│  Autoscaler (evict, recover)       │
└────────────┬───────────────────────┘
             │
             ▼
    RunPod Serverless Endpoints
    (one per model, vLLM/rvLLM)
```

## Providers

Currently supported:
- **RunPod** — serverless GPU endpoints via REST API

The `Provider` trait is designed for extension. Adding a new provider means implementing ~10 async methods.

## Backends

| Backend | Status | Image |
|---------|--------|-------|
| vLLM | Supported (default) | `runpod/worker-v1-vllm:*` |
| rvLLM | Supported | `pyshx/rvllm-runpod:latest` |
| TGI | Planned | — |
| llama.cpp | Planned | — |

## License

MIT
