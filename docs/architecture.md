# Architecture

## Overview

rpx is a Rust binary that manages serverless GPU endpoints across cloud providers and exposes them as a unified OpenAI-compatible API.

```
                    Clients (OpenAI SDK, curl)
                              │
                              ▼
                 ┌────────────────────────┐
                 │       rpx serve        │
                 │                        │
                 │  ┌────────────────┐    │
                 │  │    Gateway     │    │
                 │  │  (axum HTTP)   │    │
                 │  │                │    │
                 │  │ • /v1/chat/... │    │
                 │  │ • /v1/models   │    │
                 │  │ • /v1/rpx/...  │    │
                 │  └───────┬────────┘    │
                 │          │             │
                 │  ┌───────┴────────┐    │
                 │  │   Auth Layer   │    │
                 │  │  API keys      │    │
                 │  │  Rate limiting  │    │
                 │  │  Budget check   │    │
                 │  └───────┬────────┘    │
                 │          │             │
                 │  ┌───────┴────────┐    │
                 │  │  Model Router  │    │
                 │  │  alias → state │    │
                 │  │  → endpoint    │    │
                 │  └───────┬────────┘    │
                 │          │             │
                 │          ├── Hot/Warm: invoke directly
                 │          ├── Cold: deploy → queue → invoke
                 │          └── Error: 503
                 │                        │
                 │  ┌────────────────┐    │
                 │  │ Model Manager  │    │
                 │  │  deploy/       │    │
                 │  │  undeploy      │    │
                 │  └───────┬────────┘    │
                 │          │             │
                 │  ┌───────┴────────┐    │
                 │  │  Autoscaler    │    │
                 │  │  (background)  │    │
                 │  │  eviction +    │    │
                 │  │  error retry   │    │
                 │  └────────────────┘    │
                 └────────────┬───────────┘
                              │
                              ▼
                    Provider Layer (trait)
                              │
                 ┌────────────┴───────────┐
                 │    RunPod REST API      │
                 │  • POST /v1/templates   │
                 │  • POST /v1/endpoints   │
                 │  • POST /v2/{id}/runsync│
                 └────────────┬───────────┘
                              │
                              ▼
                 RunPod Serverless Endpoints
                 (one vLLM/rvLLM per model)
```

## Crate structure

```
rpx/
├── crates/
│   ├── rpx-core/              # Library — all business logic
│   │   └── src/
│   │       ├── provider/      # Provider trait + RunPod impl
│   │       ├── backend/       # Backend trait + vLLM/rvLLM
│   │       ├── catalog/       # GPU pricing + selection
│   │       ├── model/         # HuggingFace metadata + VRAM sizing
│   │       ├── config.rs      # RpxConfig, credentials, endpoint store
│   │       ├── deploy.rs      # Plan resolution + execution
│   │       ├── fleet/         # FleetConfig, ModelState FSM, persistence
│   │       ├── gateway/       # Multi-model axum gateway
│   │       ├── orchestrator/  # Ties everything together
│   │       └── proxy/         # SSE streaming + model name rewriting
│   └── rpx-cli/               # Binary — thin CLI layer
│       └── src/
│           ├── commands/       # login, deploy, serve, status, list, destroy, proxy
│           ├── ui.rs           # lipgloss styled output
│           └── main.rs
├── catalog/
│   └── gpus.toml              # Static GPU pricing (embedded at compile time)
└── docs/
```

## Key abstractions

### Provider trait

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    async fn validate_auth(&self) -> Result<()>;
    async fn create_endpoint(&self, config: &EndpointConfig) -> Result<Endpoint>;
    async fn get_endpoint(&self, id: &str) -> Result<Endpoint>;
    async fn delete_endpoint(&self, id: &str) -> Result<()>;
    async fn list_endpoints(&self) -> Result<Vec<Endpoint>>;
    async fn invoke(&self, endpoint: &Endpoint, req: InvocationRequest) -> Result<InvocationResponse>;
    // ...
}
```

Adding a new provider (Vast.ai, Beam, Lambda Labs) means implementing this trait.

### Backend trait

```rust
pub trait Backend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn default_image(&self) -> &str;
    fn env_vars(&self, model_id: &str, config: &ModelConfig) -> HashMap<String, String>;
    fn estimate_vram_gb(&self, model_params_billions: f64, dtype: &str) -> f64;
    fn openai_native(&self) -> bool;
    fn default_port(&self) -> u16;
}
```

Adding a new inference engine (TGI, llama.cpp, SGLang) means implementing this trait.

### Model lifecycle

```
COLD ──request──▶ DEPLOYING ──success──▶ WARM ◀──▶ HOT
  ▲                   │                    │
  │                   │ fail               │ eviction_timeout
  │                   ▼                    │
  │                ERROR ──retry──▶────────┘
  └────────────────evict───────────────────┘
```

- **Cold**: No RunPod endpoint exists. First request triggers deploy.
- **Deploying**: Template + endpoint being created. Requests queued (120s timeout).
- **Warm**: Endpoint exists. RunPod manages worker scaling (0 to max). Evicted after idle timeout.
- **Hot**: Always ≥ 1 worker. Never evicted.
- **Error**: Deploy failed. Auto-retry with exponential backoff.

rpx manages endpoint lifecycle. RunPod manages worker scaling within an endpoint.

## Deploy flow

```
rpx deploy <model>
    │
    ├── 1. Fetch model metadata from HuggingFace API
    │      (param count, architecture, gating)
    │
    ├── 2. Select backend (vLLM default, or user override)
    │
    ├── 3. Estimate VRAM
    │      backend.estimate_vram_gb(params, dtype)
    │
    ├── 4. Select GPU from catalog
    │      cheapest GPU with vram >= estimated, on configured provider
    │
    ├── 5. Create RunPod serverless template
    │      POST /v1/templates (image + env vars + isServerless: true)
    │
    ├── 6. Create RunPod endpoint
    │      POST /v1/endpoints (templateId + gpuTypeIds + scaling)
    │
    ├── 7. Poll until ready
    │      GET /v1/endpoints/{id} every 1s
    │
    └── 8. Save to ~/.rpx/endpoints.json
```

## Request flow (gateway)

### Hot/Warm model (steady state)
```
POST /v1/chat/completions {"model": "llama-8b"}
  → Auth middleware: validate key, rate limit, budget
  → Router: resolve "llama-8b" → endpoint_id
  → provider.invoke(endpoint, request)
  → SSE stream back to client
  → Record spend
Overhead: ~1-3ms
```

### Cold model (deploy on demand)
```
POST /v1/chat/completions {"model": "mistral-7b"}
  → Auth middleware
  → Router: state == Cold
  → Spawn background: model_manager.deploy()
  → Enqueue request (120s timeout)
  → Deploy: create template → create endpoint → poll ready (~30-90s)
  → Drain queue: forward all waiting requests
  → Respond to client
```

## RunPod API mapping

rpx uses the RunPod REST API (`rest.runpod.io/v1`):

| rpx operation | RunPod API |
|---|---|
| Create template | `POST /v1/templates` (imageName, env, isServerless) |
| Create endpoint | `POST /v1/endpoints` (templateId, gpuTypeIds, scaling) |
| Get endpoint | `GET /v1/endpoints/{id}` |
| Delete endpoint | `DELETE /v1/endpoints/{id}` |
| List endpoints | `GET /v1/endpoints` |
| Invoke (sync) | `POST https://api.runpod.ai/v2/{id}/runsync` |
| Invoke (stream) | `POST .../run` + `GET .../stream/{job_id}` |

## GPU auto-selection

The catalog (`catalog/gpus.toml`) maps GPU types to VRAM and pricing:

```
Model params → Backend VRAM estimate → Filter catalog → Sort by price → Pick cheapest
```

Example:
- 7B model at fp16 → vLLM estimates 18.2 GB (7 * 2 * 1.3 overhead)
- Catalog filters: RTX 4000 Ada (20 GB, $0.40/hr), L4 (24 GB, $0.68/hr), ...
- Picks RTX 4000 Ada (cheapest with >= 18.2 GB)

## State persistence

- `~/.rpx/credentials.toml` — provider API keys
- `~/.rpx/endpoints.json` — deployed single-model endpoints
- `~/.rpx/fleet_state.json` — multi-model fleet state (model states, endpoint IDs)

Fleet state is saved every 60s and on graceful shutdown (Ctrl+C).

On startup, `rpx serve` reconciles persisted state with actual RunPod endpoints — if an endpoint was deleted externally, the model reverts to Cold.

## Auth model

Configured per fleet in `rpx.yaml`:

```yaml
api_keys:
  - key: sk-my-app
    name: my-app
    rate_limit_rpm: 120     # token-bucket rate limiter
    budget_usd: 100.0       # reject when exceeded (402)
    allowed_models:          # optional model restriction
      - llama-8b
```

No keys configured = open access (no auth required).

Management endpoints (`/v1/rpx/*`, `/health`) skip auth.
