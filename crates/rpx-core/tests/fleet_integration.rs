use std::sync::Arc;

use rpx_core::catalog::GpuCatalog;
use rpx_core::deploy;
use rpx_core::fleet::{FleetConfig, ModelTier};
use rpx_core::fleet::state::FleetState;
use rpx_core::model;
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;

/// Parse a real fleet config, resolve deploy plans for every model,
/// verify GPU selection and VRAM estimation against live HuggingFace API.
#[tokio::test]
async fn fleet_config_resolves_plans_for_all_models() {
    let yaml = r#"
gateway:
  port: 4000

provider:
  name: runpod

models:
  - id: Qwen/Qwen2.5-1.5B-Instruct
    alias: qwen-1.5b
    tier: hot
    backend: vllm
    gpu: auto
    scaling:
      min_workers: 1
      max_workers: 3

  - id: Qwen/Qwen2.5-7B-Instruct
    alias: qwen-7b
    tier: warm
    backend: vllm
    gpu: auto
    scaling:
      min_workers: 0
      max_workers: 5
      idle_timeout: 600
      eviction_timeout: 3600

  - id: mistralai/Mistral-7B-Instruct-v0.3
    alias: mistral-7b
    tier: cold

api_keys:
  - key: sk-test
    name: test
    rate_limit_rpm: 60
"#;

    // 1. Parse fleet config
    let config = FleetConfig::from_yaml(yaml).expect("fleet config should parse");
    assert_eq!(config.models.len(), 3);
    assert_eq!(config.models[0].tier, ModelTier::Hot);
    assert_eq!(config.models[1].tier, ModelTier::Warm);
    assert_eq!(config.models[2].tier, ModelTier::Cold);

    // 2. Initialize fleet state
    let model_names: Vec<String> = config.models.iter().map(|m| m.display_name()).collect();
    let state = FleetState::init_from_config(&model_names);
    for name in &model_names {
        assert!(state.get(name).unwrap().is_cold(), "{name} should start cold");
    }

    // 3. Alias map works
    let alias_map = config.alias_map();
    assert_eq!(alias_map["qwen-1.5b"], 0);
    assert_eq!(alias_map["qwen-7b"], 1);
    assert_eq!(alias_map["mistral-7b"], 2);

    // 4. Resolve deploy plan for each model using REAL HuggingFace API
    let catalog = GpuCatalog::load_embedded().unwrap();
    let provider = RunPodProvider::new(String::new()); // no real key needed for plan resolution
    let http_client = reqwest::Client::new();

    let mut results = Vec::new();

    for entry in &config.models {
        let rpx_config = entry.to_rpx_config();

        // Fetch real metadata from HuggingFace
        let metadata = model::fetch_model_metadata(&http_client, &entry.id, None)
            .await
            .unwrap_or_else(|e| panic!("failed to fetch metadata for {}: {e}", entry.id));

        // Resolve plan
        let plan = deploy::resolve_plan(&rpx_config, &metadata, &catalog, &provider)
            .unwrap_or_else(|e| panic!("failed to resolve plan for {}: {e}", entry.id));

        // Verify plan sanity
        assert!(
            plan.gpu.vram_gb as f64 >= plan.estimated_vram_gb,
            "{}: GPU {} ({} GB) < estimated VRAM {:.1} GB",
            entry.id,
            plan.gpu.gpu_name,
            plan.gpu.vram_gb,
            plan.estimated_vram_gb,
        );
        assert_eq!(plan.gpu.provider, ProviderKind::RunPod);
        assert!(plan.gpu.price_per_hour > 0.0);
        assert!(!plan.env_vars.is_empty());

        results.push((entry.display_name(), plan));
    }

    // 5. Write results to file for inspection
    let report: String = results
        .iter()
        .map(|(name, plan)| {
            format!(
                "{name}:\n  model: {}\n  backend: {}\n  vram: {:.1} GB\n  gpu: {} ({} GB)\n  cost: ${:.4}/hr\n  env_vars: {:?}\n",
                plan.model_id,
                plan.backend,
                plan.estimated_vram_gb,
                plan.gpu.gpu_name,
                plan.gpu.vram_gb,
                plan.gpu.price_per_hour,
                plan.env_vars.keys().collect::<Vec<_>>(),
            )
        })
        .collect();
    std::fs::write("/tmp/rpx_fleet_test.txt", &report).ok();
}

/// Verify the state machine works end-to-end with a realistic flow.
#[test]
fn fleet_state_full_lifecycle() {
    use rpx_core::fleet::state::ModelState;
    use rpx_core::provider::types::{Endpoint, EndpointStatus};

    let names = vec![
        "hot-model".to_string(),
        "warm-model".to_string(),
        "cold-model".to_string(),
    ];
    let mut state = FleetState::init_from_config(&names);

    // All start cold
    assert!(state.get("hot-model").unwrap().is_cold());
    assert!(state.get("warm-model").unwrap().is_cold());
    assert!(state.get("cold-model").unwrap().is_cold());

    // Deploy hot model
    state.begin_deploy("hot-model").unwrap();
    assert!(state.get("hot-model").unwrap().is_deploying());

    let ep = Endpoint {
        id: "ep-hot".to_string(),
        name: "hot-model".to_string(),
        provider: ProviderKind::RunPod,
        status: EndpointStatus::Ready,
        gpu_id: "L4".to_string(),
        invocation_url: "https://api.runpod.ai/v2/ep-hot".to_string(),
        openai_base_url: Some("https://api.runpod.ai/v2/ep-hot/openai/v1".to_string()),
        created_at: chrono::Utc::now(),
    };
    state.deploy_succeeded("hot-model", ep, ModelTier::Hot).unwrap();
    assert_eq!(state.get("hot-model").unwrap().status_str(), "hot");

    // Deploy warm model
    state.begin_deploy("warm-model").unwrap();
    let ep2 = Endpoint {
        id: "ep-warm".to_string(),
        name: "warm-model".to_string(),
        provider: ProviderKind::RunPod,
        status: EndpointStatus::Ready,
        gpu_id: "A100".to_string(),
        invocation_url: "https://api.runpod.ai/v2/ep-warm".to_string(),
        openai_base_url: None,
        created_at: chrono::Utc::now(),
    };
    state.deploy_succeeded("warm-model", ep2, ModelTier::Warm).unwrap();
    assert_eq!(state.get("warm-model").unwrap().status_str(), "warm");

    // Touch warm model (simulate request)
    state.get_mut("warm-model").unwrap().touch();

    // Evict warm model
    let evicted = state.evict("warm-model").unwrap();
    assert!(evicted.is_some());
    assert_eq!(evicted.unwrap().id, "ep-warm");
    assert!(state.get("warm-model").unwrap().is_cold());

    // Cannot evict hot model
    assert!(state.evict("hot-model").is_err());

    // Cold model stays cold (no deploy triggered)
    assert!(state.get("cold-model").unwrap().is_cold());

    // Simulate failed deploy of cold model
    state.begin_deploy("cold-model").unwrap();
    state.deploy_failed("cold-model", "GPU unavailable".to_string()).unwrap();
    assert!(state.get("cold-model").unwrap().is_error());

    // Reset error → cold
    state.reset_error("cold-model").unwrap();
    assert!(state.get("cold-model").unwrap().is_cold());

    // Persistence roundtrip
    let persisted = state.to_persisted();
    let json = serde_json::to_string_pretty(&persisted).unwrap();
    let loaded: rpx_core::fleet::state::PersistedFleetState =
        serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.models["hot-model"].state, "hot");
    assert_eq!(loaded.models["warm-model"].state, "cold"); // was evicted
    assert_eq!(loaded.models["cold-model"].state, "cold"); // was reset
    assert_eq!(
        loaded.models["hot-model"].endpoint_id,
        Some("ep-hot".to_string())
    );
}

/// Verify auth layer works correctly.
#[test]
fn auth_layer_integration() {
    use rpx_core::fleet::ApiKeyEntry;
    use rpx_core::gateway::auth::{AuthError, AuthLayer};

    let keys = vec![
        ApiKeyEntry {
            key: "sk-valid".to_string(),
            name: "my-app".to_string(),
            budget_usd: Some(100.0),
            rate_limit_rpm: Some(60),
            allowed_models: None,
        },
        ApiKeyEntry {
            key: "sk-restricted".to_string(),
            name: "restricted-app".to_string(),
            budget_usd: Some(10.0),
            rate_limit_rpm: Some(5),
            allowed_models: Some(vec!["qwen-7b".to_string()]),
        },
    ];

    let mut auth = AuthLayer::new(&keys);

    // Valid key works
    assert_eq!(auth.validate("sk-valid").unwrap(), "my-app");

    // Invalid key fails
    assert!(matches!(auth.validate("sk-bad"), Err(AuthError::InvalidKey)));

    // Model access control
    assert!(auth.is_model_allowed("sk-valid", "anything")); // unrestricted
    assert!(auth.is_model_allowed("sk-restricted", "qwen-7b")); // allowed
    assert!(!auth.is_model_allowed("sk-restricted", "llama-8b")); // denied

    // Rate limiting — sk-restricted has 5 rpm (bucket starts with 5 tokens)
    // Consume all tokens rapidly
    for i in 0..5 {
        assert!(auth.validate("sk-restricted").is_ok(), "request {i} should succeed");
    }
    // Next request should fail (bucket empty, too little time to refill)
    assert!(matches!(
        auth.validate("sk-restricted"),
        Err(AuthError::RateLimited)
    ));
}

/// Verify request queue works.
#[tokio::test]
async fn request_queue_integration() {
    use rpx_core::orchestrator::queue::RequestQueue;
    use rpx_core::provider::types::InvocationRequest;
    use std::time::Duration;

    let queue = RequestQueue::new(5, Duration::from_secs(120));

    // Enqueue 3 requests
    for i in 0..3 {
        let req = InvocationRequest {
            body: serde_json::json!({"model": "test", "id": i}),
            stream: false,
            timeout_secs: 60,
        };
        let rx = queue.enqueue(req).await;
        assert!(rx.is_some(), "request {i} should be enqueued");
    }

    // Drain
    let drained = queue.drain().await;
    assert_eq!(drained.len(), 3);
    assert_eq!(drained[0].request.body["id"], 0);
    assert_eq!(drained[2].request.body["id"], 2);

    // Queue is now empty
    let drained = queue.drain().await;
    assert!(drained.is_empty());
}
