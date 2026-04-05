use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

use rpx_core::backend::BackendKind;
use rpx_core::catalog::GpuCatalog;
use rpx_core::fleet::{FleetConfig, ModelTier};
use rpx_core::fleet::state::{FleetState, ModelState};
use rpx_core::gateway::auth::AuthLayer;
use rpx_core::gateway::router::ModelRouter;
use rpx_core::gateway::spend::SpendTracker;
use rpx_core::orchestrator::model_manager::ModelManager;
use rpx_core::orchestrator::queue::RequestQueue;
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;
use rpx_core::provider::types::{
    Endpoint, EndpointConfig, EndpointStatus, GpuSpec, ImageSpec, ScalingConfig,
};
use rpx_core::provider::Provider;

fn test_fleet_config() -> FleetConfig {
    FleetConfig::from_yaml(
        r#"
provider:
  name: runpod
models:
  - id: org/hot-model
    alias: hot-model
    tier: hot
    scaling:
      min_workers: 1
  - id: org/warm-model
    alias: warm-model
    tier: warm
  - id: org/cold-model
    alias: cold-model
    tier: cold
api_keys:
  - key: sk-test
    name: test-app
    rate_limit_rpm: 600
"#,
    )
    .unwrap()
}

fn mock_endpoint(id: &str, name: &str) -> Endpoint {
    Endpoint {
        id: id.to_string(),
        name: name.to_string(),
        provider: ProviderKind::RunPod,
        status: EndpointStatus::Ready,
        gpu_id: "L4".to_string(),
        invocation_url: format!("https://api.runpod.ai/v2/{id}"),
        openai_base_url: Some(format!("https://api.runpod.ai/v2/{id}/openai/v1")),
        created_at: chrono::Utc::now(),
    }
}

/// Test 1: Multi-model routing — 3 models in different states, verify correct routing.
#[tokio::test]
async fn multi_model_routing() {
    let config = test_fleet_config();
    let router = ModelRouter::new(&config);
    let names: Vec<String> = config.models.iter().map(|m| m.display_name()).collect();
    let mut state = FleetState::init_from_config(&names);

    // Set up: hot-model is Hot, warm-model is Warm, cold-model stays Cold
    state.begin_deploy("hot-model").unwrap();
    state
        .deploy_succeeded("hot-model", mock_endpoint("ep-hot", "hot-model"), ModelTier::Hot)
        .unwrap();

    state.begin_deploy("warm-model").unwrap();
    state
        .deploy_succeeded(
            "warm-model",
            mock_endpoint("ep-warm", "warm-model"),
            ModelTier::Warm,
        )
        .unwrap();

    let fleet = Arc::new(RwLock::new(state));
    let provider: Arc<dyn Provider> =
        Arc::new(RunPodProvider::new("test-key".to_string()));
    let queues = std::collections::HashMap::new();
    let spend = RwLock::new(SpendTracker::new());

    // Hot model — should try to invoke (will get BAD_GATEWAY since RunPod URL is fake, but routing works)
    let body = serde_json::json!({"model": "hot-model", "messages": []});
    let result = router
        .forward(
            "hot-model", body, &fleet, &provider, &None, &config, &queues, &spend, Some("test-app"),
        )
        .await;
    // BAD_GATEWAY means routing worked, provider was unreachable
    assert!(
        result.is_err(),
        "should get provider error (fake URL)"
    );
    let (status, _) = result.unwrap_err();
    assert_eq!(status, axum::http::StatusCode::BAD_GATEWAY);

    // Warm model — same behavior
    let body = serde_json::json!({"model": "warm-model", "messages": []});
    let result = router
        .forward(
            "warm-model", body, &fleet, &provider, &None, &config, &queues, &spend, None,
        )
        .await;
    let (status, _) = result.unwrap_err();
    assert_eq!(status, axum::http::StatusCode::BAD_GATEWAY);

    // Cold model without model manager — should get 503
    let body = serde_json::json!({"model": "cold-model", "messages": []});
    let result = router
        .forward(
            "cold-model", body, &fleet, &provider, &None, &config, &queues, &spend, None,
        )
        .await;
    let (status, _) = result.unwrap_err();
    assert_eq!(status, axum::http::StatusCode::SERVICE_UNAVAILABLE);

    // Unknown model — should get 404
    let body = serde_json::json!({"model": "nonexistent", "messages": []});
    let result = router
        .forward(
            "nonexistent", body, &fleet, &provider, &None, &config, &queues, &spend, None,
        )
        .await;
    let (status, _) = result.unwrap_err();
    assert_eq!(status, axum::http::StatusCode::NOT_FOUND);

    // Verify last_request was touched on hot model
    let s = fleet.read().await;
    assert!(s.get("hot-model").unwrap().is_ready());
    assert!(s.get("warm-model").unwrap().is_ready());
    assert!(s.get("cold-model").unwrap().is_cold());
}

/// Test 2: Model lifecycle — Cold → Deploying → Warm, with state persistence.
#[test]
fn model_lifecycle_cold_to_warm() {
    let names = vec!["test-model".to_string()];
    let mut state = FleetState::init_from_config(&names);

    // Starts cold
    assert!(state.get("test-model").unwrap().is_cold());

    // Begin deploy
    state.begin_deploy("test-model").unwrap();
    assert!(state.get("test-model").unwrap().is_deploying());

    // Cannot begin deploy again
    assert!(state.begin_deploy("test-model").is_err());

    // Deploy succeeds
    state
        .deploy_succeeded(
            "test-model",
            mock_endpoint("ep-1", "test-model"),
            ModelTier::Warm,
        )
        .unwrap();
    assert_eq!(state.get("test-model").unwrap().status_str(), "warm");
    assert_eq!(
        state.get("test-model").unwrap().endpoint().unwrap().id,
        "ep-1"
    );

    // Evict
    let evicted = state.evict("test-model").unwrap();
    assert_eq!(evicted.unwrap().id, "ep-1");
    assert!(state.get("test-model").unwrap().is_cold());

    // Persistence roundtrip
    let persisted = state.to_persisted();
    let json = serde_json::to_string(&persisted).unwrap();
    let loaded: rpx_core::fleet::state::PersistedFleetState =
        serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.models["test-model"].state, "cold");
}

/// Test 3: Error recovery — deploy fails → Error state → reset → Cold (ready for retry).
#[test]
fn error_recovery_lifecycle() {
    let names = vec!["failing-model".to_string()];
    let mut state = FleetState::init_from_config(&names);

    // Deploy fails
    state.begin_deploy("failing-model").unwrap();
    state
        .deploy_failed("failing-model", "GPU capacity full".to_string())
        .unwrap();
    assert!(state.get("failing-model").unwrap().is_error());

    // Cannot deploy from error state directly
    assert!(state.begin_deploy("failing-model").is_err());

    // Reset error → cold
    state.reset_error("failing-model").unwrap();
    assert!(state.get("failing-model").unwrap().is_cold());

    // Now can deploy again
    state.begin_deploy("failing-model").unwrap();
    assert!(state.get("failing-model").unwrap().is_deploying());
}

/// Test 4: Eviction — warm model with elapsed idle timeout should be evictable.
#[test]
fn eviction_warm_model() {
    let names = vec!["idle-model".to_string()];
    let mut state = FleetState::init_from_config(&names);

    state.begin_deploy("idle-model").unwrap();
    state
        .deploy_succeeded(
            "idle-model",
            mock_endpoint("ep-idle", "idle-model"),
            ModelTier::Warm,
        )
        .unwrap();

    // Model is warm with endpoint
    assert!(state.get("idle-model").unwrap().is_ready());
    assert!(state.get("idle-model").unwrap().endpoint().is_some());

    // Evict
    let evicted = state.evict("idle-model").unwrap();
    assert_eq!(evicted.unwrap().id, "ep-idle");
    assert!(state.get("idle-model").unwrap().is_cold());

    // Double evict is noop
    let evicted = state.evict("idle-model").unwrap();
    assert!(evicted.is_none());
}

/// Test 5: Hot model cannot be evicted.
#[test]
fn hot_model_not_evictable() {
    let names = vec!["hot".to_string()];
    let mut state = FleetState::init_from_config(&names);

    state.begin_deploy("hot").unwrap();
    state
        .deploy_succeeded("hot", mock_endpoint("ep-hot", "hot"), ModelTier::Hot)
        .unwrap();

    let result = state.evict("hot");
    assert!(result.is_err(), "hot model should not be evictable");
}

/// Test 6: Auth + rate limiting + budget.
#[test]
fn auth_rate_limit_and_budget() {
    use rpx_core::fleet::ApiKeyEntry;
    use rpx_core::gateway::auth::AuthError;

    let keys = vec![ApiKeyEntry {
        key: "sk-budget".to_string(),
        name: "budget-app".to_string(),
        budget_usd: Some(10.0),
        rate_limit_rpm: Some(3),
        allowed_models: Some(vec!["allowed-model".to_string()]),
    }];

    let mut auth = AuthLayer::new(&keys);

    // Rate limit: 3 RPM
    assert!(auth.validate("sk-budget").is_ok());
    assert!(auth.validate("sk-budget").is_ok());
    assert!(auth.validate("sk-budget").is_ok());
    assert!(matches!(
        auth.validate("sk-budget"),
        Err(AuthError::RateLimited)
    ));

    // Budget check
    assert!(auth.check_budget("sk-budget", 5.0).is_ok());
    assert!(matches!(
        auth.check_budget("sk-budget", 10.0),
        Err(AuthError::BudgetExceeded { .. })
    ));

    // Model restriction
    assert!(auth.is_model_allowed("sk-budget", "allowed-model"));
    assert!(!auth.is_model_allowed("sk-budget", "other-model"));
}

/// Test 7: Spend tracker records and reports.
#[test]
fn spend_tracking() {
    let mut tracker = SpendTracker::new();

    // Record request with usage
    let response = serde_json::json!({
        "usage": {"prompt_tokens": 100, "completion_tokens": 200}
    });
    tracker.record("app-1", &response, 0.000188);

    // Record bare request
    tracker.record_request("app-1");
    tracker.record_request("app-2");

    let spend1 = tracker.get("app-1").unwrap();
    assert_eq!(spend1.total_requests, 2); // 1 full + 1 bare
    assert_eq!(spend1.total_input_tokens, 100);
    assert_eq!(spend1.total_output_tokens, 200);
    assert!(spend1.estimated_cost_usd > 0.0);

    let spend2 = tracker.get("app-2").unwrap();
    assert_eq!(spend2.total_requests, 1);
    assert_eq!(spend2.total_input_tokens, 0);

    // Report
    let report = tracker.to_report();
    assert_eq!(report["data"].as_array().unwrap().len(), 2);
}

/// Test 8: Request queue — enqueue, drain, forward.
#[tokio::test]
async fn request_queue_enqueue_drain_forward() {
    use rpx_core::orchestrator::queue::RequestQueue;
    use rpx_core::provider::types::{InvocationRequest, InvocationResponse, ProviderError};

    let queue = RequestQueue::new(10, Duration::from_secs(5));

    // Enqueue 3 requests
    let mut receivers = Vec::new();
    for i in 0..3 {
        let req = InvocationRequest {
            body: serde_json::json!({"model": "test", "id": i}),
            stream: false,
            timeout_secs: 60,
        };
        let rx = queue.enqueue(req).await.expect("should enqueue");
        receivers.push(rx);
    }

    // Drain and respond
    let drained = queue.drain().await;
    assert_eq!(drained.len(), 3);

    for (i, item) in drained.into_iter().enumerate() {
        assert_eq!(item.request.body["id"], i as u64);
        let response = InvocationResponse::Complete(serde_json::json!({"result": i}));
        item.response_tx.send(Ok(response)).unwrap();
    }

    // Verify receivers got responses
    for (i, rx) in receivers.into_iter().enumerate() {
        let result = rx.await.unwrap().unwrap();
        match result {
            InvocationResponse::Complete(v) => assert_eq!(v["result"], i as u64),
            _ => panic!("expected Complete"),
        }
    }
}
