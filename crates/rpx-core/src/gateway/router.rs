use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Response, StatusCode, header};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use tokio::sync::RwLock;

use crate::fleet::FleetConfig;
use crate::fleet::state::FleetState;
use crate::gateway::spend::SpendTracker;
use crate::orchestrator::model_manager::ModelManager;
use crate::orchestrator::queue::RequestQueue;
use crate::provider::Provider;
use crate::provider::types::InvocationResponse;
use crate::proxy::{streaming, translator};

pub struct ModelRouter {
    alias_to_index: HashMap<String, usize>,
}

impl ModelRouter {
    pub fn new(config: &FleetConfig) -> Self {
        Self {
            alias_to_index: config.alias_map(),
        }
    }

    pub fn resolve_model(&self, alias: &str) -> Option<usize> {
        self.alias_to_index.get(alias).copied()
    }

    /// Forward a request to the appropriate endpoint.
    /// For cold models: triggers deploy and holds request until ready.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward(
        &self,
        model_alias: &str,
        body: serde_json::Value,
        fleet: &Arc<RwLock<FleetState>>,
        provider: &Arc<dyn Provider>,
        model_manager: &Option<Arc<ModelManager>>,
        config: &FleetConfig,
        request_queues: &HashMap<String, Arc<RequestQueue>>,
        spend: &RwLock<SpendTracker>,
        key_name: Option<&str>,
    ) -> Result<Response<Body>, (StatusCode, String)> {
        let model_idx = self.resolve_model(model_alias).ok_or((
            StatusCode::NOT_FOUND,
            format!(
                r#"{{"error":{{"message":"model '{model_alias}' not found","type":"model_not_found"}}}}"#
            ),
        ))?;

        // Read current state
        let current_state = {
            let state = fleet.read().await;
            let ms = state.get(model_alias).ok_or((
                StatusCode::INTERNAL_SERVER_ERROR,
                r#"{"error":{"message":"model state missing","type":"internal_error"}}"#
                    .to_string(),
            ))?;
            ms.status_str()
        };

        match current_state {
            "hot" | "warm" => {
                // Model is ready — forward directly
                let endpoint = {
                    let state = fleet.read().await;
                    state
                        .get(model_alias)
                        .and_then(|s| s.endpoint().cloned())
                        .ok_or((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            r#"{"error":{"message":"endpoint missing for ready model","type":"internal_error"}}"#.to_string(),
                        ))?
                };

                // Touch last_request
                {
                    let mut s = fleet.write().await;
                    if let Some(ms) = s.get_mut(model_alias) {
                        ms.touch();
                    }
                }

                let result = invoke_endpoint(body, &endpoint, provider).await;

                // Record request (we can't read the response body without consuming it,
                // so we record at least the request count. For full token tracking,
                // the provider-level invoke could return usage metadata separately.)
                if result.is_ok() {
                    if let Some(kn) = key_name {
                        let mut s = spend.write().await;
                        s.record_request(kn);
                    }
                }

                result
            }

            "cold" => {
                // Trigger deploy if model_manager is available
                let Some(mm) = model_manager else {
                    return Err((
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!(
                            r#"{{"error":{{"message":"model '{model_alias}' is cold and no orchestrator configured","type":"model_cold"}}}}"#
                        ),
                    ));
                };

                let entry = &config.models[model_idx];
                let queue = request_queues.get(model_alias);

                // Spawn deploy + queue drain in background
                let mm_clone = mm.clone();
                let entry_clone = entry.clone();
                let fleet_clone = fleet.clone();
                let provider_clone = provider.clone();
                let queue_clone = queue.cloned();
                let alias = model_alias.to_string();
                tokio::spawn(async move {
                    if mm_clone.deploy(&entry_clone, &fleet_clone).await.is_ok() {
                        // Deploy succeeded — drain queue and forward requests
                        if let Some(q) = &queue_clone {
                            let endpoint = {
                                let s = fleet_clone.read().await;
                                s.get(&alias).and_then(|ms| ms.endpoint().cloned())
                            };
                            if let Some(ep) = endpoint {
                                let queued = q.drain().await;
                                tracing::info!(
                                    model = %alias,
                                    queued = queued.len(),
                                    "draining request queue after deploy"
                                );
                                for item in queued {
                                    let result = provider_clone.invoke(&ep, item.request).await;
                                    let _ = item.response_tx.send(result);
                                }
                            }
                        }
                    } else {
                        // Deploy failed — reject all queued requests
                        if let Some(q) = &queue_clone {
                            let queued = q.drain().await;
                            for item in queued {
                                let _ = item.response_tx.send(Err(
                                    crate::provider::types::ProviderError::Other(
                                        "model deploy failed".to_string(),
                                    ),
                                ));
                            }
                        }
                    }
                });

                // Try to hold the request and wait for deploy
                if let Some(q) = queue {
                    let request = translator::to_invocation_request(body);
                    let timeout = q.timeout();

                    if let Some(rx) = q.enqueue(request).await {
                        // Wait for the response with timeout
                        match tokio::time::timeout(timeout, rx).await {
                            Ok(Ok(Ok(response))) => match response {
                                InvocationResponse::Complete(v) => {
                                    Ok(Json(v).into_response())
                                }
                                InvocationResponse::Stream(stream) => {
                                    let sse = stream.map(|c| {
                                        c.map_err(|e| std::io::Error::other(e.to_string()))
                                    });
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .header(header::CONTENT_TYPE, "text/event-stream")
                                        .header(header::CACHE_CONTROL, "no-cache")
                                        .body(Body::from_stream(sse))
                                        .map_err(|e| {
                                            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                                        })
                                }
                            },
                            Ok(Ok(Err(e))) => Err((
                                StatusCode::BAD_GATEWAY,
                                format!(r#"{{"error":{{"message":"{e}","type":"provider_error"}}}}"#),
                            )),
                            Ok(Err(_)) => Err((
                                StatusCode::SERVICE_UNAVAILABLE,
                                r#"{"error":{"message":"deploy completed but response channel dropped","type":"internal_error"}}"#.to_string(),
                            )),
                            Err(_) => Err((
                                StatusCode::GATEWAY_TIMEOUT,
                                format!(
                                    r#"{{"error":{{"message":"model '{model_alias}' deploy timed out after {}s","type":"deploy_timeout"}},"retry_after":30}}"#,
                                    timeout.as_secs()
                                ),
                            )),
                        }
                    } else {
                        Err((
                            StatusCode::SERVICE_UNAVAILABLE,
                            format!(
                                r#"{{"error":{{"message":"model '{model_alias}' deploy queue full","type":"queue_full"}},"retry_after":30}}"#
                            ),
                        ))
                    }
                } else {
                    Err((
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!(
                            r#"{{"error":{{"message":"model '{model_alias}' is cold — deploying","type":"model_cold"}},"retry_after":30}}"#
                        ),
                    ))
                }
            }

            "deploying" => {
                // Model is already being deployed — hold request in queue
                if let Some(q) = request_queues.get(model_alias) {
                    let request = translator::to_invocation_request(body);
                    let timeout = q.timeout();

                    if let Some(rx) = q.enqueue(request).await {
                        match tokio::time::timeout(timeout, rx).await {
                            Ok(Ok(Ok(response))) => match response {
                                InvocationResponse::Complete(v) => {
                                    Ok(Json(v).into_response())
                                }
                                InvocationResponse::Stream(stream) => {
                                    let sse = stream.map(|c| {
                                        c.map_err(|e| std::io::Error::other(e.to_string()))
                                    });
                                    Response::builder()
                                        .status(StatusCode::OK)
                                        .header(header::CONTENT_TYPE, "text/event-stream")
                                        .header(header::CACHE_CONTROL, "no-cache")
                                        .body(Body::from_stream(sse))
                                        .map_err(|e| {
                                            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                                        })
                                }
                            },
                            Ok(Ok(Err(e))) => Err((
                                StatusCode::BAD_GATEWAY,
                                format!(r#"{{"error":{{"message":"{e}","type":"provider_error"}}}}"#),
                            )),
                            Ok(Err(_)) => Err((
                                StatusCode::SERVICE_UNAVAILABLE,
                                r#"{"error":{"message":"response channel dropped","type":"internal_error"}}"#.to_string(),
                            )),
                            Err(_) => Err((
                                StatusCode::GATEWAY_TIMEOUT,
                                format!(
                                    r#"{{"error":{{"message":"model '{model_alias}' deploy timed out","type":"deploy_timeout"}},"retry_after":15}}"#
                                ),
                            )),
                        }
                    } else {
                        Err((
                            StatusCode::SERVICE_UNAVAILABLE,
                            r#"{"error":{"message":"deploy queue full","type":"queue_full"},"retry_after":15}"#.to_string(),
                        ))
                    }
                } else {
                    Err((
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!(
                            r#"{{"error":{{"message":"model '{model_alias}' is deploying","type":"model_deploying"}},"retry_after":15}}"#
                        ),
                    ))
                }
            }

            "error" => Err((
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    r#"{{"error":{{"message":"model '{model_alias}' is in error state","type":"model_error"}}}}"#
                ),
            )),

            _ => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    r#"{{"error":{{"message":"model '{model_alias}' in unknown state","type":"internal_error"}}}}"#
                ),
            )),
        }
    }
}

/// Forward a request to a specific endpoint. Reuses proxy streaming logic.
async fn invoke_endpoint(
    body: serde_json::Value,
    endpoint: &crate::provider::types::Endpoint,
    provider: &Arc<dyn Provider>,
) -> Result<Response<Body>, (StatusCode, String)> {
    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let request = translator::to_invocation_request(body);

    let response = provider
        .invoke(endpoint, request)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                format!(r#"{{"error":{{"message":"{e}","type":"provider_error"}}}}"#),
            )
        })?;

    match response {
        InvocationResponse::Complete(v) => Ok(Json(v).into_response()),
        InvocationResponse::Stream(stream) => {
            if !is_stream {
                let mut collected = String::new();
                let mut pinned = stream;
                while let Some(chunk_result) = pinned.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            for line in text.lines() {
                                if let Some(data) = streaming::parse_sse_line(line) {
                                    collected = data;
                                }
                            }
                        }
                        Err(e) => return Err((StatusCode::BAD_GATEWAY, e.to_string())),
                    }
                }
                let json: serde_json::Value =
                    serde_json::from_str(&collected).unwrap_or(serde_json::Value::Null);
                Ok(Json(json).into_response())
            } else {
                let sse_stream = stream
                    .map(|chunk| chunk.map_err(|e| std::io::Error::other(e.to_string())));
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .header(header::CACHE_CONTROL, "no-cache")
                    .header(header::CONNECTION, "keep-alive")
                    .body(Body::from_stream(sse_stream))
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fleet::FleetConfig;

    fn test_config() -> FleetConfig {
        FleetConfig::from_yaml(
            r#"
provider:
  name: runpod
models:
  - id: org/model-a
    alias: model-a
    tier: hot
    scaling:
      min_workers: 1
  - id: org/model-b
    alias: model-b
    tier: warm
  - id: org/model-c
    alias: model-c
    tier: cold
"#,
        )
        .unwrap()
    }

    #[test]
    fn resolves_known_aliases() {
        let router = ModelRouter::new(&test_config());
        assert_eq!(router.resolve_model("model-a"), Some(0));
        assert_eq!(router.resolve_model("model-b"), Some(1));
        assert_eq!(router.resolve_model("model-c"), Some(2));
    }

    #[test]
    fn unknown_alias_returns_none() {
        let router = ModelRouter::new(&test_config());
        assert_eq!(router.resolve_model("nonexistent"), None);
    }

    #[tokio::test]
    async fn forward_unknown_model_returns_404() {
        let config = test_config();
        let router = ModelRouter::new(&config);
        let fleet = Arc::new(RwLock::new(FleetState::init_from_config(
            &config.models.iter().map(|m| m.display_name()).collect::<Vec<_>>(),
        )));
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::runpod::RunPodProvider::new("test".to_string()));
        let queues = HashMap::new();

        let body = serde_json::json!({"model": "nonexistent", "messages": []});
        let result = router
            .forward("nonexistent", body, &fleet, &provider, &None, &config, &queues, &tokio::sync::RwLock::new(crate::gateway::spend::SpendTracker::new()), None)
            .await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn forward_cold_without_manager_returns_503() {
        let config = test_config();
        let router = ModelRouter::new(&config);
        let fleet = Arc::new(RwLock::new(FleetState::init_from_config(
            &config.models.iter().map(|m| m.display_name()).collect::<Vec<_>>(),
        )));
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::runpod::RunPodProvider::new("test".to_string()));
        let queues = HashMap::new();

        let body = serde_json::json!({"model": "model-c", "messages": []});
        let result = router
            .forward("model-c", body, &fleet, &provider, &None, &config, &queues, &tokio::sync::RwLock::new(crate::gateway::spend::SpendTracker::new()), None)
            .await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn forward_error_model_returns_503() {
        let config = test_config();
        let router = ModelRouter::new(&config);
        let names: Vec<String> = config.models.iter().map(|m| m.display_name()).collect();
        let mut state = FleetState::init_from_config(&names);
        state.begin_deploy("model-a").unwrap();
        state
            .deploy_failed("model-a", "GPU unavailable".to_string())
            .unwrap();
        let fleet = Arc::new(RwLock::new(state));
        let provider: Arc<dyn Provider> =
            Arc::new(crate::provider::runpod::RunPodProvider::new("test".to_string()));
        let queues = HashMap::new();

        let body = serde_json::json!({"model": "model-a", "messages": []});
        let result = router
            .forward("model-a", body, &fleet, &provider, &None, &config, &queues, &tokio::sync::RwLock::new(crate::gateway::spend::SpendTracker::new()), None)
            .await;
        assert!(result.is_err());
        let (status, body) = result.unwrap_err();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("error state"));
    }
}
