use std::sync::Arc;

use tokio::sync::RwLock;

use crate::catalog::GpuCatalog;
use crate::deploy;
use crate::fleet::ModelEntry;
use crate::fleet::state::FleetState;
use crate::model;
use crate::provider::Provider;
use crate::provider::types::EndpointStatus;

pub struct ModelManager {
    provider: Arc<dyn Provider>,
    catalog: GpuCatalog,
    http_client: reqwest::Client,
}

impl ModelManager {
    pub fn new(provider: Arc<dyn Provider>, catalog: GpuCatalog) -> Self {
        Self {
            provider,
            catalog,
            http_client: reqwest::Client::new(),
        }
    }

    /// Deploy a model: fetch metadata, resolve plan, create endpoint, poll until ready.
    /// On failure, transitions state to Error before returning.
    pub async fn deploy(
        &self,
        entry: &ModelEntry,
        state: &Arc<RwLock<FleetState>>,
    ) -> Result<(), DeployError> {
        let model_name = entry.display_name();

        // Cold → Deploying
        {
            let mut s = state.write().await;
            s.begin_deploy(&model_name)
                .map_err(|e| DeployError::State(e.to_string()))?;
        }
        tracing::info!(model = %model_name, "deploying model");

        // Run the deploy pipeline, transition to Error on any failure
        match self.deploy_inner(entry, &model_name).await {
            Ok(endpoint) => {
                let mut s = state.write().await;
                s.deploy_succeeded(&model_name, endpoint, entry.tier)
                    .map_err(|e| DeployError::State(e.to_string()))?;
                tracing::info!(model = %model_name, tier = %entry.tier, "model deployed");
                Ok(())
            }
            Err(e) => {
                let mut s = state.write().await;
                let _ = s.deploy_failed(&model_name, e.to_string());
                Err(e)
            }
        }
    }

    async fn deploy_inner(
        &self,
        entry: &ModelEntry,
        model_name: &str,
    ) -> Result<crate::provider::types::Endpoint, DeployError> {
        // Fetch metadata
        let metadata = model::fetch_model_metadata(&self.http_client, &entry.id, None)
            .await
            .map_err(|e| DeployError::Metadata(format!("metadata fetch failed: {e}")))?;

        // Resolve plan
        let rpx_config = entry.to_rpx_config();
        let plan = deploy::resolve_plan(
            &rpx_config,
            &metadata,
            &self.catalog,
            self.provider.as_ref(),
        )
        .map_err(|e| DeployError::Plan(format!("plan resolution failed: {e}")))?;

        tracing::info!(
            model = %model_name,
            gpu = %plan.gpu.gpu_name,
            vram_gb = format!("{:.1}", plan.estimated_vram_gb),
            "resolved deploy plan"
        );

        // Execute plan (create RunPod endpoint)
        let endpoint = deploy::execute_plan(&plan, self.provider.as_ref())
            .await
            .map_err(|e| DeployError::Provider(format!("endpoint creation failed: {e}")))?;

        // Poll until ready
        self.poll_until_ready(&endpoint.id, 180).await
    }

    /// Delete the RunPod endpoint for a model.
    pub async fn undeploy(&self, endpoint_id: &str) -> Result<(), DeployError> {
        self.provider
            .delete_endpoint(endpoint_id)
            .await
            .map_err(|e| DeployError::Provider(e.to_string()))
    }

    async fn poll_until_ready(
        &self,
        endpoint_id: &str,
        max_attempts: u32,
    ) -> Result<crate::provider::types::Endpoint, DeployError> {
        for attempt in 0..max_attempts {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;

            match self.provider.get_endpoint(endpoint_id).await {
                Ok(ep) if ep.status == EndpointStatus::Ready => return Ok(ep),
                Ok(ep) if matches!(ep.status, EndpointStatus::Error(_)) => {
                    return Err(DeployError::Provider(format!(
                        "endpoint entered error state: {}",
                        ep.status
                    )));
                }
                Ok(_) => {
                    if attempt % 10 == 0 && attempt > 0 {
                        tracing::debug!(endpoint = %endpoint_id, attempt, "still waiting...");
                    }
                }
                Err(e) if attempt < max_attempts - 1 => {
                    tracing::warn!(endpoint = %endpoint_id, error = %e, "poll error, retrying");
                }
                Err(e) => return Err(DeployError::Provider(e.to_string())),
            }
        }
        Err(DeployError::Timeout(format!(
            "endpoint {endpoint_id} did not become ready after {max_attempts}s"
        )))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeployError {
    #[error("state error: {0}")]
    State(String),
    #[error("metadata: {0}")]
    Metadata(String),
    #[error("plan: {0}")]
    Plan(String),
    #[error("provider: {0}")]
    Provider(String),
    #[error("timeout: {0}")]
    Timeout(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fleet::{ModelScaling, ModelTier};
    use crate::provider::runpod::RunPodProvider;

    fn test_entry(alias: &str, tier: ModelTier) -> ModelEntry {
        ModelEntry {
            id: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            alias: Some(alias.to_string()),
            tier,
            backend: "vllm".to_string(),
            gpu: "auto".to_string(),
            gpu_count: 1,
            dtype: "auto".to_string(),
            scaling: ModelScaling {
                min_workers: if tier == ModelTier::Hot { 1 } else { 0 },
                ..Default::default()
            },
            backend_args: Default::default(),
        }
    }

    #[test]
    fn model_entry_converts_to_rpx_config() {
        let entry = test_entry("qwen-small", ModelTier::Hot);
        let config = entry.to_rpx_config();
        assert_eq!(config.model, "Qwen/Qwen2.5-1.5B-Instruct");
        assert_eq!(config.name, Some("qwen-small".to_string()));
        assert_eq!(config.backend, "vllm");
        assert_eq!(config.scaling.min_workers, 1);
    }

    #[test]
    fn model_manager_creates_with_catalog() {
        let provider = Arc::new(RunPodProvider::new("test".to_string()));
        let catalog = GpuCatalog::load_embedded().unwrap();
        let _manager = ModelManager::new(provider, catalog);
    }
}
