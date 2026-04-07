pub mod autoscaler;
pub mod model_manager;
pub mod queue;

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::catalog::GpuCatalog;
use crate::config::Credentials;
use crate::fleet::{FleetConfig, ModelTier};
use crate::fleet::state::FleetState;
use crate::fleet::store::FleetStore;
use crate::gateway::GatewayServer;
use crate::provider::ProviderKind;
use crate::provider::runpod::RunPodProvider;
use crate::provider::Provider;

use autoscaler::Autoscaler;
use model_manager::ModelManager;

pub struct Orchestrator {
    config: FleetConfig,
    state: Arc<RwLock<FleetState>>,
    provider: Arc<dyn Provider>,
    model_manager: Arc<ModelManager>,
    store: FleetStore,
}

impl Orchestrator {
    /// Initialize the orchestrator: load config, credentials, state, reconcile.
    pub async fn init(
        config: FleetConfig,
        store_path: std::path::PathBuf,
        credentials_path: std::path::PathBuf,
    ) -> Result<Self, OrchestratorError> {
        // Resolve provider + credentials
        let creds = Credentials::load(&credentials_path)
            .map_err(|e| OrchestratorError::Config(format!("credentials: {e}")))?;

        let provider_kind = match config.provider.name.as_str() {
            "runpod" => ProviderKind::RunPod,
            other => {
                return Err(OrchestratorError::Config(format!(
                    "unsupported provider: {other}"
                )))
            }
        };

        let api_key = creds
            .api_key_for_or_env(provider_kind)
            .ok_or_else(|| {
                OrchestratorError::Config(format!(
                    "no API key for {provider_kind}. Run `rpx login` first."
                ))
            })?;

        let provider: Arc<dyn Provider> = match provider_kind {
            ProviderKind::RunPod => Arc::new(RunPodProvider::new(api_key)),
            _ => {
                return Err(OrchestratorError::Config(format!(
                    "{provider_kind} not yet supported"
                )))
            }
        };

        // Load catalog
        let catalog = GpuCatalog::load_embedded()
            .map_err(|e| OrchestratorError::Config(format!("catalog: {e}")))?;

        // Initialize fleet state — try to restore from persisted state
        let store = FleetStore::new(store_path);
        let model_names: Vec<String> =
            config.models.iter().map(|m| m.display_name()).collect();
        let mut fleet_state = FleetState::init_from_config(&model_names);

        // Reconcile: try persisted state first, then discover from provider
        let persisted = store.load().unwrap_or_default();
        let mut restored_count = 0;

        // Phase 1: Restore from persisted state file (if it exists)
        for (model_name, persisted_model) in &persisted.models {
            if let Some(endpoint_id) = &persisted_model.endpoint_id {
                match provider.get_endpoint(endpoint_id).await {
                    Ok(endpoint) => {
                        let tier = config
                            .models
                            .iter()
                            .find(|m| m.display_name() == *model_name)
                            .map(|m| m.tier)
                            .unwrap_or(crate::fleet::ModelTier::Warm);

                        if fleet_state.get(model_name).is_some_and(|s| s.is_cold()) {
                            let _ = fleet_state.begin_deploy(model_name);
                            let _ = fleet_state.deploy_succeeded(model_name, endpoint, tier);
                            restored_count += 1;
                            tracing::info!(
                                model = %model_name,
                                endpoint = %endpoint_id,
                                "restored endpoint from persisted state"
                            );
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            model = %model_name,
                            endpoint = %endpoint_id,
                            "persisted endpoint no longer exists — starting cold"
                        );
                    }
                }
            }
        }

        // Phase 2: Discover existing endpoints from provider (for Cloud Run / fresh starts)
        // Match by endpoint name against config model aliases
        if restored_count == 0 {
            tracing::info!("no persisted state — discovering existing endpoints from provider");
            if let Ok(live_endpoints) = provider.list_endpoints().await {
                for endpoint in live_endpoints {
                    // Match endpoint name to model config
                    for entry in &config.models {
                        let model_name = entry.display_name();
                        let matches = endpoint.name.contains(&model_name)
                            || endpoint.name.contains(&entry.id.replace('/', "-").to_lowercase());

                        if matches && fleet_state.get(&model_name).is_some_and(|s| s.is_cold()) {
                            let _ = fleet_state.begin_deploy(&model_name);
                            let _ = fleet_state.deploy_succeeded(
                                &model_name,
                                endpoint.clone(),
                                entry.tier,
                            );
                            tracing::info!(
                                model = %model_name,
                                endpoint_id = %endpoint.id,
                                endpoint_name = %endpoint.name,
                                "discovered existing endpoint from provider"
                            );
                            break;
                        }
                    }
                }
            }
        }

        let state = Arc::new(RwLock::new(fleet_state));
        let model_manager = Arc::new(ModelManager::new(
            provider.clone(),
            catalog,
            store.path().to_path_buf(),
        ));

        Ok(Self {
            config,
            state,
            provider,
            model_manager,
            store,
        })
    }

    /// Start the orchestrator: deploy hot models, spawn autoscaler, run gateway.
    pub async fn run(&self) -> Result<(), OrchestratorError> {
        // 1. Deploy all hot models at startup
        let hot_models: Vec<_> = self
            .config
            .models
            .iter()
            .filter(|m| m.tier == ModelTier::Hot)
            .cloned()
            .collect();

        for entry in &hot_models {
            tracing::info!(model = %entry.display_name(), "deploying hot model at startup");
            if let Err(e) = self.model_manager.deploy(entry, &self.state).await {
                tracing::error!(
                    model = %entry.display_name(),
                    error = %e,
                    "failed to deploy hot model — will retry via autoscaler"
                );
            }
        }

        // 2. Spawn autoscaler
        let autoscaler = Autoscaler::new(self.provider.clone(), self.config.clone());
        let autoscaler_state = self.state.clone();
        tokio::spawn(async move {
            autoscaler.run(autoscaler_state).await;
        });

        // 3. Spawn periodic state persistence
        let persist_state = self.state.clone();
        let persist_store_path = self.store.path().to_path_buf();
        tokio::spawn(async move {
            let store = FleetStore::new(persist_store_path);
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                let s = persist_state.read().await;
                let persisted = s.to_persisted();
                if let Err(e) = store.save(&persisted) {
                    tracing::warn!(error = %e, "failed to persist fleet state");
                }
            }
        });

        // 4. Run the gateway (blocks until shutdown)
        let gateway = GatewayServer::new(
            self.config.clone(),
            self.state.clone(),
            self.provider.clone(),
        )
        .with_model_manager(self.model_manager.clone());

        gateway
            .run()
            .await
            .map_err(|e| OrchestratorError::Gateway(e.to_string()))?;

        // Save state on shutdown
        tracing::info!("saving fleet state before shutdown");
        let s = self.state.read().await;
        if let Err(e) = self.store.save(&s.to_persisted()) {
            tracing::error!(error = %e, "failed to save fleet state on shutdown");
        } else {
            tracing::info!("fleet state saved to {}", self.store.path().display());
        }

        Ok(())
    }

    pub fn state(&self) -> &Arc<RwLock<FleetState>> {
        &self.state
    }

    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("config error: {0}")]
    Config(String),
    #[error("gateway error: {0}")]
    Gateway(String),
}
