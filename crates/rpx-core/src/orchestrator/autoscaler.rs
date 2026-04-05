use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use crate::fleet::FleetConfig;
use crate::fleet::state::{FleetState, ModelState};
use crate::provider::Provider;

pub struct Autoscaler {
    provider: Arc<dyn Provider>,
    config: FleetConfig,
    interval: Duration,
}

impl Autoscaler {
    pub fn new(provider: Arc<dyn Provider>, config: FleetConfig) -> Self {
        Self {
            provider,
            config,
            interval: Duration::from_secs(30),
        }
    }

    pub async fn run(&self, state: Arc<RwLock<FleetState>>) {
        let mut interval = tokio::time::interval(self.interval);
        loop {
            interval.tick().await;
            self.tick(&state).await;
        }
    }

    async fn tick(&self, state: &Arc<RwLock<FleetState>>) {
        // Collect actions under read lock, execute under write lock
        let actions = {
            let snapshot = state.read().await;
            let alias_map = self.config.alias_map();
            let mut actions = Vec::new();

            for (model_name, model_state) in snapshot.all() {
                let config_idx = match alias_map.get(model_name) {
                    Some(idx) => *idx,
                    None => continue,
                };
                let entry = &self.config.models[config_idx];

                match model_state {
                    ModelState::Warm {
                        last_request,
                        endpoint,
                        ..
                    } => {
                        let eviction_timeout =
                            Duration::from_secs(entry.scaling.eviction_timeout);
                        if last_request.elapsed() > eviction_timeout {
                            actions.push(AutoscaleAction::Evict {
                                model: model_name.clone(),
                                endpoint_id: endpoint.id.clone(),
                            });
                        }
                    }
                    ModelState::Error {
                        last_attempt,
                        retry_count,
                        ..
                    } => {
                        let backoff = Duration::from_secs(30 * 2u64.pow(*retry_count));
                        if last_attempt.elapsed() > backoff {
                            actions.push(AutoscaleAction::ResetError {
                                model: model_name.clone(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            actions
        };

        // Execute actions outside the read lock
        for action in actions {
            match action {
                AutoscaleAction::Evict {
                    model,
                    endpoint_id,
                } => {
                    tracing::info!(model = %model, "evicting idle model");
                    let _ = self.provider.delete_endpoint(&endpoint_id).await;
                    let mut s = state.write().await;
                    let _ = s.evict(&model);
                }
                AutoscaleAction::ResetError { model } => {
                    tracing::info!(model = %model, "resetting error state for retry");
                    let mut s = state.write().await;
                    let _ = s.reset_error(&model);
                }
            }
        }
    }
}

enum AutoscaleAction {
    Evict { model: String, endpoint_id: String },
    ResetError { model: String },
}
