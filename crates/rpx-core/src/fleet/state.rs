use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::provider::types::Endpoint;
use super::ModelTier;

/// Runtime state for a single model in the fleet.
#[derive(Debug, Clone)]
pub enum ModelState {
    Cold,
    Deploying {
        started_at: Instant,
    },
    Warm {
        endpoint: Endpoint,
        last_request: Instant,
    },
    Hot {
        endpoint: Endpoint,
        last_request: Instant,
    },
    Error {
        message: String,
        last_attempt: Instant,
        retry_count: u32,
    },
}

impl ModelState {
    pub fn is_cold(&self) -> bool {
        matches!(self, Self::Cold)
    }

    pub fn is_deploying(&self) -> bool {
        matches!(self, Self::Deploying { .. })
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Warm { .. } | Self::Hot { .. })
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    pub fn endpoint(&self) -> Option<&Endpoint> {
        match self {
            Self::Warm { endpoint, .. } | Self::Hot { endpoint, .. } => Some(endpoint),
            _ => None,
        }
    }

    pub fn status_str(&self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Deploying { .. } => "deploying",
            Self::Warm { .. } => "warm",
            Self::Hot { .. } => "hot",
            Self::Error { .. } => "error",
        }
    }

    pub fn touch(&mut self) {
        match self {
            Self::Warm { last_request, .. } | Self::Hot { last_request, .. } => {
                *last_request = Instant::now();
            }
            _ => {}
        }
    }
}

/// The complete runtime state of the fleet.
#[derive(Debug)]
pub struct FleetState {
    models: HashMap<String, ModelState>,
}

impl FleetState {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Initialize state for all models based on their tier.
    /// Hot models start as Cold (will be deployed at startup).
    /// Warm and Cold models start as Cold.
    pub fn init_from_config(model_names: &[String]) -> Self {
        let models = model_names
            .iter()
            .map(|name| (name.clone(), ModelState::Cold))
            .collect();
        Self { models }
    }

    pub fn get(&self, model_name: &str) -> Option<&ModelState> {
        self.models.get(model_name)
    }

    pub fn get_mut(&mut self, model_name: &str) -> Option<&mut ModelState> {
        self.models.get_mut(model_name)
    }

    pub fn set(&mut self, model_name: &str, state: ModelState) {
        self.models.insert(model_name.to_string(), state);
    }

    pub fn all(&self) -> &HashMap<String, ModelState> {
        &self.models
    }

    /// Transition a model from Cold → Deploying.
    /// Returns Err if model is not in Cold state.
    pub fn begin_deploy(&mut self, model_name: &str) -> Result<(), StateError> {
        let state = self
            .models
            .get(model_name)
            .ok_or_else(|| StateError::UnknownModel(model_name.to_string()))?;

        match state {
            ModelState::Cold => {
                self.models.insert(
                    model_name.to_string(),
                    ModelState::Deploying {
                        started_at: Instant::now(),
                    },
                );
                Ok(())
            }
            _ => Err(StateError::InvalidTransition {
                model: model_name.to_string(),
                from: state.status_str(),
                to: "deploying",
            }),
        }
    }

    /// Transition a model from Deploying → Warm or Hot (based on tier).
    pub fn deploy_succeeded(
        &mut self,
        model_name: &str,
        endpoint: Endpoint,
        tier: ModelTier,
    ) -> Result<(), StateError> {
        let state = self
            .models
            .get(model_name)
            .ok_or_else(|| StateError::UnknownModel(model_name.to_string()))?;

        if !state.is_deploying() {
            return Err(StateError::InvalidTransition {
                model: model_name.to_string(),
                from: state.status_str(),
                to: "warm/hot",
            });
        }

        let now = Instant::now();
        let new_state = match tier {
            ModelTier::Hot => ModelState::Hot {
                endpoint,
                last_request: now,
            },
            _ => ModelState::Warm {
                endpoint,
                last_request: now,
            },
        };
        self.models.insert(model_name.to_string(), new_state);
        Ok(())
    }

    /// Transition a model from Deploying → Error.
    pub fn deploy_failed(
        &mut self,
        model_name: &str,
        message: String,
    ) -> Result<(), StateError> {
        let state = self
            .models
            .get(model_name)
            .ok_or_else(|| StateError::UnknownModel(model_name.to_string()))?;

        if !state.is_deploying() {
            return Err(StateError::InvalidTransition {
                model: model_name.to_string(),
                from: state.status_str(),
                to: "error",
            });
        }

        self.models.insert(
            model_name.to_string(),
            ModelState::Error {
                message,
                last_attempt: Instant::now(),
                retry_count: 0,
            },
        );
        Ok(())
    }

    /// Evict a Warm model → Cold (deletes endpoint externally).
    pub fn evict(&mut self, model_name: &str) -> Result<Option<Endpoint>, StateError> {
        let state = self
            .models
            .get(model_name)
            .ok_or_else(|| StateError::UnknownModel(model_name.to_string()))?;

        match state {
            ModelState::Warm { endpoint, .. } => {
                let ep = endpoint.clone();
                self.models
                    .insert(model_name.to_string(), ModelState::Cold);
                Ok(Some(ep))
            }
            ModelState::Cold => Ok(None),
            _ => Err(StateError::InvalidTransition {
                model: model_name.to_string(),
                from: state.status_str(),
                to: "cold",
            }),
        }
    }

    /// Reset an Error model → Cold (ready for retry).
    pub fn reset_error(&mut self, model_name: &str) -> Result<(), StateError> {
        let state = self
            .models
            .get(model_name)
            .ok_or_else(|| StateError::UnknownModel(model_name.to_string()))?;

        if !state.is_error() {
            return Err(StateError::InvalidTransition {
                model: model_name.to_string(),
                from: state.status_str(),
                to: "cold",
            });
        }

        self.models
            .insert(model_name.to_string(), ModelState::Cold);
        Ok(())
    }
}

impl Default for FleetState {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of fleet state for persistence.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistedFleetState {
    pub models: HashMap<String, PersistedModelState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedModelState {
    pub endpoint_id: Option<String>,
    pub state: String,
    pub last_request: Option<String>,
    pub error_message: Option<String>,
    pub retry_count: Option<u32>,
}

impl FleetState {
    /// Snapshot the runtime state for persistence.
    pub fn to_persisted(&self) -> PersistedFleetState {
        let models = self
            .models
            .iter()
            .map(|(name, state)| {
                let persisted = match state {
                    ModelState::Cold => PersistedModelState {
                        endpoint_id: None,
                        state: "cold".to_string(),
                        last_request: None,
                        error_message: None,
                        retry_count: None,
                    },
                    ModelState::Deploying { .. } => PersistedModelState {
                        endpoint_id: None,
                        state: "deploying".to_string(),
                        last_request: None,
                        error_message: None,
                        retry_count: None,
                    },
                    ModelState::Warm { endpoint, .. } => PersistedModelState {
                        endpoint_id: Some(endpoint.id.clone()),
                        state: "warm".to_string(),
                        last_request: Some(chrono::Utc::now().to_rfc3339()),
                        error_message: None,
                        retry_count: None,
                    },
                    ModelState::Hot { endpoint, .. } => PersistedModelState {
                        endpoint_id: Some(endpoint.id.clone()),
                        state: "hot".to_string(),
                        last_request: Some(chrono::Utc::now().to_rfc3339()),
                        error_message: None,
                        retry_count: None,
                    },
                    ModelState::Error {
                        message,
                        retry_count,
                        ..
                    } => PersistedModelState {
                        endpoint_id: None,
                        state: "error".to_string(),
                        last_request: None,
                        error_message: Some(message.clone()),
                        retry_count: Some(*retry_count),
                    },
                };
                (name.clone(), persisted)
            })
            .collect();

        PersistedFleetState { models }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("unknown model: {0}")]
    UnknownModel(String),

    #[error("invalid transition for '{model}': {from} → {to}")]
    InvalidTransition {
        model: String,
        from: &'static str,
        to: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderKind;
    use crate::provider::types::EndpointStatus;

    fn test_endpoint(id: &str) -> Endpoint {
        Endpoint {
            id: id.to_string(),
            name: format!("test-{id}"),
            provider: ProviderKind::RunPod,
            status: EndpointStatus::Ready,
            gpu_id: "L4".to_string(),
            invocation_url: format!("https://api.runpod.ai/v2/{id}"),
            openai_base_url: Some(format!("https://api.runpod.ai/v2/{id}/openai/v1")),
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn init_all_cold() {
        let names = vec!["llama".to_string(), "qwen".to_string()];
        let state = FleetState::init_from_config(&names);
        assert!(state.get("llama").unwrap().is_cold());
        assert!(state.get("qwen").unwrap().is_cold());
        assert!(state.get("nonexistent").is_none());
    }

    #[test]
    fn cold_to_deploying() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        assert!(state.get("test").unwrap().is_deploying());
    }

    #[test]
    fn deploying_to_warm() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        state
            .deploy_succeeded("test", test_endpoint("ep-1"), ModelTier::Warm)
            .unwrap();

        let s = state.get("test").unwrap();
        assert!(s.is_ready());
        assert_eq!(s.status_str(), "warm");
        assert_eq!(s.endpoint().unwrap().id, "ep-1");
    }

    #[test]
    fn deploying_to_hot() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        state
            .deploy_succeeded("test", test_endpoint("ep-2"), ModelTier::Hot)
            .unwrap();

        assert_eq!(state.get("test").unwrap().status_str(), "hot");
    }

    #[test]
    fn deploying_to_error() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        state
            .deploy_failed("test", "GPU unavailable".to_string())
            .unwrap();

        let s = state.get("test").unwrap();
        assert!(s.is_error());
        assert_eq!(s.status_str(), "error");
    }

    #[test]
    fn evict_warm_to_cold() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        state
            .deploy_succeeded("test", test_endpoint("ep-3"), ModelTier::Warm)
            .unwrap();

        let evicted = state.evict("test").unwrap();
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().id, "ep-3");
        assert!(state.get("test").unwrap().is_cold());
    }

    #[test]
    fn evict_cold_is_noop() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        let evicted = state.evict("test").unwrap();
        assert!(evicted.is_none());
    }

    #[test]
    fn cannot_evict_hot() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        state
            .deploy_succeeded("test", test_endpoint("ep-4"), ModelTier::Hot)
            .unwrap();

        let result = state.evict("test");
        assert!(result.is_err());
    }

    #[test]
    fn error_reset_to_cold() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        state
            .deploy_failed("test", "timeout".to_string())
            .unwrap();
        state.reset_error("test").unwrap();
        assert!(state.get("test").unwrap().is_cold());
    }

    #[test]
    fn invalid_double_deploy() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        let result = state.begin_deploy("test");
        assert!(result.is_err());
    }

    #[test]
    fn touch_updates_last_request() {
        let mut state = FleetState::init_from_config(&["test".to_string()]);
        state.begin_deploy("test").unwrap();
        state
            .deploy_succeeded("test", test_endpoint("ep-5"), ModelTier::Warm)
            .unwrap();

        let before = match state.get("test").unwrap() {
            ModelState::Warm { last_request, .. } => *last_request,
            _ => panic!("expected warm"),
        };

        std::thread::sleep(std::time::Duration::from_millis(10));
        state.get_mut("test").unwrap().touch();

        let after = match state.get("test").unwrap() {
            ModelState::Warm { last_request, .. } => *last_request,
            _ => panic!("expected warm"),
        };
        assert!(after > before);
    }

    #[test]
    fn unknown_model_errors() {
        let mut state = FleetState::new();
        assert!(state.begin_deploy("ghost").is_err());
        assert!(state.evict("ghost").is_err());
    }

    #[test]
    fn persistence_roundtrip() {
        let mut state = FleetState::init_from_config(&[
            "cold-model".to_string(),
            "warm-model".to_string(),
        ]);

        state.begin_deploy("warm-model").unwrap();
        state
            .deploy_succeeded("warm-model", test_endpoint("ep-persist"), ModelTier::Warm)
            .unwrap();

        let persisted = state.to_persisted();
        assert_eq!(persisted.models["cold-model"].state, "cold");
        assert_eq!(persisted.models["warm-model"].state, "warm");
        assert_eq!(
            persisted.models["warm-model"].endpoint_id,
            Some("ep-persist".to_string())
        );

        // Verify serialization works
        let json = serde_json::to_string_pretty(&persisted).unwrap();
        let loaded: PersistedFleetState = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.models.len(), 2);
    }
}
