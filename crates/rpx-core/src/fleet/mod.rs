pub mod state;
pub mod store;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::provider::types::ScalingConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct FleetConfig {
    #[serde(default)]
    pub gateway: GatewayConfig,
    pub provider: ProviderConfig,
    pub models: Vec<ModelEntry>,
    #[serde(default)]
    pub api_keys: Vec<ApiKeyEntry>,
    #[serde(default)]
    pub defaults: FleetDefaults,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GatewayConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_host")]
    pub host: String,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            host: default_host(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub tier: ModelTier,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_auto")]
    pub gpu: String,
    #[serde(default = "default_gpu_count")]
    pub gpu_count: u8,
    #[serde(default = "default_auto")]
    pub dtype: String,
    #[serde(default)]
    pub scaling: ModelScaling,
    #[serde(default)]
    pub backend_args: HashMap<String, serde_json::Value>,
}

impl ModelEntry {
    /// The name used for API routing — alias if set, otherwise derived from model ID.
    pub fn display_name(&self) -> String {
        self.alias
            .clone()
            .unwrap_or_else(|| sanitize_model_id(&self.id))
    }

    /// Convert to an RpxConfig for use with resolve_plan().
    pub fn to_rpx_config(&self) -> crate::config::RpxConfig {
        crate::config::RpxConfig {
            version: "1".to_string(),
            name: Some(self.display_name()),
            model: self.id.clone(),
            backend: self.backend.clone(),
            provider: "auto".to_string(),
            gpu: self.gpu.clone(),
            gpu_count: self.gpu_count,
            dtype: self.dtype.clone(),
            backend_args: self.backend_args.clone(),
            scaling: crate::config::ScalingDefaults {
                min_workers: self.scaling.min_workers,
                max_workers: self.scaling.max_workers,
                idle_timeout: self.scaling.idle_timeout,
            },
            secrets: Default::default(),
            constraints: Default::default(),
        }
    }

    pub fn to_scaling_config(&self) -> ScalingConfig {
        ScalingConfig {
            min_workers: self.scaling.min_workers,
            max_workers: self.scaling.max_workers,
            idle_timeout_secs: self.scaling.idle_timeout,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    Hot,
    #[default]
    Warm,
    Cold,
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hot => write!(f, "hot"),
            Self::Warm => write!(f, "warm"),
            Self::Cold => write!(f, "cold"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelScaling {
    #[serde(default)]
    pub min_workers: u32,
    #[serde(default = "default_max_workers")]
    pub max_workers: u32,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u32,
    #[serde(default = "default_eviction_timeout")]
    pub eviction_timeout: u64,
}

impl Default for ModelScaling {
    fn default() -> Self {
        Self {
            min_workers: 0,
            max_workers: 3,
            idle_timeout: 300,
            eviction_timeout: 3600,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApiKeyEntry {
    pub key: String,
    pub name: String,
    #[serde(default)]
    pub budget_usd: Option<f64>,
    #[serde(default)]
    pub rate_limit_rpm: Option<u32>,
    #[serde(default)]
    pub allowed_models: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FleetDefaults {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_auto")]
    pub gpu: String,
    #[serde(default = "default_auto")]
    pub dtype: String,
}

impl Default for FleetDefaults {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            gpu: default_auto(),
            dtype: default_auto(),
        }
    }
}

impl FleetConfig {
    pub fn from_yaml(content: &str) -> Result<Self, FleetConfigError> {
        let config: FleetConfig =
            serde_yaml::from_str(content).map_err(FleetConfigError::Parse)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), FleetConfigError> {
        if self.models.is_empty() {
            return Err(FleetConfigError::Validation(
                "at least one model must be defined".to_string(),
            ));
        }

        // Check for duplicate aliases
        let mut seen = std::collections::HashSet::new();
        for model in &self.models {
            let name = model.display_name();
            if !seen.insert(name.clone()) {
                return Err(FleetConfigError::Validation(format!(
                    "duplicate model alias: {name}"
                )));
            }
        }

        // Hot models must have min_workers >= 1
        for model in &self.models {
            if model.tier == ModelTier::Hot && model.scaling.min_workers < 1 {
                return Err(FleetConfigError::Validation(format!(
                    "hot model '{}' must have min_workers >= 1",
                    model.display_name()
                )));
            }
        }

        Ok(())
    }

    /// Build a map of display_name → index for fast alias lookup.
    pub fn alias_map(&self) -> HashMap<String, usize> {
        self.models
            .iter()
            .enumerate()
            .map(|(i, m)| (m.display_name(), i))
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FleetConfigError {
    #[error("failed to parse fleet config: {0}")]
    Parse(#[from] serde_yaml::Error),
    #[error("invalid fleet config: {0}")]
    Validation(String),
}

fn sanitize_model_id(id: &str) -> String {
    id.to_lowercase()
        .replace(['/', '.', '_'], "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn default_port() -> u16 { 4000 }
fn default_host() -> String { "0.0.0.0".to_string() }
fn default_backend() -> String { "vllm".to_string() }
fn default_auto() -> String { "auto".to_string() }
fn default_gpu_count() -> u8 { 1 }
fn default_max_workers() -> u32 { 3 }
fn default_idle_timeout() -> u32 { 300 }
fn default_eviction_timeout() -> u64 { 3600 }

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_YAML: &str = r#"
provider:
  name: runpod
models:
  - id: Qwen/Qwen2.5-7B-Instruct
    alias: qwen-7b
    tier: hot
    scaling:
      min_workers: 1
"#;

    const FULL_YAML: &str = r#"
gateway:
  port: 8080
  host: "127.0.0.1"
provider:
  name: runpod
models:
  - id: meta-llama/Llama-3.1-8B-Instruct
    alias: llama-8b
    tier: hot
    backend: vllm
    gpu: l4
    scaling:
      min_workers: 1
      max_workers: 5
      idle_timeout: 300
  - id: Qwen/Qwen2.5-72B-Instruct
    alias: qwen-72b
    tier: warm
    gpu: a100-80gb
    scaling:
      min_workers: 0
      max_workers: 3
      idle_timeout: 600
      eviction_timeout: 7200
  - id: mistralai/Mistral-7B-Instruct-v0.3
    alias: mistral-7b
    tier: cold
api_keys:
  - key: sk-test-key
    name: test-app
    budget_usd: 50.0
    rate_limit_rpm: 120
  - key: sk-admin
    name: admin
defaults:
  backend: vllm
  gpu: auto
"#;

    #[test]
    fn parse_minimal_config() {
        let config = FleetConfig::from_yaml(MINIMAL_YAML).unwrap();
        assert_eq!(config.provider.name, "runpod");
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models[0].display_name(), "qwen-7b");
        assert_eq!(config.models[0].tier, ModelTier::Hot);
        assert_eq!(config.gateway.port, 4000); // default
    }

    #[test]
    fn parse_full_config() {
        let config = FleetConfig::from_yaml(FULL_YAML).unwrap();
        assert_eq!(config.gateway.port, 8080);
        assert_eq!(config.gateway.host, "127.0.0.1");
        assert_eq!(config.models.len(), 3);
        assert_eq!(config.api_keys.len(), 2);

        let llama = &config.models[0];
        assert_eq!(llama.tier, ModelTier::Hot);
        assert_eq!(llama.gpu, "l4");
        assert_eq!(llama.scaling.min_workers, 1);
        assert_eq!(llama.scaling.max_workers, 5);

        let qwen = &config.models[1];
        assert_eq!(qwen.tier, ModelTier::Warm);
        assert_eq!(qwen.scaling.eviction_timeout, 7200);

        let mistral = &config.models[2];
        assert_eq!(mistral.tier, ModelTier::Cold);
        assert_eq!(mistral.scaling.min_workers, 0); // default
    }

    #[test]
    fn alias_defaults_from_model_id() {
        let yaml = r#"
provider:
  name: runpod
models:
  - id: meta-llama/Llama-3.1-8B-Instruct
    tier: warm
"#;
        let config = FleetConfig::from_yaml(yaml).unwrap();
        assert_eq!(
            config.models[0].display_name(),
            "meta-llama-llama-3-1-8b-instruct"
        );
    }

    #[test]
    fn alias_map_works() {
        let config = FleetConfig::from_yaml(FULL_YAML).unwrap();
        let map = config.alias_map();
        assert_eq!(map["llama-8b"], 0);
        assert_eq!(map["qwen-72b"], 1);
        assert_eq!(map["mistral-7b"], 2);
    }

    #[test]
    fn validate_rejects_empty_models() {
        let yaml = r#"
provider:
  name: runpod
models: []
"#;
        let err = FleetConfig::from_yaml(yaml).unwrap_err();
        assert!(err.to_string().contains("at least one model"));
    }

    #[test]
    fn validate_rejects_duplicate_aliases() {
        let yaml = r#"
provider:
  name: runpod
models:
  - id: model-a
    alias: same-name
    tier: warm
  - id: model-b
    alias: same-name
    tier: warm
"#;
        let err = FleetConfig::from_yaml(yaml).unwrap_err();
        assert!(err.to_string().contains("duplicate model alias"));
    }

    #[test]
    fn validate_hot_requires_min_workers() {
        let yaml = r#"
provider:
  name: runpod
models:
  - id: test/model
    tier: hot
    scaling:
      min_workers: 0
"#;
        let err = FleetConfig::from_yaml(yaml).unwrap_err();
        assert!(err.to_string().contains("min_workers >= 1"));
    }

    #[test]
    fn model_entry_to_rpx_config() {
        let config = FleetConfig::from_yaml(FULL_YAML).unwrap();
        let rpx_config = config.models[0].to_rpx_config();
        assert_eq!(rpx_config.model, "meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(rpx_config.name, Some("llama-8b".to_string()));
        assert_eq!(rpx_config.gpu, "l4");
        assert_eq!(rpx_config.scaling.min_workers, 1);
    }

    #[test]
    fn api_key_parsing() {
        let config = FleetConfig::from_yaml(FULL_YAML).unwrap();
        let key = &config.api_keys[0];
        assert_eq!(key.key, "sk-test-key");
        assert_eq!(key.name, "test-app");
        assert_eq!(key.budget_usd, Some(50.0));
        assert_eq!(key.rate_limit_rpm, Some(120));
        assert!(key.allowed_models.is_none());
    }

    #[test]
    fn model_tier_display() {
        assert_eq!(ModelTier::Hot.to_string(), "hot");
        assert_eq!(ModelTier::Warm.to_string(), "warm");
        assert_eq!(ModelTier::Cold.to_string(), "cold");
    }

    #[test]
    fn model_tier_default_is_warm() {
        assert_eq!(ModelTier::default(), ModelTier::Warm);
    }
}
