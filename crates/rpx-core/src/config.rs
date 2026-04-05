use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::backend::BackendKind;
use crate::provider::ProviderKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpxConfig {
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default)]
    pub name: Option<String>,
    pub model: String,
    #[serde(default = "default_backend_str")]
    pub backend: String,
    #[serde(default = "default_auto")]
    pub provider: String,
    #[serde(default = "default_auto")]
    pub gpu: String,
    #[serde(default = "default_gpu_count")]
    pub gpu_count: u8,
    #[serde(default = "default_auto")]
    pub dtype: String,
    #[serde(default)]
    pub backend_args: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub scaling: ScalingDefaults,
    #[serde(default)]
    pub secrets: HashMap<String, String>,
    #[serde(default)]
    pub constraints: Constraints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDefaults {
    #[serde(default)]
    pub min_workers: u32,
    #[serde(default = "default_max_workers")]
    pub max_workers: u32,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u32,
}

impl Default for ScalingDefaults {
    fn default() -> Self {
        Self {
            min_workers: 0,
            max_workers: 3,
            idle_timeout: 300,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Constraints {
    pub max_price_per_hour: Option<f64>,
    #[serde(default)]
    pub preferred_regions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub dtype: String,
    pub max_model_len: Option<u32>,
    pub gpu_memory_utilization: Option<f64>,
    pub tensor_parallel_size: Option<u8>,
    pub max_num_seqs: Option<u32>,
    pub extra: HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    pub fn from_rpx_config(config: &RpxConfig) -> Self {
        let mut extra = config.backend_args.clone();
        let max_model_len = extra
            .remove("max_model_len")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let gpu_memory_utilization = extra
            .remove("gpu_memory_utilization")
            .and_then(|v| v.as_f64());
        let tensor_parallel_size = extra
            .remove("tensor_parallel_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u8);
        let max_num_seqs = extra
            .remove("max_num_seqs")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        Self {
            dtype: config.dtype.clone(),
            max_model_len,
            gpu_memory_utilization,
            tensor_parallel_size,
            max_num_seqs,
            extra,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Credentials {
    #[serde(default)]
    pub runpod: Option<ProviderCredential>,
    #[serde(default)]
    pub vastai: Option<ProviderCredential>,
    #[serde(default)]
    pub beam: Option<ProviderCredential>,
    #[serde(default)]
    pub huggingface: Option<HfCredential>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCredential {
    pub api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfCredential {
    pub token: String,
}

impl RpxConfig {
    pub fn from_yaml(content: &str) -> Result<Self, toml::de::Error> {
        // We use YAML in the spec but serde works on any format.
        // For now, parse as TOML since we already have the dep.
        // TODO: switch to serde_yaml if users prefer YAML syntax.
        toml::from_str(content)
    }

    pub fn resolved_backend(&self) -> Option<BackendKind> {
        if self.backend == "auto" {
            return None;
        }
        BackendKind::from_str_loose(&self.backend)
    }

    pub fn resolved_provider(&self) -> Option<ProviderKind> {
        match self.provider.to_lowercase().as_str() {
            "runpod" => Some(ProviderKind::RunPod),
            "vastai" | "vast.ai" | "vast" => Some(ProviderKind::VastAi),
            "beam" => Some(ProviderKind::Beam),
            _ => None,
        }
    }

    pub fn resolve_secret(&self, key: &str) -> Option<String> {
        let value = self.secrets.get(key)?;
        if let Some(env_key) = value.strip_prefix("env:") {
            std::env::var(env_key).ok()
        } else {
            Some(value.clone())
        }
    }
}

impl Credentials {
    pub fn load(path: &Path) -> Result<Self, CredentialsError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| CredentialsError::Io(e, path.to_path_buf()))?;
        toml::from_str(&content)
            .map_err(|e| CredentialsError::Parse(e, path.to_path_buf()))
    }

    pub fn save(&self, path: &Path) -> Result<(), CredentialsError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CredentialsError::Io(e, parent.to_path_buf()))?;
        }
        let content = toml::to_string_pretty(self)
            .map_err(CredentialsError::Serialize)?;
        std::fs::write(path, content)
            .map_err(|e| CredentialsError::Io(e, path.to_path_buf()))
    }

    pub fn api_key_for(&self, provider: ProviderKind) -> Option<&str> {
        match provider {
            ProviderKind::RunPod => self.runpod.as_ref().map(|c| c.api_key.as_str()),
            ProviderKind::VastAi => self.vastai.as_ref().map(|c| c.api_key.as_str()),
            ProviderKind::Beam => self.beam.as_ref().map(|c| c.api_key.as_str()),
        }
    }

    pub fn api_key_for_or_env(&self, provider: ProviderKind) -> Option<String> {
        if let Some(key) = self.api_key_for(provider) {
            return Some(key.to_string());
        }
        let env_var = match provider {
            ProviderKind::RunPod => "RUNPOD_API_KEY",
            ProviderKind::VastAi => "VASTAI_API_KEY",
            ProviderKind::Beam => "BEAM_API_KEY",
        };
        std::env::var(env_var).ok()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CredentialsError {
    #[error("failed to read {1}: {0}")]
    Io(std::io::Error, PathBuf),
    #[error("failed to parse {1}: {0}")]
    Parse(toml::de::Error, PathBuf),
    #[error("failed to serialize credentials: {0}")]
    Serialize(toml::ser::Error),
}

// --- Endpoint state persistence ---

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EndpointStore {
    pub endpoints: Vec<StoredEndpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEndpoint {
    pub name: String,
    pub id: String,
    pub provider: String,
    pub model_id: String,
    pub backend: String,
    pub gpu: String,
    pub invocation_url: String,
    pub openai_base_url: Option<String>,
    pub created_at: String,
}

impl EndpointStore {
    pub fn load(path: &Path) -> Result<Self, StoreError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| StoreError::Io(e, path.to_path_buf()))?;
        serde_json::from_str(&content)
            .map_err(|e| StoreError::Parse(e, path.to_path_buf()))
    }

    pub fn save(&self, path: &Path) -> Result<(), StoreError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| StoreError::Io(e, parent.to_path_buf()))?;
        }
        let content = serde_json::to_string_pretty(self)
            .map_err(StoreError::Serialize)?;
        std::fs::write(path, content)
            .map_err(|e| StoreError::Io(e, path.to_path_buf()))
    }

    pub fn upsert(&mut self, endpoint: StoredEndpoint) {
        if let Some(existing) = self
            .endpoints
            .iter_mut()
            .find(|e| e.name == endpoint.name || e.id == endpoint.id)
        {
            *existing = endpoint;
        } else {
            self.endpoints.push(endpoint);
        }
    }

    pub fn find_by_name_or_id(&self, query: &str) -> Option<&StoredEndpoint> {
        self.endpoints
            .iter()
            .find(|e| e.name == query || e.id == query)
    }

    pub fn remove(&mut self, query: &str) -> bool {
        let len_before = self.endpoints.len();
        self.endpoints.retain(|e| e.name != query && e.id != query);
        self.endpoints.len() < len_before
    }
}

impl StoredEndpoint {
    pub fn from_endpoint(
        ep: &crate::provider::types::Endpoint,
        model_id: &str,
        backend: &str,
    ) -> Self {
        Self {
            name: ep.name.clone(),
            id: ep.id.clone(),
            provider: ep.provider.to_string(),
            model_id: model_id.to_string(),
            backend: backend.to_string(),
            gpu: ep.gpu_id.clone(),
            invocation_url: ep.invocation_url.clone(),
            openai_base_url: ep.openai_base_url.clone(),
            created_at: ep.created_at.to_rfc3339(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("failed to read {1}: {0}")]
    Io(std::io::Error, PathBuf),
    #[error("failed to parse {1}: {0}")]
    Parse(serde_json::Error, PathBuf),
    #[error("failed to serialize: {0}")]
    Serialize(serde_json::Error),
}

fn default_version() -> String { "1".to_string() }
fn default_backend_str() -> String { "auto".to_string() }
fn default_auto() -> String { "auto".to_string() }
fn default_gpu_count() -> u8 { 1 }
fn default_max_workers() -> u32 { 3 }
fn default_idle_timeout() -> u32 { 300 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let toml = r#"model = "Qwen/Qwen2.5-7B-Instruct""#;
        let config: RpxConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.model, "Qwen/Qwen2.5-7B-Instruct");
        assert_eq!(config.backend, "auto");
        assert_eq!(config.provider, "auto");
        assert_eq!(config.gpu, "auto");
        assert_eq!(config.gpu_count, 1);
    }

    #[test]
    fn parse_full_config() {
        let toml = r#"
            version = "1"
            name = "my-service"
            model = "meta-llama/Llama-3.1-8B-Instruct"
            backend = "vllm"
            provider = "runpod"
            gpu = "l4"
            gpu_count = 1
            dtype = "float16"

            [backend_args]
            max_model_len = 8192
            gpu_memory_utilization = 0.9

            [scaling]
            min_workers = 0
            max_workers = 5
            idle_timeout = 300

            [secrets]
            hf_token = "env:HF_TOKEN"

            [constraints]
            max_price_per_hour = 2.5
            preferred_regions = ["us-east"]
        "#;
        let config: RpxConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.name.as_deref(), Some("my-service"));
        assert_eq!(config.resolved_backend(), Some(BackendKind::Vllm));
        assert_eq!(config.resolved_provider(), Some(ProviderKind::RunPod));
        assert_eq!(config.scaling.max_workers, 5);
        assert_eq!(config.constraints.max_price_per_hour, Some(2.5));
    }

    #[test]
    fn resolved_backend_auto_returns_none() {
        let toml = r#"model = "test""#;
        let config: RpxConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.resolved_backend(), None);
    }

    #[test]
    fn resolved_backend_parses_variants() {
        for (input, expected) in [
            ("vllm", Some(BackendKind::Vllm)),
            ("rvllm", Some(BackendKind::Rvllm)),
            ("tgi", Some(BackendKind::Tgi)),
            ("llamacpp", Some(BackendKind::LlamaCpp)),
            ("unknown", None),
        ] {
            let toml = format!("model = \"test\"\nbackend = \"{input}\"");
            let config: RpxConfig = toml::from_str(&toml).unwrap();
            assert_eq!(config.resolved_backend(), expected, "input: {input}");
        }
    }

    #[test]
    fn model_config_extracts_backend_args() {
        let toml = r#"
            model = "test"
            dtype = "float16"
            [backend_args]
            max_model_len = 4096
            gpu_memory_utilization = 0.85
            tensor_parallel_size = 2
            max_num_seqs = 128
            custom_flag = true
        "#;
        let config: RpxConfig = toml::from_str(toml).unwrap();
        let mc = ModelConfig::from_rpx_config(&config);
        assert_eq!(mc.max_model_len, Some(4096));
        assert_eq!(mc.gpu_memory_utilization, Some(0.85));
        assert_eq!(mc.tensor_parallel_size, Some(2));
        assert_eq!(mc.max_num_seqs, Some(128));
        assert!(mc.extra.contains_key("custom_flag"));
    }

    #[test]
    fn resolve_secret_from_env() {
        std::env::set_var("RPX_TEST_TOKEN", "secret123");
        let toml = r#"
            model = "test"
            [secrets]
            hf_token = "env:RPX_TEST_TOKEN"
            literal = "plaintext"
        "#;
        let config: RpxConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.resolve_secret("hf_token"), Some("secret123".to_string()));
        assert_eq!(config.resolve_secret("literal"), Some("plaintext".to_string()));
        assert_eq!(config.resolve_secret("missing"), None);
        std::env::remove_var("RPX_TEST_TOKEN");
    }

    #[test]
    fn credentials_default_when_missing() {
        let creds = Credentials::load(Path::new("/nonexistent/path/creds.toml")).unwrap();
        assert!(creds.runpod.is_none());
        assert!(creds.vastai.is_none());
    }

    #[test]
    fn credentials_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("credentials.toml");

        let creds = Credentials {
            runpod: Some(ProviderCredential { api_key: "rpa_test".to_string() }),
            ..Default::default()
        };
        creds.save(&path).unwrap();

        let loaded = Credentials::load(&path).unwrap();
        assert_eq!(loaded.api_key_for(ProviderKind::RunPod), Some("rpa_test"));
        assert_eq!(loaded.api_key_for(ProviderKind::VastAi), None);
    }

    #[test]
    fn credentials_env_fallback() {
        let creds = Credentials::default();
        assert!(creds.api_key_for(ProviderKind::RunPod).is_none());

        std::env::set_var("RUNPOD_API_KEY", "env_key");
        assert_eq!(creds.api_key_for_or_env(ProviderKind::RunPod), Some("env_key".to_string()));
        std::env::remove_var("RUNPOD_API_KEY");
    }

    #[test]
    fn endpoint_store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("endpoints.json");

        let mut store = EndpointStore::default();
        store.upsert(StoredEndpoint {
            name: "my-llama".to_string(),
            id: "ep-123".to_string(),
            provider: "RunPod".to_string(),
            model_id: "meta-llama/Llama-3.1-8B".to_string(),
            backend: "vLLM".to_string(),
            gpu: "NVIDIA L4".to_string(),
            invocation_url: "https://api.runpod.ai/v2/ep-123".to_string(),
            openai_base_url: Some("https://api.runpod.ai/v2/ep-123/openai/v1".to_string()),
            created_at: "2026-04-05T12:00:00Z".to_string(),
        });
        store.save(&path).unwrap();

        let loaded = EndpointStore::load(&path).unwrap();
        assert_eq!(loaded.endpoints.len(), 1);
        assert_eq!(loaded.endpoints[0].name, "my-llama");
        assert_eq!(loaded.endpoints[0].id, "ep-123");
    }

    #[test]
    fn endpoint_store_find_by_name_or_id() {
        let mut store = EndpointStore::default();
        store.upsert(StoredEndpoint {
            name: "my-model".to_string(),
            id: "ep-456".to_string(),
            provider: "RunPod".to_string(),
            model_id: "test".to_string(),
            backend: "vLLM".to_string(),
            gpu: "L4".to_string(),
            invocation_url: "https://example.com".to_string(),
            openai_base_url: None,
            created_at: "2026-04-05T12:00:00Z".to_string(),
        });

        assert!(store.find_by_name_or_id("my-model").is_some());
        assert!(store.find_by_name_or_id("ep-456").is_some());
        assert!(store.find_by_name_or_id("nonexistent").is_none());
    }

    #[test]
    fn endpoint_store_upsert_updates_existing() {
        let mut store = EndpointStore::default();
        store.upsert(StoredEndpoint {
            name: "my-model".to_string(),
            id: "ep-1".to_string(),
            provider: "RunPod".to_string(),
            model_id: "old".to_string(),
            backend: "vLLM".to_string(),
            gpu: "L4".to_string(),
            invocation_url: "https://old.com".to_string(),
            openai_base_url: None,
            created_at: "2026-04-05T12:00:00Z".to_string(),
        });
        store.upsert(StoredEndpoint {
            name: "my-model".to_string(),
            id: "ep-1".to_string(),
            provider: "RunPod".to_string(),
            model_id: "new".to_string(),
            backend: "vLLM".to_string(),
            gpu: "A100".to_string(),
            invocation_url: "https://new.com".to_string(),
            openai_base_url: None,
            created_at: "2026-04-05T13:00:00Z".to_string(),
        });

        assert_eq!(store.endpoints.len(), 1);
        assert_eq!(store.endpoints[0].model_id, "new");
    }

    #[test]
    fn endpoint_store_remove() {
        let mut store = EndpointStore::default();
        store.upsert(StoredEndpoint {
            name: "to-delete".to_string(),
            id: "ep-del".to_string(),
            provider: "RunPod".to_string(),
            model_id: "test".to_string(),
            backend: "vLLM".to_string(),
            gpu: "L4".to_string(),
            invocation_url: "https://example.com".to_string(),
            openai_base_url: None,
            created_at: "2026-04-05T12:00:00Z".to_string(),
        });

        assert!(store.remove("to-delete"));
        assert!(store.endpoints.is_empty());
        assert!(!store.remove("nonexistent"));
    }

    #[test]
    fn endpoint_store_default_when_missing() {
        let store = EndpointStore::load(Path::new("/nonexistent/endpoints.json")).unwrap();
        assert!(store.endpoints.is_empty());
    }
}
