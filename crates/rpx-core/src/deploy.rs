use crate::backend::{self, BackendKind};
use crate::catalog::{GpuCatalog, GpuSelection};
use crate::config::{ModelConfig, RpxConfig};
use crate::model::ModelMetadata;
use crate::model::sizing;
use crate::provider::types::*;
use crate::provider::{Provider, ProviderKind};

/// The complete plan for a deployment, resolved from config + model metadata.
#[derive(Debug)]
pub struct DeployPlan {
    pub endpoint_name: String,
    pub model_id: String,
    pub backend: BackendKind,
    pub gpu: GpuSelection,
    pub gpu_count: u8,
    pub scaling: ScalingConfig,
    pub env_vars: std::collections::HashMap<String, String>,
    pub image: ImageSpec,
    pub estimated_vram_gb: f64,
}

/// Resolve a deployment plan from config and model metadata.
pub fn resolve_plan(
    config: &RpxConfig,
    metadata: &ModelMetadata,
    catalog: &GpuCatalog,
    provider: &dyn Provider,
) -> Result<DeployPlan, DeployError> {
    // 1. Select backend
    let backend_kind = config
        .resolved_backend()
        .unwrap_or_else(backend::auto_select_backend);

    let backend = backend::get_backend(backend_kind)
        .map_err(|e| DeployError::Backend(e.to_string()))?;

    // 2. Determine model parameters
    let params_b = metadata
        .parameters_billions
        .or_else(|| sizing::estimate_params_from_name(&config.model))
        .ok_or_else(|| {
            DeployError::ModelSizing(format!(
                "cannot determine parameter count for {}. Specify --gpu manually.",
                config.model
            ))
        })?;

    // 3. Estimate VRAM (use active params for MoE models)
    let effective_params = sizing::effective_params_for_vram(&config.model, params_b);
    let estimated_vram = backend.estimate_vram_gb(effective_params, &config.dtype);

    // 4. Select GPU
    let provider_filter = config.resolved_provider().or(Some(provider.kind()));
    let gpu = if config.gpu == "auto" {
        catalog.select_cheapest(
            estimated_vram,
            provider_filter,
            config.constraints.max_price_per_hour,
            config.gpu_count,
        ).map_err(|e| DeployError::GpuSelection(e.to_string()))?
    } else {
        let pk = provider_filter.unwrap_or(ProviderKind::RunPod);
        catalog
            .find_gpu(&config.gpu, pk)
            .ok_or_else(|| {
                DeployError::GpuSelection(format!(
                    "GPU '{}' not found in catalog for {}",
                    config.gpu, pk
                ))
            })?
    };

    // 5. Resolve image
    // default_image() may include a tag (e.g., "repo:tag"). Split it so
    // we don't create invalid double-tag references like "repo:tag:latest".
    let (default_repo, default_tag) = {
        let img = backend.default_image();
        if let Some((repo, tag)) = img.rsplit_once(':') {
            // Check it's a tag not a port (e.g., ghcr.io:443)
            if !tag.contains('/') {
                (repo.to_string(), tag.to_string())
            } else {
                (img.to_string(), "latest".to_string())
            }
        } else {
            (img.to_string(), "latest".to_string())
        }
    };

    let image = if provider.supports_native_template(&backend_kind) {
        provider
            .native_template(&backend_kind)
            .unwrap_or_else(|| ImageSpec::PrebuiltImage {
                registry_url: default_repo.clone(),
                tag: default_tag.clone(),
            })
    } else {
        ImageSpec::PrebuiltImage {
            registry_url: default_repo,
            tag: default_tag,
        }
    };

    // 6. Build env vars
    let model_config = ModelConfig::from_rpx_config(config);
    let mut env_vars = backend.env_vars(&config.model, &model_config);

    // Add HF token if available
    if let Some(token) = config.resolve_secret("hf_token") {
        env_vars.insert("HF_TOKEN".to_string(), token);
    }

    // 7. Derive endpoint name
    let endpoint_name = config
        .name
        .clone()
        .unwrap_or_else(|| sanitize_name(&config.model));

    Ok(DeployPlan {
        endpoint_name,
        model_id: config.model.clone(),
        backend: backend_kind,
        gpu,
        gpu_count: config.gpu_count,
        scaling: ScalingConfig {
            min_workers: config.scaling.min_workers,
            max_workers: config.scaling.max_workers,
            idle_timeout_secs: config.scaling.idle_timeout,
        },
        env_vars,
        image,
        estimated_vram_gb: estimated_vram,
    })
}

/// Execute a deployment plan against a provider.
pub async fn execute_plan(
    plan: &DeployPlan,
    provider: &dyn Provider,
) -> Result<Endpoint, DeployError> {
    let config = EndpointConfig {
        name: plan.endpoint_name.clone(),
        model_id: plan.model_id.clone(),
        backend: plan.backend,
        gpu: GpuSpec {
            id: plan.gpu.gpu_id.clone(),
            name: plan.gpu.gpu_name.clone(),
            provider_gpu_id: plan.gpu.provider_gpu_id.clone(),
            vram_gb: plan.gpu.vram_gb,
            price_per_sec: plan.gpu.price_per_sec,
            multi_gpu_max: 8,
        },
        gpu_count: plan.gpu_count,
        scaling: plan.scaling.clone(),
        env_vars: plan.env_vars.clone(),
        image: plan.image.clone(),
    };

    provider
        .create_endpoint(&config)
        .await
        .map_err(|e| DeployError::Provider(e.to_string()))
}

/// Sanitize a model ID into a valid endpoint name.
fn sanitize_name(model_id: &str) -> String {
    model_id
        .to_lowercase()
        .replace(['/', '.', '_'], "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum DeployError {
    #[error("backend error: {0}")]
    Backend(String),

    #[error("model sizing: {0}")]
    ModelSizing(String),

    #[error("GPU selection: {0}")]
    GpuSelection(String),

    #[error("provider error: {0}")]
    Provider(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_name_basic() {
        assert_eq!(
            sanitize_name("meta-llama/Llama-3.1-8B-Instruct"),
            "meta-llama-llama-3-1-8b-instruct"
        );
    }

    #[test]
    fn sanitize_name_dots_and_underscores() {
        assert_eq!(sanitize_name("Qwen/Qwen2.5-7B"), "qwen-qwen2-5-7b");
        assert_eq!(sanitize_name("some_model_name"), "some-model-name");
    }

    #[test]
    fn sanitize_name_strips_leading_trailing_dashes() {
        assert_eq!(sanitize_name("/model/"), "model");
    }

    #[test]
    fn resolve_plan_with_auto_backend() {
        use crate::provider::runpod::RunPodProvider;

        let config: RpxConfig = toml::from_str(r#"model = "test/model-7B""#).unwrap();
        let metadata = ModelMetadata {
            model_id: "test/model-7B".to_string(),
            pipeline_tag: Some("text-generation".to_string()),
            parameters_billions: Some(7.0),
            gated: false,
        };
        let catalog = GpuCatalog::load_embedded().unwrap();
        let provider = RunPodProvider::new("fake-key".to_string());

        let plan = resolve_plan(&config, &metadata, &catalog, &provider).unwrap();
        assert_eq!(plan.backend, BackendKind::Vllm); // auto-selects vLLM
        assert!(plan.gpu.vram_gb as f64 >= plan.estimated_vram_gb);
        assert_eq!(plan.gpu.provider, ProviderKind::RunPod);
    }

    #[test]
    fn resolve_plan_with_explicit_gpu() {
        use crate::provider::runpod::RunPodProvider;

        let config: RpxConfig = toml::from_str(r#"
            model = "test/model-7B"
            gpu = "a100-80gb"
        "#).unwrap();
        let metadata = ModelMetadata {
            model_id: "test/model-7B".to_string(),
            pipeline_tag: None,
            parameters_billions: Some(7.0),
            gated: false,
        };
        let catalog = GpuCatalog::load_embedded().unwrap();
        let provider = RunPodProvider::new("fake-key".to_string());

        let plan = resolve_plan(&config, &metadata, &catalog, &provider).unwrap();
        assert_eq!(plan.gpu.gpu_id, "a100-80gb");
    }

    #[test]
    fn resolve_plan_falls_back_to_name_estimation() {
        use crate::provider::runpod::RunPodProvider;

        let config: RpxConfig = toml::from_str(r#"model = "org/model-13B-chat""#).unwrap();
        let metadata = ModelMetadata {
            model_id: "org/model-13B-chat".to_string(),
            pipeline_tag: None,
            parameters_billions: None, // HF didn't return this
            gated: false,
        };
        let catalog = GpuCatalog::load_embedded().unwrap();
        let provider = RunPodProvider::new("fake-key".to_string());

        let plan = resolve_plan(&config, &metadata, &catalog, &provider).unwrap();
        // Should have parsed 13B from name
        assert!(plan.estimated_vram_gb > 20.0); // 13B fp16 ≈ 33.8 GB
    }

    #[test]
    fn resolve_plan_error_when_no_params() {
        use crate::provider::runpod::RunPodProvider;

        let config: RpxConfig = toml::from_str(r#"model = "bert-base-uncased""#).unwrap();
        let metadata = ModelMetadata {
            model_id: "bert-base-uncased".to_string(),
            pipeline_tag: None,
            parameters_billions: None,
            gated: false,
        };
        let catalog = GpuCatalog::load_embedded().unwrap();
        let provider = RunPodProvider::new("fake-key".to_string());

        let result = resolve_plan(&config, &metadata, &catalog, &provider);
        assert!(result.is_err());
    }
}
