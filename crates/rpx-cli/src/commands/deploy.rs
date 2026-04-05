use std::path::Path;
use std::sync::Arc;

use clap::Args;
use rpx_core::catalog::GpuCatalog;
use rpx_core::config::{Credentials, EndpointStore, RpxConfig, StoredEndpoint};
use rpx_core::deploy;
use rpx_core::model;
use rpx_core::provider::types::EndpointStatus;
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;
use rpx_core::provider::Provider;

use crate::ui;

#[derive(Args)]
pub struct DeployArgs {
    /// HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
    pub model: String,

    /// Backend to use
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// Provider to deploy on
    #[arg(long, default_value = "auto")]
    pub provider: String,

    /// GPU type (auto-selected if not specified)
    #[arg(long, default_value = "auto")]
    pub gpu: String,

    /// Number of GPUs for tensor parallelism
    #[arg(long, default_value = "1")]
    pub gpu_count: u8,

    /// Minimum workers (0 = scale to zero)
    #[arg(long, default_value = "0")]
    pub min_workers: u32,

    /// Maximum workers
    #[arg(long, default_value = "3")]
    pub max_workers: u32,

    /// Endpoint name (derived from model if not specified)
    #[arg(long)]
    pub name: Option<String>,

    /// Don't wait for endpoint to become ready
    #[arg(long)]
    pub no_wait: bool,

    /// Only show what would be deployed, don't actually create the endpoint
    #[arg(long)]
    pub dry_run: bool,
}

pub async fn run(args: DeployArgs) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(&args)?;
    let home = dirs_next::home_dir().ok_or("cannot determine home directory")?;

    let provider_kind = config
        .resolved_provider()
        .unwrap_or(ProviderKind::RunPod);

    // 1. Fetch model metadata (no API key needed)
    let mut spinner = ui::InlineSpinner::new(&format!("Resolving model {}", config.model));
    spinner.tick();
    let hf_token = config.resolve_secret("hf_token");
    let http_client = reqwest::Client::new();
    let metadata = model::fetch_model_metadata(
        &http_client,
        &config.model,
        hf_token.as_deref(),
    )
    .await?;
    spinner.finish(&format!("Resolved {}", config.model));

    if metadata.gated && hf_token.is_none() {
        return Err(format!(
            "model {} is gated. Set HF_TOKEN env var or add secrets.hf_token to rpx.yaml",
            config.model
        )
        .into());
    }

    // 2. Resolve plan (no API key needed — only uses local provider traits)
    let catalog = GpuCatalog::load_embedded()?;
    let stub_provider = RunPodProvider::new(String::new());
    let plan = deploy::resolve_plan(&config, &metadata, &catalog, &stub_provider)?;

    eprintln!(
        "{} {} on {} ({}, {}x {})",
        ui::dim("→"),
        plan.model_id,
        plan.gpu.gpu_name,
        plan.backend,
        plan.gpu_count,
        plan.gpu.provider,
    );
    eprintln!(
        "  {} VRAM: {:.1} GB | Cost: ${:.2}/hr",
        ui::dim("│"),
        plan.estimated_vram_gb,
        plan.gpu.price_per_hour,
    );

    // Dry run stops here
    if args.dry_run {
        eprintln!();
        eprintln!("{}", ui::endpoint_card(&ui::EndpointCardInfo {
            name: &plan.endpoint_name,
            id: "(dry run)",
            provider: &plan.gpu.provider.to_string(),
            gpu: &plan.gpu.gpu_name,
            backend: &plan.backend.to_string(),
            status: "Pending",
            vram: plan.estimated_vram_gb,
            cost_per_hour: plan.gpu.price_per_hour,
        }));
        return Ok(());
    }

    // 3. Now we need the API key to actually create the endpoint
    let creds = Credentials::load(&home.join(".rpx/credentials.toml"))?;
    let api_key = creds
        .api_key_for_or_env(provider_kind)
        .ok_or_else(|| {
            format!(
                "no API key for {}. Run `rpx login {}` first.",
                provider_kind,
                provider_kind.to_string().to_lowercase()
            )
        })?;

    let provider: Arc<dyn Provider> = match provider_kind {
        ProviderKind::RunPod => Arc::new(RunPodProvider::new(api_key)),
        _ => return Err(format!("{provider_kind} provider not yet implemented").into()),
    };

    // 4. Create endpoint
    let mut spinner = ui::InlineSpinner::new("Creating endpoint...");
    spinner.tick();
    let endpoint = deploy::execute_plan(&plan, provider.as_ref()).await?;
    spinner.finish("Endpoint created");

    // 5. Poll until ready
    if !args.no_wait && endpoint.status != EndpointStatus::Ready {
        let mut spinner = ui::InlineSpinner::new("Waiting for endpoint to become ready...");
        let mut attempts = 0;
        let max_attempts = 180;

        loop {
            spinner.tick();
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            attempts += 1;

            match provider.get_endpoint(&endpoint.id).await {
                Ok(ep) if ep.status == EndpointStatus::Ready => {
                    spinner.finish("Endpoint ready");
                    break;
                }
                Ok(ep) if matches!(ep.status, EndpointStatus::Error(_)) => {
                    spinner.fail(&format!("Endpoint error: {}", ep.status));
                    break;
                }
                Ok(_) => {
                    if attempts >= max_attempts {
                        spinner.fail("Timed out waiting for endpoint");
                        eprintln!("{}", ui::dim("  Check status with: rpx status ") + &endpoint.name);
                        break;
                    }
                }
                Err(_) if attempts < max_attempts => continue,
                Err(e) => {
                    spinner.fail(&format!("Failed to check status: {e}"));
                    break;
                }
            }
        }
    }

    // 6. Print result
    eprintln!();
    eprintln!("{}", ui::endpoint_card(&ui::EndpointCardInfo {
        name: &endpoint.name,
        id: &endpoint.id,
        provider: &plan.gpu.provider.to_string(),
        gpu: &plan.gpu.gpu_name,
        backend: &plan.backend.to_string(),
        status: &endpoint.status.to_string(),
        vram: plan.estimated_vram_gb,
        cost_per_hour: plan.gpu.price_per_hour,
    }));

    // 7. Save to local state
    let store_path = home.join(".rpx/endpoints.json");
    let mut store = EndpointStore::load(&store_path)?;
    store.upsert(StoredEndpoint::from_endpoint(
        &endpoint,
        &plan.model_id,
        &plan.backend.to_string(),
    ));
    store.save(&store_path)?;

    Ok(())
}

fn load_config(args: &DeployArgs) -> Result<RpxConfig, Box<dyn std::error::Error>> {
    let mut config = if Path::new("rpx.toml").exists() {
        let content = std::fs::read_to_string("rpx.toml")?;
        toml::from_str(&content)?
    } else {
        RpxConfig {
            version: "1".to_string(),
            name: None,
            model: args.model.clone(),
            backend: "auto".to_string(),
            provider: "auto".to_string(),
            gpu: "auto".to_string(),
            gpu_count: 1,
            dtype: "auto".to_string(),
            backend_args: Default::default(),
            scaling: Default::default(),
            secrets: Default::default(),
            constraints: Default::default(),
        }
    };

    config.model = args.model.clone();
    if args.backend != "auto" {
        config.backend = args.backend.clone();
    }
    if args.provider != "auto" {
        config.provider = args.provider.clone();
    }
    if args.gpu != "auto" {
        config.gpu = args.gpu.clone();
    }
    config.gpu_count = args.gpu_count;
    config.scaling.min_workers = args.min_workers;
    config.scaling.max_workers = args.max_workers;
    if let Some(name) = &args.name {
        config.name = Some(name.clone());
    }

    Ok(config)
}
