use clap::Args;
use rpx_core::config::{Credentials, EndpointStore};
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;
use rpx_core::provider::Provider;

use crate::ui;

#[derive(Args)]
pub struct StatusArgs {
    /// Endpoint name or ID
    pub endpoint: String,
}

pub async fn run(args: StatusArgs) -> Result<(), Box<dyn std::error::Error>> {
    let home = dirs_next::home_dir().ok_or("cannot determine home directory")?;
    let creds = Credentials::load(&home.join(".rpx/credentials.toml"))?;

    let api_key = creds
        .api_key_for_or_env(ProviderKind::RunPod)
        .ok_or("no RunPod API key. Run `rpx login` first.")?;

    let store = EndpointStore::load(&home.join(".rpx/endpoints.json"))?;
    let endpoint_id = store
        .find_by_name_or_id(&args.endpoint)
        .map(|e| e.id.clone())
        .unwrap_or_else(|| args.endpoint.clone());

    let provider = RunPodProvider::new(api_key);
    let endpoint = provider.get_endpoint(&endpoint_id).await?;

    eprintln!("{}", ui::key_value("Name", &endpoint.name));
    eprintln!("{}", ui::key_value("ID", &endpoint.id));
    eprintln!("{}", ui::key_value("Provider", &endpoint.provider.to_string()));
    eprintln!("{}", ui::key_value("GPU", &endpoint.gpu_id));
    eprintln!("{}", ui::key_value("Status", &endpoint.status.to_string()));
    if let Some(url) = &endpoint.openai_base_url {
        eprintln!("{}", ui::key_value("OpenAI", url));
    }

    Ok(())
}
