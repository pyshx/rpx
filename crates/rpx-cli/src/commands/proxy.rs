use std::sync::Arc;

use clap::Args;
use rpx_core::config::{Credentials, EndpointStore};
use rpx_core::provider::types::Endpoint;
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;
use rpx_core::provider::Provider;
use rpx_core::proxy::ProxyServer;

use crate::ui;

#[derive(Args)]
pub struct ProxyArgs {
    /// Endpoint name or ID
    pub endpoint: String,

    /// Local port to listen on
    #[arg(long, default_value = "4000")]
    pub port: u16,
}

pub async fn run(args: ProxyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let home = dirs_next::home_dir().ok_or("cannot determine home directory")?;
    let creds = Credentials::load(&home.join(".rpx/credentials.toml"))?;

    let api_key = creds
        .api_key_for_or_env(ProviderKind::RunPod)
        .ok_or("no RunPod API key. Run `rpx login` first.")?;

    let provider = Arc::new(RunPodProvider::new(api_key));

    let store = EndpointStore::load(&home.join(".rpx/endpoints.json"))?;
    let endpoint = if let Some(stored) = store.find_by_name_or_id(&args.endpoint) {
        Endpoint {
            id: stored.id.clone(),
            name: stored.name.clone(),
            provider: ProviderKind::RunPod,
            status: rpx_core::provider::types::EndpointStatus::Ready,
            gpu_id: stored.gpu.clone(),
            invocation_url: stored.invocation_url.clone(),
            openai_base_url: stored.openai_base_url.clone(),
            created_at: chrono::Utc::now(),
        }
    } else {
        provider.get_endpoint(&args.endpoint).await?
    };

    eprintln!(
        "{} Starting proxy for {}",
        ui::success("✓"),
        ui::title(&endpoint.name),
    );
    eprintln!(
        "  {} http://127.0.0.1:{}/v1",
        ui::label("Listening:"),
        args.port,
    );
    eprintln!();
    eprintln!("{}", ui::dim("  Use with any OpenAI client:"));
    eprintln!(
        "{}",
        ui::dim(&format!(
            "    export OPENAI_BASE_URL=http://127.0.0.1:{}/v1",
            args.port
        )),
    );
    eprintln!("{}", ui::dim("    export OPENAI_API_KEY=rpx"));

    let server = ProxyServer::new(endpoint, provider, args.port);
    server.run().await?;

    Ok(())
}
