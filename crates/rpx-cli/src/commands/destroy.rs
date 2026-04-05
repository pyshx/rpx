use clap::Args;
use rpx_core::config::{Credentials, EndpointStore};
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;
use rpx_core::provider::Provider;

use crate::ui;

#[derive(Args)]
pub struct DestroyArgs {
    /// Endpoint name or ID
    pub endpoint: String,

    /// Skip confirmation prompt
    #[arg(long, short = 'y')]
    pub yes: bool,
}

pub async fn run(args: DestroyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let home = dirs_next::home_dir().ok_or("cannot determine home directory")?;

    let store_path = home.join(".rpx/endpoints.json");
    let mut store = EndpointStore::load(&store_path)?;
    let endpoint_id = store
        .find_by_name_or_id(&args.endpoint)
        .map(|e| e.id.clone())
        .unwrap_or_else(|| args.endpoint.clone());

    if !args.yes {
        eprint!(
            "{} Destroy endpoint '{}'? [y/N] ",
            ui::error("!"),
            args.endpoint,
        );
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("{}", ui::dim("Cancelled."));
            return Ok(());
        }
    }

    let creds = Credentials::load(&home.join(".rpx/credentials.toml"))?;
    let api_key = creds
        .api_key_for_or_env(ProviderKind::RunPod)
        .ok_or("no RunPod API key. Run `rpx login` first.")?;

    let mut spinner = ui::InlineSpinner::new(&format!("Destroying {}...", args.endpoint));
    spinner.tick();

    let provider = RunPodProvider::new(api_key);
    provider.delete_endpoint(&endpoint_id).await?;

    store.remove(&args.endpoint);
    store.save(&store_path)?;

    spinner.finish(&format!("Endpoint '{}' destroyed", args.endpoint));
    Ok(())
}
