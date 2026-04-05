use rpx_core::config::Credentials;
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;
use rpx_core::provider::Provider;

use crate::ui;

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let home = dirs_next::home_dir().ok_or("cannot determine home directory")?;
    let creds = Credentials::load(&home.join(".rpx/credentials.toml"))?;

    let api_key = creds
        .api_key_for_or_env(ProviderKind::RunPod)
        .ok_or("no RunPod API key. Run `rpx login` first.")?;

    let provider = RunPodProvider::new(api_key);
    let endpoints = provider.list_endpoints().await?;

    if endpoints.is_empty() {
        eprintln!("{}", ui::dim("No endpoints found."));
        return Ok(());
    }

    let columns = [("NAME", 22), ("ID", 16), ("STATUS", 14), ("GPU", 20)];
    eprintln!("{}", ui::table_header(&columns));

    for ep in &endpoints {
        let status_str = ep.status.to_string();
        eprintln!(
            "{}",
            ui::table_row(&[
                (&ep.name, 22),
                (&ep.id, 16),
                (&status_str, 14),
                (&ep.gpu_id, 20),
            ])
        );
    }

    Ok(())
}
