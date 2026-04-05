use std::path::PathBuf;

use clap::Args;
use rpx_core::fleet::FleetConfig;
use rpx_core::orchestrator::Orchestrator;

use crate::ui;

#[derive(Args)]
pub struct ServeArgs {
    /// Path to fleet config file
    #[arg(short, long, default_value = "rpx.yaml")]
    pub config: PathBuf,

    /// Only show what would be deployed, don't start
    #[arg(long)]
    pub dry_run: bool,
}

pub async fn run(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let config_content = std::fs::read_to_string(&args.config).map_err(|e| {
        format!(
            "failed to read {}: {e}",
            args.config.display()
        )
    })?;

    let config = FleetConfig::from_yaml(&config_content)?;

    eprintln!("{}", ui::title("rpx serve"));
    eprintln!(
        "  {} models, {} API keys",
        config.models.len(),
        config.api_keys.len(),
    );
    eprintln!();

    for (i, model) in config.models.iter().enumerate() {
        eprintln!(
            "  {} {} {} {}",
            ui::dim(&format!("{}.", i + 1)),
            model.display_name(),
            ui::dim(&format!("({})", model.id)),
            match model.tier {
                rpx_core::fleet::ModelTier::Hot => ui::success("hot"),
                rpx_core::fleet::ModelTier::Warm => ui::label("warm"),
                rpx_core::fleet::ModelTier::Cold => ui::dim("cold"),
            },
        );
    }
    eprintln!();

    if args.dry_run {
        eprintln!("{}", ui::dim("Dry run — not starting gateway."));
        return Ok(());
    }

    let home = dirs_next::home_dir().ok_or("cannot determine home directory")?;
    let store_path = home.join(".rpx/fleet_state.json");
    let credentials_path = home.join(".rpx/credentials.toml");

    let orchestrator = Orchestrator::init(config.clone(), store_path, credentials_path).await?;

    eprintln!(
        "{} Gateway on http://{}:{}",
        ui::success("✓"),
        config.gateway.host,
        config.gateway.port,
    );

    orchestrator.run().await?;

    Ok(())
}
