use std::path::PathBuf;

use clap::Args;
use rpx_core::config::{Credentials, ProviderCredential};
use rpx_core::provider::ProviderKind;

use crate::ui;

#[derive(Args)]
pub struct LoginArgs {
    /// Provider to authenticate with
    #[arg(default_value = "runpod")]
    pub provider: String,

    /// API key (if not provided, reads from stdin)
    #[arg(long)]
    pub api_key: Option<String>,
}

pub async fn run(args: LoginArgs) -> Result<(), Box<dyn std::error::Error>> {
    let provider = match args.provider.to_lowercase().as_str() {
        "runpod" => ProviderKind::RunPod,
        "vastai" | "vast.ai" => ProviderKind::VastAi,
        "beam" => ProviderKind::Beam,
        other => return Err(format!("unknown provider: {other}").into()),
    };

    let api_key = match args.api_key {
        Some(key) => key,
        None => {
            eprint!("{} Enter {} API key: ", ui::label("?"), provider);
            let mut key = String::new();
            std::io::stdin().read_line(&mut key)?;
            key.trim().to_string()
        }
    };

    if api_key.is_empty() {
        return Err("API key cannot be empty".into());
    }

    let creds_path = credentials_path()?;
    let mut creds = Credentials::load(&creds_path)?;

    let cred = ProviderCredential {
        api_key: api_key.clone(),
    };
    match provider {
        ProviderKind::RunPod => creds.runpod = Some(cred),
        ProviderKind::VastAi => creds.vastai = Some(cred),
        ProviderKind::Beam => creds.beam = Some(cred),
    }

    creds.save(&creds_path)?;
    eprintln!(
        "{} Saved {} credentials to {}",
        ui::success("✓"),
        provider,
        ui::dim(&creds_path.display().to_string()),
    );
    Ok(())
}

fn credentials_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let home = dirs_next::home_dir().ok_or("cannot determine home directory")?;
    Ok(home.join(".rpx").join("credentials.toml"))
}
