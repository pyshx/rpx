mod commands;
mod ui;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "rpx",
    about = "Deploy ML models to serverless GPUs — one command",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Store your provider API key
    Login(commands::login::LoginArgs),
    /// Deploy a model to a serverless GPU endpoint
    Deploy(commands::deploy::DeployArgs),
    /// Show endpoint status
    Status(commands::status::StatusArgs),
    /// List all managed endpoints
    List,
    /// Delete an endpoint
    Destroy(commands::destroy::DestroyArgs),
    /// Start a local OpenAI-compatible proxy for an endpoint
    Proxy(commands::proxy::ProxyArgs),
    /// Start the multi-model inference gateway
    Serve(commands::serve::ServeArgs),
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rpx=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Login(args) => commands::login::run(args).await,
        Commands::Deploy(args) => commands::deploy::run(args).await,
        Commands::Status(args) => commands::status::run(args).await,
        Commands::List => commands::list::run().await,
        Commands::Destroy(args) => commands::destroy::run(args).await,
        Commands::Proxy(args) => commands::proxy::run(args).await,
        Commands::Serve(args) => commands::serve::run(args).await,
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
