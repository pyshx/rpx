pub mod auth;
pub mod router;
pub mod spend;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, Response, StatusCode, header},
    middleware::{self, Next},
    response::Json,
    routing::{get, post},
};
use tokio::sync::RwLock;

use crate::fleet::FleetConfig;
use crate::fleet::state::FleetState;
use crate::orchestrator::model_manager::ModelManager;
use crate::orchestrator::queue::RequestQueue;
use crate::provider::Provider;

use auth::AuthLayer;
use router::ModelRouter;

pub struct GatewayServer {
    config: FleetConfig,
    state: Arc<RwLock<FleetState>>,
    provider: Arc<dyn Provider>,
    model_manager: Option<Arc<ModelManager>>,
}

pub struct GatewayState {
    pub fleet: Arc<RwLock<FleetState>>,
    pub provider: Arc<dyn Provider>,
    pub auth: RwLock<AuthLayer>,
    pub router: Arc<ModelRouter>,
    pub config: FleetConfig,
    pub spend: RwLock<spend::SpendTracker>,
    pub model_manager: Option<Arc<ModelManager>>,
    pub request_queues: HashMap<String, Arc<RequestQueue>>,
}

impl GatewayServer {
    pub fn new(
        config: FleetConfig,
        state: Arc<RwLock<FleetState>>,
        provider: Arc<dyn Provider>,
    ) -> Self {
        Self {
            config,
            state,
            provider,
            model_manager: None,
        }
    }

    pub fn with_model_manager(mut self, mm: Arc<ModelManager>) -> Self {
        self.model_manager = Some(mm);
        self
    }

    pub async fn run(&self) -> Result<(), GatewayError> {
        // Create per-model request queues for cold models
        let request_queues: HashMap<String, Arc<RequestQueue>> = self
            .config
            .models
            .iter()
            .map(|m| {
                (
                    m.display_name(),
                    Arc::new(RequestQueue::new(50, Duration::from_secs(120))),
                )
            })
            .collect();

        let state = Arc::new(GatewayState {
            fleet: self.state.clone(),
            provider: self.provider.clone(),
            auth: RwLock::new(AuthLayer::new(&self.config.api_keys)),
            router: Arc::new(ModelRouter::new(&self.config)),
            config: self.config.clone(),
            spend: RwLock::new(spend::SpendTracker::new()),
            model_manager: self.model_manager.clone(),
            request_queues,
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(handle_completion))
            .route("/v1/completions", post(handle_completion))
            .route("/v1/embeddings", post(handle_completion))
            .route("/v1/models", get(handle_list_models))
            .route("/v1/rpx/fleet", get(handle_fleet_status))
            .route("/v1/rpx/spend", get(handle_spend_report))
            .route("/health", get(health))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state);

        let addr: SocketAddr = format!("{}:{}", self.config.gateway.host, self.config.gateway.port)
            .parse()
            .map_err(|e| GatewayError::Config(format!("invalid address: {e}")))?;

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| GatewayError::Bind(e, self.config.gateway.port))?;

        tracing::info!("rpx gateway listening on http://{addr}");

        let shutdown = async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown signal received");
        };

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(|e| GatewayError::Serve(e.to_string()))?;

        Ok(())
    }
}

async fn auth_middleware(
    State(state): State<Arc<GatewayState>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response<Body>, (StatusCode, String)> {
    if state.auth.read().await.is_open() {
        return Ok(next.run(request).await);
    }

    if request.uri().path() == "/health"
        || request.uri().path().starts_with("/v1/rpx/")
    {
        return Ok(next.run(request).await);
    }

    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let key = auth_header.ok_or((
        StatusCode::UNAUTHORIZED,
        r#"{"error":{"message":"missing Authorization header","type":"auth_error"}}"#.to_string(),
    ))?;

    let mut auth = state.auth.write().await;
    auth.validate(key).map_err(|e| {
        let status = match &e {
            auth::AuthError::InvalidKey => StatusCode::UNAUTHORIZED,
            auth::AuthError::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            auth::AuthError::BudgetExceeded { .. } => StatusCode::PAYMENT_REQUIRED,
        };
        (
            status,
            format!(r#"{{"error":{{"message":"{e}","type":"auth_error"}}}}"#),
        )
    })?;

    // Check budget
    let spend = state.spend.read().await;
    let key_name = auth.key_name(key).unwrap_or("");
    if let Some(spend_record) = spend.get(key_name) {
        let current_spend = spend_record.estimated_cost_usd;
        drop(spend);
        auth.check_budget(key, current_spend).map_err(|e| {
            (
                StatusCode::PAYMENT_REQUIRED,
                format!(r#"{{"error":{{"message":"{e}","type":"budget_exceeded"}}}}"#),
            )
        })?;
    }

    // Store key name in request extensions for downstream handlers
    let key_name = auth.key_name(key).unwrap_or("").to_string();
    drop(auth);

    let mut request = request;
    request.extensions_mut().insert(AuthKeyName(key_name));

    Ok(next.run(request).await)
}

#[derive(Clone)]
struct AuthKeyName(String);

async fn handle_completion(
    State(state): State<Arc<GatewayState>>,
    key_name: Option<axum::extract::Extension<AuthKeyName>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response<Body>, (StatusCode, String)> {
    let model_alias = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or((
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"missing 'model' field","type":"invalid_request"}}"#
                .to_string(),
        ))?
        .to_string();

    let kn = key_name.map(|k| k.0.0.clone());

    state
        .router
        .forward(
            &model_alias,
            body,
            &state.fleet,
            &state.provider,
            &state.model_manager,
            &state.config,
            &state.request_queues,
            &state.spend,
            kn.as_deref(),
        )
        .await
}

async fn handle_list_models(
    State(state): State<Arc<GatewayState>>,
) -> Json<serde_json::Value> {
    let models: Vec<serde_json::Value> = state
        .config
        .models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.display_name(),
                "object": "model",
                "owned_by": "rpx",
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": models,
    }))
}

async fn handle_fleet_status(
    State(state): State<Arc<GatewayState>>,
) -> Json<serde_json::Value> {
    let fleet = state.fleet.read().await;
    let models: Vec<serde_json::Value> = state
        .config
        .models
        .iter()
        .map(|entry| {
            let name = entry.display_name();
            let model_state = fleet.get(&name);
            let (status, endpoint_id) = match model_state {
                Some(s) => (
                    s.status_str().to_string(),
                    s.endpoint().map(|e| e.id.clone()),
                ),
                None => ("unknown".to_string(), None),
            };
            serde_json::json!({
                "name": name,
                "model_id": entry.id,
                "tier": entry.tier.to_string(),
                "status": status,
                "endpoint_id": endpoint_id,
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "fleet_status",
        "models": models,
    }))
}

async fn handle_spend_report(
    State(state): State<Arc<GatewayState>>,
) -> Json<serde_json::Value> {
    let spend = state.spend.read().await;
    Json(spend.to_report())
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("config error: {0}")]
    Config(String),
    #[error("failed to bind to port {1}: {0}")]
    Bind(std::io::Error, u16),
    #[error("gateway error: {0}")]
    Serve(String),
}
