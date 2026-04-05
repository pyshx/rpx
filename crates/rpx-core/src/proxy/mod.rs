pub mod streaming;
pub mod translator;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Response, StatusCode, header},
    response::{IntoResponse, Json},
    routing::{get, post},
};
use futures::StreamExt;

use crate::provider::types::{Endpoint, InvocationResponse};
use crate::provider::Provider;

pub struct ProxyServer {
    endpoint: Endpoint,
    provider: Arc<dyn Provider>,
    port: u16,
}

struct ProxyState {
    endpoint: Endpoint,
    provider: Arc<dyn Provider>,
}

impl ProxyServer {
    pub fn new(endpoint: Endpoint, provider: Arc<dyn Provider>, port: u16) -> Self {
        Self {
            endpoint,
            provider,
            port,
        }
    }

    pub async fn run(&self) -> Result<(), ProxyError> {
        let state = Arc::new(ProxyState {
            endpoint: self.endpoint.clone(),
            provider: self.provider.clone(),
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(forward_request))
            .route("/v1/completions", post(forward_request))
            .route("/v1/embeddings", post(forward_request))
            .route("/v1/models", get(list_models))
            .route("/health", get(health))
            .with_state(state);

        let addr = SocketAddr::from(([127, 0, 0, 1], self.port));
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| ProxyError::Bind(e, self.port))?;

        tracing::info!("OpenAI proxy listening on http://{addr}");

        axum::serve(listener, app)
            .await
            .map_err(|e| ProxyError::Serve(e.to_string()))?;

        Ok(())
    }
}

/// Unified handler for all forwarded requests (chat/completions/embeddings).
/// Returns JSON for non-streaming, SSE for streaming.
async fn forward_request(
    State(state): State<Arc<ProxyState>>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response<Body>, (StatusCode, String)> {
    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let request = translator::to_invocation_request(body);

    let response = state
        .provider
        .invoke(&state.endpoint, request)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    match response {
        InvocationResponse::Complete(v) => {
            Ok(Json(v).into_response())
        }
        InvocationResponse::Stream(stream) => {
            if !is_stream {
                // Caller didn't ask for streaming but provider returned a stream.
                // Collect the full response.
                let mut collected = String::new();
                let mut pinned = stream;
                while let Some(chunk_result) = pinned.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            // Parse last SSE data line as the response
                            for line in text.lines() {
                                if let Some(data) = streaming::parse_sse_line(line) {
                                    collected = data;
                                }
                            }
                        }
                        Err(e) => {
                            return Err((StatusCode::BAD_GATEWAY, e.to_string()));
                        }
                    }
                }
                let json: serde_json::Value = serde_json::from_str(&collected)
                    .unwrap_or(serde_json::Value::Null);
                Ok(Json(json).into_response())
            } else {
                // Stream SSE back to the client
                let sse_stream = stream.map(|chunk_result| {
                    chunk_result.map_err(|e| std::io::Error::other(e.to_string()))
                });

                let body = Body::from_stream(sse_stream);

                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .header(header::CACHE_CONTROL, "no-cache")
                    .header(header::CONNECTION, "keep-alive")
                    .body(body)
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
            }
        }
    }
}

async fn list_models(
    State(state): State<Arc<ProxyState>>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.endpoint.name,
            "object": "model",
            "owned_by": "rpx",
        }]
    }))
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("failed to bind to port {1}: {0}")]
    Bind(std::io::Error, u16),

    #[error("proxy server error: {0}")]
    Serve(String),
}
