use std::collections::HashMap;
use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::backend::BackendKind;
use super::ProviderKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSpec {
    pub id: String,
    pub name: String,
    pub provider_gpu_id: String,
    pub vram_gb: u32,
    pub price_per_sec: f64,
    pub multi_gpu_max: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub min_workers: u32,
    pub max_workers: u32,
    pub idle_timeout_secs: u32,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_workers: 0,
            max_workers: 3,
            idle_timeout_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageSpec {
    NativeTemplate { template_id: String },
    PrebuiltImage { registry_url: String, tag: String },
    CustomImage { image_url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub name: String,
    pub model_id: String,
    pub backend: BackendKind,
    pub gpu: GpuSpec,
    pub gpu_count: u8,
    pub scaling: ScalingConfig,
    pub env_vars: HashMap<String, String>,
    pub image: ImageSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub id: String,
    pub name: String,
    pub provider: ProviderKind,
    pub status: EndpointStatus,
    pub gpu_id: String,
    pub invocation_url: String,
    pub openai_base_url: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EndpointStatus {
    Building,
    Initializing,
    Ready,
    Scaling,
    Idle,
    Error(String),
    Terminated,
}

impl std::fmt::Display for EndpointStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Building => write!(f, "Building"),
            Self::Initializing => write!(f, "Initializing"),
            Self::Ready => write!(f, "Ready"),
            Self::Scaling => write!(f, "Scaling"),
            Self::Idle => write!(f, "Idle"),
            Self::Error(e) => write!(f, "Error: {e}"),
            Self::Terminated => write!(f, "Terminated"),
        }
    }
}

#[derive(Debug)]
pub struct InvocationRequest {
    pub body: serde_json::Value,
    pub stream: bool,
    pub timeout_secs: u32,
}

pub type SseStream = Pin<Box<dyn Stream<Item = Result<bytes::Bytes, ProviderError>> + Send>>;

pub enum InvocationResponse {
    Complete(serde_json::Value),
    Stream(SseStream),
}

impl std::fmt::Debug for InvocationResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Complete(v) => f.debug_tuple("Complete").field(v).finish(),
            Self::Stream(_) => f.debug_tuple("Stream").field(&"..").finish(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("authentication failed: {0}")]
    Auth(String),

    #[error("endpoint not found: {0}")]
    NotFound(String),

    #[error("provider API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("not supported: {0}")]
    Unsupported(String),

    #[error("timeout: {0}")]
    Timeout(String),

    #[error("{0}")]
    Other(String),
}
