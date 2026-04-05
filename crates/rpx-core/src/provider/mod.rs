pub mod runpod;
pub mod types;

use async_trait::async_trait;
use types::*;

use crate::backend::BackendKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProviderKind {
    RunPod,
    VastAi,
    Beam,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RunPod => write!(f, "RunPod"),
            Self::VastAi => write!(f, "Vast.ai"),
            Self::Beam => write!(f, "Beam"),
        }
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn kind(&self) -> ProviderKind;

    async fn validate_auth(&self) -> Result<(), ProviderError>;

    async fn list_gpus(&self) -> Result<Vec<GpuSpec>, ProviderError>;

    async fn create_endpoint(&self, config: &EndpointConfig) -> Result<Endpoint, ProviderError>;

    async fn get_endpoint(&self, id: &str) -> Result<Endpoint, ProviderError>;

    async fn update_endpoint(
        &self,
        id: &str,
        config: &EndpointConfig,
    ) -> Result<Endpoint, ProviderError>;

    async fn delete_endpoint(&self, id: &str) -> Result<(), ProviderError>;

    async fn list_endpoints(&self) -> Result<Vec<Endpoint>, ProviderError>;

    async fn invoke(
        &self,
        endpoint: &Endpoint,
        request: InvocationRequest,
    ) -> Result<InvocationResponse, ProviderError>;

    fn supports_native_template(&self, backend: &BackendKind) -> bool;

    fn native_template(&self, backend: &BackendKind) -> Option<ImageSpec>;
}
