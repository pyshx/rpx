mod api;
mod invoke;

use async_trait::async_trait;

use crate::backend::BackendKind;
use super::{Provider, ProviderKind};
use super::types::*;

pub struct RunPodProvider {
    client: reqwest::Client,
    api_key: String,
    rest_base: String,
}

impl RunPodProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            rest_base: "https://rest.runpod.io/v1".to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_base_url(api_key: String, rest_base: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            rest_base,
        }
    }
}

#[async_trait]
impl Provider for RunPodProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::RunPod
    }

    async fn validate_auth(&self) -> Result<(), ProviderError> {
        api::validate_auth(&self.client, &self.api_key, &self.rest_base).await
    }

    async fn list_gpus(&self) -> Result<Vec<GpuSpec>, ProviderError> {
        Ok(vec![]) // RunPod doesn't expose GPU listing via REST
    }

    async fn create_endpoint(&self, config: &EndpointConfig) -> Result<Endpoint, ProviderError> {
        api::create_endpoint(&self.client, &self.api_key, &self.rest_base, config).await
    }

    async fn get_endpoint(&self, id: &str) -> Result<Endpoint, ProviderError> {
        api::get_endpoint(&self.client, &self.api_key, &self.rest_base, id).await
    }

    async fn update_endpoint(
        &self,
        id: &str,
        config: &EndpointConfig,
    ) -> Result<Endpoint, ProviderError> {
        api::update_endpoint(&self.client, &self.api_key, &self.rest_base, id, config).await
    }

    async fn delete_endpoint(&self, id: &str) -> Result<(), ProviderError> {
        api::delete_endpoint(&self.client, &self.api_key, &self.rest_base, id).await
    }

    async fn list_endpoints(&self) -> Result<Vec<Endpoint>, ProviderError> {
        api::list_endpoints(&self.client, &self.api_key, &self.rest_base).await
    }

    async fn invoke(
        &self,
        endpoint: &Endpoint,
        request: InvocationRequest,
    ) -> Result<InvocationResponse, ProviderError> {
        invoke::invoke(&self.client, &self.api_key, endpoint, request).await
    }

    fn supports_native_template(&self, _backend: &BackendKind) -> bool {
        // RunPod doesn't expose reusable template IDs we can hardcode.
        // We always create a new template with the docker image + env vars.
        false
    }

    fn native_template(&self, _backend: &BackendKind) -> Option<ImageSpec> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_native_templates() {
        let provider = RunPodProvider::new("test".to_string());
        assert!(!provider.supports_native_template(&BackendKind::Vllm));
        assert!(!provider.supports_native_template(&BackendKind::Rvllm));
        assert!(provider.native_template(&BackendKind::Vllm).is_none());
    }
}
