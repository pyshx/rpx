use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::provider::ProviderKind;
use crate::provider::types::*;

pub async fn validate_auth(client: &Client, api_key: &str, base: &str) -> Result<(), ProviderError> {
    let resp = client
        .get(format!("{base}/endpoints"))
        .bearer_auth(api_key)
        .send()
        .await?;

    if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err(ProviderError::Auth("invalid RunPod API key".to_string()));
    }
    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }
    Ok(())
}

// --- Template API (env vars + image go here) ---

#[derive(Debug, Serialize)]
struct CreateTemplateRequest {
    name: String,
    #[serde(rename = "imageName")]
    image_name: String,
    #[serde(rename = "isServerless")]
    is_serverless: bool,
    #[serde(rename = "containerDiskInGb")]
    container_disk_in_gb: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    env: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
struct TemplateResponse {
    id: String,
}

async fn create_template(
    client: &Client,
    api_key: &str,
    base: &str,
    name: &str,
    image: &str,
    env_vars: &std::collections::HashMap<String, String>,
) -> Result<String, ProviderError> {
    let body = CreateTemplateRequest {
        name: format!("rpx-{name}-{}", &uuid::Uuid::new_v4().to_string()[..8]),
        image_name: image.to_string(),
        is_serverless: true,
        container_disk_in_gb: 50,
        env: if env_vars.is_empty() {
            None
        } else {
            Some(env_vars.clone())
        },
    };

    let resp = client
        .post(format!("{base}/templates"))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let tmpl: TemplateResponse = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse template response: {e}")))?;

    Ok(tmpl.id)
}

// --- Endpoint API ---

#[derive(Debug, Serialize)]
struct CreateEndpointRequest {
    name: String,
    #[serde(rename = "templateId")]
    template_id: String,
    #[serde(rename = "gpuTypeIds")]
    gpu_type_ids: Vec<String>,
    #[serde(rename = "gpuCount")]
    gpu_count: u8,
    #[serde(rename = "workersMin")]
    workers_min: u32,
    #[serde(rename = "workersMax")]
    workers_max: u32,
    #[serde(rename = "idleTimeout")]
    idle_timeout: u32,
}

#[derive(Debug, Deserialize)]
struct RunPodEndpointResponse {
    id: String,
    name: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(rename = "gpuTypeIds", default)]
    gpu_type_ids: Option<Vec<String>>,
    #[serde(rename = "templateId", default)]
    #[allow(dead_code)]
    template_id: Option<String>,
}

pub async fn create_endpoint(
    client: &Client,
    api_key: &str,
    base: &str,
    config: &EndpointConfig,
) -> Result<Endpoint, ProviderError> {
    // Step 1: Resolve or create template
    let template_id = match &config.image {
        ImageSpec::NativeTemplate { template_id } => template_id.clone(),
        ImageSpec::PrebuiltImage { registry_url, tag } => {
            let image = format!("{registry_url}:{tag}");
            create_template(client, api_key, base, &config.name, &image, &config.env_vars).await?
        }
        ImageSpec::CustomImage { image_url } => {
            create_template(client, api_key, base, &config.name, image_url, &config.env_vars).await?
        }
    };

    // Step 2: Create endpoint with template
    let body = CreateEndpointRequest {
        name: config.name.clone(),
        template_id,
        gpu_type_ids: vec![config.gpu.provider_gpu_id.clone()],
        gpu_count: config.gpu_count,
        workers_min: config.scaling.min_workers,
        workers_max: config.scaling.max_workers,
        idle_timeout: config.scaling.idle_timeout_secs,
    };

    let resp = client
        .post(format!("{base}/endpoints"))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let rp: RunPodEndpointResponse = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse response: {e}")))?;

    Ok(to_endpoint(&rp, &config.gpu.provider_gpu_id))
}

pub async fn get_endpoint(
    client: &Client,
    api_key: &str,
    base: &str,
    id: &str,
) -> Result<Endpoint, ProviderError> {
    let resp = client
        .get(format!("{base}/endpoints/{id}"))
        .bearer_auth(api_key)
        .send()
        .await?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(ProviderError::NotFound(id.to_string()));
    }
    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let rp: RunPodEndpointResponse = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse response: {e}")))?;

    Ok(to_endpoint(&rp, ""))
}

pub async fn update_endpoint(
    client: &Client,
    api_key: &str,
    base: &str,
    id: &str,
    config: &EndpointConfig,
) -> Result<Endpoint, ProviderError> {
    let body = serde_json::json!({
        "workersMin": config.scaling.min_workers,
        "workersMax": config.scaling.max_workers,
        "idleTimeout": config.scaling.idle_timeout_secs,
    });

    let resp = client
        .patch(format!("{base}/endpoints/{id}"))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let rp: RunPodEndpointResponse = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse response: {e}")))?;

    Ok(to_endpoint(&rp, &config.gpu.provider_gpu_id))
}

pub async fn delete_endpoint(
    client: &Client,
    api_key: &str,
    base: &str,
    id: &str,
) -> Result<(), ProviderError> {
    let resp = client
        .delete(format!("{base}/endpoints/{id}"))
        .bearer_auth(api_key)
        .send()
        .await?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(ProviderError::NotFound(id.to_string()));
    }
    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }
    Ok(())
}

pub async fn list_endpoints(
    client: &Client,
    api_key: &str,
    base: &str,
) -> Result<Vec<Endpoint>, ProviderError> {
    let resp = client
        .get(format!("{base}/endpoints"))
        .bearer_auth(api_key)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let endpoints: Vec<RunPodEndpointResponse> = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse response: {e}")))?;

    Ok(endpoints.iter().map(|rp| to_endpoint(rp, "")).collect())
}

fn to_endpoint(rp: &RunPodEndpointResponse, gpu_id: &str) -> Endpoint {
    let status = match rp.status.as_deref() {
        Some("READY") | Some("ready") => EndpointStatus::Ready,
        Some("BUILDING") | Some("building") => EndpointStatus::Building,
        Some("INITIALIZING") | Some("initializing") => EndpointStatus::Initializing,
        Some("ERROR") | Some("error") => {
            EndpointStatus::Error("endpoint in error state".to_string())
        }
        _ => EndpointStatus::Idle,
    };

    let gpu_str = rp
        .gpu_type_ids
        .as_ref()
        .and_then(|ids| ids.first().cloned())
        .unwrap_or_else(|| gpu_id.to_string());

    Endpoint {
        id: rp.id.clone(),
        name: rp.name.clone().unwrap_or_else(|| rp.id.clone()),
        provider: ProviderKind::RunPod,
        status,
        gpu_id: gpu_str,
        invocation_url: format!("https://api.runpod.ai/v2/{}", rp.id),
        openai_base_url: Some(format!(
            "https://api.runpod.ai/v2/{}/openai/v1",
            rp.id
        )),
        created_at: chrono::Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers};

    #[tokio::test]
    async fn validate_auth_success() {
        let server = MockServer::start().await;
        Mock::given(matchers::method("GET"))
            .and(matchers::path("/endpoints"))
            .and(matchers::header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;

        let client = Client::new();
        let result = validate_auth(&client, "test-key", &server.uri()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn validate_auth_unauthorized() {
        let server = MockServer::start().await;
        Mock::given(matchers::method("GET"))
            .and(matchers::path("/endpoints"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = Client::new();
        let result = validate_auth(&client, "bad-key", &server.uri()).await;
        assert!(matches!(result, Err(ProviderError::Auth(_))));
    }

    #[tokio::test]
    async fn create_endpoint_with_template() {
        let server = MockServer::start().await;

        // Mock template creation
        Mock::given(matchers::method("POST"))
            .and(matchers::path("/templates"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "tmpl-abc123"
                })),
            )
            .mount(&server)
            .await;

        // Mock endpoint creation
        Mock::given(matchers::method("POST"))
            .and(matchers::path("/endpoints"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "ep-abc123",
                    "name": "my-endpoint",
                    "status": "INITIALIZING",
                    "gpuTypeIds": ["NVIDIA L4"]
                })),
            )
            .mount(&server)
            .await;

        let client = Client::new();
        let config = EndpointConfig {
            name: "my-endpoint".to_string(),
            model_id: "test/model".to_string(),
            backend: crate::backend::BackendKind::Vllm,
            gpu: GpuSpec {
                id: "l4".to_string(),
                name: "NVIDIA L4".to_string(),
                provider_gpu_id: "NVIDIA L4".to_string(),
                vram_gb: 24,
                price_per_sec: 0.000188,
                multi_gpu_max: 1,
            },
            gpu_count: 1,
            scaling: ScalingConfig::default(),
            env_vars: {
                let mut m = std::collections::HashMap::new();
                m.insert("MODEL_NAME".to_string(), "test/model".to_string());
                m
            },
            image: ImageSpec::PrebuiltImage {
                registry_url: "rpxai/vllm".to_string(),
                tag: "latest".to_string(),
            },
        };

        let endpoint = create_endpoint(&client, "test-key", &server.uri(), &config)
            .await
            .unwrap();
        assert_eq!(endpoint.id, "ep-abc123");
        assert_eq!(endpoint.name, "my-endpoint");
        assert_eq!(endpoint.status, EndpointStatus::Initializing);
    }

    #[tokio::test]
    async fn get_endpoint_success() {
        let server = MockServer::start().await;
        Mock::given(matchers::method("GET"))
            .and(matchers::path("/endpoints/ep-123"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "ep-123",
                    "name": "test-ep",
                    "status": "READY",
                    "gpuTypeIds": ["NVIDIA A100 80GB"]
                })),
            )
            .mount(&server)
            .await;

        let client = Client::new();
        let ep = get_endpoint(&client, "test-key", &server.uri(), "ep-123")
            .await
            .unwrap();
        assert_eq!(ep.status, EndpointStatus::Ready);
        assert_eq!(ep.gpu_id, "NVIDIA A100 80GB");
    }

    #[tokio::test]
    async fn get_endpoint_not_found() {
        let server = MockServer::start().await;
        Mock::given(matchers::method("GET"))
            .and(matchers::path("/endpoints/nonexistent"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = Client::new();
        let result = get_endpoint(&client, "test-key", &server.uri(), "nonexistent").await;
        assert!(matches!(result, Err(ProviderError::NotFound(_))));
    }

    #[tokio::test]
    async fn delete_endpoint_success() {
        let server = MockServer::start().await;
        Mock::given(matchers::method("DELETE"))
            .and(matchers::path("/endpoints/ep-del"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let client = Client::new();
        let result = delete_endpoint(&client, "test-key", &server.uri(), "ep-del").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn list_endpoints_success() {
        let server = MockServer::start().await;
        Mock::given(matchers::method("GET"))
            .and(matchers::path("/endpoints"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"id": "ep-1", "name": "first", "status": "READY"},
                    {"id": "ep-2", "name": "second", "status": "IDLE"},
                ])),
            )
            .mount(&server)
            .await;

        let client = Client::new();
        let endpoints = list_endpoints(&client, "test-key", &server.uri())
            .await
            .unwrap();
        assert_eq!(endpoints.len(), 2);
        assert_eq!(endpoints[0].name, "first");
        assert_eq!(endpoints[1].name, "second");
    }

    #[test]
    fn to_endpoint_maps_status() {
        let rp = RunPodEndpointResponse {
            id: "test-123".to_string(),
            name: Some("my-endpoint".to_string()),
            status: Some("READY".to_string()),
            gpu_type_ids: Some(vec!["NVIDIA L4".to_string()]),
            template_id: None,
        };
        let ep = to_endpoint(&rp, "");
        assert_eq!(ep.status, EndpointStatus::Ready);
        assert_eq!(ep.name, "my-endpoint");
        assert_eq!(ep.gpu_id, "NVIDIA L4");
        assert!(ep.invocation_url.contains("test-123"));
    }

    #[test]
    fn to_endpoint_defaults_to_idle() {
        let rp = RunPodEndpointResponse {
            id: "test".to_string(),
            name: None,
            status: None,
            gpu_type_ids: None,
            template_id: None,
        };
        let ep = to_endpoint(&rp, "fallback-gpu");
        assert_eq!(ep.status, EndpointStatus::Idle);
        assert_eq!(ep.name, "test");
        assert_eq!(ep.gpu_id, "fallback-gpu");
    }
}
