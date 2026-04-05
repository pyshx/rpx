pub mod sizing;

use serde::Deserialize;

/// Metadata about a HuggingFace model, fetched from the API.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_id: String,
    pub pipeline_tag: Option<String>,
    pub parameters_billions: Option<f64>,
    pub gated: bool,
}

/// Raw response from `https://huggingface.co/api/models/{id}`
#[derive(Debug, Deserialize)]
struct HfModelResponse {
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    safetensors: Option<SafetensorsInfo>,
    #[serde(default)]
    gated: HfGated,
}

#[derive(Debug, Deserialize)]
struct SafetensorsInfo {
    #[serde(default)]
    total: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
enum HfGated {
    Bool(bool),
    String(String),
    #[default]
    None,
}

impl HfGated {
    fn is_gated(&self) -> bool {
        match self {
            Self::Bool(b) => *b,
            Self::String(s) => s != "false",
            Self::None => false,
        }
    }
}

pub async fn fetch_model_metadata(
    client: &reqwest::Client,
    model_id: &str,
    hf_token: Option<&str>,
) -> Result<ModelMetadata, ModelError> {
    let url = format!("https://huggingface.co/api/models/{model_id}");
    let mut request = client.get(&url);
    if let Some(token) = hf_token {
        request = request.bearer_auth(token);
    }

    let response = request
        .send()
        .await
        .map_err(|e| ModelError::Network(e.to_string()))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(ModelError::NotFound(model_id.to_string()));
    }
    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(ModelError::Gated(model_id.to_string()));
    }
    if !response.status().is_success() {
        return Err(ModelError::Api {
            status: response.status().as_u16(),
            message: response.text().await.unwrap_or_default(),
        });
    }

    let hf: HfModelResponse = response
        .json()
        .await
        .map_err(|e| ModelError::Parse(e.to_string()))?;

    let parameters_billions = hf
        .safetensors
        .and_then(|s| s.total)
        .map(|total_params| total_params as f64 / 1_000_000_000.0);

    Ok(ModelMetadata {
        model_id: model_id.to_string(),
        pipeline_tag: hf.pipeline_tag,
        parameters_billions,
        gated: hf.gated.is_gated(),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model not found: {0}")]
    NotFound(String),

    #[error("model {0} is gated — set HF_TOKEN or `secrets.hf_token` in rpx.yaml")]
    Gated(String),

    #[error("HuggingFace API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("network error: {0}")]
    Network(String),

    #[error("failed to parse model metadata: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_gated_parsing() {
        let g = HfGated::Bool(true);
        assert!(g.is_gated());

        let g = HfGated::Bool(false);
        assert!(!g.is_gated());

        let g = HfGated::String("auto".to_string());
        assert!(g.is_gated());

        let g = HfGated::String("false".to_string());
        assert!(!g.is_gated());

        let g = HfGated::None;
        assert!(!g.is_gated());
    }
}
