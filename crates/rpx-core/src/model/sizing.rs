use crate::backend::{self, BackendKind};

/// Estimate the VRAM needed for a model, given its parameter count and dtype.
/// Returns VRAM in GB. Uses the backend's estimate if available.
pub fn estimate_vram(
    params_billions: f64,
    dtype: &str,
    backend_kind: BackendKind,
) -> Result<f64, SizingError> {
    let backend = backend::get_backend(backend_kind)
        .map_err(|e| SizingError::BackendUnavailable(e.to_string()))?;
    Ok(backend.estimate_vram_gb(params_billions, dtype))
}

/// Fallback: estimate parameter count from model name heuristics.
/// Looks for patterns like "7B", "70B", "1.5B", "8b" in the model ID.
pub fn estimate_params_from_name(model_id: &str) -> Option<f64> {
    // Look for patterns like "7B", "70B", "1.5B", "0.5B" (case insensitive)
    let re_pattern = regex_lite::Regex::new(r"(?i)(\d+\.?\d*)b(?:\b|[^a-zA-Z])").ok()?;
    let captures = re_pattern.captures(model_id)?;
    let num_str = captures.get(1)?.as_str();
    num_str.parse::<f64>().ok()
}

#[derive(Debug, thiserror::Error)]
pub enum SizingError {
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_vram_for_vllm() {
        let vram = estimate_vram(7.0, "float16", BackendKind::Vllm).unwrap();
        // 7B * 2 bytes * 1.3 overhead = 18.2 GB
        assert!((vram - 18.2).abs() < 0.1);
    }

    #[test]
    fn estimate_vram_for_rvllm() {
        let vram = estimate_vram(7.0, "float16", BackendKind::Rvllm).unwrap();
        // 7B * 2 bytes * 1.2 overhead = 16.8 GB
        assert!((vram - 16.8).abs() < 0.1);
    }

    #[test]
    fn estimate_params_common_patterns() {
        assert_eq!(estimate_params_from_name("meta-llama/Llama-3.1-8B-Instruct"), Some(8.0));
        assert_eq!(estimate_params_from_name("Qwen/Qwen2.5-7B-Instruct"), Some(7.0));
        assert_eq!(estimate_params_from_name("Qwen/Qwen2.5-72B-Instruct"), Some(72.0));
        assert_eq!(estimate_params_from_name("Qwen/Qwen2.5-1.5B-Instruct"), Some(1.5));
        assert_eq!(estimate_params_from_name("Qwen/Qwen2.5-0.5B-Instruct"), Some(0.5));
    }

    #[test]
    fn estimate_params_no_match() {
        assert_eq!(estimate_params_from_name("bert-base-uncased"), None);
        assert_eq!(estimate_params_from_name("gpt2"), None);
    }

    #[test]
    fn estimate_params_case_insensitive() {
        assert_eq!(estimate_params_from_name("some-model-7b-chat"), Some(7.0));
        assert_eq!(estimate_params_from_name("some-model-7B-chat"), Some(7.0));
    }
}
