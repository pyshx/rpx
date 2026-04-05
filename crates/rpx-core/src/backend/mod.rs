pub mod rvllm;
pub mod vllm;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Vllm,
    Rvllm,
    Tgi,
    LlamaCpp,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vllm => write!(f, "vLLM"),
            Self::Rvllm => write!(f, "rvLLM"),
            Self::Tgi => write!(f, "TGI"),
            Self::LlamaCpp => write!(f, "llama.cpp"),
        }
    }
}

impl BackendKind {
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "vllm" => Some(Self::Vllm),
            "rvllm" => Some(Self::Rvllm),
            "tgi" => Some(Self::Tgi),
            "llamacpp" | "llama.cpp" | "llama-cpp" => Some(Self::LlamaCpp),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    TextGeneration,
    Embedding,
    ImageGeneration,
    AudioTranscription,
}

pub trait Backend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn default_image(&self) -> &str;
    fn env_vars(&self, model_id: &str, config: &ModelConfig) -> HashMap<String, String>;
    fn estimate_vram_gb(&self, model_params_billions: f64, dtype: &str) -> f64;
    fn openai_native(&self) -> bool;
    fn default_port(&self) -> u16;
}

pub fn get_backend(kind: BackendKind) -> Result<Box<dyn Backend>, BackendError> {
    match kind {
        BackendKind::Vllm => Ok(Box::new(vllm::VllmBackend)),
        BackendKind::Rvllm => Ok(Box::new(rvllm::RvllmBackend)),
        BackendKind::Tgi => Err(BackendError::NotImplemented(kind)),
        BackendKind::LlamaCpp => Err(BackendError::NotImplemented(kind)),
    }
}

/// Select the best backend for a given model automatically.
/// Prefers vLLM as the most widely supported option.
pub fn auto_select_backend() -> BackendKind {
    BackendKind::Vllm
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend {0} is not yet implemented")]
    NotImplemented(BackendKind),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_display() {
        assert_eq!(BackendKind::Vllm.to_string(), "vLLM");
        assert_eq!(BackendKind::Rvllm.to_string(), "rvLLM");
        assert_eq!(BackendKind::Tgi.to_string(), "TGI");
        assert_eq!(BackendKind::LlamaCpp.to_string(), "llama.cpp");
    }

    #[test]
    fn backend_kind_from_str_loose() {
        assert_eq!(BackendKind::from_str_loose("vllm"), Some(BackendKind::Vllm));
        assert_eq!(BackendKind::from_str_loose("VLLM"), Some(BackendKind::Vllm));
        assert_eq!(BackendKind::from_str_loose("rvllm"), Some(BackendKind::Rvllm));
        assert_eq!(BackendKind::from_str_loose("llamacpp"), Some(BackendKind::LlamaCpp));
        assert_eq!(BackendKind::from_str_loose("llama.cpp"), Some(BackendKind::LlamaCpp));
        assert_eq!(BackendKind::from_str_loose("llama-cpp"), Some(BackendKind::LlamaCpp));
        assert_eq!(BackendKind::from_str_loose("unknown"), None);
    }

    #[test]
    fn get_backend_returns_implemented() {
        assert!(get_backend(BackendKind::Vllm).is_ok());
        assert!(get_backend(BackendKind::Rvllm).is_ok());
    }

    #[test]
    fn get_backend_errors_for_unimplemented() {
        assert!(get_backend(BackendKind::Tgi).is_err());
        assert!(get_backend(BackendKind::LlamaCpp).is_err());
    }

    #[test]
    fn auto_select_defaults_to_vllm() {
        assert_eq!(auto_select_backend(), BackendKind::Vllm);
    }
}
