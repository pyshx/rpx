use std::collections::HashMap;

use crate::config::ModelConfig;
use super::{Backend, BackendKind};

pub struct RvllmBackend;

impl Backend for RvllmBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Rvllm
    }

    fn default_image(&self) -> &str {
        "pyshx/rvllm-runpod:latest"
    }

    fn env_vars(&self, model_id: &str, config: &ModelConfig) -> HashMap<String, String> {
        let mut vars = HashMap::new();
        vars.insert("MODEL_ID".to_string(), model_id.to_string());
        vars.insert("DTYPE".to_string(), config.dtype.clone());

        if let Some(len) = config.max_model_len {
            vars.insert("MAX_MODEL_LEN".to_string(), len.to_string());
        }
        if let Some(util) = config.gpu_memory_utilization {
            vars.insert("GPU_MEMORY_UTILIZATION".to_string(), util.to_string());
        }
        if let Some(tp) = config.tensor_parallel_size {
            vars.insert("TENSOR_PARALLEL_SIZE".to_string(), tp.to_string());
        }
        if let Some(seqs) = config.max_num_seqs {
            vars.insert("MAX_NUM_SEQS".to_string(), seqs.to_string());
        }

        for (key, value) in &config.extra {
            let env_key = key.to_uppercase();
            if let Some(s) = value.as_str() {
                vars.insert(env_key, s.to_string());
            } else {
                vars.insert(env_key, value.to_string());
            }
        }

        vars
    }

    fn estimate_vram_gb(&self, model_params_billions: f64, dtype: &str) -> f64 {
        let bytes_per_param = match dtype {
            "float32" | "fp32" => 4.0,
            "bfloat16" | "float16" | "fp16" | "half" | "auto" => 2.0,
            "int8" | "q8" => 1.0,
            "int4" | "q4" => 0.5,
            _ => 2.0,
        };
        let model_gb = model_params_billions * bytes_per_param;
        // rvLLM has lower overhead than vLLM (~20% for KV cache + runtime)
        model_gb * 1.2
    }

    fn openai_native(&self) -> bool {
        true
    }

    fn default_port(&self) -> u16 {
        8000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dtype: &str) -> ModelConfig {
        ModelConfig {
            dtype: dtype.to_string(),
            max_model_len: None,
            gpu_memory_utilization: None,
            tensor_parallel_size: None,
            max_num_seqs: None,
            extra: HashMap::new(),
        }
    }

    #[test]
    fn rvllm_uses_model_id_env_var() {
        let config = make_config("auto");
        let vars = RvllmBackend.env_vars("Qwen/Qwen2.5-7B", &config);
        assert_eq!(vars["MODEL_ID"], "Qwen/Qwen2.5-7B");
        assert!(!vars.contains_key("MODEL_NAME")); // that's vLLM's key
    }

    #[test]
    fn rvllm_lower_overhead_than_vllm() {
        use crate::backend::vllm::VllmBackend;
        let rvllm_vram = RvllmBackend.estimate_vram_gb(7.0, "float16");
        let vllm_vram = VllmBackend.estimate_vram_gb(7.0, "float16");
        assert!(rvllm_vram < vllm_vram, "rvLLM should have lower overhead");
    }

    #[test]
    fn rvllm_is_openai_native() {
        assert!(RvllmBackend.openai_native());
    }

    #[test]
    fn rvllm_vram_estimate_fp16() {
        // 7B * 2 bytes = 14 GB * 1.2 overhead = 16.8 GB
        let vram = RvllmBackend.estimate_vram_gb(7.0, "float16");
        assert!((vram - 16.8).abs() < 0.1);
    }
}
