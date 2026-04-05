use std::collections::HashMap;

use crate::config::ModelConfig;
use super::{Backend, BackendKind};

pub struct VllmBackend;

impl Backend for VllmBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vllm
    }

    fn default_image(&self) -> &str {
        "runpod/worker-v1-vllm:stable-cuda12.1.0"
    }

    fn env_vars(&self, model_id: &str, config: &ModelConfig) -> HashMap<String, String> {
        let mut vars = HashMap::new();
        vars.insert("MODEL_NAME".to_string(), model_id.to_string());
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

        // Pass through any extra backend_args as env vars (uppercased)
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
            _ => 2.0, // default to fp16
        };
        let model_gb = model_params_billions * bytes_per_param;
        // vLLM KV cache + runtime overhead: ~30% on top of model weights
        model_gb * 1.3
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
    fn vllm_is_openai_native() {
        assert!(VllmBackend.openai_native());
    }

    #[test]
    fn vllm_default_port() {
        assert_eq!(VllmBackend.default_port(), 8000);
    }

    #[test]
    fn vllm_env_vars_basic() {
        let config = make_config("float16");
        let vars = VllmBackend.env_vars("meta-llama/Llama-3.1-8B", &config);
        assert_eq!(vars["MODEL_NAME"], "meta-llama/Llama-3.1-8B");
        assert_eq!(vars["DTYPE"], "float16");
    }

    #[test]
    fn vllm_env_vars_with_overrides() {
        let config = ModelConfig {
            dtype: "auto".to_string(),
            max_model_len: Some(8192),
            gpu_memory_utilization: Some(0.9),
            tensor_parallel_size: Some(2),
            max_num_seqs: Some(256),
            extra: HashMap::new(),
        };
        let vars = VllmBackend.env_vars("test/model", &config);
        assert_eq!(vars["MAX_MODEL_LEN"], "8192");
        assert_eq!(vars["GPU_MEMORY_UTILIZATION"], "0.9");
        assert_eq!(vars["TENSOR_PARALLEL_SIZE"], "2");
        assert_eq!(vars["MAX_NUM_SEQS"], "256");
    }

    #[test]
    fn vllm_vram_estimate_fp16() {
        // 7B model at fp16 = 7 * 2 = 14 GB model + 30% overhead ≈ 18.2 GB
        let vram = VllmBackend.estimate_vram_gb(7.0, "float16");
        assert!((vram - 18.2).abs() < 0.1);
    }

    #[test]
    fn vllm_vram_estimate_fp32() {
        // 7B model at fp32 = 7 * 4 = 28 GB + 30% = 36.4 GB
        let vram = VllmBackend.estimate_vram_gb(7.0, "float32");
        assert!((vram - 36.4).abs() < 0.1);
    }

    #[test]
    fn vllm_vram_estimate_int4() {
        // 70B model at int4 = 70 * 0.5 = 35 GB + 30% = 45.5 GB
        let vram = VllmBackend.estimate_vram_gb(70.0, "int4");
        assert!((vram - 45.5).abs() < 0.1);
    }

    #[test]
    fn vllm_vram_unknown_dtype_defaults_to_fp16() {
        let vram_unknown = VllmBackend.estimate_vram_gb(7.0, "mystery");
        let vram_fp16 = VllmBackend.estimate_vram_gb(7.0, "float16");
        assert!((vram_unknown - vram_fp16).abs() < f64::EPSILON);
    }

    #[test]
    fn vllm_extra_args_passed_through() {
        let mut extra = HashMap::new();
        extra.insert("enforce_eager".to_string(), serde_json::json!(true));
        extra.insert("custom_opt".to_string(), serde_json::json!("value"));
        let config = ModelConfig {
            dtype: "auto".to_string(),
            max_model_len: None,
            gpu_memory_utilization: None,
            tensor_parallel_size: None,
            max_num_seqs: None,
            extra,
        };
        let vars = VllmBackend.env_vars("test/model", &config);
        assert_eq!(vars["ENFORCE_EAGER"], "true");
        assert_eq!(vars["CUSTOM_OPT"], "value");
    }
}
