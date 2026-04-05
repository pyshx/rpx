use rpx_core::backend::BackendKind;
use rpx_core::catalog::GpuCatalog;
use rpx_core::config::RpxConfig;
use rpx_core::deploy;
use rpx_core::model;
use rpx_core::provider::ProviderKind;
use rpx_core::provider::runpod::RunPodProvider;

/// Test the full plan resolution pipeline against the real HuggingFace API.
/// No RunPod API key needed — this only resolves the plan, doesn't execute it.
#[tokio::test]
async fn resolve_plan_for_qwen_2_5_1_5b() {
    let client = reqwest::Client::new();
    let metadata = model::fetch_model_metadata(&client, "Qwen/Qwen2.5-1.5B-Instruct", None)
        .await
        .expect("should fetch model metadata from HuggingFace");

    assert_eq!(metadata.model_id, "Qwen/Qwen2.5-1.5B-Instruct");
    assert!(
        metadata.parameters_billions.is_some(),
        "HF should report param count via safetensors"
    );
    let params = metadata.parameters_billions.unwrap();
    assert!(
        (1.0..=2.0).contains(&params),
        "expected ~1.5B params, got {params}"
    );
    assert!(!metadata.gated, "Qwen2.5-1.5B should not be gated");

    // Now run the full plan resolution
    let config: RpxConfig = toml::from_str(r#"model = "Qwen/Qwen2.5-1.5B-Instruct""#).unwrap();
    let catalog = GpuCatalog::load_embedded().unwrap();
    let provider = RunPodProvider::new("fake-key".to_string());

    let plan = deploy::resolve_plan(&config, &metadata, &catalog, &provider)
        .expect("plan resolution should succeed");

    assert_eq!(plan.backend, BackendKind::Vllm);
    assert_eq!(plan.gpu.provider, ProviderKind::RunPod);
    // 1.5B at fp16 ≈ 3.9 GB VRAM — should pick cheapest GPU (>=4GB)
    assert!(
        plan.estimated_vram_gb < 10.0,
        "1.5B model should need < 10 GB VRAM, got {}",
        plan.estimated_vram_gb
    );
    assert!(
        plan.gpu.vram_gb as f64 >= plan.estimated_vram_gb,
        "selected GPU ({} GB) should have enough VRAM for {:.1} GB",
        plan.gpu.vram_gb,
        plan.estimated_vram_gb
    );

    eprintln!("Model:    {}", plan.model_id);
    eprintln!("Params:   {params:.2}B");
    eprintln!("Backend:  {}", plan.backend);
    eprintln!("VRAM:     {:.1} GB estimated", plan.estimated_vram_gb);
    eprintln!("GPU:      {} ({} GB)", plan.gpu.gpu_name, plan.gpu.vram_gb);
    eprintln!("Cost:     ${:.2}/hr", plan.gpu.price_per_hour);
    eprintln!("Provider: {}", plan.gpu.provider);
}

#[tokio::test]
async fn resolve_plan_for_llama_8b() {
    let client = reqwest::Client::new();
    let metadata = model::fetch_model_metadata(
        &client,
        "meta-llama/Llama-3.1-8B-Instruct",
        None,
    )
    .await;

    // Llama 3.1 is gated — if we don't have a token, we should get a gated error
    match metadata {
        Ok(m) => {
            // If HF returns metadata without auth (sometimes possible for config)
            let config: RpxConfig =
                toml::from_str(r#"model = "meta-llama/Llama-3.1-8B-Instruct""#).unwrap();
            let catalog = GpuCatalog::load_embedded().unwrap();
            let provider = RunPodProvider::new("fake-key".to_string());

            if let Some(params) = m.parameters_billions {
                let plan = deploy::resolve_plan(&config, &m, &catalog, &provider)
                    .expect("should resolve plan");
                eprintln!("Llama 8B: {params:.1}B, VRAM: {:.1} GB, GPU: {}", plan.estimated_vram_gb, plan.gpu.gpu_name);
                assert!(plan.estimated_vram_gb > 15.0, "8B model should need > 15 GB VRAM");
            }
        }
        Err(model::ModelError::Gated(_)) => {
            eprintln!("Llama 3.1 is gated (expected without HF_TOKEN)");
        }
        Err(e) => {
            panic!("unexpected error fetching Llama metadata: {e}");
        }
    }
}

#[tokio::test]
async fn name_estimation_fallback() {
    let client = reqwest::Client::new();
    // Use a model where safetensors info might be missing — fall back to name parsing
    let config: RpxConfig =
        toml::from_str(r#"model = "some-org/fictional-model-13B-chat""#).unwrap();
    let metadata = model::ModelMetadata {
        model_id: "some-org/fictional-model-13B-chat".to_string(),
        pipeline_tag: None,
        parameters_billions: None, // simulate HF not having safetensors info
        gated: false,
    };
    let catalog = GpuCatalog::load_embedded().unwrap();
    let provider = RunPodProvider::new("fake-key".to_string());

    let plan = deploy::resolve_plan(&config, &metadata, &catalog, &provider)
        .expect("should fall back to name-based estimation");

    // 13B parsed from name, fp16 → ~26 GB model + 30% overhead ≈ 33.8 GB
    assert!(
        plan.estimated_vram_gb > 25.0,
        "13B model should need > 25 GB, got {}",
        plan.estimated_vram_gb
    );
    eprintln!(
        "13B fallback: VRAM {:.1} GB, GPU: {} ({} GB)",
        plan.estimated_vram_gb, plan.gpu.gpu_name, plan.gpu.vram_gb
    );
}
