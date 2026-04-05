pub mod data;

use serde::{Deserialize, Serialize};

use crate::provider::ProviderKind;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCatalog {
    pub gpu: Vec<GpuEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuEntry {
    pub id: String,
    pub name: String,
    pub vram_gb: u32,
    pub architecture: String,
    #[serde(default)]
    pub availability: Vec<GpuAvailability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAvailability {
    pub provider: String,
    pub provider_gpu_id: String,
    pub price_per_sec: f64,
    #[serde(default)]
    pub regions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GpuSelection {
    pub gpu_id: String,
    pub gpu_name: String,
    pub vram_gb: u32,
    pub provider: ProviderKind,
    pub provider_gpu_id: String,
    pub price_per_sec: f64,
    pub price_per_hour: f64,
}

impl GpuCatalog {
    pub fn load_embedded() -> Result<Self, CatalogError> {
        data::load_embedded_catalog()
    }

    /// Select the cheapest GPU that has enough VRAM for the given requirement.
    /// Optionally filter by provider and max hourly price.
    pub fn select_cheapest(
        &self,
        vram_needed_gb: f64,
        provider_filter: Option<ProviderKind>,
        max_price_per_hour: Option<f64>,
        gpu_count: u8,
    ) -> Result<GpuSelection, CatalogError> {
        let effective_vram_needed = vram_needed_gb / gpu_count as f64;

        let mut candidates: Vec<GpuSelection> = self
            .gpu
            .iter()
            .filter(|g| g.vram_gb as f64 >= effective_vram_needed)
            .flat_map(|g| {
                g.availability.iter().filter_map(move |a| {
                    let provider = match a.provider.as_str() {
                        "runpod" => ProviderKind::RunPod,
                        "vastai" => ProviderKind::VastAi,
                        "beam" => ProviderKind::Beam,
                        _ => return None,
                    };

                    if let Some(filter) = provider_filter {
                        if provider != filter {
                            return None;
                        }
                    }

                    let total_price_per_sec = a.price_per_sec * gpu_count as f64;
                    let price_per_hour = total_price_per_sec * 3600.0;

                    if let Some(max) = max_price_per_hour {
                        if price_per_hour > max {
                            return None;
                        }
                    }

                    Some(GpuSelection {
                        gpu_id: g.id.clone(),
                        gpu_name: g.name.clone(),
                        vram_gb: g.vram_gb,
                        provider,
                        provider_gpu_id: a.provider_gpu_id.clone(),
                        price_per_sec: total_price_per_sec,
                        price_per_hour,
                    })
                })
            })
            .collect();

        candidates.sort_by(|a, b| {
            a.price_per_sec
                .partial_cmp(&b.price_per_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
            .into_iter()
            .next()
            .ok_or(CatalogError::NoSuitableGpu {
                vram_needed_gb,
                provider_filter,
            })
    }

    /// Find a specific GPU by its catalog ID.
    pub fn find_gpu(&self, gpu_id: &str, provider: ProviderKind) -> Option<GpuSelection> {
        let provider_str = match provider {
            ProviderKind::RunPod => "runpod",
            ProviderKind::VastAi => "vastai",
            ProviderKind::Beam => "beam",
        };

        self.gpu.iter().find(|g| g.id == gpu_id).and_then(|g| {
            g.availability
                .iter()
                .find(|a| a.provider == provider_str)
                .map(|a| GpuSelection {
                    gpu_id: g.id.clone(),
                    gpu_name: g.name.clone(),
                    vram_gb: g.vram_gb,
                    provider,
                    provider_gpu_id: a.provider_gpu_id.clone(),
                    price_per_sec: a.price_per_sec,
                    price_per_hour: a.price_per_sec * 3600.0,
                })
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("failed to parse GPU catalog: {0}")]
    Parse(#[from] toml::de::Error),

    #[error(
        "no GPU found with >= {vram_needed_gb:.1} GB VRAM{}",
        provider_filter.map(|p| format!(" on {p}")).unwrap_or_default()
    )]
    NoSuitableGpu {
        vram_needed_gb: f64,
        provider_filter: Option<ProviderKind>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_catalog() -> GpuCatalog {
        GpuCatalog::load_embedded().expect("embedded catalog should parse")
    }

    #[test]
    fn embedded_catalog_loads() {
        let catalog = test_catalog();
        assert!(!catalog.gpu.is_empty(), "catalog should have GPUs");
    }

    #[test]
    fn catalog_has_expected_gpus() {
        let catalog = test_catalog();
        let ids: Vec<&str> = catalog.gpu.iter().map(|g| g.id.as_str()).collect();
        assert!(ids.contains(&"l4"), "should have L4");
        assert!(ids.contains(&"a100-80gb"), "should have A100 80GB");
    }

    #[test]
    fn select_cheapest_for_small_model() {
        let catalog = test_catalog();
        // 3B fp16 ≈ 8 GB VRAM needed
        let selection = catalog
            .select_cheapest(8.0, Some(ProviderKind::RunPod), None, 1)
            .unwrap();
        // Should pick cheapest GPU with >= 8 GB
        assert!(selection.vram_gb >= 8);
        assert_eq!(selection.provider, ProviderKind::RunPod);
    }

    #[test]
    fn select_cheapest_for_large_model() {
        let catalog = test_catalog();
        // 70B fp16 ≈ 182 GB VRAM needed — no single GPU has this
        let result = catalog.select_cheapest(182.0, Some(ProviderKind::RunPod), None, 1);
        assert!(result.is_err());
    }

    #[test]
    fn select_cheapest_with_multi_gpu() {
        let catalog = test_catalog();
        // 182 GB needed / 4 GPUs = 45.5 GB per GPU
        let selection = catalog
            .select_cheapest(182.0, Some(ProviderKind::RunPod), None, 4)
            .unwrap();
        assert!(selection.vram_gb as f64 >= 45.5);
    }

    #[test]
    fn select_cheapest_respects_price_cap() {
        let catalog = test_catalog();
        // Very low price cap should exclude expensive GPUs
        let result = catalog.select_cheapest(8.0, Some(ProviderKind::RunPod), Some(0.01), 1);
        // $0.01/hr is extremely low — might not find anything
        assert!(result.is_err() || result.unwrap().price_per_hour <= 0.01);
    }

    #[test]
    fn select_cheapest_sorts_by_price() {
        let catalog = test_catalog();
        let cheap = catalog
            .select_cheapest(8.0, Some(ProviderKind::RunPod), None, 1)
            .unwrap();
        let expensive = catalog
            .select_cheapest(50.0, Some(ProviderKind::RunPod), None, 1)
            .unwrap();
        assert!(cheap.price_per_sec <= expensive.price_per_sec);
    }

    #[test]
    fn find_gpu_by_id() {
        let catalog = test_catalog();
        let gpu = catalog.find_gpu("a100-80gb", ProviderKind::RunPod);
        assert!(gpu.is_some());
        let gpu = gpu.unwrap();
        assert_eq!(gpu.vram_gb, 80);
        assert_eq!(gpu.provider, ProviderKind::RunPod);
    }

    #[test]
    fn find_gpu_returns_none_for_unknown() {
        let catalog = test_catalog();
        assert!(catalog.find_gpu("nonexistent", ProviderKind::RunPod).is_none());
    }
}
