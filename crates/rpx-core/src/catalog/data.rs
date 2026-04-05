use super::{GpuCatalog, CatalogError};

const EMBEDDED_CATALOG: &str = include_str!("../../../../catalog/gpus.toml");

pub fn load_embedded_catalog() -> Result<GpuCatalog, CatalogError> {
    Ok(toml::from_str(EMBEDDED_CATALOG)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_is_valid_toml() {
        let catalog = load_embedded_catalog().unwrap();
        assert!(!catalog.gpu.is_empty());
    }

    #[test]
    fn every_gpu_has_availability() {
        let catalog = load_embedded_catalog().unwrap();
        for gpu in &catalog.gpu {
            assert!(
                !gpu.availability.is_empty(),
                "GPU {} has no availability entries",
                gpu.id
            );
        }
    }

    #[test]
    fn every_gpu_has_positive_vram() {
        let catalog = load_embedded_catalog().unwrap();
        for gpu in &catalog.gpu {
            assert!(gpu.vram_gb > 0, "GPU {} has zero VRAM", gpu.id);
        }
    }

    #[test]
    fn prices_are_positive() {
        let catalog = load_embedded_catalog().unwrap();
        for gpu in &catalog.gpu {
            for avail in &gpu.availability {
                assert!(
                    avail.price_per_sec > 0.0,
                    "GPU {} on {} has non-positive price",
                    gpu.id,
                    avail.provider
                );
            }
        }
    }
}
