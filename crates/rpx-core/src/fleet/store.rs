use std::path::PathBuf;

use super::state::PersistedFleetState;

pub struct FleetStore {
    path: PathBuf,
}

impl FleetStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn load(&self) -> Result<PersistedFleetState, FleetStoreError> {
        if !self.path.exists() {
            return Ok(PersistedFleetState::default());
        }
        let content = std::fs::read_to_string(&self.path)
            .map_err(|e| FleetStoreError::Io(e, self.path.clone()))?;
        serde_json::from_str(&content)
            .map_err(|e| FleetStoreError::Parse(e, self.path.clone()))
    }

    pub fn save(&self, state: &PersistedFleetState) -> Result<(), FleetStoreError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| FleetStoreError::Io(e, parent.to_path_buf()))?;
        }
        let content = serde_json::to_string_pretty(state)
            .map_err(FleetStoreError::Serialize)?;
        std::fs::write(&self.path, content)
            .map_err(|e| FleetStoreError::Io(e, self.path.clone()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FleetStoreError {
    #[error("failed to read {1}: {0}")]
    Io(std::io::Error, PathBuf),
    #[error("failed to parse {1}: {0}")]
    Parse(serde_json::Error, PathBuf),
    #[error("failed to serialize fleet state: {0}")]
    Serialize(serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fleet::state::PersistedModelState;
    use std::collections::HashMap;

    #[test]
    fn load_returns_default_when_missing() {
        let store = FleetStore::new(PathBuf::from("/nonexistent/fleet.json"));
        let state = store.load().unwrap();
        assert!(state.models.is_empty());
    }

    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = FleetStore::new(dir.path().join("fleet_state.json"));

        let mut models = HashMap::new();
        models.insert(
            "test-model".to_string(),
            PersistedModelState {
                endpoint_id: Some("ep-123".to_string()),
                state: "warm".to_string(),
                last_request: Some("2026-04-05T12:00:00Z".to_string()),
                error_message: None,
                retry_count: None,
            },
        );
        let state = PersistedFleetState { models };

        store.save(&state).unwrap();
        let loaded = store.load().unwrap();
        assert_eq!(loaded.models.len(), 1);
        assert_eq!(loaded.models["test-model"].state, "warm");
        assert_eq!(
            loaded.models["test-model"].endpoint_id,
            Some("ep-123".to_string())
        );
    }

    #[test]
    fn creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let store = FleetStore::new(dir.path().join("deep/nested/fleet.json"));
        let state = PersistedFleetState::default();
        store.save(&state).unwrap();
        assert!(dir.path().join("deep/nested/fleet.json").exists());
    }
}
