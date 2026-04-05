use std::collections::HashMap;
use std::time::Instant;

use crate::fleet::ApiKeyEntry;

pub struct AuthLayer {
    keys: HashMap<String, KeyState>,
}

struct KeyState {
    entry: ApiKeyEntry,
    bucket: TokenBucket,
}

struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl AuthLayer {
    pub fn new(api_keys: &[ApiKeyEntry]) -> Self {
        let keys = api_keys
            .iter()
            .map(|entry| {
                let rpm = entry.rate_limit_rpm.unwrap_or(600) as f64;
                let bucket = TokenBucket {
                    tokens: rpm,
                    max_tokens: rpm,
                    refill_rate: rpm / 60.0,
                    last_refill: Instant::now(),
                };
                (
                    entry.key.clone(),
                    KeyState {
                        entry: entry.clone(),
                        bucket,
                    },
                )
            })
            .collect();

        Self { keys }
    }

    /// Validate an API key and consume a rate limit token.
    /// Returns the key name on success.
    pub fn validate(&mut self, key: &str) -> Result<&str, AuthError> {
        let state = self
            .keys
            .get_mut(key)
            .ok_or(AuthError::InvalidKey)?;

        // Refill bucket
        let now = Instant::now();
        let elapsed = now.duration_since(state.bucket.last_refill).as_secs_f64();
        state.bucket.tokens =
            (state.bucket.tokens + elapsed * state.bucket.refill_rate).min(state.bucket.max_tokens);
        state.bucket.last_refill = now;

        // Consume token
        if state.bucket.tokens < 1.0 {
            return Err(AuthError::RateLimited);
        }
        state.bucket.tokens -= 1.0;

        Ok(&state.entry.name)
    }

    /// Check if a key has exceeded its budget.
    pub fn check_budget(&self, key: &str, current_spend_usd: f64) -> Result<(), AuthError> {
        let state = self.keys.get(key).ok_or(AuthError::InvalidKey)?;
        if let Some(budget) = state.entry.budget_usd {
            if current_spend_usd >= budget {
                return Err(AuthError::BudgetExceeded {
                    spent: current_spend_usd,
                    budget,
                });
            }
        }
        Ok(())
    }

    /// Check if a key is allowed to access a specific model.
    pub fn is_model_allowed(&self, key: &str, model_alias: &str) -> bool {
        let Some(state) = self.keys.get(key) else {
            return false;
        };
        match &state.entry.allowed_models {
            None => true, // no restriction
            Some(allowed) => allowed.iter().any(|m| m == model_alias),
        }
    }

    /// Get the name associated with an API key.
    pub fn key_name(&self, key: &str) -> Option<&str> {
        self.keys.get(key).map(|s| s.entry.name.as_str())
    }

    /// Check if no API keys are configured (open access mode).
    pub fn is_open(&self) -> bool {
        self.keys.is_empty()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("invalid API key")]
    InvalidKey,
    #[error("rate limit exceeded")]
    RateLimited,
    #[error("budget exceeded: ${spent:.2} of ${budget:.2}")]
    BudgetExceeded { spent: f64, budget: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keys() -> Vec<ApiKeyEntry> {
        vec![
            ApiKeyEntry {
                key: "sk-valid".to_string(),
                name: "test-app".to_string(),
                budget_usd: Some(100.0),
                rate_limit_rpm: Some(60),
                allowed_models: None,
            },
            ApiKeyEntry {
                key: "sk-restricted".to_string(),
                name: "restricted".to_string(),
                budget_usd: None,
                rate_limit_rpm: Some(10),
                allowed_models: Some(vec!["llama-8b".to_string()]),
            },
        ]
    }

    #[test]
    fn valid_key_succeeds() {
        let mut auth = AuthLayer::new(&test_keys());
        let name = auth.validate("sk-valid").unwrap();
        assert_eq!(name, "test-app");
    }

    #[test]
    fn invalid_key_fails() {
        let mut auth = AuthLayer::new(&test_keys());
        assert!(matches!(
            auth.validate("sk-invalid"),
            Err(AuthError::InvalidKey)
        ));
    }

    #[test]
    fn rate_limiting_works() {
        let keys = vec![ApiKeyEntry {
            key: "sk-limited".to_string(),
            name: "limited".to_string(),
            budget_usd: None,
            rate_limit_rpm: Some(2), // only 2 requests per minute
            allowed_models: None,
        }];
        let mut auth = AuthLayer::new(&keys);

        assert!(auth.validate("sk-limited").is_ok());
        assert!(auth.validate("sk-limited").is_ok());
        // Third request should fail (bucket started with 2 tokens)
        assert!(matches!(
            auth.validate("sk-limited"),
            Err(AuthError::RateLimited)
        ));
    }

    #[test]
    fn model_restriction_works() {
        let auth = AuthLayer::new(&test_keys());
        assert!(auth.is_model_allowed("sk-valid", "anything")); // no restriction
        assert!(auth.is_model_allowed("sk-restricted", "llama-8b")); // allowed
        assert!(!auth.is_model_allowed("sk-restricted", "qwen-72b")); // not allowed
    }

    #[test]
    fn open_access_when_no_keys() {
        let auth = AuthLayer::new(&[]);
        assert!(auth.is_open());
    }

    #[test]
    fn not_open_when_keys_exist() {
        let auth = AuthLayer::new(&test_keys());
        assert!(!auth.is_open());
    }

    #[test]
    fn budget_enforcement() {
        let auth = AuthLayer::new(&test_keys());
        // sk-valid has $100 budget
        assert!(auth.check_budget("sk-valid", 50.0).is_ok());
        assert!(auth.check_budget("sk-valid", 99.99).is_ok());
        assert!(matches!(
            auth.check_budget("sk-valid", 100.0),
            Err(AuthError::BudgetExceeded { .. })
        ));
        assert!(matches!(
            auth.check_budget("sk-valid", 150.0),
            Err(AuthError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn no_budget_means_unlimited() {
        let auth = AuthLayer::new(&test_keys());
        // sk-restricted has no budget
        assert!(auth.check_budget("sk-restricted", 999999.0).is_ok());
    }

    #[test]
    fn key_name_lookup() {
        let auth = AuthLayer::new(&test_keys());
        assert_eq!(auth.key_name("sk-valid"), Some("test-app"));
        assert_eq!(auth.key_name("sk-restricted"), Some("restricted"));
        assert_eq!(auth.key_name("sk-missing"), None);
    }
}
