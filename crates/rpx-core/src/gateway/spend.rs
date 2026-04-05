use std::collections::HashMap;

/// Tracks per-key usage and estimated cost.
pub struct SpendTracker {
    records: HashMap<String, KeySpend>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KeySpend {
    pub key_name: String,
    pub total_requests: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub estimated_cost_usd: f64,
}

impl SpendTracker {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Record usage from a completed request.
    /// Extracts token counts from the OpenAI `usage` field if present.
    pub fn record(
        &mut self,
        key_name: &str,
        response: &serde_json::Value,
        gpu_price_per_sec: f64,
    ) {
        let record = self.records.entry(key_name.to_string()).or_insert(KeySpend {
            key_name: key_name.to_string(),
            total_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            estimated_cost_usd: 0.0,
        });

        record.total_requests += 1;

        if let Some(usage) = response.get("usage") {
            let input = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            record.total_input_tokens += input;
            record.total_output_tokens += output;

            // Rough cost estimate: assume ~100 tokens/sec throughput on average
            let total_tokens = input + output;
            let estimated_seconds = total_tokens as f64 / 100.0;
            record.estimated_cost_usd += estimated_seconds * gpu_price_per_sec;
        }
    }

    /// Record that a request was made (without token-level detail).
    pub fn record_request(&mut self, key_name: &str) {
        let record = self.records.entry(key_name.to_string()).or_insert(KeySpend {
            key_name: key_name.to_string(),
            total_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            estimated_cost_usd: 0.0,
        });
        record.total_requests += 1;
    }

    pub fn get(&self, key_name: &str) -> Option<&KeySpend> {
        self.records.get(key_name)
    }

    pub fn all(&self) -> Vec<&KeySpend> {
        self.records.values().collect()
    }

    pub fn to_report(&self) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = self
            .records
            .values()
            .map(|s| serde_json::to_value(s).unwrap_or_default())
            .collect();

        serde_json::json!({
            "object": "spend_report",
            "data": entries,
        })
    }
}

impl Default for SpendTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_tracks_tokens() {
        let mut tracker = SpendTracker::new();
        let response = serde_json::json!({
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            }
        });

        tracker.record("my-app", &response, 0.000188);
        let spend = tracker.get("my-app").unwrap();
        assert_eq!(spend.total_requests, 1);
        assert_eq!(spend.total_input_tokens, 50);
        assert_eq!(spend.total_output_tokens, 100);
        assert!(spend.estimated_cost_usd > 0.0);
    }

    #[test]
    fn record_accumulates() {
        let mut tracker = SpendTracker::new();
        let response = serde_json::json!({
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        });

        tracker.record("app", &response, 0.0001);
        tracker.record("app", &response, 0.0001);

        let spend = tracker.get("app").unwrap();
        assert_eq!(spend.total_requests, 2);
        assert_eq!(spend.total_input_tokens, 20);
        assert_eq!(spend.total_output_tokens, 40);
    }

    #[test]
    fn record_without_usage_field() {
        let mut tracker = SpendTracker::new();
        let response = serde_json::json!({"id": "chatcmpl-1"});

        tracker.record("app", &response, 0.0001);
        let spend = tracker.get("app").unwrap();
        assert_eq!(spend.total_requests, 1);
        assert_eq!(spend.total_input_tokens, 0);
        assert_eq!(spend.estimated_cost_usd, 0.0);
    }

    #[test]
    fn report_format() {
        let mut tracker = SpendTracker::new();
        let response = serde_json::json!({
            "usage": {"prompt_tokens": 50, "completion_tokens": 100}
        });
        tracker.record("app-1", &response, 0.0001);

        let report = tracker.to_report();
        assert_eq!(report["object"], "spend_report");
        assert!(report["data"].as_array().unwrap().len() == 1);
    }

    #[test]
    fn unknown_key_returns_none() {
        let tracker = SpendTracker::new();
        assert!(tracker.get("nonexistent").is_none());
    }
}
