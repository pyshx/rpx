use crate::provider::types::InvocationRequest;

/// Convert an OpenAI-format request body into a provider InvocationRequest.
/// The body is passed through as-is for providers with OpenAI-native backends.
pub fn to_invocation_request(body: serde_json::Value) -> InvocationRequest {
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    InvocationRequest {
        body,
        stream,
        timeout_secs: 600,
    }
}

/// Rewrite the "model" field in a request body.
/// If the model matches `served_name` (or is absent), replace with `target`.
pub fn rewrite_request_model(
    body: &mut serde_json::Value,
    served_name: &str,
    target: &str,
) {
    if let Some(obj) = body.as_object_mut() {
        match obj.get("model").and_then(|v| v.as_str()) {
            Some(m) if m == served_name => {
                obj.insert("model".to_string(), serde_json::json!(target));
            }
            None => {
                obj.insert("model".to_string(), serde_json::json!(target));
            }
            _ => {} // preserve explicit foreign model
        }
    }
}

/// Recursively rewrite model names in a response payload.
/// Replaces `target` back to `served_name` in model fields.
pub fn rewrite_response_model(
    payload: &mut serde_json::Value,
    served_name: &str,
    target: &str,
) {
    match payload {
        serde_json::Value::Object(map) => {
            // Rewrite "model" field
            if let Some(model_val) = map.get("model") {
                if model_val.as_str() == Some(target) {
                    map.insert(
                        "model".to_string(),
                        serde_json::json!(served_name),
                    );
                }
            }
            // Rewrite "id" for model objects
            if map.get("object").and_then(|v| v.as_str()) == Some("model") {
                if let Some(id_val) = map.get("id") {
                    if id_val.as_str() == Some(target) {
                        map.insert(
                            "id".to_string(),
                            serde_json::json!(served_name),
                        );
                    }
                }
            }
            // Recurse into values
            for value in map.values_mut() {
                rewrite_response_model(value, served_name, target);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                rewrite_response_model(item, served_name, target);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_streaming_request() {
        let body = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });
        let req = to_invocation_request(body);
        assert!(req.stream);
    }

    #[test]
    fn non_streaming_by_default() {
        let body = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let req = to_invocation_request(body);
        assert!(!req.stream);
    }

    #[test]
    fn body_passed_through() {
        let body = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
        });
        let req = to_invocation_request(body.clone());
        assert_eq!(req.body, body);
    }

    #[test]
    fn rewrite_request_model_matching() {
        let mut body = serde_json::json!({
            "model": "my-llama",
            "messages": [],
        });
        rewrite_request_model(&mut body, "my-llama", "meta-llama/Llama-3.1-8B");
        assert_eq!(body["model"], "meta-llama/Llama-3.1-8B");
    }

    #[test]
    fn rewrite_request_model_missing() {
        let mut body = serde_json::json!({ "messages": [] });
        rewrite_request_model(&mut body, "my-llama", "meta-llama/Llama-3.1-8B");
        assert_eq!(body["model"], "meta-llama/Llama-3.1-8B");
    }

    #[test]
    fn rewrite_request_model_foreign_preserved() {
        let mut body = serde_json::json!({
            "model": "some-other-model",
            "messages": [],
        });
        rewrite_request_model(&mut body, "my-llama", "meta-llama/Llama-3.1-8B");
        assert_eq!(body["model"], "some-other-model");
    }

    #[test]
    fn rewrite_response_model_basic() {
        let mut resp = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "meta-llama/Llama-3.1-8B",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
            }],
        });
        rewrite_response_model(&mut resp, "my-llama", "meta-llama/Llama-3.1-8B");
        assert_eq!(resp["model"], "my-llama");
    }

    #[test]
    fn rewrite_response_model_nested() {
        let mut resp = serde_json::json!({
            "object": "list",
            "data": [{
                "object": "model",
                "id": "meta-llama/Llama-3.1-8B",
                "model": "meta-llama/Llama-3.1-8B",
            }],
        });
        rewrite_response_model(&mut resp, "my-llama", "meta-llama/Llama-3.1-8B");
        assert_eq!(resp["data"][0]["id"], "my-llama");
        assert_eq!(resp["data"][0]["model"], "my-llama");
    }

    #[test]
    fn rewrite_response_model_no_match() {
        let mut resp = serde_json::json!({
            "model": "other-model",
            "text": "hello",
        });
        rewrite_response_model(&mut resp, "my-llama", "meta-llama/Llama-3.1-8B");
        assert_eq!(resp["model"], "other-model");
    }
}
