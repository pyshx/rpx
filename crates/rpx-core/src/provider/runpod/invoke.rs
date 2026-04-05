use reqwest::Client;

use crate::provider::types::*;

const INVOCATION_BASE: &str = "https://api.runpod.ai/v2";

pub async fn invoke(
    client: &Client,
    api_key: &str,
    endpoint: &Endpoint,
    request: InvocationRequest,
) -> Result<InvocationResponse, ProviderError> {
    if request.stream {
        invoke_streaming(client, api_key, endpoint, request).await
    } else {
        invoke_sync(client, api_key, endpoint, request).await
    }
}

async fn invoke_sync(
    client: &Client,
    api_key: &str,
    endpoint: &Endpoint,
    request: InvocationRequest,
) -> Result<InvocationResponse, ProviderError> {
    let url = format!("{INVOCATION_BASE}/{}/runsync", endpoint.id);

    let body = serde_json::json!({
        "input": request.body,
    });

    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .timeout(std::time::Duration::from_secs(request.timeout_secs as u64))
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let result: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse runsync response: {e}")))?;

    // RunPod wraps the response in {"id": ..., "status": ..., "output": ...}
    let output = result
        .get("output")
        .cloned()
        .unwrap_or(result);

    Ok(InvocationResponse::Complete(output))
}

async fn invoke_streaming(
    client: &Client,
    api_key: &str,
    endpoint: &Endpoint,
    request: InvocationRequest,
) -> Result<InvocationResponse, ProviderError> {
    // Submit async job
    let run_url = format!("{INVOCATION_BASE}/{}/run", endpoint.id);
    let body = serde_json::json!({
        "input": request.body,
    });

    let resp = client
        .post(&run_url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let run_result: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse run response: {e}")))?;

    let job_id = run_result["id"]
        .as_str()
        .ok_or_else(|| ProviderError::Other("missing job id in run response".to_string()))?
        .to_string();

    // Poll the stream endpoint
    let stream_url = format!("{INVOCATION_BASE}/{}/stream/{}", endpoint.id, job_id);
    let api_key_owned = api_key.to_string();
    let client_clone = client.clone();

    let stream = futures::stream::unfold(
        StreamState::Polling,
        move |state| {
            let url = stream_url.clone();
            let key = api_key_owned.clone();
            let client = client_clone.clone();

            async move {
                match state {
                    StreamState::Done => None,
                    StreamState::Polling => {
                        match poll_stream(&client, &key, &url).await {
                            Ok((chunks, done)) => {
                                let bytes = bytes::Bytes::from(
                                    chunks
                                        .into_iter()
                                        .map(|c| format!("data: {c}\n\n"))
                                        .collect::<String>(),
                                );
                                let next_state = if done {
                                    StreamState::Done
                                } else {
                                    StreamState::Polling
                                };
                                Some((Ok(bytes), next_state))
                            }
                            Err(e) => Some((Err(e), StreamState::Done)),
                        }
                    }
                }
            }
        },
    );

    Ok(InvocationResponse::Stream(Box::pin(stream)))
}

#[derive(Debug)]
enum StreamState {
    Polling,
    Done,
}

async fn poll_stream(
    client: &Client,
    api_key: &str,
    url: &str,
) -> Result<(Vec<String>, bool), ProviderError> {
    let resp = client
        .get(url)
        .bearer_auth(api_key)
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(ProviderError::Api {
            status: resp.status().as_u16(),
            message: resp.text().await.unwrap_or_default(),
        });
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ProviderError::Other(format!("failed to parse stream response: {e}")))?;

    let chunks: Vec<String> = body
        .get("stream")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| serde_json::to_string(v).ok())
                .collect()
        })
        .unwrap_or_default();

    let done = body
        .get("status")
        .and_then(|s| s.as_str())
        .map(|s| s == "COMPLETED")
        .unwrap_or(false);

    Ok((chunks, done))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_state_transitions() {
        // Just verify the enum exists and is constructable
        let _polling = StreamState::Polling;
        let _done = StreamState::Done;
    }
}
