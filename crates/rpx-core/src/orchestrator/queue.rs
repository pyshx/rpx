use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use crate::provider::types::{InvocationRequest, InvocationResponse, ProviderError};

/// A channel-based request queue for models that are deploying.
/// Requests wait until the model is ready, then get forwarded.
pub struct RequestQueue {
    sender: mpsc::Sender<QueuedRequest>,
    receiver: tokio::sync::Mutex<mpsc::Receiver<QueuedRequest>>,
    timeout: Duration,
}

pub struct QueuedRequest {
    pub request: InvocationRequest,
    pub response_tx: oneshot::Sender<Result<InvocationResponse, ProviderError>>,
}

impl RequestQueue {
    pub fn new(capacity: usize, timeout: Duration) -> Self {
        let (sender, receiver) = mpsc::channel(capacity);
        Self {
            sender,
            receiver: tokio::sync::Mutex::new(receiver),
            timeout,
        }
    }

    /// Enqueue a request. Returns a future that resolves when the request is processed.
    /// Returns None if the queue is full.
    pub async fn enqueue(
        &self,
        request: InvocationRequest,
    ) -> Option<oneshot::Receiver<Result<InvocationResponse, ProviderError>>> {
        let (response_tx, response_rx) = oneshot::channel();
        let queued = QueuedRequest {
            request,
            response_tx,
        };
        self.sender.try_send(queued).ok()?;
        Some(response_rx)
    }

    /// Drain all queued requests (called when model becomes ready).
    pub async fn drain(&self) -> Vec<QueuedRequest> {
        let mut receiver = self.receiver.lock().await;
        let mut requests = Vec::new();
        while let Ok(req) = receiver.try_recv() {
            requests.push(req);
        }
        requests
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_and_drain() {
        let queue = RequestQueue::new(10, Duration::from_secs(30));

        let body = serde_json::json!({"model": "test", "messages": []});
        let req = InvocationRequest {
            body,
            stream: false,
            timeout_secs: 60,
        };
        let rx = queue.enqueue(req).await;
        assert!(rx.is_some());

        let drained = queue.drain().await;
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].request.body["model"], "test");
    }

    #[tokio::test]
    async fn full_queue_rejects() {
        let queue = RequestQueue::new(1, Duration::from_secs(30));

        let req1 = InvocationRequest {
            body: serde_json::json!({}),
            stream: false,
            timeout_secs: 60,
        };
        let req2 = InvocationRequest {
            body: serde_json::json!({}),
            stream: false,
            timeout_secs: 60,
        };

        assert!(queue.enqueue(req1).await.is_some());
        assert!(queue.enqueue(req2).await.is_none()); // full
    }

    #[tokio::test]
    async fn drain_empty_queue() {
        let queue = RequestQueue::new(10, Duration::from_secs(30));
        let drained = queue.drain().await;
        assert!(drained.is_empty());
    }
}
