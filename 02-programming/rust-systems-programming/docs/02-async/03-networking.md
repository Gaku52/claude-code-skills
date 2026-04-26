# Networking -- reqwest/hyper, WebSocket, tonic

> Master implementation patterns for HTTP clients/servers, WebSocket, and gRPC as Rust's asynchronous networking stack.

## What You Will Learn in This Chapter

1. **HTTP Client** -- REST API calls and connection pool management with reqwest
2. **HTTP Server (low-level)** -- Direct HTTP server implementation with hyper
3. **WebSocket** -- Real-time bidirectional communication with tokio-tungstenite
4. **gRPC** -- High-performance Protocol Buffers-based RPC with tonic
5. **DNS Resolution and TLS** -- Secure connections with trust-dns and rustls
6. **Connection Management** -- Connection pooling, retries, and timeout strategies


## Prerequisites

Reading the following before this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the contents of [Async Patterns -- Stream, Concurrency Limits, Retry](./02-async-patterns.md)

---

## 1. The Big Picture of the HTTP Stack

```
┌───────────────── Rust HTTP Ecosystem ────────────────────┐
│                                                          │
│  Application Layer                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  reqwest  │  │   Axum   │  │  tonic   │              │
│  │ (Client)  │  │ (Server) │  │ (gRPC)   │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │              │              │                    │
│  ┌────┴──────────────┴──────────────┴────┐              │
│  │              hyper (HTTP/1, HTTP/2)     │              │
│  └─────────────────┬─────────────────────┘              │
│                    │                                     │
│  ┌─────────────────┴─────────────────────┐              │
│  │          tokio (async I/O runtime)     │              │
│  └─────────────────┬─────────────────────┘              │
│                    │                                     │
│  ┌─────────────────┴─────────────────────┐              │
│  │  mio (epoll/kqueue/IOCP abstraction)   │              │
│  └───────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

### Role of Each Layer

| Layer | Crate | Role |
|---|---|---|
| Application | reqwest / Axum / tonic | High-level API. The layer developers normally interact with |
| HTTP Protocol | hyper | Parsing/generation of HTTP/1.1 and HTTP/2 |
| Async Runtime | tokio | Task scheduling, async I/O |
| OS Abstraction | mio | epoll (Linux) / kqueue (macOS) / IOCP (Windows) |
| TLS | rustls / native-tls | TLS 1.2/1.3 encryption |
| DNS | trust-dns / system resolver | Hostname resolution |

### HTTP Request Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│            Internal Flow of HTTP Request Processing         │
│                                                             │
│  1. DNS Resolution                                          │
│     app.example.com → 203.0.113.50                         │
│         │                                                   │
│  2. TCP Connection (fetch from pool or create new)          │
│         │                                                   │
│  3. TLS Handshake (for HTTPS)                               │
│     ┌──────────┐     ┌──────────┐                          │
│     │ Client   │────→│ Server   │  ClientHello             │
│     │          │←────│          │  ServerHello + Cert       │
│     │          │────→│          │  Key Exchange + Finished  │
│     │          │←────│          │  Finished                 │
│     └──────────┘     └──────────┘                          │
│         │                                                   │
│  4. Send HTTP Request                                       │
│     GET /api/users HTTP/1.1                                 │
│     Host: app.example.com                                   │
│     Authorization: Bearer xxx                               │
│         │                                                   │
│  5. Receive HTTP Response                                   │
│     HTTP/1.1 200 OK                                         │
│     Content-Type: application/json                          │
│     {"users": [...]}                                        │
│         │                                                   │
│  6. Return Connection to Pool (Keep-Alive)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. reqwest -- HTTP Client

### 2.1 Building and Configuring a Client

reqwest's `Client` holds a connection pool internally. It is a hard rule to share a single instance (or a small number per host group) across the entire application.

```rust
use reqwest::{Client, ClientBuilder, redirect::Policy};
use std::time::Duration;

/// Example of building a Client for production
fn build_production_client() -> reqwest::Result<Client> {
    ClientBuilder::new()
        // Timeout settings
        .timeout(Duration::from_secs(30))           // Overall request timeout
        .connect_timeout(Duration::from_secs(5))    // TCP connection establishment timeout
        .read_timeout(Duration::from_secs(15))      // Read timeout

        // Connection pool
        .pool_max_idle_per_host(20)                  // Max idle connections per host
        .pool_idle_timeout(Duration::from_secs(90))  // TTL for idle connections

        // TLS
        .min_tls_version(reqwest::tls::Version::TLS_1_2)
        .danger_accept_invalid_certs(false)          // Always false in production

        // Redirects
        .redirect(Policy::limited(10))               // Up to 10 redirects

        // Headers
        .user_agent("my-rust-app/1.0")
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert("Accept", "application/json".parse().unwrap());
            headers
        })

        // Compression
        .gzip(true)
        .brotli(true)
        .deflate(true)

        .build()
}
```

### 2.2 Basic HTTP Requests

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct User {
    id: u64,
    login: String,
    name: Option<String>,
    public_repos: Option<u32>,
}

#[derive(Serialize)]
struct CreateIssue {
    title: String,
    body: String,
    labels: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    data: T,
    total_count: Option<u64>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build the client (with connection pool -- should be reused)
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(10)
        .user_agent("my-rust-app/1.0")
        .build()?;

    // GET request + JSON deserialization
    let user: User = client
        .get("https://api.github.com/users/rust-lang")
        .header("Accept", "application/json")
        .send()
        .await?
        .error_for_status()?  // Convert 4xx/5xx into errors
        .json()
        .await?;
    println!("User: {} ({})", user.login, user.name.unwrap_or_default());

    // POST request
    let issue = CreateIssue {
        title: "Bug report".into(),
        body: "Description here".into(),
        labels: vec!["bug".into(), "triage".into()],
    };
    let response = client
        .post("https://api.example.com/issues")
        .bearer_auth("your-token")
        .json(&issue)
        .send()
        .await?;
    println!("Status: {}", response.status());

    // PUT request
    let response = client
        .put("https://api.example.com/users/1")
        .json(&serde_json::json!({
            "name": "Updated Name",
            "email": "new@example.com"
        }))
        .send()
        .await?;
    println!("PUT Status: {}", response.status());

    // DELETE request
    let response = client
        .delete("https://api.example.com/users/1")
        .bearer_auth("your-token")
        .send()
        .await?;
    println!("DELETE Status: {}", response.status());

    // PATCH request
    let response = client
        .patch("https://api.example.com/users/1")
        .json(&serde_json::json!({"name": "Patched"}))
        .send()
        .await?;
    println!("PATCH Status: {}", response.status());

    Ok(())
}
```

### 2.3 Detailed Response Handling

```rust
use reqwest::{Client, StatusCode};
use std::collections::HashMap;

async fn inspect_response(client: &Client, url: &str) -> anyhow::Result<()> {
    let response = client.get(url).send().await?;

    // Status code
    let status = response.status();
    println!("Status: {} ({})", status.as_u16(), status.canonical_reason().unwrap_or(""));

    // Response headers
    for (name, value) in response.headers() {
        println!("  {}: {}", name, value.to_str().unwrap_or("<binary>"));
    }

    // Branch processing based on Content-Type
    let content_type = response.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("application/json") {
        let json: serde_json::Value = response.json().await?;
        println!("JSON: {}", serde_json::to_string_pretty(&json)?);
    } else if content_type.contains("text/") {
        let text = response.text().await?;
        println!("Text ({} bytes): {}", text.len(), &text[..200.min(text.len())]);
    } else {
        let bytes = response.bytes().await?;
        println!("Binary: {} bytes", bytes.len());
    }

    Ok(())
}

/// Error handling per status code
async fn handle_api_response(client: &Client, url: &str) -> anyhow::Result<String> {
    let response = client.get(url).send().await?;

    match response.status() {
        StatusCode::OK => Ok(response.text().await?),
        StatusCode::NOT_FOUND => {
            anyhow::bail!("Resource not found: {}", url);
        }
        StatusCode::UNAUTHORIZED => {
            anyhow::bail!("Authentication required");
        }
        StatusCode::TOO_MANY_REQUESTS => {
            // Rate limit -- check the Retry-After header
            let retry_after = response.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(60);
            anyhow::bail!("Rate limited. Please retry after {} seconds", retry_after);
        }
        status if status.is_server_error() => {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Server error {}: {}", status, body);
        }
        status => {
            anyhow::bail!("Unexpected status: {}", status);
        }
    }
}
```

### 2.4 Streaming Download

```rust
use futures::StreamExt;
use tokio::io::AsyncWriteExt;

async fn download_file(url: &str, path: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let response = client.get(url).send().await?.error_for_status()?;

    let total_size = response.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(path).await?;
    let mut downloaded: u64 = 0;

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if total_size > 0 {
            let pct = (downloaded as f64 / total_size as f64) * 100.0;
            eprint!("\rProgress: {:.1}%", pct);
        }
    }
    eprintln!("\nDone: {} bytes", downloaded);

    Ok(())
}

/// Resumable download
async fn resumable_download(url: &str, path: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();

    // Check the size of the existing file
    let existing_size = match tokio::fs::metadata(path).await {
        Ok(meta) => meta.len(),
        Err(_) => 0,
    };

    let mut request = client.get(url);
    if existing_size > 0 {
        // Resume from the middle using the Range header
        request = request.header("Range", format!("bytes={}-", existing_size));
        println!("Resuming from {} bytes", existing_size);
    }

    let response = request.send().await?.error_for_status()?;

    // Confirm 206 Partial Content
    let is_partial = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let total_size = if is_partial {
        // Obtain the total size from Content-Range: bytes 1000-9999/10000
        response.headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split('/').last())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    } else {
        response.content_length().unwrap_or(0)
    };

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(is_partial)
        .write(!is_partial)
        .truncate(!is_partial)
        .open(path)
        .await?;

    let mut downloaded = if is_partial { existing_size } else { 0 };

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        if total_size > 0 {
            let pct = (downloaded as f64 / total_size as f64) * 100.0;
            eprint!("\rProgress: {:.1}% ({}/{})", pct, downloaded, total_size);
        }
    }
    eprintln!("\nDownload complete: {} bytes", downloaded);

    Ok(())
}
```

### 2.5 Concurrent Requests and Rate Limiting

```rust
use futures::stream::{self, StreamExt};
use reqwest::Client;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Bulk requests with a concurrency limit (Semaphore approach)
async fn fetch_all_with_limit(
    client: &Client,
    urls: Vec<String>,
    max_concurrent: usize,
) -> Vec<Result<String, reqwest::Error>> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent));

    let tasks: Vec<_> = urls.into_iter().map(|url| {
        let client = client.clone();
        let semaphore = semaphore.clone();
        tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            client.get(&url).send().await?.text().await
        })
    }).collect();

    let mut results = Vec::new();
    for task in tasks {
        match task.await {
            Ok(result) => results.push(result),
            Err(e) => results.push(Err(reqwest::Error::from(
                std::io::Error::new(std::io::ErrorKind::Other, e)
            ).into())),
        }
    }
    results
}

/// Concurrent requests using buffer_unordered (Stream approach)
async fn fetch_all_stream(
    client: &Client,
    urls: Vec<String>,
    concurrency: usize,
) -> Vec<(String, Result<String, String>)> {
    stream::iter(urls)
        .map(|url| {
            let client = client.clone();
            async move {
                let result = client.get(&url)
                    .send()
                    .await
                    .and_then(|r| Ok(r))
                    .map_err(|e| e.to_string());

                let body = match result {
                    Ok(resp) => resp.text().await.map_err(|e| e.to_string()),
                    Err(e) => Err(e),
                };
                (url, body)
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await
}

/// Sequential requests with rate limiting
async fn rate_limited_requests(
    client: &Client,
    urls: Vec<String>,
    requests_per_second: u32,
) -> Vec<Result<String, reqwest::Error>> {
    let interval = std::time::Duration::from_millis(1000 / requests_per_second as u64);
    let mut results = Vec::new();

    for url in urls {
        let start = std::time::Instant::now();
        let result = client.get(&url).send().await?.text().await;
        results.push(result);

        // Wait until the next request
        let elapsed = start.elapsed();
        if elapsed < interval {
            tokio::time::sleep(interval - elapsed).await;
        }
    }

    results
}
```

### 2.6 Multipart File Upload

```rust
use reqwest::{Client, multipart};
use tokio::fs::File;
use tokio::io::AsyncReadExt;

async fn upload_file(
    client: &Client,
    url: &str,
    file_path: &str,
    field_name: &str,
) -> anyhow::Result<reqwest::Response> {
    // Read file
    let mut file = File::open(file_path).await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;

    let file_name = std::path::Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("upload")
        .to_string();

    // Infer MIME type
    let mime_type = match file_path.rsplit('.').next() {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("pdf") => "application/pdf",
        Some("json") => "application/json",
        Some("csv") => "text/csv",
        _ => "application/octet-stream",
    };

    let part = multipart::Part::bytes(buffer)
        .file_name(file_name)
        .mime_str(mime_type)?;

    let form = multipart::Form::new()
        .part(field_name.to_string(), part)
        .text("description", "Uploaded from Rust client");

    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    Ok(response)
}

/// Concurrent upload of multiple files
async fn upload_multiple_files(
    client: &Client,
    url: &str,
    file_paths: &[&str],
) -> anyhow::Result<reqwest::Response> {
    let mut form = multipart::Form::new();

    for (i, path) in file_paths.iter().enumerate() {
        let mut file = File::open(path).await?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;

        let file_name = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();

        let part = multipart::Part::bytes(buffer)
            .file_name(file_name);

        form = form.part(format!("file_{}", i), part);
    }

    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    Ok(response)
}
```

### 2.7 HTTP Client with Retry

```rust
use reqwest::{Client, Response, StatusCode};
use std::time::Duration;

/// Retry with exponential backoff
async fn retry_request<F, Fut>(
    max_retries: u32,
    base_delay: Duration,
    mut request_fn: F,
) -> anyhow::Result<Response>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = reqwest::Result<Response>>,
{
    let mut last_error = None;

    for attempt in 0..=max_retries {
        if attempt > 0 {
            let delay = base_delay * 2u32.pow(attempt - 1);
            // Add jitter (0.5x to 1.5x)
            let jitter = rand::random::<f64>() + 0.5;
            let actual_delay = Duration::from_millis(
                (delay.as_millis() as f64 * jitter) as u64
            );
            tracing::warn!(
                "Retry {}/{}: retrying in {}ms",
                attempt, max_retries, actual_delay.as_millis()
            );
            tokio::time::sleep(actual_delay).await;
        }

        match (request_fn)().await {
            Ok(response) => {
                let status = response.status();
                // Retryable status codes
                if status == StatusCode::TOO_MANY_REQUESTS
                    || status == StatusCode::SERVICE_UNAVAILABLE
                    || status == StatusCode::GATEWAY_TIMEOUT
                    || status == StatusCode::BAD_GATEWAY
                {
                    last_error = Some(anyhow::anyhow!("Retryable error: {}", status));
                    continue;
                }
                return Ok(response);
            }
            Err(e) => {
                if e.is_connect() || e.is_timeout() {
                    last_error = Some(anyhow::anyhow!("Connection error: {}", e));
                    continue;
                }
                return Err(e.into());
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Maximum retry count exceeded")))
}

/// Usage example
async fn fetch_with_retry(client: &Client, url: &str) -> anyhow::Result<String> {
    let url = url.to_string();
    let client = client.clone();

    let response = retry_request(3, Duration::from_millis(500), || {
        let client = client.clone();
        let url = url.clone();
        async move {
            client.get(&url).send().await
        }
    }).await?;

    Ok(response.text().await?)
}
```

---

## 3. hyper -- Low-Level HTTP Server

### 3.1 Implementing an HTTP Server with hyper

While reqwest is a high-level HTTP client, hyper provides the implementation of the HTTP protocol itself. Axum uses hyper internally, but there are cases where you use hyper directly.

```rust
use hyper::{body::Incoming, server::conn::http1, Request, Response};
use hyper::body::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::net::TcpListener;

async fn handle_request(
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let (method, path) = (req.method().clone(), req.uri().path().to_string());

    let response = match (method.as_str(), path.as_str()) {
        ("GET", "/") => {
            Response::builder()
                .status(200)
                .header("Content-Type", "text/plain; charset=utf-8")
                .body(Full::new(Bytes::from("Hello from hyper!")))
                .unwrap()
        }
        ("GET", "/health") => {
            Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(r#"{"status":"ok"}"#)))
                .unwrap()
        }
        _ => {
            Response::builder()
                .status(404)
                .body(Full::new(Bytes::from("Not Found")))
                .unwrap()
        }
    };

    Ok(response)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    let listener = TcpListener::bind(addr).await?;
    println!("Listening on http://{}", addr);

    loop {
        let (stream, remote_addr) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(handle_request))
                .await
            {
                eprintln!("Error serving {}: {}", remote_addr, err);
            }
        });
    }
}
```

### 3.2 HTTP/2 Server with hyper

```rust
use hyper::{body::Incoming, server::conn::http2, Request, Response};
use hyper::body::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioExecutor};
use std::convert::Infallible;
use tokio::net::TcpListener;

async fn handle(
    _req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    Ok(Response::new(Full::new(Bytes::from("Hello HTTP/2!"))))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("0.0.0.0:3000").await?;

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::spawn(async move {
            // HTTP/2 starts in h2c (cleartext) mode
            // TLS is required in production
            if let Err(err) = http2::Builder::new(TokioExecutor::new())
                .serve_connection(io, service_fn(handle))
                .await
            {
                eprintln!("Error: {}", err);
            }
        });
    }
}
```

### 3.3 Server with Graceful Shutdown

```rust
use hyper::{body::Incoming, server::conn::http1, Request, Response};
use hyper::body::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use tokio::net::TcpListener;
use tokio::signal;

async fn handle(
    _req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    Ok(Response::new(Full::new(Bytes::from("OK"))))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("0.0.0.0:3000").await?;
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

    println!("Server started. Press Ctrl+C to terminate");

    // Ctrl+C handler
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        signal::ctrl_c().await.unwrap();
        println!("\nShutdown initiated...");
        let _ = shutdown_tx_clone.send(());
    });

    loop {
        tokio::select! {
            Ok((stream, addr)) = listener.accept() => {
                let io = TokioIo::new(stream);
                let mut shutdown_rx = shutdown_tx.subscribe();

                tokio::spawn(async move {
                    let conn = http1::Builder::new()
                        .serve_connection(io, service_fn(handle));

                    // graceful shutdown: wait for in-flight requests to complete
                    tokio::pin!(conn);
                    tokio::select! {
                        result = &mut conn => {
                            if let Err(e) = result {
                                eprintln!("Error serving {}: {}", addr, e);
                            }
                        }
                        _ = shutdown_rx.recv() => {
                            conn.as_mut().graceful_shutdown();
                            // Let the remaining processing complete
                            if let Err(e) = conn.await {
                                eprintln!("Error during shutdown: {}", e);
                            }
                        }
                    }
                });
            }
            _ = shutdown_rx.recv() => {
                println!("Stop accepting new connections");
                break;
            }
        }
    }

    println!("Server shutdown complete");
    Ok(())
}
```

---

## 4. WebSocket

### 4.1 WebSocket Client

```rust
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = "wss://echo.websocket.org";
    let (ws_stream, response) = connect_async(url).await?;
    println!("Connected: {}", response.status());

    let (mut write, mut read) = ws_stream.split();

    // Send task
    let send_task = tokio::spawn(async move {
        for i in 1..=5 {
            let msg = Message::Text(format!("Message #{}", i));
            write.send(msg).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        write.send(Message::Close(None)).await.unwrap();
    });

    // Receive task
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => println!("Received: {}", text),
                Ok(Message::Binary(data)) => println!("Binary: {} bytes", data.len()),
                Ok(Message::Ping(data)) => println!("Ping: {:?}", data),
                Ok(Message::Close(_)) => {
                    println!("Disconnected");
                    break;
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    let _ = tokio::join!(send_task, recv_task);
    Ok(())
}
```

### WebSocket Message Flow

```
┌──────────┐                         ┌──────────┐
│  Client  │                         │  Server  │
│          │── Text("hello") ──────→ │          │
│          │                         │          │
│          │←── Text("hello") ────── │          │
│          │                         │          │
│          │── Ping ───────────────→ │          │
│          │←── Pong ────────────── │          │
│          │                         │          │
│          │── Binary(data) ──────→ │          │
│          │                         │          │
│          │── Close ─────────────→ │          │
│          │←── Close ────────────── │          │
└──────────┘                         └──────────┘
     ▲                                    ▲
     │  Full duplex: can send/receive simultaneously │
     │  Use split() to separate read/write           │
     └────────────────────────────────────┘
```

### 4.2 WebSocket Client with Auto-Reconnect

```rust
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use std::time::Duration;
use tokio::sync::mpsc;

/// WebSocket client with reconnect logic
struct ReconnectingWsClient {
    url: String,
    max_retries: u32,
    base_delay: Duration,
}

impl ReconnectingWsClient {
    fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            max_retries: 10,
            base_delay: Duration::from_secs(1),
        }
    }

    async fn run(
        &self,
        outgoing_rx: &mut mpsc::Receiver<String>,
        incoming_tx: mpsc::Sender<String>,
    ) {
        let mut retry_count = 0u32;

        loop {
            match self.connect_and_handle(outgoing_rx, &incoming_tx).await {
                Ok(()) => {
                    println!("WebSocket disconnected normally");
                    break;
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count > self.max_retries {
                        eprintln!("Maximum retry count exceeded: {}", e);
                        break;
                    }

                    let delay = self.base_delay * 2u32.pow((retry_count - 1).min(6));
                    eprintln!(
                        "Connection error (attempt {}): {}. Reconnecting in {} seconds...",
                        retry_count, e, delay.as_secs()
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn connect_and_handle(
        &self,
        outgoing_rx: &mut mpsc::Receiver<String>,
        incoming_tx: &mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        println!("WebSocket connected: {}", self.url);

        let (mut write, mut read) = ws_stream.split();

        loop {
            tokio::select! {
                // Receive messages from the server
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if incoming_tx.send(text).await.is_err() {
                                break; // Receiver was dropped
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            write.send(Message::Pong(data)).await?;
                        }
                        Some(Ok(Message::Close(_))) => {
                            return Ok(()); // Normal disconnect
                        }
                        Some(Err(e)) => {
                            return Err(e.into());
                        }
                        None => {
                            return Err(anyhow::anyhow!("Stream ended"));
                        }
                        _ => {}
                    }
                }
                // Outgoing messages from the application
                msg = outgoing_rx.recv() => {
                    match msg {
                        Some(text) => {
                            write.send(Message::Text(text)).await?;
                        }
                        None => {
                            // Send channel was closed
                            write.send(Message::Close(None)).await?;
                            return Ok(());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
```

### 4.3 WebSocket Server (Axum Integration)

```rust
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Shared state: implement a chat room with a broadcast channel
#[derive(Clone)]
struct ChatState {
    tx: broadcast::Sender<String>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<ChatState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: ChatState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // Server -> Client (broadcast receive -> WebSocket send)
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Client -> Server (WebSocket receive -> broadcast send)
    let tx = state.tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                let _ = tx.send(text);
            }
        }
    });

    // When either task finishes, stop both
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

// Starting the WebSocket server
// #[tokio::main]
// async fn main() {
//     let (tx, _) = broadcast::channel(100);
//     let state = ChatState { tx };
//     let app = Router::new()
//         .route("/ws", get(ws_handler))
//         .with_state(state);
//     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
//     axum::serve(listener, app).await.unwrap();
// }
```

### 4.4 Heartbeat (Periodic Ping) Management

```rust
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;
use std::time::Duration;

/// WebSocket connection management with heartbeat
async fn ws_with_heartbeat(
    ws_stream: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>
    >,
    heartbeat_interval: Duration,
    pong_timeout: Duration,
) -> anyhow::Result<()> {
    let (mut write, mut read) = ws_stream.split();
    let mut heartbeat = tokio::time::interval(heartbeat_interval);
    let mut last_pong = std::time::Instant::now();

    loop {
        tokio::select! {
            // Process received messages
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        println!("Received: {}", text);
                    }
                    Some(Ok(Message::Pong(_))) => {
                        last_pong = std::time::Instant::now();
                    }
                    Some(Ok(Message::Ping(data))) => {
                        write.send(Message::Pong(data)).await?;
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        println!("Connection closed");
                        break;
                    }
                    Some(Err(e)) => {
                        eprintln!("Error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }

            // Periodic heartbeat
            _ = heartbeat.tick() => {
                // Pong timeout check
                if last_pong.elapsed() > pong_timeout {
                    eprintln!("Pong timeout. Disconnecting");
                    break;
                }
                write.send(Message::Ping(vec![].into())).await?;
            }
        }
    }

    // Clean disconnect
    let _ = write.send(Message::Close(None)).await;
    Ok(())
}
```

---

## 5. gRPC -- tonic

### 5.1 Protocol Buffers Definition

```protobuf
// proto/greeter.proto
syntax = "proto3";
package greeter;

service Greeter {
    // Unary RPC: single request -> single response
    rpc SayHello (HelloRequest) returns (HelloReply);

    // Server Streaming RPC: single request -> stream response
    rpc SayHelloStream (HelloRequest) returns (stream HelloReply);

    // Client Streaming RPC: stream request -> single response
    rpc RecordGreetings (stream HelloRequest) returns (GreetingSummary);

    // Bidirectional Streaming RPC: stream <-> stream
    rpc Chat (stream ChatMessage) returns (stream ChatMessage);
}

message HelloRequest {
    string name = 1;
}

message HelloReply {
    string message = 1;
    int64 timestamp = 2;
}

message GreetingSummary {
    int32 count = 1;
    string summary = 2;
}

message ChatMessage {
    string user = 1;
    string content = 2;
    int64 timestamp = 3;
}
```

### 5.2 Build Script

```rust
// build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")   // Output destination (optional)
        .compile_protos(
            &["proto/greeter.proto"],
            &["proto"],
        )?;
    Ok(())
}
```

### 5.3 gRPC Server Implementation in 4 Patterns

```rust
// src/server.rs
use tonic::{transport::Server, Request, Response, Status, Streaming};
use greeter::greeter_server::{Greeter, GreeterServer};
use greeter::{HelloReply, HelloRequest, GreetingSummary, ChatMessage};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use std::time::{SystemTime, UNIX_EPOCH};

pub mod greeter {
    tonic::include_proto!("greeter");
}

fn now_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

#[derive(Debug, Default)]
pub struct MyGreeter;

#[tonic::async_trait]
impl Greeter for MyGreeter {
    // 1. Unary RPC
    async fn say_hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloReply>, Status> {
        let metadata = request.metadata();
        // Retrieve information from metadata
        let request_id = metadata
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown");
        println!("Unary RPC: request_id={}", request_id);

        let name = request.into_inner().name;
        if name.is_empty() {
            return Err(Status::invalid_argument("name is required"));
        }

        let reply = HelloReply {
            message: format!("Hello, {}!", name),
            timestamp: now_timestamp(),
        };
        Ok(Response::new(reply))
    }

    // 2. Server Streaming RPC
    type SayHelloStreamStream = ReceiverStream<Result<HelloReply, Status>>;

    async fn say_hello_stream(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<Self::SayHelloStreamStream>, Status> {
        let name = request.into_inner().name;
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        tokio::spawn(async move {
            for i in 1..=5 {
                let reply = HelloReply {
                    message: format!("#{}: Hello, {}!", i, name),
                    timestamp: now_timestamp(),
                };
                if tx.send(Ok(reply)).await.is_err() { break; }
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // 3. Client Streaming RPC
    async fn record_greetings(
        &self,
        request: Request<Streaming<HelloRequest>>,
    ) -> Result<Response<GreetingSummary>, Status> {
        let mut stream = request.into_inner();
        let mut names = Vec::new();

        while let Some(req) = stream.next().await {
            let req = req?;
            println!("Client Stream received: {}", req.name);
            names.push(req.name);
        }

        let summary = GreetingSummary {
            count: names.len() as i32,
            summary: format!("Names received: {}", names.join(", ")),
        };

        Ok(Response::new(summary))
    }

    // 4. Bidirectional Streaming RPC
    type ChatStream = ReceiverStream<Result<ChatMessage, Status>>;

    async fn chat(
        &self,
        request: Request<Streaming<ChatMessage>>,
    ) -> Result<Response<Self::ChatStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::spawn(async move {
            while let Some(result) = inbound.next().await {
                match result {
                    Ok(msg) => {
                        println!("[{}] {}", msg.user, msg.content);
                        // Echo back (in practice, broadcast etc.)
                        let reply = ChatMessage {
                            user: "Server".to_string(),
                            content: format!("Echo: {}", msg.content),
                            timestamp: now_timestamp(),
                        };
                        if tx.send(Ok(reply)).await.is_err() { break; }
                    }
                    Err(e) => {
                        eprintln!("Stream error: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    println!("gRPC server started: {}", addr);

    Server::builder()
        .add_service(GreeterServer::new(MyGreeter::default()))
        .serve(addr)
        .await?;
    Ok(())
}
```

### 5.4 gRPC Client (All 4 Patterns)

```rust
use greeter::greeter_client::GreeterClient;
use greeter::{HelloRequest, ChatMessage};
use tokio_stream::StreamExt;

pub mod greeter {
    tonic::include_proto!("greeter");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = GreeterClient::connect("http://[::1]:50051").await?;

    // 1. Unary RPC
    println!("=== Unary RPC ===");
    let mut request = tonic::Request::new(HelloRequest {
        name: "Rust".into(),
    });
    // Attach metadata
    request.metadata_mut().insert(
        "x-request-id",
        "req-12345".parse().unwrap(),
    );
    let response = client.say_hello(request).await?;
    println!("Response: {}", response.into_inner().message);

    // 2. Server Streaming RPC
    println!("\n=== Server Streaming RPC ===");
    let request = tonic::Request::new(HelloRequest {
        name: "World".into(),
    });
    let mut stream = client.say_hello_stream(request).await?.into_inner();
    while let Some(reply) = stream.next().await {
        let reply = reply?;
        println!("Stream: {} (ts={})", reply.message, reply.timestamp);
    }

    // 3. Client Streaming RPC
    println!("\n=== Client Streaming RPC ===");
    let names = vec!["Alice", "Bob", "Charlie"];
    let request_stream = tokio_stream::iter(
        names.into_iter().map(|name| HelloRequest { name: name.into() })
    );
    let response = client.record_greetings(request_stream).await?;
    let summary = response.into_inner();
    println!("Summary: {} entries -- {}", summary.count, summary.summary);

    // 4. Bidirectional Streaming RPC
    println!("\n=== Bidirectional Streaming RPC ===");
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    // Outgoing stream
    tokio::spawn(async move {
        for i in 1..=3 {
            let msg = ChatMessage {
                user: "Client".to_string(),
                content: format!("Message #{}", i),
                timestamp: 0,
            };
            tx.send(msg).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    });

    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);
    let mut inbound = client.chat(outbound).await?.into_inner();

    while let Some(reply) = inbound.next().await {
        let reply = reply?;
        println!("[{}] {}", reply.user, reply.content);
    }

    Ok(())
}
```

### 5.5 gRPC Authentication and Interceptors

```rust
use tonic::{transport::Channel, Request, Status};
use tonic::service::Interceptor;

/// Authentication interceptor
#[derive(Clone)]
struct AuthInterceptor {
    token: String,
}

impl Interceptor for AuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        let token = format!("Bearer {}", self.token);
        request.metadata_mut().insert(
            "authorization",
            token.parse().map_err(|_| Status::internal("invalid token"))?,
        );
        // Also attach a request ID
        let request_id = uuid::Uuid::new_v4().to_string();
        request.metadata_mut().insert(
            "x-request-id",
            request_id.parse().unwrap(),
        );
        Ok(request)
    }
}

/// Server-side authentication check
fn check_auth(request: &Request<impl std::fmt::Debug>) -> Result<String, Status> {
    let token = request.metadata()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Status::unauthenticated("no authentication token"))?;

    let token = token
        .strip_prefix("Bearer ")
        .ok_or_else(|| Status::unauthenticated("invalid Bearer token format"))?;

    // Token verification (in practice, JWT verification etc.)
    if token == "valid-token" {
        Ok("user-123".to_string())
    } else {
        Err(Status::unauthenticated("invalid token"))
    }
}

// Usage example
// let channel = Channel::from_static("http://[::1]:50051").connect().await?;
// let interceptor = AuthInterceptor { token: "valid-token".into() };
// let client = GreeterClient::with_interceptor(channel, interceptor);
```

### 5.6 gRPC Server -- TLS and Health Checks

```rust
use tonic::transport::{Server, ServerTlsConfig, Identity};
use tonic_health::server::HealthReporter;

async fn start_grpc_server() -> Result<(), Box<dyn std::error::Error>> {
    // TLS configuration
    let cert = tokio::fs::read("server.pem").await?;
    let key = tokio::fs::read("server.key").await?;
    let identity = Identity::from_pem(cert, key);

    let tls_config = ServerTlsConfig::new().identity(identity);

    // Health check service
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<GreeterServer<MyGreeter>>()
        .await;

    let addr = "[::1]:50051".parse()?;

    Server::builder()
        .tls_config(tls_config)?
        .add_service(health_service)
        .add_service(GreeterServer::new(MyGreeter::default()))
        .serve(addr)
        .await?;

    Ok(())
}
```

### The 4 Communication Patterns of gRPC

```
┌─────────────────────────────────────────────────────────┐
│                gRPC Communication Patterns              │
│                                                         │
│  1. Unary (single -> single)                            │
│     Client ──── Request ────→ Server                    │
│     Client ←─── Response ──── Server                    │
│                                                         │
│  2. Server Streaming (single -> stream)                 │
│     Client ──── Request ────→ Server                    │
│     Client ←─── Response 1 ── Server                    │
│     Client ←─── Response 2 ── Server                    │
│     Client ←─── Response N ── Server                    │
│                                                         │
│  3. Client Streaming (stream -> single)                 │
│     Client ──── Request 1 ──→ Server                    │
│     Client ──── Request 2 ──→ Server                    │
│     Client ──── Request N ──→ Server                    │
│     Client ←─── Response ──── Server                    │
│                                                         │
│  4. Bidirectional Streaming (stream <-> stream)         │
│     Client ──── Request 1 ──→ Server                    │
│     Client ←─── Response 1 ── Server                    │
│     Client ──── Request 2 ──→ Server                    │
│     Client ←─── Response 2 ── Server                    │
│     (requests and responses flow independently)         │
└─────────────────────────────────────────────────────────┘
```

---

## 6. DNS Resolution and TLS

### 6.1 Custom DNS Resolver

```rust
use reqwest::dns::{Resolve, Resolving, Name};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::collections::HashMap;

/// Static DNS resolution (for testing or service discovery)
struct StaticResolver {
    overrides: HashMap<String, Vec<SocketAddr>>,
}

impl StaticResolver {
    fn new() -> Self {
        Self {
            overrides: HashMap::new(),
        }
    }

    fn add_override(&mut self, host: &str, addr: IpAddr, port: u16) {
        self.overrides
            .entry(host.to_string())
            .or_default()
            .push(SocketAddr::new(addr, port));
    }
}

impl Resolve for StaticResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        if let Some(addrs) = self.overrides.get(&host) {
            let addrs: Vec<SocketAddr> = addrs.clone();
            Box::pin(async move {
                Ok(Box::new(addrs.into_iter()) as Box<dyn Iterator<Item = SocketAddr> + Send>)
            })
        } else {
            // Fallback: use system DNS
            Box::pin(async move {
                let addrs = tokio::net::lookup_host(format!("{}:0", host))
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
                Ok(Box::new(addrs) as Box<dyn Iterator<Item = SocketAddr> + Send>)
            })
        }
    }
}

// Usage example
// let mut resolver = StaticResolver::new();
// resolver.add_override("api.local", IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
// let client = reqwest::Client::builder()
//     .dns_resolver(Arc::new(resolver))
//     .build()?;
```

### 6.2 Customizing TLS Configuration

```rust
use reqwest::Client;
use std::time::Duration;

/// Build a secure client using rustls
fn build_secure_client() -> reqwest::Result<Client> {
    // Add a custom CA certificate
    let ca_cert = std::fs::read("ca.pem").expect("Failed to read CA certificate");
    let ca = reqwest::Certificate::from_pem(&ca_cert)?;

    // Client certificate (mTLS)
    let client_cert = std::fs::read("client.pem").expect("Failed to read client certificate");
    let client_key = std::fs::read("client.key").expect("Failed to read client key");
    let identity = reqwest::Identity::from_pem(
        &[client_cert, client_key].concat()
    )?;

    Client::builder()
        .use_rustls_tls()
        .add_root_certificate(ca)
        .identity(identity)
        .min_tls_version(reqwest::tls::Version::TLS_1_2)
        .timeout(Duration::from_secs(30))
        .build()
}

/// Client that allows self-signed certificates (development only)
fn build_dev_client() -> reqwest::Result<Client> {
    Client::builder()
        .danger_accept_invalid_certs(true)  // Never use in production!
        .timeout(Duration::from_secs(30))
        .build()
}
```

---

## 7. Connection Pool and Connection Management

### How a Connection Pool Works

```
┌──────────────── Connection Pool Management ───────────┐
│                                                         │
│  reqwest::Client                                        │
│  ┌────────────────────────────────────────────────┐    │
│  │  Connection Pool                                │    │
│  │                                                 │    │
│  │  api.example.com:443                           │    │
│  │  ├── Conn #1 [IDLE]     <- reusable            │    │
│  │  ├── Conn #2 [ACTIVE]   <- handling request    │    │
│  │  └── Conn #3 [IDLE]     <- reusable            │    │
│  │                                                 │    │
│  │  db.example.com:5432                           │    │
│  │  ├── Conn #1 [ACTIVE]                          │    │
│  │  └── Conn #2 [IDLE]                            │    │
│  │                                                 │    │
│  │  Settings:                                      │    │
│  │  pool_max_idle_per_host = 10                    │    │
│  │  pool_idle_timeout = 90s                        │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  New request:                                           │
│  1. Get an idle connection from the pool (if any)       │
│  2. Otherwise, new TCP connection + TLS handshake       │
│  3. After receiving response, return connection to pool │
│  4. Auto-disconnect after idle timeout                  │
└─────────────────────────────────────────────────────────┘
```

### Guidelines for Connection Pool Optimization

```rust
use reqwest::Client;
use std::time::Duration;

/// Client configurations by use case
mod pool_configs {
    use super::*;

    /// For high throughput (large numbers of concurrent requests)
    pub fn high_throughput() -> Client {
        Client::builder()
            .pool_max_idle_per_host(50)
            .pool_idle_timeout(Duration::from_secs(120))
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true)
            .build()
            .unwrap()
    }

    /// For low latency (a small number of critical requests)
    pub fn low_latency() -> Client {
        Client::builder()
            .pool_max_idle_per_host(5)
            .pool_idle_timeout(Duration::from_secs(300))
            .tcp_keepalive(Duration::from_secs(30))
            .tcp_nodelay(true)
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    /// For memory efficiency (resource-constrained environments)
    pub fn memory_efficient() -> Client {
        Client::builder()
            .pool_max_idle_per_host(2)
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .unwrap()
    }
}
```

---

## 8. Comparison Tables

### Communication Protocol Comparison

| Property | REST (HTTP) | WebSocket | gRPC |
|---|---|---|---|
| Protocol | HTTP/1.1, HTTP/2 | WebSocket over TCP | HTTP/2 |
| Data Format | JSON / XML | Text / Binary | Protocol Buffers |
| Direction | Request-Response | Bidirectional | 4 patterns (Unary/Stream) |
| Schema | OpenAPI (optional) | None | .proto (required) |
| Performance | Medium | High | Very High |
| Browser Support | Full | Full | Via grpc-web |
| Use Case | CRUD API | Chat, real-time | Inter-microservice |
| Serialization Speed | Slow (text) | N/A | Fast (binary) |
| Payload Size | Large | Variable | Small |
| Code Generation | Optional | None | Required |
| Error Handling | HTTP status code | App-defined | Status + Code |
| Streaming | SSE / chunked | Native | Native |
| Load Balancing | L7 (HTTP) | L4/L7 | L7 (HTTP/2) |

### Rust HTTP Client Library Comparison

| Library | Level | Use | Features | Async |
|---|---|---|---|---|
| reqwest | High | REST API calls | Concise, automatic Cookie/redirect | Yes |
| hyper | Low | Custom HTTP impl | High performance, full control | Yes |
| ureq | Sync | Synchronous HTTP client | Simple, no async needed | No |
| surf | Mid | For async-std | Middleware support | Yes |
| isahc | Mid | curl-based | HTTP/2, connection pool | Yes |
| attohttpc | Lightweight | For embedded | Few deps, small binary | No |

### WebSocket Library Comparison

| Library | Runtime | Level | Features |
|---|---|---|---|
| tokio-tungstenite | tokio | Mid | tokio integration of tungstenite. Most widely used |
| tungstenite | sync | Low | Foundational lib supporting both sync and async |
| axum::extract::ws | tokio | High | Built into Axum. Server-oriented |
| warp::ws | tokio | High | Built into Warp. Server-oriented |

### gRPC Library Comparison

| Library | Features | Use |
|---|---|---|
| tonic | Pure Rust. tokio-based | Recommended. Most mature |
| grpc-rs | C++ gRPC bindings | Only for special requirements |
| volo-grpc | Provided by CloudWeGo | High-performance microservices |

---

## 9. Anti-Patterns

### Anti-Pattern 1: Creating a Client Per Request

```rust
// NG: Create Client every time -> connection pool not utilized, TLS handshake every time
async fn bad_fetch(url: &str) -> reqwest::Result<String> {
    let client = reqwest::Client::new(); // created every time
    client.get(url).send().await?.text().await
}

// OK: Share Client across the entire application
use std::sync::LazyLock;
static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .pool_max_idle_per_host(20)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap()
});

async fn good_fetch(url: &str) -> reqwest::Result<String> {
    HTTP_CLIENT.get(url).send().await?.text().await
}
```

### Anti-Pattern 2: Not Handling WebSocket Ping/Pong

```rust
// NG: Ignoring Ping -> server times out and disconnects
while let Some(msg) = read.next().await {
    if let Ok(Message::Text(text)) = msg {
        process(text);
    }
    // Ignoring Ping!
}

// OK: Auto-respond to Ping
while let Some(msg) = read.next().await {
    match msg? {
        Message::Text(text) => process(&text),
        Message::Ping(data) => {
            write.send(Message::Pong(data)).await?;
        }
        Message::Close(_) => break,
        _ => {}
    }
}
```

### Anti-Pattern 3: No Timeout Configured

```rust
// NG: No timeout -> blocks forever on an unresponsive server
let response = reqwest::get("https://slow-server.example.com").await?;

// OK: Always set a timeout
let client = reqwest::Client::builder()
    .timeout(std::time::Duration::from_secs(30))
    .connect_timeout(std::time::Duration::from_secs(5))
    .build()?;
let response = client.get("https://slow-server.example.com").send().await?;
```

### Anti-Pattern 4: Forgetting to Call error_for_status()

```rust
// NG: 4xx/5xx is not treated as error -> silent failure
let body: MyData = client.get(url).send().await?.json().await?;
// Panic when trying to deserialize 404 HTML!

// OK: Explicitly check with error_for_status()
let body: MyData = client.get(url)
    .send()
    .await?
    .error_for_status()?  // 4xx/5xx -> Err
    .json()
    .await?;
```

### Anti-Pattern 5: Inappropriate Use of gRPC Error Statuses

```rust
// NG: Map all errors to Internal
impl Greeter for MyGreeter {
    async fn say_hello(&self, req: Request<HelloRequest>) -> Result<Response<HelloReply>, Status> {
        let name = req.into_inner().name;
        do_something(&name).map_err(|e| Status::internal(e.to_string()))?; // All Internal!
        todo!()
    }
}

// OK: Use the appropriate status code per case
impl Greeter for MyGreeter {
    async fn say_hello(&self, req: Request<HelloRequest>) -> Result<Response<HelloReply>, Status> {
        let name = req.into_inner().name;
        if name.is_empty() {
            return Err(Status::invalid_argument("name cannot be empty"));
        }
        let user = find_user(&name).await
            .map_err(|_| Status::not_found(format!("User '{}' not found", name)))?;
        let result = process(user).await
            .map_err(|e| Status::internal(format!("Processing error: {}", e)))?;
        todo!()
    }
}
```

### gRPC Status Code List

| Code | Name | Use |
|---|---|---|
| 0 | OK | Success |
| 1 | CANCELLED | Cancelled by client |
| 2 | UNKNOWN | Unknown error |
| 3 | INVALID_ARGUMENT | Invalid argument |
| 4 | DEADLINE_EXCEEDED | Timeout |
| 5 | NOT_FOUND | Resource not found |
| 6 | ALREADY_EXISTS | Resource already exists |
| 7 | PERMISSION_DENIED | Insufficient permissions |
| 8 | RESOURCE_EXHAUSTED | Resource exhausted (rate limit) |
| 9 | FAILED_PRECONDITION | Precondition failed |
| 10 | ABORTED | Aborted (contention etc.) |
| 11 | OUT_OF_RANGE | Out of range |
| 12 | UNIMPLEMENTED | Not implemented |
| 13 | INTERNAL | Internal error |
| 14 | UNAVAILABLE | Service temporarily unavailable |
| 16 | UNAUTHENTICATED | Authentication failed |

---

## 10. Practical Patterns

### 10.1 API Client Wrapper

```rust
use reqwest::Client;
use serde::{de::DeserializeOwned, Serialize};
use std::time::Duration;

/// Type-safe API client
struct ApiClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl ApiClient {
    fn new(base_url: &str, api_key: &str) -> anyhow::Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
            .build()?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        })
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> anyhow::Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    async fn post<T: DeserializeOwned, B: Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(body)
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    async fn delete(&self, path: &str) -> anyhow::Result<()> {
        let url = format!("{}{}", self.base_url, path);
        self.client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

// Usage example:
// let api = ApiClient::new("https://api.example.com", "my-key")?;
// let users: Vec<User> = api.get("/users").await?;
// let new_user: User = api.post("/users", &CreateUser { name: "Alice".into() }).await?;
```

### 10.2 Server-Sent Events (SSE) Client

```rust
use futures::StreamExt;
use reqwest::Client;

#[derive(Debug)]
struct SseEvent {
    event_type: String,
    data: String,
    id: Option<String>,
}

/// Receive an SSE stream
async fn consume_sse(client: &Client, url: &str) -> anyhow::Result<()> {
    let response = client
        .get(url)
        .header("Accept", "text/event-stream")
        .send()
        .await?
        .error_for_status()?;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        buffer.push_str(&text);

        // Split by event delimiter (blank line)
        while let Some(pos) = buffer.find("\n\n") {
            let event_text = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            if let Some(event) = parse_sse_event(&event_text) {
                println!("[{}] {}", event.event_type, event.data);
            }
        }
    }

    Ok(())
}

fn parse_sse_event(text: &str) -> Option<SseEvent> {
    let mut event_type = "message".to_string();
    let mut data_lines = Vec::new();
    let mut id = None;

    for line in text.lines() {
        if let Some(value) = line.strip_prefix("event: ") {
            event_type = value.to_string();
        } else if let Some(value) = line.strip_prefix("data: ") {
            data_lines.push(value.to_string());
        } else if let Some(value) = line.strip_prefix("id: ") {
            id = Some(value.to_string());
        }
    }

    if data_lines.is_empty() {
        return None;
    }

    Some(SseEvent {
        event_type,
        data: data_lines.join("\n"),
        id,
    })
}
```

### 10.3 Direct TCP Socket Operations

```rust
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, AsyncBufReadExt};

/// Simple echo server
async fn echo_server(addr: &str) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    println!("Echo server started: {}", addr);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        println!("Connected: {}", peer_addr);

        tokio::spawn(async move {
            if let Err(e) = handle_echo_client(stream).await {
                eprintln!("Client error ({}): {}", peer_addr, e);
            }
            println!("Disconnected: {}", peer_addr);
        });
    }
}

async fn handle_echo_client(mut stream: TcpStream) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 { break; } // Connection closed

        writer.write_all(line.as_bytes()).await?;
        writer.flush().await?;
    }

    Ok(())
}

/// TCP client
async fn echo_client(addr: &str) -> anyhow::Result<()> {
    let mut stream = TcpStream::connect(addr).await?;
    println!("Connected to server: {}", addr);

    stream.write_all(b"Hello, server!\n").await?;

    let mut buf = vec![0u8; 1024];
    let n = stream.read(&mut buf).await?;
    println!("Response: {}", String::from_utf8_lossy(&buf[..n]));

    Ok(())
}
```

---

## FAQ

### Q1: Should I use reqwest or hyper?

**A:** For typical REST API calls, reqwest is sufficient. Use hyper directly only when you need low-level HTTP control such as implementing a custom protocol or building a framework. Axum uses hyper internally, but you almost never need to touch hyper directly. The decision criteria are:

- **Use reqwest when**: REST API client, file downloads, sending webhooks, integrating external services
- **Use hyper when**: custom proxy server, HTTP extensions for proprietary protocols, extreme performance tuning

### Q2: How do I implement authentication in gRPC?

**A:** Attach a Bearer token to the metadata using tonic's `Interceptor`. On the server side, validate the token from the request metadata. For JWT tokens, combining with the `jsonwebtoken` crate is common.

```rust
let channel = tonic::transport::Channel::from_static("http://[::1]:50051")
    .connect()
    .await?;

let client = GreeterClient::with_interceptor(channel, |mut req: tonic::Request<()>| {
    req.metadata_mut().insert(
        "authorization",
        "Bearer my-token".parse().unwrap(),
    );
    Ok(req)
});
```

### Q3: What are the benefits of HTTP/2?

**A:** It allows multiplexing several requests over a single TCP connection, and reduces overhead via header compression (HPACK). gRPC requires HTTP/2, and streaming RPCs are realized using HTTP/2's frame features. Specific benefits include:

- **Multiplexing**: handle multiple requests/responses simultaneously over one TCP connection
- **Header compression**: HPACK significantly reduces header size
- **Server push**: proactive sends from server to client
- **Stream priority**: prioritize important requests
- **Flow control**: per-stream flow control

### Q4: Should I use WebSocket or SSE (Server-Sent Events)?

**A:** Choose based on use case:

- **WebSocket**: when you need bidirectional communication (chat, games, collaboration)
- **SSE**: when one-way notifications from the server are sufficient (dashboard updates, log streaming)

SSE runs over HTTP/1.1, is easy to implement, and provides automatic reconnection. WebSocket allows bidirectional communication and binary data, but is more complex to implement.

### Q5: How do I manage cookies in reqwest?

**A:** Enabling `cookie_store(true)` automatically manages session cookies.

```rust
let client = reqwest::Client::builder()
    .cookie_store(true)
    .build()?;

// Login request
client.post("https://example.com/login")
    .json(&credentials)
    .send()
    .await?;

// Subsequent requests automatically include the cookies
let profile = client.get("https://example.com/profile")
    .send()
    .await?;
```

### Q6: What is the difference between gRPC's deadline and timeout?

**A:** A deadline is "complete by this point in time"; a timeout is "complete within this duration." gRPC internally converts to a deadline. Because deadlines propagate across services, you can manage timeouts consistently across an entire microservice chain.

```rust
use std::time::Duration;

let mut request = tonic::Request::new(HelloRequest { name: "test".into() });
request.set_timeout(Duration::from_secs(5)); // Set timeout to 5 seconds
```

### Q7: What are the best practices for many concurrent HTTP requests?

**A:** Follow these three principles:

1. **Share the Client**: share a single `reqwest::Client` across all requests to leverage the connection pool
2. **Limit concurrency**: control the number of simultaneous requests with `Semaphore` or `buffer_unordered`
3. **Set timeouts**: set timeouts both on individual requests and on the Client as a whole

```rust
// Recommended pattern
let results = stream::iter(urls)
    .map(|url| {
        let client = client.clone();
        async move { client.get(&url).send().await?.text().await }
    })
    .buffer_unordered(10) // up to 10 concurrent
    .collect::<Vec<_>>()
    .await;
```

---

## Summary

| Topic | Key Points |
|---|---|
| reqwest | High-level HTTP client. Client must be reused. Built-in connection pool |
| hyper | Low-level HTTP. For framework building or custom proxies |
| WebSocket | Bidirectional comms with tokio-tungstenite. Use split() to separate read/write. Ping/Pong required |
| gRPC | tonic + .proto. 4 communication patterns. High-performance microservice comms |
| Connection Pool | Share Client via static/Arc to leverage pool. Configurable per host |
| Streaming | Sequential processing with bytes_stream() / Server Streaming. Memory-efficient |
| Ping/Pong | Required for WebSocket keepalive. Also useful for timeout detection |
| TLS | rustls is the recommended default. mTLS is achieved via client certificates |
| Retry | Exponential backoff + jitter. Identify retryable status codes |
| Errors | Always use error_for_status(). For gRPC, use appropriate status codes |

## Recommended Next Reading

- [Axum](./04-axum-web.md) -- Server-side implementation with a web framework
- [Serde](../04-ecosystem/02-serde.md) -- JSON/Protobuf serialization
- [Async Patterns](./02-async-patterns.md) -- Implementation of retry and concurrency limits

## References

1. **reqwest documentation**: https://docs.rs/reqwest/latest/reqwest/
2. **hyper documentation**: https://docs.rs/hyper/latest/hyper/
3. **tokio-tungstenite**: https://docs.rs/tokio-tungstenite/latest/tokio_tungstenite/
4. **tonic (gRPC for Rust)**: https://docs.rs/tonic/latest/tonic/
5. **tonic-health**: https://docs.rs/tonic-health/latest/tonic_health/
6. **tower-http**: https://docs.rs/tower-http/latest/tower_http/
7. **rustls**: https://docs.rs/rustls/latest/rustls/
