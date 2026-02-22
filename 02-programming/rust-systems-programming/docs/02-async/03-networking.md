# ネットワーク — reqwest/hyper、WebSocket、tonic

> Rust の非同期ネットワークスタックとして HTTP クライアント/サーバー、WebSocket、gRPC の実装パターンを習得する

## この章で学ぶこと

1. **HTTPクライアント** — reqwest による REST API 呼び出しと接続プール管理
2. **HTTPサーバー (低レベル)** — hyper による HTTP サーバーの直接実装
3. **WebSocket** — tokio-tungstenite によるリアルタイム双方向通信
4. **gRPC** — tonic による Protocol Buffers ベースの高性能 RPC
5. **DNS解決とTLS** — trust-dns と rustls による安全な接続
6. **接続管理** — 接続プール、リトライ、タイムアウト戦略

---

## 1. HTTPスタックの全体像

```
┌───────────────── Rust HTTP エコシステム ─────────────────┐
│                                                          │
│  アプリケーション層                                       │
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
│  │     mio (epoll/kqueue/IOCP 抽象化)     │              │
│  └───────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

### 各レイヤーの役割

| レイヤー | クレート | 役割 |
|---|---|---|
| アプリケーション | reqwest / Axum / tonic | 高レベルAPI。開発者が通常触る層 |
| HTTPプロトコル | hyper | HTTP/1.1 と HTTP/2 のパース・生成 |
| 非同期ランタイム | tokio | タスクスケジューリング、非同期I/O |
| OS抽象化 | mio | epoll (Linux) / kqueue (macOS) / IOCP (Windows) |
| TLS | rustls / native-tls | TLS 1.2/1.3 暗号化 |
| DNS | trust-dns / system resolver | ホスト名解決 |

### HTTPリクエストの処理フロー

```
┌─────────────────────────────────────────────────────────────┐
│                HTTPリクエスト処理の内部フロー                   │
│                                                             │
│  1. DNS解決                                                 │
│     app.example.com → 203.0.113.50                         │
│         │                                                   │
│  2. TCP接続 (接続プールから取得 or 新規作成)                   │
│         │                                                   │
│  3. TLSハンドシェイク (HTTPS の場合)                          │
│     ┌──────────┐     ┌──────────┐                          │
│     │ Client   │────→│ Server   │  ClientHello             │
│     │          │←────│          │  ServerHello + Cert       │
│     │          │────→│          │  Key Exchange + Finished  │
│     │          │←────│          │  Finished                 │
│     └──────────┘     └──────────┘                          │
│         │                                                   │
│  4. HTTP リクエスト送信                                      │
│     GET /api/users HTTP/1.1                                 │
│     Host: app.example.com                                   │
│     Authorization: Bearer xxx                               │
│         │                                                   │
│  5. HTTP レスポンス受信                                      │
│     HTTP/1.1 200 OK                                         │
│     Content-Type: application/json                          │
│     {"users": [...]}                                        │
│         │                                                   │
│  6. 接続をプールに返却 (Keep-Alive)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. reqwest — HTTPクライアント

### 2.1 Client の構築と設定

reqwest の `Client` は内部に接続プールを持つ。アプリケーション全体で1つ（またはホスト群ごとに少数）のインスタンスを共有するのが鉄則である。

```rust
use reqwest::{Client, ClientBuilder, redirect::Policy};
use std::time::Duration;

/// 本番向けの Client 構築例
fn build_production_client() -> reqwest::Result<Client> {
    ClientBuilder::new()
        // タイムアウト設定
        .timeout(Duration::from_secs(30))           // リクエスト全体のタイムアウト
        .connect_timeout(Duration::from_secs(5))    // TCP接続確立のタイムアウト
        .read_timeout(Duration::from_secs(15))      // 読み取りタイムアウト

        // 接続プール
        .pool_max_idle_per_host(20)                  // ホストごとのアイドル接続数上限
        .pool_idle_timeout(Duration::from_secs(90))  // アイドル接続のTTL

        // TLS
        .min_tls_version(reqwest::tls::Version::TLS_1_2)
        .danger_accept_invalid_certs(false)          // 本番では必ず false

        // リダイレクト
        .redirect(Policy::limited(10))               // 最大10回のリダイレクト

        // ヘッダー
        .user_agent("my-rust-app/1.0")
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert("Accept", "application/json".parse().unwrap());
            headers
        })

        // 圧縮
        .gzip(true)
        .brotli(true)
        .deflate(true)

        .build()
}
```

### 2.2 基本的なHTTPリクエスト

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
    // クライアントの構築 (接続プール付き — 再利用すべき)
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(10)
        .user_agent("my-rust-app/1.0")
        .build()?;

    // GET リクエスト + JSON デシリアライズ
    let user: User = client
        .get("https://api.github.com/users/rust-lang")
        .header("Accept", "application/json")
        .send()
        .await?
        .error_for_status()?  // 4xx/5xx をエラーに変換
        .json()
        .await?;
    println!("User: {} ({})", user.login, user.name.unwrap_or_default());

    // POST リクエスト
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

    // PUT リクエスト
    let response = client
        .put("https://api.example.com/users/1")
        .json(&serde_json::json!({
            "name": "Updated Name",
            "email": "new@example.com"
        }))
        .send()
        .await?;
    println!("PUT Status: {}", response.status());

    // DELETE リクエスト
    let response = client
        .delete("https://api.example.com/users/1")
        .bearer_auth("your-token")
        .send()
        .await?;
    println!("DELETE Status: {}", response.status());

    // PATCH リクエスト
    let response = client
        .patch("https://api.example.com/users/1")
        .json(&serde_json::json!({"name": "Patched"}))
        .send()
        .await?;
    println!("PATCH Status: {}", response.status());

    Ok(())
}
```

### 2.3 レスポンスの詳細処理

```rust
use reqwest::{Client, StatusCode};
use std::collections::HashMap;

async fn inspect_response(client: &Client, url: &str) -> anyhow::Result<()> {
    let response = client.get(url).send().await?;

    // ステータスコード
    let status = response.status();
    println!("Status: {} ({})", status.as_u16(), status.canonical_reason().unwrap_or(""));

    // レスポンスヘッダー
    for (name, value) in response.headers() {
        println!("  {}: {}", name, value.to_str().unwrap_or("<binary>"));
    }

    // Content-Type に基づく処理分岐
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

/// ステータスコード別のエラーハンドリング
async fn handle_api_response(client: &Client, url: &str) -> anyhow::Result<String> {
    let response = client.get(url).send().await?;

    match response.status() {
        StatusCode::OK => Ok(response.text().await?),
        StatusCode::NOT_FOUND => {
            anyhow::bail!("リソースが見つかりません: {}", url);
        }
        StatusCode::UNAUTHORIZED => {
            anyhow::bail!("認証が必要です");
        }
        StatusCode::TOO_MANY_REQUESTS => {
            // Rate limit — Retry-After ヘッダーを確認
            let retry_after = response.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(60);
            anyhow::bail!("レート制限。{}秒後にリトライしてください", retry_after);
        }
        status if status.is_server_error() => {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("サーバーエラー {}: {}", status, body);
        }
        status => {
            anyhow::bail!("予期しないステータス: {}", status);
        }
    }
}
```

### 2.4 ストリーミングダウンロード

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
            eprint!("\r進捗: {:.1}%", pct);
        }
    }
    eprintln!("\n完了: {} バイト", downloaded);

    Ok(())
}

/// レジューム対応ダウンロード
async fn resumable_download(url: &str, path: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();

    // 既存ファイルのサイズを確認
    let existing_size = match tokio::fs::metadata(path).await {
        Ok(meta) => meta.len(),
        Err(_) => 0,
    };

    let mut request = client.get(url);
    if existing_size > 0 {
        // Range ヘッダーで途中から再開
        request = request.header("Range", format!("bytes={}-", existing_size));
        println!("{}バイトから再開", existing_size);
    }

    let response = request.send().await?.error_for_status()?;

    // 206 Partial Content の確認
    let is_partial = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let total_size = if is_partial {
        // Content-Range: bytes 1000-9999/10000 からトータルサイズを取得
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
            eprint!("\r進捗: {:.1}% ({}/{})", pct, downloaded, total_size);
        }
    }
    eprintln!("\nダウンロード完了: {} バイト", downloaded);

    Ok(())
}
```

### 2.5 並行リクエストとレート制限

```rust
use futures::stream::{self, StreamExt};
use reqwest::Client;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// 並行数を制限した一括リクエスト (Semaphore方式)
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

/// buffer_unordered を使った並行リクエスト (Stream方式)
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

/// レート制限付き連続リクエスト
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

        // 次のリクエストまで待機
        let elapsed = start.elapsed();
        if elapsed < interval {
            tokio::time::sleep(interval - elapsed).await;
        }
    }

    results
}
```

### 2.6 マルチパートファイルアップロード

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
    // ファイル読み込み
    let mut file = File::open(file_path).await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;

    let file_name = std::path::Path::new(file_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("upload")
        .to_string();

    // MIME タイプ推定
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

/// 複数ファイルの同時アップロード
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

### 2.7 リトライ付きHTTPクライアント

```rust
use reqwest::{Client, Response, StatusCode};
use std::time::Duration;

/// 指数バックオフ付きリトライ
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
            // ジッタを追加 (0.5 ~ 1.5 倍)
            let jitter = rand::random::<f64>() + 0.5;
            let actual_delay = Duration::from_millis(
                (delay.as_millis() as f64 * jitter) as u64
            );
            tracing::warn!(
                "リトライ {}/{}: {}ms 後に再試行",
                attempt, max_retries, actual_delay.as_millis()
            );
            tokio::time::sleep(actual_delay).await;
        }

        match (request_fn)().await {
            Ok(response) => {
                let status = response.status();
                // リトライ可能なステータスコード
                if status == StatusCode::TOO_MANY_REQUESTS
                    || status == StatusCode::SERVICE_UNAVAILABLE
                    || status == StatusCode::GATEWAY_TIMEOUT
                    || status == StatusCode::BAD_GATEWAY
                {
                    last_error = Some(anyhow::anyhow!("リトライ可能なエラー: {}", status));
                    continue;
                }
                return Ok(response);
            }
            Err(e) => {
                if e.is_connect() || e.is_timeout() {
                    last_error = Some(anyhow::anyhow!("接続エラー: {}", e));
                    continue;
                }
                return Err(e.into());
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("最大リトライ回数超過")))
}

/// 使用例
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

## 3. hyper — 低レベルHTTPサーバー

### 3.1 hyper によるHTTPサーバー実装

reqwest が高レベルHTTPクライアントであるのに対し、hyper は HTTP プロトコル自体の実装を提供する。Axum は内部で hyper を使っているが、直接 hyper を使うケースもある。

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

### 3.2 hyper による HTTP/2 サーバー

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
            // HTTP/2 は h2c (cleartext) モードで起動
            // 本番では TLS が必要
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

### 3.3 Graceful Shutdown 付きサーバー

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

    println!("サーバー起動。Ctrl+C で終了");

    // Ctrl+C ハンドラ
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        signal::ctrl_c().await.unwrap();
        println!("\nシャットダウン開始...");
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

                    // graceful shutdown: 進行中のリクエストの完了を待つ
                    tokio::pin!(conn);
                    tokio::select! {
                        result = &mut conn => {
                            if let Err(e) = result {
                                eprintln!("Error serving {}: {}", addr, e);
                            }
                        }
                        _ = shutdown_rx.recv() => {
                            conn.as_mut().graceful_shutdown();
                            // 残りの処理を完了させる
                            if let Err(e) = conn.await {
                                eprintln!("Error during shutdown: {}", e);
                            }
                        }
                    }
                });
            }
            _ = shutdown_rx.recv() => {
                println!("新規接続の受付を停止");
                break;
            }
        }
    }

    println!("サーバー停止完了");
    Ok(())
}
```

---

## 4. WebSocket

### 4.1 WebSocket クライアント

```rust
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = "wss://echo.websocket.org";
    let (ws_stream, response) = connect_async(url).await?;
    println!("接続成功: {}", response.status());

    let (mut write, mut read) = ws_stream.split();

    // 送信タスク
    let send_task = tokio::spawn(async move {
        for i in 1..=5 {
            let msg = Message::Text(format!("メッセージ #{}", i));
            write.send(msg).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        write.send(Message::Close(None)).await.unwrap();
    });

    // 受信タスク
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => println!("受信: {}", text),
                Ok(Message::Binary(data)) => println!("バイナリ: {} bytes", data.len()),
                Ok(Message::Ping(data)) => println!("Ping: {:?}", data),
                Ok(Message::Close(_)) => {
                    println!("切断");
                    break;
                }
                Err(e) => {
                    eprintln!("エラー: {}", e);
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

### WebSocket のメッセージフロー

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
     │    全二重: 送受信を同時に実行可能    │
     │    split() で read/write を分離     │
     └────────────────────────────────────┘
```

### 4.2 自動再接続付き WebSocket クライアント

```rust
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use std::time::Duration;
use tokio::sync::mpsc;

/// 再接続ロジック付きWebSocketクライアント
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
                    println!("WebSocket 正常切断");
                    break;
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count > self.max_retries {
                        eprintln!("最大リトライ回数超過: {}", e);
                        break;
                    }

                    let delay = self.base_delay * 2u32.pow((retry_count - 1).min(6));
                    eprintln!(
                        "接続エラー ({}回目): {}。{}秒後に再接続...",
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
        println!("WebSocket 接続成功: {}", self.url);

        let (mut write, mut read) = ws_stream.split();

        loop {
            tokio::select! {
                // サーバーからのメッセージ受信
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if incoming_tx.send(text).await.is_err() {
                                break; // 受信側がドロップされた
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            write.send(Message::Pong(data)).await?;
                        }
                        Some(Ok(Message::Close(_))) => {
                            return Ok(()); // 正常切断
                        }
                        Some(Err(e)) => {
                            return Err(e.into());
                        }
                        None => {
                            return Err(anyhow::anyhow!("ストリーム終了"));
                        }
                        _ => {}
                    }
                }
                // アプリからの送信メッセージ
                msg = outgoing_rx.recv() => {
                    match msg {
                        Some(text) => {
                            write.send(Message::Text(text)).await?;
                        }
                        None => {
                            // 送信チャネルがクローズ
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

### 4.3 WebSocket サーバー (Axum統合)

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

/// 共有状態: ブロードキャストチャネルでチャットルームを実現
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

    // サーバー → クライアント (ブロードキャスト受信 → WebSocket送信)
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // クライアント → サーバー (WebSocket受信 → ブロードキャスト送信)
    let tx = state.tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                let _ = tx.send(text);
            }
        }
    });

    // どちらかのタスクが終了したら両方を停止
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

// WebSocket サーバーの起動
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

### 4.4 Heartbeat (定期Ping) 管理

```rust
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;
use std::time::Duration;

/// Heartbeat 付き WebSocket 接続管理
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
            // 受信メッセージの処理
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        println!("受信: {}", text);
                    }
                    Some(Ok(Message::Pong(_))) => {
                        last_pong = std::time::Instant::now();
                    }
                    Some(Ok(Message::Ping(data))) => {
                        write.send(Message::Pong(data)).await?;
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        println!("接続終了");
                        break;
                    }
                    Some(Err(e)) => {
                        eprintln!("エラー: {}", e);
                        break;
                    }
                    _ => {}
                }
            }

            // 定期 Heartbeat
            _ = heartbeat.tick() => {
                // Pong タイムアウトチェック
                if last_pong.elapsed() > pong_timeout {
                    eprintln!("Pong タイムアウト。接続切断");
                    break;
                }
                write.send(Message::Ping(vec![].into())).await?;
            }
        }
    }

    // クリーンな切断
    let _ = write.send(Message::Close(None)).await;
    Ok(())
}
```

---

## 5. gRPC — tonic

### 5.1 Protocol Buffers 定義

```protobuf
// proto/greeter.proto
syntax = "proto3";
package greeter;

service Greeter {
    // Unary RPC: 単一リクエスト → 単一レスポンス
    rpc SayHello (HelloRequest) returns (HelloReply);

    // Server Streaming RPC: 単一リクエスト → ストリームレスポンス
    rpc SayHelloStream (HelloRequest) returns (stream HelloReply);

    // Client Streaming RPC: ストリームリクエスト → 単一レスポンス
    rpc RecordGreetings (stream HelloRequest) returns (GreetingSummary);

    // Bidirectional Streaming RPC: ストリーム ↔ ストリーム
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

### 5.2 ビルドスクリプト

```rust
// build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/generated")   // 生成先 (任意)
        .compile_protos(
            &["proto/greeter.proto"],
            &["proto"],
        )?;
    Ok(())
}
```

### 5.3 4パターンの gRPC サーバー実装

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
        // メタデータからの情報取得
        let request_id = metadata
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown");
        println!("Unary RPC: request_id={}", request_id);

        let name = request.into_inner().name;
        if name.is_empty() {
            return Err(Status::invalid_argument("name は必須です"));
        }

        let reply = HelloReply {
            message: format!("こんにちは、{}!", name),
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
                    message: format!("{}回目: こんにちは、{}!", i, name),
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
            println!("Client Stream 受信: {}", req.name);
            names.push(req.name);
        }

        let summary = GreetingSummary {
            count: names.len() as i32,
            summary: format!("受信した名前: {}", names.join(", ")),
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
                        // エコーバック (実際にはブロードキャストなど)
                        let reply = ChatMessage {
                            user: "Server".to_string(),
                            content: format!("Echo: {}", msg.content),
                            timestamp: now_timestamp(),
                        };
                        if tx.send(Ok(reply)).await.is_err() { break; }
                    }
                    Err(e) => {
                        eprintln!("ストリームエラー: {}", e);
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
    println!("gRPC サーバー起動: {}", addr);

    Server::builder()
        .add_service(GreeterServer::new(MyGreeter::default()))
        .serve(addr)
        .await?;
    Ok(())
}
```

### 5.4 gRPC クライアント (全4パターン)

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
    // メタデータの付与
    request.metadata_mut().insert(
        "x-request-id",
        "req-12345".parse().unwrap(),
    );
    let response = client.say_hello(request).await?;
    println!("応答: {}", response.into_inner().message);

    // 2. Server Streaming RPC
    println!("\n=== Server Streaming RPC ===");
    let request = tonic::Request::new(HelloRequest {
        name: "World".into(),
    });
    let mut stream = client.say_hello_stream(request).await?.into_inner();
    while let Some(reply) = stream.next().await {
        let reply = reply?;
        println!("ストリーム: {} (ts={})", reply.message, reply.timestamp);
    }

    // 3. Client Streaming RPC
    println!("\n=== Client Streaming RPC ===");
    let names = vec!["Alice", "Bob", "Charlie"];
    let request_stream = tokio_stream::iter(
        names.into_iter().map(|name| HelloRequest { name: name.into() })
    );
    let response = client.record_greetings(request_stream).await?;
    let summary = response.into_inner();
    println!("サマリー: {} 件 — {}", summary.count, summary.summary);

    // 4. Bidirectional Streaming RPC
    println!("\n=== Bidirectional Streaming RPC ===");
    let (tx, rx) = tokio::sync::mpsc::channel(32);

    // 送信ストリーム
    tokio::spawn(async move {
        for i in 1..=3 {
            let msg = ChatMessage {
                user: "Client".to_string(),
                content: format!("メッセージ #{}", i),
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

### 5.5 gRPC 認証とインターセプター

```rust
use tonic::{transport::Channel, Request, Status};
use tonic::service::Interceptor;

/// 認証インターセプター
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
        // リクエストIDも付与
        let request_id = uuid::Uuid::new_v4().to_string();
        request.metadata_mut().insert(
            "x-request-id",
            request_id.parse().unwrap(),
        );
        Ok(request)
    }
}

/// サーバー側の認証チェック
fn check_auth(request: &Request<impl std::fmt::Debug>) -> Result<String, Status> {
    let token = request.metadata()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| Status::unauthenticated("認証トークンなし"))?;

    let token = token
        .strip_prefix("Bearer ")
        .ok_or_else(|| Status::unauthenticated("Bearer トークン形式不正"))?;

    // トークン検証 (実際にはJWT検証など)
    if token == "valid-token" {
        Ok("user-123".to_string())
    } else {
        Err(Status::unauthenticated("無効なトークン"))
    }
}

// 使用例
// let channel = Channel::from_static("http://[::1]:50051").connect().await?;
// let interceptor = AuthInterceptor { token: "valid-token".into() };
// let client = GreeterClient::with_interceptor(channel, interceptor);
```

### 5.6 gRPC サーバー — TLS と健全性チェック

```rust
use tonic::transport::{Server, ServerTlsConfig, Identity};
use tonic_health::server::HealthReporter;

async fn start_grpc_server() -> Result<(), Box<dyn std::error::Error>> {
    // TLS 設定
    let cert = tokio::fs::read("server.pem").await?;
    let key = tokio::fs::read("server.key").await?;
    let identity = Identity::from_pem(cert, key);

    let tls_config = ServerTlsConfig::new().identity(identity);

    // Health check サービス
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

### gRPC の4つの通信パターン

```
┌─────────────────────────────────────────────────────────┐
│                 gRPC 通信パターン                        │
│                                                         │
│  1. Unary (単一 → 単一)                                 │
│     Client ──── Request ────→ Server                    │
│     Client ←─── Response ──── Server                    │
│                                                         │
│  2. Server Streaming (単一 → ストリーム)                  │
│     Client ──── Request ────→ Server                    │
│     Client ←─── Response 1 ── Server                    │
│     Client ←─── Response 2 ── Server                    │
│     Client ←─── Response N ── Server                    │
│                                                         │
│  3. Client Streaming (ストリーム → 単一)                  │
│     Client ──── Request 1 ──→ Server                    │
│     Client ──── Request 2 ──→ Server                    │
│     Client ──── Request N ──→ Server                    │
│     Client ←─── Response ──── Server                    │
│                                                         │
│  4. Bidirectional Streaming (ストリーム ↔ ストリーム)     │
│     Client ──── Request 1 ──→ Server                    │
│     Client ←─── Response 1 ── Server                    │
│     Client ──── Request 2 ──→ Server                    │
│     Client ←─── Response 2 ── Server                    │
│     (リクエストとレスポンスは独立して流れる)               │
└─────────────────────────────────────────────────────────┘
```

---

## 6. DNS解決とTLS

### 6.1 カスタムDNSリゾルバ

```rust
use reqwest::dns::{Resolve, Resolving, Name};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::collections::HashMap;

/// 静的DNS解決 (テスト用やサービスディスカバリ向け)
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
            // フォールバック: システムDNSを使用
            Box::pin(async move {
                let addrs = tokio::net::lookup_host(format!("{}:0", host))
                    .await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;
                Ok(Box::new(addrs) as Box<dyn Iterator<Item = SocketAddr> + Send>)
            })
        }
    }
}

// 使用例
// let mut resolver = StaticResolver::new();
// resolver.add_override("api.local", IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
// let client = reqwest::Client::builder()
//     .dns_resolver(Arc::new(resolver))
//     .build()?;
```

### 6.2 TLS設定のカスタマイズ

```rust
use reqwest::Client;
use std::time::Duration;

/// rustls を使ったセキュアなクライアント構築
fn build_secure_client() -> reqwest::Result<Client> {
    // カスタムCA証明書の追加
    let ca_cert = std::fs::read("ca.pem").expect("CA証明書の読み込み失敗");
    let ca = reqwest::Certificate::from_pem(&ca_cert)?;

    // クライアント証明書 (mTLS)
    let client_cert = std::fs::read("client.pem").expect("クライアント証明書の読み込み失敗");
    let client_key = std::fs::read("client.key").expect("クライアント鍵の読み込み失敗");
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

/// 自己署名証明書を許可するクライアント (開発環境のみ)
fn build_dev_client() -> reqwest::Result<Client> {
    Client::builder()
        .danger_accept_invalid_certs(true)  // 本番では絶対に使わない!
        .timeout(Duration::from_secs(30))
        .build()
}
```

---

## 7. 接続プールと接続管理

### 接続プールの動作原理

```
┌──────────────────── 接続プール管理 ────────────────────┐
│                                                         │
│  reqwest::Client                                        │
│  ┌────────────────────────────────────────────────┐    │
│  │  Connection Pool                                │    │
│  │                                                 │    │
│  │  api.example.com:443                           │    │
│  │  ├── Conn #1 [IDLE]     ← 再利用可能           │    │
│  │  ├── Conn #2 [ACTIVE]   ← リクエスト処理中     │    │
│  │  └── Conn #3 [IDLE]     ← 再利用可能           │    │
│  │                                                 │    │
│  │  db.example.com:5432                           │    │
│  │  ├── Conn #1 [ACTIVE]                          │    │
│  │  └── Conn #2 [IDLE]                            │    │
│  │                                                 │    │
│  │  設定:                                          │    │
│  │  pool_max_idle_per_host = 10                    │    │
│  │  pool_idle_timeout = 90s                        │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  新しいリクエスト:                                       │
│  1. プールからアイドル接続を取得 (あれば)               │
│  2. なければ新規TCP接続 + TLSハンドシェイク             │
│  3. レスポンス受信後、接続をプールに返却                 │
│  4. アイドルタイムアウト後に自動切断                     │
└─────────────────────────────────────────────────────────┘
```

### 接続プール最適化の指針

```rust
use reqwest::Client;
use std::time::Duration;

/// ユースケース別の Client 設定
mod pool_configs {
    use super::*;

    /// 高スループット向け (大量の並行リクエスト)
    pub fn high_throughput() -> Client {
        Client::builder()
            .pool_max_idle_per_host(50)
            .pool_idle_timeout(Duration::from_secs(120))
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true)
            .build()
            .unwrap()
    }

    /// 低レイテンシ向け (少数のクリティカルリクエスト)
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

    /// メモリ節約向け (リソース制約環境)
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

## 8. 比較表

### 通信プロトコル比較

| 特性 | REST (HTTP) | WebSocket | gRPC |
|---|---|---|---|
| プロトコル | HTTP/1.1, HTTP/2 | WebSocket over TCP | HTTP/2 |
| データ形式 | JSON / XML | テキスト / バイナリ | Protocol Buffers |
| 通信方向 | リクエスト-レスポンス | 双方向 | 4パターン (Unary/Stream) |
| スキーマ | OpenAPI (任意) | なし | .proto (必須) |
| パフォーマンス | 中 | 高 | 非常に高い |
| ブラウザ対応 | 完全 | 完全 | grpc-web 経由 |
| ユースケース | CRUD API | チャット、リアルタイム | マイクロサービス間 |
| シリアライズ速度 | 遅い (テキスト) | N/A | 速い (バイナリ) |
| ペイロードサイズ | 大きい | 可変 | 小さい |
| コード生成 | 任意 | なし | 必須 |
| エラーハンドリング | HTTPステータスコード | アプリ定義 | Status + Code |
| ストリーミング | SSE / chunked | ネイティブ | ネイティブ |
| ロードバランシング | L7 (HTTP) | L4/L7 | L7 (HTTP/2) |

### Rust HTTP クライアントライブラリ比較

| ライブラリ | レベル | 用途 | 特徴 | 非同期 |
|---|---|---|---|---|
| reqwest | 高レベル | REST API 呼び出し | 簡潔、Cookie/リダイレクト自動 | Yes |
| hyper | 低レベル | カスタムHTTP実装 | 高性能、フルコントロール | Yes |
| ureq | 同期 | 同期HTTPクライアント | シンプル、async不要 | No |
| surf | 中レベル | async-std 向け | ミドルウェア対応 | Yes |
| isahc | 中レベル | curl ベース | HTTP/2、接続プール | Yes |
| attohttpc | 軽量 | 組み込み向け | 依存少、小バイナリ | No |

### WebSocket ライブラリ比較

| ライブラリ | ランタイム | レベル | 特徴 |
|---|---|---|---|
| tokio-tungstenite | tokio | 中レベル | tungstenite のtokio統合。最も広く使われる |
| tungstenite | sync | 低レベル | 同期/非同期両対応の基盤ライブラリ |
| axum::extract::ws | tokio | 高レベル | Axum組み込み。サーバー向け |
| warp::ws | tokio | 高レベル | Warp組み込み。サーバー向け |

### gRPC ライブラリ比較

| ライブラリ | 特徴 | 用途 |
|---|---|---|
| tonic | Pure Rust。tokio ベース | 推奨。最も成熟 |
| grpc-rs | C++ gRPC バインディング | 特殊要件時のみ |
| volo-grpc | CloudWeGo 提供 | 高性能マイクロサービス |

---

## 9. アンチパターン

### アンチパターン1: Client のリクエスト毎生成

```rust
// NG: 毎回 Client を作成 → 接続プール未活用、TLSハンドシェイクが毎回発生
async fn bad_fetch(url: &str) -> reqwest::Result<String> {
    let client = reqwest::Client::new(); // 毎回生成
    client.get(url).send().await?.text().await
}

// OK: Client をアプリケーション全体で共有
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

### アンチパターン2: WebSocket の Ping/Pong 未対応

```rust
// NG: Ping を無視 → サーバーが接続をタイムアウトで切断
while let Some(msg) = read.next().await {
    if let Ok(Message::Text(text)) = msg {
        process(text);
    }
    // Ping を無視!
}

// OK: Ping に自動応答
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

### アンチパターン3: タイムアウト未設定

```rust
// NG: タイムアウトなし → 応答しないサーバーで永久にブロック
let response = reqwest::get("https://slow-server.example.com").await?;

// OK: 必ずタイムアウトを設定
let client = reqwest::Client::builder()
    .timeout(std::time::Duration::from_secs(30))
    .connect_timeout(std::time::Duration::from_secs(5))
    .build()?;
let response = client.get("https://slow-server.example.com").send().await?;
```

### アンチパターン4: error_for_status() の呼び忘れ

```rust
// NG: 4xx/5xx でもエラーにならない → サイレントに失敗
let body: MyData = client.get(url).send().await?.json().await?;
// 404 のHTML をデシリアライズしようとしてパニック!

// OK: error_for_status() で明示的にチェック
let body: MyData = client.get(url)
    .send()
    .await?
    .error_for_status()?  // 4xx/5xx → Err に変換
    .json()
    .await?;
```

### アンチパターン5: gRPC のエラーステータス不適切な使用

```rust
// NG: 全てのエラーを Internal にする
impl Greeter for MyGreeter {
    async fn say_hello(&self, req: Request<HelloRequest>) -> Result<Response<HelloReply>, Status> {
        let name = req.into_inner().name;
        do_something(&name).map_err(|e| Status::internal(e.to_string()))?; // 全部 Internal!
        todo!()
    }
}

// OK: 適切なステータスコードを使い分ける
impl Greeter for MyGreeter {
    async fn say_hello(&self, req: Request<HelloRequest>) -> Result<Response<HelloReply>, Status> {
        let name = req.into_inner().name;
        if name.is_empty() {
            return Err(Status::invalid_argument("name は空にできません"));
        }
        let user = find_user(&name).await
            .map_err(|_| Status::not_found(format!("ユーザー '{}' が見つかりません", name)))?;
        let result = process(user).await
            .map_err(|e| Status::internal(format!("処理エラー: {}", e)))?;
        todo!()
    }
}
```

### gRPC ステータスコード一覧

| コード | 名前 | 用途 |
|---|---|---|
| 0 | OK | 成功 |
| 1 | CANCELLED | クライアントがキャンセル |
| 2 | UNKNOWN | 不明なエラー |
| 3 | INVALID_ARGUMENT | 不正な引数 |
| 4 | DEADLINE_EXCEEDED | タイムアウト |
| 5 | NOT_FOUND | リソース未発見 |
| 6 | ALREADY_EXISTS | リソース重複 |
| 7 | PERMISSION_DENIED | 権限不足 |
| 8 | RESOURCE_EXHAUSTED | リソース枯渇 (レート制限) |
| 9 | FAILED_PRECONDITION | 前提条件不成立 |
| 10 | ABORTED | 中断 (競合など) |
| 11 | OUT_OF_RANGE | 範囲外 |
| 12 | UNIMPLEMENTED | 未実装 |
| 13 | INTERNAL | 内部エラー |
| 14 | UNAVAILABLE | サービス一時的に利用不可 |
| 16 | UNAUTHENTICATED | 認証失敗 |

---

## 10. 実践パターン

### 10.1 API クライアントラッパー

```rust
use reqwest::Client;
use serde::{de::DeserializeOwned, Serialize};
use std::time::Duration;

/// 型安全なAPIクライアント
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

// 使用例:
// let api = ApiClient::new("https://api.example.com", "my-key")?;
// let users: Vec<User> = api.get("/users").await?;
// let new_user: User = api.post("/users", &CreateUser { name: "Alice".into() }).await?;
```

### 10.2 Server-Sent Events (SSE) クライアント

```rust
use futures::StreamExt;
use reqwest::Client;

#[derive(Debug)]
struct SseEvent {
    event_type: String,
    data: String,
    id: Option<String>,
}

/// SSE ストリームの受信
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

        // イベント区切り (空行) で分割
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

### 10.3 TCP ソケット直接操作

```rust
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, AsyncBufReadExt};

/// シンプルなエコーサーバー
async fn echo_server(addr: &str) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    println!("Echo サーバー起動: {}", addr);

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        println!("接続: {}", peer_addr);

        tokio::spawn(async move {
            if let Err(e) = handle_echo_client(stream).await {
                eprintln!("クライアントエラー ({}): {}", peer_addr, e);
            }
            println!("切断: {}", peer_addr);
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
        if n == 0 { break; } // 接続切断

        writer.write_all(line.as_bytes()).await?;
        writer.flush().await?;
    }

    Ok(())
}

/// TCP クライアント
async fn echo_client(addr: &str) -> anyhow::Result<()> {
    let mut stream = TcpStream::connect(addr).await?;
    println!("サーバーに接続: {}", addr);

    stream.write_all(b"Hello, server!\n").await?;

    let mut buf = vec![0u8; 1024];
    let n = stream.read(&mut buf).await?;
    println!("応答: {}", String::from_utf8_lossy(&buf[..n]));

    Ok(())
}
```

---

## FAQ

### Q1: reqwest と hyper のどちらを使うべき?

**A:** 一般的なREST API呼び出しなら reqwest で十分です。カスタムプロトコル実装やフレームワーク構築など HTTP の低レベル制御が必要な場合のみ hyper を直接使います。Axum は内部で hyper を使っていますが、直接触る必要はほぼありません。判断基準は以下の通りです:

- **reqwest を使う場合**: REST API クライアント、ファイルダウンロード、Webhook 送信、外部サービス連携
- **hyper を使う場合**: カスタムプロキシサーバー、独自プロトコルのHTTP拡張、極限のパフォーマンスチューニング

### Q2: gRPC で認証はどう実装する?

**A:** tonic の `Interceptor` でメタデータに Bearer トークンを付与します。サーバー側ではリクエストのメタデータからトークンを検証します。JWT トークンの場合は `jsonwebtoken` クレートと組み合わせるのが一般的です。

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

### Q3: HTTP/2 のメリットは?

**A:** 単一TCP接続上で複数リクエストを多重化でき、ヘッダー圧縮 (HPACK) によりオーバーヘッドを削減します。gRPC は HTTP/2 を前提としており、ストリーミング RPC も HTTP/2 のフレーム機能で実現されています。具体的なメリットは:

- **多重化**: 1つのTCP接続で複数のリクエスト/レスポンスを同時処理
- **ヘッダー圧縮**: HPACK によりヘッダーサイズを大幅に削減
- **サーバープッシュ**: サーバーからクライアントへの事前送信
- **ストリーム優先度**: 重要なリクエストを優先的に処理
- **フロー制御**: ストリームごとのフロー制御

### Q4: WebSocket と SSE (Server-Sent Events) のどちらを使うべき?

**A:** 用途に応じて使い分けます:

- **WebSocket**: 双方向通信が必要な場合（チャット、ゲーム、コラボレーション）
- **SSE**: サーバーからの一方向通知で十分な場合（ダッシュボード更新、ログストリーミング）

SSE は HTTP/1.1 上で動作し、実装が簡単で自動再接続機能があります。一方 WebSocket は双方向通信が可能でバイナリデータも送れますが、実装が複雑になります。

### Q5: reqwest で Cookie を管理するには?

**A:** `cookie_store(true)` を有効にすると、セッションCookieが自動管理されます。

```rust
let client = reqwest::Client::builder()
    .cookie_store(true)
    .build()?;

// ログインリクエスト
client.post("https://example.com/login")
    .json(&credentials)
    .send()
    .await?;

// 以降のリクエストには自動でCookieが付与される
let profile = client.get("https://example.com/profile")
    .send()
    .await?;
```

### Q6: gRPC のデッドラインとタイムアウトの違いは?

**A:** デッドライン（Deadline）は「この時刻までに完了」、タイムアウトは「この時間内に完了」です。gRPC では内部的にデッドラインに変換されます。デッドラインはサービス間で伝播するため、マイクロサービスチェーン全体で一貫したタイムアウト管理ができます。

```rust
use std::time::Duration;

let mut request = tonic::Request::new(HelloRequest { name: "test".into() });
request.set_timeout(Duration::from_secs(5)); // タイムアウトを5秒に設定
```

### Q7: 大量の並行HTTPリクエストでのベストプラクティスは?

**A:** 以下の3つの原則を守ります:

1. **Client の共有**: 1つの `reqwest::Client` を全リクエストで共有し、接続プールを活用
2. **並行数の制限**: `Semaphore` や `buffer_unordered` で同時リクエスト数を制御
3. **タイムアウトの設定**: 個別リクエストとClient全体の両方でタイムアウトを設定

```rust
// 推奨パターン
let results = stream::iter(urls)
    .map(|url| {
        let client = client.clone();
        async move { client.get(&url).send().await?.text().await }
    })
    .buffer_unordered(10) // 最大10並行
    .collect::<Vec<_>>()
    .await;
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| reqwest | 高レベル HTTP クライアント。Client は再利用必須。接続プール内蔵 |
| hyper | 低レベル HTTP。フレームワーク構築やカスタムプロキシ用 |
| WebSocket | tokio-tungstenite で双方向通信。split() で送受信分離。Ping/Pong必須 |
| gRPC | tonic + .proto。4つの通信パターン。高性能マイクロサービス通信 |
| 接続プール | Client を static/Arc で共有して接続プール活用。ホスト別に設定可能 |
| ストリーミング | bytes_stream() / Server Streaming で逐次処理。メモリ効率が高い |
| Ping/Pong | WebSocket の keepalive に必須。タイムアウト検出にも活用 |
| TLS | rustls がデフォルト推奨。mTLS はクライアント証明書で実現 |
| リトライ | 指数バックオフ + ジッタ。リトライ可能なステータスコードを判別 |
| エラー | error_for_status() の使用必須。gRPC は適切なステータスコードを使い分け |

## 次に読むべきガイド

- [Axum](./04-axum-web.md) — Webフレームワークでのサーバーサイド実装
- [Serde](../04-ecosystem/02-serde.md) — JSON/Protobuf のシリアライズ
- [非同期パターン](./02-async-patterns.md) — リトライ、並行制限の実装

## 参考文献

1. **reqwest documentation**: https://docs.rs/reqwest/latest/reqwest/
2. **hyper documentation**: https://docs.rs/hyper/latest/hyper/
3. **tokio-tungstenite**: https://docs.rs/tokio-tungstenite/latest/tokio_tungstenite/
4. **tonic (gRPC for Rust)**: https://docs.rs/tonic/latest/tonic/
5. **tonic-health**: https://docs.rs/tonic-health/latest/tonic_health/
6. **tower-http**: https://docs.rs/tower-http/latest/tower_http/
7. **rustls**: https://docs.rs/rustls/latest/rustls/
