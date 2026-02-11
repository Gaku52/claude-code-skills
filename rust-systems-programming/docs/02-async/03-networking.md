# ネットワーク — reqwest/hyper、WebSocket、tonic

> Rust の非同期ネットワークスタックとして HTTP クライアント/サーバー、WebSocket、gRPC の実装パターンを習得する

## この章で学ぶこと

1. **HTTPクライアント** — reqwest による REST API 呼び出しと接続プール管理
2. **WebSocket** — tokio-tungstenite によるリアルタイム双方向通信
3. **gRPC** — tonic による Protocol Buffers ベースの高性能 RPC

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

---

## 2. reqwest — HTTPクライアント

### コード例1: 基本的なHTTPリクエスト

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct User {
    id: u64,
    login: String,
    name: Option<String>,
}

#[derive(Serialize)]
struct CreateIssue {
    title: String,
    body: String,
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
    };
    let response = client
        .post("https://api.example.com/issues")
        .bearer_auth("your-token")
        .json(&issue)
        .send()
        .await?;
    println!("Status: {}", response.status());

    Ok(())
}
```

### コード例2: ストリーミングダウンロード

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
```

---

## 3. WebSocket

### コード例3: WebSocket クライアント

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

---

## 4. gRPC — tonic

### コード例4: Protocol Buffers 定義とサーバー

```protobuf
// proto/greeter.proto
syntax = "proto3";
package greeter;

service Greeter {
    rpc SayHello (HelloRequest) returns (HelloReply);
    rpc SayHelloStream (HelloRequest) returns (stream HelloReply);
}

message HelloRequest {
    string name = 1;
}

message HelloReply {
    string message = 1;
}
```

```rust
// build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/greeter.proto")?;
    Ok(())
}
```

```rust
// src/server.rs
use tonic::{transport::Server, Request, Response, Status};
use greeter::greeter_server::{Greeter, GreeterServer};
use greeter::{HelloReply, HelloRequest};
use tokio_stream::wrappers::ReceiverStream;

pub mod greeter {
    tonic::include_proto!("greeter");
}

#[derive(Debug, Default)]
pub struct MyGreeter;

#[tonic::async_trait]
impl Greeter for MyGreeter {
    async fn say_hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloReply>, Status> {
        let name = request.into_inner().name;
        let reply = HelloReply {
            message: format!("こんにちは、{}!", name),
        };
        Ok(Response::new(reply))
    }

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
                };
                if tx.send(Ok(reply)).await.is_err() { break; }
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    Server::builder()
        .add_service(GreeterServer::new(MyGreeter::default()))
        .serve(addr)
        .await?;
    Ok(())
}
```

### コード例5: gRPC クライアント

```rust
use greeter::greeter_client::GreeterClient;
use greeter::HelloRequest;

pub mod greeter {
    tonic::include_proto!("greeter");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = GreeterClient::connect("http://[::1]:50051").await?;

    // Unary RPC
    let request = tonic::Request::new(HelloRequest {
        name: "Rust".into(),
    });
    let response = client.say_hello(request).await?;
    println!("応答: {}", response.into_inner().message);

    // Server Streaming RPC
    let request = tonic::Request::new(HelloRequest {
        name: "World".into(),
    });
    let mut stream = client.say_hello_stream(request).await?.into_inner();
    while let Some(reply) = stream.message().await? {
        println!("ストリーム: {}", reply.message);
    }

    Ok(())
}
```

---

## 5. 比較表

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

### Rust HTTP クライアントライブラリ比較

| ライブラリ | レベル | 用途 | 特徴 |
|---|---|---|---|
| reqwest | 高レベル | REST API 呼び出し | 簡潔、Cookie/リダイレクト自動 |
| hyper | 低レベル | カスタムHTTP実装 | 高性能、フルコントロール |
| ureq | 同期 | 同期HTTPクライアント | シンプル、async不要 |
| surf | 中レベル | async-std 向け | ミドルウェア対応 |

---

## 6. アンチパターン

### アンチパターン1: Client のリクエスト毎生成

```rust
// NG: 毎回 Client を作成 → 接続プール未活用、TLSハンドシェイクが毎回発生
async fn bad_fetch(url: &str) -> reqwest::Result<String> {
    let client = reqwest::Client::new(); // 毎回生成
    client.get(url).send().await?.text().await
}

// OK: Client をアプリケーション全体で共有
use once_cell::sync::Lazy;
static HTTP_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
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

---

## FAQ

### Q1: reqwest と hyper のどちらを使うべき?

**A:** 一般的なREST API呼び出しなら reqwest で十分です。カスタムプロトコル実装やフレームワーク構築など HTTP の低レベル制御が必要な場合のみ hyper を直接使います。Axum は内部で hyper を使っていますが、直接触る必要はほぼありません。

### Q2: gRPC で認証はどう実装する?

**A:** tonic の `Interceptor` でメタデータに Bearer トークンを付与します。

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

**A:** 単一TCP接続上で複数リクエストを多重化でき、ヘッダー圧縮 (HPACK) によりオーバーヘッドを削減します。gRPC は HTTP/2 を前提としており、ストリーミング RPC も HTTP/2 のフレーム機能で実現されています。

---

## まとめ

| 項目 | 要点 |
|---|---|
| reqwest | 高レベル HTTP クライアント。Client は再利用必須 |
| hyper | 低レベル HTTP。フレームワーク構築用 |
| WebSocket | tokio-tungstenite で双方向通信。split() で送受信分離 |
| gRPC | tonic + .proto。高性能マイクロサービス通信 |
| 接続プール | Client を static/Arc で共有して接続プール活用 |
| ストリーミング | bytes_stream() / Server Streaming で逐次処理 |
| Ping/Pong | WebSocket の keepalive に必須 |

## 次に読むべきガイド

- [Axum](./04-axum-web.md) — Webフレームワークでのサーバーサイド実装
- [Serde](../04-ecosystem/02-serde.md) — JSON/Protobuf のシリアライズ
- [非同期パターン](./02-async-patterns.md) — リトライ、並行制限の実装

## 参考文献

1. **reqwest documentation**: https://docs.rs/reqwest/latest/reqwest/
2. **tokio-tungstenite**: https://docs.rs/tokio-tungstenite/latest/tokio_tungstenite/
3. **tonic (gRPC for Rust)**: https://docs.rs/tonic/latest/tonic/
