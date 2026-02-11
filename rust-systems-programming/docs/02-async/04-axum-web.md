# Axum — ルーティング、ミドルウェア、状態管理

> Axum フレームワークを使った型安全な Web API 構築手法を、ルーティング・ミドルウェア・状態管理を軸に習得する

## この章で学ぶこと

1. **ルーティングとハンドラ** — Extractor パターンによる型安全なリクエスト解析
2. **ミドルウェア** — Tower レイヤーとカスタムミドルウェアの実装
3. **状態管理** — AppState の共有、データベース接続プール統合

---

## 1. Axum アーキテクチャ

```
┌─────────────────── Axum 処理フロー ──────────────────┐
│                                                       │
│  HTTP Request                                         │
│    │                                                  │
│    ▼                                                  │
│  ┌────────────────────────────────┐                   │
│  │  Tower Middleware Stack        │                   │
│  │  ┌─ TraceLayer ─────────────┐ │                   │
│  │  │ ┌─ CorsLayer ─────────┐ │ │                   │
│  │  │ │ ┌─ AuthLayer ─────┐ │ │ │                   │
│  │  │ │ │                  │ │ │ │                   │
│  │  │ │ │  Router          │ │ │ │                   │
│  │  │ │ │   ├─ /api/users  │ │ │ │                   │
│  │  │ │ │   ├─ /api/posts  │ │ │ │                   │
│  │  │ │ │   └─ /health     │ │ │ │                   │
│  │  │ │ │                  │ │ │ │                   │
│  │  │ │ │  Extractors:     │ │ │ │                   │
│  │  │ │ │  Path, Query,    │ │ │ │                   │
│  │  │ │ │  Json, State     │ │ │ │                   │
│  │  │ │ └──────────────────┘ │ │ │                   │
│  │  │ └──────────────────────┘ │ │                   │
│  │  └──────────────────────────┘ │                   │
│  └────────────────────────────────┘                   │
│    │                                                  │
│    ▼                                                  │
│  HTTP Response                                        │
└───────────────────────────────────────────────────────┘
```

---

## 2. ルーティングとハンドラ

### コード例1: 基本的なCRUD API

```rust
use axum::{
    extract::{Path, Query, State, Json},
    http::StatusCode,
    routing::{get, post, put, delete},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}

#[derive(Deserialize)]
struct CreateUser {
    name: String,
    email: String,
}

#[derive(Deserialize)]
struct ListParams {
    page: Option<u32>,
    limit: Option<u32>,
}

type AppState = Arc<RwLock<Vec<User>>>;

// GET /users?page=1&limit=10
async fn list_users(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> Json<Vec<User>> {
    let users = state.read().await;
    let page = params.page.unwrap_or(1) as usize;
    let limit = params.limit.unwrap_or(10) as usize;
    let start = (page - 1) * limit;
    let slice: Vec<User> = users.iter().skip(start).take(limit).cloned().collect();
    Json(slice)
}

// GET /users/:id
async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<Json<User>, StatusCode> {
    let users = state.read().await;
    users.iter()
        .find(|u| u.id == id)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

// POST /users
async fn create_user(
    State(state): State<AppState>,
    Json(input): Json<CreateUser>,
) -> (StatusCode, Json<User>) {
    let mut users = state.write().await;
    let id = users.len() as u64 + 1;
    let user = User { id, name: input.name, email: input.email };
    users.push(user.clone());
    (StatusCode::CREATED, Json(user))
}

// DELETE /users/:id
async fn delete_user(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> StatusCode {
    let mut users = state.write().await;
    let len_before = users.len();
    users.retain(|u| u.id != id);
    if users.len() < len_before {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

#[tokio::main]
async fn main() {
    let state: AppState = Arc::new(RwLock::new(Vec::new()));

    let app = Router::new()
        .route("/users", get(list_users).post(create_user))
        .route("/users/{id}", get(get_user).delete(delete_user))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("サーバー起動: http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
```

### コード例2: カスタム Extractor

```rust
use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{request::Parts, StatusCode, header},
};

/// Bearer トークンを抽出するカスタム Extractor
struct BearerToken(String);

#[async_trait]
impl<S> FromRequestParts<S> for BearerToken
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        let auth_header = parts.headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or((StatusCode::UNAUTHORIZED, "Authorization ヘッダーなし"))?;

        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or((StatusCode::UNAUTHORIZED, "Bearer トークン形式が不正"))?;

        Ok(BearerToken(token.to_string()))
    }
}

// ハンドラで使用
async fn protected_handler(
    BearerToken(token): BearerToken,
) -> String {
    format!("認証済み。トークン: {}...", &token[..8.min(token.len())])
}
```

---

## 3. ミドルウェア

### コード例3: Tower レイヤーによるミドルウェアスタック

```rust
use axum::{Router, routing::get, middleware};
use tower_http::{
    cors::{CorsLayer, Any},
    trace::TraceLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
    limit::RequestBodyLimitLayer,
};
use std::time::Duration;

fn create_router() -> Router {
    Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/api/data", get(handler))
        // ミドルウェアは下から上に適用される
        .layer(CompressionLayer::new())               // レスポンス圧縮
        .layer(RequestBodyLimitLayer::new(1024 * 1024)) // 1MB制限
        .layer(TimeoutLayer::new(Duration::from_secs(30))) // タイムアウト
        .layer(CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any))
        .layer(TraceLayer::new_for_http())             // リクエストログ
}

async fn handler() -> &'static str {
    "data"
}
```

### ミドルウェア適用順序

```
┌──────────────────────────────────────────┐
│        リクエスト処理の順序               │
│                                          │
│  Request → TraceLayer (最後に追加)       │
│          → CorsLayer                     │
│          → TimeoutLayer                  │
│          → RequestBodyLimitLayer         │
│          → CompressionLayer (最初に追加)  │
│          → Handler                       │
│                                          │
│  Response ← 逆順で返る                   │
│                                          │
│  ※ .layer() の呼び出し順序と             │
│    実行順序は逆になる                     │
└──────────────────────────────────────────┘
```

### コード例4: カスタムミドルウェア

```rust
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
    http::StatusCode,
};
use std::time::Instant;

/// リクエストの処理時間を計測するミドルウェア
async fn timing_middleware(
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = Instant::now();

    let response = next.run(request).await;

    let elapsed = start.elapsed();
    tracing::info!(
        "{} {} → {} ({:?})",
        method, uri, response.status(), elapsed
    );

    response
}

/// API キー認証ミドルウェア
async fn api_key_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let api_key = request.headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok());

    match api_key {
        Some(key) if key == "secret-key-123" => {
            Ok(next.run(request).await)
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

// 適用
// let app = Router::new()
//     .route("/api/data", get(handler))
//     .layer(axum::middleware::from_fn(timing_middleware))
//     .layer(axum::middleware::from_fn(api_key_middleware));
```

---

## 4. 状態管理とDB統合

### コード例5: SQLx + Axum 統合

```rust
use axum::{
    extract::{Path, State, Json},
    http::StatusCode,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

#[derive(Debug, Serialize, sqlx::FromRow)]
struct Todo {
    id: i64,
    title: String,
    completed: bool,
}

#[derive(Deserialize)]
struct CreateTodo {
    title: String,
}

#[derive(Clone)]
struct AppState {
    db: PgPool,
}

async fn list_todos(
    State(state): State<AppState>,
) -> Result<Json<Vec<Todo>>, StatusCode> {
    let todos = sqlx::query_as::<_, Todo>("SELECT id, title, completed FROM todos")
        .fetch_all(&state.db)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(todos))
}

async fn create_todo(
    State(state): State<AppState>,
    Json(input): Json<CreateTodo>,
) -> Result<(StatusCode, Json<Todo>), StatusCode> {
    let todo = sqlx::query_as::<_, Todo>(
        "INSERT INTO todos (title, completed) VALUES ($1, false) RETURNING id, title, completed"
    )
    .bind(&input.title)
    .fetch_one(&state.db)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((StatusCode::CREATED, Json(todo)))
}

async fn toggle_todo(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<Todo>, StatusCode> {
    let todo = sqlx::query_as::<_, Todo>(
        "UPDATE todos SET completed = NOT completed WHERE id = $1 RETURNING id, title, completed"
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(todo))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::init();

    let pool = PgPoolOptions::new()
        .max_connections(20)
        .connect(&std::env::var("DATABASE_URL")?)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;

    let state = AppState { db: pool };

    let app = Router::new()
        .route("/todos", get(list_todos).post(create_todo))
        .route("/todos/{id}/toggle", post(toggle_todo))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

### コード例6: エラーハンドリング

```rust
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// アプリケーション統一エラー型
enum AppError {
    NotFound(String),
    BadRequest(String),
    Internal(anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::Internal(e) => {
                tracing::error!("内部エラー: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "内部サーバーエラー".into())
            }
        };

        let body = Json(json!({
            "error": {
                "status": status.as_u16(),
                "message": message,
            }
        }));

        (status, body).into_response()
    }
}

// anyhow::Error から自動変換
impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Internal(e)
    }
}

// ハンドラの戻り値型
async fn handler() -> Result<Json<serde_json::Value>, AppError> {
    let data = fetch_data().await.map_err(AppError::Internal)?;
    Ok(Json(data))
}

async fn fetch_data() -> anyhow::Result<serde_json::Value> {
    Ok(json!({"status": "ok"}))
}
```

---

## 5. 比較表

### Rust Web フレームワーク比較

| 特性 | Axum | Actix-web | Rocket | Warp |
|---|---|---|---|---|
| ベース | hyper + Tower | 独自アクターシステム | 独自 | hyper |
| 型安全性 | 非常に高い | 高い | 非常に高い | 高い |
| パフォーマンス | 非常に高い | 非常に高い | 高い | 非常に高い |
| エコシステム | Tower 互換 | 独自 | 独自 | 独自 |
| 学習コスト | 中 | 中 | 低 | 中〜高 |
| 採用トレンド | 急成長中 | 安定 | 安定 | やや減少 |

### Extractor 一覧

| Extractor | 用途 | 例 |
|---|---|---|
| `Path<T>` | URLパスパラメータ | `/users/{id}` → `Path(id): Path<u64>` |
| `Query<T>` | クエリ文字列 | `?page=1` → `Query(p): Query<Params>` |
| `Json<T>` | JSONリクエストボディ | `Json(body): Json<CreateUser>` |
| `State<T>` | 共有状態 | DB プール、設定 |
| `HeaderMap` | 全ヘッダー | 認証、コンテンツネゴシエーション |
| `Extension<T>` | ミドルウェア注入値 | 認証済みユーザー情報 |
| `Multipart` | ファイルアップロード | `multipart::Multipart` |

---

## 6. アンチパターン

### アンチパターン1: ハンドラ内でブロッキング処理

```rust
// NG: async ハンドラ内でブロッキング I/O
async fn bad_handler() -> String {
    let data = std::fs::read_to_string("large.csv").unwrap(); // ブロック!
    process(data)
}

// OK: spawn_blocking で逃がす
async fn good_handler() -> String {
    let data = tokio::task::spawn_blocking(|| {
        std::fs::read_to_string("large.csv").unwrap()
    }).await.unwrap();
    process(data)
}

fn process(data: String) -> String { data }
```

### アンチパターン2: グローバル可変状態の直接操作

```rust
// NG: static mut は unsafe かつデータ競合の原因
// static mut COUNTER: u64 = 0;

// OK: State + Arc<AtomicU64>
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone)]
struct Metrics {
    request_count: Arc<AtomicU64>,
}

async fn handler(State(metrics): State<Metrics>) -> String {
    let count = metrics.request_count.fetch_add(1, Ordering::Relaxed);
    format!("リクエスト#{}", count + 1)
}
```

---

## FAQ

### Q1: Router のネスティング方法は?

**A:** `Router::nest` でプレフィックス付きサブルーターを構成します。

```rust
let api = Router::new()
    .route("/users", get(list_users))
    .route("/posts", get(list_posts));

let app = Router::new()
    .nest("/api/v1", api)           // /api/v1/users, /api/v1/posts
    .route("/health", get(health)); // /health
```

### Q2: テストはどう書く?

**A:** `axum::body::Body` と `tower::ServiceExt` を使います。

```rust
#[tokio::test]
async fn test_list_users() {
    let app = create_router();
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .uri("/users")
                .body(axum::body::Body::empty())
                .unwrap()
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
```

### Q3: 静的ファイル配信は?

**A:** `tower-http::services::ServeDir` を使います。

```rust
use tower_http::services::ServeDir;

let app = Router::new()
    .route("/api/data", get(handler))
    .nest_service("/static", ServeDir::new("./public"));
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Router | `route()` でパス+メソッド、`nest()` でグループ化 |
| Extractor | Path, Query, Json, State で型安全にリクエスト解析 |
| ミドルウェア | Tower Layer で共通処理をスタック |
| 状態管理 | `with_state()` で AppState を全ハンドラに共有 |
| エラー | `IntoResponse` を impl した統一エラー型 |
| DB統合 | PgPool を AppState に格納。sqlx でクエリ |
| テスト | `oneshot()` でインメモリテスト |

## 次に読むべきガイド

- [データベース](../04-ecosystem/03-database.md) — SQLx/Diesel/SeaORM の詳細
- [テスト](../04-ecosystem/01-testing.md) — 統合テスト・プロパティテスト
- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — API設計原則

## 参考文献

1. **Axum Documentation**: https://docs.rs/axum/latest/axum/
2. **Tower Service trait**: https://docs.rs/tower/latest/tower/trait.Service.html
3. **Axum Examples (GitHub)**: https://github.com/tokio-rs/axum/tree/main/examples
