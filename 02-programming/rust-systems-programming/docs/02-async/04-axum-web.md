# Axum — ルーティング、ミドルウェア、状態管理

> Axum フレームワークを使った型安全な Web API 構築手法を、ルーティング・ミドルウェア・状態管理を軸に習得する

## この章で学ぶこと

1. **ルーティングとハンドラ** — Extractor パターンによる型安全なリクエスト解析
2. **ミドルウェア** — Tower レイヤーとカスタムミドルウェアの実装
3. **状態管理** — AppState の共有、データベース接続プール統合
4. **エラーハンドリング** — 統一エラー型と適切なレスポンス変換
5. **WebSocket とSSE** — リアルタイム通信の統合
6. **テスト** — ハンドラ単体テストと統合テスト
7. **本番運用** — Graceful Shutdown、ヘルスチェック、メトリクス

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

### Axum の設計哲学

| 設計原則 | 説明 |
|---|---|
| Tower 統合 | ミドルウェアは全て Tower Service/Layer として実装 |
| 型安全 | Extractor による型レベルでのリクエスト解析・バリデーション |
| Macro-free | derive マクロを使わずコンパイラが型推論でルーティング検証 |
| hyper ベース | 内部で hyper を使用し高パフォーマンスを実現 |
| エコシステム | tower-http の全レイヤーがそのまま使える |

### Axum の依存関係

```
┌─────────────── 依存ツリー ─────────────────┐
│                                             │
│  axum (Web フレームワーク)                  │
│    ├── axum-core (Extractor トレイト)       │
│    ├── tower (Service, Layer)               │
│    │     ├── tower-service                  │
│    │     └── tower-layer                    │
│    ├── tower-http (HTTP ミドルウェア)        │
│    │     ├── TraceLayer                     │
│    │     ├── CorsLayer                      │
│    │     ├── CompressionLayer               │
│    │     └── TimeoutLayer                   │
│    ├── hyper (HTTP/1, HTTP/2)               │
│    ├── tokio (非同期ランタイム)              │
│    └── matchit (URLルーティング)             │
│                                             │
│  追加パッケージ:                             │
│    ├── axum-extra (TypedHeader 等)          │
│    └── axum-macros (#[debug_handler])       │
└─────────────────────────────────────────────┘
```

---

## 2. ルーティングとハンドラ

### 2.1 基本的なCRUD API

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
struct UpdateUser {
    name: Option<String>,
    email: Option<String>,
}

#[derive(Deserialize)]
struct ListParams {
    page: Option<u32>,
    limit: Option<u32>,
    sort: Option<String>,
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

// PUT /users/:id
async fn update_user(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    Json(input): Json<UpdateUser>,
) -> Result<Json<User>, StatusCode> {
    let mut users = state.write().await;
    let user = users.iter_mut()
        .find(|u| u.id == id)
        .ok_or(StatusCode::NOT_FOUND)?;

    if let Some(name) = input.name {
        user.name = name;
    }
    if let Some(email) = input.email {
        user.email = email;
    }

    Ok(Json(user.clone()))
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
        .route("/users/{id}", get(get_user).put(update_user).delete(delete_user))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("サーバー起動: http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
```

### 2.2 ルーターのネスティングとマージ

```rust
use axum::{Router, routing::get};

/// API v1 のルーター
fn api_v1_routes() -> Router<AppState> {
    let users = Router::new()
        .route("/", get(list_users).post(create_user))
        .route("/{id}", get(get_user).put(update_user).delete(delete_user));

    let posts = Router::new()
        .route("/", get(list_posts).post(create_post))
        .route("/{id}", get(get_post).put(update_post).delete(delete_post));

    let comments = Router::new()
        .route("/", get(list_comments).post(create_comment));

    Router::new()
        .nest("/users", users)
        .nest("/posts", posts)
        .nest("/posts/{post_id}/comments", comments)
}

/// API v2 のルーター (v1 を拡張)
fn api_v2_routes() -> Router<AppState> {
    let users = Router::new()
        .route("/", get(list_users_v2).post(create_user_v2))
        .route("/{id}", get(get_user_v2));

    Router::new()
        .nest("/users", users)
}

/// アプリケーション全体のルーター構築
fn create_app(state: AppState) -> Router {
    // 管理者ルート (別のミドルウェアスタック)
    let admin = Router::new()
        .route("/stats", get(admin_stats))
        .route("/users", get(admin_list_users))
        .layer(axum::middleware::from_fn(admin_auth_middleware));

    Router::new()
        // API バージョニング
        .nest("/api/v1", api_v1_routes())
        .nest("/api/v2", api_v2_routes())
        // 管理者ルート
        .nest("/admin", admin)
        // ヘルスチェック
        .route("/health", get(health_check))
        // 静的ファイル
        .nest_service("/static", tower_http::services::ServeDir::new("./public"))
        // フォールバック
        .fallback(fallback_handler)
        .with_state(state)
}

async fn health_check() -> &'static str {
    "OK"
}

async fn fallback_handler() -> (StatusCode, &'static str) {
    (StatusCode::NOT_FOUND, "ルートが見つかりません")
}
```

### 2.3 カスタム Extractor

```rust
use axum::{
    async_trait,
    extract::{FromRequestParts, FromRequest, Request},
    http::{request::Parts, StatusCode, header},
    body::Body,
    Json,
};
use serde::de::DeserializeOwned;

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

/// 認証済みユーザー情報を抽出する Extractor
#[derive(Debug, Clone)]
struct AuthUser {
    user_id: String,
    role: String,
}

#[async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        let auth_header = parts.headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                let body = serde_json::json!({"error": "認証が必要です"});
                (StatusCode::UNAUTHORIZED, Json(body))
            })?;

        let token = auth_header.strip_prefix("Bearer ").ok_or_else(|| {
            let body = serde_json::json!({"error": "Bearer トークン形式が不正"});
            (StatusCode::UNAUTHORIZED, Json(body))
        })?;

        // トークン検証 (実際にはJWT検証など)
        validate_token(token).await.map_err(|e| {
            let body = serde_json::json!({"error": format!("トークン無効: {}", e)});
            (StatusCode::UNAUTHORIZED, Json(body))
        })
    }
}

async fn validate_token(token: &str) -> Result<AuthUser, String> {
    // JWT 検証ロジック (簡略版)
    if token.starts_with("valid-") {
        Ok(AuthUser {
            user_id: "user-123".to_string(),
            role: "admin".to_string(),
        })
    } else {
        Err("無効なトークン".to_string())
    }
}

/// バリデーション付き JSON Extractor
struct ValidatedJson<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state)
            .await
            .map_err(|e| {
                let body = serde_json::json!({"error": format!("JSON パースエラー: {}", e)});
                (StatusCode::BAD_REQUEST, Json(body))
            })?;

        value.validate().map_err(|errors| {
            let body = serde_json::json!({"error": "バリデーションエラー", "details": errors});
            (StatusCode::UNPROCESSABLE_ENTITY, Json(body))
        })?;

        Ok(ValidatedJson(value))
    }
}

/// バリデーショントレイト
trait Validate {
    fn validate(&self) -> Result<(), Vec<String>>;
}

impl Validate for CreateUser {
    fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push("name は必須です".to_string());
        }
        if self.name.len() > 100 {
            errors.push("name は100文字以内です".to_string());
        }
        if !self.email.contains('@') {
            errors.push("email の形式が不正です".to_string());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

// ハンドラで使用
async fn protected_handler(
    auth: AuthUser,
) -> String {
    format!("認証済み。ユーザー: {} (役割: {})", auth.user_id, auth.role)
}

async fn create_user_validated(
    ValidatedJson(input): ValidatedJson<CreateUser>,
) -> (StatusCode, Json<User>) {
    let user = User {
        id: 1,
        name: input.name,
        email: input.email,
    };
    (StatusCode::CREATED, Json(user))
}
```

### 2.4 複数パスパラメータとタプル Extractor

```rust
use axum::extract::Path;

// 単一パスパラメータ
async fn get_user(Path(id): Path<u64>) -> String {
    format!("User {}", id)
}

// 複数パスパラメータ (タプル)
async fn get_user_post(
    Path((user_id, post_id)): Path<(u64, u64)>,
) -> String {
    format!("User {} の Post {}", user_id, post_id)
}

// 名前付きパスパラメータ (構造体)
#[derive(Deserialize)]
struct PostParams {
    user_id: u64,
    post_id: u64,
    comment_id: Option<u64>,
}

async fn get_comment(
    Path(params): Path<PostParams>,
) -> String {
    format!(
        "User {} / Post {} / Comment {:?}",
        params.user_id, params.post_id, params.comment_id
    )
}

// ルーター定義
// Router::new()
//     .route("/users/{id}", get(get_user))
//     .route("/users/{user_id}/posts/{post_id}", get(get_user_post))
//     .route("/users/{user_id}/posts/{post_id}/comments/{comment_id}", get(get_comment))
```

---

## 3. ミドルウェア

### 3.1 Tower レイヤーによるミドルウェアスタック

```rust
use axum::{Router, routing::get, middleware};
use tower_http::{
    cors::{CorsLayer, Any},
    trace::TraceLayer,
    compression::CompressionLayer,
    timeout::TimeoutLayer,
    limit::RequestBodyLimitLayer,
    set_header::SetResponseHeaderLayer,
    catch_panic::CatchPanicLayer,
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
};
use std::time::Duration;
use http::HeaderName;

fn create_router() -> Router {
    // リクエストID ヘッダー名
    let x_request_id = HeaderName::from_static("x-request-id");

    Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/api/data", get(handler))
        // ミドルウェアは下から上に適用される (最後に追加したものが最初に実行)
        .layer(CompressionLayer::new())                     // レスポンス圧縮
        .layer(RequestBodyLimitLayer::new(1024 * 1024))     // 1MB制限
        .layer(TimeoutLayer::new(Duration::from_secs(30)))  // タイムアウト
        .layer(CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .max_age(Duration::from_secs(3600)))
        .layer(CatchPanicLayer::new())                      // パニックキャッチ
        .layer(PropagateRequestIdLayer::new(x_request_id.clone()))
        .layer(SetRequestIdLayer::new(x_request_id, MakeRequestUuid))
        .layer(TraceLayer::new_for_http())                  // リクエストログ
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
│          → SetRequestIdLayer             │
│          → PropagateRequestIdLayer       │
│          → CatchPanicLayer               │
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
│                                          │
│  ※ ルート個別にミドルウェアを             │
│    適用する場合は route_layer() を使用    │
└──────────────────────────────────────────┘
```

### 3.2 カスタムミドルウェア

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

/// レート制限ミドルウェア (簡易版)
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;

#[derive(Clone)]
struct RateLimiter {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    max_requests: usize,
    window: Duration,
}

impl RateLimiter {
    fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window,
        }
    }

    async fn check(&self, key: &str) -> bool {
        let mut requests = self.requests.lock().await;
        let now = Instant::now();
        let entry = requests.entry(key.to_string()).or_default();

        // 古いエントリを削除
        entry.retain(|t| now.duration_since(*t) < self.window);

        if entry.len() >= self.max_requests {
            false
        } else {
            entry.push(now);
            true
        }
    }
}

async fn rate_limit_middleware(
    State(limiter): State<RateLimiter>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // IPアドレスでレート制限
    let ip = request.headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    if !limiter.check(&ip).await {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    Ok(next.run(request).await)
}

// 適用例
// let limiter = RateLimiter::new(100, Duration::from_secs(60));
// let app = Router::new()
//     .route("/api/data", get(handler))
//     .layer(axum::middleware::from_fn(timing_middleware))
//     .layer(axum::middleware::from_fn(api_key_middleware))
//     .layer(axum::middleware::from_fn_with_state(limiter, rate_limit_middleware));
```

### 3.3 ルート別ミドルウェア

```rust
use axum::{Router, routing::get, middleware};

fn create_router_with_route_middleware() -> Router<AppState> {
    // 認証が必要なルート
    let protected = Router::new()
        .route("/profile", get(get_profile))
        .route("/settings", get(get_settings).put(update_settings))
        .layer(middleware::from_fn(auth_middleware));

    // 認証不要なルート
    let public = Router::new()
        .route("/", get(index))
        .route("/login", post(login))
        .route("/register", post(register));

    // 管理者ルート (認証 + 権限チェック)
    let admin = Router::new()
        .route("/users", get(admin_list_users))
        .route("/users/{id}", delete(admin_delete_user))
        .layer(middleware::from_fn(admin_middleware))
        .layer(middleware::from_fn(auth_middleware));

    Router::new()
        .merge(public)
        .merge(protected)
        .nest("/admin", admin)
}
```

---

## 4. 状態管理とDB統合

### 4.1 AppState の設計パターン

```rust
use axum::extract::FromRef;
use sqlx::PgPool;
use reqwest::Client as HttpClient;

/// アプリケーションの共有状態
#[derive(Clone)]
struct AppState {
    db: PgPool,
    http_client: HttpClient,
    config: AppConfig,
    cache: Arc<tokio::sync::RwLock<HashMap<String, CacheEntry>>>,
}

#[derive(Clone)]
struct AppConfig {
    jwt_secret: String,
    external_api_url: String,
    max_upload_size: usize,
}

#[derive(Clone)]
struct CacheEntry {
    value: String,
    expires_at: std::time::Instant,
}

/// FromRef で個別の状態要素を抽出可能にする
impl FromRef<AppState> for PgPool {
    fn from_ref(state: &AppState) -> Self {
        state.db.clone()
    }
}

impl FromRef<AppState> for HttpClient {
    fn from_ref(state: &AppState) -> Self {
        state.http_client.clone()
    }
}

impl FromRef<AppState> for AppConfig {
    fn from_ref(state: &AppState) -> Self {
        state.config.clone()
    }
}

// ハンドラで個別に取得できる
async fn handler_with_db(
    State(db): State<PgPool>,
) -> Result<Json<Vec<User>>, StatusCode> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&db)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(users))
}

async fn handler_with_config(
    State(config): State<AppConfig>,
) -> String {
    format!("API URL: {}", config.external_api_url)
}
```

### 4.2 SQLx + Axum 統合

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
    created_at: chrono::NaiveDateTime,
}

#[derive(Deserialize)]
struct CreateTodo {
    title: String,
}

#[derive(Deserialize)]
struct UpdateTodo {
    title: Option<String>,
    completed: Option<bool>,
}

#[derive(Clone)]
struct AppState {
    db: PgPool,
}

async fn list_todos(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> Result<Json<Vec<Todo>>, AppError> {
    let page = params.page.unwrap_or(1) as i64;
    let limit = params.limit.unwrap_or(20) as i64;
    let offset = (page - 1) * limit;

    let todos = sqlx::query_as::<_, Todo>(
        "SELECT id, title, completed, created_at FROM todos ORDER BY created_at DESC LIMIT $1 OFFSET $2"
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(&state.db)
    .await?;

    Ok(Json(todos))
}

async fn get_todo(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<Todo>, AppError> {
    let todo = sqlx::query_as::<_, Todo>(
        "SELECT id, title, completed, created_at FROM todos WHERE id = $1"
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(AppError::NotFound(format!("Todo {} が見つかりません", id)))?;

    Ok(Json(todo))
}

async fn create_todo(
    State(state): State<AppState>,
    Json(input): Json<CreateTodo>,
) -> Result<(StatusCode, Json<Todo>), AppError> {
    if input.title.trim().is_empty() {
        return Err(AppError::BadRequest("title は空にできません".into()));
    }

    let todo = sqlx::query_as::<_, Todo>(
        "INSERT INTO todos (title, completed) VALUES ($1, false) RETURNING id, title, completed, created_at"
    )
    .bind(&input.title)
    .fetch_one(&state.db)
    .await?;

    Ok((StatusCode::CREATED, Json(todo)))
}

async fn update_todo(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(input): Json<UpdateTodo>,
) -> Result<Json<Todo>, AppError> {
    // 部分更新: COALESCE で NULL の場合は既存値を維持
    let todo = sqlx::query_as::<_, Todo>(
        r#"UPDATE todos
           SET title = COALESCE($1, title),
               completed = COALESCE($2, completed)
           WHERE id = $3
           RETURNING id, title, completed, created_at"#
    )
    .bind(input.title)
    .bind(input.completed)
    .bind(id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(AppError::NotFound(format!("Todo {} が見つかりません", id)))?;

    Ok(Json(todo))
}

async fn delete_todo(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<StatusCode, AppError> {
    let result = sqlx::query("DELETE FROM todos WHERE id = $1")
        .bind(id)
        .execute(&state.db)
        .await?;

    if result.rows_affected() == 0 {
        Err(AppError::NotFound(format!("Todo {} が見つかりません", id)))
    } else {
        Ok(StatusCode::NO_CONTENT)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ログ初期化
    tracing_subscriber::fmt()
        .with_env_filter("info,sqlx=warn")
        .init();

    // DB接続
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .min_connections(5)
        .acquire_timeout(std::time::Duration::from_secs(5))
        .idle_timeout(std::time::Duration::from_secs(600))
        .connect(&std::env::var("DATABASE_URL")?)
        .await?;

    // マイグレーション実行
    sqlx::migrate!("./migrations").run(&pool).await?;

    let state = AppState { db: pool };

    let app = Router::new()
        .route("/todos", get(list_todos).post(create_todo))
        .route("/todos/{id}", get(get_todo).put(update_todo).delete(delete_todo))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("サーバー起動: http://localhost:3000");
    axum::serve(listener, app).await?;
    Ok(())
}
```

### 4.3 トランザクションの使用

```rust
use sqlx::{PgPool, Postgres, Transaction};

async fn transfer_funds(
    State(state): State<AppState>,
    Json(input): Json<TransferRequest>,
) -> Result<Json<TransferResult>, AppError> {
    // トランザクション開始
    let mut tx: Transaction<'_, Postgres> = state.db.begin().await?;

    // 送金元の残高チェック
    let sender = sqlx::query_as::<_, Account>(
        "SELECT * FROM accounts WHERE id = $1 FOR UPDATE"
    )
    .bind(input.from_account)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or(AppError::NotFound("送金元アカウントが見つかりません".into()))?;

    if sender.balance < input.amount {
        return Err(AppError::BadRequest("残高不足".into()));
    }

    // 送金元から引き落とし
    sqlx::query("UPDATE accounts SET balance = balance - $1 WHERE id = $2")
        .bind(input.amount)
        .bind(input.from_account)
        .execute(&mut *tx)
        .await?;

    // 送金先に入金
    sqlx::query("UPDATE accounts SET balance = balance + $1 WHERE id = $2")
        .bind(input.amount)
        .bind(input.to_account)
        .execute(&mut *tx)
        .await?;

    // 取引履歴の記録
    sqlx::query(
        "INSERT INTO transactions (from_account, to_account, amount) VALUES ($1, $2, $3)"
    )
    .bind(input.from_account)
    .bind(input.to_account)
    .bind(input.amount)
    .execute(&mut *tx)
    .await?;

    // コミット
    tx.commit().await?;

    Ok(Json(TransferResult {
        success: true,
        message: format!("{}円を送金しました", input.amount),
    }))
}
```

---

## 5. エラーハンドリング

### 5.1 統一エラー型

```rust
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// アプリケーション統一エラー型
#[derive(Debug)]
enum AppError {
    NotFound(String),
    BadRequest(String),
    Unauthorized(String),
    Forbidden(String),
    Conflict(String),
    Internal(anyhow::Error),
    Database(sqlx::Error),
    Validation(Vec<String>),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "not_found", msg.clone()),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg.clone()),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, "unauthorized", msg.clone()),
            AppError::Forbidden(msg) => (StatusCode::FORBIDDEN, "forbidden", msg.clone()),
            AppError::Conflict(msg) => (StatusCode::CONFLICT, "conflict", msg.clone()),
            AppError::Internal(e) => {
                tracing::error!("内部エラー: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "内部サーバーエラー".into())
            }
            AppError::Database(e) => {
                tracing::error!("データベースエラー: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "database_error", "データベースエラー".into())
            }
            AppError::Validation(errors) => {
                let msg = errors.join(", ");
                (StatusCode::UNPROCESSABLE_ENTITY, "validation_error", msg)
            }
        };

        let body = Json(json!({
            "error": {
                "type": error_type,
                "status": status.as_u16(),
                "message": message,
            }
        }));

        (status, body).into_response()
    }
}

// 自動変換の実装
impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Internal(e)
    }
}

impl From<sqlx::Error> for AppError {
    fn from(e: sqlx::Error) -> Self {
        match &e {
            sqlx::Error::RowNotFound => AppError::NotFound("リソースが見つかりません".into()),
            sqlx::Error::Database(db_err) => {
                // PostgreSQL の一意制約違反
                if db_err.code().as_deref() == Some("23505") {
                    AppError::Conflict("リソースが既に存在します".into())
                } else {
                    AppError::Database(e)
                }
            }
            _ => AppError::Database(e),
        }
    }
}

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        AppError::Internal(e.into())
    }
}

// ハンドラの戻り値型
async fn handler() -> Result<Json<serde_json::Value>, AppError> {
    let data = fetch_data().await?;
    Ok(Json(data))
}

async fn fetch_data() -> anyhow::Result<serde_json::Value> {
    Ok(json!({"status": "ok"}))
}
```

### 5.2 エラーレスポンスの詳細化

```rust
use axum::response::IntoResponse;

/// 詳細なエラー情報を含むレスポンス
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    status: u16,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Vec<FieldError>>,
}

#[derive(Debug, Serialize)]
struct FieldError {
    field: String,
    message: String,
    code: String,
}

/// バリデーションエラーの生成ヘルパー
fn validation_error(errors: Vec<FieldError>) -> AppError {
    AppError::Validation(
        errors.iter().map(|e| format!("{}: {}", e.field, e.message)).collect()
    )
}

// 使用例
async fn create_user_handler(
    Json(input): Json<CreateUser>,
) -> Result<Json<User>, AppError> {
    let mut errors = Vec::new();

    if input.name.is_empty() {
        errors.push(FieldError {
            field: "name".into(),
            message: "名前は必須です".into(),
            code: "required".into(),
        });
    }

    if !input.email.contains('@') {
        errors.push(FieldError {
            field: "email".into(),
            message: "有効なメールアドレスを入力してください".into(),
            code: "invalid_format".into(),
        });
    }

    if !errors.is_empty() {
        return Err(validation_error(errors));
    }

    // 処理続行...
    todo!()
}
```

---

## 6. WebSocket とSSE

### 6.1 WebSocket ハンドラ

```rust
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::sync::broadcast;

#[derive(Clone)]
struct WsState {
    tx: broadcast::Sender<String>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<WsState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: WsState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // サーバー → クライアント
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // クライアント → サーバー
    let tx = state.tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // ブロードキャスト
                    let _ = tx.send(text);
                }
                Message::Ping(data) => {
                    // Pong はaxumが自動応答
                    tracing::debug!("Ping received: {} bytes", data.len());
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}
```

### 6.2 Server-Sent Events (SSE)

```rust
use axum::{
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
    Router,
};
use futures::stream::{self, Stream};
use std::{convert::Infallible, time::Duration};
use tokio_stream::StreamExt;

async fn sse_handler() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::repeat_with(|| {
        let data = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "value": rand::random::<f64>(),
        });
        Event::default()
            .event("update")
            .data(data.to_string())
    })
    .map(Ok)
    .throttle(Duration::from_secs(1));

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
}

/// DB変更を監視するSSEストリーム
async fn sse_notifications(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(100);

    // DB変更監視タスク (PostgreSQL LISTEN/NOTIFY)
    let db = state.db.clone();
    tokio::spawn(async move {
        let mut listener = sqlx::postgres::PgListener::connect_with(&db).await.unwrap();
        listener.listen("notifications").await.unwrap();

        loop {
            match listener.recv().await {
                Ok(notification) => {
                    let payload = notification.payload().to_string();
                    if tx.send(payload).await.is_err() { break; }
                }
                Err(e) => {
                    tracing::error!("Listener error: {}", e);
                    break;
                }
            }
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|payload| {
            Ok(Event::default()
                .event("notification")
                .data(payload))
        });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ルーター
// Router::new()
//     .route("/events", get(sse_handler))
//     .route("/notifications", get(sse_notifications))
```

---

## 7. ファイルアップロード

### 7.1 マルチパートアップロード

```rust
use axum::{
    extract::Multipart,
    http::StatusCode,
    Json,
};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

#[derive(Serialize)]
struct UploadResult {
    files: Vec<UploadedFile>,
}

#[derive(Serialize)]
struct UploadedFile {
    original_name: String,
    saved_name: String,
    size: usize,
    content_type: String,
}

const MAX_FILE_SIZE: usize = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES: &[&str] = &["image/jpeg", "image/png", "image/gif", "application/pdf"];

async fn upload_handler(
    mut multipart: Multipart,
) -> Result<Json<UploadResult>, AppError> {
    let mut files = Vec::new();
    let upload_dir = "./uploads";

    // アップロードディレクトリ作成
    tokio::fs::create_dir_all(upload_dir).await
        .map_err(|e| AppError::Internal(e.into()))?;

    while let Some(field) = multipart.next_field().await
        .map_err(|e| AppError::BadRequest(format!("マルチパートエラー: {}", e)))? {

        let name = field.name().unwrap_or("unknown").to_string();
        let file_name = field.file_name()
            .unwrap_or("unnamed")
            .to_string();
        let content_type = field.content_type()
            .unwrap_or("application/octet-stream")
            .to_string();

        // ファイルタイプ検証
        if !ALLOWED_TYPES.contains(&content_type.as_str()) {
            return Err(AppError::BadRequest(
                format!("許可されていないファイルタイプ: {}", content_type)
            ));
        }

        // ファイルデータ読み込み
        let data = field.bytes().await
            .map_err(|e| AppError::BadRequest(format!("読み込みエラー: {}", e)))?;

        // サイズ検証
        if data.len() > MAX_FILE_SIZE {
            return Err(AppError::BadRequest(
                format!("ファイルサイズ上限超過: {} bytes (最大 {})", data.len(), MAX_FILE_SIZE)
            ));
        }

        // 安全なファイル名生成
        let extension = std::path::Path::new(&file_name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin");
        let saved_name = format!("{}.{}", Uuid::new_v4(), extension);
        let file_path = format!("{}/{}", upload_dir, saved_name);

        // ファイル保存
        let mut file = tokio::fs::File::create(&file_path).await
            .map_err(|e| AppError::Internal(e.into()))?;
        file.write_all(&data).await
            .map_err(|e| AppError::Internal(e.into()))?;

        files.push(UploadedFile {
            original_name: file_name,
            saved_name,
            size: data.len(),
            content_type,
        });
    }

    if files.is_empty() {
        return Err(AppError::BadRequest("ファイルが見つかりません".into()));
    }

    Ok(Json(UploadResult { files }))
}
```

---

## 8. テスト

### 8.1 ハンドラの単体テスト

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode, Method},
    };
    use tower::ServiceExt; // oneshot

    fn create_test_app() -> Router {
        let state: AppState = Arc::new(RwLock::new(vec![
            User { id: 1, name: "Alice".into(), email: "alice@example.com".into() },
            User { id: 2, name: "Bob".into(), email: "bob@example.com".into() },
        ]));

        Router::new()
            .route("/users", get(list_users).post(create_user))
            .route("/users/{id}", get(get_user).delete(delete_user))
            .with_state(state)
    }

    #[tokio::test]
    async fn test_list_users() {
        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/users")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let users: Vec<User> = serde_json::from_slice(&body).unwrap();
        assert_eq!(users.len(), 2);
        assert_eq!(users[0].name, "Alice");
    }

    #[tokio::test]
    async fn test_get_user_found() {
        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/users/1")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let user: User = serde_json::from_slice(&body).unwrap();
        assert_eq!(user.name, "Alice");
    }

    #[tokio::test]
    async fn test_get_user_not_found() {
        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/users/999")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_create_user() {
        let app = create_test_app();

        let body = serde_json::json!({
            "name": "Charlie",
            "email": "charlie@example.com"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/users")
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let user: User = serde_json::from_slice(&body).unwrap();
        assert_eq!(user.name, "Charlie");
        assert_eq!(user.id, 3);
    }

    #[tokio::test]
    async fn test_delete_user() {
        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/users/1")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_delete_user_not_found() {
        let app = create_test_app();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/users/999")
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
```

### 8.2 DB統合テスト

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use sqlx::PgPool;

    /// テスト用DBプールの作成 (テストごとにトランザクションでロールバック)
    async fn setup_test_db() -> PgPool {
        let url = std::env::var("TEST_DATABASE_URL")
            .unwrap_or_else(|_| "postgres://test:test@localhost/test_db".into());
        let pool = PgPool::connect(&url).await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    #[sqlx::test]
    async fn test_create_and_get_todo(pool: PgPool) {
        let state = AppState { db: pool };
        let app = Router::new()
            .route("/todos", get(list_todos).post(create_todo))
            .route("/todos/{id}", get(get_todo))
            .with_state(state);

        // 作成
        let create_body = serde_json::json!({"title": "テストTodo"});
        let response = app.clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/todos")
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(&create_body).unwrap()))
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let created: Todo = serde_json::from_slice(&body).unwrap();
        assert_eq!(created.title, "テストTodo");
        assert!(!created.completed);

        // 取得
        let response = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/todos/{}", created.id))
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let fetched: Todo = serde_json::from_slice(&body).unwrap();
        assert_eq!(fetched.id, created.id);
    }
}
```

### 8.3 テストヘルパー

```rust
/// テスト用HTTPクライアント
struct TestClient {
    app: Router,
}

impl TestClient {
    fn new(app: Router) -> Self {
        Self { app }
    }

    async fn get(&self, uri: &str) -> axum::response::Response {
        self.app.clone()
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
    }

    async fn post_json<T: Serialize>(&self, uri: &str, body: &T) -> axum::response::Response {
        self.app.clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(uri)
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(body).unwrap()))
                    .unwrap()
            )
            .await
            .unwrap()
    }

    async fn put_json<T: Serialize>(&self, uri: &str, body: &T) -> axum::response::Response {
        self.app.clone()
            .oneshot(
                Request::builder()
                    .method(Method::PUT)
                    .uri(uri)
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(body).unwrap()))
                    .unwrap()
            )
            .await
            .unwrap()
    }

    async fn delete(&self, uri: &str) -> axum::response::Response {
        self.app.clone()
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
    }

    async fn get_with_auth(&self, uri: &str, token: &str) -> axum::response::Response {
        self.app.clone()
            .oneshot(
                Request::builder()
                    .uri(uri)
                    .header("Authorization", format!("Bearer {}", token))
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
    }
}

/// レスポンスボディのデシリアライズヘルパー
async fn body_json<T: serde::de::DeserializeOwned>(response: axum::response::Response) -> T {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn test_with_helper() {
    let client = TestClient::new(create_test_app());

    let resp = client.get("/users").await;
    assert_eq!(resp.status(), StatusCode::OK);

    let users: Vec<User> = body_json(resp).await;
    assert_eq!(users.len(), 2);
}
```

---

## 9. 本番運用パターン

### 9.1 Graceful Shutdown

```rust
use axum::Router;
use tokio::signal;
use std::time::Duration;

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("Ctrl+C シグナルハンドラの登録に失敗");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("SIGTERM ハンドラの登録に失敗")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Ctrl+C 受信"),
        _ = terminate => tracing::info!("SIGTERM 受信"),
    }

    tracing::info!("シャットダウン開始...");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = create_app().await?;

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("サーバー起動: http://localhost:3000");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("サーバー停止完了");
    Ok(())
}
```

### 9.2 ヘルスチェックとメトリクス

```rust
use axum::{routing::get, Json, Router};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    uptime_seconds: u64,
    version: String,
    checks: HealthChecks,
}

#[derive(Serialize)]
struct HealthChecks {
    database: CheckResult,
    redis: CheckResult,
}

#[derive(Serialize)]
struct CheckResult {
    status: String,
    latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let uptime = state.start_time.elapsed().as_secs();

    // DB チェック
    let db_check = {
        let start = Instant::now();
        match sqlx::query("SELECT 1").execute(&state.db).await {
            Ok(_) => CheckResult {
                status: "up".into(),
                latency_ms: start.elapsed().as_millis() as u64,
                error: None,
            },
            Err(e) => CheckResult {
                status: "down".into(),
                latency_ms: start.elapsed().as_millis() as u64,
                error: Some(e.to_string()),
            },
        }
    };

    // Redis チェック (省略、同様のパターン)
    let redis_check = CheckResult {
        status: "up".into(),
        latency_ms: 1,
        error: None,
    };

    Json(HealthResponse {
        status: if db_check.status == "up" { "healthy" } else { "unhealthy" }.into(),
        uptime_seconds: uptime,
        version: env!("CARGO_PKG_VERSION").into(),
        checks: HealthChecks {
            database: db_check,
            redis: redis_check,
        },
    })
}

/// Readiness チェック (Kubernetes 向け)
async fn readiness_check(State(state): State<AppState>) -> StatusCode {
    match sqlx::query("SELECT 1").execute(&state.db).await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Liveness チェック (Kubernetes 向け)
async fn liveness_check() -> StatusCode {
    StatusCode::OK
}

// Router::new()
//     .route("/health", get(health_check))
//     .route("/ready", get(readiness_check))
//     .route("/live", get(liveness_check))
```

### 9.3 構造化ログ設定

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn setup_tracing() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            EnvFilter::new("info,tower_http=debug,axum=trace,sqlx=warn")
        });

    // JSONフォーマットのログ (本番環境)
    if std::env::var("RUST_LOG_FORMAT").as_deref() == Ok("json") {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        // 人間に読みやすいフォーマット (開発環境)
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().pretty())
            .init();
    }
}
```

### 9.4 CORS と セキュリティヘッダー

```rust
use tower_http::cors::{CorsLayer, AllowOrigin};
use axum::http::{header, HeaderValue, Method};

fn cors_layer() -> CorsLayer {
    // 開発環境
    if cfg!(debug_assertions) {
        CorsLayer::very_permissive()
    } else {
        // 本番環境: 許可するオリジンを明示
        CorsLayer::new()
            .allow_origin(AllowOrigin::list([
                "https://app.example.com".parse::<HeaderValue>().unwrap(),
                "https://admin.example.com".parse::<HeaderValue>().unwrap(),
            ]))
            .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
            .allow_credentials(true)
            .max_age(Duration::from_secs(3600))
    }
}

/// セキュリティヘッダーミドルウェア
async fn security_headers(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'".parse().unwrap(),
    );

    response
}
```

---

## 10. 比較表

### Rust Web フレームワーク比較

| 特性 | Axum | Actix-web | Rocket | Warp |
|---|---|---|---|---|
| ベース | hyper + Tower | 独自アクターシステム | 独自 | hyper |
| 型安全性 | 非常に高い | 高い | 非常に高い | 高い |
| パフォーマンス | 非常に高い | 非常に高い | 高い | 非常に高い |
| エコシステム | Tower 互換 | 独自 | 独自 | 独自 |
| 学習コスト | 中 | 中 | 低 | 中~高 |
| 採用トレンド | 急成長中 | 安定 | 安定 | やや減少 |
| WebSocket | 組み込み | 組み込み | 外部ライブラリ | 組み込み |
| SSE | 組み込み | 外部ライブラリ | 外部ライブラリ | 限定的 |
| ミドルウェア | Tower Layer | Transform | Fairing | Filter |
| テスト | oneshot() | TestServer | local client | test::request() |

### Extractor 一覧

| Extractor | 用途 | 例 |
|---|---|---|
| `Path<T>` | URLパスパラメータ | `/users/{id}` -> `Path(id): Path<u64>` |
| `Query<T>` | クエリ文字列 | `?page=1` -> `Query(p): Query<Params>` |
| `Json<T>` | JSONリクエストボディ | `Json(body): Json<CreateUser>` |
| `State<T>` | 共有状態 | DB プール、設定 |
| `HeaderMap` | 全ヘッダー | 認証、コンテンツネゴシエーション |
| `Extension<T>` | ミドルウェア注入値 | 認証済みユーザー情報 |
| `Multipart` | ファイルアップロード | `multipart::Multipart` |
| `ConnectInfo<T>` | 接続情報 | クライアントIPアドレス |
| `OriginalUri` | 元のURI | ネスト前のフルパス |
| `MatchedPath` | マッチしたルート | メトリクス用ラベル |
| `Host` | Hostヘッダー | マルチテナント判定 |

### レスポンス型一覧

| 型 | 用途 | ステータスコード |
|---|---|---|
| `String` / `&str` | テキストレスポンス | 200 |
| `Json<T>` | JSONレスポンス | 200 |
| `(StatusCode, Json<T>)` | ステータス + JSON | 任意 |
| `StatusCode` | ステータスのみ | 任意 |
| `Html<String>` | HTMLレスポンス | 200 |
| `Redirect` | リダイレクト | 301/302/307/308 |
| `Sse<S>` | Server-Sent Events | 200 |
| `Response<Body>` | カスタムレスポンス | 任意 |
| `impl IntoResponse` | カスタム型 | 任意 |

---

## 11. アンチパターン

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

### アンチパターン3: Extractor の順序ミス

```rust
// NG: Body を消費する Extractor (Json) は最後に置く必要がある
// async fn bad(Json(body): Json<MyType>, State(state): State<AppState>) { }
// ↑ コンパイルは通るが、Bodyを先に消費するとStateが取れない場合がある

// OK: Body 非消費 Extractor を先に、Body 消費 Extractor を最後に
async fn good(
    State(state): State<AppState>,     // Body を消費しない
    Path(id): Path<u64>,               // Body を消費しない
    Query(params): Query<ListParams>,  // Body を消費しない
    Json(body): Json<CreateUser>,      // Body を消費する → 最後
) -> Result<Json<User>, AppError> {
    todo!()
}
```

### アンチパターン4: エラーの握りつぶし

```rust
// NG: unwrap() でパニック → サーバークラッシュ
async fn bad_handler(State(state): State<AppState>) -> Json<Vec<User>> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&state.db)
        .await
        .unwrap(); // パニック!
    Json(users)
}

// OK: Result + AppError で適切にエラーレスポンス
async fn good_handler(
    State(state): State<AppState>,
) -> Result<Json<Vec<User>>, AppError> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&state.db)
        .await?; // AppError に自動変換
    Ok(Json(users))
}
```

### アンチパターン5: Router のクローンコスト無視

```rust
// NG: 巨大な状態をクローンしやすい構造にしてしまう
#[derive(Clone)]
struct HeavyState {
    data: Vec<u8>, // 巨大なデータ → 毎リクエストでクローン
}

// OK: Arc で包んで参照カウントのみクローン
#[derive(Clone)]
struct LightState {
    data: Arc<Vec<u8>>,     // Arc のクローンは軽量
    db: PgPool,              // PgPool は内部で Arc を使用
    cache: Arc<RwLock<HashMap<String, String>>>,
}
```

---

## FAQ

### Q1: Router のネスティング方法は?

**A:** `Router::nest` でプレフィックス付きサブルーターを構成します。`Router::merge` は同じレベルでルーターを結合します。

```rust
let api = Router::new()
    .route("/users", get(list_users))
    .route("/posts", get(list_posts));

let app = Router::new()
    .nest("/api/v1", api)           // /api/v1/users, /api/v1/posts
    .route("/health", get(health)); // /health
```

### Q2: テストはどう書く?

**A:** `tower::ServiceExt::oneshot` を使ってインメモリでリクエストを送信します。DB統合テストでは `sqlx::test` マクロが便利です。

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

**A:** `tower-http::services::ServeDir` を使います。SPA のフォールバックも対応できます。

```rust
use tower_http::services::{ServeDir, ServeFile};

let app = Router::new()
    .route("/api/data", get(handler))
    // 静的ファイル
    .nest_service("/static", ServeDir::new("./public"))
    // SPA フォールバック: 該当ファイルがなければ index.html を返す
    .fallback_service(
        ServeDir::new("./dist").not_found_service(ServeFile::new("./dist/index.html"))
    );
```

### Q4: Axum でセッション管理するには?

**A:** `axum-extra` の `CookieJar` または `tower-sessions` クレートを使います。

```rust
use axum_extra::extract::cookie::{CookieJar, Cookie};

async fn login(jar: CookieJar) -> (CookieJar, &'static str) {
    let jar = jar.add(Cookie::new("session_id", "abc123"));
    (jar, "ログイン成功")
}

async fn profile(jar: CookieJar) -> String {
    match jar.get("session_id") {
        Some(cookie) => format!("セッション: {}", cookie.value()),
        None => "未ログイン".into(),
    }
}
```

### Q5: Axum と Actix-web のどちらを選ぶべき?

**A:** 以下の基準で選択します:

- **Axum**: Tower エコシステムとの統合、tokio のみで統一、活発な開発、新規プロジェクト推奨
- **Actix-web**: 既存の Actix-web プロジェクト、アクターモデルが必要な場合、成熟した安定性

パフォーマンスは両者とも非常に高く、実用上の差はほぼありません。

### Q6: #[debug_handler] は何のため?

**A:** `axum-macros` クレートの属性マクロで、ハンドラの型エラーを分かりやすいメッセージに変換します。開発時に有用です。

```rust
use axum_macros::debug_handler;

#[debug_handler]
async fn handler(
    State(state): State<AppState>,
    Json(body): Json<CreateUser>,
) -> Result<Json<User>, AppError> {
    // 型が合わない場合、通常のエラーよりも分かりやすいメッセージが出る
    todo!()
}
```

### Q7: 複数のDBプールを使いたい場合は?

**A:** AppState に複数のプールを持たせ、`FromRef` で個別に抽出可能にします。

```rust
#[derive(Clone)]
struct AppState {
    primary_db: PgPool,
    read_replica: PgPool,
    analytics_db: PgPool,
}

// 読み取り専用クエリはレプリカを使用
async fn list_users(
    State(state): State<AppState>,
) -> Result<Json<Vec<User>>, AppError> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&state.read_replica) // レプリカDB
        .await?;
    Ok(Json(users))
}

// 書き込みはプライマリを使用
async fn create_user(
    State(state): State<AppState>,
    Json(input): Json<CreateUser>,
) -> Result<(StatusCode, Json<User>), AppError> {
    let user = sqlx::query_as::<_, User>(
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *"
    )
    .bind(&input.name)
    .bind(&input.email)
    .fetch_one(&state.primary_db) // プライマリDB
    .await?;
    Ok((StatusCode::CREATED, Json(user)))
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Router | `route()` でパス+メソッド、`nest()` でグループ化、`merge()` で結合 |
| Extractor | Path, Query, Json, State で型安全にリクエスト解析。順序に注意 |
| ミドルウェア | Tower Layer で共通処理をスタック。`from_fn` でカスタム実装 |
| 状態管理 | `with_state()` で AppState を全ハンドラに共有。`FromRef` で個別抽出 |
| エラー | `IntoResponse` を impl した統一エラー型。From 変換で自動対応 |
| DB統合 | PgPool を AppState に格納。sqlx でクエリ。トランザクション対応 |
| テスト | `oneshot()` でインメモリテスト。TestClient パターン推奨 |
| WebSocket | `WebSocketUpgrade` + `split()` で送受信分離 |
| SSE | `Sse<Stream>` で一方向ストリーミング。KeepAlive 設定必須 |
| 本番運用 | Graceful Shutdown、ヘルスチェック、CORS、セキュリティヘッダー |

## 次に読むべきガイド

- [データベース](../04-ecosystem/03-database.md) — SQLx/Diesel/SeaORM の詳細
- [テスト](../04-ecosystem/01-testing.md) — 統合テスト・プロパティテスト
- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — API設計原則
- [ネットワーク](./03-networking.md) — reqwest/WebSocket/gRPC クライアント

## 参考文献

1. **Axum Documentation**: https://docs.rs/axum/latest/axum/
2. **Axum Examples (GitHub)**: https://github.com/tokio-rs/axum/tree/main/examples
3. **Tower Service trait**: https://docs.rs/tower/latest/tower/trait.Service.html
4. **tower-http**: https://docs.rs/tower-http/latest/tower_http/
5. **axum-extra**: https://docs.rs/axum-extra/latest/axum_extra/
6. **sqlx**: https://docs.rs/sqlx/latest/sqlx/
7. **tracing-subscriber**: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/
