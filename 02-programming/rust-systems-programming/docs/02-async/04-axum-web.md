# Axum — Routing, Middleware, and State Management

> Master type-safe Web API construction techniques using the Axum framework, focusing on routing, middleware, and state management.

## What you will learn in this chapter

1. **Routing and Handlers** — Type-safe request parsing through the Extractor pattern
2. **Middleware** — Implementing Tower layers and custom middleware
3. **State Management** — Sharing AppState and integrating database connection pools
4. **Error Handling** — Unified error types and proper response conversion
5. **WebSocket and SSE** — Integrating real-time communication
6. **Testing** — Unit testing handlers and integration testing
7. **Production Operations** — Graceful Shutdown, health checks, metrics


## Prerequisites

Reading the following beforehand will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Content of [Networking — reqwest/hyper, WebSocket, tonic](./03-networking.md)

---

## 1. Axum Architecture

```
┌─────────────────── Axum Processing Flow ─────────────┐
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

### Axum's Design Philosophy

| Design Principle | Description |
|---|---|
| Tower Integration | All middleware is implemented as Tower Service/Layer |
| Type Safety | Type-level request parsing and validation through Extractors |
| Macro-free | The compiler verifies routing through type inference without using derive macros |
| hyper-based | Internally uses hyper to achieve high performance |
| Ecosystem | All tower-http layers can be used as-is |

### Axum's Dependencies

```
┌─────────────── Dependency Tree ───────────────┐
│                                             │
│  axum (Web framework)                       │
│    ├── axum-core (Extractor traits)         │
│    ├── tower (Service, Layer)               │
│    │     ├── tower-service                  │
│    │     └── tower-layer                    │
│    ├── tower-http (HTTP middleware)         │
│    │     ├── TraceLayer                     │
│    │     ├── CorsLayer                      │
│    │     ├── CompressionLayer               │
│    │     └── TimeoutLayer                   │
│    ├── hyper (HTTP/1, HTTP/2)               │
│    ├── tokio (Async runtime)                │
│    └── matchit (URL routing)                │
│                                             │
│  Additional packages:                       │
│    ├── axum-extra (TypedHeader, etc.)       │
│    └── axum-macros (#[debug_handler])       │
└─────────────────────────────────────────────┘
```

---

## 2. Routing and Handlers

### 2.1 Basic CRUD API

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
    println!("Server started: http://localhost:3000");
    axum::serve(listener, app).await.unwrap();
}
```

### 2.2 Router Nesting and Merging

```rust
use axum::{Router, routing::get};

/// API v1 router
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

/// API v2 router (extends v1)
fn api_v2_routes() -> Router<AppState> {
    let users = Router::new()
        .route("/", get(list_users_v2).post(create_user_v2))
        .route("/{id}", get(get_user_v2));

    Router::new()
        .nest("/users", users)
}

/// Build the full application router
fn create_app(state: AppState) -> Router {
    // Admin routes (separate middleware stack)
    let admin = Router::new()
        .route("/stats", get(admin_stats))
        .route("/users", get(admin_list_users))
        .layer(axum::middleware::from_fn(admin_auth_middleware));

    Router::new()
        // API versioning
        .nest("/api/v1", api_v1_routes())
        .nest("/api/v2", api_v2_routes())
        // Admin routes
        .nest("/admin", admin)
        // Health check
        .route("/health", get(health_check))
        // Static files
        .nest_service("/static", tower_http::services::ServeDir::new("./public"))
        // Fallback
        .fallback(fallback_handler)
        .with_state(state)
}

async fn health_check() -> &'static str {
    "OK"
}

async fn fallback_handler() -> (StatusCode, &'static str) {
    (StatusCode::NOT_FOUND, "Route not found")
}
```

### 2.3 Custom Extractors

```rust
use axum::{
    async_trait,
    extract::{FromRequestParts, FromRequest, Request},
    http::{request::Parts, StatusCode, header},
    body::Body,
    Json,
};
use serde::de::DeserializeOwned;

/// Custom Extractor for extracting Bearer tokens
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
            .ok_or((StatusCode::UNAUTHORIZED, "Missing Authorization header"))?;

        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or((StatusCode::UNAUTHORIZED, "Invalid Bearer token format"))?;

        Ok(BearerToken(token.to_string()))
    }
}

/// Extractor for retrieving authenticated user information
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
                let body = serde_json::json!({"error": "Authentication required"});
                (StatusCode::UNAUTHORIZED, Json(body))
            })?;

        let token = auth_header.strip_prefix("Bearer ").ok_or_else(|| {
            let body = serde_json::json!({"error": "Invalid Bearer token format"});
            (StatusCode::UNAUTHORIZED, Json(body))
        })?;

        // Token validation (in practice, JWT validation, etc.)
        validate_token(token).await.map_err(|e| {
            let body = serde_json::json!({"error": format!("Invalid token: {}", e)});
            (StatusCode::UNAUTHORIZED, Json(body))
        })
    }
}

async fn validate_token(token: &str) -> Result<AuthUser, String> {
    // JWT validation logic (simplified)
    if token.starts_with("valid-") {
        Ok(AuthUser {
            user_id: "user-123".to_string(),
            role: "admin".to_string(),
        })
    } else {
        Err("Invalid token".to_string())
    }
}

/// JSON Extractor with validation
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
                let body = serde_json::json!({"error": format!("JSON parse error: {}", e)});
                (StatusCode::BAD_REQUEST, Json(body))
            })?;

        value.validate().map_err(|errors| {
            let body = serde_json::json!({"error": "Validation error", "details": errors});
            (StatusCode::UNPROCESSABLE_ENTITY, Json(body))
        })?;

        Ok(ValidatedJson(value))
    }
}

/// Validation trait
trait Validate {
    fn validate(&self) -> Result<(), Vec<String>>;
}

impl Validate for CreateUser {
    fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push("name is required".to_string());
        }
        if self.name.len() > 100 {
            errors.push("name must be 100 characters or fewer".to_string());
        }
        if !self.email.contains('@') {
            errors.push("email format is invalid".to_string());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

// Use in handlers
async fn protected_handler(
    auth: AuthUser,
) -> String {
    format!("Authenticated. User: {} (role: {})", auth.user_id, auth.role)
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

### 2.4 Multiple Path Parameters and Tuple Extractors

```rust
use axum::extract::Path;

// Single path parameter
async fn get_user(Path(id): Path<u64>) -> String {
    format!("User {}", id)
}

// Multiple path parameters (tuple)
async fn get_user_post(
    Path((user_id, post_id)): Path<(u64, u64)>,
) -> String {
    format!("Post {} of User {}", post_id, user_id)
}

// Named path parameters (struct)
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

// Router definition
// Router::new()
//     .route("/users/{id}", get(get_user))
//     .route("/users/{user_id}/posts/{post_id}", get(get_user_post))
//     .route("/users/{user_id}/posts/{post_id}/comments/{comment_id}", get(get_comment))
```

---

## 3. Middleware

### 3.1 Middleware Stack via Tower Layers

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
    // Request ID header name
    let x_request_id = HeaderName::from_static("x-request-id");

    Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/api/data", get(handler))
        // Middleware is applied bottom-to-top (the last one added runs first)
        .layer(CompressionLayer::new())                     // Response compression
        .layer(RequestBodyLimitLayer::new(1024 * 1024))     // 1MB limit
        .layer(TimeoutLayer::new(Duration::from_secs(30)))  // Timeout
        .layer(CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .max_age(Duration::from_secs(3600)))
        .layer(CatchPanicLayer::new())                      // Catch panics
        .layer(PropagateRequestIdLayer::new(x_request_id.clone()))
        .layer(SetRequestIdLayer::new(x_request_id, MakeRequestUuid))
        .layer(TraceLayer::new_for_http())                  // Request logging
}

async fn handler() -> &'static str {
    "data"
}
```

### Middleware Application Order

```
┌──────────────────────────────────────────┐
│      Order of request processing         │
│                                          │
│  Request → TraceLayer (added last)       │
│          → SetRequestIdLayer             │
│          → PropagateRequestIdLayer       │
│          → CatchPanicLayer               │
│          → CorsLayer                     │
│          → TimeoutLayer                  │
│          → RequestBodyLimitLayer         │
│          → CompressionLayer (added first)│
│          → Handler                       │
│                                          │
│  Response ← returns in reverse order     │
│                                          │
│  * The order of .layer() calls and       │
│    the execution order are reversed      │
│                                          │
│  * To apply middleware to individual     │
│    routes, use route_layer()             │
└──────────────────────────────────────────┘
```

### 3.2 Custom Middleware

```rust
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
    http::StatusCode,
};
use std::time::Instant;

/// Middleware that measures request processing time
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

/// API key authentication middleware
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

/// Rate limiting middleware (simplified)
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

        // Remove old entries
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
    // Rate limit by IP address
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

// Usage example
// let limiter = RateLimiter::new(100, Duration::from_secs(60));
// let app = Router::new()
//     .route("/api/data", get(handler))
//     .layer(axum::middleware::from_fn(timing_middleware))
//     .layer(axum::middleware::from_fn(api_key_middleware))
//     .layer(axum::middleware::from_fn_with_state(limiter, rate_limit_middleware));
```

### 3.3 Per-Route Middleware

```rust
use axum::{Router, routing::get, middleware};

fn create_router_with_route_middleware() -> Router<AppState> {
    // Routes that require authentication
    let protected = Router::new()
        .route("/profile", get(get_profile))
        .route("/settings", get(get_settings).put(update_settings))
        .layer(middleware::from_fn(auth_middleware));

    // Routes that do not require authentication
    let public = Router::new()
        .route("/", get(index))
        .route("/login", post(login))
        .route("/register", post(register));

    // Admin routes (authentication + permission check)
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

## 4. State Management and DB Integration

### 4.1 AppState Design Patterns

```rust
use axum::extract::FromRef;
use sqlx::PgPool;
use reqwest::Client as HttpClient;

/// Shared application state
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

/// FromRef enables extracting individual state components
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

// Handlers can pull individual components
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

### 4.2 SQLx + Axum Integration

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
    .ok_or(AppError::NotFound(format!("Todo {} not found", id)))?;

    Ok(Json(todo))
}

async fn create_todo(
    State(state): State<AppState>,
    Json(input): Json<CreateTodo>,
) -> Result<(StatusCode, Json<Todo>), AppError> {
    if input.title.trim().is_empty() {
        return Err(AppError::BadRequest("title cannot be empty".into()));
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
    // Partial update: COALESCE keeps existing values when NULL is passed
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
    .ok_or(AppError::NotFound(format!("Todo {} not found", id)))?;

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
        Err(AppError::NotFound(format!("Todo {} not found", id)))
    } else {
        Ok(StatusCode::NO_CONTENT)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,sqlx=warn")
        .init();

    // DB connection
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .min_connections(5)
        .acquire_timeout(std::time::Duration::from_secs(5))
        .idle_timeout(std::time::Duration::from_secs(600))
        .connect(&std::env::var("DATABASE_URL")?)
        .await?;

    // Run migrations
    sqlx::migrate!("./migrations").run(&pool).await?;

    let state = AppState { db: pool };

    let app = Router::new()
        .route("/todos", get(list_todos).post(create_todo))
        .route("/todos/{id}", get(get_todo).put(update_todo).delete(delete_todo))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("Server started: http://localhost:3000");
    axum::serve(listener, app).await?;
    Ok(())
}
```

### 4.3 Using Transactions

```rust
use sqlx::{PgPool, Postgres, Transaction};

async fn transfer_funds(
    State(state): State<AppState>,
    Json(input): Json<TransferRequest>,
) -> Result<Json<TransferResult>, AppError> {
    // Begin transaction
    let mut tx: Transaction<'_, Postgres> = state.db.begin().await?;

    // Check sender's balance
    let sender = sqlx::query_as::<_, Account>(
        "SELECT * FROM accounts WHERE id = $1 FOR UPDATE"
    )
    .bind(input.from_account)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or(AppError::NotFound("Sender account not found".into()))?;

    if sender.balance < input.amount {
        return Err(AppError::BadRequest("Insufficient balance".into()));
    }

    // Debit from sender
    sqlx::query("UPDATE accounts SET balance = balance - $1 WHERE id = $2")
        .bind(input.amount)
        .bind(input.from_account)
        .execute(&mut *tx)
        .await?;

    // Credit to recipient
    sqlx::query("UPDATE accounts SET balance = balance + $1 WHERE id = $2")
        .bind(input.amount)
        .bind(input.to_account)
        .execute(&mut *tx)
        .await?;

    // Record the transaction
    sqlx::query(
        "INSERT INTO transactions (from_account, to_account, amount) VALUES ($1, $2, $3)"
    )
    .bind(input.from_account)
    .bind(input.to_account)
    .bind(input.amount)
    .execute(&mut *tx)
    .await?;

    // Commit
    tx.commit().await?;

    Ok(Json(TransferResult {
        success: true,
        message: format!("Transferred {} yen", input.amount),
    }))
}
```

---

## 5. Error Handling

### 5.1 Unified Error Type

```rust
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Unified application error type
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
                tracing::error!("Internal error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "Internal server error".into())
            }
            AppError::Database(e) => {
                tracing::error!("Database error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "database_error", "Database error".into())
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

// Implementing automatic conversions
impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Internal(e)
    }
}

impl From<sqlx::Error> for AppError {
    fn from(e: sqlx::Error) -> Self {
        match &e {
            sqlx::Error::RowNotFound => AppError::NotFound("Resource not found".into()),
            sqlx::Error::Database(db_err) => {
                // PostgreSQL unique constraint violation
                if db_err.code().as_deref() == Some("23505") {
                    AppError::Conflict("Resource already exists".into())
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

// Handler return type
async fn handler() -> Result<Json<serde_json::Value>, AppError> {
    let data = fetch_data().await?;
    Ok(Json(data))
}

async fn fetch_data() -> anyhow::Result<serde_json::Value> {
    Ok(json!({"status": "ok"}))
}
```

### 5.2 Detailed Error Responses

```rust
use axum::response::IntoResponse;

/// Response containing detailed error information
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

/// Helper for generating validation errors
fn validation_error(errors: Vec<FieldError>) -> AppError {
    AppError::Validation(
        errors.iter().map(|e| format!("{}: {}", e.field, e.message)).collect()
    )
}

// Usage example
async fn create_user_handler(
    Json(input): Json<CreateUser>,
) -> Result<Json<User>, AppError> {
    let mut errors = Vec::new();

    if input.name.is_empty() {
        errors.push(FieldError {
            field: "name".into(),
            message: "Name is required".into(),
            code: "required".into(),
        });
    }

    if !input.email.contains('@') {
        errors.push(FieldError {
            field: "email".into(),
            message: "Please enter a valid email address".into(),
            code: "invalid_format".into(),
        });
    }

    if !errors.is_empty() {
        return Err(validation_error(errors));
    }

    // Continue processing...
    todo!()
}
```

---

## 6. WebSocket and SSE

### 6.1 WebSocket Handler

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

    // Server -> Client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Client -> Server
    let tx = state.tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Broadcast
                    let _ = tx.send(text);
                }
                Message::Ping(data) => {
                    // Pong is automatically returned by axum
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

/// SSE stream that watches for DB changes
async fn sse_notifications(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(100);

    // DB change-watching task (PostgreSQL LISTEN/NOTIFY)
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

// Router
// Router::new()
//     .route("/events", get(sse_handler))
//     .route("/notifications", get(sse_notifications))
```

---

## 7. File Upload

### 7.1 Multipart Upload

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

    // Create upload directory
    tokio::fs::create_dir_all(upload_dir).await
        .map_err(|e| AppError::Internal(e.into()))?;

    while let Some(field) = multipart.next_field().await
        .map_err(|e| AppError::BadRequest(format!("Multipart error: {}", e)))? {

        let name = field.name().unwrap_or("unknown").to_string();
        let file_name = field.file_name()
            .unwrap_or("unnamed")
            .to_string();
        let content_type = field.content_type()
            .unwrap_or("application/octet-stream")
            .to_string();

        // Validate file type
        if !ALLOWED_TYPES.contains(&content_type.as_str()) {
            return Err(AppError::BadRequest(
                format!("Disallowed file type: {}", content_type)
            ));
        }

        // Read file data
        let data = field.bytes().await
            .map_err(|e| AppError::BadRequest(format!("Read error: {}", e)))?;

        // Validate size
        if data.len() > MAX_FILE_SIZE {
            return Err(AppError::BadRequest(
                format!("File size exceeds limit: {} bytes (max {})", data.len(), MAX_FILE_SIZE)
            ));
        }

        // Generate a safe filename
        let extension = std::path::Path::new(&file_name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin");
        let saved_name = format!("{}.{}", Uuid::new_v4(), extension);
        let file_path = format!("{}/{}", upload_dir, saved_name);

        // Save the file
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
        return Err(AppError::BadRequest("No files found".into()));
    }

    Ok(Json(UploadResult { files }))
}
```

---

## 8. Testing

### 8.1 Unit Testing Handlers

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

### 8.2 DB Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use sqlx::PgPool;

    /// Build a test DB pool (rolled back per test inside a transaction)
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

        // Create
        let create_body = serde_json::json!({"title": "Test Todo"});
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
        assert_eq!(created.title, "Test Todo");
        assert!(!created.completed);

        // Retrieve
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

### 8.3 Test Helpers

```rust
/// Test HTTP client
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

/// Helper to deserialize a response body
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

## 9. Production Operation Patterns

### 9.1 Graceful Shutdown

```rust
use axum::Router;
use tokio::signal;
use std::time::Duration;

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Ctrl+C received"),
        _ = terminate => tracing::info!("SIGTERM received"),
    }

    tracing::info!("Beginning shutdown...");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = create_app().await?;

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("Server started: http://localhost:3000");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}
```

### 9.2 Health Checks and Metrics

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

    // DB check
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

    // Redis check (omitted; same pattern)
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

/// Readiness check (for Kubernetes)
async fn readiness_check(State(state): State<AppState>) -> StatusCode {
    match sqlx::query("SELECT 1").execute(&state.db).await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Liveness check (for Kubernetes)
async fn liveness_check() -> StatusCode {
    StatusCode::OK
}

// Router::new()
//     .route("/health", get(health_check))
//     .route("/ready", get(readiness_check))
//     .route("/live", get(liveness_check))
```

### 9.3 Structured Logging Setup

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn setup_tracing() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            EnvFilter::new("info,tower_http=debug,axum=trace,sqlx=warn")
        });

    // JSON-formatted logs (production)
    if std::env::var("RUST_LOG_FORMAT").as_deref() == Ok("json") {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        // Human-readable format (development)
        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer().pretty())
            .init();
    }
}
```

### 9.4 CORS and Security Headers

```rust
use tower_http::cors::{CorsLayer, AllowOrigin};
use axum::http::{header, HeaderValue, Method};

fn cors_layer() -> CorsLayer {
    // Development environment
    if cfg!(debug_assertions) {
        CorsLayer::very_permissive()
    } else {
        // Production: explicitly list allowed origins
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

/// Security headers middleware
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

## 10. Comparison Tables

### Rust Web Framework Comparison

| Characteristic | Axum | Actix-web | Rocket | Warp |
|---|---|---|---|---|
| Foundation | hyper + Tower | Custom actor system | Custom | hyper |
| Type safety | Very high | High | Very high | High |
| Performance | Very high | Very high | High | Very high |
| Ecosystem | Tower-compatible | Custom | Custom | Custom |
| Learning curve | Medium | Medium | Low | Medium-high |
| Adoption trend | Rapidly growing | Stable | Stable | Slightly declining |
| WebSocket | Built-in | Built-in | External library | Built-in |
| SSE | Built-in | External library | External library | Limited |
| Middleware | Tower Layer | Transform | Fairing | Filter |
| Testing | oneshot() | TestServer | local client | test::request() |

### List of Extractors

| Extractor | Purpose | Example |
|---|---|---|
| `Path<T>` | URL path parameter | `/users/{id}` -> `Path(id): Path<u64>` |
| `Query<T>` | Query string | `?page=1` -> `Query(p): Query<Params>` |
| `Json<T>` | JSON request body | `Json(body): Json<CreateUser>` |
| `State<T>` | Shared state | DB pool, configuration |
| `HeaderMap` | All headers | Authentication, content negotiation |
| `Extension<T>` | Middleware-injected value | Authenticated user info |
| `Multipart` | File upload | `multipart::Multipart` |
| `ConnectInfo<T>` | Connection info | Client IP address |
| `OriginalUri` | Original URI | Full path before nesting |
| `MatchedPath` | Matched route | Label for metrics |
| `Host` | Host header | Multi-tenant determination |

### List of Response Types

| Type | Purpose | Status Code |
|---|---|---|
| `String` / `&str` | Text response | 200 |
| `Json<T>` | JSON response | 200 |
| `(StatusCode, Json<T>)` | Status + JSON | Any |
| `StatusCode` | Status only | Any |
| `Html<String>` | HTML response | 200 |
| `Redirect` | Redirect | 301/302/307/308 |
| `Sse<S>` | Server-Sent Events | 200 |
| `Response<Body>` | Custom response | Any |
| `impl IntoResponse` | Custom type | Any |

---

## 11. Anti-Patterns

### Anti-pattern 1: Blocking Operations Inside a Handler

```rust
// BAD: Blocking I/O inside an async handler
async fn bad_handler() -> String {
    let data = std::fs::read_to_string("large.csv").unwrap(); // Blocks!
    process(data)
}

// GOOD: Offload via spawn_blocking
async fn good_handler() -> String {
    let data = tokio::task::spawn_blocking(|| {
        std::fs::read_to_string("large.csv").unwrap()
    }).await.unwrap();
    process(data)
}

fn process(data: String) -> String { data }
```

### Anti-pattern 2: Direct Manipulation of Global Mutable State

```rust
// BAD: static mut is unsafe and a source of data races
// static mut COUNTER: u64 = 0;

// GOOD: State + Arc<AtomicU64>
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone)]
struct Metrics {
    request_count: Arc<AtomicU64>,
}

async fn handler(State(metrics): State<Metrics>) -> String {
    let count = metrics.request_count.fetch_add(1, Ordering::Relaxed);
    format!("Request #{}", count + 1)
}
```

### Anti-pattern 3: Wrong Extractor Order

```rust
// BAD: Body-consuming extractors (Json) must come last
// async fn bad(Json(body): Json<MyType>, State(state): State<AppState>) { }
// ^ Compiles, but State may not be available if the body is consumed first

// GOOD: Non-body-consuming extractors first; body-consuming extractor last
async fn good(
    State(state): State<AppState>,     // Does not consume the body
    Path(id): Path<u64>,               // Does not consume the body
    Query(params): Query<ListParams>,  // Does not consume the body
    Json(body): Json<CreateUser>,      // Consumes the body -> last
) -> Result<Json<User>, AppError> {
    todo!()
}
```

### Anti-pattern 4: Swallowing Errors

```rust
// BAD: unwrap() panics -> crashes the server
async fn bad_handler(State(state): State<AppState>) -> Json<Vec<User>> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&state.db)
        .await
        .unwrap(); // Panic!
    Json(users)
}

// GOOD: Result + AppError properly produces error responses
async fn good_handler(
    State(state): State<AppState>,
) -> Result<Json<Vec<User>>, AppError> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&state.db)
        .await?; // Auto-converted to AppError
    Ok(Json(users))
}
```

### Anti-pattern 5: Ignoring Router Cloning Costs

```rust
// BAD: A structure that makes massive state easy to clone
#[derive(Clone)]
struct HeavyState {
    data: Vec<u8>, // Huge data -> cloned on every request
}

// GOOD: Wrap with Arc so only the reference count is cloned
#[derive(Clone)]
struct LightState {
    data: Arc<Vec<u8>>,     // Cloning Arc is cheap
    db: PgPool,              // PgPool internally uses Arc
    cache: Arc<RwLock<HashMap<String, String>>>,
}
```

---

## FAQ

### Q1: How do I nest routers?

**A:** Use `Router::nest` to compose subrouters with a prefix. `Router::merge` combines routers at the same level.

```rust
let api = Router::new()
    .route("/users", get(list_users))
    .route("/posts", get(list_posts));

let app = Router::new()
    .nest("/api/v1", api)           // /api/v1/users, /api/v1/posts
    .route("/health", get(health)); // /health
```

### Q2: How do I write tests?

**A:** Use `tower::ServiceExt::oneshot` to send in-memory requests. For DB integration tests, the `sqlx::test` macro is convenient.

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

### Q3: How do I serve static files?

**A:** Use `tower-http::services::ServeDir`. SPA fallbacks are also supported.

```rust
use tower_http::services::{ServeDir, ServeFile};

let app = Router::new()
    .route("/api/data", get(handler))
    // Static files
    .nest_service("/static", ServeDir::new("./public"))
    // SPA fallback: return index.html when the file doesn't exist
    .fallback_service(
        ServeDir::new("./dist").not_found_service(ServeFile::new("./dist/index.html"))
    );
```

### Q4: How do I manage sessions in Axum?

**A:** Use `axum-extra`'s `CookieJar` or the `tower-sessions` crate.

```rust
use axum_extra::extract::cookie::{CookieJar, Cookie};

async fn login(jar: CookieJar) -> (CookieJar, &'static str) {
    let jar = jar.add(Cookie::new("session_id", "abc123"));
    (jar, "Login successful")
}

async fn profile(jar: CookieJar) -> String {
    match jar.get("session_id") {
        Some(cookie) => format!("Session: {}", cookie.value()),
        None => "Not logged in".into(),
    }
}
```

### Q5: Should I choose Axum or Actix-web?

**A:** Choose based on the following criteria:

- **Axum**: Tower ecosystem integration, unified on tokio only, active development, recommended for new projects
- **Actix-web**: Existing Actix-web projects, when an actor model is needed, mature stability

Performance is very high for both, and there is essentially no practical difference.

### Q6: What is `#[debug_handler]` for?

**A:** It is an attribute macro from the `axum-macros` crate that translates handler type errors into clearer messages. It is useful during development.

```rust
use axum_macros::debug_handler;

#[debug_handler]
async fn handler(
    State(state): State<AppState>,
    Json(body): Json<CreateUser>,
) -> Result<Json<User>, AppError> {
    // When types don't match, you get more readable error messages than usual
    todo!()
}
```

### Q7: How do I use multiple DB pools?

**A:** Hold multiple pools in AppState and use `FromRef` to extract them individually.

```rust
#[derive(Clone)]
struct AppState {
    primary_db: PgPool,
    read_replica: PgPool,
    analytics_db: PgPool,
}

// Use the replica for read-only queries
async fn list_users(
    State(state): State<AppState>,
) -> Result<Json<Vec<User>>, AppError> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users")
        .fetch_all(&state.read_replica) // Replica DB
        .await?;
    Ok(Json(users))
}

// Use the primary for writes
async fn create_user(
    State(state): State<AppState>,
    Json(input): Json<CreateUser>,
) -> Result<(StatusCode, Json<User>), AppError> {
    let user = sqlx::query_as::<_, User>(
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *"
    )
    .bind(&input.name)
    .bind(&input.email)
    .fetch_one(&state.primary_db) // Primary DB
    .await?;
    Ok((StatusCode::CREATED, Json(user)))
}
```

---

## Summary

| Topic | Key Points |
|---|---|
| Router | `route()` for path + method, `nest()` for grouping, `merge()` for combining |
| Extractor | Path, Query, Json, State for type-safe request parsing. Mind the order |
| Middleware | Stack common processing with Tower Layer. Implement custom middleware via `from_fn` |
| State management | Share AppState across all handlers via `with_state()`. Extract individually with `FromRef` |
| Error handling | Unified error type that implements `IntoResponse`. Auto-handled through From conversions |
| DB integration | Store PgPool in AppState. Query via sqlx. Transaction-aware |
| Testing | In-memory tests with `oneshot()`. The TestClient pattern is recommended |
| WebSocket | `WebSocketUpgrade` + `split()` separates send and receive |
| SSE | Use `Sse<Stream>` for one-way streaming. KeepAlive configuration is essential |
| Production operations | Graceful Shutdown, health checks, CORS, security headers |

## Recommended Next Reading

- [Database](../04-ecosystem/03-database.md) — Details on SQLx/Diesel/SeaORM
- [Testing](../04-ecosystem/01-testing.md) — Integration tests and property tests
- [Best Practices](../04-ecosystem/04-best-practices.md) — API design principles
- [Networking](./03-networking.md) — reqwest/WebSocket/gRPC clients

## References

1. **Axum Documentation**: https://docs.rs/axum/latest/axum/
2. **Axum Examples (GitHub)**: https://github.com/tokio-rs/axum/tree/main/examples
3. **Tower Service trait**: https://docs.rs/tower/latest/tower/trait.Service.html
4. **tower-http**: https://docs.rs/tower-http/latest/tower_http/
5. **axum-extra**: https://docs.rs/axum-extra/latest/axum_extra/
6. **sqlx**: https://docs.rs/sqlx/latest/sqlx/
7. **tracing-subscriber**: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/
