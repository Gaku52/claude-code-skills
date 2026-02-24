# Rust ベストプラクティス

> clippy・API 設計・エラーハンドリング・テスト戦略など、Rust で品質の高いコードを書くための実践的な指針を体系的に学ぶ。本章は Rust 中級〜上級者が実務で参照するリファレンスとして設計されている。

## この章で学ぶこと

1. **コード品質ツール** — clippy、rustfmt、cargo-audit による自動品質管理
2. **API 設計原則** — 型駆動設計、ビルダーパターン、ゼロコスト抽象化
3. **エラーハンドリング** — thiserror/anyhow の使い分け、エラー設計の原則
4. **テスト戦略** — ユニットテスト、統合テスト、プロパティベーステスト、モック
5. **パフォーマンス** — メモリ割り当て削減、プロファイリング、並列化
6. **プロジェクト構成** — ワークスペース管理、ドキュメント、CI/CD 統合

---

## 1. clippy による静的解析

### 1.1 品質管理ツールチェーン全体像

```
Rust 品質管理ツールチェーン
============================

ソースコード
    |
    v
[rustfmt]     --> コードフォーマット統一
    |
    v
[clippy]      --> 静的解析 (700+ lint ルール)
    |
    v
[cargo test]  --> ユニットテスト + 統合テスト + doc テスト
    |
    v
[cargo audit] --> 依存関係の脆弱性チェック
    |
    v
[cargo deny]  --> ライセンス・重複依存チェック
    |
    v
[cargo semver-checks] --> セマンティックバージョニング検証
    |
    v
本番ビルド

ツール間の役割分担:
┌──────────────────┬───────────────────────────────────┐
│ ツール            │ 検出対象                           │
├──────────────────┼───────────────────────────────────┤
│ rustfmt          │ フォーマット不統一                   │
│ clippy           │ コードの匂い、非効率、バグの種       │
│ cargo test       │ ロジックエラー、回帰バグ             │
│ cargo audit      │ 既知の脆弱性（CVE）                 │
│ cargo deny       │ ライセンス違反、重複依存             │
│ cargo semver-checks│ 後方互換性のない API 変更          │
│ miri             │ 未定義動作（unsafe コード）          │
│ cargo-fuzz       │ ファジングによるクラッシュ検出        │
└──────────────────┴───────────────────────────────────┘
```

### 1.2 clippy 設定と主要 lint

```toml
# Cargo.toml での clippy 設定
[lints.clippy]
# pedantic レベル（厳格）
pedantic = { level = "warn", priority = -1 }
# 個別に許可
module_name_repetitions = "allow"
must_use_candidate = "allow"

# 追加の警告
nursery = { level = "warn", priority = -1 }
unwrap_used = "deny"
expect_used = "warn"
panic = "deny"

# セキュリティ関連
# SQL インジェクション等の防止
# 暗号の不適切な使用の検出
```

```rust
// clippy が検出する典型的な改善点

// === 不要なクローン ===
// [NG] 不要なクローン
let s = String::from("hello");
let t = s.clone();  // clippy: redundant_clone
println!("{}", t);

// [OK]
let s = String::from("hello");
println!("{}", s);

// === 非効率な文字列結合 ===
// [NG] 非効率な文字列結合
let mut result = String::new();
for item in items {
    result = result + &item.to_string();  // clippy: string_add
}

// [OK] 効率的な結合
let result: String = items.iter().map(|i| i.to_string()).collect();

// === map + unwrap ===
// [NG] map + unwrap
let values: Vec<i32> = strings.iter().map(|s| s.parse().unwrap()).collect();

// [OK] filter_map
let values: Vec<i32> = strings.iter().filter_map(|s| s.parse().ok()).collect();

// === 不要な if let ===
// [NG]
if let Some(value) = option {
    do_something(value);
} else {
    // 何もしない
}

// [OK]
if let Some(value) = option {
    do_something(value);
}

// === 複雑な型の繰り返し ===
// [NG] 同じ型を何度も書く
fn process(data: HashMap<String, Vec<(u64, String, bool)>>) -> HashMap<String, Vec<(u64, String, bool)>> {
    data
}

// [OK] 型エイリアスを使う
type DataMap = HashMap<String, Vec<(u64, String, bool)>>;
fn process(data: DataMap) -> DataMap {
    data
}

// === match の改善 ===
// [NG] 単純な match は if let で
match result {
    Ok(value) => do_something(value),
    Err(_) => (),
}

// [OK]
if let Ok(value) = result {
    do_something(value);
}

// === Iterator のメソッドチェーン ===
// [NG] 手動ループ
let mut count = 0;
for item in &items {
    if item.is_active() {
        count += 1;
    }
}

// [OK] イテレータメソッド
let count = items.iter().filter(|i| i.is_active()).count();
```

### 1.3 clippy の lint レベル詳細

```
clippy lint レベルの階層:

 correctness（正確性）  --- デフォルト: deny
   → 明らかなバグ、未定義動作の可能性
   → 例: approx_constant, infinite_loop, invalid_regex

 suspicious（疑わしい）  --- デフォルト: warn
   → バグの可能性が高いコード
   → 例: suspicious_arithmetic, suspicious_else_formatting

 style（スタイル）  --- デフォルト: warn
   → より慣用的に書ける箇所
   → 例: needless_return, redundant_closure, manual_map

 complexity（複雑性）  --- デフォルト: warn
   → 簡略化できるコード
   → 例: needless_bool, too_many_arguments

 perf（パフォーマンス）  --- デフォルト: warn
   → パフォーマンス改善の余地
   → 例: box_vec, unnecessary_to_owned

 pedantic（衒学的）  --- デフォルト: allow
   → 厳格だが議論の余地がある改善
   → 例: must_use_candidate, missing_errors_doc

 nursery（実験的）  --- デフォルト: allow
   → まだ安定していない新しい lint
```

### 1.4 rustfmt 設定

```toml
# rustfmt.toml
edition = "2021"
max_width = 100
tab_spaces = 4
use_field_init_shorthand = true
use_try_shorthand = true
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true
# nightly 限定の設定（必要に応じて）
# wrap_comments = true
# format_code_in_doc_comments = true
# normalize_comments = true
```

### 1.5 cargo-deny によるサプライチェーンセキュリティ

```toml
# deny.toml
[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
]
copyleft = "deny"  # GPL 系を禁止

[bans]
multiple-versions = "warn"  # 同じクレートの複数バージョン
wildcards = "deny"          # ワイルドカード依存を禁止
deny = [
    # 特定のクレートを禁止
    { name = "openssl", wrappers = ["openssl-sys"] },
]

[advisories]
vulnerability = "deny"
unmaintained = "warn"
unsound = "deny"
yanked = "deny"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

---

## 2. API 設計原則

### 2.1 型駆動設計

```
型駆動設計の考え方
====================

不正な状態を型で表現不可能にする

[NG] 文字列で状態管理
  status: String  --> "active", "inactive", "pending", "actve" (typo!)

[OK] 列挙型で状態管理
  enum Status { Active, Inactive, Pending }
  --> コンパイル時に不正な状態を排除

[NG] Option の連鎖
  name: Option<String>, email: Option<String>  --> 片方だけ None?

[OK] 状態ごとの型
  enum User {
      Anonymous,
      Registered { name: String, email: String },
  }

型駆動設計の原則:
  1. 不正な状態は型レベルで排除する
  2. newtypeパターンでドメイン概念を表現する
  3. 列挙型で状態の網羅性を保証する
  4. PhantomData で型レベルの状態遷移を実現する
```

### 2.2 newtype パターンによるドメイン型

```rust
/// newtype パターン: 基本型にドメイン意味を付与
/// 異なる ID 型を混同するバグを防止

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserId(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderId(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductId(String);

impl UserId {
    pub fn new(id: impl Into<String>) -> Result<Self, ValidationError> {
        let id = id.into();
        if id.is_empty() {
            return Err(ValidationError::new("UserId", "ID は空にできません"));
        }
        if id.len() > 64 {
            return Err(ValidationError::new("UserId", "ID は64文字以内"));
        }
        Ok(Self(id))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// newtype による型安全性
fn get_user_orders(user_id: &UserId, order_repo: &OrderRepo) -> Vec<Order> {
    order_repo.find_by_user(user_id)
}

// コンパイルエラー！ OrderId を UserId の位置に渡せない
// let orders = get_user_orders(&order_id, &repo);  // エラー！

// 正しい使い方
let user_id = UserId::new("user-123")?;
let orders = get_user_orders(&user_id, &repo);
```

### 2.3 ビルダーパターン（Typestate）

```rust
/// 型安全なビルダーパターン（Typestate パターン）
/// コンパイル時に必須フィールドの設定を強制する
pub struct RequestBuilder<S: BuilderState> {
    url: String,
    method: String,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
    timeout: Option<std::time::Duration>,
    _state: std::marker::PhantomData<S>,
}

pub struct NoUrl;
pub struct HasUrl;

pub trait BuilderState {}
impl BuilderState for NoUrl {}
impl BuilderState for HasUrl {}

impl RequestBuilder<NoUrl> {
    pub fn new() -> Self {
        RequestBuilder {
            url: String::new(),
            method: "GET".to_string(),
            headers: Vec::new(),
            body: None,
            timeout: None,
            _state: std::marker::PhantomData,
        }
    }

    /// URL を設定（必須。これを呼ばないと build() できない）
    pub fn url(self, url: impl Into<String>) -> RequestBuilder<HasUrl> {
        RequestBuilder {
            url: url.into(),
            method: self.method,
            headers: self.headers,
            body: self.body,
            timeout: self.timeout,
            _state: std::marker::PhantomData,
        }
    }
}

impl RequestBuilder<HasUrl> {
    pub fn method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    pub fn body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = Some(body.into());
        self
    }

    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// ビルド: HasUrl 状態でのみ呼べる
    pub fn build(self) -> Request {
        Request {
            url: self.url,
            method: self.method,
            headers: self.headers,
            body: self.body,
            timeout: self.timeout.unwrap_or(std::time::Duration::from_secs(30)),
        }
    }
}

// 使用例: url() を呼ばないと build() できない
let req = RequestBuilder::new()
    .url("https://api.example.com")
    .method("POST")
    .header("Content-Type", "application/json")
    .body(b"{\"key\": \"value\"}")
    .timeout(std::time::Duration::from_secs(10))
    .build();

// コンパイルエラー！ url() を呼んでいない
// let req = RequestBuilder::new().build();
```

### 2.4 より複雑な Typestate の例

```rust
/// 接続のライフサイクルを型で表現
/// Disconnected -> Connected -> Authenticated の順でのみ遷移可能

pub struct Connection<S: ConnectionState> {
    inner: TcpStream,
    _state: PhantomData<S>,
}

pub struct Disconnected;
pub struct Connected;
pub struct Authenticated {
    user_id: String,
}

pub trait ConnectionState {}
impl ConnectionState for Disconnected {}
impl ConnectionState for Connected {}
impl ConnectionState for Authenticated {}

impl Connection<Disconnected> {
    pub fn new(addr: &str) -> Result<Connection<Connected>, IoError> {
        let stream = TcpStream::connect(addr)?;
        Ok(Connection {
            inner: stream,
            _state: PhantomData,
        })
    }
}

impl Connection<Connected> {
    pub fn authenticate(
        self,
        username: &str,
        password: &str,
    ) -> Result<Connection<Authenticated>, AuthError> {
        // 認証ロジック...
        Ok(Connection {
            inner: self.inner,
            _state: PhantomData,
        })
    }

    pub fn disconnect(self) {
        // 接続を閉じる
        drop(self.inner);
    }
}

impl Connection<Authenticated> {
    /// 認証済み接続でのみクエリを実行可能
    pub fn query(&mut self, sql: &str) -> Result<QueryResult, DbError> {
        // クエリ実行...
        todo!()
    }

    pub fn disconnect(self) {
        drop(self.inner);
    }
}

// 使用例
let conn = Connection::<Disconnected>::new("localhost:5432")?;
let auth_conn = conn.authenticate("user", "pass")?;
let result = auth_conn.query("SELECT * FROM users")?;

// コンパイルエラー: 未認証接続ではクエリできない
// let conn = Connection::<Disconnected>::new("localhost:5432")?;
// conn.query("SELECT 1");  // エラー！
```

### 2.5 関数の引数設計

```rust
// === 所有権の最小化 ===

// [NG] 不必要に所有権を要求
fn count_words(text: String) -> usize {
    text.split_whitespace().count()
}

// [OK] 参照で十分
fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

// === impl Into<T> による柔軟な API ===

// [NG] String のみ受け付け
fn greet(name: String) {
    println!("Hello, {}!", name);
}
// greet("world".to_string());  // 呼び出し側でtoString必須

// [OK] &str でも String でも受け付け
fn greet(name: impl Into<String>) {
    let name = name.into();
    println!("Hello, {}!", name);
}
// greet("world");  // OK
// greet(String::from("world"));  // OK

// === スライス vs Vec ===

// [NG] Vec を要求
fn sum(numbers: Vec<i32>) -> i32 {
    numbers.iter().sum()
}

// [OK] スライスを受け取る（Vec, 配列, どちらでもOK）
fn sum(numbers: &[i32]) -> i32 {
    numbers.iter().sum()
}

// === AsRef による汎用的な参照変換 ===
fn read_file(path: impl AsRef<std::path::Path>) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}
// read_file("config.toml");                       // &str
// read_file(String::from("config.toml"));         // String
// read_file(std::path::PathBuf::from("config.toml")); // PathBuf

// === Cow による遅延クローン ===
use std::borrow::Cow;

fn normalize_name(name: &str) -> Cow<'_, str> {
    if name.contains(char::is_uppercase) {
        // 変換が必要な場合のみ新しい String を作成
        Cow::Owned(name.to_lowercase())
    } else {
        // そのまま参照を返す（割り当てなし）
        Cow::Borrowed(name)
    }
}
```

### 2.6 戻り値の設計

```rust
// === イテレータを返す（遅延評価） ===

// [NG] Vec を返す（全要素をヒープに割り当て）
fn even_numbers(data: &[i32]) -> Vec<i32> {
    data.iter().filter(|&&x| x % 2 == 0).copied().collect()
}

// [OK] impl Iterator を返す（遅延評価、割り当てなし）
fn even_numbers(data: &[i32]) -> impl Iterator<Item = i32> + '_ {
    data.iter().filter(|&&x| x % 2 == 0).copied()
}

// === Option<&T> vs &Option<T> ===

// [OK] Option<&T> を返す（呼び出し側が値の存在を判断）
fn find_user(&self, id: &UserId) -> Option<&User> {
    self.users.get(id)
}

// === Result にコンテキストを付ける ===
use anyhow::Context;

fn load_config(path: &str) -> anyhow::Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("設定ファイルの読み込みに失敗: {}", path))?;

    let config: Config = toml::from_str(&content)
        .with_context(|| format!("設定ファイルのパースに失敗: {}", path))?;

    config.validate()
        .with_context(|| format!("設定のバリデーションに失敗: {}", path))?;

    Ok(config)
}
```

---

## 3. エラーハンドリング戦略

### 3.1 エラー型の設計

```rust
use thiserror::Error;

/// アプリケーション全体のエラー型
#[derive(Error, Debug)]
pub enum AppError {
    #[error("データベースエラー: {0}")]
    Database(#[from] sqlx::Error),

    #[error("認証エラー: {0}")]
    Auth(#[from] AuthError),

    #[error("バリデーションエラー: {field} - {message}")]
    Validation { field: String, message: String },

    #[error("リソースが見つかりません: {resource_type} (id={id})")]
    NotFound { resource_type: String, id: String },

    #[error("外部サービスエラー: {service} - {0}", service = .service)]
    ExternalService {
        service: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("レート制限超過: {retry_after_secs}秒後にリトライしてください")]
    RateLimited { retry_after_secs: u64 },

    #[error("内部エラー")]
    Internal(#[source] anyhow::Error),
}

/// 認証固有のエラー型
#[derive(Error, Debug)]
pub enum AuthError {
    #[error("トークンが無効です")]
    InvalidToken,
    #[error("トークンの有効期限切れ")]
    TokenExpired,
    #[error("権限不足: {required} が必要です")]
    InsufficientPermissions { required: String },
    #[error("アカウントがロックされています")]
    AccountLocked,
}

/// HTTP ステータスコードへの変換
impl AppError {
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::Database(_) => 500,
            AppError::Auth(AuthError::InvalidToken) => 401,
            AppError::Auth(AuthError::TokenExpired) => 401,
            AppError::Auth(AuthError::InsufficientPermissions { .. }) => 403,
            AppError::Auth(AuthError::AccountLocked) => 423,
            AppError::Validation { .. } => 400,
            AppError::NotFound { .. } => 404,
            AppError::ExternalService { .. } => 502,
            AppError::RateLimited { .. } => 429,
            AppError::Internal(_) => 500,
        }
    }

    /// ユーザーに見せても安全なメッセージか判定
    pub fn is_safe_to_expose(&self) -> bool {
        matches!(
            self,
            AppError::Validation { .. }
            | AppError::NotFound { .. }
            | AppError::Auth(_)
            | AppError::RateLimited { .. }
        )
    }

    /// エラーレスポンスの生成
    pub fn to_response(&self) -> ErrorResponse {
        ErrorResponse {
            status: self.status_code(),
            message: if self.is_safe_to_expose() {
                self.to_string()
            } else {
                "内部エラーが発生しました".to_string()
            },
            error_code: self.error_code(),
        }
    }

    fn error_code(&self) -> &'static str {
        match self {
            AppError::Database(_) => "DATABASE_ERROR",
            AppError::Auth(AuthError::InvalidToken) => "INVALID_TOKEN",
            AppError::Auth(AuthError::TokenExpired) => "TOKEN_EXPIRED",
            AppError::Auth(AuthError::InsufficientPermissions { .. }) => "FORBIDDEN",
            AppError::Auth(AuthError::AccountLocked) => "ACCOUNT_LOCKED",
            AppError::Validation { .. } => "VALIDATION_ERROR",
            AppError::NotFound { .. } => "NOT_FOUND",
            AppError::ExternalService { .. } => "EXTERNAL_SERVICE_ERROR",
            AppError::RateLimited { .. } => "RATE_LIMITED",
            AppError::Internal(_) => "INTERNAL_ERROR",
        }
    }
}

/// Result 型エイリアス
pub type AppResult<T> = Result<T, AppError>;
```

### 3.2 thiserror vs anyhow の使い分け

```
┌──────────────┬───────────────────────┬───────────────────────┐
│              │ thiserror             │ anyhow                │
├──────────────┼───────────────────────┼───────────────────────┤
│ 用途         │ ライブラリ             │ アプリケーション       │
│ エラー型     │ 独自 enum 定義         │ anyhow::Error (動的)   │
│ パターンマッチ│ 可能                  │ downcast が必要        │
│ コンテキスト  │ 手動で付与             │ .context() で簡単      │
│ 依存関係     │ 少ない                │ anyhow 1つ            │
│ エラーチェーン│ #[from], #[source]    │ 自動                  │
│ 表示         │ #[error("...")] で定義 │ Display 自動実装       │
└──────────────┴───────────────────────┴───────────────────────┘

使い分けのガイドライン:
  - ライブラリのパブリック API → thiserror
  - アプリケーションの内部ロジック → anyhow
  - CLI ツールの main() → anyhow::Result<()>
  - Web サーバーのハンドラ → thiserror（ステータスコード変換のため）
```

```rust
// === ライブラリでの thiserror 使用例 ===
// my_library/src/lib.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("不正な構文 (行 {line}, 列 {column}): {message}")]
    Syntax {
        line: usize,
        column: usize,
        message: String,
    },
    #[error("未知のトークン: {0}")]
    UnknownToken(String),
    #[error("IO エラー")]
    Io(#[from] std::io::Error),
}

// 呼び出し側でパターンマッチ可能
pub fn parse(input: &str) -> Result<Ast, ParseError> {
    // ...
    Err(ParseError::Syntax {
        line: 10,
        column: 5,
        message: "予期しない ')' ".to_string(),
    })
}


// === アプリケーションでの anyhow 使用例 ===
// my_app/src/main.rs
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let config = load_config("config.toml")
        .context("アプリケーションの設定読み込みに失敗")?;

    let db = connect_database(&config.database_url)
        .context("データベース接続に失敗")?;

    let server = start_server(&config, db)
        .context("サーバーの起動に失敗")?;

    server.run().context("サーバーの実行中にエラー")
}

// エラーチェーン表示例:
// Error: サーバーの起動に失敗
//
// Caused by:
//     0: データベース接続に失敗
//     1: Connection refused (os error 111)
```

### 3.3 エラーハンドリングのパターン

```rust
// === パターン1: ? 演算子による伝播 ===
fn process_user(id: &UserId) -> AppResult<UserResponse> {
    let user = user_repo.find(id)
        .map_err(AppError::Database)?
        .ok_or_else(|| AppError::NotFound {
            resource_type: "User".to_string(),
            id: id.to_string(),
        })?;

    let profile = profile_service.get(&user.profile_id)
        .map_err(AppError::Database)?;

    Ok(UserResponse::from(user, profile))
}

// === パターン2: エラーの変換と集約 ===
fn validate_user_input(input: &CreateUserInput) -> AppResult<()> {
    let mut errors = Vec::new();

    if input.name.is_empty() {
        errors.push(("name", "名前は必須です"));
    }
    if input.name.len() > 100 {
        errors.push(("name", "名前は100文字以内です"));
    }
    if !input.email.contains('@') {
        errors.push(("email", "有効なメールアドレスを入力してください"));
    }

    if let Some((field, message)) = errors.first() {
        return Err(AppError::Validation {
            field: field.to_string(),
            message: message.to_string(),
        });
    }

    Ok(())
}

// === パターン3: リトライ付きエラーハンドリング ===
async fn with_retry<F, Fut, T, E>(
    f: F,
    max_retries: u32,
    initial_delay: Duration,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut delay = initial_delay;
    let mut last_error = None;

    for attempt in 0..=max_retries {
        match f().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                tracing::warn!(
                    attempt = attempt + 1,
                    max_retries,
                    error = %e,
                    "リトライ中..."
                );
                last_error = Some(e);
                if attempt < max_retries {
                    tokio::time::sleep(delay).await;
                    delay *= 2; // 指数バックオフ
                }
            }
        }
    }

    Err(last_error.unwrap())
}

// === パターン4: カスタムエラーコンテキスト ===
trait ResultExt<T> {
    fn with_not_found(self, resource_type: &str, id: &str) -> AppResult<T>;
}

impl<T> ResultExt<T> for Option<T> {
    fn with_not_found(self, resource_type: &str, id: &str) -> AppResult<T> {
        self.ok_or_else(|| AppError::NotFound {
            resource_type: resource_type.to_string(),
            id: id.to_string(),
        })
    }
}

// 使用例
let user = user_repo.find(&user_id)?
    .with_not_found("User", user_id.as_str())?;
```

---

## 4. テスト戦略

### 4.1 テストの構成と種類

```
テストピラミッド:

          /\
         /  \       E2E テスト（少数、低速、高コスト）
        /    \
       /------\
      / 統合    \    統合テスト（中程度）
     / テスト    \
    /------------\
   / ユニットテスト \  ユニットテスト（多数、高速、低コスト）
  /________________\

Rust のテスト配置:
  src/
  ├── lib.rs          # #[cfg(test)] mod tests { ... }  ← ユニットテスト
  ├── module_a.rs     # #[cfg(test)] mod tests { ... }  ← ユニットテスト
  └── module_b.rs     # #[cfg(test)] mod tests { ... }  ← ユニットテスト
  tests/
  ├── integration_a.rs  ← 統合テスト（別クレートとしてコンパイル）
  └── integration_b.rs  ← 統合テスト
  benches/
  └── benchmark.rs      ← ベンチマーク（criterion）
```

### 4.2 ユニットテストのパターン

```rust
// src/domain/price.rs
pub fn calculate_price(base: f64, tax_rate: f64, discount: Option<f64>) -> f64 {
    let discounted = match discount {
        Some(d) if (0.0..=1.0).contains(&d) => base * (1.0 - d),
        _ => base,
    };
    discounted * (1.0 + tax_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    // === 基本テスト ===
    #[test]
    fn test_price_without_discount() {
        let result = calculate_price(100.0, 0.1, None);
        assert!((result - 110.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_price_with_discount() {
        let result = calculate_price(100.0, 0.1, Some(0.2));
        assert!((result - 88.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalid_discount_ignored() {
        let result = calculate_price(100.0, 0.1, Some(1.5));
        assert!((result - 110.0).abs() < f64::EPSILON);
    }

    // === テーブル駆動テスト ===
    #[test]
    fn test_price_calculation_table() {
        let cases = vec![
            // (base, tax_rate, discount, expected, description)
            (100.0, 0.10, None,       110.0, "割引なし・税率10%"),
            (100.0, 0.10, Some(0.2),  88.0,  "20%割引・税率10%"),
            (100.0, 0.08, Some(0.5),  54.0,  "50%割引・税率8%"),
            (0.0,   0.10, None,       0.0,   "0円の商品"),
            (100.0, 0.0,  Some(0.0),  100.0, "割引0%・税率0%"),
        ];

        for (base, tax, discount, expected, desc) in cases {
            let result = calculate_price(base, tax, discount);
            assert!(
                (result - expected).abs() < 0.001,
                "Failed for case '{}': expected {}, got {}",
                desc, expected, result
            );
        }
    }

    // === プロパティベーステスト ===
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn price_is_always_non_negative(
            base in 0.0f64..10000.0,
            tax in 0.0f64..0.5,
            discount in proptest::option::of(0.0f64..1.0),
        ) {
            let result = calculate_price(base, tax, discount);
            prop_assert!(result >= 0.0, "Price must be non-negative: {}", result);
        }

        #[test]
        fn discount_reduces_price(
            base in 1.0f64..10000.0,
            tax in 0.0f64..0.5,
            discount in 0.01f64..1.0,
        ) {
            let without = calculate_price(base, tax, None);
            let with = calculate_price(base, tax, Some(discount));
            prop_assert!(with <= without, "Discount should reduce price");
        }
    }
}
```

### 4.3 モックとテストダブル

```rust
// === トレイトベースのモック ===

// プロダクションコード
#[async_trait]
pub trait UserRepository: Send + Sync {
    async fn find(&self, id: &UserId) -> Result<Option<User>, DbError>;
    async fn save(&self, user: &User) -> Result<(), DbError>;
    async fn delete(&self, id: &UserId) -> Result<(), DbError>;
}

pub struct UserService<R: UserRepository> {
    repo: R,
}

impl<R: UserRepository> UserService<R> {
    pub fn new(repo: R) -> Self {
        Self { repo }
    }

    pub async fn get_user(&self, id: &UserId) -> AppResult<User> {
        self.repo
            .find(id)
            .await
            .map_err(AppError::Database)?
            .ok_or_else(|| AppError::NotFound {
                resource_type: "User".to_string(),
                id: id.to_string(),
            })
    }
}

// テスト用モック
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct MockUserRepository {
        users: Mutex<HashMap<String, User>>,
        save_called: Mutex<Vec<User>>,
    }

    impl MockUserRepository {
        fn new() -> Self {
            Self {
                users: Mutex::new(HashMap::new()),
                save_called: Mutex::new(Vec::new()),
            }
        }

        fn with_user(self, user: User) -> Self {
            self.users.lock().unwrap().insert(user.id.to_string(), user);
            self
        }

        fn saved_users(&self) -> Vec<User> {
            self.save_called.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl UserRepository for MockUserRepository {
        async fn find(&self, id: &UserId) -> Result<Option<User>, DbError> {
            Ok(self.users.lock().unwrap().get(id.as_str()).cloned())
        }

        async fn save(&self, user: &User) -> Result<(), DbError> {
            self.save_called.lock().unwrap().push(user.clone());
            self.users.lock().unwrap()
                .insert(user.id.to_string(), user.clone());
            Ok(())
        }

        async fn delete(&self, id: &UserId) -> Result<(), DbError> {
            self.users.lock().unwrap().remove(id.as_str());
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_get_user_found() {
        let user = User {
            id: UserId::new("user-1").unwrap(),
            name: "Test User".to_string(),
        };
        let repo = MockUserRepository::new().with_user(user.clone());
        let service = UserService::new(repo);

        let result = service.get_user(&UserId::new("user-1").unwrap()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().name, "Test User");
    }

    #[tokio::test]
    async fn test_get_user_not_found() {
        let repo = MockUserRepository::new();
        let service = UserService::new(repo);

        let result = service.get_user(&UserId::new("nonexistent").unwrap()).await;
        assert!(matches!(result, Err(AppError::NotFound { .. })));
    }
}
```

### 4.4 統合テストとテストヘルパー

```rust
// tests/common/mod.rs - テストヘルパー
pub struct TestApp {
    pub db: PgPool,
    pub addr: String,
    pub client: reqwest::Client,
}

impl TestApp {
    pub async fn spawn() -> Self {
        let db = setup_test_database().await;
        let app = build_app(db.clone()).await;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", listener.local_addr().unwrap());

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        Self {
            db,
            addr,
            client: reqwest::Client::new(),
        }
    }

    pub async fn create_test_user(&self, name: &str) -> User {
        sqlx::query_as!(
            User,
            "INSERT INTO users (name) VALUES ($1) RETURNING *",
            name
        )
        .fetch_one(&self.db)
        .await
        .unwrap()
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.addr, path)
    }
}

impl Drop for TestApp {
    fn drop(&mut self) {
        // テストDB のクリーンアップ
    }
}

// tests/api/users.rs
use crate::common::TestApp;

#[tokio::test]
async fn test_create_user_success() {
    let app = TestApp::spawn().await;

    let response = app.client
        .post(app.url("/api/users"))
        .json(&serde_json::json!({
            "name": "Test User",
            "email": "test@example.com"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 201);

    let user: UserResponse = response.json().await.unwrap();
    assert_eq!(user.name, "Test User");
}

#[tokio::test]
async fn test_create_user_validation_error() {
    let app = TestApp::spawn().await;

    let response = app.client
        .post(app.url("/api/users"))
        .json(&serde_json::json!({
            "name": "",
            "email": "invalid-email"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
}
```

### 4.5 ドキュメントテスト

```rust
/// 2つの数値を加算する。
///
/// # Examples
///
/// ```
/// use my_crate::add;
///
/// assert_eq!(add(2, 3), 5);
/// assert_eq!(add(-1, 1), 0);
/// ```
///
/// # Panics
///
/// オーバーフローが発生する場合にパニックする（debug ビルドのみ）。
///
/// ```should_panic
/// use my_crate::add;
/// let _ = add(i32::MAX, 1);  // debug ビルドでパニック
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// ファイルを読み込んでパースする。
///
/// # Errors
///
/// - ファイルが見つからない場合
/// - パースに失敗した場合
///
/// # Examples
///
/// ```no_run
/// use my_crate::parse_config;
///
/// let config = parse_config("config.toml")?;
/// println!("{:?}", config);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn parse_config(path: &str) -> anyhow::Result<Config> {
    // ...
    todo!()
}
```

### 4.6 CI/CD 設定（GitHub Actions）

```yaml
# .github/workflows/rust.yml
name: Rust CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: -Dwarnings

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - uses: Swatinem/rust-cache@v2

      - name: Format check
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Test
        run: cargo test --all-features

      - name: Doc test
        run: cargo doc --no-deps --all-features

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Security audit
        run: |
          cargo install cargo-audit
          cargo audit

      - name: License check
        run: |
          cargo install cargo-deny
          cargo deny check

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Code coverage
        run: cargo tarpaulin --all-features --workspace --out xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # MSRV (Minimum Supported Rust Version) チェック
  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@1.75.0  # MSRV
      - run: cargo check --all-features
```

---

## 5. パフォーマンスのベストプラクティス

### 5.1 割り当て回避の比較表

| パターン | ヒープ割り当て | 推奨度 | 説明 |
|---|---|---|---|
| `String` を引数に | 毎回 | 低 | 呼び出し側でクローン必要 |
| `&str` を引数に | なし | 高 | 借用で十分な場合 |
| `impl Into<String>` | 必要時のみ | 高 | 柔軟な API |
| `Cow<'_, str>` | 必要時のみ | 中 | 所有/借用を動的切替 |
| `Vec<T>` を返す | 毎回 | 中 | イテレータを返す方が良い場合あり |
| `impl Iterator` を返す | なし | 高 | 遅延評価 |
| `Box<dyn Trait>` | 毎回 | 低 | 動的ディスパッチ + ヒープ割り当て |
| `impl Trait` | なし | 高 | 静的ディスパッチ、インライン化可能 |
| `Arc<T>` | 最初のみ | 中 | 参照カウント（共有所有権） |

### 5.2 具体的な最適化パターン

```rust
// === 1. String の割り当て削減 ===

// [NG] 毎回新しい String を作成
fn format_name(first: &str, last: &str) -> String {
    let mut result = String::new();
    result.push_str(first);
    result.push(' ');
    result.push_str(last);
    result  // 3回のリアロケーション可能性
}

// [OK] 容量を事前確保
fn format_name(first: &str, last: &str) -> String {
    let mut result = String::with_capacity(first.len() + 1 + last.len());
    result.push_str(first);
    result.push(' ');
    result.push_str(last);
    result  // リアロケーションなし
}

// [Best] format! マクロ
fn format_name(first: &str, last: &str) -> String {
    format!("{} {}", first, last)
}


// === 2. Vec の事前確保 ===

// [NG] 都度リアロケーション
fn collect_even(data: &[i32]) -> Vec<i32> {
    let mut result = Vec::new();
    for &n in data {
        if n % 2 == 0 {
            result.push(n);  // 容量不足時にリアロケーション
        }
    }
    result
}

// [OK] 容量を事前確保
fn collect_even(data: &[i32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(data.len() / 2);
    for &n in data {
        if n % 2 == 0 {
            result.push(n);
        }
    }
    result
}

// [Best] イテレータを使用
fn collect_even(data: &[i32]) -> Vec<i32> {
    data.iter().copied().filter(|n| n % 2 == 0).collect()
}


// === 3. 不要なクローンの削除 ===

// [NG] 不要なクローン
struct Config {
    name: String,
    values: Vec<String>,
}

impl Config {
    // name のクローンが不要
    fn get_name(&self) -> String {
        self.name.clone()  // 毎回ヒープ割り当て
    }
}

// [OK] 参照を返す
impl Config {
    fn name(&self) -> &str {
        &self.name
    }
}


// === 4. SmallVec による小さな配列の最適化 ===
use smallvec::SmallVec;

// 通常は要素数が少ない場合、ヒープ割り当てを回避
fn parse_tags(input: &str) -> SmallVec<[&str; 4]> {
    // 4個以下ならスタック上に配置
    // 5個以上になったら自動的にヒープに切り替え
    input.split(',').map(str::trim).collect()
}


// === 5. 文字列インターン / アリーナアロケータ ===
use bumpalo::Bump;

fn process_many_strings(inputs: &[String]) {
    let arena = Bump::new();

    // アリーナに確保: 一括解放されるため個別の dealloc 不要
    let processed: Vec<&str> = inputs
        .iter()
        .map(|s| {
            let allocated = arena.alloc_str(&s.to_uppercase());
            &*allocated
        })
        .collect();

    // processed を使用...

    // arena がドロップされると全て一括解放
}
```

### 5.3 プロファイリングツール

```
パフォーマンス計測ツール:

┌──────────────────┬──────────────────────────────────────┐
│ ツール            │ 用途                                  │
├──────────────────┼──────────────────────────────────────┤
│ criterion        │ マイクロベンチマーク（統計的に信頼性高）  │
│ flamegraph       │ CPU プロファイル可視化                  │
│ perf (Linux)     │ CPU サンプリングプロファイラ             │
│ Instruments (Mac)│ macOS 用プロファイラ                   │
│ heaptrack        │ ヒープ割り当てプロファイリング           │
│ dhat             │ ヒープ割り当て分析（Valgrind）          │
│ cachegrind       │ キャッシュヒット率分析                  │
│ tokio-console    │ async タスクのデバッグ・モニタリング     │
└──────────────────┴──────────────────────────────────────┘
```

```rust
// criterion によるベンチマーク
// benches/benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn fibonacci_iterative(n: u64) -> u64 {
    if n <= 1 { return n; }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fibonacci");

    for size in [10, 20, 30].iter() {
        group.bench_with_input(
            BenchmarkId::new("recursive", size),
            size,
            |b, &n| b.iter(|| fibonacci(black_box(n))),
        );
        group.bench_with_input(
            BenchmarkId::new("iterative", size),
            size,
            |b, &n| b.iter(|| fibonacci_iterative(black_box(n))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fibonacci);
criterion_main!(benches);
```

### 5.4 並列化（rayon）

```rust
use rayon::prelude::*;

// === CPU バウンドの並列化 ===

// 逐次処理
fn process_images_sequential(images: &[Image]) -> Vec<ProcessedImage> {
    images.iter().map(|img| process_image(img)).collect()
}

// 並列処理（rayon）
fn process_images_parallel(images: &[Image]) -> Vec<ProcessedImage> {
    images.par_iter().map(|img| process_image(img)).collect()
}

// 並列ソート
fn sort_large_data(data: &mut [f64]) {
    data.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
}

// 並列リデュース
fn parallel_sum(data: &[f64]) -> f64 {
    data.par_iter().sum()
}

// カスタム並列度
fn with_custom_thread_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        // このブロック内の par_iter は4スレッドで実行
        let result: Vec<_> = (0..1000)
            .into_par_iter()
            .map(|i| expensive_computation(i))
            .collect();
    });
}
```

---

## 6. プロジェクト構成

### 6.1 ワークスペース構成

```
推奨プロジェクト構成:

my-project/
├── Cargo.toml          # ワークスペースルート
├── deny.toml           # cargo-deny 設定
├── rustfmt.toml        # フォーマット設定
├── .github/
│   └── workflows/
│       └── ci.yml
├── crates/
│   ├── my-app/         # バイナリクレート
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs
│   ├── my-core/        # コアロジック（ドメイン）
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── domain/
│   │       ├── service/
│   │       └── error.rs
│   ├── my-api/         # Web API レイヤー
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── routes/
│   │       ├── middleware/
│   │       └── extractors/
│   ├── my-db/          # データベースレイヤー
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── models/
│   │       ├── repositories/
│   │       └── migrations/
│   └── my-shared/      # 共通型・ユーティリティ
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           └── types.rs
└── tests/              # ワークスペースレベルの統合テスト
    └── e2e/
```

```toml
# ルート Cargo.toml
[workspace]
resolver = "2"
members = [
    "crates/my-app",
    "crates/my-core",
    "crates/my-api",
    "crates/my-db",
    "crates/my-shared",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
rust-version = "1.75"

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
thiserror = "2"
anyhow = "1"
sqlx = { version = "0.8", features = ["runtime-tokio", "postgres"] }

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -1 }
unwrap_used = "deny"
```

### 6.2 依存関係の方向

```
クレート間の依存関係:

  ┌──────────┐
  │  my-app  │ (バイナリ: main.rs)
  └──────────┘
    │  │  │
    │  │  └─────────────────────┐
    │  └───────────┐            │
    ▼              ▼            ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  my-api  │  │  my-db   │  │  my-core │
  └──────────┘  └──────────┘  └──────────┘
    │              │            │
    │              │            │
    ▼              ▼            ▼
  ┌──────────────────────────────────────┐
  │            my-shared                  │
  └──────────────────────────────────────┘

  ルール:
  - 矢印の方向にのみ依存可能
  - my-shared は他のクレートに依存しない
  - my-core は my-db に依存しない（トレイトで抽象化）
  - 循環依存は禁止
```

### 6.3 feature フラグの活用

```toml
# crates/my-core/Cargo.toml
[features]
default = []
# テスト用のヘルパーを公開
test-helpers = []
# ベンチマーク用の内部関数公開
bench = []
# 追加機能
advanced-analytics = ["dep:stats"]
```

```rust
// feature フラグによる条件付きコンパイル
#[cfg(feature = "test-helpers")]
pub mod test_helpers {
    use super::*;

    pub fn create_test_user() -> User {
        User {
            id: UserId::new("test-user").unwrap(),
            name: "Test User".to_string(),
            email: Email::new("test@example.com").unwrap(),
        }
    }
}
```

---

## 7. アンチパターン

### 7.1 過剰な unwrap/expect

**問題**: `unwrap()` はパニックを引き起こし、サーバーアプリケーションではプロセスが停止する。

```rust
// [NG] 連続する unwrap
let config = std::fs::read_to_string("config.toml").unwrap();
let port: u16 = env::var("PORT").unwrap().parse().unwrap();

// [OK] 適切なエラーハンドリング
let config = std::fs::read_to_string("config.toml")
    .context("Failed to read config.toml")?;
let port: u16 = env::var("PORT")
    .unwrap_or_else(|_| "8080".to_string())
    .parse()
    .context("Invalid PORT value")?;

// unwrap() が許される場面:
// 1. テストコード
#[test]
fn test_something() {
    let result = my_function().unwrap();  // テストでは OK
}

// 2. 論理的に失敗しないことが証明できる場合（コメント必須）
let regex = Regex::new(r"^\d+$").unwrap();  // 定数パターンは失敗しない

// 3. プロトタイプ・実験コード（本番前に除去）
```

### 7.2 不要な Clone の多用

**問題**: 所有権の問題を Clone で回避すると、不要なメモリ割り当てが増え、パフォーマンスが劣化する。

```rust
// [NG] 不要なクローン
fn process(items: &[String]) {
    for item in items {
        let owned = item.clone();  // 毎回ヒープ割り当て
        do_something(owned);
    }
}

// [OK] 借用で十分
fn process(items: &[String]) {
    for item in items {
        do_something_ref(item);  // &str で受け取る
    }
}

// [NG] Arc の不要なクローン
fn process_shared(data: Arc<Vec<String>>) {
    let cloned = data.clone();  // Arc のクローンは安い...
    let cloned2 = data.clone(); // ...が、不要なら避ける

    // 1つの関数内で同じ Arc を複数回クローンする必要はない
    use_data(&data);
    use_data_again(&data);
}

// [OK]
fn process_shared(data: &Arc<Vec<String>>) {
    use_data(data);
    use_data_again(data);
}
```

### 7.3 過剰な抽象化

```rust
// [NG] 不要なトレイト・ジェネリクス
trait Addable<T> {
    fn add(&self, other: &T) -> T;
}

impl Addable<i32> for i32 {
    fn add(&self, other: &i32) -> i32 {
        self + other
    }
}

// [OK] 単純に関数で十分
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// 抽象化が正当化される場面:
// - 複数の実装が実際に存在する場合
// - テスト時にモックが必要な場合
// - 将来的な拡張が確実に予想される場合
```

### 7.4 unsafe の不適切な使用

```rust
// [NG] 安全な代替手段があるのに unsafe を使う
unsafe fn get_element(slice: &[i32], index: usize) -> i32 {
    *slice.get_unchecked(index)  // 境界チェックをスキップ
}

// [OK] 安全な方法
fn get_element(slice: &[i32], index: usize) -> Option<i32> {
    slice.get(index).copied()
}

// unsafe が正当化される場面:
// - FFI（外部関数インターフェース）
// - パフォーマンスクリティカルなホットパス（プロファイリングで確認済み）
// - 低レベルのメモリ操作（カスタムアロケータなど）
// ※ 必ず Safety コメントを書く

/// # Safety
///
/// `ptr` は有効な `T` へのポインタであり、
/// 少なくとも `len` 個の連続した `T` が読み取り可能でなければならない。
unsafe fn read_buffer<T>(ptr: *const T, len: usize) -> Vec<T>
where
    T: Copy,
{
    let slice = std::slice::from_raw_parts(ptr, len);
    slice.to_vec()
}
```

---

## 8. ドキュメント作成のベストプラクティス

### 8.1 ドキュメントコメントの規約

```rust
/// ユーザーアカウントを表すドメインモデル。
///
/// ユーザーは作成後、アクティブ化してからでないと
/// 操作を行うことはできない。
///
/// # Examples
///
/// ```
/// use my_crate::User;
///
/// let user = User::new("alice", "alice@example.com")?;
/// assert_eq!(user.name(), "alice");
/// assert!(!user.is_active());
/// # Ok::<(), my_crate::Error>(())
/// ```
///
/// # Errors
///
/// - 名前が空文字列の場合
/// - メールアドレスの形式が不正な場合
pub struct User {
    // ...
}

/// ユーザーを作成する。
///
/// 新しいユーザーは非アクティブ状態で作成される。
/// [`User::activate`] を呼んでアクティブ化する必要がある。
///
/// # Arguments
///
/// * `name` - ユーザー名（1文字以上、100文字以内）
/// * `email` - メールアドレス（RFC 5322 準拠）
///
/// # Errors
///
/// * [`ValidationError::EmptyName`] - 名前が空の場合
/// * [`ValidationError::InvalidEmail`] - メールが不正な場合
///
/// # Examples
///
/// ```
/// let user = User::new("alice", "alice@example.com")?;
/// # Ok::<(), my_crate::Error>(())
/// ```
pub fn new(name: &str, email: &str) -> Result<Self, ValidationError> {
    // ...
    todo!()
}
```

---

## API 設計の指針比較表

| 原則 | 良い例 | 悪い例 |
|---|---|---|
| **型で不変条件を表現** | `NonZeroU32` | `u32`（0チェックを忘れる） |
| **所有権を最小限に** | `fn process(data: &[u8])` | `fn process(data: Vec<u8>)` |
| **列挙型で網羅性** | `match` の全パターン | `if/else` の連鎖 |
| **newtype で意味付け** | `struct UserId(u64)` | `u64`（何の ID?） |
| **ビルダーで複雑な構築** | `Config::builder().port(8080).build()` | `Config::new(8080, ...)` 引数多数 |
| **From/Into で変換** | `impl From<String> for Name` | 手動変換関数 |
| **イテレータを返す** | `fn items(&self) -> impl Iterator` | `fn items(&self) -> Vec<T>` |
| **エラー型を分ける** | ライブラリ用/アプリ用で分離 | 全部 `String` で返す |

---

## FAQ

### Q1: clippy の警告をすべて修正すべきですか？

**A**: `warn` レベルの lint は基本的に修正すべきです。ただし、`pedantic` レベルには過度に厳格なものもあるため、プロジェクトに合わせて `#[allow]` で個別に許可できます。`deny` に設定した lint は CI で必ずチェックしてください。チームで合意した設定を `Cargo.toml` の `[lints]` セクションに記述し、全員が同じルールを使うようにしましょう。

### Q2: anyhow と thiserror はどう使い分けますか？

**A**:
- **thiserror**: ライブラリで使用。呼び出し側がエラー型をパターンマッチで処理できる
- **anyhow**: アプリケーションで使用。エラーの種類よりもコンテキスト情報が重要な場合
ライブラリでは `thiserror` で型を定義し、アプリケーションの `main` や CLI では `anyhow::Result` で統一するのが一般的です。Web サーバーのハンドラではステータスコード変換が必要なので thiserror が適しています。

### Q3: Rust のパフォーマンスチューニングの進め方は？

**A**: 以下の順序で進めます:
1. **ベンチマーク** を `criterion` で作成し、現状を計測
2. **プロファイル** を `perf`/`flamegraph` で実行し、ホットスポットを特定
3. **アルゴリズム改善** が最優先（O(n^2) -> O(n log n) など）
4. **割り当て削減** — Clone の除去、`String` -> `&str`、`Vec` -> スライス
5. **並列化** — rayon による data parallelism
6. **unsafe** — 最終手段。必ずベンチマークで効果を確認
計測なしの最適化は避けてください。

### Q4: feature フラグはどう設計すべきですか？

**A**: 以下の原則に従います:
- `default` フィーチャーは最小限に（ユーザーが opt-in する設計）
- 重い依存関係は feature フラグで制御
- テスト用ヘルパーは `test-helpers` feature で公開
- `serde` サポートは `serde` feature で opt-in

### Q5: unsafe コードはどう管理すべきですか？

**A**:
- 最小限の範囲で使用する（unsafe ブロックを小さく保つ）
- 必ず `# Safety` ドキュメントを書く
- 可能なら安全な抽象をラッパーとして提供する
- Miri (`cargo +nightly miri test`) で未定義動作を検出
- `unsafe_code` lint を `deny` に設定し、明示的に許可する

---

## まとめ

| 項目 | 要点 |
|---|---|
| clippy | 700+ の lint で一般的なミスを防止。CI で必須 |
| rustfmt | コードフォーマットの統一。チーム開発で必須 |
| cargo-deny | サプライチェーンセキュリティ。ライセンス管理 |
| API 設計 | 型で不変条件を表現。所有権は最小限に |
| エラー処理 | ライブラリは thiserror、アプリは anyhow |
| テスト | ユニット + 統合 + プロパティベース + ドキュメントテスト |
| パフォーマンス | 計測優先。不要な Clone と割り当てを排除 |
| プロジェクト構成 | ワークスペースで分割。依存の方向を一方向に |
| ドキュメント | Examples, Errors, Safety を必ず書く |

---

## 演習問題

### 演習1: 型駆動設計

以下の構造体を、不正な状態が表現不可能になるよう再設計せよ。

```rust
struct User {
    name: String,           // 空文字列もOK？
    email: String,          // 不正な形式もOK？
    age: i32,               // 負の値もOK？
    role: String,           // "admin", "user", "guest" 以外もOK？
    verified: bool,
    verification_code: Option<String>, // verified=true なのに code がある？
}
```

### 演習2: エラー型の設計

EC サイトの注文処理で発生しうるエラーを thiserror で定義せよ。最低5種類のエラーバリアントを含め、HTTP ステータスコードへの変換メソッドも実装すること。

### 演習3: テスト戦略

以下の関数に対して、(1) 通常のユニットテスト、(2) テーブル駆動テスト、(3) プロパティベーステスト を書け。

```rust
pub fn parse_csv_line(line: &str) -> Vec<String> {
    // カンマ区切りでフィールドを分割
    // ダブルクォートで囲まれたフィールド内のカンマは無視
    // ダブルクォート内の "" はエスケープされた "
    todo!()
}
```

### 演習4: パフォーマンス改善

以下のコードのパフォーマンスを改善せよ。改善前後で criterion ベンチマークを書き、効果を測定すること。

```rust
fn count_word_frequencies(text: &str) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        let word = word.to_lowercase();
        let count = freq.entry(word.clone()).or_insert(0);
        *count += 1;
    }
    freq
}
```

### 演習5: ワークスペース設計

ブログシステム（記事の CRUD、ユーザー認証、コメント機能）のワークスペース構成を設計せよ。各クレートの責任と依存関係を図示し、Cargo.toml を書くこと。

---

## 次に読むべきガイド

- [FFI](../03-systems/02-ffi-interop.md) — 他言語連携のベストプラクティス
- [非同期プログラミング](../02-advanced/01-async.md) — async Rust の設計パターン

## 参考文献

1. **Rust API Guidelines**: [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) — 公式 API 設計ガイドライン
2. **clippy Lints**: [Clippy Lint List](https://rust-lang.github.io/rust-clippy/master/) — 全 lint の一覧と説明
3. **Rust Design Patterns**: [Rust Design Patterns](https://rust-unofficial.github.io/patterns/) — Rust 特有の設計パターン集
4. **Rust Performance Book**: [The Rust Performance Book](https://nnethercote.github.io/perf-book/) — パフォーマンスチューニング
5. **Error Handling in Rust**: [Error Handling Survey](https://blog.burntsushi.net/rust-error-handling/) — エラーハンドリングの包括的ガイド
6. **Cargo Book**: [The Cargo Book](https://doc.rust-lang.org/cargo/) — Cargo の公式ドキュメント
