# テスト — proptest、criterion

> Rust のテストエコシステムを単体テスト・統合テスト・プロパティテスト・ベンチマークまで網羅的に習得する

## この章で学ぶこと

1. **標準テスト** — #[test], #[cfg(test)], assert マクロ、テスト構成
2. **プロパティテスト** — proptest / quickcheck による性質ベーステスト
3. **ベンチマーク** — criterion による統計的パフォーマンス測定
4. **テスト設計パターン** — モック、スタブ、フィクスチャ、テスタブルアーキテクチャ
5. **テスト自動化** — カバレッジ計測、CI 統合、ミューテーションテスト

---

## 1. テスト体系の全体像

```
┌────────────────── Rust テスト体系 ──────────────────┐
│                                                      │
│  ┌─ 単体テスト (Unit Test) ──────────────────────┐  │
│  │  src/ 内に #[cfg(test)] mod tests で定義       │  │
│  │  プライベート関数もテスト可能                    │  │
│  │  $ cargo test --lib                            │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ 統合テスト (Integration Test) ────────────────┐  │
│  │  tests/ ディレクトリに配置                      │  │
│  │  公開APIのみテスト (外部クレートとして扱う)      │  │
│  │  $ cargo test --test test_name                 │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ ドキュメントテスト (Doc Test) ────────────────┐  │
│  │  /// コメント内のコードブロック                  │  │
│  │  ドキュメントと同時にテストも維持                │  │
│  │  $ cargo test --doc                            │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ プロパティテスト ─────────────────────────────┐  │
│  │  proptest / quickcheck                          │  │
│  │  ランダム入力で性質を検証                        │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ ベンチマーク ─────────────────────────────────┐  │
│  │  criterion / divan                              │  │
│  │  統計的なパフォーマンス計測                      │  │
│  │  $ cargo bench                                 │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 1.1 テストファイル配置の規約

```
my-project/
├── src/
│   ├── lib.rs              # ライブラリルート + 単体テスト
│   ├── parser.rs           # 各モジュール内に #[cfg(test)] mod tests
│   ├── validator.rs
│   └── utils/
│       └── mod.rs
├── tests/                   # 統合テスト
│   ├── common/
│   │   └── mod.rs          # テスト共通ヘルパー（テストとして実行されない）
│   ├── api_test.rs         # 各ファイルが独立したテストクレート
│   ├── parser_test.rs
│   └── e2e/
│       └── main.rs         # サブディレクトリはバイナリクレートとして扱う
├── benches/
│   ├── parser_bench.rs     # criterion ベンチマーク
│   └── sorting_bench.rs
└── examples/
    └── basic.rs            # cargo test --examples でテスト可能
```

---

## 2. 標準テスト

### コード例1: 単体テストの基本パターン

```rust
// src/lib.rs
pub struct Calculator;

impl Calculator {
    pub fn add(a: i64, b: i64) -> i64 {
        a + b
    }

    pub fn divide(a: f64, b: f64) -> Result<f64, &'static str> {
        if b == 0.0 {
            Err("ゼロ除算エラー")
        } else {
            Ok(a / b)
        }
    }

    /// 内部ヘルパー (プライベート)
    fn clamp(value: i64, min: i64, max: i64) -> i64 {
        value.max(min).min(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_basic() {
        assert_eq!(Calculator::add(2, 3), 5);
    }

    #[test]
    fn test_add_negative() {
        assert_eq!(Calculator::add(-1, 1), 0);
        assert_eq!(Calculator::add(-5, -3), -8);
    }

    #[test]
    fn test_divide_success() {
        let result = Calculator::divide(10.0, 3.0).unwrap();
        assert!((result - 3.333).abs() < 0.001, "結果が期待値と異なる: {}", result);
    }

    #[test]
    fn test_divide_by_zero() {
        let result = Calculator::divide(1.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "ゼロ除算エラー");
    }

    // プライベート関数もテスト可能
    #[test]
    fn test_clamp() {
        assert_eq!(Calculator::clamp(5, 0, 10), 5);
        assert_eq!(Calculator::clamp(-1, 0, 10), 0);
        assert_eq!(Calculator::clamp(15, 0, 10), 10);
    }

    // パニックの検証
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_index_panic() {
        let v = vec![1, 2, 3];
        let _ = v[5];
    }

    // 条件付きスキップ
    #[test]
    #[ignore = "CI環境でのみ実行"]
    fn test_slow_integration() {
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}
```

### 2.1 assert マクロの詳細と活用

```rust
#[cfg(test)]
mod assert_examples {
    // --- 基本的な assert ---
    #[test]
    fn test_assert_basic() {
        let x = 42;
        assert!(x > 0);
        assert!(x > 0, "x は正数であるべき: x = {}", x);
    }

    // --- 等値比較 ---
    #[test]
    fn test_assert_eq() {
        let expected = vec![1, 2, 3];
        let actual = vec![1, 2, 3];
        // 失敗時: left と right の値が表示される
        assert_eq!(expected, actual);
        assert_eq!(expected, actual, "ベクタが一致しない");
    }

    // --- 非等値比較 ---
    #[test]
    fn test_assert_ne() {
        let a = "hello";
        let b = "world";
        assert_ne!(a, b);
    }

    // --- 浮動小数点の比較（epsilon ベース） ---
    #[test]
    fn test_float_comparison() {
        let result = 0.1 + 0.2;
        let expected = 0.3;
        let epsilon = 1e-10;

        assert!(
            (result - expected).abs() < epsilon,
            "浮動小数点比較失敗: {} != {} (差: {})",
            result, expected, (result - expected).abs()
        );
    }

    // --- Result を返すテスト ---
    #[test]
    fn test_with_result() -> Result<(), Box<dyn std::error::Error>> {
        let value: i32 = "42".parse()?;
        assert_eq!(value, 42);
        Ok(())
    }

    // --- debug_assert（リリースビルドでは除去） ---
    #[test]
    fn test_debug_assert() {
        debug_assert!(true, "デバッグビルドでのみチェック");
        debug_assert_eq!(1 + 1, 2);
        debug_assert_ne!(1, 2);
    }

    // --- カスタム assert マクロ ---
    macro_rules! assert_between {
        ($value:expr, $min:expr, $max:expr) => {
            assert!(
                $value >= $min && $value <= $max,
                "{} は {} から {} の範囲外です (実際: {})",
                stringify!($value), $min, $max, $value
            );
        };
    }

    #[test]
    fn test_custom_assert() {
        let score = 85;
        assert_between!(score, 0, 100);
    }
}
```

### 2.2 テストの構造化パターン

```rust
// テストモジュールの階層化
#[cfg(test)]
mod tests {
    use super::*;

    // --- Arrange-Act-Assert (AAA) パターン ---
    #[test]
    fn test_user_registration() {
        // Arrange: テストデータの準備
        let email = "test@example.com";
        let password = "SecureP@ss123";
        let mut service = UserService::new(MockDatabase::new());

        // Act: テスト対象の実行
        let result = service.register(email, password);

        // Assert: 結果の検証
        assert!(result.is_ok());
        let user = result.unwrap();
        assert_eq!(user.email, email);
        assert!(user.password_hash != password); // ハッシュ化されている
    }

    // --- Given-When-Then パターン ---
    mod given_valid_input {
        use super::*;

        mod when_parsing {
            use super::*;

            #[test]
            fn then_returns_correct_value() {
                let input = "42";
                let result: Result<i32, _> = input.parse();
                assert_eq!(result.unwrap(), 42);
            }
        }
    }

    mod given_invalid_input {
        use super::*;

        mod when_parsing {
            use super::*;

            #[test]
            fn then_returns_error() {
                let input = "not_a_number";
                let result: Result<i32, _> = input.parse();
                assert!(result.is_err());
            }
        }
    }

    // --- テストフィクスチャ ---
    struct TestFixture {
        db: MockDatabase,
        service: UserService,
        admin_user: User,
    }

    impl TestFixture {
        fn new() -> Self {
            let db = MockDatabase::new();
            let service = UserService::new(db.clone());
            let admin_user = User {
                id: 1,
                email: "admin@example.com".to_string(),
                role: Role::Admin,
                ..Default::default()
            };
            TestFixture { db, service, admin_user }
        }

        fn with_users(mut self, count: usize) -> Self {
            for i in 0..count {
                self.db.insert_user(User {
                    id: i as u64 + 100,
                    email: format!("user{}@example.com", i),
                    role: Role::User,
                    ..Default::default()
                });
            }
            self
        }
    }

    #[test]
    fn test_with_fixture() {
        let fixture = TestFixture::new().with_users(5);
        let users = fixture.service.list_users(&fixture.admin_user);
        assert_eq!(users.unwrap().len(), 5);
    }
}
```

### コード例2: 統合テストとテストヘルパー

```rust
// tests/common/mod.rs — テストヘルパー
pub struct TestContext {
    pub temp_dir: tempfile::TempDir,
    pub config: String,
}

impl TestContext {
    pub fn new() -> Self {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{ "data_dir": "{}" }}"#,
            temp_dir.path().display()
        );
        TestContext { temp_dir, config }
    }

    pub fn write_test_file(&self, name: &str, content: &str) {
        let path = self.temp_dir.path().join(name);
        std::fs::write(path, content).unwrap();
    }
}

// tests/integration_test.rs
mod common;

use my_lib::Calculator;

#[test]
fn test_full_workflow() {
    let ctx = common::TestContext::new();
    ctx.write_test_file("input.txt", "10\n20\n30");

    let path = ctx.temp_dir.path().join("input.txt");
    let content = std::fs::read_to_string(path).unwrap();
    let sum: i64 = content.lines()
        .filter_map(|line| line.parse::<i64>().ok())
        .fold(0, |acc, x| Calculator::add(acc, x));

    assert_eq!(sum, 60);
}
```

### 2.3 高度な統合テストパターン

```rust
// tests/api_integration.rs — Web API の統合テスト
use axum::http::StatusCode;
use reqwest::Client;
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// テスト用のサーバーを起動するヘルパー
async fn spawn_test_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let app = my_app::create_app().await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // サーバー起動を待つ
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    (addr, handle)
}

#[tokio::test]
async fn test_health_check() {
    let (addr, _handle) = spawn_test_server().await;
    let client = Client::new();

    let response = client
        .get(format!("http://{}/health", addr))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_create_and_get_user() {
    let (addr, _handle) = spawn_test_server().await;
    let client = Client::new();
    let base_url = format!("http://{}", addr);

    // ユーザー作成
    let create_response = client
        .post(format!("{}/api/users", base_url))
        .json(&serde_json::json!({
            "name": "テストユーザー",
            "email": "test@example.com"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(create_response.status(), StatusCode::CREATED);
    let created_user: serde_json::Value = create_response.json().await.unwrap();
    let user_id = created_user["id"].as_u64().unwrap();

    // ユーザー取得
    let get_response = client
        .get(format!("{}/api/users/{}", base_url, user_id))
        .send()
        .await
        .unwrap();

    assert_eq!(get_response.status(), StatusCode::OK);
    let user: serde_json::Value = get_response.json().await.unwrap();
    assert_eq!(user["name"], "テストユーザー");
    assert_eq!(user["email"], "test@example.com");
}
```

### 2.4 Testcontainers による DB テスト

```rust
// tests/database_test.rs — testcontainers を使った実際の DB テスト
use testcontainers::{clients::Cli, images::postgres::Postgres};
use sqlx::PgPool;

async fn setup_test_db(docker: &Cli) -> (PgPool, testcontainers::Container<Postgres>) {
    let container = docker.run(Postgres::default());
    let port = container.get_host_port_ipv4(5432);
    let database_url = format!(
        "postgres://postgres:postgres@localhost:{}/postgres",
        port
    );

    let pool = PgPool::connect(&database_url).await.unwrap();

    // マイグレーション実行
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .unwrap();

    (pool, container)
}

#[tokio::test]
async fn test_user_repository() {
    let docker = Cli::default();
    let (pool, _container) = setup_test_db(&docker).await;

    let repo = UserRepository::new(pool.clone());

    // ユーザー作成
    let user = repo.create("test@example.com", "Test User").await.unwrap();
    assert_eq!(user.email, "test@example.com");

    // ユーザー検索
    let found = repo.find_by_email("test@example.com").await.unwrap();
    assert!(found.is_some());
    assert_eq!(found.unwrap().name, "Test User");

    // 存在しないユーザー
    let not_found = repo.find_by_email("nobody@example.com").await.unwrap();
    assert!(not_found.is_none());
}
```

### コード例3: 非同期テスト

```rust
// tokio のテストマクロ
#[tokio::test]
async fn test_async_operation() {
    let result = async_fetch_data("test").await;
    assert!(result.is_ok());
}

// タイムアウト付きテスト
#[tokio::test(flavor = "multi_thread")]
async fn test_with_timeout() {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        long_running_task(),
    ).await;
    assert!(result.is_ok(), "タイムアウト!");
}

// テスト用の時間制御
#[tokio::test]
async fn test_time_control() {
    tokio::time::pause(); // 仮想時間モード
    let start = tokio::time::Instant::now();
    tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
    // 実際には即座に完了 (仮想時間が進む)
    assert!(start.elapsed() >= tokio::time::Duration::from_secs(3600));
}

async fn async_fetch_data(_: &str) -> Result<String, String> { Ok("data".into()) }
async fn long_running_task() -> String { "done".into() }
```

### 2.5 非同期テストの高度なパターン

```rust
use tokio::sync::mpsc;
use std::sync::Arc;

// --- チャネルを使ったテスト ---
#[tokio::test]
async fn test_producer_consumer() {
    let (tx, mut rx) = mpsc::channel::<String>(10);

    // プロデューサー
    let producer = tokio::spawn(async move {
        for i in 0..5 {
            tx.send(format!("message_{}", i)).await.unwrap();
        }
    });

    // コンシューマー
    let mut received = Vec::new();
    while let Some(msg) = rx.recv().await {
        received.push(msg);
    }

    producer.await.unwrap();
    assert_eq!(received.len(), 5);
    assert_eq!(received[0], "message_0");
    assert_eq!(received[4], "message_4");
}

// --- 並行処理のテスト ---
#[tokio::test]
async fn test_concurrent_access() {
    use tokio::sync::RwLock;

    let data = Arc::new(RwLock::new(Vec::<i32>::new()));
    let mut handles = Vec::new();

    // 複数の書き込みタスク
    for i in 0..10 {
        let data = Arc::clone(&data);
        handles.push(tokio::spawn(async move {
            let mut guard = data.write().await;
            guard.push(i);
        }));
    }

    // 全タスクの完了を待つ
    for handle in handles {
        handle.await.unwrap();
    }

    let result = data.read().await;
    assert_eq!(result.len(), 10);
    // 順序は保証されないが、全要素が含まれるはず
    let mut sorted: Vec<i32> = result.clone();
    sorted.sort();
    assert_eq!(sorted, (0..10).collect::<Vec<_>>());
}

// --- リトライ機能のテスト ---
#[tokio::test]
async fn test_retry_logic() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let attempt_count = Arc::new(AtomicUsize::new(0));

    let count = Arc::clone(&attempt_count);
    let result = retry_with_backoff(3, || {
        let count = Arc::clone(&count);
        async move {
            let attempt = count.fetch_add(1, Ordering::SeqCst);
            if attempt < 2 {
                Err("一時的なエラー".to_string())
            } else {
                Ok("成功".to_string())
            }
        }
    }).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "成功");
    assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
}

async fn retry_with_backoff<F, Fut, T, E>(
    max_retries: usize,
    f: F,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    let mut last_err = None;
    for i in 0..max_retries {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = Some(e);
                if i + 1 < max_retries {
                    tokio::time::sleep(std::time::Duration::from_millis(10 * (i as u64 + 1))).await;
                }
            }
        }
    }
    Err(last_err.unwrap())
}
```

### 2.6 ドキュメントテスト

```rust
/// 2つの数値を加算する。
///
/// # 使用例
///
/// ```
/// use my_lib::add;
///
/// assert_eq!(add(2, 3), 5);
/// assert_eq!(add(-1, 1), 0);
/// ```
///
/// # パニック
///
/// オーバーフロー時にパニックする（デバッグビルド）。
///
/// ```should_panic
/// use my_lib::add;
///
/// // i64::MAX + 1 はオーバーフロー
/// let _ = add(i64::MAX, 1);
/// ```
///
/// # エラーハンドリングの例
///
/// ```
/// use my_lib::divide;
///
/// let result = divide(10.0, 0.0);
/// assert!(result.is_err());
/// ```
///
/// # 非表示コード（セットアップコード）
///
/// ```
/// # // この行はドキュメントには表示されないがテストでは実行される
/// # use my_lib::Calculator;
/// let calc = Calculator::new();
/// assert_eq!(calc.add(1, 2), 3);
/// ```
///
/// # コンパイルのみ（実行しない）
///
/// ```no_run
/// use my_lib::Server;
///
/// // 実行はしないがコンパイルは通ることを確認
/// let server = Server::bind("0.0.0.0:8080").await.unwrap();
/// server.run().await;
/// ```
///
/// # コンパイルもしない（ドキュメント表示用）
///
/// ```ignore
/// // このコードはテストされない（外部依存がある場合等）
/// let result = external_api::call().await;
/// ```
///
/// # テキストブロック（コードとして扱わない）
///
/// ```text
/// これはテキストとして表示されるだけで、テストもコンパイルもされない。
/// ```
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

### 2.7 rstest によるパラメータ化テスト

```rust
// Cargo.toml:
// [dev-dependencies]
// rstest = "0.18"

use rstest::rstest;

// --- パラメータ化テスト（テーブル駆動テスト） ---
#[rstest]
#[case("", true)]
#[case(" ", true)]
#[case("hello", false)]
#[case("  spaces  ", false)]
fn test_is_blank(#[case] input: &str, #[case] expected: bool) {
    assert_eq!(input.trim().is_empty(), expected);
}

// --- フィクスチャ ---
#[rstest::fixture]
fn database() -> MockDatabase {
    let db = MockDatabase::new();
    db.seed_test_data();
    db
}

#[rstest::fixture]
fn service(database: MockDatabase) -> UserService {
    UserService::new(database)
}

#[rstest]
fn test_find_user(service: UserService) {
    let user = service.find_by_id(1).unwrap();
    assert_eq!(user.name, "テストユーザー");
}

// --- 複数パラメータの組み合わせ ---
#[rstest]
fn test_format_combinations(
    #[values("json", "yaml", "toml")] format: &str,
    #[values(true, false)] pretty: bool,
) {
    let config = FormatConfig { format: format.to_string(), pretty };
    let result = serialize_config(&config);
    assert!(result.is_ok());
}

// --- 非同期フィクスチャ ---
#[rstest]
#[tokio::test]
async fn test_async_with_fixture(
    #[future] async_database: MockDatabase,
) {
    let db = async_database.await;
    assert!(db.is_connected());
}
```

---

## 3. プロパティテスト

### テスト手法の比較

```
┌─────────── テスト手法の比較 ───────────┐
│                                         │
│  Example-based Test (従来型):           │
│    入力: 具体的な値                      │
│    assert_eq!(sort(vec![3,1,2]),         │
│               vec![1,2,3]);             │
│    → 特定のケースのみ検証               │
│                                         │
│  Property-based Test:                   │
│    入力: ランダム生成 (数百〜数千パターン)│
│    proptest! {                          │
│      fn test(v: Vec<i32>) {            │
│        let sorted = sort(v);            │
│        assert!(is_sorted(&sorted));     │
│      }                                  │
│    }                                    │
│    → 性質 (invariant) を検証             │
│    → 反例を自動的に最小化 (shrinking)    │
└─────────────────────────────────────────┘
```

### コード例4: proptest の使用

```rust
use proptest::prelude::*;

/// テスト対象: カスタムソート
fn insertion_sort(mut arr: Vec<i32>) -> Vec<i32> {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
    arr
}

proptest! {
    // ソート後のベクタは昇順である
    #[test]
    fn test_sort_is_ordered(mut v in prop::collection::vec(any::<i32>(), 0..100)) {
        let sorted = insertion_sort(v.clone());
        for window in sorted.windows(2) {
            prop_assert!(window[0] <= window[1],
                "ソートされていない: {} > {}", window[0], window[1]);
        }
    }

    // ソート後の長さは変わらない
    #[test]
    fn test_sort_preserves_length(v in prop::collection::vec(any::<i32>(), 0..100)) {
        let sorted = insertion_sort(v.clone());
        prop_assert_eq!(v.len(), sorted.len());
    }

    // ソート後は元の要素を全て含む
    #[test]
    fn test_sort_preserves_elements(v in prop::collection::vec(any::<i32>(), 0..100)) {
        let mut original = v.clone();
        let mut sorted = insertion_sort(v);
        original.sort();
        sorted.sort();
        prop_assert_eq!(original, sorted);
    }

    // カスタム戦略: メールアドレスのバリデーション
    #[test]
    fn test_email_validation(
        local in "[a-z][a-z0-9]{0,15}",
        domain in "[a-z]{2,10}",
        tld in "(com|org|net|io)"
    ) {
        let email = format!("{}@{}.{}", local, domain, tld);
        prop_assert!(is_valid_email(&email),
            "有効なメールアドレスが拒否された: {}", email);
    }
}

fn is_valid_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}
```

### 3.1 カスタム戦略 (Strategy) の実装

```rust
use proptest::prelude::*;
use proptest::strategy::Strategy;

// --- カスタムデータ型の生成戦略 ---
#[derive(Debug, Clone, PartialEq)]
struct Money {
    amount: i64,     // セント単位
    currency: String,
}

impl Money {
    fn new(amount: i64, currency: &str) -> Self {
        Money { amount, currency: currency.to_string() }
    }

    fn add(&self, other: &Money) -> Result<Money, String> {
        if self.currency != other.currency {
            return Err("通貨が異なります".to_string());
        }
        Ok(Money::new(self.amount + other.amount, &self.currency))
    }
}

// Money のための Arbitrary 実装
fn money_strategy() -> impl Strategy<Value = Money> {
    (
        -1_000_000i64..1_000_000i64,
        prop::sample::select(vec!["JPY", "USD", "EUR"]),
    ).prop_map(|(amount, currency)| Money::new(amount, currency))
}

// 同一通貨の Money ペア戦略
fn same_currency_pair() -> impl Strategy<Value = (Money, Money)> {
    prop::sample::select(vec!["JPY", "USD", "EUR"]).prop_flat_map(|currency| {
        (
            (-1_000_000i64..1_000_000i64).prop_map(move |amt| Money::new(amt, &currency)),
            (-1_000_000i64..1_000_000i64).prop_map(move |amt| Money::new(amt, &currency)),
        )
    })
}

proptest! {
    // 加算の可換性: a + b == b + a
    #[test]
    fn test_money_addition_commutative((a, b) in same_currency_pair()) {
        let ab = a.add(&b).unwrap();
        let ba = b.add(&a).unwrap();
        prop_assert_eq!(ab.amount, ba.amount);
    }

    // ゼロの加算: a + 0 == a
    #[test]
    fn test_money_addition_identity(a in money_strategy()) {
        let zero = Money::new(0, &a.currency);
        let result = a.add(&zero).unwrap();
        prop_assert_eq!(result.amount, a.amount);
    }

    // 異なる通貨の加算はエラー
    #[test]
    fn test_different_currency_fails(
        amount1 in -1_000_000i64..1_000_000i64,
        amount2 in -1_000_000i64..1_000_000i64,
    ) {
        let jpy = Money::new(amount1, "JPY");
        let usd = Money::new(amount2, "USD");
        prop_assert!(jpy.add(&usd).is_err());
    }
}

// --- JSON のラウンドトリップテスト ---
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct UserProfile {
    name: String,
    age: u8,
    tags: Vec<String>,
}

fn user_profile_strategy() -> impl Strategy<Value = UserProfile> {
    (
        "[a-zA-Z]{1,20}",
        0u8..150u8,
        prop::collection::vec("[a-z]{1,10}", 0..5),
    ).prop_map(|(name, age, tags)| UserProfile { name, age, tags })
}

proptest! {
    // シリアライズ → デシリアライズで元に戻る
    #[test]
    fn test_json_roundtrip(profile in user_profile_strategy()) {
        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: UserProfile = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(profile, deserialized);
    }
}
```

### 3.2 proptest の設定とチューニング

```rust
use proptest::prelude::*;
use proptest::test_runner::Config;

// --- テスト設定のカスタマイズ ---
proptest! {
    #![proptest_config(Config::with_cases(10_000))]

    #[test]
    fn test_with_more_cases(x in any::<i32>()) {
        // デフォルト256回 → 10,000回のテスト
        prop_assert!(x.checked_add(0) == Some(x));
    }
}

// ProptestConfig の詳細設定
fn custom_config() -> ProptestConfig {
    ProptestConfig {
        cases: 1000,            // テストケース数
        max_shrink_iters: 10000, // 最大シュリンク反復回数
        max_shrink_time: 30000,  // シュリンク最大時間（ミリ秒）
        fork: false,            // フォークプロセスで実行
        timeout: 60000,         // テストタイムアウト（ミリ秒）
        ..ProptestConfig::default()
    }
}

// --- 回帰テストファイル ---
// proptest-regressions/ ディレクトリに失敗ケースが保存される
// このディレクトリは Git にコミットすること

// proptest-regressions/my_tests.txt の例:
// # seed = "cc deadbeef12345678..."
// # 失敗した入力を再現するシード
```

---

## 4. ベンチマーク

### コード例5: criterion ベンチマーク

```rust
// benches/sorting_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn fibonacci_recursive(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        n => fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2),
    }
}

fn fibonacci_iterative(n: u64) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let temp = b;
        b = a + b;
        a = temp;
    }
    a
}

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fibonacci");

    for n in [10, 20, 30].iter() {
        group.bench_with_input(
            BenchmarkId::new("recursive", n),
            n,
            |b, &n| b.iter(|| fibonacci_recursive(black_box(n))),
        );
        group.bench_with_input(
            BenchmarkId::new("iterative", n),
            n,
            |b, &n| b.iter(|| fibonacci_iterative(black_box(n))),
        );
    }
    group.finish();
}

fn bench_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting");

    for size in [100, 1_000, 10_000].iter() {
        let mut data: Vec<i32> = (0..*size).rev().collect();

        group.bench_with_input(
            BenchmarkId::new("std_sort", size),
            &data,
            |b, data| b.iter(|| {
                let mut d = data.clone();
                d.sort();
                black_box(d)
            }),
        );

        group.bench_with_input(
            BenchmarkId::new("std_sort_unstable", size),
            &data,
            |b, data| b.iter(|| {
                let mut d = data.clone();
                d.sort_unstable();
                black_box(d)
            }),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fibonacci, bench_sorting);
criterion_main!(benches);

// Cargo.toml:
// [[bench]]
// name = "sorting_bench"
// harness = false
//
// [dev-dependencies]
// criterion = { version = "0.5", features = ["html_reports"] }
```

### ベンチマーク結果の読み方

```
┌──────────── criterion 出力の解釈 ──────────────┐
│                                                  │
│  fibonacci/iterative/20                          │
│                  time:   [12.3 ns 12.5 ns 12.7 ns]
│                          ~~~~~~~ ~~~~~~~ ~~~~~~~
│                          下限95%  中央値  上限95%
│                                                  │
│  change: [-2.1234% -0.5678% +1.0123%]           │
│          ~~~~~~~~  ~~~~~~~~  ~~~~~~~~            │
│          最小変化   推定変化   最大変化             │
│          (95% 信頼区間)                           │
│                                                  │
│  Performance has improved. (p < 0.05)            │
│  → 統計的に有意な性能改善                         │
│                                                  │
│  HTML レポート:                                   │
│  target/criterion/report/index.html              │
│  → グラフで時系列変化を可視化                     │
└──────────────────────────────────────────────────┘
```

### 4.1 高度なベンチマーク設定

```rust
// benches/advanced_bench.rs
use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, BenchmarkId, BatchSize,
    measurement::WallTime,
    PlotConfiguration, AxisScale,
};
use std::time::Duration;

fn bench_with_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("with_setup");

    // ベンチマークグループの設定
    group.sample_size(100);              // サンプル数
    group.measurement_time(Duration::from_secs(10)); // 計測時間
    group.warm_up_time(Duration::from_secs(3));      // ウォームアップ
    group.noise_threshold(0.05);         // ノイズ閾値（5%）
    group.confidence_level(0.95);        // 信頼区間
    group.significance_level(0.05);      // 有意水準

    // プロット設定
    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);
    group.plot_config(plot_config);

    // --- セットアップ付きベンチマーク ---
    for size in [100, 1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("hashmap_insert", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || {
                        // セットアップ（計測時間に含まない）
                        let data: Vec<(String, i32)> = (0..size)
                            .map(|i| (format!("key_{}", i), i as i32))
                            .collect();
                        data
                    },
                    |data| {
                        // 計測対象のコード
                        let mut map = std::collections::HashMap::new();
                        for (k, v) in data {
                            map.insert(k, v);
                        }
                        black_box(map)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// --- スループットベンチマーク ---
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    for size in [1024, 4096, 16384, 65536].iter() {
        let data = vec![0u8; *size];

        group.throughput(criterion::Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("compress", size),
            &data,
            |b, data| {
                b.iter(|| {
                    // 圧縮ベンチマーク
                    let mut encoder = flate2::write::GzEncoder::new(
                        Vec::new(),
                        flate2::Compression::default(),
                    );
                    std::io::Write::write_all(&mut encoder, data).unwrap();
                    black_box(encoder.finish().unwrap())
                });
            },
        );
    }

    group.finish();
}

// --- 非同期ベンチマーク ---
fn bench_async(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("async_task", |b| {
        b.to_async(&rt).iter(|| async {
            let result = async_computation().await;
            black_box(result)
        });
    });
}

async fn async_computation() -> Vec<i32> {
    let mut handles = Vec::new();
    for i in 0..10 {
        handles.push(tokio::spawn(async move { i * i }));
    }
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.unwrap());
    }
    results
}

criterion_group!(benches, bench_with_setup, bench_throughput, bench_async);
criterion_main!(benches);
```

### 4.2 divan ベンチマーク（代替ツール）

```rust
// benches/divan_bench.rs
// Cargo.toml:
// [[bench]]
// name = "divan_bench"
// harness = false
//
// [dev-dependencies]
// divan = "0.1"

fn main() {
    divan::main();
}

#[divan::bench]
fn simple_bench() -> Vec<i32> {
    (0..1000).collect()
}

#[divan::bench(args = [100, 1000, 10000])]
fn bench_with_args(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

#[divan::bench(types = [Vec<i32>, Vec<u64>, Vec<f64>])]
fn bench_generic<T: Default + Clone>() -> Vec<T> {
    vec![T::default(); 1000]
}

// divan の利点:
// - #[divan::bench] 属性だけで簡単にベンチマーク定義
// - 型パラメータのベンチマーク
// - 引数のベンチマーク
// - criterion より設定がシンプル
```

---

## 5. テスト戦略

### コード例6: モック/スタブ

```rust
/// テスト可能な設計: trait でインターフェースを定義
trait EmailSender {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), String>;
}

struct SmtpSender;
impl EmailSender for SmtpSender {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), String> {
        // 実際のSMTP送信
        Ok(())
    }
}

/// テスト用モック
#[cfg(test)]
struct MockEmailSender {
    sent: std::cell::RefCell<Vec<(String, String, String)>>,
    should_fail: bool,
}

#[cfg(test)]
impl MockEmailSender {
    fn new() -> Self {
        MockEmailSender {
            sent: std::cell::RefCell::new(Vec::new()),
            should_fail: false,
        }
    }

    fn with_failure() -> Self {
        MockEmailSender {
            sent: std::cell::RefCell::new(Vec::new()),
            should_fail: true,
        }
    }

    fn sent_count(&self) -> usize {
        self.sent.borrow().len()
    }
}

#[cfg(test)]
impl EmailSender for MockEmailSender {
    fn send(&self, to: &str, subject: &str, body: &str) -> Result<(), String> {
        if self.should_fail {
            return Err("送信失敗".into());
        }
        self.sent.borrow_mut().push((to.into(), subject.into(), body.into()));
        Ok(())
    }
}

// ビジネスロジック
fn notify_user(sender: &dyn EmailSender, user_email: &str) -> Result<(), String> {
    sender.send(user_email, "通知", "処理が完了しました")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notify_sends_email() {
        let mock = MockEmailSender::new();
        notify_user(&mock, "user@example.com").unwrap();
        assert_eq!(mock.sent_count(), 1);
    }

    #[test]
    fn test_notify_handles_failure() {
        let mock = MockEmailSender::with_failure();
        let result = notify_user(&mock, "user@example.com");
        assert!(result.is_err());
    }
}
```

### 5.1 mockall を使った自動モック生成

```rust
// Cargo.toml:
// [dev-dependencies]
// mockall = "0.12"

use mockall::{automock, predicate::*};

#[automock]
trait UserRepository {
    fn find_by_id(&self, id: u64) -> Option<User>;
    fn find_by_email(&self, email: &str) -> Option<User>;
    fn save(&self, user: &User) -> Result<(), RepositoryError>;
    fn delete(&self, id: u64) -> Result<(), RepositoryError>;
}

#[automock]
trait NotificationService {
    fn send_email(&self, to: &str, subject: &str, body: &str) -> Result<(), String>;
    fn send_sms(&self, to: &str, message: &str) -> Result<(), String>;
}

struct UserService<R: UserRepository, N: NotificationService> {
    repo: R,
    notification: N,
}

impl<R: UserRepository, N: NotificationService> UserService<R, N> {
    fn register(&self, email: &str, name: &str) -> Result<User, String> {
        // 既存ユーザーチェック
        if self.repo.find_by_email(email).is_some() {
            return Err("メールアドレスが既に登録されています".to_string());
        }

        let user = User {
            id: 0,
            email: email.to_string(),
            name: name.to_string(),
        };
        self.repo.save(&user).map_err(|e| e.to_string())?;
        self.notification.send_email(email, "登録完了", "登録が完了しました")?;
        Ok(user)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::predicate::*;

    #[test]
    fn test_register_new_user() {
        let mut mock_repo = MockUserRepository::new();
        let mut mock_notif = MockNotificationService::new();

        // find_by_email は None を返す（既存ユーザーなし）
        mock_repo
            .expect_find_by_email()
            .with(eq("new@example.com"))
            .times(1)
            .returning(|_| None);

        // save は成功
        mock_repo
            .expect_save()
            .times(1)
            .returning(|_| Ok(()));

        // メール送信は成功
        mock_notif
            .expect_send_email()
            .with(eq("new@example.com"), eq("登録完了"), always())
            .times(1)
            .returning(|_, _, _| Ok(()));

        let service = UserService {
            repo: mock_repo,
            notification: mock_notif,
        };

        let result = service.register("new@example.com", "新規ユーザー");
        assert!(result.is_ok());
    }

    #[test]
    fn test_register_duplicate_email() {
        let mut mock_repo = MockUserRepository::new();
        let mock_notif = MockNotificationService::new();

        // 既存ユーザーが見つかる
        mock_repo
            .expect_find_by_email()
            .returning(|_| Some(User {
                id: 1,
                email: "existing@example.com".to_string(),
                name: "既存ユーザー".to_string(),
            }));

        // save は呼ばれない
        mock_repo.expect_save().never();

        let service = UserService {
            repo: mock_repo,
            notification: mock_notif,
        };

        let result = service.register("existing@example.com", "テスト");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("既に登録"));
    }
}
```

### 5.2 wiremock を使った HTTP モック

```rust
// Cargo.toml:
// [dev-dependencies]
// wiremock = "0.6"

use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path, query_param, body_json};

#[tokio::test]
async fn test_external_api_call() {
    // モックサーバーの起動
    let mock_server = MockServer::start().await;

    // モックレスポンスの設定
    Mock::given(method("GET"))
        .and(path("/api/users/42"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "id": 42,
                    "name": "テストユーザー",
                    "email": "test@example.com"
                }))
        )
        .expect(1)  // 1回だけ呼ばれることを期待
        .mount(&mock_server)
        .await;

    // テスト対象のクライアント
    let client = ApiClient::new(&mock_server.uri());
    let user = client.get_user(42).await.unwrap();

    assert_eq!(user.name, "テストユーザー");
}

#[tokio::test]
async fn test_api_error_handling() {
    let mock_server = MockServer::start().await;

    // 500 エラーのモック
    Mock::given(method("GET"))
        .and(path("/api/users/999"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock_server)
        .await;

    let client = ApiClient::new(&mock_server.uri());
    let result = client.get_user(999).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_api_retry_on_timeout() {
    let mock_server = MockServer::start().await;

    // 最初の2回はタイムアウト、3回目で成功
    Mock::given(method("GET"))
        .and(path("/api/data"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"status": "ok"}))
                .set_delay(std::time::Duration::from_millis(100))
        )
        .mount(&mock_server)
        .await;

    let client = ApiClient::new(&mock_server.uri())
        .with_timeout(std::time::Duration::from_secs(5))
        .with_retries(3);

    let result = client.get_data().await;
    assert!(result.is_ok());
}
```

### 5.3 スナップショットテスト

```rust
// Cargo.toml:
// [dev-dependencies]
// insta = { version = "1", features = ["json", "yaml"] }

use insta::{assert_snapshot, assert_json_snapshot, assert_yaml_snapshot};

#[test]
fn test_html_rendering() {
    let html = render_template("welcome", &context);
    // スナップショットファイルに保存された期待値と比較
    assert_snapshot!(html);
}

#[test]
fn test_api_response_format() {
    let response = create_user_response(&user);
    // JSON 形式でスナップショット
    assert_json_snapshot!(response, {
        ".id" => "[id]",           // 動的な値をマスク
        ".created_at" => "[date]", // 日時をマスク
    });
}

#[test]
fn test_config_serialization() {
    let config = Config::default();
    assert_yaml_snapshot!(config);
}

// スナップショットの更新:
// cargo insta test         # テスト実行
// cargo insta review       # 差分レビュー & 承認
// cargo insta accept       # 全承認
```

### 5.4 ファジングテスト

```rust
// Cargo.toml:
// [dependencies]
// # fuzz ターゲットは別途 cargo-fuzz で管理
//
// fuzz/Cargo.toml:
// [dependencies]
// libfuzzer-sys = "0.4"
// arbitrary = { version = "1", features = ["derive"] }

// fuzz/fuzz_targets/parse_input.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 任意のバイト列をパーサーに渡す
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = my_lib::parse(input);
    }
});

// Arbitrary トレイトを使った構造化ファジング
use arbitrary::Arbitrary;

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    name: String,
    age: u8,
    values: Vec<i32>,
}

fuzz_target!(|input: FuzzInput| {
    let _ = my_lib::process_user(&input.name, input.age, &input.values);
});

// 実行方法:
// cargo install cargo-fuzz
// cargo fuzz init           # fuzz/ ディレクトリ作成
// cargo fuzz add parse_input  # ファズターゲット追加
// cargo +nightly fuzz run parse_input  # ファジング実行（nightly 必須）
// cargo +nightly fuzz run parse_input -- -max_total_time=60  # 60秒で終了
```

---

## 6. テストカバレッジとCI統合

### 6.1 カバレッジ計測

```bash
# --- cargo-llvm-cov ---
cargo install cargo-llvm-cov

# テストカバレッジの計測
cargo llvm-cov                     # コンソール出力
cargo llvm-cov --html              # HTML レポート
cargo llvm-cov --lcov --output-path lcov.info  # LCOV 形式

# 特定のテストのみ
cargo llvm-cov --lib               # 単体テストのみ
cargo llvm-cov --test integration  # 統合テストのみ
cargo llvm-cov --all-features      # 全 feature

# ワークスペース全体
cargo llvm-cov --workspace

# カバレッジ閾値の確認
cargo llvm-cov --fail-under-lines 80  # 行カバレッジ 80% 未満で失敗

# --- tarpaulin (Linux のみ) ---
cargo install cargo-tarpaulin
cargo tarpaulin --out html --all-features
```

### 6.2 ミューテーションテスト

```bash
# cargo-mutants — ミューテーションテスト
cargo install cargo-mutants

# ミューテーションテスト実行
cargo mutants                      # 全ミュータント実行
cargo mutants -- -p my-crate       # 特定クレートのみ
cargo mutants --list               # ミュータントの一覧
cargo mutants --timeout-multiplier 3  # タイムアウト倍率

# 結果の読み方:
# caught: テストがミュータントを検出（良い）
# missed: テストがミュータントを見逃した（テスト追加が必要）
# unviable: コンパイルできないミュータント（無視してOK）
# timeout: タイムアウト（テストが遅すぎる可能性）
```

### 6.3 テストの高速化

```bash
# --- nextest: 高速テストランナー ---
cargo install cargo-nextest

# テスト実行（並列度が高い）
cargo nextest run                  # デフォルトの並列実行
cargo nextest run --workspace      # ワークスペース全体
cargo nextest run -p my-crate      # 特定クレート
cargo nextest run --retries 2      # 失敗時リトライ
cargo nextest run --test-threads 8 # スレッド数指定

# テストのフィルタリング
cargo nextest run -E 'test(test_parse)'     # 名前でフィルタ
cargo nextest run -E 'package(my-crate)'    # パッケージでフィルタ
cargo nextest run -E 'kind(test)'           # テスト種類でフィルタ

# テスト結果の保存
cargo nextest run --message-format json > results.json
```

```toml
# .config/nextest.toml — nextest の設定
[store]
dir = "target/nextest"

[profile.default]
retries = 0
slow-timeout = { period = "60s", terminate-after = 2 }
fail-fast = true

[profile.ci]
retries = 2
fail-fast = false
slow-timeout = { period = "120s", terminate-after = 3 }

# テストグループ（リソース制限）
[test-groups.serial-db]
max-threads = 1

[[profile.default.overrides]]
filter = "test(/db_/)"
test-group = "serial-db"
```

---

## 7. 比較表

### テストフレームワーク比較

| フレームワーク | 種類 | 特徴 | 用途 |
|---|---|---|---|
| 標準 #[test] | 単体/統合 | 組み込み、設定不要 | 基本テスト |
| proptest | プロパティ | 自動生成+shrinking | 仕様の検証 |
| quickcheck | プロパティ | Haskell 由来 | 軽量プロパティテスト |
| criterion | ベンチマーク | 統計的分析、HTML | パフォーマンス回帰 |
| divan | ベンチマーク | #[divan::bench] | シンプルなベンチ |
| rstest | パラメータ化 | #[rstest] + fixtures | テーブル駆動テスト |
| mockall | モック | 自動モック生成 | 依存の差し替え |
| wiremock | HTTPモック | 非同期対応 | 外部APIテスト |
| insta | スナップショット | JSON/YAML対応 | 出力の回帰テスト |
| cargo-fuzz | ファジング | libFuzzer ベース | セキュリティテスト |
| nextest | テストランナー | 高速並列実行 | CI/CD |

### assert マクロ比較

| マクロ | 用途 | 失敗時メッセージ |
|---|---|---|
| `assert!(expr)` | 真偽値 | "assertion failed" |
| `assert_eq!(a, b)` | 等値比較 | left と right の値を表示 |
| `assert_ne!(a, b)` | 非等値 | left と right が等しい |
| `debug_assert!()` | デバッグビルドのみ | リリースでは除去 |
| `prop_assert!()` | proptest 内 | 反例の最小化を実行 |
| `assert_snapshot!()` | insta | スナップショットとの差分 |

### テスト実行コマンド比較

| コマンド | 対象 | 説明 |
|---|---|---|
| `cargo test` | 全テスト | 単体+統合+ドキュメントテスト |
| `cargo test --lib` | 単体テスト | src/ 内の #[test] のみ |
| `cargo test --test name` | 特定統合テスト | tests/name.rs |
| `cargo test --doc` | ドキュメントテスト | /// 内のコードブロック |
| `cargo test --examples` | サンプルコード | examples/ のコンパイルテスト |
| `cargo test name` | 名前フィルタ | テスト名に "name" を含むもの |
| `cargo test -- --nocapture` | 出力表示 | println! の出力を表示 |
| `cargo test -- --test-threads=1` | シリアル実行 | 並列実行を無効化 |
| `cargo test -- --ignored` | 無視テスト | #[ignore] のテストを実行 |
| `cargo test -- --include-ignored` | 全テスト | 無視テストも含め全実行 |

---

## 8. アンチパターン

### アンチパターン1: テスト間の状態共有

```rust
// NG: 静的変数でテスト間に状態が漏れる
static mut COUNTER: u32 = 0;

#[test]
fn test_a() {
    unsafe { COUNTER += 1; }
    // テスト実行順序に依存!
}

#[test]
fn test_b() {
    unsafe { assert_eq!(COUNTER, 0); } // test_a が先に実行されると失敗!
}

// OK: 各テストが独立したセットアップ
#[test]
fn test_a_isolated() {
    let mut counter = 0u32;
    counter += 1;
    assert_eq!(counter, 1);
}

#[test]
fn test_b_isolated() {
    let counter = 0u32;
    assert_eq!(counter, 0);
}
```

### アンチパターン2: ベンチマークでの最適化除去

```rust
// NG: コンパイラが結果を使わないコードを除去する
fn bench_bad(c: &mut Criterion) {
    c.bench_function("sum", |b| {
        b.iter(|| {
            let sum: u64 = (0..1000).sum(); // 最適化で除去される可能性
        });
    });
}

// OK: black_box で最適化を防ぐ
use criterion::black_box;
fn bench_good(c: &mut Criterion) {
    c.bench_function("sum", |b| {
        b.iter(|| {
            let sum: u64 = (0..1000).sum();
            black_box(sum) // コンパイラに「この値は使われる」と伝える
        });
    });
}
```

### アンチパターン3: テストでの実装の過剰検証

```rust
// NG: 実装の詳細を検証している
#[test]
fn test_sort_uses_quicksort() {
    // 内部でクイックソートを使っているかチェック ← 実装詳細
    let comparisons = count_comparisons(sort, &data);
    assert!(comparisons < n * log2(n) * 1.5);
}

// OK: 振る舞い（インターフェース）を検証
#[test]
fn test_sort_produces_ordered_output() {
    let input = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let result = sort(&input);
    assert_eq!(result, vec![1, 1, 2, 3, 4, 5, 6, 9]);
}
```

### アンチパターン4: テストの重複

```rust
// NG: ほぼ同じテストのコピー&ペースト
#[test]
fn test_parse_int_positive() {
    assert_eq!(parse("42"), Ok(42));
}
#[test]
fn test_parse_int_positive_2() {
    assert_eq!(parse("100"), Ok(100));
}
#[test]
fn test_parse_int_positive_3() {
    assert_eq!(parse("999"), Ok(999));
}

// OK: パラメータ化テスト
#[rstest]
#[case("42", 42)]
#[case("100", 100)]
#[case("999", 999)]
#[case("0", 0)]
#[case("-1", -1)]
fn test_parse_int(#[case] input: &str, #[case] expected: i32) {
    assert_eq!(parse(input), Ok(expected));
}
```

### アンチパターン5: 遅すぎるテスト

```rust
// NG: テストが遅い（I/O、ネットワーク、大量データ）
#[test]
fn test_slow_network_call() {
    let response = reqwest::blocking::get("https://api.example.com/data").unwrap();
    assert_eq!(response.status(), 200);
}

// OK: モックを使って高速化
#[tokio::test]
async fn test_fast_with_mock() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/data"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock_server)
        .await;

    let response = client.get(&format!("{}/data", mock_server.uri()))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
}
```

---

## 9. テスタブルアーキテクチャ

### 9.1 依存性注入パターン

```rust
// --- trait による依存性注入 ---
pub trait Clock {
    fn now(&self) -> chrono::DateTime<chrono::Utc>;
}

pub struct SystemClock;
impl Clock for SystemClock {
    fn now(&self) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc::now()
    }
}

#[cfg(test)]
pub struct FixedClock(pub chrono::DateTime<chrono::Utc>);

#[cfg(test)]
impl Clock for FixedClock {
    fn now(&self) -> chrono::DateTime<chrono::Utc> {
        self.0
    }
}

// 時刻に依存するビジネスロジック
pub struct TokenService<C: Clock> {
    clock: C,
    expiry_duration: chrono::Duration,
}

impl<C: Clock> TokenService<C> {
    pub fn new(clock: C, expiry_hours: i64) -> Self {
        TokenService {
            clock,
            expiry_duration: chrono::Duration::hours(expiry_hours),
        }
    }

    pub fn create_token(&self, user_id: u64) -> Token {
        let now = self.clock.now();
        Token {
            user_id,
            created_at: now,
            expires_at: now + self.expiry_duration,
        }
    }

    pub fn is_expired(&self, token: &Token) -> bool {
        self.clock.now() > token.expires_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_token_expiry() {
        // 固定時刻でテスト
        let fixed_time = chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let clock = FixedClock(fixed_time);
        let service = TokenService::new(clock, 24);

        let token = service.create_token(42);
        assert_eq!(token.user_id, 42);
        assert_eq!(
            token.expires_at,
            chrono::Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap()
        );
    }

    #[test]
    fn test_expired_token() {
        // 1日後の時刻でチェック
        let future_time = chrono::Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap();
        let clock = FixedClock(future_time);
        let service = TokenService::new(clock, 24);

        let token = Token {
            user_id: 42,
            created_at: chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            expires_at: chrono::Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
        };

        assert!(service.is_expired(&token));
    }
}
```

### 9.2 テストダブルの種類

```
┌────────────── テストダブルの種類 ──────────────┐
│                                                  │
│  Dummy:  引数を満たすだけ（メソッドは呼ばれない）│
│  Stub:   固定値を返す                            │
│  Spy:    呼び出しを記録する                      │
│  Mock:   期待される呼び出しを検証する            │
│  Fake:   動作する簡易実装（インメモリDB等）      │
│                                                  │
│  使い分け:                                       │
│  - 戻り値の制御 → Stub                          │
│  - 呼び出し回数の検証 → Mock (mockall)          │
│  - 副作用の記録 → Spy                           │
│  - 統合テスト → Fake (インメモリ実装)            │
└──────────────────────────────────────────────────┘
```

```rust
// Fake パターン: インメモリリポジトリ
pub struct InMemoryUserRepository {
    users: std::sync::Mutex<Vec<User>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl InMemoryUserRepository {
    pub fn new() -> Self {
        InMemoryUserRepository {
            users: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }
}

impl UserRepository for InMemoryUserRepository {
    fn find_by_id(&self, id: u64) -> Option<User> {
        self.users.lock().unwrap()
            .iter()
            .find(|u| u.id == id)
            .cloned()
    }

    fn find_by_email(&self, email: &str) -> Option<User> {
        self.users.lock().unwrap()
            .iter()
            .find(|u| u.email == email)
            .cloned()
    }

    fn save(&self, user: &User) -> Result<(), RepositoryError> {
        let mut users = self.users.lock().unwrap();
        let mut new_user = user.clone();
        if new_user.id == 0 {
            new_user.id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        users.push(new_user);
        Ok(())
    }

    fn delete(&self, id: u64) -> Result<(), RepositoryError> {
        let mut users = self.users.lock().unwrap();
        users.retain(|u| u.id != id);
        Ok(())
    }
}
```

---

## FAQ

### Q1: テストの実行順序は?

**A:** Rust のテストはデフォルトで並列に実行されます。`cargo test -- --test-threads=1` でシリアル実行に変更できます。テスト間に依存関係がある設計は避けるべきです。

### Q2: テストカバレッジの計測方法は?

**A:** `cargo-llvm-cov` または `tarpaulin` を使います。

```bash
# cargo-llvm-cov
cargo install cargo-llvm-cov
cargo llvm-cov --html         # HTML レポート生成
cargo llvm-cov --open         # ブラウザで開く

# tarpaulin (Linux のみ)
cargo install cargo-tarpaulin
cargo tarpaulin --out html
```

### Q3: proptest で失敗した場合のデバッグ方法は?

**A:** proptest は失敗した入力を自動的に最小化 (shrink) して再テストします。失敗したシードは `proptest-regressions/` ディレクトリに保存され、次回以降のテストで自動的に再実行されます。

```
# proptest-regressions/test_name.txt
# 失敗した入力のシードが保存される
cc deadbeef12345678
```

### Q4: テストのフィルタリング方法は?

**A:** テスト名やモジュール名でフィルタリングできます。

```bash
# 名前に "parse" を含むテストのみ
cargo test parse

# 特定モジュールのテスト
cargo test tests::given_valid_input

# 正規表現フィルタ（nextest）
cargo nextest run -E 'test(/test_parse_.*/)'

# 無視テストの実行
cargo test -- --ignored

# 全テスト（無視含む）
cargo test -- --include-ignored
```

### Q5: テストのデバッグ方法は?

**A:** 複数の方法があります。

```bash
# println! の出力を表示
cargo test -- --nocapture

# 特定のテストのみ実行 + 出力表示
cargo test test_my_function -- --nocapture

# RUST_LOG で詳細ログ
RUST_LOG=debug cargo test -- --nocapture

# デバッガ接続（lldb/gdb）
# 1. テストバイナリのパスを確認
cargo test --no-run --message-format=json | jq '.executable'
# 2. デバッガで実行
lldb target/debug/deps/my_crate-abc123 -- test_my_function
```

### Q6: 統合テストのコンパイルが遅い場合は?

**A:** 統合テストは各ファイルが独立したバイナリクレートとしてコンパイルされます。ファイル数を減らすことで改善できます。

```rust
// NG: 多数の統合テストファイル（それぞれ独立コンパイル）
// tests/test_auth.rs
// tests/test_users.rs
// tests/test_orders.rs
// tests/test_payments.rs
// → 4 つのバイナリがコンパイルされる

// OK: 単一のエントリポイントからモジュールを読み込む
// tests/integration.rs
mod auth;
mod users;
mod orders;
mod payments;
// → 1 つのバイナリで全テスト実行
```

### Q7: feature ごとにテストを分ける方法は?

**A:** `#[cfg(feature = "...")]` をテストモジュールに適用します。

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // 常に実行されるテスト
    #[test]
    fn test_basic() {
        assert!(true);
    }

    // json feature 有効時のみ
    #[cfg(feature = "json")]
    mod json_tests {
        use super::*;

        #[test]
        fn test_json_serialize() {
            let value = MyStruct { name: "test".into() };
            let json = serde_json::to_string(&value).unwrap();
            assert!(json.contains("test"));
        }
    }

    // async feature 有効時のみ
    #[cfg(feature = "async")]
    mod async_tests {
        use super::*;

        #[tokio::test]
        async fn test_async_operation() {
            let result = async_function().await;
            assert!(result.is_ok());
        }
    }
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| #[test] | 標準テスト。#[cfg(test)] で本番ビルドから除外 |
| 統合テスト | tests/ ディレクトリ。公開 API のみテスト |
| doc テスト | /// コメント内コード。ドキュメントとテストを同時管理 |
| proptest | ランダム入力で性質を検証。反例を自動最小化 |
| criterion | 統計的ベンチマーク。回帰検知に有効 |
| divan | シンプルなベンチマーク。属性マクロで定義 |
| rstest | パラメータ化テスト。フィクスチャサポート |
| mockall | trait ベースの自動モック生成 |
| wiremock | HTTP モック。非同期テスト対応 |
| insta | スナップショットテスト。JSON/YAML 対応 |
| モック | trait ベースで依存を差し替え |
| テスト隔離 | 各テストは独立。状態共有は禁止 |
| black_box | ベンチマークの最適化除去を防止 |
| nextest | 高速テストランナー。CI/CD に最適 |
| cargo-llvm-cov | コードカバレッジ計測 |
| cargo-mutants | ミューテーションテスト |
| cargo-fuzz | ファジングテスト |

## 次に読むべきガイド

- [Serde](./02-serde.md) — テストフィクスチャの読み込みに活用
- [ベストプラクティス](./04-best-practices.md) — テスタブルな設計パターン
- [Cargo/ワークスペース](./00-cargo-workspace.md) — テスト構成とプロファイル

## 参考文献

1. **The Rust Book — Testing**: https://doc.rust-lang.org/book/ch11-00-testing.html
2. **proptest book**: https://proptest-rs.github.io/proptest/intro.html
3. **criterion.rs User Guide**: https://bheisler.github.io/criterion.rs/book/
4. **mockall documentation**: https://docs.rs/mockall/
5. **rstest documentation**: https://docs.rs/rstest/
6. **wiremock documentation**: https://docs.rs/wiremock/
7. **insta documentation**: https://insta.rs/
8. **cargo-nextest documentation**: https://nexte.st/
9. **cargo-llvm-cov documentation**: https://github.com/taiki-e/cargo-llvm-cov
10. **cargo-fuzz documentation**: https://rust-fuzz.github.io/book/cargo-fuzz.html
