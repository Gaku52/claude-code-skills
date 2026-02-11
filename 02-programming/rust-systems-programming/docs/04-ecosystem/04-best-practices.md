# Rust ベストプラクティス

> clippy・API 設計・エラーハンドリング・テスト戦略など、Rust で品質の高いコードを書くための実践的な指針を体系的に学ぶ

## この章で学ぶこと

1. **コード品質ツール** — clippy、rustfmt、cargo-audit による自動品質管理
2. **API 設計原則** — 型駆動設計、ビルダーパターン、ゼロコスト抽象化
3. **プロジェクト構成** — ワークスペース管理、ドキュメント、CI/CD 統合

---

## 1. clippy による静的解析

```
Rust 品質管理ツールチェーン
============================

ソースコード
    |
    v
[rustfmt]     --> コードフォーマット統一
    |
    v
[clippy]      --> 静的解析 (500+ lint ルール)
    |
    v
[cargo test]  --> ユニットテスト + 統合テスト
    |
    v
[cargo audit] --> 依存関係の脆弱性チェック
    |
    v
[cargo deny]  --> ライセンス・重複依存チェック
    |
    v
本番ビルド
```

### コード例 1: clippy 設定と主要 lint

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
```

```rust
// clippy が検出する典型的な改善点

// [NG] 不要なクローン
let s = String::from("hello");
let t = s.clone();  // clippy: redundant_clone
println!("{}", t);

// [OK]
let s = String::from("hello");
println!("{}", s);

// [NG] 非効率な文字列結合
let mut result = String::new();
for item in items {
    result = result + &item.to_string();  // clippy: string_add
}

// [OK] 効率的な結合
let result: String = items.iter().map(|i| i.to_string()).collect();

// [NG] map + unwrap
let values: Vec<i32> = strings.iter().map(|s| s.parse().unwrap()).collect();

// [OK] filter_map
let values: Vec<i32> = strings.iter().filter_map(|s| s.parse().ok()).collect();
```

### コード例 2: rustfmt 設定

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
```

---

## 2. API 設計原則

### 型駆動設計

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
```

### コード例 3: ビルダーパターン

```rust
/// 型安全なビルダーパターン（Typestate パターン）
pub struct RequestBuilder<S: BuilderState> {
    url: String,
    method: String,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
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
            _state: std::marker::PhantomData,
        }
    }

    pub fn url(self, url: impl Into<String>) -> RequestBuilder<HasUrl> {
        RequestBuilder {
            url: url.into(),
            method: self.method,
            headers: self.headers,
            body: self.body,
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

    pub fn build(self) -> Request {
        Request {
            url: self.url,
            method: self.method,
            headers: self.headers,
            body: self.body,
        }
    }
}

// 使用例: url() を呼ばないと build() できない
let req = RequestBuilder::new()
    .url("https://api.example.com")
    .method("POST")
    .header("Content-Type", "application/json")
    .body(b"{\"key\": \"value\"}")
    .build();
```

### コード例 4: エラー設計

```rust
use thiserror::Error;

/// アプリケーションエラー型
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

    #[error("内部エラー")]
    Internal(#[source] anyhow::Error),
}

#[derive(Error, Debug)]
pub enum AuthError {
    #[error("トークンが無効です")]
    InvalidToken,
    #[error("トークンの有効期限切れ")]
    TokenExpired,
    #[error("権限不足: {required} が必要です")]
    InsufficientPermissions { required: String },
}

/// HTTP ステータスコードへの変換
impl AppError {
    pub fn status_code(&self) -> u16 {
        match self {
            AppError::Database(_) => 500,
            AppError::Auth(_) => 401,
            AppError::Validation { .. } => 400,
            AppError::NotFound { .. } => 404,
            AppError::Internal(_) => 500,
        }
    }
}

/// Result 型エイリアス
pub type AppResult<T> = Result<T, AppError>;
```

---

## 3. テスト戦略

### コード例 5: テストの構成

```rust
// src/lib.rs
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

    // ユニットテスト
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

    // プロパティベーステスト（proptest）
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn price_is_always_positive(
            base in 0.01f64..10000.0,
            tax in 0.0f64..0.5,
            discount in proptest::option::of(0.0f64..1.0),
        ) {
            let result = calculate_price(base, tax, discount);
            prop_assert!(result > 0.0);
        }
    }
}

// tests/integration_test.rs（統合テスト）
#[test]
fn test_full_workflow() {
    // 統合テストはここに
}
```

### コード例 6: CI/CD 設定（GitHub Actions）

```yaml
# .github/workflows/rust.yml
name: Rust CI
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - name: Format check
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --all-targets -- -D warnings

      - name: Test
        run: cargo test --all-features

      - name: Security audit
        run: |
          cargo install cargo-audit
          cargo audit

      - name: Doc test
        run: cargo doc --no-deps
```

---

## 4. パフォーマンスのベストプラクティス

### 割り当て回避の比較表

| パターン | 割り当て | 推奨度 | 説明 |
|---|---|---|---|
| `String` を引数に | 毎回 | 低 | 呼び出し側でクローン必要 |
| `&str` を引数に | なし | 高 | 借用で十分な場合 |
| `impl Into<String>` | 必要時のみ | 高 | 柔軟な API |
| `Cow<'_, str>` | 必要時のみ | 中 | 所有/借用を動的切替 |
| `Vec<T>` を返す | 毎回 | 中 | イテレータを返す方が良い場合あり |
| `impl Iterator` を返す | なし | 高 | 遅延評価 |

### API 設計の指針比較表

| 原則 | 良い例 | 悪い例 |
|---|---|---|
| **型で不変条件を表現** | `NonZeroU32` | `u32`（0チェックを忘れる） |
| **所有権を最小限に** | `fn process(data: &[u8])` | `fn process(data: Vec<u8>)` |
| **列挙型で網羅性** | `match` の全パターン | `if/else` の連鎖 |
| **newtype で意味付け** | `struct UserId(u64)` | `u64`（何の ID?） |
| **ビルダーで複雑な構築** | `Config::builder().port(8080).build()` | `Config::new(8080, ...)` 引数多数 |
| **From/Into で変換** | `impl From<String> for Name` | 手動変換関数 |

---

## アンチパターン

### 1. 過剰な unwrap/expect

**問題**: `unwrap()` はパニックを引き起こし、サーバーアプリケーションではプロセスが停止する。

```rust
// [NG]
let config = std::fs::read_to_string("config.toml").unwrap();
let port: u16 = env::var("PORT").unwrap().parse().unwrap();

// [OK] 適切なエラーハンドリング
let config = std::fs::read_to_string("config.toml")
    .context("Failed to read config.toml")?;
let port: u16 = env::var("PORT")
    .unwrap_or_else(|_| "8080".to_string())
    .parse()
    .context("Invalid PORT value")?;
```

### 2. 不要な Clone の多用

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
```

---

## FAQ

### Q1: clippy の警告をすべて修正すべきですか？

**A**: `warn` レベルの lint は基本的に修正すべきです。ただし、`pedantic` レベルには過度に厳格なものもあるため、プロジェクトに合わせて `#[allow]` で個別に許可できます。`deny` に設定した lint は CI で必ずチェックしてください。

### Q2: anyhow と thiserror はどう使い分けますか？

**A**:
- **thiserror**: ライブラリで使用。呼び出し側がエラー型をパターンマッチで処理できる
- **anyhow**: アプリケーションで使用。エラーの種類よりもコンテキスト情報が重要な場合
ライブラリでは `thiserror` で型を定義し、アプリケーションの `main` や CLI では `anyhow::Result` で統一するのが一般的です。

### Q3: Rust のパフォーマンスチューニングの進め方は？

**A**: 以下の順序で進めます:
1. **ベンチマーク** を `criterion` で作成し、現状を計測
2. **プロファイル** を `perf`/`flamegraph` で実行し、ホットスポットを特定
3. **アルゴリズム改善** が最優先（O(n^2) -> O(n log n) など）
4. **割り当て削減** — Clone の除去、`String` -> `&str`、`Vec` -> スライス
5. **並列化** — rayon による data parallelism
計測なしの最適化は避けてください。

---

## まとめ

| 項目 | 要点 |
|---|---|
| clippy | 500+ の lint で一般的なミスを防止。CI で必須 |
| rustfmt | コードフォーマットの統一。チーム開発で必須 |
| API 設計 | 型で不変条件を表現。所有権は最小限に |
| エラー処理 | ライブラリは thiserror、アプリは anyhow |
| テスト | ユニット + 統合 + プロパティベース。カバレッジ監視 |
| パフォーマンス | 計測優先。不要な Clone と割り当てを排除 |

## 次に読むべきガイド

- [FFI](../03-systems/02-ffi-interop.md) — 他言語連携のベストプラクティス
- [非同期プログラミング](../02-advanced/01-async.md) — async Rust の設計パターン

## 参考文献

1. **Rust API Guidelines**: [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) — 公式 API 設計ガイドライン
2. **clippy Lints**: [Clippy Lint List](https://rust-lang.github.io/rust-clippy/master/) — 全 lint の一覧と説明
3. **Rust Design Patterns**: [Rust Design Patterns](https://rust-unofficial.github.io/patterns/) — Rust 特有の設計パターン集
