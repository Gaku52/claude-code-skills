# テスト — proptest、criterion

> Rust のテストエコシステムを単体テスト・統合テスト・プロパティテスト・ベンチマークまで網羅的に習得する

## この章で学ぶこと

1. **標準テスト** — #[test], #[cfg(test)], assert マクロ、テスト構成
2. **プロパティテスト** — proptest / quickcheck による性質ベーステスト
3. **ベンチマーク** — criterion による統計的パフォーマンス測定

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

---

## 6. 比較表

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

### assert マクロ比較

| マクロ | 用途 | 失敗時メッセージ |
|---|---|---|
| `assert!(expr)` | 真偽値 | "assertion failed" |
| `assert_eq!(a, b)` | 等値比較 | left と right の値を表示 |
| `assert_ne!(a, b)` | 非等値 | left と right が等しい |
| `debug_assert!()` | デバッグビルドのみ | リリースでは除去 |
| `prop_assert!()` | proptest 内 | 反例の最小化を実行 |

---

## 7. アンチパターン

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

---

## まとめ

| 項目 | 要点 |
|---|---|
| #[test] | 標準テスト。#[cfg(test)] で本番ビルドから除外 |
| 統合テスト | tests/ ディレクトリ。公開 API のみテスト |
| doc テスト | /// コメント内コード。ドキュメントとテストを同時管理 |
| proptest | ランダム入力で性質を検証。反例を自動最小化 |
| criterion | 統計的ベンチマーク。回帰検知に有効 |
| モック | trait ベースで依存を差し替え |
| テスト隔離 | 各テストは独立。状態共有は禁止 |
| black_box | ベンチマークの最適化除去を防止 |

## 次に読むべきガイド

- [Serde](./02-serde.md) — テストフィクスチャの読み込みに活用
- [ベストプラクティス](./04-best-practices.md) — テスタブルな設計パターン
- [Cargo/ワークスペース](./00-cargo-workspace.md) — テスト構成とプロファイル

## 参考文献

1. **The Rust Book — Testing**: https://doc.rust-lang.org/book/ch11-00-testing.html
2. **proptest book**: https://proptest-rs.github.io/proptest/intro.html
3. **criterion.rs User Guide**: https://bheisler.github.io/criterion.rs/book/
