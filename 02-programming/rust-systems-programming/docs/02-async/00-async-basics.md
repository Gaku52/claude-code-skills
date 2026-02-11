# async/await基礎 — Rustの非同期プログラミングモデル

> Future trait を中心とした Rust 非同期ランタイムの仕組みと async/await 構文の基礎を理解する

## この章で学ぶこと

1. **Future trait の仕組み** — Poll ベースの遅延評価モデルとゼロコスト抽象化
2. **async/await 構文** — 非同期関数の定義・呼び出し・合成パターン
3. **ランタイムの役割** — Executor、Reactor、Waker の協調動作

---

## 1. 同期 vs 非同期の全体像

```
┌──────────────────── 同期 (Sync) ────────────────────┐
│                                                      │
│  Thread 1: [タスクA ██████████████████████████████]  │
│  Thread 2: [タスクB ██████████████████████████████]  │
│  Thread 3: [タスクC ██████████████████████████████]  │
│                                                      │
│  → スレッド数 = 同時タスク数 (10K接続 = 10Kスレッド)   │
└──────────────────────────────────────────────────────┘

┌──────────────────── 非同期 (Async) ──────────────────┐
│                                                      │
│  Thread 1: [A██][B██][A██][C████][B██][A██]         │
│  Thread 2: [C██][A██][B████][C██][B██]              │
│                                                      │
│  → 少数スレッドで多数タスクを処理                      │
│  → I/O待ち中に他のタスクを実行                        │
└──────────────────────────────────────────────────────┘
```

---

## 2. Future trait の核心

### コード例1: Future trait の定義

```rust
use std::pin::Pin;
use std::task::{Context, Poll};

// 標準ライブラリの Future trait (簡略化)
pub trait Future {
    type Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

// Poll 列挙体
pub enum Poll<T> {
    Ready(T),   // 完了。値 T を返す
    Pending,     // 未完了。Waker で再通知される
}
```

### コード例2: 手動 Future 実装

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// 指定時間後に完了する自作 Future
struct Delay {
    when: Instant,
}

impl Delay {
    fn new(duration: Duration) -> Self {
        Delay {
            when: Instant::now() + duration,
        }
    }
}

impl Future for Delay {
    type Output = String;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if Instant::now() >= self.when {
            Poll::Ready("時間経過!".to_string())
        } else {
            // 実際のランタイムではタイマーを登録して Waker を呼ぶ
            // ここでは即座に再ポーリングを要求 (ビジーウェイト)
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[tokio::main]
async fn main() {
    let msg = Delay::new(Duration::from_secs(1)).await;
    println!("{}", msg); // "時間経過!"
}
```

---

## 3. async/await 構文

### コード例3: 基本的な async 関数

```rust
use tokio::time::{sleep, Duration};

/// async fn は Future を返す関数のシンタックスシュガー
async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    // .await で Future の完了を待つ
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}

/// 上記は以下と等価 (脱糖後)
fn fetch_data_desugared(url: &str) -> impl Future<Output = Result<String, reqwest::Error>> + '_ {
    async move {
        let response = reqwest::get(url).await?;
        let body = response.text().await?;
        Ok(body)
    }
}

#[tokio::main]
async fn main() {
    match fetch_data("https://httpbin.org/get").await {
        Ok(body) => println!("取得成功: {}バイト", body.len()),
        Err(e) => eprintln!("エラー: {}", e),
    }
}
```

### 非同期処理の実行フロー

```
┌─────────────────────────────────────────────────┐
│            async fn fetch_data() の内部           │
│                                                   │
│  ① async fn 呼び出し                              │
│     → Future (状態マシン) を生成 (まだ実行しない)   │
│                                                   │
│  ② .await                                        │
│     → Executor が poll() を呼ぶ                   │
│                                                   │
│  ③ Poll::Pending (I/O未完了)                      │
│     → Waker を登録してタスクを中断                  │
│     → Executor は他のタスクを実行                   │
│                                                   │
│  ④ I/O 完了通知 (Reactor → Waker)                 │
│     → Executor がタスクを再スケジュール             │
│                                                   │
│  ⑤ 再 poll() → Poll::Ready(value)                │
│     → .await が value を返す                      │
└─────────────────────────────────────────────────┘
```

---

## 4. 複数 Future の合成

### コード例4: join! と select!

```rust
use tokio::time::{sleep, Duration};

async fn task_a() -> String {
    sleep(Duration::from_millis(100)).await;
    "A完了".to_string()
}

async fn task_b() -> String {
    sleep(Duration::from_millis(200)).await;
    "B完了".to_string()
}

async fn task_c() -> String {
    sleep(Duration::from_millis(50)).await;
    "C完了".to_string()
}

#[tokio::main]
async fn main() {
    // join! — 全ての Future を並行実行し、全完了を待つ
    let (a, b, c) = tokio::join!(task_a(), task_b(), task_c());
    println!("{}, {}, {}", a, b, c);
    // 200ms で全完了 (最も遅い task_b に合わせる)

    // select! — 最初に完了した Future の結果を取得
    tokio::select! {
        val = task_a() => println!("Aが先: {}", val),
        val = task_b() => println!("Bが先: {}", val),
        val = task_c() => println!("Cが先: {}", val),
    }
    // "Cが先: C完了" (50ms で最速)
}
```

### コード例5: エラーハンドリング付き並行処理

```rust
use anyhow::Result;

async fn fetch_user(id: u64) -> Result<String> {
    // API呼び出しをシミュレート
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    Ok(format!("User#{}", id))
}

async fn fetch_profile(id: u64) -> Result<String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
    Ok(format!("Profile#{}", id))
}

#[tokio::main]
async fn main() -> Result<()> {
    // try_join! — いずれかがエラーなら即座に返す
    let (user, profile) = tokio::try_join!(
        fetch_user(1),
        fetch_profile(1),
    )?;

    println!("{}, {}", user, profile);

    // JoinSet — 動的にタスクを追加して全完了を待つ
    let mut set = tokio::task::JoinSet::new();
    for id in 1..=5 {
        set.spawn(fetch_user(id));
    }

    while let Some(result) = set.join_next().await {
        let user = result??;
        println!("取得: {}", user);
    }

    Ok(())
}
```

---

## 5. Pin と Unpin

```
┌─────────────────────────────────────────────┐
│              Pin の必要性                      │
│                                               │
│  async fn foo() {                            │
│      let data = vec![1, 2, 3];               │
│      let ref_to_data = &data;  ← 自己参照    │
│      some_async_op().await;    ← 中断点      │
│      println!("{:?}", ref_to_data);           │
│  }                                            │
│                                               │
│  中断時に Future がメモリ上で移動すると       │
│  ref_to_data が無効になる → Pin で防ぐ       │
│                                               │
│  Pin<&mut T>:                                │
│    T が Unpin なら → 移動OK (大半の型)       │
│    T が !Unpin なら → 移動禁止 (async 生成物) │
└─────────────────────────────────────────────┘
```

---

## 6. 比較表

### ランタイム比較

| 項目 | tokio | async-std | smol |
|---|---|---|---|
| エコシステム | 最大規模 | 中規模 | 軽量 |
| マルチスレッド | デフォルト対応 | デフォルト対応 | 対応 |
| I/O | 独自 (mio ベース) | 独自 | polling ベース |
| タイマー | `tokio::time` | `async_std::task` | `async-io` |
| 採用実績 | Axum, tonic 等 | 一部 | 組み込み向け |
| 依存サイズ | 中 | 中 | 小 |

### 同期 vs 非同期の選択基準

| 基準 | 同期処理が適切 | 非同期処理が適切 |
|---|---|---|
| I/O パターン | CPU集中的処理 | I/O集中的処理 |
| 同時接続数 | 少数 (〜100) | 多数 (1K〜100K+) |
| レイテンシ要件 | 予測可能性重視 | スループット重視 |
| コード複雑性 | シンプルさ重視 | 多少の複雑さ許容 |
| ライブラリ | 同期APIのみの場合 | 非同期エコシステム活用 |
| デバッグ | スタックトレース明快 | 非同期対応ツール必要 |

---

## 7. アンチパターン

### アンチパターン1: async 内でのブロッキング呼び出し

```rust
// NG: async 内で std::thread::sleep (ランタイム全体をブロック!)
async fn bad_delay() {
    std::thread::sleep(std::time::Duration::from_secs(5));
    // 他の全タスクが5秒間停止する
}

// OK: tokio の非同期スリープを使う
async fn good_delay() {
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    // 他のタスクは中断中も実行を続ける
}

// OK: CPU集中処理は spawn_blocking で逃がす
async fn cpu_heavy() {
    let result = tokio::task::spawn_blocking(|| {
        // 重い同期計算
        (0..10_000_000u64).sum::<u64>()
    }).await.unwrap();
    println!("結果: {}", result);
}
```

### アンチパターン2: Future を .await せずに放置

```rust
// NG: async fn の戻り値を無視 (何も実行されない!)
async fn send_notification() {
    println!("通知送信");
}

async fn bad_example() {
    send_notification(); // ← .await なし! 実行されない!
    // コンパイラ警告: unused future
}

// OK: 必ず .await するか spawn する
async fn good_example() {
    send_notification().await;           // パターン1: 同期的に待つ
    tokio::spawn(send_notification());   // パターン2: バックグラウンド実行
}
```

---

## FAQ

### Q1: `#[tokio::main]` は何をしているの?

**A:** 非同期ランタイム(Executor)を起動し、`main` 関数の `async` ブロックを実行するマクロです。

```rust
// これは:
#[tokio::main]
async fn main() { /* ... */ }

// 以下と等価:
fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { /* ... */ })
}
```

### Q2: `Send` + `'static` 制約が必要な理由は?

**A:** `tokio::spawn` はタスクを別スレッドで実行する可能性があるため、`Send`(スレッド間移動可能)と `'static`(借用を含まない)が必要です。

```rust
// エラー: ローカル参照を含む Future は spawn できない
async fn bad() {
    let data = String::from("hello");
    let r = &data;
    tokio::spawn(async move {
        // println!("{}", r); // ← コンパイルエラー: &String は 'static でない
    });
}

// OK: 所有権を move する
async fn good() {
    let data = String::from("hello");
    tokio::spawn(async move {
        println!("{}", data); // OK: data の所有権を移動
    });
}
```

### Q3: async trait はどう書く?

**A:** Rust 1.75+ では `async fn` を trait 内で直接使えます。それ以前は `async-trait` クレートを使います。

```rust
// Rust 1.75+ (ネイティブ対応)
trait Service {
    async fn call(&self, req: Request) -> Response;
}

// Rust 1.74以前 (async-trait クレート)
use async_trait::async_trait;

#[async_trait]
trait Service {
    async fn call(&self, req: Request) -> Response;
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Future trait | `poll()` が `Ready` か `Pending` を返す遅延評価モデル |
| async/await | Future を生成・待機するシンタックスシュガー |
| ランタイム | Executor (タスク実行) + Reactor (I/O監視) + Waker (通知) |
| tokio | 最も広く使われる非同期ランタイム |
| join! | 複数 Future の並行実行・全完了待ち |
| select! | 複数 Future のうち最初の完了を取得 |
| Pin | 自己参照を含む Future のメモリ安全性を保証 |
| spawn_blocking | 同期処理をブロッキングスレッドプールに逃がす |

## 次に読むべきガイド

- [Tokioランタイム](./01-tokio-runtime.md) — タスク管理とチャネルの詳細
- [非同期パターン](./02-async-patterns.md) — Stream、並行制限、リトライ
- [ネットワーク](./03-networking.md) — HTTP/WebSocket/gRPC

## 参考文献

1. **Asynchronous Programming in Rust**: https://rust-lang.github.io/async-book/
2. **Tokio Tutorial**: https://tokio.rs/tokio/tutorial
3. **Pin and suffering (Fasterthanlime)**: https://fasterthanli.me/articles/pin-and-suffering
