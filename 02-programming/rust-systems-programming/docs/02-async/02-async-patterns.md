# 非同期パターン — Stream、並行制限、リトライ

> 実践的な非同期設計パターンとして Stream 処理、並行度制御、リトライ戦略、バックプレッシャーを体系的に学ぶ

## この章で学ぶこと

1. **Stream** — 非同期イテレータの概念と操作 (map, filter, buffer)
2. **並行制限パターン** — セマフォ、buffer_unordered、レートリミッター
3. **リトライとタイムアウト** — 指数バックオフ、サーキットブレーカー

---

## 1. Stream の基本

```
┌─────────────────── Iterator vs Stream ──────────────┐
│                                                      │
│  Iterator (同期):                                    │
│    fn next(&mut self) -> Option<Item>                │
│    → 即座に次の値を返す                               │
│                                                      │
│  Stream (非同期):                                     │
│    fn poll_next(Pin<&mut Self>, &mut Context)        │
│         -> Poll<Option<Item>>                        │
│    → Ready(Some(item)) : 値あり                      │
│    → Ready(None)       : ストリーム終了               │
│    → Pending           : まだ準備できていない          │
│                                                      │
│  StreamExt トレイト:                                  │
│    .next().await   .map()   .filter()                │
│    .take()   .collect()   .for_each()                │
└──────────────────────────────────────────────────────┘
```

### コード例1: Stream の作成と操作

```rust
use futures::stream::{self, StreamExt};
use tokio::time::{sleep, Duration, interval};

#[tokio::main]
async fn main() {
    // iter から Stream を作成
    let sum: i32 = stream::iter(1..=10)
        .filter(|x| futures::future::ready(x % 2 == 0))
        .map(|x| x * x)
        .fold(0, |acc, x| async move { acc + x })
        .await;
    println!("偶数の二乗和: {}", sum); // 220

    // 非同期変換を含む Stream
    let results: Vec<String> = stream::iter(vec!["a", "b", "c"])
        .then(|item| async move {
            sleep(Duration::from_millis(50)).await;
            format!("processed_{}", item)
        })
        .collect()
        .await;
    println!("{:?}", results);

    // interval から無限 Stream
    let mut ticker = tokio::time::interval(Duration::from_millis(100));
    let ticks: Vec<_> = stream::poll_fn(|cx| ticker.poll_tick(cx).map(Some))
        .take(5)
        .map(|instant| format!("{:?}", instant.elapsed()))
        .collect()
        .await;
    println!("Ticks: {:?}", ticks);
}
```

### コード例2: カスタム Stream

```rust
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// 指定範囲の素数を非同期で生成する Stream
struct PrimeStream {
    current: u64,
    max: u64,
}

impl PrimeStream {
    fn new(max: u64) -> Self {
        PrimeStream { current: 2, max }
    }

    fn is_prime(n: u64) -> bool {
        if n < 2 { return false; }
        if n < 4 { return true; }
        if n % 2 == 0 { return false; }
        let mut i = 3;
        while i * i <= n {
            if n % i == 0 { return false; }
            i += 2;
        }
        true
    }
}

impl Stream for PrimeStream {
    type Item = u64;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<u64>> {
        while self.current <= self.max {
            let n = self.current;
            self.current += 1;
            if Self::is_prime(n) {
                return Poll::Ready(Some(n));
            }
        }
        Poll::Ready(None)
    }
}

// 使用例:
// use futures::StreamExt;
// let primes: Vec<u64> = PrimeStream::new(50).collect().await;
// // [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```

---

## 2. 並行制限パターン

### 並行度制御の全体像

```
┌─────────────── 並行度制御パターン ───────────────┐
│                                                    │
│  1. buffer_unordered(N)                           │
│     Stream の各要素を最大N並行で処理               │
│     完了順に結果を返す                             │
│                                                    │
│  2. Semaphore                                      │
│     明示的なリソースガード                          │
│     任意の非同期処理に適用可能                      │
│                                                    │
│  3. JoinSet + カウンタ                             │
│     動的タスクの並行数を手動管理                    │
│                                                    │
│  Input   ─┬─ [Task 1] ─┐                         │
│  Stream    ├─ [Task 2] ─┼─→ Output Stream         │
│            ├─ [Task 3] ─┤   (最大N並行)           │
│            │  (待機中)   │                         │
│            └─ ...       ─┘                         │
└────────────────────────────────────────────────────┘
```

### コード例3: buffer_unordered で並行制限

```rust
use futures::stream::{self, StreamExt};
use tokio::time::{sleep, Duration};

async fn fetch_page(url: String) -> Result<String, String> {
    sleep(Duration::from_millis(100)).await;
    Ok(format!("Content of {}", url))
}

#[tokio::main]
async fn main() {
    let urls: Vec<String> = (1..=20)
        .map(|i| format!("https://example.com/page/{}", i))
        .collect();

    // 最大5並行でフェッチ
    let results: Vec<_> = stream::iter(urls)
        .map(|url| fetch_page(url))    // Stream<Future>
        .buffer_unordered(5)            // 最大5並行実行
        .collect()
        .await;

    let success = results.iter().filter(|r| r.is_ok()).count();
    println!("成功: {}/20", success);
}
```

### コード例4: Semaphore による同時接続制限

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let semaphore = Arc::new(Semaphore::new(3)); // 最大3同時
    let mut handles = Vec::new();

    for i in 0..10 {
        let sem = semaphore.clone();
        let handle = tokio::spawn(async move {
            // permit を取得するまで待機
            let _permit = sem.acquire().await.unwrap();
            println!("[{}] 開始", i);
            sleep(Duration::from_millis(200)).await;
            println!("[{}] 完了", i);
            // _permit が drop されると自動的に permit を返却
        });
        handles.push(handle);
    }

    for h in handles {
        h.await.unwrap();
    }
}
```

---

## 3. リトライとタイムアウト

### コード例5: 指数バックオフ付きリトライ

```rust
use std::time::Duration;
use tokio::time::sleep;

/// 指数バックオフ付きリトライ
async fn retry_with_backoff<F, Fut, T, E>(
    mut operation: F,
    max_retries: u32,
    initial_delay: Duration,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut delay = initial_delay;
    let mut last_err = None;

    for attempt in 0..=max_retries {
        match operation().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                if attempt < max_retries {
                    eprintln!(
                        "試行 {}/{} 失敗: {}。{:?} 後にリトライ",
                        attempt + 1, max_retries, e, delay
                    );
                    sleep(delay).await;
                    delay = delay.mul_f64(2.0).min(Duration::from_secs(30));
                }
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap())
}

// 使用例:
// let result = retry_with_backoff(
//     || async { reqwest::get("https://api.example.com/data").await },
//     3,
//     Duration::from_millis(500),
// ).await?;
```

### コード例6: タイムアウトラッパー

```rust
use tokio::time::{timeout, Duration};

async fn fetch_with_timeout(url: &str) -> anyhow::Result<String> {
    // 個別リクエストのタイムアウト
    let response = timeout(
        Duration::from_secs(10),
        reqwest::get(url),
    )
    .await
    .map_err(|_| anyhow::anyhow!("リクエストタイムアウト (10秒)"))?
    .map_err(|e| anyhow::anyhow!("HTTPエラー: {}", e))?;

    let body = timeout(
        Duration::from_secs(30),
        response.text(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("ボディ読み取りタイムアウト (30秒)"))?
    .map_err(|e| anyhow::anyhow!("読み取りエラー: {}", e))?;

    Ok(body)
}
```

---

## 4. バックプレッシャー

```
┌──────────── バックプレッシャーの仕組み ────────────┐
│                                                     │
│  Producer (高速)                                    │
│    │                                                │
│    ▼                                                │
│  [Buffer: capacity = 32]                            │
│    │                                                │
│    │  バッファ満杯時:                                │
│    │  → bounded:  send().await がブロック (推奨)     │
│    │  → unbounded: メモリ無限消費 (危険)             │
│    │                                                │
│    ▼                                                │
│  Consumer (低速)                                    │
│                                                     │
│  適切なバッファサイズ:                                │
│    ピーク流量 x 処理遅延 x 2 (安全マージン)         │
└─────────────────────────────────────────────────────┘
```

### コード例7: バックプレッシャー対応パイプライン

```rust
use tokio::sync::mpsc;
use futures::stream::StreamExt;

struct Pipeline;

impl Pipeline {
    async fn run() {
        let (raw_tx, raw_rx) = mpsc::channel::<Vec<u8>>(64);
        let (parsed_tx, parsed_rx) = mpsc::channel::<serde_json::Value>(32);
        let (result_tx, mut result_rx) = mpsc::channel::<String>(16);

        // Stage 1: データ取得
        tokio::spawn(async move {
            for i in 0..100 {
                let data = format!(r#"{{"id": {}}}"#, i).into_bytes();
                if raw_tx.send(data).await.is_err() { break; }
            }
        });

        // Stage 2: パース (バッファが満杯なら待機)
        tokio::spawn(async move {
            let mut rx = tokio_stream::wrappers::ReceiverStream::new(raw_rx);
            while let Some(data) = rx.next().await {
                if let Ok(json) = serde_json::from_slice(&data) {
                    if parsed_tx.send(json).await.is_err() { break; }
                }
            }
        });

        // Stage 3: 変換
        tokio::spawn(async move {
            let mut rx = tokio_stream::wrappers::ReceiverStream::new(parsed_rx);
            while let Some(value) = rx.next().await {
                let result = format!("Processed: {}", value);
                if result_tx.send(result).await.is_err() { break; }
            }
        });

        // 結果収集
        while let Some(result) = result_rx.recv().await {
            println!("{}", result);
        }
    }
}
```

---

## 5. 比較表

### 並行制限手法の比較

| 手法 | 粒度 | 適用対象 | 利点 | 欠点 |
|---|---|---|---|---|
| `buffer_unordered(N)` | Stream 要素 | Stream パイプライン | 簡潔 | Stream限定 |
| `Semaphore` | 任意ブロック | どこでも | 柔軟 | ボイラープレート多 |
| `JoinSet` + カウンタ | タスク | 動的タスク生成 | 制御しやすい | 手動管理 |
| `mpsc(N)` | メッセージ | Producer-Consumer | バックプレッシャー | チャネル設計必要 |

### リトライ戦略の比較

| 戦略 | 遅延パターン | ユースケース | リスク |
|---|---|---|---|
| 即時リトライ | なし | 一時的ロック競合 | サーバー過負荷 |
| 固定遅延 | 常に同じ間隔 | 定期ポーリング | 効率が悪い |
| 指数バックオフ | 2倍ずつ増加 | API呼び出し | 収束が遅い |
| ジッター付きバックオフ | ランダム要素追加 | 分散システム (推奨) | 実装が少し複雑 |
| サーキットブレーカー | 一定失敗で停止 | 障害伝播防止 | 状態管理が必要 |

---

## 6. アンチパターン

### アンチパターン1: 無制限並行

```rust
// NG: 10,000 URLを一斉にフェッチ → ファイルディスクリプタ枯渇
let handles: Vec<_> = urls.iter()
    .map(|url| tokio::spawn(fetch(url.clone())))
    .collect();

// OK: buffer_unordered で制限
let results: Vec<_> = stream::iter(urls)
    .map(|url| fetch(url))
    .buffer_unordered(50) // 最大50並行
    .collect()
    .await;
```

### アンチパターン2: リトライなしの一発勝負

```rust
// NG: ネットワーク一時障害で即座に失敗
let data = reqwest::get(url).await?;

// OK: リトライ + タイムアウト + ロギング
let data = retry_with_backoff(
    || async {
        timeout(Duration::from_secs(10), reqwest::get(url)).await?
    },
    3,
    Duration::from_millis(500),
).await?;
```

---

## FAQ

### Q1: `buffer_unordered` と `buffered` の違いは?

**A:** `buffered(N)` は入力順に結果を返します。`buffer_unordered(N)` は完了順に返します。レイテンシが均一でない場合は `buffer_unordered` の方がスループットが高くなります。

### Q2: Stream はいつ使うべき?

**A:** 以下の場面で有効です: (1) 大量データを逐次処理する時 (2) WebSocket のような継続的なデータ受信 (3) ページネーション付きAPI (4) イベント駆動処理。小規模なデータセットなら `Vec` + `join_all` で十分です。

### Q3: バックプレッシャーが効いているか確認する方法は?

**A:** `mpsc::Sender::send().await` の待機時間を計測するか、チャネルの `capacity()` をモニタリングします。メトリクスライブラリ (metrics クレート) と組み合わせてダッシュボードで可視化するのが本番環境での推奨手法です。

---

## まとめ

| 項目 | 要点 |
|---|---|
| Stream | 非同期イテレータ。`StreamExt` で map/filter/collect |
| buffer_unordered | Stream の並行制限。最も簡潔な手法 |
| Semaphore | 汎用的な並行度ガード |
| リトライ | 指数バックオフ + ジッターが推奨 |
| タイムアウト | `tokio::time::timeout` で個別・全体を制御 |
| バックプレッシャー | bounded チャネルで自然な流量制御 |
| パイプライン | ステージ間をチャネルで接続 |

## 次に読むべきガイド

- [ネットワーク](./03-networking.md) — HTTP/WebSocket/gRPCの非同期パターン適用
- [Axum](./04-axum-web.md) — Webフレームワークでの実践
- [並行性](../03-systems/01-concurrency.md) — スレッドレベルの並行制御

## 参考文献

1. **futures crate (StreamExt)**: https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html
2. **Tokio — Streams**: https://tokio.rs/tokio/tutorial/streams
3. **Tower (middleware/retry/rate-limit)**: https://docs.rs/tower/latest/tower/
