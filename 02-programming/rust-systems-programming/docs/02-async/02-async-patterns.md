# 非同期パターン — Stream、並行制限、リトライ

> 実践的な非同期設計パターンとして Stream 処理、並行度制御、リトライ戦略、バックプレッシャーを体系的に学ぶ

## この章で学ぶこと

1. **Stream** — 非同期イテレータの概念と操作 (map, filter, buffer)
2. **並行制限パターン** — セマフォ、buffer_unordered、レートリミッター
3. **リトライとタイムアウト** — 指数バックオフ、サーキットブレーカー
4. **バックプレッシャー** — bounded チャネルとパイプライン設計
5. **ファンアウト/ファンイン** — 分散処理と結果集約

---

## 1. Stream の基本

### 1.1 Iterator と Stream の対比

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
│    .chain()  .zip()  .enumerate()                    │
│    .fold()   .scan()  .flat_map()                    │
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

    // chain — 2つの Stream を連結
    let first = stream::iter(vec![1, 2, 3]);
    let second = stream::iter(vec![4, 5, 6]);
    let combined: Vec<i32> = first.chain(second).collect().await;
    println!("連結: {:?}", combined); // [1, 2, 3, 4, 5, 6]

    // zip — 2つの Stream を並行に処理してペアにする
    let names = stream::iter(vec!["Alice", "Bob", "Carol"]);
    let ages = stream::iter(vec![30, 25, 35]);
    let pairs: Vec<_> = names.zip(ages).collect().await;
    println!("ペア: {:?}", pairs); // [("Alice", 30), ("Bob", 25), ("Carol", 35)]

    // scan — 状態を持つ変換
    let running_total: Vec<i32> = stream::iter(vec![1, 2, 3, 4, 5])
        .scan(0, |state, x| {
            *state += x;
            futures::future::ready(Some(*state))
        })
        .collect()
        .await;
    println!("累積和: {:?}", running_total); // [1, 3, 6, 10, 15]
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

### コード例3: async_stream を使った簡易 Stream 作成

```rust
use async_stream::stream;
use futures::StreamExt;
use tokio::time::{sleep, Duration};

/// async_stream マクロで簡潔に Stream を作成
fn countdown(from: u32) -> impl futures::Stream<Item = u32> {
    stream! {
        for i in (0..=from).rev() {
            sleep(Duration::from_millis(100)).await;
            yield i;
        }
    }
}

/// ページネーション付き API から全データを取得する Stream
fn fetch_all_pages(base_url: &str) -> impl futures::Stream<Item = Vec<String>> + '_ {
    stream! {
        let mut page = 1;
        loop {
            let url = format!("{}/items?page={}&per_page=50", base_url, page);
            // let response = reqwest::get(&url).await.unwrap();
            // let items: Vec<String> = response.json().await.unwrap();
            let items: Vec<String> = (0..50)
                .map(|i| format!("item_{}_{}", page, i))
                .collect();

            if items.is_empty() {
                break; // 最終ページ
            }

            let is_last = items.len() < 50;
            yield items;

            if is_last {
                break;
            }
            page += 1;

            // レート制限対策
            sleep(Duration::from_millis(100)).await;
        }
    }
}

#[tokio::main]
async fn main() {
    // カウントダウン
    let nums: Vec<u32> = countdown(5).collect().await;
    println!("カウントダウン: {:?}", nums); // [5, 4, 3, 2, 1, 0]

    // 全ページ取得
    let mut total = 0;
    let mut pages = std::pin::pin!(fetch_all_pages("https://api.example.com"));
    while let Some(items) = pages.next().await {
        total += items.len();
        println!("ページ取得: {} アイテム (累計: {})", items.len(), total);
    }
}
```

### コード例4: ReceiverStream とチャネル変換

```rust
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel::<i32>(32);

    // プロデューサー
    tokio::spawn(async move {
        for i in 0..20 {
            let _ = tx.send(i).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    });

    // mpsc::Receiver を Stream に変換
    let stream = ReceiverStream::new(rx);

    // Stream 操作を適用
    let results: Vec<i32> = stream
        .filter(|x| futures::future::ready(*x % 3 == 0))   // 3の倍数
        .map(|x| x * 10)                                     // 10倍
        .take(4)                                              // 最初の4つ
        .collect()
        .await;

    println!("結果: {:?}", results); // [0, 30, 60, 90]
}
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
│  2. buffered(N)                                    │
│     Stream の各要素を最大N並行で処理               │
│     入力順に結果を返す                             │
│                                                    │
│  3. Semaphore                                      │
│     明示的なリソースガード                          │
│     任意の非同期処理に適用可能                      │
│                                                    │
│  4. JoinSet + カウンタ                             │
│     動的タスクの並行数を手動管理                    │
│                                                    │
│  Input   ─┬─ [Task 1] ─┐                         │
│  Stream    ├─ [Task 2] ─┼─→ Output Stream         │
│            ├─ [Task 3] ─┤   (最大N並行)           │
│            │  (待機中)   │                         │
│            └─ ...       ─┘                         │
└────────────────────────────────────────────────────┘
```

### コード例5: buffer_unordered で並行制限

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

### コード例6: buffered vs buffer_unordered の違い

```rust
use futures::stream::{self, StreamExt};
use tokio::time::{sleep, Duration, Instant};

async fn variable_delay(id: u32) -> (u32, Duration) {
    let delay = Duration::from_millis(if id % 2 == 0 { 200 } else { 50 });
    sleep(delay).await;
    (id, delay)
}

#[tokio::main]
async fn main() {
    let start = Instant::now();

    // buffered(3): 入力順に結果を返す
    println!("=== buffered(3) ===");
    let results: Vec<_> = stream::iter(1..=6)
        .map(|id| variable_delay(id))
        .buffered(3)
        .collect()
        .await;
    for (id, delay) in &results {
        println!("  id={}, delay={:?}", id, delay);
    }
    // 入力順: 1, 2, 3, 4, 5, 6 (遅いタスクがあると後続も待たされる)

    println!("=== buffer_unordered(3) ===");
    let results: Vec<_> = stream::iter(1..=6)
        .map(|id| variable_delay(id))
        .buffer_unordered(3)
        .collect()
        .await;
    for (id, delay) in &results {
        println!("  id={}, delay={:?}", id, delay);
    }
    // 完了順: 奇数(50ms)が先に返る → スループットが高い

    println!("合計時間: {:?}", start.elapsed());
}
```

### コード例7: Semaphore による同時接続制限

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
            println!("[{}] 開始 (残り permit: {})",
                i, sem.available_permits());
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

### コード例8: Semaphore でのリソースプール

```rust
use std::sync::Arc;
use tokio::sync::{Semaphore, OwnedSemaphorePermit};

/// 接続プール的な使い方
struct ConnectionPool {
    semaphore: Arc<Semaphore>,
    max_connections: usize,
}

struct PooledConnection {
    id: usize,
    _permit: OwnedSemaphorePermit,
}

impl ConnectionPool {
    fn new(max: usize) -> Self {
        ConnectionPool {
            semaphore: Arc::new(Semaphore::new(max)),
            max_connections: max,
        }
    }

    async fn acquire(&self) -> PooledConnection {
        let permit = self.semaphore.clone().acquire_owned().await.unwrap();
        let id = self.max_connections - self.semaphore.available_permits();
        println!("接続取得: #{} (残り: {})", id, self.semaphore.available_permits());
        PooledConnection { id, _permit: permit }
    }

    fn available(&self) -> usize {
        self.semaphore.available_permits()
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        println!("接続返却: #{}", self.id);
        // OwnedSemaphorePermit の drop で permit が自動返却される
    }
}

#[tokio::main]
async fn main() {
    let pool = Arc::new(ConnectionPool::new(3));

    let mut handles = Vec::new();
    for i in 0..8 {
        let p = pool.clone();
        handles.push(tokio::spawn(async move {
            let conn = p.acquire().await;
            println!("タスク {}: 接続 #{} を使用中", i, conn.id);
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            println!("タスク {}: 処理完了", i);
            // conn がドロップされ、接続がプールに返却される
        }));
    }

    for h in handles { h.await.unwrap(); }
    println!("最終利用可能接続数: {}", pool.available());
}
```

### コード例9: レートリミッター

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration, Instant};

/// トークンバケット方式のレートリミッター
struct RateLimiter {
    semaphore: Arc<Semaphore>,
    refill_interval: Duration,
}

impl RateLimiter {
    fn new(max_requests_per_second: usize) -> Self {
        let semaphore = Arc::new(Semaphore::new(max_requests_per_second));
        let sem_clone = semaphore.clone();
        let interval = Duration::from_secs(1) / max_requests_per_second as u32;

        // トークン補充タスク
        tokio::spawn(async move {
            loop {
                sleep(interval).await;
                if sem_clone.available_permits() < max_requests_per_second {
                    sem_clone.add_permits(1);
                }
            }
        });

        RateLimiter {
            semaphore,
            refill_interval: interval,
        }
    }

    async fn acquire(&self) {
        self.semaphore.acquire().await.unwrap().forget();
    }
}

#[tokio::main]
async fn main() {
    let limiter = Arc::new(RateLimiter::new(5)); // 5 req/s
    let start = Instant::now();

    let mut handles = Vec::new();
    for i in 0..15 {
        let l = limiter.clone();
        handles.push(tokio::spawn(async move {
            l.acquire().await;
            println!("[{:?}] リクエスト {}", start.elapsed(), i);
        }));
    }

    for h in handles { h.await.unwrap(); }
}
```

---

## 3. リトライとタイムアウト

### コード例10: 指数バックオフ付きリトライ

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

### コード例11: ジッター付き指数バックオフ

```rust
use rand::Rng;
use std::time::Duration;
use tokio::time::sleep;

/// ジッター付き指数バックオフ
/// 複数クライアントのリトライが同時に発生するのを防ぐ
async fn retry_with_jitter<F, Fut, T, E>(
    mut operation: F,
    config: RetryConfig,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut rng = rand::thread_rng();
    let mut delay = config.initial_delay;
    let mut last_err = None;

    for attempt in 0..=config.max_retries {
        match operation().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                if attempt < config.max_retries {
                    // ジッター: 0〜delay の範囲でランダム化
                    let jittered_delay = match config.jitter_strategy {
                        JitterStrategy::Full => {
                            Duration::from_millis(rng.gen_range(0..=delay.as_millis() as u64))
                        }
                        JitterStrategy::Equal => {
                            let half = delay / 2;
                            half + Duration::from_millis(
                                rng.gen_range(0..=half.as_millis() as u64)
                            )
                        }
                        JitterStrategy::Decorrelated => {
                            let min = config.initial_delay;
                            let max = delay * 3;
                            Duration::from_millis(
                                rng.gen_range(min.as_millis() as u64..=max.as_millis() as u64)
                            )
                        }
                    };

                    eprintln!(
                        "試行 {}/{} 失敗: {}。{:?} 後にリトライ",
                        attempt + 1, config.max_retries, e, jittered_delay
                    );
                    sleep(jittered_delay).await;
                    delay = (delay.mul_f64(config.multiplier))
                        .min(config.max_delay);
                }
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap())
}

struct RetryConfig {
    max_retries: u32,
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
    jitter_strategy: JitterStrategy,
}

enum JitterStrategy {
    Full,          // [0, delay]
    Equal,         // [delay/2, delay]
    Decorrelated,  // [initial, delay*3]
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter_strategy: JitterStrategy::Full,
        }
    }
}
```

### コード例12: サーキットブレーカー

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,     // 正常動作
    Open,       // 障害検出、リクエスト遮断
    HalfOpen,   // 回復テスト中
}

struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure: RwLock<Option<Instant>>,
    config: CircuitBreakerConfig,
}

struct CircuitBreakerConfig {
    failure_threshold: u32,   // Open に遷移する失敗回数
    success_threshold: u32,   // HalfOpen から Closed に戻る成功回数
    timeout: Duration,        // Open から HalfOpen に遷移する時間
}

impl CircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Self {
        CircuitBreaker {
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure: RwLock::new(None),
            config,
        }
    }

    async fn call<F, Fut, T, E>(&self, operation: F) -> Result<T, CircuitError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
    {
        // 現在の状態を確認
        let state = *self.state.read().await;

        match state {
            CircuitState::Open => {
                // タイムアウト経過していれば HalfOpen に遷移
                let last_failure = self.last_failure.read().await;
                if let Some(last) = *last_failure {
                    if last.elapsed() >= self.config.timeout {
                        drop(last_failure);
                        *self.state.write().await = CircuitState::HalfOpen;
                        self.success_count.store(0, Ordering::SeqCst);
                        // HalfOpen で処理を試行
                    } else {
                        return Err(CircuitError::Open);
                    }
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {}
        }

        // 操作を実行
        match operation().await {
            Ok(value) => {
                self.on_success().await;
                Ok(value)
            }
            Err(e) => {
                self.on_failure().await;
                Err(CircuitError::Operation(e))
            }
        }
    }

    async fn on_success(&self) {
        let state = *self.state.read().await;
        match state {
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count >= self.config.success_threshold {
                    *self.state.write().await = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::SeqCst);
                    println!("[CB] Closed に回復");
                }
            }
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            _ => {}
        }
    }

    async fn on_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        *self.last_failure.write().await = Some(Instant::now());

        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => {
                if count >= self.config.failure_threshold {
                    *self.state.write().await = CircuitState::Open;
                    println!("[CB] Open に遷移 (失敗 {} 回)", count);
                }
            }
            CircuitState::HalfOpen => {
                *self.state.write().await = CircuitState::Open;
                println!("[CB] HalfOpen → Open に戻る");
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
enum CircuitError<E> {
    Open,          // サーキットが開いている
    Operation(E),  // 操作自体のエラー
}
```

### コード例13: タイムアウトラッパー

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

/// 全体タイムアウト付きのバッチ処理
async fn batch_with_timeout(urls: Vec<String>, total_timeout: Duration) -> Vec<Result<String, String>> {
    let result = timeout(total_timeout, async {
        let mut results = Vec::new();
        for url in urls {
            match fetch_with_timeout(&url).await {
                Ok(body) => results.push(Ok(body)),
                Err(e) => results.push(Err(e.to_string())),
            }
        }
        results
    }).await;

    match result {
        Ok(results) => results,
        Err(_) => vec![Err("バッチ全体のタイムアウト".to_string())],
    }
}
```

### コード例14: 条件付きリトライ

```rust
use std::time::Duration;

/// エラーの種類に応じてリトライ判断
async fn smart_retry<F, Fut, T>(
    mut operation: F,
    max_retries: u32,
) -> Result<T, anyhow::Error>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, anyhow::Error>>,
{
    let mut delay = Duration::from_millis(100);

    for attempt in 0..=max_retries {
        match operation().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                // リトライ可能かどうか判断
                if !is_retryable(&e) {
                    return Err(e); // 即座にエラーを返す
                }

                if attempt < max_retries {
                    eprintln!(
                        "リトライ可能エラー (試行 {}/{}): {}",
                        attempt + 1, max_retries, e
                    );
                    tokio::time::sleep(delay).await;
                    delay = delay.mul_f64(2.0).min(Duration::from_secs(30));
                } else {
                    return Err(e);
                }
            }
        }
    }

    unreachable!()
}

fn is_retryable(error: &anyhow::Error) -> bool {
    let error_string = error.to_string();

    // リトライすべきエラー
    if error_string.contains("timeout")
        || error_string.contains("connection reset")
        || error_string.contains("503")
        || error_string.contains("429")
        || error_string.contains("temporary")
    {
        return true;
    }

    // リトライすべきでないエラー
    if error_string.contains("404")
        || error_string.contains("401")
        || error_string.contains("400")
        || error_string.contains("invalid")
    {
        return false;
    }

    // デフォルト: リトライする
    true
}
```

---

## 4. バックプレッシャー

### 4.1 バックプレッシャーの仕組み

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

### コード例15: バックプレッシャー対応パイプライン

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

### コード例16: 多段パイプラインとモニタリング

```rust
use tokio::sync::mpsc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// パイプラインのメトリクス
struct PipelineMetrics {
    stage1_processed: AtomicU64,
    stage2_processed: AtomicU64,
    stage3_processed: AtomicU64,
    stage1_backpressure: AtomicU64, // send が待機した回数
    stage2_backpressure: AtomicU64,
}

impl PipelineMetrics {
    fn new() -> Arc<Self> {
        Arc::new(PipelineMetrics {
            stage1_processed: AtomicU64::new(0),
            stage2_processed: AtomicU64::new(0),
            stage3_processed: AtomicU64::new(0),
            stage1_backpressure: AtomicU64::new(0),
            stage2_backpressure: AtomicU64::new(0),
        })
    }

    fn report(&self) {
        println!(
            "パイプライン状態: S1={} S2={} S3={} BP1={} BP2={}",
            self.stage1_processed.load(Ordering::Relaxed),
            self.stage2_processed.load(Ordering::Relaxed),
            self.stage3_processed.load(Ordering::Relaxed),
            self.stage1_backpressure.load(Ordering::Relaxed),
            self.stage2_backpressure.load(Ordering::Relaxed),
        );
    }
}

async fn monitored_pipeline() {
    let metrics = PipelineMetrics::new();
    let (tx1, mut rx1) = mpsc::channel::<String>(32);
    let (tx2, mut rx2) = mpsc::channel::<String>(16);

    // メトリクスモニター
    let m = metrics.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            interval.tick().await;
            m.report();
        }
    });

    // Stage 1: 高速プロデューサー
    let m1 = metrics.clone();
    tokio::spawn(async move {
        for i in 0..1000 {
            let capacity_before = tx1.capacity();
            tx1.send(format!("data_{}", i)).await.unwrap();
            m1.stage1_processed.fetch_add(1, Ordering::Relaxed);
            if tx1.capacity() == 0 {
                m1.stage1_backpressure.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    // Stage 2: 中速変換
    let m2 = metrics.clone();
    tokio::spawn(async move {
        while let Some(data) = rx1.recv().await {
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            let transformed = format!("transformed_{}", data);
            tx2.send(transformed).await.unwrap();
            m2.stage2_processed.fetch_add(1, Ordering::Relaxed);
        }
    });

    // Stage 3: 低速コンシューマー
    let m3 = metrics.clone();
    while let Some(data) = rx2.recv().await {
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        m3.stage3_processed.fetch_add(1, Ordering::Relaxed);
    }
}
```

---

## 5. ファンアウト/ファンイン パターン

### コード例17: 分散処理と結果集約

```rust
use futures::stream::{self, StreamExt};
use tokio::sync::mpsc;

/// ファンアウト: 1つの入力を複数ワーカーに分配
/// ファンイン: 複数ワーカーの結果を1つに集約
async fn fan_out_fan_in(
    items: Vec<u32>,
    num_workers: usize,
) -> Vec<String> {
    let (result_tx, mut result_rx) = mpsc::channel::<String>(100);

    // ファンアウト: アイテムをワーカーに分配
    let chunks: Vec<Vec<u32>> = items
        .chunks((items.len() + num_workers - 1) / num_workers)
        .map(|c| c.to_vec())
        .collect();

    for (worker_id, chunk) in chunks.into_iter().enumerate() {
        let tx = result_tx.clone();
        tokio::spawn(async move {
            for item in chunk {
                // 各ワーカーの処理
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                let result = format!("Worker{}: processed item {}", worker_id, item);
                if tx.send(result).await.is_err() { break; }
            }
        });
    }

    // result_tx のオリジナルを drop して、全ワーカー完了でチャネルが閉じるようにする
    drop(result_tx);

    // ファンイン: 全ワーカーの結果を集約
    let mut results = Vec::new();
    while let Some(result) = result_rx.recv().await {
        results.push(result);
    }

    results
}

#[tokio::main]
async fn main() {
    let items: Vec<u32> = (1..=100).collect();
    let results = fan_out_fan_in(items, 4).await;
    println!("処理完了: {} 件", results.len());
}
```

### コード例18: マップリデュースパターン

```rust
use futures::future::join_all;
use std::collections::HashMap;

/// Map フェーズ: テキストを単語に分割して (word, 1) ペアを生成
async fn map_phase(text: String) -> Vec<(String, u32)> {
    tokio::task::spawn_blocking(move || {
        text.split_whitespace()
            .map(|word| (word.to_lowercase(), 1))
            .collect()
    }).await.unwrap()
}

/// Shuffle フェーズ: 同じキーのペアをグループ化
fn shuffle_phase(mapped: Vec<Vec<(String, u32)>>) -> HashMap<String, Vec<u32>> {
    let mut grouped: HashMap<String, Vec<u32>> = HashMap::new();
    for pairs in mapped {
        for (word, count) in pairs {
            grouped.entry(word).or_default().push(count);
        }
    }
    grouped
}

/// Reduce フェーズ: グループ化された値を集約
async fn reduce_phase(word: String, counts: Vec<u32>) -> (String, u32) {
    let total: u32 = counts.iter().sum();
    (word, total)
}

#[tokio::main]
async fn main() {
    let texts = vec![
        "hello world hello rust".to_string(),
        "rust is fast and hello world".to_string(),
        "async rust is awesome hello".to_string(),
    ];

    // Map フェーズ (並行実行)
    let map_futures: Vec<_> = texts.into_iter().map(|t| map_phase(t)).collect();
    let mapped = join_all(map_futures).await;

    // Shuffle フェーズ
    let grouped = shuffle_phase(mapped);

    // Reduce フェーズ (並行実行)
    let reduce_futures: Vec<_> = grouped.into_iter()
        .map(|(word, counts)| reduce_phase(word, counts))
        .collect();
    let mut results = join_all(reduce_futures).await;

    // 結果をソート (出現回数降順)
    results.sort_by(|a, b| b.1.cmp(&a.1));

    println!("=== Word Count ===");
    for (word, count) in &results {
        println!("  {}: {}", word, count);
    }
}
```

---

## 6. 比較表

### 並行制限手法の比較

| 手法 | 粒度 | 適用対象 | 利点 | 欠点 |
|---|---|---|---|---|
| `buffer_unordered(N)` | Stream 要素 | Stream パイプライン | 簡潔 | Stream限定 |
| `buffered(N)` | Stream 要素 | 順序保証が必要な場合 | 順序維持 | 遅いタスクがボトルネック |
| `Semaphore` | 任意ブロック | どこでも | 柔軟 | ボイラープレート多 |
| `JoinSet` + カウンタ | タスク | 動的タスク生成 | 制御しやすい | 手動管理 |
| `mpsc(N)` | メッセージ | Producer-Consumer | バックプレッシャー | チャネル設計必要 |
| `RateLimiter` | リクエスト | API呼び出し | レート保証 | 実装が複雑 |

### リトライ戦略の比較

| 戦略 | 遅延パターン | ユースケース | リスク |
|---|---|---|---|
| 即時リトライ | なし | 一時的ロック競合 | サーバー過負荷 |
| 固定遅延 | 常に同じ間隔 | 定期ポーリング | 効率が悪い |
| 指数バックオフ | 2倍ずつ増加 | API呼び出し | 収束が遅い |
| ジッター付きバックオフ | ランダム要素追加 | 分散システム (推奨) | 実装が少し複雑 |
| サーキットブレーカー | 一定失敗で停止 | 障害伝播防止 | 状態管理が必要 |
| 条件付きリトライ | エラー種別で判断 | 精密なエラーハンドリング | 条件定義が必要 |

### Stream メソッドの比較

| メソッド | 用途 | 非同期変換 | 順序 |
|---|---|---|---|
| `.map()` | 同期変換 | 非対応 | 維持 |
| `.then()` | 非同期変換 | 対応 | 維持 (逐次) |
| `.buffered(N)` | 非同期変換 | 対応 (並行) | 維持 |
| `.buffer_unordered(N)` | 非同期変換 | 対応 (並行) | 完了順 |
| `.filter()` | 同期フィルタ | 非対応 | 維持 |
| `.filter_map()` | 同期変換+フィルタ | 非対応 | 維持 |
| `.flat_map()` | 展開 | 非対応 | 維持 |
| `.scan()` | 状態付き変換 | 対応 | 維持 |
| `.fold()` | 集約 | 対応 | N/A |
| `.for_each()` | 副作用 | 対応 | 維持 |
| `.for_each_concurrent(N)` | 並行副作用 | 対応 | 不定 |

---

## 7. アンチパターン

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

### アンチパターン3: Stream の collect 前に全要素をメモリに載せる

```rust
// NG: 大量データを一度に collect してメモリ消費
let all_items: Vec<HugeStruct> = huge_stream.collect().await; // OOM のリスク

// OK: for_each で逐次処理 (メモリ一定)
huge_stream
    .for_each(|item| async {
        process_item(item).await;
    })
    .await;

// OK: チャンク処理
use futures::stream::StreamExt;
let mut stream = huge_stream.chunks(100); // 100件ずつ
while let Some(chunk) = stream.next().await {
    process_batch(chunk).await;
}
```

### アンチパターン4: サーキットブレーカーなしのカスケード障害

```rust
// NG: 下流サービスが落ちると上流も全部詰まる
async fn bad_handler(req: Request) -> Response {
    let user = user_service.get_user(req.user_id).await?;       // タイムアウト待ち
    let profile = profile_service.get_profile(req.user_id).await?;  // さらにタイムアウト
    // 全リクエストがブロックされ、スレッドプール枯渇
    Response::ok(json!({ "user": user, "profile": profile }))
}

// OK: サーキットブレーカーでフォールバック
async fn good_handler(req: Request, cb: &CircuitBreaker) -> Response {
    let user = match cb.call(|| user_service.get_user(req.user_id)).await {
        Ok(user) => user,
        Err(CircuitError::Open) => {
            return Response::service_unavailable("ユーザーサービス利用不可");
        }
        Err(CircuitError::Operation(e)) => {
            return Response::internal_error(format!("エラー: {}", e));
        }
    };
    // ...
}
```

---

## 8. 実践パターン集

### 8.1 バッチ処理パターン

```rust
use futures::stream::{self, StreamExt};
use tokio::time::{sleep, Duration, Instant};

/// 大量のアイテムを効率的にバッチ処理
async fn batch_process(
    items: Vec<u32>,
    batch_size: usize,
    max_concurrent_batches: usize,
) -> Vec<String> {
    let start = Instant::now();

    let results: Vec<Vec<String>> = stream::iter(
        items.chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
    )
    .map(|batch| async move {
        // 各バッチの処理
        let batch_results: Vec<String> = batch.iter()
            .map(|item| format!("processed_{}", item))
            .collect();
        sleep(Duration::from_millis(100)).await; // I/O処理のシミュレート
        batch_results
    })
    .buffer_unordered(max_concurrent_batches)
    .collect()
    .await;

    let flat_results: Vec<String> = results.into_iter().flatten().collect();
    println!("バッチ処理完了: {} 件 ({:?})", flat_results.len(), start.elapsed());
    flat_results
}

#[tokio::main]
async fn main() {
    let items: Vec<u32> = (1..=1000).collect();
    let results = batch_process(items, 50, 10).await;
    println!("結果: {} 件", results.len());
}
```

### 8.2 デバウンスパターン

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, Instant};

/// デバウンス: 最後のイベントから一定時間経過後に処理を実行
async fn debounce<T: Send + 'static>(
    mut rx: mpsc::Receiver<T>,
    delay: Duration,
    mut handler: impl FnMut(T) + Send + 'static,
) {
    let mut last_value: Option<T> = None;
    let mut deadline = Instant::now() + delay;

    loop {
        tokio::select! {
            Some(value) = rx.recv() => {
                last_value = Some(value);
                deadline = Instant::now() + delay;
            }
            _ = sleep_until(deadline) => {
                if let Some(value) = last_value.take() {
                    handler(value);
                    deadline = Instant::now() + Duration::from_secs(86400); // 次のイベントまで待機
                }
            }
            else => break,
        }
    }
}

async fn sleep_until(deadline: Instant) {
    let now = Instant::now();
    if now < deadline {
        sleep(deadline - now).await;
    }
}

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel::<String>(32);

    // デバウンスハンドラー
    tokio::spawn(debounce(rx, Duration::from_millis(300), |value| {
        println!("デバウンス実行: {}", value);
    }));

    // 高頻度でイベントを送信
    for i in 0..10 {
        let _ = tx.send(format!("event_{}", i)).await;
        sleep(Duration::from_millis(50)).await;
    }

    // 最後のイベントのみ処理される
    sleep(Duration::from_millis(500)).await;
}
```

### 8.3 スロットルパターン

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, Instant};

/// スロットル: 一定間隔で最新の値を処理
async fn throttle<T: Send + Clone + 'static>(
    mut rx: mpsc::Receiver<T>,
    interval: Duration,
    mut handler: impl FnMut(T) + Send + 'static,
) {
    let mut last_value: Option<T> = None;
    let mut last_execution = Instant::now() - interval; // 初回は即実行

    loop {
        tokio::select! {
            Some(value) = rx.recv() => {
                let elapsed = last_execution.elapsed();
                if elapsed >= interval {
                    // インターバル経過済み → 即実行
                    handler(value);
                    last_execution = Instant::now();
                    last_value = None;
                } else {
                    // 次のインターバルまで保持
                    last_value = Some(value);
                }
            }
            _ = sleep(interval.saturating_sub(last_execution.elapsed())), if last_value.is_some() => {
                if let Some(value) = last_value.take() {
                    handler(value);
                    last_execution = Instant::now();
                }
            }
            else => break,
        }
    }
}
```

### 8.4 並行キャッシュパターン

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tokio::time::{Duration, Instant};

/// TTL付き非同期キャッシュ
struct AsyncCache<V: Clone> {
    entries: Mutex<HashMap<String, CacheEntry<V>>>,
    ttl: Duration,
}

struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
}

impl<V: Clone + Send + 'static> AsyncCache<V> {
    fn new(ttl: Duration) -> Arc<Self> {
        let cache = Arc::new(AsyncCache {
            entries: Mutex::new(HashMap::new()),
            ttl,
        });

        // 期限切れエントリのクリーンアップタスク
        let cache_clone = cache.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(ttl);
            loop {
                interval.tick().await;
                let mut entries = cache_clone.entries.lock().await;
                entries.retain(|_, entry| entry.inserted_at.elapsed() < cache_clone.ttl);
            }
        });

        cache
    }

    async fn get(&self, key: &str) -> Option<V> {
        let entries = self.entries.lock().await;
        entries.get(key)
            .filter(|entry| entry.inserted_at.elapsed() < self.ttl)
            .map(|entry| entry.value.clone())
    }

    async fn set(&self, key: String, value: V) {
        let mut entries = self.entries.lock().await;
        entries.insert(key, CacheEntry {
            value,
            inserted_at: Instant::now(),
        });
    }

    /// キャッシュミス時に非同期で値を取得してキャッシュ
    async fn get_or_insert<F, Fut>(&self, key: &str, factory: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = V>,
    {
        if let Some(value) = self.get(key).await {
            return value;
        }

        let value = factory().await;
        self.set(key.to_string(), value.clone()).await;
        value
    }
}

#[tokio::main]
async fn main() {
    let cache = AsyncCache::new(Duration::from_secs(60));

    // キャッシュを使ったデータ取得
    let user = cache.get_or_insert("user_1", || async {
        println!("DBからユーザー取得...");
        tokio::time::sleep(Duration::from_millis(100)).await;
        "Alice".to_string()
    }).await;
    println!("ユーザー: {}", user);

    // 2回目はキャッシュヒット
    let user = cache.get_or_insert("user_1", || async {
        println!("この行は実行されない");
        "Bob".to_string()
    }).await;
    println!("ユーザー (キャッシュ): {}", user); // Alice
}
```

---

## FAQ

### Q1: `buffer_unordered` と `buffered` の違いは?

**A:** `buffered(N)` は入力順に結果を返します。`buffer_unordered(N)` は完了順に返します。レイテンシが均一でない場合は `buffer_unordered` の方がスループットが高くなります。

### Q2: Stream はいつ使うべき?

**A:** 以下の場面で有効です:
1. 大量データを逐次処理する時 (メモリ効率)
2. WebSocket のような継続的なデータ受信
3. ページネーション付き API からのデータ取得
4. イベント駆動処理
5. ETL パイプライン

小規模なデータセットなら `Vec` + `join_all` で十分です。

### Q3: バックプレッシャーが効いているか確認する方法は?

**A:** `mpsc::Sender::send().await` の待機時間を計測するか、チャネルの `capacity()` をモニタリングします。メトリクスライブラリ (metrics クレート) と組み合わせてダッシュボードで可視化するのが本番環境での推奨手法です。

```rust
let (tx, rx) = mpsc::channel(32);

// capacity でバックプレッシャーの程度を推定
let remaining = tx.capacity();
if remaining < 4 {
    eprintln!("警告: チャネルバッファがほぼ満杯 (残り: {})", remaining);
}
```

### Q4: サーキットブレーカーのパラメータ設定の目安は?

**A:** 一般的な目安は以下の通りです。

| パラメータ | 推奨値 | 考慮事項 |
|---|---|---|
| failure_threshold | 5〜10 | 過敏すぎると正常時にも Open になる |
| timeout (Open → HalfOpen) | 30〜60秒 | 下流の回復時間に合わせる |
| success_threshold | 3〜5 | HalfOpen で安定確認する回数 |

### Q5: FuturesUnordered と buffer_unordered の違いは?

**A:** `FuturesUnordered` は Future のコレクションとして直接使え、動的に Future を追加・削除できます。`buffer_unordered` は Stream のアダプタとして使います。動的にタスクを追加する必要がある場合は `FuturesUnordered`、Stream パイプラインの一部として使う場合は `buffer_unordered` が適切です。

```rust
use futures::stream::FuturesUnordered;
use futures::StreamExt;

let mut futures = FuturesUnordered::new();

// 動的に Future を追加
futures.push(async { 1 });
futures.push(async { 2 });

// 完了順に取得
while let Some(result) = futures.next().await {
    println!("完了: {}", result);
    // 条件に応じて新しい Future を追加
    if result < 5 {
        futures.push(async move { result + 10 });
    }
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Stream | 非同期イテレータ。`StreamExt` で map/filter/collect |
| async_stream | `stream!` マクロで簡潔に Stream を作成 |
| ReceiverStream | mpsc::Receiver を Stream に変換 |
| buffer_unordered | Stream の並行制限。完了順で高スループット |
| buffered | Stream の並行制限。入力順を維持 |
| Semaphore | 汎用的な並行度ガード |
| リトライ | 指数バックオフ + ジッターが推奨 |
| サーキットブレーカー | カスケード障害の防止 |
| タイムアウト | `tokio::time::timeout` で個別・全体を制御 |
| バックプレッシャー | bounded チャネルで自然な流量制御 |
| パイプライン | ステージ間をチャネルで接続 |
| ファンアウト/ファンイン | 分散処理と結果集約のパターン |
| デバウンス | 最後のイベント後に一定時間経過で実行 |
| スロットル | 一定間隔で最新の値を処理 |
| 非同期キャッシュ | TTL付きの並行安全なキャッシュ |

## 次に読むべきガイド

- [ネットワーク](./03-networking.md) — HTTP/WebSocket/gRPCの非同期パターン適用
- [Axum](./04-axum-web.md) — Webフレームワークでの実践
- [並行性](../03-systems/01-concurrency.md) — スレッドレベルの並行制御

## 参考文献

1. **futures crate (StreamExt)**: https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html
2. **Tokio — Streams**: https://tokio.rs/tokio/tutorial/streams
3. **Tower (middleware/retry/rate-limit)**: https://docs.rs/tower/latest/tower/
4. **async-stream crate**: https://docs.rs/async-stream/latest/async_stream/
5. **AWS Blog — Exponential Backoff And Jitter**: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
