# Async Patterns -- Streams, Concurrency Limits, and Retries

> Systematically learn practical async design patterns including Stream processing, concurrency control, retry strategies, and backpressure.

## What You Will Learn in This Chapter

1. **Stream** -- Concept of asynchronous iterators and operations (map, filter, buffer)
2. **Concurrency Limit Patterns** -- Semaphores, buffer_unordered, rate limiters
3. **Retry and Timeout** -- Exponential backoff, circuit breakers
4. **Backpressure** -- Bounded channels and pipeline design
5. **Fan-out / Fan-in** -- Distributed processing and result aggregation


## Prerequisites

The following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Tokio Runtime -- Task Management and Channels](./01-tokio-runtime.md)

---

## 1. Stream Basics

### 1.1 Comparison Between Iterator and Stream

```
┌─────────────────── Iterator vs Stream ──────────────┐
│                                                      │
│  Iterator (synchronous):                             │
│    fn next(&mut self) -> Option<Item>                │
│    → Returns the next value immediately              │
│                                                      │
│  Stream (asynchronous):                              │
│    fn poll_next(Pin<&mut Self>, &mut Context)        │
│         -> Poll<Option<Item>>                        │
│    → Ready(Some(item)) : value available             │
│    → Ready(None)       : end of stream               │
│    → Pending           : not ready yet               │
│                                                      │
│  StreamExt trait:                                    │
│    .next().await   .map()   .filter()                │
│    .take()   .collect()   .for_each()                │
│    .chain()  .zip()  .enumerate()                    │
│    .fold()   .scan()  .flat_map()                    │
└──────────────────────────────────────────────────────┘
```

### Example 1: Creating and Operating on Streams

```rust
use futures::stream::{self, StreamExt};
use tokio::time::{sleep, Duration, interval};

#[tokio::main]
async fn main() {
    // Create a Stream from an iter
    let sum: i32 = stream::iter(1..=10)
        .filter(|x| futures::future::ready(x % 2 == 0))
        .map(|x| x * x)
        .fold(0, |acc, x| async move { acc + x })
        .await;
    println!("Sum of squares of even numbers: {}", sum); // 220

    // Stream with asynchronous transformation
    let results: Vec<String> = stream::iter(vec!["a", "b", "c"])
        .then(|item| async move {
            sleep(Duration::from_millis(50)).await;
            format!("processed_{}", item)
        })
        .collect()
        .await;
    println!("{:?}", results);

    // Infinite Stream from interval
    let mut ticker = tokio::time::interval(Duration::from_millis(100));
    let ticks: Vec<_> = stream::poll_fn(|cx| ticker.poll_tick(cx).map(Some))
        .take(5)
        .map(|instant| format!("{:?}", instant.elapsed()))
        .collect()
        .await;
    println!("Ticks: {:?}", ticks);

    // chain -- concatenate two Streams
    let first = stream::iter(vec![1, 2, 3]);
    let second = stream::iter(vec![4, 5, 6]);
    let combined: Vec<i32> = first.chain(second).collect().await;
    println!("Concatenated: {:?}", combined); // [1, 2, 3, 4, 5, 6]

    // zip -- process two Streams concurrently and pair them
    let names = stream::iter(vec!["Alice", "Bob", "Carol"]);
    let ages = stream::iter(vec![30, 25, 35]);
    let pairs: Vec<_> = names.zip(ages).collect().await;
    println!("Pairs: {:?}", pairs); // [("Alice", 30), ("Bob", 25), ("Carol", 35)]

    // scan -- stateful transformation
    let running_total: Vec<i32> = stream::iter(vec![1, 2, 3, 4, 5])
        .scan(0, |state, x| {
            *state += x;
            futures::future::ready(Some(*state))
        })
        .collect()
        .await;
    println!("Running total: {:?}", running_total); // [1, 3, 6, 10, 15]
}
```

### Example 2: Custom Stream

```rust
use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Stream that asynchronously generates prime numbers within a specified range
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

// Usage example:
// use futures::StreamExt;
// let primes: Vec<u64> = PrimeStream::new(50).collect().await;
// // [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```

### Example 3: Simple Stream Creation Using async_stream

```rust
use async_stream::stream;
use futures::StreamExt;
use tokio::time::{sleep, Duration};

/// Concisely create a Stream with the async_stream macro
fn countdown(from: u32) -> impl futures::Stream<Item = u32> {
    stream! {
        for i in (0..=from).rev() {
            sleep(Duration::from_millis(100)).await;
            yield i;
        }
    }
}

/// Stream that fetches all data from a paginated API
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
                break; // last page
            }

            let is_last = items.len() < 50;
            yield items;

            if is_last {
                break;
            }
            page += 1;

            // Rate limit mitigation
            sleep(Duration::from_millis(100)).await;
        }
    }
}

#[tokio::main]
async fn main() {
    // Countdown
    let nums: Vec<u32> = countdown(5).collect().await;
    println!("Countdown: {:?}", nums); // [5, 4, 3, 2, 1, 0]

    // Fetch all pages
    let mut total = 0;
    let mut pages = std::pin::pin!(fetch_all_pages("https://api.example.com"));
    while let Some(items) = pages.next().await {
        total += items.len();
        println!("Page fetched: {} items (cumulative: {})", items.len(), total);
    }
}
```

### Example 4: ReceiverStream and Channel Conversion

```rust
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel::<i32>(32);

    // Producer
    tokio::spawn(async move {
        for i in 0..20 {
            let _ = tx.send(i).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    });

    // Convert mpsc::Receiver into a Stream
    let stream = ReceiverStream::new(rx);

    // Apply Stream operations
    let results: Vec<i32> = stream
        .filter(|x| futures::future::ready(*x % 3 == 0))   // multiples of 3
        .map(|x| x * 10)                                     // multiply by 10
        .take(4)                                              // first 4
        .collect()
        .await;

    println!("Results: {:?}", results); // [0, 30, 60, 90]
}
```

---

## 2. Concurrency Limit Patterns

### Overview of Concurrency Control

```
┌─────────────── Concurrency Control Patterns ──────┐
│                                                    │
│  1. buffer_unordered(N)                           │
│     Process Stream elements with up to N concurrency│
│     Returns results in completion order            │
│                                                    │
│  2. buffered(N)                                    │
│     Process Stream elements with up to N concurrency│
│     Returns results in input order                 │
│                                                    │
│  3. Semaphore                                      │
│     Explicit resource guard                        │
│     Applicable to any async operation              │
│                                                    │
│  4. JoinSet + counter                              │
│     Manually manage concurrency for dynamic tasks  │
│                                                    │
│  Input   ─┬─ [Task 1] ─┐                         │
│  Stream    ├─ [Task 2] ─┼─→ Output Stream         │
│            ├─ [Task 3] ─┤   (max N concurrent)    │
│            │  (waiting) │                         │
│            └─ ...       ─┘                         │
└────────────────────────────────────────────────────┘
```

### Example 5: Concurrency Limiting with buffer_unordered

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

    // Fetch with up to 5 concurrent requests
    let results: Vec<_> = stream::iter(urls)
        .map(|url| fetch_page(url))    // Stream<Future>
        .buffer_unordered(5)            // execute up to 5 concurrently
        .collect()
        .await;

    let success = results.iter().filter(|r| r.is_ok()).count();
    println!("Success: {}/20", success);
}
```

### Example 6: Difference Between buffered and buffer_unordered

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

    // buffered(3): returns results in input order
    println!("=== buffered(3) ===");
    let results: Vec<_> = stream::iter(1..=6)
        .map(|id| variable_delay(id))
        .buffered(3)
        .collect()
        .await;
    for (id, delay) in &results {
        println!("  id={}, delay={:?}", id, delay);
    }
    // Input order: 1, 2, 3, 4, 5, 6 (slow tasks make subsequent ones wait)

    println!("=== buffer_unordered(3) ===");
    let results: Vec<_> = stream::iter(1..=6)
        .map(|id| variable_delay(id))
        .buffer_unordered(3)
        .collect()
        .await;
    for (id, delay) in &results {
        println!("  id={}, delay={:?}", id, delay);
    }
    // Completion order: odd ids (50ms) return first → higher throughput

    println!("Total time: {:?}", start.elapsed());
}
```

### Example 7: Limiting Simultaneous Connections with Semaphore

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let semaphore = Arc::new(Semaphore::new(3)); // max 3 simultaneous
    let mut handles = Vec::new();

    for i in 0..10 {
        let sem = semaphore.clone();
        let handle = tokio::spawn(async move {
            // Wait until a permit is acquired
            let _permit = sem.acquire().await.unwrap();
            println!("[{}] Started (remaining permits: {})",
                i, sem.available_permits());
            sleep(Duration::from_millis(200)).await;
            println!("[{}] Finished", i);
            // Permit is automatically returned when _permit is dropped
        });
        handles.push(handle);
    }

    for h in handles {
        h.await.unwrap();
    }
}
```

### Example 8: Resource Pool with Semaphore

```rust
use std::sync::Arc;
use tokio::sync::{Semaphore, OwnedSemaphorePermit};

/// Connection pool style usage
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
        println!("Connection acquired: #{} (remaining: {})", id, self.semaphore.available_permits());
        PooledConnection { id, _permit: permit }
    }

    fn available(&self) -> usize {
        self.semaphore.available_permits()
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        println!("Connection returned: #{}", self.id);
        // Permit is automatically returned when OwnedSemaphorePermit is dropped
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
            println!("Task {}: using connection #{}", i, conn.id);
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            println!("Task {}: processing complete", i);
            // conn is dropped, returning the connection to the pool
        }));
    }

    for h in handles { h.await.unwrap(); }
    println!("Final available connections: {}", pool.available());
}
```

### Example 9: Rate Limiter

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration, Instant};

/// Token bucket style rate limiter
struct RateLimiter {
    semaphore: Arc<Semaphore>,
    refill_interval: Duration,
}

impl RateLimiter {
    fn new(max_requests_per_second: usize) -> Self {
        let semaphore = Arc::new(Semaphore::new(max_requests_per_second));
        let sem_clone = semaphore.clone();
        let interval = Duration::from_secs(1) / max_requests_per_second as u32;

        // Token refill task
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
            println!("[{:?}] Request {}", start.elapsed(), i);
        }));
    }

    for h in handles { h.await.unwrap(); }
}
```

---

## 3. Retry and Timeout

### Example 10: Retry with Exponential Backoff

```rust
use std::time::Duration;
use tokio::time::sleep;

/// Retry with exponential backoff
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
                        "Attempt {}/{} failed: {}. Retrying after {:?}",
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

// Usage example:
// let result = retry_with_backoff(
//     || async { reqwest::get("https://api.example.com/data").await },
//     3,
//     Duration::from_millis(500),
// ).await?;
```

### Example 11: Exponential Backoff with Jitter

```rust
use rand::Rng;
use std::time::Duration;
use tokio::time::sleep;

/// Exponential backoff with jitter
/// Prevents simultaneous retries from multiple clients
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
                    // Jitter: randomize within the range 0..delay
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
                        "Attempt {}/{} failed: {}. Retrying after {:?}",
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

### Example 12: Circuit Breaker

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,     // normal operation
    Open,       // failure detected, requests blocked
    HalfOpen,   // recovery testing
}

struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure: RwLock<Option<Instant>>,
    config: CircuitBreakerConfig,
}

struct CircuitBreakerConfig {
    failure_threshold: u32,   // failure count to transition to Open
    success_threshold: u32,   // success count to return from HalfOpen to Closed
    timeout: Duration,        // duration before transitioning from Open to HalfOpen
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
        // Check current state
        let state = *self.state.read().await;

        match state {
            CircuitState::Open => {
                // If timeout elapsed, transition to HalfOpen
                let last_failure = self.last_failure.read().await;
                if let Some(last) = *last_failure {
                    if last.elapsed() >= self.config.timeout {
                        drop(last_failure);
                        *self.state.write().await = CircuitState::HalfOpen;
                        self.success_count.store(0, Ordering::SeqCst);
                        // Try processing in HalfOpen
                    } else {
                        return Err(CircuitError::Open);
                    }
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {}
        }

        // Execute the operation
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
                    println!("[CB] Recovered to Closed");
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
                    println!("[CB] Transitioned to Open (failures: {})", count);
                }
            }
            CircuitState::HalfOpen => {
                *self.state.write().await = CircuitState::Open;
                println!("[CB] HalfOpen → Open reverted");
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
enum CircuitError<E> {
    Open,          // circuit is open
    Operation(E),  // error from the operation itself
}
```

### Example 13: Timeout Wrapper

```rust
use tokio::time::{timeout, Duration};

async fn fetch_with_timeout(url: &str) -> anyhow::Result<String> {
    // Per-request timeout
    let response = timeout(
        Duration::from_secs(10),
        reqwest::get(url),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Request timeout (10s)"))?
    .map_err(|e| anyhow::anyhow!("HTTP error: {}", e))?;

    let body = timeout(
        Duration::from_secs(30),
        response.text(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Body read timeout (30s)"))?
    .map_err(|e| anyhow::anyhow!("Read error: {}", e))?;

    Ok(body)
}

/// Batch processing with overall timeout
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
        Err(_) => vec![Err("Overall batch timeout".to_string())],
    }
}
```

### Example 14: Conditional Retry

```rust
use std::time::Duration;

/// Decide whether to retry based on error type
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
                // Determine whether retryable
                if !is_retryable(&e) {
                    return Err(e); // Return error immediately
                }

                if attempt < max_retries {
                    eprintln!(
                        "Retryable error (attempt {}/{}): {}",
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

    // Errors that should be retried
    if error_string.contains("timeout")
        || error_string.contains("connection reset")
        || error_string.contains("503")
        || error_string.contains("429")
        || error_string.contains("temporary")
    {
        return true;
    }

    // Errors that should not be retried
    if error_string.contains("404")
        || error_string.contains("401")
        || error_string.contains("400")
        || error_string.contains("invalid")
    {
        return false;
    }

    // Default: retry
    true
}
```

---

## 4. Backpressure

### 4.1 How Backpressure Works

```
┌──────────── How Backpressure Works ────────────┐
│                                                     │
│  Producer (fast)                                    │
│    │                                                │
│    ▼                                                │
│  [Buffer: capacity = 32]                            │
│    │                                                │
│    │  When buffer is full:                          │
│    │  → bounded:  send().await blocks (recommended) │
│    │  → unbounded: unbounded memory use (dangerous) │
│    │                                                │
│    ▼                                                │
│  Consumer (slow)                                    │
│                                                     │
│  Appropriate buffer size:                           │
│    peak throughput x processing latency x 2 (safety margin) │
└─────────────────────────────────────────────────────┘
```

### Example 15: Backpressure-Aware Pipeline

```rust
use tokio::sync::mpsc;
use futures::stream::StreamExt;

struct Pipeline;

impl Pipeline {
    async fn run() {
        let (raw_tx, raw_rx) = mpsc::channel::<Vec<u8>>(64);
        let (parsed_tx, parsed_rx) = mpsc::channel::<serde_json::Value>(32);
        let (result_tx, mut result_rx) = mpsc::channel::<String>(16);

        // Stage 1: data fetch
        tokio::spawn(async move {
            for i in 0..100 {
                let data = format!(r#"{{"id": {}}}"#, i).into_bytes();
                if raw_tx.send(data).await.is_err() { break; }
            }
        });

        // Stage 2: parse (waits if buffer is full)
        tokio::spawn(async move {
            let mut rx = tokio_stream::wrappers::ReceiverStream::new(raw_rx);
            while let Some(data) = rx.next().await {
                if let Ok(json) = serde_json::from_slice(&data) {
                    if parsed_tx.send(json).await.is_err() { break; }
                }
            }
        });

        // Stage 3: transform
        tokio::spawn(async move {
            let mut rx = tokio_stream::wrappers::ReceiverStream::new(parsed_rx);
            while let Some(value) = rx.next().await {
                let result = format!("Processed: {}", value);
                if result_tx.send(result).await.is_err() { break; }
            }
        });

        // Result collection
        while let Some(result) = result_rx.recv().await {
            println!("{}", result);
        }
    }
}
```

### Example 16: Multi-stage Pipeline with Monitoring

```rust
use tokio::sync::mpsc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Pipeline metrics
struct PipelineMetrics {
    stage1_processed: AtomicU64,
    stage2_processed: AtomicU64,
    stage3_processed: AtomicU64,
    stage1_backpressure: AtomicU64, // number of times send waited
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
            "Pipeline state: S1={} S2={} S3={} BP1={} BP2={}",
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

    // Metrics monitor
    let m = metrics.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            interval.tick().await;
            m.report();
        }
    });

    // Stage 1: fast producer
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

    // Stage 2: medium-speed transform
    let m2 = metrics.clone();
    tokio::spawn(async move {
        while let Some(data) = rx1.recv().await {
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            let transformed = format!("transformed_{}", data);
            tx2.send(transformed).await.unwrap();
            m2.stage2_processed.fetch_add(1, Ordering::Relaxed);
        }
    });

    // Stage 3: slow consumer
    let m3 = metrics.clone();
    while let Some(data) = rx2.recv().await {
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        m3.stage3_processed.fetch_add(1, Ordering::Relaxed);
    }
}
```

---

## 5. Fan-out / Fan-in Patterns

### Example 17: Distributed Processing and Result Aggregation

```rust
use futures::stream::{self, StreamExt};
use tokio::sync::mpsc;

/// Fan-out: distribute one input to multiple workers
/// Fan-in: aggregate results from multiple workers into one
async fn fan_out_fan_in(
    items: Vec<u32>,
    num_workers: usize,
) -> Vec<String> {
    let (result_tx, mut result_rx) = mpsc::channel::<String>(100);

    // Fan-out: distribute items to workers
    let chunks: Vec<Vec<u32>> = items
        .chunks((items.len() + num_workers - 1) / num_workers)
        .map(|c| c.to_vec())
        .collect();

    for (worker_id, chunk) in chunks.into_iter().enumerate() {
        let tx = result_tx.clone();
        tokio::spawn(async move {
            for item in chunk {
                // Each worker's processing
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                let result = format!("Worker{}: processed item {}", worker_id, item);
                if tx.send(result).await.is_err() { break; }
            }
        });
    }

    // Drop the original result_tx so the channel closes once all workers finish
    drop(result_tx);

    // Fan-in: aggregate results from all workers
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
    println!("Processing complete: {} items", results.len());
}
```

### Example 18: MapReduce Pattern

```rust
use futures::future::join_all;
use std::collections::HashMap;

/// Map phase: split text into words and produce (word, 1) pairs
async fn map_phase(text: String) -> Vec<(String, u32)> {
    tokio::task::spawn_blocking(move || {
        text.split_whitespace()
            .map(|word| (word.to_lowercase(), 1))
            .collect()
    }).await.unwrap()
}

/// Shuffle phase: group pairs with the same key
fn shuffle_phase(mapped: Vec<Vec<(String, u32)>>) -> HashMap<String, Vec<u32>> {
    let mut grouped: HashMap<String, Vec<u32>> = HashMap::new();
    for pairs in mapped {
        for (word, count) in pairs {
            grouped.entry(word).or_default().push(count);
        }
    }
    grouped
}

/// Reduce phase: aggregate grouped values
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

    // Map phase (concurrent execution)
    let map_futures: Vec<_> = texts.into_iter().map(|t| map_phase(t)).collect();
    let mapped = join_all(map_futures).await;

    // Shuffle phase
    let grouped = shuffle_phase(mapped);

    // Reduce phase (concurrent execution)
    let reduce_futures: Vec<_> = grouped.into_iter()
        .map(|(word, counts)| reduce_phase(word, counts))
        .collect();
    let mut results = join_all(reduce_futures).await;

    // Sort results (descending by occurrence count)
    results.sort_by(|a, b| b.1.cmp(&a.1));

    println!("=== Word Count ===");
    for (word, count) in &results {
        println!("  {}: {}", word, count);
    }
}
```

---

## 6. Comparison Tables

### Comparison of Concurrency Limiting Techniques

| Technique | Granularity | Applies To | Pros | Cons |
|---|---|---|---|---|
| `buffer_unordered(N)` | Stream element | Stream pipelines | Concise | Stream-only |
| `buffered(N)` | Stream element | When ordering is required | Preserves order | Slow tasks become bottleneck |
| `Semaphore` | Arbitrary block | Anywhere | Flexible | More boilerplate |
| `JoinSet` + counter | Task | Dynamic task creation | Easy to control | Manual management |
| `mpsc(N)` | Message | Producer-Consumer | Backpressure | Channel design needed |
| `RateLimiter` | Request | API calls | Rate guarantee | Complex implementation |

### Comparison of Retry Strategies

| Strategy | Delay Pattern | Use Case | Risk |
|---|---|---|---|
| Immediate retry | None | Transient lock contention | Server overload |
| Fixed delay | Always same interval | Periodic polling | Inefficient |
| Exponential backoff | Doubles each time | API calls | Slow convergence |
| Backoff with jitter | Adds random element | Distributed systems (recommended) | Slightly more complex |
| Circuit breaker | Stop after fixed failures | Failure propagation prevention | State management needed |
| Conditional retry | Decide by error type | Precise error handling | Conditions must be defined |

### Comparison of Stream Methods

| Method | Purpose | Async Transform | Order |
|---|---|---|---|
| `.map()` | Sync transform | Not supported | Preserved |
| `.then()` | Async transform | Supported | Preserved (sequential) |
| `.buffered(N)` | Async transform | Supported (concurrent) | Preserved |
| `.buffer_unordered(N)` | Async transform | Supported (concurrent) | Completion order |
| `.filter()` | Sync filter | Not supported | Preserved |
| `.filter_map()` | Sync transform + filter | Not supported | Preserved |
| `.flat_map()` | Expand | Not supported | Preserved |
| `.scan()` | Stateful transform | Supported | Preserved |
| `.fold()` | Aggregate | Supported | N/A |
| `.for_each()` | Side effects | Supported | Preserved |
| `.for_each_concurrent(N)` | Concurrent side effects | Supported | Indeterminate |

---

## 7. Anti-Patterns

### Anti-pattern 1: Unlimited Concurrency

```rust
// NG: fetching 10,000 URLs at once → file descriptor exhaustion
let handles: Vec<_> = urls.iter()
    .map(|url| tokio::spawn(fetch(url.clone())))
    .collect();

// OK: limit with buffer_unordered
let results: Vec<_> = stream::iter(urls)
    .map(|url| fetch(url))
    .buffer_unordered(50) // up to 50 concurrent
    .collect()
    .await;
```

### Anti-pattern 2: One-shot Without Retry

```rust
// NG: fails immediately on transient network issue
let data = reqwest::get(url).await?;

// OK: retry + timeout + logging
let data = retry_with_backoff(
    || async {
        timeout(Duration::from_secs(10), reqwest::get(url)).await?
    },
    3,
    Duration::from_millis(500),
).await?;
```

### Anti-pattern 3: Loading All Stream Elements into Memory Before collect

```rust
// NG: collecting huge data at once consumes memory
let all_items: Vec<HugeStruct> = huge_stream.collect().await; // OOM risk

// OK: process sequentially with for_each (constant memory)
huge_stream
    .for_each(|item| async {
        process_item(item).await;
    })
    .await;

// OK: chunk processing
use futures::stream::StreamExt;
let mut stream = huge_stream.chunks(100); // 100 items at a time
while let Some(chunk) = stream.next().await {
    process_batch(chunk).await;
}
```

### Anti-pattern 4: Cascading Failures Without Circuit Breaker

```rust
// NG: when downstream service fails, upstream all gets stuck
async fn bad_handler(req: Request) -> Response {
    let user = user_service.get_user(req.user_id).await?;       // waits for timeout
    let profile = profile_service.get_profile(req.user_id).await?;  // another timeout
    // All requests blocked, thread pool exhausted
    Response::ok(json!({ "user": user, "profile": profile }))
}

// OK: fallback with circuit breaker
async fn good_handler(req: Request, cb: &CircuitBreaker) -> Response {
    let user = match cb.call(|| user_service.get_user(req.user_id)).await {
        Ok(user) => user,
        Err(CircuitError::Open) => {
            return Response::service_unavailable("User service unavailable");
        }
        Err(CircuitError::Operation(e)) => {
            return Response::internal_error(format!("Error: {}", e));
        }
    };
    // ...
}
```

---

## 8. Practical Pattern Collection

### 8.1 Batch Processing Pattern

```rust
use futures::stream::{self, StreamExt};
use tokio::time::{sleep, Duration, Instant};

/// Efficiently batch-process a large number of items
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
        // Per-batch processing
        let batch_results: Vec<String> = batch.iter()
            .map(|item| format!("processed_{}", item))
            .collect();
        sleep(Duration::from_millis(100)).await; // simulate I/O
        batch_results
    })
    .buffer_unordered(max_concurrent_batches)
    .collect()
    .await;

    let flat_results: Vec<String> = results.into_iter().flatten().collect();
    println!("Batch processing complete: {} items ({:?})", flat_results.len(), start.elapsed());
    flat_results
}

#[tokio::main]
async fn main() {
    let items: Vec<u32> = (1..=1000).collect();
    let results = batch_process(items, 50, 10).await;
    println!("Results: {} items", results.len());
}
```

### 8.2 Debounce Pattern

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, Instant};

/// Debounce: execute the handler after a fixed time has elapsed since the last event
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
                    deadline = Instant::now() + Duration::from_secs(86400); // wait until next event
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

    // Debounce handler
    tokio::spawn(debounce(rx, Duration::from_millis(300), |value| {
        println!("Debounce executed: {}", value);
    }));

    // Send events at high frequency
    for i in 0..10 {
        let _ = tx.send(format!("event_{}", i)).await;
        sleep(Duration::from_millis(50)).await;
    }

    // Only the last event is processed
    sleep(Duration::from_millis(500)).await;
}
```

### 8.3 Throttle Pattern

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, Instant};

/// Throttle: process the latest value at fixed intervals
async fn throttle<T: Send + Clone + 'static>(
    mut rx: mpsc::Receiver<T>,
    interval: Duration,
    mut handler: impl FnMut(T) + Send + 'static,
) {
    let mut last_value: Option<T> = None;
    let mut last_execution = Instant::now() - interval; // execute immediately on first call

    loop {
        tokio::select! {
            Some(value) = rx.recv() => {
                let elapsed = last_execution.elapsed();
                if elapsed >= interval {
                    // Interval has elapsed → execute immediately
                    handler(value);
                    last_execution = Instant::now();
                    last_value = None;
                } else {
                    // Hold until next interval
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

### 8.4 Concurrent Cache Pattern

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tokio::time::{Duration, Instant};

/// Async cache with TTL
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

        // Cleanup task for expired entries
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

    /// On cache miss, asynchronously fetch and cache the value
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

    // Fetch data using cache
    let user = cache.get_or_insert("user_1", || async {
        println!("Fetching user from DB...");
        tokio::time::sleep(Duration::from_millis(100)).await;
        "Alice".to_string()
    }).await;
    println!("User: {}", user);

    // Second call hits the cache
    let user = cache.get_or_insert("user_1", || async {
        println!("This line is not executed");
        "Bob".to_string()
    }).await;
    println!("User (cached): {}", user); // Alice
}
```

---

## FAQ

### Q1: What is the difference between `buffer_unordered` and `buffered`?

**A:** `buffered(N)` returns results in input order. `buffer_unordered(N)` returns them in completion order. When latency is non-uniform, `buffer_unordered` typically yields higher throughput.

### Q2: When should I use Streams?

**A:** They are useful in the following scenarios:
1. Processing large amounts of data sequentially (memory efficiency)
2. Continuous data reception such as WebSockets
3. Fetching data from paginated APIs
4. Event-driven processing
5. ETL pipelines

For small data sets, `Vec` + `join_all` is sufficient.

### Q3: How do I verify that backpressure is effective?

**A:** Measure the wait time of `mpsc::Sender::send().await` or monitor the channel's `capacity()`. The recommended production approach is to combine this with a metrics library (the metrics crate) and visualize it on a dashboard.

```rust
let (tx, rx) = mpsc::channel(32);

// Use capacity to estimate the level of backpressure
let remaining = tx.capacity();
if remaining < 4 {
    eprintln!("Warning: channel buffer nearly full (remaining: {})", remaining);
}
```

### Q4: What are good guidelines for circuit breaker parameters?

**A:** General guidelines are as follows.

| Parameter | Recommended Value | Considerations |
|---|---|---|
| failure_threshold | 5-10 | Too sensitive causes Open during normal operation |
| timeout (Open → HalfOpen) | 30-60s | Match downstream recovery time |
| success_threshold | 3-5 | Number of confirmations for stability in HalfOpen |

### Q5: What is the difference between FuturesUnordered and buffer_unordered?

**A:** `FuturesUnordered` can be used directly as a collection of Futures, allowing dynamic addition and removal of Futures. `buffer_unordered` is used as a Stream adapter. Use `FuturesUnordered` when you need to add tasks dynamically; use `buffer_unordered` when used as part of a Stream pipeline.

```rust
use futures::stream::FuturesUnordered;
use futures::StreamExt;

let mut futures = FuturesUnordered::new();

// Add Futures dynamically
futures.push(async { 1 });
futures.push(async { 2 });

// Retrieve in completion order
while let Some(result) = futures.next().await {
    println!("Completed: {}", result);
    // Add a new Future based on conditions
    if result < 5 {
        futures.push(async move { result + 10 });
    }
}
```

---

## Summary

| Item | Key Point |
|---|---|
| Stream | Async iterator. map/filter/collect via `StreamExt` |
| async_stream | Concisely create Streams using the `stream!` macro |
| ReceiverStream | Convert mpsc::Receiver into a Stream |
| buffer_unordered | Stream concurrency limit. High throughput in completion order |
| buffered | Stream concurrency limit. Preserves input order |
| Semaphore | General-purpose concurrency guard |
| Retry | Exponential backoff + jitter recommended |
| Circuit breaker | Prevents cascading failures |
| Timeout | Control individually or globally with `tokio::time::timeout` |
| Backpressure | Natural flow control via bounded channels |
| Pipeline | Connect stages with channels |
| Fan-out / Fan-in | Pattern for distributed processing and result aggregation |
| Debounce | Execute after a fixed time elapses since the last event |
| Throttle | Process the latest value at fixed intervals |
| Async cache | Concurrent-safe cache with TTL |

## Recommended Next Reading

- [Networking](./03-networking.md) -- Applying async patterns to HTTP/WebSocket/gRPC
- [Axum](./04-axum-web.md) -- Practice with a web framework
- [Concurrency](../03-systems/01-concurrency.md) -- Thread-level concurrency control

## References

1. **futures crate (StreamExt)**: https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html
2. **Tokio -- Streams**: https://tokio.rs/tokio/tutorial/streams
3. **Tower (middleware/retry/rate-limit)**: https://docs.rs/tower/latest/tower/
4. **async-stream crate**: https://docs.rs/async-stream/latest/async_stream/
5. **AWS Blog -- Exponential Backoff And Jitter**: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
