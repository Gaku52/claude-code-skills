# async/await Basics — Rust's Asynchronous Programming Model

> Understand the inner workings of Rust's async runtime centered on the Future trait and the basics of async/await syntax

## What You'll Learn in This Chapter

1. **How the Future trait works** — Poll-based lazy evaluation model and zero-cost abstraction
2. **async/await syntax** — Defining, calling, and composing async functions
3. **Role of the runtime** — Coordination between Executor, Reactor, and Waker
4. **Pin and Unpin** — Memory pinning that guarantees the safety of self-referential types
5. **Lifetimes and async** — Borrowing and ownership in async functions


## Prerequisites

The following knowledge will help you understand this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. Sync vs Async Big Picture

### 1.1 Basic Concept Diagram

```
┌──────────────────── Sync ───────────────────────────┐
│                                                      │
│  Thread 1: [Task A ██████████████████████████████]  │
│  Thread 2: [Task B ██████████████████████████████]  │
│  Thread 3: [Task C ██████████████████████████████]  │
│                                                      │
│  → Threads = concurrent tasks (10K conns = 10K thr) │
└──────────────────────────────────────────────────────┘

┌──────────────────── Async ───────────────────────────┐
│                                                      │
│  Thread 1: [A██][B██][A██][C████][B██][A██]         │
│  Thread 2: [C██][A██][B████][C██][B██]              │
│                                                      │
│  → Few threads handle many tasks                     │
│  → Run other tasks while waiting on I/O              │
└──────────────────────────────────────────────────────┘
```

### 1.2 Why Async Is Necessary

In the synchronous model, an entire thread is blocked while waiting for an I/O operation (network communication, disk access, etc.) to complete. Handling 10,000 concurrent connections would require 10,000 threads, causing memory consumption (about 8MB stack per thread = 80GB) and context switch overhead to balloon dramatically.

In the asynchronous model, threads can be used for other tasks while waiting on I/O, so a small number of threads (typically equal to the number of CPU cores) can efficiently handle tens to hundreds of thousands of concurrent connections.

```rust
// Sync model: thread pool size caps concurrent processing
fn sync_handler(stream: TcpStream) {
    // While this function blocks, the thread cannot do other work
    let data = read_from_db();       // ~10ms blocked
    let enriched = call_api(data);   // ~50ms blocked
    stream.write_all(&enriched);     // ~1ms blocked
}

// Async model: thread handles other tasks while waiting on I/O
async fn async_handler(stream: TcpStream) {
    // Suspends at .await and resumes upon completion
    let data = read_from_db().await;       // suspend → resume
    let enriched = call_api(data).await;   // suspend → resume
    stream.write_all(&enriched).await;     // suspend → resume
}
```

### 1.3 Characteristics of Rust's Async Model

Rust's async model has important characteristics that differ from other languages.

| Feature | Rust | Go | JavaScript | Python |
|---|---|---|---|---|
| Execution model | Zero-cost Future | Goroutine (with GC) | Event loop | Coroutines |
| Runtime | User chosen (tokio etc.) | Built into language | Built into V8 engine | asyncio standard |
| Scheduling | Cooperative | Preemptive | Cooperative | Cooperative |
| Memory allocation | On stack (zero-allocation possible) | Heap (Goroutine stack) | Heap (Promise) | Heap |
| Threading model | Multi-thread capable | M:N scheduling | Single-threaded | Single-threaded (GIL) |
| Type safety | Compile-time verified | Runtime panics possible | None | Type hints only |

Rust's async aims for "zero-cost abstraction"; the async/await syntax is converted into a state machine at compile time. Since the runtime is not built into the language, you can choose runtimes such as tokio, async-std, or smol depending on your use case.

---

## 2. The Heart of the Future Trait

### 2.1 Definition of the Future Trait

```rust
use std::pin::Pin;
use std::task::{Context, Poll};

// Future trait from the standard library (simplified)
pub trait Future {
    type Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

// Poll enum
pub enum Poll<T> {
    Ready(T),   // Done. Returns value T
    Pending,    // Not done. Will be re-notified via Waker
}
```

The `Future` trait is the most basic abstraction for asynchronous computation. Each time the `poll` method is called, the computation makes progress; if completed it returns `Ready(value)`, otherwise it returns `Pending`. When returning `Pending`, it registers the `Waker` from the `Context` and notifies the runtime once it is ready.

### 2.2 Manual Future Implementation

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// A custom Future that completes after a specified duration
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
            Poll::Ready("Time elapsed!".to_string())
        } else {
            // A real runtime would register a timer and call the Waker
            // Here we just request immediate re-polling (busy wait)
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[tokio::main]
async fn main() {
    let msg = Delay::new(Duration::from_secs(1)).await;
    println!("{}", msg); // "Time elapsed!"
}
```

### 2.3 Future as a State Machine

An async block is converted by the compiler into a state machine. Each `.await` point becomes a state transition.

```rust
// This async function:
async fn example() -> u64 {
    let a = step_one().await;    // suspend point 1
    let b = step_two(a).await;   // suspend point 2
    a + b
}

// The compiler conceptually generates a state machine like this:
enum ExampleFuture {
    // Initial state: waiting for step_one to complete
    State0 {
        step_one_future: StepOneFuture,
    },
    // After step_one completes: waiting for step_two to complete
    State1 {
        a: u64,
        step_two_future: StepTwoFuture,
    },
    // Done state
    Done,
}

impl Future for ExampleFuture {
    type Output = u64;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u64> {
        loop {
            match self.as_mut().get_mut() {
                ExampleFuture::State0 { step_one_future } => {
                    // Poll step_one
                    match Pin::new(step_one_future).poll(cx) {
                        Poll::Ready(a) => {
                            // Transition to next state
                            *self.as_mut().get_mut() = ExampleFuture::State1 {
                                a,
                                step_two_future: step_two(a),
                            };
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ExampleFuture::State1 { a, step_two_future } => {
                    match Pin::new(step_two_future).poll(cx) {
                        Poll::Ready(b) => {
                            let result = *a + b;
                            *self.as_mut().get_mut() = ExampleFuture::Done;
                            return Poll::Ready(result);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ExampleFuture::Done => panic!("polled after completion"),
            }
        }
    }
}
```

This transformation enables async processing without heap allocation. Each state is represented as an enum variant, with the necessary variables held as fields of the variant.

### 2.4 How the Waker Works

The `Waker` is the mechanism by which a Future notifies the runtime that it should be polled again.

```
┌──────────── Waker Lifecycle ───────────────────┐
│                                                  │
│  ① Executor calls Future::poll()                │
│     Passes Context containing Waker              │
│                                                  │
│  ② Future returns Pending                        │
│     → Registers Waker with the I/O driver       │
│     → Executor runs other tasks                  │
│                                                  │
│  ③ I/O event occurs (data arrives, etc.)        │
│     → Reactor calls Waker.wake()                 │
│                                                  │
│  ④ Executor re-queues the task                  │
│     → Re-runs poll() in next polling cycle      │
│                                                  │
│  ⑤ Future returns Ready(value)                   │
│     → Task complete                              │
└──────────────────────────────────────────────────┘
```

```rust
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Future that waits until a value is set
struct SharedState {
    completed: bool,
    value: Option<String>,
    waker: Option<Waker>,
}

struct WaitForValue {
    shared: Arc<Mutex<SharedState>>,
}

impl Future for WaitForValue {
    type Output = String;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<String> {
        let mut state = self.shared.lock().unwrap();

        if state.completed {
            // Done if the value has been set
            Poll::Ready(state.value.take().unwrap())
        } else {
            // Save the Waker so we can be notified when the value is set
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Externally set the value and trigger the Waker
fn set_value(shared: &Arc<Mutex<SharedState>>, value: String) {
    let mut state = shared.lock().unwrap();
    state.value = Some(value);
    state.completed = true;
    // Notify if a Waker is registered
    if let Some(waker) = state.waker.take() {
        waker.wake(); // Ask the Executor to re-poll the task
    }
}
```

### 2.5 Custom Future: Timer with Retry

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use pin_project::pin_project;

/// A Future that adds a timeout to an inner Future
#[pin_project]
struct WithTimeout<F> {
    #[pin]
    future: F,
    #[pin]
    delay: tokio::time::Sleep,
    timed_out: bool,
}

impl<F> WithTimeout<F> {
    fn new(future: F, timeout: std::time::Duration) -> Self {
        WithTimeout {
            future,
            delay: tokio::time::sleep(timeout),
            timed_out: false,
        }
    }
}

impl<F: Future> Future for WithTimeout<F> {
    type Output = Result<F::Output, &'static str>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        // First check the timeout
        if this.delay.poll(cx).is_ready() {
            return Poll::Ready(Err("Timeout"));
        }

        // Poll the wrapped Future
        match this.future.poll(cx) {
            Poll::Ready(value) => Poll::Ready(Ok(value)),
            Poll::Pending => Poll::Pending,
        }
    }
}

// Usage example
#[tokio::main]
async fn main() {
    let slow_operation = async {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        "Done"
    };

    let result = WithTimeout::new(
        slow_operation,
        std::time::Duration::from_secs(2),
    ).await;

    match result {
        Ok(value) => println!("Success: {}", value),
        Err(e) => println!("Error: {}", e), // "Error: Timeout"
    }
}
```

---

## 3. async/await Syntax

### 3.1 Basic async Functions

```rust
use tokio::time::{sleep, Duration};

/// async fn is syntactic sugar for a function that returns a Future
async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    // .await waits for the Future to complete
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}

/// The above is equivalent to (after desugaring):
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
        Ok(body) => println!("Fetched: {} bytes", body.len()),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### 3.2 Execution Flow of Async Operations

```
┌─────────────────────────────────────────────────┐
│        Inside async fn fetch_data()              │
│                                                   │
│  ① async fn invocation                           │
│     → Creates Future (state machine, not yet run)│
│                                                   │
│  ② .await                                        │
│     → Executor calls poll()                      │
│                                                   │
│  ③ Poll::Pending (I/O not complete)              │
│     → Register Waker and suspend the task        │
│     → Executor runs other tasks                  │
│                                                   │
│  ④ I/O completion notice (Reactor → Waker)       │
│     → Executor reschedules the task              │
│                                                   │
│  ⑤ Re-poll() → Poll::Ready(value)                │
│     → .await returns value                       │
└─────────────────────────────────────────────────┘
```

### 3.3 async Blocks and Closures

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // async block: like an anonymous async function
    let greeting = async {
        sleep(Duration::from_millis(100)).await;
        "Hello from async block!"
    };
    println!("{}", greeting.await);

    // async move block: moves ownership of captured variables
    let name = String::from("Rust");
    let greet = async move {
        // Ownership of `name` is moved into this block
        format!("Hello, {}!", name)
    };
    println!("{}", greet.await);
    // println!("{}", name); // Compile error: name has been moved

    // Async closures (nightly feature, or use async block as workaround)
    let urls = vec!["https://a.com", "https://b.com", "https://c.com"];
    let fetch_all = urls.iter().map(|url| {
        let url = url.to_string(); // Clone outside the closure
        async move {
            // reqwest::get(&url).await
            format!("fetched: {}", url)
        }
    });

    let results: Vec<String> = futures::future::join_all(fetch_all).await;
    for r in &results {
        println!("{}", r);
    }
}
```

### 3.4 Lifetimes in async Functions

When the arguments of an async function include lifetimes, they affect the lifetime of the returned Future.

```rust
// async function taking a reference
// The returned Future is bound to the argument's lifetime
async fn process_data(data: &[u8]) -> usize {
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    data.len()
}

// Desugared form of the above:
// fn process_data<'a>(data: &'a [u8]) -> impl Future<Output = usize> + 'a

#[tokio::main]
async fn main() {
    let data = vec![1, 2, 3, 4, 5];

    // OK: data lives until the .await completes
    let len = process_data(&data).await;
    println!("Length: {}", len);

    // NG: spawn requires 'static
    // tokio::spawn(process_data(&data)); // Compile error!

    // OK: move ownership to make it 'static
    let data_clone = data.clone();
    tokio::spawn(async move {
        let len = process_data(&data_clone).await;
        println!("Inside spawned task: length = {}", len);
    }).await.unwrap();
}
```

### 3.5 Recursive async Functions

async functions normally cannot be called recursively because the size of the state machine generated by the compiler would be infinite. We use `Box::pin` to work around this.

```rust
use std::pin::Pin;
use std::future::Future;

/// Example of asynchronously traversing a directory tree
async fn traverse_directory(path: std::path::PathBuf) -> Vec<String> {
    let mut results = Vec::new();

    if let Ok(mut entries) = tokio::fs::read_dir(&path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                // Recursive call: place Future on the heap with Box::pin
                let sub_results: Pin<Box<dyn Future<Output = Vec<String>> + Send>>
                    = Box::pin(traverse_directory(entry_path));
                results.extend(sub_results.await);
            } else {
                results.push(entry_path.display().to_string());
            }
        }
    }

    results
}

// More concise notation (using the async-recursion crate)
// #[async_recursion::async_recursion]
// async fn traverse_directory(path: PathBuf) -> Vec<String> {
//     // Normal recursive calls work
//     traverse_directory(sub_path).await;
// }
```

---

## 4. Composing Multiple Futures

### 4.1 join! and select!

```rust
use tokio::time::{sleep, Duration};

async fn task_a() -> String {
    sleep(Duration::from_millis(100)).await;
    "A done".to_string()
}

async fn task_b() -> String {
    sleep(Duration::from_millis(200)).await;
    "B done".to_string()
}

async fn task_c() -> String {
    sleep(Duration::from_millis(50)).await;
    "C done".to_string()
}

#[tokio::main]
async fn main() {
    // join! — Run all Futures concurrently and wait for all to complete
    let (a, b, c) = tokio::join!(task_a(), task_b(), task_c());
    println!("{}, {}, {}", a, b, c);
    // All complete at 200ms (matches the slowest, task_b)

    // select! — Get the result of the first Future to complete
    tokio::select! {
        val = task_a() => println!("A first: {}", val),
        val = task_b() => println!("B first: {}", val),
        val = task_c() => println!("C first: {}", val),
    }
    // "C first: C done" (fastest at 50ms)
}
```

### 4.2 Concurrent Processing with Error Handling

```rust
use anyhow::Result;

async fn fetch_user(id: u64) -> Result<String> {
    // Simulate API call
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    Ok(format!("User#{}", id))
}

async fn fetch_profile(id: u64) -> Result<String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
    Ok(format!("Profile#{}", id))
}

#[tokio::main]
async fn main() -> Result<()> {
    // try_join! — returns immediately if any one errors
    let (user, profile) = tokio::try_join!(
        fetch_user(1),
        fetch_profile(1),
    )?;

    println!("{}, {}", user, profile);

    // JoinSet — dynamically add tasks and wait for all to complete
    let mut set = tokio::task::JoinSet::new();
    for id in 1..=5 {
        set.spawn(fetch_user(id));
    }

    while let Some(result) = set.join_next().await {
        let user = result??;
        println!("Fetched: {}", user);
    }

    Ok(())
}
```

### 4.3 Extension Methods via FutureExt

```rust
use futures::FutureExt;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // fuse() — conversion to safely use with select!
    let mut future_a = Box::pin(sleep(Duration::from_millis(100)).fuse());
    let mut future_b = Box::pin(sleep(Duration::from_millis(200)).fuse());

    // Loop until both are done
    let mut a_done = false;
    let mut b_done = false;

    while !a_done || !b_done {
        tokio::select! {
            _ = &mut future_a, if !a_done => {
                println!("A done");
                a_done = true;
            }
            _ = &mut future_b, if !b_done => {
                println!("B done");
                b_done = true;
            }
        }
    }

    // map() — transform the result of a Future
    let result = async { 42 }
        .map(|x| x * 2)
        .await;
    println!("Result: {}", result); // 84

    // then() — generate a new Future from the result of one
    let result = async { 10 }
        .then(|x| async move {
            sleep(Duration::from_millis(10)).await;
            x + 5
        })
        .await;
    println!("Result: {}", result); // 15
}
```

### 4.4 join_all and try_join_all

```rust
use futures::future::{join_all, try_join_all};
use anyhow::Result;

async fn fetch_item(id: u32) -> Result<String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(id as u64 * 10)).await;
    if id == 5 {
        anyhow::bail!("Failed to fetch item 5");
    }
    Ok(format!("Item#{}", id))
}

#[tokio::main]
async fn main() -> Result<()> {
    // join_all — collect results of all Futures into a Vec (including errors)
    let futures: Vec<_> = (1..=10).map(|id| fetch_item(id)).collect();
    let results: Vec<Result<String>> = join_all(futures).await;

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(item) => println!("[{}] {}", i, item),
            Err(e) => eprintln!("[{}] Error: {}", i, e),
        }
    }

    // try_join_all — return immediately on the first error
    let futures: Vec<_> = (1..=3).map(|id| fetch_item(id)).collect();
    let results: Vec<String> = try_join_all(futures).await?;
    println!("All succeeded: {:?}", results);

    Ok(())
}
```

### 4.5 Advanced Use of select!

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, interval};

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(32);
    let mut heartbeat = interval(Duration::from_secs(5));
    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    // Sender side
    tokio::spawn(async move {
        for i in 0..10 {
            sleep(Duration::from_millis(500)).await;
            let _ = tx.send(format!("Message #{}", i)).await;
        }
    });

    loop {
        tokio::select! {
            // Specifying `biased;` checks branches in order from top to bottom
            biased;

            // Shutdown signal (highest priority)
            _ = &mut shutdown => {
                println!("Ctrl+C received. Shutting down...");
                break;
            }

            // Message received
            Some(msg) = rx.recv() => {
                println!("Received: {}", msg);
            }

            // Heartbeat
            _ = heartbeat.tick() => {
                println!("Heartbeat sent");
            }

            // When all channels are closed
            else => {
                println!("All channels closed. Exiting.");
                break;
            }
        }
    }
}
```

---

## 5. Pin and Unpin

### 5.1 Why Pin Is Needed

```
┌─────────────────────────────────────────────┐
│              Why Pin Is Needed                │
│                                               │
│  async fn foo() {                            │
│      let data = vec![1, 2, 3];               │
│      let ref_to_data = &data;  ← self-ref    │
│      some_async_op().await;    ← suspend pt  │
│      println!("{:?}", ref_to_data);           │
│  }                                            │
│                                               │
│  If the Future moves in memory while         │
│  suspended, ref_to_data becomes invalid.     │
│  Pin prevents this.                           │
│                                               │
│  Pin<&mut T>:                                │
│    If T: Unpin → can move (most types)       │
│    If T: !Unpin → cannot move (async output) │
└─────────────────────────────────────────────┘
```

### 5.2 Practical Use of Pin

```rust
use std::pin::Pin;
use std::future::Future;
use tokio::time::{sleep, Duration};

// pin! macro (tokio::pin! or std::pin::pin!) for stack pinning
#[tokio::main]
async fn main() {
    // Pin on the stack
    let future = sleep(Duration::from_millis(100));
    tokio::pin!(future);

    // Now usable as Pin<&mut Sleep>
    // Required when you need to use it as &mut inside select!
    tokio::select! {
        _ = &mut future => {
            println!("Sleep done");
        }
    }

    // Box::pin pins on the heap
    let boxed_future: Pin<Box<dyn Future<Output = ()>>> =
        Box::pin(async {
            sleep(Duration::from_millis(100)).await;
            println!("Boxed Future done");
        });
    boxed_future.await;
}

// When dynamic dispatch is required (trait object)
async fn execute_any(future: Pin<Box<dyn Future<Output = String> + Send>>) -> String {
    future.await
}
```

### 5.3 Understanding Unpin

```rust
use std::marker::Unpin;
use std::pin::Pin;

// Most types automatically implement Unpin
// → Even with Pin<&mut T>, they can be moved freely
struct MyStruct {
    value: i32,
}
// MyStruct: Unpin (auto-implemented)

// Futures generated by async blocks/functions are !Unpin
// → Cannot be moved while pinned
fn takes_unpin<T: Unpin>(_: &T) {}

fn example() {
    let x = MyStruct { value: 42 };
    takes_unpin(&x); // OK: MyStruct is Unpin

    // let future = async { 42 };
    // takes_unpin(&future); // Compile error: async is !Unpin

    // Manually implementing Unpin (usually unnecessary):
    // For custom Futures that contain no self-references
    struct SimpleFuture;
    impl Unpin for SimpleFuture {}
}
```

---

## 6. Executor / Reactor / Waker Architecture

### 6.1 Cooperation of the Three Components

```
┌──────────────── Async Runtime Structure ───────────────┐
│                                                           │
│  ┌─────────────┐     ┌─────────────┐                    │
│  │  Executor    │     │  Reactor    │                    │
│  │ (run tasks)  │     │ (watch I/O) │                    │
│  │             │     │             │                    │
│  │ task queue  │     │  epoll /    │                    │
│  │  ┌───┐┌───┐│     │  kqueue /   │                    │
│  │  │T1 ││T2 ││     │  IOCP       │                    │
│  │  └───┘└───┘│     │             │                    │
│  │  ┌───┐┌───┐│     │  ┌────────┐ │                    │
│  │  │T3 ││T4 ││     │  │sockets │ │                    │
│  │  └───┘└───┘│     │  │ timers │ │                    │
│  └──────┬──────┘     │  │ files  │ │                    │
│         │            │  └────────┘ │                    │
│         │ poll()     └──────┬──────┘                    │
│         │                   │                            │
│         │    ┌──────────────┘                            │
│         │    │ wake()                                    │
│         ▼    ▼                                           │
│  ┌─────────────────┐                                    │
│  │     Waker       │                                    │
│  │ (notification)  │                                    │
│  │                 │                                    │
│  │  Reactor → Waker.wake()                              │
│  │        → Executor re-queues the task                 │
│  └─────────────────┘                                    │
└───────────────────────────────────────────────────────────┘
```

### 6.2 Implementing a Mini Executor

```rust
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

/// Minimal Executor implementation
struct MiniExecutor {
    queue: VecDeque<Pin<Box<dyn Future<Output = ()>>>>,
}

/// Waker used by tasks to request re-polling
struct MiniWaker;

impl Wake for MiniWaker {
    fn wake(self: Arc<Self>) {
        // A real runtime would re-queue the task here
        // This minimal version does nothing (busy polling)
    }
}

impl MiniExecutor {
    fn new() -> Self {
        MiniExecutor {
            queue: VecDeque::new(),
        }
    }

    fn spawn(&mut self, future: impl Future<Output = ()> + 'static) {
        self.queue.push_back(Box::pin(future));
    }

    fn run(&mut self) {
        let waker = Waker::from(Arc::new(MiniWaker));
        let mut cx = Context::from_waker(&waker);

        while let Some(mut future) = self.queue.pop_front() {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(()) => {
                    // Task done
                }
                Poll::Pending => {
                    // Not done: push back to the end of the queue (busy polling)
                    self.queue.push_back(future);
                }
            }
        }
    }
}

// Usage example (works without tokio)
fn main() {
    let mut executor = MiniExecutor::new();

    executor.spawn(async {
        println!("Task 1 start");
        // Note: real async I/O requires runtime support
        println!("Task 1 done");
    });

    executor.spawn(async {
        println!("Task 2 start");
        println!("Task 2 done");
    });

    executor.run();
}
```

---

## 7. Async Error Handling Patterns

### 7.1 Result and the ? Operator

```rust
use anyhow::{Context, Result};
use tokio::time::{sleep, Duration};

#[derive(Debug)]
struct ApiResponse {
    status: u16,
    body: String,
}

async fn fetch_api(url: &str) -> Result<ApiResponse> {
    let response = reqwest::get(url)
        .await
        .context(format!("HTTP request failed: {}", url))?;

    let status = response.status().as_u16();
    let body = response.text()
        .await
        .context("Failed to read response body")?;

    Ok(ApiResponse { status, body })
}

async fn fetch_with_fallback(primary: &str, fallback: &str) -> Result<String> {
    // Try primary, fall back if it fails
    match fetch_api(primary).await {
        Ok(resp) if resp.status == 200 => Ok(resp.body),
        Ok(resp) => {
            eprintln!("Primary returned status {}. Switching to fallback", resp.status);
            let resp = fetch_api(fallback).await?;
            Ok(resp.body)
        }
        Err(e) => {
            eprintln!("Primary error: {}. Switching to fallback", e);
            let resp = fetch_api(fallback).await?;
            Ok(resp.body)
        }
    }
}
```

### 7.2 Custom Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum ServiceError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Timeout: exceeded {0} seconds")]
    Timeout(u64),

    #[error("Rate limit: retry after {retry_after} seconds")]
    RateLimit { retry_after: u64 },

    #[error("Auth failed: {0}")]
    Auth(String),

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

async fn call_service(url: &str, token: &str) -> Result<String, ServiceError> {
    let client = reqwest::Client::new();
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        client.get(url).bearer_auth(token).send(),
    )
    .await
    .map_err(|_| ServiceError::Timeout(30))?
    .map_err(ServiceError::Network)?;

    match response.status().as_u16() {
        200 => Ok(response.text().await.map_err(ServiceError::Network)?),
        401 => Err(ServiceError::Auth("Invalid token".into())),
        429 => {
            let retry = response
                .headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(60);
            Err(ServiceError::RateLimit { retry_after: retry })
        }
        _ => Err(ServiceError::Internal(
            anyhow::anyhow!("Unexpected status: {}", response.status()),
        )),
    }
}
```

---

## 8. Comparison Tables

### 8.1 Runtime Comparison

| Item | tokio | async-std | smol |
|---|---|---|---|
| Ecosystem | Largest | Medium | Lightweight |
| Multi-threaded | Default | Default | Supported |
| I/O | Custom (mio-based) | Custom | polling-based |
| Timers | `tokio::time` | `async_std::task` | `async-io` |
| Adoption | Axum, tonic, etc. | Some | Embedded-oriented |
| Dependency size | Medium | Medium | Small |
| Work stealing | Yes | Yes | Yes |
| Custom runtime construction | Builder API | Limited | Easy |

### 8.2 Sync vs Async Selection Criteria

| Criterion | Sync is appropriate | Async is appropriate |
|---|---|---|
| I/O pattern | CPU-bound work | I/O-bound work |
| Concurrent connections | Few (~100) | Many (1K~100K+) |
| Latency requirements | Predictability prioritized | Throughput prioritized |
| Code complexity | Simplicity prioritized | Some complexity acceptable |
| Libraries | Sync-only API | Leveraging async ecosystem |
| Debugging | Clear stack traces | Async-aware tooling required |
| Memory usage | Thread stack (8MB/thread) | Future (hundreds of bytes/task) |
| Startup cost | Thread creation (~ms) | Task spawn (~μs) |

### 8.3 Comparison of Future Composition Methods

| Method | Use case | Behavior | On error |
|---|---|---|---|
| `join!` | Concurrent · wait for all | Wait until all Futures complete | Check each individually after completion |
| `try_join!` | Concurrent · abort on first error | Returns immediately if any errors | Others dropped (cancelled) |
| `select!` | Concurrent · first completion | Get result of the first to complete | Others dropped (cancelled) |
| `join_all` | Dynamic count · wait for all | Take Vec<Future> | Returns Vec<Result> |
| `try_join_all` | Dynamic count · abort on first error | Take Vec<Future> | Aborts on first error |
| `FuturesUnordered` | Dynamic count · in completion order | Returns results as a Stream | Handle individually |

---

## 9. Anti-Patterns

### 9.1 Blocking Calls Inside async

```rust
// NG: std::thread::sleep inside async (blocks the entire runtime!)
async fn bad_delay() {
    std::thread::sleep(std::time::Duration::from_secs(5));
    // All other tasks are stalled for 5 seconds
}

// OK: use tokio's async sleep
async fn good_delay() {
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    // Other tasks continue running while suspended
}

// OK: offload CPU-intensive work with spawn_blocking
async fn cpu_heavy() {
    let result = tokio::task::spawn_blocking(|| {
        // Heavy synchronous computation
        (0..10_000_000u64).sum::<u64>()
    }).await.unwrap();
    println!("Result: {}", result);
}

// OK: wrap blocking libraries with spawn_blocking
async fn read_file_blocking(path: String) -> std::io::Result<String> {
    tokio::task::spawn_blocking(move || {
        std::fs::read_to_string(path)
    }).await.unwrap()
}
```

### 9.2 Forgetting to .await a Future

```rust
// NG: ignoring the return value of an async fn (nothing executes!)
async fn send_notification() {
    println!("Notification sent");
}

async fn bad_example() {
    send_notification(); // ← no .await! does not run!
    // Compiler warning: unused future
}

// OK: always .await or spawn
async fn good_example() {
    send_notification().await;           // Pattern 1: wait synchronously
    tokio::spawn(send_notification());   // Pattern 2: run in background
}
```

### 9.3 Unnecessary Use of Arc/Mutex

```rust
// NG: using std::sync::Mutex in an async context
use std::sync::Mutex;

async fn bad_shared_state() {
    let data = std::sync::Arc::new(Mutex::new(Vec::new()));
    let d = data.clone();
    tokio::spawn(async move {
        let mut lock = d.lock().unwrap(); // Risk of holding lock across .await
        // Long async work...
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await; // ← deadlock cause!
        lock.push(42);
    });
}

// OK: use tokio::sync::Mutex (safe to hold the lock across .await)
async fn good_shared_state() {
    let data = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let d = data.clone();
    tokio::spawn(async move {
        let mut lock = d.lock().await; // async lock acquisition
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        lock.push(42);
    });
}

// Best: minimize lock granularity
async fn best_shared_state() {
    let data = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let d = data.clone();
    tokio::spawn(async move {
        // Async work
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // Hold the lock for the minimum scope
        {
            let mut lock = d.lock().unwrap();
            lock.push(42);
        } // ← lock released immediately here
    });
}
```

### 9.4 Allocating Large Memory Inside an async Function

```rust
// NG: the async function's stack frame (state machine) becomes huge
async fn bad_large_stack() {
    let buffer = [0u8; 1_000_000]; // 1MB array embedded in the Future state
    some_async_op().await;
    println!("{}", buffer.len());
}

// OK: place on the heap with Box
async fn good_large_heap() {
    let buffer = vec![0u8; 1_000_000]; // placed on the heap
    some_async_op().await;
    println!("{}", buffer.len());
}

async fn some_async_op() {
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
}
```

---

## 10. Practical Pattern Catalog

### 10.1 Graceful Shutdown

```rust
use tokio::sync::watch;
use tokio::time::{sleep, Duration};

async fn graceful_shutdown_example() {
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    // Worker task group
    let mut handles = Vec::new();
    for i in 0..4 {
        let mut rx = shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = rx.changed() => {
                        if *rx.borrow() {
                            println!("Worker {} shutting down...", i);
                            // Cleanup work
                            sleep(Duration::from_millis(100)).await;
                            println!("Worker {} stopped", i);
                            return;
                        }
                    }
                    _ = sleep(Duration::from_secs(1)) => {
                        println!("Worker {} working...", i);
                    }
                }
            }
        });
        handles.push(handle);
    }

    // Shutdown after 3 seconds
    sleep(Duration::from_secs(3)).await;
    println!("Sending shutdown signal");
    let _ = shutdown_tx.send(true);

    // Wait for all workers to finish
    for handle in handles {
        let _ = handle.await;
    }
    println!("All workers stopped. Program exiting.");
}
```

### 10.2 Async Iteration (for-await-style Pattern)

```rust
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

async fn async_iteration_example() {
    let (tx, rx) = mpsc::channel::<i32>(32);

    // Producer
    tokio::spawn(async move {
        for i in 0..20 {
            let _ = tx.send(i).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    });

    // Consumer: process as a Stream
    let stream = ReceiverStream::new(rx);
    let results: Vec<i32> = stream
        .filter(|x| *x % 2 == 0)       // even numbers only
        .map(|x| x * x)                 // square
        .take(5)                         // first 5
        .collect()
        .await;

    println!("Result: {:?}", results); // [0, 4, 16, 36, 64]
}
```

### 10.3 CancellationToken Pattern

```rust
use tokio_util::sync::CancellationToken;
use tokio::time::{sleep, Duration};

async fn cancellation_token_example() {
    let token = CancellationToken::new();

    // Child token: automatically cancelled when the parent is cancelled
    let child_token = token.child_token();

    let task = tokio::spawn({
        let token = child_token.clone();
        async move {
            loop {
                tokio::select! {
                    _ = token.cancelled() => {
                        println!("Cancelled. Cleaning up...");
                        // Release resources, etc.
                        break;
                    }
                    _ = sleep(Duration::from_secs(1)) => {
                        println!("Working...");
                    }
                }
            }
        }
    });

    // Cancel after 3 seconds
    sleep(Duration::from_secs(3)).await;
    token.cancel();
    let _ = task.await;
    println!("Task was cancelled successfully");
}
```

### 10.4 Async Resource Management (Drop and Async)

```rust
/// Pattern for managing resources that require async cleanup
struct AsyncResource {
    name: String,
    // Signal for async cleanup
    cleanup_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl AsyncResource {
    async fn new(name: &str) -> Self {
        println!("Creating resource '{}'", name);
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Wait for cleanup in the background
        let resource_name = name.to_string();
        tokio::spawn(async move {
            let _ = rx.await;
            // Async cleanup work
            println!("Async cleanup of resource '{}' complete", resource_name);
        });

        AsyncResource {
            name: name.to_string(),
            cleanup_tx: Some(tx),
        }
    }

    async fn use_resource(&self) {
        println!("Using resource '{}'", self.name);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

impl Drop for AsyncResource {
    fn drop(&mut self) {
        // Drop is synchronous, so signal the background task to handle cleanup
        if let Some(tx) = self.cleanup_tx.take() {
            let _ = tx.send(());
        }
        println!("Dropping resource '{}' (sync part)", self.name);
    }
}
```

---

## FAQ

### Q1: What does `#[tokio::main]` do?

**A:** It is a macro that starts an async runtime (Executor) and runs the `async` block in `main`.

```rust
// This:
#[tokio::main]
async fn main() { /* ... */ }

// Is equivalent to:
fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { /* ... */ })
}
```

### Q2: Why are `Send` + `'static` constraints required?

**A:** Because `tokio::spawn` may run the task on another thread, it requires `Send` (movable between threads) and `'static` (no borrowed references).

```rust
// Error: a Future containing a local reference cannot be spawned
async fn bad() {
    let data = String::from("hello");
    let r = &data;
    tokio::spawn(async move {
        // println!("{}", r); // ← Compile error: &String is not 'static
    });
}

// OK: move ownership
async fn good() {
    let data = String::from("hello");
    tokio::spawn(async move {
        println!("{}", data); // OK: ownership of data is moved
    });
}
```

### Q3: How do I write an async trait?

**A:** From Rust 1.75+, you can use `async fn` directly inside traits. Before that, use the `async-trait` crate.

```rust
// Rust 1.75+ (native support)
trait Service {
    async fn call(&self, req: Request) -> Response;
}

// Note: async fn in traits is not Send by default
// If a Send bound is needed, declare it explicitly:
trait SendService: Send + Sync {
    fn call(&self, req: Request) -> impl Future<Output = Response> + Send;
}

// Rust 1.74 and earlier (async-trait crate)
use async_trait::async_trait;

#[async_trait]
trait Service {
    async fn call(&self, req: Request) -> Response;
}
```

### Q4: When should I use `tokio::spawn` vs `tokio::join!`?

**A:** `join!` runs multiple Futures concurrently within the current task. `spawn` creates a new task that runs detached.

```rust
// join!: concurrent, but within the same task. Easy to cancel
let (a, b) = tokio::join!(future_a, future_b);

// spawn: independent task. Requires Send + 'static
let handle = tokio::spawn(future_a);
// Get the result via JoinHandle, or detach
```

**Selection guide:**
- Short concurrent work where you need the result → `join!`
- Independent background tasks → `spawn`
- Lifetime constraints (containing references) → `join!`
- Dynamic number of tasks → `JoinSet` (spawn-based)

### Q5: How do I debug async code?

**A:** Make use of the following tools and techniques.

```rust
// 1. tokio-console (tracing-based debugging tool)
// Cargo.toml:
// [dependencies]
// console-subscriber = "0.4"

#[tokio::main]
async fn main() {
    console_subscriber::init(); // Enable tokio-console
    // ... application code
}

// 2. Logging with tracing
use tracing::{info, instrument};

#[instrument] // Automatically traces function calls
async fn process_request(id: u64) -> String {
    info!("Request processing start");
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    info!("Request processing done");
    format!("Result: {}", id)
}

// 3. Task dump (tokio's unstable feature)
// RUSTFLAGS="--cfg tokio_unstable" cargo run
// enables tokio::runtime::Handle::dump()
```

### Q6: When does calling synchronous code inside async fn become a problem?

**A:** It becomes a problem when the synchronous code "blocks for a long time." Short synchronous work on the order of a microsecond is fine. As a rule of thumb, consider `spawn_blocking` for work that takes 10 to 100μs or more.

```rust
// No problem: short synchronous work
async fn ok_example() {
    let hash = sha256(&data); // microsecond order
    // ...
}

// Problematic: long synchronous work
async fn bad_example() {
    let compressed = zstd::compress(&large_data, 19); // millisecond–second order
    // → should be offloaded with spawn_blocking
}
```

---

## Summary

| Item | Key Point |
|---|---|
| Future trait | A lazy-evaluation model where `poll()` returns `Ready` or `Pending` |
| async/await | Syntactic sugar for creating and awaiting Futures |
| State machine | async blocks are converted into state machines at compile time (zero cost) |
| Runtime | Executor (run tasks) + Reactor (watch I/O) + Waker (notification) |
| tokio | The most widely used async runtime |
| join! | Concurrent execution of multiple Futures, waiting for all to complete |
| try_join! | Concurrent execution, aborting on the first error |
| select! | Get the first completion among multiple Futures |
| Pin | Guarantees memory safety for Futures containing self-references |
| Unpin | Marker trait for movable types (regular types implement it automatically) |
| spawn_blocking | Offload synchronous work to the blocking thread pool |
| CancellationToken | Recommended pattern for cooperative task cancellation |
| Send + 'static | Required for spawn. Use join! when references are involved |
| Error handling | The combination of thiserror + anyhow is practical |

## Recommended Next Guides

- [Tokio Runtime](./01-tokio-runtime.md) — Details on task management and channels
- [Async Patterns](./02-async-patterns.md) — Streams, concurrency limits, retries
- [Networking](./03-networking.md) — HTTP/WebSocket/gRPC

## References

1. **Asynchronous Programming in Rust**: https://rust-lang.github.io/async-book/
2. **Tokio Tutorial**: https://tokio.rs/tokio/tutorial
3. **Pin and suffering (Fasterthanlime)**: https://fasterthanli.me/articles/pin-and-suffering
4. **Jon Gjengset - Decrusting the tokio crate**: https://www.youtube.com/watch?v=o2ob8zkeq2s
5. **Without Boats - Zero-cost async IO**: https://without.boats/blog/zero-cost-async-io/
6. **Tokio Console**: https://github.com/tokio-rs/console
