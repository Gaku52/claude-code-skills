# Tokio Runtime — Task Management and Channels

> Master the internals of Tokio's multi-threaded runtime, task spawning, and asynchronous message passing using channels.

## What You'll Learn in This Chapter

1. **Runtime configuration** — Choosing and configuring multi-threaded vs. single-threaded runtimes
2. **Task management** — spawn, JoinSet, abort, task-local storage
3. **Asynchronous channels** — When to use mpsc, oneshot, broadcast, and watch
4. **Synchronization primitives** — Mutex, RwLock, Semaphore, Notify, Barrier
5. **Task-local storage** — Per-task context propagation with task_local!


## Prerequisites

Reading the following beforehand will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Content of [async/await Basics — Rust's Asynchronous Programming Model](./00-async-basics.md)

---

## 1. Tokio Runtime Architecture

### 1.1 Overall Structure

```
┌────────────────────── Tokio Runtime ──────────────────────┐
│                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Worker      │  │ Worker      │  │ Worker      │      │
│  │ Thread #1   │  │ Thread #2   │  │ Thread #N   │      │
│  │             │  │             │  │             │      │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │      │
│  │ │ Local Q │ │  │ │ Local Q │ │  │ │ Local Q │ │      │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │              │
│  ┌──────┴────────────────┴────────────────┴──────┐      │
│  │              Global Task Queue                  │      │
│  │          (Work-Stealing Scheduler)             │      │
│  └─────────────────────┬─────────────────────────┘      │
│                        │                                  │
│  ┌─────────────────────┴─────────────────────────┐      │
│  │           I/O Driver (mio/epoll/kqueue)        │      │
│  │           Timer Driver                          │      │
│  └────────────────────────────────────────────────┘      │
│                                                            │
│  ┌────────────────────────────────────────────────┐      │
│  │        Blocking Thread Pool (spawn_blocking)    │      │
│  └────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────┘
```

### 1.2 How the Work-Stealing Scheduler Works

```
┌──────────── Work-Stealing Scheduler ──────────────┐
│                                                      │
│  Worker #1          Worker #2          Worker #3    │
│  ┌──────┐          ┌──────┐          ┌──────┐    │
│  │ T1   │          │ T4   │          │      │    │
│  │ T2   │          │ T5   │          │(empty)│   │
│  │ T3   │          │      │          │      │    │
│  └──────┘          └──────┘          └──────┘    │
│      │                                    ▲        │
│      │           Work-stealing             │        │
│      └────────────────────────────────────┘        │
│      Worker #3 steals a task from Worker #1        │
│                                                      │
│  Benefits:                                           │
│  - Automatic load balancing                          │
│  - Minimization of idle threads                      │
│  - Cache locality preserved (local queue first)      │
└──────────────────────────────────────────────────────┘
```

### 1.3 Role of the I/O Driver

The I/O Driver abstracts the OS event-notification mechanism (Linux: epoll, macOS: kqueue, Windows: IOCP) and bridges asynchronous I/O events to the Tokio runtime.

```
┌─────────── I/O Driver Operation Flow ──────────┐
│                                                  │
│  (1) A task requests a socket read              │
│     → Register interest with I/O Driver         │
│       (epoll_ctl)                               │
│     → Task returns Pending and is suspended     │
│                                                  │
│  (2) I/O Driver monitors events                 │
│     → Blocks on epoll_wait                      │
│       (other tasks keep running)                │
│                                                  │
│  (3) Data arrives                               │
│     → epoll_wait returns                        │
│     → The corresponding Waker is called         │
│                                                  │
│  (4) Task is rescheduled                        │
│     → A worker thread polls() it again          │
│     → It returns Ready(data)                    │
└──────────────────────────────────────────────────┘
```

---

## 2. Building and Configuring the Runtime

### Code Example 1: Choosing a Runtime Configuration

```rust
// Pattern 1: Simple setup via macro (multi-threaded)
#[tokio::main]
async fn main() {
    println!("Multi-threaded runtime");
}

// Pattern 2: Single-threaded (for tests and lightweight use cases)
#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("Single-threaded runtime");
}

// Pattern 3: Specify the number of worker threads
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    println!("4-thread runtime");
}

// Pattern 4: Manual build (fine-grained control)
fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)                       // Number of worker threads
        .max_blocking_threads(64)                // Upper bound on blocking threads
        .thread_name("my-worker")                // Thread name
        .thread_stack_size(3 * 1024 * 1024)     // Stack size: 3MB
        .enable_all()                             // Enable I/O + Timer drivers
        .on_thread_start(|| {
            println!("Thread start: {:?}", std::thread::current().id());
        })
        .on_thread_stop(|| {
            println!("Thread stop: {:?}", std::thread::current().id());
        })
        .build()
        .unwrap();

    runtime.block_on(async {
        println!("Manually built runtime");
    });
}

// Pattern 5: Manual build of a single-threaded runtime
fn main_single() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        println!("Single-threaded runtime (manual)");
    });
}
```

### Code Example 2: Using Multiple Runtimes Selectively

```rust
use tokio::runtime::Runtime;

/// Separate runtimes for CPU-bound work and I/O-bound work
struct AppRuntime {
    io_runtime: Runtime,
    cpu_runtime: Runtime,
}

impl AppRuntime {
    fn new() -> Self {
        let io_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("io-worker")
            .enable_all()
            .build()
            .unwrap();

        let cpu_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .thread_name("cpu-worker")
            .enable_all()
            .build()
            .unwrap();

        AppRuntime { io_runtime, cpu_runtime }
    }

    /// Run an I/O-bound task
    fn spawn_io<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.io_runtime.spawn(future)
    }

    /// Run a CPU-bound task on the blocking pool
    fn spawn_cpu<F, R>(&self, f: F) -> tokio::task::JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.cpu_runtime.spawn_blocking(f)
    }
}
```

### Code Example 3: Monitoring Runtime Metrics

```rust
use tokio::runtime::Handle;

#[tokio::main]
async fn main() {
    let handle = Handle::current();

    // Obtain runtime metrics (requires tokio_unstable)
    // RUSTFLAGS="--cfg tokio_unstable" cargo run
    #[cfg(tokio_unstable)]
    {
        let metrics = handle.metrics();
        println!("Worker threads: {}", metrics.num_workers());
        println!("Active tasks: {}", metrics.active_tasks_count());
        println!("Blocking threads: {}", metrics.num_blocking_threads());

        // Periodically emit metrics
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                let m = Handle::current().metrics();
                println!(
                    "[metrics] active_tasks={}, blocking_threads={}",
                    m.active_tasks_count(),
                    m.num_blocking_threads(),
                );
            }
        });
    }

    // Main application logic
    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
}
```

---

## 3. Task Management

### Code Example 4: spawn and JoinHandle

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // spawn — run a new task asynchronously
    let handle = tokio::spawn(async {
        sleep(Duration::from_millis(100)).await;
        42
    });

    // Retrieve the result via JoinHandle
    let result = handle.await.unwrap();
    println!("Result: {}", result); // Result: 42

    // Tasks created with spawn start executing immediately.
    // .await is for "retrieving" the result, not "starting" it.
    let h1 = tokio::spawn(async { sleep(Duration::from_secs(1)).await; "A" });
    let h2 = tokio::spawn(async { sleep(Duration::from_secs(1)).await; "B" });

    // h1 and h2 run concurrently → about 1 second total
    let (a, b) = (h1.await.unwrap(), h2.await.unwrap());
    println!("{}, {}", a, b);

    // Use is_finished() on JoinHandle to check completion (non-blocking)
    let h3 = tokio::spawn(async {
        sleep(Duration::from_secs(2)).await;
        "done"
    });

    // Polling-style completion check
    for _ in 0..5 {
        if h3.is_finished() {
            println!("Task complete!");
            break;
        }
        println!("Still running...");
        sleep(Duration::from_millis(500)).await;
    }
    let result = h3.await.unwrap();
    println!("Result: {}", result);
}
```

### Code Example 5: Dynamic Task Management with JoinSet

```rust
use tokio::task::JoinSet;
use tokio::time::{sleep, Duration};

async fn process_item(id: u32) -> String {
    sleep(Duration::from_millis(50 * id as u64)).await;
    format!("Item#{} processing complete", id)
}

#[tokio::main]
async fn main() {
    let mut set = JoinSet::new();

    // Add tasks dynamically
    for id in 1..=10 {
        set.spawn(process_item(id));
    }

    // Retrieve results in completion order
    while let Some(result) = set.join_next().await {
        match result {
            Ok(msg) => println!("{}", msg),
            Err(e) => eprintln!("Task error: {}", e),
        }
    }

    // Cancel all tasks with abort_all
    let mut set2 = JoinSet::new();
    for i in 0..5 {
        set2.spawn(async move {
            sleep(Duration::from_secs(10)).await;
            i
        });
    }
    set2.abort_all(); // Cancel all tasks immediately

    // Use len() on JoinSet to check the active task count
    println!("Remaining tasks: {}", set2.len());
}
```

### Code Example 6: Task Execution with Concurrency Limit Using JoinSet

```rust
use tokio::task::JoinSet;
use tokio::time::{sleep, Duration};

/// Run at most max_concurrent tasks at once
async fn process_with_limit(items: Vec<u32>, max_concurrent: usize) -> Vec<String> {
    let mut set = JoinSet::new();
    let mut results = Vec::new();
    let mut iter = items.into_iter();

    // Initially submit max_concurrent tasks
    for _ in 0..max_concurrent {
        if let Some(item) = iter.next() {
            set.spawn(async move {
                sleep(Duration::from_millis(100)).await;
                format!("Item#{} processing complete", item)
            });
        }
    }

    // Submit the next task each time one completes
    while let Some(result) = set.join_next().await {
        match result {
            Ok(msg) => results.push(msg),
            Err(e) => eprintln!("Error: {}", e),
        }

        // Add the next task if there are remaining items
        if let Some(item) = iter.next() {
            set.spawn(async move {
                sleep(Duration::from_millis(100)).await;
                format!("Item#{} processing complete", item)
            });
        }
    }

    results
}

#[tokio::main]
async fn main() {
    let items: Vec<u32> = (1..=20).collect();
    let results = process_with_limit(items, 5).await; // up to 5 concurrent
    println!("Processing complete: {} items", results.len());
}
```

### How Task Cancellation Works

```
┌────────────── Task Lifecycle ──────────────────────┐
│                                                     │
│  spawn()                                            │
│    │                                                │
│    ▼                                                │
│  Running ──── .await on poll ────→ Suspended        │
│    │              │                    │            │
│    │              │ I/O Ready          │            │
│    │              ◄────────────────────┘            │
│    │                                                │
│    ├── abort() ──→ Cancelled (JoinError)           │
│    │                                                │
│    └── normal completion ──→ Completed (Ok(T))      │
│                                                     │
│  drop(JoinHandle):                                  │
│    Task continues (detached). Result unobtainable.  │
│                                                     │
│  When abort() is called:                            │
│    - Cancellation occurs at the next .await point   │
│    - In-flight synchronous code runs to completion  │
│    - Drop is run correctly (RAII-safe)              │
└─────────────────────────────────────────────────────┘
```

### Code Example 7: Task Cancellation and Cleanup

```rust
use tokio::time::{sleep, Duration};

struct Resource {
    name: String,
}

impl Resource {
    fn new(name: &str) -> Self {
        println!("  [{}] Resource created", name);
        Resource { name: name.to_string() }
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        println!("  [{}] Resource released (Drop)", self.name);
    }
}

#[tokio::main]
async fn main() {
    // Cancellation via abort
    let handle = tokio::spawn(async {
        let _resource = Resource::new("task-resource");
        println!("  Task started. Long operation...");
        sleep(Duration::from_secs(60)).await;
        println!("  Task complete (this line is never executed)");
    });

    // Cancel after 500ms
    sleep(Duration::from_millis(500)).await;
    handle.abort();

    match handle.await {
        Ok(_) => println!("Task completed normally"),
        Err(e) if e.is_cancelled() => println!("Task was cancelled"),
        Err(e) => println!("Task panicked: {}", e),
    }
    // Output:
    //   [task-resource] Resource created
    //   Task started. Long operation...
    //   [task-resource] Resource released (Drop)
    //   Task was cancelled
}
```

### Code Example 8: Task-Local Storage

```rust
use tokio::task_local;

task_local! {
    static REQUEST_ID: String;
    static USER_ID: u64;
}

async fn handle_request() {
    let req_id = REQUEST_ID.with(|id| id.clone());
    let user_id = USER_ID.with(|id| *id);
    println!("Request {} (user {}): processing...", req_id, user_id);

    // Accessible from sub-functions as well
    process_data().await;
}

async fn process_data() {
    let req_id = REQUEST_ID.with(|id| id.clone());
    println!("  [{}] processing data...", req_id);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    println!("  [{}] data processing complete", req_id);
}

#[tokio::main]
async fn main() {
    // Run with scoped task_local! variables
    let handle1 = tokio::spawn(
        REQUEST_ID.scope("req-001".to_string(),
            USER_ID.scope(42,
                handle_request()
            )
        )
    );

    let handle2 = tokio::spawn(
        REQUEST_ID.scope("req-002".to_string(),
            USER_ID.scope(99,
                handle_request()
            )
        )
    );

    let _ = tokio::join!(handle1, handle2);
}
```

### Code Example 9: spawn_local and LocalSet

```rust
use tokio::task::LocalSet;
use std::rc::Rc;

#[tokio::main]
async fn main() {
    // LocalSet: an environment for running !Send Futures
    let local = LocalSet::new();

    local.run_until(async {
        // Rc is not Send, but it works with spawn_local
        let data = Rc::new(vec![1, 2, 3]);

        let data_clone = data.clone();
        tokio::task::spawn_local(async move {
            println!("Local task: {:?}", data_clone);
        }).await.unwrap();

        // Spawn multiple local tasks
        let mut handles = Vec::new();
        for i in 0..5 {
            let d = data.clone();
            let handle = tokio::task::spawn_local(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(i * 10)).await;
                format!("Task {} completed with data len {}", i, d.len())
            });
            handles.push(handle);
        }

        for handle in handles {
            println!("{}", handle.await.unwrap());
        }
    }).await;
}
```

---

## 4. Channels

### Code Example 10: mpsc (Many-to-One) Channel

```rust
use tokio::sync::mpsc;

#[derive(Debug)]
enum Command {
    Get { key: String },
    Set { key: String, value: String },
    Delete { key: String },
    Shutdown,
}

#[tokio::main]
async fn main() {
    // Buffered mpsc channel
    let (tx, mut rx) = mpsc::channel::<Command>(32);

    // Worker task (receiver)
    let worker = tokio::spawn(async move {
        let mut store = std::collections::HashMap::new();

        while let Some(cmd) = rx.recv().await {
            match cmd {
                Command::Set { key, value } => {
                    println!("[worker] SET {} = {}", key, value);
                    store.insert(key, value);
                }
                Command::Get { key } => {
                    let val = store.get(&key).cloned().unwrap_or_default();
                    println!("[worker] GET {} => {}", key, val);
                }
                Command::Delete { key } => {
                    if store.remove(&key).is_some() {
                        println!("[worker] DELETE {} (success)", key);
                    } else {
                        println!("[worker] DELETE {} (key not present)", key);
                    }
                }
                Command::Shutdown => {
                    println!("[worker] shutdown");
                    break;
                }
            }
        }
        println!("[worker] final store state: {:?}", store);
    });

    // Multiple senders
    let tx2 = tx.clone();
    tokio::spawn(async move {
        tx2.send(Command::Set {
            key: "name".into(),
            value: "Alice".into(),
        }).await.unwrap();
    });

    let tx3 = tx.clone();
    tokio::spawn(async move {
        tx3.send(Command::Set {
            key: "role".into(),
            value: "admin".into(),
        }).await.unwrap();
    });

    tx.send(Command::Set {
        key: "age".into(),
        value: "30".into(),
    }).await.unwrap();

    tx.send(Command::Get { key: "name".into() }).await.unwrap();
    tx.send(Command::Delete { key: "role".into() }).await.unwrap();
    tx.send(Command::Shutdown).await.unwrap();

    worker.await.unwrap();
}
```

### Code Example 11: Request-Response Pattern with mpsc

```rust
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
enum DbCommand {
    Get {
        key: String,
        reply: oneshot::Sender<Option<String>>,
    },
    Set {
        key: String,
        value: String,
        reply: oneshot::Sender<bool>,
    },
}

/// Database actor task
async fn db_actor(mut rx: mpsc::Receiver<DbCommand>) {
    let mut store = std::collections::HashMap::new();

    while let Some(cmd) = rx.recv().await {
        match cmd {
            DbCommand::Get { key, reply } => {
                let value = store.get(&key).cloned();
                let _ = reply.send(value);
            }
            DbCommand::Set { key, value, reply } => {
                store.insert(key, value);
                let _ = reply.send(true);
            }
        }
    }
}

/// Database client
#[derive(Clone)]
struct DbClient {
    tx: mpsc::Sender<DbCommand>,
}

impl DbClient {
    async fn get(&self, key: &str) -> Option<String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send(DbCommand::Get {
            key: key.to_string(),
            reply: reply_tx,
        }).await.ok()?;
        reply_rx.await.ok()?
    }

    async fn set(&self, key: &str, value: &str) -> bool {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send(DbCommand::Set {
            key: key.to_string(),
            value: value.to_string(),
            reply: reply_tx,
        }).await.ok();
        reply_rx.await.unwrap_or(false)
    }
}

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(db_actor(rx));

    let client = DbClient { tx };

    // Multiple clients accessing concurrently
    let c1 = client.clone();
    let c2 = client.clone();

    let h1 = tokio::spawn(async move {
        c1.set("name", "Alice").await;
        c1.set("email", "alice@example.com").await;
    });

    let h2 = tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let name = c2.get("name").await;
        println!("name = {:?}", name);
        let email = c2.get("email").await;
        println!("email = {:?}", email);
    });

    let _ = tokio::join!(h1, h2);
}
```

### Code Example 12: oneshot (One-to-One, Single-Use)

```rust
use tokio::sync::oneshot;

async fn compute_answer(reply: oneshot::Sender<u64>) {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let answer = 42;
    let _ = reply.send(answer); // Send only once
}

#[tokio::main]
async fn main() {
    let (tx, rx) = oneshot::channel();

    tokio::spawn(compute_answer(tx));

    // Receive with a timeout
    tokio::select! {
        result = rx => {
            println!("Answer: {}", result.unwrap());
        }
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(5)) => {
            println!("Timeout");
        }
    }

    // Detect drop of a oneshot
    let (tx2, rx2) = oneshot::channel::<String>();
    drop(tx2); // Drop the sender side

    match rx2.await {
        Ok(value) => println!("Received: {}", value),
        Err(_) => println!("Sender closed (no value sent)"),
    }
}
```

### Code Example 13: broadcast and watch

```rust
use tokio::sync::{broadcast, watch};

#[tokio::main]
async fn main() {
    // ── broadcast — many-to-many. Send a clone to every receiver ──
    let (btx, _) = broadcast::channel::<String>(16);
    let mut brx1 = btx.subscribe();
    let mut brx2 = btx.subscribe();

    btx.send("Broadcast!".into()).unwrap();

    println!("rx1: {}", brx1.recv().await.unwrap());
    println!("rx2: {}", brx2.recv().await.unwrap());

    // Late subscriber: only receives messages sent after subscribing
    let mut brx3 = btx.subscribe();
    btx.send("New message".into()).unwrap();
    println!("rx3: {}", brx3.recv().await.unwrap());

    // Error handling for buffer overflow
    let (btx2, _) = broadcast::channel::<u32>(2); // Buffer size 2
    let mut brx = btx2.subscribe();
    btx2.send(1).unwrap();
    btx2.send(2).unwrap();
    btx2.send(3).unwrap(); // Buffer overflow → the oldest message is lost

    match brx.recv().await {
        Ok(val) => println!("Received: {}", val),
        Err(broadcast::error::RecvError::Lagged(n)) => {
            println!("{} messages were lost", n);
        }
        Err(broadcast::error::RecvError::Closed) => {
            println!("Channel closed");
        }
    }

    // ── watch — keeps only the latest value. Ideal for config-change notifications ──
    let (wtx, mut wrx) = watch::channel("initial value".to_string());

    tokio::spawn(async move {
        loop {
            wrx.changed().await.unwrap();
            println!("Config changed: {}", *wrx.borrow());
        }
    });

    wtx.send("updated value 1".into()).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    wtx.send("updated value 2".into()).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
}
```

### Code Example 14: Configuration Hot-Reload Using watch

```rust
use tokio::sync::watch;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
struct AppConfig {
    log_level: String,
    max_connections: u32,
    feature_flags: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            log_level: "info".to_string(),
            max_connections: 100,
            feature_flags: vec![],
        }
    }
}

async fn config_watcher(config_tx: watch::Sender<Arc<AppConfig>>) {
    // Watch the configuration file for changes (simplified version)
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));

    loop {
        interval.tick().await;

        // Load the configuration file (in practice, use a filesystem watcher)
        match tokio::fs::read_to_string("config.toml").await {
            Ok(content) => {
                match toml::from_str::<AppConfig>(&content) {
                    Ok(new_config) => {
                        println!("Config reloaded: {:?}", new_config);
                        let _ = config_tx.send(Arc::new(new_config));
                    }
                    Err(e) => eprintln!("Config parse error: {}", e),
                }
            }
            Err(_) => {} // Skip if the file does not exist
        }
    }
}

async fn worker(id: u32, mut config_rx: watch::Receiver<Arc<AppConfig>>) {
    loop {
        // Wait for config change notifications
        tokio::select! {
            Ok(()) = config_rx.changed() => {
                let config = config_rx.borrow().clone();
                println!("Worker {} received new config: log_level={}", id, config.log_level);
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                let config = config_rx.borrow().clone();
                println!(
                    "Worker {} processing (max_conn={})",
                    id, config.max_connections
                );
            }
        }
    }
}
```

---

## 5. Synchronization Primitives

### Code Example 15: tokio::sync::Mutex and RwLock

```rust
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

#[derive(Debug)]
struct SharedCache {
    data: std::collections::HashMap<String, String>,
    hits: u64,
    misses: u64,
}

#[tokio::main]
async fn main() {
    // Mutex: it is safe to hold the lock across await points
    let cache = Arc::new(Mutex::new(SharedCache {
        data: std::collections::HashMap::new(),
        hits: 0,
        misses: 0,
    }));

    // Access from multiple tasks
    let mut handles = Vec::new();
    for i in 0..10 {
        let cache = cache.clone();
        handles.push(tokio::spawn(async move {
            let mut c = cache.lock().await; // Asynchronously acquire the lock
            c.data.insert(format!("key_{}", i), format!("value_{}", i));
            // Awaiting while holding the lock is safe (with tokio::sync::Mutex)
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }));
    }
    for h in handles { h.await.unwrap(); }

    // RwLock: reads are concurrent; writes are exclusive
    let config = Arc::new(RwLock::new(vec!["setting1".to_string()]));

    // Reader tasks (can run concurrently)
    let c1 = config.clone();
    let c2 = config.clone();
    let r1 = tokio::spawn(async move {
        let guard = c1.read().await;
        println!("Reader 1: {:?}", *guard);
    });
    let r2 = tokio::spawn(async move {
        let guard = c2.read().await;
        println!("Reader 2: {:?}", *guard);
    });

    // Writer task (exclusive)
    let c3 = config.clone();
    let w1 = tokio::spawn(async move {
        let mut guard = c3.write().await;
        guard.push("setting2".to_string());
        println!("Writer: setting added");
    });

    let _ = tokio::join!(r1, r2, w1);
}
```

### Code Example 16: Notify — Event Notification

```rust
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let notify = Arc::new(Notify::new());

    // Waiter
    let n1 = notify.clone();
    let waiter = tokio::spawn(async move {
        println!("Waiting for notification...");
        n1.notified().await;
        println!("Notification received!");
    });

    // Notifier
    sleep(Duration::from_millis(500)).await;
    println!("Sending notification");
    notify.notify_one(); // Wake one waiting task

    waiter.await.unwrap();

    // notify_waiters(): wake all waiting tasks
    let notify = Arc::new(Notify::new());
    let mut handles = Vec::new();

    for i in 0..5 {
        let n = notify.clone();
        handles.push(tokio::spawn(async move {
            n.notified().await;
            println!("Worker {} woke up", i);
        }));
    }

    sleep(Duration::from_millis(100)).await;
    notify.notify_waiters(); // Wake everyone

    for h in handles { h.await.unwrap(); }
}
```

### Code Example 17: Barrier — Synchronization Point

```rust
use std::sync::Arc;
use tokio::sync::Barrier;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let barrier = Arc::new(Barrier::new(4)); // Wait until 4 tasks are gathered

    let mut handles = Vec::new();
    for i in 0..4 {
        let b = barrier.clone();
        handles.push(tokio::spawn(async move {
            // Phase 1: initialization
            println!("Worker {}: initializing...", i);
            sleep(Duration::from_millis(i as u64 * 100)).await;
            println!("Worker {}: initialization complete. Waiting at barrier", i);

            // Wait until all workers have arrived
            let result = b.wait().await;
            if result.is_leader() {
                println!("Worker {} is the leader (last to arrive)", i);
            }

            // Phase 2: run once everyone has arrived
            println!("Worker {}: starting main processing!", i);
        }));
    }

    for h in handles { h.await.unwrap(); }
}
```

---

## 6. Comparison Tables

### Channel Type Comparison

| Channel | Senders | Receivers | Buffer | Use Case |
|---|---|---|---|---|
| `mpsc` | Many | One | Bounded size | Command queues, worker pools |
| `mpsc::unbounded` | Many | One | Unlimited | Simple queue when memory is acceptable |
| `oneshot` | One | One | Single message | Request-response |
| `broadcast` | Many | Many | Ring buffer | Event notification, Pub/Sub |
| `watch` | One | Many | Latest value only | Config changes, state monitoring |

### spawn Type Comparison

| API | Thread | Use Case | Constraints |
|---|---|---|---|
| `tokio::spawn` | Worker | Async tasks | `Send + 'static` |
| `spawn_blocking` | Blocking pool | Sync I/O, CPU work | `Send + 'static` |
| `spawn_local` | Current thread | `!Send` Futures | Only inside `LocalSet` |
| `block_on` | Calling thread | Outside runtime → inside | Cannot be used inside the runtime |

### Synchronization Primitive Comparison

| Primitive | Use Case | Difference From the std Version |
|---|---|---|
| `tokio::sync::Mutex` | Exclusive lock | Safe to hold across `.await` |
| `tokio::sync::RwLock` | Read/write lock | Async-context aware |
| `tokio::sync::Semaphore` | Resource limiting | Async `acquire().await` |
| `tokio::sync::Notify` | Event notification | No std equivalent. Condition-variable-like usage |
| `tokio::sync::Barrier` | Synchronization point | Async-aware. Waits until all tasks arrive |
| `tokio::sync::OnceCell` | Lazy initialization | Supports async initializer functions |

### When to Use std::sync vs tokio::sync

| Situation | Recommendation | Reason |
|---|---|---|
| Lock scope contains no `.await` | `std::sync::Mutex` | Lightweight, low overhead |
| Lock scope contains `.await` | `tokio::sync::Mutex` | Prevents deadlock |
| Short atomic operations | `std::sync::atomic` | Lightest weight |
| Notification across multiple tasks | `tokio::sync::Notify` | Async support is required |
| Configuration sharing (read-heavy) | `Arc<std::sync::RwLock>` | Read locks are lightweight |
| Configuration change notifications | `tokio::sync::watch` | Built-in change notification |

---

## 7. Anti-patterns

### Anti-pattern 1: Inappropriate Channel Buffer Sizes

```rust
// BAD: a buffer size of 1 nearly always blocks the sender
let (tx, rx) = mpsc::channel(1);

// BAD: unbounded consumes memory without limit
let (tx, rx) = mpsc::unbounded_channel();
// → if the receiver is slow, the sender uses memory indefinitely

// GOOD: buffer size based on expected load
let (tx, rx) = mpsc::channel(256);  // 2-4x the expected peak load

// GOOD: design that takes back-pressure into account
let (tx, rx) = mpsc::channel(64);
// send().await waits when the buffer is full → natural flow control
```

### Anti-pattern 2: Leaving Task Leaks

```rust
// BAD: discarding the JoinHandle from spawn
fn start_background() {
    tokio::spawn(async {
        loop {
            // A task that runs forever — no one can stop it
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });
    // JoinHandle is dropped → the task is detached and uncontrollable
}

// GOOD: keep the JoinHandle and shut down gracefully
struct Service {
    handle: tokio::task::JoinHandle<()>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
}

impl Service {
    fn new() -> Self {
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() { break; }
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                        println!("Processing...");
                    }
                }
            }
            println!("Stopped gracefully");
        });
        Service { handle, shutdown_tx }
    }

    async fn stop(self) {
        let _ = self.shutdown_tx.send(true);
        let _ = self.handle.await;
    }
}
```

### Anti-pattern 3: Unnecessary Use of tokio::sync::Mutex

```rust
// BAD: using tokio::sync::Mutex even though there is no await
use tokio::sync::Mutex;
async fn bad_counter(counter: &Mutex<u64>) {
    let mut c = counter.lock().await;
    *c += 1;
    // No await point → std::sync::Mutex would suffice
}

// GOOD: when there is no await, std::sync::Mutex is lighter
use std::sync::Mutex;
async fn good_counter(counter: &Mutex<u64>) {
    let mut c = counter.lock().unwrap();
    *c += 1;
    // drop(c); — released automatically at end of scope
}

// BEST: for a simple counter, use atomics
use std::sync::atomic::{AtomicU64, Ordering};
async fn best_counter(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}
```

### Anti-pattern 4: Calling block_on Inside the Runtime

```rust
// BAD: calling block_on from within an async runtime panics
#[tokio::main]
async fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    // rt.block_on(some_future()); // panic: "Cannot start a runtime from within a runtime"
}

// GOOD: use .await inside an async runtime
#[tokio::main]
async fn main() {
    let result = some_future().await;
    println!("{}", result);
}

// GOOD: when calling async code from a synchronous context
fn sync_function() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async { some_future().await });
    println!("{}", result);
}

async fn some_future() -> String {
    "result".to_string()
}
```

---

## 8. Practical Patterns

### 8.1 Actor Pattern

```rust
use tokio::sync::{mpsc, oneshot};

/// Messages that the actor processes
enum ActorMessage {
    Increment,
    Decrement,
    GetValue { reply: oneshot::Sender<i64> },
}

/// Counter actor
struct CounterActor {
    value: i64,
    rx: mpsc::Receiver<ActorMessage>,
}

impl CounterActor {
    fn new(rx: mpsc::Receiver<ActorMessage>) -> Self {
        CounterActor { value: 0, rx }
    }

    async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                ActorMessage::Increment => self.value += 1,
                ActorMessage::Decrement => self.value -= 1,
                ActorMessage::GetValue { reply } => {
                    let _ = reply.send(self.value);
                }
            }
        }
    }
}

/// Handle to the actor (cloneable)
#[derive(Clone)]
struct CounterHandle {
    tx: mpsc::Sender<ActorMessage>,
}

impl CounterHandle {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel(32);
        let actor = CounterActor::new(rx);
        tokio::spawn(actor.run());
        CounterHandle { tx }
    }

    async fn increment(&self) {
        let _ = self.tx.send(ActorMessage::Increment).await;
    }

    async fn decrement(&self) {
        let _ = self.tx.send(ActorMessage::Decrement).await;
    }

    async fn get_value(&self) -> i64 {
        let (reply_tx, reply_rx) = oneshot::channel();
        let _ = self.tx.send(ActorMessage::GetValue { reply: reply_tx }).await;
        reply_rx.await.unwrap_or(0)
    }
}

#[tokio::main]
async fn main() {
    let counter = CounterHandle::new();

    // Concurrent operations from multiple tasks
    let mut handles = Vec::new();
    for _ in 0..100 {
        let c = counter.clone();
        handles.push(tokio::spawn(async move {
            c.increment().await;
        }));
    }
    for h in handles { h.await.unwrap(); }

    let value = counter.get_value().await;
    println!("Final value: {}", value); // 100
}
```

### 8.2 Worker Pool Pattern

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

struct Job {
    id: u32,
    data: String,
}

struct WorkerPool {
    job_tx: mpsc::Sender<Job>,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl WorkerPool {
    fn new(num_workers: usize) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>(100);
        let job_rx = std::sync::Arc::new(tokio::sync::Mutex::new(job_rx));
        let mut handles = Vec::new();

        for worker_id in 0..num_workers {
            let rx = job_rx.clone();
            let handle = tokio::spawn(async move {
                loop {
                    let job = {
                        let mut rx = rx.lock().await;
                        rx.recv().await
                    };

                    match job {
                        Some(job) => {
                            println!("Worker {}: processing job {} ({})",
                                worker_id, job.id, job.data);
                            sleep(Duration::from_millis(100)).await;
                            println!("Worker {}: job {} complete", worker_id, job.id);
                        }
                        None => {
                            println!("Worker {}: shutdown", worker_id);
                            break;
                        }
                    }
                }
            });
            handles.push(handle);
        }

        WorkerPool { job_tx, handles }
    }

    async fn submit(&self, job: Job) -> Result<(), mpsc::error::SendError<Job>> {
        self.job_tx.send(job).await
    }

    async fn shutdown(self) {
        drop(self.job_tx); // Close the channel
        for handle in self.handles {
            let _ = handle.await;
        }
    }
}

#[tokio::main]
async fn main() {
    let pool = WorkerPool::new(4);

    // Submit jobs
    for i in 0..20 {
        pool.submit(Job {
            id: i,
            data: format!("data_{}", i),
        }).await.unwrap();
    }

    pool.shutdown().await;
    println!("All jobs complete");
}
```

### 8.3 Periodic Task Scheduler

```rust
use tokio::time::{interval, Duration, Instant};

struct Scheduler {
    tasks: Vec<ScheduledTask>,
}

struct ScheduledTask {
    name: String,
    interval: Duration,
    task: Box<dyn Fn() -> tokio::task::JoinHandle<()> + Send + Sync>,
}

impl Scheduler {
    fn new() -> Self {
        Scheduler { tasks: Vec::new() }
    }

    fn add_task<F, Fut>(&mut self, name: &str, interval_ms: u64, task_fn: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.tasks.push(ScheduledTask {
            name: name.to_string(),
            interval: Duration::from_millis(interval_ms),
            task: Box::new(move || tokio::spawn(task_fn())),
        });
    }

    async fn run(self, shutdown: tokio::sync::watch::Receiver<bool>) {
        let mut handles = Vec::new();

        for scheduled in &self.tasks {
            let name = scheduled.name.clone();
            let interval_duration = scheduled.interval;
            let mut shutdown_rx = shutdown.clone();

            // Scheduler loop for each task
            let handle = tokio::spawn(async move {
                let mut ticker = interval(interval_duration);

                loop {
                    tokio::select! {
                        _ = ticker.tick() => {
                            let start = Instant::now();
                            println!("[{}] starting", name);
                            // Run the actual task here
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            println!("[{}] completed ({:?})", name, start.elapsed());
                        }
                        Ok(()) = shutdown_rx.changed() => {
                            if *shutdown_rx.borrow() {
                                println!("[{}] shutdown", name);
                                break;
                            }
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for h in handles {
            let _ = h.await;
        }
    }
}
```

---

## FAQ

### Q1: What is the difference between `spawn_blocking` and `block_on`?

**A:** `spawn_blocking` is a means of offloading blocking work from inside an async runtime onto a dedicated thread. `block_on` is a means of starting an async runtime from synchronous code.

```rust
// spawn_blocking: from async → run sync work on a separate thread
async fn example() {
    let result = tokio::task::spawn_blocking(|| {
        std::fs::read_to_string("large.txt") // Blocking I/O
    }).await.unwrap();
}

// block_on: from sync code → run async code
fn sync_context() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async { fetch_data().await });
}
```

### Q2: What happens to branches that are not selected by `select!`?

**A:** The Futures of unselected branches are dropped. To avoid resource leaks, manage processes that need cleanup of intermediate state explicitly with a `CancellationToken`.

```rust
use tokio_util::sync::CancellationToken;

async fn careful_select() {
    let token = CancellationToken::new();

    let token_clone = token.clone();
    let task = tokio::spawn(async move {
        tokio::select! {
            _ = token_clone.cancelled() => {
                println!("Cancelled. Cleaning up...");
                // Explicitly release resources
            }
            result = long_running_task() => {
                println!("Task complete: {:?}", result);
            }
        }
    });

    // Cancel based on a condition
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    token.cancel();
    let _ = task.await;
}

async fn long_running_task() -> String {
    tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
    "done".to_string()
}
```

### Q3: What is the performance difference between tokio and OS threads?

**A:** A tokio task consumes about 256 bytes, while an OS thread consumes about 8MB (stack). For 100,000 concurrent connections, tokio is overwhelmingly more favorable. However, for CPU-bound work a thread pool is more appropriate.

| Metric | Tokio Task | OS Thread |
|---|---|---|
| Memory (initial) | ~256 bytes | ~8 MB (stack) |
| Spawn cost | ~a few microseconds | ~a few milliseconds |
| Context switch | User space (~ns) | Kernel space (~μs) |
| Memory at 100k tasks | ~25 MB | ~800 GB (impractical) |
| Best suited for | I/O-bound work | CPU-bound work |

### Q4: What is a good rule of thumb for mpsc channel buffer sizes?

**A:** A common guideline is 2 to 4 times the peak number of messages per second. Too small and the sender blocks; too large and memory is wasted. The key is to choose a size that allows back-pressure to work effectively.

```rust
// Web server (assumed 100 req/s): around 256
let (tx, rx) = mpsc::channel(256);

// High-throughput pipeline (10K msg/s): 1024-4096
let (tx, rx) = mpsc::channel(4096);

// Low-frequency commands (config changes, etc.): 8-32
let (tx, rx) = mpsc::channel(16);
```

### Q5: What happens if a task panics?

**A:** If a task spawned with `tokio::spawn` panics, other tasks are not affected. `.await` on the JoinHandle returns a `JoinError`.

```rust
#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        panic!("Panic inside task!");
    });

    match handle.await {
        Ok(_) => println!("Completed normally"),
        Err(e) if e.is_panic() => {
            // Retrieve the panic value
            let panic_value = e.into_panic();
            if let Some(msg) = panic_value.downcast_ref::<&str>() {
                eprintln!("Panic: {}", msg);
            }
        }
        Err(e) if e.is_cancelled() => println!("Cancelled"),
        Err(e) => eprintln!("Other error: {}", e),
    }
    // Other tasks continue to run normally
}
```

---

## Summary

| Item | Key Point |
|---|---|
| Runtime selection | `multi_thread` is the default. Use `current_thread` for tests |
| Work-stealing | Automatic load balancing. Local queue → global queue → steal from other threads |
| Task spawn | `Send + 'static` bound. Retrieve results via JoinHandle |
| JoinSet | Dynamic task collection. Retrieve results in completion order. Bulk cancellation possible |
| spawn_local | For `!Send` Futures. Usable only inside a LocalSet |
| task_local! | Per-task context information (request IDs, etc.) |
| mpsc | The most general-purpose. A queue with back-pressure |
| oneshot | Request-response pattern |
| broadcast | Pub/Sub pattern |
| watch | State monitoring and configuration-change notifications |
| Mutex / RwLock | Use the tokio versions for critical sections that contain awaits |
| Notify | Event notifications. Condition-variable-like usage |
| Barrier | Synchronization point across all tasks |
| Shutdown | Graceful stop using watch + select! |
| Actor pattern | Safe message passing using mpsc + oneshot |

## Recommended Next Reads

- [Async Patterns](./02-async-patterns.md) — Stream, concurrency limiting, retry patterns
- [Networking](./03-networking.md) — Asynchronous HTTP/WebSocket/gRPC
- [Concurrency](../03-systems/01-concurrency.md) — Fundamentals of threads and locks

## References

1. **Tokio Documentation**: https://docs.rs/tokio/latest/tokio/
2. **Tokio Tutorial**: https://tokio.rs/tokio/tutorial
3. **Alice Ryhl - Actors with Tokio**: https://ryhl.io/blog/actors-with-tokio/
4. **Tokio Mini-Redis (educational implementation)**: https://github.com/tokio-rs/mini-redis
5. **Tokio Metrics**: https://docs.rs/tokio-metrics/latest/tokio_metrics/
6. **Jon Gjengset - Decrusting tokio**: https://www.youtube.com/watch?v=o2ob8zkeq2s
