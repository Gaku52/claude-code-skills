# Concurrency — Threads, Mutex/RwLock, rayon

> Master compile-time data race prevention through Rust's ownership system, and practical concurrent processing using threads, locks, and parallel iterators

## What You'll Learn in This Chapter

1. **Threads and ownership** — std::thread, Send/Sync traits, scoped threads
2. **Synchronization primitives** — Mutex, RwLock, Condvar, Atomic, Barrier
3. **Data parallelism** — Parallel iterators and work splitting with rayon
4. **Message passing via channels** — mpsc, crossbeam-channel
5. **Advanced concurrency patterns** — Lock-free data structures, Producer-Consumer, work stealing


## Prerequisites

Reading the following beforehand will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Content of [Memory Layout — Stack/Heap, repr](./00-memory-layout.md)

---

## 1. Rust's Concurrency Safety Model

```
┌────────── Rust's Concurrency Safety Guarantees ──────────┐
│                                                            │
│  Compile-time checks                                       │
│  ┌──────────────────────────────────────────┐              │
│  │  Send: type T can move to another thread │              │
│  │  Sync: &T can be shared across threads   │              │
│  │                                          │              │
│  │  Send + Sync examples:                   │              │
│  │    i32, String, Vec<T>, Arc<T>           │              │
│  │                                          │              │
│  │  !Send examples:                         │              │
│  │    Rc<T>, *mut T                         │              │
│  │                                          │              │
│  │  !Sync examples:                         │              │
│  │    Cell<T>, RefCell<T>                   │              │
│  └──────────────────────────────────────────┘              │
│                                                            │
│  → Data races are compile errors                           │
│  → Deadlocks cannot be prevented (logical errors)          │
└────────────────────────────────────────────────────────────┘
```

### Detailed Rules for Send / Sync

```rust
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

// Send: type T can be "moved with ownership" to another thread
// Sync: &T can be "referenced concurrently" from multiple threads

// Basic rule:
// T: Sync  ⟺  &T: Send
// In other words, "shared references can be sent" = "can be shared"

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

fn check_traits() {
    // Primitives: Send + Sync
    assert_send::<i32>();
    assert_sync::<i32>();
    assert_send::<String>();
    assert_sync::<String>();

    // Arc<T>: Send + Sync if T is Send + Sync
    assert_send::<Arc<i32>>();
    assert_sync::<Arc<i32>>();

    // Rc<T>: !Send, !Sync (reference count is non-atomic)
    // assert_send::<Rc<i32>>();  // Compile error!
    // assert_sync::<Rc<i32>>();  // Compile error!

    // Cell<T>: Send but !Sync (interior mutability is not thread-safe)
    assert_send::<Cell<i32>>();
    // assert_sync::<Cell<i32>>();  // Compile error!

    // Mutex<T>: Send + Sync if T is Send
    assert_send::<std::sync::Mutex<Vec<i32>>>();
    assert_sync::<std::sync::Mutex<Vec<i32>>>();
}

fn main() {
    check_traits();
    println!("All type checks passed");
}
```

### Code Example: Practical Meaning of Send/Sync

```rust
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    // NG: Passing Rc to a thread → compile error
    // let rc = Rc::new(42);
    // std::thread::spawn(move || {
    //     println!("{}", rc);
    // });
    // error: `Rc<i32>` cannot be sent between threads safely

    // OK: Use Arc to share between threads
    let arc = Arc::new(42);
    let arc_clone = Arc::clone(&arc);
    let handle = std::thread::spawn(move || {
        println!("Other thread: {}", arc_clone);
    });
    handle.join().unwrap();
    println!("Main: {}", arc);

    // Manually mark with unsafe impl Send/Sync (dangerous!)
    // Use only when you guarantee safety yourself
    struct MyWrapper(*mut u8);
    unsafe impl Send for MyWrapper {}
    unsafe impl Sync for MyWrapper {}
}
```

---

## 2. Basic Thread Operations

### Code Example 1: Spawning Threads and join

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // Spawning a thread
    let handle = thread::spawn(|| {
        for i in 1..=5 {
            println!("[Child thread] count: {}", i);
            thread::sleep(Duration::from_millis(100));
        }
        42 // Return value
    });

    // Main thread processing
    for i in 1..=3 {
        println!("[Main] count: {}", i);
        thread::sleep(Duration::from_millis(150));
    }

    // Wait for completion with join and obtain return value
    let result = handle.join().unwrap();
    println!("Child thread return value: {}", result);

    // Use the thread builder to specify name and stack size
    let builder = thread::Builder::new()
        .name("worker-1".into())
        .stack_size(4 * 1024 * 1024); // 4MB

    let handle = builder.spawn(|| {
        println!("Thread name: {:?}", thread::current().name());
    }).unwrap();
    handle.join().unwrap();
}
```

### Code Example 2: Scoped Threads (Rust 1.63+)

```rust
use std::thread;

fn main() {
    let data = vec![1, 2, 3, 4, 5];
    let mut results = vec![0; 5];

    // Threads inside scope are auto-joined when the scope ends
    // → references to local variables can be passed safely
    thread::scope(|s| {
        // Data-reader thread
        s.spawn(|| {
            let sum: i32 = data.iter().sum();
            println!("Sum: {}", sum);
        });

        // Data-mutating threads (split borrowing)
        for (i, slot) in results.iter_mut().enumerate() {
            s.spawn(move || {
                *slot = data[i] * data[i];
            });
        }
    }); // ← blocks until all threads complete

    println!("Squares: {:?}", results); // [1, 4, 9, 16, 25]
}
```

### Code Example: Building Your Own Thread Pool

```rust
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<mpsc::Sender<Job>>,
}

struct Worker {
    id: usize,
    handle: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        assert!(size > 0, "thread count must be at least 1");

        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));

        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            let receiver = Arc::clone(&receiver);
            let handle = thread::Builder::new()
                .name(format!("pool-worker-{}", id))
                .spawn(move || loop {
                    let job = receiver.lock().unwrap().recv();
                    match job {
                        Ok(job) => {
                            println!("[Worker {}] starting job execution", id);
                            job();
                        }
                        Err(_) => {
                            println!("[Worker {}] channel closed, exiting", id);
                            break;
                        }
                    }
                })
                .unwrap();

            workers.push(Worker {
                id,
                handle: Some(handle),
            });
        }

        ThreadPool {
            workers,
            sender: Some(sender),
        }
    }

    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.as_ref().unwrap().send(job).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Drop sender → close the channel → workers exit their loops
        drop(self.sender.take());

        for worker in &mut self.workers {
            println!("Waiting for worker {} to finish...", worker.id);
            if let Some(handle) = worker.handle.take() {
                handle.join().unwrap();
            }
        }
    }
}

fn main() {
    let pool = ThreadPool::new(4);

    for i in 0..8 {
        pool.execute(move || {
            println!("Task {} running (thread: {:?})", i, thread::current().name());
            thread::sleep(std::time::Duration::from_millis(100));
            println!("Task {} done", i);
        });
    }

    // When pool is dropped → wait for all tasks to complete
    drop(pool);
    println!("All tasks completed");
}
```

---

## 3. Synchronization Primitives

### Code Example 3: Mutex and RwLock

```rust
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

fn main() {
    // Mutex — exclusive lock (restricts both reads and writes to a single thread)
    let counter = Arc::new(Mutex::new(0u64));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                let mut num = counter.lock().unwrap();
                *num += 1;
                // The lock is released automatically when num is dropped
            }
        }));
    }

    for h in handles { h.join().unwrap(); }
    println!("Counter: {}", *counter.lock().unwrap()); // 10000

    // RwLock — concurrent reads, exclusive writes
    let config = Arc::new(RwLock::new(String::from("initial config")));

    let config_reader = Arc::clone(&config);
    let reader = thread::spawn(move || {
        let cfg = config_reader.read().unwrap(); // read lock
        println!("Config: {}", *cfg);
    });

    let config_writer = Arc::clone(&config);
    let writer = thread::spawn(move || {
        let mut cfg = config_writer.write().unwrap(); // write lock
        *cfg = "updated config".to_string();
    });

    reader.join().unwrap();
    writer.join().unwrap();
}
```

### How Locks Behave

```
┌─────────── Mutex vs RwLock ─────────────┐
│                                          │
│  Mutex:                                  │
│    Thread A: [LOCK████████UNLOCK]        │
│    Thread B:      [wait][LOCK████UNLOCK] │
│    Thread C:           [wait....][LOCK]  │
│    → Only one at a time                  │
│                                          │
│  RwLock:                                 │
│    Reader A: [RLOCK██████████RUNLOCK]    │
│    Reader B: [RLOCK██████████RUNLOCK]    │
│    Reader C: [RLOCK██████████RUNLOCK]    │
│    → Concurrent reads OK                 │
│                                          │
│    Writer:   [wait.........][WLOCK██UNL] │
│    → Writes are exclusive (waits for     │
│      all readers to finish)              │
└──────────────────────────────────────────┘
```

### Code Example: Handling Mutex Poisoning

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));

    // Spawn a thread that panics
    let data_clone = Arc::clone(&data);
    let handle = thread::spawn(move || {
        let mut guard = data_clone.lock().unwrap();
        guard.push(4);
        panic!("intentional panic!"); // ← lock is poisoned on panic
    });

    // Panics are returned as Result::Err from join
    let _ = handle.join();

    // Handling a poisoned Mutex
    match data.lock() {
        Ok(guard) => {
            println!("Normal lock: {:?}", *guard);
        }
        Err(poisoned) => {
            // Poisoned, but the inner data is still accessible
            println!("Poisoning detected! data: {:?}", *poisoned.into_inner());
            // You must judge data integrity yourself
        }
    }

    // parking_lot::Mutex never poisons (no unwrap needed)
    // use parking_lot::Mutex;
    // let m = Mutex::new(42);
    // let guard = m.lock(); // returns MutexGuard directly, not Result
}
```

### Code Example 4: Atomic Operations

```rust
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let counter = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    let mut handles = vec![];

    for _ in 0..4 {
        let counter = Arc::clone(&counter);
        let running = Arc::clone(&running);
        handles.push(thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                counter.fetch_add(1, Ordering::Relaxed);
                // Relaxed: weakest ordering guarantee (sufficient for a counter)
            }
        }));
    }

    thread::sleep(std::time::Duration::from_millis(100));
    running.store(false, Ordering::Relaxed); // tell all threads to stop

    for h in handles { h.join().unwrap(); }
    println!("Counter: {}", counter.load(Ordering::Relaxed));
}
```

### Code Example: Understanding the Differences in Ordering

```rust
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    // Types of Ordering and their uses
    // Relaxed:    Weakest. No ordering guarantees with other operations. Good for counters.
    // Acquire:    Reads/writes after this load cannot be reordered before it.
    // Release:    Reads/writes before this store cannot be reordered after it.
    // AcqRel:     Acquire + Release. Used for RMW (Read-Modify-Write) operations.
    // SeqCst:     Strongest. Total order of all operations across all threads.

    // Release-Acquire pair example: publishing data via a flag
    let data = Arc::new(AtomicU32::new(0));
    let ready = Arc::new(AtomicBool::new(false));

    let data_clone = Arc::clone(&data);
    let ready_clone = Arc::clone(&ready);

    // Producer
    let producer = thread::spawn(move || {
        data_clone.store(42, Ordering::Relaxed);      // write data
        ready_clone.store(true, Ordering::Release);    // Release: prior writes complete first
    });

    // Consumer
    let data_clone2 = Arc::clone(&data);
    let ready_clone2 = Arc::clone(&ready);

    let consumer = thread::spawn(move || {
        // Acquire: after reading ready=true, the load of data is guaranteed to be 42
        while !ready_clone2.load(Ordering::Acquire) {
            std::hint::spin_loop(); // busy-wait
        }
        let value = data_clone2.load(Ordering::Relaxed);
        assert_eq!(value, 42); // guaranteed by Release-Acquire pairing
        println!("Data: {}", value);
    });

    producer.join().unwrap();
    consumer.join().unwrap();
}
```

### Code Example 5: Condvar (Condition Variable)

```rust
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::collections::VecDeque;

/// A thread-safe queue using Condvar
struct BlockingQueue<T> {
    queue: Mutex<VecDeque<T>>,
    not_empty: Condvar,
}

impl<T> BlockingQueue<T> {
    fn new() -> Self {
        BlockingQueue {
            queue: Mutex::new(VecDeque::new()),
            not_empty: Condvar::new(),
        }
    }

    fn push(&self, item: T) {
        let mut q = self.queue.lock().unwrap();
        q.push_back(item);
        self.not_empty.notify_one(); // wake one waiting thread
    }

    fn pop(&self) -> T {
        let mut q = self.queue.lock().unwrap();
        while q.is_empty() {
            q = self.not_empty.wait(q).unwrap(); // wait for notification
        }
        q.pop_front().unwrap()
    }
}

fn main() {
    let queue = Arc::new(BlockingQueue::new());

    let consumer_queue = Arc::clone(&queue);
    let consumer = thread::spawn(move || {
        for _ in 0..5 {
            let item = consumer_queue.pop();
            println!("Consumed: {}", item);
        }
    });

    for i in 0..5 {
        queue.push(i);
        thread::sleep(std::time::Duration::from_millis(50));
    }

    consumer.join().unwrap();
}
```

### Code Example: Synchronization with Barrier

```rust
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

fn main() {
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = vec![];

    for id in 0..num_threads {
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            // Phase 1: each thread initializes independently
            println!("[Thread {}] Phase 1: initializing...", id);
            thread::sleep(std::time::Duration::from_millis(100 * id as u64));
            println!("[Thread {}] Phase 1: initialization complete", id);

            // Wait until all threads finish initialization
            let result = barrier.wait();
            if result.is_leader() {
                println!("=== All threads initialized, starting Phase 2 ===");
            }

            // Phase 2: all threads start processing simultaneously
            let start = Instant::now();
            println!("[Thread {}] Phase 2: started at {:?}", id, start.elapsed());

            // Sync again
            barrier.wait();
            println!("[Thread {}] Phase 2: complete", id);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
```

### Code Example: One-Time Initialization with Once and OnceLock

```rust
use std::sync::{Once, OnceLock};

// Once: runs the initialization exactly once
static INIT: Once = Once::new();
static mut CONFIG: Option<String> = None;

fn get_config_legacy() -> &'static str {
    INIT.call_once(|| {
        // Executed only on the first call
        unsafe {
            CONFIG = Some("production".to_string());
        }
    });
    unsafe { CONFIG.as_ref().unwrap().as_str() }
}

// OnceLock (Rust 1.70+): type-safe one-time initialization
static CONFIG_NEW: OnceLock<String> = OnceLock::new();

fn get_config() -> &'static str {
    CONFIG_NEW.get_or_init(|| {
        println!("Initializing config...");
        "production".to_string()
    })
}

fn main() {
    // Initialization runs only once even when called multiple times
    println!("1st: {}", get_config());
    println!("2nd: {}", get_config());
    println!("3rd: {}", get_config());
    // Output:
    // Initializing config...
    // 1st: production
    // 2nd: production
    // 3rd: production
}
```

---

## 4. Message Passing via Channels

### Code Example: mpsc Channel

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    // mpsc: Multiple Producer, Single Consumer
    let (tx, rx) = mpsc::channel();

    // Multiple producers
    for id in 0..3 {
        let tx = tx.clone();
        thread::spawn(move || {
            for i in 0..5 {
                let msg = format!("Producer {} - Message {}", id, i);
                tx.send(msg).unwrap();
                thread::sleep(Duration::from_millis(50));
            }
        });
    }

    // Drop the original tx (only the clones remain)
    drop(tx);

    // Consumer: receive all messages
    let mut count = 0;
    while let Ok(msg) = rx.recv() {
        println!("Received: {}", msg);
        count += 1;
    }
    println!("Received {} messages in total", count); // 15

    // sync_channel: buffered channel
    let (tx, rx) = mpsc::sync_channel::<i32>(3); // buffer size 3

    thread::spawn(move || {
        for i in 0..10 {
            println!("Sending: {} (waits if buffer is full)", i);
            tx.send(i).unwrap(); // blocks if the buffer is full
        }
    });

    thread::sleep(Duration::from_millis(500));
    while let Ok(v) = rx.recv() {
        println!("Received: {}", v);
    }
}
```

### Code Example: Multi-Consumer with crossbeam-channel

```rust
use std::thread;
use std::time::Duration;

// crossbeam-channel: MPMC (Multiple Producer, Multiple Consumer)
// Cargo.toml: crossbeam-channel = "0.5"

fn crossbeam_example() {
    use crossbeam_channel::{bounded, select, unbounded, Receiver, Sender};

    // bounded channel (with buffer)
    let (tx, rx): (Sender<String>, Receiver<String>) = bounded(10);

    // Multiple consumers
    let mut consumers = vec![];
    for id in 0..3 {
        let rx = rx.clone();
        consumers.push(thread::spawn(move || {
            let mut processed = 0;
            while let Ok(msg) = rx.recv() {
                println!("[Consumer {}] processing: {}", id, msg);
                processed += 1;
            }
            println!("[Consumer {}] processed {} items in total", id, processed);
        }));
    }

    // Producer
    for i in 0..30 {
        tx.send(format!("Job {}", i)).unwrap();
    }
    drop(tx); // close the channel

    for c in consumers {
        c.join().unwrap();
    }
}

// select! macro: wait on multiple channels
fn select_example() {
    use crossbeam_channel::{bounded, select, after, tick};

    let (tx1, rx1) = bounded(1);
    let (tx2, rx2) = bounded(1);

    // Timers
    let timeout = after(Duration::from_secs(1));
    let ticker = tick(Duration::from_millis(200));

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        tx1.send("data from channel 1").unwrap();
    });

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        tx2.send("data from channel 2").unwrap();
    });

    loop {
        select! {
            recv(rx1) -> msg => {
                match msg {
                    Ok(m) => println!("rx1: {}", m),
                    Err(_) => println!("rx1 closed"),
                }
            }
            recv(rx2) -> msg => {
                match msg {
                    Ok(m) => println!("rx2: {}", m),
                    Err(_) => println!("rx2 closed"),
                }
            }
            recv(ticker) -> _ => {
                println!("tick");
            }
            recv(timeout) -> _ => {
                println!("timeout!");
                break;
            }
        }
    }
}

fn main() {
    println!("=== crossbeam MPMC ===");
    crossbeam_example();
    println!("\n=== select! ===");
    select_example();
}
```

---

## 5. rayon — Data Parallelism

### Code Example 6: Parallel Iterators

```rust
use rayon::prelude::*;

fn main() {
    let data: Vec<u64> = (0..10_000_000).collect();

    // Parallel map + filter + sum
    let sum: u64 = data.par_iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * x)
        .sum();
    println!("Sum of squares of evens: {}", sum);

    // Parallel sort
    let mut nums: Vec<i32> = (0..1_000_000).rev().collect();
    nums.par_sort_unstable();
    assert!(nums.windows(2).all(|w| w[0] <= w[1]));

    // Parallel for_each
    let results: Vec<String> = (0..100)
        .into_par_iter()
        .map(|i| format!("result#{}", i))
        .collect();
    println!("Result count: {}", results.len());
}
```

### Code Example 7: Custom Thread Pool

```rust
use rayon::ThreadPoolBuilder;

fn main() {
    // Custom thread pool (control number of threads)
    let pool = ThreadPoolBuilder::new()
        .num_threads(4)
        .thread_name(|index| format!("worker-{}", index))
        .build()
        .unwrap();

    pool.install(|| {
        let data: Vec<i64> = (0..1_000_000).collect();
        let sum: i64 = data.par_iter().sum();
        println!("Sum: {}", sum);
    });

    // join — run two computations in parallel
    let (left, right) = rayon::join(
        || expensive_computation_a(),
        || expensive_computation_b(),
    );
    println!("A={}, B={}", left, right);
}

fn expensive_computation_a() -> u64 { (0..1_000_000u64).sum() }
fn expensive_computation_b() -> u64 { (0..500_000u64).map(|x| x * 2).sum() }
```

### Code Example: Advanced Use of rayon

```rust
use rayon::prelude::*;
use std::collections::HashMap;

/// Aggregation via parallel reduce
fn parallel_word_count(texts: &[String]) -> HashMap<String, usize> {
    texts
        .par_iter()
        .fold(
            || HashMap::new(),
            |mut map, text| {
                for word in text.split_whitespace() {
                    *map.entry(word.to_lowercase()).or_insert(0) += 1;
                }
                map
            },
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (key, count) in b {
                    *a.entry(key).or_insert(0) += count;
                }
                a
            },
        )
}

/// Parallel find (first element satisfying the condition)
fn parallel_find_prime(range: std::ops::Range<u64>) -> Option<u64> {
    range.into_par_iter().find_any(|&n| is_prime(n))
}

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let limit = (n as f64).sqrt() as u64;
    (3..=limit).step_by(2).all(|i| n % i != 0)
}

/// Parallel chunk processing
fn parallel_chunk_processing(data: &[u8]) -> Vec<u32> {
    data.par_chunks(1024)
        .map(|chunk| {
            // Process each chunk (e.g., checksum computation)
            chunk.iter().map(|&b| b as u32).sum()
        })
        .collect()
}

fn main() {
    // Parallel word count
    let texts: Vec<String> = (0..1000)
        .map(|i| format!("hello world rust programming hello rust {}", i))
        .collect();
    let counts = parallel_word_count(&texts);
    println!("Occurrences of 'hello': {}", counts.get("hello").unwrap_or(&0));
    println!("Occurrences of 'rust': {}", counts.get("rust").unwrap_or(&0));

    // Parallel search
    let prime = parallel_find_prime(1_000_000..1_001_000);
    println!("Found prime: {:?}", prime);

    // Chunk processing
    let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
    let checksums = parallel_chunk_processing(&data);
    println!("Chunk count: {}, first checksum: {}", checksums.len(), checksums[0]);
}
```

---

## 6. Advanced Concurrency Patterns

### Code Example: Read-Copy-Update (RCU) Pattern

```rust
use std::sync::{Arc, atomic::{AtomicPtr, Ordering}};
use std::thread;

/// Arc-based Read-Copy-Update pattern
/// Reads are lock-free; writes create new data and replace it
struct RcuConfig {
    data: std::sync::RwLock<Arc<ConfigData>>,
}

#[derive(Clone, Debug)]
struct ConfigData {
    database_url: String,
    max_connections: u32,
    timeout_ms: u64,
}

impl RcuConfig {
    fn new(data: ConfigData) -> Self {
        RcuConfig {
            data: std::sync::RwLock::new(Arc::new(data)),
        }
    }

    /// Read: get a clone of the Arc (fast)
    fn read(&self) -> Arc<ConfigData> {
        Arc::clone(&self.data.read().unwrap())
    }

    /// Update: replace with new data
    fn update<F>(&self, f: F)
    where
        F: FnOnce(&ConfigData) -> ConfigData,
    {
        let mut guard = self.data.write().unwrap();
        let new_data = f(&guard);
        *guard = Arc::new(new_data);
    }
}

fn main() {
    let config = Arc::new(RcuConfig::new(ConfigData {
        database_url: "postgres://localhost/mydb".to_string(),
        max_connections: 10,
        timeout_ms: 5000,
    }));

    // Reader threads
    let mut readers = vec![];
    for id in 0..5 {
        let config = Arc::clone(&config);
        readers.push(thread::spawn(move || {
            for _ in 0..100 {
                let data = config.read();
                // Reads are lock-free (just Arc::clone)
                let _url = &data.database_url;
                let _max = data.max_connections;
            }
            println!("[Reader {}] complete", id);
        }));
    }

    // Writer thread
    let config_writer = Arc::clone(&config);
    let writer = thread::spawn(move || {
        for i in 0..10 {
            config_writer.update(|old| ConfigData {
                max_connections: old.max_connections + 1,
                ..old.clone()
            });
            thread::sleep(std::time::Duration::from_millis(10));
        }
        println!("[Writer] complete");
    });

    for r in readers { r.join().unwrap(); }
    writer.join().unwrap();

    let final_config = config.read();
    println!("Final max_connections: {}", final_config.max_connections);
}
```

### Code Example: Double Buffering Pattern

```rust
use std::sync::{Arc, RwLock, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

/// Double buffer: separate reads and writes
struct DoubleBuffer<T: Clone> {
    buffers: [RwLock<T>; 2],
    active: std::sync::atomic::AtomicUsize, // currently active buffer (0 or 1)
}

impl<T: Clone> DoubleBuffer<T> {
    fn new(initial: T) -> Self {
        DoubleBuffer {
            buffers: [
                RwLock::new(initial.clone()),
                RwLock::new(initial),
            ],
            active: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Read: from the active buffer (minimum lock contention)
    fn read(&self) -> std::sync::RwLockReadGuard<'_, T> {
        let idx = self.active.load(Ordering::Acquire);
        self.buffers[idx].read().unwrap()
    }

    /// Write: write to the inactive buffer and swap
    fn write<F>(&self, update_fn: F)
    where
        F: FnOnce(&T) -> T,
    {
        let active = self.active.load(Ordering::Acquire);
        let inactive = 1 - active;

        // Write new data to the inactive buffer
        {
            let current = self.buffers[active].read().unwrap();
            let new_data = update_fn(&current);
            let mut inactive_guard = self.buffers[inactive].write().unwrap();
            *inactive_guard = new_data;
        }

        // Swap the buffers
        self.active.store(inactive, Ordering::Release);
    }
}

fn main() {
    let buffer = Arc::new(DoubleBuffer::new(vec![0u64; 100]));

    let running = Arc::new(AtomicBool::new(true));

    // Reader thread
    let buf_reader = Arc::clone(&buffer);
    let run_r = Arc::clone(&running);
    let reader = thread::spawn(move || {
        let mut reads = 0u64;
        while run_r.load(Ordering::Relaxed) {
            let data = buf_reader.read();
            let _sum: u64 = data.iter().sum();
            reads += 1;
        }
        println!("Read count: {}", reads);
    });

    // Writer thread
    let buf_writer = Arc::clone(&buffer);
    for i in 0..100 {
        buf_writer.write(|old| {
            old.iter().map(|x| x + 1).collect()
        });
        thread::sleep(Duration::from_millis(1));
    }

    running.store(false, Ordering::Relaxed);
    reader.join().unwrap();

    let final_data = buffer.read();
    println!("Final value[0]: {}", final_data[0]); // 100
}
```

### Code Example: Sharded Lock Pattern

```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher, DefaultHasher};
use std::sync::RwLock;

/// Improve concurrency via hash-based sharding
struct ShardedMap<K, V> {
    shards: Vec<RwLock<HashMap<K, V>>>,
    num_shards: usize,
}

impl<K: Hash + Eq + Clone, V: Clone> ShardedMap<K, V> {
    fn new(num_shards: usize) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(RwLock::new(HashMap::new()));
        }
        ShardedMap { shards, num_shards }
    }

    fn shard_index(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.num_shards
    }

    fn insert(&self, key: K, value: V) {
        let idx = self.shard_index(&key);
        let mut shard = self.shards[idx].write().unwrap();
        shard.insert(key, value);
    }

    fn get(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let shard = self.shards[idx].read().unwrap();
        shard.get(key).cloned()
    }

    fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write().unwrap();
        shard.remove(key)
    }

    fn len(&self) -> usize {
        self.shards.iter()
            .map(|s| s.read().unwrap().len())
            .sum()
    }
}

fn main() {
    use std::sync::Arc;
    use std::thread;

    let map = Arc::new(ShardedMap::<String, u64>::new(16));
    let mut handles = vec![];

    // Concurrent writes
    for t in 0..8 {
        let map = Arc::clone(&map);
        handles.push(thread::spawn(move || {
            for i in 0..10_000 {
                let key = format!("key_{}_{}", t, i);
                map.insert(key, i as u64);
            }
        }));
    }

    for h in handles { h.join().unwrap(); }
    println!("ShardedMap entry count: {}", map.len()); // 80000

    // Concurrent reads
    let value = map.get(&"key_3_500".to_string());
    println!("key_3_500 = {:?}", value);
}
```

---

## 7. Comparison Tables

### Synchronization Primitive Comparison

| Primitive | Use Case | Overhead | Characteristics |
|---|---|---|---|
| `Mutex<T>` | Exclusive access | Medium | Most general-purpose |
| `RwLock<T>` | Concurrent reads | Medium–High | Suited for read-heavy/write-light |
| `Atomic*` | Single-value operations | Low | Lock-free. Good for counters |
| `Condvar` | Wait/notify | Medium | Combined with Mutex |
| `Barrier` | Sync point | Low | Wait for all threads to arrive |
| `Once` / `OnceLock` | One-time initialization | Lowest | Replacement for lazy_static |
| `mpsc::channel` | Message send/receive | Medium | MPSC only |
| `crossbeam::channel` | Message send/receive | Low–Medium | MPMC, supports select |

### Concurrency Pattern Comparison

| Pattern | When to Use | Implementation in Rust |
|---|---|---|
| Shared memory | Direct sharing of state | Arc<Mutex<T>>, Arc<RwLock<T>> |
| Message passing | Isolated state | mpsc, crossbeam-channel |
| Data parallelism | Same operation on lots of data | rayon par_iter |
| Actor model | Independent entities | tokio::spawn + mpsc |
| Lock-free | Ultra-high frequency access | Atomic*, crossbeam |
| RCU | Read-dominant | Arc + RwLock / AtomicPtr |
| Sharding | Concurrent access to many keys | ShardedMap (DashMap) |
| Double buffer | Read/write separation | Atomic index + buffer pair |

### When to Use Which Ordering

| Ordering | Use Case | Performance | Guarantee |
|---|---|---|---|
| `Relaxed` | Counters, simple flags | Highest | No ordering guarantee |
| `Acquire` | Lock acquisition, data read | High | Subsequent reads/writes don't move before |
| `Release` | Lock release, data publication | High | Preceding reads/writes don't move after |
| `AcqRel` | CAS (compare-and-swap) | Medium | Acquire + Release |
| `SeqCst` | When total order is required | Low | Order is consistent across all threads |

---

## 8. Anti-patterns

### Anti-pattern 1: Lock Granularity Too Coarse

```rust
use std::sync::{Arc, Mutex};

// NG: Protect the whole struct with a single Mutex
struct BadCache {
    data: Mutex<(Vec<String>, std::collections::HashMap<String, String>)>,
}
// → Locks the whole thing even when only data or only index is needed

// OK: Fine-grained locking
struct GoodCache {
    data: Mutex<Vec<String>>,
    index: Mutex<std::collections::HashMap<String, String>>,
}
// → data and index can be locked independently
// * Beware deadlocks: always lock in the same order
```

### Anti-pattern 2: Holding a Mutex Lock During Long Operations

```rust
use std::sync::{Arc, Mutex};

// NG: Network call while holding the lock
fn bad_update(cache: &Mutex<Vec<String>>) {
    let mut data = cache.lock().unwrap();
    let new_item = fetch_from_network(); // long-blocking call!
    data.push(new_item);
}

// OK: Do work outside the lock; minimal lock scope
fn good_update(cache: &Mutex<Vec<String>>) {
    let new_item = fetch_from_network(); // fetched outside the lock
    let mut data = cache.lock().unwrap();
    data.push(new_item);
    // Lock-hold time minimized
}

fn fetch_from_network() -> String { "data".into() }
```

### Anti-pattern 3: Deadlock

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// NG: Reverse lock order → risk of deadlock
fn deadlock_example() {
    let a = Arc::new(Mutex::new(1));
    let b = Arc::new(Mutex::new(2));

    let a1 = Arc::clone(&a);
    let b1 = Arc::clone(&b);
    let t1 = thread::spawn(move || {
        let _a = a1.lock().unwrap(); // order: A → B
        thread::sleep(std::time::Duration::from_millis(10));
        let _b = b1.lock().unwrap();
    });

    let a2 = Arc::clone(&a);
    let b2 = Arc::clone(&b);
    let t2 = thread::spawn(move || {
        let _b = b2.lock().unwrap(); // order: B → A (reversed!) → deadlock!
        thread::sleep(std::time::Duration::from_millis(10));
        let _a = a2.lock().unwrap();
    });

    // t1.join().unwrap();
    // t2.join().unwrap();
}

// OK: Always lock in the same order
fn no_deadlock() {
    let a = Arc::new(Mutex::new(1));
    let b = Arc::new(Mutex::new(2));

    let a1 = Arc::clone(&a);
    let b1 = Arc::clone(&b);
    let t1 = thread::spawn(move || {
        let _a = a1.lock().unwrap(); // always order A → B
        let _b = b1.lock().unwrap();
    });

    let a2 = Arc::clone(&a);
    let b2 = Arc::clone(&b);
    let t2 = thread::spawn(move || {
        let _a = a2.lock().unwrap(); // always order A → B
        let _b = b2.lock().unwrap();
    });

    t1.join().unwrap();
    t2.join().unwrap();
}

fn main() {
    // deadlock_example(); // this may deadlock
    no_deadlock(); // safe
    println!("No deadlock");
}
```

### Anti-pattern 4: Unnecessary Locks / Overuse of Atomics

```rust
use std::sync::atomic::{AtomicU64, Ordering};

// NG: Using Atomic in a single-threaded context
fn single_threaded_bad() {
    let counter = AtomicU64::new(0);
    for _ in 0..1_000_000 {
        counter.fetch_add(1, Ordering::SeqCst); // wasted overhead
    }
}

// OK: A regular variable suffices for single-threaded use
fn single_threaded_good() {
    let mut counter: u64 = 0;
    for _ in 0..1_000_000 {
        counter += 1;
    }
}

// NG: Using SeqCst everywhere (stricter ordering than necessary)
fn overly_strict() {
    let counter = AtomicU64::new(0);
    counter.fetch_add(1, Ordering::SeqCst); // Relaxed is enough for a counter
}

// OK: Choose the Ordering that fits the use case
fn appropriate_ordering() {
    let counter = AtomicU64::new(0);
    counter.fetch_add(1, Ordering::Relaxed); // counter → Relaxed is enough
}

fn main() {
    single_threaded_good();
    appropriate_ordering();
}
```


---

## Hands-on Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate the input data
- Implement appropriate error handling
- Also write tests

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data-processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the impact with benchmarks
---

## FAQ

### Q1: Should I use `Arc<Mutex<T>>` or `Arc<RwLock<T>>`?

**A:** When writes are infrequent and reads are frequent, `RwLock` has the advantage. When writes are frequent or lock-hold times are short, `Mutex` has lower overhead. When in doubt, start with `Mutex`.

### Q2: What's the difference between `parking_lot`'s Mutex and the standard library's?

**A:** `parking_lot::Mutex` (1) has no poisoning (no unwrap needed), (2) is smaller (8 bytes vs 40 bytes), (3) supports a fairness option, and (4) supports `const fn`. It's recommended for performance-critical use cases.

### Q3: When should I use rayon?

**A:** It's ideal for CPU-bound work where data can be split and processed independently. Its appeal is the ease of parallelizing—just change `.iter()` to `.par_iter()`. For I/O-bound work, use tokio.

### Q4: Should I use channels or shared memory?

**A:** Generally, channels (message passing) are preferred. Keeping state confined to each thread and exchanging messages tends to produce fewer bugs. However, when large amounts of data must be shared frequently, shared memory is more efficient.

```rust
// Pattern: when channels fit
// - Pipeline processing (A → B → C)
// - Producer-Consumer
// - Event notification
// - Distributed aggregation

// Pattern: when shared memory fits
// - Frequent reads (caches, configuration)
// - Sharing large amounts of data (DB connection pools)
// - High-frequency counters/metrics
```

### Q5: When should I use `thread::scope` vs `thread::spawn`?

**A:** When you want to reference local variables, `thread::scope` is safer and more convenient. There's no `'static` lifetime requirement, so heap allocation via Arc isn't needed. When you need to manage threads outside of a scope (e.g., background threads), `thread::spawn` is required.

### Q6: What is DashMap?

**A:** `DashMap` is a thread-safe HashMap provided by the `dashmap` crate. It uses sharding internally (the ShardedMap pattern shown earlier) and delivers better concurrency than `Arc<RwLock<HashMap>>`.

```rust
// Cargo.toml: dashmap = "5"
use dashmap::DashMap;

fn main() {
    let map = DashMap::new();
    map.insert("key1", 42);
    map.insert("key2", 100);

    // Read
    if let Some(value) = map.get("key1") {
        println!("key1 = {}", *value);
    }

    // Update
    map.entry("key1").and_modify(|v| *v += 1).or_insert(0);

    // Iterate
    for entry in map.iter() {
        println!("{} = {}", entry.key(), entry.value());
    }
}
```

---

## Summary

| Item | Key Point |
|---|---|
| Send / Sync | Marker traits that prevent data races at compile time |
| thread::scope | Safely pass references to local variables to child threads |
| Mutex | Exclusive lock. The most fundamental synchronization mechanism |
| RwLock | Concurrent-read lock. Suited for read-dominant scenarios |
| Atomic | Lock-free single-value operations. For counters and flags |
| Condvar | Wait on a condition. Producer-Consumer pattern |
| Barrier | Synchronization point for all threads. For phase-based processing |
| OnceLock | Thread-safe one-time initialization |
| mpsc / crossbeam | Channel-based message passing |
| rayon | Easy data parallelism with `.par_iter()`. For CPU-bound work |
| ShardedMap / DashMap | High-concurrency thread-safe HashMap |
| Ordering | Use Relaxed < Acquire/Release < SeqCst as appropriate |

## Recommended Next Reads

- [FFI](./02-ffi-interop.md) — The intersection of thread safety and FFI
- [Tokio Runtime](../02-async/01-tokio-runtime.md) — Concurrent management of async tasks
- [Memory Layout](./00-memory-layout.md) — False sharing and cache lines

## References

1. **Rust Book — Fearless Concurrency**: https://doc.rust-lang.org/book/ch16-00-concurrency.html
2. **Rayon documentation**: https://docs.rs/rayon/latest/rayon/
3. **The Rustonomicon — Concurrency**: https://doc.rust-lang.org/nomicon/concurrency.html
4. **crossbeam documentation**: https://docs.rs/crossbeam/latest/crossbeam/
5. **DashMap documentation**: https://docs.rs/dashmap/latest/dashmap/
