# Threads and Processes

> Concurrent processing is the technology of "advancing multiple tasks simultaneously." Understanding the difference between processes and threads is the starting point for concurrent programming.

## What You Will Learn in This Chapter

- [ ] Understand the difference between processes and threads
- [ ] Grasp the threading model of each language
- [ ] Understand the basics of synchronization primitives
- [ ] Master the causes and avoidance strategies for data races
- [ ] Understand detection and prevention of deadlocks and livelocks
- [ ] Understand thread pools and task-based concurrency
- [ ] Gain practical understanding of OS-level process management


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Concurrency vs. Parallelism

### 1.1 Basic Concepts

```
Concurrency:
  "A structure for managing multiple tasks"
  Can be achieved even on a single CPU core by switching between tasks

  Task A: ──▓▓──────▓▓──────▓▓──
  Task B: ────▓▓──────▓▓──────▓▓

Parallelism:
  "Physically executing multiple tasks at the same time"
  Requires multiple CPU cores

  Core 1: ──▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓──  Task A
  Core 2: ──▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓──  Task B

  Rob Pike: "Concurrency is about dealing with lots of things at once.
             Parallelism is about doing lots of things at once."
```

### 1.2 Relationship Between Concurrency and Parallelism

```
┌─────────────────────────────────────────────────────────────┐
│          Relationship Between Concurrency and Parallelism    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Concurrent but not parallel:                                │
│    Switching between multiple tasks on a single-core CPU     │
│    Example: Node.js event loop                               │
│                                                              │
│  Parallel but not concurrent:                                │
│    Applying the same operation to multiple data items (SIMD) │
│    Example: GPU matrix operations                            │
│                                                              │
│  Both concurrent and parallel:                               │
│    Managing multiple tasks simultaneously on multiple cores  │
│    Example: Multi-threaded web server                        │
│                                                              │
│  Neither:                                                    │
│    Single-threaded sequential processing                     │
│    Example: Simple script                                    │
│                                                              │
│            ┌────────────────────────┐                        │
│            │      Concurrency       │                        │
│            │  ┌─────────────────┐   │                        │
│            │  │ Concurrent +    │   │                        │
│            │  │ Parallel        │   │                        │
│            │  │ (multithreaded) │   │                        │
│  ┌────────┤  └─────────────────┘   │                        │
│  │Parallel│  Concurrent only       │                        │
│  │ (SIMD) │  (event loop)          │                        │
│  └────────┴────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Why Is Concurrent Processing Needed?

```
1. Improved responsiveness:
   - UI does not freeze (heavy processing runs in the background)
   - Web server handles multiple requests simultaneously

2. Improved throughput:
   - Effectively utilize I/O wait time (I/O-bound processing)
   - Efficiently use multi-core CPUs (CPU-bound processing)

3. Efficient resource utilization:
   - Execute other processing during network I/O wait time
   - Overlap disk I/O and computation

I/O-bound vs. CPU-bound:
  I/O-bound: Wait time for network, disk, DB, etc. is dominant
    → Concurrent processing (async/await, threads, goroutines) is effective
    → Other work can proceed during wait time

  CPU-bound: CPU computation is dominant
    → Parallel processing (multiprocess, multithreading) is effective
    → Requires physically using multiple CPU cores
```

---

## 2. Process vs. Thread

### 2.1 Fundamental Differences

```
┌──────────────┬──────────────────┬──────────────────┐
│              │ Process           │ Thread            │
├──────────────┼──────────────────┼──────────────────┤
│ Memory space │ Independent       │ Shared            │
│              │ (isolated)        │                   │
│ Creation cost│ High (ms order)   │ Low (μs order)    │
│ Communication│ IPC (pipes, etc.) │ Shared memory     │
│ Safety       │ High (isolated)   │ Low (race risk)   │
│ Overhead     │ Large             │ Small             │
│ Context      │ All registers +   │ Registers +       │
│ switch       │ page table switch │ stack only        │
│ Fault impact │ Does not affect   │ Affects all       │
│              │ other processes   │ threads in the    │
│              │                   │ same process      │
│ Use case     │ Independent       │ Concurrency within│
│              │ programs          │ a single program  │
└──────────────┴──────────────────┴──────────────────┘
```

### 2.2 Memory Model Details

```
Process:
  ┌─────────────┐  ┌─────────────┐
  │ Process A   │  │ Process B   │
  │ ┌─────────┐ │  │ ┌─────────┐ │
  │ │ Code    │ │  │ │ Code    │ │
  │ │ Data    │ │  │ │ Data    │ │
  │ │ Heap    │ │  │ │ Heap    │ │
  │ │ Stack   │ │  │ │ Stack   │ │
  │ │ FD table│ │  │ │ FD table│ │
  │ └─────────┘ │  │ └─────────┘ │
  └─────────────┘  └─────────────┘
  ← Completely isolated (separate virtual address spaces) →

Thread:
  ┌──────────────────────────────────┐
  │ Process                          │
  │ ┌────────────────────────────┐   │
  │ │ Shared: Code, Data, Heap   │   │
  │ │         FD table, Signals  │   │
  │ └────────────────────────────┘   │
  │ ┌────────┐ ┌────────┐ ┌────────┐│
  │ │Thread 1│ │Thread 2│ │Thread 3││
  │ │Stack   │ │Stack   │ │Stack   ││
  │ │Registers│ │Registers│ │Registers││
  │ │TLS     │ │TLS     │ │TLS     ││
  │ └────────┘ └────────┘ └────────┘│
  └──────────────────────────────────┘
  * TLS = Thread Local Storage
```

### 2.3 Process Details

```
Process lifecycle:

  ┌──────┐    fork()    ┌──────┐
  │ New  │ ──────────→ │ Ready │
  └──────┘              └──┬───┘
                           │ Scheduled
                           ▼
  ┌──────┐   Timeout     ┌──────┐
  │Wait  │ ←──────────── │Running│
  └──┬───┘   / I/O done  └──┬───┘
     │       / Signal        │ exit()
     │ I/O complete          ▼
     └────────────────→ ┌──────┐
                        │Termin.│
                        └──────┘

Process states:
  - New:        Just created
  - Ready:      Waiting for CPU allocation
  - Running:    Executing on CPU
  - Waiting:    Waiting for I/O or event
  - Terminated: Execution complete

Process Control Block (PCB):
  - Process ID (PID)
  - Process state
  - Program counter
  - CPU register contents
  - Memory management info (page table)
  - I/O status info (open files)
  - Accounting info (CPU usage time, etc.)
```

### 2.4 Types of Thread Models

```
1:1 Model (Native threads):
  1 user thread ↔ 1 kernel thread
  Languages: C, Java, Rust, C++
  Advantage: True parallel execution, uses kernel scheduler
  Disadvantage: Relatively high thread creation cost

N:1 Model (Green threads):
  N user threads ↔ 1 kernel thread
  Languages: Early Java (Green Threads), Ruby (Fiber)
  Advantage: Very low thread creation cost
  Disadvantage: Cannot leverage multiple cores

M:N Model (Hybrid):
  M user threads ↔ N kernel threads
  Languages: Go (goroutine), Erlang (process), Java (Virtual Threads 21+)
  Advantage: Lightweight and can leverage multiple cores
  Disadvantage: Scheduler implementation is complex

┌────────────────────────────────────────────────────────┐
│             M:N Model (Go example)                      │
│                                                        │
│  goroutine  goroutine  goroutine  goroutine  goroutine │
│     G1         G2         G3         G4         G5     │
│     │          │          │          │          │      │
│     └────┬─────┘          └────┬─────┘          │      │
│          │                     │                │      │
│      ┌───┴───┐             ┌───┴───┐        ┌───┴───┐ │
│      │  P1   │             │  P2   │        │  P3   │ │
│      │(logical)│           │(logical)│      │(logical)│ │
│      └───┬───┘             └───┬───┘        └───┬───┘ │
│          │                     │                │      │
│      ┌───┴───┐             ┌───┴───┐        ┌───┴───┐ │
│      │  M1   │             │  M2   │        │  M3   │ │
│      │(OS)   │             │(OS)   │        │(OS)   │ │
│      └───────┘             └───────┘        └───────┘ │
│                                                        │
│  G = Goroutine, P = Processor (logical), M = Machine (OS) │
└────────────────────────────────────────────────────────┘
```

---

## 3. Basic Thread Operations

### 3.1 Threads in Python

```python
# Python: threading module
import threading
import time

def worker(name, delay):
    print(f"Thread {name} started")
    time.sleep(delay)  # Simulate I/O
    print(f"Thread {name} finished after {delay}s")

# Creating and running threads
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(f"Worker-{i}", i * 0.5))
    threads.append(t)
    t.start()

for t in threads:
    t.join()  # Wait for all threads to complete

print("All threads completed")

# Note: Python's GIL (Global Interpreter Lock)
# → In CPython, only one thread can execute Python code at a time
# → Use multiprocessing for CPU-intensive tasks
# → threading is effective for I/O-intensive tasks

# Daemon threads
daemon_thread = threading.Thread(target=worker, args=("Daemon", 10), daemon=True)
daemon_thread.start()
# Daemon threads are automatically terminated when the main thread exits

# Thread pool (concurrent.futures)
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_url(url):
    """Fetch data from a URL (simulated)"""
    time.sleep(1)  # Simulate network latency
    return f"Data from {url}"

urls = [
    "https://api.example.com/users",
    "https://api.example.com/posts",
    "https://api.example.com/comments",
    "https://api.example.com/albums",
    "https://api.example.com/photos",
]

# Concurrent execution with a pool of up to 3 threads
with ThreadPoolExecutor(max_workers=3) as executor:
    # submit: Submit tasks individually
    futures = {executor.submit(fetch_url, url): url for url in urls}

    for future in as_completed(futures):
        url = futures[future]
        try:
            data = future.result()
            print(f"{url}: {data}")
        except Exception as e:
            print(f"{url} generated an exception: {e}")

# map: Submit all tasks at once
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch_url, urls))
    for url, data in zip(urls, results):
        print(f"{url}: {data}")
```

### 3.2 Multiprocessing in Python

```python
# Python: multiprocessing (for CPU-intensive tasks, bypassing the GIL)
from multiprocessing import Process, Pool, Queue, Value, Array
import os

def cpu_intensive_task(n):
    """CPU-intensive computation"""
    result = sum(i * i for i in range(n))
    return result

# Direct usage of Process
def worker_process(name):
    pid = os.getpid()
    print(f"Process {name} (PID: {pid}) started")
    result = cpu_intensive_task(10_000_000)
    print(f"Process {name} result: {result}")

processes = []
for i in range(4):
    p = Process(target=worker_process, args=(f"P-{i}",))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

# Process pool
with Pool(processes=4) as pool:
    # map: Execute all tasks synchronously
    results = pool.map(cpu_intensive_task, [1_000_000] * 8)
    print(f"Results: {results}")

    # apply_async: Submit tasks asynchronously
    async_results = [
        pool.apply_async(cpu_intensive_task, (1_000_000,))
        for _ in range(8)
    ]
    results = [r.get(timeout=30) for r in async_results]
    print(f"Async results: {results}")

# Inter-process communication: Queue
def producer(queue):
    for i in range(10):
        queue.put(f"item-{i}")
    queue.put(None)  # Termination signal

def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")

q = Queue()
prod = Process(target=producer, args=(q,))
cons = Process(target=consumer, args=(q,))
prod.start()
cons.start()
prod.join()
cons.join()

# Shared memory: Value, Array
counter = Value('i', 0)  # 'i' = int type
shared_array = Array('d', [0.0] * 10)  # 'd' = double type

def increment(counter, lock):
    for _ in range(100000):
        with lock:
            counter.value += 1

lock = multiprocessing.Lock()
ps = [Process(target=increment, args=(counter, lock)) for _ in range(4)]
for p in ps:
    p.start()
for p in ps:
    p.join()
print(f"Counter: {counter.value}")  # → 400000
```

### 3.3 Threads in Rust

```rust
// Rust: std::thread (OS threads)
use std::thread;
use std::time::Duration;

fn main() {
    // Spawning a thread
    let handle = thread::spawn(|| {
        println!("Hello from thread! ID: {:?}", thread::current().id());
        thread::sleep(Duration::from_millis(100));
        42  // Thread return value
    });

    let result = handle.join().unwrap();  // → 42
    println!("Thread returned: {}", result);

    // Spawning multiple threads
    let handles: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                println!("Thread {} started", i);
                thread::sleep(Duration::from_millis(100 * i as u64));
                println!("Thread {} finished", i);
                i * i
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    println!("Results: {:?}", results);  // [0, 1, 4, 9, 16]

    // Thread builder (setting name and stack size)
    let builder = thread::Builder::new()
        .name("custom-thread".into())
        .stack_size(32 * 1024);  // 32KB stack

    let handle = builder.spawn(|| {
        let name = thread::current().name().unwrap_or("unnamed").to_string();
        println!("Running in thread: {}", name);
    }).unwrap();

    handle.join().unwrap();
}

// Data sharing (Arc + Mutex)
use std::sync::{Arc, Mutex};

fn shared_counter_example() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            // Lock is released when MutexGuard is dropped
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
    println!("Result: {}", *counter.lock().unwrap());  // → 10
}

// RwLock (read-write lock)
use std::sync::RwLock;

fn rwlock_example() {
    let config = Arc::new(RwLock::new(HashMap::new()));

    // Multiple reader threads
    let readers: Vec<_> = (0..5)
        .map(|i| {
            let config = Arc::clone(&config);
            thread::spawn(move || {
                let data = config.read().unwrap();
                println!("Reader {}: {:?}", i, *data);
            })
        })
        .collect();

    // One writer thread
    {
        let config = Arc::clone(&config);
        thread::spawn(move || {
            let mut data = config.write().unwrap();
            data.insert("key", "value");
            println!("Writer: inserted data");
        })
        .join()
        .unwrap();
    }

    for r in readers {
        r.join().unwrap();
    }
}

// Atomic operations (lock-free)
use std::sync::atomic::{AtomicI64, Ordering};

fn atomic_example() {
    let counter = Arc::new(AtomicI64::new(0));
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                for _ in 0..1000 {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    println!("Atomic counter: {}", counter.load(Ordering::Relaxed)); // → 10000
}
```

### 3.4 Goroutines in Go

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "sync/atomic"
    "time"
)

// Go: goroutines (lightweight threads)
func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d started (goroutine)\n", id)
    time.Sleep(time.Second)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    // Set the number of CPU cores to use
    runtime.GOMAXPROCS(runtime.NumCPU())

    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go worker(i, &wg) // Launch a goroutine (go keyword)
    }
    wg.Wait() // Wait for all goroutines to complete

    // Goroutines are far more lightweight than OS threads
    // → Millions of goroutines can run simultaneously
    // → Go runtime performs M:N scheduling
    // → Each goroutine's stack starts at 2KB (grows dynamically)

    // Example of spawning a large number of goroutines
    var wg2 sync.WaitGroup
    count := int64(0)
    for i := 0; i < 100000; i++ {
        wg2.Add(1)
        go func() {
            defer wg2.Done()
            atomic.AddInt64(&count, 1)
        }()
    }
    wg2.Wait()
    fmt.Printf("Count: %d\n", count) // → 100000
}

// Thread-safe counter using Mutex
type SafeCounter struct {
    mu sync.Mutex
    v  map[string]int
}

func (c *SafeCounter) Inc(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.v[key]++
}

func (c *SafeCounter) Value(key string) int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.v[key]
}

// Cache using RWMutex
type Cache struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()         // Read lock (multiple simultaneous reads OK)
    defer c.mu.RUnlock()
    val, ok := c.data[key]
    return val, ok
}

func (c *Cache) Set(key string, value interface{}) {
    c.mu.Lock()          // Write lock (exclusive)
    defer c.mu.Unlock()
    c.data[key] = value
}

// sync.Once: Guarantee execution only once
var (
    instance *Database
    once     sync.Once
)

type Database struct {
    // ...
}

func GetDB() *Database {
    once.Do(func() {
        instance = &Database{}
        fmt.Println("Database initialized")
    })
    return instance
}
```

### 3.5 Threads in Java

```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

// Java: Thread class and Runnable interface
public class ThreadExample {

    // Method 1: Extend the Thread class
    static class MyThread extends Thread {
        @Override
        public void run() {
            System.out.println("Thread: " + getName() + " running");
        }
    }

    // Method 2: Implement the Runnable interface
    static class MyRunnable implements Runnable {
        @Override
        public void run() {
            System.out.println("Runnable running in: " + Thread.currentThread().getName());
        }
    }

    public static void main(String[] args) throws Exception {
        // Thread class
        MyThread t1 = new MyThread();
        t1.start();
        t1.join();

        // Runnable (lambda expression)
        Thread t2 = new Thread(() -> {
            System.out.println("Lambda thread running");
        });
        t2.start();
        t2.join();

        // ExecutorService (thread pool)
        ExecutorService executor = Executors.newFixedThreadPool(4);

        // submit: Returns a Future
        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(1000);
            return 42;
        });
        System.out.println("Result: " + future.get()); // → 42

        // invokeAll: Wait for all tasks to complete
        var tasks = List.of(
            (Callable<String>) () -> { Thread.sleep(100); return "A"; },
            (Callable<String>) () -> { Thread.sleep(200); return "B"; },
            (Callable<String>) () -> { Thread.sleep(300); return "C"; }
        );
        var futures = executor.invokeAll(tasks);
        for (var f : futures) {
            System.out.println(f.get());
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);

        // Java 21+: Virtual Threads (Project Loom)
        // Lightweight virtual threads (similar to goroutines)
        try (var vExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 100000; i++) {
                final int id = i;
                vExecutor.submit(() -> {
                    // Execute in each virtual thread
                    Thread.sleep(Duration.ofMillis(100));
                    return "VThread-" + id;
                });
            }
        }
    }
}

// AtomicInteger: Lock-free counter
class AtomicCounter {
    private final AtomicInteger count = new AtomicInteger(0);

    public void increment() {
        count.incrementAndGet();
    }

    public int get() {
        return count.get();
    }

    // CAS (Compare-And-Swap) operation
    public boolean compareAndSet(int expected, int newValue) {
        return count.compareAndSet(expected, newValue);
    }
}
```

### 3.6 Threads in C++

```cpp
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <vector>
#include <iostream>

// C++11 and later: std::thread
void basic_thread_example() {
    // Thread creation
    std::thread t([] {
        std::cout << "Hello from thread " << std::this_thread::get_id() << std::endl;
    });
    t.join(); // Wait for completion

    // detach: Detach the thread (make it a daemon thread)
    std::thread daemon([] {
        // Background processing...
    });
    daemon.detach();
}

// Mutual exclusion with Mutex
std::mutex mtx;
int shared_counter = 0;

void increment_with_mutex() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx); // RAII lock
        shared_counter++;
    } // lock_guard's destructor unlocks
}

// shared_mutex (read-write lock, C++17)
std::shared_mutex rw_mutex;
std::map<std::string, std::string> cache;

std::string read_cache(const std::string& key) {
    std::shared_lock lock(rw_mutex); // Read lock
    auto it = cache.find(key);
    return it != cache.end() ? it->second : "";
}

void write_cache(const std::string& key, const std::string& value) {
    std::unique_lock lock(rw_mutex); // Write lock
    cache[key] = value;
}

// std::async and std::future
void async_example() {
    // Execute an async task
    auto future = std::async(std::launch::async, [] {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 42;
    });

    // Do other work...
    std::cout << "Waiting for result..." << std::endl;

    int result = future.get(); // Retrieve result (blocking)
    std::cout << "Result: " << result << std::endl;
}

// Atomic operations
std::atomic<int> atomic_counter{0};

void atomic_increment() {
    for (int i = 0; i < 100000; i++) {
        atomic_counter.fetch_add(1, std::memory_order_relaxed);
    }
}
```

---

## 4. Synchronization Primitives

### 4.1 Basic Synchronization Primitives

```
Mutex (Mutual Exclusion):
  Only one thread can access the resource at a time

  Thread 1: lock() → [in use] → unlock()
  Thread 2: lock() → [waiting...] → [in use] → unlock()

  Use case: Exclusive access to shared resources
  Caution: Forgetting to lock causes data races; forgetting to unlock causes deadlocks

RWLock (Read-Write Lock):
  Multiple simultaneous reads allowed, writes are exclusive

  Reader 1: read_lock() → [reading] → unlock()
  Reader 2: read_lock() → [reading] → unlock()  ← simultaneous OK
  Writer:   write_lock() → [waiting...] → [writing] → unlock()

  Use case: Data that is read frequently but written infrequently (config, cache, etc.)

Semaphore:
  A counter that limits the number of simultaneous accesses
  acquire() decrements the counter (waits if 0)
  release() increments the counter

  Example: Maximum 5 simultaneous DB connections
  Semaphore(5):
    Thread 1: acquire() → [connected] → release()
    Thread 2: acquire() → [connected] → release()
    ...
    Thread 6: acquire() → [waiting...] → (until a slot opens)

Condition Variable:
  Makes a thread wait until a specific condition is met
  wait(): Wait until the condition is met
  notify_one(): Wake up one waiting thread
  notify_all(): Wake up all waiting threads

  Use case: Producer-Consumer pattern, event waiting

Barrier:
  All threads wait until a specified number have arrived

  Thread 1: ──processing── barrier.wait() ──processing──
  Thread 2: ──processing── barrier.wait() ──processing──
  Thread 3: ──processing── barrier.wait() ──processing──
                           ↑ All wait until everyone arrives
```

### 4.2 Synchronization in TypeScript / JavaScript

```typescript
// JavaScript is single-threaded but synchronization of concurrent processing is needed

// Semaphore implementation
class Semaphore {
    private queue: Array<() => void> = [];
    private count: number;

    constructor(maxConcurrency: number) {
        this.count = maxConcurrency;
    }

    async acquire(): Promise<void> {
        if (this.count > 0) {
            this.count--;
            return;
        }
        return new Promise<void>((resolve) => {
            this.queue.push(resolve);
        });
    }

    release(): void {
        const next = this.queue.shift();
        if (next) {
            next();
        } else {
            this.count++;
        }
    }
}

// Usage example: Limit concurrency of API calls
async function fetchWithLimit(urls: string[], maxConcurrency: number) {
    const semaphore = new Semaphore(maxConcurrency);
    const results = await Promise.all(
        urls.map(async (url) => {
            await semaphore.acquire();
            try {
                const response = await fetch(url);
                return response.json();
            } finally {
                semaphore.release();
            }
        })
    );
    return results;
}

// Mutex implementation (async/await based)
class AsyncMutex {
    private locked = false;
    private queue: Array<() => void> = [];

    async lock(): Promise<() => void> {
        if (!this.locked) {
            this.locked = true;
            return () => this.unlock();
        }
        return new Promise<() => void>((resolve) => {
            this.queue.push(() => {
                resolve(() => this.unlock());
            });
        });
    }

    private unlock(): void {
        const next = this.queue.shift();
        if (next) {
            next();
        } else {
            this.locked = false;
        }
    }
}

// Web Workers (parallel processing in the browser)
// main.js
const worker = new Worker("worker.js");
worker.postMessage({ data: [1, 2, 3, 4, 5], operation: "sum" });
worker.onmessage = (event) => {
    console.log("Result from worker:", event.data);
};

// worker.js
// self.onmessage = (event) => {
//     const { data, operation } = event.data;
//     if (operation === "sum") {
//         const result = data.reduce((a, b) => a + b, 0);
//         self.postMessage(result);
//     }
// };

// SharedArrayBuffer and Atomics (shared memory)
const buffer = new SharedArrayBuffer(1024);
const view = new Int32Array(buffer);

// Main thread
Atomics.store(view, 0, 42);
Atomics.notify(view, 0);

// Worker thread
// Atomics.wait(view, 0, 0); // Wait until value is no longer 0
// const value = Atomics.load(view, 0); // → 42
```

### 4.3 Condition Variables in Practice

```python
import threading
from collections import deque

# Producer-Consumer pattern (using condition variables)
class BoundedBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def produce(self, item):
        with self.not_full:
            while len(self.buffer) >= self.capacity:
                self.not_full.wait()  # Wait until buffer has space
            self.buffer.append(item)
            self.not_empty.notify()   # Wake up consumer

    def consume(self):
        with self.not_empty:
            while len(self.buffer) == 0:
                self.not_empty.wait()  # Wait until an item arrives
            item = self.buffer.popleft()
            self.not_full.notify()     # Wake up producer
            return item

# Usage example
buffer = BoundedBuffer(capacity=5)

def producer():
    for i in range(20):
        buffer.produce(f"item-{i}")
        print(f"Produced: item-{i}")

def consumer(name):
    for _ in range(10):
        item = buffer.consume()
        print(f"Consumer {name} consumed: {item}")

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer, args=("A",))
t3 = threading.Thread(target=consumer, args=("B",))

t1.start(); t2.start(); t3.start()
t1.join(); t2.join(); t3.join()
```

---

## 5. Data Races and Avoidance Strategies

### 5.1 Principles of Data Races

```
Data Race:
  Two or more threads access the same memory simultaneously,
  at least one is writing, and there is no synchronization

  Thread 1: read(x)=0 → write(x)=1
  Thread 2: read(x)=0 → write(x)=1
  Result: x=1 (expected x=2)

Specific race scenario:

  Time →
  Thread 1:  read x (=0)  →  compute 0+1  →  write x (=1)
  Thread 2:       read x (=0)  →  compute 0+1  →  write x (=1)

  Expected: x = 2
  Actual:   x = 1 (Thread 2 overwrites Thread 1's write)

  This is the "TOCTOU (Time of Check to Time of Use)" problem

Avoidance strategies:
  1. Use Mutex for exclusive access control
  2. Use atomic operations
  3. Don't share data (message passing)
  4. Use immutable data (functional approach)
  5. Prevent at compile time with the type system (Rust)
```

### 5.2 Compile-Time Data Race Prevention in Rust

```rust
// Rust: Prevents data races at compile time
use std::thread;
use std::sync::{Arc, Mutex};

fn main() {
    let mut data = vec![1, 2, 3];

    // Compile error: Cannot pass a mutable reference to multiple threads
    // thread::spawn(|| { data.push(4); });
    // thread::spawn(|| { data.push(5); });
    // Error: `data` does not implement `Send`
    // → Rust's ownership system prevents data races at compile time

    // Safe sharing with Arc<Mutex<T>>
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let d1 = Arc::clone(&data);
    let d2 = Arc::clone(&data);

    let h1 = thread::spawn(move || {
        d1.lock().unwrap().push(4);
    });
    let h2 = thread::spawn(move || {
        d2.lock().unwrap().push(5);
    });

    h1.join().unwrap();
    h2.join().unwrap();
    println!("{:?}", data.lock().unwrap()); // [1, 2, 3, 4, 5] or [1, 2, 3, 5, 4]

    // Send and Sync traits
    // Send: The value can be moved to another thread
    // Sync: A reference can be shared with another thread (&T is Send)
    //
    // Most types are Send + Sync
    // Rc<T> is NOT Send (not thread-safe)
    // → Use Arc<T> (atomic reference counting)
    //
    // Cell<T>, RefCell<T> are NOT Sync
    // → Use Mutex<T>, RwLock<T>
}
```

### 5.3 Common Concurrency Bugs

```
1. TOCTOU (Time of Check to Time of Use):
   if (file.exists()) {     // At check time
       file.read();          // At use time ← another thread may delete it in between
   }

2. ABA Problem:
   Thread 1: read A → (pause)
   Thread 2: A → B → A change
   Thread 1: read A → mistakenly judges "unchanged"
   → Cannot be detected by CAS (Compare-And-Swap)

3. Memory Visibility Problem:
   Each CPU core has its own cache
   Even if Thread 1 writes to memory,
   Thread 2's cache may not reflect the change
   → Solve with memory barriers / volatile / Atomic operations

4. False Sharing:
   Different threads access different variables on the same cache line
   → Cache invalidation occurs frequently, degrading performance
   → Separate cache lines with padding
```

---

## 6. Deadlocks and Their Avoidance

### 6.1 Conditions for Deadlock

```
Deadlock: A state where two or more threads wait indefinitely
          for each other to release their locks

  Thread 1: lock(A) → waiting for lock(B) → waits forever
  Thread 2: lock(B) → waiting for lock(A) → waits forever

  ┌──────────┐  waiting for lock(B) ┌──────────┐
  │ Thread 1 │ ────────────────────→│ Thread 2 │
  │ (holds A)│ ←────────────────────│ (holds B)│
  └──────────┘  waiting for lock(A) └──────────┘

Four Coffman conditions for deadlock:
  1. Mutual exclusion: A resource can only be used by one thread at a time
  2. Hold and wait: A thread holds a resource while waiting for another
  3. No preemption: Resources cannot be forcibly taken from a thread
  4. Circular wait: Threads form a circular chain of waiting

→ Breaking even one condition prevents deadlock
```

### 6.2 Deadlock Avoidance Strategies

```python
import threading

# Deadlock example
lock_a = threading.Lock()
lock_b = threading.Lock()

# BAD: Potential deadlock
def thread1_bad():
    lock_a.acquire()
    # time.sleep(0.001)  # Deadlock depending on timing
    lock_b.acquire()
    # Processing...
    lock_b.release()
    lock_a.release()

def thread2_bad():
    lock_b.acquire()  # Acquires locks in reverse order
    lock_a.acquire()
    # Processing...
    lock_a.release()
    lock_b.release()

# Strategy 1: Consistent lock ordering
def thread1_good():
    lock_a.acquire()  # Always A → B order
    lock_b.acquire()
    # Processing...
    lock_b.release()
    lock_a.release()

def thread2_good():
    lock_a.acquire()  # Always A → B order (consistent)
    lock_b.acquire()
    # Processing...
    lock_b.release()
    lock_a.release()

# Strategy 2: Lock with timeout
def thread_with_timeout():
    acquired_a = lock_a.acquire(timeout=1.0)
    if not acquired_a:
        return  # Timeout → retry

    acquired_b = lock_b.acquire(timeout=1.0)
    if not acquired_b:
        lock_a.release()  # Release held lock too
        return  # Timeout → retry

    try:
        # Processing...
        pass
    finally:
        lock_b.release()
        lock_a.release()

# Strategy 3: Safe lock management with context manager
from contextlib import contextmanager

@contextmanager
def acquire_locks(*locks):
    """Acquire multiple locks in sorted order"""
    sorted_locks = sorted(locks, key=id)
    acquired = []
    try:
        for lock in sorted_locks:
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()

# Usage example
def safe_thread():
    with acquire_locks(lock_a, lock_b):
        # Processing... (lock order is automatically consistent)
        pass
```

### 6.3 Livelocks and Starvation

```
Livelock:
  A state where threads try to avoid deadlock
  but keep yielding to each other indefinitely
  (Like two people in a hallway trying to pass but always stepping the same way)

  Thread 1: lock(A) → try lock(B) → fail → unlock(A) → retry
  Thread 2: lock(B) → try lock(A) → fail → unlock(B) → retry
  → Retries forever

  Countermeasure: Random backoff (wait a random time before retrying)

Starvation:
  A state where a specific thread can never acquire a resource
  Example: A low-priority thread is always deprioritized

  Countermeasures:
  - Locks that guarantee fairness (FIFO queue)
  - Protocols to prevent priority inversion
  - Timeouts and retries
```

---

## 7. Thread Pools and Task-Based Concurrency

### 7.1 Thread Pool Concept

```
Thread Pool:
  Pre-create threads and reuse them by submitting tasks

  ┌──────────────────────────────────────┐
  │ Thread Pool                          │
  │                                      │
  │  ┌─────────────┐  ┌──────────────┐  │
  │  │ Task Queue   │  │ Worker Pool  │  │
  │  │              │  │              │  │
  │  │ [Task 1] ─────→ │ Thread 1 ○  │  │
  │  │ [Task 2] ─────→ │ Thread 2 ○  │  │
  │  │ [Task 3]    │  │ Thread 3 ●  │  │
  │  │ [Task 4]    │  │ Thread 4 ●  │  │
  │  │ ...         │  │              │  │
  │  └─────────────┘  └──────────────┘  │
  │  ○ = idle  ● = running              │
  └──────────────────────────────────────┘

Benefits:
  - Reduced thread creation cost (reuse)
  - Control over concurrent execution count
  - Efficient resource utilization
  - Priority-based task scheduling
```

### 7.2 Rust's rayon (Data Parallelism)

```rust
// rayon: Rust's data parallelism library
use rayon::prelude::*;

fn main() {
    let numbers: Vec<i64> = (0..10_000_000).collect();

    // Parallel map + sum
    let sum: i64 = numbers.par_iter()
        .map(|&n| n * n)
        .sum();

    // Parallel filter + collect
    let evens: Vec<i64> = numbers.par_iter()
        .filter(|&&n| n % 2 == 0)
        .copied()
        .collect();

    // Parallel sort
    let mut data = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
    data.par_sort();

    // Custom thread pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let result: i64 = numbers.par_iter().sum();
        println!("Sum: {}", result);
    });

    // Parallel for_each
    (0..100).into_par_iter().for_each(|i| {
        // Each iteration runs in parallel
        println!("Processing: {}", i);
    });
}
```

### 7.3 Work Stealing

```
Work Stealing Algorithm:
  Each worker thread has a local queue,
  and when its own queue is empty, it "steals" tasks from other threads' queues

  Thread 1: [T1, T2, T3] → Execute T1
  Thread 2: [T4]         → Execute T4
  Thread 3: []           → Steal T3 from Thread 1

  Benefits:
  - Load balancing happens automatically
  - Local queue access requires no locking (deque double-ended operations)
  - Stealing occurs rarely, so overhead is small

  Languages/libraries that use it:
  - Go's runtime scheduler
  - Rust's rayon
  - Java's ForkJoinPool
  - .NET's Task Parallel Library (TPL)
  - Tokio (Rust's async runtime)
```

---

## 8. Inter-Process Communication (IPC)

### 8.1 Types of IPC

```
┌────────────────┬───────────────────────────────────────┐
│ IPC Method      │ Characteristics                       │
├────────────────┼───────────────────────────────────────┤
│ Pipe           │ Unidirectional, parent-child, byte    │
│                │ stream                                │
│ Named pipe     │ Bidirectional possible, non-parent-   │
│                │ child processes                       │
│ Socket         │ Can work across networks, bidirectional│
│ Shared memory  │ Fastest, requires synchronization     │
│ Message queue  │ Asynchronous, message-based           │
│ Signal         │ Asynchronous notification, no payload │
│ File           │ Simplest, low performance             │
│ mmap           │ Maps a file to memory                 │
│ Unix domain    │ Same machine only, faster than TCP/IP │
│ socket         │                                       │
└────────────────┴───────────────────────────────────────┘
```

### 8.2 Examples of Pipes and Sockets

```python
import subprocess
import socket
import os

# Pipe: Communication with subprocess
result = subprocess.run(
    ["ls", "-la"],
    capture_output=True,
    text=True,
)
print(result.stdout)

# Pipe chain: ls | grep ".py" | wc -l
p1 = subprocess.Popen(["ls"], stdout=subprocess.PIPE)
p2 = subprocess.Popen(["grep", ".py"], stdin=p1.stdout, stdout=subprocess.PIPE)
p3 = subprocess.Popen(["wc", "-l"], stdin=p2.stdout, stdout=subprocess.PIPE)
p1.stdout.close()
p2.stdout.close()
output = p3.communicate()[0]
print(f"Python files: {output.decode().strip()}")

# os.pipe(): Low-level pipe
read_fd, write_fd = os.pipe()

pid = os.fork()
if pid == 0:
    # Child process
    os.close(read_fd)
    os.write(write_fd, b"Hello from child!")
    os.close(write_fd)
    os._exit(0)
else:
    # Parent process
    os.close(write_fd)
    message = os.read(read_fd, 1024)
    os.close(read_fd)
    os.waitpid(pid, 0)
    print(f"Parent received: {message.decode()}")

# Unix domain socket
SOCKET_PATH = "/tmp/example.sock"

# Server
def server():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(SOCKET_PATH)
    server_socket.listen(1)

    conn, _ = server_socket.accept()
    data = conn.recv(1024)
    print(f"Server received: {data.decode()}")
    conn.send(b"Response from server")
    conn.close()
    server_socket.close()

# Client
def client():
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_socket.connect(SOCKET_PATH)
    client_socket.send(b"Hello from client")
    response = client_socket.recv(1024)
    print(f"Client received: {response.decode()}")
    client_socket.close()
```

---

## 9. Context Switching and Scheduling

### 9.1 Context Switching

```
Context Switch:
  The operation of saving the state of the currently running task
  and restoring the state of the next task when the CPU switches tasks

  Process context switch:
    1. Save CPU registers
    2. Save program counter
    3. Switch page table (TLB flush)
    4. Invalidate cache (partially)
    → Cost: Several microseconds

  Thread context switch:
    1. Save CPU registers
    2. Switch stack pointer
    → Cost: Hundreds of nanoseconds (no page table switch)

  Goroutine context switch:
    1. Save a small number of registers
    2. Switch stack pointer
    → Cost: Tens of nanoseconds (no kernel involvement)

Cost comparison:
  Process switch:   ~5-10 μs
  Thread switch:    ~0.5-1 μs
  Goroutine switch: ~0.01-0.1 μs
  Async task switch: ~0.001-0.01 μs
```

### 9.2 Scheduling Algorithms

```
OS Scheduling:
  1. FCFS (First-Come, First-Served): Arrival order
  2. SJF (Shortest Job First): Shortest job priority
  3. Priority scheduling: Based on priority
  4. Round Robin: Fair switching with time slices
  5. CFS (Completely Fair Scheduler): Linux default
     - Tracks the virtual execution time of each task
     - Executes the task with the least execution time next
     - Managed with a red-black tree (O(log n))

Go's Scheduler:
  GMP Model:
  - G (Goroutine): Execution unit
  - M (Machine): OS thread
  - P (Processor): Logical processor

  Features:
  - Load balancing with work stealing
  - Preemptive (Go 1.14+)
  - Detects blocking on network I/O and moves
    the goroutine to a different OS thread
```

---

## 10. Practical Concurrency Patterns

### 10.1 Worker Pool Pattern

```go
package main

import (
    "fmt"
    "sync"
)

// Worker Pool pattern
func workerPool() {
    const numWorkers = 4
    jobs := make(chan int, 100)
    results := make(chan int, 100)

    // Start workers
    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for job := range jobs {
                result := job * job // Processing
                results <- result
            }
        }(w)
    }

    // Submit jobs
    go func() {
        for i := 0; i < 20; i++ {
            jobs <- i
        }
        close(jobs)
    }()

    // Collect results
    go func() {
        wg.Wait()
        close(results)
    }()

    for result := range results {
        fmt.Println(result)
    }
}
```

### 10.2 Fan-out / Fan-in Pattern

```go
// Fan-out: Distribute one input to multiple workers
// Fan-in: Merge output from multiple workers into one

func fanOutFanIn() {
    // Input channel
    input := make(chan int)
    go func() {
        for i := 0; i < 100; i++ {
            input <- i
        }
        close(input)
    }()

    // Fan-out: Distribute to 3 workers
    numWorkers := 3
    workers := make([]<-chan int, numWorkers)
    for i := 0; i < numWorkers; i++ {
        workers[i] = worker(input)
    }

    // Fan-in: Merge output from all workers
    merged := merge(workers...)

    for result := range merged {
        fmt.Println(result)
    }
}

func worker(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for n := range input {
            output <- n * n // Processing
        }
    }()
    return output
}

func merge(channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for v := range c {
                merged <- v
            }
        }(ch)
    }

    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}
```

### 10.3 Pipeline Pattern

```go
// Pipeline: Data passes through multiple stages sequentially

func pipeline() {
    // Stage 1: Generate numbers
    gen := func(nums ...int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for _, n := range nums {
                out <- n
            }
        }()
        return out
    }

    // Stage 2: Square
    square := func(in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for n := range in {
                out <- n * n
            }
        }()
        return out
    }

    // Stage 3: Filter (even numbers only)
    filterEven := func(in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for n := range in {
                if n%2 == 0 {
                    out <- n
                }
            }
        }()
        return out
    }

    // Connect the pipeline
    result := filterEven(square(gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))

    for v := range result {
        fmt.Println(v) // 4, 16, 36, 64, 100
    }
}
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important aspect. Understanding deepens not just through theory but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It is particularly important during code reviews and architecture design.

---

## Summary

| Concept | Description | Representative Languages |
|------|------|---------|
| OS threads | Kernel-managed, heavy | C, Java, Rust |
| Lightweight threads | Runtime-managed, light | Go (goroutine), Erlang |
| Virtual Threads | Lightweight threads on JVM | Java 21+ |
| Mutex | Mutual exclusion | All languages |
| RWLock | Read-write lock | All languages |
| Semaphore | Limit concurrent access count | All languages |
| Condition Variable | Conditional wait and notification | All languages |
| Arc / shared ownership | Share data between threads | Rust |
| Atomic | Lock-free synchronization | C++, Rust, Java |
| GIL | Python constraint | CPython |
| Thread Pool | Thread reuse | Java, Python, C++ |
| Work Stealing | Dynamic load balancing | Go, Rust (rayon), Java |

Principles for correctly implementing concurrent processing:

1. **Minimize shared mutable state**: Prefer immutable data and message passing
2. **Set appropriate lock granularity**: Too coarse reduces throughput; too fine increases deadlock risk
3. **Unify lock ordering**: The most fundamental measure to prevent deadlocks
4. **Manage locks with RAII**: Prevent forgetting to release locks
5. **Choose the right concurrency model**: async for I/O-bound, threads/processes for CPU-bound
6. **Testing and debugging**: Leverage ThreadSanitizer and race detection tools

---

## Recommended Next Guides

---

## References
1. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." 2nd Ed, 2020.
2. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.16, 2023.
3. Goetz, B. "Java Concurrency in Practice." Addison-Wesley, 2006.
4. Pike, R. "Concurrency is not Parallelism." Go Blog, 2012.
5. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Wiley, 2018.
6. Love, R. "Linux Kernel Development." 3rd Ed, Addison-Wesley, 2010.
7. Butcher, P. "Seven Concurrency Models in Seven Weeks." Pragmatic, 2014.
8. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
