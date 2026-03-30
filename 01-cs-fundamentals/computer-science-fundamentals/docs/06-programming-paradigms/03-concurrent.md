# Concurrent and Parallel Programming

> Concurrency is about dealing with lots of things at once. Parallelism is about doing lots of things at once. —Rob Pike

## What You Will Learn in This Chapter

- [ ] Clearly explain the difference between concurrency and parallelism
- [ ] Understand the causes of and countermeasures for deadlocks, race conditions, and starvation
- [ ] Understand the mechanics of async/await and event loops
- [ ] Understand Go's goroutine/channel (CSP model)
- [ ] Understand the principles and use cases of the actor model
- [ ] Know the concepts behind lock-free algorithms
- [ ] Master practical concurrency patterns and anti-patterns


## Prerequisites

Before reading this guide, familiarity with the following will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Familiarity with the content in [Functional Programming](./02-functional.md)

---

## 1. Concurrency vs Parallelism

### 1.1 Clarifying the Basic Concepts

```
Concurrency: Managing multiple tasks "logically" at the same time
  → Achievable even with a single CPU core (time slicing)
  → A structural concern: "How do we organize multiple tasks?"
  → A state where the execution periods of multiple tasks overlap

Parallelism: Executing multiple tasks "physically" at the same time
  → Requires multiple CPU cores
  → An execution concern: "How do we physically run things simultaneously?"
  → A state where multiple tasks are executing at the same instant

  Analogy:

  Concurrency: One chef cooking multiple dishes by switching between them
              Preparing sauce while pasta is boiling,
              arranging a salad while sauce is simmering
              → A single person can efficiently handle multiple tasks

  Parallelism: Multiple chefs cooking different dishes simultaneously
              Chef A handles pasta, Chef B handles salad, Chef C handles dessert
              → Physical simultaneous execution

  Key insight: Concurrency is a superset that encompasses parallelism
  Parallelism is one form of concurrency (concurrent processing that
  happens to execute physically at the same time)

  ┌─────────────────────────────────┐
  │         Concurrency             │
  │  ┌───────────────────────┐     │
  │  │     Parallelism       │     │
  │  └───────────────────────┘     │
  │  Async I/O, Coroutines, etc.   │
  └─────────────────────────────────┘
```

### 1.2 Why Do We Need Concurrent Processing?

```
Reasons why concurrent processing is necessary:

1. Effective use of I/O wait time
   - Network communication: milliseconds to seconds
   - Disk I/O: milliseconds
   - User input: seconds to minutes
   → Keep the CPU busy by performing other work during wait times

2. Ensuring responsiveness
   - GUI applications: Don't block the main thread
   - Web servers: Accept other requests while processing one
   - Games: Manage rendering, physics, AI, and input simultaneously

3. Improving throughput
   - Utilizing multi-core CPUs
   - Accelerating batch processing
   - Handling large numbers of concurrent requests

4. Real-time requirements
   - Continuous monitoring of sensor data
   - Stream processing (log aggregation, event processing)
   - Real-time communication (chat, video calls)

Performance comparison example (web scraping 100 pages):
┌──────────────────┬──────────────┬──────────────┐
│ Approach          │ Time         │ Notes        │
├──────────────────┼──────────────┼──────────────┤
│ Sequential        │ ~100 sec     │ 1 sec/page   │
│ Multi-threaded(10)│ ~10 sec      │ 10 concurrent│
│ async/await       │ ~2-3 sec     │ 100 conc. I/O│
│ Multi-process(10) │ ~10 sec      │ CPU-limited  │
└──────────────────┴──────────────┴──────────────┘
```

### 1.3 Differences Between Processes, Threads, and Coroutines

```
Units of concurrent execution:

┌──────────────┬───────────────┬──────────────┬──────────────┐
│              │ Process       │ Thread       │ Coroutine    │
├──────────────┼───────────────┼──────────────┼──────────────┤
│ Memory space │ Isolated      │ Shared       │ Shared       │
│ Creation cost│ High          │ Medium       │ Low          │
│ Context      │ Heavy         │ Medium       │ Light        │
│ switch       │               │              │              │
│ Communication│ IPC           │ Shared memory│ Function call│
│              │ (pipes, etc.) │ + locks      │ + yield      │
│ Parallelism  │ ✅ True       │ ✅ True*     │ ❌ Concurrency│
│              │   parallel    │   parallel   │   only       │
│ Safety       │ High          │ Low (races)  │ High (single)│
│              │ (isolation)   │              │              │
│ Scalability  │ ~hundreds     │ ~thousands   │ Millions     │
│              │               │              │ possible     │
│ Examples     │ multiprocessing│ threading   │ asyncio      │
│              │ fork          │ pthread      │ goroutine    │
│ Best for     │ CPU-bound     │ General      │ I/O-bound   │
└──────────────┴───────────────┴──────────────┴──────────────┘

* Due to Python's GIL (Global Interpreter Lock),
  CPython threads cannot achieve true parallel CPU execution
```

---

## 2. Concurrency Problems and Solutions

### 2.1 Race Conditions

```python
import threading
import time

# === Race Condition Demo ===

# ❌ Code that causes a race condition
counter = 0

def increment_unsafe():
    """Unsafe increment"""
    global counter
    for _ in range(100000):
        counter += 1
        # Internal operations:
        # 1. Read the value of counter (read)
        # 2. Increment the value by 1 (increment)
        # 3. Write the value back to counter (write)
        # → If two threads execute step 1 simultaneously,
        #   one increment is lost!

# Execute with 2 threads
threads = [threading.Thread(target=increment_unsafe) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Expected: 200000, Actual: {counter}")
# → The actual value is often less than 200000 (e.g., 183421)


# ✅ Solution 1: Lock (Mutex)
counter = 0
lock = threading.Lock()

def increment_with_lock():
    """Increment protected by a lock"""
    global counter
    for _ in range(100000):
        with lock:  # Mutual exclusion: only 1 thread executes
            counter += 1

threads = [threading.Thread(target=increment_with_lock) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"With lock: {counter}")  # Always 200000


# ✅ Solution 2: Thread-safe data structures
import queue

def producer(q: queue.Queue, items: list):
    """Producer: puts items into the queue"""
    for item in items:
        q.put(item)
        time.sleep(0.01)
    q.put(None)  # Termination signal

def consumer(q: queue.Queue, name: str):
    """Consumer: retrieves items from the queue"""
    while True:
        item = q.get()  # Blocking (thread-safe)
        if item is None:
            q.put(None)  # Propagate termination to other consumers
            break
        print(f"[{name}] Processing: {item}")
        q.task_done()

# Communicate via a thread-safe queue
q = queue.Queue(maxsize=10)
prod = threading.Thread(target=producer, args=(q, list(range(20))))
cons1 = threading.Thread(target=consumer, args=(q, "Consumer-1"))
cons2 = threading.Thread(target=consumer, args=(q, "Consumer-2"))

prod.start()
cons1.start()
cons2.start()

prod.join()
cons1.join()
cons2.join()
```

```java
// Race condition solutions in Java

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class ConcurrencyDemo {

    // ✅ Solution 1: synchronized keyword
    private int counter = 0;

    public synchronized void incrementSync() {
        counter++;
    }

    // ✅ Solution 2: AtomicInteger (CAS operation)
    private AtomicInteger atomicCounter = new AtomicInteger(0);

    public void incrementAtomic() {
        atomicCounter.incrementAndGet();
        // Compare-And-Swap: fast without locks
    }

    // ✅ Solution 3: ReentrantLock (explicit lock)
    private final ReentrantLock lock = new ReentrantLock();
    private int lockedCounter = 0;

    public void incrementWithLock() {
        lock.lock();
        try {
            lockedCounter++;
        } finally {
            lock.unlock();  // Always release
        }
    }

    // ✅ Solution 4: Concurrent collections
    // ConcurrentHashMap, CopyOnWriteArrayList, BlockingQueue, etc.
    // → Thread-safe collection implementations provided in the standard library
}
```

### 2.2 Deadlock

```python
import threading
import time

# === Deadlock Demo ===

lock_a = threading.Lock()
lock_b = threading.Lock()

# ❌ Code that causes a deadlock
def thread_1():
    with lock_a:
        print("Thread-1: acquired lock_a")
        time.sleep(0.1)  # While waiting for lock_b...
        with lock_b:     # Thread-2 holds lock_b!
            print("Thread-1: acquired lock_b")

def thread_2():
    with lock_b:
        print("Thread-2: acquired lock_b")
        time.sleep(0.1)  # While waiting for lock_a...
        with lock_a:     # Thread-1 holds lock_a!
            print("Thread-2: acquired lock_a")

# → Thread-1 waits for lock_b, Thread-2 waits for lock_a
# → Both wait forever = deadlock


# ✅ Solution 1: Enforce a consistent lock acquisition order
def thread_1_fixed():
    with lock_a:       # Always acquire lock_a → lock_b in order
        print("Thread-1: acquired lock_a")
        with lock_b:
            print("Thread-1: acquired lock_b")

def thread_2_fixed():
    with lock_a:       # Always acquire lock_a → lock_b in order (consistent)
        print("Thread-2: acquired lock_a")
        with lock_b:
            print("Thread-2: acquired lock_b")


# ✅ Solution 2: Lock acquisition with timeout
def thread_with_timeout():
    acquired_a = lock_a.acquire(timeout=1.0)  # Timeout after 1 second
    if not acquired_a:
        print("lock_a acquisition timed out, retrying")
        return False

    try:
        acquired_b = lock_b.acquire(timeout=1.0)
        if not acquired_b:
            print("lock_b acquisition timed out, releasing and retrying")
            return False
        try:
            # Critical section
            pass
        finally:
            lock_b.release()
    finally:
        lock_a.release()
    return True


# ✅ Solution 3: Safe lock management with a context manager
import contextlib

@contextlib.contextmanager
def acquire_locks(*locks, timeout=5.0):
    """Context manager for safely acquiring multiple locks"""
    acquired = []
    try:
        for lock in sorted(locks, key=id):  # Unify acquisition order by id
            if lock.acquire(timeout=timeout):
                acquired.append(lock)
            else:
                raise TimeoutError(f"Lock acquisition timed out")
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()

# Usage example
# with acquire_locks(lock_a, lock_b):
#     # Safely execute the critical section
#     pass
```

```
Coffman Conditions for Deadlock:
Deadlock occurs when all four conditions are met simultaneously

1. Mutual Exclusion
   A resource can only be used by one process at a time

2. Hold and Wait
   A process holds resources while waiting for additional ones

3. No Preemption
   Resources held by a process cannot be forcibly taken away

4. Circular Wait
   Processes are waiting for resources in a circular chain

Prevention strategies (breaking any one condition is sufficient):
┌──────────────┬─────────────────────────────────────┐
│ Condition     │ Prevention Strategy                  │
├──────────────┼─────────────────────────────────────┤
│ Mutual       │ Lock-free algorithms                 │
│ exclusion    │                                      │
│ Hold and wait│ Acquire all resources at once         │
│ No preemption│ Forced release via timeout            │
│ Circular wait│ Assign resource ordering and enforce  │
│              │ consistent acquisition order          │
└──────────────┴─────────────────────────────────────┘
```

### 2.3 Other Concurrency Problems

```python
# === Livelock ===
# Similar to deadlock, but threads are "active" yet make no progress

# Example: Two people unable to pass each other in a hallway
# A and B both try to yield, continuously moving in the same direction

# Solution: Random backoff
import random

def polite_worker(name: str, resource_a, resource_b):
    while True:
        if resource_a.acquire(timeout=0.1):
            if resource_b.acquire(timeout=0.1):
                # Both acquired successfully
                try:
                    print(f"{name}: executing task")
                    break
                finally:
                    resource_b.release()
                    resource_a.release()
            else:
                resource_a.release()
                # Retry with random backoff
                time.sleep(random.uniform(0.01, 0.1))
        else:
            time.sleep(random.uniform(0.01, 0.1))


# === Starvation ===
# A specific thread is perpetually unable to acquire a resource

# Solution: Fair Lock
# Java's ReentrantLock(true) guarantees fairness
# In Python, use queue.PriorityQueue to assign priorities to tasks

# === Memory Visibility Problem ===
# Writes by one thread are not visible to other threads

# Java: volatile keyword guarantees visibility
# Python: GIL implicitly guarantees this (CPython only)
# C++: std::atomic, memory_order for explicit control
```

### 2.4 Comparison of Concurrency Models

```
Major concurrency models:

┌───────────────┬──────────────────────┬───────────────────┬──────────────┐
│ Model          │ Characteristics      │ Advantages         │ Language/FW  │
├───────────────┼──────────────────────┼───────────────────┼──────────────┤
│ Threads        │ OS threads with      │ Familiar           │ Java, C++   │
│ + Locks        │ shared memory comm.  │ Low-level control  │ Python      │
│                │                      │                    │             │
│ Actor          │ Independent entities │ Strong for         │ Erlang/OTP  │
│ Model          │ Message passing      │ distributed systems│ Akka(Scala) │
│                │ No shared state      │ Fault tolerant     │             │
│                │                      │ Scalable           │             │
│                │                      │                    │             │
│ CSP            │ Communication via    │ Simple and safe    │ Go          │
│                │ channels             │ High performance   │ Clojure     │
│                │ goroutines (light)   │ Easy to reason     │             │
│                │                      │ about              │             │
│                │                      │                    │             │
│ async/await    │ Cooperative          │ Optimal for I/O    │ JavaScript  │
│                │ multitasking         │ No locks needed    │ Python      │
│                │ Event loop           │ Easy to implement  │ Rust, C#    │
│                │ Single-threaded      │                    │             │
│                │                      │                    │             │
│ STM            │ Transactional        │ Composable         │ Haskell     │
│                │ Optimistic memory    │ No deadlocks       │ Clojure     │
│                │ control              │ Easy to reason     │             │
│                │                      │ about              │             │
│                │                      │                    │             │
│ Data-parallel  │ SIMD/GPU-based       │ Strong for large   │ CUDA        │
│                │ Apply same operation │ data volumes       │ OpenCL      │
│                │ in parallel          │ Optimal for        │ numpy       │
│                │                      │ numerical computing│             │
└───────────────┴──────────────────────┴───────────────────┴──────────────┘
```

---

## 3. async/await (Asynchronous Programming)

### 3.1 How the Event Loop Works

```python
# === How async/await Works ===

# Conceptual diagram of the event loop:
#
# ┌──────────────────────────────────────────┐
# │            Event Loop                     │
# │                                           │
# │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐    │
# │  │Task1│  │Task2│  │Task3│  │Task4│    │
# │  │await│  │ready│  │await│  │ready│    │
# │  └─────┘  └─────┘  └─────┘  └─────┘    │
# │                                           │
# │  1. Pick one ready task                   │
# │  2. Execute until it hits await           │
# │  3. On await, start I/O, move task to     │
# │     waiting state                         │
# │  4. Move tasks with completed I/O to      │
# │     ready state                           │
# │  5. Go back to step 1                     │
# └──────────────────────────────────────────┘

import asyncio
import time


# Basic async/await
async def fetch_data(name: str, delay: float) -> str:
    """Async function simulating I/O"""
    print(f"  [{name}] Started (waiting: {delay}s)")
    await asyncio.sleep(delay)  # I/O wait (yields control to other tasks)
    print(f"  [{name}] Completed")
    return f"{name}_data"


async def main():
    start = time.perf_counter()

    # ❌ Sequential execution: total 3 seconds
    # result1 = await fetch_data("API-1", 1.0)
    # result2 = await fetch_data("API-2", 1.0)
    # result3 = await fetch_data("API-3", 1.0)

    # ✅ Concurrent execution: total ~1 second (time of the slowest task)
    results = await asyncio.gather(
        fetch_data("API-1", 1.0),
        fetch_data("API-2", 0.5),
        fetch_data("API-3", 0.8),
    )

    elapsed = time.perf_counter() - start
    print(f"  Results: {results}")
    print(f"  Elapsed time: {elapsed:.2f}s")  # ~1.0 seconds

asyncio.run(main())
```

### 3.2 Practical Asynchronous Patterns

```python
import asyncio
import aiohttp
from typing import Any
from dataclasses import dataclass


# === Concurrent HTTP Requests ===

async def fetch_url(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch data from a single URL"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            return {"url": url, "status": resp.status, "data": data}
    except Exception as e:
        return {"url": url, "status": "error", "error": str(e)}


async def fetch_all(urls: list[str], max_concurrent: int = 10) -> list[dict]:
    """Fetch multiple URLs concurrently (with concurrency limit)"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(session, url):
        async with semaphore:  # Limit concurrent connections
            return await fetch_url(session, url)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_limit(session, url) for url in urls]
        return await asyncio.gather(*tasks)


# === Processing with Timeout ===

async def with_timeout(coro, timeout_seconds: float, default=None):
    """Execute a coroutine with a timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"Timed out ({timeout_seconds}s)")
        return default


# === Async Processing with Retry ===

async def retry_async(
    coro_func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Retry with exponential backoff"""
    current_delay = delay
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_func()
        except exceptions as e:
            if attempt == max_retries:
                raise
            print(f"Retry {attempt}/{max_retries}: {e}")
            await asyncio.sleep(current_delay)
            current_delay *= backoff


# === Async Generator (Stream Processing) ===

async def event_stream(interval: float = 1.0):
    """Generator that asynchronously produces events"""
    event_id = 0
    while True:
        await asyncio.sleep(interval)
        event_id += 1
        yield {"id": event_id, "timestamp": time.time(), "data": f"event_{event_id}"}


async def process_events():
    """Process the event stream"""
    async for event in event_stream(0.5):
        print(f"Event received: {event}")
        if event["id"] >= 5:
            break


# === Producer-Consumer Pattern (Async Version) ===

async def async_producer(queue: asyncio.Queue, items: list):
    """Async producer"""
    for item in items:
        await asyncio.sleep(0.1)  # Simulates time-consuming generation
        await queue.put(item)
        print(f"[Producer] Enqueued: {item}")
    await queue.put(None)  # Termination signal

async def async_consumer(queue: asyncio.Queue, name: str):
    """Async consumer"""
    while True:
        item = await queue.get()
        if item is None:
            await queue.put(None)  # Propagate to other consumers
            break
        await asyncio.sleep(0.2)  # Simulates time-consuming processing
        print(f"[{name}] Processed: {item}")
        queue.task_done()

async def producer_consumer_demo():
    """Producer-Consumer pattern demo"""
    queue = asyncio.Queue(maxsize=5)

    # Run 1 producer + 3 consumers concurrently
    await asyncio.gather(
        async_producer(queue, list(range(10))),
        async_consumer(queue, "Consumer-A"),
        async_consumer(queue, "Consumer-B"),
        async_consumer(queue, "Consumer-C"),
    )
```

```javascript
// JavaScript async/await

// Promise-based asynchronous processing
async function fetchUserWithPosts(userId) {
    // Fetch user info and posts concurrently
    const [user, posts] = await Promise.all([
        fetch(`/api/users/${userId}`).then(r => r.json()),
        fetch(`/api/users/${userId}/posts`).then(r => r.json()),
    ]);

    return { ...user, posts };
}

// Error handling
async function safelyFetch(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return { ok: true, data: await response.json() };
    } catch (error) {
        return { ok: false, error: error.message };
    }
}

// Promise.allSettled: Wait for all Promises to complete (even if some fail)
async function fetchMultiple(urls) {
    const results = await Promise.allSettled(
        urls.map(url => fetch(url).then(r => r.json()))
    );

    return results.map((result, i) => ({
        url: urls[i],
        status: result.status,
        data: result.status === 'fulfilled' ? result.value : null,
        error: result.status === 'rejected' ? result.reason.message : null,
    }));
}

// Rate limiter
class AsyncRateLimiter {
    constructor(maxConcurrent) {
        this.maxConcurrent = maxConcurrent;
        this.running = 0;
        this.queue = [];
    }

    async execute(fn) {
        if (this.running >= this.maxConcurrent) {
            await new Promise(resolve => this.queue.push(resolve));
        }

        this.running++;
        try {
            return await fn();
        } finally {
            this.running--;
            if (this.queue.length > 0) {
                this.queue.shift()();
            }
        }
    }
}

// Usage example: max 5 concurrent requests
const limiter = new AsyncRateLimiter(5);
const results = await Promise.all(
    urls.map(url => limiter.execute(() => fetch(url)))
);
```

### 3.3 Python's GIL and multiprocessing

```python
# === Python's GIL (Global Interpreter Lock) ===

# What is the GIL:
# An exclusive lock introduced by CPython for memory management safety
# → Only one thread can execute Python bytecode at a time within a process
# → Multi-threaded code does not parallelize CPU-bound work

# I/O-bound → Threads are fine (GIL is released during I/O waits)
# CPU-bound → Use multiprocessing

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def cpu_heavy_task(n: int) -> int:
    """CPU-intensive computation"""
    total = 0
    for i in range(n):
        total += i * i
    return total


# ❌ Multi-threaded: Does not speed up CPU-bound work due to the GIL
def with_threads(tasks: list[int]) -> list[int]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(cpu_heavy_task, tasks))

# ✅ Multi-process: True parallel execution
def with_processes(tasks: list[int]) -> list[int]:
    with ProcessPoolExecutor(max_workers=4) as executor:
        return list(executor.map(cpu_heavy_task, tasks))


# Benchmark
tasks = [10_000_000] * 4

start = time.perf_counter()
with_threads(tasks)
thread_time = time.perf_counter() - start

start = time.perf_counter()
with_processes(tasks)
process_time = time.perf_counter() - start

print(f"Threads: {thread_time:.2f}s")    # e.g., 8.5s (slow due to GIL)
print(f"Processes: {process_time:.2f}s")  # e.g., 2.5s (parallel on 4 cores)


# === concurrent.futures: Unified Interface ===

from concurrent.futures import as_completed

def process_batch(items: list, worker_func, max_workers: int = 4):
    """Execute batch processing in parallel and collect results"""
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit each task
        future_to_item = {
            executor.submit(worker_func, item): item
            for item in items
        }

        # Retrieve results in completion order
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result(timeout=30)
                results[item] = {"status": "success", "result": result}
            except Exception as e:
                results[item] = {"status": "error", "error": str(e)}

    return results
```

---

## 4. Go's goroutine/channel (CSP Model)

### 4.1 goroutine Basics

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// goroutine: lightweight thread (a few KB; OS threads are several MB)
// → Can run hundreds of thousands of goroutines concurrently

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d: started\n", id)
    time.Sleep(time.Second)
    fmt.Printf("Worker %d: completed\n", id)
}

func main() {
    var wg sync.WaitGroup

    // Launch 10 goroutines
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go worker(i, &wg)  // Launch goroutine with go keyword
    }

    wg.Wait()  // Wait for all goroutines to complete
    fmt.Println("All workers completed")
}
```

### 4.2 Communication via Channels

```go
package main

import (
    "fmt"
    "time"
)

// "Don't communicate by sharing memory,
//  share memory by communicating." — Go Proverb

// === Channel Basics ===

// Producer-Consumer Pattern
func producer(ch chan<- int) {
    for i := 0; i < 10; i++ {
        ch <- i  // Send to channel
        fmt.Printf("[Producer] Sent: %d\n", i)
        time.Sleep(100 * time.Millisecond)
    }
    close(ch)  // Close channel (no more sends)
}

func consumer(ch <-chan int, name string) {
    for val := range ch {  // Receive until channel is closed
        fmt.Printf("[%s] Received: %d\n", name, val)
        time.Sleep(200 * time.Millisecond)
    }
}


// === Fan-out / Fan-in Pattern ===

func fanOut(input <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        ch := make(chan int)
        channels[i] = ch
        go func(out chan<- int, workerID int) {
            for val := range input {
                // Simulate heavy processing
                result := val * val
                fmt.Printf("[Worker-%d] %d → %d\n", workerID, val, result)
                out <- result
            }
            close(out)
        }(ch, i)
    }
    return channels
}

func fanIn(channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for val := range c {
                merged <- val
            }
        }(ch)
    }

    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}


// === select Statement: Waiting on Multiple Channels ===

func selectDemo() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    timeout := time.After(3 * time.Second)

    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "data from ch1"
    }()

    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "data from ch2"
    }()

    for i := 0; i < 2; i++ {
        select {
        case msg := <-ch1:
            fmt.Printf("Received from ch1: %s\n", msg)
        case msg := <-ch2:
            fmt.Printf("Received from ch2: %s\n", msg)
        case <-timeout:
            fmt.Println("Timeout!")
            return
        }
    }
}


// === Pipeline Pattern ===

func generate(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

func filter(in <-chan int, predicate func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            if predicate(n) {
                out <- n
            }
        }
        close(out)
    }()
    return out
}

func pipelineMain() {
    // Pipeline: generate → square → filter
    numbers := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    squared := square(numbers)
    filtered := filter(squared, func(n int) bool { return n > 20 })

    for result := range filtered {
        fmt.Println(result)  // 25, 36, 49, 64, 81, 100
    }
}


// === Cancellation with Context ===

import "context"

func longRunningTask(ctx context.Context, id int) error {
    for i := 0; ; i++ {
        select {
        case <-ctx.Done():
            // Cancelled
            fmt.Printf("Task %d: cancelled (reason: %v)\n", id, ctx.Err())
            return ctx.Err()
        default:
            // Continue processing
            fmt.Printf("Task %d: step %d\n", id, i)
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func contextDemo() {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    go longRunningTask(ctx, 1)
    go longRunningTask(ctx, 2)

    time.Sleep(3 * time.Second)
    // → Auto-cancelled after 2 seconds
}
```

---

## 5. The Actor Model

### 5.1 Principles of the Actor Model

```
Actor Model (Carl Hewitt, 1973):

  Core concepts:
  - Everything is an actor (an independent unit of computation)
  - An actor can only do three things:
    1. Receive and process messages
    2. Create new actors
    3. Send messages to other actors
  - No shared state whatsoever
  - Messages are sent and received asynchronously

  Mailbox model:
  ┌─────────────────────────────────────┐
  │          Actor A                     │
  │  ┌──────────┐  ┌─────────────┐     │
  │  │ Mailbox  │→ │ Behavior    │     │
  │  │ msg1     │  │ (has state) │     │
  │  │ msg2     │  │             │     │
  │  │ msg3     │  │ → process   │──→ Send to Actor B
  │  └──────────┘  └─────────────┘     │
  └─────────────────────────────────────┘

  Advantages:
  - No shared state means no locks needed
  - High affinity with distributed systems
  - Fault tolerance (Let it crash philosophy)
  - Scalability
```

### 5.2 Actor Implementation in Erlang/Elixir

```elixir
# Elixir's actor model (GenServer)

defmodule BankAccount do
  use GenServer

  # Client API
  def start_link(initial_balance) do
    GenServer.start_link(__MODULE__, initial_balance)
  end

  def deposit(pid, amount) do
    GenServer.call(pid, {:deposit, amount})
  end

  def withdraw(pid, amount) do
    GenServer.call(pid, {:withdraw, amount})
  end

  def balance(pid) do
    GenServer.call(pid, :balance)
  end

  # Server callbacks
  @impl true
  def init(initial_balance) do
    {:ok, %{balance: initial_balance, transactions: []}}
  end

  @impl true
  def handle_call({:deposit, amount}, _from, state) when amount > 0 do
    new_state = %{
      state |
      balance: state.balance + amount,
      transactions: [{:deposit, amount} | state.transactions]
    }
    {:reply, {:ok, new_state.balance}, new_state}
  end

  @impl true
  def handle_call({:withdraw, amount}, _from, state) when amount > 0 do
    if state.balance >= amount do
      new_state = %{
        state |
        balance: state.balance - amount,
        transactions: [{:withdraw, amount} | state.transactions]
      }
      {:reply, {:ok, new_state.balance}, new_state}
    else
      {:reply, {:error, :insufficient_funds}, state}
    end
  end

  @impl true
  def handle_call(:balance, _from, state) do
    {:reply, state.balance, state}
  end
end

# Usage example
{:ok, account} = BankAccount.start_link(10000)
BankAccount.deposit(account, 5000)    # {:ok, 15000}
BankAccount.withdraw(account, 3000)   # {:ok, 12000}
BankAccount.balance(account)          # 12000

# Safe even under concurrent access (messages are processed sequentially)
```

### 5.3 Actor Model Implementation in Python

```python
import asyncio
from typing import Any, Callable
from dataclasses import dataclass, field


class Actor:
    """Simple actor implementation"""

    def __init__(self, name: str):
        self.name = name
        self._mailbox: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """Start the actor"""
        self._running = True
        while self._running:
            message = await self._mailbox.get()
            if message is None:  # Stop signal
                self._running = False
                break
            await self.handle_message(message)

    async def send(self, message: Any):
        """Send a message"""
        await self._mailbox.put(message)

    async def stop(self):
        """Stop the actor"""
        await self._mailbox.put(None)

    async def handle_message(self, message: Any):
        """Message handler (override in subclass)"""
        raise NotImplementedError


@dataclass
class Transfer:
    from_account: str
    to_account: str
    amount: int
    reply_to: asyncio.Queue


class BankAccountActor(Actor):
    """Bank account actor"""

    def __init__(self, name: str, initial_balance: int = 0):
        super().__init__(name)
        self._balance = initial_balance

    async def handle_message(self, message: dict):
        match message:
            case {"action": "deposit", "amount": amount, "reply": reply}:
                self._balance += amount
                await reply.put({"status": "ok", "balance": self._balance})

            case {"action": "withdraw", "amount": amount, "reply": reply}:
                if self._balance >= amount:
                    self._balance -= amount
                    await reply.put({"status": "ok", "balance": self._balance})
                else:
                    await reply.put({"status": "error", "reason": "insufficient funds"})

            case {"action": "balance", "reply": reply}:
                await reply.put({"balance": self._balance})


async def actor_demo():
    """Actor model demo"""
    # Create and start the actor
    account = BankAccountActor("account-1", 10000)
    task = asyncio.create_task(account.start())

    # Send messages and receive replies
    reply = asyncio.Queue()
    await account.send({"action": "deposit", "amount": 5000, "reply": reply})
    result = await reply.get()
    print(f"Deposit result: {result}")  # {"status": "ok", "balance": 15000}

    await account.send({"action": "balance", "reply": reply})
    result = await reply.get()
    print(f"Balance: {result}")  # {"balance": 15000}

    await account.stop()
    await task
```

---

## 6. Lock-Free Programming

```
Lock-free algorithms:

  Problems with locks (Mutex):
  - Risk of deadlock
  - Priority inversion
  - Context switch overhead
  - Scalability limitations

  Lock-free techniques:
  1. CAS (Compare-And-Swap) Operations
     - Atomically "compare and swap if equal"
     - Supported at the hardware level

  2. Immutable Data Structures
     - Immutable data → no locks needed
     - Persistent Data Structures

  3. Lock-Free Queue / Stack
     - CAS-based implementations
     - High throughput

  How CAS works:
  ┌─────────────────────────────────────┐
  │ CAS(memory_location, expected, new) │
  │                                     │
  │ if memory_location == expected:     │
  │     memory_location = new           │
  │     return true  // success         │
  │ else:                               │
  │     return false // another thread  │
  │                  // already changed │
  │                  // → retry         │
  └─────────────────────────────────────┘
```

```python
# Atomic operations in Python (simplified implementation)
import threading

class AtomicInteger:
    """Atomic integer (CAS-style implementation)"""

    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()  # Python lacks true CAS, so we use a lock as a substitute

    @property
    def value(self) -> int:
        return self._value

    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """CAS: Update to new_value if current value equals expected"""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

    def increment(self) -> int:
        """Atomic increment"""
        while True:
            current = self._value
            if self.compare_and_swap(current, current + 1):
                return current + 1

    def decrement(self) -> int:
        """Atomic decrement"""
        while True:
            current = self._value
            if self.compare_and_swap(current, current - 1):
                return current - 1

    def add_and_get(self, delta: int) -> int:
        """Atomic addition"""
        while True:
            current = self._value
            if self.compare_and_swap(current, current + delta):
                return current + delta
```

---

## 7. Practical Concurrency Patterns

### 7.1 Commonly Used Patterns

```python
# === Worker Pool Pattern ===

import asyncio
from typing import Callable, Any


async def worker_pool(
    tasks: list[Any],
    worker_func: Callable,
    max_workers: int = 10,
    progress_callback: Callable = None
) -> list[Any]:
    """Process multiple tasks concurrently using a worker pool"""
    semaphore = asyncio.Semaphore(max_workers)
    results = [None] * len(tasks)
    completed = 0

    async def process(index: int, task: Any):
        nonlocal completed
        async with semaphore:
            result = await worker_func(task)
            results[index] = result
            completed += 1
            if progress_callback:
                progress_callback(completed, len(tasks))

    await asyncio.gather(*(
        process(i, task) for i, task in enumerate(tasks)
    ))
    return results


# === Circuit Breaker Pattern ===

import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"        # Normal
    OPEN = "open"            # Tripped
    HALF_OPEN = "half_open"  # Partially open

class CircuitBreaker:
    """Circuit Breaker: blocks requests after consecutive failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            # Transition to HALF_OPEN after timeout elapses
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    async def call(self, func: Callable, *args, **kwargs):
        """Execute a request through the circuit breaker"""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise RuntimeError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)

            if current_state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0

            return result

        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                print(f"Circuit breaker: OPEN ({self._failure_count} failures)")

            raise


# === Bulkhead Pattern ===

class Bulkhead:
    """Bulkhead: isolate resources per service"""

    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    async def execute(self, func: Callable, *args, **kwargs):
        if self._semaphore.locked():
            raise RuntimeError(
                f"Bulkhead '{self.name}' capacity exceeded "
                f"(active: {self._active})"
            )

        async with self._semaphore:
            self._active += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self._active -= 1


# Set up bulkheads per service
api_bulkhead = Bulkhead("External API", max_concurrent=10)
db_bulkhead = Bulkhead("Database", max_concurrent=20)

# Even if the API service is overloaded, the DB service is not affected
# await api_bulkhead.execute(call_external_api, url)
# await db_bulkhead.execute(query_database, sql)
```

### 7.2 Concurrency Anti-Patterns

```
Concurrency anti-patterns:

1. ❌ Excessive concurrency
   - Creating a thread per request (C10K problem)
   - → Use event loops or thread pools

2. ❌ Lock granularity too coarse
   - Locking the entire operation (reduces concurrency)
   - → Minimize the scope of locks

3. ❌ Lock granularity too fine
   - Overly fine-grained locks (increased overhead)
   - → Find the appropriate granularity

4. ❌ Nested locks
   - A breeding ground for deadlocks
   - → Unify lock ordering, or design to avoid locks entirely

5. ❌ Busy waiting (spin lock)
   - 100% CPU usage while waiting
   - → Use condition variables or semaphores to wait

6. ❌ Fire and forget
   - Ignoring errors from async tasks
   - → Always handle errors and confirm completion

7. ❌ Shared mutable state
   - Modifying global variables from multiple threads
   - → Use immutable data or message passing

8. ❌ Using non-thread-safe libraries
   - Using libraries with shared state in a multi-threaded context
   - → Verify thread safety in the documentation
```

---

## 8. Concurrency in Rust

```rust
// Rust: Compile-time safety guarantees through the ownership system

use std::thread;
use std::sync::{Arc, Mutex, mpsc};

// === Threads + Message Passing ===

fn channel_example() {
    let (tx, rx) = mpsc::channel();

    // Sender thread
    thread::spawn(move || {
        let messages = vec!["hello", "from", "thread"];
        for msg in messages {
            tx.send(msg).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    });

    // Receiver (main thread)
    for received in rx {
        println!("Received: {}", received);
    }
}

// === Shared State (Arc + Mutex) ===

fn shared_state_example() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());  // 10
}

// Problems the Rust compiler prevents:
// - Data races: Concurrent &mut T access is forbidden at compile time
// - Dangling pointers: Prevented by the ownership system
// - Send/Sync traits: Prohibit sending non-thread-safe types across threads

// === async/await (tokio runtime) ===

use tokio;

#[tokio::main]
async fn main() {
    let handle1 = tokio::spawn(async {
        // Async task 1
        tokio::time::sleep(Duration::from_secs(1)).await;
        "result1"
    });

    let handle2 = tokio::spawn(async {
        // Async task 2
        tokio::time::sleep(Duration::from_secs(2)).await;
        "result2"
    });

    // Concurrent execution
    let (result1, result2) = tokio::join!(handle1, handle2);
    println!("{:?}, {:?}", result1, result2);
}
```

---

## 9. Design Guidelines for Practical Concurrency

```
Concurrency design guidelines:

1. First, consider whether concurrency is actually needed
   - Don't force concurrency if sequential processing is sufficient
   - Weigh the cost of added complexity vs. performance gains

2. Choose the simplest model
   I/O-bound → async/await
   CPU-bound → Multi-process (Python) / Multi-threaded
   Distributed → Message queues (RabbitMQ, Kafka, etc.)
   Real-time → Actor model or CSP

3. Minimize shared mutable state
   - Prefer immutable data structures
   - Use locks only when necessary
   - Keep lock scope minimal

4. Design error handling
   - Always set timeouts
   - Define retry strategies
   - Consider introducing circuit breakers
   - Limit fault propagation (bulkhead)

5. Testing strategy
   - Race conditions are hard to test → prevent them by design
   - Use pure functions extensively for testability
   - Adopt stress testing and chaos engineering

6. Monitoring and debugging
   - Include thread/task IDs in structured logs
   - Metrics: concurrency level, latency, error rate
   - Leverage deadlock detection tools

Technology selection quick reference:
┌───────────────────────┬───────────────────────────┐
│ Requirement            │ Recommended Technology     │
├───────────────────────┼───────────────────────────┤
│ Concurrent web API     │ async/await               │
│ requests               │                           │
│ Image/video batch      │ Multi-process             │
│ processing             │                           │
│ WebSocket server       │ async/await + events      │
│ Distributed task queue │ Celery / Kafka / RabbitMQ │
│ Real-time game         │ Actor model / ECS         │
│ Data pipeline          │ Apache Spark / Flink      │
│ Microservice           │ gRPC / message queues     │
│ communication          │                           │
└───────────────────────┴───────────────────────────┘
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is paramount. Understanding deepens not just through theory but by actually writing code and observing its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping ahead to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge covered in this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Concurrency vs Parallelism | Concurrency = structure (possible on 1 core), Parallelism = execution (requires multiple cores) |
| Race Condition | Shared state + non-atomic operations. Prevent with locks/CAS/immutability |
| Deadlock | Circular resource waiting. Prevent by breaking one of the Coffman conditions |
| async/await | Single-threaded async. Optimal for I/O-bound work |
| goroutine/channel | CSP model. Lightweight threads + channel communication |
| Actor Model | Message passing. No shared state. Strong for distributed systems |
| Lock-Free | CAS operations eliminate locks. High throughput |
| GIL (Python) | Use multi-process for CPU-bound work |
| Patterns | Worker Pool, Circuit Breaker, Bulkhead |
| Design Guidelines | Minimize shared state, always set timeouts, choose the simplest model |

---

## Recommended Next Reading

---

## References
1. Pike, R. "Concurrency Is Not Parallelism." Waza Conference, 2012.
2. Goetz, B. "Java Concurrency in Practice." Addison-Wesley, 2006.
3. Armstrong, J. "Programming Erlang: Software for a Concurrent World." 2nd Edition, Pragmatic Bookshelf, 2013.
4. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, 2019.
6. Hewitt, C. "A Universal Modular ACTOR Formalism for Artificial Intelligence." 1973.
7. Hoare, C.A.R. "Communicating Sequential Processes." Prentice Hall, 1985.
8. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." Morgan Kaufmann, 2012.
9. Nystrom, R. "Game Programming Patterns." Genever Benning, 2014.
10. Butcher, P. "Seven Concurrency Models in Seven Weeks." Pragmatic Bookshelf, 2014.
