# Concurrency Models Overview

> Three major models for programs to "do multiple things at once": multithreading, event loops, and the actor model. This guide compares the mechanics, advantages, and disadvantages of each.

## Learning Objectives

- [ ] Understand the difference between concurrency and parallelism
- [ ] Grasp the characteristics of the three major concurrency models
- [ ] Learn the use cases each model excels at
- [ ] Understand the CSP (Communicating Sequential Processes) model
- [ ] Acquire criteria for selecting concurrency models in practice

## Prerequisites

Having the following knowledge will deepen your understanding of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Synchronous vs Asynchronous](./00-sync-vs-async.md)

---

## 1. Concurrency vs Parallelism

### 1.1 Fundamental Concepts

```
Concurrency:
  -> Progresses multiple tasks by "switching between" them
  -> Possible even with a single CPU core
  -> A matter of "structure"

  Core 1: [Task A] [Task B] [Task A] [Task C] [Task B]

Parallelism:
  -> Executes multiple tasks "simultaneously"
  -> Requires multiple CPU cores
  -> A matter of "execution"

  Core 1: [Task A] [Task A] [Task A]
  Core 2: [Task B] [Task B] [Task B]
  Core 3: [Task C] [Task C] [Task C]

Rob Pike (Go designer):
  "Concurrency is about dealing with lots of things at once.
   Parallelism is about doing lots of things at once."
```

### 1.2 Relationship Between Concurrency and Parallelism

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Concurrent but not parallel:                      │
│   -> Switching tasks on a single core               │
│   -> Example: Node.js event loop                    │
│   -> 1 core handling multiple requests              │
│                                                     │
│   Parallel but not concurrent:                      │
│   -> SIMD (applying the same instruction to         │
│      multiple data)                                 │
│   -> Example: GPU matrix operations                 │
│   -> Parallel execution of the same operation       │
│      (not separate tasks)                           │
│                                                     │
│   Both concurrent and parallel:                     │
│   -> Multiple tasks running simultaneously on       │
│      multiple cores                                 │
│   -> Example: Go goroutines + multicore             │
│   -> Multiple tasks progressing simultaneously      │
│      on multiple cores                              │
│                                                     │
│   Neither:                                          │
│   -> Sequential execution of a single task          │
│      on a single core                               │
│   -> Example: an ordinary for loop                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1.3 Levels of Concurrency

```
Level 1: Process-level concurrency
  -> OS schedules multiple processes
  -> Memory spaces are isolated between processes
  -> Communication via IPC (Inter-Process Communication)
  -> Example: fork(), multi-process web servers (Apache prefork)

Level 2: Thread-level concurrency
  -> Multiple threads within a single process
  -> Shared memory space -> synchronization required
  -> Example: Java threads, C++ std::thread, Python threading

Level 3: Coroutine/fiber-level concurrency
  -> Switching in user space (no kernel involvement)
  -> Extremely lightweight
  -> Example: Go goroutines, Kotlin coroutines, Python asyncio

Level 4: Instruction-level parallelism (ILP)
  -> CPU executes instructions in parallel via pipelining/superscalar
  -> Transparent to the programmer
  -> Example: CPU out-of-order execution

Level 5: Data-level parallelism (DLP)
  -> Applies the same instruction to multiple data elements
  -> Example: SIMD (SSE, AVX), GPU CUDA/OpenCL
```

### 1.4 Concurrency and Parallelism in Code

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

func main() {
    // Concurrent but not parallel (using only 1 core)
    runtime.GOMAXPROCS(1) // Limit the number of OS threads to 1

    var wg sync.WaitGroup
    start := time.Now()

    for i := 0; i < 4; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            time.Sleep(100 * time.Millisecond)
            fmt.Printf("goroutine %d done at %v\n", id, time.Since(start))
        }(i)
    }
    wg.Wait()
    fmt.Printf("1 core: total %v\n\n", time.Since(start))
    // -> About 100ms (I/O waits are processed concurrently, so total is ~100ms)

    // Both concurrent and parallel (using all cores)
    runtime.GOMAXPROCS(runtime.NumCPU())
    start = time.Now()

    for i := 0; i < 4; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            // CPU-intensive processing
            sum := 0
            for j := 0; j < 100_000_000; j++ {
                sum += j
            }
            fmt.Printf("goroutine %d done at %v\n", id, time.Since(start))
        }(i)
    }
    wg.Wait()
    fmt.Printf("all cores: total %v\n", time.Since(start))
    // -> CPU-intensive work is accelerated through parallel execution
}
```

```python
import asyncio
import time
import multiprocessing

# Concurrency (asyncio) and parallelism (multiprocessing) in Python

# Concurrency: efficiently handling I/O waits
async def concurrent_io():
    """Single-threaded, but processes other tasks while waiting on I/O"""
    start = time.time()

    async def fetch(name: str, delay: float) -> str:
        await asyncio.sleep(delay)  # Simulate I/O wait
        return f"{name} done"

    # Run 4 I/O tasks concurrently
    results = await asyncio.gather(
        fetch("A", 0.1),
        fetch("B", 0.2),
        fetch("C", 0.15),
        fetch("D", 0.1),
    )
    print(f"Concurrent I/O: {time.time() - start:.3f}s")
    # -> About 0.2 seconds (the time of the slowest task)
    return results

# Parallelism: processing CPU-intensive work across multiple cores
def cpu_heavy(n: int) -> int:
    """CPU-intensive processing"""
    return sum(range(n))

def parallel_cpu():
    """Parallel execution across multiple processes"""
    start = time.time()
    with multiprocessing.Pool(4) as pool:
        results = pool.map(cpu_heavy, [10_000_000] * 4)
    print(f"Parallel CPU: {time.time() - start:.3f}s")
    return results

if __name__ == "__main__":
    asyncio.run(concurrent_io())
    parallel_cpu()
```

---

## 2. Multithreading Model

### 2.1 Basic Structure

```
Mechanism:
  -> Creates multiple OS threads
  -> Exchanges data via shared memory
  -> Uses locks (Mutex) for mutual exclusion

  Thread 1 ─────────────────────────→
  Thread 2 ─────────────────────────→
  Thread 3 ─────────────────────────→
       ↕ Shared Memory ↕
  ┌──────────────────┐
  │  Shared State    │ <- Locked with Mutex
  └──────────────────┘

Advantages:
  ✓ True parallel execution (utilizes multiple cores)
  ✓ Well suited for CPU-intensive tasks
  ✓ OS handles scheduling automatically

Disadvantages:
  ✗ Complex lock management for shared state
  ✗ Deadlocks, race conditions
  ✗ Thread creation overhead (~1MB/thread)
  ✗ Difficult to debug

Representative languages: Java, C++, Python (with GIL), Rust
```

### 2.2 Java Multithreading

```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

public class MultithreadingExamples {

    // Basic: Thread class
    public void basicThread() {
        Thread thread = new Thread(() -> {
            System.out.println("Running in thread: " + Thread.currentThread().getName());
        });
        thread.start();
    }

    // Thread pool: ExecutorService
    public void threadPool() {
        // Pool size matched to CPU count
        int poolSize = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(poolSize);

        List<Future<String>> futures = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            futures.add(executor.submit(() -> {
                Thread.sleep(100);
                return "Task " + taskId + " completed";
            }));
        }

        // Collect results
        for (Future<String> future : futures) {
            try {
                System.out.println(future.get(5, TimeUnit.SECONDS));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        executor.shutdown();
    }

    // Managing shared state: synchronized
    private int counter = 0;

    public synchronized void incrementSafe() {
        counter++;
    }

    // Finer-grained lock control: ReentrantLock
    private final ReentrantLock lock = new ReentrantLock();
    private final Map<String, String> sharedMap = new HashMap<>();

    public void updateMap(String key, String value) {
        lock.lock();
        try {
            sharedMap.put(key, value);
        } finally {
            lock.unlock(); // Always release the lock in finally
        }
    }

    // Lock-free: Atomic variables
    private final AtomicLong atomicCounter = new AtomicLong(0);

    public void atomicIncrement() {
        atomicCounter.incrementAndGet(); // Thread-safe via CAS operation
    }

    // Producer-Consumer pattern
    public void producerConsumer() {
        BlockingQueue<String> queue = new LinkedBlockingQueue<>(100);

        // Producer
        new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                try {
                    queue.put("Item " + i); // Blocks if the queue is full
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }).start();

        // Consumer
        new Thread(() -> {
            while (true) {
                try {
                    String item = queue.take(); // Blocks if the queue is empty
                    process(item);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }).start();
    }
}
```

### 2.3 Rust Thread Model

```rust
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

// Rust: ownership system guarantees thread safety at compile time

fn basic_threading() {
    let mut handles = vec![];

    for i in 0..4 {
        let handle = thread::spawn(move || {
            println!("Thread {} running", i);
            i * 2
        });
        handles.push(handle);
    }

    let results: Vec<i32> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    println!("Results: {:?}", results);
}

// Shared state with Arc<Mutex<T>>
fn shared_state() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            // Lock is automatically released when the scope ends
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());
}

// RwLock: multiple concurrent reads, exclusive writes
fn read_write_lock() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));

    // Multiple reader threads
    let mut readers = vec![];
    for i in 0..5 {
        let data = Arc::clone(&data);
        readers.push(thread::spawn(move || {
            let values = data.read().unwrap(); // Read lock
            println!("Reader {}: {:?}", i, *values);
        }));
    }

    // Writer thread
    {
        let data = Arc::clone(&data);
        thread::spawn(move || {
            let mut values = data.write().unwrap(); // Write lock
            values.push(4);
        }).join().unwrap();
    }

    for reader in readers {
        reader.join().unwrap();
    }
}

// Compile error example: Rust prevents data races
// fn compile_error() {
//     let mut data = vec![1, 2, 3];
//
//     // Compile error: cannot move data into multiple threads
//     thread::spawn(|| { data.push(4); });
//     thread::spawn(|| { data.push(5); });
//
//     // -> Must use Arc<Mutex<Vec<i32>>> instead
// }
```

### 2.4 Dangers of Multithreading

```
Deadlock:
  Thread A: lock(X) -> lock(Y)
  Thread B: lock(Y) -> lock(X)
  -> Each thread waits indefinitely for the other to release its lock

  Prevention strategies:
  1. Unify lock acquisition order
  2. Timed locks (tryLock)
  3. Lock hierarchy protocol

Race condition:
  Thread A: read(x) -> x + 1 -> write(x)
  Thread B: read(x) -> x + 1 -> write(x)
  -> Both read x simultaneously and write the same value + 1
  -> x only increases by 1 (should increase by 2)

  Prevention strategies:
  1. Mutex / synchronized
  2. Atomic operations (CAS)
  3. Immutable data structures

Priority Inversion:
  -> Low-priority thread holds a lock
  -> High-priority thread waits for the lock
  -> Medium-priority thread preempts the low-priority thread
  -> High-priority thread waits forever

  Prevention strategies:
  1. Priority Inheritance Protocol
  2. Minimize lock hold time
```

```java
// Deadlock example
public class DeadlockExample {
    private final Object lockA = new Object();
    private final Object lockB = new Object();

    // Potential deadlock
    public void method1() {
        synchronized (lockA) {
            System.out.println("Thread 1: locked A");
            // Meanwhile, Thread 2 acquires lockB
            synchronized (lockB) {
                System.out.println("Thread 1: locked B");
            }
        }
    }

    public void method2() {
        synchronized (lockB) {
            System.out.println("Thread 2: locked B");
            // lockA is held by Thread 1 -> deadlock
            synchronized (lockA) {
                System.out.println("Thread 2: locked A");
            }
        }
    }

    // Fix: unify lock order
    public void method1Fixed() {
        synchronized (lockA) {  // Always acquire A first
            synchronized (lockB) {
                System.out.println("Thread 1: locked A, B");
            }
        }
    }

    public void method2Fixed() {
        synchronized (lockA) {  // Always acquire A first
            synchronized (lockB) {
                System.out.println("Thread 2: locked A, B");
            }
        }
    }
}
```

---

## 3. Event Loop Model

### 3.1 Basic Structure

```
Mechanism:
  -> Processes an event queue on a single thread
  -> I/O is non-blocking (delegated to the OS)
  -> Adds callbacks to the queue when I/O completes

  ┌──────────────────────────────────────┐
  │           Event Loop                 │
  │  ┌──────────────────────────────┐   │
  │  │ 1. Execute call stack        │   │
  │  │ 2. Process microtasks        │   │
  │  │ 3. Execute one macrotask     │   │
  │  │ 4. -> Return to step 1      │   │
  │  └──────────────────────────────┘   │
  │         ↑ Completion notification   │
  │  ┌──────────────────────────────┐   │
  │  │ OS / libuv (I/O management)  │   │
  │  │ Network, files, timers       │   │
  │  └──────────────────────────────┘   │
  └──────────────────────────────────────┘

Advantages:
  ✓ No lock needed for shared state (single-threaded)
  ✓ Excels at handling massive concurrent connections (solves C10K)
  ✓ Memory efficient

Disadvantages:
  ✗ Not suited for CPU-intensive tasks (blocks the event loop)
  ✗ Cannot directly utilize multiple cores
  ✗ Callback hell (improved with async/await)

Representative: JavaScript (Node.js/browser), Python (asyncio)
```

### 3.2 Node.js Event Loop in Detail

```
Node.js Event Loop Phases:

  ┌───────────────────────┐
  │        timers          │ <- setTimeout, setInterval
  ├───────────────────────┤
  │    pending callbacks   │ <- Deferred I/O callbacks
  ├───────────────────────┤
  │     idle, prepare      │ <- Internal use
  ├───────────────────────┤
  │         poll           │ <- Retrieve I/O events, execute callbacks
  ├───────────────────────┤
  │        check           │ <- setImmediate
  ├───────────────────────┤
  │    close callbacks     │ <- close events
  └───────────────────────┘

Microtasks vs Macrotasks:
  Microtasks (high priority):
    - Promise.then/catch/finally
    - process.nextTick (Node.js)
    - queueMicrotask
    -> All processed between each phase

  Macrotasks (normal priority):
    - setTimeout/setInterval
    - setImmediate
    - I/O callbacks
    -> Processed one at a time
```

```typescript
// Verifying event loop execution order
console.log('1. Synchronous code');

setTimeout(() => {
  console.log('5. setTimeout (macrotask)');
}, 0);

Promise.resolve().then(() => {
  console.log('3. Promise (microtask)');
}).then(() => {
  console.log('4. Promise chain (microtask)');
});

process.nextTick(() => {
  console.log('2. nextTick (microtask, highest priority)');
});

console.log('1.5. Synchronous code 2');

// Output order:
// 1. Synchronous code
// 1.5. Synchronous code 2
// 2. nextTick (microtask, highest priority)
// 3. Promise (microtask)
// 4. Promise chain (microtask)
// 5. setTimeout (macrotask)
```

### 3.3 Browser Event Loop

```typescript
// The browser event loop differs slightly from Node.js
// requestAnimationFrame has its own timing

console.log('1. Synchronous');

requestAnimationFrame(() => {
  console.log('4. requestAnimationFrame (before paint)');
});

setTimeout(() => {
  console.log('5. setTimeout');
}, 0);

Promise.resolve().then(() => {
  console.log('2. Promise (microtask)');
});

queueMicrotask(() => {
  console.log('3. queueMicrotask');
});

// Output order:
// 1. Synchronous
// 2. Promise (microtask)
// 3. queueMicrotask
// 4. requestAnimationFrame (before the next paint frame)
// 5. setTimeout
```

### 3.4 Detecting Event Loop Blocking

```typescript
// Detecting event loop blocking
function detectEventLoopBlocking(): void {
  let lastCheck = Date.now();

  setInterval(() => {
    const now = Date.now();
    const lag = now - lastCheck - 100; // Set at 100ms intervals

    if (lag > 50) { // Delay over 50ms
      console.warn(`Event loop blocked for ${lag}ms`);
      // Could capture a stack trace here, etc.
    }

    lastCheck = now;
  }, 100);
}

// Node.js: monitorEventLoopDelay API (v11.10+)
import { monitorEventLoopDelay } from 'perf_hooks';

const h = monitorEventLoopDelay({ resolution: 20 });
h.enable();

setInterval(() => {
  console.log({
    min: h.min / 1e6,       // nanoseconds -> milliseconds
    max: h.max / 1e6,
    mean: h.mean / 1e6,
    p99: h.percentile(99) / 1e6,
  });
  h.reset();
}, 5000);
```

### 3.5 Python asyncio Event Loop

```python
import asyncio
import time

# Python's asyncio event loop

# Basic coroutine
async def fetch_data(name: str, delay: float) -> str:
    print(f"[{time.time():.3f}] {name}: start")
    await asyncio.sleep(delay)  # Simulate I/O wait
    print(f"[{time.time():.3f}] {name}: done")
    return f"{name} result"

# Concurrent task execution
async def main():
    # Concurrent execution with asyncio.create_task
    task1 = asyncio.create_task(fetch_data("API-1", 0.2))
    task2 = asyncio.create_task(fetch_data("API-2", 0.3))
    task3 = asyncio.create_task(fetch_data("DB", 0.1))

    # Wait for all to complete
    results = await asyncio.gather(task1, task2, task3)
    print(f"All results: {results}")

    # With timeout
    try:
        result = await asyncio.wait_for(
            fetch_data("Slow API", 5.0),
            timeout=1.0
        )
    except asyncio.TimeoutError:
        print("Timeout!")

asyncio.run(main())

# Using a custom event loop
# uvloop: a high-performance event loop based on libuv
import uvloop

async def high_performance_main():
    # uvloop is 2-4x faster than CPython's default event loop
    pass

uvloop.install()
asyncio.run(high_performance_main())
```

---

## 4. Actor Model

### 4.1 Basic Structure

```
Mechanism:
  -> Everything is an "actor" (an independent process)
  -> Actors communicate via message passing
  -> No shared state (each actor owns its own state)

  ┌─────────┐  Message     ┌─────────┐
  │ Actor A │────────────→│ Actor B │
  │ state_a │             │ state_b │
  └─────────┘             └─────────┘
       │                       │
       │  Message              │ Message
       ↓                       ↓
  ┌─────────┐             ┌─────────┐
  │ Actor C │             │ Actor D │
  │ state_c │             │ state_d │
  └─────────┘             └─────────┘

Advantages:
  ✓ No shared state (no locks needed)
  ✓ Naturally extends to distributed systems
  ✓ Fault tolerance (actor restarts)
  ✓ Scalability

Disadvantages:
  ✗ Message passing overhead
  ✗ Difficult to debug (asynchronous messages)
  ✗ Steep learning curve

Representative: Erlang/Elixir (BEAM), Akka (Scala/Java)
```

### 4.2 Erlang/Elixir Actor Model

```elixir
# Elixir: a typical implementation of the actor model

# GenServer: a generic server process
defmodule CounterServer do
  use GenServer

  # Client API
  def start_link(initial_value \\ 0) do
    GenServer.start_link(__MODULE__, initial_value, name: __MODULE__)
  end

  def increment() do
    GenServer.cast(__MODULE__, :increment)  # Asynchronous message
  end

  def get_value() do
    GenServer.call(__MODULE__, :get_value)  # Synchronous message (waits for reply)
  end

  # Server callbacks
  @impl true
  def init(initial_value) do
    {:ok, initial_value}
  end

  @impl true
  def handle_cast(:increment, state) do
    {:noreply, state + 1}
  end

  @impl true
  def handle_call(:get_value, _from, state) do
    {:reply, state, state}
  end
end

# Supervisor: fault tolerance through supervision trees
defmodule MyApp.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, :ok, opts)
  end

  @impl true
  def init(:ok) do
    children = [
      {CounterServer, 0},
      {WebSocketHandler, []},
      {DatabasePool, pool_size: 10},
    ]

    # one_for_one: if one process crashes, only restart that one
    # one_for_all: if one crashes, restart all
    # rest_for_one: restart all processes started after the crashed one
    Supervisor.init(children, strategy: :one_for_one)
  end
end

# Spawning massive numbers of processes (lightweight: ~2KB/process)
defmodule MassiveSpawn do
  def run(count) do
    pids = for _ <- 1..count do
      spawn(fn ->
        receive do
          {:ping, sender} -> send(sender, :pong)
        end
      end)
    end

    # Send a message to all processes
    for pid <- pids do
      send(pid, {:ping, self()})
    end

    # Receive responses from all processes
    for _ <- pids do
      receive do
        :pong -> :ok
      end
    end

    IO.puts("#{count} processes completed")
  end
end

# MassiveSpawn.run(1_000_000)  # 1 million processes work practically
```

### 4.3 Akka (Scala/Java)

```scala
import akka.actor.{Actor, ActorSystem, Props, ActorRef}

// Akka: the actor model on the JVM

// Actor definition
class CounterActor extends Actor {
  private var count = 0

  def receive: Receive = {
    case "increment" =>
      count += 1

    case "get" =>
      sender() ! count  // Send reply

    case "reset" =>
      count = 0
  }
}

// Usage example
object Main extends App {
  val system = ActorSystem("MySystem")
  val counter = system.actorOf(Props[CounterActor], "counter")

  // Send message (asynchronous, fire-and-forget)
  counter ! "increment"
  counter ! "increment"
  counter ! "increment"

  // Wait for reply (Ask pattern)
  import akka.pattern.ask
  import scala.concurrent.duration._
  implicit val timeout: akka.util.Timeout = 5.seconds

  val future = counter ? "get"  // Returns Future[Any]
  future.foreach(println)       // 3
}

// Supervision
class ParentActor extends Actor {
  import akka.actor.SupervisorStrategy._
  import scala.concurrent.duration._

  override val supervisorStrategy = OneForOneStrategy(
    maxNrOfRetries = 3,
    withinTimeRange = 1.minute
  ) {
    case _: ArithmeticException => Resume    // Resume
    case _: NullPointerException => Restart  // Restart
    case _: Exception => Escalate            // Escalate to parent
  }

  val child: ActorRef = context.actorOf(Props[ChildActor], "child")

  def receive: Receive = {
    case msg => child forward msg
  }
}
```

### 4.4 Actor Model Patterns

```
Pattern 1: Request-Reply
  Client ──Request──→ Actor ──Reply──→ Client
  -> Synchronous interaction (with timeout)

Pattern 2: Fire-and-Forget
  Sender ──Message──→ Actor
  -> No reply needed, asynchronous

Pattern 3: Publish-Subscribe
  Publisher ──Message──→ EventBus ──→ Subscriber1
                                  ──→ Subscriber2
                                  ──→ Subscriber3

Pattern 4: Scatter-Gather
  Coordinator ──Task──→ Worker1 ──Result──→ Aggregator
              ──Task──→ Worker2 ──Result──→
              ──Task──→ Worker3 ──Result──→

Pattern 5: Pipeline
  Stage1 ──→ Stage2 ──→ Stage3 ──→ Stage4
  -> Each stage is an actor
  -> Backpressure is possible

Pattern 6: Circuit Breaker
  -> Blocks messages to an actor after a certain number of consecutive failures
  -> Retries after a specified time
  -> Prevents failure cascading
```

---

## 5. CSP (Communicating Sequential Processes)

### 5.1 Basic Structure

```
Mechanism:
  -> Lightweight threads (goroutines) x many
  -> Data is sent and received through channels
  -> "Don't communicate by sharing memory; share memory by communicating"

  goroutine 1 ───→ [channel] ───→ goroutine 2
  goroutine 3 ───→ [channel] ───→ goroutine 4

Advantages:
  ✓ Lightweight (goroutine: ~2KB, thread: ~1MB)
  ✓ Safe communication via channels
  ✓ Runtime handles scheduling automatically
  ✓ Automatic multicore utilization

Representative: Go, Clojure (core.async)
```

### 5.2 Go Channel Patterns

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Pattern 1: Generator
func fibonacci(ctx context.Context) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        a, b := 0, 1
        for {
            select {
            case <-ctx.Done():
                return
            case ch <- a:
                a, b = b, a+b
            }
        }
    }()
    return ch
}

// Pattern 2: Fan-Out / Fan-In
func fanOut(input <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        channels[i] = process(input)
    }
    return channels
}

func fanIn(channels ...<-chan int) <-chan int {
    merged := make(chan int)
    var wg sync.WaitGroup

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

func process(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for v := range input {
            output <- v * 2 // Processing
        }
    }()
    return output
}

// Pattern 3: Pipeline
func source(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            out <- n
        }
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            out <- n * n
        }
    }()
    return out
}

func filter(in <-chan int, predicate func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            if predicate(n) {
                out <- n
            }
        }
    }()
    return out
}

func main() {
    // Pipeline: source -> square -> filter
    numbers := source(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    squared := square(numbers)
    even := filter(squared, func(n int) bool { return n%2 == 0 })

    for n := range even {
        fmt.Println(n) // 4, 16, 36, 64, 100
    }
}

// Pattern 4: Worker Pool
func workerPool(jobs <-chan int, results chan<- int, workerCount int) {
    var wg sync.WaitGroup

    for i := 0; i < workerCount; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for job := range jobs {
                result := processJob(job) // Actual processing
                results <- result
            }
        }(i)
    }

    go func() {
        wg.Wait()
        close(results)
    }()
}

func processJob(job int) int {
    time.Sleep(10 * time.Millisecond) // Simulate processing
    return job * 2
}

// Pattern 5: Multiplexing with Select
func multiplexer(ctx context.Context) {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()

    heartbeat := make(chan struct{})
    data := make(chan string)

    for {
        select {
        case <-ctx.Done():
            fmt.Println("Shutting down")
            return
        case <-ticker.C:
            fmt.Println("Tick")
        case <-heartbeat:
            fmt.Println("Heartbeat received")
        case msg := <-data:
            fmt.Println("Data:", msg)
        }
    }
}

// Pattern 6: Rate Limiter
func rateLimiter(requests <-chan string, ratePerSecond int) <-chan string {
    output := make(chan string)
    ticker := time.NewTicker(time.Second / time.Duration(ratePerSecond))

    go func() {
        defer close(output)
        defer ticker.Stop()
        for req := range requests {
            <-ticker.C // Rate limiting
            output <- req
        }
    }()

    return output
}
```

### 5.3 CSP vs Actor Model Comparison

```
                    CSP (Go)               Actor Model (Erlang)
Communication      Channels (anonymous)     Messages (addressed)
Process identity   Anonymous (connected     PID / name
                   via channels)
Blocking           Can block on send/recv   Non-blocking (mailbox)
Synchronization    Synchronous by default   Asynchronous by default
Distribution       Not supported by default Native support
Fault tolerance    Must be implemented      Supervisor trees
                   manually
Buffering          Buffered channels        Unbounded mailboxes
                   available

CSP strengths:
  -> Synchronous communication makes data flow explicit
  -> select for waiting on multiple channels
  -> Channel directionality (send-only/receive-only) expressed in the type system

Actor model strengths:
  -> Naturally extends to distributed environments
  -> Rich fault tolerance frameworks
  -> Per-actor GC and memory management
```

---

## 6. Other Concurrency Models

### 6.1 Software Transactional Memory (STM)

```
STM: memory operations similar to database transactions

  atomically $ do
    balance1 <- readTVar account1
    balance2 <- readTVar account2
    writeTVar account1 (balance1 - 100)
    writeTVar account2 (balance2 + 100)
  -- Automatically retries on conflict

Advantages:
  ✓ Optimistic concurrency control (no locks)
  ✓ Composable (transactions can be combined)
  ✓ No deadlocks

Disadvantages:
  ✗ Cannot be used with side-effecting operations
  ✗ Retry overhead
  ✗ Potential for livelock (constant conflicts)

Representative: Haskell (STM), Clojure (Ref/STM)
```

```haskell
-- Haskell STM: bank account transfer
import Control.Concurrent.STM

type Account = TVar Int

transfer :: Account -> Account -> Int -> STM ()
transfer from to amount = do
    fromBalance <- readTVar from
    toBalance <- readTVar to
    if fromBalance >= amount
        then do
            writeTVar from (fromBalance - amount)
            writeTVar to (toBalance + amount)
        else retry  -- Wait for retry if insufficient balance

main :: IO ()
main = do
    account1 <- newTVarIO 1000
    account2 <- newTVarIO 500

    -- Execute atomically (auto-retries on conflict)
    atomically $ transfer account1 account2 200

    balance1 <- readTVarIO account1
    balance2 <- readTVarIO account2
    putStrLn $ "Account 1: " ++ show balance1  -- 800
    putStrLn $ "Account 2: " ++ show balance2  -- 700
```

### 6.2 Structured Concurrency

```
Structured Concurrency:
  -> Provides scopes (boundaries) for concurrent tasks
  -> Child tasks must complete when the parent task ends
  -> Prevents resource leaks
  -> Adopted in Kotlin coroutines, Swift concurrency, Java 21

Unstructured concurrency:
  func parent() {
    spawn(child1)  // Fire and forget -> potential leak
    spawn(child2)  // Fire and forget
    return         // Does not wait for child task completion
  }

Structured concurrency:
  func parent() {
    taskGroup {
      spawn(child1)  // Spawned within a scope
      spawn(child2)
    }  // <- Waits for all child tasks to complete here
    // Execution does not proceed until child1 and child2 are done
  }
```

```swift
// Swift: Structured Concurrency
func fetchDashboard() async throws -> Dashboard {
    // TaskGroup: structured concurrent execution
    try await withThrowingTaskGroup(of: DashboardComponent.self) { group in
        group.addTask { try await fetchUser() }
        group.addTask { try await fetchOrders() }
        group.addTask { try await fetchNotifications() }

        var components: [DashboardComponent] = []
        for try await component in group {
            components.append(component)
        }
        // When exiting the group, all tasks have completed
        return Dashboard(components: components)
    }
}

// TaskGroup cancellation
func fetchWithCancellation() async throws -> Data {
    try await withThrowingTaskGroup(of: Data.self) { group in
        group.addTask {
            try await fetchFromServer1() // Slow
        }
        group.addTask {
            try await fetchFromServer2() // Fast
        }

        // Use the first completed result, cancel the rest
        guard let result = try await group.next() else {
            throw FetchError.noResult
        }
        group.cancelAll() // Cancel remaining tasks
        return result
    }
}
```

```kotlin
// Kotlin: Structured Concurrency with Coroutines
import kotlinx.coroutines.*

suspend fun fetchDashboard(): Dashboard = coroutineScope {
    // Waits for all tasks within coroutineScope to complete
    val userDeferred = async { fetchUser() }
    val ordersDeferred = async { fetchOrders() }
    val notifsDeferred = async { fetchNotifications() }

    Dashboard(
        user = userDeferred.await(),
        orders = ordersDeferred.await(),
        notifications = notifsDeferred.await()
    )
    // If any throws an exception, the others are automatically cancelled
}

// SupervisorScope: one child's failure does not affect other children
suspend fun resilientFetch(): Dashboard = supervisorScope {
    val user = async { fetchUser() }
    val orders = async {
        try { fetchOrders() } catch (e: Exception) { emptyList() }
    }
    val notifs = async {
        try { fetchNotifications() } catch (e: Exception) { emptyList() }
    }

    Dashboard(
        user = user.await(),
        orders = orders.await(),
        notifications = notifs.await()
    )
}
```

### 6.3 Data Parallel Model

```
Data Parallelism:
  -> Applies the same operation to multiple data elements simultaneously
  -> GPU computing, SIMD instructions, MapReduce

  CPU SIMD:
    Normal:   a[0]*b[0]  a[1]*b[1]  a[2]*b[2]  a[3]*b[3]  (4 multiplications)
    SIMD:    [a[0] a[1] a[2] a[3]] * [b[0] b[1] b[2] b[3]]  (4 multiplications in 1 instruction)

  GPU (CUDA):
    -> Thousands of cores execute the same kernel
    -> Optimal for matrix operations and machine learning

  MapReduce:
    Map:    [data1, data2, data3, ...] -> [result1, result2, result3, ...]
    Reduce: [result1, result2, result3, ...] -> finalResult
    -> Data processing in distributed environments (Hadoop, Spark)
```

```python
# Python: data parallelism examples

# 1. Data parallelism with multiprocessing.Pool
from multiprocessing import Pool

def square(x):
    return x ** 2

with Pool(4) as p:
    results = p.map(square, range(100))
    # 100 data elements processed in parallel across 4 processes

# 2. NumPy: vectorization (implicit data parallelism)
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
c = a * b  # Element-wise multiplication (utilizes SIMD)
# [10, 40, 90, 160, 250]

# 3. concurrent.futures: high-level API
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# I/O concurrency: ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    urls = ["https://api.example.com/1", "https://api.example.com/2"]
    futures = [executor.submit(fetch_url, url) for url in urls]
    results = [f.result() for f in futures]

# CPU parallelism: ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=4) as executor:
    data_chunks = [data[i::4] for i in range(4)]
    futures = [executor.submit(process_chunk, chunk) for chunk in data_chunks]
    results = [f.result() for f in futures]
```

---

## 7. Comparison and Selection

### 7.1 Comprehensive Comparison Table

```
┌──────────────┬────────────┬────────────┬────────────┬────────────┐
│              │ Multi-     │ Event      │ Actor      │ CSP        │
│              │ threading  │ Loop       │ Model      │            │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ CPU-intensive│ Excellent  │ Poor       │ Good       │ Excellent  │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ I/O-intensive│ Good       │ Excellent  │ Excellent  │ Excellent  │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ Concurrent   │ ~thousands │ ~100K      │ ~millions  │ ~millions  │
│ connections  │            │            │            │            │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ Safety       │ Low (locks)│ Medium     │ High       │ High       │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ Debugging    │ Difficult  │ Moderate   │ Difficult  │ Moderate   │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ Memory       │ Low        │ High       │ High       │ High       │
│ efficiency   │            │            │            │            │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ Distribution │ Manual     │ Manual     │ Native     │ Manual     │
│ support      │            │            │            │            │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ Learning     │ Medium     │ Low        │ High       │ Medium     │
│ curve        │            │            │            │            │
└──────────────┴────────────┴────────────┴────────────┴────────────┘

Selection guidelines:
  Web APIs (I/O-intensive) -> Event loop or CSP
  Real-time communication -> Actor model
  Image/video processing -> Multithreading
  Microservices -> Actor model or CSP
  Distributed systems -> Actor model
  High-performance servers -> CSP (Go)
  Frontend -> Event loop (JS)
```

### 7.2 Recommended Models by Use Case

```typescript
// Use case 1: REST API server
// -> Event loop (Node.js) or CSP (Go)

// Node.js: high productivity, NPM ecosystem
import express from 'express';
const app = express();

app.get('/api/users/:id', async (req, res) => {
  const user = await db.getUser(req.params.id);
  res.json(user);
});

// Go: high performance, static typing
// func handleUser(w http.ResponseWriter, r *http.Request) {
//     user, err := db.GetUser(r.PathValue("id"))
//     json.NewEncoder(w).Encode(user)
// }
```

```elixir
# Use case 2: Real-time chat
# -> Actor model (Elixir/Phoenix)

# Each chat room is an actor (process)
defmodule ChatRoom do
  use GenServer

  def start_link(room_id) do
    GenServer.start_link(__MODULE__, room_id, name: via(room_id))
  end

  def join(room_id, user_id) do
    GenServer.call(via(room_id), {:join, user_id})
  end

  def send_message(room_id, user_id, message) do
    GenServer.cast(via(room_id), {:message, user_id, message})
  end

  # State: member list
  def init(room_id) do
    {:ok, %{room_id: room_id, members: MapSet.new()}}
  end

  def handle_call({:join, user_id}, _from, state) do
    new_state = %{state | members: MapSet.put(state.members, user_id)}
    {:reply, :ok, new_state}
  end

  def handle_cast({:message, user_id, message}, state) do
    # Broadcast to all members
    Enum.each(state.members, fn member ->
      send_to_user(member, %{from: user_id, text: message})
    end)
    {:noreply, state}
  end

  defp via(room_id), do: {:via, Registry, {ChatRegistry, room_id}}
end
```

```go
// Use case 3: Data pipeline
// -> CSP (Go)

package main

import (
    "encoding/json"
    "fmt"
    "sync"
)

// ETL pipeline: Extract -> Transform -> Load
type Record struct {
    ID   int
    Data string
}

func extract(source string) <-chan Record {
    out := make(chan Record, 100) // Buffered channel
    go func() {
        defer close(out)
        // Read from data source
        for i := 0; i < 1000; i++ {
            out <- Record{ID: i, Data: fmt.Sprintf("raw-%d", i)}
        }
    }()
    return out
}

func transform(in <-chan Record, workers int) <-chan Record {
    out := make(chan Record, 100)
    var wg sync.WaitGroup

    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for record := range in {
                // Transform data
                record.Data = fmt.Sprintf("transformed-%s", record.Data)
                out <- record
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

func load(in <-chan Record) {
    for record := range in {
        // Write to database
        data, _ := json.Marshal(record)
        fmt.Println(string(data))
    }
}

func main() {
    // Build the pipeline
    records := extract("data-source")
    transformed := transform(records, 4) // 4 workers for concurrent transformation
    load(transformed)
}
```

### 7.3 Practical Selection Flowchart

```
Evaluate requirements:

1. What is the dominant workload?
   ├── I/O-intensive (APIs, databases, files)
   │   ├── Concurrent connections < 1,000 -> Anything works
   │   ├── Concurrent connections 1,000-10,000
   │   │   ├── Team experience: JS -> Node.js (event loop)
   │   │   ├── Team experience: Python -> asyncio
   │   │   └── Performance priority -> Go (CSP)
   │   └── Concurrent connections > 10,000
   │       ├── Go (CSP)
   │       └── Erlang/Elixir (actor model)
   │
   └── CPU-intensive (computation, image processing, ML)
       ├── Simple parallelism -> Multithreading (Rust, C++, Java)
       ├── Data parallelism -> GPU / SIMD
       └── Pipeline -> Go (CSP) or threads + queues

2. Is distribution required?
   ├── Yes -> Actor model (Erlang/Elixir, Akka)
   └── No -> Other models are sufficient

3. Is fault tolerance critical?
   ├── Yes -> Actor model (supervisor trees)
   └── No -> Other models are sufficient

4. Real-time requirements?
   ├── WebSocket / SSE -> Event loop or actor model
   └── REST API -> Event loop or CSP
```

---

## 8. Hybrid Approaches

### 8.1 Node.js: Event Loop + Worker Threads

```typescript
// Node.js: event loop for I/O, Worker Threads for CPU
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { cpus } from 'os';

// Main thread: request handling (I/O)
if (isMainThread) {
  const numCPUs = cpus().length;
  const workerPool: Worker[] = [];
  const taskQueue: { data: any; resolve: Function; reject: Function }[] = [];

  // Initialize worker pool
  for (let i = 0; i < numCPUs; i++) {
    const worker = new Worker(__filename);
    worker.on('message', (result) => {
      // Process the next task
      const nextTask = taskQueue.shift();
      if (nextTask) {
        worker.postMessage(nextTask.data);
      }
    });
    workerPool.push(worker);
  }

  // Offload CPU-intensive tasks to workers
  function offloadToWorker(data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const freeWorker = workerPool.find(w => !w.isBusy);
      if (freeWorker) {
        freeWorker.postMessage(data);
      } else {
        taskQueue.push({ data, resolve, reject });
      }
    });
  }

  // Express server
  import express from 'express';
  const app = express();

  app.post('/api/process-image', async (req, res) => {
    // I/O: receive request (event loop)
    const image = await receiveUpload(req);

    // CPU: image processing (worker thread)
    const processed = await offloadToWorker({ type: 'resize', image });

    // I/O: upload to S3 (event loop)
    const url = await uploadToS3(processed);

    res.json({ url });
  });
}

// Worker thread: CPU-intensive processing
if (!isMainThread) {
  parentPort?.on('message', (data) => {
    const result = heavyComputation(data);
    parentPort?.postMessage(result);
  });
}
```

### 8.2 Python: asyncio + multiprocessing

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# CPU-intensive processing
def cpu_bound_task(data: bytes) -> bytes:
    """CPU-intensive tasks like image processing (executed in separate process)"""
    import hashlib
    result = hashlib.pbkdf2_hmac('sha256', data, b'salt', 100000)
    return result

# I/O-intensive processing
async def io_bound_task(url: str) -> dict:
    """I/O-intensive tasks like API calls (executed with asyncio)"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

# Hybrid: asyncio + ProcessPoolExecutor
async def hybrid_handler(request_data: dict) -> dict:
    loop = asyncio.get_event_loop()

    # I/O: fetch external data asynchronously
    external_data = await io_bound_task("https://api.example.com/data")

    # CPU: heavy computation in a separate process
    with ProcessPoolExecutor(max_workers=4) as pool:
        computed = await loop.run_in_executor(
            pool,
            cpu_bound_task,
            external_data["payload"].encode()
        )

    # I/O: save results
    await save_result(computed)

    return {"status": "ok", "hash": computed.hex()}
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is most important. Understanding deepens not through theory alone but by actually writing code and observing how it behaves.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend building a solid understanding of the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Model | Core Concept | Representative Languages | Use Cases |
|-------|-------------|-------------------------|-----------|
| Multithreading | Shared memory + locks | Java, C++, Rust | CPU-intensive, legacy systems |
| Event loop | Single thread + async I/O | JS, Python | Web APIs, frontend |
| Actor model | Message passing | Erlang, Elixir | Distributed, real-time, high fault tolerance |
| CSP | Lightweight threads + channels | Go | High-performance servers |
| STM | Transactional memory | Haskell, Clojure | Complex shared state management |
| Structured concurrency | Scoped concurrency | Kotlin, Swift, Java 21 | Modern applications |

### Principles for Selection

```
1. Prioritize simplicity
   -> Do not choose an overly complex model
   -> If an event loop is sufficient, multithreading is unnecessary

2. Match the team's skill set
   -> Productivity is higher with familiar languages and models
   -> Allocate sufficient learning time when introducing a new model

3. Match the bottleneck
   -> I/O-intensive -> Event loop / CSP
   -> CPU-intensive -> Multithreading / data parallelism
   -> Mixed -> Hybrid approach

4. Consider scalability requirements
   -> Vertical scaling -> Multithreading
   -> Horizontal scaling -> Actor model / CSP
```

---

## Recommended Next Guides

---

## References
1. Hoare, C.A.R. "Communicating Sequential Processes." 1978.
2. Hewitt, C. "A Universal Modular Actor Formalism." 1973.
3. Pike, R. "Concurrency Is Not Parallelism." Waza Conference, 2012.
4. Armstrong, J. "Programming Erlang: Software for a Concurrent World." Pragmatic Bookshelf, 2013.
5. Goetz, B. "Java Concurrency in Practice." Addison-Wesley, 2006.
6. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, 2019.
7. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
8. Elizarov, R. "Structured Concurrency." Kotlin Blog, 2018.
9. Apple Developer Documentation. "Swift Concurrency."
10. Peierls, T. "STM in Haskell." Journal of Functional Programming, 2005.
