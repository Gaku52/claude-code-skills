# Structured Concurrency

> Structured concurrency is a paradigm for "structurally managing the lifetime of concurrent operations." Through Kotlin coroutines, Swift structured concurrency, and Python TaskGroup, it enables safe concurrent programming.

## What You Will Learn in This Chapter

- [ ] Understand the principles of structured concurrency
- [ ] Grasp the problems of unstructured concurrency
- [ ] Learn implementations in various languages
- [ ] Understand cancellation propagation mechanisms
- [ ] Grasp integration with error handling
- [ ] Master practical application patterns

## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Retry Strategies](./02-retry-and-backoff.md)

---

## 1. What Is Structured Concurrency

### 1.1 Core Concept

```
Unstructured Concurrency (Traditional):
  -> Tasks are "fired and forgotten"
  -> Child tasks survive after parent terminates
  -> Errors are silently swallowed in child tasks
  -> Resource leaks

  function process() {
    startBackgroundTask(); // Fire and forget
    startAnotherTask();    // Who manages this task's lifetime?
  } // Tasks continue running after process returns

Structured Concurrency:
  -> Child tasks complete within the parent's scope
  -> Parent waits for all child tasks to complete
  -> If one child task fails, others are cancelled
  -> No resource leaks

  async function process() {
    await Promise.all([  // Wait for all child tasks to complete
      task1(),
      task2(),
    ]);
  } // All tasks are guaranteed to be complete here
```

### 1.2 Comparison with Structured Programming

```
Structured Programming (1968, Dijkstra):
  -> Eliminated goto, structured the control flow
  -> Scopes made explicit with if/else, while, for
  -> Clear entry and exit points in code

  Unstructured: goto label;  // No telling where it jumps
  Structured:   if (...) { ... }  // Scope is clear

Structured Concurrency (2018, Elizarov, Syme):
  -> Eliminated "fire and forget," structured the lifetime of concurrent operations
  -> Task scopes made explicit
  -> Clear start and end points for tasks

  Unstructured: Task.run(() => ...) // No telling where it ends
  Structured:   async with TaskGroup() { ... } // Completion guaranteed within scope

Common Principles:
  -> Clarification of control flow
  -> Scope-based resource management
  -> Improved readability and debuggability
```

### 1.3 Problems with Unstructured Concurrency

```
Problem 1: Resource Leaks
  function startProcessing() {
    setTimeout(() => {
      // Who manages this callback?
      // It outlives the scope of startProcessing
      processData();
    }, 5000);
  }

Problem 2: Swallowed Errors
  function fetchAll() {
    fetch('/api/users');     // No one catches errors
    fetch('/api/products');  // Same issue
  }

Problem 3: Difficulty of Cancellation
  function loadDashboard() {
    const p1 = fetch('/api/users');
    const p2 = fetch('/api/stats');
    // If the user navigates away, cancelling p1 and p2 is difficult
    // Each needs its own AbortController
  }

Problem 4: Difficulty of Debugging
  -> Stack traces of async tasks are fragmented
  -> Parent-child relationships are unclear
  -> Tracking which task is running at which point is difficult
```

---

## 2. Kotlin Coroutines

### 2.1 coroutineScope: The Basics of Structured Concurrency

```kotlin
import kotlinx.coroutines.*

// coroutineScope: Scope for structured concurrency
suspend fun loadDashboard(): Dashboard = coroutineScope {
    // Launch child coroutines
    val userDeferred = async { fetchUser() }
    val ordersDeferred = async { fetchOrders() }
    val statsDeferred = async { fetchStats() }

    // Wait for all results
    Dashboard(
        user = userDeferred.await(),
        orders = ordersDeferred.await(),
        stats = statsDeferred.await(),
    )
    // When exiting coroutineScope, all child coroutines are guaranteed to be complete
    // If one throws an exception, the others are cancelled
}
```

### 2.2 supervisorScope: Handling Child Errors Independently

```kotlin
// supervisorScope: Child errors do not affect siblings
suspend fun loadDashboardResilient(): Dashboard = supervisorScope {
    val user = async { fetchUser() }
    val orders = async {
        try { fetchOrders() }
        catch (e: Exception) { emptyList() } // Fallback
    }
    val stats = async {
        try { fetchStats() }
        catch (e: Exception) { Stats.empty() }
    }

    Dashboard(
        user = user.await(),
        orders = orders.await(),
        stats = stats.await(),
    )
}

// Choosing between coroutineScope and supervisorScope
//
// coroutineScope:
//   -> When all tasks must succeed
//   -> One failure -> cancel all
//   -> Example: Transaction-like operations
//
// supervisorScope:
//   -> When individual tasks are independent
//   -> One failure does not stop the others
//   -> Example: Loading individual dashboard components
```

### 2.3 Cancellation in Kotlin

```kotlin
import kotlinx.coroutines.*

// Cancellation basics
suspend fun processWithCancellation() {
    val job = CoroutineScope(Dispatchers.Default).launch {
        try {
            repeat(1000) { i ->
                println("Processing $i...")
                delay(100) // Cancellation point
            }
        } catch (e: CancellationException) {
            println("Cancelled!")
            // Cleanup operations
        } finally {
            // Release resources
            withContext(NonCancellable) {
                // This block executes even after cancellation
                cleanup()
            }
        }
    }

    delay(500)
    job.cancel() // Request cancellation
    job.join()   // Wait for cancellation to complete
}

// Best practices for cancellation support
suspend fun downloadFile(url: String, dest: File) = coroutineScope {
    val response = httpClient.get(url)
    val channel = response.bodyAsChannel()

    dest.outputStream().use { output ->
        val buffer = ByteArray(8192)
        while (true) {
            // Periodically check for cancellation with ensureActive()
            ensureActive()

            val bytesRead = channel.readAvailable(buffer)
            if (bytesRead == -1) break

            output.write(buffer, 0, bytesRead)
        }
    }
}

// Processing with timeout
suspend fun fetchWithTimeout(): Result {
    return withTimeout(5000) { // 5-second timeout
        fetchData()
    }
    // Throws TimeoutCancellationException on timeout
}

// Return null on timeout
suspend fun fetchWithTimeoutOrNull(): Result? {
    return withTimeoutOrNull(5000) {
        fetchData()
    }
    // Returns null on timeout (no exception)
}
```

### 2.4 Advanced Patterns in Kotlin

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

// Fan-out: One producer, multiple consumers
suspend fun fanOutExample() = coroutineScope {
    val channel = produce {
        repeat(100) { send(it) }
    }

    // Process with 5 workers
    repeat(5) { workerId ->
        launch {
            for (item in channel) {
                println("Worker $workerId processing $item")
                processItem(item)
            }
        }
    }
}

// Fan-in: Multiple producers, one consumer
suspend fun fanInExample() = coroutineScope {
    val results = Channel<ProcessResult>()

    // Multiple producers
    val sources = listOf("api-1", "api-2", "api-3")
    sources.forEach { source ->
        launch {
            val data = fetchFromSource(source)
            results.send(ProcessResult(source, data))
        }
    }

    // Collect all results
    launch {
        val allResults = mutableListOf<ProcessResult>()
        repeat(sources.size) {
            allResults.add(results.receive())
        }
        processAllResults(allResults)
        results.close()
    }
}

// Race pattern: Return the first success
suspend fun raceExample(): String = coroutineScope {
    select<String> {
        async { fetchFromPrimary() }.onAwait { it }
        async { fetchFromSecondary() }.onAwait { it }
        async { fetchFromTertiary() }.onAwait { it }
    }
    // Returns the first to complete, cancels the rest
}

// Concurrent processing with backpressure
fun processWithBackpressure(items: List<Item>): Flow<Result> = flow {
    coroutineScope {
        val semaphore = Semaphore(10) // Concurrency limit
        items.map { item ->
            async {
                semaphore.withPermit {
                    processItem(item)
                }
            }
        }.forEach { deferred ->
            emit(deferred.await())
        }
    }
}

// Structured concurrency with built-in error recovery
suspend fun resilientDashboard(): Dashboard = supervisorScope {
    val user = async {
        retryWithBackoff(maxRetries = 3) { fetchUser() }
    }

    val orders = async {
        try {
            withTimeout(5000) { fetchOrders() }
        } catch (e: Exception) {
            logger.warn("Failed to fetch orders: ${e.message}")
            emptyList()
        }
    }

    val recommendations = async {
        try {
            withTimeoutOrNull(3000) { fetchRecommendations() }
                ?: Recommendations.default()
        } catch (e: Exception) {
            Recommendations.default()
        }
    }

    Dashboard(
        user = user.await(),
        orders = orders.await(),
        recommendations = recommendations.await(),
    )
}
```

---

## 3. Swift Structured Concurrency

### 3.1 async let: Static Concurrency

```swift
// Swift: Concurrently execute a static number of tasks with async let
func loadDashboard() async throws -> Dashboard {
    async let user = fetchUser()           // Start concurrently
    async let orders = fetchOrders()       // Start concurrently
    async let stats = fetchStats()         // Start concurrently

    return try await Dashboard(
        user: user,
        orders: orders,
        stats: stats,
    )
    // Waits for all async let bindings to complete
    // If one throws, the others are automatically cancelled
}
```

### 3.2 TaskGroup: Dynamic Concurrency

```swift
// TaskGroup: Dynamic number of tasks
func processItems(_ items: [Item]) async throws -> [Result] {
    try await withThrowingTaskGroup(of: Result.self) { group in
        for item in items {
            group.addTask {
                try await processItem(item)
            }
        }

        var results: [Result] = []
        for try await result in group {
            results.append(result)
        }
        return results
    }
    // Outside TaskGroup scope = all tasks guaranteed complete
}

// TaskGroup with concurrency limit
func processWithConcurrencyLimit(
    items: [Item],
    maxConcurrent: Int = 5
) async throws -> [Result] {
    try await withThrowingTaskGroup(of: Result.self) { group in
        var results: [Result] = []
        var iterator = items.makeIterator()
        var inFlight = 0

        // Submit initial batch
        while inFlight < maxConcurrent, let item = iterator.next() {
            group.addTask { try await processItem(item) }
            inFlight += 1
        }

        // Submit next task as each one completes
        for try await result in group {
            results.append(result)
            inFlight -= 1
            if let item = iterator.next() {
                group.addTask { try await processItem(item) }
                inFlight += 1
            }
        }

        return results
    }
}
```

### 3.3 Cancellation in Swift

```swift
// Checking for and responding to cancellation
func downloadFile(url: URL) async throws -> Data {
    var data = Data()
    let (bytes, _) = try await URLSession.shared.bytes(from: url)

    for try await byte in bytes {
        // Periodically check for cancellation
        try Task.checkCancellation()
        data.append(byte)
    }

    return data
}

// Cleanup on cancellation
func processWithCleanup() async throws {
    let tempFile = createTempFile()

    do {
        try await longRunningProcess(tempFile)
    } catch is CancellationError {
        // Cleanup on cancellation
        try? FileManager.default.removeItem(at: tempFile)
        throw CancellationError()
    }
}

// withTaskCancellationHandler: Cancellation handler
func fetchData() async throws -> Data {
    let handle = startNetworkRequest()

    return try await withTaskCancellationHandler {
        // Main operation
        try await handle.result()
    } onCancel: {
        // Cancel network request on cancellation
        handle.cancel()
    }
}

// Implementing timeout
func fetchWithTimeout<T>(
    seconds: TimeInterval,
    operation: @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask {
            try await operation()
        }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw TimeoutError()
        }
        // Return the first to complete
        let result = try await group.next()!
        group.cancelAll() // Cancel the rest
        return result
    }
}
```

### 3.4 Actor: Preventing Data Races

```swift
// Actor: Thread-safe data access
actor UserCache {
    private var cache: [String: User] = [:]
    private var inFlightRequests: [String: Task<User, Error>] = [:]

    func getUser(id: String) async throws -> User {
        // Cache hit
        if let cached = cache[id] {
            return cached
        }

        // Wait if a request for the same user is already in flight
        if let existing = inFlightRequests[id] {
            return try await existing.value
        }

        // Start a new request
        let task = Task {
            let user = try await fetchUser(id: id)
            cache[id] = user
            inFlightRequests[id] = nil
            return user
        }

        inFlightRequests[id] = task
        return try await task.value
    }

    func invalidate(id: String) {
        cache.removeValue(forKey: id)
    }

    func invalidateAll() {
        cache.removeAll()
    }
}

// GlobalActor: Guaranteeing execution in a specific context
@globalActor
actor DatabaseActor {
    static let shared = DatabaseActor()
}

@DatabaseActor
class DatabaseManager {
    private var connection: Connection?

    func query(_ sql: String) async throws -> [Row] {
        // Executes in the DatabaseActor context
        // Automatically thread-safe
        guard let conn = connection else {
            throw DatabaseError.notConnected
        }
        return try await conn.execute(sql)
    }
}

// Sendable protocol: Guaranteeing concurrency-safe types
struct UserData: Sendable {
    let id: String
    let name: String
    let email: String
}

// @Sendable closure
func processInBackground(_ data: UserData) {
    Task.detached { @Sendable in
        // data is Sendable, so it can be safely passed
        await processUser(data)
    }
}
```

---

## 4. Python TaskGroup (3.11+)

### 4.1 Basic Usage

```python
import asyncio

# Python 3.11+: TaskGroup
async def load_dashboard():
    async with asyncio.TaskGroup() as tg:
        user_task = tg.create_task(fetch_user())
        orders_task = tg.create_task(fetch_orders())
        stats_task = tg.create_task(fetch_stats())

    # All tasks are complete after exiting async with
    # If one raises an exception -> others are cancelled -> ExceptionGroup is raised
    return Dashboard(
        user=user_task.result(),
        orders=orders_task.result(),
        stats=stats_task.result(),
    )
```

### 4.2 Handling ExceptionGroup

```python
# Handling ExceptionGroup (Python 3.11+)
async def resilient_load():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task_a())
            tg.create_task(task_b())
    except* ValueError as eg:
        print(f"ValueError group: {eg.exceptions}")
        for exc in eg.exceptions:
            print(f"  - {exc}")
    except* TypeError as eg:
        print(f"TypeError group: {eg.exceptions}")
    except* ConnectionError as eg:
        print(f"ConnectionError group: {eg.exceptions}")

# ExceptionGroup structure
# ExceptionGroup wraps multiple exceptions
# except* selectively catches only specific exception types
# Remaining exceptions are re-raised

# Multiple except* blocks
async def handle_multiple_errors():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task_that_may_raise_value_error())
            tg.create_task(task_that_may_raise_type_error())
            tg.create_task(task_that_may_raise_io_error())
    except* ValueError as eg:
        # Handle only ValueError
        for exc in eg.exceptions:
            log_validation_error(exc)
    except* (TypeError, IOError) as eg:
        # Handle TypeError and IOError together
        for exc in eg.exceptions:
            log_system_error(exc)
    # Exception types not handled above are re-raised
```

### 4.3 Cancellation

```python
import asyncio
from contextlib import asynccontextmanager


# TaskGroup with timeout
async def load_with_timeout():
    try:
        async with asyncio.timeout(5.0):
            async with asyncio.TaskGroup() as tg:
                user_task = tg.create_task(fetch_user())
                orders_task = tg.create_task(fetch_orders())
    except TimeoutError:
        print("Dashboard loading timed out")
        return Dashboard.default()

    return Dashboard(
        user=user_task.result(),
        orders=orders_task.result(),
    )


# Cancellation-aware task
async def cancellable_download(url: str, dest: str) -> None:
    """Download with cancellation support"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            with open(dest, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    # asyncio.CancelledError propagates automatically
                    f.write(chunk)


# shield: Protect from cancellation
async def critical_operation():
    """Protect a critical operation from cancellation"""
    # Wrapping with shield prevents external cancellation from propagating inside
    result = await asyncio.shield(save_to_database(data))
    return result


# Cancellation handling pattern
async def process_with_cleanup():
    """Execute cleanup on cancellation"""
    resource = await acquire_resource()
    try:
        await long_running_process(resource)
    except asyncio.CancelledError:
        # Cleanup on cancellation
        await cleanup_resource(resource)
        raise  # CancelledError must always be re-raised
    finally:
        await release_resource(resource)
```

### 4.4 Advanced Patterns

```python
import asyncio
from typing import TypeVar, Callable, Awaitable, AsyncIterator
from dataclasses import dataclass

T = TypeVar('T')
R = TypeVar('R')


# Batch processing with concurrency limit
async def map_concurrent(
    items: list[T],
    func: Callable[[T], Awaitable[R]],
    max_concurrent: int = 10,
) -> list[R]:
    """Process items with a concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[R] = [None] * len(items)  # type: ignore

    async def process_with_limit(index: int, item: T) -> None:
        async with semaphore:
            results[index] = await func(item)

    async with asyncio.TaskGroup() as tg:
        for i, item in enumerate(items):
            tg.create_task(process_with_limit(i, item))

    return results


# Usage example
async def main():
    urls = [f"https://api.example.com/items/{i}" for i in range(100)]
    results = await map_concurrent(
        urls,
        fetch_url,
        max_concurrent=20,
    )


# Race: Return the first success
async def race(*coros: Awaitable[T]) -> T:
    """Return the result of the first coroutine to succeed"""
    async with asyncio.TaskGroup() as tg:
        done = asyncio.Event()
        result_holder: list[T] = []

        async def run_and_signal(coro: Awaitable[T]) -> None:
            try:
                result = await coro
                if not done.is_set():
                    result_holder.append(result)
                    done.set()
            except Exception:
                pass  # Ignore failures

        for coro in coros:
            tg.create_task(run_and_signal(coro))

        # Note: TaskGroup waits for all tasks to complete
        # TaskGroup is not ideal for the race pattern
        # asyncio.wait(return_when=FIRST_COMPLETED) is more appropriate

    if result_holder:
        return result_holder[0]
    raise RuntimeError("All tasks failed")


# Proper race implementation using asyncio.wait
async def race_proper(*coros: Awaitable[T]) -> T:
    """Wait for the first completion using asyncio.wait"""
    tasks = [asyncio.ensure_future(c) for c in coros]

    try:
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Return the result of the first completed task
        result_task = done.pop()
        return result_task.result()

    except Exception:
        # Cancel all tasks on error
        for task in tasks:
            task.cancel()
        raise


# Pipeline: Process in stages
async def pipeline_example():
    """Multi-stage pipeline"""
    queue1: asyncio.Queue[RawData] = asyncio.Queue(maxsize=100)
    queue2: asyncio.Queue[ProcessedData] = asyncio.Queue(maxsize=100)

    async def stage1_fetch():
        """Stage 1: Data fetching"""
        for url in urls:
            data = await fetch_data(url)
            await queue1.put(data)
        await queue1.put(None)  # Termination signal

    async def stage2_process():
        """Stage 2: Data processing"""
        while True:
            data = await queue1.get()
            if data is None:
                await queue2.put(None)
                break
            processed = await process_data(data)
            await queue2.put(processed)

    async def stage3_save():
        """Stage 3: Data saving"""
        while True:
            data = await queue2.get()
            if data is None:
                break
            await save_data(data)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(stage1_fetch())
        tg.create_task(stage2_process())
        tg.create_task(stage3_save())


# Resource management with structured concurrency
@asynccontextmanager
async def managed_workers(
    num_workers: int,
    work_queue: asyncio.Queue,
    handler: Callable,
):
    """Lifecycle management for a worker pool"""
    async def worker(worker_id: int):
        while True:
            try:
                item = await asyncio.wait_for(work_queue.get(), timeout=1.0)
                await handler(worker_id, item)
                work_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    tasks = []
    try:
        for i in range(num_workers):
            task = asyncio.create_task(worker(i))
            tasks.append(task)
        yield tasks
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# Usage example
async def process_with_workers():
    queue: asyncio.Queue = asyncio.Queue()

    # Add items to the queue
    for item in items:
        await queue.put(item)

    async with managed_workers(5, queue, process_item):
        await queue.join()  # Wait for all items to be processed
```

---

## 5. Structured Concurrency in JavaScript/TypeScript

### 5.1 Promise.all -- Basic Concurrent Processing

```typescript
// Promise.all: Wait for all tasks to complete (partial structured concurrency)
async function loadDashboard(): Promise<Dashboard> {
  const [user, orders, stats] = await Promise.all([
    fetchUser(),
    fetchOrders(),
    fetchStats(),
  ]);

  return { user, orders, stats };
}

// Limitations:
// 1. If one fails, the others immediately reject, but are NOT cancelled
// 2. The remaining Promises continue executing in the background
// 3. There is no explicit cancellation mechanism
```

### 5.2 Promise.allSettled -- Error Resilience

```typescript
// Promise.allSettled: Wait for all tasks to complete (regardless of success/failure)
async function loadDashboardResilient(): Promise<Dashboard> {
  const results = await Promise.allSettled([
    fetchUser(),
    fetchOrders(),
    fetchStats(),
  ]);

  const user = results[0].status === 'fulfilled'
    ? results[0].value
    : null;

  const orders = results[1].status === 'fulfilled'
    ? results[1].value
    : [];

  const stats = results[2].status === 'fulfilled'
    ? results[2].value
    : Stats.default();

  if (!user) {
    throw new Error('Failed to fetch user');
  }

  return { user, orders, stats };
}

// Helper functions for easier use
function extractResult<T>(result: PromiseSettledResult<T>): T | null {
  return result.status === 'fulfilled' ? result.value : null;
}

function extractResults<T extends readonly unknown[]>(
  results: { [K in keyof T]: PromiseSettledResult<T[K]> },
): { [K in keyof T]: T[K] | null } {
  return results.map(extractResult) as any;
}
```

### 5.3 Pseudo-Structured Concurrency with AbortController

```typescript
// Cancellation-aware concurrent processing using AbortController
class StructuredScope {
  private controller = new AbortController();
  private tasks: Promise<any>[] = [];

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  addTask<T>(fn: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const task = fn(this.signal);
    this.tasks.push(task);
    return task;
  }

  async run<T>(
    fn: (scope: StructuredScope) => Promise<T>,
  ): Promise<T> {
    try {
      const result = await fn(this);
      // Wait for remaining tasks to complete
      await Promise.allSettled(this.tasks);
      return result;
    } catch (error) {
      // Cancel all tasks on error
      this.controller.abort();
      // Wait for cancellation to complete
      await Promise.allSettled(this.tasks);
      throw error;
    }
  }

  cancel(reason?: string): void {
    this.controller.abort(reason);
  }
}

// Usage example
async function loadWithScope(): Promise<Dashboard> {
  const scope = new StructuredScope();

  return scope.run(async (s) => {
    const userPromise = s.addTask(async (signal) => {
      const response = await fetch('/api/user', { signal });
      return response.json();
    });

    const ordersPromise = s.addTask(async (signal) => {
      const response = await fetch('/api/orders', { signal });
      return response.json();
    });

    const [user, orders] = await Promise.all([userPromise, ordersPromise]);
    return { user, orders, stats: null };
  });
}

// Scope with timeout
async function loadWithTimeout(): Promise<Dashboard> {
  const scope = new StructuredScope();

  // Cancel on timeout
  const timeout = setTimeout(() => scope.cancel('timeout'), 5000);

  try {
    return await scope.run(async (s) => {
      const user = await s.addTask((signal) =>
        fetchWithSignal('/api/user', signal)
      );
      const orders = await s.addTask((signal) =>
        fetchWithSignal('/api/orders', signal)
      );
      return { user, orders, stats: null };
    });
  } finally {
    clearTimeout(timeout);
  }
}
```

### 5.4 Concurrency-Limited Processing

```typescript
// Semaphore-based concurrency limiting
class AsyncSemaphore {
  private current = 0;
  private queue: Array<() => void> = [];

  constructor(private readonly limit: number) {}

  async acquire(): Promise<void> {
    if (this.current < this.limit) {
      this.current++;
      return;
    }

    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
    });
  }

  release(): void {
    if (this.queue.length > 0) {
      const next = this.queue.shift()!;
      next();
    } else {
      this.current--;
    }
  }

  async withPermit<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

// Concurrency-limited map
async function mapConcurrent<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  concurrency: number = 10,
): Promise<R[]> {
  const semaphore = new AsyncSemaphore(concurrency);
  return Promise.all(
    items.map((item) =>
      semaphore.withPermit(() => fn(item))
    ),
  );
}

// Usage example
const results = await mapConcurrent(
  urls,
  async (url) => {
    const response = await fetch(url);
    return response.json();
  },
  5, // Max 5 concurrent
);
```

---

## 6. Structured Concurrency in Rust

### 6.1 tokio::select! Macro

```rust
use tokio::time::{sleep, Duration};

// Wait for the first completion with select!
async fn fetch_with_timeout() -> Result<Data, Error> {
    tokio::select! {
        result = fetch_data() => result,
        _ = sleep(Duration::from_secs(5)) => {
            Err(Error::Timeout)
        }
    }
    // Returns the result of whichever completes first
    // The other is cancelled (Future is dropped)
}

// Receiving from multiple sources
async fn handle_messages(
    mut ws_rx: WebSocketReceiver,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            msg = ws_rx.next() => {
                match msg {
                    Some(Ok(message)) => handle_message(message).await,
                    Some(Err(e)) => {
                        eprintln!("WebSocket error: {}", e);
                        break;
                    }
                    None => break,
                }
            }
            _ = shutdown_rx.changed() => {
                println!("Shutdown signal received");
                break;
            }
        }
    }
}
```

### 6.2 tokio::spawn and JoinSet

```rust
use tokio::task::JoinSet;

// JoinSet: Structured management of concurrent tasks
async fn process_items(items: Vec<Item>) -> Vec<Result<ProcessResult, Error>> {
    let mut set = JoinSet::new();

    for item in items {
        set.spawn(async move {
            process_item(item).await
        });
    }

    let mut results = Vec::new();
    while let Some(result) = set.join_next().await {
        match result {
            Ok(process_result) => results.push(process_result),
            Err(join_error) => {
                eprintln!("Task panicked: {}", join_error);
            }
        }
    }

    results
}

// JoinSet + concurrency limit
async fn process_with_limit(
    items: Vec<Item>,
    max_concurrent: usize,
) -> Vec<ProcessResult> {
    let mut set = JoinSet::new();
    let mut results = Vec::new();
    let mut iter = items.into_iter();

    // Submit initial batch
    for _ in 0..max_concurrent {
        if let Some(item) = iter.next() {
            set.spawn(async move { process_item(item).await });
        }
    }

    // Submit next task as each one completes
    while let Some(result) = set.join_next().await {
        if let Ok(Ok(r)) = result {
            results.push(r);
        }
        if let Some(item) = iter.next() {
            set.spawn(async move { process_item(item).await });
        }
    }

    results
}

// Scoped tasks (Rust-specific)
// Local task management using tokio::task::LocalSet
async fn scoped_tasks() {
    let local = tokio::task::LocalSet::new();

    local.run_until(async {
        let handle1 = tokio::task::spawn_local(async {
            // Local task (Send not required)
            process_local_data().await
        });

        let handle2 = tokio::task::spawn_local(async {
            process_another_local_data().await
        });

        let (r1, r2) = tokio::join!(handle1, handle2);
        println!("Results: {:?}, {:?}", r1, r2);
    }).await;
    // All local tasks are complete when exiting LocalSet scope
}
```

---

## 7. Principles of Structured Concurrency

### 7.1 Three Core Principles

```
Three Principles:

  1. Child tasks live within the parent's scope
     -> Parent terminates = children terminate (leak prevention)
     -> Task lifetime matches the scope
     -> Parent-child relationships are clear when debugging

  2. Error propagation
     -> Child errors propagate to the parent
     -> Errors are not silently swallowed
     -> ExceptionGroup (Python) can handle multiple errors

  3. Cancellation propagation
     -> If the parent is cancelled, children are cancelled too
     -> If one child fails, siblings are also cancelled (coroutineScope)
     -> Cancellation is cooperative

Benefits:
  + Resource leak prevention
  + Reliable error handling
  + Code readability (clear scopes)
  + Ease of debugging
  + Improved testability
  + Ease of reasoning (function exit = all child tasks complete)
```

### 7.2 Cooperative Cancellation

```
Cancellation is a "request," not a "force":

  Cooperative cancellation:
    -> The task that receives the cancellation request voluntarily stops
    -> Tasks check for cancellation at safe stopping points
    -> An opportunity for cleanup is provided

  Cancellation points in each language:
    Kotlin: delay(), yield(), ensureActive(), suspend functions
    Swift:  Task.checkCancellation(), await
    Python: await (asyncio.CancelledError is raised)
    Rust:   When a Future's poll returns Pending

  Best practices on cancellation:
    1. Re-raise CancelledError/CancellationException
    2. Release resources in a finally block
    3. Protect critical sections from cancellation
       -> Kotlin: withContext(NonCancellable)
       -> Python: asyncio.shield()
    4. Periodically check for cancellation
```

### 7.3 Design Pattern Comparison

```
Pattern 1: All or Nothing (all succeed or all fail)
  -> Kotlin: coroutineScope
  -> Swift:  withThrowingTaskGroup
  -> Python: asyncio.TaskGroup
  -> Use case: Transaction-like operations, when all data is required

Pattern 2: Best Effort (succeed as much as possible)
  -> Kotlin: supervisorScope
  -> Swift:  withTaskGroup (with individual error handling)
  -> Python: TaskGroup + except*
  -> JS/TS: Promise.allSettled
  -> Use case: Dashboards, when partial results are acceptable

Pattern 3: First Success (adopt the first success)
  -> Kotlin: select
  -> Swift:  TaskGroup + cancelAll
  -> Python: asyncio.wait(FIRST_COMPLETED)
  -> JS/TS: Promise.race
  -> Use case: Hedge requests, multi-source fetching

Pattern 4: Fan-Out/Fan-In
  -> Multiple producers and consumers
  -> Combined with channels or queues
  -> Use case: Pipeline processing, parallel data processing
```

---

## 8. Practical Application Patterns

### 8.1 Concurrent Microservice Calls

```typescript
// Concurrent API calls in BFF (Backend for Frontend) pattern
class DashboardBFF {
  async getDashboard(userId: string): Promise<DashboardResponse> {
    const [
      userResult,
      ordersResult,
      notificationsResult,
      recommendationsResult,
    ] = await Promise.allSettled([
      // Required: User information
      this.userService.getUser(userId),
      // Required: Order history
      this.orderService.getOrders(userId),
      // Optional: Notifications (failure is acceptable)
      this.notificationService.getUnread(userId),
      // Optional: Recommendations (failure is acceptable)
      this.recommendationService.getForUser(userId),
    ]);

    // Check required data
    if (userResult.status === 'rejected') {
      throw new ServiceError('Failed to fetch user data', userResult.reason);
    }
    if (ordersResult.status === 'rejected') {
      throw new ServiceError('Failed to fetch orders', ordersResult.reason);
    }

    return {
      user: userResult.value,
      orders: ordersResult.value,
      notifications: notificationsResult.status === 'fulfilled'
        ? notificationsResult.value
        : [],
      recommendations: recommendationsResult.status === 'fulfilled'
        ? recommendationsResult.value
        : [],
    };
  }
}
```

### 8.2 Batch Processing

```python
import asyncio
from typing import TypeVar, Callable, Awaitable

T = TypeVar('T')
R = TypeVar('R')


async def batch_process(
    items: list[T],
    processor: Callable[[T], Awaitable[R]],
    batch_size: int = 50,
    max_concurrent: int = 10,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[R], list[tuple[T, Exception]]]:
    """Batch processing using structured concurrency"""
    results: list[R] = []
    errors: list[tuple[T, Exception]] = []
    completed = 0

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_item(item: T) -> tuple[T, R | None, Exception | None]:
            async with semaphore:
                try:
                    result = await processor(item)
                    return (item, result, None)
                except Exception as e:
                    return (item, None, e)

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(process_item(item))
                for item in batch
            ]

        for task in tasks:
            item, result, error = task.result()
            if error:
                errors.append((item, error))
            else:
                results.append(result)

        completed += len(batch)
        if on_progress:
            on_progress(completed, len(items))

    return results, errors


# Usage example
async def main():
    users = await fetch_all_users()

    results, errors = await batch_process(
        users,
        send_notification,
        batch_size=100,
        max_concurrent=20,
        on_progress=lambda done, total: print(f"{done}/{total}"),
    )

    print(f"Sent: {len(results)}, Failed: {len(errors)}")
    for user, error in errors:
        print(f"  Failed for {user.id}: {error}")
```

### 8.3 Health Checks

```kotlin
// Health check for multiple dependent services
data class HealthStatus(
    val service: String,
    val healthy: Boolean,
    val latencyMs: Long,
    val error: String? = null,
)

suspend fun checkAllHealth(): List<HealthStatus> = supervisorScope {
    val services = mapOf(
        "database" to { checkDatabase() },
        "redis" to { checkRedis() },
        "elasticsearch" to { checkElasticsearch() },
        "external-api" to { checkExternalApi() },
    )

    services.map { (name, check) ->
        async {
            val start = System.currentTimeMillis()
            try {
                withTimeout(5000) { check() }
                HealthStatus(
                    service = name,
                    healthy = true,
                    latencyMs = System.currentTimeMillis() - start,
                )
            } catch (e: Exception) {
                HealthStatus(
                    service = name,
                    healthy = false,
                    latencyMs = System.currentTimeMillis() - start,
                    error = e.message,
                )
            }
        }
    }.awaitAll()
}
```

---

## 9. Testing Strategies

### 9.1 Testing Structured Concurrency

```kotlin
// Kotlin: Test dispatchers
@Test
fun `dashboard loads all data concurrently`() = runTest {
    val userService = FakeUserService()
    val orderService = FakeOrderService()

    val dashboard = loadDashboard(userService, orderService)

    assertEquals("Taro Tanaka", dashboard.user.name)
    assertEquals(3, dashboard.orders.size)
}

@Test
fun `partial failure returns fallback data`() = runTest {
    val userService = FakeUserService()
    val orderService = FailingOrderService()

    val dashboard = loadDashboardResilient(userService, orderService)

    assertEquals("Taro Tanaka", dashboard.user.name)
    assertEquals(emptyList(), dashboard.orders) // Fallback
}

@Test
fun `cancellation propagates to child tasks`() = runTest {
    val job = launch {
        loadDashboard(
            SlowUserService(delay = 10.seconds),
            SlowOrderService(delay = 10.seconds),
        )
    }

    advanceTimeBy(1.seconds)
    job.cancel()

    assertTrue(job.isCancelled)
    // Verify that child tasks are also cancelled
}
```

```python
# Python: Testing structured concurrency
import pytest
import asyncio


@pytest.mark.asyncio
async def test_task_group_all_succeed():
    """All tasks succeed"""
    results = []

    async with asyncio.TaskGroup() as tg:
        async def task(value):
            await asyncio.sleep(0.01)
            results.append(value)

        tg.create_task(task(1))
        tg.create_task(task(2))
        tg.create_task(task(3))

    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_task_group_one_fails():
    """When one task fails, the others are cancelled"""
    with pytest.raises(ExceptionGroup) as exc_info:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(asyncio.sleep(10))  # This gets cancelled
            tg.create_task(failing_task())       # This fails

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ValueError)


@pytest.mark.asyncio
async def test_cancellation_propagation():
    """Cancellation propagates to child tasks"""
    cancelled = asyncio.Event()

    async def cancellable_task():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_timeout_with_task_group():
    """Timeout cancels the entire TaskGroup"""
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(asyncio.sleep(10))
                tg.create_task(asyncio.sleep(10))
```

---

## 10. Anti-Patterns

### 10.1 Patterns to Avoid

```
Anti-Pattern 1: Fire and Forget
  x Bad:
    function handleRequest() {
      sendEmail(user.email);  // Does not wait for result, does not detect errors
      return { ok: true };
    }

  o Good:
    function handleRequest() {
      // Enqueue to a job queue (reliable async processing)
      await jobQueue.enqueue('send-email', { email: user.email });
      return { ok: true };
    }

Anti-Pattern 2: Ignoring Cancellation Indefinitely
  x Bad:
    async def process():
        while True:
            data = compute_heavy()  # No cancellation point
            results.append(data)

  o Good:
    async def process():
        while True:
            await asyncio.sleep(0)  # Cancellation check
            data = compute_heavy()
            results.append(data)

Anti-Pattern 3: Swallowing CancelledError
  x Bad:
    async def task():
        try:
            await operation()
        except Exception:  # Also catches CancelledError
            pass

  o Good:
    async def task():
        try:
            await operation()
        except asyncio.CancelledError:
            raise  # Must always re-raise
        except Exception:
            pass

Anti-Pattern 4: Unnecessary Global Scope
  x Bad (Kotlin):
    fun handleRequest() {
      GlobalScope.launch { ... }  // No lifecycle management
    }

  o Good (Kotlin):
    suspend fun handleRequest() = coroutineScope {
      launch { ... }  // Managed within scope
    }

Anti-Pattern 5: Excessive Concurrency
  x Bad:
    // Process all 100,000 items simultaneously
    await Promise.all(items.map(item => process(item)));

  o Good:
    // Limit concurrency
    await mapConcurrent(items, process, 20);
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Create test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
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

Extend the basic implementation with the following features.

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
        """Delete by key"""
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
    assert ex.add("d", 4) == False  # Size limit
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

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be aware of algorithm complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point to keep in mind when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend building a solid understanding of the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Language | Structured Concurrency | Scope | Cancellation | Error Propagation |
|----------|----------------------|-------|--------------|-------------------|
| Kotlin | coroutineScope | Waits for all children | CancellationException | Automatic propagation |
| Kotlin | supervisorScope | Waits for all children | Independent | Independent handling |
| Swift | async let | Waits for all children | Automatic cancellation | Propagated via throws |
| Swift | TaskGroup | Waits for all children | cancelAll() | Propagated via throws |
| Python | asyncio.TaskGroup | Waits for all children | CancelledError | ExceptionGroup |
| Rust | tokio JoinSet | Explicit waiting | Future drop | JoinError |
| JS/TS | Promise.all | Explicit waiting | AbortController | reject propagation |

---

## Recommended Next Guides

---

## References
1. Elizarov, R. "Structured Concurrency." vorpus.org, 2018.
2. Swift Evolution. "SE-0304: Structured Concurrency."
3. Python Documentation. "asyncio -- TaskGroup." docs.python.org.
4. Kotlin Documentation. "Coroutines guide." kotlinlang.org.
5. Smith, N. "Notes on structured concurrency, or: Go statement considered harmful." 2018.
6. Tokio Documentation. "Working with Tasks." tokio.rs.
7. Apple Developer. "Concurrency -- Swift Programming Language." developer.apple.com.
8. Syme, D. "The early history of F# async." fsharpforfunandprofit.com.
9. Sustrik, M. "Structured Concurrency." 250bpm.com, 2016.
10. Nygard, M. "Release It!" Pragmatic Bookshelf, 2018.
