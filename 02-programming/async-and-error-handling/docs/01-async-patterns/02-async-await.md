# async/await

> async/await is syntactic sugar that lets you "read asynchronous code as if it were synchronous." It provides an intuitive way to write Promise-based asynchronous operations. This guide covers implementations in JavaScript, Python, Rust, and C#, along with concurrent execution patterns.

## What You Will Learn in This Chapter

- [ ] Understand the mechanism and underlying principles of async/await
- [ ] Grasp the differences in async/await across languages
- [ ] Learn efficient concurrent execution patterns
- [ ] Master error handling best practices
- [ ] Implement cancellation, timeout, and retry patterns
- [ ] Understand testing and debugging techniques


## Prerequisites

Before reading this guide, the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Promises](./01-promises.md)

---

## 1. async/await Fundamentals

### 1.1 Concept and Underlying Principles

async/await is syntactic sugar that allows asynchronous operations to be written as if they were synchronous code. Internally, it is based on Promises (JavaScript), Futures (Rust), or Coroutines (Python).

```
async function:
  -> Returns a Promise
  -> Return values are automatically wrapped with Promise.resolve()
  -> Thrown values are wrapped with Promise.reject()

await expression:
  -> Suspends function execution until the Promise resolves
  -> Returns the resolved value
  -> Can only be used inside async functions (Top-Level Await added in ES2022)
  -> Rejected Promises are thrown as exceptions

Internal behavior:
  async function f() {
    const a = await fetchA();  // Suspends here
    const b = await fetchB();  // Resumes after a resolves
    return a + b;
  }

  // Equivalent to:
  function f() {
    return fetchA()
      .then(a => fetchB().then(b => a + b));
  }
```

### 1.2 async/await as a State Machine

The compiler (or engine) transforms async functions into state machines. This enables efficient suspension and resumption.

```typescript
// Code written by the developer
async function process() {
  console.log("Step 1");
  const a = await fetchA();
  console.log("Step 2");
  const b = await fetchB(a);
  console.log("Step 3");
  return a + b;
}

// Conceptual transformation inside the engine (pseudocode)
function process() {
  let state = 0;
  let a: any, b: any;

  function step(value?: any): Promise<any> {
    switch (state) {
      case 0:
        console.log("Step 1");
        state = 1;
        return fetchA().then(step);
      case 1:
        a = value;
        console.log("Step 2");
        state = 2;
        return fetchB(a).then(step);
      case 2:
        b = value;
        console.log("Step 3");
        return Promise.resolve(a + b);
    }
  }
  return step();
}
```

### 1.3 Relationship with the Microtask Queue

```typescript
// async/await uses the microtask queue
async function demo() {
  console.log("1: async function starts (executes synchronously)");
  const result = await Promise.resolve("hello");
  // ^ Suspends here and places the continuation in the microtask queue
  console.log("3: Resumes after await (executes as a microtask)");
  return result;
}

console.log("0: Before the call");
demo().then(() => console.log("4: then callback"));
console.log("2: After the call (executes synchronously)");

// Output order:
// 0: Before the call
// 1: async function starts (executes synchronously)
// 2: After the call (executes synchronously)
// 3: Resumes after await (executes as a microtask)
// 4: then callback
```

### 1.4 Top-Level Await (ES2022)

```typescript
// Top-Level Await is available in ES modules

// config.ts - Asynchronous loading of configuration
const response = await fetch("/api/config");
export const config = await response.json();

// main.ts - Automatically awaited on import
import { config } from "./config.ts";
console.log(config.apiKey); // Executes after configuration loading completes

// Caveats:
// 1. Cannot be used with CommonJS (require)
// 2. Affects module loading order
// 3. Be cautious of circular dependencies
// 4. Convenient for server-side initialization
```

---

## 2. JavaScript/TypeScript

### 2.1 Basic Patterns

```typescript
// Basic async function
async function getUserProfile(userId: string): Promise<UserProfile> {
  const user = await userRepo.findById(userId);
  if (!user) throw new Error("User not found");

  const [orders, reviews] = await Promise.all([
    orderRepo.findByUserId(userId),
    reviewRepo.findByUserId(userId),
  ]);

  return { user, orders, reviews };
}

// async with arrow functions
const fetchData = async (url: string): Promise<Response> => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }
  return response;
};

// async in methods
class UserService {
  async findById(id: string): Promise<User | null> {
    const cached = await this.cache.get(`user:${id}`);
    if (cached) return cached;

    const user = await this.db.query("SELECT * FROM users WHERE id = $1", [id]);
    if (user) {
      await this.cache.set(`user:${id}`, user, { ttl: 300 });
    }
    return user;
  }

  // Note: async cannot be used with getters
  // async get name() {} // SyntaxError

  // Alternative pattern
  async getName(): Promise<string> {
    const profile = await this.loadProfile();
    return profile.name;
  }
}
```

### 2.2 Error Handling

```typescript
// Basic error handling with try/catch
async function safeGetUser(userId: string): Promise<User | null> {
  try {
    return await userRepo.findById(userId);
  } catch (error) {
    logger.error("Failed to get user", { userId, error });
    return null;
  }
}

// Fine-grained error handling with multiple async operations
async function processOrder(orderId: string): Promise<OrderResult> {
  // Step 1: Retrieve the order
  let order: Order;
  try {
    order = await orderRepo.findById(orderId);
  } catch (error) {
    throw new OrderNotFoundError(orderId, { cause: error });
  }

  // Step 2: Check inventory
  try {
    await inventoryService.checkAvailability(order.items);
  } catch (error) {
    if (error instanceof OutOfStockError) {
      return { status: "out_of_stock", items: error.unavailableItems };
    }
    throw error; // Re-throw unexpected errors
  }

  // Step 3: Process payment
  try {
    const payment = await paymentService.charge(order.total, order.paymentMethod);
    return { status: "completed", payment };
  } catch (error) {
    // Restore inventory on payment failure
    await inventoryService.release(order.items);
    throw new PaymentFailedError(orderId, { cause: error });
  }
}

// Result type pattern (without exceptions)
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

async function safeAsync<T>(
  fn: () => Promise<T>
): Promise<Result<T>> {
  try {
    const value = await fn();
    return { ok: true, value };
  } catch (error) {
    return { ok: false, error: error as Error };
  }
}

// Usage example
async function handleRequest() {
  const userResult = await safeAsync(() => getUser("123"));
  if (!userResult.ok) {
    console.error("User fetch failed:", userResult.error.message);
    return;
  }

  const ordersResult = await safeAsync(() => getOrders(userResult.value.id));
  if (!ordersResult.ok) {
    console.error("Orders fetch failed:", ordersResult.error.message);
    return;
  }

  return { user: userResult.value, orders: ordersResult.value };
}
```

### 2.3 Sequential vs Concurrent

```typescript
// Sequential execution (serial)
async function sequential(): Promise<void> {
  const a = await fetchA(); // 100ms
  const b = await fetchB(); // 200ms
  // Total: 300ms (serial)
}

// Concurrent execution
async function concurrent(): Promise<void> {
  const [a, b] = await Promise.all([
    fetchA(), // 100ms |
    fetchB(), // 200ms | concurrent
  ]);        //       |
  // Total: 200ms (concurrent)
}

// Important: Execution starts at the moment a Promise is created
async function earlyStart(): Promise<void> {
  // Create Promises first (execution starts)
  const promiseA = fetchA(); // Starts immediately
  const promiseB = fetchB(); // Starts immediately

  // Wait for both results
  const a = await promiseA;
  const b = await promiseB;
  // Equivalent concurrent execution to Promise.all
}

// However, be careful with error handling
async function earlyStartWithErrorHandling(): Promise<void> {
  const promiseA = fetchA();
  const promiseB = fetchB();

  // If promiseB rejects first, we're still waiting on promiseA's await
  // -> May trigger an unhandled rejection warning
  // -> Using Promise.all is safer
  try {
    const [a, b] = await Promise.all([promiseA, promiseB]);
  } catch (error) {
    // Catches all errors
  }
}
```

### 2.4 Iteration Patterns

```typescript
// Bad: for...of + await (sequential execution)
async function processSequential(urls: string[]): Promise<Response[]> {
  const results: Response[] = [];
  for (const url of urls) {
    const response = await fetch(url); // One at a time...
    results.push(response);
  }
  return results;
}

// Good: Promise.all (fully concurrent)
async function processAllConcurrent(urls: string[]): Promise<Response[]> {
  return Promise.all(urls.map(url => fetch(url)));
}

// Good: Concurrency-limited execution (up to N at a time)
async function processWithConcurrencyLimit<T>(
  items: T[],
  fn: (item: T) => Promise<any>,
  limit: number
): Promise<any[]> {
  const results: any[] = [];
  const executing: Promise<void>[] = [];

  for (const [index, item] of items.entries()) {
    const promise = fn(item).then(result => {
      results[index] = result;
    });

    executing.push(promise);

    if (executing.length >= limit) {
      await Promise.race(executing);
      // Remove completed Promises
      const completed = executing.findIndex(
        p => p === Promise.race([p]).then(() => p)
      );
    }
  }

  await Promise.all(executing);
  return results;
}

// A more refined concurrency limiter: Semaphore
class Semaphore {
  private queue: (() => void)[] = [];
  private running = 0;

  constructor(private readonly limit: number) {}

  async acquire(): Promise<void> {
    if (this.running < this.limit) {
      this.running++;
      return;
    }
    return new Promise<void>(resolve => {
      this.queue.push(resolve);
    });
  }

  release(): void {
    this.running--;
    const next = this.queue.shift();
    if (next) {
      this.running++;
      next();
    }
  }

  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

// Semaphore usage example
async function fetchAllWithLimit(urls: string[], limit: number) {
  const semaphore = new Semaphore(limit);
  return Promise.all(
    urls.map(url => semaphore.run(() => fetch(url)))
  );
}

// for-await-of (async iteration)
async function* fetchPages(url: string): AsyncGenerator<Item[]> {
  let nextUrl: string | null = url;
  while (nextUrl) {
    const response = await fetch(nextUrl);
    const data = await response.json();
    yield data.items;
    nextUrl = data.nextPage;
  }
}

async function getAllItems(url: string): Promise<Item[]> {
  const allItems: Item[] = [];
  for await (const page of fetchPages(url)) {
    allItems.push(...page);
    console.log(`Fetched ${page.length} items, total: ${allItems.length}`);
  }
  return allItems;
}

// Async iteration of ReadableStream
async function readStream(stream: ReadableStream<Uint8Array>) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let result = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      result += decoder.decode(value, { stream: true });
    }
  } finally {
    reader.releaseLock();
  }
  return result;
}
```

### 2.5 Cancellation with AbortController

```typescript
// Basic cancellation
async function fetchWithCancel(
  url: string,
  signal?: AbortSignal
): Promise<Response> {
  const response = await fetch(url, { signal });
  return response;
}

const controller = new AbortController();
const promise = fetchWithCancel("/api/data", controller.signal);

// Cancel as needed
setTimeout(() => controller.abort(), 5000);

try {
  const result = await promise;
} catch (error) {
  if (error instanceof DOMException && error.name === "AbortError") {
    console.log("Request was cancelled");
  } else {
    throw error;
  }
}

// Generic function with timeout
async function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  message = "Operation timed out"
): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), ms);

  try {
    const result = await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        controller.signal.addEventListener("abort", () => {
          reject(new Error(message));
        });
      }),
    ]);
    return result;
  } finally {
    clearTimeout(timeoutId);
  }
}

// AbortSignal.timeout() (newer API)
async function fetchWithTimeout(url: string): Promise<Response> {
  return fetch(url, {
    signal: AbortSignal.timeout(5000), // 5-second timeout
  });
}

// Batch cancellation of multiple requests
class RequestManager {
  private controller = new AbortController();

  async fetch(url: string): Promise<Response> {
    return fetch(url, { signal: this.controller.signal });
  }

  cancelAll(): void {
    this.controller.abort();
    this.controller = new AbortController(); // Reset
  }
}

// Usage in React
function useAsyncEffect(
  effect: (signal: AbortSignal) => Promise<void>,
  deps: React.DependencyList
): void {
  React.useEffect(() => {
    const controller = new AbortController();
    effect(controller.signal).catch(error => {
      if (error.name !== "AbortError") {
        console.error(error);
      }
    });
    return () => controller.abort(); // Cleanup
  }, deps);
}

// Usage example
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = React.useState<User | null>(null);

  useAsyncEffect(async (signal) => {
    const response = await fetch(`/api/users/${userId}`, { signal });
    const data = await response.json();
    setUser(data);
  }, [userId]);

  return user ? <div>{user.name}</div> : <div>Loading...</div>;
}
```

### 2.6 Retry Patterns

```typescript
// Retry with exponential backoff
async function withRetry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    baseDelay?: number;
    maxDelay?: number;
    shouldRetry?: (error: Error, attempt: number) => boolean;
    onRetry?: (error: Error, attempt: number) => void;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 30000,
    shouldRetry = () => true,
    onRetry,
  } = options;

  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === maxRetries || !shouldRetry(lastError, attempt)) {
        throw lastError;
      }

      onRetry?.(lastError, attempt);

      // Exponential backoff + jitter
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
        maxDelay
      );
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

// Usage example
const data = await withRetry(
  () => fetch("/api/data").then(r => r.json()),
  {
    maxRetries: 3,
    baseDelay: 1000,
    shouldRetry: (error, attempt) => {
      // Only retry on network errors or 5xx
      if (error instanceof TypeError) return true; // Network error
      if (error instanceof HttpError && error.status >= 500) return true;
      return false;
    },
    onRetry: (error, attempt) => {
      console.log(`Retry ${attempt + 1}: ${error.message}`);
    },
  }
);

// Retry with circuit breaker
class CircuitBreaker {
  private failures = 0;
  private lastFailureTime = 0;
  private state: "closed" | "open" | "half-open" = "closed";

  constructor(
    private readonly threshold: number = 5,
    private readonly resetTimeout: number = 60000
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === "open") {
      if (Date.now() - this.lastFailureTime > this.resetTimeout) {
        this.state = "half-open";
      } else {
        throw new Error("Circuit breaker is open");
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = 0;
    this.state = "closed";
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
    if (this.failures >= this.threshold) {
      this.state = "open";
    }
  }
}
```

---

## 3. Python

### 3.1 asyncio Basics

```python
import asyncio
from typing import Any

# Basic async function
async def get_user_profile(user_id: str) -> dict:
    user = await user_repo.find_by_id(user_id)
    if not user:
        raise ValueError("User not found")

    # Concurrent execution with asyncio.gather
    orders, reviews = await asyncio.gather(
        order_repo.find_by_user_id(user_id),
        review_repo.find_by_user_id(user_id),
    )
    return {"user": user, "orders": orders, "reviews": reviews}

# Execution
async def main():
    profile = await get_user_profile("user-123")
    print(profile)

asyncio.run(main())
```

### 3.2 Task Management

```python
import asyncio

# Creating tasks and concurrent execution
async def process_items(items: list[str]) -> list[dict]:
    tasks = [asyncio.create_task(fetch_item(item)) for item in items]
    return await asyncio.gather(*tasks)

# Task cancellation
async def cancellable_operation():
    task = asyncio.create_task(long_running_operation())

    # Cancel after 5 seconds
    await asyncio.sleep(5)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled")

# TaskGroup (Python 3.11+) - Structured concurrency
async def structured_concurrency():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_users())
        task2 = tg.create_task(fetch_orders())
        task3 = tg.create_task(fetch_products())
    # Reaches here after all tasks complete
    # If any task raises an exception,
    # the others are also cancelled
    users = task1.result()
    orders = task2.result()
    products = task3.result()
    return users, orders, products

# Error handling with TaskGroup
async def safe_task_group():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(might_fail_1())
            tg.create_task(might_fail_2())
    except* ValueError as eg:
        # Handle ExceptionGroup (Python 3.11+)
        for exc in eg.exceptions:
            print(f"ValueError: {exc}")
    except* TypeError as eg:
        for exc in eg.exceptions:
            print(f"TypeError: {exc}")
```

### 3.3 Timeouts and Deadlines

```python
import asyncio

# Timeout with wait_for
async def with_timeout():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        print("Timed out")

# asyncio.timeout (Python 3.11+)
async def modern_timeout():
    async with asyncio.timeout(5.0):
        result = await slow_operation()
        return result

# Deadline
async def with_deadline():
    deadline = asyncio.get_event_loop().time() + 10.0
    async with asyncio.timeout_at(deadline):
        await step1()
        await step2()  # Executes within remaining time
        await step3()  # All within 10 seconds total

# Processing partial completion with asyncio.wait
async def partial_results():
    tasks = [
        asyncio.create_task(fetch(url))
        for url in urls
    ]

    # Get the first completed result
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
        print(f"Completed: {task.result()}")

    # Cancel the rest
    for task in pending:
        task.cancel()

# Streaming results with as_completed
async def stream_results():
    tasks = [
        asyncio.create_task(fetch(url))
        for url in urls
    ]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Got result: {result}")
        # Processed in completion order
```

### 3.4 Async Context Managers and Iterators

```python
import asyncio
from contextlib import asynccontextmanager

# Async context manager (class-based)
class AsyncDatabaseConnection:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn = None

    async def __aenter__(self):
        self.conn = await asyncpg.connect(self.dsn)
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            await self.conn.close()
        return False  # Re-raise the exception

# Usage
async def query_users():
    async with AsyncDatabaseConnection("postgresql://...") as conn:
        rows = await conn.fetch("SELECT * FROM users")
        return rows

# Decorator-based async context manager
@asynccontextmanager
async def managed_transaction(pool):
    conn = await pool.acquire()
    tx = conn.transaction()
    await tx.start()
    try:
        yield conn
        await tx.commit()
    except Exception:
        await tx.rollback()
        raise
    finally:
        await pool.release(conn)

# Async iterator
class AsyncPaginator:
    def __init__(self, url: str, page_size: int = 100):
        self.url = url
        self.page_size = page_size
        self.page = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.page += 1
        data = await fetch_page(self.url, self.page, self.page_size)
        if not data:
            raise StopAsyncIteration
        return data

# Usage
async def process_all_pages():
    async for page in AsyncPaginator("/api/users"):
        for user in page:
            await process_user(user)

# Async generator
async def async_range(start: int, stop: int, delay: float = 0.1):
    for i in range(start, stop):
        await asyncio.sleep(delay)
        yield i

async def use_async_generator():
    async for value in async_range(0, 10):
        print(value)
```

### 3.5 Practical HTTP Client with aiohttp

```python
import aiohttp
import asyncio
from typing import Any

class AsyncHttpClient:
    def __init__(self, base_url: str, max_concurrent: int = 10):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def get(self, path: str) -> dict[str, Any]:
        async with self.semaphore:
            async with self.session.get(path) as response:
                response.raise_for_status()
                return await response.json()

    async def get_many(self, paths: list[str]) -> list[dict[str, Any]]:
        tasks = [self.get(path) for path in paths]
        return await asyncio.gather(*tasks)

    async def get_with_retry(
        self, path: str, max_retries: int = 3
    ) -> dict[str, Any]:
        for attempt in range(max_retries):
            try:
                return await self.get(path)
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise
                delay = 2 ** attempt
                print(f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                await asyncio.sleep(delay)

# Usage example
async def main():
    async with AsyncHttpClient("https://api.example.com") as client:
        # Fetch 100 users concurrently (up to 10 at a time)
        paths = [f"/users/{i}" for i in range(100)]
        users = await client.get_many(paths)
        print(f"Fetched {len(users)} users")
```

---

## 4. Rust

### 4.1 The Future Trait and async/await

```rust
// Rust: async/await (tokio runtime)
use tokio;

// Rust's async fn returns a type that implements the Future trait
// Future trait:
// trait Future {
//     type Output;
//     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
// }

async fn get_user_profile(user_id: &str) -> Result<UserProfile, AppError> {
    let user = user_repo.find_by_id(user_id).await?;

    // Concurrent execution with tokio::join!
    let (orders, reviews) = tokio::join!(
        order_repo.find_by_user_id(user_id),
        review_repo.find_by_user_id(user_id),
    );

    Ok(UserProfile {
        user,
        orders: orders?,
        reviews: reviews?,
    })
}

#[tokio::main]
async fn main() {
    let profile = get_user_profile("user-123").await.unwrap();
    println!("{:?}", profile);
}

// Characteristics of Rust's async:
// -> Zero-cost abstraction (compiles to a state machine)
// -> Runtime is separate (tokio, async-std, smol)
// -> Futures are lazy (not executed until awaited)
// -> Send + 'static constraints (must be movable across threads)
```

### 4.2 tokio Concurrent Execution Patterns

```rust
use tokio;
use tokio::time::{timeout, Duration};

// Spawning tasks with tokio::spawn
async fn spawn_tasks() -> Result<(), Box<dyn std::error::Error>> {
    let handle1 = tokio::spawn(async {
        // Independent task
        fetch_users().await
    });

    let handle2 = tokio::spawn(async {
        fetch_orders().await
    });

    // Get both results
    let (users, orders) = (handle1.await??, handle2.await??);
    println!("Users: {}, Orders: {}", users.len(), orders.len());
    Ok(())
}

// Racing with tokio::select!
async fn fetch_with_timeout() -> Result<Data, AppError> {
    tokio::select! {
        result = fetch_data() => {
            result.map_err(|e| AppError::Fetch(e))
        }
        _ = tokio::time::sleep(Duration::from_secs(5)) => {
            Err(AppError::Timeout)
        }
    }
}

// Using tokio::select! to take the first response
async fn fastest_mirror(mirrors: Vec<String>) -> Result<Data, AppError> {
    tokio::select! {
        result = fetch_from(&mirrors[0]) => result,
        result = fetch_from(&mirrors[1]) => result,
        result = fetch_from(&mirrors[2]) => result,
    }
}

// Timeout
async fn with_timeout() -> Result<Data, AppError> {
    match timeout(Duration::from_secs(10), fetch_data()).await {
        Ok(Ok(data)) => Ok(data),
        Ok(Err(e)) => Err(AppError::Fetch(e)),
        Err(_) => Err(AppError::Timeout),
    }
}

// Buffered channel
async fn producer_consumer() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);

    // Producer
    let producer = tokio::spawn(async move {
        for i in 0..1000 {
            tx.send(format!("message {}", i)).await.unwrap();
        }
    });

    // Consumer
    let consumer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            process_message(&msg).await;
        }
    });

    let _ = tokio::join!(producer, consumer);
}
```

### 4.3 Stream (Async Iterator)

```rust
use tokio_stream::{self as stream, StreamExt};
use futures::stream::{self, Stream};

// Creating and consuming a Stream
async fn process_stream() {
    let mut stream = stream::iter(vec![1, 2, 3, 4, 5])
        .map(|x| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            x * 2
        })
        .buffered(3); // Process up to 3 concurrently

    while let Some(value) = stream.next().await {
        println!("Got: {}", value);
    }
}

// Custom Stream
fn countdown(from: u32) -> impl Stream<Item = u32> {
    stream::unfold(from, |state| async move {
        if state == 0 {
            None
        } else {
            tokio::time::sleep(Duration::from_secs(1)).await;
            Some((state, state - 1))
        }
    })
}

// Stream composition
async fn merged_streams() {
    let stream1 = stream::iter(vec![1, 3, 5]);
    let stream2 = stream::iter(vec![2, 4, 6]);

    let mut merged = stream::select(stream1, stream2);
    while let Some(value) = merged.next().await {
        println!("{}", value);
    }
}

// Processing with concurrency limit
async fn process_with_limit(
    items: Vec<String>,
    limit: usize,
) -> Vec<Result<Data, Error>> {
    stream::iter(items)
        .map(|item| async move { fetch_data(&item).await })
        .buffer_unordered(limit)
        .collect()
        .await
}
```

### 4.4 Error Handling and the ? Operator

```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum AppError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Timeout")]
    Timeout,
}

// Concise error handling with the ? operator
async fn get_user_with_orders(user_id: &str) -> Result<UserWithOrders, AppError> {
    let user = db::find_user(user_id)
        .await?  // sqlx::Error -> AppError::Database
        .ok_or_else(|| AppError::NotFound(user_id.to_string()))?;

    let orders = api::fetch_orders(user_id)
        .await?; // reqwest::Error -> AppError::Network

    Ok(UserWithOrders { user, orders })
}

// Retry
async fn with_retry<T, E, F, Fut>(
    mut f: F,
    max_retries: u32,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut attempt = 0;
    loop {
        match f().await {
            Ok(value) => return Ok(value),
            Err(e) if attempt < max_retries => {
                attempt += 1;
                eprintln!("Attempt {} failed: {:?}, retrying...", attempt, e);
                tokio::time::sleep(Duration::from_millis(
                    100 * 2u64.pow(attempt)
                )).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

---

## 5. Go (goroutine + channel)

Go does not have async/await syntax, but achieves equivalent functionality through goroutines and channels.

### 5.1 Basic Patterns

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// In Go, all functions appear "synchronous"
// Asynchrony is achieved through goroutines
func getUserProfile(ctx context.Context, userID string) (*UserProfile, error) {
    user, err := userRepo.FindByID(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("find user: %w", err)
    }

    // Concurrent execution via goroutine + channel
    type ordersResult struct {
        orders []Order
        err    error
    }
    type reviewsResult struct {
        reviews []Review
        err     error
    }

    ordersCh := make(chan ordersResult, 1)
    reviewsCh := make(chan reviewsResult, 1)

    go func() {
        orders, err := orderRepo.FindByUserID(ctx, userID)
        ordersCh <- ordersResult{orders, err}
    }()

    go func() {
        reviews, err := reviewRepo.FindByUserID(ctx, userID)
        reviewsCh <- reviewsResult{reviews, err}
    }()

    or := <-ordersCh
    rr := <-reviewsCh

    if or.err != nil {
        return nil, fmt.Errorf("find orders: %w", or.err)
    }
    if rr.err != nil {
        return nil, fmt.Errorf("find reviews: %w", rr.err)
    }

    return &UserProfile{
        User:    user,
        Orders:  or.orders,
        Reviews: rr.reviews,
    }, nil
}
```

### 5.2 Concurrent Execution with errgroup

```go
import "golang.org/x/sync/errgroup"

func getDashboard(ctx context.Context, userID string) (*Dashboard, error) {
    var (
        profile       *Profile
        notifications []Notification
        stats         *Stats
    )

    g, ctx := errgroup.WithContext(ctx)

    g.Go(func() error {
        var err error
        profile, err = getProfile(ctx, userID)
        return err
    })

    g.Go(func() error {
        var err error
        notifications, err = getNotifications(ctx, userID)
        return err
    })

    g.Go(func() error {
        var err error
        stats, err = getStats(ctx, userID)
        return err
    })

    if err := g.Wait(); err != nil {
        return nil, err
    }

    return &Dashboard{
        Profile:       profile,
        Notifications: notifications,
        Stats:         stats,
    }, nil
}

// errgroup with concurrency limit
func processItems(ctx context.Context, items []string) error {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(10) // Up to 10 concurrent

    for _, item := range items {
        item := item // Capture loop variable (before Go 1.21)
        g.Go(func() error {
            return processItem(ctx, item)
        })
    }

    return g.Wait()
}
```

### 5.3 Cancellation and Timeout with Context

```go
import (
    "context"
    "time"
)

// Timeout
func fetchWithTimeout(url string) ([]byte, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err // May contain context.DeadlineExceeded
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}

// Cancellation propagation
func longOperation(ctx context.Context) error {
    for i := 0; i < 100; i++ {
        select {
        case <-ctx.Done():
            return ctx.Err() // context.Canceled or DeadlineExceeded
        default:
            // Continue processing
            if err := doStep(ctx, i); err != nil {
                return err
            }
        }
    }
    return nil
}

// Cancellation from parent context
func handler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context() // Cancelled when the client disconnects

    result, err := longOperation(ctx)
    if err != nil {
        if ctx.Err() != nil {
            // Client disconnected
            return
        }
        http.Error(w, err.Error(), 500)
        return
    }

    json.NewEncoder(w).Encode(result)
}
```

---

## 6. C#

### 6.1 Task-Based async/await

```csharp
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

// Basic
public async Task<UserProfile> GetUserProfileAsync(string userId)
{
    var user = await _userRepo.FindByIdAsync(userId);
    if (user == null)
        throw new NotFoundException($"User {userId} not found");

    var (orders, reviews) = await (
        _orderRepo.FindByUserIdAsync(userId),
        _reviewRepo.FindByUserIdAsync(userId)
    ).WhenAll();

    return new UserProfile(user, orders, reviews);
}

// ValueTask (value type, lightweight; effective when cache hits are frequent)
public ValueTask<User?> GetUserAsync(string userId)
{
    if (_cache.TryGetValue(userId, out var cached))
    {
        return ValueTask.FromResult(cached); // No heap allocation
    }

    return new ValueTask<User?>(GetUserFromDbAsync(userId));
}

private async Task<User?> GetUserFromDbAsync(string userId)
{
    var user = await _db.QueryAsync<User>(
        "SELECT * FROM Users WHERE Id = @Id", new { Id = userId });
    if (user != null)
    {
        _cache.Set(userId, user);
    }
    return user;
}

// CancellationToken
public async Task<Data> FetchDataAsync(
    string url,
    CancellationToken cancellationToken = default)
{
    using var client = new HttpClient();
    var response = await client.GetAsync(url, cancellationToken);
    response.EnsureSuccessStatusCode();

    var content = await response.Content.ReadAsStringAsync(cancellationToken);
    return JsonSerializer.Deserialize<Data>(content)!;
}

// Usage example
var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
try
{
    var data = await FetchDataAsync("https://api.example.com", cts.Token);
}
catch (OperationCanceledException)
{
    Console.WriteLine("Operation was cancelled or timed out");
}
```

### 6.2 Concurrent Execution Patterns

```csharp
// Task.WhenAll
public async Task<Dashboard> GetDashboardAsync(string userId)
{
    var profileTask = GetProfileAsync(userId);
    var notificationsTask = GetNotificationsAsync(userId);
    var statsTask = GetStatsAsync(userId);

    await Task.WhenAll(profileTask, notificationsTask, statsTask);

    return new Dashboard
    {
        Profile = await profileTask,       // Already completed
        Notifications = await notificationsTask,
        Stats = await statsTask,
    };
}

// Task.WhenAny (use the first completion)
public async Task<Data> FetchFromFastestAsync(IEnumerable<string> urls)
{
    var tasks = urls.Select(url => FetchDataAsync(url)).ToList();
    var completed = await Task.WhenAny(tasks);
    return await completed;
}

// Concurrency limiting with SemaphoreSlim
public async Task ProcessAllAsync(
    IEnumerable<string> items,
    int maxConcurrency = 10)
{
    using var semaphore = new SemaphoreSlim(maxConcurrency);
    var tasks = items.Select(async item =>
    {
        await semaphore.WaitAsync();
        try
        {
            await ProcessItemAsync(item);
        }
        finally
        {
            semaphore.Release();
        }
    });

    await Task.WhenAll(tasks);
}

// IAsyncEnumerable (C# 8.0+)
public async IAsyncEnumerable<User> GetAllUsersAsync(
    [EnumeratorCancellation] CancellationToken ct = default)
{
    int page = 0;
    while (true)
    {
        var users = await _db.GetUsersPageAsync(page++, 100, ct);
        if (users.Count == 0) yield break;

        foreach (var user in users)
        {
            yield return user;
        }
    }
}

// Consumption
await foreach (var user in GetAllUsersAsync())
{
    Console.WriteLine(user.Name);
}
```

### 6.3 ConfigureAwait and Synchronization Context

```csharp
// Considerations for UI threads
// WPF and Windows Forms have a SynchronizationContext
public async void Button_Click(object sender, EventArgs e)
{
    var data = await FetchDataAsync("/api/data");
    // ^ Returns to the UI thread by default

    // UI update (executes on the UI thread)
    textBox.Text = data.ToString();
}

// Use ConfigureAwait(false) in library code
public async Task<Data> FetchDataLibraryAsync(string url)
{
    var response = await _client.GetAsync(url)
        .ConfigureAwait(false); // Continue on thread pool

    var content = await response.Content.ReadAsStringAsync()
        .ConfigureAwait(false);

    return JsonSerializer.Deserialize<Data>(content)!;
}

// Avoiding deadlocks
// Bad: Calling async from synchronous method causes deadlock
public Data GetDataSync()
{
    // Calling .Result on the UI thread causes deadlock!
    return FetchDataAsync("/api/data").Result;
}

// Good: async all the way (make everything async)
public async Task<Data> GetDataAsync()
{
    return await FetchDataAsync("/api/data");
}
```

---

## 7. Kotlin

### 7.1 Coroutine Basics

```kotlin
import kotlinx.coroutines.*

// suspend function
suspend fun getUserProfile(userId: String): UserProfile {
    val user = userRepo.findById(userId)
        ?: throw NotFoundException("User $userId not found")

    // Concurrent execution with coroutineScope
    return coroutineScope {
        val ordersDeferred = async { orderRepo.findByUserId(userId) }
        val reviewsDeferred = async { reviewRepo.findByUserId(userId) }

        UserProfile(
            user = user,
            orders = ordersDeferred.await(),
            reviews = reviewsDeferred.await()
        )
    }
}

// CoroutineScope and dispatchers
fun main() = runBlocking {
    // Dispatchers.IO: For I/O operations
    val data = withContext(Dispatchers.IO) {
        fetchFromNetwork()
    }

    // Dispatchers.Default: For CPU-intensive tasks
    val processed = withContext(Dispatchers.Default) {
        heavyComputation(data)
    }

    println(processed)
}

// Structured concurrency
suspend fun processDashboard(userId: String): Dashboard {
    return coroutineScope {
        val profile = async { getProfile(userId) }
        val notifications = async { getNotifications(userId) }
        val stats = async { getStats(userId) }

        // If any fails, the others are automatically cancelled
        Dashboard(
            profile = profile.await(),
            notifications = notifications.await(),
            stats = stats.await()
        )
    }
}
```

### 7.2 Flow (Cold Stream)

```kotlin
import kotlinx.coroutines.flow.*

// Creating a Flow
fun fetchUsers(): Flow<User> = flow {
    var page = 0
    while (true) {
        val users = api.getUsers(page++)
        if (users.isEmpty()) break
        users.forEach { emit(it) }
    }
}

// Transforming and consuming a Flow
suspend fun processUsers() {
    fetchUsers()
        .filter { it.isActive }
        .map { enrichUser(it) }
        .buffer(10) // Buffering for concurrency
        .collect { user ->
            println("Processed: ${user.name}")
        }
}

// StateFlow (Hot Stream, for UI)
class UserViewModel : ViewModel() {
    private val _uiState = MutableStateFlow<UiState>(UiState.Loading)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    fun loadUser(userId: String) {
        viewModelScope.launch {
            _uiState.value = UiState.Loading
            try {
                val user = userRepo.findById(userId)
                _uiState.value = UiState.Success(user)
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.message ?: "Unknown error")
            }
        }
    }
}
```

### 7.3 Cancellation and Timeout

```kotlin
import kotlinx.coroutines.*

// withTimeout
suspend fun fetchWithTimeout(): Data {
    return withTimeout(5000L) { // 5-second timeout
        fetchData()
    }
    // TimeoutCancellationException is thrown
}

// withTimeoutOrNull (returns null instead of throwing)
suspend fun safeFetch(): Data? {
    return withTimeoutOrNull(5000L) {
        fetchData()
    }
}

// Cooperative cancellation
suspend fun cancellableOperation() {
    for (i in 0..1000) {
        // Check for cancellation
        ensureActive() // Throws CancellationException

        // Or insert a suspension point with yield()
        yield()

        // Processing
        processItem(i)
    }
}

// Job cancellation
fun main() = runBlocking {
    val job = launch {
        repeat(1000) { i ->
            println("Processing $i...")
            delay(100)
        }
    }

    delay(500)
    job.cancelAndJoin() // Cancel and wait for completion
    println("Cancelled")
}
```

---

## 8. Swift

### 8.1 Structured Concurrency

```swift
import Foundation

// async function
func getUserProfile(userId: String) async throws -> UserProfile {
    let user = try await userRepo.findById(userId)

    // Concurrent execution with async let (structured concurrency)
    async let orders = orderRepo.findByUserId(userId)
    async let reviews = reviewRepo.findByUserId(userId)

    return UserProfile(
        user: user,
        orders: try await orders,
        reviews: try await reviews
    )
}

// TaskGroup
func fetchAllUsers(ids: [String]) async throws -> [User] {
    try await withThrowingTaskGroup(of: User.self) { group in
        for id in ids {
            group.addTask {
                try await fetchUser(id)
            }
        }

        var users: [User] = []
        for try await user in group {
            users.append(user)
        }
        return users
    }
}

// Task cancellation
func cancellableOperation() async throws {
    for i in 0..<1000 {
        // Check for cancellation
        try Task.checkCancellation()

        await processItem(i)
    }
}

// Actor (preventing data races)
actor UserCache {
    private var cache: [String: User] = [:]

    func get(_ id: String) -> User? {
        return cache[id]
    }

    func set(_ id: String, user: User) {
        cache[id] = user
    }

    func getOrFetch(_ id: String) async throws -> User {
        if let cached = cache[id] {
            return cached
        }
        let user = try await fetchUser(id)
        cache[id] = user
        return user
    }
}
```

### 8.2 AsyncSequence

```swift
// AsyncSequence
func fetchPages(url: URL) -> AsyncStream<[Item]> {
    AsyncStream { continuation in
        Task {
            var nextURL: URL? = url
            while let currentURL = nextURL {
                let (data, _) = try await URLSession.shared.data(from: currentURL)
                let page = try JSONDecoder().decode(Page.self, from: data)
                continuation.yield(page.items)
                nextURL = page.nextURL
            }
            continuation.finish()
        }
    }
}

// Consumption
func processAllPages() async {
    for await items in fetchPages(url: apiURL) {
        for item in items {
            print(item)
        }
    }
}

// URLSession bytes (streaming)
func downloadWithProgress(url: URL) async throws {
    let (bytes, response) = try await URLSession.shared.bytes(from: url)
    let totalSize = response.expectedContentLength
    var receivedSize: Int64 = 0

    for try await byte in bytes {
        receivedSize += 1
        if receivedSize % 1024 == 0 {
            let progress = Double(receivedSize) / Double(totalSize)
            print("Progress: \(Int(progress * 100))%")
        }
    }
}
```

---

## 9. Efficient Patterns

### 9.1 Execution Based on Dependency Graphs

```typescript
// Pattern 1: Early await (when there are dependencies)
async function orderPipeline(userId: string) {
  const user = await getUser(userId);         // Need user first
  const cart = await getCart(user.cartId);      // Depends on user
  const total = calculateTotal(cart.items);     // Synchronous
  const payment = await processPayment(total);  // Depends on total
  return payment;
}

// Pattern 2: Concurrent execution of independent tasks
async function dashboardData(userId: string) {
  // Fetch independent data concurrently
  const [profile, notifications, stats, feed] = await Promise.all([
    getProfile(userId),
    getNotifications(userId),
    getStats(userId),
    getFeed(userId),
  ]);
  return { profile, notifications, stats, feed };
}

// Pattern 3: Staged concurrent execution
async function complexPipeline(userId: string) {
  // Stage 1: Fetch user
  const user = await getUser(userId);

  // Stage 2: Three tasks that depend on user, in parallel
  const [orders, reviews, wishlist] = await Promise.all([
    getOrders(user.id),
    getReviews(user.id),
    getWishlist(user.id),
  ]);

  // Stage 3: Process tasks that depend on orders, in parallel
  const orderDetails = await Promise.all(
    orders.map(order => getOrderDetails(order.id))
  );

  return { user, orders: orderDetails, reviews, wishlist };
}

// Pattern 4: Automatic dependency graph resolution
type TaskDef<T> = {
  deps: string[];
  run: (results: Record<string, any>) => Promise<T>;
};

async function runTaskGraph(
  tasks: Record<string, TaskDef<any>>
): Promise<Record<string, any>> {
  const results: Record<string, any> = {};
  const completed = new Set<string>();
  const running = new Map<string, Promise<void>>();

  async function runTask(name: string): Promise<void> {
    if (completed.has(name)) return;
    if (running.has(name)) return running.get(name)!;

    const task = tasks[name];
    const promise = (async () => {
      // Execute dependent tasks first
      await Promise.all(task.deps.map(dep => runTask(dep)));
      results[name] = await task.run(results);
      completed.add(name);
    })();

    running.set(name, promise);
    await promise;
  }

  await Promise.all(Object.keys(tasks).map(name => runTask(name)));
  return results;
}

// Usage example
const result = await runTaskGraph({
  user: {
    deps: [],
    run: () => getUser("123"),
  },
  orders: {
    deps: ["user"],
    run: (r) => getOrders(r.user.id),
  },
  reviews: {
    deps: ["user"],
    run: (r) => getReviews(r.user.id),
  },
  recommendations: {
    deps: ["orders", "reviews"],
    run: (r) => getRecommendations(r.orders, r.reviews),
  },
});
```

### 9.2 Cache Patterns

```typescript
// Async cache (prevents duplicate requests)
class AsyncCache<K, V> {
  private cache = new Map<string, { value: V; expiresAt: number }>();
  private pending = new Map<string, Promise<V>>();

  constructor(
    private readonly fetcher: (key: K) => Promise<V>,
    private readonly keyFn: (key: K) => string = String,
    private readonly ttl: number = 60000
  ) {}

  async get(key: K): Promise<V> {
    const cacheKey = this.keyFn(key);

    // Cache hit
    const cached = this.cache.get(cacheKey);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }

    // If a request for the same key is in progress, wait for it (dedup)
    const pending = this.pending.get(cacheKey);
    if (pending) {
      return pending;
    }

    // New fetch
    const promise = this.fetcher(key)
      .then(value => {
        this.cache.set(cacheKey, {
          value,
          expiresAt: Date.now() + this.ttl,
        });
        return value;
      })
      .finally(() => {
        this.pending.delete(cacheKey);
      });

    this.pending.set(cacheKey, promise);
    return promise;
  }

  invalidate(key: K): void {
    this.cache.delete(this.keyFn(key));
  }

  clear(): void {
    this.cache.clear();
    this.pending.clear();
  }
}

// Usage example
const userCache = new AsyncCache<string, User>(
  (userId) => fetchUser(userId),
  (key) => key,
  5 * 60 * 1000 // 5 minutes
);

// Even if the same user is requested simultaneously, only one API call is made
const [user1, user2] = await Promise.all([
  userCache.get("user-123"),
  userCache.get("user-123"),
]);
```

### 9.3 Batch Processing and Debouncing

```typescript
// DataLoader pattern (solving the N+1 problem)
class DataLoader<K, V> {
  private batch: Map<K, {
    resolve: (value: V) => void;
    reject: (error: Error) => void;
  }[]> = new Map();
  private scheduled = false;

  constructor(
    private readonly batchFn: (keys: K[]) => Promise<Map<K, V>>
  ) {}

  async load(key: K): Promise<V> {
    return new Promise<V>((resolve, reject) => {
      if (!this.batch.has(key)) {
        this.batch.set(key, []);
      }
      this.batch.get(key)!.push({ resolve, reject });

      if (!this.scheduled) {
        this.scheduled = true;
        // Schedule batch execution as a microtask
        queueMicrotask(() => this.executeBatch());
      }
    });
  }

  private async executeBatch(): Promise<void> {
    const batch = this.batch;
    this.batch = new Map();
    this.scheduled = false;

    const keys = Array.from(batch.keys());
    try {
      const results = await this.batchFn(keys);
      for (const [key, callbacks] of batch) {
        const value = results.get(key);
        if (value !== undefined) {
          callbacks.forEach(cb => cb.resolve(value));
        } else {
          callbacks.forEach(cb => cb.reject(new Error(`Not found: ${key}`)));
        }
      }
    } catch (error) {
      for (const callbacks of batch.values()) {
        callbacks.forEach(cb => cb.reject(error as Error));
      }
    }
  }
}

// Usage example
const userLoader = new DataLoader<string, User>(
  async (ids) => {
    // SELECT * FROM users WHERE id IN (...)
    const users = await db.query(
      `SELECT * FROM users WHERE id = ANY($1)`, [ids]
    );
    return new Map(users.map(u => [u.id, u]));
  }
);

// Even when called individually, they are batched
async function resolveComment(comment: Comment) {
  const author = await userLoader.load(comment.authorId); // |
  const editor = await userLoader.load(comment.editorId); // | Becomes 1 SQL query
  return { ...comment, author, editor };                   // |
}

// Async debounce
function asyncDebounce<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  ms: number
): T {
  let timeoutId: NodeJS.Timeout;
  let pendingResolve: ((value: any) => void) | null = null;
  let pendingReject: ((error: any) => void) | null = null;

  return ((...args: any[]) => {
    return new Promise((resolve, reject) => {
      // Abort previous request
      if (pendingReject) {
        pendingReject(new Error("Debounced"));
      }

      pendingResolve = resolve;
      pendingReject = reject;

      clearTimeout(timeoutId);
      timeoutId = setTimeout(async () => {
        try {
          const result = await fn(...args);
          pendingResolve?.(result);
        } catch (error) {
          pendingReject?.(error);
        } finally {
          pendingResolve = null;
          pendingReject = null;
        }
      }, ms);
    });
  }) as T;
}
```

### 9.4 for-await-of Patterns

```typescript
// Async generator
async function* fetchPages(url: string): AsyncGenerator<Item[]> {
  let nextUrl: string | null = url;
  while (nextUrl) {
    const response = await fetch(nextUrl);
    const data = await response.json();
    yield data.items;
    nextUrl = data.nextPage;
  }
}

for await (const page of fetchPages("/api/users")) {
  console.log(`Got ${page.length} users`);
}

// Pipeline: Async iterator with transformations
async function* map<T, U>(
  source: AsyncIterable<T>,
  fn: (item: T) => U | Promise<U>
): AsyncGenerator<U> {
  for await (const item of source) {
    yield await fn(item);
  }
}

async function* filter<T>(
  source: AsyncIterable<T>,
  predicate: (item: T) => boolean | Promise<boolean>
): AsyncGenerator<T> {
  for await (const item of source) {
    if (await predicate(item)) {
      yield item;
    }
  }
}

async function* take<T>(
  source: AsyncIterable<T>,
  count: number
): AsyncGenerator<T> {
  let taken = 0;
  for await (const item of source) {
    yield item;
    if (++taken >= count) break;
  }
}

// Pipeline usage example
const activeUsers = take(
  filter(
    map(
      fetchPages("/api/users"),
      page => page  // Get user array from page
    ),
    users => users.length > 0
  ),
  10 // Up to the first 10 pages
);

for await (const users of activeUsers) {
  await processUsers(users);
}
```

---

## 10. Common Mistakes and Anti-Patterns

### 10.1 Unnecessary Sequential Execution

```typescript
// Bad: Sequential await (when they could run concurrently)
const users = await getUsers();
const orders = await getOrders();
const products = await getProducts();
// -> Independent tasks running serially

// Good: Concurrent execution
const [users, orders, products] = await Promise.all([
  getUsers(), getOrders(), getProducts(),
]);
```

### 10.2 await Inside Loops

```typescript
// Bad: await inside a loop
for (const id of ids) {
  const data = await fetch(`/api/${id}`); // One at a time...
}

// Good: Concurrent execution
const results = await Promise.all(
  ids.map(id => fetch(`/api/${id}`))
);

// Good: Concurrency-limited (for large numbers of requests)
const semaphore = new Semaphore(10);
const results = await Promise.all(
  ids.map(id => semaphore.run(() => fetch(`/api/${id}`)))
);
```

### 10.3 Mixing async/await and .then()

```typescript
// Bad: Mixing .then() inside an async function
async function mixed() {
  return fetchData().then(data => data.value); // Mixed
}
// Good: Consistent style
async function clean() {
  const data = await fetchData();
  return data.value;
}
```

### 10.4 Unnecessary async

```typescript
// Bad: Unnecessary async (just return the Promise directly)
async function wrapper() {
  return await fetchData(); // Unnecessary async/await
}

// Good: Return directly (but be aware of error stack traces)
function wrapper() {
  return fetchData();
}

// Note: async/await is needed when try/catch is present
async function withErrorHandling() {
  try {
    return await fetchData(); // await is needed here
  } catch (error) {
    return fallbackData;
  }
}
```

### 10.5 Missing Error Handling

```typescript
// Bad: Unhandled rejection
async function firAndForget() {
  fetchData(); // Neither await nor catch!
}

// Good: Error handling even for fire-and-forget
async function safeFireAndForget() {
  fetchData().catch(error => {
    logger.error("Background task failed", error);
  });
}

// Bad: Promise.all fails entirely on partial failure
const results = await Promise.all([
  fetchA(), // Succeeds
  fetchB(), // Fails -> entire thing rejects
  fetchC(), // Succeeds but result is lost
]);

// Good: allSettled allows partial success
const results = await Promise.allSettled([
  fetchA(), fetchB(), fetchC(),
]);

const successful = results
  .filter((r): r is PromiseFulfilledResult<any> => r.status === "fulfilled")
  .map(r => r.value);

const failed = results
  .filter((r): r is PromiseRejectedResult => r.status === "rejected")
  .map(r => r.reason);
```

### 10.6 Memory Leaks

```typescript
// Bad: Forgetting cleanup
class DataFetcher {
  private intervalId?: NodeJS.Timeout;

  start() {
    this.intervalId = setInterval(async () => {
      const data = await fetchData();
      this.processData(data);
    }, 1000);
  }
  // Leaks if stop() is never called
}

// Good: Proper cleanup
class DataFetcher {
  private controller = new AbortController();
  private intervalId?: NodeJS.Timeout;

  start() {
    this.intervalId = setInterval(async () => {
      try {
        const data = await fetchData({ signal: this.controller.signal });
        this.processData(data);
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // Normal cancellation
        }
        throw error;
      }
    }, 1000);
  }

  stop() {
    this.controller.abort();
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }
}
```

---

## 11. Testing

### 11.1 Testing in JavaScript/TypeScript

```typescript
import { describe, it, expect, vi } from "vitest";

// Basic async test
describe("UserService", () => {
  it("should fetch user profile", async () => {
    const service = new UserService(mockRepo);
    const profile = await service.getUserProfile("user-123");

    expect(profile.user.id).toBe("user-123");
    expect(profile.orders).toHaveLength(3);
  });

  // Error case
  it("should throw on missing user", async () => {
    const service = new UserService(emptyRepo);

    await expect(
      service.getUserProfile("nonexistent")
    ).rejects.toThrow("User not found");
  });

  // Timeout test
  it("should timeout after 5 seconds", async () => {
    vi.useFakeTimers();

    const promise = fetchWithTimeout("/api/slow", 5000);

    // Advance time
    vi.advanceTimersByTime(5000);

    await expect(promise).rejects.toThrow("Timeout");

    vi.useRealTimers();
  });

  // Concurrent execution test
  it("should fetch in parallel", async () => {
    const startTime = Date.now();
    const callOrder: string[] = [];

    const mockFetchA = async () => {
      callOrder.push("A-start");
      await new Promise(r => setTimeout(r, 100));
      callOrder.push("A-end");
      return "A";
    };

    const mockFetchB = async () => {
      callOrder.push("B-start");
      await new Promise(r => setTimeout(r, 100));
      callOrder.push("B-end");
      return "B";
    };

    const [a, b] = await Promise.all([mockFetchA(), mockFetchB()]);

    expect(a).toBe("A");
    expect(b).toBe("B");
    // Verify concurrent execution: both start first
    expect(callOrder[0]).toBe("A-start");
    expect(callOrder[1]).toBe("B-start");
  });

  // Retry test
  it("should retry on failure", async () => {
    let attempts = 0;
    const unreliable = async () => {
      attempts++;
      if (attempts < 3) throw new Error("Temporary failure");
      return "success";
    };

    const result = await withRetry(unreliable, { maxRetries: 3, baseDelay: 10 });
    expect(result).toBe("success");
    expect(attempts).toBe(3);
  });

  // AbortController test
  it("should cancel on abort", async () => {
    const controller = new AbortController();

    const promise = fetchWithCancel("/api/data", controller.signal);
    controller.abort();

    await expect(promise).rejects.toThrow("AbortError");
  });
});
```

### 11.2 Testing in Python

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

# Using pytest-asyncio
@pytest.mark.asyncio
async def test_get_user_profile():
    mock_repo = AsyncMock()
    mock_repo.find_by_id.return_value = {"id": "123", "name": "Alice"}

    service = UserService(mock_repo)
    profile = await service.get_user_profile("123")

    assert profile["user"]["name"] == "Alice"
    mock_repo.find_by_id.assert_awaited_once_with("123")

@pytest.mark.asyncio
async def test_timeout():
    async def slow_operation():
        await asyncio.sleep(10)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=0.1)

@pytest.mark.asyncio
async def test_concurrent_execution():
    results = []

    async def task(name: str, delay: float):
        results.append(f"{name}-start")
        await asyncio.sleep(delay)
        results.append(f"{name}-end")
        return name

    a, b = await asyncio.gather(
        task("A", 0.1),
        task("B", 0.1),
    )

    assert a == "A"
    assert b == "B"
    assert results[0] == "A-start"
    assert results[1] == "B-start"

@pytest.mark.asyncio
async def test_task_cancellation():
    cancelled = False

    async def cancellable():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    task = asyncio.create_task(cancellable())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled is True
```

### 11.3 Testing in Rust

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_get_user_profile() {
        let repo = MockUserRepo::new();
        repo.expect_find_by_id()
            .returning(|_| Ok(User { id: "123".into(), name: "Alice".into() }));

        let profile = get_user_profile(&repo, "123").await.unwrap();
        assert_eq!(profile.user.name, "Alice");
    }

    #[tokio::test]
    async fn test_timeout() {
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            async {
                tokio::time::sleep(Duration::from_secs(10)).await;
                "done"
            }
        ).await;

        assert!(result.is_err()); // Elapsed error
    }

    #[tokio::test]
    async fn test_concurrent_tasks() {
        let (a, b) = tokio::join!(
            async { 1 + 1 },
            async { 2 + 2 },
        );

        assert_eq!(a, 2);
        assert_eq!(b, 4);
    }

    #[tokio::test]
    async fn test_cancellation() {
        let handle = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(100)).await;
            42
        });

        handle.abort();
        let result = handle.await;
        assert!(result.unwrap_err().is_cancelled());
    }
}
```

---

## 12. Debugging Techniques

### 12.1 AsyncLocalStorage (Node.js)

```typescript
import { AsyncLocalStorage } from "node:async_hooks";

// Request tracking
const requestContext = new AsyncLocalStorage<{
  requestId: string;
  startTime: number;
}>();

// Set context in middleware
app.use((req, res, next) => {
  const context = {
    requestId: crypto.randomUUID(),
    startTime: Date.now(),
  };
  requestContext.run(context, next);
});

// Access context from anywhere
async function processOrder(orderId: string) {
  const ctx = requestContext.getStore()!;
  logger.info(`[${ctx.requestId}] Processing order ${orderId}`);

  const result = await orderService.process(orderId);

  logger.info(
    `[${ctx.requestId}] Order processed in ${Date.now() - ctx.startTime}ms`
  );
  return result;
}
```

### 12.2 Profiling Async Operations

```typescript
// Measuring execution time
async function withTiming<T>(
  label: string,
  fn: () => Promise<T>
): Promise<T> {
  const start = performance.now();
  try {
    const result = await fn();
    const duration = performance.now() - start;
    console.log(`[${label}] completed in ${duration.toFixed(2)}ms`);
    return result;
  } catch (error) {
    const duration = performance.now() - start;
    console.error(`[${label}] failed after ${duration.toFixed(2)}ms`);
    throw error;
  }
}

// Usage
const user = await withTiming("getUser", () => getUser("123"));

// Visualizing parallel execution
async function traceParallel(
  tasks: Record<string, () => Promise<any>>
): Promise<Record<string, any>> {
  const startTime = performance.now();
  const timeline: { name: string; start: number; end: number }[] = [];

  const entries = Object.entries(tasks);
  const results = await Promise.all(
    entries.map(async ([name, fn]) => {
      const taskStart = performance.now() - startTime;
      const result = await fn();
      const taskEnd = performance.now() - startTime;
      timeline.push({ name, start: taskStart, end: taskEnd });
      return [name, result] as const;
    })
  );

  // Timeline output
  console.log("=== Execution Timeline ===");
  for (const entry of timeline.sort((a, b) => a.start - b.start)) {
    const bar = " ".repeat(Math.floor(entry.start / 10))
      + "=".repeat(Math.floor((entry.end - entry.start) / 10));
    console.log(`${entry.name.padEnd(20)} |${bar}| ${(entry.end - entry.start).toFixed(1)}ms`);
  }

  return Object.fromEntries(results);
}

// Usage
await traceParallel({
  users: () => fetchUsers(),
  orders: () => fetchOrders(),
  products: () => fetchProducts(),
});
// Output:
// === Execution Timeline ===
// users                |====      | 45.2ms
// orders               |========  | 82.1ms
// products             |=====     | 53.7ms
```

### 12.3 Detecting Unhandled Rejections

```typescript
// Global handler in Node.js
process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
  // Log the application state
  // Consider terminating the process in production
  process.exit(1);
});

// Browser
window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled rejection:", event.reason);
  event.preventDefault(); // Suppress default console output
  // Send to error tracking service
  errorTracker.captureException(event.reason);
});

// Detect unhandled Promises with ESLint rules
// .eslintrc.json
// {
//   "rules": {
//     "no-floating-promises": "error",  // @typescript-eslint
//     "require-await": "warn"
//   }
// }
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying how it works.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world work?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

### Cross-Language Comparison

| Language | async Syntax | Concurrent Execution | Runtime | Cancellation |
|----------|-------------|---------------------|---------|-------------|
| JS/TS | async/await | Promise.all | Event loop | AbortController |
| Python | async/await | asyncio.gather | asyncio | Task.cancel() |
| Rust | async/await | tokio::join! | tokio/async-std | tokio::select! |
| Go | goroutine | go + channel | Built-in runtime | context.Context |
| C# | async/await | Task.WhenAll | CLR | CancellationToken |
| Kotlin | suspend | coroutineScope | Dispatchers | Job.cancel() |
| Swift | async/await | async let | Swift Runtime | Task.cancel() |

### Design Guidelines

| Pattern | When to Use | Caveats |
|---------|-------------|---------|
| Sequential await | Operations with dependencies | Avoid unnecessary sequential execution |
| Promise.all | Multiple independent operations | One failure causes total failure |
| Promise.allSettled | When partial success is acceptable | Results need classification |
| Semaphore | Rate-limiting large concurrency | Prevents resource exhaustion |
| for-await-of | Stream processing | Watch out for backpressure |
| DataLoader | Solving the N+1 problem | Design of the batch window |
| Circuit breaker | Calls to unstable services | State management complexity |

---

## Recommended Next Guides

---

## References
1. MDN Web Docs. "async function."
2. Python Documentation. "Coroutines and Tasks."
3. Tokio Documentation. "Tutorial."
4. Kotlin Documentation. "Coroutines Guide."
5. Swift Documentation. "Concurrency."
6. C# Documentation. "Asynchronous programming with async and await."
7. Go Blog. "Go Concurrency Patterns."
8. Node.js Documentation. "Async Hooks."
