# async/await (Asynchronous Programming)

> async/await is a mechanism that "allows other tasks to proceed while waiting for I/O." It is a foundational technology for modern server-side and UI programming that efficiently handles a large number of concurrent connections without using threads.

## Learning Objectives

- [ ] Understand the necessity of asynchronous processing and the fundamental differences from synchronous processing
- [ ] Accurately grasp the operating principles of event loops
- [ ] Understand the syntax and internal mechanisms of async/await
- [ ] Organize the concepts and relationships of Promise / Future / Task
- [ ] Compare the differences in asynchronous models across languages (JavaScript, Python, Rust, Go, Java, C#)
- [ ] Master design patterns for error handling, cancellation, and timeouts
- [ ] Recognize anti-patterns and avoid pitfalls in production code


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Threads and Processes](./00-threads-and-processes.md)

---

## 1. Why Asynchronous Processing Is Necessary

### 1.1 Limitations of Synchronous Processing

Modern applications frequently perform I/O operations such as network communication, database access, and file operations. These I/O operations take orders of magnitude more time compared to CPU computations.

```
Latency Comparison by Operation (Approximate Values):

+---------------------------------+--------------------+--------------+
| Operation                       | Latency            | CPU Equiv.   |
+---------------------------------+--------------------+--------------+
| L1 Cache Reference              | 1 ns               | 1 second     |
| L2 Cache Reference              | 4 ns               | 4 seconds    |
| Main Memory Reference           | 100 ns             | 1.5 minutes  |
| SSD Random Read                 | 16,000 ns          | 4.4 hours    |
| HDD Seek                        | 2,000,000 ns       | 23 days      |
| Same Datacenter RTT             | 500,000 ns         | 5.7 days     |
| Intercontinental Network RTT    | 150,000,000 ns     | 4.7 years    |
+---------------------------------+--------------------+--------------+

> Network I/O is approximately 100 million times slower than CPU computation
> Leaving the CPU idle during I/O wait is extremely inefficient
```

In synchronous processing (blocking I/O), the entire thread is halted until I/O completes. When a web server adopts a 1-request = 1-thread model, the number of concurrent connections is limited by the number of threads.

```
Synchronous Web Server Behavior:

Thread-1: --[Request Received]--################--[DB Response]--[Send Response]--
Thread-2: --[Request Received]--##########--[API Response]--[Send Response]------
Thread-3: --[Request Received]--####################--[File Response]--[Send]---
Thread-4: --Waiting (thread pool exhausted)------------------------------------------
           ## = Blocked on I/O (CPU is doing nothing)

Issues:
  - Each thread consumes approximately 1 MB of stack memory
  - 10,000 concurrent connections -> 10 GB of memory required
  - Context switch overhead
  - C10K problem (the 10,000 concurrent connections barrier)
```

### 1.2 The Asynchronous Solution

Asynchronous processing (non-blocking I/O) only initiates an I/O operation and proceeds with other tasks while waiting for the completion notification.

```
Asynchronous Web Server Behavior (Event Loop Model):

Event Loop (1 thread):
  --[Req-A Recv]--[DB Issue]--[Req-B Recv]--[API Issue]--[Req-C Recv]--
  --[File Issue]--[DB Response->Res-A Send]--[API Response->Res-B Send]--
  --[File Response->Res-C Send]--[Req-D Recv]--...

  -> Can handle thousands to tens of thousands of concurrent connections with 1 thread
  -> Effectively utilizes I/O wait time for processing other requests
  -> Dramatically reduces memory usage

Benefits:
  +----------------------------+------------+---------------+
  | Metric                     | Sync Model | Async Model   |
  +----------------------------+------------+---------------+
  | Memory at 10,000 conns     | ~10 GB     | ~100 MB       |
  | Context Switches           | Frequent   | Minimal       |
  | Throughput                 | Medium     | High          |
  | CPU Utilization Efficiency | Low        | High          |
  | Programming Complexity     | Low        | Medium-High   |
  +----------------------------+------------+---------------+
```

### 1.3 Historical Evolution of Asynchronous Processing

Asynchronous programming has evolved incrementally from callback hell to async/await.

```
Evolution of Asynchronous Programming:

Stage 1: Callbacks
  +-- The most primitive approach
  +-- Pass completion handler as function argument
  +-- Problem: Callback hell (Pyramid of Doom)

Stage 2: Promise / Future
  +-- Objects representing the result of async operations
  +-- Describe sequential processing via method chaining
  +-- Problem: Nesting can still become deep

Stage 3: async/await
  +-- Syntactic sugar over Promise/Future
  +-- Write async code that looks like sync code
  +-- The current mainstream paradigm

Stage 4: Structured Concurrency
  +-- Structurally manage the lifecycle of async tasks
  +-- Python: TaskGroup, Kotlin: coroutineScope
  +-- Safely handle cancellation and error propagation
```

---

## 2. How Event Loops Work

### 2.1 What Is an Event Loop

To understand async/await, you need to accurately grasp the mechanism of the event loop that underlies it. An event loop is an infinite loop that monitors I/O events and executes corresponding callbacks.

```
Basic Structure of an Event Loop:

+------------------------------------------------------------+
|                     Event Loop                              |
|                                                             |
|  while (true) {                                             |
|    1. Execute callbacks from the timer queue                |
|    2. I/O polling (epoll/kqueue/IOCP)                       |
|    3. Execute callbacks for completed I/O                   |
|    4. Process the microtask queue                           |
|    5. Sleep if there are no tasks to execute                |
|  }                                                          |
|                                                             |
|  +-----------+    +-----------+    +----------------+       |
|  | Timer     |    |  I/O      |    | Microtask      |       |
|  | Queue     |    |  Queue    |    | Queue          |       |
|  |           |    |           |    |                |       |
|  | setTimeout|    | fs.read   |    | Promise.then   |       |
|  | setInterval|   | net.req   |    | queueMicro..   |       |
|  +-----------+    +-----------+    +----------------+       |
+------------------------------------------------------------+
```

### 2.2 Node.js Event Loop (libuv)

The Node.js event loop is based on the libuv library and executes the following phases in order.

```
Detailed Phases of the Node.js Event Loop:

   +-----------------------------+
+->|        timers               | <- setTimeout, setInterval callbacks
|  +--------------+--------------+
|  +--------------+--------------+
|  |     pending callbacks       | <- System callbacks like TCP errors
|  +--------------+--------------+
|  +--------------+--------------+
|  |       idle, prepare         | <- Internal use only
|  +--------------+--------------+      +-----------------+
|  +--------------+--------------+      |   incoming:     |
|  |          poll               |<-----|  connections,   |
|  +--------------+--------------+      |  data, etc.     |
|  +--------------+--------------+      +-----------------+
|  |          check              | <- setImmediate callbacks
|  +--------------+--------------+
|  +--------------+--------------+
|  |      close callbacks        | <- socket.on('close', ...) etc.
|  +--------------+--------------+
+------------------+

* The microtask queue (Promises) is processed between each phase
```

### 2.3 Execution Order in JavaScript

Let's trace the execution order to deepen our understanding of the event loop.

```javascript
// Execution order quiz: What is the output order of the following code?
console.log('1: Script start');

setTimeout(() => {
    console.log('2: setTimeout');
}, 0);

Promise.resolve()
    .then(() => {
        console.log('3: Promise.then');
    })
    .then(() => {
        console.log('4: Promise.then (chained)');
    });

queueMicrotask(() => {
    console.log('5: queueMicrotask');
});

console.log('6: Script end');

// Output order:
// 1: Script start
// 6: Script end
// 3: Promise.then         <- Microtask (immediately after sync code completes)
// 5: queueMicrotask       <- Microtask
// 4: Promise.then (chained) <- Microtask
// 2: setTimeout           <- Macrotask (next event loop iteration)
```

```
Execution Order Diagram:

Call Stack              Microtask Queue          Macrotask Queue
+---------------+       +---------------+       +---------------+
| console.log   |       |               |       |               |
| ('1: ...')    |       |               |       |               |
+---------------+       |               |       |               |
| setTimeout    | ------+---------------+------>| callback-2    |
+---------------+       |               |       |               |
| Promise       | ----->| callback-3    |       |               |
| .resolve()    |       |               |       |               |
+---------------+       |               |       |               |
| queueMicro    | ----->| callback-5    |       |               |
| task          |       |               |       |               |
+---------------+       |               |       |               |
| console.log   |       |               |       |               |
| ('6: ...')    |       |               |       |               |
+---------------+       +---------------+       +---------------+
   v Stack empty              v Execute all         v Execute next
   Step 1: Output 1,6   Step 2: Output 3,5,4   Step 3: Output 2
```

---

## 3. Fundamental Concepts of Promise / Future / Task

### 3.1 Promise State Transitions

Promise (JavaScript) / Future (Rust, Dart) / Task (C#) are objects representing the result of asynchronous operations. They share a common state transition model.

```
Promise State Transition Diagram:

                     +----------+
                     | Pending  | <- Initial state (in progress)
                     +----+-----+
                          |
              +-----------+-----------+
              |                       |
         +----v-----+          +-----v------+
         | Fulfilled|          |  Rejected  |
         | (Success)|          |  (Failure) |
         +----------+          +------------+
              |                       |
              v                       v
         .then(value)            .catch(error)
         receives the value      handles the error

Note: The transition Pending -> Fulfilled / Rejected is one-way
      Once settled, the state cannot be changed (Immutable Settlement)
```

### 3.2 Correspondence Table of Async Primitives Across Languages

```
+------------------+----------------+---------------+-----------------+
| Concept          | JavaScript     | Python        | Rust            |
+------------------+----------------+---------------+-----------------+
| Async Result     | Promise        | Coroutine     | Future          |
| Get Success Value| .then()        | await         | .await          |
| Error Handling   | .catch()       | try/except    | ? / match       |
| Concurrent Exec  | Promise.all    | gather        | join!           |
| Racing           | Promise.race   | wait_for      | select!         |
| Timeout          | AbortSignal    | timeout()     | timeout()       |
| Cancellation     | AbortController| Task.cancel   | drop            |
+------------------+----------------+---------------+-----------------+
| Concept          | C#             | Go            | Java            |
+------------------+----------------+---------------+-----------------+
| Async Result     | Task<T>        | chan T        | CompletableFuture |
| Get Success Value| await          | <-ch          | .get() / .join() |
| Error Handling   | try/catch      | error return  | .exceptionally() |
| Concurrent Exec  | Task.WhenAll   | WaitGroup     | allOf           |
| Racing           | Task.WhenAny   | select        | anyOf           |
| Timeout          | CancellationToken| context     | .orTimeout()    |
| Cancellation     | CancellationToken| context     | .cancel()       |
+------------------+----------------+---------------+-----------------+
```

### 3.3 Evolution from Callbacks to Promise to async/await

```javascript
// ===== Stage 1: Callback Hell =====
function loadDashboard(userId, callback) {
    getUser(userId, function(err, user) {
        if (err) return callback(err);
        getPosts(user.id, function(err, posts) {
            if (err) return callback(err);
            getComments(posts[0].id, function(err, comments) {
                if (err) return callback(err);
                getNotifications(user.id, function(err, notifications) {
                    if (err) return callback(err);
                    callback(null, {
                        user: user,
                        posts: posts,
                        comments: comments,
                        notifications: notifications
                    });
                });
            });
        });
    });
}
// Issues: Deep nesting, scattered error handling, poor readability

// ===== Stage 2: Promise Chain =====
function loadDashboard(userId) {
    let userData;
    return getUser(userId)
        .then(user => {
            userData = user;
            return getPosts(user.id);
        })
        .then(posts => {
            return getComments(posts[0].id)
                .then(comments => ({ posts, comments }));
        })
        .then(({ posts, comments }) => {
            return getNotifications(userData.id)
                .then(notifications => ({
                    user: userData,
                    posts,
                    comments,
                    notifications
                }));
        })
        .catch(err => {
            console.error('Dashboard load failed:', err);
            throw err;
        });
}
// Improvement: Flat chain, unified error handling
// Issue: Still complex, managing intermediate variables is cumbersome

// ===== Stage 3: async/await =====
async function loadDashboard(userId) {
    try {
        const user = await getUser(userId);
        const posts = await getPosts(user.id);
        const comments = await getComments(posts[0].id);
        const notifications = await getNotifications(user.id);

        return { user, posts, comments, notifications };
    } catch (err) {
        console.error('Dashboard load failed:', err);
        throw err;
    }
}
// Improvement: Looks like synchronous code, intuitive error handling
// Note: The above executes sequentially. Use Promise.all for concurrent operations

// ===== Stage 3.5: async/await + Concurrency Optimization =====
async function loadDashboard(userId) {
    try {
        const user = await getUser(userId);

        // posts and notifications are independent, so run concurrently
        const [posts, notifications] = await Promise.all([
            getPosts(user.id),
            getNotifications(user.id),
        ]);

        // comments depends on posts, so execute sequentially
        const comments = await getComments(posts[0].id);

        return { user, posts, comments, notifications };
    } catch (err) {
        console.error('Dashboard load failed:', err);
        throw err;
    }
}
```

---

## 4. JavaScript async/await in Detail

### 4.1 Basic Syntax and Internal Behavior

```javascript
// ===== Basic: async functions always return a Promise =====
async function greet(name) {
    return `Hello, ${name}!`;
}

// The above is equivalent to:
function greet(name) {
    return Promise.resolve(`Hello, ${name}!`);
}

// ===== await: Wait for a Promise to resolve =====
async function fetchUserProfile(userId) {
    // await suspends function execution until the Promise resolves
    // and returns control to the event loop
    const response = await fetch(`/api/users/${userId}`);

    // When the response arrives, execution resumes here
    if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
    }

    const data = await response.json();
    return data;
}

// ===== await can only be used inside async functions =====
// ES2022+: Top-level await is available (ESM modules only)
const config = await import('./config.js');
const data = await fetch('/api/data').then(r => r.json());
```

### 4.2 Promise Concurrency Patterns

```javascript
// ===== Promise.all: Completes when all succeed, rejects immediately on any failure =====
async function fetchAllResources() {
    try {
        const [users, products, orders] = await Promise.all([
            fetch('/api/users').then(r => r.json()),
            fetch('/api/products').then(r => r.json()),
            fetch('/api/orders').then(r => r.json()),
        ]);
        return { users, products, orders };
    } catch (error) {
        // If any one fails, other operations are also interrupted
        console.error('One of the requests failed:', error);
        throw error;
    }
}

// ===== Promise.allSettled: Get all results (regardless of success/failure) =====
async function fetchAllResourcesGracefully() {
    const results = await Promise.allSettled([
        fetch('/api/users').then(r => r.json()),
        fetch('/api/products').then(r => r.json()),
        fetch('/api/orders').then(r => r.json()),
    ]);

    const data = {};
    results.forEach((result, index) => {
        const keys = ['users', 'products', 'orders'];
        if (result.status === 'fulfilled') {
            data[keys[index]] = result.value;
        } else {
            console.warn(`${keys[index]} failed:`, result.reason);
            data[keys[index]] = []; // Fallback value
        }
    });
    return data;
}

// ===== Promise.race: Returns the first to complete (success or failure) =====
async function fetchWithTimeout(url, timeoutMs = 5000) {
    const controller = new AbortController();

    const result = await Promise.race([
        fetch(url, { signal: controller.signal }),
        new Promise((_, reject) => {
            setTimeout(() => {
                controller.abort();
                reject(new Error(`Timeout after ${timeoutMs}ms`));
            }, timeoutMs);
        }),
    ]);

    return result.json();
}

// ===== Promise.any: Returns the first success (AggregateError if all fail) =====
async function fetchFromMultipleCDNs(path) {
    try {
        const response = await Promise.any([
            fetch(`https://cdn1.example.com${path}`),
            fetch(`https://cdn2.example.com${path}`),
            fetch(`https://cdn3.example.com${path}`),
        ]);
        return response;
    } catch (error) {
        // AggregateError: All CDNs failed
        console.error('All CDNs failed:', error.errors);
        throw error;
    }
}
```

### 4.3 Error Handling Best Practices

```javascript
// ===== Pattern 1: Centralized try/catch =====
async function processOrder(orderId) {
    try {
        const order = await fetchOrder(orderId);
        const payment = await processPayment(order);
        const shipping = await arrangeShipping(order, payment);
        await sendConfirmationEmail(order, shipping);
        return { success: true, trackingNumber: shipping.trackingNumber };
    } catch (error) {
        // Handle based on error type
        if (error instanceof PaymentError) {
            await refundIfNeeded(orderId);
            return { success: false, reason: 'payment_failed' };
        }
        if (error instanceof ShippingError) {
            await notifyManualReview(orderId, error);
            return { success: false, reason: 'shipping_unavailable' };
        }
        // Re-throw unexpected errors
        throw error;
    }
}

// ===== Pattern 2: Result Type Pattern (Go-style error handling) =====
async function safeAsync(asyncFn) {
    try {
        const result = await asyncFn();
        return [result, null];
    } catch (error) {
        return [null, error];
    }
}

// Usage example
async function loadUserData(userId) {
    const [user, userErr] = await safeAsync(() => fetchUser(userId));
    if (userErr) {
        console.error('User fetch failed:', userErr);
        return null;
    }

    const [posts, postsErr] = await safeAsync(() => fetchPosts(user.id));
    if (postsErr) {
        console.warn('Posts fetch failed, continuing without posts');
    }

    return { user, posts: posts || [] };
}

// ===== Pattern 3: Cancellation with AbortController =====
class ApiClient {
    constructor() {
        this.controllers = new Map();
    }

    async fetch(key, url, options = {}) {
        // Cancel existing request
        if (this.controllers.has(key)) {
            this.controllers.get(key).abort();
        }

        const controller = new AbortController();
        this.controllers.set(key, controller);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
            });
            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log(`Request ${key} was cancelled`);
                return null;
            }
            throw error;
        } finally {
            this.controllers.delete(key);
        }
    }

    cancelAll() {
        for (const controller of this.controllers.values()) {
            controller.abort();
        }
        this.controllers.clear();
    }
}
```

### 4.4 Async Iterators

```javascript
// ===== for await...of: Sequential processing of async data streams =====

// Async generator function
async function* fetchPages(baseUrl) {
    let page = 1;
    let hasMore = true;

    while (hasMore) {
        const response = await fetch(`${baseUrl}?page=${page}`);
        const data = await response.json();

        yield data.items;

        hasMore = data.hasNextPage;
        page++;
    }
}

// Usage: Process all paginated data
async function processAllItems(baseUrl) {
    const allItems = [];

    for await (const items of fetchPages(baseUrl)) {
        allItems.push(...items);
        console.log(`Processed page, total items: ${allItems.length}`);
    }

    return allItems;
}

// Reading from a ReadableStream (Web Streams API)
async function readStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let result = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        result += decoder.decode(value, { stream: true });
    }

    return result;
}
```

---

## 5. Python asyncio in Detail

### 5.1 Coroutine Basics

```python
import asyncio
from typing import Any

# ===== Coroutine function: Defined with async def =====
async def fetch_data(url: str) -> dict:
    """Fetch data asynchronously"""
    # Calling a coroutine function returns a coroutine object
    # It is not executed until awaited
    await asyncio.sleep(1)  # Simulating an I/O operation
    return {"url": url, "data": "sample"}

# ===== How to run coroutines =====

# Method 1: asyncio.run() (main entry point)
async def main():
    result = await fetch_data("https://api.example.com/data")
    print(result)

asyncio.run(main())

# Method 2: Directly manipulate the event loop (low-level)
loop = asyncio.get_event_loop()
result = loop.run_until_complete(fetch_data("https://api.example.com"))
loop.close()
```

### 5.2 Concurrent Execution Patterns

```python
import asyncio
import aiohttp
from dataclasses import dataclass

@dataclass
class UserDashboard:
    user: dict
    posts: list
    notifications: list

async def fetch_json(session: aiohttp.ClientSession, url: str) -> Any:
    """Generic JSON fetch function"""
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()

# ===== asyncio.gather: Concurrent execution (most common) =====
async def load_dashboard(user_id: int) -> UserDashboard:
    async with aiohttp.ClientSession() as session:
        base = "https://api.example.com"
        user, posts, notifications = await asyncio.gather(
            fetch_json(session, f"{base}/users/{user_id}"),
            fetch_json(session, f"{base}/users/{user_id}/posts"),
            fetch_json(session, f"{base}/users/{user_id}/notifications"),
        )
        return UserDashboard(user=user, posts=posts, notifications=notifications)

# ===== asyncio.gather: return_exceptions=True to catch exceptions =====
async def load_dashboard_safe(user_id: int) -> dict:
    async with aiohttp.ClientSession() as session:
        base = "https://api.example.com"
        results = await asyncio.gather(
            fetch_json(session, f"{base}/users/{user_id}"),
            fetch_json(session, f"{base}/users/{user_id}/posts"),
            fetch_json(session, f"{base}/users/{user_id}/notifications"),
            return_exceptions=True,  # Return exceptions as values
        )
        return {
            "user": results[0] if not isinstance(results[0], Exception) else None,
            "posts": results[1] if not isinstance(results[1], Exception) else [],
            "notifications": results[2] if not isinstance(results[2], Exception) else [],
        }

# ===== TaskGroup: Structured Concurrency (Python 3.11+) =====
async def load_dashboard_structured(user_id: int) -> UserDashboard:
    """Structured concurrency pattern using TaskGroup"""
    results = {}

    async with aiohttp.ClientSession() as session:
        base = "https://api.example.com"

        async with asyncio.TaskGroup() as tg:
            async def fetch_and_store(key: str, url: str):
                results[key] = await fetch_json(session, url)

            tg.create_task(fetch_and_store("user", f"{base}/users/{user_id}"))
            tg.create_task(fetch_and_store("posts", f"{base}/users/{user_id}/posts"))
            tg.create_task(fetch_and_store("notifs", f"{base}/users/{user_id}/notifications"))

        # Reaching outside the TaskGroup = all tasks completed
        # If any task throws an exception, all other tasks are cancelled
        # and the exceptions are aggregated as an ExceptionGroup

    return UserDashboard(
        user=results["user"],
        posts=results["posts"],
        notifications=results["notifs"],
    )
```

### 5.3 Timeout and Cancellation

```python
import asyncio

# ===== asyncio.timeout: Timeout control (Python 3.11+) =====
async def fetch_with_timeout(url: str, timeout_sec: float = 5.0):
    try:
        async with asyncio.timeout(timeout_sec):
            return await fetch_data(url)
    except TimeoutError:
        print(f"Timeout after {timeout_sec}s: {url}")
        return None

# ===== asyncio.wait_for: Traditional timeout approach =====
async def fetch_with_wait_for(url: str, timeout_sec: float = 5.0):
    try:
        return await asyncio.wait_for(
            fetch_data(url),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        print(f"Timeout: {url}")
        return None

# ===== Task Cancellation =====
async def cancellable_operation():
    task = asyncio.create_task(long_running_operation())

    # Cancel after 5 seconds
    await asyncio.sleep(5)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled")

# ===== Limiting concurrency with a Semaphore =====
async def fetch_many_urls(urls: list[str], max_concurrent: int = 10):
    """Fetch a large number of URLs with limited concurrent connections"""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def limited_fetch(url: str):
        async with semaphore:  # Limit concurrency to max_concurrent
            return await fetch_data(url)

    results = await asyncio.gather(
        *[limited_fetch(url) for url in urls]
    )
    return results
```

### 5.4 Async Context Managers and Iterators

```python
import asyncio
from contextlib import asynccontextmanager

# ===== Async Context Manager =====
class AsyncDatabaseConnection:
    """Example of an async database connection"""

    async def __aenter__(self):
        self.conn = await create_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
        return False  # Propagate exceptions

    async def query(self, sql: str):
        return await self.conn.execute(sql)

# Usage
async def get_users():
    async with AsyncDatabaseConnection() as db:
        return await db.query("SELECT * FROM users")

# ===== Decorator-based approach =====
@asynccontextmanager
async def managed_resource(name: str):
    resource = await acquire_resource(name)
    try:
        yield resource
    finally:
        await release_resource(resource)

# ===== Async Iterator =====
class AsyncPaginator:
    """Async iterator with pagination"""

    def __init__(self, base_url: str, page_size: int = 20):
        self.base_url = base_url
        self.page_size = page_size
        self.current_page = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.current_page += 1
        data = await fetch_data(
            f"{self.base_url}?page={self.current_page}&size={self.page_size}"
        )
        if not data.get("items"):
            raise StopAsyncIteration
        return data["items"]

# Usage
async def process_all_users():
    async for page in AsyncPaginator("https://api.example.com/users"):
        for user in page:
            await process_user(user)

# ===== Async Generator =====
async def event_stream(url: str):
    """Async generator for receiving Server-Sent Events"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            async for line in response.content:
                decoded = line.decode('utf-8').strip()
                if decoded.startswith('data:'):
                    yield decoded[5:].strip()
```

---

## 6. Rust async/await in Detail

### 6.1 The Future Trait and Polling Model

Rust's async model is fundamentally different from other languages. Futures are zero-cost abstractions, and nothing is executed until `.await` is called (lazy evaluation).

```rust
// ===== Future Trait Definition (Standard Library) =====
pub trait Future {
    type Output;

    // Progress is checked each time poll is called
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

pub enum Poll<T> {
    Ready(T),    // Complete: value is available
    Pending,     // Incomplete: still waiting
}

// ===== Example of manually implementing a Future =====
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

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
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.when {
            Poll::Ready(())
        } else {
            // Register the waker to request re-polling later
            let waker = cx.waker().clone();
            let when = self.when;
            std::thread::spawn(move || {
                let now = Instant::now();
                if now < when {
                    std::thread::sleep(when - now);
                }
                waker.wake();  // Notify the runtime to re-poll
            });
            Poll::Pending
        }
    }
}

// Usage
async fn example() {
    println!("Start");
    Delay::new(Duration::from_secs(2)).await;
    println!("2 seconds elapsed");
}
```

```
Rust's Future Polling Model:

  Runtime (Executor)                Future
  +--------------------+               +---------------+
  |                    |  poll()       |               |
  |  1. Execute poll --+-------------->|  Process      |
  |                    |  Pending      |  Not yet done |
  |  2. Execute other <-+--------------+               |
  |     tasks          |              |  (Waker reg.) |
  |                    |              +---------------+
  |  ...time passes... |
  |                    |  wake()
  |  3. Waker notif. <-+------------ OS/Timer
  |                    |
  |  4. Re-poll     --+-------------->|  Resume       |
  |                    |  Ready(val)  |  Done!        |
  |  5. Get value   <--+--------------+               |
  +--------------------+               +---------------+

  * JavaScript/Python: Runtime directly drives coroutines
  * Rust: Runtime polls Futures to check progress
  -> Rust allows finer-grained control (zero-cost abstraction)
```

### 6.2 tokio Runtime Details

```rust
use tokio;
use tokio::time::{sleep, timeout, Duration};
use tokio::sync::{mpsc, Semaphore};
use std::sync::Arc;

// ===== Basic async main =====
#[tokio::main]
async fn main() {
    let result = fetch_data("https://api.example.com").await;
    match result {
        Ok(data) => println!("Data: {}", data),
        Err(e) => eprintln!("Error: {}", e),
    }
}

// ===== Concurrent Execution Patterns =====
async fn load_dashboard(user_id: u32) -> Result<Dashboard, AppError> {
    // tokio::join!: Execute all concurrently and wait for all to complete
    let (user_result, posts_result, notifs_result) = tokio::join!(
        fetch_user(user_id),
        fetch_posts(user_id),
        fetch_notifications(user_id),
    );

    Ok(Dashboard {
        user: user_result?,
        posts: posts_result?,
        notifications: notifs_result?,
    })
}

// ===== tokio::select!: Process the first to complete =====
async fn fetch_with_fallback(primary: &str, fallback: &str) -> String {
    tokio::select! {
        result = fetch_data(primary) => {
            match result {
                Ok(data) => data,
                Err(_) => fetch_data(fallback).await.unwrap_or_default(),
            }
        }
        _ = sleep(Duration::from_secs(3)) => {
            // Primary timed out -> use fallback
            fetch_data(fallback).await.unwrap_or_default()
        }
    }
}

// ===== Task Spawning: Background execution =====
async fn spawn_example() {
    let handle = tokio::spawn(async {
        // Execute asynchronously as a separate task
        heavy_computation().await
    });

    // Execute other work concurrently
    do_other_work().await;

    // Get the task result
    let result = handle.await.expect("Task panicked");
    println!("Result: {:?}", result);
}

// ===== Limiting concurrency with a Semaphore =====
async fn fetch_many_urls(urls: Vec<String>) -> Vec<Result<String, reqwest::Error>> {
    let semaphore = Arc::new(Semaphore::new(10)); // Max 10 concurrent
    let mut handles = vec![];

    for url in urls {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let handle = tokio::spawn(async move {
            let result = reqwest::get(&url).await?.text().await;
            drop(permit); // Explicitly release the semaphore
            result
        });
        handles.push(handle);
    }

    let mut results = vec![];
    for handle in handles {
        results.push(handle.await.unwrap());
    }
    results
}
```

### 6.3 Rust Streams (Async Iterators)

```rust
use tokio_stream::{self as stream, StreamExt};

// ===== Stream: Async Iterator =====
async fn process_stream() {
    let mut stream = stream::iter(vec![1, 2, 3, 4, 5])
        .map(|x| async move {
            sleep(Duration::from_millis(100)).await;
            x * 2
        })
        .buffered(3); // Process up to 3 concurrently

    while let Some(value) = stream.next().await {
        println!("Value: {}", value);
    }
}

// ===== Channel-based Stream =====
async fn event_producer(tx: mpsc::Sender<String>) {
    for i in 0..100 {
        tx.send(format!("Event {}", i)).await.unwrap();
        sleep(Duration::from_millis(10)).await;
    }
}

async fn event_consumer(mut rx: mpsc::Receiver<String>) {
    while let Some(event) = rx.recv().await {
        println!("Received: {}", event);
    }
}

async fn channel_example() {
    let (tx, rx) = mpsc::channel(32); // Buffer size 32

    tokio::spawn(event_producer(tx));
    event_consumer(rx).await;
}
```

---

## 7. Go Concurrency (goroutine + channel)

Go does not adopt async/await; instead, it uses goroutines and channels based on the CSP (Communicating Sequential Processes) model.

### 7.1 goroutine Basics

```go
package main

import (
    "fmt"
    "net/http"
    "sync"
    "time"
    "context"
    "encoding/json"
    "io"
)

// ===== goroutine: Launch a lightweight thread with the go keyword =====
func fetchURL(url string) ([]byte, error) {
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    return io.ReadAll(resp.Body)
}

func main() {
    // Normal function call (synchronous)
    data, _ := fetchURL("https://api.example.com/data")
    fmt.Println(string(data))

    // Execute asynchronously with goroutine
    go func() {
        data, _ := fetchURL("https://api.example.com/other")
        fmt.Println(string(data))
    }()

    // Wait so main doesn't exit before goroutine completes
    time.Sleep(2 * time.Second)
}
```

### 7.2 Communication via Channels

```go
// ===== channel: Communication channels between goroutines =====

type User struct {
    ID   int    `json:"id"`
    Name string `json:"name"`
}

func fetchUser(id int, ch chan<- User) {
    resp, err := http.Get(fmt.Sprintf("https://api.example.com/users/%d", id))
    if err != nil {
        return
    }
    defer resp.Body.Close()

    var user User
    json.NewDecoder(resp.Body).Decode(&user)
    ch <- user // Send to channel
}

func main() {
    ch := make(chan User, 3) // Buffered channel

    // Launch 3 goroutines concurrently
    for i := 1; i <= 3; i++ {
        go fetchUser(i, ch)
    }

    // Collect 3 results
    users := make([]User, 0, 3)
    for i := 0; i < 3; i++ {
        user := <-ch // Receive from channel
        users = append(users, user)
    }
    fmt.Println(users)
}

// ===== select: Multiplexing multiple channels =====
func fetchWithTimeout(url string, timeout time.Duration) ([]byte, error) {
    resultCh := make(chan []byte, 1)
    errCh := make(chan error, 1)

    go func() {
        data, err := fetchURL(url)
        if err != nil {
            errCh <- err
            return
        }
        resultCh <- data
    }()

    select {
    case data := <-resultCh:
        return data, nil
    case err := <-errCh:
        return nil, err
    case <-time.After(timeout):
        return nil, fmt.Errorf("timeout after %v", timeout)
    }
}

// ===== WaitGroup: Wait for goroutines to complete =====
func fetchMultipleURLs(urls []string) map[string][]byte {
    var mu sync.Mutex
    var wg sync.WaitGroup
    results := make(map[string][]byte)

    for _, url := range urls {
        wg.Add(1)
        go func(u string) {
            defer wg.Done()
            data, err := fetchURL(u)
            if err != nil {
                return
            }
            mu.Lock()
            results[u] = data
            mu.Unlock()
        }(url)
    }

    wg.Wait() // Wait for all goroutines to complete
    return results
}

// ===== context: Cancellation and Timeout =====
func fetchWithContext(ctx context.Context, url string) ([]byte, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    return io.ReadAll(resp.Body)
}

func main() {
    // Context with 5-second timeout
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    data, err := fetchWithContext(ctx, "https://api.example.com/data")
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            fmt.Println("Request timed out")
        }
        return
    }
    fmt.Println(string(data))
}
```

```
Go goroutine Scheduling Model (M:N Mapping):

  OS Threads (M)         goroutines (G)       Processors (P)
  +------------+           +----++----+         +------------+
  | Thread-1   |<----------| G1 || G2 |<--------| P1         |
  |            |           +----++----+         | LocalQ     |
  +------------+           +----++----+         +------------+
  +------------+           | G3 || G4 |
  | Thread-2   |<----------+----++----+         +------------+
  |            |           +----+               | P2         |
  +------------+           | G5 |<--------------| LocalQ     |
  +------------+           +----+               +------------+
  | Thread-3   | (idle)
  |            |           +----++----++----+
  +------------+           | G6 || G7 || G8 |   <- Global Queue
                           +----++----++----+

  - G (goroutine): ~2-8 KB stack (dynamically grows)
  - M (Machine/OS Thread): OS kernel thread
  - P (Processor): Logical processor (set by GOMAXPROCS)

  Work Stealing:
    If P1's queue is empty -> "steal" a G from P2's queue
    -> Load is balanced evenly across all processors
```

---

## 8. C# and Java Async Models

### 8.1 C# async/await (Task-based)

```csharp
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

// ===== Basic async/await =====
public class ApiClient
{
    private readonly HttpClient _httpClient = new HttpClient();

    public async Task<string> FetchDataAsync(string url)
    {
        // await: Asynchronously wait for the Task to complete
        HttpResponseMessage response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();
        string content = await response.Content.ReadAsStringAsync();
        return content;
    }

    // ===== Concurrent Execution: Task.WhenAll =====
    public async Task<Dashboard> LoadDashboardAsync(int userId)
    {
        // Start 3 tasks simultaneously
        Task<User> userTask = FetchUserAsync(userId);
        Task<List<Post>> postsTask = FetchPostsAsync(userId);
        Task<List<Notification>> notifsTask = FetchNotificationsAsync(userId);

        // Wait for all tasks to complete
        await Task.WhenAll(userTask, postsTask, notifsTask);

        return new Dashboard
        {
            User = userTask.Result,
            Posts = postsTask.Result,
            Notifications = notifsTask.Result,
        };
    }

    // ===== CancellationToken: Cancellation control =====
    public async Task<string> FetchWithCancellationAsync(
        string url,
        CancellationToken cancellationToken)
    {
        try
        {
            HttpResponseMessage response = await _httpClient.GetAsync(
                url, cancellationToken);
            return await response.Content.ReadAsStringAsync(cancellationToken);
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("Request was cancelled");
            return null;
        }
    }

    // ===== Request with Timeout =====
    public async Task<string> FetchWithTimeoutAsync(
        string url, TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        return await FetchWithCancellationAsync(url, cts.Token);
    }
}

// ===== IAsyncEnumerable: Async Streams (C# 8.0+) =====
public class PaginatedApi
{
    public async IAsyncEnumerable<List<Item>> FetchAllPagesAsync(
        string baseUrl,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        int page = 1;
        bool hasMore = true;

        while (hasMore && !ct.IsCancellationRequested)
        {
            var response = await FetchPageAsync($"{baseUrl}?page={page}", ct);
            yield return response.Items;
            hasMore = response.HasNextPage;
            page++;
        }
    }
}

// Usage
await foreach (var items in api.FetchAllPagesAsync("/api/items"))
{
    foreach (var item in items)
    {
        await ProcessItemAsync(item);
    }
}
```

### 8.2 Java Virtual Threads (Project Loom, Java 21+)

```java
import java.net.http.*;
import java.net.URI;
import java.time.Duration;
import java.util.concurrent.*;
import java.util.List;
import java.util.stream.Collectors;

// ===== Virtual Threads: Lightweight Threads (Java 21+) =====
public class AsyncJavaExample {

    private static final HttpClient client = HttpClient.newHttpClient();

    public static String fetchData(String url) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(url))
            .timeout(Duration.ofSeconds(10))
            .build();

        HttpResponse<String> response = client.send(
            request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public static void main(String[] args) throws Exception {
        // ===== Method 1: Directly start a Virtual Thread =====
        Thread vThread = Thread.ofVirtual().start(() -> {
            try {
                String data = fetchData("https://api.example.com/data");
                System.out.println(data);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        vThread.join();

        // ===== Method 2: ExecutorService + Virtual Threads =====
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<String>> futures = List.of(
                executor.submit(() -> fetchData("https://api.example.com/users")),
                executor.submit(() -> fetchData("https://api.example.com/posts")),
                executor.submit(() -> fetchData("https://api.example.com/notifications"))
            );

            for (Future<String> future : futures) {
                System.out.println(future.get()); // Get result
            }
        }

        // ===== Method 3: StructuredTaskScope (Structured Concurrency Preview) =====
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            Future<String> users = scope.fork(() ->
                fetchData("https://api.example.com/users"));
            Future<String> posts = scope.fork(() ->
                fetchData("https://api.example.com/posts"));

            scope.join();           // Wait for all tasks to complete
            scope.throwIfFailed();  // Throw exception if any failed

            System.out.println(users.resultNow());
            System.out.println(posts.resultNow());
        }
    }
}

// ===== CompletableFuture: Traditional Async API =====
public class CompletableFutureExample {

    public static CompletableFuture<String> fetchAsync(String url) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                return fetchData(url);
            } catch (Exception e) {
                throw new CompletionException(e);
            }
        });
    }

    public static void main(String[] args) {
        // Chaining: thenApply, thenCompose, thenCombine
        CompletableFuture<String> future = fetchAsync("/api/users/1")
            .thenApply(json -> parseUser(json))        // Transform
            .thenCompose(user -> fetchAsync(            // Next async operation
                "/api/posts?userId=" + user.getId()))
            .exceptionally(ex -> {                     // Error handling
                System.err.println("Error: " + ex.getMessage());
                return "[]";
            });

        // Concurrent execution: allOf
        CompletableFuture<Void> all = CompletableFuture.allOf(
            fetchAsync("/api/users"),
            fetchAsync("/api/posts"),
            fetchAsync("/api/notifications")
        );
        all.join(); // Wait for all to complete
    }
}
```

---

## 9. Comparative Analysis of Async Runtimes

### 9.1 Architecture Comparison Table

```
+------------+-----------------+------------------+-------------------+
| Feature    | JavaScript      | Python           | Rust              |
|            | (Node.js)       | (asyncio)        | (tokio)           |
+------------+-----------------+------------------+-------------------+
| Execution  | Single-threaded | Single-threaded  | Multi-threaded    |
| Model      | Event loop      | Event loop       | Work stealing     |
+------------+-----------------+------------------+-------------------+
| I/O        | libuv           | selector/epoll   | mio(epoll/kqueue) |
| Backend    | (cross-platform)| (OS-dependent)   | (cross-platform)  |
+------------+-----------------+------------------+-------------------+
| CPU        | Worker Threads  | multiprocessing  | Native threads    |
| Parallelism| (limited)       | (bypasses GIL)   | (full parallelism)|
+------------+-----------------+------------------+-------------------+
| Memory     | V8 heap         | Object           | Zero-cost         |
| Overhead   | (mid-coroutine) | (mid-coroutine)  | (state machine)   |
+------------+-----------------+------------------+-------------------+
| Cancellation| AbortController| Task.cancel()    | drop (RAII)       |
+------------+-----------------+------------------+-------------------+
| Maturity   | Very high       | High             | High              |
+------------+-----------------+------------------+-------------------+
| Learning   | Low             | Medium           | High              |
| Curve      |                 |                  |                   |
+------------+-----------------+------------------+-------------------+

+------------+-----------------+------------------+-------------------+
| Feature    | Go              | C#               | Java (21+)        |
+------------+-----------------+------------------+-------------------+
| Execution  | M:N scheduling  | Thread pool      | Virtual Threads   |
| Model      | Work stealing   | + async/await    | M:N scheduling    |
+------------+-----------------+------------------+-------------------+
| I/O        | netpoller       | OS-dependent     | NIO / Virtual     |
| Backend    | (built-in)      | (IOCP/epoll)     | Thread integration|
+------------+-----------------+------------------+-------------------+
| CPU        | Auto-distributed| Task Parallel Lib| Thread pool       |
| Parallelism| (GOMAXPROCS)    | (Parallel.For)   | (ForkJoinPool)    |
+------------+-----------------+------------------+-------------------+
| Memory     | goroutine: ~2KB | Task: ~300B      | VThread: ~100s B  |
| Overhead   | (dynamic stack) | (heap-allocated)  | (continuation)   |
+------------+-----------------+------------------+-------------------+
| Cancellation| context.Context| CancellationToken| Thread.interrupt  |
+------------+-----------------+------------------+-------------------+
| Maturity   | Very high       | Very high        | Growing           |
+------------+-----------------+------------------+-------------------+
| Learning   | Low             | Medium           | Low (VThread)     |
| Curve      |                 |                  |                   |
+------------+-----------------+------------------+-------------------+
```

### 9.2 Recommended Models by Use Case

```
Async Model Selection Guide by Use Case:

+-----------------------------+------------------------------------+
| Use Case                    | Recommended Model                  |
+-----------------------------+------------------------------------+
| Web Frontend                | JavaScript async/await             |
| (Browser)                   | -> Only option, mature ecosystem   |
+-----------------------------+------------------------------------+
| REST API Server             | Go (goroutine) / Node.js           |
| (High throughput)           | -> Simple and high-performance     |
+-----------------------------+------------------------------------+
| Microservices               | Go / Rust / Java (VThread)         |
| (Low latency)               | -> Go: dev speed, Rust: max perf  |
+-----------------------------+------------------------------------+
| Data Pipelines              | Python (asyncio + multiprocessing) |
| (Mixed I/O + CPU)           | -> Rich data processing libraries  |
+-----------------------------+------------------------------------+
| Real-time Communication     | Go / Rust / Node.js                |
| (WebSocket etc.)            | -> Strong at massive connections   |
+-----------------------------+------------------------------------+
| Enterprise                  | C# / Java                          |
| (Leveraging existing assets)| -> Ecosystem and compatibility     |
+-----------------------------+------------------------------------+
| Embedded/IoT                | Rust (embassy/no_std)              |
| (Resource constraints)      | -> Zero-cost, no runtime possible  |
+-----------------------------+------------------------------------+
```

---

## 10. Anti-Patterns and Pitfalls

### 10.1 Anti-Pattern 1: Sequential Await Trap

Sequentially awaiting independent async operations serializes what could otherwise run concurrently.

```javascript
// ===== BAD: Sequential await (unnecessarily slow) =====
async function loadPageData(userId) {
    // These 3 operations don't depend on each other, but execute serially
    const user = await fetchUser(userId);        // 200ms wait
    const posts = await fetchPosts(userId);       // 300ms wait
    const notifications = await fetchNotifications(userId); // 150ms wait
    // Total: 200 + 300 + 150 = 650ms

    return { user, posts, notifications };
}

// ===== GOOD: Concurrent await (fast) =====
async function loadPageData(userId) {
    // Execute concurrently with Promise.all
    const [user, posts, notifications] = await Promise.all([
        fetchUser(userId),        // -+
        fetchPosts(userId),       // -| concurrent execution
        fetchNotifications(userId), // -+
    ]);
    // Total: max(200, 300, 150) = 300ms (approximately 54% faster)

    return { user, posts, notifications };
}

// ===== BEST: Optimization considering dependencies =====
async function loadPageData(userId) {
    // Step 1: user is needed by other operations, so fetch it first
    const user = await fetchUser(userId);

    // Step 2: posts and notifications can run concurrently
    const [posts, notifications] = await Promise.all([
        fetchPosts(user.id),
        fetchNotifications(user.id),
    ]);

    // Step 3: comments depends on posts, so execute sequentially
    const comments = await fetchComments(posts[0]?.id);

    return { user, posts, notifications, comments };
}
```

```
Sequential vs Concurrent Execution Time Comparison:

Sequential Execution:
  fetchUser --[200ms]--+
                       |
  fetchPosts ----------+--[300ms]--+
                                   |
  fetchNotifs --------- -----------+--[150ms]--+
                                               |
  Total: ------------------------------------------+ 650ms

Concurrent Execution:
  fetchUser --[200ms]--+
  fetchPosts --[300ms]-+ <- Slowest operation is the bottleneck
  fetchNotifs [150ms]--+
                       |
  Total: --------  ----+ 300ms (54% reduction)
```

### 10.2 Anti-Pattern 2: Improper Mixing of async/await

```javascript
// ===== BAD: Confusion from mixing Promise and async/await =====
async function processData() {
    // Trying to use await inside .then() (no error, but confusing)
    const result = fetch('/api/data')
        .then(async (response) => {
            const json = await response.json();
            const processed = await processJson(json);
            return processed;
        })
        .catch(err => {
            console.error(err);
            return null;
        });

    return result; // This could become Promise<Promise<...>>
}

// ===== GOOD: Consistent async/await =====
async function processData() {
    try {
        const response = await fetch('/api/data');
        const json = await response.json();
        const processed = await processJson(json);
        return processed;
    } catch (err) {
        console.error(err);
        return null;
    }
}
```

### 10.3 Anti-Pattern 3: Fire-and-Forget

```javascript
// ===== BAD: Calling an async function without await =====
async function handleRequest(req, res) {
    const data = await fetchData(req.params.id);

    // Calling audit log asynchronously without await
    saveAuditLog(req.user, 'data_access');  // <- Floating Promise

    // Issue 1: Errors are not caught (Unhandled Promise Rejection)
    // Issue 2: Log may be lost on process exit
    // Issue 3: Difficult to debug

    res.json(data);
}

// ===== GOOD: Explicit error handling for intentional fire-and-forget =====
async function handleRequest(req, res) {
    const data = await fetchData(req.params.id);

    // Method 1: Explicitly handle errors with catch
    saveAuditLog(req.user, 'data_access')
        .catch(err => console.error('Audit log failed:', err));

    // Method 2: Use void operator to indicate intentionality
    void saveAuditLog(req.user, 'data_access')
        .catch(err => console.error('Audit log failed:', err));

    res.json(data);
}

// ===== BEST: Use a background task queue =====
async function handleRequest(req, res) {
    const data = await fetchData(req.params.id);

    // Enqueue to task queue (with retry and monitoring)
    backgroundQueue.enqueue('audit_log', {
        user: req.user,
        action: 'data_access',
        timestamp: Date.now(),
    });

    res.json(data);
}
```

### 10.4 Anti-Pattern 4: Synchronous Blocking Inside async Functions

```python
import asyncio
import time

# ===== BAD: Blocking synchronously inside async function =====
async def process_data():
    data = await fetch_data()

    # time.sleep blocks the event loop!
    time.sleep(5)  # <- All async processing halts for 5 seconds

    # CPU-intensive work also blocks the event loop
    result = heavy_computation(data)  # <- No other processing progresses during computation

    return result

# ===== GOOD: Use async sleep and thread pool =====
async def process_data():
    data = await fetch_data()

    # asyncio.sleep does not block other processing
    await asyncio.sleep(5)

    # Run CPU-intensive work in a thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, heavy_computation, data)

    return result
```

### 10.5 Anti-Pattern 5: Deadlock

```python
# ===== BAD: Nested asyncio.run() call inside synchronous code =====
import asyncio

async def inner():
    return "result"

async def outer():
    # Calling asyncio.run() while an event loop is already running causes deadlock
    result = asyncio.run(inner())  # RuntimeError!
    return result

# ===== GOOD: Use await =====
async def outer():
    result = await inner()  # Correct approach
    return result
```

```csharp
// ===== BAD: Deadlock from .Result / .Wait() in C# =====
// Dangerous in environments with SynchronizationContext (ASP.NET / WPF)
public string GetData()
{
    // Synchronously waiting on an async method with .Result
    // -> Deadlock due to SynchronizationContext capture
    var result = FetchDataAsync().Result;  // <- Deadlock!
    return result;
}

// ===== GOOD: Propagate async all the way =====
public async Task<string> GetDataAsync()
{
    var result = await FetchDataAsync();  // Correct
    return result;
}

// ConfigureAwait(false) to ignore SynchronizationContext
public async Task<string> GetDataAsync()
{
    var result = await FetchDataAsync().ConfigureAwait(false);
    return result;
}
```

---

## 11. Design Patterns and Best Practices

### 11.1 Retry Pattern (Exponential Backoff)

```javascript
// ===== Retry with Exponential Backoff =====
async function fetchWithRetry(url, options = {}) {
    const {
        maxRetries = 3,
        baseDelay = 1000,    // Initial wait: 1 second
        maxDelay = 30000,    // Maximum wait: 30 seconds
        backoffFactor = 2,   // Multiplier
        retryableStatuses = [408, 429, 500, 502, 503, 504],
    } = options;

    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const response = await fetch(url);

            if (response.ok) {
                return await response.json();
            }

            if (!retryableStatuses.includes(response.status)) {
                throw new Error(`Non-retryable HTTP ${response.status}`);
            }

            lastError = new Error(`HTTP ${response.status}`);
        } catch (error) {
            lastError = error;

            if (error.name === 'AbortError') {
                throw error; // Don't retry cancellations
            }
        }

        if (attempt < maxRetries) {
            // Exponential backoff + jitter
            const delay = Math.min(
                baseDelay * Math.pow(backoffFactor, attempt),
                maxDelay
            );
            const jitter = delay * (0.5 + Math.random() * 0.5);

            console.log(
                `Retry ${attempt + 1}/${maxRetries} after ${Math.round(jitter)}ms`
            );
            await new Promise(resolve => setTimeout(resolve, jitter));
        }
    }

    throw lastError;
}

// Usage
const data = await fetchWithRetry('https://api.example.com/data', {
    maxRetries: 5,
    baseDelay: 500,
});
```

```
Exponential Backoff Visualization:

Retry Count   Wait Time (no jitter)    Wait Time (with jitter, approx.)
    0          Send request              Send request
    v          Failure                   Failure
    1          1,000ms                   500-1,500ms
    v          Failure                   Failure
    2          2,000ms                   1,000-3,000ms
    v          Failure                   Failure
    3          4,000ms                   2,000-6,000ms
    v          Failure                   Failure
    4          8,000ms                   4,000-12,000ms
    v          Failure                   Failure
    5          Give up                   Give up

  Why Jitter Matters:
    When many clients fail simultaneously,
    if all clients retry at the same timing,
    a "retry storm" hits the server.
    Adding random jitter distributes the load.
```

### 11.2 Circuit Breaker Pattern

```javascript
// ===== Circuit Breaker: Preventing cascading failures =====
class CircuitBreaker {
    constructor(options = {}) {
        this.failureThreshold = options.failureThreshold || 5;
        this.resetTimeout = options.resetTimeout || 60000; // 60 seconds
        this.state = 'CLOSED';  // CLOSED -> OPEN -> HALF_OPEN -> CLOSED
        this.failureCount = 0;
        this.lastFailureTime = null;
        this.successCount = 0;
    }

    async execute(asyncFn) {
        if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailureTime > this.resetTimeout) {
                this.state = 'HALF_OPEN';
                this.successCount = 0;
            } else {
                throw new Error('Circuit breaker is OPEN');
            }
        }

        try {
            const result = await asyncFn();
            this._onSuccess();
            return result;
        } catch (error) {
            this._onFailure();
            throw error;
        }
    }

    _onSuccess() {
        if (this.state === 'HALF_OPEN') {
            this.successCount++;
            if (this.successCount >= 3) {
                this.state = 'CLOSED';
                this.failureCount = 0;
            }
        } else {
            this.failureCount = 0;
        }
    }

    _onFailure() {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
        }
    }
}

// Usage
const breaker = new CircuitBreaker({
    failureThreshold: 5,
    resetTimeout: 30000,
});

async function callExternalService() {
    return breaker.execute(async () => {
        const response = await fetch('https://external-api.example.com/data');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    });
}
```

```
Circuit Breaker State Transitions:

  +----------+  Failures exceed threshold  +----------+
  | CLOSED   | --------------------------> |  OPEN    |
  | (Normal) |                             | (Tripped)|
  |          |                             |          |
  | Passes   |                             | Immediate|
  | requests |                             | error    |
  +----^-----+                             +----+-----+
       |                                        |
       | Consecutive successes recover           | Reset timeout elapsed
       |                                        |
  +----+-----+                             +----v-----+
  |          | <--------------------------- | HALF_OPEN|
  |          |  Test succeeds               | (Probing)|
  |          |                             |          |
  +----------+  Test fails -> back to      | Limited  |
                OPEN                       | requests |
                                           | allowed  |
                                           +----------+
```

### 11.3 Bulkhead Pattern (Resource Isolation with Semaphores)

```python
import asyncio

class BulkheadExecutor:
    """
    Bulkhead Pattern: Isolate concurrency limits per service to prevent
    a failure in one service from cascading to others
    """

    def __init__(self):
        self.semaphores = {}

    def register(self, service_name: str, max_concurrent: int):
        """Register a concurrency limit per service"""
        self.semaphores[service_name] = asyncio.Semaphore(max_concurrent)

    async def execute(self, service_name: str, coro):
        """Execute an async operation within the specified service's bulkhead"""
        sem = self.semaphores.get(service_name)
        if sem is None:
            raise ValueError(f"Unknown service: {service_name}")

        async with sem:
            return await coro

# Usage
bulkhead = BulkheadExecutor()
bulkhead.register("user_service", max_concurrent=20)
bulkhead.register("payment_service", max_concurrent=5)   # More restrictive
bulkhead.register("notification_service", max_concurrent=50)

async def handle_order(order_id: int):
    user = await bulkhead.execute(
        "user_service", fetch_user(order_id)
    )
    payment = await bulkhead.execute(
        "payment_service", process_payment(order_id)
    )
    await bulkhead.execute(
        "notification_service", send_notification(user, payment)
    )
```

---

## 12. Exercises

### 12.1 Beginner Exercises

**Exercise 1: Basic Async Data Fetching**

Implement a `fetchAllUsers` function in JavaScript that meets the following requirements:

- Parameters: An array of user IDs `userIds: number[]`
- Concurrently fetch user information from `/api/users/:id`
- Return all user information as an array
- Throw an error if any request fails

```javascript
// Solution template
async function fetchAllUsers(userIds) {
    // Implement here
}

// Test
// const users = await fetchAllUsers([1, 2, 3, 4, 5]);
// console.log(users); // [{id:1,name:"Alice"}, {id:2,...}, ...]
```

<details>
<summary>Solution (click to expand)</summary>

```javascript
async function fetchAllUsers(userIds) {
    const promises = userIds.map(id =>
        fetch(`/api/users/${id}`).then(res => {
            if (!res.ok) throw new Error(`User ${id}: HTTP ${res.status}`);
            return res.json();
        })
    );

    return Promise.all(promises);
}

// Improved version: Limit concurrent connections
async function fetchAllUsersLimited(userIds, concurrency = 5) {
    const results = [];
    const executing = new Set();

    for (const id of userIds) {
        const promise = fetch(`/api/users/${id}`)
            .then(res => res.json())
            .then(user => {
                executing.delete(promise);
                return user;
            });

        executing.add(promise);
        results.push(promise);

        if (executing.size >= concurrency) {
            await Promise.race(executing);
        }
    }

    return Promise.all(results);
}
```

</details>

**Exercise 2: Implement Fetching with Timeout in Python**

Implement a function in Python that meets the following requirements:

- Concurrently fetch multiple URLs
- Set an individual timeout (default 5 seconds) for each request
- Exclude timed-out URLs from results, returning only successful ones
- Support an overall timeout (default 30 seconds)

```python
# Solution template
async def fetch_urls_with_timeout(
    urls: list[str],
    per_request_timeout: float = 5.0,
    total_timeout: float = 30.0,
) -> dict[str, str]:
    """Return a mapping of successful URLs to their results"""
    # Implement here
    pass
```

<details>
<summary>Solution (click to expand)</summary>

```python
import asyncio
import aiohttp

async def fetch_urls_with_timeout(
    urls: list[str],
    per_request_timeout: float = 5.0,
    total_timeout: float = 30.0,
) -> dict[str, str]:
    results = {}

    async def fetch_one(session: aiohttp.ClientSession, url: str):
        try:
            async with asyncio.timeout(per_request_timeout):
                async with session.get(url) as resp:
                    text = await resp.text()
                    results[url] = text
        except (TimeoutError, aiohttp.ClientError) as e:
            print(f"Failed {url}: {e}")

    async with asyncio.timeout(total_timeout):
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[fetch_one(session, url) for url in urls],
                return_exceptions=True,
            )

    return results
```

</details>

### 12.2 Intermediate Exercises

**Exercise 3: Async Rate Limiter**

Implement a rate limiter in JavaScript that meets the following requirements:

- Use a token bucket algorithm
- Limit to a maximum of N requests per second
- Automatically wait for token replenishment when the limit is exceeded
- Provide an async/await-compatible interface

```javascript
// Solution template
class RateLimiter {
    constructor(maxTokens, refillRate) {
        // Implement here
    }

    async acquire() {
        // Acquire a token (wait if limit exceeded)
    }
}

// Usage
// const limiter = new RateLimiter(10, 10); // Max 10 tokens, 10 tokens/sec refill
// await limiter.acquire(); // Acquire token
// await fetch('/api/data'); // Execute request
```

<details>
<summary>Solution (click to expand)</summary>

```javascript
class RateLimiter {
    constructor(maxTokens, refillRatePerSecond) {
        this.maxTokens = maxTokens;
        this.tokens = maxTokens;
        this.refillRate = refillRatePerSecond;
        this.lastRefillTime = Date.now();
        this.waitQueue = [];
    }

    _refill() {
        const now = Date.now();
        const elapsed = (now - this.lastRefillTime) / 1000;
        this.tokens = Math.min(
            this.maxTokens,
            this.tokens + elapsed * this.refillRate
        );
        this.lastRefillTime = now;
    }

    async acquire() {
        this._refill();

        if (this.tokens >= 1) {
            this.tokens -= 1;
            return;
        }

        // Insufficient tokens: wait for replenishment
        const waitTime = ((1 - this.tokens) / this.refillRate) * 1000;

        return new Promise(resolve => {
            setTimeout(() => {
                this._refill();
                this.tokens -= 1;
                resolve();
                // Process any items waiting in queue
                this._processQueue();
            }, waitTime);
        });
    }

    _processQueue() {
        while (this.waitQueue.length > 0) {
            this._refill();
            if (this.tokens < 1) break;
            this.tokens -= 1;
            const resolve = this.waitQueue.shift();
            resolve();
        }
    }
}

// Usage: Rate-limit API calls
async function fetchWithRateLimit(urls) {
    const limiter = new RateLimiter(10, 10);
    const results = [];

    for (const url of urls) {
        await limiter.acquire();
        results.push(fetch(url).then(r => r.json()));
    }

    return Promise.all(results);
}
```

</details>

### 12.3 Advanced Exercises

**Exercise 4: Async Task Scheduler**

Implement a task scheduler that meets the following requirements. Language is your choice.

Requirements:
- Priority task queue (high / medium / low)
- Configurable concurrency limit
- Declarable task dependencies (task A runs after task B completes)
- Timeout functionality
- Cancellation functionality

```javascript
// Interface example
class TaskScheduler {
    constructor(maxConcurrent) { /* ... */ }

    addTask(id, asyncFn, options) {
        // options: { priority, dependsOn, timeout }
    }

    async run() {
        // Execute all tasks according to dependencies and priorities
    }

    cancel(taskId) { /* ... */ }
}
```

<details>
<summary>Solution (click to expand)</summary>

```javascript
class TaskScheduler {
    constructor(maxConcurrent = 5) {
        this.maxConcurrent = maxConcurrent;
        this.tasks = new Map();
        this.results = new Map();
        this.running = new Set();
        this.completed = new Set();
        this.cancelled = new Set();
    }

    addTask(id, asyncFn, options = {}) {
        const { priority = 'medium', dependsOn = [], timeout = 0 } = options;
        const priorityValue = { high: 0, medium: 1, low: 2 }[priority];

        this.tasks.set(id, {
            id,
            fn: asyncFn,
            priority: priorityValue,
            dependsOn,
            timeout,
        });
    }

    cancel(taskId) {
        this.cancelled.add(taskId);
    }

    _getReadyTasks() {
        const ready = [];
        for (const [id, task] of this.tasks) {
            if (this.completed.has(id) || this.running.has(id) || this.cancelled.has(id)) {
                continue;
            }
            const depsReady = task.dependsOn.every(dep => this.completed.has(dep));
            if (depsReady) {
                ready.push(task);
            }
        }
        return ready.sort((a, b) => a.priority - b.priority);
    }

    async _executeTask(task) {
        this.running.add(task.id);

        try {
            let result;
            if (task.timeout > 0) {
                result = await Promise.race([
                    task.fn(this.results),
                    new Promise((_, reject) =>
                        setTimeout(() => reject(new Error(`Task ${task.id} timed out`)), task.timeout)
                    ),
                ]);
            } else {
                result = await task.fn(this.results);
            }
            this.results.set(task.id, { status: 'fulfilled', value: result });
        } catch (error) {
            this.results.set(task.id, { status: 'rejected', reason: error });
        } finally {
            this.running.delete(task.id);
            this.completed.add(task.id);
        }
    }

    async run() {
        while (true) {
            const allDone = [...this.tasks.keys()].every(
                id => this.completed.has(id) || this.cancelled.has(id)
            );
            if (allDone) break;

            const ready = this._getReadyTasks();
            const slots = this.maxConcurrent - this.running.size;

            if (ready.length === 0 && this.running.size === 0) {
                // Deadlock detection: no executable or running tasks
                const remaining = [...this.tasks.keys()].filter(
                    id => !this.completed.has(id) && !this.cancelled.has(id)
                );
                throw new Error(`Deadlock detected. Blocked tasks: ${remaining.join(', ')}`);
            }

            const toRun = ready.slice(0, slots);
            const promises = toRun.map(task => this._executeTask(task));

            if (promises.length > 0) {
                await Promise.race([...promises, ...Array.from(this.running)].filter(Boolean));
            } else if (this.running.size > 0) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        return this.results;
    }
}

// Usage
const scheduler = new TaskScheduler(3);

scheduler.addTask('fetchUser', async () => {
    return await fetch('/api/users/1').then(r => r.json());
}, { priority: 'high' });

scheduler.addTask('fetchPosts', async (results) => {
    const user = results.get('fetchUser').value;
    return await fetch(`/api/posts?userId=${user.id}`).then(r => r.json());
}, { priority: 'medium', dependsOn: ['fetchUser'], timeout: 5000 });

scheduler.addTask('fetchAnalytics', async () => {
    return await fetch('/api/analytics').then(r => r.json());
}, { priority: 'low' });

const results = await scheduler.run();
```

</details>

---

## 13. FAQ (Frequently Asked Questions)

### Q1: What is the difference between async/await and multithreading?

**A1:** async/await provides "concurrency," while multithreading provides "parallelism."

```
Concurrency:
  A single executor switches between multiple tasks
  -> Achievable even with 1 core
  -> Proceeds with other work during I/O wait

  Thread-1: -[Task-A]-[Task-B]-[Task-A]-[Task-C]-[Task-B]-

Parallelism:
  Multiple executors process simultaneously
  -> Requires multiple cores
  -> Speeds up CPU-intensive processing

  Core-1: -[Task-A]-[Task-A]-[Task-A]-
  Core-2: -[Task-B]-[Task-B]-[Task-B]-
  Core-3: -[Task-C]-[Task-C]-[Task-C]-

Where async/await is effective:
  - Network I/O (HTTP, DB, WebSocket)
  - File I/O
  - Timer operations
  -> Tasks with lots of waiting and little CPU usage

Where multithreading is effective:
  - Image processing / video encoding
  - Scientific computing / machine learning
  - Cryptographic operations
  -> Computation tasks that fully utilize the CPU
```

### Q2: What happens if you forget await in JavaScript?

**A2:** The Promise object is returned as-is, causing unintended behavior.

```javascript
// ===== Behavior when await is forgotten =====

async function getUser(id) {
    const response = fetch(`/api/users/${id}`);  // Missing await!

    // response is a Promise object, not a Response object
    console.log(response);          // Promise { <pending> }
    console.log(response.status);   // undefined (Promise has no status property)

    // JSON parsing also fails
    // response.json() -> TypeError: response.json is not a function
}

// ===== Correct way =====
async function getUser(id) {
    const response = await fetch(`/api/users/${id}`);  // With await
    console.log(response);          // Response {...}
    console.log(response.status);   // 200
    return await response.json();
}

// ===== Detectable with ESLint =====
// eslint rule: "@typescript-eslint/no-floating-promises": "error"
// -> Detects unhandled Promises and reports them as errors
```

### Q3: Can Python's asyncio utilize multiple cores?

**A3:** asyncio alone cannot utilize multiple cores (due to the GIL constraint). For CPU-intensive processing, combine it with `multiprocessing` or `ProcessPoolExecutor`.

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# CPU-intensive task
def cpu_heavy_task(data):
    """CPU-intensive processing affected by the GIL"""
    return sum(x ** 2 for x in range(data))

async def process_with_multiprocessing(items: list[int]) -> list[int]:
    """CPU parallel processing with asyncio + ProcessPoolExecutor"""
    loop = asyncio.get_event_loop()

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Execute in parallel on each CPU core
        futures = [
            loop.run_in_executor(executor, cpu_heavy_task, item)
            for item in items
        ]
        results = await asyncio.gather(*futures)

    return results

# Hybrid of I/O and CPU
async def hybrid_pipeline(urls: list[str]) -> list:
    # Phase 1: I/O-intensive (concurrent with asyncio)
    raw_data = await asyncio.gather(
        *[fetch_data(url) for url in urls]
    )

    # Phase 2: CPU-intensive (parallel with ProcessPoolExecutor)
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        processed = await asyncio.gather(
            *[loop.run_in_executor(executor, process, data) for data in raw_data]
        )

    return processed
```

### Q4: Why does the return type of async fn in Rust become complex?

**A4:** Rust's Futures generate a unique anonymous type (state machine) for each async fn, making it difficult to explicitly specify the type.

```rust
// The Future type generated by async fn is anonymous
async fn fetch_data() -> String {
    // The compiler transforms this function into a state machine like:
    // enum FetchDataFuture {
    //     State0 { ... },  // Before await
    //     State1 { ... },  // After first await
    //     State2 { ... },  // After second await
    //     Done,
    // }
    let resp = reqwest::get("https://example.com").await.unwrap();
    resp.text().await.unwrap()
}

// When holding as a function pointer:
// Method 1: impl Future (static dispatch, recommended)
fn make_future() -> impl Future<Output = String> {
    fetch_data()
}

// Method 2: Box<dyn Future> (dynamic dispatch, heap allocation)
fn make_boxed_future() -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(fetch_data())
}

// Method 3: Hold as trait object (when storing in collections)
async fn run_multiple() {
    let futures: Vec<Pin<Box<dyn Future<Output = String>>>> = vec![
        Box::pin(fetch_data()),
        Box::pin(another_async_fn()),
    ];

    for future in futures {
        let result = future.await;
        println!("{}", result);
    }
}
```

### Q5: Are there cases where async/await should not be used?

**A5:** The following cases are where async/await is inappropriate or unnecessary.

```
Cases where async/await is inappropriate:

1. CPU-intensive processing only
   -> Use multithreading/parallel processing instead
   -> async/await is designed to optimize I/O wait time

2. All processing completes synchronously
   -> Unnecessary overhead is introduced
   -> Just write synchronous functions as-is

3. Concurrent processing with extensive shared state
   -> async/await + shared state = risk of data races
   -> Actor model or CSP (Go channels) is safer

4. Ultra-low latency requirements
   -> Event loop overhead can become problematic
   -> io_uring (Linux) or direct system calls are effective

5. Thinly wrapping existing synchronous APIs
   -> Wrapping in async is meaningless if internals are synchronous
   -> Python: Properly offload with asyncio.to_thread()
```

### Q6: Why didn't Go adopt async/await?

**A6:** Go's design philosophy prioritizes "simplicity" above all else. With the CSP model of goroutines + channels, all functions are implicitly async-capable, eliminating the need for syntactic distinctions like async/await.

```go
// Go: All functions "look synchronous"
func fetchUser(id int) (User, error) {
    // Even when I/O occurs internally, the goroutine scheduler
    // automatically context-switches
    resp, err := http.Get(fmt.Sprintf("/api/users/%d", id))
    if err != nil {
        return User{}, err
    }
    defer resp.Body.Close()

    var user User
    err = json.NewDecoder(resp.Body).Decode(&user)
    return user, err
}

// Concurrent execution is natural with goroutines + channels
func main() {
    ch := make(chan User, 3)
    for i := 1; i <= 3; i++ {
        go func(id int) {
            user, _ := fetchUser(id)
            ch <- user
        }(i)
    }

    for i := 0; i < 3; i++ {
        user := <-ch
        fmt.Println(user)
    }
}
```

Go's advantage: async does not infect function signatures (the function coloring problem does not exist).
Go's disadvantage: Goroutine leak detection is difficult; explicit cancellation is required (context package).

---

## 14. Debugging and Testing Async Code

### 14.1 Debugging Techniques

```javascript
// ===== Async Processing Debugging Techniques =====

// 1. Enable async stack traces
// Node.js: --async-stack-traces flag (enabled by default, v12+)

// 2. Promise labeling
async function fetchUser(id) {
    const promise = fetch(`/api/users/${id}`)
        .then(r => r.json());

    // Add a label for debugging
    promise._debugLabel = `fetchUser(${id})`;
    return promise;
}

// 3. Detect unhandled Promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection:', reason);
    console.error('Promise:', promise._debugLabel || promise);
    // In production, safely terminate the process
    process.exit(1);
});

// 4. Measuring async timing
async function timedFetch(label, asyncFn) {
    const start = performance.now();
    try {
        const result = await asyncFn();
        const elapsed = performance.now() - start;
        console.log(`[${label}] completed in ${elapsed.toFixed(1)}ms`);
        return result;
    } catch (error) {
        const elapsed = performance.now() - start;
        console.error(`[${label}] failed after ${elapsed.toFixed(1)}ms:`, error);
        throw error;
    }
}

// Usage
const user = await timedFetch('fetchUser', () => fetchUser(1));
```

### 14.2 Testing Techniques

```javascript
// ===== Async Testing with Jest/Vitest =====

// Basic: async test function
test('fetchUser returns user data', async () => {
    const user = await fetchUser(1);
    expect(user).toHaveProperty('name');
    expect(user.id).toBe(1);
});

// Mocking: Mock external APIs
test('loadDashboard aggregates data', async () => {
    // Mock fetch
    global.fetch = jest.fn()
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ id: 1, name: 'Alice' }),
        })
        .mockResolvedValueOnce({
            ok: true,
            json: async () => [{ id: 1, title: 'Post 1' }],
        });

    const dashboard = await loadDashboard(1);
    expect(dashboard.user.name).toBe('Alice');
    expect(dashboard.posts).toHaveLength(1);
});

// Timeout test: fake timers
test('fetchWithTimeout throws on timeout', async () => {
    jest.useFakeTimers();

    const promise = fetchWithTimeout('/api/slow', 1000);

    // Advance time
    jest.advanceTimersByTime(1500);

    await expect(promise).rejects.toThrow('Timeout');

    jest.useRealTimers();
});

// Error cases
test('processOrder handles payment failure', async () => {
    mockPaymentService.mockRejectedValue(new PaymentError('Declined'));

    const result = await processOrder('order-123');

    expect(result.success).toBe(false);
    expect(result.reason).toBe('payment_failed');
});
```

```python
# ===== Async Testing with pytest-asyncio (Python) =====
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_fetch_user():
    """Basic async function test"""
    user = await fetch_user(1)
    assert user["name"] == "Alice"
    assert user["id"] == 1

@pytest.mark.asyncio
async def test_load_dashboard_concurrent():
    """Concurrent execution test"""
    with patch("module.fetch_user", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"id": 1, "name": "Alice"}

        dashboard = await load_dashboard(1)
        assert dashboard.user["name"] == "Alice"

@pytest.mark.asyncio
async def test_timeout_behavior():
    """Timeout test"""
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await asyncio.sleep(1.0)
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just from theory alone, but from actually writing code and observing its behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge of this topic is frequently used in everyday development work. It is particularly important during code reviews and architecture design.

---

## 15. Summary

### 15.1 Comprehensive Comparison of Language-Specific Async Models

| Language | Async Syntax | Runtime | Execution Model | Memory Efficiency | Learning Curve | Application Domain |
|----------|-------------|---------|----------------|-------------------|---------------|-------------------|
| JavaScript | async/await | Event loop (libuv) | Single-threaded | Medium | Low | Web in general |
| Python | async/await | asyncio | Single-threaded (GIL) | Medium | Medium | I/O-intensive |
| Rust | async/await | tokio/async-std | Multi-threaded | High (zero-cost) | High | Systems/high-performance |
| Go | goroutine+chan | Built-in runtime | M:N scheduling | High | Low | Servers/infrastructure |
| C# | async/await | TaskScheduler | Thread pool | Medium | Medium | Enterprise |
| Java | Virtual Threads | Loom | M:N scheduling | High | Low | Enterprise |

### 15.2 Key Principles Checklist

```
Principles of Asynchronous Programming:

[Fundamental Principles]
  [ ] Use async/await for I/O-bound tasks
  [ ] Use thread/process pools for CPU-bound tasks
  [ ] Execute independent async operations concurrently (Promise.all / gather / join!)
  [ ] Always handle errors (don't leave unhandled Promise rejections)

[Design Principles]
  [ ] async propagates to callers (async all the way)
  [ ] Don't block-call async code from synchronous code
  [ ] Limit concurrency with semaphores
  [ ] Always set timeouts

[Operational Principles]
  [ ] Prevent cascading failures with circuit breakers
  [ ] Retry with exponential backoff + jitter
  [ ] Provide cancellation mechanisms (AbortController / context / CancellationToken)
  [ ] Monitor async processing (timing measurement, error rates)
```

---

## Recommended Next Reading


---

## References

1. Hoare, C.A.R. "Communicating Sequential Processes." *Communications of the ACM*, vol. 21, no. 8, 1978, pp. 666-677. The original paper on the CSP model. Theoretical foundation for Go's goroutine + channel.
2. "Asynchronous Programming in Rust." The Rust Async Book, rust-lang.github.io/async-book/. The official guide to asynchronous programming in Rust. Detailed explanation of the Future trait and polling model.
3. "Node.js Event Loop, Timers, and process.nextTick()." Node.js Documentation, nodejs.org/en/guides/event-loop-timers-and-nexttick. Official explanation of the Node.js event loop. Details behavior by phase.
4. Python Software Foundation. "asyncio -- Asynchronous I/O." Python Documentation, docs.python.org/3/library/asyncio.html. Official reference for Python asyncio. Includes usage of TaskGroup and timeout.
5. Cleary, Stephen. "Async in C# 5.0." O'Reilly Media, 2012. Best practices for async/await in C#. Explains SynchronizationContext and deadlock avoidance.
6. Go Authors. "Effective Go: Concurrency." go.dev/doc/effective_go#concurrency. Official guide to Go concurrency patterns. Design philosophy of goroutines and channels.
7. Goetz, Brian et al. "JEP 444: Virtual Threads." OpenJDK, openjdk.org/jeps/444. Specification for Java Virtual Threads. Design of lightweight threads via Project Loom.
```
