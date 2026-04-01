# Synchronous vs Asynchronous

> Synchronous processing means "wait for the previous operation to finish before starting the next one," while asynchronous processing means "do other work during wait times." The key to web application performance is handling I/O waits efficiently.

## What You Will Learn in This Chapter

- [ ] Understand the fundamental difference between synchronous and asynchronous processing
- [ ] Grasp the meaning of blocking and non-blocking
- [ ] Understand concretely why asynchronous processing is necessary
- [ ] Compare the synchronous/asynchronous models across different languages
- [ ] Learn about typical real-world scenarios and optimal choices

## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Fundamental Concepts of Synchronous vs Asynchronous

### 1.1 Visual Understanding

```
Synchronous Processing:
  Task A ████████████████████
  Task B                     ████████████████████
  Task C                                         ████████████████████
  → Executed in order. The next one waits until the previous one finishes
  → Total time = A + B + C

Asynchronous Processing:
  Task A ████──────────████
  Task B     ████──────────████
  Task C         ████──────────████
  → Other tasks proceed during I/O waits (──)
  → Total time ≒ max(A, B, C)

Concrete example: 3 API calls (200ms each)
  Synchronous:  200 + 200 + 200 = 600ms
  Asynchronous: max(200, 200, 200) = 200ms (3x faster)
```

### 1.2 Understanding Through an Everyday Analogy

The difference between synchronous and asynchronous processing is easy to understand when compared to ordering at a restaurant.

```
Synchronous processing (one waiter fully attending one table at a time):
  Table 1: Take order → Wait for food to be ready → Serve → Bill
  Table 2:                                              Take order → Wait for food to be ready → Serve → Bill
  Table 3:                                                                                           Take order → ...
  → The waiter stands idle even while waiting for food to be prepared
  → Extremely inefficient

Asynchronous processing (one waiter efficiently handling multiple tables):
  Table 1: Take order → (pass to kitchen) → ... → Food ready! Serve
  Table 2:              Take order → (pass to kitchen) → ... → Food ready! Serve
  Table 3:                          Take order → (pass to kitchen) → ...
  → Attends to other tables while waiting for food
  → One waiter can efficiently handle many tables
```

### 1.3 Definitions in Programming

```
Synchronous:
  - The caller waits for the operation to complete before proceeding
  - Order of execution is guaranteed
  - Code flow is linear and easy to understand
  - Results are received directly as function return values

Asynchronous:
  - The caller proceeds without waiting for the operation to complete
  - Results are delivered later (via callbacks, Promises, events, etc.)
  - Order of execution can become non-deterministic
  - More complex, but uses resources more efficiently
```

---

## 2. Blocking vs Non-Blocking

### 2.1 Basic Concepts

```
Blocking I/O:
  → Thread is suspended until I/O completes
  → Thread does not consume CPU, but remains occupied

  Thread1: [Receive request] → [DB query... 100ms wait...] → [Response]
  Thread2: [Receive request] → [API call... 200ms wait...] → [Response]
  Thread3: [Receive request] → [File read... 50ms wait...] → [Response]
  → Concurrent connections are limited by the number of threads

Non-Blocking I/O:
  → Control returns immediately after I/O starts
  → Notification via callback/event upon completion

  Thread1: [Request 1] [Request 2] [Request 3] [DB result handling] [API result handling]
  → A single thread can handle many requests
  → The Node.js model
```

### 2.2 Blocking I/O in Detail

With blocking I/O, the thread is blocked until the OS system call (read, write, connect, etc.) completes.

```typescript
// Blocking I/O conceptual illustration (pseudocode)
function handleRequest(socket: Socket): void {
  // 1. Read request (blocks)
  const request = socket.read(); // ← Thread stops here

  // 2. Query DB (blocks)
  const data = database.query("SELECT * FROM users"); // ← Thread stops here

  // 3. Call external API (blocks)
  const externalData = http.get("https://api.example.com/data"); // ← Thread stops here

  // 4. Write response (blocks)
  socket.write(buildResponse(data, externalData)); // ← Thread stops here
}
```

```java
// Java: Traditional blocking server
import java.net.ServerSocket;
import java.net.Socket;
import java.io.*;

public class BlockingServer {
    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(8080);

        while (true) {
            // accept() blocks until a client connects
            Socket clientSocket = serverSocket.accept();

            // Assign one thread per connection
            new Thread(() -> {
                try {
                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(clientSocket.getInputStream())
                    );
                    PrintWriter writer = new PrintWriter(
                        clientSocket.getOutputStream(), true
                    );

                    // readLine() blocks until data arrives
                    String line = reader.readLine();
                    writer.println("Echo: " + line);

                    clientSocket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }).start();
        }
        // Problem: 10,000 connections = 10,000 threads (1MB each) = 10GB memory
    }
}
```

### 2.3 Non-Blocking I/O in Detail

```typescript
// Node.js: Non-blocking I/O
import * as http from 'http';
import * as fs from 'fs';

const server = http.createServer(async (req, res) => {
  // Non-blocking: control returns immediately after I/O starts
  // Other requests can be processed in the meantime
  try {
    const data = await fs.promises.readFile('data.json', 'utf8');
    const parsed = JSON.parse(data);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(parsed));
  } catch (err) {
    res.writeHead(500);
    res.end('Internal Server Error');
  }
});

server.listen(8080);
// A single thread can handle tens of thousands of concurrent connections
```

### 2.4 Caution: Don't Confuse Blocking and Non-Blocking

"Synchronous" and "blocking," as well as "asynchronous" and "non-blocking," are closely related but are strictly separate concepts.

```
                  Blocking                Non-Blocking
Synchronous       Synchronous Blocking    Synchronous Non-Blocking
                  (typical I/O)           (polling)
Asynchronous      Asynchronous Blocking   Asynchronous Non-Blocking
                  (select/poll)           (epoll/kqueue/IOCP)

Synchronous Blocking:
  → Calling read() suspends the thread until data arrives
  → Simplest but does not scale

Synchronous Non-Blocking (polling):
  → Calling read() immediately returns EWOULDBLOCK if no data is available
  → Application must check repeatedly
  → Prone to wasting CPU time

Asynchronous Non-Blocking:
  → Initiates I/O and returns immediately
  → Receives notification upon completion
  → Most efficient (the Node.js and nginx model)
```

### 2.5 OS-Level I/O Multiplexing

```
Linux:
  select()  → Limited number of monitored fds (1024)
  poll()    → No fd count limit, but scans all fds every time
  epoll()   → Event-driven, highly efficient (Linux 2.6+)

macOS/BSD:
  kqueue()  → Equivalent to epoll, for BSD-based OSes

Windows:
  IOCP (I/O Completion Ports) → Completion port model

Node.js's libuv:
  → Abstracts the optimal mechanism for each OS
  → Linux: epoll, macOS: kqueue, Windows: IOCP
  → File I/O: thread pool (4 threads by default)
  → Network I/O: OS async I/O
```

```c
// How to use epoll (C language, Linux)
#include <sys/epoll.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int main() {
    int epoll_fd = epoll_create1(0);

    struct epoll_event event;
    event.events = EPOLLIN;  // Monitor for readable events
    event.data.fd = socket_fd;

    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, socket_fd, &event);

    struct epoll_event events[MAX_EVENTS];

    while (1) {
        // Wait for events (blocks, but monitors multiple fds simultaneously)
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);

        for (int i = 0; i < nfds; i++) {
            if (events[i].events & EPOLLIN) {
                // Data is readable
                handle_read(events[i].data.fd);
            }
        }
    }
}
```

---

## 3. Why Asynchronous Processing Is Necessary

### 3.1 CPU Cycles vs I/O Wait Times

```
CPU Cycles vs I/O Wait Times (approximate):

  Operation                 Time            CPU Cycle Equivalent
  ─────────────────────────────────────────────────
  L1 Cache                  1ns             1 cycle
  L2 Cache                  4ns             4 cycles
  L3 Cache                  12ns            12 cycles
  Main Memory               100ns           100 cycles
  SSD Random Read           16us            16,000 cycles
  SSD Sequential            50us            50,000 cycles
  HDD Random Read           4ms             4,000,000 cycles
  Network (same DC)         500us           500,000 cycles
  Network (same country)    30ms            30,000,000 cycles
  Network (intercontinental) 150ms          150,000,000 cycles
  TLS Handshake             250ms           250,000,000 cycles

  → During network I/O, the CPU does "nothing" for 150 million cycles
  → Asynchronous processing makes effective use of this wait time

In human time (if 1 CPU cycle = 1 second):
  L1 Cache       → 1 second
  Main Memory    → 1 minute 40 seconds
  SSD Read       → 4.5 hours
  HDD Read       → 46 days
  Network        → 4.8 years (!)
```

### 3.2 Concrete Effect: Web Server Response Time

```typescript
// Synchronous processing (not recommended in Node.js)
function syncHandler(req: Request): Response {
  const user = db.getUserSync(req.userId);      // 10ms wait
  const orders = db.getOrdersSync(user.id);     // 15ms wait
  const recommendations = api.getRecsSync(user); // 50ms wait
  return { user, orders, recommendations };
  // Total: 75ms (sequential execution)
}

// Asynchronous processing (concurrent execution)
async function asyncHandler(req: Request): Promise<Response> {
  const user = await db.getUser(req.userId);    // 10ms
  // After getting user, run the rest concurrently
  const [orders, recommendations] = await Promise.all([
    db.getOrders(user.id),                      // 15ms ┐
    api.getRecs(user),                           // 50ms ┤ concurrent
  ]);                                            //      ┘ max = 50ms
  return { user, orders, recommendations };
  // Total: 10 + 50 = 60ms (20% faster)
}
```

### 3.3 Impact on Throughput

```
Blocking server (thread pool approach):
  Thread count: 200 (Java Tomcat default)
  Average processing time per request: 100ms (of which I/O wait: 80ms)
  Max throughput: 200 / 0.1 = 2,000 req/sec

Non-blocking server (event loop approach):
  Thread count: 1 (Node.js)
  CPU execution time per request: 20ms (I/O wait time is used for other processing)
  Max throughput: 1 / 0.02 = 50 req/sec (when CPU-bound)
  However, there is no limit on concurrent connections
  → Even with 10,000 concurrent connections, memory usage remains low

Actual benchmarks (approximate):
  ┌────────────────────┬──────────────────┬───────────────────┐
  │ Server             │ 1,000 concurrent │ 10,000 concurrent │
  ├────────────────────┼──────────────────┼───────────────────┤
  │ Apache (prefork)   │ 5,000 req/s      │ Out of memory     │
  │ Nginx              │ 20,000 req/s     │ 18,000 req/s      │
  │ Node.js            │ 15,000 req/s     │ 12,000 req/s      │
  │ Go net/http        │ 25,000 req/s     │ 22,000 req/s      │
  └────────────────────┴──────────────────┴───────────────────┘
  * Actual numbers vary significantly depending on workload and hardware
```

### 3.4 The C10K Problem

```
The C10K Problem (proposed by Dan Kegel in 1999):
  → Can a single server handle 10,000 concurrent connections?

Traditional approach (one thread per connection):
  10,000 connections x 1MB/thread = 10GB memory
  → Enormous thread context switching overhead
  → Practically impossible

Solutions:
  1. Event-driven (epoll/kqueue) + non-blocking I/O
     → Nginx, Node.js, HAProxy
  2. Lightweight threads / coroutines
     → Go (goroutine: ~2KB), Erlang (process: ~2KB)
  3. Asynchronous I/O (io_uring, IOCP)
     → Latest Linux kernels (5.1+)

Current challenge: The C10M Problem
  → Handling 10 million connections on a single server
  → Kernel bypass (DPDK, XDP), user-space networking
```

### 3.5 Real-World Effects of Asynchronous Processing

```typescript
// E-commerce product page: sequential version
async function getProductPageSync(productId: string) {
  const start = Date.now();

  const product = await getProduct(productId);           // 20ms
  const reviews = await getReviews(productId);           // 30ms
  const relatedProducts = await getRelated(productId);   // 25ms
  const inventory = await getInventory(productId);       // 15ms
  const pricing = await getPricing(productId);           // 10ms
  const seller = await getSeller(product.sellerId);      // 20ms

  console.log(`Sequential execution: ${Date.now() - start}ms`);
  // → 120ms
  return { product, reviews, relatedProducts, inventory, pricing, seller };
}

// E-commerce product page: optimized version
async function getProductPageOptimized(productId: string) {
  const start = Date.now();

  // Stage 1: Run independent tasks concurrently
  const [product, reviews, relatedProducts, inventory, pricing] =
    await Promise.all([
      getProduct(productId),           // 20ms ┐
      getReviews(productId),           // 30ms ┤
      getRelated(productId),           // 25ms ┤ concurrent
      getInventory(productId),         // 15ms ┤
      getPricing(productId),           // 10ms ┘
    ]);
  // Stage 1: max(20, 30, 25, 15, 10) = 30ms

  // Stage 2: Processing that depends on product
  const seller = await getSeller(product.sellerId); // 20ms

  console.log(`Optimized version: ${Date.now() - start}ms`);
  // → 50ms (58% faster)
  return { product, reviews, relatedProducts, inventory, pricing, seller };
}
```

---

## 4. Asynchronous Models Across Languages

### 4.1 Model Overview

```
┌──────────────┬───────────────────────────────┐
│ Language     │ Asynchronous Model            │
├──────────────┼───────────────────────────────┤
│ JavaScript   │ Event loop + Promise          │
│ Python       │ asyncio (event loop)          │
│ Rust         │ async/await + runtime (tokio) │
│ Go           │ goroutine + channel           │
│ Java         │ Threads + CompletableFuture   │
│ Kotlin       │ coroutines                    │
│ Swift        │ structured concurrency        │
│ Elixir       │ Actor model (BEAM)            │
│ C#           │ Task + async/await            │
│ C++          │ std::async + co_await (C++20) │
└──────────────┴───────────────────────────────┘

Three major approaches:
  1. Event loop (JS, Python): Single-threaded + async I/O
  2. Green threads (Go, Erlang): Many lightweight threads
  3. OS threads + async (Java, C#): Thread pool + Future
```

### 4.2 JavaScript / TypeScript

```typescript
// JavaScript: Single-threaded + event loop
// Shared model for browser and Node.js

// 1. Promise-based
function fetchUserData(userId: string): Promise<User> {
  return fetch(`/api/users/${userId}`)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    });
}

// 2. async/await (syntactic sugar over Promises)
async function fetchUserData(userId: string): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

// 3. Node.js specific: Worker Threads (for CPU-intensive tasks)
import { Worker, isMainThread, parentPort } from 'worker_threads';

if (isMainThread) {
  const worker = new Worker(__filename);
  worker.on('message', (result) => {
    console.log('Computation result:', result);
  });
  worker.postMessage({ data: largeArray });
} else {
  parentPort?.on('message', (msg) => {
    // Execute CPU-intensive processing in a worker thread
    const result = heavyComputation(msg.data);
    parentPort?.postMessage(result);
  });
}
```

### 4.3 Python

```python
import asyncio
import aiohttp

# Python: asyncio event loop
# Due to the GIL (Global Interpreter Lock),
# use multiprocessing for CPU parallelism and asyncio for I/O concurrency

# Basic async function
async def fetch_user(user_id: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as resp:
            return await resp.json()

# Concurrent execution
async def fetch_all_users(user_ids: list[str]) -> list[dict]:
    tasks = [fetch_user(uid) for uid in user_ids]
    return await asyncio.gather(*tasks)

# Execution
async def main():
    users = await fetch_all_users(["user-1", "user-2", "user-3"])
    for user in users:
        print(user["name"])

asyncio.run(main())

# CPU-intensive: multiprocessing
from concurrent.futures import ProcessPoolExecutor
import asyncio

async def cpu_intensive_async(data_list: list) -> list:
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        results = await asyncio.gather(*[
            loop.run_in_executor(pool, heavy_computation, data)
            for data in data_list
        ])
    return results
```

### 4.4 Go

```go
package main

import (
    "fmt"
    "net/http"
    "sync"
    "time"
)

// Go: goroutine + channel
// Goroutines are lightweight threads (~2KB), scheduled by the runtime on top of OS threads

// Basic asynchronous execution
func fetchURL(url string, ch chan<- string, wg *sync.WaitGroup) {
    defer wg.Done()
    resp, err := http.Get(url)
    if err != nil {
        ch <- fmt.Sprintf("Error: %s", err)
        return
    }
    defer resp.Body.Close()
    ch <- fmt.Sprintf("%s: %d", url, resp.StatusCode)
}

func main() {
    urls := []string{
        "https://api.example.com/users",
        "https://api.example.com/orders",
        "https://api.example.com/products",
    }

    ch := make(chan string, len(urls))
    var wg sync.WaitGroup

    for _, url := range urls {
        wg.Add(1)
        go fetchURL(url, ch, &wg) // Concurrent execution with goroutines
    }

    // Wait for all to complete
    go func() {
        wg.Wait()
        close(ch)
    }()

    for result := range ch {
        fmt.Println(result)
    }
}

// Waiting on multiple channels with select
func fetchWithTimeout(url string, timeout time.Duration) (string, error) {
    ch := make(chan string, 1)
    errCh := make(chan error, 1)

    go func() {
        resp, err := http.Get(url)
        if err != nil {
            errCh <- err
            return
        }
        defer resp.Body.Close()
        ch <- resp.Status
    }()

    select {
    case result := <-ch:
        return result, nil
    case err := <-errCh:
        return "", err
    case <-time.After(timeout):
        return "", fmt.Errorf("timeout after %v", timeout)
    }
}
```

### 4.5 Rust

```rust
use tokio;
use reqwest;

// Rust: async/await + runtime (tokio)
// Zero-cost abstraction: async functions compile to state machines
// Futures are lazy: they don't execute until .await is called

async fn fetch_user(user_id: &str) -> Result<User, reqwest::Error> {
    let url = format!("https://api.example.com/users/{}", user_id);
    let user: User = reqwest::get(&url)
        .await?
        .json()
        .await?;
    Ok(user)
}

// Concurrent execution
async fn fetch_all_data(user_id: &str) -> Result<Dashboard, AppError> {
    // Concurrent execution with tokio::join!
    let (user, orders, notifications) = tokio::join!(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_notifications(user_id),
    );

    Ok(Dashboard {
        user: user?,
        orders: orders?,
        notifications: notifications?,
    })
}

// Background tasks with tokio::spawn
async fn background_processing() {
    let handle = tokio::spawn(async {
        // Execute in the background
        heavy_async_work().await
    });

    // Continue with other work
    do_other_work().await;

    // Retrieve background task result
    let result = handle.await.unwrap();
}

#[tokio::main]
async fn main() {
    let dashboard = fetch_all_data("user-123").await.unwrap();
    println!("{:?}", dashboard);
}
```

### 4.6 Java

```java
import java.util.concurrent.*;

// Java: CompletableFuture (Java 8+)
// Virtual Threads (Java 21+ / Project Loom)

public class AsyncExample {

    // CompletableFuture-based
    public CompletableFuture<Dashboard> getDashboard(String userId) {
        CompletableFuture<User> userFuture =
            CompletableFuture.supplyAsync(() -> userRepo.findById(userId));

        CompletableFuture<List<Order>> ordersFuture =
            CompletableFuture.supplyAsync(() -> orderRepo.findByUserId(userId));

        CompletableFuture<List<Notification>> notifFuture =
            CompletableFuture.supplyAsync(() -> notifRepo.findByUserId(userId));

        // Combine when all complete
        return CompletableFuture.allOf(userFuture, ordersFuture, notifFuture)
            .thenApply(v -> new Dashboard(
                userFuture.join(),
                ordersFuture.join(),
                notifFuture.join()
            ));
    }

    // Java 21: Virtual Threads (Project Loom)
    public Dashboard getDashboardVirtualThreads(String userId) throws Exception {
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            var userTask = scope.fork(() -> userRepo.findById(userId));
            var ordersTask = scope.fork(() -> orderRepo.findByUserId(userId));
            var notifTask = scope.fork(() -> notifRepo.findByUserId(userId));

            scope.join();
            scope.throwIfFailed();

            return new Dashboard(
                userTask.get(),
                ordersTask.get(),
                notifTask.get()
            );
        }
    }
}
```

### 4.7 C#

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

// C#: Task + async/await
// .NET's async model is one of the most mature

public class AsyncService
{
    private readonly HttpClient _httpClient;

    // async/await basics
    public async Task<Dashboard> GetDashboardAsync(string userId)
    {
        var user = await GetUserAsync(userId);

        // Concurrent execution
        var ordersTask = GetOrdersAsync(userId);
        var notificationsTask = GetNotificationsAsync(userId);

        await Task.WhenAll(ordersTask, notificationsTask);

        return new Dashboard
        {
            User = user,
            Orders = ordersTask.Result,
            Notifications = notificationsTask.Result
        };
    }

    // Cancellation token support
    public async Task<User> GetUserAsync(
        string userId,
        CancellationToken cancellationToken = default)
    {
        var response = await _httpClient.GetAsync(
            $"/api/users/{userId}",
            cancellationToken
        );
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<User>(
            cancellationToken: cancellationToken
        );
    }

    // ValueTask: hot path optimization
    public ValueTask<CachedData> GetCachedDataAsync(string key)
    {
        if (_cache.TryGetValue(key, out var cached))
        {
            // No heap allocation on cache hit
            return new ValueTask<CachedData>(cached);
        }

        // Async processing only on cache miss
        return new ValueTask<CachedData>(FetchAndCacheAsync(key));
    }
}
```

---

## 5. Choosing Between Synchronous and Asynchronous

### 5.1 Decision Criteria

```
Synchronous is appropriate:
  + CPU-intensive computation (numerical calculations, encryption, image processing)
  + Simple scripts and batch processing
  + Processing with little I/O
  + Processing that requires sequential execution (order guarantee needed)
  + When debuggability is important
  + Short-lived operations

Asynchronous is appropriate:
  + Network I/O (API calls, DB connections)
  + File I/O (large-scale file operations)
  + Servers handling many concurrent connections
  + Client apps where blocking the UI is undesirable
  + Real-time processing (WebSocket, chat)
  + Inter-microservice communication

Caveats:
  → Making CPU-intensive processing async is pointless
  → Don't block the event loop (a cardinal rule in Node.js)
  → Consider the overhead of async (context switching, memory)
```

### 5.2 Scenario-Based Guide

```typescript
// Scenario 1: File processing
// Good: Process many files concurrently with async
async function processFiles(filePaths: string[]): Promise<void> {
  const CONCURRENCY = 10; // Up to 10 files simultaneously
  const results: string[] = [];

  for (let i = 0; i < filePaths.length; i += CONCURRENCY) {
    const batch = filePaths.slice(i, i + CONCURRENCY);
    const batchResults = await Promise.all(
      batch.map(async (filePath) => {
        const content = await fs.promises.readFile(filePath, 'utf8');
        return processContent(content);
      })
    );
    results.push(...batchResults);
  }
}

// Bad: For a single small file, synchronous is fine
// (startup scripts, config loading, etc.)
const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));
```

```python
# Scenario 2: Web scraping
import asyncio
import aiohttp
from typing import List, Dict

# Good: Fetch many URLs concurrently with async
async def scrape_urls(urls: list[str]) -> list[dict]:
    semaphore = asyncio.Semaphore(20)  # Limit concurrent connections

    async def fetch_one(session: aiohttp.ClientSession, url: str) -> dict:
        async with semaphore:
            async with session.get(url) as response:
                html = await response.text()
                return {"url": url, "status": response.status, "html": html}

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# 100 URLs with 20 concurrent connections
# Synchronous: 100 x 200ms = 20 seconds
# Asynchronous: 100 / 20 x 200ms = 1 second (20x faster)
```

```go
// Scenario 3: Microservice API Gateway
package main

import (
    "context"
    "net/http"
    "time"
    "encoding/json"
    "golang.org/x/sync/errgroup"
)

type AggregatedResponse struct {
    User          *User          `json:"user"`
    Orders        []Order        `json:"orders"`
    Notifications []Notification `json:"notifications"`
}

// Good: Call multiple microservices concurrently
func aggregateHandler(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
    defer cancel()

    userID := r.URL.Query().Get("user_id")
    var resp AggregatedResponse

    g, ctx := errgroup.WithContext(ctx)

    g.Go(func() error {
        user, err := fetchUser(ctx, userID)
        if err != nil {
            return err
        }
        resp.User = user
        return nil
    })

    g.Go(func() error {
        orders, err := fetchOrders(ctx, userID)
        if err != nil {
            return err
        }
        resp.Orders = orders
        return nil
    })

    g.Go(func() error {
        notifs, err := fetchNotifications(ctx, userID)
        if err != nil {
            return err
        }
        resp.Notifications = notifs
        return nil
    })

    if err := g.Wait(); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    json.NewEncoder(w).Encode(resp)
}
```

### 5.3 Anti-Patterns

```typescript
// Bad: Anti-pattern 1 - Running CPU-intensive work on the event loop
async function badImageProcessing(images: Buffer[]): Promise<Buffer[]> {
  // This blocks the event loop
  return images.map(img => {
    // Heavy image processing (synchronously occupies CPU)
    return sharp(img).resize(800, 600).toBuffer(); // ← Synchronous API
  });
}

// Good: Delegate CPU-intensive work to Worker Threads
import { Worker } from 'worker_threads';

async function goodImageProcessing(images: Buffer[]): Promise<Buffer[]> {
  const worker = new Worker('./image-worker.js');
  return new Promise((resolve, reject) => {
    worker.postMessage(images);
    worker.on('message', resolve);
    worker.on('error', reject);
  });
}

// Bad: Anti-pattern 2 - Unnecessary async
async function unnecessary(): Promise<number> {
  return 1 + 1; // ← No point making synchronous-sufficient code async
}

// Bad: Anti-pattern 3 - Ignoring async results
function fireAndForget(data: Data): void {
  saveToDatabase(data); // Promise result ignored → errors become invisible
}

// Good: Handle results properly
async function properSave(data: Data): Promise<void> {
  try {
    await saveToDatabase(data);
  } catch (error) {
    logger.error('Failed to save data', error);
    throw error; // Propagate to caller
  }
}
```

---

## 6. Patterns Frequently Encountered in Practice

### 6.1 Async Operations with Timeout

```typescript
// Fetch with timeout
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = 5000,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeoutMs}ms: ${url}`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// Usage
try {
  const response = await fetchWithTimeout('https://api.example.com/data', {}, 3000);
  const data = await response.json();
} catch (error) {
  console.error('Request failed:', error.message);
}
```

### 6.2 Async Operations with Retry

```typescript
// Retry with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    initialDelayMs?: number;
    maxDelayMs?: number;
    backoffMultiplier?: number;
    retryableErrors?: (error: unknown) => boolean;
  } = {},
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelayMs = 1000,
    maxDelayMs = 30000,
    backoffMultiplier = 2,
    retryableErrors = () => true,
  } = options;

  let lastError: unknown;
  let delay = initialDelayMs;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxRetries || !retryableErrors(error)) {
        throw error;
      }

      // Add jitter (random variation)
      const jitter = delay * 0.1 * Math.random();
      const actualDelay = Math.min(delay + jitter, maxDelayMs);

      console.warn(
        `Attempt ${attempt + 1} failed, retrying in ${actualDelay}ms...`,
        error
      );

      await new Promise(resolve => setTimeout(resolve, actualDelay));
      delay *= backoffMultiplier;
    }
  }

  throw lastError;
}

// Usage
const data = await retryWithBackoff(
  () => fetchWithTimeout('https://api.example.com/data'),
  {
    maxRetries: 3,
    initialDelayMs: 1000,
    retryableErrors: (error) => {
      // Retry only on 5xx errors
      return error instanceof Error && error.message.includes('5');
    },
  }
);
```

### 6.3 Concurrency Limiting (Semaphore Pattern)

```typescript
// Semaphore: limits the number of concurrent executions
class Semaphore {
  private permits: number;
  private queue: (() => void)[] = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
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
      this.permits++;
    }
  }

  async use<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

// Usage: API calls with 5 concurrent connections
const semaphore = new Semaphore(5);
const urls = Array.from({ length: 100 }, (_, i) => `https://api.example.com/item/${i}`);

const results = await Promise.all(
  urls.map(url =>
    semaphore.use(async () => {
      const response = await fetch(url);
      return response.json();
    })
  )
);
```

### 6.4 Cancellable Async Operations

```typescript
// Cancellation using AbortController
class CancellableTask<T> {
  private controller: AbortController;
  private promise: Promise<T>;

  constructor(executor: (signal: AbortSignal) => Promise<T>) {
    this.controller = new AbortController();
    this.promise = executor(this.controller.signal);
  }

  get result(): Promise<T> {
    return this.promise;
  }

  cancel(reason?: string): void {
    this.controller.abort(reason);
  }
}

// Usage: Auto-cancel search
let currentSearch: CancellableTask<SearchResult[]> | null = null;

async function search(query: string): Promise<SearchResult[]> {
  // Cancel previous search
  currentSearch?.cancel('New search started');

  currentSearch = new CancellableTask(async (signal) => {
    const response = await fetch(`/api/search?q=${query}`, { signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  });

  return currentSearch.result;
}
```

---

## 7. Performance Measurement and Optimization

### 7.1 Benchmarking Async Operations

```typescript
// Measuring processing time
async function benchmark<T>(
  name: string,
  fn: () => Promise<T>,
  iterations: number = 10,
): Promise<{ name: string; avg: number; min: number; max: number; p95: number }> {
  const times: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }

  times.sort((a, b) => a - b);

  return {
    name,
    avg: times.reduce((sum, t) => sum + t, 0) / times.length,
    min: times[0],
    max: times[times.length - 1],
    p95: times[Math.floor(times.length * 0.95)],
  };
}

// Comparison test
async function compareSyncVsAsync(): Promise<void> {
  const syncResult = await benchmark('Sequential', async () => {
    const a = await fetchA();
    const b = await fetchB();
    const c = await fetchC();
    return { a, b, c };
  });

  const asyncResult = await benchmark('Parallel', async () => {
    const [a, b, c] = await Promise.all([
      fetchA(),
      fetchB(),
      fetchC(),
    ]);
    return { a, b, c };
  });

  console.table([syncResult, asyncResult]);
  // ┌─────────────┬──────────┬──────────┬──────────┬──────────┐
  // │ name        │ avg      │ min      │ max      │ p95      │
  // ├─────────────┼──────────┼──────────┼──────────┼──────────┤
  // │ Sequential  │ 312.5ms  │ 301.2ms  │ 325.8ms  │ 321.3ms  │
  // │ Parallel    │ 105.3ms  │ 100.1ms  │ 115.2ms  │ 112.7ms  │
  // └─────────────┴──────────┴──────────┴──────────┴──────────┘
}
```

### 7.2 Common Bottlenecks and Countermeasures

```
Bottleneck 1: Insufficient DB connection pool
  Symptom: DB connection wait under high concurrent requests
  Solution: Set pool size appropriately (CPU cores x 2 + number of disks)

Bottleneck 2: External API rate limiting
  Symptom: 429 Too Many Requests
  Solution: Limit concurrency with semaphore, use rate-limiting libraries

Bottleneck 3: Memory leaks (Promise accumulation)
  Symptom: Heap memory continuously increasing
  Solution: Release unnecessary Promise references, use WeakRef

Bottleneck 4: Event loop blocking
  Symptom: Sudden spikes in response time
  Solution: Move CPU work to Worker Threads, detect with tools like blocked-at

Bottleneck 5: DNS resolution delay
  Symptom: Only the first request is slow
  Solution: DNS prefetch, use keep-alive connections
```

---

## 8. Testing Async Operations

### 8.1 Basic Test Patterns

```typescript
import { describe, it, expect, vi } from 'vitest';

// Testing async functions
describe('fetchUserData', () => {
  // Basic test
  it('successfully retrieves user data', async () => {
    const user = await fetchUserData('user-123');
    expect(user).toEqual({
      id: 'user-123',
      name: 'Test User',
    });
  });

  // Error test
  it('throws an error for non-existent user', async () => {
    await expect(fetchUserData('nonexistent'))
      .rejects.toThrow('User not found');
  });

  // Timeout test
  it('throws an error on timeout', async () => {
    vi.useFakeTimers();

    const promise = fetchWithTimeout('https://slow.api.com', {}, 3000);

    vi.advanceTimersByTime(3000);

    await expect(promise).rejects.toThrow('timeout');

    vi.useRealTimers();
  });

  // Using mocks
  it('tests with mocked API', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ id: '123', name: 'Test' }),
    });

    global.fetch = mockFetch;

    const result = await fetchUserData('123');
    expect(result.name).toBe('Test');
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/users/123')
    );
  });
});
```

### 8.2 Testing Concurrent Processing

```typescript
describe('Promise.all pattern tests', () => {
  it('confirms order independence of concurrent execution', async () => {
    const results: string[] = [];

    const task1 = async () => {
      await sleep(100);
      results.push('task1');
      return 'result1';
    };

    const task2 = async () => {
      await sleep(50);
      results.push('task2');
      return 'result2';
    };

    const [r1, r2] = await Promise.all([task1(), task2()]);

    expect(r1).toBe('result1');
    expect(r2).toBe('result2');
    // task2 completes first, but the result order is preserved
    expect(results).toEqual(['task2', 'task1']);
  });

  it('handles partial failures', async () => {
    const results = await Promise.allSettled([
      Promise.resolve('success'),
      Promise.reject(new Error('failure')),
      Promise.resolve('success2'),
    ]);

    expect(results[0]).toEqual({ status: 'fulfilled', value: 'success' });
    expect(results[1].status).toBe('rejected');
    expect(results[2]).toEqual({ status: 'fulfilled', value: 'success2' });
  });
});
```

---

## 9. Debugging Techniques

### 9.1 Debugging Async Operations

```typescript
// Tracing async operations using async_hooks (Node.js)
import { AsyncLocalStorage } from 'async_hooks';

const requestStorage = new AsyncLocalStorage<{ requestId: string }>();

// Propagate request ID as context
async function handleRequest(req: Request): Promise<Response> {
  const requestId = generateRequestId();

  return requestStorage.run({ requestId }, async () => {
    logger.info(`[${requestId}] Request started`);

    const user = await getUser(req.userId);
    logger.info(`[${requestId}] User retrieved`);

    const data = await processData(user);
    logger.info(`[${requestId}] Data processing complete`);

    return new Response(JSON.stringify(data));
  });
}

// Request ID can be retrieved from any async operation
function getRequestId(): string {
  return requestStorage.getStore()?.requestId ?? 'unknown';
}
```

### 9.2 Detecting Unhandled Rejections

```typescript
// Node.js: Detect unhandled Promise Rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Promise Rejection:', reason);
  console.error('Promise:', promise);
  // In production, log and send alerts
  logger.error('Unhandled Promise Rejection', {
    reason: reason instanceof Error ? reason.message : String(reason),
    stack: reason instanceof Error ? reason.stack : undefined,
  });
});

// Browser
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled Promise Rejection:', event.reason);
  event.preventDefault(); // Suppress default console error
  // Report to error tracking service
  errorTracker.captureException(event.reason);
});
```

### 9.3 Profiling Async Operations

```typescript
// Visualizing async operation performance
class AsyncProfiler {
  private traces: Map<string, { start: number; end?: number }[]> = new Map();

  wrap<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const entry = { start: performance.now() };

    if (!this.traces.has(name)) {
      this.traces.set(name, []);
    }
    this.traces.get(name)!.push(entry);

    return fn().finally(() => {
      entry.end = performance.now();
    });
  }

  report(): void {
    console.log('\n=== Async Performance Report ===');
    for (const [name, entries] of this.traces) {
      const durations = entries
        .filter(e => e.end !== undefined)
        .map(e => e.end! - e.start);
      const avg = durations.reduce((s, d) => s + d, 0) / durations.length;
      const max = Math.max(...durations);
      console.log(`${name}: calls=${entries.length}, avg=${avg.toFixed(1)}ms, max=${max.toFixed(1)}ms`);
    }
  }
}

// Usage
const profiler = new AsyncProfiler();

const user = await profiler.wrap('getUser', () => getUser(userId));
const [orders, reviews] = await Promise.all([
  profiler.wrap('getOrders', () => getOrders(user.id)),
  profiler.wrap('getReviews', () => getReviews(user.id)),
]);

profiler.report();
// === Async Performance Report ===
// getUser: calls=1, avg=12.3ms, max=12.3ms
// getOrders: calls=1, avg=18.7ms, max=18.7ms
// getReviews: calls=1, avg=45.2ms, max=45.2ms
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Write test code as well

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
        """Main data processing logic"""
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
        assert False, "Should have raised an exception"
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
        """Add item (with size limit)"""
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
        """Get statistics"""
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
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing and running code to see how things work.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend solidly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Synchronous | Asynchronous |
|---------|-------------|--------------|
| Execution | Waits in order | Processes other work during wait times |
| I/O | Blocking | Non-blocking |
| Performance | Wastes time on I/O waits | Makes effective use of I/O waits |
| Complexity | Simple | Callbacks/Promises |
| Best for | CPU-intensive | I/O-intensive |
| Scalability | Limited by thread count | Handles many concurrent connections |
| Debugging | Easy (linear stack traces) | Difficult (async stack traces) |
| Memory | ~1MB per thread | ~2KB per event/goroutine |

### Decision Flowchart

```
What type of processing?
├── CPU-intensive (computation, encryption, image processing)
│   ├── Single task → Synchronous
│   └── Parallel computation needed → Worker Threads / multiprocessing
├── I/O-intensive (API, DB, files)
│   ├── Single operation → async/await
│   ├── Multiple independent I/O → Promise.all / gather / join!
│   └── Streams → Observable / AsyncIterator / Channel
└── Mixed
    ├── I/O → Asynchronous
    └── CPU → Delegate to workers
```

---

## Recommended Next Reads

---

## References
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Node.js Documentation. "Don't Block the Event Loop."
3. Kegel, D. "The C10K Problem." 1999. http://www.kegel.com/c10k.html
4. Pike, R. "Concurrency Is Not Parallelism." Waza Conference, 2012.
5. Mozilla Developer Network. "Asynchronous JavaScript." MDN Web Docs.
6. Python Documentation. "asyncio - Asynchronous I/O." docs.python.org.
7. Tokio Documentation. "Tutorial." tokio.rs.
8. Microsoft. "Asynchronous programming with async and await." docs.microsoft.com.
9. OpenJDK. "JEP 444: Virtual Threads." openjdk.org.
10. Nginx Documentation. "Inside NGINX: How We Designed for Performance & Scale."
