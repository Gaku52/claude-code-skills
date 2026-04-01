# Cancellation

> Cancellation of asynchronous operations is often overlooked, but it is a critical technique that directly impacts UX and resource management. This guide covers AbortController, timeouts, and cancellation token implementations.

## Learning Objectives

- [ ] Understand scenarios where cancellation of async operations is necessary
- [ ] Learn how to use AbortController
- [ ] Learn timeout pattern implementations
- [ ] Compare cancellation mechanisms across languages
- [ ] Master production-level cancellation design
- [ ] Learn how to implement testable cancellation logic


## Prerequisites

Understanding the following will help you get the most out of this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with [Event Loop](./00-event-loop.md)

---

## 1. Why Cancellation Is Necessary

```
Scenarios where cancellation is needed:
  1. User navigates to a different page -> Cancel the previous page's API request
  2. Search box input -> Cancel the previous search request
  3. Timeout -> Cancel if no response within a certain time
  4. Component unmount -> Cancel ongoing operations
  5. User clicks the "Cancel" button
  6. Cleanup of in-progress operations during server shutdown
  7. Cancel operations when resource limits are reached
  8. Cancel one of competing operations (race pattern)

Without cancellation:
  -> Unnecessary network requests remain
  -> Memory leaks (setState after component destruction)
  -> Race conditions (old results overwrite new results)
  -> Wasted server resources
  -> Degraded user experience (displaying stale data)
  -> DB connection pool exhaustion
```

### 1.1 Types of Cancellation

```
Types of cancellation:

1. User-initiated cancellation
   -> Cancel button press
   -> Page navigation
   -> Component unmount
   -> Closing a browser tab

2. System-initiated cancellation
   -> Timeout
   -> Server shutdown
   -> Resource limit reached
   -> Cancellation propagation from parent task

3. Logic-initiated cancellation
   -> New request cancels the previous request
   -> Cancel others once the first result is obtained
   -> Cancel processing when conditions change

Cancellation levels:
  +----------------------------------------+
  | Application level                      |
  |  -> UI events, routing                 |
  |                                        |
  |  +------------------------------------+|
  |  | Service level                      ||
  |  |  -> API calls, batch processing    ||
  |  |                                    ||
  |  |  +--------------------------------+||
  |  |  | Resource level                 |||
  |  |  |  -> DB connections, file handles|||
  |  |  +--------------------------------+||
  |  +------------------------------------+|
  +----------------------------------------+
```

---

## 2. AbortController (Web Standard)

### 2.1 Basic Usage

```typescript
// Cancelling a fetch request
const controller = new AbortController();
const { signal } = controller;

// Start request
const promise = fetch('/api/data', { signal })
  .then(res => res.json())
  .catch(err => {
    if (err.name === 'AbortError') {
      console.log('Request was cancelled');
    } else {
      throw err;
    }
  });

// Cancel after 3 seconds
setTimeout(() => controller.abort(), 3000);

// Specify cancellation reason with AbortSignal.reason (2022+ spec)
controller.abort(new Error('User navigated away'));
controller.abort('timeout'); // Strings are also valid

// Get cancellation reason with signal.reason
signal.addEventListener('abort', () => {
  console.log('Abort reason:', signal.reason);
});
```

### 2.2 Usage Pattern in React

```typescript
// Usage in React
function SearchResults({ query }: { query: string }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetch(`/api/search?q=${query}`, { signal: controller.signal })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setResults(data);
        setLoading(false);
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          setError(err.message);
          setLoading(false);
        }
        // Ignore AbortError (component destroyed or query changed)
      });

    // Cleanup: Cancel on component destruction or query change
    return () => controller.abort();
  }, [query]);

  return (
    <div>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      <ul>{results.map(r => <li key={r.id}>{r.name}</li>)}</ul>
    </div>
  );
}
```

### 2.3 Advanced AbortSignal Usage

```typescript
// AbortSignal.timeout(): Signal with timeout
const response = await fetch('/api/data', {
  signal: AbortSignal.timeout(5000), // 5-second timeout
});

// AbortSignal.any(): Combine multiple signals (2023+)
const userCancel = new AbortController();
const timeoutSignal = AbortSignal.timeout(30000);
const shutdownSignal = getShutdownSignal();

const combinedSignal = AbortSignal.any([
  userCancel.signal,
  timeoutSignal,
  shutdownSignal,
]);

fetch('/api/data', { signal: combinedSignal });
// -> Cancelled by user cancellation, timeout, or shutdown — whichever comes first

// Passing AbortSignal to custom APIs
class DataFetcher {
  async fetchWithRetry(
    url: string,
    options: { signal?: AbortSignal; maxRetries?: number } = {},
  ): Promise<Response> {
    const { signal, maxRetries = 3 } = options;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      // Check for cancellation
      signal?.throwIfAborted();

      try {
        const response = await fetch(url, { signal });
        if (response.ok) return response;

        if (response.status >= 500 && attempt < maxRetries) {
          // Retry on server errors
          await this.delay(Math.pow(2, attempt) * 1000, signal);
          continue;
        }

        throw new Error(`HTTP ${response.status}`);
      } catch (err) {
        if ((err as Error).name === 'AbortError') throw err;
        if (attempt === maxRetries) throw err;
      }
    }

    throw new Error('Max retries exceeded');
  }

  private delay(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(signal.reason);
        return;
      }

      const timer = setTimeout(resolve, ms);

      signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(signal.reason);
      }, { once: true });
    });
  }
}
```

### 2.4 AbortController in Node.js

```typescript
import { readFile, writeFile } from 'fs/promises';
import { createReadStream } from 'fs';
import { pipeline } from 'stream/promises';
import { setTimeout as sleep } from 'timers/promises';

// Cancelling fs/promises operations
const controller = new AbortController();
const { signal } = controller;

// Cancel file reading
try {
  const data = await readFile('large-file.txt', { signal });
} catch (err) {
  if ((err as NodeJS.ErrnoException).code === 'ABORT_ERR') {
    console.log('File read cancelled');
  }
}

// Cancelling timers/promises
try {
  await sleep(60000, null, { signal }); // 60-second wait
} catch (err) {
  // Resolves immediately when signal is aborted
}

// Cancelling streams
const readStream = createReadStream('data.csv', { signal });
const writeStream = createWriteStream('output.csv', { signal });

try {
  await pipeline(readStream, transformStream, writeStream, { signal });
} catch (err) {
  if ((err as Error).name === 'AbortError') {
    console.log('Pipeline cancelled');
  }
}

// Cancelling EventEmitter
import { once } from 'events';

const controller2 = new AbortController();
try {
  const [data] = await once(emitter, 'data', { signal: controller2.signal });
} catch (err) {
  // Wait is cancelled when signal is aborted
}

// Detecting request cancellation in HTTP server
import http from 'http';

const server = http.createServer(async (req, res) => {
  const controller = new AbortController();

  // Cancel when client disconnects
  req.on('close', () => {
    if (!res.writableEnded) {
      controller.abort();
    }
  });

  try {
    const data = await fetchExpensiveData({ signal: controller.signal });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      // Client disconnected, no response needed
      return;
    }
    res.writeHead(500);
    res.end('Internal Server Error');
  }
});
```

---

## 3. Timeout Patterns

### 3.1 Basic Timeout

```typescript
// Fetch with timeout
async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeoutMs?: number } = {},
): Promise<Response> {
  const { timeoutMs = 5000, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if ((error as Error).name === 'AbortError') {
      throw new TimeoutError(`Request timed out after ${timeoutMs}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// Custom error class
class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}
```

### 3.2 Promise Timeout Wrapper

```typescript
// Promise timeout wrapper
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new TimeoutError(`Timeout after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]);
}

// Usage
const data = await withTimeout(fetchData(), 5000);

// Cancellable timeout (prevents resource leaks)
function withCancellableTimeout<T>(
  promise: Promise<T>,
  ms: number,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;

    const timeoutId = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new TimeoutError(`Timeout after ${ms}ms`));
      }
    }, ms);

    const cleanup = () => {
      clearTimeout(timeoutId);
    };

    // AbortSignal cancellation
    signal?.addEventListener('abort', () => {
      if (!settled) {
        settled = true;
        cleanup();
        reject(signal.reason);
      }
    }, { once: true });

    promise.then(
      value => {
        if (!settled) {
          settled = true;
          cleanup();
          resolve(value);
        }
      },
      error => {
        if (!settled) {
          settled = true;
          cleanup();
          reject(error);
        }
      },
    );
  });
}
```

### 3.3 Gradual Timeout

```typescript
// Gradual timeout: Warning -> Timeout -> Force terminate
class GradualTimeout {
  private timers: NodeJS.Timeout[] = [];

  constructor(
    private warningMs: number,
    private timeoutMs: number,
    private forceMs: number,
  ) {}

  async execute<T>(
    fn: (signal: AbortSignal) => Promise<T>,
    callbacks: {
      onWarning?: () => void;
      onTimeout?: () => void;
      onForce?: () => void;
    } = {},
  ): Promise<T> {
    const controller = new AbortController();

    // Stage 1: Warning
    this.timers.push(
      setTimeout(() => {
        callbacks.onWarning?.();
        console.warn(`Operation running for ${this.warningMs}ms`);
      }, this.warningMs),
    );

    // Stage 2: Timeout (cooperative cancellation)
    this.timers.push(
      setTimeout(() => {
        callbacks.onTimeout?.();
        controller.abort(new TimeoutError(`Timeout after ${this.timeoutMs}ms`));
      }, this.timeoutMs),
    );

    // Stage 3: Force terminate
    this.timers.push(
      setTimeout(() => {
        callbacks.onForce?.();
        console.error('Force terminating operation');
        // Force termination logic
      }, this.forceMs),
    );

    try {
      return await fn(controller.signal);
    } finally {
      this.timers.forEach(clearTimeout);
      this.timers = [];
    }
  }
}

// Usage example
const gradual = new GradualTimeout(5000, 10000, 30000);

const result = await gradual.execute(
  async (signal) => {
    return await fetchLargeDataset(signal);
  },
  {
    onWarning: () => showSlowOperationBanner(),
    onTimeout: () => logSlowOperation(),
    onForce: () => alertOpsTeam(),
  },
);
```

---

## 4. Cancellation Mechanisms Across Languages

### 4.1 Python Cancellation

```python
import asyncio

async def cancellable_task():
    try:
        while True:
            data = await fetch_data()
            process(data)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        # Cleanup processing
        print("Task was cancelled")
        await cleanup_resources()
        raise  # Re-raise (propagate the cancellation)

async def main():
    task = asyncio.create_task(cancellable_task())

    await asyncio.sleep(5)
    task.cancel()  # Cancel

    try:
        await task
    except asyncio.CancelledError:
        print("Task was successfully cancelled")

# Timeout
async def with_timeout():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=5.0)
    except asyncio.TimeoutError:
        print("Timeout")

# TaskGroup (Python 3.11+)
async def parallel_with_cancellation():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(operation_a())
        task2 = tg.create_task(operation_b())
        # If one raises an error, all other tasks are cancelled

# Shield (prevent cancellation propagation)
async def critical_operation():
    # Wrapping with shield() prevents external cancellation from propagating
    result = await asyncio.shield(important_db_write())
    return result

# Structured concurrency (Python 3.12+, anyio)
import anyio

async def structured_cancellation():
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker, "task1")
        tg.start_soon(worker, "task2")
        # All tasks are cancelled when leaving the scope

    # cancel_scope for timeout
    with anyio.move_on_after(5.0) as scope:
        await slow_operation()
    if scope.cancelled_caught:
        print("Timed out")
```

### 4.2 Go Cancellation (Context)

```go
package main

import (
    "context"
    "fmt"
    "net/http"
    "time"
)

// context.WithCancel: Manual cancellation
func manualCancel() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel() // Always call (prevents resource leaks)

    go func() {
        select {
        case <-ctx.Done():
            fmt.Println("Cancelled:", ctx.Err())
            return
        case result := <-doWork():
            fmt.Println("Result:", result)
        }
    }()

    time.Sleep(2 * time.Second)
    cancel() // Cancel
}

// context.WithTimeout: Timeout
func withTimeout() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    req, _ := http.NewRequestWithContext(ctx, "GET", "https://api.example.com/data", nil)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            fmt.Println("Timeout")
        }
        return
    }
    defer resp.Body.Close()
}

// context.WithDeadline: Deadline
func withDeadline() {
    deadline := time.Now().Add(10 * time.Second)
    ctx, cancel := context.WithDeadline(context.Background(), deadline)
    defer cancel()

    // Propagate context to DB query
    rows, err := db.QueryContext(ctx, "SELECT * FROM users")
    // ...
}

// context.WithCancelCause (Go 1.20+): Cancellation reason
func withCancelCause() {
    ctx, cancel := context.WithCancelCause(context.Background())

    go func() {
        // Cancel with a cause error attached
        cancel(fmt.Errorf("user interrupted"))
    }()

    <-ctx.Done()
    fmt.Println("Reason:", context.Cause(ctx))
}

// Context propagation in HTTP handlers
func handler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context() // Automatically cancelled on client disconnect

    // Add timeout to child context
    ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()

    // Pass context to all downstream calls
    data, err := fetchData(ctx)
    if err != nil {
        if ctx.Err() == context.Canceled {
            // Client disconnected
            return
        }
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    json.NewEncoder(w).Encode(data)
}

// Cancellation handling in goroutines
func worker(ctx context.Context, jobs <-chan Job) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case job, ok := <-jobs:
            if !ok {
                return nil // Channel was closed
            }
            if err := processJob(ctx, job); err != nil {
                return fmt.Errorf("job %s failed: %w", job.ID, err)
            }
        }
    }
}
```

### 4.3 C# Cancellation (CancellationToken)

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

// Create a token with CancellationTokenSource
public class DataService
{
    public async Task<Data> FetchDataAsync(CancellationToken cancellationToken = default)
    {
        // Periodically check for cancellation
        cancellationToken.ThrowIfCancellationRequested();

        using var httpClient = new HttpClient();
        var response = await httpClient.GetAsync(
            "https://api.example.com/data",
            cancellationToken
        );

        return await response.Content.ReadFromJsonAsync<Data>(
            cancellationToken: cancellationToken
        );
    }
}

// Usage example
public async Task Main()
{
    // Manual cancellation
    using var cts = new CancellationTokenSource();

    var task = service.FetchDataAsync(cts.Token);

    // Cancel after 5 seconds
    cts.CancelAfter(TimeSpan.FromSeconds(5));

    try
    {
        var data = await task;
    }
    catch (OperationCanceledException)
    {
        Console.WriteLine("Operation was cancelled");
    }

    // LinkedToken: Combine multiple cancellation sources
    using var userCts = new CancellationTokenSource();
    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
        userCts.Token,
        timeoutCts.Token
    );

    await service.FetchDataAsync(linkedCts.Token);
}
```

### 4.4 Rust Cancellation

```rust
use tokio::select;
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration};
use tokio_util::sync::CancellationToken;

// Cancellation with tokio::select!
async fn cancellable_operation(cancel_rx: oneshot::Receiver<()>) -> Result<Data, Error> {
    select! {
        result = fetch_data() => result,
        _ = cancel_rx => {
            println!("Operation cancelled");
            Err(Error::Cancelled)
        }
    }
}

// CancellationToken (tokio-util)
async fn with_cancellation_token() {
    let token = CancellationToken::new();
    let child_token = token.child_token();

    // Worker task
    let handle = tokio::spawn(async move {
        loop {
            select! {
                _ = child_token.cancelled() => {
                    println!("Worker cancelled");
                    break;
                }
                _ = do_work() => {}
            }
        }
    });

    // Cancel after 5 seconds
    tokio::time::sleep(Duration::from_secs(5)).await;
    token.cancel();

    handle.await.unwrap();
}

// Automatic cancellation via Drop trait
struct AutoCancelGuard {
    token: CancellationToken,
}

impl Drop for AutoCancelGuard {
    fn drop(&mut self) {
        self.token.cancel();
        // Automatically cancels when leaving scope
    }
}

// Timeout
async fn with_timeout() -> Result<Data, Error> {
    match timeout(Duration::from_secs(5), fetch_data()).await {
        Ok(result) => result,
        Err(_) => Err(Error::Timeout),
    }
}
```

---

## 5. Cancellation Design Principles

```
1. Cancellation is cooperative
   -> Rather than forcibly stopping a process,
     it notifies that "cancellation has been requested"
   -> The process cleans up before stopping

2. Guarantee cleanup
   -> Release resources in finally blocks
   -> Roll back DB transactions
   -> Delete temporary files
   -> Release locks

3. Design cancellable APIs
   -> Accept AbortSignal / CancellationToken / Context as parameters
   -> Document behavior on cancellation
   -> Clarify handling of partial results

4. Watch out for race conditions
   -> Cancellation and completion can occur simultaneously
   -> Perform state checks appropriately
   -> Handle the "already cancelled" state

5. Cancellation propagation
   -> Parent cancellation should propagate to children
   -> Child cancellation should not propagate to parent (normally)
   -> Design cancellation to propagate in a tree structure
```

### 5.1 Cancellable API Design

```typescript
// === Good API Design ===

// 1. Accept signal as part of options
interface FetchOptions {
  signal?: AbortSignal;
  timeout?: number;
  retries?: number;
}

async function fetchData(url: string, options: FetchOptions = {}): Promise<Data> {
  const { signal, timeout = 30000, retries = 3 } = options;

  // Combine external signal with timeout
  const timeoutSignal = AbortSignal.timeout(timeout);
  const combinedSignal = signal
    ? AbortSignal.any([signal, timeoutSignal])
    : timeoutSignal;

  for (let attempt = 0; attempt <= retries; attempt++) {
    combinedSignal.throwIfAborted();

    try {
      const response = await fetch(url, { signal: combinedSignal });
      if (response.ok) return await response.json();
    } catch (err) {
      if ((err as Error).name === 'AbortError') throw err;
      if (attempt === retries) throw err;
      await delay(Math.pow(2, attempt) * 1000, combinedSignal);
    }
  }

  throw new Error('Unreachable');
}

// 2. Cancellable iterator
async function* paginatedFetch<T>(
  baseUrl: string,
  signal?: AbortSignal,
): AsyncGenerator<T[]> {
  let page = 1;
  let hasMore = true;

  while (hasMore) {
    signal?.throwIfAborted();

    const response = await fetch(`${baseUrl}?page=${page}`, { signal });
    const data = await response.json();

    yield data.items;

    hasMore = data.hasMore;
    page++;
  }
}

// Usage example
const controller = new AbortController();

for await (const items of paginatedFetch<User>('/api/users', controller.signal)) {
  for (const user of items) {
    processUser(user);
  }
}

// 3. Cancellable batch processing
class BatchProcessor<T, R> {
  async process(
    items: T[],
    processor: (item: T, signal: AbortSignal) => Promise<R>,
    options: {
      signal?: AbortSignal;
      batchSize?: number;
      concurrency?: number;
      onProgress?: (completed: number, total: number) => void;
    } = {},
  ): Promise<{ results: R[]; errors: Array<{ item: T; error: Error }> }> {
    const { signal, batchSize = 100, concurrency = 5, onProgress } = options;
    const results: R[] = [];
    const errors: Array<{ item: T; error: Error }> = [];
    let completed = 0;

    for (let i = 0; i < items.length; i += batchSize) {
      signal?.throwIfAborted();

      const batch = items.slice(i, i + batchSize);

      // Process with concurrency limit
      const batchPromises = batch.map(async (item) => {
        try {
          signal?.throwIfAborted();
          const result = await processor(item, signal!);
          results.push(result);
        } catch (err) {
          if ((err as Error).name === 'AbortError') throw err;
          errors.push({ item, error: err as Error });
        } finally {
          completed++;
          onProgress?.(completed, items.length);
        }
      });

      await Promise.all(batchPromises);
    }

    return { results, errors };
  }
}
```

### 5.2 Cancellation Token Tree

```typescript
// Cancellation propagation tree
class CancellationScope {
  private controller: AbortController;
  private children: CancellationScope[] = [];
  private cleanupFns: Array<() => void | Promise<void>> = [];

  constructor(parentSignal?: AbortSignal) {
    this.controller = new AbortController();

    // Propagate parent's cancellation
    if (parentSignal) {
      parentSignal.addEventListener('abort', () => {
        this.cancel(parentSignal.reason);
      }, { once: true });
    }
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  // Create a child scope
  createChild(): CancellationScope {
    const child = new CancellationScope(this.signal);
    this.children.push(child);
    return child;
  }

  // Register a cleanup function
  onCancel(fn: () => void | Promise<void>): void {
    this.cleanupFns.push(fn);
  }

  // Execute cancellation
  async cancel(reason?: any): Promise<void> {
    if (this.controller.signal.aborted) return;

    // Cancel child scopes first
    await Promise.allSettled(
      this.children.map(child => child.cancel(reason)),
    );

    // Execute cleanup
    await Promise.allSettled(
      this.cleanupFns.map(fn => fn()),
    );

    this.controller.abort(reason);
  }
}

// Usage example
const rootScope = new CancellationScope();

const dbScope = rootScope.createChild();
dbScope.onCancel(async () => {
  await db.rollback();
  console.log('DB transaction rolled back');
});

const fileScope = rootScope.createChild();
fileScope.onCancel(async () => {
  await tempFile.delete();
  console.log('Temp file deleted');
});

// Cancelling root cancels all children
await rootScope.cancel('User requested cancellation');
```

---

## 6. Testable Cancellation Logic

```typescript
// === Tests ===
import { describe, it, expect, vi } from 'vitest';

describe('fetchWithTimeout', () => {
  it('throws TimeoutError on timeout', async () => {
    // Mock a slow response
    global.fetch = vi.fn().mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 10000)),
    );

    await expect(
      fetchWithTimeout('/api/slow', { timeoutMs: 100 }),
    ).rejects.toThrow(TimeoutError);
  });

  it('throws AbortError when signal is aborted', async () => {
    const controller = new AbortController();

    // Cancel immediately
    controller.abort();

    await expect(
      fetchWithTimeout('/api/data', { signal: controller.signal }),
    ).rejects.toThrow('AbortError');
  });

  it('returns a normal response', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ data: 'test' }), { status: 200 }),
    );

    const response = await fetchWithTimeout('/api/data', { timeoutMs: 5000 });
    expect(response.status).toBe(200);
  });

  it('executes cleanup after cancellation', async () => {
    const cleanup = vi.fn();
    const controller = new AbortController();

    const scope = new CancellationScope(controller.signal);
    scope.onCancel(cleanup);

    controller.abort();

    // Verify that cleanup was called
    await vi.waitFor(() => {
      expect(cleanup).toHaveBeenCalled();
    });
  });
});

// React component tests
import { render, screen, waitFor, act } from '@testing-library/react';

describe('SearchResults', () => {
  it('cancels the previous request when query changes', async () => {
    const abortSpy = vi.spyOn(AbortController.prototype, 'abort');

    const { rerender } = render(<SearchResults query="hello" />);

    // Change query
    rerender(<SearchResults query="world" />);

    // Verify that the previous request was cancelled
    expect(abortSpy).toHaveBeenCalled();
  });

  it('cancels the request on unmount', async () => {
    const abortSpy = vi.spyOn(AbortController.prototype, 'abort');

    const { unmount } = render(<SearchResults query="test" />);

    unmount();

    expect(abortSpy).toHaveBeenCalled();
  });
});
```

---

## 7. Practical Patterns

### 7.1 Debounced Cancellation

```typescript
// Search input debounce + cancellation
function useDebounceSearch(delayMs: number = 300) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const search = useCallback((searchQuery: string) => {
    // Clear previous debounce timer
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Cancel previous request
    controllerRef.current?.abort();

    if (!searchQuery.trim()) {
      setResults([]);
      setLoading(false);
      return;
    }

    setLoading(true);

    timeoutRef.current = window.setTimeout(async () => {
      const controller = new AbortController();
      controllerRef.current = controller;

      try {
        const response = await fetch(
          `/api/search?q=${encodeURIComponent(searchQuery)}`,
          { signal: controller.signal },
        );
        const data = await response.json();
        setResults(data);
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          console.error('Search error:', err);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }, delayMs);
  }, [delayMs]);

  // Cleanup
  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return { query, setQuery: (q: string) => { setQuery(q); search(q); }, results, loading };
}
```

### 7.2 Preventing Race Conditions

```typescript
// Pattern to only honor the latest request
function useLatestRequest<T>(
  fetchFn: (signal: AbortSignal) => Promise<T>,
  deps: any[],
): { data: T | null; loading: boolean; error: Error | null } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError(null);

    fetchFn(controller.signal)
      .then(result => {
        // Check if this request is still the latest
        if (requestId === requestIdRef.current) {
          setData(result);
          setLoading(false);
        }
      })
      .catch(err => {
        if (err.name !== 'AbortError' && requestId === requestIdRef.current) {
          setError(err);
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, deps);

  return { data, loading, error };
}
```

### 7.3 Using the First Result from Concurrent Operations

```typescript
// Use the first result from multiple sources and cancel the rest
async function raceWithCancellation<T>(
  tasks: Array<(signal: AbortSignal) => Promise<T>>,
): Promise<T> {
  const controller = new AbortController();

  try {
    const result = await Promise.race(
      tasks.map(task => task(controller.signal)),
    );
    return result;
  } finally {
    controller.abort(); // Cancel remaining tasks
  }
}

// Usage example: Use the fastest result from multiple API endpoints
const data = await raceWithCancellation([
  (signal) => fetch('https://api1.example.com/data', { signal }).then(r => r.json()),
  (signal) => fetch('https://api2.example.com/data', { signal }).then(r => r.json()),
  (signal) => fetch('https://api3.example.com/data', { signal }).then(r => r.json()),
]);
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

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
        """Main processing logic"""
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

Extend the basic implementation to add the following features.

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
    """Efficient search using hash map"""
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

**Key Points:**
- Be conscious of algorithm time complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Check configuration file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Check executing user's permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Steps

1. **Check the error message**: Read the stack trace to identify where the error occurred
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Form hypotheses**: List possible causes
4. **Verify step by step**: Verify hypotheses using log output or a debugger
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input/output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O wait**: Check disk and network I/O status
4. **Check concurrent connections**: Check connection pool status

| Issue Type | Diagnostic Tool | Countermeasure |
|-----------|----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

Here is a summary of criteria for making technology choices.

| Criterion | When to prioritize | When acceptable to compromise |
|-----------|-------------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin dashboards, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Growing services | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+--------------------------------------------------+
|          Architecture Selection Flow              |
+--------------------------------------------------+
|                                                   |
|  (1) Team size?                                   |
|    +-- Small (1-5 people) -> Monolith             |
|    +-- Large (10+ people) -> Go to (2)            |
|                                                   |
|  (2) Deploy frequency?                            |
|    +-- Weekly or less -> Monolith + module split   |
|    +-- Daily/multiple -> Go to (3)                |
|                                                   |
|  (3) Team independence?                           |
|    +-- High -> Microservices                      |
|    +-- Medium -> Modular monolith                 |
|                                                   |
+--------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A faster approach in the short term can become technical debt in the long term
- Conversely, over-engineering incurs high short-term costs and can delay projects

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables best-fit choices but increases operational costs

**3. Level of Abstraction**
- Higher abstraction provides greater reusability but can make debugging more difficult
- Lower abstraction is more intuitive but more prone to code duplication

```python
# Design decision recording template
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and challenges"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Context\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "+" if c['type'] == 'positive' else "!"
            md += f"- [{icon}] {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in professional practice?

Knowledge of this topic is frequently used in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Technique | Language/Environment | Use Case |
|-----------|---------------------|----------|
| AbortController | JS/TS | fetch, events, streams |
| AbortSignal.timeout() | JS/TS | Timeouts |
| AbortSignal.any() | JS/TS | Combining multiple conditions |
| asyncio.cancel() | Python | asyncio tasks |
| asyncio.TaskGroup | Python | Structured concurrency |
| Context.cancel() | Go | goroutines, HTTP, DB |
| context.WithTimeout | Go | Timeouts |
| CancellationToken | C# | Tasks |
| CancellationTokenSource | C# | Token creation |
| tokio::select! | Rust | Async branching |
| CancellationToken | Rust (tokio-util) | Token-based cancellation |
| Drop trait | Rust | Automatic cancellation on scope exit |

| Design Principle | Description |
|-----------------|-------------|
| Cooperative cancellation | Notify -> Cleanup -> Stop, not force stop |
| Guarantee cleanup | Release resources with finally / defer / Drop |
| Cancellation propagation | Parent -> child propagation, tree structure |
| Race condition handling | Account for simultaneous cancellation and completion |
| Testability | Make signal injectable from outside |

---

## 8. FAQ

### Q1: Can AbortController be reused?

No. Once `abort()` has been called on an AbortController, it cannot be restored. Create a new AbortController for each new request. The signal's `aborted` property cannot be changed once it becomes `true`.

### Q2: How should partial results from cancelled operations be handled?

This should be clarified at design time. There are three options: (1) discard partial results and start over, (2) save partial results and make them resumable (checkpoint pattern), (3) return partial results as-is. For batch processing, (2) is common, and for file uploads, chunk-level checkpoints are effective.

### Q3: What should you be careful about when testing cancellation?

Tests that depend on timing tend to be flaky. Use techniques such as calling `AbortController.abort()` immediately to test the cancellation path, controlling timers with `vi.useFakeTimers()`, and injecting `signal` as a mock. Designing tests that don't depend on non-deterministic timing is important.

### Q4: What APIs besides fetch support AbortSignal?

In Node.js, `fs/promises` (readFile, writeFile, etc.), `timers/promises` (setTimeout), `events.once()`, `stream.pipeline()`, `child_process.exec()`, and more support it. In browsers, you can pass signal as an option to `addEventListener`, and `ReadableStream`, `WritableStream`, `Blob.text()`, etc. also support it. You can add support to custom APIs using `signal?.throwIfAborted()` and `signal?.addEventListener('abort', ...)`.

### Q5: What are the differences between Go's context.Context and JavaScript's AbortSignal?

Go's Context provides unified handling of cancellation plus timeout (WithTimeout/WithDeadline) and value propagation (WithValue). JavaScript's AbortSignal is cancellation-only, with timeouts achieved through `AbortSignal.timeout()`. Go has a convention of passing Context as the first argument, and all blocking operations support Context. JavaScript's AbortSignal adoption is still ongoing, and some APIs don't support it yet.

### Q6: How do you integrate cancellation with error handling?

Cancellation is commonly treated as a type of error. In TypeScript, you distinguish it with `err.name === 'AbortError'`, and in Go with `errors.Is(err, context.Canceled)`. Since cancellation errors usually don't need to be displayed to the user, they receive special treatment in the error handling layer. It is also recommended to set the log level lower than that of regular errors.

```typescript
// Example of integrating error handling and cancellation
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number,
    public readonly isCancellation: boolean = false,
  ) {
    super(message);
    this.name = 'AppError';
  }

  static fromError(err: unknown): AppError {
    if (err instanceof AppError) return err;

    const error = err as Error;
    if (error.name === 'AbortError') {
      return new AppError('Operation cancelled', 'CANCELLED', 499, true);
    }
    if (error.name === 'TimeoutError') {
      return new AppError('Operation timed out', 'TIMEOUT', 504, false);
    }
    return new AppError(error.message, 'INTERNAL', 500, false);
  }
}

// Integrated handling in middleware
async function errorHandler(ctx: Context, next: () => Promise<void>) {
  try {
    await next();
  } catch (err) {
    const appError = AppError.fromError(err);

    if (appError.isCancellation) {
      // Log cancellation at debug level
      logger.debug('Request cancelled', { path: ctx.path });
      return; // No response needed
    }

    // Log regular errors at error level
    logger.error('Request failed', {
      path: ctx.path,
      code: appError.code,
      message: appError.message,
    });

    ctx.status = appError.statusCode;
    ctx.body = { error: { code: appError.code, message: appError.message } };
  }
}
```

---

## Recommended Next Guides

---

## References
1. MDN Web Docs. "AbortController."
2. Node.js Documentation. "AbortController."
3. Go Documentation. "context package."
4. Python Documentation. "asyncio - Tasks."
5. Microsoft Docs. "Cancellation in Managed Threads."
6. Tokio Documentation. "Cancellation."
