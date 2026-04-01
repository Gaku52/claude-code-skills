# Promise

> A Promise is an object representing a "future value." It resolves callback hell and makes asynchronous processing chainable. Master the usage of Promise.all, Promise.race, and Promise.allSettled.

## What You Will Learn in This Chapter

- [ ] Understand the three states and operational principles of a Promise
- [ ] Grasp Promise chaining and error propagation
- [ ] Learn concurrent execution patterns with Promises
- [ ] Compare Promise equivalents across different languages
- [ ] Master Promise patterns and anti-patterns used in production


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [Callbacks](./00-callbacks.md)

---

## 1. Promise Basics

### 1.1 The Three States of a Promise

```
The three states of a Promise:
  pending  -> fulfilled (success) -> holds a value
           -> rejected (failure)  -> holds an error

  +---------+
  | pending |
  +----+----+
  +----+----+
  v         v
+----------+ +----------+
|fulfilled | | rejected |
| (value)  | | (error)  |
+----------+ +----------+

  Once fulfilled/rejected, it cannot change (immutable)
  -> This is called "settled"

State transition rules:
  1. pending -> fulfilled (transitions via resolve)
  2. pending -> rejected (transitions via reject)
  3. fulfilled -> cannot change
  4. rejected -> cannot change
  5. pending -> pending (stays as is)
```

### 1.2 Creating and Consuming a Promise

```javascript
// Creating a Promise
const promise = new Promise((resolve, reject) => {
  // Asynchronous processing
  setTimeout(() => {
    const success = Math.random() > 0.5;
    if (success) {
      resolve("Success!");         // Transitions to fulfilled state
    } else {
      reject(new Error("Failed")); // Transitions to rejected state
    }
  }, 1000);
});

// Consuming a Promise
promise
  .then(value => console.log(value))   // On fulfilled
  .catch(error => console.error(error)) // On rejected
  .finally(() => console.log("Done"));  // Either way
```

### 1.3 Immediately Resolved Promises

```typescript
// Promise that immediately becomes fulfilled
const resolved = Promise.resolve(42);
const resolvedObj = Promise.resolve({ name: "Taro" });

// Promise that immediately becomes rejected
const rejected = Promise.reject(new Error("Error"));

// If the value is a Promise, it is returned as-is (not wrapped)
const original = Promise.resolve(42);
const same = Promise.resolve(original);
console.log(original === same); // true

// Thenable objects (objects with a then method)
const thenable = {
  then(resolve) {
    resolve(42);
  }
};
const fromThenable = Promise.resolve(thenable);
fromThenable.then(value => console.log(value)); // 42
```

### 1.4 Promise Execution Timing

```typescript
// The Promise callback (executor) is executed synchronously
console.log('1. before');

const p = new Promise((resolve) => {
  console.log('2. executor (synchronous execution)');
  resolve('value');
});

console.log('3. after');

p.then((value) => {
  console.log('4. then (asynchronous execution, microtask)');
});

console.log('5. end');

// Output order:
// 1. before
// 2. executor (synchronous execution)
// 3. after
// 5. end
// 4. then (asynchronous execution, microtask)
```

---

## 2. Promise Chaining

### 2.1 Basic Chaining

```javascript
// then() returns a new Promise -> chainable
fetchUser(userId)
  .then(user => fetchOrders(user.id))         // Returns a Promise
  .then(orders => orders.filter(o => o.active)) // Returns a value -> wrapped with Promise.resolve()
  .then(activeOrders => {
    console.log(`${activeOrders.length} active orders`);
    return activeOrders;
  })
  .catch(error => {
    // Catches errors from anywhere in the chain
    console.error("Error:", error.message);
  });

// Error propagation
//  then -> then -> then -> catch
//    | error occurs          ^
//    +----------------------+
//    skipped
```

### 2.2 How Chaining Works

```typescript
// The value of the Promise returned by then() is determined by the callback's return value

// Case 1: Return a value -> Promise.resolve(value)
Promise.resolve(1)
  .then(x => x + 1)  // Promise.resolve(2)
  .then(x => x * 3)  // Promise.resolve(6)
  .then(x => console.log(x)); // 6

// Case 2: Return a Promise -> that Promise is used
Promise.resolve(1)
  .then(x => Promise.resolve(x + 1))  // Promise<2>
  .then(x => fetch(`/api/${x}`))       // fetch's Promise
  .then(response => response.json());

// Case 3: Throw an error -> Promise.reject(error)
Promise.resolve(1)
  .then(x => {
    if (x < 10) throw new Error('Too small');
    return x;
  })
  .catch(err => console.error(err.message)); // "Too small"

// Case 4: Return nothing -> Promise.resolve(undefined)
Promise.resolve(1)
  .then(x => { console.log(x); }) // undefined
  .then(x => console.log(x));     // undefined
```

### 2.3 Error Handling in Detail

```typescript
// catch is a shortcut for then(undefined, onRejected)
promise.catch(fn);
// is equivalent to promise.then(undefined, fn);

// However, there is a subtle difference in behavior
promise
  .then(
    value => { throw new Error('Error inside then'); },
    error => console.log('rejected:', error) // <- Does NOT catch errors inside then
  );

promise
  .then(value => { throw new Error('Error inside then'); })
  .catch(error => console.log('caught:', error)); // <- Catches errors inside then too

// Error recovery mid-chain
fetchUser(userId)
  .then(user => fetchAvatar(user.avatarId))
  .catch(error => {
    console.warn('Avatar fetch failed, using default');
    return '/images/default-avatar.png'; // Recovery value
  })
  .then(avatarUrl => {
    // Reaches here even after an error (with recovery value)
    displayAvatar(avatarUrl);
  });

// Segmented error handling with multiple catches
fetchUser(userId)
  .then(user => {
    return fetchOrders(user.id);
  })
  .catch(error => {
    // Error from fetchUser or fetchOrders
    console.error('Data fetch error:', error);
    return []; // Recovery with empty array
  })
  .then(orders => {
    return calculateTotal(orders);
  })
  .catch(error => {
    // Error from calculateTotal only
    console.error('Calculation error:', error);
    return 0;
  })
  .then(total => {
    displayTotal(total);
  });
```

### 2.4 Using finally

```typescript
// finally: Executes regardless of success or failure
// Does not alter the value (transparent)

async function fetchData(url: string): Promise<Data> {
  showLoadingSpinner();

  return fetch(url)
    .then(response => response.json())
    .finally(() => {
      // Hide spinner (whether success or failure)
      hideLoadingSpinner();
    });
}

// finally passes the value through (does not change it)
Promise.resolve(42)
  .finally(() => {
    console.log('cleanup');
    return 100; // Ignored
  })
  .then(value => console.log(value)); // 42 (not 100)

// However, throwing inside finally propagates the error
Promise.resolve(42)
  .finally(() => {
    throw new Error('cleanup failed');
  })
  .catch(err => console.error(err.message)); // "cleanup failed"
```

---

## 3. Concurrent Execution Patterns

### 3.1 Promise.all

```typescript
// Promise.all: Succeeds if all succeed. Fails if even one fails
const [users, orders, products] = await Promise.all([
  fetchUsers(),      // 100ms
  fetchOrders(),     // 200ms
  fetchProducts(),   // 150ms
]);
// Total: max(100, 200, 150) = 200ms

// Type-safe usage (TypeScript)
interface DashboardData {
  users: User[];
  orders: Order[];
  stats: Stats;
}

async function getDashboard(): Promise<DashboardData> {
  const [users, orders, stats] = await Promise.all([
    fetchUsers(),                    // Promise<User[]>
    fetchOrders(),                   // Promise<Order[]>
    fetchStats(),                    // Promise<Stats>
  ] as const);

  return { users, orders, stats };
}

// Dynamic arrays
async function fetchAllUserData(userIds: string[]): Promise<User[]> {
  return Promise.all(
    userIds.map(id => fetchUser(id))
  );
}

// Note: If even one fails, the entire operation fails
try {
  const results = await Promise.all([
    fetchFromAPI1(), // Succeeds
    fetchFromAPI2(), // Fails -> entire operation fails
    fetchFromAPI3(), // Succeeds but result is discarded
  ]);
} catch (error) {
  // Only the error from fetchFromAPI2
  console.error('One of the requests failed:', error);
}
```

### 3.2 Promise.allSettled

```typescript
// Promise.allSettled: Gets all results (both successes and failures)
// Added in ES2020
const results = await Promise.allSettled([
  fetchFromAPI1(),   // Succeeds
  fetchFromAPI2(),   // Fails
  fetchFromAPI3(),   // Succeeds
]);
// results = [
//   { status: "fulfilled", value: data1 },
//   { status: "rejected", reason: Error },
//   { status: "fulfilled", value: data3 },
// ]

// Practical example: Tolerating partial failure
async function fetchMultipleAPIs(urls: string[]): Promise<{
  succeeded: { url: string; data: any }[];
  failed: { url: string; error: Error }[];
}> {
  const results = await Promise.allSettled(
    urls.map(async url => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return { url, data: await response.json() };
    })
  );

  const succeeded = results
    .filter((r): r is PromiseFulfilledResult<{ url: string; data: any }> =>
      r.status === 'fulfilled'
    )
    .map(r => r.value);

  const failed = results
    .filter((r): r is PromiseRejectedResult => r.status === 'rejected')
    .map((r, i) => ({ url: urls[i], error: r.reason }));

  return { succeeded, failed };
}

// Usage
const { succeeded, failed } = await fetchMultipleAPIs([
  'https://api1.example.com/data',
  'https://api2.example.com/data',
  'https://api3.example.com/data',
]);

console.log(`${succeeded.length} succeeded, ${failed.length} failed`);
```

### 3.3 Promise.race

```typescript
// Promise.race: Returns the first to complete
const fastest = await Promise.race([
  fetchFromServer1(), // 100ms
  fetchFromServer2(), // 50ms  <- This wins
  fetchFromServer3(), // 200ms
]);

// Practical example 1: Timeout implementation
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms);
  });

  return Promise.race([promise, timeout]);
}

// Usage
try {
  const data = await withTimeout(fetchData(), 5000);
} catch (error) {
  if (error.message.includes('Timeout')) {
    console.error('Request timed out');
  }
}

// Practical example 2: Cancellable Promise
function cancellable<T>(promise: Promise<T>): {
  promise: Promise<T>;
  cancel: () => void;
} {
  let cancelFn: () => void;

  const cancelPromise = new Promise<never>((_, reject) => {
    cancelFn = () => reject(new Error('Cancelled'));
  });

  return {
    promise: Promise.race([promise, cancelPromise]),
    cancel: cancelFn!,
  };
}

const { promise, cancel } = cancellable(fetchLargeData());
// Cancel after 5 seconds
setTimeout(cancel, 5000);
```

### 3.4 Promise.any

```typescript
// Promise.any: Returns the first to succeed (ES2021)
const firstSuccess = await Promise.any([
  fetchFromServer1(), // Fails
  fetchFromServer2(), // Succeeds <- This is returned
  fetchFromServer3(), // Succeeds
]);
// Only throws AggregateError if all fail

// Practical example: Fallback servers
async function fetchWithFallback(urls: string[]): Promise<Response> {
  try {
    return await Promise.any(
      urls.map(url => fetch(url).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r;
      }))
    );
  } catch (error) {
    if (error instanceof AggregateError) {
      console.error('All servers failed:', error.errors);
      throw new Error('All fallback servers failed');
    }
    throw error;
  }
}

// Usage
const response = await fetchWithFallback([
  'https://primary.example.com/api/data',
  'https://secondary.example.com/api/data',
  'https://tertiary.example.com/api/data',
]);

// Practical example: Fastest DNS resolution
async function resolveFastest(hostname: string): Promise<string> {
  return Promise.any([
    resolveViaDoH('https://dns.google/resolve', hostname),
    resolveViaDoH('https://cloudflare-dns.com/dns-query', hostname),
    resolveViaSystem(hostname),
  ]);
}
```

### 3.5 Comparison of the Four Concurrent Methods

```
+------------------+-------------+-------------+--------------+
| Method           | Success     | Failure     | Use Case     |
|                  | Condition   | Condition   |              |
+------------------+-------------+-------------+--------------+
| Promise.all      | All succeed | Any one     | All data     |
|                  |             | fails       | required     |
+------------------+-------------+-------------+--------------+
| Promise.         | Always      | Never       | Partial      |
| allSettled       | succeeds    |             | failure OK   |
+------------------+-------------+-------------+--------------+
| Promise.race     | First       | First       | Timeout      |
|                  | result      | result      |              |
+------------------+-------------+-------------+--------------+
| Promise.any      | First       | All fail    | Fallback     |
|                  | success     |             |              |
+------------------+-------------+-------------+--------------+
```

---

## 4. Common Mistakes and Anti-Patterns

### 4.1 Forgetting to Return a Promise

```typescript
// Bad: Forgetting to return a Promise
async function bad() {
  fetchData(); // No await or return -> does not wait for result
}

// Good: Fixed
async function good() {
  return fetchData(); // or await fetchData();
}

// Bad: Forgetting to return Promises in map
async function badMap(items: Item[]) {
  items.map(async item => {
    await processItem(item); // Not returned -> cannot wait for completion
  });
}

// Good: Wait with Promise.all
async function goodMap(items: Item[]) {
  await Promise.all(
    items.map(async item => {
      await processItem(item);
    })
  );
}
```

### 4.2 Unnecessary Promise Wrappers

```typescript
// Bad: Unnecessary Promise wrapper
async function unnecessary() {
  return new Promise((resolve) => {
    resolve(fetchData()); // fetchData() already returns a Promise
  });
}
// Good: Return directly
async function correct() {
  return fetchData();
}

// Bad: Unnecessary async
async function alsoUnnecessary() {
  return 42; // async is unnecessary (just returning a synchronous value)
}
// Good: Remove async if unnecessary
function simple(): number {
  return 42;
}

// However, async is useful when you want errors to become Promise.reject
async function withErrorHandling(): Promise<number> {
  const value = validate(input); // validate may throw
  return value; // Since it's an async function, throw is automatically converted to reject
}
```

### 4.3 async with forEach

```typescript
// Bad: async with forEach (uncontrollable concurrency)
items.forEach(async (item) => {
  await processItem(item); // All start simultaneously, cannot wait for completion
});
console.log('done'); // <- Executes BEFORE processItem completes!

// Good: Sequential execution with for...of
for (const item of items) {
  await processItem(item); // Processes one at a time in order
}
console.log('done'); // Executes after all items are complete

// Good: Concurrent execution with Promise.all
await Promise.all(items.map(item => processItem(item)));
console.log('done'); // Executes after all items are complete

// Good: Controlled concurrency with for...of + batching
async function processBatch<T>(
  items: T[],
  fn: (item: T) => Promise<void>,
  batchSize: number,
): Promise<void> {
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    await Promise.all(batch.map(fn));
  }
}

await processBatch(items, processItem, 5); // Batch processing with 5 concurrent
```

### 4.4 Promise Without catch

```typescript
// Bad: Promise without catch
fetchData().then(data => use(data));
// -> UnhandledPromiseRejection on rejection

// Good: Fixed
fetchData().then(data => use(data)).catch(handleError);

// Good: try-catch with async/await
async function handler() {
  try {
    const data = await fetchData();
    use(data);
  } catch (error) {
    handleError(error);
  }
}

// Also set up a global handler
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Log recording, alert sending, etc.
});
```

### 4.5 Nesting then Chains

```typescript
// Bad: then inside then (callback hell returns)
fetchUser(userId).then(user => {
  fetchOrders(user.id).then(orders => {
    fetchOrderDetails(orders[0].id).then(details => {
      console.log(details); // Deep nesting
    });
  });
});

// Good: Flatten with chaining
fetchUser(userId)
  .then(user => fetchOrders(user.id))
  .then(orders => fetchOrderDetails(orders[0].id))
  .then(details => console.log(details))
  .catch(error => console.error(error));

// Better: Even simpler with async/await
async function getDetails(userId: string) {
  const user = await fetchUser(userId);
  const orders = await fetchOrders(user.id);
  const details = await fetchOrderDetails(orders[0].id);
  return details;
}
```

---

## 5. Concurrency Limiting

### 5.1 Promise Pool

```typescript
// Promise pool that limits concurrent execution
async function promisePool<T>(
  tasks: (() => Promise<T>)[],
  concurrency: number,
): Promise<T[]> {
  const results: T[] = [];
  const executing = new Set<Promise<void>>();

  for (const [index, task] of tasks.entries()) {
    const promise = task().then(result => {
      results[index] = result;
    });

    executing.add(promise);
    promise.finally(() => executing.delete(promise));

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
}

// Usage: Fetch 1000 URLs with 5 concurrent
const urls = Array.from({ length: 1000 }, (_, i) =>
  `https://api.example.com/item/${i}`
);
const tasks = urls.map(url => () => fetch(url).then(r => r.json()));
const results = await promisePool(tasks, 5);
```

### 5.2 Semaphore-Based Concurrency Limiting

```typescript
class AsyncSemaphore {
  private permits: number;
  private waiting: (() => void)[] = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }
    return new Promise<void>(resolve => {
      this.waiting.push(resolve);
    });
  }

  release(): void {
    if (this.waiting.length > 0) {
      const resolve = this.waiting.shift()!;
      resolve();
    } else {
      this.permits++;
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

// Usage
const semaphore = new AsyncSemaphore(3); // Max 3 concurrent

const results = await Promise.all(
  urls.map(url =>
    semaphore.withPermit(() => fetch(url).then(r => r.json()))
  )
);
```

### 5.3 Queue-Based Concurrency Limiting

```typescript
class AsyncQueue<T> {
  private concurrency: number;
  private running = 0;
  private queue: {
    fn: () => Promise<T>;
    resolve: (value: T) => void;
    reject: (reason: any) => void;
  }[] = [];

  constructor(concurrency: number) {
    this.concurrency = concurrency;
  }

  add(fn: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({ fn, resolve, reject });
      this.processNext();
    });
  }

  private async processNext(): Promise<void> {
    if (this.running >= this.concurrency || this.queue.length === 0) {
      return;
    }

    this.running++;
    const { fn, resolve, reject } = this.queue.shift()!;

    try {
      const result = await fn();
      resolve(result);
    } catch (error) {
      reject(error);
    } finally {
      this.running--;
      this.processNext();
    }
  }

  get size(): number {
    return this.queue.length;
  }

  get pending(): number {
    return this.running;
  }
}

// Usage
const queue = new AsyncQueue<Response>(5);

const results = await Promise.all(
  urls.map(url =>
    queue.add(() => fetch(url))
  )
);
```

---

## 6. Practical Patterns

### 6.1 Retry Pattern

```typescript
async function retryPromise<T>(
  fn: () => Promise<T>,
  options: {
    retries?: number;
    delay?: number;
    backoff?: number;
    shouldRetry?: (error: unknown) => boolean;
    onRetry?: (error: unknown, attempt: number) => void;
  } = {},
): Promise<T> {
  const {
    retries = 3,
    delay = 1000,
    backoff = 2,
    shouldRetry = () => true,
    onRetry,
  } = options;

  let lastError: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === retries || !shouldRetry(error)) {
        throw error;
      }

      const waitTime = delay * Math.pow(backoff, attempt);
      const jitter = waitTime * 0.1 * Math.random();

      onRetry?.(error, attempt + 1);

      await new Promise(resolve =>
        setTimeout(resolve, waitTime + jitter)
      );
    }
  }

  throw lastError;
}

// Usage
const data = await retryPromise(
  () => fetch('https://api.example.com/data').then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }),
  {
    retries: 3,
    delay: 1000,
    shouldRetry: (error) => {
      if (error instanceof Error) {
        return error.message.includes('5') || error.message.includes('429');
      }
      return false;
    },
    onRetry: (error, attempt) => {
      console.warn(`Attempt ${attempt} failed:`, error);
    },
  }
);
```

### 6.2 Cache Pattern

```typescript
// Promise caching (preventing duplicate requests)
class PromiseCache<K, V> {
  private cache = new Map<K, Promise<V>>();
  private ttl: number;

  constructor(ttlMs: number = 60000) {
    this.ttl = ttlMs;
  }

  get(key: K, factory: () => Promise<V>): Promise<V> {
    const existing = this.cache.get(key);
    if (existing) return existing;

    const promise = factory().then(value => {
      // Delete cache after TTL
      setTimeout(() => this.cache.delete(key), this.ttl);
      return value;
    }).catch(error => {
      // Delete cache immediately on error (allow retry next time)
      this.cache.delete(key);
      throw error;
    });

    this.cache.set(key, promise);
    return promise;
  }

  invalidate(key: K): void {
    this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }
}

// Usage
const userCache = new PromiseCache<string, User>(30000); // 30-second TTL

async function getUser(userId: string): Promise<User> {
  return userCache.get(userId, () =>
    fetch(`/api/users/${userId}`).then(r => r.json())
  );
}

// Even if the same user is requested simultaneously, the API is called only once
const [user1, user2] = await Promise.all([
  getUser('user-123'),
  getUser('user-123'), // Cache hit (same Promise)
]);
```

### 6.3 Debounce Pattern

```typescript
// Promise-based debounce
function debouncePromise<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  delay: number,
): T {
  let timeoutId: ReturnType<typeof setTimeout>;
  let pendingResolve: ((value: any) => void) | null = null;
  let pendingReject: ((reason: any) => void) | null = null;

  return ((...args: Parameters<T>): Promise<ReturnType<T>> => {
    return new Promise((resolve, reject) => {
      // Cancel the previous pending
      if (pendingReject) {
        pendingReject(new Error('Debounced'));
      }

      clearTimeout(timeoutId);
      pendingResolve = resolve;
      pendingReject = reject;

      timeoutId = setTimeout(async () => {
        try {
          const result = await fn(...args);
          pendingResolve?.(result);
        } catch (error) {
          pendingReject?.(error);
        }
        pendingResolve = null;
        pendingReject = null;
      }, delay);
    });
  }) as T;
}

// Usage: Search API
const debouncedSearch = debouncePromise(
  (query: string) => fetch(`/api/search?q=${query}`).then(r => r.json()),
  300,
);

// Even if called multiple times within 300ms, only the last call executes
input.addEventListener('input', async (e) => {
  try {
    const results = await debouncedSearch(e.target.value);
    renderResults(results);
  } catch (error) {
    if (error.message !== 'Debounced') {
      console.error(error);
    }
  }
});
```

### 6.4 Pipeline Pattern

```typescript
// Promise pipeline: Build processing step by step
type AsyncPipe<T, R> = (input: T) => Promise<R>;

function pipeline<T>(...fns: AsyncPipe<any, any>[]): AsyncPipe<T, any> {
  return async (input: T) => {
    let result: any = input;
    for (const fn of fns) {
      result = await fn(result);
    }
    return result;
  };
}

// Usage
const processOrder = pipeline<OrderInput>(
  validateOrder,        // OrderInput -> ValidatedOrder
  calculatePricing,     // ValidatedOrder -> PricedOrder
  applyDiscounts,       // PricedOrder -> DiscountedOrder
  processPayment,       // DiscountedOrder -> PaidOrder
  createShipment,       // PaidOrder -> ShippedOrder
  sendConfirmation,     // ShippedOrder -> ConfirmedOrder
);

const order = await processOrder({
  items: [{ productId: 'p-1', quantity: 2 }],
  customerId: 'c-123',
});
```

---

## 7. Promise Equivalents in Other Languages

### 7.1 Python: asyncio.Future / coroutine

```python
import asyncio

# Python's coroutine is equivalent to JavaScript's async function
async def fetch_user(user_id: str) -> dict:
    # Use await to wait for another coroutine
    await asyncio.sleep(0.1)  # Simulate I/O
    return {"id": user_id, "name": "Taro"}

# asyncio.gather = Promise.all
async def fetch_all():
    users, orders, stats = await asyncio.gather(
        fetch_user("u-1"),
        fetch_orders("u-1"),
        fetch_stats(),
    )
    return {"users": users, "orders": orders, "stats": stats}

# asyncio.wait = More fine-grained control
async def fetch_with_timeout():
    tasks = [
        asyncio.create_task(fetch_user("u-1")),
        asyncio.create_task(fetch_orders("u-1")),
    ]

    done, pending = await asyncio.wait(
        tasks,
        timeout=5.0,
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()  # Cancel timed-out tasks

    return [task.result() for task in done]

# asyncio.TaskGroup (Python 3.11+) = Structured concurrency
async def structured_fetch():
    async with asyncio.TaskGroup() as tg:
        user_task = tg.create_task(fetch_user("u-1"))
        orders_task = tg.create_task(fetch_orders("u-1"))

    # All tasks are complete by the time you exit the TaskGroup
    return user_task.result(), orders_task.result()
```

### 7.2 Rust: Future

```rust
use tokio;
use futures::future;

// Rust's Future = JavaScript's Promise
// However, it's lazy: it does not execute until .await

async fn fetch_user(user_id: &str) -> Result<User, AppError> {
    // An async function returns Future<Output = Result<User, AppError>>
    let url = format!("https://api.example.com/users/{}", user_id);
    let user: User = reqwest::get(&url).await?.json().await?;
    Ok(user)
}

// tokio::join! = Promise.all
async fn fetch_all(user_id: &str) -> Result<Dashboard, AppError> {
    let (user, orders, stats) = tokio::join!(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_stats(),
    );

    Ok(Dashboard {
        user: user?,
        orders: orders?,
        stats: stats?,
    })
}

// tokio::select! = Promise.race
async fn fetch_with_timeout(user_id: &str) -> Result<User, AppError> {
    tokio::select! {
        result = fetch_user(user_id) => result,
        _ = tokio::time::sleep(Duration::from_secs(5)) => {
            Err(AppError::Timeout)
        }
    }
}

// futures::future::join_all = Promise.all (for dynamic count)
async fn fetch_all_users(user_ids: Vec<String>) -> Vec<Result<User, AppError>> {
    let futures: Vec<_> = user_ids.iter()
        .map(|id| fetch_user(id))
        .collect();

    future::join_all(futures).await
}
```

### 7.3 Java: CompletableFuture

```java
import java.util.concurrent.*;

// Java's CompletableFuture = JavaScript's Promise

public class CompletableFutureExamples {

    // Basic creation
    CompletableFuture<User> fetchUser(String userId) {
        return CompletableFuture.supplyAsync(() -> {
            // Runs on a background thread
            return userRepo.findById(userId);
        });
    }

    // Chaining (equivalent to then)
    CompletableFuture<String> getUserName(String userId) {
        return fetchUser(userId)
            .thenApply(user -> user.getName())        // map
            .thenApply(name -> name.toUpperCase());    // map
    }

    // Equivalent to flatMap
    CompletableFuture<List<Order>> getUserOrders(String userId) {
        return fetchUser(userId)
            .thenCompose(user -> fetchOrders(user.getId())); // flatMap
    }

    // Equivalent to Promise.all
    CompletableFuture<Dashboard> getDashboard(String userId) {
        CompletableFuture<User> userF = fetchUser(userId);
        CompletableFuture<List<Order>> ordersF = fetchOrders(userId);
        CompletableFuture<Stats> statsF = fetchStats(userId);

        return CompletableFuture.allOf(userF, ordersF, statsF)
            .thenApply(v -> new Dashboard(
                userF.join(),
                ordersF.join(),
                statsF.join()
            ));
    }

    // Equivalent to Promise.race
    CompletableFuture<User> fetchFastest(String userId) {
        return CompletableFuture.anyOf(
            fetchFromPrimary(userId),
            fetchFromSecondary(userId)
        ).thenApply(result -> (User) result);
    }

    // Error handling
    CompletableFuture<User> fetchWithFallback(String userId) {
        return fetchUser(userId)
            .exceptionally(error -> {
                // Equivalent to catch
                System.err.println("Fetch failed: " + error.getMessage());
                return User.defaultUser();
            });
    }

    // Timeout (Java 9+)
    CompletableFuture<User> fetchWithTimeout(String userId) {
        return fetchUser(userId)
            .orTimeout(5, TimeUnit.SECONDS)
            .exceptionally(error -> {
                if (error instanceof TimeoutException) {
                    return User.defaultUser();
                }
                throw new CompletionException(error);
            });
    }
}
```

### 7.4 C#: Task

```csharp
using System;
using System.Threading.Tasks;

// C#'s Task = JavaScript's Promise

public class TaskExamples
{
    // Basic
    async Task<User> FetchUserAsync(string userId)
    {
        var response = await httpClient.GetAsync($"/api/users/{userId}");
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<User>();
    }

    // Task.WhenAll = Promise.all
    async Task<Dashboard> GetDashboardAsync(string userId)
    {
        var userTask = FetchUserAsync(userId);
        var ordersTask = FetchOrdersAsync(userId);
        var statsTask = FetchStatsAsync(userId);

        await Task.WhenAll(userTask, ordersTask, statsTask);

        return new Dashboard
        {
            User = userTask.Result,
            Orders = ordersTask.Result,
            Stats = statsTask.Result,
        };
    }

    // Task.WhenAny = Promise.race
    async Task<User> FetchFastestAsync(string userId)
    {
        var task1 = FetchFromPrimaryAsync(userId);
        var task2 = FetchFromSecondaryAsync(userId);

        var completed = await Task.WhenAny(task1, task2);
        return await completed;
    }

    // Cancellation token
    async Task<User> FetchWithCancellationAsync(
        string userId,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        var response = await httpClient.GetAsync(
            $"/api/users/{userId}", ct
        );
        return await response.Content.ReadFromJsonAsync<User>(ct);
    }
}
```

---

## 8. Testing Promises

### 8.1 Basic Tests

```typescript
import { describe, it, expect, vi } from 'vitest';

describe('Promise Pattern Tests', () => {
  it('tests normal Promise resolution', async () => {
    const result = await Promise.resolve(42);
    expect(result).toBe(42);
  });

  it('tests Promise rejection', async () => {
    await expect(Promise.reject(new Error('test')))
      .rejects.toThrow('test');
  });

  it('tests Promise.all behavior', async () => {
    const results = await Promise.all([
      Promise.resolve(1),
      Promise.resolve(2),
      Promise.resolve(3),
    ]);
    expect(results).toEqual([1, 2, 3]);
  });

  it('tests Promise.all failure', async () => {
    await expect(
      Promise.all([
        Promise.resolve(1),
        Promise.reject(new Error('fail')),
        Promise.resolve(3),
      ])
    ).rejects.toThrow('fail');
  });

  it('tests Promise.allSettled behavior', async () => {
    const results = await Promise.allSettled([
      Promise.resolve('ok'),
      Promise.reject(new Error('fail')),
    ]);

    expect(results[0]).toEqual({ status: 'fulfilled', value: 'ok' });
    expect(results[1].status).toBe('rejected');
  });
});
```

### 8.2 Mocking Asynchronous Functions

```typescript
describe('Mocking Async Functions', () => {
  it('tests with mocked fetch', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ id: '123', name: 'Test' }),
    });

    global.fetch = mockFetch;

    const user = await fetchUser('123');
    expect(user.name).toBe('Test');
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/users/123')
    );
  });

  it('tests retry logic', async () => {
    const mockFn = vi.fn()
      .mockRejectedValueOnce(new Error('fail'))
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValue('success');

    const result = await retryPromise(mockFn, { retries: 3, delay: 10 });
    expect(result).toBe('success');
    expect(mockFn).toHaveBeenCalledTimes(3);
  });

  it('tests timeout', async () => {
    vi.useFakeTimers();

    const slowPromise = new Promise(resolve =>
      setTimeout(() => resolve('done'), 10000)
    );

    const promise = withTimeout(slowPromise, 5000);

    vi.advanceTimersByTime(5000);

    await expect(promise).rejects.toThrow('Timeout');

    vi.useRealTimers();
  });

  it('tests concurrency limiting', async () => {
    let concurrent = 0;
    let maxConcurrent = 0;

    const tasks = Array.from({ length: 10 }, () => async () => {
      concurrent++;
      maxConcurrent = Math.max(maxConcurrent, concurrent);
      await new Promise(r => setTimeout(r, 50));
      concurrent--;
      return 'done';
    });

    await promisePool(tasks, 3);
    expect(maxConcurrent).toBeLessThanOrEqual(3);
  });
});
```

---

## 9. Promise Internals

### 9.1 Simplified Promise Implementation

```typescript
// Simplified implementation to understand how Promises work internally
class SimplePromise<T> {
  private state: 'pending' | 'fulfilled' | 'rejected' = 'pending';
  private value: T | undefined;
  private reason: any;
  private onFulfilledCallbacks: ((value: T) => void)[] = [];
  private onRejectedCallbacks: ((reason: any) => void)[] = [];

  constructor(executor: (
    resolve: (value: T) => void,
    reject: (reason: any) => void,
  ) => void) {
    const resolve = (value: T) => {
      if (this.state !== 'pending') return;
      this.state = 'fulfilled';
      this.value = value;
      this.onFulfilledCallbacks.forEach(cb => cb(value));
    };

    const reject = (reason: any) => {
      if (this.state !== 'pending') return;
      this.state = 'rejected';
      this.reason = reason;
      this.onRejectedCallbacks.forEach(cb => cb(reason));
    };

    try {
      executor(resolve, reject);
    } catch (error) {
      reject(error);
    }
  }

  then<U>(
    onFulfilled?: (value: T) => U | SimplePromise<U>,
    onRejected?: (reason: any) => U | SimplePromise<U>,
  ): SimplePromise<U> {
    return new SimplePromise<U>((resolve, reject) => {
      const handleFulfilled = (value: T) => {
        queueMicrotask(() => {
          try {
            if (onFulfilled) {
              const result = onFulfilled(value);
              if (result instanceof SimplePromise) {
                result.then(resolve, reject);
              } else {
                resolve(result);
              }
            } else {
              resolve(value as any);
            }
          } catch (error) {
            reject(error);
          }
        });
      };

      const handleRejected = (reason: any) => {
        queueMicrotask(() => {
          try {
            if (onRejected) {
              const result = onRejected(reason);
              if (result instanceof SimplePromise) {
                result.then(resolve, reject);
              } else {
                resolve(result);
              }
            } else {
              reject(reason);
            }
          } catch (error) {
            reject(error);
          }
        });
      };

      switch (this.state) {
        case 'fulfilled':
          handleFulfilled(this.value!);
          break;
        case 'rejected':
          handleRejected(this.reason);
          break;
        case 'pending':
          this.onFulfilledCallbacks.push(handleFulfilled);
          this.onRejectedCallbacks.push(handleRejected);
          break;
      }
    });
  }

  catch<U>(onRejected: (reason: any) => U | SimplePromise<U>): SimplePromise<U> {
    return this.then(undefined, onRejected);
  }

  static resolve<T>(value: T): SimplePromise<T> {
    return new SimplePromise(resolve => resolve(value));
  }

  static reject(reason: any): SimplePromise<never> {
    return new SimplePromise((_, reject) => reject(reason));
  }

  static all<T>(promises: SimplePromise<T>[]): SimplePromise<T[]> {
    return new SimplePromise((resolve, reject) => {
      const results: T[] = [];
      let completed = 0;

      if (promises.length === 0) {
        resolve([]);
        return;
      }

      promises.forEach((promise, index) => {
        promise.then(
          value => {
            results[index] = value;
            completed++;
            if (completed === promises.length) {
              resolve(results);
            }
          },
          reject,
        );
      });
    });
  }
}
```


---

## Hands-On Exercises

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
    print(f"Speedup:             {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure results with benchmarks
---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining hands-on experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying its behavior.

### Q2: What are common mistakes that beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts covered in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

Knowledge of this topic is frequently used in everyday development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Method | Behavior | Use Case |
|--------|----------|----------|
| Promise.all | Succeeds when all succeed | Independent multiple API calls |
| Promise.allSettled | Waits for all to complete | Tolerating partial failure |
| Promise.race | Fastest result | Timeout implementation |
| Promise.any | First success | Fallback servers |

### Promise Best Practices

```
1. Always handle errors
   -> .catch() or try-catch

2. Avoid unnecessary Promise wrapping
   -> async functions already return Promises

3. Use Promise.all for parallelizable operations
   -> Avoid wasteful sequential awaits

4. Limit concurrency for large-scale parallel operations
   -> Semaphore or pool pattern

5. Use Promise caching to prevent duplicate requests
   -> Consolidate simultaneous requests for the same key into one
```

---

## Recommended Next Guides

---

## References
1. MDN Web Docs. "Promise."
2. Archibald, J. "JavaScript Promises: An Introduction." web.dev.
3. Promises/A+ Specification. promisesaplus.com.
4. ECMAScript Language Specification. "Promise Objects."
5. Tokio Documentation. "Working with Futures." tokio.rs.
6. Python Documentation. "asyncio - Tasks and Coroutines."
7. Oracle. "CompletableFuture." docs.oracle.com.
8. Microsoft. "Task-based asynchronous pattern." docs.microsoft.com.
