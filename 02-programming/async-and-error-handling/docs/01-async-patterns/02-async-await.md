# async/await

> async/await は「非同期コードを同期的に読める」構文糖。Promise ベースの非同期処理を直感的に書ける。JavaScript, Python, Rust, C# での実装と、並行実行パターンを解説。

## この章で学ぶこと

- [ ] async/await の仕組みと動作原理を理解する
- [ ] 各言語での async/await の違いを把握する
- [ ] 効率的な並行実行パターンを学ぶ
- [ ] エラーハンドリングのベストプラクティスを身につける
- [ ] キャンセレーション・タイムアウト・リトライパターンを実装できる
- [ ] テスト手法とデバッグ技術を理解する

---

## 1. async/await の基本

### 1.1 概念と動作原理

async/await は、非同期処理を同期的なコードのように記述できる構文糖（Syntactic Sugar）である。内部的には Promise（JavaScript）や Future（Rust）、Coroutine（Python）をベースとしている。

```
async 関数:
  → Promise を返す関数
  → return 値は自動的に Promise.resolve() でラップ
  → throw した値は Promise.reject() でラップ

await 式:
  → Promise が解決されるまで関数の実行を一時停止
  → 解決された値を返す
  → async 関数内でのみ使用可能（ES2022 で Top-Level Await 追加）
  → rejected な Promise は例外としてスローされる

内部動作:
  async function f() {
    const a = await fetchA();  // ここで一時停止
    const b = await fetchB();  // a が解決後に再開
    return a + b;
  }

  // 以下と同等:
  function f() {
    return fetchA()
      .then(a => fetchB().then(b => a + b));
  }
```

### 1.2 ステートマシンとしての async/await

コンパイラ（またはエンジン）は async 関数をステートマシンに変換する。これにより、一時停止と再開が効率的に行える。

```typescript
// 開発者が書くコード
async function process() {
  console.log("Step 1");
  const a = await fetchA();
  console.log("Step 2");
  const b = await fetchB(a);
  console.log("Step 3");
  return a + b;
}

// エンジン内部での概念的な変換（擬似コード）
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

### 1.3 マイクロタスクキューとの関係

```typescript
// async/await はマイクロタスクキューを使用する
async function demo() {
  console.log("1: async 関数開始（同期的に実行）");
  const result = await Promise.resolve("hello");
  // ↑ ここで一時停止し、マイクロタスクキューに継続処理を入れる
  console.log("3: await 後の再開（マイクロタスクとして実行）");
  return result;
}

console.log("0: 呼び出し前");
demo().then(() => console.log("4: then コールバック"));
console.log("2: 呼び出し後（同期的に実行）");

// 出力順序:
// 0: 呼び出し前
// 1: async 関数開始（同期的に実行）
// 2: 呼び出し後（同期的に実行）
// 3: await 後の再開（マイクロタスクとして実行）
// 4: then コールバック
```

### 1.4 Top-Level Await（ES2022）

```typescript
// ES モジュールでは Top-Level Await が使用可能

// config.ts - 設定の非同期読み込み
const response = await fetch("/api/config");
export const config = await response.json();

// main.ts - インポート時に自動的に待機される
import { config } from "./config.ts";
console.log(config.apiKey); // 設定読み込み完了後に実行される

// 注意点:
// 1. CommonJS (require) では使用不可
// 2. モジュールのロード順序に影響する
// 3. 循環依存に注意
// 4. サーバーサイドでの初期化に便利
```

---

## 2. JavaScript/TypeScript

### 2.1 基本パターン

```typescript
// 基本的な async 関数
async function getUserProfile(userId: string): Promise<UserProfile> {
  const user = await userRepo.findById(userId);
  if (!user) throw new Error("User not found");

  const [orders, reviews] = await Promise.all([
    orderRepo.findByUserId(userId),
    reviewRepo.findByUserId(userId),
  ]);

  return { user, orders, reviews };
}

// アロー関数での async
const fetchData = async (url: string): Promise<Response> => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }
  return response;
};

// メソッドでの async
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

  // getter では async を使えないので注意
  // async get name() {} // SyntaxError

  // 代替パターン
  async getName(): Promise<string> {
    const profile = await this.loadProfile();
    return profile.name;
  }
}
```

### 2.2 エラーハンドリング

```typescript
// try/catch による基本的なエラーハンドリング
async function safeGetUser(userId: string): Promise<User | null> {
  try {
    return await userRepo.findById(userId);
  } catch (error) {
    logger.error("Failed to get user", { userId, error });
    return null;
  }
}

// 複数の非同期処理での細かいエラーハンドリング
async function processOrder(orderId: string): Promise<OrderResult> {
  // Step 1: 注文取得
  let order: Order;
  try {
    order = await orderRepo.findById(orderId);
  } catch (error) {
    throw new OrderNotFoundError(orderId, { cause: error });
  }

  // Step 2: 在庫確認
  try {
    await inventoryService.checkAvailability(order.items);
  } catch (error) {
    if (error instanceof OutOfStockError) {
      return { status: "out_of_stock", items: error.unavailableItems };
    }
    throw error; // 予期しないエラーは再スロー
  }

  // Step 3: 決済処理
  try {
    const payment = await paymentService.charge(order.total, order.paymentMethod);
    return { status: "completed", payment };
  } catch (error) {
    // 決済失敗時は在庫を戻す
    await inventoryService.release(order.items);
    throw new PaymentFailedError(orderId, { cause: error });
  }
}

// Result 型パターン（例外を使わない）
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

// 使用例
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

### 2.3 逐次 vs 並行

```typescript
// 逐次実行（直列）
async function sequential(): Promise<void> {
  const a = await fetchA(); // 100ms
  const b = await fetchB(); // 200ms
  // 合計: 300ms（直列）
}

// 並行実行
async function concurrent(): Promise<void> {
  const [a, b] = await Promise.all([
    fetchA(), // 100ms ┐
    fetchB(), // 200ms ┤ 並行
  ]);        //       ┘
  // 合計: 200ms（並行）
}

// 重要: Promise の作成時点で実行が開始される
async function earlyStart(): Promise<void> {
  // Promise を先に作成（実行開始）
  const promiseA = fetchA(); // 即座に開始
  const promiseB = fetchB(); // 即座に開始

  // 両方の結果を待つ
  const a = await promiseA;
  const b = await promiseB;
  // Promise.all と同等の並行実行になる
}

// ただし、エラーハンドリングに注意
async function earlyStartWithErrorHandling(): Promise<void> {
  const promiseA = fetchA();
  const promiseB = fetchB();

  // promiseB が先に reject しても、promiseA の await で待機中
  // → unhandled rejection 警告が出る可能性がある
  // → Promise.all を使う方が安全
  try {
    const [a, b] = await Promise.all([promiseA, promiseB]);
  } catch (error) {
    // 全てのエラーをキャッチ
  }
}
```

### 2.4 イテレーション・パターン

```typescript
// ❌ for...of + await（逐次実行）
async function processSequential(urls: string[]): Promise<Response[]> {
  const results: Response[] = [];
  for (const url of urls) {
    const response = await fetch(url); // 1件ずつ...
    results.push(response);
  }
  return results;
}

// ✅ Promise.all（全並行）
async function processAllConcurrent(urls: string[]): Promise<Response[]> {
  return Promise.all(urls.map(url => fetch(url)));
}

// ✅ 制限付き並行実行（同時N件まで）
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
      // 完了した Promise を除去
      const completed = executing.findIndex(
        p => p === Promise.race([p]).then(() => p)
      );
    }
  }

  await Promise.all(executing);
  return results;
}

// より洗練された並行制限: セマフォ
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

// セマフォの使用例
async function fetchAllWithLimit(urls: string[], limit: number) {
  const semaphore = new Semaphore(limit);
  return Promise.all(
    urls.map(url => semaphore.run(() => fetch(url)))
  );
}

// for-await-of（非同期イテレーション）
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

// ReadableStream の非同期イテレーション
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

### 2.5 AbortController によるキャンセレーション

```typescript
// 基本的なキャンセレーション
async function fetchWithCancel(
  url: string,
  signal?: AbortSignal
): Promise<Response> {
  const response = await fetch(url, { signal });
  return response;
}

const controller = new AbortController();
const promise = fetchWithCancel("/api/data", controller.signal);

// 必要に応じてキャンセル
setTimeout(() => controller.abort(), 5000);

try {
  const result = await promise;
} catch (error) {
  if (error instanceof DOMException && error.name === "AbortError") {
    console.log("リクエストがキャンセルされました");
  } else {
    throw error;
  }
}

// タイムアウト付きの汎用関数
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

// AbortSignal.timeout()（新しい API）
async function fetchWithTimeout(url: string): Promise<Response> {
  return fetch(url, {
    signal: AbortSignal.timeout(5000), // 5秒タイムアウト
  });
}

// 複数リクエストの一括キャンセル
class RequestManager {
  private controller = new AbortController();

  async fetch(url: string): Promise<Response> {
    return fetch(url, { signal: this.controller.signal });
  }

  cancelAll(): void {
    this.controller.abort();
    this.controller = new AbortController(); // リセット
  }
}

// React での使用例
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
    return () => controller.abort(); // クリーンアップ
  }, deps);
}

// 使用例
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

### 2.6 リトライパターン

```typescript
// 指数バックオフ付きリトライ
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

      // 指数バックオフ + ジッター
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt) + Math.random() * 1000,
        maxDelay
      );
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}

// 使用例
const data = await withRetry(
  () => fetch("/api/data").then(r => r.json()),
  {
    maxRetries: 3,
    baseDelay: 1000,
    shouldRetry: (error, attempt) => {
      // ネットワークエラーや 5xx のみリトライ
      if (error instanceof TypeError) return true; // ネットワークエラー
      if (error instanceof HttpError && error.status >= 500) return true;
      return false;
    },
    onRetry: (error, attempt) => {
      console.log(`Retry ${attempt + 1}: ${error.message}`);
    },
  }
);

// サーキットブレーカー付きリトライ
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

### 3.1 asyncio の基本

```python
import asyncio
from typing import Any

# 基本的な async 関数
async def get_user_profile(user_id: str) -> dict:
    user = await user_repo.find_by_id(user_id)
    if not user:
        raise ValueError("User not found")

    # asyncio.gather で並行実行
    orders, reviews = await asyncio.gather(
        order_repo.find_by_user_id(user_id),
        review_repo.find_by_user_id(user_id),
    )
    return {"user": user, "orders": orders, "reviews": reviews}

# 実行
async def main():
    profile = await get_user_profile("user-123")
    print(profile)

asyncio.run(main())
```

### 3.2 Task の管理

```python
import asyncio

# タスクの作成と並行実行
async def process_items(items: list[str]) -> list[dict]:
    tasks = [asyncio.create_task(fetch_item(item)) for item in items]
    return await asyncio.gather(*tasks)

# タスクのキャンセル
async def cancellable_operation():
    task = asyncio.create_task(long_running_operation())

    # 5秒後にキャンセル
    await asyncio.sleep(5)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("タスクがキャンセルされました")

# TaskGroup（Python 3.11+）- 構造化並行性
async def structured_concurrency():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_users())
        task2 = tg.create_task(fetch_orders())
        task3 = tg.create_task(fetch_products())
    # 全タスク完了後にここに到達
    # いずれかのタスクが例外を発生させると、
    # 他のタスクもキャンセルされる
    users = task1.result()
    orders = task2.result()
    products = task3.result()
    return users, orders, products

# TaskGroup でのエラーハンドリング
async def safe_task_group():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(might_fail_1())
            tg.create_task(might_fail_2())
    except* ValueError as eg:
        # ExceptionGroup をハンドリング（Python 3.11+）
        for exc in eg.exceptions:
            print(f"ValueError: {exc}")
    except* TypeError as eg:
        for exc in eg.exceptions:
            print(f"TypeError: {exc}")
```

### 3.3 タイムアウトとデッドライン

```python
import asyncio

# wait_for によるタイムアウト
async def with_timeout():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        print("タイムアウト")

# asyncio.timeout（Python 3.11+）
async def modern_timeout():
    async with asyncio.timeout(5.0):
        result = await slow_operation()
        return result

# デッドライン
async def with_deadline():
    deadline = asyncio.get_event_loop().time() + 10.0
    async with asyncio.timeout_at(deadline):
        await step1()
        await step2()  # 残り時間内で実行
        await step3()  # 全体で10秒以内

# asyncio.wait で部分的な完了を処理
async def partial_results():
    tasks = [
        asyncio.create_task(fetch(url))
        for url in urls
    ]

    # 最初に完了した結果を取得
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
        print(f"Completed: {task.result()}")

    # 残りをキャンセル
    for task in pending:
        task.cancel()

# as_completed でストリーミング処理
async def stream_results():
    tasks = [
        asyncio.create_task(fetch(url))
        for url in urls
    ]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Got result: {result}")
        # 完了順に処理される
```

### 3.4 非同期コンテキストマネージャーとイテレータ

```python
import asyncio
from contextlib import asynccontextmanager

# 非同期コンテキストマネージャー（クラスベース）
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
        return False  # 例外を再送出

# 使用
async def query_users():
    async with AsyncDatabaseConnection("postgresql://...") as conn:
        rows = await conn.fetch("SELECT * FROM users")
        return rows

# デコレータベースの非同期コンテキストマネージャー
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

# 非同期イテレータ
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

# 使用
async def process_all_pages():
    async for page in AsyncPaginator("/api/users"):
        for user in page:
            await process_user(user)

# 非同期ジェネレーター
async def async_range(start: int, stop: int, delay: float = 0.1):
    for i in range(start, stop):
        await asyncio.sleep(delay)
        yield i

async def use_async_generator():
    async for value in async_range(0, 10):
        print(value)
```

### 3.5 aiohttp を使った実践的な HTTP クライアント

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

# 使用例
async def main():
    async with AsyncHttpClient("https://api.example.com") as client:
        # 並行で100件のユーザーを取得（同時10件まで）
        paths = [f"/users/{i}" for i in range(100)]
        users = await client.get_many(paths)
        print(f"Fetched {len(users)} users")
```

---

## 4. Rust

### 4.1 Future トレイトと async/await

```rust
// Rust: async/await（tokio ランタイム）
use tokio;

// Rust の async fn は Future トレイトを実装する型を返す
// Future トレイト:
// trait Future {
//     type Output;
//     fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
// }

async fn get_user_profile(user_id: &str) -> Result<UserProfile, AppError> {
    let user = user_repo.find_by_id(user_id).await?;

    // tokio::join! で並行実行
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

// Rust の async の特徴:
// → ゼロコスト抽象化（ステートマシンにコンパイル）
// → ランタイムが分離（tokio, async-std, smol）
// → Future は lazy（await するまで実行されない）
// → Send + 'static の制約（スレッド間移動可能）
```

### 4.2 tokio の並行実行パターン

```rust
use tokio;
use tokio::time::{timeout, Duration};

// tokio::spawn でタスクを生成
async fn spawn_tasks() -> Result<(), Box<dyn std::error::Error>> {
    let handle1 = tokio::spawn(async {
        // 独立したタスク
        fetch_users().await
    });

    let handle2 = tokio::spawn(async {
        fetch_orders().await
    });

    // 両方の結果を取得
    let (users, orders) = (handle1.await??, handle2.await??);
    println!("Users: {}, Orders: {}", users.len(), orders.len());
    Ok(())
}

// tokio::select! でレース
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

// tokio::select! で最初の応答を採用
async fn fastest_mirror(mirrors: Vec<String>) -> Result<Data, AppError> {
    tokio::select! {
        result = fetch_from(&mirrors[0]) => result,
        result = fetch_from(&mirrors[1]) => result,
        result = fetch_from(&mirrors[2]) => result,
    }
}

// timeout
async fn with_timeout() -> Result<Data, AppError> {
    match timeout(Duration::from_secs(10), fetch_data()).await {
        Ok(Ok(data)) => Ok(data),
        Ok(Err(e)) => Err(AppError::Fetch(e)),
        Err(_) => Err(AppError::Timeout),
    }
}

// バッファ付きチャネル
async fn producer_consumer() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);

    // プロデューサー
    let producer = tokio::spawn(async move {
        for i in 0..1000 {
            tx.send(format!("message {}", i)).await.unwrap();
        }
    });

    // コンシューマー
    let consumer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            process_message(&msg).await;
        }
    });

    let _ = tokio::join!(producer, consumer);
}
```

### 4.3 Stream（非同期イテレータ）

```rust
use tokio_stream::{self as stream, StreamExt};
use futures::stream::{self, Stream};

// Stream の作成と消費
async fn process_stream() {
    let mut stream = stream::iter(vec![1, 2, 3, 4, 5])
        .map(|x| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            x * 2
        })
        .buffered(3); // 3つまで並行処理

    while let Some(value) = stream.next().await {
        println!("Got: {}", value);
    }
}

// カスタム Stream
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

// Stream の合成
async fn merged_streams() {
    let stream1 = stream::iter(vec![1, 3, 5]);
    let stream2 = stream::iter(vec![2, 4, 6]);

    let mut merged = stream::select(stream1, stream2);
    while let Some(value) = merged.next().await {
        println!("{}", value);
    }
}

// 並行制限付きの処理
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

### 4.4 エラーハンドリングと ? 演算子

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

// ? 演算子で簡潔なエラーハンドリング
async fn get_user_with_orders(user_id: &str) -> Result<UserWithOrders, AppError> {
    let user = db::find_user(user_id)
        .await?  // sqlx::Error → AppError::Database
        .ok_or_else(|| AppError::NotFound(user_id.to_string()))?;

    let orders = api::fetch_orders(user_id)
        .await?; // reqwest::Error → AppError::Network

    Ok(UserWithOrders { user, orders })
}

// リトライ
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

## 5. Go（goroutine + channel）

Go は async/await 構文を持たないが、goroutine と channel で同等の機能を実現する。

### 5.1 基本パターン

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Go では全ての関数が「同期的」に見える
// 非同期性は goroutine で実現する
func getUserProfile(ctx context.Context, userID string) (*UserProfile, error) {
    user, err := userRepo.FindByID(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("find user: %w", err)
    }

    // 並行実行は goroutine + channel で
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

### 5.2 errgroup による並行実行

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

// 並行制限付き errgroup
func processItems(ctx context.Context, items []string) error {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(10) // 同時に10件まで

    for _, item := range items {
        item := item // ループ変数のキャプチャ（Go 1.21 以前）
        g.Go(func() error {
            return processItem(ctx, item)
        })
    }

    return g.Wait()
}
```

### 5.3 Context によるキャンセルとタイムアウト

```go
import (
    "context"
    "time"
)

// タイムアウト
func fetchWithTimeout(url string) ([]byte, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err // context.DeadlineExceeded が含まれる可能性
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}

// キャンセル伝播
func longOperation(ctx context.Context) error {
    for i := 0; i < 100; i++ {
        select {
        case <-ctx.Done():
            return ctx.Err() // context.Canceled or DeadlineExceeded
        default:
            // 処理を続行
            if err := doStep(ctx, i); err != nil {
                return err
            }
        }
    }
    return nil
}

// 親コンテキストからのキャンセル
func handler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context() // クライアントが接続を切るとキャンセルされる

    result, err := longOperation(ctx)
    if err != nil {
        if ctx.Err() != nil {
            // クライアントが切断した
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

### 6.1 Task ベースの async/await

```csharp
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

// 基本
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

// ValueTask（値型で軽量、キャッシュヒットが多い場合に有効）
public ValueTask<User?> GetUserAsync(string userId)
{
    if (_cache.TryGetValue(userId, out var cached))
    {
        return ValueTask.FromResult(cached); // ヒープアロケーションなし
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

// 使用例
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

### 6.2 並行実行パターン

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
        Profile = await profileTask,       // 既に完了済み
        Notifications = await notificationsTask,
        Stats = await statsTask,
    };
}

// Task.WhenAny（最初の完了を利用）
public async Task<Data> FetchFromFastestAsync(IEnumerable<string> urls)
{
    var tasks = urls.Select(url => FetchDataAsync(url)).ToList();
    var completed = await Task.WhenAny(tasks);
    return await completed;
}

// SemaphoreSlim で並行制限
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

// IAsyncEnumerable（C# 8.0+）
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

// 消費
await foreach (var user in GetAllUsersAsync())
{
    Console.WriteLine(user.Name);
}
```

### 6.3 ConfigureAwait と同期コンテキスト

```csharp
// UI スレッドでの注意点
// WPF や Windows Forms では SynchronizationContext がある
public async void Button_Click(object sender, EventArgs e)
{
    var data = await FetchDataAsync("/api/data");
    // ↑ デフォルトでは UI スレッドに戻る

    // UI の更新（UI スレッドで実行される）
    textBox.Text = data.ToString();
}

// ライブラリコードでは ConfigureAwait(false) を使う
public async Task<Data> FetchDataLibraryAsync(string url)
{
    var response = await _client.GetAsync(url)
        .ConfigureAwait(false); // スレッドプールで継続

    var content = await response.Content.ReadAsStringAsync()
        .ConfigureAwait(false);

    return JsonSerializer.Deserialize<Data>(content)!;
}

// デッドロック回避
// ❌ 同期メソッドから async を呼ぶとデッドロック
public Data GetDataSync()
{
    // UI スレッドで .Result を呼ぶとデッドロック！
    return FetchDataAsync("/api/data").Result;
}

// ✅ async all the way（全てを async にする）
public async Task<Data> GetDataAsync()
{
    return await FetchDataAsync("/api/data");
}
```

---

## 7. Kotlin

### 7.1 Coroutine の基本

```kotlin
import kotlinx.coroutines.*

// suspend 関数
suspend fun getUserProfile(userId: String): UserProfile {
    val user = userRepo.findById(userId)
        ?: throw NotFoundException("User $userId not found")

    // coroutineScope で並行実行
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

// CoroutineScope とディスパッチャー
fun main() = runBlocking {
    // Dispatchers.IO: I/O操作向け
    val data = withContext(Dispatchers.IO) {
        fetchFromNetwork()
    }

    // Dispatchers.Default: CPU集中タスク向け
    val processed = withContext(Dispatchers.Default) {
        heavyComputation(data)
    }

    println(processed)
}

// 構造化並行性
suspend fun processDashboard(userId: String): Dashboard {
    return coroutineScope {
        val profile = async { getProfile(userId) }
        val notifications = async { getNotifications(userId) }
        val stats = async { getStats(userId) }

        // いずれかが失敗すると、他も自動キャンセル
        Dashboard(
            profile = profile.await(),
            notifications = notifications.await(),
            stats = stats.await()
        )
    }
}
```

### 7.2 Flow（Cold Stream）

```kotlin
import kotlinx.coroutines.flow.*

// Flow の作成
fun fetchUsers(): Flow<User> = flow {
    var page = 0
    while (true) {
        val users = api.getUsers(page++)
        if (users.isEmpty()) break
        users.forEach { emit(it) }
    }
}

// Flow の変換と消費
suspend fun processUsers() {
    fetchUsers()
        .filter { it.isActive }
        .map { enrichUser(it) }
        .buffer(10) // バッファリングで並行化
        .collect { user ->
            println("Processed: ${user.name}")
        }
}

// StateFlow（Hot Stream, UI 向け）
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

### 7.3 キャンセルとタイムアウト

```kotlin
import kotlinx.coroutines.*

// withTimeout
suspend fun fetchWithTimeout(): Data {
    return withTimeout(5000L) { // 5秒タイムアウト
        fetchData()
    }
    // TimeoutCancellationException がスローされる
}

// withTimeoutOrNull（例外ではなく null を返す）
suspend fun safeFetch(): Data? {
    return withTimeoutOrNull(5000L) {
        fetchData()
    }
}

// キャンセルの協調
suspend fun cancellableOperation() {
    for (i in 0..1000) {
        // キャンセルチェック
        ensureActive() // CancellationException をスロー

        // または yield() でサスペンドポイントを挿入
        yield()

        // 処理
        processItem(i)
    }
}

// Job のキャンセル
fun main() = runBlocking {
    val job = launch {
        repeat(1000) { i ->
            println("Processing $i...")
            delay(100)
        }
    }

    delay(500)
    job.cancelAndJoin() // キャンセルして完了を待つ
    println("Cancelled")
}
```

---

## 8. Swift

### 8.1 Structured Concurrency

```swift
import Foundation

// async 関数
func getUserProfile(userId: String) async throws -> UserProfile {
    let user = try await userRepo.findById(userId)

    // async let で並行実行（構造化並行性）
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

// Task のキャンセル
func cancellableOperation() async throws {
    for i in 0..<1000 {
        // キャンセルチェック
        try Task.checkCancellation()

        await processItem(i)
    }
}

// Actor（データ競合の防止）
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

// 消費
func processAllPages() async {
    for await items in fetchPages(url: apiURL) {
        for item in items {
            print(item)
        }
    }
}

// URLSession の bytes（ストリーミング）
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

## 9. 効率的なパターン

### 9.1 依存関係グラフに基づく実行

```typescript
// パターン1: 早期 await（依存関係がある場合）
async function orderPipeline(userId: string) {
  const user = await getUser(userId);         // まず user が必要
  const cart = await getCart(user.cartId);      // user に依存
  const total = calculateTotal(cart.items);     // 同期処理
  const payment = await processPayment(total);  // total に依存
  return payment;
}

// パターン2: 独立タスクの並行実行
async function dashboardData(userId: string) {
  // 独立したデータを並行取得
  const [profile, notifications, stats, feed] = await Promise.all([
    getProfile(userId),
    getNotifications(userId),
    getStats(userId),
    getFeed(userId),
  ]);
  return { profile, notifications, stats, feed };
}

// パターン3: 段階的な並行実行
async function complexPipeline(userId: string) {
  // Stage 1: user を取得
  const user = await getUser(userId);

  // Stage 2: user に依存する3つを並行
  const [orders, reviews, wishlist] = await Promise.all([
    getOrders(user.id),
    getReviews(user.id),
    getWishlist(user.id),
  ]);

  // Stage 3: orders に依存する処理を並行
  const orderDetails = await Promise.all(
    orders.map(order => getOrderDetails(order.id))
  );

  return { user, orders: orderDetails, reviews, wishlist };
}

// パターン4: 依存関係グラフの自動解決
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
      // 依存タスクを先に実行
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

// 使用例
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

### 9.2 キャッシュパターン

```typescript
// 非同期キャッシュ（重複リクエスト防止）
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

    // キャッシュヒット
    const cached = this.cache.get(cacheKey);
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }

    // 同じキーのリクエストが進行中ならそれを待つ（dedup）
    const pending = this.pending.get(cacheKey);
    if (pending) {
      return pending;
    }

    // 新規フェッチ
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

// 使用例
const userCache = new AsyncCache<string, User>(
  (userId) => fetchUser(userId),
  (key) => key,
  5 * 60 * 1000 // 5分
);

// 同時に同じユーザーをリクエストしてもAPI呼び出しは1回
const [user1, user2] = await Promise.all([
  userCache.get("user-123"),
  userCache.get("user-123"),
]);
```

### 9.3 バッチ処理とデバウンス

```typescript
// DataLoader パターン（N+1 問題の解決）
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
        // マイクロタスクでバッチ実行をスケジュール
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

// 使用例
const userLoader = new DataLoader<string, User>(
  async (ids) => {
    // SELECT * FROM users WHERE id IN (...)
    const users = await db.query(
      `SELECT * FROM users WHERE id = ANY($1)`, [ids]
    );
    return new Map(users.map(u => [u.id, u]));
  }
);

// 個別に呼んでもバッチ化される
async function resolveComment(comment: Comment) {
  const author = await userLoader.load(comment.authorId); // ┐
  const editor = await userLoader.load(comment.editorId); // ┤ 1回のSQLに
  return { ...comment, author, editor };                   // ┘
}

// 非同期デバウンス
function asyncDebounce<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  ms: number
): T {
  let timeoutId: NodeJS.Timeout;
  let pendingResolve: ((value: any) => void) | null = null;
  let pendingReject: ((error: any) => void) | null = null;

  return ((...args: any[]) => {
    return new Promise((resolve, reject) => {
      // 前回のリクエストを中断
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

### 9.4 for-await-of パターン

```typescript
// 非同期ジェネレーター
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

// パイプライン: 変換付き非同期イテレータ
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

// パイプライン使用例
const activeUsers = take(
  filter(
    map(
      fetchPages("/api/users"),
      page => page  // ページからユーザー配列を取得
    ),
    users => users.length > 0
  ),
  10 // 最初の10ページまで
);

for await (const users of activeUsers) {
  await processUsers(users);
}
```

---

## 10. よくある間違いとアンチパターン

### 10.1 不必要な逐次実行

```typescript
// ❌ await の逐次実行（並行可能なのに）
const users = await getUsers();
const orders = await getOrders();
const products = await getProducts();
// → 独立なのに直列実行している

// ✅ 並行実行
const [users, orders, products] = await Promise.all([
  getUsers(), getOrders(), getProducts(),
]);
```

### 10.2 ループ内の await

```typescript
// ❌ ループ内の await
for (const id of ids) {
  const data = await fetch(`/api/${id}`); // 1件ずつ...
}

// ✅ 並行実行
const results = await Promise.all(
  ids.map(id => fetch(`/api/${id}`))
);

// ✅ 制限付き並行（大量のリクエストの場合）
const semaphore = new Semaphore(10);
const results = await Promise.all(
  ids.map(id => semaphore.run(() => fetch(`/api/${id}`)))
);
```

### 10.3 async/await と .then() の混在

```typescript
// ❌ async 関数内で .then() を混在
async function mixed() {
  return fetchData().then(data => data.value); // 混在
}
// ✅ 統一
async function clean() {
  const data = await fetchData();
  return data.value;
}
```

### 10.4 不要な async

```typescript
// ❌ 不要な async（Promise をそのまま返せばいい）
async function wrapper() {
  return await fetchData(); // 不要な async/await
}

// ✅ そのまま返す（ただしエラースタックトレースに注意）
function wrapper() {
  return fetchData();
}

// 注意: try/catch がある場合は async/await が必要
async function withErrorHandling() {
  try {
    return await fetchData(); // ここでは await が必要
  } catch (error) {
    return fallbackData;
  }
}
```

### 10.5 エラーハンドリングの漏れ

```typescript
// ❌ unhandled rejection
async function firAndForget() {
  fetchData(); // await も catch もない！
}

// ✅ fire-and-forget でもエラーハンドリング
async function safeFireAndForget() {
  fetchData().catch(error => {
    logger.error("Background task failed", error);
  });
}

// ❌ Promise.all の部分的失敗で全体が失敗
const results = await Promise.all([
  fetchA(), // 成功
  fetchB(), // 失敗 → 全体が reject
  fetchC(), // 成功だが結果が失われる
]);

// ✅ allSettled で部分的な成功を許容
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

### 10.6 メモリリーク

```typescript
// ❌ クリーンアップ忘れ
class DataFetcher {
  private intervalId?: NodeJS.Timeout;

  start() {
    this.intervalId = setInterval(async () => {
      const data = await fetchData();
      this.processData(data);
    }, 1000);
  }
  // stop() が呼ばれないとリーク
}

// ✅ 適切なクリーンアップ
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
          return; // 正常なキャンセル
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

## 11. テスト

### 11.1 JavaScript/TypeScript でのテスト

```typescript
import { describe, it, expect, vi } from "vitest";

// 基本的な async テスト
describe("UserService", () => {
  it("should fetch user profile", async () => {
    const service = new UserService(mockRepo);
    const profile = await service.getUserProfile("user-123");

    expect(profile.user.id).toBe("user-123");
    expect(profile.orders).toHaveLength(3);
  });

  // エラーケース
  it("should throw on missing user", async () => {
    const service = new UserService(emptyRepo);

    await expect(
      service.getUserProfile("nonexistent")
    ).rejects.toThrow("User not found");
  });

  // タイムアウトのテスト
  it("should timeout after 5 seconds", async () => {
    vi.useFakeTimers();

    const promise = fetchWithTimeout("/api/slow", 5000);

    // 時間を進める
    vi.advanceTimersByTime(5000);

    await expect(promise).rejects.toThrow("Timeout");

    vi.useRealTimers();
  });

  // 並行実行のテスト
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
    // 並行実行の確認: 両方が先に開始される
    expect(callOrder[0]).toBe("A-start");
    expect(callOrder[1]).toBe("B-start");
  });

  // リトライのテスト
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

  // AbortController のテスト
  it("should cancel on abort", async () => {
    const controller = new AbortController();

    const promise = fetchWithCancel("/api/data", controller.signal);
    controller.abort();

    await expect(promise).rejects.toThrow("AbortError");
  });
});
```

### 11.2 Python でのテスト

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

# pytest-asyncio を使用
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

### 11.3 Rust でのテスト

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

## 12. デバッグ技術

### 12.1 AsyncLocalStorage（Node.js）

```typescript
import { AsyncLocalStorage } from "node:async_hooks";

// リクエスト追跡
const requestContext = new AsyncLocalStorage<{
  requestId: string;
  startTime: number;
}>();

// ミドルウェアでコンテキストを設定
app.use((req, res, next) => {
  const context = {
    requestId: crypto.randomUUID(),
    startTime: Date.now(),
  };
  requestContext.run(context, next);
});

// どこからでもコンテキストにアクセス
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

### 12.2 非同期処理のプロファイリング

```typescript
// 実行時間の計測
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

// 使用
const user = await withTiming("getUser", () => getUser("123"));

// 並行実行の可視化
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

  // タイムライン出力
  console.log("=== Execution Timeline ===");
  for (const entry of timeline.sort((a, b) => a.start - b.start)) {
    const bar = " ".repeat(Math.floor(entry.start / 10))
      + "=".repeat(Math.floor((entry.end - entry.start) / 10));
    console.log(`${entry.name.padEnd(20)} |${bar}| ${(entry.end - entry.start).toFixed(1)}ms`);
  }

  return Object.fromEntries(results);
}

// 使用
await traceParallel({
  users: () => fetchUsers(),
  orders: () => fetchOrders(),
  products: () => fetchProducts(),
});
// 出力:
// === Execution Timeline ===
// users                |====      | 45.2ms
// orders               |========  | 82.1ms
// products             |=====     | 53.7ms
```

### 12.3 Unhandled Rejection の検出

```typescript
// Node.js でのグローバルハンドラー
process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
  // アプリケーションの状態をログに記録
  // 本番環境ではプロセスを終了させることを検討
  process.exit(1);
});

// ブラウザ
window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled rejection:", event.reason);
  event.preventDefault(); // デフォルトのコンソール出力を抑制
  // エラートラッキングサービスに送信
  errorTracker.captureException(event.reason);
});

// ESLint ルールで未処理 Promise を検出
// .eslintrc.json
// {
//   "rules": {
//     "no-floating-promises": "error",  // @typescript-eslint
//     "require-await": "warn"
//   }
// }
```

---

## まとめ

### 言語間比較

| 言語 | async 構文 | 並行実行 | ランタイム | キャンセル |
|------|-----------|---------|-----------|-----------|
| JS/TS | async/await | Promise.all | イベントループ | AbortController |
| Python | async/await | asyncio.gather | asyncio | Task.cancel() |
| Rust | async/await | tokio::join! | tokio/async-std | tokio::select! |
| Go | goroutine | go + channel | ランタイム内蔵 | context.Context |
| C# | async/await | Task.WhenAll | CLR | CancellationToken |
| Kotlin | suspend | coroutineScope | Dispatchers | Job.cancel() |
| Swift | async/await | async let | Swift Runtime | Task.cancel() |

### 設計方針

| パターン | 使うべき場面 | 注意点 |
|---------|-------------|--------|
| 逐次 await | 依存関係がある処理 | 不要な逐次実行を避ける |
| Promise.all | 独立した複数の処理 | 1つ失敗で全体が失敗 |
| Promise.allSettled | 部分的成功を許容 | 結果の分類が必要 |
| セマフォ | 大量の並行制限 | リソース枯渇の防止 |
| for-await-of | ストリーム処理 | バックプレッシャーに注意 |
| DataLoader | N+1 問題の解決 | バッチウィンドウの設計 |
| サーキットブレーカー | 不安定なサービス呼び出し | 状態管理の複雑さ |

---

## 次に読むべきガイド
-> [[03-reactive-streams.md]] -- Reactive Streams

---

## 参考文献
1. MDN Web Docs. "async function."
2. Python Documentation. "Coroutines and Tasks."
3. Tokio Documentation. "Tutorial."
4. Kotlin Documentation. "Coroutines Guide."
5. Swift Documentation. "Concurrency."
6. C# Documentation. "Asynchronous programming with async and await."
7. Go Blog. "Go Concurrency Patterns."
8. Node.js Documentation. "Async Hooks."
