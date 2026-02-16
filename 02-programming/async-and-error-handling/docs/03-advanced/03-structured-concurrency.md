# 構造化並行性

> 構造化並行性は「並行処理のライフタイムを構造的に管理する」パラダイム。Kotlin coroutines、Swift structured concurrency、Python TaskGroup を通じて、安全な並行プログラミングを実現する。

## この章で学ぶこと

- [ ] 構造化並行性の原則を理解する
- [ ] 非構造化並行性の問題を把握する
- [ ] 各言語での実装を学ぶ
- [ ] キャンセル伝播の仕組みを理解する
- [ ] エラーハンドリングとの統合を把握する
- [ ] 実務での適用パターンを習得する

---

## 1. 構造化並行性とは

### 1.1 基本概念

```
非構造化並行性（従来）:
  → タスクを「起動して放置」
  → 親が終了しても子タスクが残る
  → エラーが子タスクで握りつぶされる
  → リソースリーク

  function process() {
    startBackgroundTask(); // 起動して放置
    startAnotherTask();    // 誰がこの寿命を管理する？
  } // process 終了後もタスクが動き続ける

構造化並行性:
  → 子タスクは親のスコープ内で完了する
  → 親は全ての子タスクの完了を待つ
  → 1つの子タスクが失敗したら、他もキャンセル
  → リソースリークなし

  async function process() {
    await Promise.all([  // 全ての子タスクの完了を待つ
      task1(),
      task2(),
    ]);
  } // ここで全タスクが完了していることが保証
```

### 1.2 構造化プログラミングとの対比

```
構造化プログラミング（1968, Dijkstra）:
  → goto を排除し、制御フローを構造化
  → if/else, while, for でスコープを明確に
  → コードの開始点と終了点が明確

  非構造化: goto label;  // どこに飛ぶかわからない
  構造化:   if (...) { ... }  // スコープが明確

構造化並行性（2018, Elizarov, Syme）:
  → 「起動して放置」を排除し、並行処理のライフタイムを構造化
  → タスクのスコープを明確に
  → タスクの開始点と終了点が明確

  非構造化: Task.run(() => ...) // どこで終わるかわからない
  構造化:   async with TaskGroup() { ... } // スコープ内で完了保証

共通する原則:
  → 制御フローの明確化
  → スコープベースのリソース管理
  → 可読性とデバッグ容易性の向上
```

### 1.3 非構造化並行性の問題

```
問題1: リソースリーク
  function startProcessing() {
    setTimeout(() => {
      // このコールバックは誰が管理する？
      // startProcessing のスコープを超えて生存
      processData();
    }, 5000);
  }

問題2: エラーの握りつぶし
  function fetchAll() {
    fetch('/api/users');     // エラーが起きても誰も catch しない
    fetch('/api/products');  // 同上
  }

問題3: キャンセルの困難
  function loadDashboard() {
    const p1 = fetch('/api/users');
    const p2 = fetch('/api/stats');
    // ユーザーがページ遷移した場合、p1とp2をキャンセルするのが困難
    // 個別にAbortControllerを管理する必要がある
  }

問題4: デバッグの困難
  → 非同期タスクのスタックトレースが途切れる
  → 親子関係が不明確
  → どのタスクがどの時点で動いているか追跡困難
```

---

## 2. Kotlin Coroutines

### 2.1 coroutineScope: 構造化並行性の基本

```kotlin
import kotlinx.coroutines.*

// coroutineScope: 構造化並行性のスコープ
suspend fun loadDashboard(): Dashboard = coroutineScope {
    // 子 coroutine を起動
    val userDeferred = async { fetchUser() }
    val ordersDeferred = async { fetchOrders() }
    val statsDeferred = async { fetchStats() }

    // 全ての結果を待つ
    Dashboard(
        user = userDeferred.await(),
        orders = ordersDeferred.await(),
        stats = statsDeferred.await(),
    )
    // coroutineScope を抜けるとき、全子coroutineが完了していることが保証
    // 1つが例外を投げたら、他もキャンセルされる
}
```

### 2.2 supervisorScope: 子のエラーを独立に処理

```kotlin
// supervisorScope: 子のエラーが他に影響しない
suspend fun loadDashboardResilient(): Dashboard = supervisorScope {
    val user = async { fetchUser() }
    val orders = async {
        try { fetchOrders() }
        catch (e: Exception) { emptyList() } // フォールバック
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

// coroutineScope vs supervisorScope の使い分け
//
// coroutineScope:
//   → 全タスクが成功する必要がある場合
//   → 1つ失敗 → 全てキャンセル
//   → 例: トランザクション的な処理
//
// supervisorScope:
//   → 個々のタスクが独立している場合
//   → 1つ失敗しても他は続行
//   → 例: ダッシュボードの各コンポーネント読み込み
```

### 2.3 Kotlin のキャンセル処理

```kotlin
import kotlinx.coroutines.*

// キャンセルの基本
suspend fun processWithCancellation() {
    val job = CoroutineScope(Dispatchers.Default).launch {
        try {
            repeat(1000) { i ->
                println("Processing $i...")
                delay(100) // キャンセルポイント
            }
        } catch (e: CancellationException) {
            println("Cancelled!")
            // クリーンアップ処理
        } finally {
            // リソース解放
            withContext(NonCancellable) {
                // キャンセル後でもこのブロック内は実行される
                cleanup()
            }
        }
    }

    delay(500)
    job.cancel() // キャンセル要求
    job.join()   // キャンセル完了を待つ
}

// キャンセル対応のベストプラクティス
suspend fun downloadFile(url: String, dest: File) = coroutineScope {
    val response = httpClient.get(url)
    val channel = response.bodyAsChannel()

    dest.outputStream().use { output ->
        val buffer = ByteArray(8192)
        while (true) {
            // ensureActive() で定期的にキャンセルチェック
            ensureActive()

            val bytesRead = channel.readAvailable(buffer)
            if (bytesRead == -1) break

            output.write(buffer, 0, bytesRead)
        }
    }
}

// タイムアウト付き処理
suspend fun fetchWithTimeout(): Result {
    return withTimeout(5000) { // 5秒タイムアウト
        fetchData()
    }
    // タイムアウト時は TimeoutCancellationException が発生
}

// タイムアウトで null を返す
suspend fun fetchWithTimeoutOrNull(): Result? {
    return withTimeoutOrNull(5000) {
        fetchData()
    }
    // タイムアウト時は null を返す（例外なし）
}
```

### 2.4 Kotlin の高度なパターン

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

// Fan-out: 1つのプロデューサー、複数のコンシューマー
suspend fun fanOutExample() = coroutineScope {
    val channel = produce {
        repeat(100) { send(it) }
    }

    // 5つのワーカーで処理
    repeat(5) { workerId ->
        launch {
            for (item in channel) {
                println("Worker $workerId processing $item")
                processItem(item)
            }
        }
    }
}

// Fan-in: 複数のプロデューサー、1つのコンシューマー
suspend fun fanInExample() = coroutineScope {
    val results = Channel<ProcessResult>()

    // 複数のプロデューサー
    val sources = listOf("api-1", "api-2", "api-3")
    sources.forEach { source ->
        launch {
            val data = fetchFromSource(source)
            results.send(ProcessResult(source, data))
        }
    }

    // 全結果を収集
    launch {
        val allResults = mutableListOf<ProcessResult>()
        repeat(sources.size) {
            allResults.add(results.receive())
        }
        processAllResults(allResults)
        results.close()
    }
}

// 競争パターン: 最初の成功を返す
suspend fun raceExample(): String = coroutineScope {
    select<String> {
        async { fetchFromPrimary() }.onAwait { it }
        async { fetchFromSecondary() }.onAwait { it }
        async { fetchFromTertiary() }.onAwait { it }
    }
    // 最初に完了したものを返す、残りはキャンセル
}

// バックプレッシャー対応の並行処理
fun processWithBackpressure(items: List<Item>): Flow<Result> = flow {
    coroutineScope {
        val semaphore = Semaphore(10) // 同時実行数制限
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

// エラー回復を組み込んだ構造化並行性
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

### 3.1 async let: 静的な並行性

```swift
// Swift: async let で静的な数のタスクを並行実行
func loadDashboard() async throws -> Dashboard {
    async let user = fetchUser()           // 並行開始
    async let orders = fetchOrders()       // 並行開始
    async let stats = fetchStats()         // 並行開始

    return try await Dashboard(
        user: user,
        orders: orders,
        stats: stats,
    )
    // 全ての async let の完了を待つ
    // 1つが throw したら、他は自動キャンセル
}
```

### 3.2 TaskGroup: 動的な並行性

```swift
// TaskGroup: 動的な数のタスク
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
    // TaskGroup スコープ外 = 全タスク完了保証
}

// 並行数制限付きTaskGroup
func processWithConcurrencyLimit(
    items: [Item],
    maxConcurrent: Int = 5
) async throws -> [Result] {
    try await withThrowingTaskGroup(of: Result.self) { group in
        var results: [Result] = []
        var iterator = items.makeIterator()
        var inFlight = 0

        // 初期バッチを投入
        while inFlight < maxConcurrent, let item = iterator.next() {
            group.addTask { try await processItem(item) }
            inFlight += 1
        }

        // 1つ完了するたびに次を投入
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

### 3.3 Swift のキャンセル処理

```swift
// キャンセルの確認と対応
func downloadFile(url: URL) async throws -> Data {
    var data = Data()
    let (bytes, _) = try await URLSession.shared.bytes(from: url)

    for try await byte in bytes {
        // 定期的にキャンセルチェック
        try Task.checkCancellation()
        data.append(byte)
    }

    return data
}

// キャンセル対応のクリーンアップ
func processWithCleanup() async throws {
    let tempFile = createTempFile()

    do {
        try await longRunningProcess(tempFile)
    } catch is CancellationError {
        // キャンセル時のクリーンアップ
        try? FileManager.default.removeItem(at: tempFile)
        throw CancellationError()
    }
}

// withTaskCancellationHandler: キャンセルハンドラー
func fetchData() async throws -> Data {
    let handle = startNetworkRequest()

    return try await withTaskCancellationHandler {
        // メイン処理
        try await handle.result()
    } onCancel: {
        // キャンセル時にネットワークリクエストを中止
        handle.cancel()
    }
}

// タイムアウトの実装
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
        // 最初に完了したものを返す
        let result = try await group.next()!
        group.cancelAll() // 残りをキャンセル
        return result
    }
}
```

### 3.4 Actor: データ競合の防止

```swift
// Actor: スレッドセーフなデータアクセス
actor UserCache {
    private var cache: [String: User] = [:]
    private var inFlightRequests: [String: Task<User, Error>] = [:]

    func getUser(id: String) async throws -> User {
        // キャッシュヒット
        if let cached = cache[id] {
            return cached
        }

        // 同じユーザーのリクエストが進行中なら待つ
        if let existing = inFlightRequests[id] {
            return try await existing.value
        }

        // 新しいリクエストを開始
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

// GlobalActor: 特定のコンテキストでの実行保証
@globalActor
actor DatabaseActor {
    static let shared = DatabaseActor()
}

@DatabaseActor
class DatabaseManager {
    private var connection: Connection?

    func query(_ sql: String) async throws -> [Row] {
        // DatabaseActor のコンテキストで実行される
        // 自動的にスレッドセーフ
        guard let conn = connection else {
            throw DatabaseError.notConnected
        }
        return try await conn.execute(sql)
    }
}

// Sendable プロトコル: 並行安全な型の保証
struct UserData: Sendable {
    let id: String
    let name: String
    let email: String
}

// @Sendable クロージャ
func processInBackground(_ data: UserData) {
    Task.detached { @Sendable in
        // data は Sendable なので安全に渡せる
        await processUser(data)
    }
}
```

---

## 4. Python TaskGroup（3.11+）

### 4.1 基本的な使い方

```python
import asyncio

# Python 3.11+: TaskGroup
async def load_dashboard():
    async with asyncio.TaskGroup() as tg:
        user_task = tg.create_task(fetch_user())
        orders_task = tg.create_task(fetch_orders())
        stats_task = tg.create_task(fetch_stats())

    # async with を抜けると全タスク完了
    # 1つが例外 → 他もキャンセル → ExceptionGroup が送出
    return Dashboard(
        user=user_task.result(),
        orders=orders_task.result(),
        stats=stats_task.result(),
    )
```

### 4.2 ExceptionGroup のハンドリング

```python
# ExceptionGroup のハンドリング（Python 3.11+）
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

# ExceptionGroup の構造
# ExceptionGroup は複数の例外をラップする
# except* は特定の型の例外だけを選択的にキャッチ
# 残りの例外は再送出される

# 複数の except* ブロック
async def handle_multiple_errors():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task_that_may_raise_value_error())
            tg.create_task(task_that_may_raise_type_error())
            tg.create_task(task_that_may_raise_io_error())
    except* ValueError as eg:
        # ValueError だけを処理
        for exc in eg.exceptions:
            log_validation_error(exc)
    except* (TypeError, IOError) as eg:
        # TypeError と IOError をまとめて処理
        for exc in eg.exceptions:
            log_system_error(exc)
    # 上記で処理されなかった例外型は再送出される
```

### 4.3 キャンセル処理

```python
import asyncio
from contextlib import asynccontextmanager


# タイムアウト付きTaskGroup
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


# キャンセル対応のタスク
async def cancellable_download(url: str, dest: str) -> None:
    """キャンセル対応のダウンロード処理"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            with open(dest, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    # asyncio.CancelledError は自動的に伝播
                    f.write(chunk)


# shield: キャンセルから保護
async def critical_operation():
    """クリティカルな操作をキャンセルから保護"""
    # shield で囲むと、外部からのキャンセルが内部に伝播しない
    result = await asyncio.shield(save_to_database(data))
    return result


# キャンセルハンドリングのパターン
async def process_with_cleanup():
    """キャンセル時にクリーンアップを実行"""
    resource = await acquire_resource()
    try:
        await long_running_process(resource)
    except asyncio.CancelledError:
        # キャンセル時のクリーンアップ
        await cleanup_resource(resource)
        raise  # CancelledError は必ず再送出
    finally:
        await release_resource(resource)
```

### 4.4 高度なパターン

```python
import asyncio
from typing import TypeVar, Callable, Awaitable, AsyncIterator
from dataclasses import dataclass

T = TypeVar('T')
R = TypeVar('R')


# 並行数制限付きバッチ処理
async def map_concurrent(
    items: list[T],
    func: Callable[[T], Awaitable[R]],
    max_concurrent: int = 10,
) -> list[R]:
    """アイテムを並行数制限付きで処理"""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[R] = [None] * len(items)  # type: ignore

    async def process_with_limit(index: int, item: T) -> None:
        async with semaphore:
            results[index] = await func(item)

    async with asyncio.TaskGroup() as tg:
        for i, item in enumerate(items):
            tg.create_task(process_with_limit(i, item))

    return results


# 使用例
async def main():
    urls = [f"https://api.example.com/items/{i}" for i in range(100)]
    results = await map_concurrent(
        urls,
        fetch_url,
        max_concurrent=20,
    )


# レース: 最初の成功を返す
async def race(*coros: Awaitable[T]) -> T:
    """複数のコルーチンのうち、最初に成功したものを返す"""
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
                pass  # 失敗は無視

        for coro in coros:
            tg.create_task(run_and_signal(coro))

        # 注意: TaskGroup は全タスク完了を待つ
        # raceパターンにはTaskGroupは不向き
        # asyncio.wait(return_when=FIRST_COMPLETED) を使う方が適切

    if result_holder:
        return result_holder[0]
    raise RuntimeError("All tasks failed")


# asyncio.wait を使った適切なレース実装
async def race_proper(*coros: Awaitable[T]) -> T:
    """asyncio.wait で最初の完了を待つ"""
    tasks = [asyncio.ensure_future(c) for c in coros]

    try:
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )

        # 残りのタスクをキャンセル
        for task in pending:
            task.cancel()

        # キャンセル完了を待つ
        if pending:
            await asyncio.wait(pending)

        # 最初に完了したタスクの結果を返す
        result_task = done.pop()
        return result_task.result()

    except Exception:
        # エラー時は全タスクをキャンセル
        for task in tasks:
            task.cancel()
        raise


# パイプライン: ステージごとに処理
async def pipeline_example():
    """マルチステージパイプライン"""
    queue1: asyncio.Queue[RawData] = asyncio.Queue(maxsize=100)
    queue2: asyncio.Queue[ProcessedData] = asyncio.Queue(maxsize=100)

    async def stage1_fetch():
        """ステージ1: データ取得"""
        for url in urls:
            data = await fetch_data(url)
            await queue1.put(data)
        await queue1.put(None)  # 終了シグナル

    async def stage2_process():
        """ステージ2: データ処理"""
        while True:
            data = await queue1.get()
            if data is None:
                await queue2.put(None)
                break
            processed = await process_data(data)
            await queue2.put(processed)

    async def stage3_save():
        """ステージ3: データ保存"""
        while True:
            data = await queue2.get()
            if data is None:
                break
            await save_data(data)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(stage1_fetch())
        tg.create_task(stage2_process())
        tg.create_task(stage3_save())


# 構造化並行性でのリソース管理
@asynccontextmanager
async def managed_workers(
    num_workers: int,
    work_queue: asyncio.Queue,
    handler: Callable,
):
    """ワーカープールのライフサイクル管理"""
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


# 使用例
async def process_with_workers():
    queue: asyncio.Queue = asyncio.Queue()

    # キューにアイテムを追加
    for item in items:
        await queue.put(item)

    async with managed_workers(5, queue, process_item):
        await queue.join()  # 全アイテムの処理完了を待つ
```

---

## 5. JavaScript/TypeScript の構造化並行性

### 5.1 Promise.all — 基本的な並行処理

```typescript
// Promise.all: 全タスクの完了を待つ（部分的な構造化並行性）
async function loadDashboard(): Promise<Dashboard> {
  const [user, orders, stats] = await Promise.all([
    fetchUser(),
    fetchOrders(),
    fetchStats(),
  ]);

  return { user, orders, stats };
}

// 制限事項:
// 1. 1つ失敗すると他も即座にrejectされるが、キャンセルはされない
// 2. 残りのPromiseはバックグラウンドで実行を続ける
// 3. 明示的なキャンセル機構がない
```

### 5.2 Promise.allSettled — エラー耐性

```typescript
// Promise.allSettled: 全タスクの完了を待つ（成否を問わず）
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

// ヘルパー関数で使いやすく
function extractResult<T>(result: PromiseSettledResult<T>): T | null {
  return result.status === 'fulfilled' ? result.value : null;
}

function extractResults<T extends readonly unknown[]>(
  results: { [K in keyof T]: PromiseSettledResult<T[K]> },
): { [K in keyof T]: T[K] | null } {
  return results.map(extractResult) as any;
}
```

### 5.3 AbortController による擬似的な構造化並行性

```typescript
// AbortController を使ったキャンセル対応の並行処理
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
      // 残りのタスクの完了を待つ
      await Promise.allSettled(this.tasks);
      return result;
    } catch (error) {
      // エラー時は全タスクをキャンセル
      this.controller.abort();
      // キャンセルの完了を待つ
      await Promise.allSettled(this.tasks);
      throw error;
    }
  }

  cancel(reason?: string): void {
    this.controller.abort(reason);
  }
}

// 使用例
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

// タイムアウト付きスコープ
async function loadWithTimeout(): Promise<Dashboard> {
  const scope = new StructuredScope();

  // タイムアウトでキャンセル
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

### 5.4 並行数制限付き処理

```typescript
// セマフォベースの並行数制限
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

// 並行数制限付き map
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

// 使用例
const results = await mapConcurrent(
  urls,
  async (url) => {
    const response = await fetch(url);
    return response.json();
  },
  5, // 最大5並行
);
```

---

## 6. Rust の構造化並行性

### 6.1 tokio::select! マクロ

```rust
use tokio::time::{sleep, Duration};

// select! で最初の完了を待つ
async fn fetch_with_timeout() -> Result<Data, Error> {
    tokio::select! {
        result = fetch_data() => result,
        _ = sleep(Duration::from_secs(5)) => {
            Err(Error::Timeout)
        }
    }
    // 最初に完了した方の結果を返す
    // もう一方はキャンセルされる（Future がドロップ）
}

// 複数のソースからの受信
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

### 6.2 tokio::spawn と JoinSet

```rust
use tokio::task::JoinSet;

// JoinSet: 構造化された並行タスクの管理
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

// JoinSet + 並行数制限
async fn process_with_limit(
    items: Vec<Item>,
    max_concurrent: usize,
) -> Vec<ProcessResult> {
    let mut set = JoinSet::new();
    let mut results = Vec::new();
    let mut iter = items.into_iter();

    // 初期バッチを投入
    for _ in 0..max_concurrent {
        if let Some(item) = iter.next() {
            set.spawn(async move { process_item(item).await });
        }
    }

    // 1つ完了するたびに次を投入
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

// スコープ付きタスク（Rust 特有）
// tokio::task::LocalSet を使ったローカルタスク管理
async fn scoped_tasks() {
    let local = tokio::task::LocalSet::new();

    local.run_until(async {
        let handle1 = tokio::task::spawn_local(async {
            // ローカルタスク（Send不要）
            process_local_data().await
        });

        let handle2 = tokio::task::spawn_local(async {
            process_another_local_data().await
        });

        let (r1, r2) = tokio::join!(handle1, handle2);
        println!("Results: {:?}, {:?}", r1, r2);
    }).await;
    // LocalSet のスコープを抜けると全ローカルタスクが完了
}
```

---

## 7. 構造化並行性の原則

### 7.1 3つの核心原則

```
3つの原則:

  1. 子タスクは親のスコープ内で生存
     → 親が終了 = 子も終了（リーク防止）
     → タスクのライフタイムがスコープと一致
     → デバッグ時にタスクの親子関係が明確

  2. エラーの伝播
     → 子のエラーは親に伝播する
     → 握りつぶされない
     → ExceptionGroup（Python）で複数エラーを扱える

  3. キャンセルの伝播
     → 親がキャンセルされたら子もキャンセル
     → 1つの子が失敗したら兄弟もキャンセル（coroutineScope）
     → キャンセルは協調的（cooperative）

メリット:
  ✓ リソースリーク防止
  ✓ エラーの確実な処理
  ✓ コードの可読性（スコープが明確）
  ✓ デバッグの容易さ
  ✓ テスタビリティの向上
  ✓ 推論の容易さ（関数の終了 = 全子タスクの終了）
```

### 7.2 キャンセルの協調性

```
キャンセルは「要求」であり「強制」ではない:

  協調的キャンセル:
    → キャンセル要求を受け取ったタスクが自発的に停止
    → タスクは安全な停止ポイントでキャンセルをチェック
    → クリーンアップの機会が与えられる

  各言語のキャンセルポイント:
    Kotlin: delay(), yield(), ensureActive(), suspend関数
    Swift:  Task.checkCancellation(), await
    Python: await（asyncio.CancelledError が送出）
    Rust:   Future の poll が Pending を返す時

  キャンセル時のベストプラクティス:
    1. CancelledError/CancellationException は再送出する
    2. finally ブロックでリソースを解放する
    3. クリティカルセクションはキャンセルから保護する
       → Kotlin: withContext(NonCancellable)
       → Python: asyncio.shield()
    4. 定期的にキャンセルチェックを行う
```

### 7.3 設計パターンの比較

```
パターン1: All or Nothing（全部成功 or 全部失敗）
  → Kotlin: coroutineScope
  → Swift:  withThrowingTaskGroup
  → Python: asyncio.TaskGroup
  → 用途: トランザクション的な処理、全データが必要な場合

パターン2: Best Effort（できるだけ成功）
  → Kotlin: supervisorScope
  → Swift:  withTaskGroup（エラーを個別ハンドリング）
  → Python: TaskGroup + except*
  → JS/TS: Promise.allSettled
  → 用途: ダッシュボード、部分的な結果でOKの場合

パターン3: First Success（最初の成功を採用）
  → Kotlin: select
  → Swift:  TaskGroup + cancelAll
  → Python: asyncio.wait(FIRST_COMPLETED)
  → JS/TS: Promise.race
  → 用途: ヘッジリクエスト、マルチソース取得

パターン4: Fan-Out/Fan-In
  → 複数のプロデューサーとコンシューマー
  → チャネルやキューを組み合わせ
  → 用途: パイプライン処理、並列データ処理
```

---

## 8. 実務での適用パターン

### 8.1 マイクロサービスの並行呼び出し

```typescript
// BFF（Backend for Frontend）パターンでの並行API呼び出し
class DashboardBFF {
  async getDashboard(userId: string): Promise<DashboardResponse> {
    const [
      userResult,
      ordersResult,
      notificationsResult,
      recommendationsResult,
    ] = await Promise.allSettled([
      // 必須: ユーザー情報
      this.userService.getUser(userId),
      // 必須: 注文履歴
      this.orderService.getOrders(userId),
      // オプション: 通知（失敗しても可）
      this.notificationService.getUnread(userId),
      // オプション: レコメンド（失敗しても可）
      this.recommendationService.getForUser(userId),
    ]);

    // 必須データのチェック
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

### 8.2 バッチ処理

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
    """構造化並行性を使ったバッチ処理"""
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


# 使用例
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

### 8.3 ヘルスチェック

```kotlin
// 複数の依存サービスのヘルスチェック
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

## 9. テスト戦略

### 9.1 構造化並行性のテスト

```kotlin
// Kotlin: テスト用のディスパッチャー
@Test
fun `dashboard loads all data concurrently`() = runTest {
    val userService = FakeUserService()
    val orderService = FakeOrderService()

    val dashboard = loadDashboard(userService, orderService)

    assertEquals("田中太郎", dashboard.user.name)
    assertEquals(3, dashboard.orders.size)
}

@Test
fun `partial failure returns fallback data`() = runTest {
    val userService = FakeUserService()
    val orderService = FailingOrderService()

    val dashboard = loadDashboardResilient(userService, orderService)

    assertEquals("田中太郎", dashboard.user.name)
    assertEquals(emptyList(), dashboard.orders) // フォールバック
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
    // 子タスクもキャンセルされていることを確認
}
```

```python
# Python: 構造化並行性のテスト
import pytest
import asyncio


@pytest.mark.asyncio
async def test_task_group_all_succeed():
    """全タスクが成功する場合"""
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
    """1つのタスクが失敗すると他もキャンセルされる"""
    with pytest.raises(ExceptionGroup) as exc_info:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(asyncio.sleep(10))  # これはキャンセルされる
            tg.create_task(failing_task())       # これが失敗

    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ValueError)


@pytest.mark.asyncio
async def test_cancellation_propagation():
    """キャンセルが子タスクに伝播する"""
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
    """タイムアウトでTaskGroup全体がキャンセルされる"""
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(asyncio.sleep(10))
                tg.create_task(asyncio.sleep(10))
```

---

## 10. アンチパターン

### 10.1 避けるべきパターン

```
アンチパターン1: Fire and Forget（起動して放置）
  ✗ Bad:
    function handleRequest() {
      sendEmail(user.email);  // 結果を待たない、エラーも検知しない
      return { ok: true };
    }

  ✓ Good:
    function handleRequest() {
      // ジョブキューに投入（信頼性のある非同期処理）
      await jobQueue.enqueue('send-email', { email: user.email });
      return { ok: true };
    }

アンチパターン2: 無限のキャンセル無視
  ✗ Bad:
    async def process():
        while True:
            data = compute_heavy()  # キャンセルポイントがない
            results.append(data)

  ✓ Good:
    async def process():
        while True:
            await asyncio.sleep(0)  # キャンセルチェック
            data = compute_heavy()
            results.append(data)

アンチパターン3: CancelledError の握りつぶし
  ✗ Bad:
    async def task():
        try:
            await operation()
        except Exception:  # CancelledError も catch してしまう
            pass

  ✓ Good:
    async def task():
        try:
            await operation()
        except asyncio.CancelledError:
            raise  # 必ず再送出
        except Exception:
            pass

アンチパターン4: 不要なグローバルスコープ
  ✗ Bad (Kotlin):
    fun handleRequest() {
      GlobalScope.launch { ... }  // ライフサイクル管理なし
    }

  ✓ Good (Kotlin):
    suspend fun handleRequest() = coroutineScope {
      launch { ... }  // スコープ内で管理
    }

アンチパターン5: 過度な並行性
  ✗ Bad:
    // 10万件を全て同時に処理
    await Promise.all(items.map(item => process(item)));

  ✓ Good:
    // 並行数を制限
    await mapConcurrent(items, process, 20);
```

---

## まとめ

| 言語 | 構造化並行性 | スコープ | キャンセル | エラー伝播 |
|------|------------|---------|-----------|-----------|
| Kotlin | coroutineScope | 全子完了を待つ | CancellationException | 自動伝播 |
| Kotlin | supervisorScope | 全子完了を待つ | 独立 | 独立ハンドリング |
| Swift | async let | 全子完了を待つ | 自動キャンセル | throws で伝播 |
| Swift | TaskGroup | 全子完了を待つ | cancelAll() | throws で伝播 |
| Python | asyncio.TaskGroup | 全子完了を待つ | CancelledError | ExceptionGroup |
| Rust | tokio JoinSet | 明示的に待つ | Future ドロップ | JoinError |
| JS/TS | Promise.all | 明示的に待つ | AbortController | reject 伝播 |

---

## 次に読むべきガイド
→ [[../04-practical/00-api-error-design.md]] — APIエラー設計

---

## 参考文献
1. Elizarov, R. "Structured Concurrency." vorpus.org, 2018.
2. Swift Evolution. "SE-0304: Structured Concurrency."
3. Python Documentation. "asyncio — TaskGroup." docs.python.org.
4. Kotlin Documentation. "Coroutines guide." kotlinlang.org.
5. Smith, N. "Notes on structured concurrency, or: Go statement considered harmful." 2018.
6. Tokio Documentation. "Working with Tasks." tokio.rs.
7. Apple Developer. "Concurrency — Swift Programming Language." developer.apple.com.
8. Syme, D. "The early history of F# async." fsharpforfunandprofit.com.
9. Sustrik, M. "Structured Concurrency." 250bpm.com, 2016.
10. Nygard, M. "Release It!" Pragmatic Bookshelf, 2018.
