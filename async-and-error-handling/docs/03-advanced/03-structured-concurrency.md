# 構造化並行性

> 構造化並行性は「並行処理のライフタイムを構造的に管理する」パラダイム。Kotlin coroutines、Swift structured concurrency、Python TaskGroup を通じて、安全な並行プログラミングを実現する。

## この章で学ぶこと

- [ ] 構造化並行性の原則を理解する
- [ ] 非構造化並行性の問題を把握する
- [ ] 各言語での実装を学ぶ

---

## 1. 構造化並行性とは

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

---

## 2. Kotlin Coroutines

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
```

---

## 3. Swift Structured Concurrency

```swift
// Swift: TaskGroup
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
```

---

## 4. Python TaskGroup（3.11+）

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

# ExceptionGroup のハンドリング（Python 3.11+）
async def resilient_load():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(task_a())
            tg.create_task(task_b())
    except* ValueError as eg:
        print(f"ValueError: {eg.exceptions}")
    except* TypeError as eg:
        print(f"TypeError: {eg.exceptions}")
```

---

## 5. 構造化並行性の原則

```
3つの原則:

  1. 子タスクは親のスコープ内で生存
     → 親が終了 = 子も終了（リーク防止）

  2. エラーの伝播
     → 子のエラーは親に伝播する
     → 握りつぶされない

  3. キャンセルの伝播
     → 親がキャンセルされたら子もキャンセル
     → 1つの子が失敗したら兄弟もキャンセル（coroutineScope）

メリット:
  ✓ リソースリーク防止
  ✓ エラーの確実な処理
  ✓ コードの可読性（スコープが明確）
  ✓ デバッグの容易さ
```

---

## まとめ

| 言語 | 構造化並行性 | スコープ |
|------|------------|---------|
| Kotlin | coroutineScope | 全子完了を待つ |
| Swift | async let, TaskGroup | 全子完了を待つ |
| Python | asyncio.TaskGroup | 全子完了を待つ |
| JS/TS | Promise.all（部分的） | 明示的に待つ必要 |

---

## 次に読むべきガイド
→ [[../04-practical/00-api-error-design.md]] — APIエラー設計

---

## 参考文献
1. Elizarov, R. "Structured Concurrency." vorpus.org, 2018.
2. Swift Evolution. "SE-0304: Structured Concurrency."
