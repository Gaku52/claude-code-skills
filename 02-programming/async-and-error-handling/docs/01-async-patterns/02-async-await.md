# async/await

> async/await は「非同期コードを同期的に読める」構文糖。Promise ベースの非同期処理を直感的に書ける。JavaScript, Python, Rust, C# での実装と、並行実行パターンを解説。

## この章で学ぶこと

- [ ] async/await の仕組みと動作原理を理解する
- [ ] 各言語での async/await の違いを把握する
- [ ] 効率的な並行実行パターンを学ぶ

---

## 1. async/await の基本

```
async 関数:
  → Promise を返す関数
  → return 値は自動的に Promise.resolve() でラップ

await 式:
  → Promise が解決されるまで関数の実行を一時停止
  → 解決された値を返す
  → async 関数内でのみ使用可能

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

---

## 2. JavaScript/TypeScript

```typescript
// 基本
async function getUserProfile(userId: string): Promise<UserProfile> {
  const user = await userRepo.findById(userId);
  if (!user) throw new Error("User not found");

  const [orders, reviews] = await Promise.all([
    orderRepo.findByUserId(userId),
    reviewRepo.findByUserId(userId),
  ]);

  return { user, orders, reviews };
}

// エラーハンドリング
async function safeGetUser(userId: string): Promise<User | null> {
  try {
    return await userRepo.findById(userId);
  } catch (error) {
    logger.error("Failed to get user", { userId, error });
    return null;
  }
}

// 逐次 vs 並行
async function sequential(): Promise<void> {
  const a = await fetchA(); // 100ms
  const b = await fetchB(); // 200ms
  // 合計: 300ms（直列）
}

async function concurrent(): Promise<void> {
  const [a, b] = await Promise.all([
    fetchA(), // 100ms ┐
    fetchB(), // 200ms ┤ 並行
  ]);        //       ┘
  // 合計: 200ms（並行）
}
```

---

## 3. Python

```python
import asyncio

# 基本
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

# タスクの作成と並行実行
async def process_items(items: list[str]) -> list[dict]:
    tasks = [asyncio.create_task(fetch_item(item)) for item in items]
    return await asyncio.gather(*tasks)

# タイムアウト
async def with_timeout():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        print("タイムアウト")
```

---

## 4. Rust

```rust
// Rust: async/await（tokio ランタイム）
use tokio;

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

---

## 5. 効率的なパターン

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

// パターン4: for-await-of（非同期イテレーション）
async function* fetchPages(url: string) {
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
```

---

## 6. よくある間違い

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

// ❌ ループ内の await
for (const id of ids) {
  const data = await fetch(`/api/${id}`); // 1件ずつ...
}

// ✅ 並行実行
const results = await Promise.all(
  ids.map(id => fetch(`/api/${id}`))
);

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

---

## まとめ

| 言語 | async 構文 | 並行実行 | ランタイム |
|------|-----------|---------|-----------|
| JS/TS | async/await | Promise.all | イベントループ |
| Python | async/await | asyncio.gather | asyncio |
| Rust | async/await | tokio::join! | tokio/async-std |
| Go | goroutine | go + channel | ランタイム内蔵 |
| C# | async/await | Task.WhenAll | CLR |

---

## 次に読むべきガイド
→ [[03-reactive-streams.md]] — Reactive Streams

---

## 参考文献
1. MDN Web Docs. "async function."
2. Python Documentation. "Coroutines and Tasks."
3. Tokio Documentation. "Tutorial."
