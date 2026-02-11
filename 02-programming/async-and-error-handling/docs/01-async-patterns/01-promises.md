# Promise

> Promise は「将来の値」を表すオブジェクト。コールバック地獄を解消し、非同期処理をチェーン可能にする。Promise.all, Promise.race, Promise.allSettled の使い分けをマスターする。

## この章で学ぶこと

- [ ] Promise の3つの状態と動作原理を理解する
- [ ] Promise チェーンとエラー伝播を把握する
- [ ] Promise の並行実行パターンを学ぶ

---

## 1. Promise の基本

```
Promise の3つの状態:
  pending  → fulfilled（成功）→ 値を持つ
           → rejected（失敗） → エラーを持つ

  ┌─────────┐
  │ pending │
  └────┬────┘
  ┌────┴────┐
  ▼         ▼
┌──────────┐ ┌──────────┐
│fulfilled │ │ rejected │
│ (値)     │ │ (エラー) │
└──────────┘ └──────────┘

  一度 fulfilled/rejected になると変更不可（不変）
```

```javascript
// Promise の作成
const promise = new Promise((resolve, reject) => {
  // 非同期処理
  setTimeout(() => {
    const success = Math.random() > 0.5;
    if (success) {
      resolve("成功！");    // fulfilled 状態に移行
    } else {
      reject(new Error("失敗")); // rejected 状態に移行
    }
  }, 1000);
});

// Promise の消費
promise
  .then(value => console.log(value))   // fulfilled 時
  .catch(error => console.error(error)) // rejected 時
  .finally(() => console.log("完了"));  // どちらでも
```

---

## 2. Promise チェーン

```javascript
// then() は新しい Promise を返す → チェーン可能
fetchUser(userId)
  .then(user => fetchOrders(user.id))         // Promise を返す
  .then(orders => orders.filter(o => o.active)) // 値を返す → Promise.resolve() でラップ
  .then(activeOrders => {
    console.log(`${activeOrders.length} 件の有効な注文`);
    return activeOrders;
  })
  .catch(error => {
    // チェーン内のどこで発生したエラーもここでキャッチ
    console.error("エラー:", error.message);
  });

// エラーの伝播
//  then → then → then → catch
//    ↓ エラー発生        ↑
//    └──────────────────┘
//    スキップされる
```

---

## 3. 並行実行パターン

```javascript
// Promise.all: 全て成功したら成功。1つでも失敗したら失敗
const [users, orders, products] = await Promise.all([
  fetchUsers(),      // 100ms
  fetchOrders(),     // 200ms
  fetchProducts(),   // 150ms
]);
// 合計: max(100, 200, 150) = 200ms

// Promise.allSettled: 全ての結果を取得（成功も失敗も）
const results = await Promise.allSettled([
  fetchFromAPI1(),   // 成功
  fetchFromAPI2(),   // 失敗
  fetchFromAPI3(),   // 成功
]);
// results = [
//   { status: "fulfilled", value: data1 },
//   { status: "rejected", reason: Error },
//   { status: "fulfilled", value: data3 },
// ]

// Promise.race: 最初に完了したものを返す
const fastest = await Promise.race([
  fetchFromServer1(), // 100ms
  fetchFromServer2(), // 50ms  ← これが勝つ
  fetchFromServer3(), // 200ms
]);

// Promise.any: 最初に成功したものを返す（ES2021）
const firstSuccess = await Promise.any([
  fetchFromServer1(), // 失敗
  fetchFromServer2(), // 成功 ← これを返す
  fetchFromServer3(), // 成功
]);
// 全て失敗した場合のみ AggregateError
```

---

## 4. よくある間違い

```javascript
// ❌ Promise を返し忘れ
async function bad() {
  fetchData(); // await も return もない → 結果を待たない
}

// ❌ 不要な Promise ラッパー
async function unnecessary() {
  return new Promise((resolve) => {
    resolve(fetchData()); // fetchData() は既に Promise を返す
  });
}
// ✅
async function correct() {
  return fetchData(); // そのまま返す
}

// ❌ forEach で async（並行制御不能）
items.forEach(async (item) => {
  await processItem(item); // 全て同時に開始、完了を待てない
});
// ✅ for...of で逐次実行
for (const item of items) {
  await processItem(item);
}
// ✅ Promise.all で並行実行
await Promise.all(items.map(item => processItem(item)));

// ❌ catch なしの Promise
fetchData().then(data => use(data));
// → rejected 時に UnhandledPromiseRejection
// ✅
fetchData().then(data => use(data)).catch(handleError);
```

---

## 5. 並行数制限

```typescript
// 同時実行数を制限する Promise プール
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

// 使用: 1000件のURLを同時5並列でフェッチ
const urls = Array.from({ length: 1000 }, (_, i) => `https://api.example.com/item/${i}`);
const tasks = urls.map(url => () => fetch(url).then(r => r.json()));
const results = await promisePool(tasks, 5);
```

---

## まとめ

| メソッド | 動作 | ユースケース |
|---------|------|-------------|
| Promise.all | 全成功で成功 | 独立した複数のAPIコール |
| Promise.allSettled | 全完了を待つ | 部分的失敗を許容 |
| Promise.race | 最速の結果 | タイムアウト実装 |
| Promise.any | 最初の成功 | フォールバックサーバー |

---

## 次に読むべきガイド
→ [[02-async-await.md]] — async/await

---

## 参考文献
1. MDN Web Docs. "Promise."
2. Archibald, J. "JavaScript Promises: An Introduction." web.dev.
