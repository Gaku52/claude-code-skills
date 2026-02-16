# Promise

> Promise は「将来の値」を表すオブジェクト。コールバック地獄を解消し、非同期処理をチェーン可能にする。Promise.all, Promise.race, Promise.allSettled の使い分けをマスターする。

## この章で学ぶこと

- [ ] Promise の3つの状態と動作原理を理解する
- [ ] Promise チェーンとエラー伝播を把握する
- [ ] Promise の並行実行パターンを学ぶ
- [ ] 各言語の Promise 相当機能を比較する
- [ ] 実務での Promise パターンとアンチパターンを習得する

---

## 1. Promise の基本

### 1.1 Promise の3つの状態

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
  → これを「settled」（決定済み）と呼ぶ

状態遷移のルール:
  1. pending → fulfilled（resolve で遷移）
  2. pending → rejected（reject で遷移）
  3. fulfilled → 変更不可
  4. rejected → 変更不可
  5. pending → pending（そのまま）
```

### 1.2 Promise の作成と消費

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

### 1.3 即座に解決される Promise

```typescript
// 即座に fulfilled になる Promise
const resolved = Promise.resolve(42);
const resolvedObj = Promise.resolve({ name: "太郎" });

// 即座に rejected になる Promise
const rejected = Promise.reject(new Error("エラー"));

// 値が Promise の場合はそのまま返す（ラップしない）
const original = Promise.resolve(42);
const same = Promise.resolve(original);
console.log(original === same); // true

// thenableオブジェクト（then メソッドを持つオブジェクト）
const thenable = {
  then(resolve) {
    resolve(42);
  }
};
const fromThenable = Promise.resolve(thenable);
fromThenable.then(value => console.log(value)); // 42
```

### 1.4 Promise の実行タイミング

```typescript
// Promise のコールバック（executor）は同期的に実行される
console.log('1. before');

const p = new Promise((resolve) => {
  console.log('2. executor (同期実行)');
  resolve('value');
});

console.log('3. after');

p.then((value) => {
  console.log('4. then (非同期実行、マイクロタスク)');
});

console.log('5. end');

// 出力順序:
// 1. before
// 2. executor (同期実行)
// 3. after
// 5. end
// 4. then (非同期実行、マイクロタスク)
```

---

## 2. Promise チェーン

### 2.1 基本的なチェーン

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

### 2.2 チェーンの動作原理

```typescript
// then() が返す Promise の値は、コールバックの戻り値で決まる

// ケース1: 値を返す → Promise.resolve(値)
Promise.resolve(1)
  .then(x => x + 1)  // Promise.resolve(2)
  .then(x => x * 3)  // Promise.resolve(6)
  .then(x => console.log(x)); // 6

// ケース2: Promise を返す → その Promise が使われる
Promise.resolve(1)
  .then(x => Promise.resolve(x + 1))  // Promise<2>
  .then(x => fetch(`/api/${x}`))       // fetch の Promise
  .then(response => response.json());

// ケース3: エラーをスロー → Promise.reject(エラー)
Promise.resolve(1)
  .then(x => {
    if (x < 10) throw new Error('Too small');
    return x;
  })
  .catch(err => console.error(err.message)); // "Too small"

// ケース4: 何も返さない → Promise.resolve(undefined)
Promise.resolve(1)
  .then(x => { console.log(x); }) // undefined
  .then(x => console.log(x));     // undefined
```

### 2.3 エラーハンドリングの詳細

```typescript
// catch は then(undefined, onRejected) のショートカット
promise.catch(fn);
// ≡ promise.then(undefined, fn);

// ただし動作に微妙な違いがある
promise
  .then(
    value => { throw new Error('then内のエラー'); },
    error => console.log('rejected:', error) // ← then内のエラーはキャッチしない
  );

promise
  .then(value => { throw new Error('then内のエラー'); })
  .catch(error => console.log('caught:', error)); // ← then内のエラーもキャッチする

// チェーン途中でのエラーリカバリー
fetchUser(userId)
  .then(user => fetchAvatar(user.avatarId))
  .catch(error => {
    console.warn('Avatar fetch failed, using default');
    return '/images/default-avatar.png'; // リカバリー値
  })
  .then(avatarUrl => {
    // エラーがあってもここに到達（リカバリー値で）
    displayAvatar(avatarUrl);
  });

// 複数の catch でセグメント化
fetchUser(userId)
  .then(user => {
    return fetchOrders(user.id);
  })
  .catch(error => {
    // fetchUser または fetchOrders のエラー
    console.error('Data fetch error:', error);
    return []; // 空配列でリカバリー
  })
  .then(orders => {
    return calculateTotal(orders);
  })
  .catch(error => {
    // calculateTotal のエラーのみ
    console.error('Calculation error:', error);
    return 0;
  })
  .then(total => {
    displayTotal(total);
  });
```

### 2.4 finally の使い方

```typescript
// finally: 成功・失敗に関わらず実行される
// 値を変更しない（透過的）

async function fetchData(url: string): Promise<Data> {
  showLoadingSpinner();

  return fetch(url)
    .then(response => response.json())
    .finally(() => {
      // スピナーを隠す（成功でも失敗でも）
      hideLoadingSpinner();
    });
}

// finally は値を渡す（変更しない）
Promise.resolve(42)
  .finally(() => {
    console.log('cleanup');
    return 100; // 無視される
  })
  .then(value => console.log(value)); // 42（100ではない）

// ただし finally 内で throw するとエラーが伝播する
Promise.resolve(42)
  .finally(() => {
    throw new Error('cleanup failed');
  })
  .catch(err => console.error(err.message)); // "cleanup failed"
```

---

## 3. 並行実行パターン

### 3.1 Promise.all

```typescript
// Promise.all: 全て成功したら成功。1つでも失敗したら失敗
const [users, orders, products] = await Promise.all([
  fetchUsers(),      // 100ms
  fetchOrders(),     // 200ms
  fetchProducts(),   // 150ms
]);
// 合計: max(100, 200, 150) = 200ms

// 型安全な使い方（TypeScript）
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

// 動的な配列
async function fetchAllUserData(userIds: string[]): Promise<User[]> {
  return Promise.all(
    userIds.map(id => fetchUser(id))
  );
}

// 注意: 1つでも失敗すると全体が失敗
try {
  const results = await Promise.all([
    fetchFromAPI1(), // 成功
    fetchFromAPI2(), // 失敗 → 全体が失敗
    fetchFromAPI3(), // 成功だが結果は破棄される
  ]);
} catch (error) {
  // fetchFromAPI2 のエラーのみ
  console.error('One of the requests failed:', error);
}
```

### 3.2 Promise.allSettled

```typescript
// Promise.allSettled: 全ての結果を取得（成功も失敗も）
// ES2020 で追加
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

// 実用例: 部分的な成功を許容
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

// 使用
const { succeeded, failed } = await fetchMultipleAPIs([
  'https://api1.example.com/data',
  'https://api2.example.com/data',
  'https://api3.example.com/data',
]);

console.log(`${succeeded.length} succeeded, ${failed.length} failed`);
```

### 3.3 Promise.race

```typescript
// Promise.race: 最初に完了したものを返す
const fastest = await Promise.race([
  fetchFromServer1(), // 100ms
  fetchFromServer2(), // 50ms  ← これが勝つ
  fetchFromServer3(), // 200ms
]);

// 実用例1: タイムアウト実装
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms);
  });

  return Promise.race([promise, timeout]);
}

// 使用
try {
  const data = await withTimeout(fetchData(), 5000);
} catch (error) {
  if (error.message.includes('Timeout')) {
    console.error('Request timed out');
  }
}

// 実用例2: キャンセル可能なPromise
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
// 5秒後にキャンセル
setTimeout(cancel, 5000);
```

### 3.4 Promise.any

```typescript
// Promise.any: 最初に成功したものを返す（ES2021）
const firstSuccess = await Promise.any([
  fetchFromServer1(), // 失敗
  fetchFromServer2(), // 成功 ← これを返す
  fetchFromServer3(), // 成功
]);
// 全て失敗した場合のみ AggregateError

// 実用例: フォールバックサーバー
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

// 使用
const response = await fetchWithFallback([
  'https://primary.example.com/api/data',
  'https://secondary.example.com/api/data',
  'https://tertiary.example.com/api/data',
]);

// 実用例: 最速のDNS解決
async function resolveFastest(hostname: string): Promise<string> {
  return Promise.any([
    resolveViaDoH('https://dns.google/resolve', hostname),
    resolveViaDoH('https://cloudflare-dns.com/dns-query', hostname),
    resolveViaSystem(hostname),
  ]);
}
```

### 3.5 4つの並行メソッド比較

```
┌──────────────────┬─────────────┬─────────────┬──────────────┐
│ メソッド          │ 成功条件     │ 失敗条件     │ ユースケース  │
├──────────────────┼─────────────┼─────────────┼──────────────┤
│ Promise.all      │ 全て成功     │ 1つでも失敗  │ 全データ必要  │
├──────────────────┼─────────────┼─────────────┼──────────────┤
│ Promise.allSettled│ 常に成功     │ なし         │ 部分的失敗OK │
├──────────────────┼─────────────┼─────────────┼──────────────┤
│ Promise.race     │ 最初の結果   │ 最初の結果   │ タイムアウト  │
├──────────────────┼─────────────┼─────────────┼──────────────┤
│ Promise.any      │ 最初の成功   │ 全て失敗     │ フォールバック│
└──────────────────┴─────────────┴─────────────┴──────────────┘
```

---

## 4. よくある間違いとアンチパターン

### 4.1 Promise を返し忘れ

```typescript
// ❌ Promise を返し忘れ
async function bad() {
  fetchData(); // await も return もない → 結果を待たない
}

// ✅ 修正
async function good() {
  return fetchData(); // または await fetchData();
}

// ❌ map 内で Promise を返し忘れ
async function badMap(items: Item[]) {
  items.map(async item => {
    await processItem(item); // 返されない → 完了を待てない
  });
}

// ✅ 修正: Promise.all で待つ
async function goodMap(items: Item[]) {
  await Promise.all(
    items.map(async item => {
      await processItem(item);
    })
  );
}
```

### 4.2 不要な Promise ラッパー

```typescript
// ❌ 不要な Promise ラッパー
async function unnecessary() {
  return new Promise((resolve) => {
    resolve(fetchData()); // fetchData() は既に Promise を返す
  });
}
// ✅ そのまま返す
async function correct() {
  return fetchData();
}

// ❌ 不要な async
async function alsoUnnecessary() {
  return 42; // async 不要（同期値を返すだけ）
}
// ✅ 修正（async が不要なら外す）
function simple(): number {
  return 42;
}

// ただし、エラーを Promise.reject にしたい場合は async が有用
async function withErrorHandling(): Promise<number> {
  const value = validate(input); // validate が throw する可能性
  return value; // async 関数なので throw は自動的に reject に変換
}
```

### 4.3 forEach で async

```typescript
// ❌ forEach で async（並行制御不能）
items.forEach(async (item) => {
  await processItem(item); // 全て同時に開始、完了を待てない
});
console.log('done'); // ← processItem の完了前に実行される！

// ✅ for...of で逐次実行
for (const item of items) {
  await processItem(item); // 1つずつ順番に処理
}
console.log('done'); // 全件完了後に実行

// ✅ Promise.all で並行実行
await Promise.all(items.map(item => processItem(item)));
console.log('done'); // 全件完了後に実行

// ✅ for...of + バッチで制御された並行実行
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

await processBatch(items, processItem, 5); // 5並列でバッチ処理
```

### 4.4 catch なしの Promise

```typescript
// ❌ catch なしの Promise
fetchData().then(data => use(data));
// → rejected 時に UnhandledPromiseRejection

// ✅ 修正
fetchData().then(data => use(data)).catch(handleError);

// ✅ async/await で try-catch
async function handler() {
  try {
    const data = await fetchData();
    use(data);
  } catch (error) {
    handleError(error);
  }
}

// グローバルなハンドラーも設定しておく
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // ログ記録、アラート送信等
});
```

### 4.5 then チェーンのネスト

```typescript
// ❌ then の中で then（コールバック地獄の再来）
fetchUser(userId).then(user => {
  fetchOrders(user.id).then(orders => {
    fetchOrderDetails(orders[0].id).then(details => {
      console.log(details); // ネストが深い
    });
  });
});

// ✅ チェーンでフラット化
fetchUser(userId)
  .then(user => fetchOrders(user.id))
  .then(orders => fetchOrderDetails(orders[0].id))
  .then(details => console.log(details))
  .catch(error => console.error(error));

// ✅ async/await でさらにシンプル
async function getDetails(userId: string) {
  const user = await fetchUser(userId);
  const orders = await fetchOrders(user.id);
  const details = await fetchOrderDetails(orders[0].id);
  return details;
}
```

---

## 5. 並行数制限

### 5.1 Promise プール

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
const urls = Array.from({ length: 1000 }, (_, i) =>
  `https://api.example.com/item/${i}`
);
const tasks = urls.map(url => () => fetch(url).then(r => r.json()));
const results = await promisePool(tasks, 5);
```

### 5.2 セマフォベースの並行制限

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

// 使用
const semaphore = new AsyncSemaphore(3); // 最大3並列

const results = await Promise.all(
  urls.map(url =>
    semaphore.withPermit(() => fetch(url).then(r => r.json()))
  )
);
```

### 5.3 キューベースの並行制限

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

// 使用
const queue = new AsyncQueue<Response>(5);

const results = await Promise.all(
  urls.map(url =>
    queue.add(() => fetch(url))
  )
);
```

---

## 6. 実践パターン

### 6.1 リトライパターン

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

// 使用
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

### 6.2 キャッシュパターン

```typescript
// Promise のキャッシュ（重複リクエスト防止）
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
      // TTL後にキャッシュ削除
      setTimeout(() => this.cache.delete(key), this.ttl);
      return value;
    }).catch(error => {
      // エラー時は即座にキャッシュ削除（次回リトライ可能に）
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

// 使用
const userCache = new PromiseCache<string, User>(30000); // 30秒TTL

async function getUser(userId: string): Promise<User> {
  return userCache.get(userId, () =>
    fetch(`/api/users/${userId}`).then(r => r.json())
  );
}

// 同時に同じユーザーをリクエストしても、APIは1回だけ呼ばれる
const [user1, user2] = await Promise.all([
  getUser('user-123'),
  getUser('user-123'), // キャッシュヒット（同じPromise）
]);
```

### 6.3 デバウンスパターン

```typescript
// Promiseベースのデバウンス
function debouncePromise<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  delay: number,
): T {
  let timeoutId: ReturnType<typeof setTimeout>;
  let pendingResolve: ((value: any) => void) | null = null;
  let pendingReject: ((reason: any) => void) | null = null;

  return ((...args: Parameters<T>): Promise<ReturnType<T>> => {
    return new Promise((resolve, reject) => {
      // 前のPendingをキャンセル
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

// 使用: 検索API
const debouncedSearch = debouncePromise(
  (query: string) => fetch(`/api/search?q=${query}`).then(r => r.json()),
  300,
);

// 300ms以内に複数回呼んでも最後の1回だけ実行
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

### 6.4 パイプラインパターン

```typescript
// Promise パイプライン: 処理を段階的に組み立て
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

// 使用
const processOrder = pipeline<OrderInput>(
  validateOrder,        // OrderInput → ValidatedOrder
  calculatePricing,     // ValidatedOrder → PricedOrder
  applyDiscounts,       // PricedOrder → DiscountedOrder
  processPayment,       // DiscountedOrder → PaidOrder
  createShipment,       // PaidOrder → ShippedOrder
  sendConfirmation,     // ShippedOrder → ConfirmedOrder
);

const order = await processOrder({
  items: [{ productId: 'p-1', quantity: 2 }],
  customerId: 'c-123',
});
```

---

## 7. 他言語の Promise 相当

### 7.1 Python: asyncio.Future / coroutine

```python
import asyncio

# Python の coroutine は JavaScript の async 関数に相当
async def fetch_user(user_id: str) -> dict:
    # await で他の coroutine を待つ
    await asyncio.sleep(0.1)  # I/Oシミュレート
    return {"id": user_id, "name": "太郎"}

# asyncio.gather = Promise.all
async def fetch_all():
    users, orders, stats = await asyncio.gather(
        fetch_user("u-1"),
        fetch_orders("u-1"),
        fetch_stats(),
    )
    return {"users": users, "orders": orders, "stats": stats}

# asyncio.wait = より細かい制御
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
        task.cancel()  # タイムアウトしたタスクをキャンセル

    return [task.result() for task in done]

# asyncio.TaskGroup (Python 3.11+) = 構造化並行性
async def structured_fetch():
    async with asyncio.TaskGroup() as tg:
        user_task = tg.create_task(fetch_user("u-1"))
        orders_task = tg.create_task(fetch_orders("u-1"))

    # TaskGroupを抜けた時点で全タスク完了
    return user_task.result(), orders_task.result()
```

### 7.2 Rust: Future

```rust
use tokio;
use futures::future;

// Rust の Future = JavaScript の Promise
// ただし lazy: .await するまで実行されない

async fn fetch_user(user_id: &str) -> Result<User, AppError> {
    // async関数は Future<Output = Result<User, AppError>> を返す
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

// futures::future::join_all = Promise.all（動的な数）
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

// Java の CompletableFuture = JavaScript の Promise

public class CompletableFutureExamples {

    // 基本的な作成
    CompletableFuture<User> fetchUser(String userId) {
        return CompletableFuture.supplyAsync(() -> {
            // バックグラウンドスレッドで実行
            return userRepo.findById(userId);
        });
    }

    // チェーン（then相当）
    CompletableFuture<String> getUserName(String userId) {
        return fetchUser(userId)
            .thenApply(user -> user.getName())        // map
            .thenApply(name -> name.toUpperCase());    // map
    }

    // flatMap相当
    CompletableFuture<List<Order>> getUserOrders(String userId) {
        return fetchUser(userId)
            .thenCompose(user -> fetchOrders(user.getId())); // flatMap
    }

    // Promise.all 相当
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

    // Promise.race 相当
    CompletableFuture<User> fetchFastest(String userId) {
        return CompletableFuture.anyOf(
            fetchFromPrimary(userId),
            fetchFromSecondary(userId)
        ).thenApply(result -> (User) result);
    }

    // エラーハンドリング
    CompletableFuture<User> fetchWithFallback(String userId) {
        return fetchUser(userId)
            .exceptionally(error -> {
                // catch 相当
                System.err.println("Fetch failed: " + error.getMessage());
                return User.defaultUser();
            });
    }

    // タイムアウト（Java 9+）
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

// C# の Task = JavaScript の Promise

public class TaskExamples
{
    // 基本
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

    // キャンセルトークン
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

## 8. Promise のテスト

### 8.1 基本テスト

```typescript
import { describe, it, expect, vi } from 'vitest';

describe('Promise パターンのテスト', () => {
  it('正常な Promise の解決をテスト', async () => {
    const result = await Promise.resolve(42);
    expect(result).toBe(42);
  });

  it('Promise の reject をテスト', async () => {
    await expect(Promise.reject(new Error('test')))
      .rejects.toThrow('test');
  });

  it('Promise.all の動作をテスト', async () => {
    const results = await Promise.all([
      Promise.resolve(1),
      Promise.resolve(2),
      Promise.resolve(3),
    ]);
    expect(results).toEqual([1, 2, 3]);
  });

  it('Promise.all の失敗をテスト', async () => {
    await expect(
      Promise.all([
        Promise.resolve(1),
        Promise.reject(new Error('fail')),
        Promise.resolve(3),
      ])
    ).rejects.toThrow('fail');
  });

  it('Promise.allSettled の動作をテスト', async () => {
    const results = await Promise.allSettled([
      Promise.resolve('ok'),
      Promise.reject(new Error('fail')),
    ]);

    expect(results[0]).toEqual({ status: 'fulfilled', value: 'ok' });
    expect(results[1].status).toBe('rejected');
  });
});
```

### 8.2 非同期処理のモック

```typescript
describe('非同期関数のモック', () => {
  it('fetchをモックしてテスト', async () => {
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

  it('リトライロジックをテスト', async () => {
    const mockFn = vi.fn()
      .mockRejectedValueOnce(new Error('fail'))
      .mockRejectedValueOnce(new Error('fail'))
      .mockResolvedValue('success');

    const result = await retryPromise(mockFn, { retries: 3, delay: 10 });
    expect(result).toBe('success');
    expect(mockFn).toHaveBeenCalledTimes(3);
  });

  it('タイムアウトをテスト', async () => {
    vi.useFakeTimers();

    const slowPromise = new Promise(resolve =>
      setTimeout(() => resolve('done'), 10000)
    );

    const promise = withTimeout(slowPromise, 5000);

    vi.advanceTimersByTime(5000);

    await expect(promise).rejects.toThrow('Timeout');

    vi.useRealTimers();
  });

  it('並行数制限をテスト', async () => {
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

## 9. Promise の内部実装

### 9.1 簡易 Promise の実装

```typescript
// Promise の内部動作を理解するための簡易実装
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

## まとめ

| メソッド | 動作 | ユースケース |
|---------|------|-------------|
| Promise.all | 全成功で成功 | 独立した複数のAPIコール |
| Promise.allSettled | 全完了を待つ | 部分的失敗を許容 |
| Promise.race | 最速の結果 | タイムアウト実装 |
| Promise.any | 最初の成功 | フォールバックサーバー |

### Promise のベストプラクティス

```
1. 常にエラーをハンドリングする
   → .catch() または try-catch

2. 不要な Promise ラッピングを避ける
   → async 関数はすでに Promise を返す

3. 並行可能な処理は Promise.all で並行実行
   → 逐次 await の無駄を避ける

4. 大量の並行処理は並行数を制限する
   → セマフォ or プールパターン

5. Promise のキャッシュでリクエスト重複を防ぐ
   → 同じキーの同時リクエストを1つにまとめる
```

---

## 次に読むべきガイド
→ [[02-async-await.md]] — async/await

---

## 参考文献
1. MDN Web Docs. "Promise."
2. Archibald, J. "JavaScript Promises: An Introduction." web.dev.
3. Promises/A+ Specification. promisesaplus.com.
4. ECMAScript Language Specification. "Promise Objects."
5. Tokio Documentation. "Working with Futures." tokio.rs.
6. Python Documentation. "asyncio - Tasks and Coroutines."
7. Oracle. "CompletableFuture." docs.oracle.com.
8. Microsoft. "Task-based asynchronous pattern." docs.microsoft.com.
