# async/await（非同期プログラミング）

> async/await は「I/O待ちの間に他の処理を進める」仕組みである。スレッドを使わずに大量の同時接続を効率的に処理する、現代のサーバーサイド・UIプログラミングの基盤技術。

## この章で学ぶこと

- [ ] 非同期処理の必要性と、同期処理との根本的な違いを理解する
- [ ] イベントループの動作原理を正確に把握する
- [ ] async/await の構文と内部メカニズムを理解する
- [ ] Promise / Future / Task の概念と関係を整理する
- [ ] 各言語（JavaScript, Python, Rust, Go, Java, C#）の非同期モデルの違いを比較する
- [ ] エラーハンドリング・キャンセル・タイムアウトの設計パターンを習得する
- [ ] アンチパターンを認識し、本番コードでの落とし穴を回避する

---

## 1. なぜ非同期処理が必要か

### 1.1 同期処理の限界

現代のアプリケーションは、ネットワーク通信・データベースアクセス・ファイル操作といった I/O 操作を頻繁に行う。これらの I/O 操作は CPU 演算と比較して桁違いに時間がかかる。

```
操作別レイテンシの比較（概算値）:

┌───────────────────────────────┬──────────────────┬────────────┐
│ 操作                          │ レイテンシ        │ CPU換算    │
├───────────────────────────────┼──────────────────┼────────────┤
│ L1 キャッシュ参照              │ 1 ns             │ 1 秒       │
│ L2 キャッシュ参照              │ 4 ns             │ 4 秒       │
│ メインメモリ参照               │ 100 ns           │ 1.5 分     │
│ SSD ランダムリード             │ 16,000 ns        │ 4.4 時間   │
│ HDD シーク                    │ 2,000,000 ns     │ 23 日      │
│ 同一データセンター内 RTT       │ 500,000 ns       │ 5.7 日     │
│ 大陸間ネットワーク RTT         │ 150,000,000 ns   │ 4.7 年     │
└───────────────────────────────┴──────────────────┴────────────┘

→ ネットワーク I/O は CPU 演算の約1億倍遅い
→ I/O 待ちの間、CPU を遊ばせるのは極めて非効率
```

同期処理（ブロッキング I/O）では、I/O が完了するまでスレッド全体が停止する。Webサーバーで 1リクエスト＝1スレッドのモデルを採用すると、同時接続数がスレッド数に制限される。

```
同期Webサーバーの動作:

Thread-1: ──[リクエスト受信]──▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓──[DB応答]──[レスポンス送信]──
Thread-2: ──[リクエスト受信]──▓▓▓▓▓▓▓▓▓▓──[API応答]──[レスポンス送信]──────
Thread-3: ──[リクエスト受信]──▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓──[File応答]──[送信]───
Thread-4: ──待機中（スレッドプール枯渇）──────────────────────────────────
           ▓▓ = I/O待ちでブロック（CPU は何もしていない）

問題点:
  - スレッド1本あたり 1MB 程度のスタックメモリを消費
  - 1万同時接続 → 10GB のメモリが必要
  - コンテキストスイッチのオーバーヘッド
  - C10K 問題（1万同時接続の壁）
```

### 1.2 非同期処理の解決策

非同期処理（ノンブロッキング I/O）は、I/O 操作の開始だけを行い、完了通知を待つ間に他の処理を進める。

```
非同期Webサーバーの動作（イベントループ方式）:

Event Loop (1 thread):
  ──[Req-A 受信]──[DB発行]──[Req-B 受信]──[API発行]──[Req-C 受信]──
  ──[File発行]──[DB応答→Res-A送信]──[API応答→Res-B送信]──
  ──[File応答→Res-C送信]──[Req-D 受信]──...

  → 1スレッドで数千〜数万の同時接続を処理可能
  → I/O待ち時間を他のリクエスト処理に有効活用
  → メモリ使用量が劇的に削減

利点:
  ┌────────────────────────┬──────────┬─────────────┐
  │ 指標                    │ 同期方式  │ 非同期方式   │
  ├────────────────────────┼──────────┼─────────────┤
  │ 10,000 同時接続時メモリ  │ ~10 GB   │ ~100 MB     │
  │ コンテキストスイッチ     │ 頻繁     │ 最小限       │
  │ スループット             │ 中       │ 高           │
  │ CPU利用効率             │ 低       │ 高           │
  │ プログラミング複雑性     │ 低       │ 中〜高       │
  └────────────────────────┴──────────┴─────────────┘
```

### 1.3 非同期処理の歴史的進化

非同期プログラミングは、コールバック地獄からasync/awaitに至るまで段階的に進化してきた。

```
非同期プログラミングの進化:

Stage 1: コールバック（Callback）
  ├── 最も原始的な方式
  ├── 関数の引数として完了時の処理を渡す
  └── 問題: コールバック地獄（Pyramid of Doom）

Stage 2: Promise / Future
  ├── 非同期処理の結果を表すオブジェクト
  ├── メソッドチェーンで逐次処理を記述
  └── 問題: まだネストが深くなりがち

Stage 3: async/await
  ├── Promise/Future のシンタックスシュガー
  ├── 同期処理と同じ見た目で非同期処理を記述
  └── 現在の主流パラダイム

Stage 4: Structured Concurrency（構造化並行性）
  ├── 非同期タスクのライフサイクルを構造的に管理
  ├── Python: TaskGroup, Kotlin: coroutineScope
  └── キャンセル・エラー伝播を安全に処理
```

---

## 2. イベントループの動作原理

### 2.1 イベントループとは

async/await を理解するには、その基盤であるイベントループの仕組みを正確に把握する必要がある。イベントループは、I/O イベントを監視し、対応するコールバックを実行する無限ループである。

```
イベントループの基本構造:

┌──────────────────────────────────────────────────────────┐
│                     イベントループ                        │
│                                                          │
│  while (true) {                                          │
│    1. タイマーキューのコールバックを実行                    │
│    2. I/O ポーリング（epoll/kqueue/IOCP）                 │
│    3. 完了した I/O のコールバックを実行                    │
│    4. マイクロタスクキューを処理                           │
│    5. 実行すべきタスクがなければスリープ                    │
│  }                                                       │
│                                                          │
│  ┌─────────┐    ┌─────────┐    ┌──────────────┐         │
│  │ Timer   │    │  I/O    │    │ Microtask    │         │
│  │ Queue   │    │  Queue  │    │ Queue        │         │
│  │         │    │         │    │              │         │
│  │ setTimeout│  │ fs.read │    │ Promise.then │         │
│  │ setInterval│ │ net.req │    │ queueMicro.. │         │
│  └─────────┘    └─────────┘    └──────────────┘         │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Node.js のイベントループ（libuv）

Node.js のイベントループは libuv ライブラリに基づいており、以下のフェーズを順番に実行する。

```
Node.js イベントループの詳細フェーズ:

   ┌───────────────────────────┐
┌─>│        timers              │ ← setTimeout, setInterval のコールバック
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
│  │     pending callbacks     │ ← TCP エラーなどのシステムコールバック
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
│  │       idle, prepare       │ ← 内部使用のみ
│  └─────────────┬─────────────┘      ┌───────────────┐
│  ┌─────────────┴─────────────┐      │   incoming:    │
│  │          poll              │<─────┤  connections,  │
│  └─────────────┬─────────────┘      │  data, etc.    │
│  ┌─────────────┴─────────────┐      └───────────────┘
│  │          check            │ ← setImmediate のコールバック
│  └─────────────┬─────────────┘
│  ┌─────────────┴─────────────┐
│  │      close callbacks      │ ← socket.on('close', ...) 等
│  └─────────────┬─────────────┘
└─────────────────┘

※ 各フェーズの間にマイクロタスクキュー（Promise）が処理される
```

### 2.3 JavaScript での実行順序

イベントループの理解を深めるため、実行順序を追跡してみよう。

```javascript
// 実行順序クイズ: 以下のコードの出力順は？
console.log('1: スクリプト開始');

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

console.log('6: スクリプト終了');

// 出力順:
// 1: スクリプト開始
// 6: スクリプト終了
// 3: Promise.then         ← マイクロタスク（同期コード完了直後）
// 5: queueMicrotask       ← マイクロタスク
// 4: Promise.then (chained) ← マイクロタスク
// 2: setTimeout           ← マクロタスク（次のイベントループ）
```

```
実行順序の図解:

コールスタック         マイクロタスクキュー    マクロタスクキュー
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ console.log │       │             │       │             │
│ ('1: ...')  │       │             │       │             │
├─────────────┤       │             │       │             │
│ setTimeout  │ ──────┼─────────────┼──────>│ callback-2  │
├─────────────┤       │             │       │             │
│ Promise     │ ─────>│ callback-3  │       │             │
│ .resolve()  │       │             │       │             │
├─────────────┤       │             │       │             │
│ queueMicro  │ ─────>│ callback-5  │       │             │
│ task        │       │             │       │             │
├─────────────┤       │             │       │             │
│ console.log │       │             │       │             │
│ ('6: ...')  │       │             │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
   ↓ スタック空                ↓ 全て実行          ↓ 次に実行
   Step 1: 1,6出力    Step 2: 3,5,4出力    Step 3: 2出力
```

---

## 3. Promise / Future / Task の基礎概念

### 3.1 Promise の状態遷移

Promise（JavaScript）/ Future（Rust, Dart）/ Task（C#）は、非同期処理の結果を表すオブジェクトである。これらは共通の状態遷移モデルを持つ。

```
Promise の状態遷移図:

                     ┌──────────┐
                     │ Pending  │ ← 初期状態（処理中）
                     │  (待機)  │
                     └────┬─────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         ┌────▼─────┐          ┌─────▼──────┐
         │ Fulfilled│          │  Rejected  │
         │  (成功)  │          │   (失敗)   │
         └──────────┘          └────────────┘
              │                       │
              ▼                       ▼
         .then(value)            .catch(error)
         で値を受け取る           でエラーを処理

注意: Pending → Fulfilled / Rejected の遷移は一方向
      一度確定した状態は変更不可（Immutable Settlement）
```

### 3.2 各言語における非同期プリミティブの対応表

```
┌────────────────┬──────────────┬─────────────┬───────────────┐
│ 概念            │ JavaScript   │ Python      │ Rust          │
├────────────────┼──────────────┼─────────────┼───────────────┤
│ 非同期結果      │ Promise      │ Coroutine   │ Future        │
│ 成功値取得      │ .then()      │ await       │ .await        │
│ エラー処理      │ .catch()     │ try/except  │ ? / match     │
│ 並行実行        │ Promise.all  │ gather      │ join!         │
│ 競争           │ Promise.race │ wait_for    │ select!       │
│ タイムアウト    │ AbortSignal  │ timeout()   │ timeout()     │
│ キャンセル      │ AbortController│ Task.cancel│ drop          │
├────────────────┼──────────────┼─────────────┼───────────────┤
│ 概念            │ C#           │ Go          │ Java          │
├────────────────┼──────────────┼─────────────┼───────────────┤
│ 非同期結果      │ Task<T>      │ chan T      │ CompletableFuture │
│ 成功値取得      │ await        │ <-ch        │ .get() / .join() │
│ エラー処理      │ try/catch    │ error return│ .exceptionally() │
│ 並行実行        │ Task.WhenAll │ WaitGroup   │ allOf         │
│ 競争           │ Task.WhenAny │ select      │ anyOf         │
│ タイムアウト    │ CancellationToken│ context  │ .orTimeout()  │
│ キャンセル      │ CancellationToken│ context  │ .cancel()     │
└────────────────┴──────────────┴─────────────┴───────────────┘
```

### 3.3 コールバック → Promise → async/await の進化

```javascript
// ===== Stage 1: コールバック地獄 =====
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
// 問題: ネストが深い、エラー処理が散在、可読性が低い

// ===== Stage 2: Promise チェーン =====
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
// 改善: フラットなチェーン、統一的なエラー処理
// 問題: まだ複雑、中間変数の管理が面倒

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
// 改善: 同期処理と同じ見た目、直感的なエラー処理
// 注意: 上記は逐次実行。並行可能な処理は Promise.all を使う

// ===== Stage 3.5: async/await + 並行実行の最適化 =====
async function loadDashboard(userId) {
    try {
        const user = await getUser(userId);

        // posts と notifications は互いに依存しないので並行実行
        const [posts, notifications] = await Promise.all([
            getPosts(user.id),
            getNotifications(user.id),
        ]);

        // comments は posts に依存するので逐次実行
        const comments = await getComments(posts[0].id);

        return { user, posts, comments, notifications };
    } catch (err) {
        console.error('Dashboard load failed:', err);
        throw err;
    }
}
```

---

## 4. JavaScript の async/await 詳説

### 4.1 基本構文と内部動作

```javascript
// ===== 基本: async 関数は常に Promise を返す =====
async function greet(name) {
    return `Hello, ${name}!`;
}

// 上記は以下と等価:
function greet(name) {
    return Promise.resolve(`Hello, ${name}!`);
}

// ===== await: Promise の完了を待機する =====
async function fetchUserProfile(userId) {
    // await は Promise が解決されるまで関数の実行を中断し、
    // イベントループに制御を返す
    const response = await fetch(`/api/users/${userId}`);

    // レスポンスが到着したら、ここから再開
    if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
    }

    const data = await response.json();
    return data;
}

// ===== await を使えるのは async 関数内のみ =====
// ES2022 以降: トップレベル await が使用可能（ESM モジュールのみ）
const config = await import('./config.js');
const data = await fetch('/api/data').then(r => r.json());
```

### 4.2 Promise 並行処理パターン

```javascript
// ===== Promise.all: 全て成功で完了、1つでも失敗で即座に reject =====
async function fetchAllResources() {
    try {
        const [users, products, orders] = await Promise.all([
            fetch('/api/users').then(r => r.json()),
            fetch('/api/products').then(r => r.json()),
            fetch('/api/orders').then(r => r.json()),
        ]);
        return { users, products, orders };
    } catch (error) {
        // いずれか1つが失敗した場合、他の処理も中断される
        console.error('One of the requests failed:', error);
        throw error;
    }
}

// ===== Promise.allSettled: 全ての結果を取得（成功/失敗問わず） =====
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
            data[keys[index]] = []; // フォールバック値
        }
    });
    return data;
}

// ===== Promise.race: 最初に完了（成功/失敗問わず）したものを返す =====
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

// ===== Promise.any: 最初に成功したものを返す（全失敗で AggregateError） =====
async function fetchFromMultipleCDNs(path) {
    try {
        const response = await Promise.any([
            fetch(`https://cdn1.example.com${path}`),
            fetch(`https://cdn2.example.com${path}`),
            fetch(`https://cdn3.example.com${path}`),
        ]);
        return response;
    } catch (error) {
        // AggregateError: 全ての CDN が失敗
        console.error('All CDNs failed:', error.errors);
        throw error;
    }
}
```

### 4.3 エラーハンドリングのベストプラクティス

```javascript
// ===== パターン1: try/catch による集約 =====
async function processOrder(orderId) {
    try {
        const order = await fetchOrder(orderId);
        const payment = await processPayment(order);
        const shipping = await arrangeShipping(order, payment);
        await sendConfirmationEmail(order, shipping);
        return { success: true, trackingNumber: shipping.trackingNumber };
    } catch (error) {
        // エラーの種類に応じた処理
        if (error instanceof PaymentError) {
            await refundIfNeeded(orderId);
            return { success: false, reason: 'payment_failed' };
        }
        if (error instanceof ShippingError) {
            await notifyManualReview(orderId, error);
            return { success: false, reason: 'shipping_unavailable' };
        }
        // 予期しないエラーは再 throw
        throw error;
    }
}

// ===== パターン2: Result 型パターン（Go風エラーハンドリング） =====
async function safeAsync(asyncFn) {
    try {
        const result = await asyncFn();
        return [result, null];
    } catch (error) {
        return [null, error];
    }
}

// 使用例
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

// ===== パターン3: AbortController によるキャンセル =====
class ApiClient {
    constructor() {
        this.controllers = new Map();
    }

    async fetch(key, url, options = {}) {
        // 既存のリクエストをキャンセル
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

### 4.4 非同期イテレータ（Async Iterators）

```javascript
// ===== for await...of: 非同期データストリームの逐次処理 =====

// 非同期ジェネレータ関数
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

// 使用例: ページネーションの全データを処理
async function processAllItems(baseUrl) {
    const allItems = [];

    for await (const items of fetchPages(baseUrl)) {
        allItems.push(...items);
        console.log(`Processed page, total items: ${allItems.length}`);
    }

    return allItems;
}

// ReadableStream からの読み取り（Web Streams API）
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

## 5. Python の asyncio 詳説

### 5.1 コルーチンの基礎

```python
import asyncio
from typing import Any

# ===== コルーチン関数: async def で定義 =====
async def fetch_data(url: str) -> dict:
    """非同期にデータを取得する"""
    # コルーチン関数を呼び出すとコルーチンオブジェクトが返る
    # await するまで実行されない
    await asyncio.sleep(1)  # I/O 操作のシミュレーション
    return {"url": url, "data": "sample"}

# ===== コルーチンの実行方法 =====

# 方法1: asyncio.run()（メインエントリポイント）
async def main():
    result = await fetch_data("https://api.example.com/data")
    print(result)

asyncio.run(main())

# 方法2: イベントループを直接操作（低レベル）
loop = asyncio.get_event_loop()
result = loop.run_until_complete(fetch_data("https://api.example.com"))
loop.close()
```

### 5.2 並行実行パターン

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
    """汎用的な JSON フェッチ関数"""
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()

# ===== asyncio.gather: 並行実行（最も一般的） =====
async def load_dashboard(user_id: int) -> UserDashboard:
    async with aiohttp.ClientSession() as session:
        base = "https://api.example.com"
        user, posts, notifications = await asyncio.gather(
            fetch_json(session, f"{base}/users/{user_id}"),
            fetch_json(session, f"{base}/users/{user_id}/posts"),
            fetch_json(session, f"{base}/users/{user_id}/notifications"),
        )
        return UserDashboard(user=user, posts=posts, notifications=notifications)

# ===== asyncio.gather: return_exceptions=True で例外もキャッチ =====
async def load_dashboard_safe(user_id: int) -> dict:
    async with aiohttp.ClientSession() as session:
        base = "https://api.example.com"
        results = await asyncio.gather(
            fetch_json(session, f"{base}/users/{user_id}"),
            fetch_json(session, f"{base}/users/{user_id}/posts"),
            fetch_json(session, f"{base}/users/{user_id}/notifications"),
            return_exceptions=True,  # 例外を値として返す
        )
        return {
            "user": results[0] if not isinstance(results[0], Exception) else None,
            "posts": results[1] if not isinstance(results[1], Exception) else [],
            "notifications": results[2] if not isinstance(results[2], Exception) else [],
        }

# ===== TaskGroup: 構造化並行性（Python 3.11+） =====
async def load_dashboard_structured(user_id: int) -> UserDashboard:
    """TaskGroup を使った構造化並行性パターン"""
    results = {}

    async with aiohttp.ClientSession() as session:
        base = "https://api.example.com"

        async with asyncio.TaskGroup() as tg:
            async def fetch_and_store(key: str, url: str):
                results[key] = await fetch_json(session, url)

            tg.create_task(fetch_and_store("user", f"{base}/users/{user_id}"))
            tg.create_task(fetch_and_store("posts", f"{base}/users/{user_id}/posts"))
            tg.create_task(fetch_and_store("notifs", f"{base}/users/{user_id}/notifications"))

        # TaskGroup の外に到達 = 全タスク完了
        # いずれかが例外を投げた場合、他の全タスクがキャンセルされ
        # ExceptionGroup として集約される

    return UserDashboard(
        user=results["user"],
        posts=results["posts"],
        notifications=results["notifs"],
    )
```

### 5.3 タイムアウトとキャンセル

```python
import asyncio

# ===== asyncio.timeout: タイムアウト制御（Python 3.11+） =====
async def fetch_with_timeout(url: str, timeout_sec: float = 5.0):
    try:
        async with asyncio.timeout(timeout_sec):
            return await fetch_data(url)
    except TimeoutError:
        print(f"Timeout after {timeout_sec}s: {url}")
        return None

# ===== asyncio.wait_for: 従来のタイムアウト方式 =====
async def fetch_with_wait_for(url: str, timeout_sec: float = 5.0):
    try:
        return await asyncio.wait_for(
            fetch_data(url),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        print(f"Timeout: {url}")
        return None

# ===== タスクのキャンセル =====
async def cancellable_operation():
    task = asyncio.create_task(long_running_operation())

    # 5秒後にキャンセル
    await asyncio.sleep(5)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("タスクがキャンセルされました")

# ===== セマフォによる同時実行数制限 =====
async def fetch_many_urls(urls: list[str], max_concurrent: int = 10):
    """同時接続数を制限しながら大量の URL をフェッチ"""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def limited_fetch(url: str):
        async with semaphore:  # 同時実行数を max_concurrent に制限
            return await fetch_data(url)

    results = await asyncio.gather(
        *[limited_fetch(url) for url in urls]
    )
    return results
```

### 5.4 非同期コンテキストマネージャとイテレータ

```python
import asyncio
from contextlib import asynccontextmanager

# ===== 非同期コンテキストマネージャ =====
class AsyncDatabaseConnection:
    """非同期データベース接続の例"""

    async def __aenter__(self):
        self.conn = await create_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()
        return False  # 例外を伝播

    async def query(self, sql: str):
        return await self.conn.execute(sql)

# 使用例
async def get_users():
    async with AsyncDatabaseConnection() as db:
        return await db.query("SELECT * FROM users")

# ===== デコレータ方式 =====
@asynccontextmanager
async def managed_resource(name: str):
    resource = await acquire_resource(name)
    try:
        yield resource
    finally:
        await release_resource(resource)

# ===== 非同期イテレータ =====
class AsyncPaginator:
    """ページネーション付き非同期イテレータ"""

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

# 使用例
async def process_all_users():
    async for page in AsyncPaginator("https://api.example.com/users"):
        for user in page:
            await process_user(user)

# ===== 非同期ジェネレータ =====
async def event_stream(url: str):
    """Server-Sent Events を受信する非同期ジェネレータ"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            async for line in response.content:
                decoded = line.decode('utf-8').strip()
                if decoded.startswith('data:'):
                    yield decoded[5:].strip()
```

---

## 6. Rust の async/await 詳説

### 6.1 Future トレイトとポーリングモデル

Rust の非同期モデルは他の言語と根本的に異なる。Future はゼロコスト抽象化であり、`.await` を呼ぶまで何も実行されない（遅延評価）。

```rust
// ===== Future トレイトの定義（標準ライブラリ） =====
pub trait Future {
    type Output;

    // poll が呼ばれるたびに進捗をチェック
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

pub enum Poll<T> {
    Ready(T),    // 完了: 値が利用可能
    Pending,     // 未完了: まだ待機中
}

// ===== 手動で Future を実装する例 =====
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
            // ウェイカーを登録して、後で再ポーリングを要求
            let waker = cx.waker().clone();
            let when = self.when;
            std::thread::spawn(move || {
                let now = Instant::now();
                if now < when {
                    std::thread::sleep(when - now);
                }
                waker.wake();  // ランタイムに再ポーリングを通知
            });
            Poll::Pending
        }
    }
}

// 使用
async fn example() {
    println!("開始");
    Delay::new(Duration::from_secs(2)).await;
    println!("2秒経過");
}
```

```
Rust の Future ポーリングモデル:

  ランタイム (Executor)                Future
  ┌──────────────────┐               ┌─────────────┐
  │                  │  poll()       │             │
  │  1. poll 実行  ──┼──────────────>│  処理実行    │
  │                  │  Pending     │  まだ未完了   │
  │  2. 他の処理   <─┼──────────────│             │
  │     を実行       │              │  (Waker登録) │
  │                  │              └─────────────┘
  │  ...時間経過...   │
  │                  │  wake()
  │  3. Waker通知  <─┼──────────── OS/タイマー
  │                  │
  │  4. 再poll    ──┼──────────────>│  処理再開    │
  │                  │  Ready(val)  │  完了!       │
  │  5. 値を取得   <─┼──────────────│             │
  └──────────────────┘               └─────────────┘

  ※ JavaScript/Python: ランタイムがコルーチンを直接駆動
  ※ Rust: ランタイムが Future を poll して進捗を確認
  → Rust の方がきめ細かい制御が可能（ゼロコスト抽象化）
```

### 6.2 tokio ランタイムの詳細

```rust
use tokio;
use tokio::time::{sleep, timeout, Duration};
use tokio::sync::{mpsc, Semaphore};
use std::sync::Arc;

// ===== 基本的な async main =====
#[tokio::main]
async fn main() {
    let result = fetch_data("https://api.example.com").await;
    match result {
        Ok(data) => println!("Data: {}", data),
        Err(e) => eprintln!("Error: {}", e),
    }
}

// ===== 並行実行パターン =====
async fn load_dashboard(user_id: u32) -> Result<Dashboard, AppError> {
    // tokio::join!: 全て並行実行して全完了を待つ
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

// ===== tokio::select!: 最初に完了したものを処理 =====
async fn fetch_with_fallback(primary: &str, fallback: &str) -> String {
    tokio::select! {
        result = fetch_data(primary) => {
            match result {
                Ok(data) => data,
                Err(_) => fetch_data(fallback).await.unwrap_or_default(),
            }
        }
        _ = sleep(Duration::from_secs(3)) => {
            // プライマリがタイムアウト → フォールバック
            fetch_data(fallback).await.unwrap_or_default()
        }
    }
}

// ===== タスクスポーン: バックグラウンド実行 =====
async fn spawn_example() {
    let handle = tokio::spawn(async {
        // 別タスクとして非同期実行
        heavy_computation().await
    });

    // 他の処理を並行して実行
    do_other_work().await;

    // タスクの結果を取得
    let result = handle.await.expect("Task panicked");
    println!("Result: {:?}", result);
}

// ===== セマフォによる同時実行数制限 =====
async fn fetch_many_urls(urls: Vec<String>) -> Vec<Result<String, reqwest::Error>> {
    let semaphore = Arc::new(Semaphore::new(10)); // 最大10並行
    let mut handles = vec![];

    for url in urls {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let handle = tokio::spawn(async move {
            let result = reqwest::get(&url).await?.text().await;
            drop(permit); // 明示的にセマフォを解放
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

### 6.3 Rust のストリーム（非同期イテレータ）

```rust
use tokio_stream::{self as stream, StreamExt};

// ===== Stream: 非同期イテレータ =====
async fn process_stream() {
    let mut stream = stream::iter(vec![1, 2, 3, 4, 5])
        .map(|x| async move {
            sleep(Duration::from_millis(100)).await;
            x * 2
        })
        .buffered(3); // 最大3つを並行処理

    while let Some(value) = stream.next().await {
        println!("Value: {}", value);
    }
}

// ===== チャネルベースのストリーム =====
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
    let (tx, rx) = mpsc::channel(32); // バッファサイズ 32

    tokio::spawn(event_producer(tx));
    event_consumer(rx).await;
}
```

---

## 7. Go の並行処理（goroutine + channel）

Go は async/await を採用せず、CSP（Communicating Sequential Processes）モデルに基づく goroutine と channel を使う。

### 7.1 goroutine の基礎

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

// ===== goroutine: go キーワードで軽量スレッドを起動 =====
func fetchURL(url string) ([]byte, error) {
    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    return io.ReadAll(resp.Body)
}

func main() {
    // 通常の関数呼び出し（同期）
    data, _ := fetchURL("https://api.example.com/data")
    fmt.Println(string(data))

    // goroutine で非同期実行
    go func() {
        data, _ := fetchURL("https://api.example.com/other")
        fmt.Println(string(data))
    }()

    // goroutine が完了する前に main が終了しないよう待機
    time.Sleep(2 * time.Second)
}
```

### 7.2 channel による通信

```go
// ===== channel: goroutine 間の通信チャネル =====

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
    ch <- user // チャネルに送信
}

func main() {
    ch := make(chan User, 3) // バッファ付きチャネル

    // 3つの goroutine を並行起動
    for i := 1; i <= 3; i++ {
        go fetchUser(i, ch)
    }

    // 3つの結果を収集
    users := make([]User, 0, 3)
    for i := 0; i < 3; i++ {
        user := <-ch // チャネルから受信
        users = append(users, user)
    }
    fmt.Println(users)
}

// ===== select: 複数チャネルの多重化 =====
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

// ===== WaitGroup: goroutine の完了を待機 =====
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

    wg.Wait() // 全 goroutine の完了を待機
    return results
}

// ===== context: キャンセルとタイムアウト =====
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
    // 5秒のタイムアウト付きコンテキスト
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    data, err := fetchWithContext(ctx, "https://api.example.com/data")
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            fmt.Println("リクエストがタイムアウトしました")
        }
        return
    }
    fmt.Println(string(data))
}
```

```
Go の goroutine スケジューリングモデル（M:N マッピング）:

  OS スレッド (M)        goroutine (G)        プロセッサ (P)
  ┌──────────┐           ┌────┐┌────┐         ┌──────────┐
  │ Thread-1 │<──────────│ G1 ││ G2 │<────────│ P1       │
  │          │           └────┘└────┘         │ LocalQ   │
  └──────────┘           ┌────┐┌────┐         └──────────┘
  ┌──────────┐           │ G3 ││ G4 │
  │ Thread-2 │<──────────└────┘└────┘         ┌──────────┐
  │          │           ┌────┐               │ P2       │
  └──────────┘           │ G5 │<──────────────│ LocalQ   │
  ┌──────────┐           └────┘               └──────────┘
  │ Thread-3 │ (idle)
  │          │           ┌────┐┌────┐┌────┐
  └──────────┘           │ G6 ││ G7 ││ G8 │   ← Global Queue
                         └────┘└────┘└────┘

  - G (goroutine): 約 2-8 KB のスタック（動的に拡張）
  - M (Machine/OS Thread): OS カーネルスレッド
  - P (Processor): 論理プロセッサ（GOMAXPROCS で設定）

  ワークスティーリング:
    P1 のキューが空 → P2 のキューから G を「盗む」
    → 全プロセッサが均等に負荷分散
```

---

## 8. C# と Java の非同期モデル

### 8.1 C# の async/await（Task ベース）

```csharp
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

// ===== 基本的な async/await =====
public class ApiClient
{
    private readonly HttpClient _httpClient = new HttpClient();

    public async Task<string> FetchDataAsync(string url)
    {
        // await: Task の完了を非同期に待機
        HttpResponseMessage response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();
        string content = await response.Content.ReadAsStringAsync();
        return content;
    }

    // ===== 並行実行: Task.WhenAll =====
    public async Task<Dashboard> LoadDashboardAsync(int userId)
    {
        // 3つのタスクを同時に開始
        Task<User> userTask = FetchUserAsync(userId);
        Task<List<Post>> postsTask = FetchPostsAsync(userId);
        Task<List<Notification>> notifsTask = FetchNotificationsAsync(userId);

        // 全タスクの完了を待機
        await Task.WhenAll(userTask, postsTask, notifsTask);

        return new Dashboard
        {
            User = userTask.Result,
            Posts = postsTask.Result,
            Notifications = notifsTask.Result,
        };
    }

    // ===== CancellationToken: キャンセル制御 =====
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
            Console.WriteLine("リクエストがキャンセルされました");
            return null;
        }
    }

    // ===== タイムアウト付きリクエスト =====
    public async Task<string> FetchWithTimeoutAsync(
        string url, TimeSpan timeout)
    {
        using var cts = new CancellationTokenSource(timeout);
        return await FetchWithCancellationAsync(url, cts.Token);
    }
}

// ===== IAsyncEnumerable: 非同期ストリーム（C# 8.0+） =====
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

// 使用例
await foreach (var items in api.FetchAllPagesAsync("/api/items"))
{
    foreach (var item in items)
    {
        await ProcessItemAsync(item);
    }
}
```

### 8.2 Java の Virtual Threads（Project Loom, Java 21+）

```java
import java.net.http.*;
import java.net.URI;
import java.time.Duration;
import java.util.concurrent.*;
import java.util.List;
import java.util.stream.Collectors;

// ===== Virtual Threads: 軽量スレッド（Java 21+） =====
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
        // ===== 方法1: Virtual Thread を直接起動 =====
        Thread vThread = Thread.ofVirtual().start(() -> {
            try {
                String data = fetchData("https://api.example.com/data");
                System.out.println(data);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        vThread.join();

        // ===== 方法2: ExecutorService + Virtual Threads =====
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<String>> futures = List.of(
                executor.submit(() -> fetchData("https://api.example.com/users")),
                executor.submit(() -> fetchData("https://api.example.com/posts")),
                executor.submit(() -> fetchData("https://api.example.com/notifications"))
            );

            for (Future<String> future : futures) {
                System.out.println(future.get()); // 結果取得
            }
        }

        // ===== 方法3: StructuredTaskScope（Structured Concurrency Preview） =====
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            Future<String> users = scope.fork(() ->
                fetchData("https://api.example.com/users"));
            Future<String> posts = scope.fork(() ->
                fetchData("https://api.example.com/posts"));

            scope.join();           // 全タスクの完了を待機
            scope.throwIfFailed();  // いずれか失敗なら例外

            System.out.println(users.resultNow());
            System.out.println(posts.resultNow());
        }
    }
}

// ===== CompletableFuture: 従来の非同期API =====
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
        // チェーン: thenApply, thenCompose, thenCombine
        CompletableFuture<String> future = fetchAsync("/api/users/1")
            .thenApply(json -> parseUser(json))        // 変換
            .thenCompose(user -> fetchAsync(            // 次の非同期処理
                "/api/posts?userId=" + user.getId()))
            .exceptionally(ex -> {                     // エラー処理
                System.err.println("Error: " + ex.getMessage());
                return "[]";
            });

        // 並行実行: allOf
        CompletableFuture<Void> all = CompletableFuture.allOf(
            fetchAsync("/api/users"),
            fetchAsync("/api/posts"),
            fetchAsync("/api/notifications")
        );
        all.join(); // 全完了を待機
    }
}
```

---

## 9. 非同期ランタイムの比較分析

### 9.1 アーキテクチャ比較表

```
┌────────────┬─────────────────┬──────────────────┬───────────────────┐
│ 特性        │ JavaScript      │ Python           │ Rust              │
│            │ (Node.js)       │ (asyncio)        │ (tokio)           │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ 実行モデル  │ シングルスレッド │ シングルスレッド  │ マルチスレッド     │
│            │ イベントループ   │ イベントループ    │ ワークスティーリング│
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ I/Oバック   │ libuv           │ selector/epoll   │ mio(epoll/kqueue) │
│ エンド      │ (クロスPF)      │ (OS依存)         │ (クロスPF)        │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ CPU並列性   │ Worker Threads  │ multiprocessing  │ ネイティブスレッド │
│            │ (制限的)        │ (GIL回避)        │ (完全並列)        │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ メモリ      │ V8ヒープ        │ オブジェクト      │ ゼロコスト         │
│ オーバーヘッド│ (コルーチン中)  │ (コルーチン中)   │ (ステートマシン)  │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ キャンセル  │ AbortController │ Task.cancel()    │ drop (RAII)       │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ 成熟度      │ 非常に高い      │ 高い             │ 高い              │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ 学習曲線    │ 低い            │ 中程度           │ 高い              │
└────────────┴─────────────────┴──────────────────┴───────────────────┘

┌────────────┬─────────────────┬──────────────────┬───────────────────┐
│ 特性        │ Go              │ C#               │ Java (21+)        │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ 実行モデル  │ M:Nスケジューリング│ スレッドプール   │ Virtual Threads   │
│            │ ワークスティーリング│ + async/await   │ M:Nスケジューリング│
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ I/Oバック   │ netpoller       │ OS依存           │ NIO / Virtual     │
│ エンド      │ (ランタイム内蔵) │ (IOCP/epoll)    │ Thread挿入        │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ CPU並列性   │ goroutine自動分散│ Task並列ライブラリ│ スレッドプール     │
│            │ (GOMAXPROCS)    │ (Parallel.For等) │ (ForkJoinPool)    │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ メモリ      │ goroutine: ~2KB │ Task: ~300B      │ VThread: ~数百B   │
│ オーバーヘッド│ (動的スタック)  │ (ヒープ確保)     │ (継続ベース)      │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ キャンセル  │ context.Context │ CancellationToken│ Thread.interrupt  │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ 成熟度      │ 非常に高い      │ 非常に高い       │ 成長中            │
├────────────┼─────────────────┼──────────────────┼───────────────────┤
│ 学習曲線    │ 低い            │ 中程度           │ 低い(VThread)     │
└────────────┴─────────────────┴──────────────────┴───────────────────┘
```

### 9.2 用途別推奨モデル

```
ユースケース別の非同期モデル選択ガイド:

┌───────────────────────────┬──────────────────────────────────┐
│ ユースケース               │ 推奨モデル                        │
├───────────────────────────┼──────────────────────────────────┤
│ Webフロントエンド          │ JavaScript async/await           │
│ (ブラウザ)                │ → 唯一の選択肢、成熟したエコシステム│
├───────────────────────────┼──────────────────────────────────┤
│ REST API サーバー          │ Go (goroutine) / Node.js         │
│ (高スループット)           │ → シンプルかつ高性能              │
├───────────────────────────┼──────────────────────────────────┤
│ マイクロサービス           │ Go / Rust / Java (VThread)       │
│ (低レイテンシ)            │ → Go: 開発速度、Rust: 最高性能    │
├───────────────────────────┼──────────────────────────────────┤
│ データパイプライン         │ Python (asyncio + multiprocessing)│
│ (I/O + CPU混在)           │ → 豊富なデータ処理ライブラリ      │
├───────────────────────────┼──────────────────────────────────┤
│ リアルタイム通信           │ Go / Rust / Node.js              │
│ (WebSocket等)             │ → 大量接続に強い                  │
├───────────────────────────┼──────────────────────────────────┤
│ エンタープライズ           │ C# / Java                        │
│ (既存資産活用)            │ → エコシステムと互換性             │
├───────────────────────────┼──────────────────────────────────┤
│ 組み込み/IoT              │ Rust (embassy/no_std)            │
│ (リソース制約)            │ → ゼロコスト、ランタイムなし可     │
└───────────────────────────┴──────────────────────────────────┘
```

---

## 10. アンチパターンと落とし穴

### 10.1 アンチパターン1: 逐次 await の罠（Sequential Await Trap）

独立した非同期処理を逐次 await すると、本来並行実行できる処理が直列化されてしまう。

```javascript
// ===== BAD: 逐次 await（不必要に遅い） =====
async function loadPageData(userId) {
    // これら3つの処理は互いに依存しないが、直列に実行される
    const user = await fetchUser(userId);        // 200ms 待ち
    const posts = await fetchPosts(userId);       // 300ms 待ち
    const notifications = await fetchNotifications(userId); // 150ms 待ち
    // 合計: 200 + 300 + 150 = 650ms

    return { user, posts, notifications };
}

// ===== GOOD: 並行 await（高速） =====
async function loadPageData(userId) {
    // Promise.all で並行実行
    const [user, posts, notifications] = await Promise.all([
        fetchUser(userId),        // ─┐
        fetchPosts(userId),       // ─┤ 並行実行
        fetchNotifications(userId), // ─┘
    ]);
    // 合計: max(200, 300, 150) = 300ms（約54%高速化）

    return { user, posts, notifications };
}

// ===== BEST: 依存関係を考慮した最適化 =====
async function loadPageData(userId) {
    // Step 1: user は他の処理に必要なので先に取得
    const user = await fetchUser(userId);

    // Step 2: posts と notifications は並行可能
    const [posts, notifications] = await Promise.all([
        fetchPosts(user.id),
        fetchNotifications(user.id),
    ]);

    // Step 3: comments は posts に依存
    const comments = await fetchComments(posts[0]?.id);

    return { user, posts, notifications, comments };
}
```

```
逐次 vs 並行の実行時間比較:

逐次実行:
  fetchUser ──[200ms]──┐
                       │
  fetchPosts ──────────┴──[300ms]──┐
                                   │
  fetchNotifs ─────────────────────┴──[150ms]──┐
                                               │
  合計: ────────────────────────────────────────┘ 650ms

並行実行:
  fetchUser ──[200ms]──┐
  fetchPosts ──[300ms]─┤ ← 最も遅い処理がボトルネック
  fetchNotifs [150ms]──┘
                       │
  合計: ───────────────┘ 300ms（54% 削減）
```

### 10.2 アンチパターン2: async/await の不適切な混在

```javascript
// ===== BAD: Promise と async/await の混在による混乱 =====
async function processData() {
    // .then() 内で await を使おうとする（エラーにはならないが紛らわしい）
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

    return result; // これは Promise<Promise<...>> になりうる
}

// ===== GOOD: async/await で統一 =====
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

### 10.3 アンチパターン3: Fire-and-Forget（投げっぱなし）

```javascript
// ===== BAD: await なしで非同期関数を呼ぶ =====
async function handleRequest(req, res) {
    const data = await fetchData(req.params.id);

    // ログ記録を非同期で呼ぶが、await していない
    saveAuditLog(req.user, 'data_access');  // ← Promise が浮遊

    // 問題1: エラーが捕捉されない（Unhandled Promise Rejection）
    // 問題2: プロセス終了時にログが失われる可能性
    // 問題3: デバッグが困難

    res.json(data);
}

// ===== GOOD: 意図的な Fire-and-Forget には明示的にエラー処理 =====
async function handleRequest(req, res) {
    const data = await fetchData(req.params.id);

    // 方法1: catch で明示的にエラーを処理
    saveAuditLog(req.user, 'data_access')
        .catch(err => console.error('Audit log failed:', err));

    // 方法2: void 演算子で意図的であることを明示
    void saveAuditLog(req.user, 'data_access')
        .catch(err => console.error('Audit log failed:', err));

    res.json(data);
}

// ===== BEST: バックグラウンドタスクキューを使う =====
async function handleRequest(req, res) {
    const data = await fetchData(req.params.id);

    // タスクキューに投入（リトライ・監視付き）
    backgroundQueue.enqueue('audit_log', {
        user: req.user,
        action: 'data_access',
        timestamp: Date.now(),
    });

    res.json(data);
}
```

### 10.4 アンチパターン4: async 関数内での同期的ブロッキング

```python
import asyncio
import time

# ===== BAD: async 関数内で同期的にブロック =====
async def process_data():
    data = await fetch_data()

    # time.sleep はイベントループをブロックする!
    time.sleep(5)  # ← 全ての非同期処理が5秒間停止

    # CPU密集処理もイベントループをブロック
    result = heavy_computation(data)  # ← 計算中は他の処理が進まない

    return result

# ===== GOOD: 非同期スリープとスレッドプールを使う =====
async def process_data():
    data = await fetch_data()

    # asyncio.sleep は他の処理をブロックしない
    await asyncio.sleep(5)

    # CPU密集処理はスレッドプールで実行
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, heavy_computation, data)

    return result
```

### 10.5 アンチパターン5: Deadlock（デッドロック）

```python
# ===== BAD: 同期コード内から asyncio.run() をネスト呼び出し =====
import asyncio

async def inner():
    return "result"

async def outer():
    # 既にイベントループが動作中に asyncio.run() を呼ぶとデッドロック
    result = asyncio.run(inner())  # RuntimeError!
    return result

# ===== GOOD: await を使う =====
async def outer():
    result = await inner()  # 正しい方法
    return result
```

```csharp
// ===== BAD: C# での .Result / .Wait() によるデッドロック =====
// ASP.NET / WPF など SynchronizationContext がある環境で危険
public string GetData()
{
    // async メソッドを .Result で同期的に待機
    // → SynchronizationContext のキャプチャによりデッドロック
    var result = FetchDataAsync().Result;  // ← デッドロック!
    return result;
}

// ===== GOOD: async を末端まで伝播させる =====
public async Task<string> GetDataAsync()
{
    var result = await FetchDataAsync();  // 正しい
    return result;
}

// ConfigureAwait(false) で SynchronizationContext を無視
public async Task<string> GetDataAsync()
{
    var result = await FetchDataAsync().ConfigureAwait(false);
    return result;
}
```
```
