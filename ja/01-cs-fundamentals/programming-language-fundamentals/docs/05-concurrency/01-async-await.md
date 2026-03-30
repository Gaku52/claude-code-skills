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


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [スレッドとプロセス](./00-threads-and-processes.md) の内容を理解していること

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

---

## 11. 設計パターンとベストプラクティス

### 11.1 リトライパターン（Exponential Backoff）

```javascript
// ===== 指数バックオフ付きリトライ =====
async function fetchWithRetry(url, options = {}) {
    const {
        maxRetries = 3,
        baseDelay = 1000,    // 初回待機: 1秒
        maxDelay = 30000,    // 最大待機: 30秒
        backoffFactor = 2,   // 倍率
        retryableStatuses = [408, 429, 500, 502, 503, 504],
    } = options;

    let lastError;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const response = await fetch(url);

            if (response.ok) {
                return await response.json();
            }

            if (!retryableStatuses.includes(response.status)) {
                throw new Error(`Non-retryable HTTP ${response.status}`);
            }

            lastError = new Error(`HTTP ${response.status}`);
        } catch (error) {
            lastError = error;

            if (error.name === 'AbortError') {
                throw error; // キャンセルはリトライしない
            }
        }

        if (attempt < maxRetries) {
            // 指数バックオフ + ジッタ
            const delay = Math.min(
                baseDelay * Math.pow(backoffFactor, attempt),
                maxDelay
            );
            const jitter = delay * (0.5 + Math.random() * 0.5);

            console.log(
                `Retry ${attempt + 1}/${maxRetries} after ${Math.round(jitter)}ms`
            );
            await new Promise(resolve => setTimeout(resolve, jitter));
        }
    }

    throw lastError;
}

// 使用例
const data = await fetchWithRetry('https://api.example.com/data', {
    maxRetries: 5,
    baseDelay: 500,
});
```

```
指数バックオフの可視化:

リトライ回数   待機時間（ジッタなし）   待機時間（ジッタあり、概算）
    0          リクエスト送信            リクエスト送信
    ↓          失敗                      失敗
    1          1,000ms                   500〜1,500ms
    ↓          失敗                      失敗
    2          2,000ms                   1,000〜3,000ms
    ↓          失敗                      失敗
    3          4,000ms                   2,000〜6,000ms
    ↓          失敗                      失敗
    4          8,000ms                   4,000〜12,000ms
    ↓          失敗                      失敗
    5          諦める                    諦める

  ジッタ（Jitter）が重要な理由:
    多数のクライアントが同時に失敗した場合、
    全クライアントが同じタイミングでリトライすると
    サーバーに「リトライストーム」が発生する。
    ランダムなジッタを加えることで負荷を分散する。
```

### 11.2 サーキットブレーカーパターン

```javascript
// ===== サーキットブレーカー: 障害の連鎖を防ぐ =====
class CircuitBreaker {
    constructor(options = {}) {
        this.failureThreshold = options.failureThreshold || 5;
        this.resetTimeout = options.resetTimeout || 60000; // 60秒
        this.state = 'CLOSED';  // CLOSED → OPEN → HALF_OPEN → CLOSED
        this.failureCount = 0;
        this.lastFailureTime = null;
        this.successCount = 0;
    }

    async execute(asyncFn) {
        if (this.state === 'OPEN') {
            if (Date.now() - this.lastFailureTime > this.resetTimeout) {
                this.state = 'HALF_OPEN';
                this.successCount = 0;
            } else {
                throw new Error('Circuit breaker is OPEN');
            }
        }

        try {
            const result = await asyncFn();
            this._onSuccess();
            return result;
        } catch (error) {
            this._onFailure();
            throw error;
        }
    }

    _onSuccess() {
        if (this.state === 'HALF_OPEN') {
            this.successCount++;
            if (this.successCount >= 3) {
                this.state = 'CLOSED';
                this.failureCount = 0;
            }
        } else {
            this.failureCount = 0;
        }
    }

    _onFailure() {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
        }
    }
}

// 使用例
const breaker = new CircuitBreaker({
    failureThreshold: 5,
    resetTimeout: 30000,
});

async function callExternalService() {
    return breaker.execute(async () => {
        const response = await fetch('https://external-api.example.com/data');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    });
}
```

```
サーキットブレーカーの状態遷移:

  ┌──────────┐  失敗がしきい値超過   ┌──────────┐
  │ CLOSED   │ ────────────────────> │  OPEN    │
  │ (正常)   │                       │ (遮断)   │
  │          │                       │          │
  │ リクエスト│                       │ 即座に   │
  │ を通す   │                       │ エラー返却│
  └────▲─────┘                       └────┬─────┘
       │                                  │
       │ 連続成功で回復                     │ リセットタイムアウト経過
       │                                  │
  ┌────┴─────┐                       ┌────▼─────┐
  │          │ <──────────────────── │ HALF_OPEN│
  │          │  テスト成功            │ (半開)   │
  │          │                       │          │
  └──────────┘  テスト失敗 → OPEN    │ 限定的に │
                に戻る               │ リクエスト│
                                     │ を通す   │
                                     └──────────┘
```

### 11.3 バルクヘッドパターン（セマフォによるリソース分離）

```python
import asyncio

class BulkheadExecutor:
    """
    バルクヘッドパターン: サービスごとに同時実行数を分離し、
    1つのサービス障害が他のサービスに波及するのを防ぐ
    """

    def __init__(self):
        self.semaphores = {}

    def register(self, service_name: str, max_concurrent: int):
        """サービスごとの同時実行数上限を登録"""
        self.semaphores[service_name] = asyncio.Semaphore(max_concurrent)

    async def execute(self, service_name: str, coro):
        """指定サービスのバルクヘッド内で非同期処理を実行"""
        sem = self.semaphores.get(service_name)
        if sem is None:
            raise ValueError(f"Unknown service: {service_name}")

        async with sem:
            return await coro

# 使用例
bulkhead = BulkheadExecutor()
bulkhead.register("user_service", max_concurrent=20)
bulkhead.register("payment_service", max_concurrent=5)   # 少なめに制限
bulkhead.register("notification_service", max_concurrent=50)

async def handle_order(order_id: int):
    user = await bulkhead.execute(
        "user_service", fetch_user(order_id)
    )
    payment = await bulkhead.execute(
        "payment_service", process_payment(order_id)
    )
    await bulkhead.execute(
        "notification_service", send_notification(user, payment)
    )
```

---

## 12. 演習問題

### 12.1 基礎演習（Beginner）

**演習1: 非同期データ取得の基本**

以下の仕様を満たす `fetchAllUsers` 関数を JavaScript で実装せよ。

- 引数: ユーザーIDの配列 `userIds: number[]`
- `/api/users/:id` からユーザー情報を並行取得する
- 全てのユーザー情報を配列で返す
- 1つでも失敗した場合はエラーをスローする

```javascript
// 解答テンプレート
async function fetchAllUsers(userIds) {
    // ここに実装
}

// テスト
// const users = await fetchAllUsers([1, 2, 3, 4, 5]);
// console.log(users); // [{id:1,name:"Alice"}, {id:2,...}, ...]
```

<details>
<summary>解答例（クリックして展開）</summary>

```javascript
async function fetchAllUsers(userIds) {
    const promises = userIds.map(id =>
        fetch(`/api/users/${id}`).then(res => {
            if (!res.ok) throw new Error(`User ${id}: HTTP ${res.status}`);
            return res.json();
        })
    );

    return Promise.all(promises);
}

// 改良版: 同時接続数を制限
async function fetchAllUsersLimited(userIds, concurrency = 5) {
    const results = [];
    const executing = new Set();

    for (const id of userIds) {
        const promise = fetch(`/api/users/${id}`)
            .then(res => res.json())
            .then(user => {
                executing.delete(promise);
                return user;
            });

        executing.add(promise);
        results.push(promise);

        if (executing.size >= concurrency) {
            await Promise.race(executing);
        }
    }

    return Promise.all(results);
}
```

</details>

**演習2: Python でタイムアウト付きフェッチを実装**

以下の仕様を満たす関数を Python で実装せよ。

- 複数の URL を並行フェッチする
- 各リクエストに個別のタイムアウト（デフォルト5秒）を設定する
- タイムアウトした URL は結果から除外し、成功したもののみ返す
- 全体のタイムアウト（デフォルト30秒）も設定可能

```python
# 解答テンプレート
async def fetch_urls_with_timeout(
    urls: list[str],
    per_request_timeout: float = 5.0,
    total_timeout: float = 30.0,
) -> dict[str, str]:
    """成功した URL と結果のマッピングを返す"""
    # ここに実装
    pass
```

<details>
<summary>解答例（クリックして展開）</summary>

```python
import asyncio
import aiohttp

async def fetch_urls_with_timeout(
    urls: list[str],
    per_request_timeout: float = 5.0,
    total_timeout: float = 30.0,
) -> dict[str, str]:
    results = {}

    async def fetch_one(session: aiohttp.ClientSession, url: str):
        try:
            async with asyncio.timeout(per_request_timeout):
                async with session.get(url) as resp:
                    text = await resp.text()
                    results[url] = text
        except (TimeoutError, aiohttp.ClientError) as e:
            print(f"Failed {url}: {e}")

    async with asyncio.timeout(total_timeout):
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(
                *[fetch_one(session, url) for url in urls],
                return_exceptions=True,
            )

    return results
```

</details>

### 12.2 中級演習（Intermediate）

**演習3: 非同期レートリミッター**

以下の仕様を満たすレートリミッターを JavaScript で実装せよ。

- トークンバケットアルゴリズムを使用
- 1秒あたり最大 N リクエストに制限
- 制限超過時はトークン補充まで自動待機
- async/await で利用可能なインターフェース

```javascript
// 解答テンプレート
class RateLimiter {
    constructor(maxTokens, refillRate) {
        // ここに実装
    }

    async acquire() {
        // トークン取得（制限超過時は待機）
    }
}

// 使用例
// const limiter = new RateLimiter(10, 10); // 最大10トークン、毎秒10トークン補充
// await limiter.acquire(); // トークン取得
// await fetch('/api/data'); // リクエスト実行
```

<details>
<summary>解答例（クリックして展開）</summary>

```javascript
class RateLimiter {
    constructor(maxTokens, refillRatePerSecond) {
        this.maxTokens = maxTokens;
        this.tokens = maxTokens;
        this.refillRate = refillRatePerSecond;
        this.lastRefillTime = Date.now();
        this.waitQueue = [];
    }

    _refill() {
        const now = Date.now();
        const elapsed = (now - this.lastRefillTime) / 1000;
        this.tokens = Math.min(
            this.maxTokens,
            this.tokens + elapsed * this.refillRate
        );
        this.lastRefillTime = now;
    }

    async acquire() {
        this._refill();

        if (this.tokens >= 1) {
            this.tokens -= 1;
            return;
        }

        // トークン不足: 補充まで待機
        const waitTime = ((1 - this.tokens) / this.refillRate) * 1000;

        return new Promise(resolve => {
            setTimeout(() => {
                this._refill();
                this.tokens -= 1;
                resolve();
                // キューに待機中のものがあれば処理
                this._processQueue();
            }, waitTime);
        });
    }

    _processQueue() {
        while (this.waitQueue.length > 0) {
            this._refill();
            if (this.tokens < 1) break;
            this.tokens -= 1;
            const resolve = this.waitQueue.shift();
            resolve();
        }
    }
}

// 使用例: API呼び出しをレートリミット
async function fetchWithRateLimit(urls) {
    const limiter = new RateLimiter(10, 10);
    const results = [];

    for (const url of urls) {
        await limiter.acquire();
        results.push(fetch(url).then(r => r.json()));
    }

    return Promise.all(results);
}
```

</details>

### 12.3 上級演習（Advanced）

**演習4: 非同期タスクスケジューラ**

以下の仕様を満たすタスクスケジューラを実装せよ。言語は自由に選択可。

要件:
- 優先度付きタスクキュー（high / medium / low）
- 同時実行数の上限を指定可能
- タスクの依存関係を宣言可能（タスクAはタスクBの完了後に実行）
- タイムアウト機能
- キャンセル機能

```javascript
// インターフェース例
class TaskScheduler {
    constructor(maxConcurrent) { /* ... */ }

    addTask(id, asyncFn, options) {
        // options: { priority, dependsOn, timeout }
    }

    async run() {
        // 全タスクを依存関係と優先度に従って実行
    }

    cancel(taskId) { /* ... */ }
}
```

<details>
<summary>解答例（クリックして展開）</summary>

```javascript
class TaskScheduler {
    constructor(maxConcurrent = 5) {
        this.maxConcurrent = maxConcurrent;
        this.tasks = new Map();
        this.results = new Map();
        this.running = new Set();
        this.completed = new Set();
        this.cancelled = new Set();
    }

    addTask(id, asyncFn, options = {}) {
        const { priority = 'medium', dependsOn = [], timeout = 0 } = options;
        const priorityValue = { high: 0, medium: 1, low: 2 }[priority];

        this.tasks.set(id, {
            id,
            fn: asyncFn,
            priority: priorityValue,
            dependsOn,
            timeout,
        });
    }

    cancel(taskId) {
        this.cancelled.add(taskId);
    }

    _getReadyTasks() {
        const ready = [];
        for (const [id, task] of this.tasks) {
            if (this.completed.has(id) || this.running.has(id) || this.cancelled.has(id)) {
                continue;
            }
            const depsReady = task.dependsOn.every(dep => this.completed.has(dep));
            if (depsReady) {
                ready.push(task);
            }
        }
        return ready.sort((a, b) => a.priority - b.priority);
    }

    async _executeTask(task) {
        this.running.add(task.id);

        try {
            let result;
            if (task.timeout > 0) {
                result = await Promise.race([
                    task.fn(this.results),
                    new Promise((_, reject) =>
                        setTimeout(() => reject(new Error(`Task ${task.id} timed out`)), task.timeout)
                    ),
                ]);
            } else {
                result = await task.fn(this.results);
            }
            this.results.set(task.id, { status: 'fulfilled', value: result });
        } catch (error) {
            this.results.set(task.id, { status: 'rejected', reason: error });
        } finally {
            this.running.delete(task.id);
            this.completed.add(task.id);
        }
    }

    async run() {
        while (true) {
            const allDone = [...this.tasks.keys()].every(
                id => this.completed.has(id) || this.cancelled.has(id)
            );
            if (allDone) break;

            const ready = this._getReadyTasks();
            const slots = this.maxConcurrent - this.running.size;

            if (ready.length === 0 && this.running.size === 0) {
                // デッドロック検出: 実行可能なタスクも実行中タスクもない
                const remaining = [...this.tasks.keys()].filter(
                    id => !this.completed.has(id) && !this.cancelled.has(id)
                );
                throw new Error(`Deadlock detected. Blocked tasks: ${remaining.join(', ')}`);
            }

            const toRun = ready.slice(0, slots);
            const promises = toRun.map(task => this._executeTask(task));

            if (promises.length > 0) {
                await Promise.race([...promises, ...Array.from(this.running)].filter(Boolean));
            } else if (this.running.size > 0) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        return this.results;
    }
}

// 使用例
const scheduler = new TaskScheduler(3);

scheduler.addTask('fetchUser', async () => {
    return await fetch('/api/users/1').then(r => r.json());
}, { priority: 'high' });

scheduler.addTask('fetchPosts', async (results) => {
    const user = results.get('fetchUser').value;
    return await fetch(`/api/posts?userId=${user.id}`).then(r => r.json());
}, { priority: 'medium', dependsOn: ['fetchUser'], timeout: 5000 });

scheduler.addTask('fetchAnalytics', async () => {
    return await fetch('/api/analytics').then(r => r.json());
}, { priority: 'low' });

const results = await scheduler.run();
```

</details>

---

## 13. FAQ（よくある質問）

### Q1: async/await とマルチスレッドの違いは何か？

**A1:** async/await は「並行性（Concurrency）」を提供し、マルチスレッドは「並列性（Parallelism）」を提供する。

```
並行性 (Concurrency):
  1つの実行主体が複数タスクを切り替えながら処理する
  → 1コアでも実現可能
  → I/O 待ちの間に他の処理を進める

  Thread-1: ─[Task-A]─[Task-B]─[Task-A]─[Task-C]─[Task-B]─

並列性 (Parallelism):
  複数の実行主体が同時に処理する
  → 複数コアが必要
  → CPU密集処理の高速化

  Core-1: ─[Task-A]─[Task-A]─[Task-A]─
  Core-2: ─[Task-B]─[Task-B]─[Task-B]─
  Core-3: ─[Task-C]─[Task-C]─[Task-C]─

async/await が有効な場面:
  - ネットワーク I/O（HTTP, DB, WebSocket）
  - ファイル I/O
  - タイマー処理
  → 待ち時間が多く、CPU はあまり使わない処理

マルチスレッドが有効な場面:
  - 画像処理・動画エンコード
  - 科学計算・機械学習
  - 暗号処理
  → CPU をフルに使う計算処理
```

### Q2: JavaScript で await を忘れるとどうなるか？

**A2:** Promise オブジェクトがそのまま返され、意図しない動作になる。

```javascript
// ===== await を忘れた場合の挙動 =====

async function getUser(id) {
    const response = fetch(`/api/users/${id}`);  // await がない!

    // response は Response オブジェクトではなく Promise オブジェクト
    console.log(response);          // Promise { <pending> }
    console.log(response.status);   // undefined（Promise にはstatusプロパティがない）

    // JSON パースも失敗する
    // response.json() → TypeError: response.json is not a function
}

// ===== 正しい書き方 =====
async function getUser(id) {
    const response = await fetch(`/api/users/${id}`);  // await あり
    console.log(response);          // Response {...}
    console.log(response.status);   // 200
    return await response.json();
}

// ===== ESLint で検出可能 =====
// eslint ルール: "@typescript-eslint/no-floating-promises": "error"
// → await されていない Promise を検出してエラーにする
```

### Q3: Python の asyncio はマルチコアを活用できるか？

**A3:** asyncio 単体ではマルチコアを活用できない（GIL の制約）。CPU密集処理には `multiprocessing` や `ProcessPoolExecutor` を組み合わせる。

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# CPU密集処理
def cpu_heavy_task(data):
    """GIL の影響を受ける CPU 密集処理"""
    return sum(x ** 2 for x in range(data))

async def process_with_multiprocessing(items: list[int]) -> list[int]:
    """asyncio + ProcessPoolExecutor で CPU 並列処理"""
    loop = asyncio.get_event_loop()

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # 各 CPU コアで並列実行
        futures = [
            loop.run_in_executor(executor, cpu_heavy_task, item)
            for item in items
        ]
        results = await asyncio.gather(*futures)

    return results

# I/O と CPU のハイブリッド
async def hybrid_pipeline(urls: list[str]) -> list:
    # Phase 1: I/O 密集（asyncio で並行）
    raw_data = await asyncio.gather(
        *[fetch_data(url) for url in urls]
    )

    # Phase 2: CPU 密集（ProcessPoolExecutor で並列）
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        processed = await asyncio.gather(
            *[loop.run_in_executor(executor, process, data) for data in raw_data]
        )

    return processed
```

### Q4: Rust で async fn の返り値の型が複雑になるのはなぜか？

**A4:** Rust の Future は各 async fn ごとに固有の匿名型（ステートマシン）を生成するため、型を明示するのが困難である。

```rust
// async fn が生成する Future の型は匿名
async fn fetch_data() -> String {
    // コンパイラはこの関数を以下のようなステートマシンに変換:
    // enum FetchDataFuture {
    //     State0 { ... },  // await 前
    //     State1 { ... },  // 最初の await 後
    //     State2 { ... },  // 2番目の await 後
    //     Done,
    // }
    let resp = reqwest::get("https://example.com").await.unwrap();
    resp.text().await.unwrap()
}

// 関数ポインタとして保持する場合:
// 方法1: impl Future（静的ディスパッチ、推奨）
fn make_future() -> impl Future<Output = String> {
    fetch_data()
}

// 方法2: Box<dyn Future>（動的ディスパッチ、ヒープ割り当て）
fn make_boxed_future() -> Pin<Box<dyn Future<Output = String> + Send>> {
    Box::pin(fetch_data())
}

// 方法3: トレイトオブジェクトとして保持（コレクションに格納する場合）
async fn run_multiple() {
    let futures: Vec<Pin<Box<dyn Future<Output = String>>>> = vec![
        Box::pin(fetch_data()),
        Box::pin(another_async_fn()),
    ];

    for future in futures {
        let result = future.await;
        println!("{}", result);
    }
}
```

### Q5: async/await を使うべきでない場面はあるか？

**A5:** 以下の場合は async/await が不適切または不要である。

```
async/await が不適切なケース:

1. CPU 密集処理のみの場合
   → マルチスレッド/並列処理を使うべき
   → async/await は I/O 待ちの効率化が目的

2. 処理が全て同期的に完了する場合
   → 不要なオーバーヘッドが発生
   → 同期関数でそのまま書けばよい

3. 共有状態が多い並行処理
   → async/await + 共有状態 = データ競合のリスク
   → Actor モデルや CSP（Go の channel）の方が安全

4. 超低レイテンシが要求される場合
   → イベントループのオーバーヘッドが問題になりうる
   → io_uring（Linux）や直接的なシステムコールが有効

5. 既存の同期APIを薄くラップするだけの場合
   → async で包んでも内部が同期なら意味がない
   → Python: asyncio.to_thread() で適切にオフロード
```

### Q6: なぜ Go は async/await を採用しなかったのか？

**A6:** Go の設計哲学は「シンプルさ」を最優先する。goroutine + channel の CSP モデルにより、全ての関数が暗黙的に非同期対応となり、async/await のような構文的な区別が不要になる。

```go
// Go: 全ての関数が「同期に見える」
func fetchUser(id int) (User, error) {
    // 内部で I/O が発生しても、goroutine スケジューラが
    // 自動的にコンテキストスイッチする
    resp, err := http.Get(fmt.Sprintf("/api/users/%d", id))
    if err != nil {
        return User{}, err
    }
    defer resp.Body.Close()

    var user User
    err = json.NewDecoder(resp.Body).Decode(&user)
    return user, err
}

// 並行実行も goroutine + channel で自然に書ける
func main() {
    ch := make(chan User, 3)
    for i := 1; i <= 3; i++ {
        go func(id int) {
            user, _ := fetchUser(id)
            ch <- user
        }(i)
    }

    for i := 0; i < 3; i++ {
        user := <-ch
        fmt.Println(user)
    }
}
```

Go の利点: 関数のシグネチャに async が伝染しない（関数の色問題が存在しない）。
Go の欠点: goroutine のリーク検出が難しい、明示的なキャンセルが必要（context パッケージ）。

---

## 14. 非同期処理のデバッグとテスト

### 14.1 デバッグ手法

```javascript
// ===== 非同期処理のデバッグテクニック =====

// 1. async スタックトレースの有効化
// Node.js: --async-stack-traces フラグ（デフォルトで有効、v12+）

// 2. Promise のラベリング
async function fetchUser(id) {
    const promise = fetch(`/api/users/${id}`)
        .then(r => r.json());

    // デバッグ用にラベルを付ける
    promise._debugLabel = `fetchUser(${id})`;
    return promise;
}

// 3. 未処理の Promise 拒否を検出
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection:', reason);
    console.error('Promise:', promise._debugLabel || promise);
    // 本番環境ではプロセスを安全に終了
    process.exit(1);
});

// 4. 非同期タイミングの計測
async function timedFetch(label, asyncFn) {
    const start = performance.now();
    try {
        const result = await asyncFn();
        const elapsed = performance.now() - start;
        console.log(`[${label}] completed in ${elapsed.toFixed(1)}ms`);
        return result;
    } catch (error) {
        const elapsed = performance.now() - start;
        console.error(`[${label}] failed after ${elapsed.toFixed(1)}ms:`, error);
        throw error;
    }
}

// 使用例
const user = await timedFetch('fetchUser', () => fetchUser(1));
```

### 14.2 テスト手法

```javascript
// ===== Jest/Vitest での非同期テスト =====

// 基本: async テスト関数
test('fetchUser returns user data', async () => {
    const user = await fetchUser(1);
    expect(user).toHaveProperty('name');
    expect(user.id).toBe(1);
});

// モック: 外部APIのモック
test('loadDashboard aggregates data', async () => {
    // fetch をモック
    global.fetch = jest.fn()
        .mockResolvedValueOnce({
            ok: true,
            json: async () => ({ id: 1, name: 'Alice' }),
        })
        .mockResolvedValueOnce({
            ok: true,
            json: async () => [{ id: 1, title: 'Post 1' }],
        });

    const dashboard = await loadDashboard(1);
    expect(dashboard.user.name).toBe('Alice');
    expect(dashboard.posts).toHaveLength(1);
});

// タイムアウトテスト: fake timers
test('fetchWithTimeout throws on timeout', async () => {
    jest.useFakeTimers();

    const promise = fetchWithTimeout('/api/slow', 1000);

    // 時間を進める
    jest.advanceTimersByTime(1500);

    await expect(promise).rejects.toThrow('Timeout');

    jest.useRealTimers();
});

// エラーケース
test('processOrder handles payment failure', async () => {
    mockPaymentService.mockRejectedValue(new PaymentError('Declined'));

    const result = await processOrder('order-123');

    expect(result.success).toBe(false);
    expect(result.reason).toBe('payment_failed');
});
```

```python
# ===== pytest-asyncio での非同期テスト（Python） =====
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_fetch_user():
    """非同期関数の基本テスト"""
    user = await fetch_user(1)
    assert user["name"] == "Alice"
    assert user["id"] == 1

@pytest.mark.asyncio
async def test_load_dashboard_concurrent():
    """並行実行のテスト"""
    with patch("module.fetch_user", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"id": 1, "name": "Alice"}

        dashboard = await load_dashboard(1)
        assert dashboard.user["name"] == "Alice"

@pytest.mark.asyncio
async def test_timeout_behavior():
    """タイムアウトのテスト"""
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await asyncio.sleep(1.0)
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 15. まとめ

### 15.1 言語別非同期モデル総合比較

| 言語 | 非同期構文 | ランタイム | 実行モデル | メモリ効率 | 学習曲線 | 適用領域 |
|------|----------|----------|----------|----------|---------|---------|
| JavaScript | async/await | イベントループ(libuv) | シングルスレッド | 中 | 低 | Web全般 |
| Python | async/await | asyncio | シングルスレッド(GIL) | 中 | 中 | I/O密集型 |
| Rust | async/await | tokio/async-std | マルチスレッド | 高(ゼロコスト) | 高 | システム/高性能 |
| Go | goroutine+chan | ランタイム内蔵 | M:Nスケジューリング | 高 | 低 | サーバー/インフラ |
| C# | async/await | TaskScheduler | スレッドプール | 中 | 中 | エンタープライズ |
| Java | Virtual Threads | Loom | M:Nスケジューリング | 高 | 低 | エンタープライズ |

### 15.2 重要原則のチェックリスト

```
非同期プログラミングの原則:

[基本原則]
  □ I/O バウンドな処理に async/await を使う
  □ CPU バウンドな処理にはスレッド/プロセスプールを使う
  □ 独立した非同期処理は並行実行する（Promise.all / gather / join!）
  □ エラーハンドリングを必ず行う（未処理のPromise拒否を放置しない）

[設計原則]
  □ async は呼び出し元に伝染する（async all the way）
  □ 同期コード内から async コードをブロッキング呼び出ししない
  □ セマフォで同時実行数を制限する
  □ タイムアウトを必ず設定する

[運用原則]
  □ サーキットブレーカーで障害の連鎖を防ぐ
  □ リトライは指数バックオフ + ジッタで行う
  □ キャンセル機構を提供する（AbortController / context / CancellationToken）
  □ 非同期処理の監視（タイミング計測、エラーレート）を行う
```

---

## 次に読むべきガイド


---

## 参考文献

1. Hoare, C.A.R. "Communicating Sequential Processes." *Communications of the ACM*, vol. 21, no. 8, 1978, pp. 666-677. CSP モデルの原論文。Go の goroutine + channel の理論的基盤。
2. "Asynchronous Programming in Rust." The Rust Async Book, rust-lang.github.io/async-book/. Rust の非同期プログラミングの公式ガイド。Future トレイトとポーリングモデルの詳細な解説。
3. "Node.js Event Loop, Timers, and process.nextTick()." Node.js Documentation, nodejs.org/en/guides/event-loop-timers-and-nexttick. Node.js のイベントループの公式解説。フェーズごとの動作を詳述。
4. Python Software Foundation. "asyncio -- Asynchronous I/O." Python Documentation, docs.python.org/3/library/asyncio.html. Python asyncio の公式リファレンス。TaskGroup や timeout の使用方法を含む。
5. Cleary, Stephen. "Async in C# 5.0." O'Reilly Media, 2012. C# における async/await のベストプラクティス。SynchronizationContext とデッドロック回避の解説。
6. Go Authors. "Effective Go: Concurrency." go.dev/doc/effective_go#concurrency. Go の並行処理パターンの公式ガイド。goroutine と channel の設計思想。
7. Goetz, Brian et al. "JEP 444: Virtual Threads." OpenJDK, openjdk.org/jeps/444. Java Virtual Threads の仕様。Project Loom による軽量スレッドの設計。
```
