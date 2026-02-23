# 並行・並列プログラミング

> 並行性は「構造」の問題であり、並列性は「実行」の問題である。——Rob Pike

## この章で学ぶこと

- [ ] 並行（Concurrency）と並列（Parallelism）の違いを明確に説明できる
- [ ] デッドロック、競合状態、飢餓の原因と対策を理解する
- [ ] async/awaitの仕組みとイベントループを理解する
- [ ] Go の goroutine/channel（CSPモデル）を理解する
- [ ] アクターモデルの原理と適用場面を理解する
- [ ] ロックフリーアルゴリズムの概念を知る
- [ ] 実務での並行処理のパターンとアンチパターンを身につける

---

## 1. 並行 vs 並列

### 1.1 基本概念の整理

```
並行（Concurrency）: 複数のタスクを「論理的に」同時に管理
  → 1つのCPUコアでも実現可能（タイムスライシング）
  → 構造の問題: 「どう複数の仕事を整理するか」
  → 複数のタスクの実行期間が重なっている状態

並列（Parallelism）: 複数のタスクを「物理的に」同時に実行
  → 複数のCPUコアが必要
  → 実行の問題: 「どう物理的に同時実行するか」
  → 同一時刻に複数のタスクが実行されている状態

  例えで理解:

  並行: 1人のシェフが複数の料理を交互に調理
        パスタを茹でている間にソースを準備、
        ソースを煮込んでいる間にサラダを盛り付け
        → 1人でも効率的に複数タスクをこなせる

  並列: 複数のシェフが同時に別々の料理を調理
        シェフAがパスタ、シェフBがサラダ、シェフCがデザート
        → 物理的な同時実行

  重要: 並行性は並列性を包含する上位概念
  並列は並行の一形態（物理的に同時実行される並行処理）

  ┌─────────────────────────────────┐
  │         並行（Concurrency）       │
  │  ┌───────────────────────┐      │
  │  │   並列（Parallelism）  │      │
  │  └───────────────────────┘      │
  │  非同期I/O、コルーチン等         │
  └─────────────────────────────────┘
```

### 1.2 なぜ並行処理が必要なのか

```
並行処理が必要な理由:

1. I/O待ちの有効活用
   - ネットワーク通信: 数ミリ秒〜数秒
   - ディスクI/O: 数ミリ秒
   - ユーザー入力: 数秒〜数分
   → CPUを遊ばせず、待ち時間に他の処理を実行

2. 応答性の確保
   - GUIアプリケーション: メインスレッドをブロックしない
   - Webサーバー: 1リクエストの処理中に他のリクエストも受付
   - ゲーム: 描画・物理演算・AI・入力を同時管理

3. スループットの向上
   - マルチコアCPUの活用
   - バッチ処理の高速化
   - 大量のリクエストの同時処理

4. リアルタイム性
   - センサーデータの継続的な監視
   - ストリーム処理（ログ集約、イベント処理）
   - リアルタイム通信（チャット、ビデオ通話）

性能の比較例（Webスクレイピング 100ページ）:
┌──────────────────┬──────────────┬──────────────┐
│ 方式              │ 所要時間      │ 備考          │
├──────────────────┼──────────────┼──────────────┤
│ 逐次処理          │ 約100秒      │ 1ページ1秒    │
│ マルチスレッド(10) │ 約10秒       │ 10並行        │
│ async/await       │ 約2-3秒      │ 100並行I/O    │
│ マルチプロセス(10) │ 約10秒       │ CPU限界       │
└──────────────────┴──────────────┴──────────────┘
```

### 1.3 プロセス、スレッド、コルーチンの違い

```
並行実行の単位:

┌──────────────┬───────────────┬──────────────┬──────────────┐
│              │ プロセス       │ スレッド      │ コルーチン    │
├──────────────┼───────────────┼──────────────┼──────────────┤
│ メモリ空間   │ 独立          │ 共有          │ 共有          │
│ 作成コスト   │ 高い          │ 中程度        │ 低い          │
│ コンテキスト │ 重い          │ 中程度        │ 軽い          │
│ スイッチ     │               │               │               │
│ 通信方式     │ IPC           │ 共有メモリ    │ 関数呼び出し  │
│              │ (パイプ等)    │ + ロック      │ + yield       │
│ 並列性       │ ✅ 真の並列   │ ✅ 真の並列*  │ ❌ 並行のみ   │
│ 安全性       │ 高い(分離)    │ 低い(競合)    │ 高い(単一)    │
│ スケーラ     │ 数百程度      │ 数千程度      │ 数百万可能    │
│ ビリティ     │               │               │               │
│ 代表例       │ multiprocessing│ threading    │ asyncio       │
│              │ fork          │ pthread       │ goroutine     │
│ 適する処理   │ CPU-bound     │ 汎用          │ I/O-bound     │
└──────────────┴───────────────┴──────────────┴──────────────┘

* Python の GIL (Global Interpreter Lock) により、
  CPython のスレッドは真の並列CPU実行ができない
```

---

## 2. 並行性の問題と対策

### 2.1 競合状態（Race Condition）

```python
import threading
import time

# === 競合状態のデモ ===

# ❌ 競合状態が発生するコード
counter = 0

def increment_unsafe():
    """安全でないインクリメント"""
    global counter
    for _ in range(100000):
        counter += 1
        # 内部動作:
        # 1. counter の値を読む (read)
        # 2. 値を +1 する (increment)
        # 3. counter に書き戻す (write)
        # → 2つのスレッドが同時に step 1 を実行すると
        #   1回分のインクリメントが失われる！

# 2スレッドで実行
threads = [threading.Thread(target=increment_unsafe) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"期待値: 200000, 実際: {counter}")
# → 実際の値は200000未満になることが多い（例: 183421）


# ✅ 対策1: ロック（Mutex）
counter = 0
lock = threading.Lock()

def increment_with_lock():
    """ロックで保護されたインクリメント"""
    global counter
    for _ in range(100000):
        with lock:  # 排他制御: 1スレッドだけが実行
            counter += 1

threads = [threading.Thread(target=increment_with_lock) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"ロック使用: {counter}")  # 必ず200000


# ✅ 対策2: スレッドセーフなデータ構造
import queue

def producer(q: queue.Queue, items: list):
    """プロデューサー: キューにアイテムを投入"""
    for item in items:
        q.put(item)
        time.sleep(0.01)
    q.put(None)  # 終了シグナル

def consumer(q: queue.Queue, name: str):
    """コンシューマー: キューからアイテムを取得"""
    while True:
        item = q.get()  # ブロッキング（スレッドセーフ）
        if item is None:
            q.put(None)  # 他のconsumerにも終了を伝える
            break
        print(f"[{name}] 処理: {item}")
        q.task_done()

# スレッドセーフなキューで通信
q = queue.Queue(maxsize=10)
prod = threading.Thread(target=producer, args=(q, list(range(20))))
cons1 = threading.Thread(target=consumer, args=(q, "Consumer-1"))
cons2 = threading.Thread(target=consumer, args=(q, "Consumer-2"))

prod.start()
cons1.start()
cons2.start()

prod.join()
cons1.join()
cons2.join()
```

```java
// Java での競合状態対策

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class ConcurrencyDemo {

    // ✅ 対策1: synchronized キーワード
    private int counter = 0;

    public synchronized void incrementSync() {
        counter++;
    }

    // ✅ 対策2: AtomicInteger（CAS操作）
    private AtomicInteger atomicCounter = new AtomicInteger(0);

    public void incrementAtomic() {
        atomicCounter.incrementAndGet();
        // Compare-And-Swap: ロック不要で高速
    }

    // ✅ 対策3: ReentrantLock（明示的ロック）
    private final ReentrantLock lock = new ReentrantLock();
    private int lockedCounter = 0;

    public void incrementWithLock() {
        lock.lock();
        try {
            lockedCounter++;
        } finally {
            lock.unlock();  // 必ず解放
        }
    }

    // ✅ 対策4: concurrent コレクション
    // ConcurrentHashMap, CopyOnWriteArrayList, BlockingQueue 等
    // → スレッドセーフなコレクション実装が標準提供
}
```

### 2.2 デッドロック（Deadlock）

```python
import threading
import time

# === デッドロックのデモ ===

lock_a = threading.Lock()
lock_b = threading.Lock()

# ❌ デッドロックが発生するコード
def thread_1():
    with lock_a:
        print("Thread-1: lock_a 取得")
        time.sleep(0.1)  # lock_b を待つ間に...
        with lock_b:     # Thread-2 が lock_b を持っている！
            print("Thread-1: lock_b 取得")

def thread_2():
    with lock_b:
        print("Thread-2: lock_b 取得")
        time.sleep(0.1)  # lock_a を待つ間に...
        with lock_a:     # Thread-1 が lock_a を持っている！
            print("Thread-2: lock_a 取得")

# → Thread-1 は lock_b を待ち、Thread-2 は lock_a を待つ
# → どちらも永遠に待ち続ける = デッドロック


# ✅ 対策1: ロック取得の順序を統一
def thread_1_fixed():
    with lock_a:       # 常に lock_a → lock_b の順
        print("Thread-1: lock_a 取得")
        with lock_b:
            print("Thread-1: lock_b 取得")

def thread_2_fixed():
    with lock_a:       # 常に lock_a → lock_b の順（統一）
        print("Thread-2: lock_a 取得")
        with lock_b:
            print("Thread-2: lock_b 取得")


# ✅ 対策2: タイムアウト付きロック
def thread_with_timeout():
    acquired_a = lock_a.acquire(timeout=1.0)  # 1秒でタイムアウト
    if not acquired_a:
        print("lock_a 取得タイムアウト、リトライ")
        return False

    try:
        acquired_b = lock_b.acquire(timeout=1.0)
        if not acquired_b:
            print("lock_b 取得タイムアウト、リリースしてリトライ")
            return False
        try:
            # クリティカルセクション
            pass
        finally:
            lock_b.release()
    finally:
        lock_a.release()
    return True


# ✅ 対策3: コンテキストマネージャで安全にロック管理
import contextlib

@contextlib.contextmanager
def acquire_locks(*locks, timeout=5.0):
    """複数のロックを安全に取得するコンテキストマネージャ"""
    acquired = []
    try:
        for lock in sorted(locks, key=id):  # id順で取得順序を統一
            if lock.acquire(timeout=timeout):
                acquired.append(lock)
            else:
                raise TimeoutError(f"ロック取得タイムアウト")
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()

# 使用例
# with acquire_locks(lock_a, lock_b):
#     # 安全にクリティカルセクションを実行
#     pass
```

```
デッドロック発生の4条件（Coffman条件）:
すべてを同時に満たすとデッドロックが発生する

1. 相互排除（Mutual Exclusion）
   リソースは同時に1つのプロセスしか使えない

2. 保持と待ち（Hold and Wait）
   リソースを保持しながら別のリソースを待つ

3. 横取り不可（No Preemption）
   保持中のリソースを強制的に奪えない

4. 循環待ち（Circular Wait）
   プロセスが循環的にリソースを待っている

防止策（いずれか1つを崩せばよい）:
┌──────────┬─────────────────────────────────────┐
│ 条件      │ 防止策                                │
├──────────┼─────────────────────────────────────┤
│ 相互排除  │ ロックフリーアルゴリズム              │
│ 保持と待ち│ 全リソースを一度に取得                │
│ 横取り不可│ タイムアウトで強制解放                │
│ 循環待ち  │ リソースに順序をつけて取得順序を統一  │
└──────────┴─────────────────────────────────────┘
```

### 2.3 その他の並行性問題

```python
# === ライブロック（Livelock）===
# デッドロックと似ているが、スレッドが「動いている」のに進展しない

# 例: 廊下で2人がすれ違えない状況
# AとBが互いに譲ろうとして、同じ方向に動き続ける

# 対策: ランダムなバックオフ
import random

def polite_worker(name: str, resource_a, resource_b):
    while True:
        if resource_a.acquire(timeout=0.1):
            if resource_b.acquire(timeout=0.1):
                # 両方取得成功
                try:
                    print(f"{name}: 処理実行")
                    break
                finally:
                    resource_b.release()
                    resource_a.release()
            else:
                resource_a.release()
                # ランダムなバックオフで再試行
                time.sleep(random.uniform(0.01, 0.1))
        else:
            time.sleep(random.uniform(0.01, 0.1))


# === 飢餓状態（Starvation）===
# 特定のスレッドがいつまでもリソースを取得できない

# 対策: 公平なロック（Fair Lock）
# Java の ReentrantLock(true) は公平性を保証
# Python では queue.PriorityQueue でタスクに優先度を付ける

# === メモリ可視性の問題 ===
# あるスレッドの書き込みが、他のスレッドから見えない

# Java: volatile キーワードで可視性を保証
# Python: GILが暗黙的に保証（ただしCPythonのみ）
# C++: std::atomic, memory_order で明示的に制御
```

### 2.4 並行モデルの比較

```
主要な並行モデル:

┌───────────────┬──────────────────────┬───────────────────┬──────────────┐
│ モデル         │ 特徴                  │ メリット           │ 言語/FW      │
├───────────────┼──────────────────────┼───────────────────┼──────────────┤
│ スレッド       │ OSスレッドを共有      │ 馴染みやすい       │ Java, C++    │
│ +ロック        │ メモリで通信          │ 低レベル制御可能   │ Python       │
│               │                       │                    │              │
│ アクター       │ 独立したエンティティ  │ 分散に強い         │ Erlang/OTP   │
│ モデル         │ メッセージパッシング  │ 耐障害性           │ Akka(Scala)  │
│               │ 共有状態なし          │ スケーラブル       │              │
│               │                       │                    │              │
│ CSP           │ チャネルを通じた通信  │ シンプルで安全     │ Go           │
│               │ goroutine(軽量)       │ 高性能             │ Clojure      │
│               │                       │ 推論しやすい       │              │
│               │                       │                    │              │
│ async/await   │ 協調的マルチタスク    │ I/Oに最適          │ JavaScript   │
│               │ イベントループ        │ ロック不要          │ Python       │
│               │ シングルスレッド      │ 実装が容易          │ Rust, C#     │
│               │                       │                    │              │
│ STM           │ トランザクション      │ 合成可能            │ Haskell      │
│               │ メモリの楽観的制御    │ デッドロックなし    │ Clojure      │
│               │                       │ 推論が容易          │              │
│               │                       │                    │              │
│ データ並列    │ SIMD/GPUベース       │ 大量データに強い    │ CUDA         │
│               │ 同一操作を並列適用    │ 数値計算に最適      │ OpenCL       │
│               │                       │                    │ numpy        │
└───────────────┴──────────────────────┴───────────────────┴──────────────┘
```

---

## 3. async/await（非同期プログラミング）

### 3.1 イベントループの仕組み

```python
# === async/await の仕組み ===

# イベントループの概念図:
#
# ┌──────────────────────────────────────────┐
# │            イベントループ                  │
# │                                            │
# │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐    │
# │  │Task1│  │Task2│  │Task3│  │Task4│    │
# │  │await│  │ready│  │await│  │ready│    │
# │  └─────┘  └─────┘  └─────┘  └─────┘    │
# │                                            │
# │  1. ready なタスクを1つ取り出す            │
# │  2. await に達するまで実行                 │
# │  3. await で I/O 開始、タスクを待機に      │
# │  4. I/O 完了したタスクを ready に          │
# │  5. 1 に戻る                               │
# └──────────────────────────────────────────┘

import asyncio
import time


# 基本的な async/await
async def fetch_data(name: str, delay: float) -> str:
    """I/O を模擬する非同期関数"""
    print(f"  [{name}] 開始 (待機: {delay}秒)")
    await asyncio.sleep(delay)  # I/O待ち（他のタスクに制御を渡す）
    print(f"  [{name}] 完了")
    return f"{name}_data"


async def main():
    start = time.perf_counter()

    # ❌ 逐次実行: 合計3秒
    # result1 = await fetch_data("API-1", 1.0)
    # result2 = await fetch_data("API-2", 1.0)
    # result3 = await fetch_data("API-3", 1.0)

    # ✅ 並行実行: 合計約1秒（最も遅いタスクの時間）
    results = await asyncio.gather(
        fetch_data("API-1", 1.0),
        fetch_data("API-2", 0.5),
        fetch_data("API-3", 0.8),
    )

    elapsed = time.perf_counter() - start
    print(f"  結果: {results}")
    print(f"  所要時間: {elapsed:.2f}秒")  # 約1.0秒

asyncio.run(main())
```

### 3.2 実務的な非同期パターン

```python
import asyncio
import aiohttp
from typing import Any
from dataclasses import dataclass


# === 並行HTTP リクエスト ===

async def fetch_url(session: aiohttp.ClientSession, url: str) -> dict:
    """1つのURLからデータを取得"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            return {"url": url, "status": resp.status, "data": data}
    except Exception as e:
        return {"url": url, "status": "error", "error": str(e)}


async def fetch_all(urls: list[str], max_concurrent: int = 10) -> list[dict]:
    """複数URLを並行取得（同時接続数制限付き）"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(session, url):
        async with semaphore:  # 同時接続数を制限
            return await fetch_url(session, url)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_limit(session, url) for url in urls]
        return await asyncio.gather(*tasks)


# === タイムアウト付き処理 ===

async def with_timeout(coro, timeout_seconds: float, default=None):
    """タイムアウト付きでコルーチンを実行"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"タイムアウト ({timeout_seconds}秒)")
        return default


# === リトライ付き非同期処理 ===

async def retry_async(
    coro_func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """指数バックオフ付きリトライ"""
    current_delay = delay
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_func()
        except exceptions as e:
            if attempt == max_retries:
                raise
            print(f"リトライ {attempt}/{max_retries}: {e}")
            await asyncio.sleep(current_delay)
            current_delay *= backoff


# === 非同期ジェネレータ（ストリーム処理）===

async def event_stream(interval: float = 1.0):
    """イベントを非同期的に生成するジェネレータ"""
    event_id = 0
    while True:
        await asyncio.sleep(interval)
        event_id += 1
        yield {"id": event_id, "timestamp": time.time(), "data": f"event_{event_id}"}


async def process_events():
    """イベントストリームを処理"""
    async for event in event_stream(0.5):
        print(f"イベント受信: {event}")
        if event["id"] >= 5:
            break


# === Producer-Consumer パターン（非同期版）===

async def async_producer(queue: asyncio.Queue, items: list):
    """非同期プロデューサー"""
    for item in items:
        await asyncio.sleep(0.1)  # 生成に時間がかかる
        await queue.put(item)
        print(f"[Producer] 投入: {item}")
    await queue.put(None)  # 終了シグナル

async def async_consumer(queue: asyncio.Queue, name: str):
    """非同期コンシューマー"""
    while True:
        item = await queue.get()
        if item is None:
            await queue.put(None)  # 他のconsumerにも伝播
            break
        await asyncio.sleep(0.2)  # 処理に時間がかかる
        print(f"[{name}] 処理完了: {item}")
        queue.task_done()

async def producer_consumer_demo():
    """Producer-Consumer パターンのデモ"""
    queue = asyncio.Queue(maxsize=5)

    # 1 producer + 3 consumers を並行実行
    await asyncio.gather(
        async_producer(queue, list(range(10))),
        async_consumer(queue, "Consumer-A"),
        async_consumer(queue, "Consumer-B"),
        async_consumer(queue, "Consumer-C"),
    )
```

```javascript
// JavaScript の async/await

// Promise ベースの非同期処理
async function fetchUserWithPosts(userId) {
    // 並行でユーザー情報と投稿を取得
    const [user, posts] = await Promise.all([
        fetch(`/api/users/${userId}`).then(r => r.json()),
        fetch(`/api/users/${userId}/posts`).then(r => r.json()),
    ]);

    return { ...user, posts };
}

// エラーハンドリング
async function safelyFetch(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return { ok: true, data: await response.json() };
    } catch (error) {
        return { ok: false, error: error.message };
    }
}

// Promise.allSettled: 全てのPromiseが完了するまで待つ（失敗しても）
async function fetchMultiple(urls) {
    const results = await Promise.allSettled(
        urls.map(url => fetch(url).then(r => r.json()))
    );

    return results.map((result, i) => ({
        url: urls[i],
        status: result.status,
        data: result.status === 'fulfilled' ? result.value : null,
        error: result.status === 'rejected' ? result.reason.message : null,
    }));
}

// レートリミッター
class AsyncRateLimiter {
    constructor(maxConcurrent) {
        this.maxConcurrent = maxConcurrent;
        this.running = 0;
        this.queue = [];
    }

    async execute(fn) {
        if (this.running >= this.maxConcurrent) {
            await new Promise(resolve => this.queue.push(resolve));
        }

        this.running++;
        try {
            return await fn();
        } finally {
            this.running--;
            if (this.queue.length > 0) {
                this.queue.shift()();
            }
        }
    }
}

// 使用例: 最大5同時リクエスト
const limiter = new AsyncRateLimiter(5);
const results = await Promise.all(
    urls.map(url => limiter.execute(() => fetch(url)))
);
```

### 3.3 Python の GIL と multiprocessing

```python
# === Python の GIL (Global Interpreter Lock) ===

# GILとは:
# CPython がメモリ管理の安全性のために導入した排他ロック
# → 同一プロセス内で同時に1スレッドしかPythonバイトコードを実行できない
# → マルチスレッドでもCPU-bound処理は並列化されない

# I/O-bound → スレッドでOK（I/O待ちの間にGILが解放される）
# CPU-bound → multiprocessing を使う

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def cpu_heavy_task(n: int) -> int:
    """CPU集約的な処理"""
    total = 0
    for i in range(n):
        total += i * i
    return total


# ❌ マルチスレッド: CPU-bound では GIL のせいで速くならない
def with_threads(tasks: list[int]) -> list[int]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(cpu_heavy_task, tasks))

# ✅ マルチプロセス: 真の並列実行
def with_processes(tasks: list[int]) -> list[int]:
    with ProcessPoolExecutor(max_workers=4) as executor:
        return list(executor.map(cpu_heavy_task, tasks))


# ベンチマーク
tasks = [10_000_000] * 4

start = time.perf_counter()
with_threads(tasks)
thread_time = time.perf_counter() - start

start = time.perf_counter()
with_processes(tasks)
process_time = time.perf_counter() - start

print(f"スレッド: {thread_time:.2f}秒")    # 例: 8.5秒（GILで遅い）
print(f"プロセス: {process_time:.2f}秒")   # 例: 2.5秒（4コアで並列）


# === concurrent.futures: 統一的なインターフェース ===

from concurrent.futures import as_completed

def process_batch(items: list, worker_func, max_workers: int = 4):
    """バッチ処理を並列実行して結果を収集"""
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 各タスクを投入
        future_to_item = {
            executor.submit(worker_func, item): item
            for item in items
        }

        # 完了順に結果を取得
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result(timeout=30)
                results[item] = {"status": "success", "result": result}
            except Exception as e:
                results[item] = {"status": "error", "error": str(e)}

    return results
```

---

## 4. Go の goroutine/channel（CSPモデル）

### 4.1 goroutine の基本

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// goroutine: 軽量スレッド（数KB、OSスレッドは数MB）
// → 数十万の goroutine を同時実行可能

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d: 開始\n", id)
    time.Sleep(time.Second)
    fmt.Printf("Worker %d: 完了\n", id)
}

func main() {
    var wg sync.WaitGroup

    // 10個の goroutine を起動
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go worker(i, &wg)  // go キーワードで goroutine 起動
    }

    wg.Wait()  // 全 goroutine の完了を待つ
    fmt.Println("全ワーカー完了")
}
```

### 4.2 チャネルによる通信

```go
package main

import (
    "fmt"
    "time"
)

// "Don't communicate by sharing memory,
//  share memory by communicating." — Go Proverb

// === チャネルの基本 ===

// Producer-Consumer パターン
func producer(ch chan<- int) {
    for i := 0; i < 10; i++ {
        ch <- i  // チャネルに送信
        fmt.Printf("[Producer] 送信: %d\n", i)
        time.Sleep(100 * time.Millisecond)
    }
    close(ch)  // チャネルを閉じる（これ以上送信しない）
}

func consumer(ch <-chan int, name string) {
    for val := range ch {  // チャネルが閉じるまで受信
        fmt.Printf("[%s] 受信: %d\n", name, val)
        time.Sleep(200 * time.Millisecond)
    }
}


// === Fan-out / Fan-in パターン ===

func fanOut(input <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        ch := make(chan int)
        channels[i] = ch
        go func(out chan<- int, workerID int) {
            for val := range input {
                // 重い処理をシミュレート
                result := val * val
                fmt.Printf("[Worker-%d] %d → %d\n", workerID, val, result)
                out <- result
            }
            close(out)
        }(ch, i)
    }
    return channels
}

func fanIn(channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for val := range c {
                merged <- val
            }
        }(ch)
    }

    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}


// === select 文: 複数チャネルの待機 ===

func selectDemo() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    timeout := time.After(3 * time.Second)

    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "ch1のデータ"
    }()

    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "ch2のデータ"
    }()

    for i := 0; i < 2; i++ {
        select {
        case msg := <-ch1:
            fmt.Printf("ch1 受信: %s\n", msg)
        case msg := <-ch2:
            fmt.Printf("ch2 受信: %s\n", msg)
        case <-timeout:
            fmt.Println("タイムアウト！")
            return
        }
    }
}


// === Pipeline パターン ===

func generate(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

func filter(in <-chan int, predicate func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            if predicate(n) {
                out <- n
            }
        }
        close(out)
    }()
    return out
}

func pipelineMain() {
    // パイプライン: 生成 → 二乗 → フィルタ
    numbers := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    squared := square(numbers)
    filtered := filter(squared, func(n int) bool { return n > 20 })

    for result := range filtered {
        fmt.Println(result)  // 25, 36, 49, 64, 81, 100
    }
}


// === Context によるキャンセレーション ===

import "context"

func longRunningTask(ctx context.Context, id int) error {
    for i := 0; ; i++ {
        select {
        case <-ctx.Done():
            // キャンセルされた
            fmt.Printf("Task %d: キャンセル (理由: %v)\n", id, ctx.Err())
            return ctx.Err()
        default:
            // 処理を続行
            fmt.Printf("Task %d: ステップ %d\n", id, i)
            time.Sleep(500 * time.Millisecond)
        }
    }
}

func contextDemo() {
    ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
    defer cancel()

    go longRunningTask(ctx, 1)
    go longRunningTask(ctx, 2)

    time.Sleep(3 * time.Second)
    // → 2秒後に自動キャンセル
}
```

---

## 5. アクターモデル

### 5.1 アクターモデルの原理

```
アクターモデル（Carl Hewitt, 1973）:

  基本概念:
  - すべてがアクター（独立した計算単位）
  - アクターは以下の3つだけを行う:
    1. メッセージの受信と処理
    2. 新しいアクターの生成
    3. 他のアクターへのメッセージ送信
  - 共有状態は一切なし
  - メッセージは非同期に送受信

  メッセージボックスモデル:
  ┌─────────────────────────────────────┐
  │          Actor A                     │
  │  ┌──────────┐  ┌─────────────┐     │
  │  │ Mailbox  │→ │ 振る舞い     │     │
  │  │ msg1     │  │ (状態を持つ) │     │
  │  │ msg2     │  │             │     │
  │  │ msg3     │  │ → msg処理   │──→ Actor B に送信
  │  └──────────┘  └─────────────┘     │
  └─────────────────────────────────────┘

  メリット:
  - 共有状態がないためロック不要
  - 分散システムとの親和性が高い
  - 耐障害性（Let it crash 哲学）
  - スケーラビリティ
```

### 5.2 Erlang/Elixir のアクター実装

```elixir
# Elixir のアクターモデル（GenServer）

defmodule BankAccount do
  use GenServer

  # クライアントAPI
  def start_link(initial_balance) do
    GenServer.start_link(__MODULE__, initial_balance)
  end

  def deposit(pid, amount) do
    GenServer.call(pid, {:deposit, amount})
  end

  def withdraw(pid, amount) do
    GenServer.call(pid, {:withdraw, amount})
  end

  def balance(pid) do
    GenServer.call(pid, :balance)
  end

  # サーバーコールバック
  @impl true
  def init(initial_balance) do
    {:ok, %{balance: initial_balance, transactions: []}}
  end

  @impl true
  def handle_call({:deposit, amount}, _from, state) when amount > 0 do
    new_state = %{
      state |
      balance: state.balance + amount,
      transactions: [{:deposit, amount} | state.transactions]
    }
    {:reply, {:ok, new_state.balance}, new_state}
  end

  @impl true
  def handle_call({:withdraw, amount}, _from, state) when amount > 0 do
    if state.balance >= amount do
      new_state = %{
        state |
        balance: state.balance - amount,
        transactions: [{:withdraw, amount} | state.transactions]
      }
      {:reply, {:ok, new_state.balance}, new_state}
    else
      {:reply, {:error, :insufficient_funds}, state}
    end
  end

  @impl true
  def handle_call(:balance, _from, state) do
    {:reply, state.balance, state}
  end
end

# 使用例
{:ok, account} = BankAccount.start_link(10000)
BankAccount.deposit(account, 5000)    # {:ok, 15000}
BankAccount.withdraw(account, 3000)   # {:ok, 12000}
BankAccount.balance(account)          # 12000

# 並行アクセスしても安全（メッセージは順番に処理される）
```

### 5.3 Python でのアクターモデル風実装

```python
import asyncio
from typing import Any, Callable
from dataclasses import dataclass, field


class Actor:
    """シンプルなアクター実装"""

    def __init__(self, name: str):
        self.name = name
        self._mailbox: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """アクターを起動"""
        self._running = True
        while self._running:
            message = await self._mailbox.get()
            if message is None:  # 停止シグナル
                self._running = False
                break
            await self.handle_message(message)

    async def send(self, message: Any):
        """メッセージを送信"""
        await self._mailbox.put(message)

    async def stop(self):
        """アクターを停止"""
        await self._mailbox.put(None)

    async def handle_message(self, message: Any):
        """メッセージハンドラ（サブクラスでオーバーライド）"""
        raise NotImplementedError


@dataclass
class Transfer:
    from_account: str
    to_account: str
    amount: int
    reply_to: asyncio.Queue


class BankAccountActor(Actor):
    """銀行口座アクター"""

    def __init__(self, name: str, initial_balance: int = 0):
        super().__init__(name)
        self._balance = initial_balance

    async def handle_message(self, message: dict):
        match message:
            case {"action": "deposit", "amount": amount, "reply": reply}:
                self._balance += amount
                await reply.put({"status": "ok", "balance": self._balance})

            case {"action": "withdraw", "amount": amount, "reply": reply}:
                if self._balance >= amount:
                    self._balance -= amount
                    await reply.put({"status": "ok", "balance": self._balance})
                else:
                    await reply.put({"status": "error", "reason": "残高不足"})

            case {"action": "balance", "reply": reply}:
                await reply.put({"balance": self._balance})


async def actor_demo():
    """アクターモデルのデモ"""
    # アクターを作成して起動
    account = BankAccountActor("account-1", 10000)
    task = asyncio.create_task(account.start())

    # メッセージ送信と返信受信
    reply = asyncio.Queue()
    await account.send({"action": "deposit", "amount": 5000, "reply": reply})
    result = await reply.get()
    print(f"入金結果: {result}")  # {"status": "ok", "balance": 15000}

    await account.send({"action": "balance", "reply": reply})
    result = await reply.get()
    print(f"残高: {result}")  # {"balance": 15000}

    await account.stop()
    await task
```

---

## 6. ロックフリープログラミング

```
ロックフリーアルゴリズム:

  ロック（Mutex）の問題:
  - デッドロックのリスク
  - 優先度逆転
  - コンテキストスイッチのオーバーヘッド
  - スケーラビリティの限界

  ロックフリーの手法:
  1. CAS (Compare-And-Swap) 操作
     - アトミックに「比較して一致すれば書き換え」
     - ハードウェアレベルでサポート

  2. Immutable Data Structures
     - 変更不可能なデータ → ロック不要
     - 永続データ構造（Persistent Data Structures）

  3. Lock-Free Queue / Stack
     - CAS ベースの実装
     - 高スループット

  CAS の動作:
  ┌─────────────────────────────────────┐
  │ CAS(メモリ位置, 期待値, 新しい値)    │
  │                                       │
  │ if メモリ位置の値 == 期待値:          │
  │     メモリ位置の値 = 新しい値         │
  │     return true  // 成功              │
  │ else:                                 │
  │     return false // 他のスレッドが     │
  │                  // 変更済み → リトライ│
  └─────────────────────────────────────┘
```

```python
# Python での Atomic 操作（簡易的な実装）
import threading

class AtomicInteger:
    """アトミックな整数（CAS風の実装）"""

    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()  # Python では真の CAS がないためロックで代用

    @property
    def value(self) -> int:
        return self._value

    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """CAS: 現在の値が expected なら new_value に更新"""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

    def increment(self) -> int:
        """アトミックなインクリメント"""
        while True:
            current = self._value
            if self.compare_and_swap(current, current + 1):
                return current + 1

    def decrement(self) -> int:
        """アトミックなデクリメント"""
        while True:
            current = self._value
            if self.compare_and_swap(current, current - 1):
                return current - 1

    def add_and_get(self, delta: int) -> int:
        """アトミックな加算"""
        while True:
            current = self._value
            if self.compare_and_swap(current, current + delta):
                return current + delta
```

---

## 7. 実務での並行処理パターン

### 7.1 よく使われるパターン

```python
# === Worker Pool パターン ===

import asyncio
from typing import Callable, Any


async def worker_pool(
    tasks: list[Any],
    worker_func: Callable,
    max_workers: int = 10,
    progress_callback: Callable = None
) -> list[Any]:
    """ワーカープールで複数タスクを並行処理"""
    semaphore = asyncio.Semaphore(max_workers)
    results = [None] * len(tasks)
    completed = 0

    async def process(index: int, task: Any):
        nonlocal completed
        async with semaphore:
            result = await worker_func(task)
            results[index] = result
            completed += 1
            if progress_callback:
                progress_callback(completed, len(tasks))

    await asyncio.gather(*(
        process(i, task) for i, task in enumerate(tasks)
    ))
    return results


# === Circuit Breaker パターン ===

import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"        # 正常
    OPEN = "open"            # 遮断
    HALF_OPEN = "half_open"  # 半開

class CircuitBreaker:
    """サーキットブレーカー: 連続失敗時にリクエストを遮断"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            # タイムアウト経過で HALF_OPEN に移行
            if time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    async def call(self, func: Callable, *args, **kwargs):
        """サーキットブレーカーを通してリクエストを実行"""
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise RuntimeError("サーキットブレーカーが開いています")

        try:
            result = await func(*args, **kwargs)

            if current_state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0

            return result

        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                print(f"サーキットブレーカー: OPEN (失敗{self._failure_count}回)")

            raise


# === Bulkhead パターン（隔壁）===

class Bulkhead:
    """バルクヘッド: サービスごとにリソースを分離"""

    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    async def execute(self, func: Callable, *args, **kwargs):
        if self._semaphore.locked():
            raise RuntimeError(
                f"バルクヘッド '{self.name}' の容量超過 "
                f"(アクティブ: {self._active})"
            )

        async with self._semaphore:
            self._active += 1
            try:
                return await func(*args, **kwargs)
            finally:
                self._active -= 1


# サービスごとにバルクヘッドを設定
api_bulkhead = Bulkhead("外部API", max_concurrent=10)
db_bulkhead = Bulkhead("データベース", max_concurrent=20)

# APIサービスが過負荷でもDBサービスに影響しない
# await api_bulkhead.execute(call_external_api, url)
# await db_bulkhead.execute(query_database, sql)
```

### 7.2 並行処理のアンチパターン

```
並行処理のアンチパターン:

1. ❌ 必要以上の並行度
   - スレッドをリクエストごとに作成（C10K問題）
   - → イベントループやスレッドプールを使う

2. ❌ ロックの粒度が大きすぎる
   - 処理全体をロックする（コンカレンシーが下がる）
   - → ロックの範囲を最小限にする

3. ❌ ロックの粒度が小さすぎる
   - 細かすぎるロック（オーバーヘッド増大）
   - → 適切な粒度を見極める

4. ❌ ネストしたロック
   - デッドロックの温床
   - → ロックの順序を統一、またはロック不要な設計

5. ❌ busy waiting（スピンロック）
   - CPU使用率100%で待機
   - → 条件変数やセマフォで待機

6. ❌ fire and forget
   - 非同期タスクのエラーを無視
   - → 必ずエラーハンドリングと完了確認

7. ❌ 共有ミュータブル状態
   - グローバル変数を複数スレッドから変更
   - → 不変データ or メッセージパッシング

8. ❌ スレッドセーフでないライブラリの使用
   - 共有状態を持つライブラリをマルチスレッドで使用
   - → ドキュメントでスレッドセーフ性を確認
```

---

## 8. Rust の並行処理

```rust
// Rust: 所有権システムによるコンパイル時の安全性保証

use std::thread;
use std::sync::{Arc, Mutex, mpsc};

// === スレッド + メッセージパッシング ===

fn channel_example() {
    let (tx, rx) = mpsc::channel();

    // 送信側スレッド
    thread::spawn(move || {
        let messages = vec!["こんにちは", "from", "スレッド"];
        for msg in messages {
            tx.send(msg).unwrap();
            thread::sleep(Duration::from_millis(100));
        }
    });

    // 受信側（メインスレッド）
    for received in rx {
        println!("受信: {}", received);
    }
}

// === 共有状態（Arc + Mutex）===

fn shared_state_example() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("結果: {}", *counter.lock().unwrap());  // 10
}

// Rust のコンパイラが防ぐ問題:
// - データ競合: &mut T の同時アクセスをコンパイル時に禁止
// - ダングリングポインタ: 所有権システムで防止
// - Send/Sync トレイト: スレッド安全でない型の送信を禁止

// === async/await（tokio ランタイム）===

use tokio;

#[tokio::main]
async fn main() {
    let handle1 = tokio::spawn(async {
        // 非同期タスク1
        tokio::time::sleep(Duration::from_secs(1)).await;
        "結果1"
    });

    let handle2 = tokio::spawn(async {
        // 非同期タスク2
        tokio::time::sleep(Duration::from_secs(2)).await;
        "結果2"
    });

    // 並行実行
    let (result1, result2) = tokio::join!(handle1, handle2);
    println!("{:?}, {:?}", result1, result2);
}
```

---

## 9. 実務での並行処理設計指針

```
並行処理の設計指針:

1. まず並行性が必要か検討する
   - 逐次処理で十分なら無理に並行化しない
   - 複雑さが増すコスト vs 性能改善の利益

2. 最も単純なモデルを選ぶ
   I/O-bound → async/await
   CPU-bound → マルチプロセス（Python）/ マルチスレッド
   分散処理 → メッセージキュー（RabbitMQ, Kafka等）
   リアルタイム → アクターモデル or CSP

3. 共有ミュータブル状態を最小化する
   - 不変データ構造を優先
   - 必要な場合のみロックを使用
   - ロック範囲は最小限に

4. エラーハンドリングを設計する
   - タイムアウトを必ず設定
   - リトライ戦略を定義
   - サーキットブレーカーの導入を検討
   - 障害の伝播を制限（バルクヘッド）

5. テスト戦略
   - 競合状態のテストは困難 → 設計で防ぐ
   - 純粋関数を多用してテスト容易性を確保
   - stress test、chaos engineering の導入

6. 監視とデバッグ
   - 構造化ログにスレッド/タスクIDを含める
   - メトリクス: 並行度、レイテンシ、エラー率
   - デッドロック検出ツールの活用

技術選定の早見表:
┌───────────────────────┬───────────────────────────┐
│ 要件                   │ 推奨技術                   │
├───────────────────────┼───────────────────────────┤
│ Web API の並行リクエスト │ async/await                │
│ 画像/動画のバッチ処理   │ マルチプロセス             │
│ WebSocket サーバー     │ async/await + イベント      │
│ 分散タスクキュー       │ Celery / Kafka / RabbitMQ  │
│ リアルタイムゲーム     │ アクターモデル / ECS       │
│ データパイプライン     │ Apache Spark / Flink       │
│ マイクロサービス間通信 │ gRPC / メッセージキュー    │
└───────────────────────┴───────────────────────────┘
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 並行 vs 並列 | 並行=構造（1コアでも可能）、並列=実行（複数コア必要） |
| 競合状態 | 共有状態 + 非アトミック操作。ロック/CAS/不変性で防止 |
| デッドロック | 循環する待ち合わせ。Coffman条件の1つを崩して防止 |
| async/await | シングルスレッド非同期。I/O-bound に最適 |
| goroutine/channel | CSPモデル。軽量スレッド + チャネル通信 |
| アクターモデル | メッセージパッシング。共有状態なし。分散に強い |
| ロックフリー | CAS操作でロック不要。高スループット |
| GIL (Python) | CPU-bound はマルチプロセスを使う |
| パターン | Worker Pool, Circuit Breaker, Bulkhead |
| 設計指針 | 共有状態最小化、タイムアウト必須、最も単純なモデルを選ぶ |

---

## 次に読むべきガイド
→ [[../07-software-engineering/00-development-process.md]] — ソフトウェア開発プロセス

---

## 参考文献
1. Pike, R. "Concurrency Is Not Parallelism." Waza Conference, 2012.
2. Goetz, B. "Java Concurrency in Practice." Addison-Wesley, 2006.
3. Armstrong, J. "Programming Erlang: Software for a Concurrent World." 2nd Edition, Pragmatic Bookshelf, 2013.
4. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, 2019.
6. Hewitt, C. "A Universal Modular ACTOR Formalism for Artificial Intelligence." 1973.
7. Hoare, C.A.R. "Communicating Sequential Processes." Prentice Hall, 1985.
8. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." Morgan Kaufmann, 2012.
9. Nystrom, R. "Game Programming Patterns." Genever Benning, 2014.
10. Butcher, P. "Seven Concurrency Models in Seven Weeks." Pragmatic Bookshelf, 2014.
