# キャンセル処理

> 非同期処理のキャンセルは見落とされがちだが、UXとリソース管理に直結する重要な技術。AbortController、タイムアウト、キャンセルトークンの実装を解説。

## この章で学ぶこと

- [ ] 非同期処理のキャンセルが必要な場面を理解する
- [ ] AbortController の使い方を把握する
- [ ] タイムアウトパターンの実装を学ぶ
- [ ] 各言語のキャンセルメカニズムを比較する
- [ ] プロダクションレベルのキャンセル設計を習得する
- [ ] テスト可能なキャンセル処理の実装を学ぶ


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [イベントループ](./00-event-loop.md) の内容を理解していること

---

## 1. なぜキャンセルが必要か

```
キャンセルが必要な場面:
  1. ユーザーがページ遷移した → 前のページのAPIリクエストを中止
  2. 検索ボックスの入力 → 前の検索リクエストを中止
  3. タイムアウト → 一定時間内に応答がない場合に中止
  4. コンポーネントのアンマウント → 進行中の処理を中止
  5. ユーザーが「キャンセル」ボタンを押した
  6. サーバーシャットダウン時の進行中処理のクリーンアップ
  7. リソース制限に達した場合の処理中止
  8. 競合する処理の一方を中止（レースパターン）

キャンセルしないと:
  → 不要なネットワークリクエストが残る
  → メモリリーク（コンポーネント破棄後にsetState）
  → レースコンディション（古い結果が新しい結果を上書き）
  → サーバーリソースの無駄遣い
  → ユーザー体験の悪化（古いデータの表示）
  → DBコネクションの枯渇
```

### 1.1 キャンセルの分類

```
キャンセルの種類:

1. ユーザー起因のキャンセル
   → キャンセルボタン押下
   → ページ遷移
   → コンポーネントのアンマウント
   → ブラウザのタブを閉じる

2. システム起因のキャンセル
   → タイムアウト
   → サーバーシャットダウン
   → リソース制限到達
   → 親タスクのキャンセル伝搬

3. ロジック起因のキャンセル
   → 新しいリクエストによる前のリクエストの中止
   → 最初の結果が得られた時点で他を中止
   → 条件が変わった場合の処理中止

キャンセルのレベル:
  ┌────────────────────────────────────────┐
  │ アプリケーションレベル                   │
  │  → UIイベント、ルーティング             │
  │                                        │
  │  ┌────────────────────────────────────┐ │
  │  │ サービスレベル                      │ │
  │  │  → API呼び出し、バッチ処理          │ │
  │  │                                    │ │
  │  │  ┌────────────────────────────────┐ │ │
  │  │  │ リソースレベル                  │ │ │
  │  │  │  → DB接続、ファイルハンドル     │ │ │
  │  │  └────────────────────────────────┘ │ │
  │  └────────────────────────────────────┘ │
  └────────────────────────────────────────┘
```

---

## 2. AbortController（Web標準）

### 2.1 基本的な使い方

```typescript
// fetch のキャンセル
const controller = new AbortController();
const { signal } = controller;

// リクエスト開始
const promise = fetch('/api/data', { signal })
  .then(res => res.json())
  .catch(err => {
    if (err.name === 'AbortError') {
      console.log('リクエストがキャンセルされました');
    } else {
      throw err;
    }
  });

// 3秒後にキャンセル
setTimeout(() => controller.abort(), 3000);

// AbortSignal.reason でキャンセル理由を指定（2022年以降の仕様）
controller.abort(new Error('User navigated away'));
controller.abort('timeout'); // 文字列も可

// signal.reason でキャンセル理由を取得
signal.addEventListener('abort', () => {
  console.log('Abort reason:', signal.reason);
});
```

### 2.2 React での使用パターン

```typescript
// React での使用
function SearchResults({ query }: { query: string }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetch(`/api/search?q=${query}`, { signal: controller.signal })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        setResults(data);
        setLoading(false);
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          setError(err.message);
          setLoading(false);
        }
        // AbortError は無視（コンポーネント破棄 or query変更時）
      });

    // クリーンアップ: コンポーネント破棄時 or query変更時にキャンセル
    return () => controller.abort();
  }, [query]);

  return (
    <div>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      <ul>{results.map(r => <li key={r.id}>{r.name}</li>)}</ul>
    </div>
  );
}
```

### 2.3 AbortSignal の高度な使い方

```typescript
// AbortSignal.timeout(): タイムアウト付き signal
const response = await fetch('/api/data', {
  signal: AbortSignal.timeout(5000), // 5秒でタイムアウト
});

// AbortSignal.any(): 複数の signal を結合（2023年以降）
const userCancel = new AbortController();
const timeoutSignal = AbortSignal.timeout(30000);
const shutdownSignal = getShutdownSignal();

const combinedSignal = AbortSignal.any([
  userCancel.signal,
  timeoutSignal,
  shutdownSignal,
]);

fetch('/api/data', { signal: combinedSignal });
// → ユーザーキャンセル、タイムアウト、シャットダウンの いずれかで中止

// AbortSignal をカスタムAPIに渡す
class DataFetcher {
  async fetchWithRetry(
    url: string,
    options: { signal?: AbortSignal; maxRetries?: number } = {},
  ): Promise<Response> {
    const { signal, maxRetries = 3 } = options;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      // キャンセルチェック
      signal?.throwIfAborted();

      try {
        const response = await fetch(url, { signal });
        if (response.ok) return response;

        if (response.status >= 500 && attempt < maxRetries) {
          // サーバーエラーはリトライ
          await this.delay(Math.pow(2, attempt) * 1000, signal);
          continue;
        }

        throw new Error(`HTTP ${response.status}`);
      } catch (err) {
        if ((err as Error).name === 'AbortError') throw err;
        if (attempt === maxRetries) throw err;
      }
    }

    throw new Error('Max retries exceeded');
  }

  private delay(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(signal.reason);
        return;
      }

      const timer = setTimeout(resolve, ms);

      signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(signal.reason);
      }, { once: true });
    });
  }
}
```

### 2.4 Node.js での AbortController

```typescript
import { readFile, writeFile } from 'fs/promises';
import { createReadStream } from 'fs';
import { pipeline } from 'stream/promises';
import { setTimeout as sleep } from 'timers/promises';

// fs/promises のキャンセル
const controller = new AbortController();
const { signal } = controller;

// ファイル読み込みのキャンセル
try {
  const data = await readFile('large-file.txt', { signal });
} catch (err) {
  if ((err as NodeJS.ErrnoException).code === 'ABORT_ERR') {
    console.log('File read cancelled');
  }
}

// timers/promises のキャンセル
try {
  await sleep(60000, null, { signal }); // 60秒待ち
} catch (err) {
  // signal が abort されたら即座に解決
}

// Stream のキャンセル
const readStream = createReadStream('data.csv', { signal });
const writeStream = createWriteStream('output.csv', { signal });

try {
  await pipeline(readStream, transformStream, writeStream, { signal });
} catch (err) {
  if ((err as Error).name === 'AbortError') {
    console.log('Pipeline cancelled');
  }
}

// EventEmitter のキャンセル
import { once } from 'events';

const controller2 = new AbortController();
try {
  const [data] = await once(emitter, 'data', { signal: controller2.signal });
} catch (err) {
  // signal が abort されたら待機をキャンセル
}

// HTTP サーバーでのリクエストキャンセル検出
import http from 'http';

const server = http.createServer(async (req, res) => {
  const controller = new AbortController();

  // クライアントが接続を切断した場合にキャンセル
  req.on('close', () => {
    if (!res.writableEnded) {
      controller.abort();
    }
  });

  try {
    const data = await fetchExpensiveData({ signal: controller.signal });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      // クライアントが切断、レスポンス不要
      return;
    }
    res.writeHead(500);
    res.end('Internal Server Error');
  }
});
```

---

## 3. タイムアウトパターン

### 3.1 基本的なタイムアウト

```typescript
// タイムアウト付き fetch
async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeoutMs?: number } = {},
): Promise<Response> {
  const { timeoutMs = 5000, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if ((error as Error).name === 'AbortError') {
      throw new TimeoutError(`Request timed out after ${timeoutMs}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// カスタムエラークラス
class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}
```

### 3.2 Promise のタイムアウトラッパー

```typescript
// Promise のタイムアウトラッパー
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new TimeoutError(`Timeout after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]);
}

// 使用
const data = await withTimeout(fetchData(), 5000);

// キャンセル可能なタイムアウト（リソースリークを防ぐ）
function withCancellableTimeout<T>(
  promise: Promise<T>,
  ms: number,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;

    const timeoutId = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new TimeoutError(`Timeout after ${ms}ms`));
      }
    }, ms);

    const cleanup = () => {
      clearTimeout(timeoutId);
    };

    // AbortSignal のキャンセル
    signal?.addEventListener('abort', () => {
      if (!settled) {
        settled = true;
        cleanup();
        reject(signal.reason);
      }
    }, { once: true });

    promise.then(
      value => {
        if (!settled) {
          settled = true;
          cleanup();
          resolve(value);
        }
      },
      error => {
        if (!settled) {
          settled = true;
          cleanup();
          reject(error);
        }
      },
    );
  });
}
```

### 3.3 段階的タイムアウト

```typescript
// 段階的タイムアウト: 警告 → タイムアウト → 強制終了
class GradualTimeout {
  private timers: NodeJS.Timeout[] = [];

  constructor(
    private warningMs: number,
    private timeoutMs: number,
    private forceMs: number,
  ) {}

  async execute<T>(
    fn: (signal: AbortSignal) => Promise<T>,
    callbacks: {
      onWarning?: () => void;
      onTimeout?: () => void;
      onForce?: () => void;
    } = {},
  ): Promise<T> {
    const controller = new AbortController();

    // 段階1: 警告
    this.timers.push(
      setTimeout(() => {
        callbacks.onWarning?.();
        console.warn(`Operation running for ${this.warningMs}ms`);
      }, this.warningMs),
    );

    // 段階2: タイムアウト（協調的キャンセル）
    this.timers.push(
      setTimeout(() => {
        callbacks.onTimeout?.();
        controller.abort(new TimeoutError(`Timeout after ${this.timeoutMs}ms`));
      }, this.timeoutMs),
    );

    // 段階3: 強制終了
    this.timers.push(
      setTimeout(() => {
        callbacks.onForce?.();
        console.error('Force terminating operation');
        // 強制終了のロジック
      }, this.forceMs),
    );

    try {
      return await fn(controller.signal);
    } finally {
      this.timers.forEach(clearTimeout);
      this.timers = [];
    }
  }
}

// 使用例
const gradual = new GradualTimeout(5000, 10000, 30000);

const result = await gradual.execute(
  async (signal) => {
    return await fetchLargeDataset(signal);
  },
  {
    onWarning: () => showSlowOperationBanner(),
    onTimeout: () => logSlowOperation(),
    onForce: () => alertOpsTeam(),
  },
);
```

---

## 4. 各言語のキャンセルメカニズム

### 4.1 Python のキャンセル

```python
import asyncio

async def cancellable_task():
    try:
        while True:
            data = await fetch_data()
            process(data)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        # クリーンアップ処理
        print("タスクがキャンセルされました")
        await cleanup_resources()
        raise  # 再送出（キャンセルを伝播）

async def main():
    task = asyncio.create_task(cancellable_task())

    await asyncio.sleep(5)
    task.cancel()  # キャンセル

    try:
        await task
    except asyncio.CancelledError:
        print("タスクが正常にキャンセルされました")

# タイムアウト
async def with_timeout():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=5.0)
    except asyncio.TimeoutError:
        print("タイムアウト")

# TaskGroup（Python 3.11+）
async def parallel_with_cancellation():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(operation_a())
        task2 = tg.create_task(operation_b())
        # 1つがエラーになると、他の全タスクがキャンセルされる

# シールド（キャンセル伝搬の防止）
async def critical_operation():
    # shield() で囲むと、外部からのキャンセルが伝搬しない
    result = await asyncio.shield(important_db_write())
    return result

# 構造化並行処理（Python 3.12+、anyio）
import anyio

async def structured_cancellation():
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker, "task1")
        tg.start_soon(worker, "task2")
        # スコープを抜けるとすべてのタスクがキャンセルされる

    # cancel_scope でタイムアウト
    with anyio.move_on_after(5.0) as scope:
        await slow_operation()
    if scope.cancelled_caught:
        print("タイムアウトしました")
```

### 4.2 Go のキャンセル（Context）

```go
package main

import (
    "context"
    "fmt"
    "net/http"
    "time"
)

// context.WithCancel: 手動キャンセル
func manualCancel() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel() // 必ず呼ぶ（リソースリーク防止）

    go func() {
        select {
        case <-ctx.Done():
            fmt.Println("キャンセルされました:", ctx.Err())
            return
        case result := <-doWork():
            fmt.Println("結果:", result)
        }
    }()

    time.Sleep(2 * time.Second)
    cancel() // キャンセル
}

// context.WithTimeout: タイムアウト
func withTimeout() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    req, _ := http.NewRequestWithContext(ctx, "GET", "https://api.example.com/data", nil)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            fmt.Println("タイムアウト")
        }
        return
    }
    defer resp.Body.Close()
}

// context.WithDeadline: 期限
func withDeadline() {
    deadline := time.Now().Add(10 * time.Second)
    ctx, cancel := context.WithDeadline(context.Background(), deadline)
    defer cancel()

    // DBクエリにコンテキストを伝搬
    rows, err := db.QueryContext(ctx, "SELECT * FROM users")
    // ...
}

// context.WithCancelCause（Go 1.20+）: キャンセル理由
func withCancelCause() {
    ctx, cancel := context.WithCancelCause(context.Background())

    go func() {
        // エラーの原因を付与してキャンセル
        cancel(fmt.Errorf("ユーザーが中断しました"))
    }()

    <-ctx.Done()
    fmt.Println("理由:", context.Cause(ctx))
}

// HTTPハンドラでのコンテキスト伝搬
func handler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context() // クライアント切断で自動キャンセル

    // 子のコンテキストにタイムアウトを追加
    ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()

    // 全ての下流呼び出しにコンテキストを渡す
    data, err := fetchData(ctx)
    if err != nil {
        if ctx.Err() == context.Canceled {
            // クライアントが切断
            return
        }
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    json.NewEncoder(w).Encode(data)
}

// goroutine でのキャンセル対応
func worker(ctx context.Context, jobs <-chan Job) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case job, ok := <-jobs:
            if !ok {
                return nil // チャネルが閉じられた
            }
            if err := processJob(ctx, job); err != nil {
                return fmt.Errorf("job %s failed: %w", job.ID, err)
            }
        }
    }
}
```

### 4.3 C# のキャンセル（CancellationToken）

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

// CancellationTokenSource でトークンを作成
public class DataService
{
    public async Task<Data> FetchDataAsync(CancellationToken cancellationToken = default)
    {
        // 定期的にキャンセルをチェック
        cancellationToken.ThrowIfCancellationRequested();

        using var httpClient = new HttpClient();
        var response = await httpClient.GetAsync(
            "https://api.example.com/data",
            cancellationToken
        );

        return await response.Content.ReadFromJsonAsync<Data>(
            cancellationToken: cancellationToken
        );
    }
}

// 使用例
public async Task Main()
{
    // 手動キャンセル
    using var cts = new CancellationTokenSource();

    var task = service.FetchDataAsync(cts.Token);

    // 5秒後にキャンセル
    cts.CancelAfter(TimeSpan.FromSeconds(5));

    try
    {
        var data = await task;
    }
    catch (OperationCanceledException)
    {
        Console.WriteLine("操作がキャンセルされました");
    }

    // LinkedToken: 複数のキャンセルソースを結合
    using var userCts = new CancellationTokenSource();
    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
        userCts.Token,
        timeoutCts.Token
    );

    await service.FetchDataAsync(linkedCts.Token);
}
```

### 4.4 Rust のキャンセル

```rust
use tokio::select;
use tokio::sync::oneshot;
use tokio::time::{timeout, Duration};
use tokio_util::sync::CancellationToken;

// tokio::select! でキャンセル
async fn cancellable_operation(cancel_rx: oneshot::Receiver<()>) -> Result<Data, Error> {
    select! {
        result = fetch_data() => result,
        _ = cancel_rx => {
            println!("Operation cancelled");
            Err(Error::Cancelled)
        }
    }
}

// CancellationToken（tokio-util）
async fn with_cancellation_token() {
    let token = CancellationToken::new();
    let child_token = token.child_token();

    // ワーカータスク
    let handle = tokio::spawn(async move {
        loop {
            select! {
                _ = child_token.cancelled() => {
                    println!("Worker cancelled");
                    break;
                }
                _ = do_work() => {}
            }
        }
    });

    // 5秒後にキャンセル
    tokio::time::sleep(Duration::from_secs(5)).await;
    token.cancel();

    handle.await.unwrap();
}

// Drop trait による自動キャンセル
struct AutoCancelGuard {
    token: CancellationToken,
}

impl Drop for AutoCancelGuard {
    fn drop(&mut self) {
        self.token.cancel();
        // スコープを抜けるときに自動的にキャンセル
    }
}

// タイムアウト
async fn with_timeout() -> Result<Data, Error> {
    match timeout(Duration::from_secs(5), fetch_data()).await {
        Ok(result) => result,
        Err(_) => Err(Error::Timeout),
    }
}
```

---

## 5. キャンセルの設計原則

```
1. キャンセルは協調的
   → 処理を強制停止するのではなく、
     「キャンセルが要求された」ことを通知
   → 処理側がクリーンアップしてから停止

2. クリーンアップを保証
   → finally でリソース解放
   → DBトランザクションのロールバック
   → テンポラリファイルの削除
   → ロックの解放

3. キャンセル可能なAPIを設計
   → AbortSignal / CancellationToken / Context を引数に受け取る
   → キャンセル時の振る舞いをドキュメント化
   → 部分的な結果の扱いを明確にする

4. レースコンディションに注意
   → キャンセルと完了が同時に起こる可能性
   → 状態チェックを適切に行う
   → 「既にキャンセル済み」の状態を処理する

5. キャンセルの伝搬
   → 親のキャンセルは子に伝搬すべき
   → 子のキャンセルは親に伝搬すべきでない（通常）
   → ツリー構造でキャンセルが伝搬する設計
```

### 5.1 キャンセル可能なAPI設計

```typescript
// === 良いAPI設計 ===

// 1. signal を options の一部として受け取る
interface FetchOptions {
  signal?: AbortSignal;
  timeout?: number;
  retries?: number;
}

async function fetchData(url: string, options: FetchOptions = {}): Promise<Data> {
  const { signal, timeout = 30000, retries = 3 } = options;

  // 外部の signal とタイムアウトを結合
  const timeoutSignal = AbortSignal.timeout(timeout);
  const combinedSignal = signal
    ? AbortSignal.any([signal, timeoutSignal])
    : timeoutSignal;

  for (let attempt = 0; attempt <= retries; attempt++) {
    combinedSignal.throwIfAborted();

    try {
      const response = await fetch(url, { signal: combinedSignal });
      if (response.ok) return await response.json();
    } catch (err) {
      if ((err as Error).name === 'AbortError') throw err;
      if (attempt === retries) throw err;
      await delay(Math.pow(2, attempt) * 1000, combinedSignal);
    }
  }

  throw new Error('Unreachable');
}

// 2. キャンセル可能なイテレータ
async function* paginatedFetch<T>(
  baseUrl: string,
  signal?: AbortSignal,
): AsyncGenerator<T[]> {
  let page = 1;
  let hasMore = true;

  while (hasMore) {
    signal?.throwIfAborted();

    const response = await fetch(`${baseUrl}?page=${page}`, { signal });
    const data = await response.json();

    yield data.items;

    hasMore = data.hasMore;
    page++;
  }
}

// 使用例
const controller = new AbortController();

for await (const items of paginatedFetch<User>('/api/users', controller.signal)) {
  for (const user of items) {
    processUser(user);
  }
}

// 3. キャンセル可能なバッチ処理
class BatchProcessor<T, R> {
  async process(
    items: T[],
    processor: (item: T, signal: AbortSignal) => Promise<R>,
    options: {
      signal?: AbortSignal;
      batchSize?: number;
      concurrency?: number;
      onProgress?: (completed: number, total: number) => void;
    } = {},
  ): Promise<{ results: R[]; errors: Array<{ item: T; error: Error }> }> {
    const { signal, batchSize = 100, concurrency = 5, onProgress } = options;
    const results: R[] = [];
    const errors: Array<{ item: T; error: Error }> = [];
    let completed = 0;

    for (let i = 0; i < items.length; i += batchSize) {
      signal?.throwIfAborted();

      const batch = items.slice(i, i + batchSize);

      // 並行数制限付きで処理
      const batchPromises = batch.map(async (item) => {
        try {
          signal?.throwIfAborted();
          const result = await processor(item, signal!);
          results.push(result);
        } catch (err) {
          if ((err as Error).name === 'AbortError') throw err;
          errors.push({ item, error: err as Error });
        } finally {
          completed++;
          onProgress?.(completed, items.length);
        }
      });

      await Promise.all(batchPromises);
    }

    return { results, errors };
  }
}
```

### 5.2 キャンセルトークンツリー

```typescript
// キャンセルの伝搬ツリー
class CancellationScope {
  private controller: AbortController;
  private children: CancellationScope[] = [];
  private cleanupFns: Array<() => void | Promise<void>> = [];

  constructor(parentSignal?: AbortSignal) {
    this.controller = new AbortController();

    // 親のキャンセルを伝搬
    if (parentSignal) {
      parentSignal.addEventListener('abort', () => {
        this.cancel(parentSignal.reason);
      }, { once: true });
    }
  }

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  // 子スコープを作成
  createChild(): CancellationScope {
    const child = new CancellationScope(this.signal);
    this.children.push(child);
    return child;
  }

  // クリーンアップ関数を登録
  onCancel(fn: () => void | Promise<void>): void {
    this.cleanupFns.push(fn);
  }

  // キャンセル実行
  async cancel(reason?: any): Promise<void> {
    if (this.controller.signal.aborted) return;

    // 子スコープを先にキャンセル
    await Promise.allSettled(
      this.children.map(child => child.cancel(reason)),
    );

    // クリーンアップを実行
    await Promise.allSettled(
      this.cleanupFns.map(fn => fn()),
    );

    this.controller.abort(reason);
  }
}

// 使用例
const rootScope = new CancellationScope();

const dbScope = rootScope.createChild();
dbScope.onCancel(async () => {
  await db.rollback();
  console.log('DB transaction rolled back');
});

const fileScope = rootScope.createChild();
fileScope.onCancel(async () => {
  await tempFile.delete();
  console.log('Temp file deleted');
});

// ルートをキャンセルすると全ての子もキャンセル
await rootScope.cancel('User requested cancellation');
```

---

## 6. テスト可能なキャンセル処理

```typescript
// === テスト ===
import { describe, it, expect, vi } from 'vitest';

describe('fetchWithTimeout', () => {
  it('タイムアウト時にTimeoutErrorをスローする', async () => {
    // 遅いレスポンスをモック
    global.fetch = vi.fn().mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 10000)),
    );

    await expect(
      fetchWithTimeout('/api/slow', { timeoutMs: 100 }),
    ).rejects.toThrow(TimeoutError);
  });

  it('signal が abort されたらAbortErrorをスローする', async () => {
    const controller = new AbortController();

    // 即座にキャンセル
    controller.abort();

    await expect(
      fetchWithTimeout('/api/data', { signal: controller.signal }),
    ).rejects.toThrow('AbortError');
  });

  it('正常なレスポンスを返す', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ data: 'test' }), { status: 200 }),
    );

    const response = await fetchWithTimeout('/api/data', { timeoutMs: 5000 });
    expect(response.status).toBe(200);
  });

  it('キャンセル後にクリーンアップが実行される', async () => {
    const cleanup = vi.fn();
    const controller = new AbortController();

    const scope = new CancellationScope(controller.signal);
    scope.onCancel(cleanup);

    controller.abort();

    // クリーンアップが呼ばれたことを確認
    await vi.waitFor(() => {
      expect(cleanup).toHaveBeenCalled();
    });
  });
});

// React コンポーネントのテスト
import { render, screen, waitFor, act } from '@testing-library/react';

describe('SearchResults', () => {
  it('query 変更時に前のリクエストをキャンセルする', async () => {
    const abortSpy = vi.spyOn(AbortController.prototype, 'abort');

    const { rerender } = render(<SearchResults query="hello" />);

    // query を変更
    rerender(<SearchResults query="world" />);

    // 前のリクエストがキャンセルされたことを確認
    expect(abortSpy).toHaveBeenCalled();
  });

  it('アンマウント時にリクエストをキャンセルする', async () => {
    const abortSpy = vi.spyOn(AbortController.prototype, 'abort');

    const { unmount } = render(<SearchResults query="test" />);

    unmount();

    expect(abortSpy).toHaveBeenCalled();
  });
});
```

---

## 7. 実践パターン集

### 7.1 デバウンス付きキャンセル

```typescript
// 検索入力のデバウンス + キャンセル
function useDebounceSearch(delayMs: number = 300) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const search = useCallback((searchQuery: string) => {
    // 前のデバウンスタイマーをクリア
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // 前のリクエストをキャンセル
    controllerRef.current?.abort();

    if (!searchQuery.trim()) {
      setResults([]);
      setLoading(false);
      return;
    }

    setLoading(true);

    timeoutRef.current = window.setTimeout(async () => {
      const controller = new AbortController();
      controllerRef.current = controller;

      try {
        const response = await fetch(
          `/api/search?q=${encodeURIComponent(searchQuery)}`,
          { signal: controller.signal },
        );
        const data = await response.json();
        setResults(data);
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          console.error('Search error:', err);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }, delayMs);
  }, [delayMs]);

  // クリーンアップ
  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return { query, setQuery: (q: string) => { setQuery(q); search(q); }, results, loading };
}
```

### 7.2 レース条件の防止

```typescript
// 最後のリクエストのみ有効にするパターン
function useLatestRequest<T>(
  fetchFn: (signal: AbortSignal) => Promise<T>,
  deps: any[],
): { data: T | null; loading: boolean; error: Error | null } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError(null);

    fetchFn(controller.signal)
      .then(result => {
        // このリクエストが最新かチェック
        if (requestId === requestIdRef.current) {
          setData(result);
          setLoading(false);
        }
      })
      .catch(err => {
        if (err.name !== 'AbortError' && requestId === requestIdRef.current) {
          setError(err);
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, deps);

  return { data, loading, error };
}
```

### 7.3 並行処理の最初の結果を使用

```typescript
// 複数のソースから最初の結果を使用し、他をキャンセル
async function raceWithCancellation<T>(
  tasks: Array<(signal: AbortSignal) => Promise<T>>,
): Promise<T> {
  const controller = new AbortController();

  try {
    const result = await Promise.race(
      tasks.map(task => task(controller.signal)),
    );
    return result;
  } finally {
    controller.abort(); // 残りのタスクをキャンセル
  }
}

// 使用例: 複数の API エンドポイントから最速の結果を使用
const data = await raceWithCancellation([
  (signal) => fetch('https://api1.example.com/data', { signal }).then(r => r.json()),
  (signal) => fetch('https://api2.example.com/data', { signal }).then(r => r.json()),
  (signal) => fetch('https://api3.example.com/data', { signal }).then(r => r.json()),
]);
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
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

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
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

## まとめ

| 手法 | 言語/環境 | 用途 |
|------|----------|------|
| AbortController | JS/TS | fetch, イベント, ストリーム |
| AbortSignal.timeout() | JS/TS | タイムアウト |
| AbortSignal.any() | JS/TS | 複数条件の結合 |
| asyncio.cancel() | Python | asyncio タスク |
| asyncio.TaskGroup | Python | 構造化並行処理 |
| Context.cancel() | Go | goroutine, HTTP, DB |
| context.WithTimeout | Go | タイムアウト |
| CancellationToken | C# | Task |
| CancellationTokenSource | C# | トークン作成 |
| tokio::select! | Rust | 非同期分岐 |
| CancellationToken | Rust (tokio-util) | トークンベースキャンセル |
| Drop trait | Rust | スコープ終了時自動キャンセル |

| 設計原則 | 説明 |
|---------|------|
| 協調的キャンセル | 強制停止ではなく、通知→クリーンアップ→停止 |
| クリーンアップ保証 | finally / defer / Drop でリソース解放 |
| キャンセル伝搬 | 親→子への伝搬、ツリー構造 |
| レース対策 | キャンセルと完了の同時発生を考慮 |
| テスタビリティ | signal を外部から注入可能にする |

---

## 8. FAQ

### Q1: AbortController は再利用できるか？

いいえ。一度 `abort()` が呼ばれた AbortController は元に戻せない。新しいリクエストごとに新しい AbortController を作成する。signal の `aborted` プロパティは一度 `true` になると変更不可。

### Q2: キャンセルされた処理の部分的な結果はどう扱うか？

設計時に明確にすべき。選択肢は3つ: (1) 部分結果を捨てて最初からやり直す、(2) 部分結果を保存して再開可能にする（チェックポイントパターン）、(3) 部分結果をそのまま返す。バッチ処理では (2) が一般的で、ファイルアップロードではチャンク単位のチェックポイントが有効。

### Q3: キャンセル処理のテストで注意すべき点は？

タイミング依存のテストは不安定になりやすい。`AbortController.abort()` を即座に呼んでキャンセルパスをテストする、`vi.useFakeTimers()` でタイマーを制御する、`signal` をモックとして注入する、といった手法を使う。非決定的なタイミングに依存しないテスト設計が重要。

### Q4: fetch 以外でAbortSignalを使えるAPIは？

Node.js では `fs/promises`（readFile, writeFile等）、`timers/promises`（setTimeout）、`events.once()`、`stream.pipeline()`、`child_process.exec()` 等がサポートしている。ブラウザでは `addEventListener` のオプションとして signal を渡せるほか、`ReadableStream`、`WritableStream`、`Blob.text()` 等もサポートしている。カスタムAPIにも `signal?.throwIfAborted()` と `signal?.addEventListener('abort', ...)` で対応可能。

### Q5: Go の context.Context と JavaScript の AbortSignal の違いは？

Go の Context はキャンセルに加えてタイムアウト（WithTimeout/WithDeadline）と値の伝搬（WithValue）を統合的に扱う。JavaScript の AbortSignal はキャンセル専用で、タイムアウトは `AbortSignal.timeout()` で実現する。Go は Context を第一引数として渡す規約があり、全てのブロッキング操作がContext対応している。JavaScript はまだ AbortSignal の普及途上で、対応していないAPIも存在する。

### Q6: キャンセルとエラーハンドリングの統合方法は？

キャンセルはエラーの一種として扱うのが一般的。TypeScript では `err.name === 'AbortError'` で判別し、Go では `errors.Is(err, context.Canceled)` で判別する。キャンセルエラーはユーザーに表示する必要がないことが多いため、エラーハンドリング層で特別扱いする。ログレベルも通常のエラーより低く設定することが推奨される。

```typescript
// エラーハンドリングとキャンセルの統合例
class AppError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number,
    public readonly isCancellation: boolean = false,
  ) {
    super(message);
    this.name = 'AppError';
  }

  static fromError(err: unknown): AppError {
    if (err instanceof AppError) return err;

    const error = err as Error;
    if (error.name === 'AbortError') {
      return new AppError('Operation cancelled', 'CANCELLED', 499, true);
    }
    if (error.name === 'TimeoutError') {
      return new AppError('Operation timed out', 'TIMEOUT', 504, false);
    }
    return new AppError(error.message, 'INTERNAL', 500, false);
  }
}

// ミドルウェアでの統合ハンドリング
async function errorHandler(ctx: Context, next: () => Promise<void>) {
  try {
    await next();
  } catch (err) {
    const appError = AppError.fromError(err);

    if (appError.isCancellation) {
      // キャンセルはdebugレベルでログ
      logger.debug('Request cancelled', { path: ctx.path });
      return; // レスポンスは不要
    }

    // 通常のエラーはerrorレベルでログ
    logger.error('Request failed', {
      path: ctx.path,
      code: appError.code,
      message: appError.message,
    });

    ctx.status = appError.statusCode;
    ctx.body = { error: { code: appError.code, message: appError.message } };
  }
}
```

---

## 次に読むべきガイド

---

## 参考文献
1. MDN Web Docs. "AbortController."
2. Node.js Documentation. "AbortController."
3. Go Documentation. "context package."
4. Python Documentation. "asyncio - Tasks."
5. Microsoft Docs. "Cancellation in Managed Threads."
6. Tokio Documentation. "Cancellation."
