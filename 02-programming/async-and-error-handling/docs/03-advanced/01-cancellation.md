# キャンセル処理

> 非同期処理のキャンセルは見落とされがちだが、UXとリソース管理に直結する重要な技術。AbortController、タイムアウト、キャンセルトークンの実装を解説。

## この章で学ぶこと

- [ ] 非同期処理のキャンセルが必要な場面を理解する
- [ ] AbortController の使い方を把握する
- [ ] タイムアウトパターンの実装を学ぶ

---

## 1. なぜキャンセルが必要か

```
キャンセルが必要な場面:
  1. ユーザーがページ遷移した → 前のページのAPIリクエストを中止
  2. 検索ボックスの入力 → 前の検索リクエストを中止
  3. タイムアウト → 一定時間内に応答がない場合に中止
  4. コンポーネントのアンマウント → 進行中の処理を中止
  5. ユーザーが「キャンセル」ボタンを押した

キャンセルしないと:
  → 不要なネットワークリクエストが残る
  → メモリリーク（コンポーネント破棄後にsetState）
  → レースコンディション（古い結果が新しい結果を上書き）
```

---

## 2. AbortController（Web標準）

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

// React での使用
function SearchResults({ query }: { query: string }) {
  const [results, setResults] = useState([]);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`/api/search?q=${query}`, { signal: controller.signal })
      .then(res => res.json())
      .then(data => setResults(data))
      .catch(err => {
        if (err.name !== 'AbortError') throw err;
      });

    // クリーンアップ: コンポーネント破棄時 or query変更時にキャンセル
    return () => controller.abort();
  }, [query]);

  return <ul>{results.map(r => <li key={r.id}>{r.name}</li>)}</ul>;
}
```

---

## 3. タイムアウトパターン

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

// Promise のタイムアウトラッパー
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new TimeoutError(`Timeout after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]);
}

// 使用
const data = await withTimeout(fetchData(), 5000);
```

---

## 4. Python のキャンセル

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

3. キャンセル可能なAPIを設計
   → AbortSignal を引数に受け取る
   → キャンセル時の振る舞いをドキュメント化

4. レースコンディションに注意
   → キャンセルと完了が同時に起こる可能性
   → 状態チェックを適切に行う
```

---

## まとめ

| 手法 | 言語/環境 | 用途 |
|------|----------|------|
| AbortController | JS/TS | fetch, イベント |
| asyncio.cancel() | Python | asyncio タスク |
| Context.cancel() | Go | goroutine |
| CancellationToken | C# | Task |
| Drop trait | Rust | スコープ終了時 |

---

## 次に読むべきガイド
→ [[02-retry-and-backoff.md]] — リトライ戦略

---

## 参考文献
1. MDN Web Docs. "AbortController."
2. Node.js Documentation. "AbortController."
