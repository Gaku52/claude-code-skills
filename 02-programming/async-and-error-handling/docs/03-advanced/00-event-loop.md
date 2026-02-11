# イベントループ

> イベントループは Node.js とブラウザの非同期処理の心臓部。マイクロタスク、マクロタスク、実行順序を理解することで、非同期コードの振る舞いを正確に予測できる。

## この章で学ぶこと

- [ ] イベントループの仕組みと各フェーズを理解する
- [ ] マイクロタスクとマクロタスクの実行順序を把握する
- [ ] イベントループをブロックしないベストプラクティスを学ぶ

---

## 1. イベントループの全体像

```
Node.js のイベントループ（libuv ベース）:

  ┌──────────────────────────────────────┐
  │           イベントループ              │
  │                                      │
  │   ┌─────────────────────┐            │
  │   │ timers              │ ← setTimeout, setInterval │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ pending callbacks   │ ← I/O コールバック │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ idle, prepare       │ ← 内部使用 │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ poll                │ ← I/O イベントの取得 │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ check               │ ← setImmediate │
  │   └──────────┬──────────┘            │
  │   ┌──────────▼──────────┐            │
  │   │ close callbacks     │ ← close イベント │
  │   └──────────┬──────────┘            │
  │              └──→ 次のループへ        │
  └──────────────────────────────────────┘

  各フェーズの間に:
    → process.nextTick() キュー を処理
    → Promise マイクロタスクキュー を処理
```

---

## 2. マイクロタスク vs マクロタスク

```
マイクロタスク（優先度: 高）:
  → Promise.then/catch/finally
  → queueMicrotask()
  → process.nextTick()（Node.js、最優先）
  → MutationObserver（ブラウザ）

マクロタスク（優先度: 低）:
  → setTimeout / setInterval
  → setImmediate（Node.js）
  → I/O コールバック
  → UI レンダリング（ブラウザ）

実行順序:
  1. コールスタックが空になる
  2. マイクロタスクキューを全て処理
  3. マクロタスクを1つ処理
  4. → 2に戻る
```

```javascript
// 実行順序クイズ
console.log("1: 同期");

setTimeout(() => console.log("2: setTimeout"), 0);

Promise.resolve().then(() => console.log("3: Promise"));

queueMicrotask(() => console.log("4: queueMicrotask"));

console.log("5: 同期");

// 出力:
// 1: 同期
// 5: 同期
// 3: Promise        ← マイクロタスク
// 4: queueMicrotask ← マイクロタスク
// 2: setTimeout     ← マクロタスク
```

```javascript
// もう少し複雑な例
console.log("start");

setTimeout(() => {
  console.log("timeout 1");
  Promise.resolve().then(() => console.log("promise in timeout"));
}, 0);

Promise.resolve().then(() => {
  console.log("promise 1");
  setTimeout(() => console.log("timeout in promise"), 0);
});

setTimeout(() => console.log("timeout 2"), 0);

console.log("end");

// 出力:
// start
// end
// promise 1          ← マイクロタスク
// timeout 1          ← マクロタスク1
// promise in timeout ← timeout1内のマイクロタスク
// timeout 2          ← マクロタスク2
// timeout in promise ← promise1内のマクロタスク
```

---

## 3. イベントループのブロック

```
❌ イベントループをブロックする操作:
  → 同期的なファイルI/O（fs.readFileSync）
  → 重い計算（暗号化、画像処理）
  → 大きなJSONのパース
  → 無限ループ / 長時間ループ

ブロック時の影響:
  → 全ての非同期処理が停止
  → HTTPリクエストが応答不能
  → WebSocketメッセージが遅延
  → タイマーが不正確

対策:
  1. 同期APIを使わない（fs.readFile, not fs.readFileSync）
  2. CPU集約処理はWorkerスレッドに委譲
  3. 大きなループは分割（setImmediate で休憩）
```

```javascript
// ❌ ブロッキング
function processLargeArray(items) {
  for (const item of items) { // 100万件
    heavyComputation(item);   // イベントループが停止
  }
}

// ✅ 分割実行
async function processLargeArrayAsync(items, batchSize = 1000) {
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    for (const item of batch) {
      heavyComputation(item);
    }
    // バッチ間でイベントループに制御を返す
    await new Promise(resolve => setImmediate(resolve));
  }
}

// ✅ Worker Threads で並列実行
const { Worker } = require('worker_threads');
function runInWorker(data) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./heavy-task.js', { workerData: data });
    worker.on('message', resolve);
    worker.on('error', reject);
  });
}
```

---

## 4. ブラウザのイベントループ

```
ブラウザのイベントループ:

  ┌──────────────────────────────────┐
  │ 1. マクロタスク1つ実行            │
  │ 2. マイクロタスク全て実行          │
  │ 3. レンダリング（必要なら）        │
  │    → requestAnimationFrame       │
  │    → スタイル計算                 │
  │    → レイアウト                   │
  │    → ペイント                    │
  │ 4. → 1に戻る                     │
  └──────────────────────────────────┘

  重要: マイクロタスクが大量にあると
  → レンダリングが遅延
  → UIがフリーズしたように見える

requestAnimationFrame:
  → 次のレンダリング前に実行
  → アニメーションに最適（60fps = 16.6ms間隔）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| イベントループ | 非同期処理のスケジューラ |
| マイクロタスク | Promise.then、各マクロタスク後に全処理 |
| マクロタスク | setTimeout、1つずつ処理 |
| ブロック回避 | 同期I/O禁止、Worker活用 |
| ブラウザ | レンダリングはマクロタスク間 |

---

## 次に読むべきガイド
→ [[01-cancellation.md]] — キャンセル処理

---

## 参考文献
1. Node.js Documentation. "The Node.js Event Loop."
2. Jake Archibald. "In The Loop." JSConf.Asia, 2018.
