# イベントループ

> イベントループは Node.js とブラウザの非同期処理の心臓部。マイクロタスク、マクロタスク、実行順序を理解することで、非同期コードの振る舞いを正確に予測できる。

## この章で学ぶこと

- [ ] イベントループの仕組みと各フェーズを理解する
- [ ] マイクロタスクとマクロタスクの実行順序を把握する
- [ ] イベントループをブロックしないベストプラクティスを学ぶ
- [ ] Node.js と ブラウザのイベントループの違いを理解する
- [ ] Worker Threads / Web Workers の活用法を身につける
- [ ] パフォーマンス計測とデバッグ手法を習得する

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

### 1.1 各フェーズの詳細

```
timers フェーズ:
  → setTimeout() と setInterval() のコールバックを実行
  → 最小遅延は1ms（0を指定しても1msに切り上げ）
  → タイマーは「最低でもN ms後に実行」であり、正確なN ms後ではない
  → 大量のタイマーがあると、この フェーズで時間を消費する

pending callbacks フェーズ:
  → 前のイテレーションで延期されたI/Oコールバックを実行
  → TCP接続エラーなどのシステムオペレーションのコールバック
  → 例: ECONNREFUSED エラーのコールバック

idle, prepare フェーズ:
  → Node.js の内部使用のみ
  → ユーザーコードからは直接触れない

poll フェーズ（最も重要）:
  → 新しいI/Oイベントを取得し、I/Oコールバックを実行
  → fs.readFile, HTTP リクエスト応答, DB クエリ結果などを処理
  → このフェーズでブロックする可能性がある（他にタスクがない場合）
  → ブロック時間の上限は、次のtimersフェーズの最も近いタイマーまで

check フェーズ:
  → setImmediate() のコールバックを実行
  → poll フェーズの直後に実行されることが保証される
  → I/Oコールバック内では setTimeout(fn, 0) より先に実行される

close callbacks フェーズ:
  → socket.on('close', ...) などのクローズイベントを処理
  → クリーンアップ処理に使われる
```

### 1.2 実行の全体フロー

```
Node.js プロセス起動
    │
    ▼
┌──────────────────────────────────────┐
│ 1. モジュールの読み込み・コンパイル   │
│    → require() / import の解決       │
│    → トップレベルコードの同期実行     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 2. process.nextTick キューの処理     │
│    → マイクロタスクキューの処理       │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 3. イベントループ開始                │
│    ┌─→ timers                        │
│    │   → nextTick + microtasks       │
│    │   pending callbacks             │
│    │   → nextTick + microtasks       │
│    │   idle, prepare                 │
│    │   → nextTick + microtasks       │
│    │   poll (I/O待ち)                │
│    │   → nextTick + microtasks       │
│    │   check (setImmediate)          │
│    │   → nextTick + microtasks       │
│    │   close callbacks               │
│    │   → nextTick + microtasks       │
│    └─← 次のイテレーション            │
└──────────────────────────────────────┘
                   │
                   ▼ (処理するタスクが無くなったら)
┌──────────────────────────────────────┐
│ 4. プロセス終了                      │
│    → 'exit' イベント発行             │
│    → process.exit()                  │
└──────────────────────────────────────┘
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
  → requestAnimationFrame（ブラウザ、レンダリング前）
  → MessageChannel

実行順序:
  1. コールスタックが空になる
  2. マイクロタスクキューを全て処理
  3. マクロタスクを1つ処理
  4. → 2に戻る

Node.js での優先順位:
  process.nextTick > Promise microtask > setImmediate > setTimeout
```

### 2.1 基本的な実行順序

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

### 2.2 ネストした非同期処理

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

### 2.3 process.nextTick vs Promise vs queueMicrotask

```javascript
// Node.js での優先順位
console.log("1: 同期");

process.nextTick(() => {
  console.log("2: nextTick");
});

Promise.resolve().then(() => {
  console.log("3: Promise");
});

queueMicrotask(() => {
  console.log("4: queueMicrotask");
});

setImmediate(() => {
  console.log("5: setImmediate");
});

setTimeout(() => {
  console.log("6: setTimeout");
}, 0);

console.log("7: 同期");

// 出力:
// 1: 同期
// 7: 同期
// 2: nextTick           ← nextTick キュー（最優先）
// 3: Promise            ← マイクロタスクキュー
// 4: queueMicrotask     ← マイクロタスクキュー
// 5: setImmediate       ← check フェーズ
// 6: setTimeout         ← timers フェーズ
// ※ setImmediate と setTimeout(,0) の順序はタイミングにより変わる可能性あり
```

### 2.4 nextTick の再帰呼び出しの危険性

```javascript
// ❌ nextTick のスターベーション問題
// nextTick が再帰的に呼ばれると、イベントループが進まない
function recursiveNextTick() {
  process.nextTick(() => {
    console.log("nextTick");
    recursiveNextTick(); // 永遠にnextTickが実行され続ける
  });
}
recursiveNextTick();
// setTimeout のコールバックは永遠に実行されない！

// ✅ setImmediate を使う（イベントループの1イテレーションを許可）
function recursiveImmediate() {
  setImmediate(() => {
    console.log("immediate");
    recursiveImmediate(); // 他のタスクも実行される余地がある
  });
}
```

### 2.5 高度な実行順序パズル

```javascript
// async/await を含む実行順序
async function asyncA() {
  console.log("A1");
  await Promise.resolve();
  console.log("A2");
}

async function asyncB() {
  console.log("B1");
  await asyncA();
  console.log("B2");
}

console.log("start");

asyncB();

Promise.resolve().then(() => console.log("P1"));

console.log("end");

// 出力:
// start
// B1
// A1        ← asyncA の同期部分
// end
// A2        ← await 後（マイクロタスク）
// P1        ← Promise.then（マイクロタスク）
// B2        ← await asyncA() の後（マイクロタスク）

// ポイント:
// - async関数の await 前の部分は同期的に実行される
// - await は内部的に .then() に変換される
// - 各 await の続きはマイクロタスクとしてキューに入る
```

```javascript
// Promise チェーンの実行順序
Promise.resolve()
  .then(() => console.log("then 1"))
  .then(() => console.log("then 2"))
  .then(() => console.log("then 3"));

Promise.resolve()
  .then(() => console.log("then A"))
  .then(() => console.log("then B"))
  .then(() => console.log("then C"));

// 出力:
// then 1  ← 最初のPromiseチェーンの1段目
// then A  ← 2番目のPromiseチェーンの1段目
// then 2  ← 最初のPromiseチェーンの2段目
// then B  ← 2番目のPromiseチェーンの2段目
// then 3  ← 最初のPromiseチェーンの3段目
// then C  ← 2番目のPromiseチェーンの3段目

// ポイント: .then() は1段ずつマイクロタスクキューに追加される
// 最初の .then() が実行されると、次の .then() がキューに追加される
// そのため、交互に実行される（ラウンドロビン的）
```

---

## 3. イベントループのブロック

```
❌ イベントループをブロックする操作:
  → 同期的なファイルI/O（fs.readFileSync）
  → 重い計算（暗号化、画像処理）
  → 大きなJSONのパース（JSON.parse）
  → 正規表現の指数的バックトラッキング
  → 無限ループ / 長時間ループ
  → 同期的なHTTPリクエスト
  → 大きな配列のソート

ブロック時の影響:
  → 全ての非同期処理が停止
  → HTTPリクエストが応答不能
  → WebSocketメッセージが遅延
  → タイマーが不正確
  → ヘルスチェックがタイムアウト
  → クライアントがタイムアウトエラー

対策:
  1. 同期APIを使わない（fs.readFile, not fs.readFileSync）
  2. CPU集約処理はWorkerスレッドに委譲
  3. 大きなループは分割（setImmediate で休憩）
  4. ストリーミング処理で大きなデータを分割
  5. 正規表現の安全性を検証（ReDoS対策）
```

### 3.1 ブロッキングの検出と回避

```javascript
// ❌ ブロッキング
function processLargeArray(items) {
  for (const item of items) { // 100万件
    heavyComputation(item);   // イベントループが停止
  }
}

// ✅ 分割実行（setImmediate でイベントループに制御を返す）
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

### 3.2 大きなJSONのストリーミング処理

```javascript
const { createReadStream } = require('fs');
const { pipeline } = require('stream/promises');
const JSONStream = require('jsonstream2');

// ❌ 大きなJSONを一括読み込み（メモリ＋ブロッキング問題）
async function processLargeJsonBad(filePath) {
  const data = JSON.parse(await fs.readFile(filePath, 'utf8')); // 500MB → ブロック
  for (const item of data) {
    await processItem(item);
  }
}

// ✅ ストリーミングで逐次処理
async function processLargeJsonGood(filePath) {
  const stream = createReadStream(filePath)
    .pipe(JSONStream.parse('*')); // 配列の各要素を1つずつ発行

  for await (const item of stream) {
    await processItem(item);
  }
}

// ✅ NDJSON（改行区切りJSON）のストリーミング処理
const readline = require('readline');

async function processNDJSON(filePath) {
  const rl = readline.createInterface({
    input: createReadStream(filePath),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (line.trim()) {
      const item = JSON.parse(line);
      await processItem(item);
    }
  }
}
```

### 3.3 正規表現のバックトラッキング対策（ReDoS）

```javascript
// ❌ 危険な正規表現（指数的バックトラッキング）
const dangerousRegex = /^(a+)+$/;
// "aaaaaaaaaaaaaaaaab" に対して指数関数的に時間がかかる

// ❌ これもReDoS脆弱性
const emailRegex = /^([a-zA-Z0-9]+\.)*[a-zA-Z0-9]+@[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*$/;

// ✅ 安全な正規表現の書き方
// 1. バックトラッキングを避ける具体的な文字クラスを使用
const safeEmailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// 2. re2 ライブラリを使用（バックトラッキングしない正規表現エンジン）
const RE2 = require('re2');
const safeRegex = new RE2('^[a-z]+$');

// 3. タイムアウト付き正規表現実行
function safeRegexTest(regex, input, timeoutMs = 100) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(`
      const { parentPort, workerData } = require('worker_threads');
      const result = new RegExp(workerData.pattern).test(workerData.input);
      parentPort.postMessage(result);
    `, {
      eval: true,
      workerData: { pattern: regex.source, input },
    });

    const timeout = setTimeout(() => {
      worker.terminate();
      reject(new Error('Regex execution timed out'));
    }, timeoutMs);

    worker.on('message', result => {
      clearTimeout(timeout);
      resolve(result);
    });
  });
}
```

### 3.4 イベントループのモニタリング

```javascript
// イベントループの遅延を計測
function monitorEventLoop(thresholdMs = 100) {
  let lastTime = process.hrtime.bigint();

  setInterval(() => {
    const now = process.hrtime.bigint();
    const delta = Number(now - lastTime) / 1_000_000; // ns → ms
    const lag = delta - 1000; // 期待値1000msとの差

    if (lag > thresholdMs) {
      console.warn(`⚠️ Event loop lag: ${lag.toFixed(1)}ms`);
    }

    lastTime = now;
  }, 1000);
}

// perf_hooks を使った精密な計測
const { monitorEventLoopDelay } = require('perf_hooks');

const histogram = monitorEventLoopDelay({ resolution: 20 });
histogram.enable();

// 定期的に統計を出力
setInterval(() => {
  console.log({
    min: histogram.min / 1e6,      // ns → ms
    max: histogram.max / 1e6,
    mean: histogram.mean / 1e6,
    p50: histogram.percentile(50) / 1e6,
    p95: histogram.percentile(95) / 1e6,
    p99: histogram.percentile(99) / 1e6,
  });
  histogram.reset();
}, 10000);

// Prometheus メトリクスとして公開
const { collectDefaultMetrics, register, Histogram } = require('prom-client');

collectDefaultMetrics(); // デフォルトメトリクスにイベントループ遅延を含む

const eventLoopLag = new Histogram({
  name: 'nodejs_eventloop_lag_seconds',
  help: 'Lag of event loop in seconds',
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
});

// ヘルスチェックエンドポイント
app.get('/health', (req, res) => {
  const h = monitorEventLoopDelay({ resolution: 20 });
  h.enable();
  setTimeout(() => {
    h.disable();
    const p99 = h.percentile(99) / 1e6;
    if (p99 > 500) {
      res.status(503).json({ status: 'unhealthy', eventLoopLag: p99 });
    } else {
      res.status(200).json({ status: 'healthy', eventLoopLag: p99 });
    }
  }, 1000);
});
```

---

## 4. Worker Threads（Node.js）

```javascript
// === メインスレッド ===
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

// Worker プールの実装
class WorkerPool {
  constructor(workerPath, numWorkers) {
    this.workerPath = workerPath;
    this.workers = [];
    this.freeWorkers = [];
    this.taskQueue = [];

    for (let i = 0; i < numWorkers; i++) {
      this.addWorker();
    }
  }

  addWorker() {
    const worker = new Worker(this.workerPath);
    worker.on('message', (result) => {
      // タスクのPromiseを解決
      worker.currentResolve(result);
      worker.currentResolve = null;

      // キューにタスクがあれば次を実行
      if (this.taskQueue.length > 0) {
        const { data, resolve, reject } = this.taskQueue.shift();
        this.runTask(worker, data, resolve, reject);
      } else {
        this.freeWorkers.push(worker);
      }
    });

    worker.on('error', (err) => {
      if (worker.currentReject) {
        worker.currentReject(err);
      }
    });

    this.workers.push(worker);
    this.freeWorkers.push(worker);
  }

  runTask(worker, data, resolve, reject) {
    worker.currentResolve = resolve;
    worker.currentReject = reject;
    worker.postMessage(data);
  }

  execute(data) {
    return new Promise((resolve, reject) => {
      if (this.freeWorkers.length > 0) {
        const worker = this.freeWorkers.pop();
        this.runTask(worker, data, resolve, reject);
      } else {
        this.taskQueue.push({ data, resolve, reject });
      }
    });
  }

  async shutdown() {
    for (const worker of this.workers) {
      await worker.terminate();
    }
  }
}

// 使用例
const pool = new WorkerPool('./crypto-worker.js', 4); // 4ワーカー

// 並行してハッシュ計算
async function hashPasswords(passwords) {
  const results = await Promise.all(
    passwords.map(pw => pool.execute({ password: pw }))
  );
  return results;
}

// === ワーカースレッド（crypto-worker.js） ===
const { parentPort } = require('worker_threads');
const crypto = require('crypto');

parentPort.on('message', ({ password }) => {
  // CPU集約的な処理をワーカーで実行
  const hash = crypto.pbkdf2Sync(password, 'salt', 100000, 64, 'sha512');
  parentPort.postMessage(hash.toString('hex'));
});
```

### 4.1 SharedArrayBuffer による共有メモリ

```javascript
// メインスレッド
const { Worker } = require('worker_threads');

// 共有メモリバッファ（全ワーカーからアクセス可能）
const sharedBuffer = new SharedArrayBuffer(1024 * Int32Array.BYTES_PER_ELEMENT);
const sharedArray = new Int32Array(sharedBuffer);

// 複数のワーカーで共有メモリに書き込み
const workers = [];
for (let i = 0; i < 4; i++) {
  const worker = new Worker('./shared-worker.js', {
    workerData: { buffer: sharedBuffer, workerId: i },
  });
  workers.push(worker);
}

// === shared-worker.js ===
const { parentPort, workerData } = require('worker_threads');
const { buffer, workerId } = workerData;
const sharedArray = new Int32Array(buffer);

// Atomics でスレッドセーフな操作
Atomics.add(sharedArray, 0, 1); // アトミックに加算

// Atomics.wait / Atomics.notify でスレッド間同期
Atomics.wait(sharedArray, 1, 0); // sharedArray[1] が 0 の間待機
// ... 別のスレッドが Atomics.notify(sharedArray, 1) で起こす

parentPort.postMessage({ done: true, workerId });
```

---

## 5. ブラウザのイベントループ

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
  → マイクロタスクでもマクロタスクでもない独立したキュー
```

### 5.1 requestAnimationFrame の詳細

```javascript
// requestAnimationFrame はレンダリング前に実行
console.log("1: 同期");

requestAnimationFrame(() => console.log("2: rAF"));

setTimeout(() => console.log("3: setTimeout"), 0);

Promise.resolve().then(() => console.log("4: Promise"));

console.log("5: 同期");

// 出力:
// 1: 同期
// 5: 同期
// 4: Promise         ← マイクロタスク
// 2: rAF             ← レンダリング前（通常 setTimeout より先）
// 3: setTimeout      ← マクロタスク
// ※ rAF と setTimeout の順序はブラウザの実装により異なる場合あり

// === スムーズなアニメーション ===
function animate(element, targetX, duration) {
  const startX = element.offsetLeft;
  const startTime = performance.now();

  function frame(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // イージング関数
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic

    element.style.left = startX + (targetX - startX) * eased + 'px';

    if (progress < 1) {
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

// === requestIdleCallback（優先度の低いタスク）===
// ブラウザがアイドル状態の時に実行される
function processNonUrgentWork(tasks) {
  function doWork(deadline) {
    // deadline.timeRemaining() でフレーム内の残り時間を確認
    while (tasks.length > 0 && deadline.timeRemaining() > 1) {
      const task = tasks.shift();
      task();
    }

    if (tasks.length > 0) {
      requestIdleCallback(doWork);
    }
  }

  requestIdleCallback(doWork, { timeout: 5000 }); // 最大5秒待ち
}

// 使用例: アナリティクスの送信
processNonUrgentWork([
  () => sendAnalytics('page_view', { path: location.pathname }),
  () => preloadImages(nextPageImages),
  () => prefetchData('/api/next-page'),
]);
```

### 5.2 Web Workers

```javascript
// === メインスレッド ===
const worker = new Worker('worker.js');

// メッセージの送受信
worker.postMessage({ type: 'process', data: largeDataset });

worker.onmessage = (event) => {
  const { result, stats } = event.data;
  updateUI(result);
  console.log('処理統計:', stats);
};

worker.onerror = (error) => {
  console.error('Worker error:', error.message);
};

// Transferable Objects（コピーではなく所有権の移転）
const buffer = new ArrayBuffer(1024 * 1024); // 1MB
worker.postMessage({ buffer }, [buffer]); // 転送（コピーなし）
// この時点で buffer は使用不可

// === worker.js ===
self.onmessage = (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'process': {
      const startTime = performance.now();

      // CPU集約的な処理（メインスレッドをブロックしない）
      const result = data.map(item => {
        return heavyComputation(item);
      });

      const duration = performance.now() - startTime;

      self.postMessage({
        result,
        stats: {
          itemCount: data.length,
          duration: `${duration.toFixed(2)}ms`,
        },
      });
      break;
    }
  }
};

// === Comlink ライブラリで Worker をRPCのように使う ===
// メインスレッド
import * as Comlink from 'comlink';

const api = Comlink.wrap(new Worker('api-worker.js'));

// Worker のメソッドを直接呼び出すように使える
const result = await api.processData(largeDataset);
const hash = await api.hashPassword('secret');

// api-worker.js
import * as Comlink from 'comlink';

const api = {
  processData(data) {
    return data.map(item => heavyComputation(item));
  },
  hashPassword(password) {
    // CPU集約的なハッシュ計算
    return computeHash(password);
  },
};

Comlink.expose(api);
```

---

## 6. Node.js vs ブラウザ の違い

```
┌──────────────────────────────────────────────────┐
│              Node.js vs ブラウザ                   │
├─────────────────┬────────────────────────────────┤
│     Node.js     │         ブラウザ                │
├─────────────────┼────────────────────────────────┤
│ libuv ベース     │ ブラウザエンジン独自実装       │
│ 6フェーズ        │ タスクキュー + レンダリング    │
│ setImmediate ○  │ setImmediate △(IE のみ)       │
│ nextTick ○      │ nextTick ✗                    │
│ Worker Threads  │ Web Workers                    │
│ レンダリング無し │ レンダリングが挟まる           │
│ 複数タスクキュー │ 単一タスクキュー（基本）       │
│ fs, net, etc    │ DOM, fetch, etc                │
│ サーバーサイド   │ クライアントサイド             │
└─────────────────┴────────────────────────────────┘

setImmediate vs setTimeout(fn, 0):
  Node.js:
    → I/Oコールバック内: setImmediate が先
    → トップレベル: 順序不定
  ブラウザ:
    → setTimeout(fn, 0) のみ（最小遅延4ms）
    → setImmediate は非標準
```

```javascript
// Node.js: I/O コールバック内での順序
const fs = require('fs');

fs.readFile('file.txt', () => {
  setTimeout(() => console.log('timeout'), 0);
  setImmediate(() => console.log('immediate'));
});

// 出力（常にこの順序）:
// immediate    ← I/Oコールバック → check フェーズが先
// timeout

// Node.js: トップレベルでの順序（不定）
setTimeout(() => console.log('timeout'), 0);
setImmediate(() => console.log('immediate'));

// 出力（実行ごとに変わる可能性）:
// timeout   または  immediate
// immediate         timeout
// → プロセス起動時のタイミングに依存
```

---

## 7. 実践パターン

### 7.1 非同期イテレータとイベントループ

```javascript
// for-await-of とイベントループ
const { once } = require('events');
const { createReadStream } = require('fs');

async function processFile(filePath) {
  const stream = createReadStream(filePath, { encoding: 'utf8' });
  let lineCount = 0;

  for await (const chunk of stream) {
    const lines = chunk.split('\n');
    for (const line of lines) {
      lineCount++;
      await processLine(line);

      // 1000行ごとにイベントループに制御を返す
      if (lineCount % 1000 === 0) {
        await new Promise(resolve => setImmediate(resolve));
      }
    }
  }

  return lineCount;
}
```

### 7.2 Promise.all とイベントループ

```javascript
// Promise.all は全てのPromiseを同時に開始する
// → 大量のPromiseを同時実行するとリソースを圧迫

// ❌ 1万件のHTTPリクエストを同時実行
const urls = Array(10000).fill('https://api.example.com/data');
const results = await Promise.all(urls.map(url => fetch(url)));
// → ソケットの枯渇、メモリ圧迫

// ✅ 並行数を制限する
async function promisePool(tasks, concurrency = 10) {
  const results = [];
  const executing = new Set();

  for (const [index, task] of tasks.entries()) {
    const promise = task().then(result => {
      executing.delete(promise);
      return result;
    });
    executing.add(promise);
    results[index] = promise;

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  return Promise.all(results);
}

// 使用
const results = await promisePool(
  urls.map(url => () => fetch(url).then(r => r.json())),
  10, // 最大10並行
);
```

### 7.3 Graceful Shutdown

```javascript
const http = require('http');

const server = http.createServer(handler);

// 新しいリクエストの追跡
const connections = new Set();
server.on('connection', (conn) => {
  connections.add(conn);
  conn.on('close', () => connections.delete(conn));
});

// シグナルハンドリング
let isShuttingDown = false;

async function gracefulShutdown(signal) {
  if (isShuttingDown) return;
  isShuttingDown = true;

  console.log(`${signal} received. Starting graceful shutdown...`);

  // 1. 新しいリクエストの受付を停止
  server.close(() => {
    console.log('Server closed');
  });

  // 2. ヘルスチェックを不健全にする（ロードバランサーからの切り離し）
  // → /health エンドポイントで isShuttingDown をチェック

  // 3. 進行中のリクエストの完了を待つ（最大30秒）
  const forceTimeout = setTimeout(() => {
    console.log('Force shutdown: destroying remaining connections');
    connections.forEach(conn => conn.destroy());
  }, 30000);

  // 4. リソースのクリーンアップ
  try {
    await Promise.allSettled([
      db.end(),
      redis.quit(),
      messageQueue.close(),
    ]);
    console.log('Resources cleaned up');
  } catch (err) {
    console.error('Cleanup error:', err);
  }

  clearTimeout(forceTimeout);
  process.exit(0);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// ヘルスチェック
app.get('/health', (req, res) => {
  if (isShuttingDown) {
    res.status(503).json({ status: 'shutting-down' });
  } else {
    res.status(200).json({ status: 'healthy' });
  }
});
```

### 7.4 タイマーの精度問題

```javascript
// setTimeout(fn, 0) は実際には 0ms ではない
// Node.js: 最小 1ms
// ブラウザ: 最小 4ms（5回以上ネストした場合）

// 高精度タイミングが必要な場合
function preciseTimeout(callback, ms) {
  const start = performance.now();

  function check() {
    const elapsed = performance.now() - start;
    if (elapsed >= ms) {
      callback();
    } else if (ms - elapsed > 10) {
      setTimeout(check, 0); // 大まかに待つ
    } else {
      // 最後のミリ秒はビジーウェイト（精度のため）
      setImmediate(check);
    }
  }

  if (ms <= 0) {
    setImmediate(callback);
  } else {
    setTimeout(check, Math.max(0, ms - 10));
  }
}

// setInterval の「ドリフト」問題
// ❌ 1秒ごとに実行したいが、徐々にずれる
let count = 0;
const start = Date.now();
setInterval(() => {
  count++;
  const expected = count * 1000;
  const actual = Date.now() - start;
  console.log(`ドリフト: ${actual - expected}ms`);
}, 1000);

// ✅ 自己補正タイマー
function preciseInterval(callback, intervalMs) {
  let expected = Date.now() + intervalMs;

  function step() {
    const drift = Date.now() - expected;
    callback();
    expected += intervalMs;
    setTimeout(step, Math.max(0, intervalMs - drift));
  }

  setTimeout(step, intervalMs);
}
```

---

## 8. デバッグとトラブルシューティング

### 8.1 よくある問題パターン

```javascript
// 問題1: 意図しない順序でのコールバック実行
function fetchAndProcess() {
  let result = null;

  fetch('/api/data')
    .then(r => r.json())
    .then(data => { result = data; });

  console.log(result); // null！（非同期処理が完了していない）
}

// 問題2: Unhandled Promise Rejection
// Node.js 15+ ではプロセスがクラッシュする
async function riskyOperation() {
  const data = await fetch('/api/data'); // エラーをキャッチしていない
  return data.json();
}

riskyOperation(); // .catch() も try-catch もなし → UnhandledPromiseRejection

// 対策: グローバルハンドラ
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  // ログを送信し、グレースフルにシャットダウン
  gracefulShutdown('unhandledRejection');
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  // 即座にシャットダウン（状態が不整合の可能性）
  process.exit(1);
});

// 問題3: メモリリーク（イベントリスナーの解除忘れ）
const EventEmitter = require('events');
const emitter = new EventEmitter();

// ❌ リスナーが蓄積される
function handleRequest(req) {
  emitter.on('data', (data) => {
    // リクエストごとに新しいリスナーが追加される
    // → メモリリーク
  });
}

// ✅ once を使うか、手動で解除
function handleRequestFixed(req) {
  const handler = (data) => {
    // 処理
  };
  emitter.on('data', handler);

  // リクエスト終了時に解除
  req.on('close', () => {
    emitter.removeListener('data', handler);
  });
}

// MaxListenersExceededWarning の検出
emitter.setMaxListeners(20); // デフォルト10
// 警告が出たらリスナーリークを疑う
```

### 8.2 Node.js の診断ツール

```javascript
// --inspect フラグで Chrome DevTools に接続
// node --inspect server.js
// Chrome で chrome://inspect を開く

// CPU プロファイリング
const { writeHeapSnapshot } = require('v8');
const { Session } = require('inspector');

// ヒープスナップショットの取得
app.get('/debug/heap', (req, res) => {
  const filename = writeHeapSnapshot();
  res.json({ file: filename });
});

// CPU プロファイルの取得
app.get('/debug/profile', async (req, res) => {
  const session = new Session();
  session.connect();

  session.post('Profiler.enable');
  session.post('Profiler.start');

  // 10秒間プロファイリング
  await new Promise(resolve => setTimeout(resolve, 10000));

  session.post('Profiler.stop', (err, { profile }) => {
    session.disconnect();
    // profile を .cpuprofile ファイルとして保存
    fs.writeFileSync('profile.cpuprofile', JSON.stringify(profile));
    res.json({ message: 'Profile saved' });
  });
});

// async_hooks でイベントループの追跡
const async_hooks = require('async_hooks');

const resources = new Map();

const hook = async_hooks.createHook({
  init(asyncId, type, triggerAsyncId) {
    resources.set(asyncId, { type, triggerAsyncId, created: Date.now() });
  },
  destroy(asyncId) {
    resources.delete(asyncId);
  },
});

// 有効化（パフォーマンスオーバーヘッドあり、デバッグ時のみ）
hook.enable();

// アクティブな非同期リソースの表示
setInterval(() => {
  console.log(`Active async resources: ${resources.size}`);
  const types = {};
  for (const [, { type }] of resources) {
    types[type] = (types[type] || 0) + 1;
  }
  console.log(types);
}, 10000);
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| イベントループ | 非同期処理のスケジューラ、6フェーズで構成（Node.js） |
| マイクロタスク | Promise.then、各マクロタスク後に全処理 |
| マクロタスク | setTimeout、1つずつ処理 |
| process.nextTick | マイクロタスクより優先、スターベーションに注意 |
| ブロック回避 | 同期I/O禁止、Worker活用、分割実行 |
| ブラウザ | レンダリングはマクロタスク間、rAFはレンダリング前 |
| Worker Threads | CPU集約処理の委譲、SharedArrayBufferで共有メモリ |
| モニタリング | perf_hooks、async_hooks、ヒープスナップショット |
| Graceful Shutdown | シグナル処理、リソースクリーンアップ、タイムアウト |

---

## 9. FAQ

### Q1: setTimeout(fn, 0) は本当に0msなのか？

Node.js では最小遅延は1ms。ブラウザでは通常4ms（ネストが5回以上の場合）。これは仕様として定められている。正確なタイミングが必要な場合は、`performance.now()` で自己補正するか、`setImmediate`（Node.js）や `requestAnimationFrame`（ブラウザ）を使用する。

### Q2: async/await はイベントループにどう影響するか？

`async/await` は構文糖であり、内部的には Promise を使用する。`await` の直後のコードはマイクロタスクとしてキューに入る。したがって、`await` はイベントループをブロックしない。ただし、`await` する対象が同期的に重い計算を行う場合は、その計算自体がイベントループをブロックする。

### Q3: process.nextTick() と queueMicrotask() のどちらを使うべきか？

新しいコードでは `queueMicrotask()` を推奨する。`process.nextTick()` はNode.js固有であり、マイクロタスクより優先度が高いためスターベーション問題を起こす可能性がある。`queueMicrotask()` はWeb標準でありブラウザでも動作する。ただし、I/Oコールバックの前に確実に実行したい場合は `process.nextTick()` が適切。

### Q4: イベントループが空になるとプロセスは終了するか？

はい。Node.js はイベントループのキューが全て空になり、保留中のI/O操作やタイマーがなくなると自動的に終了する。`setInterval` や `server.listen()` などのアクティブなハンドルがあるとプロセスは終了しない。`unref()` を呼ぶとハンドルをイベントループのカウントから除外でき、他にアクティブなハンドルがなければプロセスが終了する。

### Q5: Deno/Bun のイベントループはNode.jsと違うか？

Deno は Tokio（Rustの非同期ランタイム）をベースとしており、Node.js の libuv とは異なるが、マイクロタスク/マクロタスクの概念は同じ。Bun は独自のイベントループ実装（JavaScriptCore + liburing on Linux）を持ち、Node.js と高い互換性を保ちながらパフォーマンスを向上させている。基本的な実行順序の規則は全環境で共通。

---

## 次に読むべきガイド
→ [[01-cancellation.md]] — キャンセル処理

---

## 参考文献
1. Node.js Documentation. "The Node.js Event Loop."
2. Jake Archibald. "In The Loop." JSConf.Asia, 2018.
3. Node.js Documentation. "Worker Threads."
4. MDN Web Docs. "The event loop." developer.mozilla.org.
5. libuv Documentation. "Design overview." docs.libuv.org.
6. Erin Zimmer. "Further Adventures of the Event Loop." JSConf EU, 2018.
