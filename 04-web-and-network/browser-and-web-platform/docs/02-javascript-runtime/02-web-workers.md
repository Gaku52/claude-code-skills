# Web Workers

> Web Workers はメインスレッドとは別のバックグラウンドスレッドで JavaScript を実行する仕組みである。重い計算処理をオフロードして UI の応答性を維持し、マルチスレッドプログラミングのパターンを Web に持ち込む。Dedicated Worker、Shared Worker、Service Worker の 3 種類を正しく使い分けることで、高パフォーマンスかつオフライン対応のアプリケーションを構築できる。

## この章で学ぶこと

- [ ] Web Worker の基本概念とブラウザのスレッドモデルを理解する
- [ ] Dedicated Worker の生成、メッセージパッシング、終了を実装できる
- [ ] Shared Worker で複数タブ間の状態共有を実現できる
- [ ] Service Worker のライフサイクルとキャッシュ戦略を設計できる
- [ ] Transferable Objects と SharedArrayBuffer の使い分けを判断できる
- [ ] Worker プールパターンによる並列処理を設計できる
- [ ] Worklet の種類と用途を把握する

---

## 前提知識

- **ブラウザのイベントループ** → 参照: [イベントループ](./01-event-loop-browser.md)
  Web Worker がメインスレッドとどのように協調動作するかを理解するために、イベントループの仕組み（タスクキュー、マイクロタスク、レンダリングタイミング）を事前に把握しておく必要がある。

- **V8 エンジンの仕組み** → 参照: [V8 エンジン](./00-v8-engine.md)
  Worker スレッドも V8 エンジンで実行されるため、JIT コンパイル、ガベージコレクション、ヒープ管理の基礎知識があると、Worker のパフォーマンスチューニングがしやすくなる。

- **マルチスレッドプログラミングの概念**
  Worker によるメッセージパッシング、Shared Worker での状態共有、SharedArrayBuffer での同期処理を理解するには、スレッド間通信、競合状態（Race Condition）、Atomics による排他制御の基本概念が必要である。

---

## 1. ブラウザのスレッドモデルと Web Worker の位置づけ

### 1.1 シングルスレッドの限界

ブラウザのメインスレッド（UI スレッド）は JavaScript の実行、DOM の更新、レイアウト計算、ペイント処理をすべて 1 つのスレッドで行う。このため、長時間かかる計算処理があると画面が固まり（ジャンク）、ユーザー体験が著しく低下する。

```
┌─────────────────────────────────────────────────────────────────┐
│                    メインスレッド（UI スレッド）                    │
│                                                                 │
│  ┌──────┐ ┌──────┐ ┌──────────────────────┐ ┌──────┐ ┌──────┐ │
│  │ JS   │ │Layout│ │  重い計算（3秒）       │ │Layout│ │Paint │ │
│  │実行  │ │      │ │  ← この間 UI が固まる │ │      │ │      │ │
│  └──────┘ └──────┘ └──────────────────────┘ └──────┘ └──────┘ │
│  0ms      16ms      33ms ─────────────────── 3033ms   3050ms   │
│                                                                 │
│  60fps を維持するには各フレームを 16.67ms 以内に処理する必要がある │
│  3 秒のブロッキングは約 180 フレーム分のドロップに相当             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Web Worker によるマルチスレッド化

Web Worker を導入すると、重い計算をバックグラウンドスレッドに移し、メインスレッドは UI 更新に専念できる。

```
┌───────────────────────────────────────────────────────────────────┐
│  メインスレッド                                                    │
│  ┌────┐ ┌──────┐ ┌────┐ ┌──────┐ ┌────┐ ┌──────┐ ┌────────────┐│
│  │ JS │ │Layout│ │ JS │ │Layout│ │ JS │ │Layout│ │結果受信+描画││
│  └────┘ └──────┘ └────┘ └──────┘ └────┘ └──────┘ └────────────┘│
│  0ms    16ms     33ms    50ms     67ms   83ms     ...           │
│    │                                                     ▲       │
│    │ postMessage                            postMessage  │       │
│    ▼                                                     │       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Worker スレッド                                         │     │
│  │  ┌──────────────────────────────────────────────────┐   │     │
│  │  │        重い計算（3秒）                             │   │     │
│  │  │        メインスレッドに影響を与えない               │   │     │
│  │  └──────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  メインスレッドは 60fps を維持し続ける                              │
└───────────────────────────────────────────────────────────────────┘
```

### 1.3 Worker の種類と全体像

```
                        Web Worker API
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        Dedicated       Shared         Service
         Worker         Worker          Worker
              │              │              │
        1 ページ       複数ページ      全ページ
        専用スレッド    共有スレッド    プロキシ型
              │              │              │
        重い計算        状態共有       キャッシュ
        データ加工     WebSocket共有   オフライン
        画像処理        DB 接続共有    Push 通知
                                      バックグラウンド同期

              ┌──────────────────────────┐
              │        Worklet           │
              │  Paint / Animation /     │
              │  Audio / Layout          │
              │  (レンダリングパイプライン │
              │   統合型の軽量 Worker)    │
              └──────────────────────────┘
```

---

## 2. Dedicated Worker

### 2.1 基本的な使い方

Dedicated Worker は最も基本的な Worker で、1 つのページ（正確には 1 つのスクリプトコンテキスト）に紐づく。

```javascript
// ===== main.js =====

// Worker の生成
// worker.js はメインスクリプトとは別のファイルとして用意する
const worker = new Worker('worker.js');

// Worker にデータを送信
worker.postMessage({
  type: 'sort',
  data: generateLargeArray(1_000_000)
});

// Worker からの結果を受信
worker.onmessage = (event) => {
  const { type, result, duration } = event.data;
  console.log(`[${type}] 完了: ${duration}ms`);
  renderResult(result);
};

// Worker 内でのエラーをキャッチ
worker.onerror = (error) => {
  console.error('Worker error:', error.message);
  console.error('ファイル:', error.filename);
  console.error('行番号:', error.lineno);
};

// Worker が不要になったら終了（リソース解放）
// worker.terminate();


// ===== worker.js =====

// Worker 側のグローバルスコープは `self`（= DedicatedWorkerGlobalScope）
// `window` や `document` は存在しない

self.onmessage = (event) => {
  const { type, data } = event.data;
  const start = performance.now();

  switch (type) {
    case 'sort': {
      const sorted = data.sort((a, b) => a - b);
      const duration = Math.round(performance.now() - start);
      self.postMessage({ type, result: sorted, duration });
      break;
    }
    case 'filter': {
      const filtered = data.filter(x => x > 0);
      const duration = Math.round(performance.now() - start);
      self.postMessage({ type, result: filtered, duration });
      break;
    }
    default:
      self.postMessage({ type: 'error', message: `未知のタイプ: ${type}` });
  }
};

// Worker 内からの自発的な終了
// self.close();
```

### 2.2 Worker でアクセス可能な API

Worker スレッドはメインスレッドとは異なるグローバルスコープを持つ。DOM には一切アクセスできないが、ネットワーク通信やストレージの一部は利用できる。

| カテゴリ | API | 利用可否 |
|----------|-----|----------|
| DOM | document, window, HTMLElement | 不可 |
| ネットワーク | fetch, XMLHttpRequest | 可 |
| WebSocket | WebSocket | 可 |
| タイマー | setTimeout, setInterval | 可 |
| ストレージ | IndexedDB | 可 |
| ストレージ | localStorage, sessionStorage | 不可 |
| URL | URL, URLSearchParams | 可 |
| 暗号 | crypto.subtle (Web Crypto API) | 可 |
| パフォーマンス | performance.now(), performance.mark() | 可 |
| コンソール | console.log() 等 | 可 |
| モジュール | importScripts() | 可 |
| エンコード | TextEncoder, TextDecoder | 可 |
| 画像処理 | createImageBitmap, OffscreenCanvas | 可 |
| 通知 | Notification (一部ブラウザ) | 制限あり |

### 2.3 Module Worker

従来の Worker は `importScripts()` でスクリプトを読み込んでいたが、ES Modules に対応した Module Worker を使うと `import`/`export` 構文が利用できる。

```javascript
// ===== main.js =====

// type: 'module' を指定すると ES Module として読み込まれる
const worker = new Worker('worker.js', { type: 'module' });

worker.postMessage({ numbers: [5, 3, 8, 1, 9, 2, 7] });

worker.onmessage = (event) => {
  console.log('統計結果:', event.data);
};


// ===== worker.js (Module Worker) =====

// ES Module の import が使える
import { mean, median, standardDeviation } from './statistics.js';

self.onmessage = (event) => {
  const { numbers } = event.data;

  const result = {
    mean: mean(numbers),
    median: median(numbers),
    stdDev: standardDeviation(numbers),
    count: numbers.length
  };

  self.postMessage(result);
};


// ===== statistics.js =====

export function mean(arr) {
  return arr.reduce((sum, v) => sum + v, 0) / arr.length;
}

export function median(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function standardDeviation(arr) {
  const avg = mean(arr);
  const squareDiffs = arr.map(v => (v - avg) ** 2);
  return Math.sqrt(mean(squareDiffs));
}
```

### 2.4 Inline Worker（Blob URL パターン）

Worker ファイルを別途用意せず、メインスクリプト内にインラインで定義するテクニック。バンドラーとの統合や単一ファイルで完結させたい場合に有用。

```javascript
// Worker のコードを文字列として定義
function createInlineWorker(workerFunction) {
  const blob = new Blob(
    [`(${workerFunction.toString()})()`],
    { type: 'application/javascript' }
  );
  const url = URL.createObjectURL(blob);
  const worker = new Worker(url);

  // Blob URL はすぐに解放可能（Worker は既にロード済み）
  URL.revokeObjectURL(url);

  return worker;
}

// 使用例
const worker = createInlineWorker(function() {
  self.onmessage = (event) => {
    const { data } = event;
    // フィボナッチ数列の計算（再帰ではなく反復で実装）
    function fibonacci(n) {
      if (n <= 1) return n;
      let prev = 0, curr = 1;
      for (let i = 2; i <= n; i++) {
        [prev, curr] = [curr, prev + curr];
      }
      return curr;
    }
    self.postMessage({
      input: data.n,
      result: fibonacci(data.n)
    });
  };
});

worker.postMessage({ n: 45 });
worker.onmessage = (e) => console.log(e.data);
// { input: 45, result: 1134903170 }
```

### 2.5 Worker プールパターン

Worker の生成にはコスト（数 ms〜数十 ms）がかかる。頻繁にタスクを投げる場合は、あらかじめ複数の Worker を生成してプールし、タスクをラウンドロビンやキュー方式で分配するパターンが有効である。

```javascript
// ===== WorkerPool.js =====

class WorkerPool {
  constructor(workerScript, poolSize = navigator.hardwareConcurrency || 4) {
    this.poolSize = poolSize;
    this.workers = [];
    this.taskQueue = [];
    this.workerStatus = [];  // true = idle, false = busy

    // Worker をプールサイズ分生成
    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(workerScript, { type: 'module' });
      this.workers.push(worker);
      this.workerStatus.push(true);  // 初期状態は idle
    }
  }

  // タスクを投入し、Promise で結果を返す
  exec(data) {
    return new Promise((resolve, reject) => {
      const task = { data, resolve, reject };

      // 空いている Worker を探す
      const idleIndex = this.workerStatus.indexOf(true);
      if (idleIndex !== -1) {
        this._runTask(idleIndex, task);
      } else {
        // 全 Worker がビジーならキューに入れる
        this.taskQueue.push(task);
      }
    });
  }

  _runTask(workerIndex, task) {
    this.workerStatus[workerIndex] = false;  // busy にする
    const worker = this.workers[workerIndex];

    const onMessage = (event) => {
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      this.workerStatus[workerIndex] = true;  // idle に戻す

      task.resolve(event.data);
      this._processQueue();  // キューに待ちタスクがあれば処理
    };

    const onError = (error) => {
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      this.workerStatus[workerIndex] = true;

      task.reject(error);
      this._processQueue();
    };

    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.postMessage(task.data);
  }

  _processQueue() {
    if (this.taskQueue.length === 0) return;
    const idleIndex = this.workerStatus.indexOf(true);
    if (idleIndex === -1) return;
    const task = this.taskQueue.shift();
    this._runTask(idleIndex, task);
  }

  // 全 Worker を終了
  terminate() {
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    this.workerStatus = [];
    // キュー内の未処理タスクを reject
    this.taskQueue.forEach(task =>
      task.reject(new Error('WorkerPool terminated'))
    );
    this.taskQueue = [];
  }

  // プールの状態を取得
  get stats() {
    return {
      poolSize: this.poolSize,
      idle: this.workerStatus.filter(s => s).length,
      busy: this.workerStatus.filter(s => !s).length,
      queued: this.taskQueue.length
    };
  }
}


// ===== 使用例 =====

const pool = new WorkerPool('compute-worker.js', 4);

// 100 個のタスクを並列実行（最大 4 並列）
const tasks = Array.from({ length: 100 }, (_, i) => ({
  id: i,
  type: 'heavyComputation',
  payload: generateData(i)
}));

const results = await Promise.all(
  tasks.map(task => pool.exec(task))
);

console.log(`全 ${results.length} タスク完了`);
console.log('プール状態:', pool.stats);

// 使い終わったらリソースを解放
pool.terminate();
```

```
Worker プールの動作イメージ:

  タスクキュー          Worker プール（サイズ = 4）
  ┌─────────┐
  │ Task 8   │     ┌────────────────────────────────┐
  │ Task 7   │     │ Worker 0: [Task 1] ■■■■□□□□    │
  │ Task 6   │     │ Worker 1: [Task 2] ■■■□□□□□    │
  │ Task 5   │     │ Worker 2: [Task 3] ■■■■■■□□    │
  │          │────→│ Worker 3: [Task 4] ■■□□□□□□    │
  └─────────┘     └────────────────────────────────┘
                          │
                          ▼ タスク完了時
                    次のタスクをキューからデキュー

  ■ = 処理中の進捗
  □ = 残り処理

  Worker 3 が Task 4 を完了
    → Worker 3 が idle になる
    → キューから Task 5 をデキュー
    → Worker 3 が Task 5 の処理を開始
```

---

## 3. メッセージパッシングの詳細

### 3.1 Structured Clone アルゴリズム

`postMessage()` で送信されるデータは、デフォルトで Structured Clone アルゴリズムによってディープコピーされる。JSON.parse(JSON.stringify()) よりも多くの型をサポートする。

| データ型 | Structured Clone | JSON | 備考 |
|----------|:----------------:|:----:|------|
| プリミティブ (string, number, boolean) | 可 | 可 | |
| null, undefined | 可 | null のみ | undefined は JSON で消える |
| Date | 可 | 文字列化 | JSON は Date を文字列にする |
| RegExp | 可 | 空オブジェクト | JSON は RegExp を {} にする |
| Map, Set | 可 | 不可 | JSON 未対応 |
| ArrayBuffer, TypedArray | 可 | 不可 | バイナリデータ |
| Blob, File | 可 | 不可 | |
| ImageData, ImageBitmap | 可 | 不可 | |
| Error | 可 | 不可 | name と message のみ |
| Function | 不可 | 不可 | 関数はクローン不可 |
| DOM ノード | 不可 | 不可 | |
| Symbol | 不可 | 不可 | |
| WeakMap, WeakRef | 不可 | 不可 | 弱参照は移行不可 |
| クラスインスタンス | プロパティのみ | プロパティのみ | プロトタイプチェーンは失われる |

### 3.2 メッセージのシリアライゼーションコスト

Structured Clone にはコピーコストが発生する。データサイズが大きいほど、コピーに要する時間は増大する。

```
Structured Clone のコスト目安（ブラウザ・環境により変動）:

  データサイズ       コピー時間の目安
  ─────────────────────────────────
    1 KB             < 0.01 ms
   10 KB             ~ 0.05 ms
  100 KB             ~ 0.5  ms
    1 MB             ~ 5    ms
   10 MB             ~ 50   ms
  100 MB             ~ 500  ms

  注意: これは一般的な傾向値であり、データの構造（ネスト深度、
  オブジェクト数）やブラウザエンジンによって大きく変動する。
  実際のアプリケーションではプロファイリングによる検証を推奨。
```

### 3.3 メッセージプロトコルの設計

複雑なアプリケーションでは、Worker とメインスレッド間のメッセージに一定のプロトコルを設けるとよい。

```javascript
// ===== message-protocol.js =====

// メッセージ型の定義（TypeScript 併用時は interface / type で定義）
const MessageType = {
  // メインスレッド → Worker
  REQUEST_COMPUTE: 'REQUEST_COMPUTE',
  REQUEST_CANCEL:  'REQUEST_CANCEL',

  // Worker → メインスレッド
  RESPONSE_SUCCESS: 'RESPONSE_SUCCESS',
  RESPONSE_ERROR:   'RESPONSE_ERROR',
  PROGRESS_UPDATE:  'PROGRESS_UPDATE',
};

// リクエスト ID を生成するユーティリティ
let requestIdCounter = 0;
function generateRequestId() {
  return `req_${Date.now()}_${++requestIdCounter}`;
}

// ===== main.js =====

const pendingRequests = new Map();

function sendRequest(worker, payload) {
  return new Promise((resolve, reject) => {
    const requestId = generateRequestId();

    pendingRequests.set(requestId, {
      resolve,
      reject,
      startTime: performance.now()
    });

    worker.postMessage({
      type: MessageType.REQUEST_COMPUTE,
      requestId,
      payload
    });
  });
}

worker.onmessage = (event) => {
  const { type, requestId, data, error, progress } = event.data;

  switch (type) {
    case MessageType.RESPONSE_SUCCESS: {
      const pending = pendingRequests.get(requestId);
      if (pending) {
        pendingRequests.delete(requestId);
        pending.resolve(data);
      }
      break;
    }
    case MessageType.RESPONSE_ERROR: {
      const pending = pendingRequests.get(requestId);
      if (pending) {
        pendingRequests.delete(requestId);
        pending.reject(new Error(error));
      }
      break;
    }
    case MessageType.PROGRESS_UPDATE: {
      console.log(`[${requestId}] 進捗: ${progress}%`);
      // UI のプログレスバーを更新するなど
      break;
    }
  }
};

// 使用例: Promise ベースで Worker を呼び出す
async function processData(rawData) {
  try {
    const result = await sendRequest(worker, {
      operation: 'analyze',
      data: rawData
    });
    console.log('結果:', result);
  } catch (err) {
    console.error('処理失敗:', err.message);
  }
}


// ===== worker.js =====

self.onmessage = (event) => {
  const { type, requestId, payload } = event.data;

  if (type === MessageType.REQUEST_COMPUTE) {
    try {
      const totalSteps = 100;
      let result = [];

      for (let i = 0; i < totalSteps; i++) {
        // 進捗を定期的に報告
        if (i % 10 === 0) {
          self.postMessage({
            type: MessageType.PROGRESS_UPDATE,
            requestId,
            progress: Math.round((i / totalSteps) * 100)
          });
        }
        // 実際の計算処理
        result.push(compute(payload.data, i));
      }

      self.postMessage({
        type: MessageType.RESPONSE_SUCCESS,
        requestId,
        data: result
      });
    } catch (err) {
      self.postMessage({
        type: MessageType.RESPONSE_ERROR,
        requestId,
        error: err.message
      });
    }
  }
};
```

---

## 4. Transferable Objects

### 4.1 所有権の移転（Transfer）

Structured Clone はデータをコピーするため、大きなバイナリデータの転送にはオーバーヘッドが大きい。Transferable Objects を使うと、データのメモリ領域の「所有権」を移転することで、ほぼゼロコストで転送できる。

```javascript
// ===== Structured Clone（コピー）と Transfer（移転）の比較 =====

// --- 方法 1: Structured Clone（デフォルト） ---
const buffer1 = new ArrayBuffer(100 * 1024 * 1024); // 100MB
console.log('送信前:', buffer1.byteLength); // 104857600

worker.postMessage(buffer1);  // コピーが発生（数十 ms）
console.log('送信後:', buffer1.byteLength); // 104857600（元のデータは残る）


// --- 方法 2: Transfer（所有権の移転） ---
const buffer2 = new ArrayBuffer(100 * 1024 * 1024); // 100MB
console.log('送信前:', buffer2.byteLength); // 104857600

worker.postMessage(buffer2, [buffer2]);  // 所有権を移転（ほぼ 0ms）
console.log('送信後:', buffer2.byteLength); // 0 （もうアクセスできない）

// 移転後に buffer2 を使おうとするとエラーにはならないが、
// byteLength は 0 になり、TypedArray のビューも空になる
```

### 4.2 Transferable な型の一覧

| 型 | 用途 | 備考 |
|----|------|------|
| ArrayBuffer | バイナリデータ全般 | 最も一般的な Transferable |
| MessagePort | Worker 間の直接通信チャネル | Channel Messaging API |
| ImageBitmap | 画像データ | createImageBitmap() で生成 |
| OffscreenCanvas | Worker 内での Canvas 描画 | 描画権限の移転 |
| ReadableStream | ストリームの所有権移転 | Streams API |
| WritableStream | ストリームの所有権移転 | Streams API |
| TransformStream | ストリームの所有権移転 | Streams API |
| VideoFrame | 動画フレーム | WebCodecs API |
| AudioData | 音声データ | WebCodecs API |

### 4.3 画像処理での Transferable Objects 活用例

```javascript
// ===== main.js: 画像のグレースケール変換を Worker に委譲 =====

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function processImage(imageElement) {
  // 画像を ImageBitmap に変換
  const bitmap = await createImageBitmap(imageElement);

  // OffscreenCanvas を使って ImageData を取得
  const offscreen = new OffscreenCanvas(bitmap.width, bitmap.height);
  const offCtx = offscreen.getContext('2d');
  offCtx.drawImage(bitmap, 0, 0);
  const imageData = offCtx.getImageData(0, 0, bitmap.width, bitmap.height);

  // ImageData の内部バッファを Transfer で Worker に送信
  // imageData.data は Uint8ClampedArray、その buffer が ArrayBuffer
  const buffer = imageData.data.buffer;

  worker.postMessage(
    {
      type: 'grayscale',
      width: imageData.width,
      height: imageData.height,
      buffer: buffer
    },
    [buffer]  // buffer を Transfer
  );
}

worker.onmessage = (event) => {
  const { width, height, buffer } = event.data;
  const resultData = new ImageData(
    new Uint8ClampedArray(buffer),
    width,
    height
  );
  ctx.putImageData(resultData, 0, 0);
};


// ===== worker.js =====

self.onmessage = (event) => {
  const { type, width, height, buffer } = event.data;

  if (type === 'grayscale') {
    const pixels = new Uint8ClampedArray(buffer);

    // グレースケール変換
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      // 輝度の加重平均（ITU-R BT.709）
      const gray = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
      pixels[i] = gray;
      pixels[i + 1] = gray;
      pixels[i + 2] = gray;
      // pixels[i + 3] はアルファ値（変更しない）
    }

    // 処理済みバッファを Transfer で返送
    self.postMessage(
      { width, height, buffer: pixels.buffer },
      [pixels.buffer]
    );
  }
};
```

### 4.4 SharedArrayBuffer と Atomics

SharedArrayBuffer は Transfer でも Clone でもなく、複数のスレッドで同一のメモリ領域を共有する仕組みである。共有メモリアクセスにはレースコンディションのリスクがあるため、Atomics API でスレッドセーフな操作を行う。

```javascript
// ===== main.js =====

// Cross-Origin Isolation が必要（HTTP ヘッダーの設定）
// Cross-Origin-Opener-Policy: same-origin
// Cross-Origin-Embedder-Policy: require-corp

// 共有メモリの確保
const sharedBuffer = new SharedArrayBuffer(4 * 1024); // 4KB
const sharedArray = new Int32Array(sharedBuffer);

// 初期値の設定
sharedArray[0] = 0; // カウンター

// Worker に共有メモリを渡す（Transfer ではなく共有）
worker.postMessage({ sharedBuffer });

// メインスレッドでカウンターを安全にインクリメント
function incrementCounter() {
  const oldValue = Atomics.add(sharedArray, 0, 1);
  console.log(`カウンター: ${oldValue} → ${oldValue + 1}`);
}

// カウンターの値を安全に読み取り
function readCounter() {
  return Atomics.load(sharedArray, 0);
}


// ===== worker.js =====

self.onmessage = (event) => {
  const { sharedBuffer } = event.data;
  const sharedArray = new Int32Array(sharedBuffer);

  // Worker 側でもカウンターを安全にインクリメント
  for (let i = 0; i < 1000; i++) {
    Atomics.add(sharedArray, 0, 1);
  }

  // Atomics.wait / Atomics.notify による同期
  // Worker スレッドで値が変わるのを待つ
  // （メインスレッドでは Atomics.wait は使用不可）
  const result = Atomics.wait(sharedArray, 1, 0);
  // result: 'ok' | 'not-equal' | 'timed-out'

  self.postMessage({ done: true, counter: Atomics.load(sharedArray, 0) });
};
```

```
SharedArrayBuffer vs Transferable Objects の使い分け:

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   Structured Clone（デフォルト）                              │
  │   ┌─────────┐    コピー    ┌─────────┐                      │
  │   │ Main    │ ──────────→ │ Worker  │   両方のスレッドで      │
  │   │ [ABCDE] │             │ [ABCDE] │   独立したコピーを持つ  │
  │   └─────────┘             └─────────┘                      │
  │                                                             │
  │   Transfer（所有権移転）                                      │
  │   ┌─────────┐    移転    ┌─────────┐                       │
  │   │ Main    │ ────────→ │ Worker  │   送信元はアクセス不可    │
  │   │ [     ] │            │ [ABCDE] │   高速（ゼロコピー）     │
  │   └─────────┘            └─────────┘                       │
  │                                                             │
  │   SharedArrayBuffer（共有メモリ）                              │
  │   ┌─────────┐            ┌─────────┐                       │
  │   │ Main    │            │ Worker  │   同一メモリ領域を参照    │
  │   │    ↓    │            │    ↓    │   Atomics で同期が必要   │
  │   │  ┌─────────────────────────┐   │                       │
  │   │  │      [ABCDE]            │   │                       │
  │   │  │   共有メモリ領域         │   │                       │
  │   │  └─────────────────────────┘   │                       │
  │   └─────────┘            └─────────┘                       │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

---

## 5. Shared Worker

### 5.1 基本概念

Shared Worker は同一オリジンの複数のページ（タブやフレーム）から接続できる Worker である。Dedicated Worker とは異なり、`onconnect` イベントで接続を管理し、`MessagePort` を介して通信する。

```javascript
// ===== main.js（各ページに配置） =====

// Shared Worker の生成
// 同じ URL を指定すると、既存の Shared Worker に接続される
const sharedWorker = new SharedWorker('shared-worker.js');

// Shared Worker との通信は port を介して行う
const port = sharedWorker.port;

// ポートを開始（onmessage を使う場合は自動で開始される）
// port.start();  // addEventListener を使う場合は明示的に呼ぶ

// メッセージの送信
port.postMessage({
  type: 'increment',
  tabId: crypto.randomUUID()
});

// メッセージの受信
port.onmessage = (event) => {
  const { type, count, connections } = event.data;
  console.log(`カウント: ${count}, 接続タブ数: ${connections}`);
  document.getElementById('counter').textContent = count;
  document.getElementById('tabs').textContent = connections;
};


// ===== shared-worker.js =====

// 接続中のポート一覧
const ports = new Set();
let counter = 0;

self.onconnect = (event) => {
  const port = event.ports[0];
  ports.add(port);

  console.log(`新しい接続。現在の接続数: ${ports.size}`);

  port.onmessage = (msgEvent) => {
    const { type, tabId } = msgEvent.data;

    switch (type) {
      case 'increment':
        counter++;
        // 全接続先に通知（ブロードキャスト）
        broadcastToAll({
          type: 'update',
          count: counter,
          connections: ports.size
        });
        break;

      case 'getState':
        // リクエスト元のみに返信
        port.postMessage({
          type: 'state',
          count: counter,
          connections: ports.size
        });
        break;
    }
  };

  // 接続が閉じられたときのクリーンアップ
  port.addEventListener('close', () => {
    ports.delete(port);
    console.log(`切断。残接続数: ${ports.size}`);
  });

  // 接続時に現在の状態を送信
  port.postMessage({
    type: 'state',
    count: counter,
    connections: ports.size
  });

  port.start();
};

function broadcastToAll(message) {
  for (const port of ports) {
    try {
      port.postMessage(message);
    } catch (e) {
      // ポートが閉じている場合は削除
      ports.delete(port);
    }
  }
}
```

### 5.2 Shared Worker の主なユースケース

```
  Shared Worker の代表的なユースケース:

  1. WebSocket 接続の共有
  ┌────────┐     ┌────────┐     ┌────────┐
  │ Tab A  │     │ Tab B  │     │ Tab C  │
  └───┬────┘     └───┬────┘     └───┬────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
           ┌─────────┴─────────┐
           │   Shared Worker   │
           │  ┌─────────────┐  │
           │  │ WebSocket   │  │  ← 接続は 1 本のみ
           │  │ connection  │  │    サーバー負荷を軽減
           │  └──────┬──────┘  │
           └─────────┼─────────┘
                     │
              ┌──────┴──────┐
              │   Server    │
              └─────────────┘

  2. 共有キャッシュ / 状態管理
     複数タブで同じデータを共有し、重複フェッチを防止

  3. ロギング / 分析の集約
     各タブのイベントを Shared Worker で集約してバッチ送信
```

---

## 6. Service Worker

### 6.1 ライフサイクル

Service Worker はページとネットワークの間に立つプログラマブルなプロキシである。通常の Worker とは大きく異なるライフサイクルを持つ。

```
Service Worker のライフサイクル:

  ┌──────────┐
  │ 未登録    │  navigator.serviceWorker.register('/sw.js')
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │ 登録中    │  ブラウザが sw.js をダウンロード・パース
  │(installing)│
  └────┬─────┘
       │  install イベント発火
       │  event.waitUntil() で非同期処理を待機可能
       ▼
  ┌──────────┐    既に古い SW がアクティブな場合
  │ 待機中    │───────────────────────────────────┐
  │(waiting)  │  古い SW が制御する全タブが閉じるまで │
  └────┬─────┘  待機する（skipWaiting() で回避可能） │
       │                                           │
       │  全クライアントが解放される                  │
       │  または self.skipWaiting() が呼ばれる        │
       ▼                                           │
  ┌──────────┐                                     │
  │ 有効化中  │  activate イベント発火                │
  │(activating)│  古いキャッシュの削除に最適          │
  └────┬─────┘                                     │
       │                                           │
       ▼                                           │
  ┌──────────┐                                     │
  │ 有効     │  fetch, push, sync 等のイベントを     │
  │(activated)│  インターセプト可能                  │
  └────┬─────┘                                     │
       │                                           │
       │  新しいバージョンの sw.js が検出される       │
       ▼                                           │
  ┌──────────┐                                     │
  │ 冗長     │  新しい SW に置き換えられた           │
  │(redundant)│  または登録/インストール失敗         │
  └──────────┘  ← ────────────────────────────────┘
```

### 6.2 登録とインストール

```javascript
// ===== app.js（メインスクリプト） =====

// Service Worker の登録
async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    console.log('Service Worker 未対応のブラウザ');
    return;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      // スコープ: この SW が制御するパスのプレフィックス
      scope: '/'
    });

    console.log('SW 登録成功:', registration.scope);

    // 更新の検出
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      console.log('新しい SW を検出:', newWorker.state);

      newWorker.addEventListener('statechange', () => {
        console.log('SW 状態変更:', newWorker.state);
        if (newWorker.state === 'activated') {
          // 新しい SW がアクティブになった
          // ユーザーにリロードを促す UI を表示するなど
          showUpdateNotification();
        }
      });
    });
  } catch (error) {
    console.error('SW 登録失敗:', error);
  }
}

// ページ読み込み後に登録
window.addEventListener('load', registerServiceWorker);
```

### 6.3 キャッシュ戦略の実装

Service Worker の最も重要な機能はネットワークリクエストのインターセプトとキャッシュ制御である。代表的なキャッシュ戦略を実装する。

```javascript
// ===== sw.js =====

const CACHE_VERSION = 'v2';
const STATIC_CACHE = `static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `dynamic-${CACHE_VERSION}`;

// プリキャッシュする静的アセット
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/styles/main.css',
  '/scripts/app.js',
  '/images/logo.png',
  '/offline.html'
];

// ===== Install: 静的アセットのプリキャッシュ =====
self.addEventListener('install', (event) => {
  console.log('[SW] Install');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('[SW] プリキャッシュ開始');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        // 待機状態をスキップ（即座にアクティブ化）
        return self.skipWaiting();
      })
  );
});

// ===== Activate: 古いキャッシュの削除 =====
self.addEventListener('activate', (event) => {
  console.log('[SW] Activate');
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(name => name !== STATIC_CACHE && name !== DYNAMIC_CACHE)
            .map(name => {
              console.log(`[SW] 古いキャッシュ削除: ${name}`);
              return caches.delete(name);
            })
        );
      })
      .then(() => {
        // 全クライアントを即座に制御下に置く
        return self.clients.claim();
      })
  );
});

// ===== Fetch: リクエストのインターセプト =====
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // API リクエストには Network First 戦略
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }

  // 静的アセットには Cache First 戦略
  if (STATIC_ASSETS.includes(url.pathname)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // その他のリクエストには Stale While Revalidate 戦略
  event.respondWith(staleWhileRevalidate(request));
});


// ===== キャッシュ戦略の実装 =====

// Cache First: キャッシュ優先、なければネットワーク
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    return caches.match('/offline.html');
  }
}

// Network First: ネットワーク優先、失敗時キャッシュ
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    return cached || new Response(
      JSON.stringify({ error: 'オフラインです' }),
      { headers: { 'Content-Type': 'application/json' } }
    );
  }
}

// Stale While Revalidate: キャッシュを即時返却しつつバックグラウンドで更新
async function staleWhileRevalidate(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request)
    .then(response => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => cached);

  // キャッシュがあれば即時返却、なければネットワークを待つ
  return cached || fetchPromise;
}
```

### 6.4 キャッシュ戦略の比較表

| 戦略 | 動作 | 最適な用途 | オフライン対応 | 鮮度 |
|------|------|-----------|:-------------:|------|
| Cache First | キャッシュ優先、なければネットワーク | 静的アセット (CSS, JS, 画像) | 高 | 低（手動更新が必要） |
| Network First | ネットワーク優先、失敗時キャッシュ | API レスポンス、頻繁に更新されるデータ | 中 | 高 |
| Stale While Revalidate | キャッシュを即時返却 + バックグラウンド更新 | ニュースフィード、SNS タイムライン | 中 | 中（次回アクセスで反映） |
| Network Only | 常にネットワーク | 非冪等リクエスト (POST)、リアルタイムデータ | 不可 | 最高 |
| Cache Only | 常にキャッシュ | プリキャッシュ済み静的リソース | 最高 | なし（ビルド時に固定） |

### 6.5 Background Sync

Service Worker のバックグラウンド同期機能を使うと、オフライン時の操作を保存しておき、ネットワーク復帰時に自動的にサーバーへ送信できる。

```javascript
// ===== app.js（メインスクリプト） =====

async function sendMessage(message) {
  // IndexedDB にメッセージを保存
  await saveToOutbox(message);

  // Background Sync の登録
  const registration = await navigator.serviceWorker.ready;
  try {
    await registration.sync.register('outbox-sync');
    console.log('Background Sync 登録完了');
  } catch (err) {
    console.error('Background Sync 未対応:', err);
    // フォールバック: 即座に送信を試みる
    await sendPendingMessages();
  }
}


// ===== sw.js =====

// sync イベントはネットワーク復帰時に自動発火
self.addEventListener('sync', (event) => {
  if (event.tag === 'outbox-sync') {
    event.waitUntil(sendPendingMessages());
  }
});

async function sendPendingMessages() {
  const messages = await getFromOutbox(); // IndexedDB から取得

  const results = await Promise.allSettled(
    messages.map(async (msg) => {
      const response = await fetch('/api/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(msg)
      });

      if (response.ok) {
        await removeFromOutbox(msg.id); // 送信成功したら削除
        return { id: msg.id, status: 'sent' };
      }
      throw new Error(`送信失敗: ${response.status}`);
    })
  );

  const failed = results.filter(r => r.status === 'rejected');
  if (failed.length > 0) {
    throw new Error(`${failed.length} 件の送信に失敗`);
    // エラーを throw すると、ブラウザは後で再試行する
  }
}
```

### 6.6 Push 通知

Service Worker は Push API と連携し、サーバーからのプッシュ通知を受信できる。ブラウザが閉じていても（バックグラウンドで）通知を表示可能。

```javascript
// ===== app.js =====

async function subscribeToPush() {
  const registration = await navigator.serviceWorker.ready;

  // 通知の許可を取得
  const permission = await Notification.requestPermission();
  if (permission !== 'granted') {
    console.log('通知が許可されていません');
    return;
  }

  // Push サブスクリプションの作成
  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true, // 可視通知のみ（Chrome の要件）
    applicationServerKey: urlBase64ToUint8Array(VAPID_PUBLIC_KEY)
  });

  // サーバーにサブスクリプション情報を送信
  await fetch('/api/push-subscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(subscription)
  });
}

function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  const rawData = atob(base64);
  return Uint8Array.from([...rawData].map(c => c.charCodeAt(0)));
}


// ===== sw.js =====

self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};

  const options = {
    body: data.body || 'お知らせがあります',
    icon: '/images/notification-icon.png',
    badge: '/images/badge.png',
    vibrate: [200, 100, 200],
    data: {
      url: data.url || '/',
      timestamp: Date.now()
    },
    actions: [
      { action: 'open', title: '開く' },
      { action: 'dismiss', title: '閉じる' }
    ]
  };

  event.waitUntil(
    self.registration.showNotification(data.title || '通知', options)
  );
});

// 通知のクリックハンドリング
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'dismiss') {
    return;
  }

  const targetUrl = event.notification.data.url;

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then(clientList => {
        // 既に開いているタブがあればフォーカス
        for (const client of clientList) {
          if (client.url === targetUrl && 'focus' in client) {
            return client.focus();
          }
        }
        // なければ新しいタブで開く
        return clients.openWindow(targetUrl);
      })
  );
});
```

---

## 7. Worker 間通信と Channel Messaging

### 7.1 MessageChannel による直接通信

通常、Worker はメインスレッドを介してしか通信できない。しかし MessageChannel を使うと、2 つの Worker 間で直接通信するチャネルを作成できる。

```javascript
// ===== main.js =====

const workerA = new Worker('workerA.js');
const workerB = new Worker('workerB.js');

// MessageChannel を作成
const channel = new MessageChannel();

// port1 を workerA に、port2 を workerB に Transfer
workerA.postMessage({ type: 'setPort', port: channel.port1 }, [channel.port1]);
workerB.postMessage({ type: 'setPort', port: channel.port2 }, [channel.port2]);

// これ以降、workerA と workerB はメインスレッドを介さず直接通信可能
// メインスレッドはボトルネックにならない


// ===== workerA.js =====

let directPort = null;

self.onmessage = (event) => {
  if (event.data.type === 'setPort') {
    directPort = event.data.port;
    directPort.onmessage = (e) => {
      console.log('[WorkerA] WorkerB からの直接メッセージ:', e.data);
    };
    // WorkerB に直接メッセージを送信
    directPort.postMessage({ from: 'A', message: '直接通信テスト' });
  }
};


// ===== workerB.js =====

let directPort = null;

self.onmessage = (event) => {
  if (event.data.type === 'setPort') {
    directPort = event.data.port;
    directPort.onmessage = (e) => {
      console.log('[WorkerB] WorkerA からの直接メッセージ:', e.data);
      // 返信
      directPort.postMessage({
        from: 'B',
        message: '了解、直接通信成功'
      });
    };
  }
};
```

```
MessageChannel による Worker 間直接通信:

  通常の通信（メインスレッド経由）:
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Worker A │ →  │  Main    │ →  │ Worker B │
  │          │ ←  │ Thread   │ ←  │          │
  └──────────┘    └──────────┘    └──────────┘
  メインスレッドがボトルネックになる可能性

  MessageChannel（直接通信）:
  ┌──────────┐                    ┌──────────┐
  │ Worker A │ ←── port1──port2 ──→ │ Worker B │
  │          │  MessageChannel     │          │
  └──────────┘                    └──────────┘
  メインスレッドを経由しない高速な通信
```

### 7.2 BroadcastChannel による多対多通信

BroadcastChannel は同一オリジンの全コンテキスト（ページ、Worker、Service Worker）にメッセージをブロードキャストする仕組みである。

```javascript
// ===== 任意のコンテキスト（ページでも Worker でも可） =====

// 同じチャネル名を指定すると自動的に接続される
const channel = new BroadcastChannel('app-events');

// メッセージの送信（全リスナーに配信される）
channel.postMessage({
  type: 'user-login',
  userId: 'user123',
  timestamp: Date.now()
});

// メッセージの受信
channel.onmessage = (event) => {
  const { type, userId } = event.data;
  if (type === 'user-login') {
    console.log(`ユーザー ${userId} がログインしました`);
    updateUI();
  }
};

// 不要になったら閉じる
// channel.close();
```

```
BroadcastChannel の通信モデル:

  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
  │  Tab A   │  │  Tab B   │  │ Worker   │  │ Service Worker│
  │          │  │          │  │          │  │              │
  │ channel  │  │ channel  │  │ channel  │  │ channel      │
  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                           │
                BroadcastChannel('app-events')
                           │
            送信者以外の全リスナーが受信
```

---

## 8. Worklet

### 8.1 Worklet と Worker の違い

Worklet はレンダリングパイプラインに統合された軽量な Worker である。通常の Worker とは異なり、ブラウザの内部処理（描画、レイアウト、オーディオ処理）に直接介入できる。

| 特性 | Worker | Worklet |
|------|--------|---------|
| スレッドモデル | 独立したバックグラウンドスレッド | レンダリングパイプライン統合 |
| 生成コスト | 比較的高い | 軽量 |
| グローバルスコープ | DedicatedWorkerGlobalScope | 各 Worklet 固有のスコープ |
| postMessage | 可 | 不可（直接通信なし） |
| DOM アクセス | 不可 | 不可 |
| 目的 | 汎用計算のオフロード | パイプライン特化処理 |
| 実行保証 | 明示的な起動・終了 | ブラウザが必要時に実行 |
| モジュール | importScripts / ES Modules | ES Modules のみ |

### 8.2 Paint Worklet（CSS Houdini）

Paint Worklet は CSS の `background-image` をプログラマブルに描画する。Canvas API に似たインターフェースで自由な描画が可能。

```javascript
// ===== main.js =====

if ('paintWorklet' in CSS) {
  CSS.paintWorklet.addModule('paint-worklet.js');
}


// ===== paint-worklet.js =====

class GradientBorderPainter {
  // CSS カスタムプロパティへの依存宣言
  static get inputProperties() {
    return [
      '--border-width',
      '--gradient-start',
      '--gradient-end'
    ];
  }

  paint(ctx, size, properties) {
    const borderWidth = parseInt(properties.get('--border-width')) || 4;
    const startColor = properties.get('--gradient-start').toString().trim()
      || '#ff6b6b';
    const endColor = properties.get('--gradient-end').toString().trim()
      || '#4ecdc4';

    // グラデーションボーダーを描画
    const gradient = ctx.createLinearGradient(0, 0, size.width, size.height);
    gradient.addColorStop(0, startColor);
    gradient.addColorStop(1, endColor);

    ctx.strokeStyle = gradient;
    ctx.lineWidth = borderWidth;
    ctx.strokeRect(
      borderWidth / 2,
      borderWidth / 2,
      size.width - borderWidth,
      size.height - borderWidth
    );
  }
}

registerPaint('gradient-border', GradientBorderPainter);


// ===== styles.css =====
/*
.card {
  --border-width: 4;
  --gradient-start: #ff6b6b;
  --gradient-end: #4ecdc4;
  background-image: paint(gradient-border);
}
*/
```

### 8.3 Audio Worklet

Audio Worklet は Web Audio API の信号処理をリアルタイムで行う。以前の ScriptProcessorNode（メインスレッドで動作）に代わる、高パフォーマンスな代替手段である。

```javascript
// ===== main.js =====

async function setupAudioWorklet() {
  const audioContext = new AudioContext();

  // Audio Worklet モジュールの登録
  await audioContext.audioWorklet.addModule('audio-processor.js');

  // カスタム AudioWorkletNode の生成
  const gainNode = new AudioWorkletNode(audioContext, 'custom-gain');

  // パラメータの制御
  const gainParam = gainNode.parameters.get('gain');
  gainParam.value = 0.5; // 音量を半分に

  // 入力 → カスタム処理 → 出力
  const source = audioContext.createMediaStreamSource(
    await navigator.mediaDevices.getUserMedia({ audio: true })
  );
  source.connect(gainNode).connect(audioContext.destination);
}


// ===== audio-processor.js =====

class CustomGainProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [{
      name: 'gain',
      defaultValue: 1.0,
      minValue: 0.0,
      maxValue: 2.0,
      automationRate: 'a-rate' // サンプル単位で変化可能
    }];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    const gain = parameters.gain;

    for (let channel = 0; channel < input.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        // gain パラメータが a-rate の場合、サンプルごとに異なる値を持つ
        const g = gain.length > 1 ? gain[i] : gain[0];
        outputChannel[i] = inputChannel[i] * g;
      }
    }

    return true; // true を返すと処理を継続
  }
}

registerProcessor('custom-gain', CustomGainProcessor);
```

---

## 9. Worker の種類の総合比較

### 9.1 機能比較表

```
┌────────────────────┬───────────────┬───────────────┬───────────────┬────────────────┐
│                    │ Dedicated     │ Shared        │ Service       │ Worklet        │
│                    │ Worker        │ Worker        │ Worker        │                │
├────────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
│ スコープ           │ 1 ページ      │ 同一オリジン  │ 同一オリジン  │ 特定処理       │
│ 接続数             │ 1             │ 複数ページ    │ 全ページ      │ N/A            │
│ DOM アクセス       │ 不可          │ 不可          │ 不可          │ 不可           │
│ ライフサイクル     │ ページと同じ  │ 全接続終了まで│ 独立（永続）  │ ブラウザ管理   │
│ オフライン対応     │ 不可          │ 不可          │ 可            │ 不可           │
│ Push 通知          │ 不可          │ 不可          │ 可            │ 不可           │
│ ネットワーク制御   │ 不可          │ 不可          │ 可            │ 不可           │
│ fetch() 利用       │ 可            │ 可            │ 可            │ 不可           │
│ IndexedDB 利用     │ 可            │ 可            │ 可            │ 不可           │
│ postMessage        │ 可            │ 可 (port経由) │ 可            │ 不可           │
│ ES Modules         │ 可            │ 可            │ 可            │ 必須           │
│ HTTPS 必須         │ 不要          │ 不要          │ 必須          │ 不要           │
│ DevTools 対応      │ 良好          │ 制限あり      │ 良好          │ 制限あり       │
│ ブラウザ対応       │ 全モダン      │ 制限あり      │ 全モダン      │ 部分的         │
├────────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
│ 主な用途           │ 重い計算      │ 状態共有      │ キャッシュ    │ 描画拡張       │
│                    │ データ加工    │ WebSocket共有 │ PWA           │ オーディオ処理 │
│                    │ 画像/動画処理 │ DB 接続共有   │ Push / Sync   │ アニメーション │
└────────────────────┴───────────────┴───────────────┴───────────────┴────────────────┘
```

### 9.2 ユースケース別の選択指針

```
どの Worker を使うべきか？ フローチャート:

  ┌─────────────────────────────────────────┐
  │  何をしたいのか？                         │
  └───────────┬─────────────────────────────┘
              │
    ┌─────────┴──────────┐
    │                    │
  重い計算を          ネットワークを
  オフロードしたい     制御したい
    │                    │
    │                    ▼
    │              ┌──────────────┐
    │              │ Service Worker│
    │              │ キャッシュ     │
    │              │ オフライン     │
    │              │ Push 通知     │
    │              └──────────────┘
    │
    ├── 1 ページでだけ使う？
    │     │
    │     ├── Yes → Dedicated Worker
    │     │
    │     └── No → 複数タブで共有したい？
    │               │
    │               ├── Yes → Shared Worker
    │               │
    │               └── No → Dedicated Worker
    │                          (各ページに 1 つ)
    │
    └── レンダリングに関わる処理？
          │
          ├── 描画のカスタマイズ → Paint Worklet
          ├── スムーズアニメーション → Animation Worklet
          ├── リアルタイム音声処理 → Audio Worklet
          └── カスタムレイアウト → Layout Worklet (実験的)
```

---

## 10. アンチパターンと改善策

### 10.1 アンチパターン 1: Worker の過剰生成

```javascript
// ===== BAD: タスクごとに Worker を生成・破棄 =====

async function processItems(items) {
  const results = [];
  for (const item of items) {
    // 毎回 Worker を生成（数 ms のオーバーヘッド x 1000回）
    const worker = new Worker('process.js');

    const result = await new Promise((resolve) => {
      worker.onmessage = (e) => {
        resolve(e.data);
        worker.terminate();  // 毎回終了
      };
      worker.postMessage(item);
    });

    results.push(result);
  }
  return results;
}
// 問題: Worker の生成・破棄コストが大きい
// 問題: 並列実行されない（逐次処理）
// 問題: メモリリークの可能性


// ===== GOOD: Worker プールで再利用 =====

const pool = new WorkerPool('process.js', 4);

async function processItems(items) {
  // 全タスクを並列にキューイング（最大 4 並列）
  const results = await Promise.all(
    items.map(item => pool.exec(item))
  );
  return results;
}
// Worker を再利用するため生成コストは初期化時のみ
// 最大並列数を制御可能
// 明示的な terminate で確実にリソース解放
```

### 10.2 アンチパターン 2: 大量データの無駄なコピー

```javascript
// ===== BAD: 大きなバッファを毎回コピー =====

function processVideoFrame(frameBuffer) {
  // 100MB のバッファが毎フレームコピーされる
  worker.postMessage({ frame: frameBuffer });
  // frameBuffer はまだメインスレッドに残っている
  // GC されるまでメモリを二重消費
}

worker.onmessage = (event) => {
  // 結果もコピーで返される
  const processedFrame = event.data.result;
  renderFrame(processedFrame);
};

// 問題: 30fps で動画処理する場合、毎秒 6GB のメモリコピーが発生
// 問題: GC 圧力が高くなり、パフォーマンスが不安定に


// ===== GOOD: Transferable Objects で所有権を移転 =====

function processVideoFrame(frameBuffer) {
  // 所有権を Worker に移転（ゼロコピー）
  worker.postMessage(
    { frame: frameBuffer },
    [frameBuffer]  // Transfer リストに含める
  );
  // frameBuffer.byteLength === 0（もう使えない）
}

worker.onmessage = (event) => {
  // Worker からも Transfer で返送
  const processedFrame = event.data.result;
  renderFrame(processedFrame);
  // 次のフレーム処理のために再び Worker に Transfer
};

// ゼロコピーなので 30fps でも問題なし
// メモリ使用量も最小限


// ===== BETTER: SharedArrayBuffer で共有メモリ（CORS 設定が必要） =====

const frameBuffer = new SharedArrayBuffer(frameSize);
const mainView = new Uint8Array(frameBuffer);
const statusArray = new Int32Array(new SharedArrayBuffer(4));
// statusArray[0]: 0 = idle, 1 = processing, 2 = done

worker.postMessage({ frameBuffer, statusArray });

function processVideoFrame(rawFrame) {
  // 共有メモリに書き込み
  mainView.set(rawFrame);
  // Worker に処理開始を通知
  Atomics.store(statusArray, 0, 1);
  Atomics.notify(statusArray, 0);
}

// コピーもメモリ移転も発生しない
```

### 10.3 アンチパターン 3: エラーハンドリングの欠如

```javascript
// ===== BAD: エラーが無視される =====

const worker = new Worker('worker.js');
worker.postMessage(data);
worker.onmessage = (e) => {
  updateUI(e.data);
};
// Worker 内でエラーが発生しても何も起きない
// Promise が永遠に resolve されない可能性


// ===== GOOD: 包括的なエラーハンドリング =====

const worker = new Worker('worker.js');

// Worker 自体のエラー（構文エラー、未キャッチ例外）
worker.onerror = (error) => {
  console.error('[Worker Error]', error.message);
  console.error('ファイル:', error.filename, '行:', error.lineno);
  error.preventDefault(); // デフォルトのエラー報告を抑制
  showErrorUI('ワーカーで予期しないエラーが発生しました');
};

// Worker 内で messageerror（デシリアライズ失敗等）
worker.onmessageerror = (event) => {
  console.error('[Message Error] メッセージの復元に失敗');
};

// タイムアウト付きのリクエスト
function requestWithTimeout(worker, data, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Worker タイムアウト: ${timeoutMs}ms`));
    }, timeoutMs);

    const handler = (event) => {
      clearTimeout(timer);
      worker.removeEventListener('message', handler);
      if (event.data.error) {
        reject(new Error(event.data.error));
      } else {
        resolve(event.data);
      }
    };

    worker.addEventListener('message', handler);
    worker.postMessage(data);
  });
}

// 使用例
try {
  const result = await requestWithTimeout(worker, taskData, 10000);
  updateUI(result);
} catch (err) {
  console.error('処理失敗:', err.message);
  showErrorUI(err.message);
}
```

---

## 11. エッジケース分析

### 11.1 エッジケース 1: Worker の同時生成数制限

ブラウザには Worker の同時生成数に実質的な上限がある。仕様上の制限ではないが、OS スレッド数やメモリの制約から、大量の Worker を同時に生成するとパフォーマンスが低下したり、生成自体が失敗したりする。

```javascript
// ===== 問題: 大量の Worker 同時生成 =====

// 100 個の Worker を一度に生成しようとする
const workers = [];
for (let i = 0; i < 100; i++) {
  try {
    workers.push(new Worker('heavy-task.js'));
  } catch (e) {
    console.error(`Worker ${i} の生成に失敗:`, e);
    // ブラウザによっては 20〜50 個程度で制限に達する
    break;
  }
}

// 結果:
// - Chrome: 概ね動作するが、スレッド数過多でコンテキストスイッチが増大
// - Firefox: 一定数を超えるとキューイングされる
// - Safari: より早い段階で制限に達する傾向
// - 全ブラウザ: メモリ消費が急増（Worker 1 つあたり数 MB のスタック領域）


// ===== 対策: Worker プール + キューイング =====

// navigator.hardwareConcurrency で論理 CPU コア数を取得
const optimalPoolSize = Math.max(1, navigator.hardwareConcurrency - 1);
// メインスレッド用に 1 コア残すのが慣例
console.log(`最適プールサイズ: ${optimalPoolSize}`);

const pool = new WorkerPool('heavy-task.js', optimalPoolSize);
// 100 個のタスクを適切な並列度で実行
const results = await Promise.all(
  tasks.map(task => pool.exec(task))
);
```

### 11.2 エッジケース 2: Service Worker のスコープ制限

Service Worker は登録時の `scope` パラメータ（またはスクリプトのディレクトリ）によって制御範囲が決まる。この制限を理解していないと、期待通りにリクエストをインターセプトできない。

```javascript
// ===== Service Worker のスコープに関するルール =====

// 1. デフォルトスコープ = sw.js のディレクトリ
//    sw.js が /scripts/sw.js にある場合
//    → スコープは /scripts/ 以下のみ

// /sw.js → スコープ: / (ルート以下すべて)
navigator.serviceWorker.register('/sw.js');

// /scripts/sw.js → スコープ: /scripts/ 以下のみ
navigator.serviceWorker.register('/scripts/sw.js');
// /index.html へのリクエストはインターセプトされない

// 明示的にスコープを広げようとするとエラー
navigator.serviceWorker.register('/scripts/sw.js', {
  scope: '/'  // SecurityError: スクリプトのディレクトリより上位は指定不可
});

// 解決策 1: sw.js をルートに配置
navigator.serviceWorker.register('/sw.js', { scope: '/' });

// 解決策 2: Service-Worker-Allowed ヘッダーをサーバーで設定
// HTTP レスポンスヘッダー: Service-Worker-Allowed: /
// これにより /scripts/sw.js でもルートスコープを取得可能
navigator.serviceWorker.register('/scripts/sw.js', { scope: '/' });


// 2. 複数の Service Worker を異なるスコープで登録
navigator.serviceWorker.register('/sw-main.js', { scope: '/' });
navigator.serviceWorker.register('/blog/sw-blog.js', { scope: '/blog/' });
// /blog/ 以下のリクエストは sw-blog.js が優先
// それ以外のリクエストは sw-main.js が処理

// 3. Service Worker の更新判定
// ブラウザはバイト単位でスクリプトを比較し、1 バイトでも変わっていれば更新する
// 24 時間に 1 回、自動的に更新チェックが行われる
// registration.update() で手動チェックも可能
```

---

## FAQ

### Q1: Web Worker、Shared Worker、Service Worker の違いは何か？

**A:** 3 種類の Worker は用途と寿命が異なる。

| Worker の種類 | 用途 | 寿命 | 共有範囲 |
|--------------|------|------|----------|
| **Dedicated Worker** | 単一ページでの並列計算 | ページが閉じるまで | 生成元のページのみ |
| **Shared Worker** | 複数タブ間の状態共有 | 全タブが閉じるまで | 同一オリジンの全タブ |
| **Service Worker** | オフライン対応・プッシュ通知 | ブラウザが管理（idle 時に停止） | 同一スコープの全ページ |

```javascript
// Dedicated Worker: 画像処理など、単一ページの重い計算に使用
const worker = new Worker('image-processor.js');
worker.postMessage(imageData);

// Shared Worker: WebSocket 接続を複数タブで共有
const sharedWorker = new SharedWorker('websocket-manager.js');
sharedWorker.port.start();
sharedWorker.port.postMessage({ type: 'subscribe', channel: 'chat' });

// Service Worker: API レスポンスをキャッシュしてオフライン対応
navigator.serviceWorker.register('/sw.js').then(registration => {
  console.log('Service Worker registered with scope:', registration.scope);
});
```

**選択基準:**
- **計算処理のみ** → Dedicated Worker
- **タブ間でリアルタイム同期** → Shared Worker
- **ネットワークリクエストの制御** → Service Worker

---

### Q2: Worker で大きなデータを転送する際の Transferable Objects とは何か？

**A:** Transferable Objects は、データをコピーせずに所有権を転送する仕組みである。大きな ArrayBuffer を Worker とやり取りする際、構造化複製アルゴリズムによる深いコピーは数百 ms かかることがあるが、転送なら 1ms 未満で完了する。

```javascript
// ===== 通常のコピー（遅い）=====
const largeBuffer = new ArrayBuffer(100 * 1024 * 1024); // 100MB
console.time('Copy');
worker.postMessage({ buffer: largeBuffer }); // 深いコピー発生（数百 ms）
console.timeEnd('Copy');
// メインスレッドでも largeBuffer は引き続き使用可能

// ===== Transferable Objects（速い）=====
const largeBuffer2 = new ArrayBuffer(100 * 1024 * 1024);
console.time('Transfer');
worker.postMessage(
  { buffer: largeBuffer2 },
  [largeBuffer2] // 第2引数で転送対象を指定
);
console.timeEnd('Transfer'); // 1ms 未満
// この後、メインスレッドで largeBuffer2 にアクセスすると TypeError
// console.log(largeBuffer2.byteLength); // Error: Detached ArrayBuffer
```

**転送可能なオブジェクト:**
- `ArrayBuffer`
- `MessagePort`
- `ImageBitmap`
- `OffscreenCanvas`
- `ReadableStream`、`WritableStream`、`TransformStream`

**注意点:**
- 転送後、元のスレッドからはアクセス不可（Detached 状態）
- 双方向転送が必要な場合、Worker から返すときも転送を指定する

```javascript
// Worker 側で処理後に返送
self.onmessage = (e) => {
  const buffer = e.data.buffer;
  // 処理実行
  processBuffer(buffer);

  // 処理済みバッファを転送で返す
  self.postMessage({ result: buffer }, [buffer]);
};
```

---

### Q3: Worker のデバッグ方法は？

**A:** Chrome DevTools と Firefox Developer Tools は Worker 専用のデバッグ機能を提供している。

**Chrome DevTools での Worker デバッグ:**

1. **Worker の一覧表示**
   `Sources` タブ → 左ペインの `Threads` セクション → 起動中の Worker が表示される

2. **ブレークポイント設定**
   Worker のスクリプトを開き、通常の JavaScript と同様にブレークポイントを設置

3. **コンソールへのアクセス**
   `Console` タブで `top` のドロップダウン → Worker を選択 → Worker のコンテキストで `console.log` を確認

4. **postMessage の追跡**
   `Performance` タブで記録 → `Main` と `Worker` のタイムラインを並べて確認 → メッセージのやり取りを可視化

**Firefox での Worker デバッグ:**

1. `about:debugging` → `This Firefox` → 対象の Worker を確認
2. `Inspect` ボタンで専用の DevTools を起動
3. `Console`、`Debugger`、`Network` タブで通常通りデバッグ

**ログ出力のベストプラクティス:**

```javascript
// Worker 側でログにタイムスタンプと Worker ID を付ける
const workerId = Math.random().toString(36).slice(2, 9);

self.onmessage = (e) => {
  console.log(`[Worker ${workerId}] ${Date.now()} - Received:`, e.data);
  const result = heavyComputation(e.data);
  console.log(`[Worker ${workerId}] ${Date.now()} - Sending result:`, result);
  self.postMessage(result);
};

// メインスレッド側でも対応するログ
worker.postMessage(data);
console.log(`[Main] ${Date.now()} - Sent to worker:`, data);

worker.onmessage = (e) => {
  console.log(`[Main] ${Date.now()} - Received from worker:`, e.data);
};
```

**Service Worker のデバッグ:**

- Chrome: `chrome://serviceworker-internals/` → 登録済み SW の一覧、強制更新、Unregister
- Firefox: `about:debugging` → `Service Workers` セクション
- Application タブ → Service Workers → `Update on reload` をチェックして開発中の再読み込みを容易化

---

## まとめ

### Web Worker の特性比較

| 項目 | Dedicated Worker | Shared Worker | Service Worker |
|------|------------------|---------------|----------------|
| **生成方法** | `new Worker(url)` | `new SharedWorker(url)` | `navigator.serviceWorker.register(url)` |
| **通信手段** | `postMessage` / `onmessage` | `port.postMessage` / `port.onmessage` | `postMessage` + `fetch` イベント |
| **DOM アクセス** | 不可 | 不可 | 不可 |
| **複数タブ共有** | 不可 | 可能 | 可能 |
| **永続性** | ページが閉じると終了 | 全タブが閉じると終了 | ブラウザが自動管理（idle で停止） |
| **主な用途** | 画像処理、暗号化、大量計算 | WebSocket 共有、状態同期 | オフライン対応、プッシュ通知 |

### キーポイント

1. **メインスレッドのブロッキングを避ける**
   16.67ms（60fps）を超える処理は Worker にオフロードする。構造化複製のコストを考慮し、大きなデータには Transferable Objects を使用する。

2. **Worker プールで並列度を制御する**
   `navigator.hardwareConcurrency` で CPU コア数を取得し、適切なプールサイズを決定する。無制限に Worker を生成すると、コンテキストスイッチとメモリ消費でパフォーマンスが悪化する。

3. **Service Worker はネットワーク層を制御する特殊な Worker**
   キャッシュ戦略（Cache First、Network First、Stale While Revalidate）を設計し、オフライン対応と高速化を両立する。ライフサイクル（installing → waiting → activated）を理解し、適切なタイミングで `skipWaiting()` と `clients.claim()` を実行する。

---

## 次に読むべきガイド

- [メモリ管理](./03-memory-management.md)
  Worker で扱う大きなバッファのメモリリークを防ぐためのベストプラクティスと、SharedArrayBuffer のメモリモデルを学ぶ。

- 非同期処理パターン
  Worker への非同期リクエストを Promise でラップする方法や、複数 Worker の結果を `Promise.all` で集約するパターンを習得する。

- パフォーマンス最適化
  Worker のオフロード効果を Performance API と Chrome DevTools で定量評価し、ボトルネックを特定する手法を学ぶ。

---

## 参考文献

1. **MDN Web Docs - Web Workers API**
   https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
   Web Worker、Shared Worker、Service Worker の仕様と API リファレンス。構造化複製アルゴリズムと Transferable Objects の詳細。

2. **Google Developers - Service Worker Lifecycle**
   https://web.dev/service-worker-lifecycle/
   Service Worker のライフサイクル（installing → waiting → activated）と、`skipWaiting()`、`clients.claim()` のタイミングをダイアグラムで解説。

3. **HTML Living Standard - Web Workers**
   https://html.spec.whatwg.org/multipage/workers.html
   Worker の仕様定義。スレッドモデル、メッセージパッシング、エラーハンドリングの標準動作を確認できる。

4. **Jake Archibald - The Offline Cookbook**
   https://jakearchibald.com/2014/offline-cookbook/
   Service Worker によるキャッシュ戦略（Cache First、Network First、Stale While Revalidate など）の実装パターン集。

5. **Surma - Is postMessage slow?**
   https://surma.dev/things/is-postmessage-slow/
   `postMessage` の構造化複製アルゴリズムのパフォーマンス測定と、Transferable Objects との比較実験。ArrayBuffer の転送で 100 倍以上の高速化を実証。
