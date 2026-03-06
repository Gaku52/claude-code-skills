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
