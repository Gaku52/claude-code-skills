# ブラウザのイベントループ

> ブラウザのイベントループは JavaScript の実行モデルの核心である。タスクキュー、マイクロタスクキュー、requestAnimationFrame、requestIdleCallback の実行順序を正確に理解することで、パフォーマンスの最適化やデバッグが格段に容易になる。本ガイドでは WHATWG HTML Living Standard に準拠した正確なモデルを、豊富なコード例・図解・演習とともに解説する。

---

## 目次

1. [この章で学ぶこと](#この章で学ぶこと)
2. [前提知識](#前提知識)
3. [イベントループの全体構造](#1-イベントループの全体構造)
4. [タスク（マクロタスク）の詳細](#2-タスクマクロタスクの詳細)
5. [マイクロタスクの詳細](#3-マイクロタスクの詳細)
6. [タスク vs マイクロタスク比較](#4-タスク-vs-マイクロタスク比較)
7. [requestAnimationFrame（rAF）](#5-requestanimationframeraf)
8. [requestIdleCallback（rIC）](#6-requestidlecallbackric)
9. [スケジューリング API 比較表](#7-スケジューリング-api-比較表)
10. [実行順序の統合モデル](#8-実行順序の統合モデル)
11. [コード例集](#9-コード例集)
12. [アンチパターン](#10-アンチパターン)
13. [エッジケース分析](#11-エッジケース分析)
14. [段階別演習](#12-段階別演習)
15. [FAQ](#13-faq)
16. [用語集](#14-用語集)
17. [まとめ](#まとめ)
18. [次に読むべきガイド](#次に読むべきガイド)
19. [参考文献](#参考文献)

---

## この章で学ぶこと

- [ ] ブラウザのイベントループが WHATWG 仕様上どのように定義されているか理解する
- [ ] マクロタスクとマイクロタスクの実行順序を正確に予測できるようになる
- [ ] requestAnimationFrame（rAF）のタイミングと活用法を習得する
- [ ] requestIdleCallback（rIC）による低優先度処理の設計手法を学ぶ
- [ ] 各スケジューリング API の使い分けを身につける
- [ ] レンダリングパイプラインとイベントループの関係を把握する
- [ ] 典型的なアンチパターンを認識し、回避できるようになる
- [ ] エッジケースにおける挙動を予測できるようになる

---

## 前提知識

本ガイドを最大限に活用するためには、以下の知識があることが望ましい。

| 分野 | 必要なレベル | 参照先 |
|------|------------|--------|
| JavaScript 基礎 | コールスタック、スコープチェーンの理解 | JS 基礎ガイド |
| Promise / async-await | 基本的な非同期処理が書ける | 非同期処理ガイド |
| DOM API | addEventListener, querySelector 等 | DOM 操作ガイド |
| ブラウザレンダリング | レイアウト・ペイントの概念 | レンダリングガイド |

---

## 1. イベントループの全体構造

### 1.1 WHATWG 仕様に基づくモデル

イベントループは、ブラウザがユーザーインタラクション、スクリプト実行、レンダリング、ネットワーク処理などの作業を協調的に処理するためのメカニズムである。WHATWG HTML Living Standard（Section 8.1.7）では、イベントループの各サイクルで行われる処理ステップが厳密に定義されている。

```
┌─────────────────────────────────────────────────────────────────┐
│                  イベントループの1サイクル                        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 1: タスクキューから最も古いタスクを1つ取得し実行       │  │
│  │         （キューが空ならスキップ）                         │  │
│  │                                                           │  │
│  │  タスクの例:                                              │  │
│  │   - setTimeout / setInterval のコールバック               │  │
│  │   - I/O コールバック（fetch, XMLHttpRequest 完了）        │  │
│  │   - UI イベントのディスパッチ（click, keydown 等）        │  │
│  │   - MessageChannel の onmessage                          │  │
│  │   - history.back() / history.forward() ナビゲーション     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 2: マイクロタスクチェックポイント                     │  │
│  │         マイクロタスクキューが空になるまで繰り返し実行     │  │
│  │                                                           │  │
│  │  マイクロタスクの例:                                      │  │
│  │   - Promise の then / catch / finally コールバック        │  │
│  │   - queueMicrotask() で登録された関数                    │  │
│  │   - MutationObserver のコールバック                       │  │
│  │   - async 関数の await 後の継続処理                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 3: レンダリング更新（ブラウザが必要と判断した場合）   │  │
│  │                                                           │  │
│  │  3a. resize / scroll イベントの発火                       │  │
│  │  3b. requestAnimationFrame コールバックの実行             │  │
│  │  3c. IntersectionObserver コールバック                    │  │
│  │  3d. Style 再計算（Recalculate Style）                   │  │
│  │  3e. Layout（Reflow）                                    │  │
│  │  3f. Paint                                               │  │
│  │  3g. Composite                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 4: アイドル期間の処理（余裕がある場合）              │  │
│  │         requestIdleCallback コールバックの実行            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│                    ① に戻る（次のサイクル）                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 重要な設計原則

イベントループの設計には、以下の根本的な原則がある。

**シングルスレッド原則**: JavaScript のメインスレッドは1つだけであり、同時に1つのタスクしか実行できない。これにより、共有状態に対するデータ競合（data race）が原理的に発生しない。一方で、長時間実行されるタスクがあると UI が固まる（ジャンク）原因となる。

**協調的マルチタスキング**: イベントループはプリエンプティブ（強制的な中断）ではなく、各タスクが自発的に制御を返すことで他のタスクに実行機会を与える。これは協調的マルチタスキングの一種であり、タスクの設計者がフレーム予算（通常 16.67ms = 1/60fps）を意識する必要がある。

**タスクの粒度制御**: タスクは1つずつ実行されるが、マイクロタスクはチェックポイントごとに全て実行される。この違いは意図的な設計であり、マイクロタスクは「現在のタスクの論理的な延長」として扱われる。

### 1.3 イベントループの種類

WHATWG 仕様では、以下の種類のイベントループが定義されている。

| 種類 | コンテキスト | 特徴 |
|------|------------|------|
| Window イベントループ | ブラウザタブ / iframe | レンダリング更新を含む完全なループ |
| Worker イベントループ | Web Worker / Service Worker | レンダリングステップなし |
| Worklet イベントループ | AudioWorklet, PaintWorklet | 制限された API セット |

本ガイドでは、最も一般的な **Window イベントループ** を中心に解説する。Worker イベントループについては別ガイドで扱う。

---

## 2. タスク（マクロタスク）の詳細

### 2.1 タスクソースとタスクキュー

仕様上、イベントループは複数のタスクキューを持つことができる。各タスクは特定の「タスクソース」から生成され、同じタスクソースからのタスクは順序が保証される。ただし、異なるタスクソース間の優先度はブラウザの実装に委ねられている。

```
タスクキューの内部構造（概念図）:

 ┌──────────────────────────────────────────────────────┐
 │ イベントループ                                        │
 │                                                      │
 │  ┌─────────────────────────┐                         │
 │  │ タスクキュー A          │  ← UI イベント用         │
 │  │ [click_cb] [scroll_cb]  │                         │
 │  └─────────────────────────┘                         │
 │                                                      │
 │  ┌─────────────────────────┐                         │
 │  │ タスクキュー B          │  ← タイマー用            │
 │  │ [timeout1] [interval2]  │                         │
 │  └─────────────────────────┘                         │
 │                                                      │
 │  ┌─────────────────────────┐                         │
 │  │ タスクキュー C          │  ← ネットワーク用        │
 │  │ [fetch_cb] [xhr_cb]     │                         │
 │  └─────────────────────────┘                         │
 │                                                      │
 │  ブラウザは各サイクルで「どのキューから取るか」を      │
 │  自由に選択できる（優先度はブラウザ依存）              │
 └──────────────────────────────────────────────────────┘
```

### 2.2 主なタスクソース一覧

| タスクソース | 生成される場面 | 備考 |
|-------------|--------------|------|
| DOM 操作 | `element.click()` のプログラム的呼び出し | ユーザー操作とは別扱い |
| ユーザーインタラクション | クリック、キー入力、スクロール | ブラウザが高優先度にしがち |
| ネットワーク | fetch / XHR の完了 | レスポンス到着時にキューイング |
| ナビゲーション | `history.pushState()` 等 | ページ遷移のための処理 |
| タイマー | `setTimeout`, `setInterval` | 遅延が保証されない点に注意 |
| MessageChannel | `port.postMessage()` | Worker との通信にも使用 |
| IndexedDB | トランザクション完了時 | 非同期 DB 操作の結果通知 |

### 2.3 setTimeout の遅延に関する仕様

`setTimeout(fn, 0)` と書いても、実際には0ms後に実行されるわけではない。WHATWG 仕様では以下のルールが定められている。

```javascript
// setTimeout のネスト制限（HTML 仕様 Section 8.6）
//
// ネストレベルが 5 を超えた場合、最小遅延は 4ms に強制される

function demonstrateNestedTimeout() {
  const start = performance.now();
  let count = 0;

  function nest() {
    count++;
    const elapsed = performance.now() - start;
    console.log(`Nest ${count}: ${elapsed.toFixed(2)}ms`);

    if (count < 10) {
      setTimeout(nest, 0);  // 0ms 指定でもネスト深くなると 4ms+ になる
    }
  }

  setTimeout(nest, 0);
}

demonstrateNestedTimeout();
// 典型的な出力:
// Nest 1: 0.10ms    ← ほぼ即座
// Nest 2: 0.20ms
// Nest 3: 0.30ms
// Nest 4: 0.40ms
// Nest 5: 0.50ms
// Nest 6: 4.50ms    ← ここからネスト制限発動
// Nest 7: 8.60ms
// Nest 8: 12.70ms
// Nest 9: 16.80ms
// Nest 10: 20.90ms
```

この挙動は「setTimeout clamping」と呼ばれ、再帰的な setTimeout によるCPU の過剰消費を防ぐための保護メカニズムである。

### 2.4 タスクの実行と長時間タスクの問題

各タスクは開始から終了まで中断されない（run-to-completion）。これはコードの予測可能性を高める一方で、長時間タスクがメインスレッドをブロックし、レンダリング更新やユーザーインタラクションへの応答を妨げるリスクがある。

```javascript
// 長時間タスクの例（アンチパターン）
button.addEventListener('click', () => {
  // この同期処理が完了するまで UI は固まる
  const result = heavyComputation(); // 200ms かかる処理
  display.textContent = result;
});

// 改善例: タスクを分割する
button.addEventListener('click', async () => {
  display.textContent = 'Computing...';

  // yieldToMain: 制御をメインスレッドに返す
  await yieldToMain();

  const result = heavyComputation();
  display.textContent = result;
});

// yieldToMain のシンプルな実装
function yieldToMain() {
  return new Promise(resolve => {
    setTimeout(resolve, 0);
  });
}
```

**Long Tasks API** を使えば、50ms を超える長時間タスクを検出できる。

```javascript
// Long Tasks API による監視
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.warn(
      `Long task detected: ${entry.duration.toFixed(1)}ms`,
      `(name: ${entry.name})`
    );
  }
});

observer.observe({ type: 'longtask', buffered: true });
```

---

## 3. マイクロタスクの詳細

### 3.1 マイクロタスクチェックポイント

マイクロタスクは、以下の「マイクロタスクチェックポイント」で実行される。

1. **各タスクの実行完了後**（イベントループの Step 2）
2. **コールバック実行後**（一部の Web API）
3. **コールスタックが空になった時点**

重要なのは、マイクロタスクの処理中に新たなマイクロタスクがキューに追加された場合、**そのマイクロタスクも同じチェックポイント内で実行される**という点である。これにより、マイクロタスクの再帰的なキューイングは無限ループを引き起こす可能性がある。

```javascript
// 危険: マイクロタスクの無限ループ
// 以下のコードはブラウザタブをフリーズさせる
function dangerousInfiniteLoop() {
  queueMicrotask(() => {
    console.log('This will repeat forever');
    dangerousInfiniteLoop();  // 新しいマイクロタスクが即座に実行される
  });
}
// dangerousInfiniteLoop(); // 絶対に実行しないこと！

// 対比: setTimeout による再帰は安全
function safeRecursion() {
  setTimeout(() => {
    console.log('This yields to the event loop');
    safeRecursion();  // 次のタスクサイクルまで待機
  }, 0);
}
```

### 3.2 Promise チェーンとマイクロタスク

Promise の `.then()` / `.catch()` / `.finally()` は、Promise が解決（settled）した時点でマイクロタスクキューに追加される。

```javascript
// Promise チェーンの実行順序を追跡する
console.log('A: sync start');

const p = new Promise((resolve) => {
  console.log('B: executor (sync)');  // executor は同期実行
  resolve('done');
  console.log('C: after resolve (still sync)');
});

p.then((val) => {
  console.log('D: first then - ' + val);
}).then(() => {
  console.log('E: second then');
});

p.then(() => {
  console.log('F: another branch then');
});

console.log('G: sync end');

// 出力順序:
// A: sync start
// B: executor (sync)
// C: after resolve (still sync)
// G: sync end
// D: first then - done
// F: another branch then    ← D と F は同じ Promise から分岐
// E: second then            ← D の .then() チェーンなので F の後
```

### 3.3 async/await とマイクロタスク

`async/await` は Promise の糖衣構文（syntactic sugar）であり、`await` の後の処理はマイクロタスクとして実行される。

```javascript
async function example() {
  console.log('1: before await');
  await Promise.resolve();
  console.log('2: after await');  // マイクロタスクとして実行
  await Promise.resolve();
  console.log('3: after second await');  // 次のマイクロタスクとして実行
}

console.log('A: start');
example();
console.log('B: end');

// 出力:
// A: start
// 1: before await
// B: end
// 2: after await
// 3: after second await
```

`await` は内部的に以下のように変換される。

```
async function f() {        →   function f() {
  console.log('before');    →     console.log('before');
  await somePromise;        →     return somePromise.then(() => {
  console.log('after');     →       console.log('after');
}                           →     });
                            →   }
```

### 3.4 queueMicrotask の活用

`queueMicrotask()` は、Promise を経由せずに直接マイクロタスクをキューに追加する API である。

```javascript
// queueMicrotask の典型的な使用例:
// バッチ処理の最適化

class BatchProcessor {
  #pending = [];
  #scheduled = false;

  add(item) {
    this.#pending.push(item);

    if (!this.#scheduled) {
      this.#scheduled = true;
      // 同期コードが全て完了した後にバッチ処理
      queueMicrotask(() => {
        this.#flush();
      });
    }
  }

  #flush() {
    const batch = this.#pending.splice(0);
    this.#scheduled = false;
    console.log(`Processing batch of ${batch.length} items:`, batch);
    // 実際の処理をここで一括実行
  }
}

const processor = new BatchProcessor();
processor.add('item1');
processor.add('item2');
processor.add('item3');
// 同期コード完了後に1回だけ flush される:
// "Processing batch of 3 items: ['item1', 'item2', 'item3']"
```

### 3.5 MutationObserver とマイクロタスク

`MutationObserver` のコールバックはマイクロタスクとして実行される。これにより、複数の DOM 変更を1つのコールバック呼び出しで効率的に処理できる。

```javascript
const observer = new MutationObserver((mutations) => {
  // ここはマイクロタスクとして実行される
  console.log(`${mutations.length} mutations observed`);
  mutations.forEach(m => {
    console.log(`  Type: ${m.type}, Target: ${m.target.id}`);
  });
});

observer.observe(document.getElementById('container'), {
  childList: true,
  subtree: true,
  attributes: true,
});

// 以下の3つの DOM 変更は、同じマイクロタスクチェックポイントで
// 1回のコールバックにまとめて通知される
const container = document.getElementById('container');
container.appendChild(document.createElement('div'));
container.setAttribute('data-count', '1');
container.firstChild.textContent = 'Hello';
```

---

## 4. タスク vs マイクロタスク比較

### 4.1 比較表

| 特性 | タスク（マクロタスク） | マイクロタスク |
|------|----------------------|--------------|
| **実行タイミング** | イベントループの各サイクルで1つずつ | チェックポイントで全て実行 |
| **レンダリングとの関係** | タスク間にレンダリング機会あり | マイクロタスク中はレンダリングなし |
| **キューの管理** | 複数のタスクキュー（優先度あり） | 単一のマイクロタスクキュー |
| **生成元** | setTimeout, I/O, UI イベント | Promise, queueMicrotask, MutationObserver |
| **優先度** | マイクロタスクより低い | タスクより高い（同サイクル内で先に実行） |
| **無限ループの危険** | 各タスク間に yield するため比較的安全 | 再帰的追加で無限ループの危険あり |
| **典型的な遅延** | 最小 4ms（ネスト制限時） | サブミリ秒 |
| **使いどころ** | 遅延実行、定期処理 | 非同期処理の継続、状態の一貫性保証 |

### 4.2 実行順序の視覚化

```
時間軸 →

タスクのみの場合:
  ┌──────┐     ┌──────┐     ┌──────┐
  │Task A│ Ren │Task B│ Ren │Task C│
  └──────┘     └──────┘     └──────┘
          ↑           ↑
      レンダリング  レンダリング

マイクロタスクありの場合:
  ┌──────┬──────────────┐     ┌──────┬───────┐
  │Task A│ Micro1 Micro2│ Ren │Task B│ Micro3│ Ren
  └──────┴──────────────┘     └──────┴───────┘
                         ↑                     ↑
                     レンダリング           レンダリング

  ポイント: マイクロタスクはタスク直後に「割り込み」実行される
           レンダリングはマイクロタスク完了後まで遅延する
```

### 4.3 実行順序の完全な例

```javascript
// コード例1: タスクとマイクロタスクの実行順序（基礎）
console.log('1: sync');

setTimeout(() => {
  console.log('2: timeout');
}, 0);

Promise.resolve().then(() => {
  console.log('3: promise');
});

queueMicrotask(() => {
  console.log('4: queueMicrotask');
});

console.log('5: sync');

// 出力:
// 1: sync
// 5: sync
// 3: promise
// 4: queueMicrotask
// 2: timeout
//
// 解説:
// (a) 同期コード: 1, 5
// (b) マイクロタスク: 3, 4 （Promise.then と queueMicrotask は同列）
// (c) タスク: 2 （setTimeout は次のタスクサイクル）
```

```javascript
// コード例2: タスク内で生成されるマイクロタスク（中級）
setTimeout(() => {
  console.log('timeout 1');
  Promise.resolve().then(() => console.log('promise in timeout 1'));
}, 0);

setTimeout(() => {
  console.log('timeout 2');
  Promise.resolve().then(() => console.log('promise in timeout 2'));
}, 0);

Promise.resolve().then(() => {
  console.log('promise 1');
  queueMicrotask(() => console.log('nested microtask'));
});

// 出力:
// promise 1
// nested microtask     ← promise 1 のマイクロタスク内で追加されたもの
// timeout 1
// promise in timeout 1 ← timeout 1 完了後のマイクロタスクチェックポイント
// timeout 2
// promise in timeout 2 ← timeout 2 完了後のマイクロタスクチェックポイント
```

---

## 5. requestAnimationFrame（rAF）

### 5.1 rAF の実行タイミング

`requestAnimationFrame` はレンダリング更新の直前に呼び出されるコールバックを登録する API である。ブラウザがレンダリングを行うと判断したサイクルにおいて、rAF コールバックが実行され、その後にスタイル再計算・レイアウト・ペイントが行われる。

```
1フレームの詳細タイムライン（60fps の場合、1フレーム = 約16.67ms）:

  0ms                              16.67ms
  │                                │
  ├── Task 実行                    │
  │     │                          │
  │     ├── Microtask チェックポイント
  │     │                          │
  │     ├── resize / scroll events │
  │     │                          │
  │     ├── rAF コールバック群      │ ← ここで DOM 変更を行う
  │     │                          │
  │     ├── Style 再計算           │
  │     ├── Layout（Reflow）       │
  │     ├── Paint                  │
  │     ├── Composite              │
  │     │                          │
  │     └── Idle（余った時間）      │ ← rIC はここで実行
  │                                │
  │◄──────── 16.67ms ─────────────►│
```

### 5.2 rAF の基本的な使い方

```javascript
// コード例3: rAF によるスムーズアニメーション
function animateElement(element, targetX, duration) {
  const startX = parseFloat(getComputedStyle(element).transform.split(',')[4]) || 0;
  const distance = targetX - startX;
  let startTime = null;

  function frame(currentTime) {
    if (startTime === null) startTime = currentTime;
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);

    // イージング関数: ease-out-cubic
    const eased = 1 - Math.pow(1 - progress, 3);

    element.style.transform = `translateX(${startX + distance * eased}px)`;

    if (progress < 1) {
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

// 使用例
const box = document.getElementById('animated-box');
animateElement(box, 300, 1000);  // 300px の位置まで 1000ms かけて移動
```

### 5.3 rAF と DOM バッチ更新

DOM の読み取りと書き込みを交互に行うと、ブラウザは「強制同期レイアウト（Forced Synchronous Layout）」を発生させてパフォーマンスが低下する。rAF を使って書き込みをバッチ処理することで、この問題を回避できる。

```javascript
// アンチパターン: 読み取りと書き込みの交互実行
// → 強制同期レイアウトが発生する
function badLayout(elements) {
  elements.forEach(el => {
    const height = el.offsetHeight;    // 読み取り（Layout を強制）
    el.style.height = height * 2 + 'px'; // 書き込み（Layout を無効化）
    // 次の読み取りで再び Layout 計算が必要になる
  });
}

// 推奨パターン: 読み取りと書き込みを分離
function goodLayout(elements) {
  // Phase 1: 全ての読み取りを先に行う
  const heights = elements.map(el => el.offsetHeight);

  // Phase 2: rAF 内で全ての書き込みを行う
  requestAnimationFrame(() => {
    elements.forEach((el, i) => {
      el.style.height = heights[i] * 2 + 'px';
    });
  });
}
```

### 5.4 rAF のキャンセル

```javascript
// rAF のキャンセル方法
let animationId = null;

function startAnimation() {
  function frame(timestamp) {
    // アニメーション処理
    updatePosition(timestamp);
    animationId = requestAnimationFrame(frame);
  }
  animationId = requestAnimationFrame(frame);
}

function stopAnimation() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

// ページ可視性の変化に応じた制御
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopAnimation();  // タブが非表示になったら停止
  } else {
    startAnimation(); // タブが表示されたら再開
  }
});
```

### 5.5 rAF の注意点

1. **非アクティブタブでの動作**: 多くのブラウザは、非アクティブタブの rAF を停止またはスロットリングする。これは省電力のための措置である。
2. **フレームレートの変動**: rAF は必ずしも 60fps で呼ばれるわけではない。ディスプレイのリフレッシュレート（120Hz, 144Hz など）やブラウザの負荷に依存する。
3. **rAF 内での rAF 登録**: rAF コールバック内で新たに `requestAnimationFrame` を呼ぶと、**次のフレーム** で実行される（同フレームでは実行されない）。

```javascript
// rAF 内での rAF 登録は次フレームになる
requestAnimationFrame(() => {
  console.log('Frame 1');
  requestAnimationFrame(() => {
    console.log('Frame 2');  // Frame 1 の次のフレームで実行
  });
});
```

### 5.6 rAF vs setTimeout(fn, 16) 比較表

| 特性 | requestAnimationFrame | setTimeout(fn, 16) |
|------|----------------------|-------------------|
| **フレーム同期** | ディスプレイリフレッシュレートに正確に同期 | ずれが蓄積する可能性あり |
| **非アクティブタブ** | 停止される（省電力） | 実行され続ける（一部ブラウザで制限あり） |
| **タイムスタンプ** | 高精度の DOMHighResTimeStamp が渡される | 自分で計測する必要あり |
| **バッテリー消費** | 低い（不要時は停止） | 高い（常時実行） |
| **レンダリングとの関係** | レンダリング直前に実行（最適なタイミング） | レンダリングとは無関係 |
| **最小間隔** | ディスプレイ依存（16.67ms @60Hz） | 4ms（ネスト制限後） |
| **用途** | アニメーション、DOM 更新 | 遅延実行、ポーリング |

---

## 6. requestIdleCallback（rIC）

### 6.1 アイドル期間の概念

ブラウザの各フレームには「予算」がある。60fps のディスプレイでは 1 フレームあたり約 16.67ms である。タスク実行、マイクロタスク処理、レンダリングが全てこの予算内に収まった場合、残りの時間が「アイドル期間」となる。`requestIdleCallback` は、このアイドル期間に低優先度の処理を実行するための API である。

```
フレーム予算の配分イメージ:

 ケース1: 処理が軽い場合（アイドル時間あり）
 ┌────────┬──────┬──────────┬──────────────────┐
 │ Task   │Micro │Rendering │   Idle (rIC)     │
 │ 3ms    │ 1ms  │  5ms     │   7.67ms         │
 └────────┴──────┴──────────┴──────────────────┘
 │◄───────────── 16.67ms ─────────────────────►│

 ケース2: 処理が重い場合（アイドル時間なし）
 ┌────────────────────┬──────┬──────────┐
 │ Task               │Micro │Rendering │
 │ 10ms               │ 2ms  │  5ms     │← 予算超過！rIC は実行されない
 └────────────────────┴──────┴──────────┘
 │◄───────────── 17ms ────────────────►│
```

### 6.2 IdleDeadline API

rIC のコールバックには `IdleDeadline` オブジェクトが渡される。これを使ってアイドル期間の残り時間を確認し、処理を適切に分割できる。

```javascript
// コード例4: rIC による段階的なデータ処理
class IdleProcessor {
  #queue = [];
  #isProcessing = false;

  enqueue(items) {
    this.#queue.push(...items);
    this.#scheduleProcessing();
  }

  #scheduleProcessing() {
    if (this.#isProcessing) return;
    this.#isProcessing = true;

    requestIdleCallback((deadline) => {
      this.#process(deadline);
    }, { timeout: 5000 });  // 最大 5 秒待っても実行されない場合は強制実行
  }

  #process(deadline) {
    // アイドル時間が残っている間、またはタイムアウトした場合に処理
    while (
      this.#queue.length > 0 &&
      (deadline.timeRemaining() > 1 || deadline.didTimeout)
    ) {
      const item = this.#queue.shift();
      this.#processItem(item);
    }

    if (this.#queue.length > 0) {
      // まだ残りがある場合、次のアイドル期間に継続
      requestIdleCallback((deadline) => {
        this.#process(deadline);
      }, { timeout: 5000 });
    } else {
      this.#isProcessing = false;
    }
  }

  #processItem(item) {
    // 個々のアイテムの処理
    console.log(`Processing: ${item}`);
  }
}

// 使用例: 1000件のアイテムをアイドル時間で段階的に処理
const processor = new IdleProcessor();
processor.enqueue(Array.from({ length: 1000 }, (_, i) => `item-${i}`));
```

### 6.3 rIC の制約と注意事項

**DOM 操作の禁止**: rIC コールバック内で DOM を変更すると、予期しないレイアウト再計算が発生する可能性がある。DOM 変更が必要な場合は、rIC 内から rAF をスケジュールし、rAF 内で DOM を操作すべきである。

```javascript
// 推奨: rIC + rAF の組み合わせ
requestIdleCallback((deadline) => {
  // アイドル時間中に計算を行う
  const results = performCalculations(deadline);

  // DOM 変更は rAF に委譲する
  requestAnimationFrame(() => {
    updateDOM(results);
  });
});
```

**ブラウザサポート**: Safari は requestIdleCallback をサポートしていない（2025年時点）。ポリフィルまたはフォールバックが必要である。

```javascript
// rIC のポリフィル（簡易版）
const requestIdleCallbackCompat =
  window.requestIdleCallback ||
  function(callback, options) {
    const start = Date.now();
    return setTimeout(() => {
      callback({
        didTimeout: false,
        timeRemaining() {
          return Math.max(0, 50 - (Date.now() - start));
        },
      });
    }, 1);
  };

const cancelIdleCallbackCompat =
  window.cancelIdleCallback ||
  function(id) {
    clearTimeout(id);
  };
```

### 6.4 rIC の典型的なユースケース

| ユースケース | 説明 | timeout の目安 |
|-------------|------|---------------|
| アナリティクス送信 | ユーザー行動ログの非同期送信 | 3000ms |
| プリフェッチ | 次のページのリソースを事前取得 | 5000ms |
| 遅延初期化 | 非クリティカル機能の初期化 | 10000ms |
| データ前処理 | 検索インデックスの構築など | なし（完了保証不要） |
| キャッシュ管理 | 不要なキャッシュエントリの削除 | なし |
| テレメトリ | パフォーマンスデータの収集・送信 | 5000ms |

---

## 7. スケジューリング API 比較表

### 7.1 全 API 横断比較

| API | 実行タイミング | 優先度 | フレーム同期 | キャンセル可能 | 最適な用途 |
|-----|--------------|--------|-------------|--------------|-----------|
| 同期コード | 即座 | 最高 | - | 不可 | 即時実行が必要な処理 |
| `queueMicrotask()` | タスク直後 | 高 | なし | 不可 | 状態の一貫性保証 |
| `Promise.then()` | タスク直後 | 高 | なし | 不可 | 非同期処理の継続 |
| `MutationObserver` | DOM 変更後 | 高 | なし | observe 解除 | DOM 変更の監視 |
| `requestAnimationFrame` | レンダリング前 | 中-高 | あり | `cancelAnimationFrame` | アニメーション、DOM 更新 |
| `setTimeout(fn, 0)` | 次サイクル以降 | 中 | なし | `clearTimeout` | 遅延実行 |
| `setInterval(fn, ms)` | 定期的 | 中 | なし | `clearInterval` | 定期ポーリング |
| `MessageChannel` | 次サイクル | 中 | なし | port.close() | ネスト制限回避 |
| `requestIdleCallback` | アイドル時 | 低 | なし | `cancelIdleCallback` | 低優先度処理 |
| `scheduler.postTask()` | 優先度依存 | 可変 | なし | `AbortController` | 優先度付きタスク |

### 7.2 scheduler.postTask()（新しい API）

`scheduler.postTask()` は、タスクに明示的な優先度を付与できる新しい API である（Chrome 94+ でサポート）。

```javascript
// scheduler.postTask() の使用例
async function demonstrateScheduler() {
  // user-blocking: ユーザー操作に影響する処理（最高優先度）
  scheduler.postTask(() => {
    console.log('user-blocking task');
  }, { priority: 'user-blocking' });

  // user-visible: 表示に影響するが即時でなくてよい処理
  scheduler.postTask(() => {
    console.log('user-visible task');
  }, { priority: 'user-visible' });

  // background: バックグラウンド処理（最低優先度）
  scheduler.postTask(() => {
    console.log('background task');
  }, { priority: 'background' });
}

// AbortController によるキャンセル
const controller = new AbortController();

scheduler.postTask(
  () => { console.log('cancellable task'); },
  { priority: 'background', signal: controller.signal }
);

controller.abort();  // タスクをキャンセル
```

---

## 8. 実行順序の統合モデル

### 8.1 全 API を含む実行順序

```javascript
// コード例5: 全スケジューリング API の実行順序
console.log('1: sync');

setTimeout(() => console.log('2: setTimeout'), 0);

Promise.resolve().then(() => console.log('3: promise'));

queueMicrotask(() => console.log('4: queueMicrotask'));

requestAnimationFrame(() => console.log('5: rAF'));

requestIdleCallback(() => console.log('6: rIC'));

console.log('7: sync end');

// 保証される順序:
// 1: sync
// 7: sync end
// 3: promise         ← マイクロタスク（Promise）
// 4: queueMicrotask  ← マイクロタスク（queueMicrotask）
//
// 以下はブラウザの判断による（相対順序は変わりうる）:
// 2: setTimeout      ← タスク
// 5: rAF             ← レンダリング前（レンダリングが発生する場合）
// 6: rIC             ← アイドル時間（余裕がある場合）
//
// 典型的な出力:
// 1, 7, 3, 4, 5, 2, 6
// または
// 1, 7, 3, 4, 2, 5, 6
```

### 8.2 実行順序の決定フローチャート

```
実行順序を予測するためのフローチャート:

  コードを読む
       │
       ▼
  ┌─────────────┐    はい    ┌──────────────────┐
  │ 同期コードか？├──────────►│ 即座に実行（順序通り）│
  └──────┬──────┘           └──────────────────┘
         │ いいえ
         ▼
  ┌─────────────────┐  はい  ┌────────────────────────┐
  │ マイクロタスクか？├──────►│ 現在のタスク完了直後に   │
  │ (Promise, queue  │       │ 全てのマイクロタスクと   │
  │  Microtask, etc) │       │ 一緒に実行              │
  └──────┬──────────┘       └────────────────────────┘
         │ いいえ
         ▼
  ┌─────────────┐    はい    ┌────────────────────────┐
  │ rAF か？     ├──────────►│ 次のレンダリングフレーム │
  │              │           │ の直前に実行             │
  └──────┬──────┘           └────────────────────────┘
         │ いいえ
         ▼
  ┌─────────────┐    はい    ┌────────────────────────┐
  │ タスクか？    ├──────────►│ タスクキューに追加      │
  │ (setTimeout) │           │ 順番が来たら実行        │
  └──────┬──────┘           └────────────────────────┘
         │ いいえ
         ▼
  ┌─────────────┐    はい    ┌────────────────────────┐
  │ rIC か？     ├──────────►│ アイドル期間に実行      │
  │              │           │（最も低い優先度）       │
  └─────────────┘           └────────────────────────┘
```

### 8.3 複雑な実行順序のトレース

以下は、複数の API が組み合わさった場合の詳細なトレースである。

```javascript
// 複合的な実行順序の例
console.log('A');

setTimeout(() => {
  console.log('B');
  queueMicrotask(() => console.log('C'));
}, 0);

requestAnimationFrame(() => {
  console.log('D');
  Promise.resolve().then(() => console.log('E'));
});

queueMicrotask(() => {
  console.log('F');
  queueMicrotask(() => console.log('G'));
});

Promise.resolve().then(() => console.log('H'));

console.log('I');

// トレース:
// Step 1 (同期): A, I
// Step 2 (マイクロタスク): F → G（F 内で追加された G も同じチェックポイント）, H
//   ※ F と H の順序は登録順で F → H、G は F の中で追加されるので F → G → H
//   正確な出力: F, G, H
// Step 3 (レンダリング判断):
//   レンダリングする場合: D → E（rAF 内の Promise）
//   レンダリングしない場合: D, E は次のレンダリングまで遅延
// Step 4 (タスク): B → C（B 内で追加された C はマイクロタスクチェックポイントで実行）
//
// 典型的な出力: A, I, F, G, H, D, E, B, C
```

