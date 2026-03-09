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

---

## 9. コード例集

### 9.1 コード例6: MessageChannel による即時タスクスケジューリング

`MessageChannel` を使うと、`setTimeout` のネスト制限（4ms）を回避して、より高速にタスクをスケジュールできる。React のスケジューラ（Scheduler パッケージ）でもこの手法が使用されている。

```javascript
// MessageChannel を使った高速タスクスケジューリング
function scheduleTask(callback) {
  const channel = new MessageChannel();
  channel.port1.onmessage = () => callback();
  channel.port2.postMessage(null);
}

// 速度比較
async function benchmark() {
  const iterations = 100;

  // setTimeout(fn, 0) の場合
  const startTimeout = performance.now();
  let countTimeout = 0;
  await new Promise(resolve => {
    function next() {
      countTimeout++;
      if (countTimeout < iterations) {
        setTimeout(next, 0);
      } else {
        resolve();
      }
    }
    setTimeout(next, 0);
  });
  const timeoutDuration = performance.now() - startTimeout;

  // MessageChannel の場合
  const startChannel = performance.now();
  let countChannel = 0;
  await new Promise(resolve => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      countChannel++;
      if (countChannel < iterations) {
        channel.port2.postMessage(null);
      } else {
        resolve();
      }
    };
    channel.port2.postMessage(null);
  });
  const channelDuration = performance.now() - startChannel;

  console.log(`setTimeout x${iterations}: ${timeoutDuration.toFixed(1)}ms`);
  console.log(`MessageChannel x${iterations}: ${channelDuration.toFixed(1)}ms`);
  // 典型的な結果（Chrome）:
  // setTimeout x100: ~450ms（ネスト制限で各 4ms+ に）
  // MessageChannel x100: ~15ms（ネスト制限なし）
}
```

### 9.2 コード例7: イベントループを活用したプログレス表示

```javascript
// 重い処理の途中でプログレスバーを更新する
async function processWithProgress(data, progressCallback) {
  const total = data.length;
  const chunkSize = 100;

  for (let i = 0; i < total; i += chunkSize) {
    const chunk = data.slice(i, i + chunkSize);

    // チャンクを処理
    for (const item of chunk) {
      processItem(item);
    }

    // プログレスを更新（DOM 操作）
    const progress = Math.min((i + chunkSize) / total, 1);
    progressCallback(progress);

    // メインスレッドに制御を返してレンダリングを許可
    await new Promise(resolve => {
      requestAnimationFrame(() => {
        // rAF 内で resolve することで、レンダリング後に次のチャンクが実行される
        resolve();
      });
    });
  }

  progressCallback(1);
}

// 使用例
const progressBar = document.getElementById('progress-bar');
const data = generateLargeDataset(10000);

processWithProgress(data, (progress) => {
  progressBar.style.width = `${progress * 100}%`;
  progressBar.textContent = `${Math.round(progress * 100)}%`;
});
```

### 9.3 コード例8: デバウンスとイベントループ

```javascript
// マイクロタスクベースのデバウンス（同一タスク内の複数呼び出しを統合）
function microtaskDebounce(fn) {
  let scheduled = false;
  let latestArgs = null;

  return function(...args) {
    latestArgs = args;
    if (!scheduled) {
      scheduled = true;
      queueMicrotask(() => {
        scheduled = false;
        fn.apply(this, latestArgs);
      });
    }
  };
}

// rAF ベースのデバウンス（フレーム単位で統合）
function rafDebounce(fn) {
  let frameId = null;
  let latestArgs = null;

  return function(...args) {
    latestArgs = args;
    if (frameId === null) {
      frameId = requestAnimationFrame(() => {
        frameId = null;
        fn.apply(this, latestArgs);
      });
    }
  };
}

// タスクベースのデバウンス（従来型、ms 指定）
function taskDebounce(fn, delay = 300) {
  let timerId = null;

  return function(...args) {
    clearTimeout(timerId);
    timerId = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

// 使い分け:
// microtaskDebounce: 同一同期コンテキスト内の重複排除
// rafDebounce: スクロールやリサイズなどフレーム単位の処理
// taskDebounce: ユーザー入力（検索ボックスなど）の待機
```

---

## 10. アンチパターン

### 10.1 アンチパターン1: マイクロタスクによるレンダリングブロック

**問題**: マイクロタスクはチェックポイント内で全て実行されるため、大量のマイクロタスクはレンダリングを長時間ブロックする。

```javascript
// NG: マイクロタスクの大量キューイング
function processAllWithMicrotasks(items) {
  items.forEach((item, index) => {
    // 10000個の Promise チェーンがマイクロタスクキューに積まれる
    Promise.resolve().then(() => {
      processItem(item);
      if (index % 100 === 0) {
        updateProgressUI(index / items.length);
        // この UI 更新はレンダリングされない！
        // 全てのマイクロタスクが完了するまでレンダリングはブロックされる
      }
    });
  });
}

// OK: タスクに分割してレンダリング機会を確保
async function processAllWithYield(items) {
  const chunkSize = 50;
  for (let i = 0; i < items.length; i += chunkSize) {
    const chunk = items.slice(i, i + chunkSize);
    chunk.forEach(item => processItem(item));

    updateProgressUI(Math.min((i + chunkSize) / items.length, 1));

    // setTimeout でメインスレッドに制御を返す
    await new Promise(resolve => setTimeout(resolve, 0));
  }
}
```

**なぜ問題なのか**: マイクロタスクキューが空になるまでレンダリングパイプラインは開始されない。10000件のマイクロタスクが積まれると、全て実行されるまで画面は更新されず、ユーザーにはフリーズしたように見える。

### 10.2 アンチパターン2: rAF 内での重い同期処理

**問題**: rAF コールバック内で重い処理を行うと、フレーム予算を超過してフレームドロップが発生する。

```javascript
// NG: rAF 内での重い処理
requestAnimationFrame(() => {
  // 大量のデータをソート（数十ms かかる可能性）
  const sorted = hugeArray.sort((a, b) => complexComparison(a, b));

  // ソート結果を DOM に反映
  sorted.forEach(item => {
    const el = document.createElement('div');
    el.textContent = item.name;
    container.appendChild(el);  // DOM 操作も重い
  });
  // フレーム予算（16.67ms）を大幅に超過 → ジャンク発生
});

// OK: 計算は事前に行い、rAF では DOM 操作のみ
const sorted = hugeArray.sort((a, b) => complexComparison(a, b));

// DOM 更新は DocumentFragment を使ってバッチ処理
requestAnimationFrame(() => {
  const fragment = document.createDocumentFragment();
  sorted.forEach(item => {
    const el = document.createElement('div');
    el.textContent = item.name;
    fragment.appendChild(el);
  });
  container.appendChild(fragment);  // 1回の DOM 操作で済む
});
```

**なぜ問題なのか**: rAF はレンダリングの直前に実行される。ここでフレーム予算を使い切ると、レンダリング自体が遅延し、ユーザーが認知できるレベルのカクつきが発生する。rAF 内では DOM の書き込みのみに集中し、計算処理は事前に完了させるべきである。

### 10.3 アンチパターン3: setInterval の不適切な使用

```javascript
// NG: setInterval で正確なタイミングを期待
let lastTime = performance.now();
setInterval(() => {
  const now = performance.now();
  const drift = now - lastTime - 1000;
  console.log(`Drift: ${drift.toFixed(1)}ms`);
  lastTime = now;
  // 長時間実行するとドリフトが蓄積する
}, 1000);

// OK: setTimeout の再帰呼び出しで自己補正
function accurateInterval(callback, interval) {
  let expected = performance.now() + interval;

  function step() {
    const now = performance.now();
    const drift = now - expected;
    callback(drift);

    expected += interval;
    // ドリフトを補正して次のタイマーを設定
    setTimeout(step, Math.max(0, interval - drift));
  }

  setTimeout(step, interval);
}

accurateInterval((drift) => {
  console.log(`Drift: ${drift.toFixed(1)}ms`);
}, 1000);
```

---

## 11. エッジケース分析

### 11.1 エッジケース1: Promise コンストラクタ内の例外

Promise コンストラクタの executor 内で同期的にスローされた例外は、Promise の reject として処理される。しかし、executor 内で非同期に（setTimeout 内で）スローされた例外は、catch できない未処理例外となる。

```javascript
// ケース A: executor 内の同期例外 → reject として捕捉可能
const p1 = new Promise((resolve, reject) => {
  throw new Error('sync error');
});
p1.catch(err => console.log('Caught:', err.message));
// 出力: Caught: sync error

// ケース B: executor 内の非同期例外 → 捕捉不可
const p2 = new Promise((resolve, reject) => {
  setTimeout(() => {
    throw new Error('async error');
    // この例外は Promise チェーンでは捕捉できない
    // window.onerror または unhandledrejection で検出される
  }, 0);
});
p2.catch(err => console.log('This will NOT be called'));

// ケース C: resolve 後の例外は無視される
const p3 = new Promise((resolve, reject) => {
  resolve('done');
  throw new Error('after resolve');
  // resolve 後の throw は無視される（Promise の状態は不変）
});
p3.then(val => console.log('Value:', val));
// 出力: Value: done
```

**イベントループとの関連**: ケース B では、`setTimeout` のコールバックは別のタスクとして実行される。そのタスク内での例外は、元の Promise チェーンとは完全に独立したコンテキストで発生するため、`.catch()` では捕捉できない。これはイベントループの「タスク境界」を跨ぐことによるものである。

### 11.2 エッジケース2: ネストされた rAF の実行フレーム

`requestAnimationFrame` 内で新たに `requestAnimationFrame` を呼ぶと、新しいコールバックは次のフレームで実行される。これはレイアウトの読み取り・書き込みパターンで活用できるが、注意が必要である。

```javascript
// 「次のフレームまで待つ」テクニック
function afterNextPaint(callback) {
  requestAnimationFrame(() => {
    // この rAF は現在のフレームのレンダリング前に実行
    requestAnimationFrame(() => {
      // この rAF は次のフレームのレンダリング前に実行
      // つまり、前のフレームのレンダリング（Paint）完了後
      callback();
    });
  });
}

// 使用例: DOM 変更後に「描画完了」を検知
element.style.display = 'block';
afterNextPaint(() => {
  // ここでは element が画面上に描画されていることが期待できる
  const rect = element.getBoundingClientRect();
  console.log('Element is now visible at:', rect);
});
```

```
ダブル rAF のタイムライン:

 Frame N                          Frame N+1
 ┌──────────────────────────────┐ ┌────────────────────────────┐
 │ rAF-1 │ Style │Layout│Paint │ │ rAF-2  │ Style│Layout│Paint│
 │(登録) │       │      │      │ │(実行)  │      │      │     │
 └───┬──────────────────────────┘ └───┬────────────────────────┘
     │                                │
     └─ rAF-2 を登録                  └─ callback 実行
                                         （Paint 後の状態が確定）
```

### 11.3 エッジケース3: async/await と実行コンテキストの保持

```javascript
// async 関数内での this の挙動
class Timer {
  name = 'MyTimer';

  async start() {
    console.log(this.name);    // 'MyTimer' （同期部分）

    await Promise.resolve();
    console.log(this.name);    // 'MyTimer' （this は保持される）

    // しかし、コールバックとして渡した場合は異なる
    setTimeout(function() {
      // console.log(this.name); // undefined （this が失われる）
    }, 0);

    setTimeout(() => {
      console.log(this.name);   // 'MyTimer' （arrow function で this 保持）
    }, 0);
  }
}

// await 前後でマイクロタスクの実行順序が変わるケース
async function tricky() {
  console.log('1');
  await null;           // マイクロタスク境界
  console.log('2');
  await null;           // マイクロタスク境界
  console.log('3');
}

console.log('A');
tricky();
console.log('B');
Promise.resolve().then(() => console.log('C'));

// 出力: A, 1, B, C, 2, 3
// 解説:
// 同期: A, 1（await null まで）, B
// マイクロタスク: C（Promise.then）, 2（await null の継続）
//   ※ await null の継続は Promise.resolve(null).then(() => ...) と等価
//   C と 2 の順序は、C が先に登録されているため C が先
// 次のマイクロタスク: 3（2の await null の継続）
```

---

## 12. 段階別演習

### 12.1 演習1（初級）: 実行順序の予測

以下のコードの出力順序を予測せよ。予測後にブラウザの DevTools コンソールで検証すること。

```javascript
// 問題1
console.log('start');

setTimeout(() => console.log('timeout'), 0);

Promise.resolve()
  .then(() => console.log('promise 1'))
  .then(() => console.log('promise 2'));

queueMicrotask(() => console.log('microtask'));

console.log('end');
```

<details>
<summary>解答</summary>

```
start
end
promise 1
microtask
promise 2
timeout
```

**解説**:
1. 同期コード: `start`, `end`
2. マイクロタスクチェックポイント:
   - `promise 1`（最初の .then、Promise.resolve() で即座にキューイング）
   - `microtask`（queueMicrotask で登録）
   - `promise 2`（promise 1 の .then が解決した後にキューイング → 同チェックポイント内で実行）
3. タスク: `timeout`

`promise 1` と `microtask` の順序は、Promise.resolve().then() と queueMicrotask() の登録順に依存する。`promise 2` は `promise 1` の実行完了後にマイクロタスクキューに追加されるが、チェックポイント内なのでそのまま実行される。
</details>

### 12.2 演習2（中級）: rAF を使ったアニメーション実装

以下の要件を満たすカウントダウンタイマーを rAF で実装せよ。

- 10 から 0 までカウントダウンする
- 各カウントの表示は正確に 1 秒間隔にする
- カウント 0 で停止し、「Complete!」と表示する
- `cancelAnimationFrame` で途中停止可能にする

```javascript
// 演習2のスケルトン
function createCountdown(element, from, onComplete) {
  let startTime = null;
  let currentCount = from;
  let animationId = null;

  function tick(timestamp) {
    // ここを実装せよ
  }

  animationId = requestAnimationFrame(tick);

  // キャンセル関数を返す
  return () => {
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  };
}
```

<details>
<summary>解答</summary>

```javascript
function createCountdown(element, from, onComplete) {
  let startTime = null;
  let currentCount = from;
  let animationId = null;

  element.textContent = String(currentCount);

  function tick(timestamp) {
    if (startTime === null) {
      startTime = timestamp;
    }

    const elapsed = timestamp - startTime;
    const newCount = from - Math.floor(elapsed / 1000);

    if (newCount !== currentCount && newCount >= 0) {
      currentCount = newCount;
      element.textContent = String(currentCount);
    }

    if (currentCount > 0) {
      animationId = requestAnimationFrame(tick);
    } else {
      element.textContent = 'Complete!';
      animationId = null;
      if (onComplete) onComplete();
    }
  }

  animationId = requestAnimationFrame(tick);

  return () => {
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  };
}

// 使用例
const display = document.getElementById('countdown');
const cancel = createCountdown(display, 10, () => {
  console.log('Countdown finished!');
});

// 5秒後に途中停止する場合:
// setTimeout(() => cancel(), 5000);
```

**ポイント**:
- `startTime` を最初のフレームで記録し、経過時間ベースでカウントを計算する
- `setInterval` ではなく rAF を使うことで、フレームに同期した滑らかな表示が可能
- キャンセル関数を返すことで外部からの停止を可能にする
</details>

### 12.3 演習3（上級）: タスクスケジューラの実装

以下の要件を満たすタスクスケジューラを実装せよ。

- 優先度付きタスクキュー（high, normal, low の 3 段階）
- 各タスクはフレーム予算（デフォルト 8ms）を超えない範囲で実行
- 予算超過時は次のフレームに延期
- タスクの追加・キャンセルが可能

```javascript
// 演習3のスケルトン
class PriorityTaskScheduler {
  #queues = { high: [], normal: [], low: [] };
  #isRunning = false;
  #frameBudget;

  constructor(frameBudgetMs = 8) {
    this.#frameBudget = frameBudgetMs;
  }

  schedule(task, priority = 'normal') {
    // ここを実装せよ
    // task は { id: string, run: () => void } の形式
  }

  cancel(taskId) {
    // ここを実装せよ
  }

  #processQueue() {
    // ここを実装せよ
  }
}
```

<details>
<summary>解答</summary>

```javascript
class PriorityTaskScheduler {
  #queues = { high: [], normal: [], low: [] };
  #isRunning = false;
  #frameBudget;
  #frameId = null;

  constructor(frameBudgetMs = 8) {
    this.#frameBudget = frameBudgetMs;
  }

  schedule(task, priority = 'normal') {
    if (!this.#queues[priority]) {
      throw new Error(`Invalid priority: ${priority}`);
    }
    this.#queues[priority].push(task);
    this.#ensureRunning();
    return task.id;
  }

  cancel(taskId) {
    for (const priority of ['high', 'normal', 'low']) {
      const index = this.#queues[priority].findIndex(t => t.id === taskId);
      if (index !== -1) {
        this.#queues[priority].splice(index, 1);
        return true;
      }
    }
    return false;
  }

  #ensureRunning() {
    if (this.#isRunning) return;
    this.#isRunning = true;
    this.#frameId = requestAnimationFrame((ts) => this.#processQueue(ts));
  }

  #getNextTask() {
    for (const priority of ['high', 'normal', 'low']) {
      if (this.#queues[priority].length > 0) {
        return this.#queues[priority].shift();
      }
    }
    return null;
  }

  #hasRemainingTasks() {
    return (
      this.#queues.high.length > 0 ||
      this.#queues.normal.length > 0 ||
      this.#queues.low.length > 0
    );
  }

  #processQueue(frameTimestamp) {
    const deadline = performance.now() + this.#frameBudget;

    while (performance.now() < deadline) {
      const task = this.#getNextTask();
      if (!task) break;

      try {
        task.run();
      } catch (err) {
        console.error(`Task ${task.id} failed:`, err);
      }
    }

    if (this.#hasRemainingTasks()) {
      this.#frameId = requestAnimationFrame((ts) => this.#processQueue(ts));
    } else {
      this.#isRunning = false;
      this.#frameId = null;
    }
  }

  destroy() {
    if (this.#frameId !== null) {
      cancelAnimationFrame(this.#frameId);
    }
    this.#queues = { high: [], normal: [], low: [] };
    this.#isRunning = false;
    this.#frameId = null;
  }
}

// 使用例
const scheduler = new PriorityTaskScheduler(8);

scheduler.schedule({
  id: 'analytics',
  run: () => sendAnalytics(),
}, 'low');

scheduler.schedule({
  id: 'render-update',
  run: () => updateCriticalUI(),
}, 'high');

const taskId = scheduler.schedule({
  id: 'prefetch',
  run: () => prefetchNextPage(),
}, 'normal');

// 不要になったらキャンセル
scheduler.cancel(taskId);
```

**設計ポイント**:
- 優先度の高いキューから順にタスクを取得する（high → normal → low）
- `performance.now()` でフレーム予算の残りを確認し、超過前にループを抜ける
- rAF を使うことでフレーム単位の処理サイクルを実現する
- `destroy()` メソッドでリソースを適切に解放する
</details>

---

## 13. FAQ

### Q1: マクロタスクとマイクロタスクの実行順序の違いは？

**回答**: マクロタスクとマイクロタスクは、イベントループにおいて異なるタイミングで処理される。

**マクロタスク（Task）**:
- `setTimeout`、`setInterval`、I/O、UI イベントなどで生成される
- イベントループは1回のループで**1つのマクロタスクのみ**を実行する
- マクロタスク間にはレンダリング更新の機会が挟まる
- タスクキューは複数存在する場合があり、ブラウザが優先度を決定する

**マイクロタスク（Microtask）**:
- `Promise.then`、`queueMicrotask`、`MutationObserver` などで生成される
- マクロタスク完了後、マイクロタスクキューが**空になるまで全て**実行される
- マイクロタスクの実行中に新たなマイクロタスクが追加された場合、それも同じチェックポイント内で実行される
- レンダリング更新はマイクロタスクキューが空になるまで行われない

**実行順序の具体例**:

```javascript
console.log('1: 同期コード');

setTimeout(() => console.log('2: マクロタスク'), 0);

Promise.resolve().then(() => {
  console.log('3: マイクロタスク');
  Promise.resolve().then(() => console.log('4: ネストされたマイクロタスク'));
});

queueMicrotask(() => console.log('5: queueMicrotask'));

console.log('6: 同期コード終了');

// 出力順序:
// 1: 同期コード
// 6: 同期コード終了
// 3: マイクロタスク
// 5: queueMicrotask
// 4: ネストされたマイクロタスク
// 2: マクロタスク
```

**重要なポイント**:
- マイクロタスクは現在のタスクの「論理的な延長」として扱われる
- マイクロタスクが大量にキューイングされるとレンダリングがブロックされ、UI が固まる原因となる
- `setTimeout(fn, 0)` でもマイクロタスクより後に実行される

### Q2: ブラウザのイベントループとNode.jsのイベントループの違いは？

**回答**: ブラウザとNode.jsのイベントループは、基本的な概念は共通しているが、構造と動作に重要な違いがある。

**ブラウザのイベントループ**:

1. **タスクソース**: UI イベント、タイマー、ネットワーク、ユーザーインタラクションなど
2. **レンダリング**: タスク実行後、マイクロタスク処理後に、レンダリング更新ステップが挿入される（必要に応じて）
3. **フレームレート**: 通常 60fps（約16.67ms/フレーム）でレンダリングが試行される
4. **優先度**: 複数のタスクキューがあるが、優先度はブラウザ実装に依存

**Node.jsのイベントループ**:

1. **フェーズベース**: イベントループは複数のフェーズ（timers、pending callbacks、idle/prepare、poll、check、close callbacks）に分かれている
2. **各フェーズの処理**: 各フェーズで対応するタスクキューを処理し、フェーズ間でマイクロタスクキューを処理する
3. **`setImmediate` vs `setTimeout`**: Node.js 固有の `setImmediate` は check フェーズで実行され、`setTimeout` は timers フェーズで実行される
4. **レンダリングなし**: Node.js はサーバー環境のため、レンダリングステップが存在しない

**具体的な違いの例**:

```javascript
// ブラウザとNode.jsで動作が異なる例
setTimeout(() => console.log('timeout 1'), 0);
setImmediate(() => console.log('immediate 1'));  // Node.js のみ

// Node.js では実行順序が不定（タイマーの精度による）
// ブラウザでは setImmediate がサポートされていない
```

```javascript
// マイクロタスク処理のタイミング
setTimeout(() => {
  console.log('timeout');
  Promise.resolve().then(() => console.log('microtask in timeout'));
}, 0);

setTimeout(() => console.log('timeout 2'), 0);

// ブラウザ: timeout → microtask in timeout → timeout 2
// Node.js v11+: 同じ（ブラウザ互換の挙動）
// Node.js v10以前: timeout → timeout 2 → microtask in timeout
```

**まとめ**:
- ブラウザのイベントループは「UI レンダリング」を中心に設計されている
- Node.js のイベントループは「I/O 処理の効率」を中心に設計されている
- Node.js v11 以降、ブラウザとの互換性が向上し、マイクロタスクの処理タイミングがブラウザと同等になった

### Q3: 長時間実行されるタスクがUIをブロックする場合の対処法は？

**回答**: 長時間タスク（Long Task）は UI の応答性を低下させる主要な原因である。50ms を超えるタスクは Long Task と見なされ、ユーザー体験に悪影響を与える。以下の対処法がある。

**1. タスク分割（Task Splitting）**

長いタスクを小さなチャンクに分割し、メインスレッドに制御を返す（yield to main thread）。

```javascript
// 悪い例: 長時間ブロック
function processLargeArray(items) {
  items.forEach(item => {
    heavyComputation(item);  // 各処理に10ms かかる
  });
  // 1000件なら10秒間 UI がブロックされる
}

// 良い例: タスク分割
async function processLargeArrayChunked(items, chunkSize = 100) {
  for (let i = 0; i < items.length; i += chunkSize) {
    const chunk = items.slice(i, i + chunkSize);
    chunk.forEach(item => heavyComputation(item));

    // チャンクごとにメインスレッドに制御を返す
    await new Promise(resolve => setTimeout(resolve, 0));
  }
}
```

**2. Web Workers の活用**

CPU 集約的な処理をバックグラウンドスレッドにオフロードする。

```javascript
// メインスレッド
const worker = new Worker('heavy-worker.js');
worker.postMessage({ data: largeDataset });
worker.onmessage = (event) => {
  updateUI(event.data.result);
};

// heavy-worker.js
self.onmessage = (event) => {
  const result = performHeavyComputation(event.data.data);
  self.postMessage({ result });
};
```

**3. requestIdleCallback の活用**

優先度の低い処理をアイドル時間に実行する。

```javascript
function processBackgroundTasks(tasks) {
  function processTasks(deadline) {
    while (deadline.timeRemaining() > 0 && tasks.length > 0) {
      const task = tasks.shift();
      task();
    }

    if (tasks.length > 0) {
      requestIdleCallback(processTasks);
    }
  }

  requestIdleCallback(processTasks);
}

// 使用例
const backgroundTasks = [
  () => preloadImage('img1.jpg'),
  () => preloadImage('img2.jpg'),
  () => buildSearchIndex(),
];
processBackgroundTasks(backgroundTasks);
```

**4. Scheduler API（実験的）**

優先度付きタスクスケジューリングを行う。

```javascript
// 高優先度タスク（ユーザーインタラクション）
scheduler.postTask(() => {
  handleUserClick();
}, { priority: 'user-blocking' });

// 中優先度タスク（レンダリング更新）
scheduler.postTask(() => {
  updateChart();
}, { priority: 'user-visible' });

// 低優先度タスク（アナリティクス送信）
scheduler.postTask(() => {
  sendAnalytics();
}, { priority: 'background' });
```

**5. パフォーマンス測定**

Long Tasks API で長時間タスクを検出する。

```javascript
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.warn('Long task detected:', {
      duration: entry.duration,
      startTime: entry.startTime,
    });
  }
});

observer.observe({ entryTypes: ['longtask'] });
```

**まとめ**:
- タスク分割で UI の応答性を維持する
- CPU 集約的な処理は Web Workers へオフロード
- 低優先度処理は requestIdleCallback でアイドル時間に実行
- Long Tasks API で問題箇所を特定し、継続的に改善する
- 目標: メインスレッドのタスクを50ms以下に保つ

### Q4: async/awaitを使えばイベントループのブロッキングは防げますか?

**回答**: `async/await` は非同期処理を同期的な見た目で書ける構文糖衣であるが、`await` が返すPromiseの解決はマイクロタスクとして処理される。つまり、`await` の前後で処理が分割されるが、各分割された処理自体がCPU集約的であればメインスレッドをブロックする。例えば、`for` ループ内で10万回の計算を行う処理を `async` 関数に入れても、ループ自体は同期的に実行される。ブロッキングを防ぐには、ループ内で明示的に `await new Promise(resolve => setTimeout(resolve, 0))` を挿入してメインスレッドに制御を返すか、Web Workers に処理をオフロードする必要がある。

### Q5: requestAnimationFrame のコールバック内で重い処理を行うとどうなりますか?

**回答**: rAF コールバックはレンダリング直前に実行されるため、コールバック内の処理が16.67msを超えると、そのフレームのレンダリングが遅延し、フレームドロップ（ジャンク）が発生する。rAF コールバック内ではDOM更新やスタイル変更など「レンダリングの準備」に必要な処理のみを行い、データの計算やネットワーク通信などは事前に完了させておくのが原則である。やむを得ず複数フレームにまたがる処理が必要な場合は、処理を小さなチャンクに分割し、各フレームで一部ずつ実行する。

---

## 14. 用語集

| 用語 | 英語 | 説明 |
|------|------|------|
| イベントループ | Event Loop | ブラウザがタスクを協調的に処理するための無限ループ機構 |
| タスク | Task（Macrotask） | setTimeout、I/O、UI イベントなどにより生成される作業単位 |
| マイクロタスク | Microtask | Promise.then や queueMicrotask で生成される高優先度の作業単位 |
| タスクキュー | Task Queue | タスクが順番に格納される FIFO キュー |
| マイクロタスクチェックポイント | Microtask Checkpoint | マイクロタスクキューを全て処理するポイント |
| レンダリングパイプライン | Rendering Pipeline | Style → Layout → Paint → Composite の処理フロー |
| フレーム予算 | Frame Budget | 1フレームに割り当てられる時間（60fps で約 16.67ms） |
| 強制同期レイアウト | Forced Synchronous Layout | DOM 読み取り前に未処理のスタイル変更を強制的にレイアウト計算すること |
| レイアウトスラッシング | Layout Thrashing | 読み取りと書き込みの交互実行による繰り返しレイアウト計算 |
| ジャンク | Jank | フレームドロップによる画面のカクつき |
| コールスタック | Call Stack | 関数呼び出しの履歴を管理するスタック構造 |
| Run-to-completion | Run-to-completion | 1つのタスクが開始されたら完了するまで中断されない性質 |
| Yield | Yield | メインスレッドに制御を返すこと |

---

## まとめ

| 概念 | 実行タイミング | 用途 | 重要な注意点 |
|------|-------------|------|-------------|
| 同期コード | 即座（コールスタック上） | 即時実行が必要な処理 | 長時間実行は UI をブロック |
| マイクロタスク | タスク完了直後、全て実行 | Promise, async/await, MutationObserver | 大量キューイングはレンダリングをブロック |
| タスク（マクロタスク） | 1つずつ、レンダリング機会を挟む | setTimeout, I/O, UI イベント | ネスト制限（4ms）に注意 |
| rAF | レンダリング直前 | アニメーション、DOM バッチ更新 | 内部で重い処理を避ける |
| rIC | アイドル時間 | 低優先度処理 | DOM 操作は禁止、Safari 未サポート |
| scheduler.postTask | 優先度に応じて | 優先度付きタスクスケジューリング | ブラウザサポートが限定的 |

### 設計指針

1. **フレーム予算を意識する**: 1フレーム 16.67ms の予算内で処理を完了する設計を心がける
2. **適切な API を選択する**: 処理の優先度と性質に応じて、マイクロタスク・タスク・rAF・rIC を使い分ける
3. **長時間タスクを分割する**: 50ms を超えるタスクは分割し、`yield to main thread` パターンを適用する
4. **読み取りと書き込みを分離する**: DOM の読み取りと書き込みを交互に行わず、バッチ処理する
5. **測定に基づく最適化**: Long Tasks API や Performance Observer を活用して、ボトルネックを特定する

---

## 次に読むべきガイド

- [Web Workers によるマルチスレッド処理](02-web-workers.md)
- [Service Worker のライフサイクルとイベントループ](03-service-workers.md)
- [ブラウザのレンダリングパイプライン詳細](04-rendering-pipeline.md)

---

## 参考文献

1. WHATWG. "HTML Living Standard -- 8.1.7 Event loops." <https://html.spec.whatwg.org/multipage/webappapis.html#event-loops> (2024)
2. Jake Archibald. "Tasks, microtasks, queues and schedules." <https://jakearchibald.com/2015/tasks-microtasks-queues-and-schedules/> (2015)
3. Philip Roberts. "What the heck is the event loop anyway?" JSConf EU 2014. <https://www.youtube.com/watch?v=8aGhZQkoFbQ>
4. MDN Web Docs. "The event loop." <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Event_loop>
5. MDN Web Docs. "requestAnimationFrame." <https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame>
6. MDN Web Docs. "requestIdleCallback." <https://developer.mozilla.org/en-US/docs/Web/API/Window/requestIdleCallback>
7. W3C. "Long Tasks API." <https://w3c.github.io/longtasks/>
8. Google Developers. "Optimize long tasks." <https://web.dev/optimize-long-tasks/> (2023)

