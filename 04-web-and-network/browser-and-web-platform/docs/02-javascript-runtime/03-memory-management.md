# JavaScriptメモリ管理

> V8ヒープ構造からGCアルゴリズム、メモリリーク検出、Chrome DevToolsプロファイリングまで――ブラウザ上で動作するJavaScriptのメモリ管理を体系的に理解し、高パフォーマンスなWebアプリケーションを構築する技術を身につける。

## この章で学ぶこと

- [ ] JavaScriptのメモリモデル（スタック・ヒープ）を正確に理解する
- [ ] V8エンジンのヒープ構造と世代別GCの動作原理を把握する
- [ ] Mark-Sweep / Mark-Compact / Scavenge の各アルゴリズムを区別できる
- [ ] 典型的なメモリリークパターン6種を識別し、対策を実装できる
- [ ] Chrome DevTools Memory パネルで Heap Snapshot 比較分析ができる
- [ ] WeakRef / WeakMap / FinalizationRegistry を適切に使い分けられる
- [ ] 本番環境でのメモリ監視戦略を設計できる

---

## 前提知識

- V8エンジンの内部構造 → 参照: [V8エンジン](./00-v8-engine.md)
- Web Workers → 参照: [Web Workers](./02-web-workers.md)
- ガベージコレクションの基本概念

---

## 1. JavaScriptのメモリモデル

### 1.1 スタックとヒープの二層構造

JavaScriptエンジンは、メモリを大きく2つの領域に分けて管理する。

```
JavaScriptメモリの二層構造:

  コールスタック (Stack)              ヒープ (Heap)
  ┌──────────────────────┐          ┌───────────────────────────────┐
  │ フレーム: main()     │          │                               │
  │  ├─ x = 42 (数値)   │          │  ┌─────────────────────┐      │
  │  ├─ y = "hello"     │          │  │ Object {a:1, b:2}   │      │
  │  └─ obj ────────────│──────→   │  └─────────────────────┘      │
  │                      │          │                               │
  │ フレーム: doWork()   │          │  ┌─────────────────────┐      │
  │  ├─ i = 0 (数値)    │          │  │ Array [1, 2, 3]     │      │
  │  ├─ arr ────────────│──────→   │  └─────────────────────┘      │
  │  └─ flag = true     │          │                               │
  │                      │          │  ┌─────────────────────┐      │
  │ フレーム: nested()   │          │  │ Function closure     │      │
  │  ├─ temp = 3.14     │          │  │  captured: [arr]     │      │
  │  └─ fn ─────────────│──────→   │  └─────────────────────┘      │
  └──────────────────────┘          │                               │
                                    │  ┌─────────────────────┐      │
  特徴:                             │  │ Map {key → value}    │      │
  - LIFO (Last In, First Out)       │  └─────────────────────┘      │
  - 固定サイズ (通常 1MB 程度)      │                               │
  - 関数終了で自動解放              │  特徴:                        │
  - プリミティブ値を直接格納        │  - 動的サイズ (数MB〜数GB)    │
                                    │  - GCによる自動解放            │
                                    │  - オブジェクト型を格納        │
                                    └───────────────────────────────┘
```

#### スタックに格納されるもの

| データ種別 | 例 | サイズ |
|-----------|-----|--------|
| 数値 (Number) | `42`, `3.14` | 8バイト (IEEE 754) |
| 真偽値 (Boolean) | `true`, `false` | タグ付きポインタ内 |
| null / undefined | `null` | タグ付きポインタ内 |
| BigInt (小さい値) | `42n` | インライン化される場合あり |
| 参照 (ポインタ) | `obj → 0x7ff...` | 8バイト (64bit環境) |

#### ヒープに格納されるもの

| データ種別 | 例 | 補足 |
|-----------|-----|------|
| Object | `{a: 1}` | Hidden Class + プロパティストレージ |
| Array | `[1, 2, 3]` | 内部的には特殊なObject |
| Function | `() => {}` | コードへの参照 + クロージャ環境 |
| String (長い) | `"hello..."` | 一定長以上はヒープ割り当て |
| Map / Set | `new Map()` | ハッシュテーブル構造 |
| RegExp | `/abc/g` | コンパイル済みパターン |
| ArrayBuffer | `new ArrayBuffer(1024)` | 連続メモリブロック |
| DOM Node参照 | `document.getElementById(...)` | C++側オブジェクトへのラッパー |

### 1.2 値のコピーと参照の共有

```javascript
// コード例 1: プリミティブのコピー vs オブジェクトの参照共有

// プリミティブ: 値がコピーされる
let a = 42;
let b = a;      // b は 42 の独立したコピーを持つ
b = 100;
console.log(a); // 42 (a は影響を受けない)

// オブジェクト: 参照が共有される
let obj1 = { name: "Alice", scores: [90, 85, 92] };
let obj2 = obj1;          // obj2 は同じオブジェクトを参照
obj2.name = "Bob";
console.log(obj1.name);   // "Bob" (obj1 も影響を受ける)

// 参照の切断: 新しいオブジェクトを代入
obj2 = { name: "Charlie", scores: [70, 80] };
console.log(obj1.name);   // "Bob" (obj1 は元のオブジェクトを参照し続ける)

// 浅いコピー: スプレッド演算子
let obj3 = { ...obj1 };
obj3.name = "Dave";
console.log(obj1.name);   // "Bob" (プリミティブプロパティは独立)
obj3.scores.push(100);
console.log(obj1.scores); // [90, 85, 92, 100] (ネストされた配列は共有!)

// 深いコピー: structuredClone (推奨)
let obj4 = structuredClone(obj1);
obj4.scores.push(200);
console.log(obj1.scores); // [90, 85, 92, 100] (影響を受けない)
```

### 1.3 GCルートと到達可能性

ガベージコレクション (GC) は「到達可能性 (Reachability)」に基づいてオブジェクトの生死を判定する。GCルートから辿れるオブジェクトは生存、辿れないオブジェクトは回収対象となる。

```
GCルートからの到達可能性判定:

  GCルート
  ├── グローバルオブジェクト (window / globalThis)
  │     ├── window.app ──→ [App Object] ──→ [Config]
  │     └── window.cache ──→ [Cache Map] ──→ [Entry1] ──→ [Data1]
  │                                        └→ [Entry2] ──→ [Data2]
  │
  ├── コールスタック上のローカル変数
  │     ├── localVar ──→ [Temp Object]
  │     └── callback ──→ [Function] ──→ (クロージャ) ──→ [Captured Vars]
  │
  ├── アクティブなタイマー
  │     ├── setInterval(fn, 1000) ──→ [fn] ──→ [Referenced Data]
  │     └── setTimeout(fn, 5000) ──→ [fn]
  │
  └── その他
        ├── Promise の then/catch コールバック
        ├── MutationObserver
        ├── MessagePort
        └── アクティブな EventListener

  到達不能 (GC対象):
  ┌─────────────────────────────────────────────────┐
  │  [Orphan Object]     どのルートからも辿れない     │
  │  [Detached DOM Tree] DOM から除去 & JS参照なし    │
  │  [Old Closure Data]  関数実行完了 & 参照消失      │
  └─────────────────────────────────────────────────┘
```

---

## 2. V8エンジンのヒープ構造

### 2.1 V8ヒープの内部構成

V8エンジン (Chrome, Node.js で使用) は、ヒープ領域を複数のスペースに分割して管理する。この構造が世代別GCの基盤となる。

```
V8 ヒープ構造 (詳細):

  ┌──────────────────────────────────────────────────────────────┐
  │                        V8 Heap                               │
  │                                                              │
  │  ┌─────────────────────────────────────────────────────┐     │
  │  │              New Space (Young Generation)            │     │
  │  │  ┌──────────────────┐  ┌──────────────────┐         │     │
  │  │  │   Semi-Space A   │  │   Semi-Space B   │         │     │
  │  │  │   (From-Space)   │  │   (To-Space)     │         │     │
  │  │  │                  │  │                  │         │     │
  │  │  │  新規オブジェクト │  │  Scavenge後の    │         │     │
  │  │  │  が割り当てられる │  │  生存者が移動    │         │     │
  │  │  │                  │  │                  │         │     │
  │  │  │  サイズ: 1〜8 MB │  │  サイズ: 1〜8 MB │         │     │
  │  │  └──────────────────┘  └──────────────────┘         │     │
  │  └─────────────────────────────────────────────────────┘     │
  │                                                              │
  │  ┌─────────────────────────────────────────────────────┐     │
  │  │              Old Space (Old Generation)              │     │
  │  │                                                     │     │
  │  │  ┌──────────────────────────────────────────┐       │     │
  │  │  │  Old Pointer Space                       │       │     │
  │  │  │  他オブジェクトへの参照を含むオブジェクト │       │     │
  │  │  └──────────────────────────────────────────┘       │     │
  │  │                                                     │     │
  │  │  ┌──────────────────────────────────────────┐       │     │
  │  │  │  Old Data Space                          │       │     │
  │  │  │  プリミティブデータのみのオブジェクト     │       │     │
  │  │  │  (文字列、ボックス化された数値など)       │       │     │
  │  │  └──────────────────────────────────────────┘       │     │
  │  │                                                     │     │
  │  │  サイズ: 数百MB〜数GB (--max-old-space-size)        │     │
  │  └─────────────────────────────────────────────────────┘     │
  │                                                              │
  │  ┌──────────────────┐  ┌──────────────────┐                  │
  │  │ Large Object     │  │ Code Space       │                  │
  │  │ Space            │  │                  │                  │
  │  │ サイズ > 閾値の  │  │ JITコンパイル    │                  │
  │  │ オブジェクト      │  │ 済みコード       │                  │
  │  │ (配列等)         │  │                  │                  │
  │  └──────────────────┘  └──────────────────┘                  │
  │                                                              │
  │  ┌──────────────────┐  ┌──────────────────┐                  │
  │  │ Map Space        │  │ Cell Space       │                  │
  │  │                  │  │                  │                  │
  │  │ Hidden Class     │  │ Cell / Property  │                  │
  │  │ (Map) 構造体     │  │ Cell             │                  │
  │  └──────────────────┘  └──────────────────┘                  │
  └──────────────────────────────────────────────────────────────┘
```

### 2.2 各スペースの役割と特性

| スペース | 役割 | 典型サイズ | GC方式 | 頻度 |
|---------|------|-----------|--------|------|
| New Space (Semi-Space A/B) | 新規オブジェクトの割り当て | 1〜8 MB | Scavenge (コピーGC) | 高頻度 (ms単位) |
| Old Pointer Space | 長寿命オブジェクト (参照含む) | 数百MB〜 | Mark-Sweep / Mark-Compact | 低頻度 |
| Old Data Space | 長寿命データ (参照なし) | 数十MB〜 | Mark-Sweep / Mark-Compact | 低頻度 |
| Large Object Space | 巨大オブジェクト (閾値超え) | 可変 | Mark-Sweep | 低頻度 |
| Code Space | JITコンパイル済みコード | 数十MB | 特殊なGC | 低頻度 |
| Map Space | Hidden Class (Map) | 数MB | Mark-Sweep | 低頻度 |

### 2.3 オブジェクトのライフサイクル

```javascript
// コード例 2: オブジェクトのライフサイクル追跡

function demonstrateLifecycle() {
  // Phase 1: New Space に割り当て
  const shortLived = { type: "temporary", data: new Array(100) };
  // → shortLived は New Space (Semi-Space A) に配置される

  // Phase 2: 関数終了で shortLived は到達不能に
  // → 次の Scavenge で回収される (New Space 内で完結)

  // Phase 3: 長寿命オブジェクトは Old Space へ昇格
  const cache = new Map();
  for (let i = 0; i < 10000; i++) {
    cache.set(i, { value: i * 2, label: `item-${i}` });
  }
  // → cache と内部オブジェクトは複数回の Scavenge を生き延び
  //   Old Space (Old Pointer Space) へ昇格 (promotion)

  return cache; // cache は呼び出し元で参照され続ける
}

// グローバルスコープで保持 → GCルートから到達可能
const globalCache = demonstrateLifecycle();

// 明示的に参照を切る → GC対象になる
// globalCache = null;
```

### 2.4 Hidden Class (Map) とインラインキャッシュ

V8はオブジェクトの「形状 (shape)」を Hidden Class (内部的には Map と呼ばれる) で管理する。同じ順序で同じプロパティを持つオブジェクトは同一の Hidden Class を共有し、プロパティアクセスが高速化される。

```javascript
// コード例 3: Hidden Class とメモリ効率

// 良い例: 同一の Hidden Class を共有
function createPoint(x, y) {
  // 全てのオブジェクトが同じ順序でプロパティを持つ
  return { x: x, y: y };
}

const points = [];
for (let i = 0; i < 10000; i++) {
  points.push(createPoint(i, i * 2));
}
// → 10000個のオブジェクトが1つの Hidden Class を共有
// → メモリ効率が良い (Hidden Class は1つだけ)

// 悪い例: 異なる Hidden Class が大量生成される
const badPoints = [];
for (let i = 0; i < 10000; i++) {
  const p = {};
  if (i % 2 === 0) {
    p.x = i;
    p.y = i * 2;
  } else {
    p.y = i * 2;  // プロパティ追加順が異なる!
    p.x = i;
  }
  badPoints.push(p);
}
// → 2種類の Hidden Class が生成される
// → インラインキャッシュのミスが発生し、プロパティアクセスが低速化

// 最悪の例: delete による Hidden Class の退化
const obj = { a: 1, b: 2, c: 3 };
delete obj.b;
// → Hidden Class が「辞書モード (slow mode)」に退化
// → ハッシュテーブルベースのルックアップになり低速化
// 対策: delete の代わりに undefined を代入する
// obj.b = undefined; // Hidden Class を壊さない
```

---

## 3. GCアルゴリズム詳解

### 3.1 Minor GC: Scavenge (コピーGC)

New Space で実行される高速なGCアルゴリズム。Cheney のコピーGCアルゴリズムに基づく。

```
Scavenge アルゴリズムの動作:

  === 初期状態 ===
  From-Space (Semi-Space A):          To-Space (Semi-Space B):
  ┌────────────────────────────┐      ┌────────────────────────────┐
  │ [Obj-A] [Dead] [Obj-B]    │      │         (空)               │
  │ [Dead] [Obj-C] [Dead]     │      │                            │
  │ [Obj-D] [Dead] [Dead]     │      │                            │
  └────────────────────────────┘      └────────────────────────────┘

  === Scavenge 実行 ===
  1. GCルートから From-Space 内の到達可能オブジェクトを走査
  2. 到達可能オブジェクトを To-Space にコピー
  3. すでに1回以上 Scavenge を生き延びたオブジェクトは Old Space へ昇格

  From-Space:                          To-Space:
  ┌────────────────────────────┐      ┌────────────────────────────┐
  │ (全てのデータは破棄)       │      │ [Obj-A'] [Obj-B'] [Obj-C']│
  │                            │      │                            │
  │                            │      │                            │
  └────────────────────────────┘      └────────────────────────────┘
                                            ↑ コンパクション済み
                                            (断片化なし)
  [Obj-D] → Old Space へ昇格 (2回目の Scavenge 生存)

  === ロール交代 ===
  旧 To-Space が新しい From-Space になる
  旧 From-Space が新しい To-Space になる (次回の Scavenge 用)
```

**Scavenge の特性:**

| 特性 | 値 |
|------|-----|
| 停止時間 | 通常 1〜10 ms |
| 対象空間 | New Space のみ (1〜8 MB) |
| アルゴリズム | コピーGC (Cheney) |
| コンパクション | コピー時に自動的に実施 |
| 空間効率 | 50% (2つの Semi-Space を使用) |
| 昇格条件 | 1回以上の Scavenge を生存 or To-Space の 25% 超過 |

### 3.2 Major GC: Mark-Sweep と Mark-Compact

Old Space で実行される、より大規模なGC。Mark-Sweep (マーク&スイープ) と Mark-Compact (マーク&コンパクト) の2つのフェーズから構成される。

#### Mark フェーズ

```
Mark フェーズ (三色マーキング):

  色の意味:
  - 白 (White): 未訪問 (GC開始時、全オブジェクトが白)
  - 灰 (Gray):  訪問済みだが子の走査が未完了
  - 黒 (Black): 訪問済みかつ子の走査も完了

  === 手順 ===

  Step 1: GCルートの直接の子を灰色にマーク
  ┌─────┐
  │Root │──→ [Obj-A: 灰] ──→ [Obj-B: 白]
  │     │──→ [Obj-C: 灰]     [Obj-D: 白] (孤立)
  └─────┘

  Step 2: 灰色オブジェクトの子を走査し、自身を黒にする
  ┌─────┐
  │Root │──→ [Obj-A: 黒] ──→ [Obj-B: 灰]
  │     │──→ [Obj-C: 黒]     [Obj-D: 白] (孤立)
  └─────┘

  Step 3: 灰色がなくなるまで繰り返す
  ┌─────┐
  │Root │──→ [Obj-A: 黒] ──→ [Obj-B: 黒]
  │     │──→ [Obj-C: 黒]     [Obj-D: 白 → 回収!]
  └─────┘

  結果: 白いままのオブジェクト = 到達不能 = GC対象
```

#### Sweep フェーズと Compact フェーズ

**Mark-Sweep** はマーク済みでない (白い) オブジェクトのメモリを解放する。高速だが断片化が発生する。

**Mark-Compact** は生存オブジェクトをメモリの一端に寄せることで断片化を解消する。Sweep より低速だが、大きなオブジェクトの割り当てに必要な連続領域を確保できる。

### 3.3 インクリメンタルマーキングと並行GC

V8は停止時間を短縮するため、複数の最適化技法を組み合わせている。

| 技法 | 説明 | 効果 |
|------|------|------|
| インクリメンタルマーキング | Mark フェーズを小さなステップに分割し、アプリケーション実行と交互に行う | 長い停止時間を回避 |
| 並行マーキング (Concurrent) | ワーカースレッドでマーキングを並行実行 | メインスレッドの停止時間を短縮 |
| 並列マーキング (Parallel) | 複数スレッドで同時にマーキング | マーキング処理自体の高速化 |
| 遅延スイープ (Lazy Sweeping) | Sweep を必要になるまで遅延 | 停止時間の分散 |
| 並行コンパクション | メモリの移動を並行実行 | コンパクション時の停止短縮 |

```javascript
// コード例 4: GC停止時間の測定

// Performance API を使ったGC停止時間の間接測定
function measureGCPauses(durationMs = 5000) {
  const pauses = [];
  let lastTime = performance.now();
  const threshold = 5; // 5ms 以上の停止をGCと推定

  const intervalId = setInterval(() => {
    const now = performance.now();
    const elapsed = now - lastTime;

    if (elapsed > threshold) {
      pauses.push({
        timestamp: now,
        duration: elapsed.toFixed(2) + "ms",
        likely: elapsed > 50 ? "Major GC" : "Minor GC"
      });
    }
    lastTime = now;
  }, 1);

  setTimeout(() => {
    clearInterval(intervalId);
    console.table(pauses);
    console.log(`検出された停止: ${pauses.length}回`);
  }, durationMs);
}

// 使用例:
// measureGCPauses(10000); // 10秒間監視
```

### 3.4 Orinoco: V8の最新GCアーキテクチャ

V8のGCサブシステム「Orinoco」は、以下の設計方針で進化を続けている。

1. **並行 (Concurrent)**: メインスレッドを停止せずにバックグラウンドでGC作業を実行
2. **並列 (Parallel)**: 複数のヘルパースレッドでGC作業を分担
3. **インクリメンタル (Incremental)**: GC作業を小さなチャンクに分割し、段階的に処理

これらの組み合わせにより、Major GC でも停止時間は数ミリ秒程度に抑えられる。

---

## 4. メモリリークの典型パターンと対策

### 4.1 パターン一覧

| # | パターン | 重篤度 | 検出難易度 | 主な対策 |
|---|---------|--------|-----------|---------|
| 1 | グローバル変数の意図しない生成 | 中 | 易 | `"use strict"`, ESLint no-implicit-globals |
| 2 | タイマーのクリア忘れ | 高 | 中 | `clearInterval` / `clearTimeout` |
| 3 | イベントリスナーの解除忘れ | 高 | 中 | `removeEventListener` / `AbortController` |
| 4 | クロージャによる意図しない参照保持 | 高 | 難 | スコープの最小化、null代入 |
| 5 | 切り離されたDOMツリー (Detached DOM) | 高 | 中 | JS側の参照をnullに |
| 6 | コンソールログでのオブジェクト保持 | 低 | 易 | 本番ではログを除去 |

### 4.2 パターン 1: グローバル変数の意図しない生成

```javascript
// アンチパターン 1: 暗黙のグローバル変数

function processData(items) {
  // "use strict" がない場合、result がグローバル変数になる
  result = items.map(item => item.value * 2); // var/let/const がない!

  // さらに危険: this が window を指すケース
  this.accumulatedData = new Array(100000).fill(0);
  // → window.accumulatedData としてグローバルに残り続ける
}

// 対策: strict mode + 適切な変数宣言
"use strict";

function processDataSafe(items) {
  const result = items.map(item => item.value * 2); // ブロックスコープ
  return result;
}
```

### 4.3 パターン 2: タイマーのクリア忘れ

```javascript
// アンチパターン 2: SPA でのタイマーリーク

class DataPollingWidget {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.data = null;

    // 問題: コンポーネント破棄時にクリアされない
    this.intervalId = setInterval(async () => {
      const response = await fetch(this.endpoint);
      this.data = await response.json(); // 大量のデータを蓄積
      this.render();
    }, 5000);
  }

  render() {
    // UIの更新処理
  }

  // 修正: 明示的なクリーンアップメソッド
  destroy() {
    clearInterval(this.intervalId);
    this.intervalId = null;
    this.data = null;
  }
}

// React での正しいパターン
function useDataPolling(endpoint, intervalMs = 5000) {
  const [data, setData] = React.useState(null);

  React.useEffect(() => {
    const controller = new AbortController();

    const poll = async () => {
      try {
        const response = await fetch(endpoint, {
          signal: controller.signal
        });
        const json = await response.json();
        setData(json);
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error("Polling error:", err);
        }
      }
    };

    const id = setInterval(poll, intervalMs);
    poll(); // 初回即時実行

    // クリーンアップ: アンマウント時に確実に停止
    return () => {
      clearInterval(id);
      controller.abort();
    };
  }, [endpoint, intervalMs]);

  return data;
}
```

### 4.4 パターン 3: イベントリスナーの解除忘れ

```javascript
// 問題のあるパターン: リスナーが蓄積する
class ScrollTracker {
  constructor() {
    this.positions = [];

    // 問題: 匿名関数なので removeEventListener できない
    window.addEventListener("scroll", () => {
      this.positions.push({
        y: window.scrollY,
        time: Date.now()
      });
    });
  }
}

// 修正版 1: 名前付き関数 + removeEventListener
class ScrollTrackerFixed {
  constructor() {
    this.positions = [];
    this.handleScroll = this.handleScroll.bind(this);
    window.addEventListener("scroll", this.handleScroll);
  }

  handleScroll() {
    this.positions.push({ y: window.scrollY, time: Date.now() });
  }

  destroy() {
    window.removeEventListener("scroll", this.handleScroll);
    this.positions = [];
  }
}

// 修正版 2: AbortController による一括管理 (推奨)
class ScrollTrackerModern {
  constructor() {
    this.positions = [];
    this.abortController = new AbortController();

    window.addEventListener("scroll", () => {
      this.positions.push({ y: window.scrollY, time: Date.now() });
    }, { signal: this.abortController.signal });

    window.addEventListener("resize", () => {
      this.positions = []; // リサイズ時にリセット
    }, { signal: this.abortController.signal });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.flush();
    }, { signal: this.abortController.signal });
  }

  flush() {
    // サーバーに送信するなどの処理
    this.positions = [];
  }

  destroy() {
    // 全てのリスナーを一括解除
    this.abortController.abort();
    this.positions = [];
  }
}
```

### 4.5 パターン 4: クロージャによる意図しない参照保持

クロージャは外側のスコープの変数を「キャプチャ」する。V8は最適化により使用されない変数をクロージャから除外するが、特定の条件下では大きなオブジェクトが保持され続ける。

```javascript
// エッジケース 1: eval が存在するとクロージャの最適化が無効化される

function createProcessor() {
  const hugeBuffer = new ArrayBuffer(100 * 1024 * 1024); // 100MB
  const metadata = { created: Date.now(), type: "buffer" };

  // hugeBuffer を使わないが、eval があるため全変数がキャプチャされる
  return function process(code) {
    // eval が存在すると、V8 はどの変数が使われるか静的に判定できない
    // そのため、スコープ内の全変数を保持してしまう
    return eval(code); // hugeBuffer も保持される!
  };
}

// 対策: クロージャのスコープを明示的に制限する
function createProcessorFixed() {
  const hugeBuffer = new ArrayBuffer(100 * 1024 * 1024);
  const result = processBuffer(hugeBuffer);

  // hugeBuffer を使い終わったら、別スコープで関数を作成
  return createCallback(result);
}

function createCallback(processedResult) {
  // このクロージャは processedResult のみをキャプチャ
  return function() {
    return processedResult;
  };
}
```

```javascript
// エッジケース 2: 複数クロージャが同一スコープを共有する場合

function setupHandlers() {
  const largeData = new Array(1000000).fill("data");
  const config = { debug: true };

  // handler1 は largeData を使う
  const handler1 = () => {
    console.log(largeData.length);
  };

  // handler2 は largeData を使わない
  // しかし、V8 の実装によっては handler1 と同じ Context オブジェクトを共有
  // → largeData への参照が handler2 にも残る可能性がある
  const handler2 = () => {
    console.log(config.debug);
  };

  // handler1 を破棄しても handler2 が生きていれば
  // largeData が解放されない可能性がある
  return { handler1, handler2 };
}

// 対策: 関数を別々のスコープで定義する
function setupHandlersSafe() {
  const handler1 = createHandler1();
  const handler2 = createHandler2();
  return { handler1, handler2 };
}

function createHandler1() {
  const largeData = new Array(1000000).fill("data");
  return () => console.log(largeData.length);
}

function createHandler2() {
  const config = { debug: true };
  return () => console.log(config.debug);
}
```

### 4.6 パターン 5: 切り離されたDOMツリー (Detached DOM)

SPAにおいて最も発生頻度が高いリークパターンの一つ。DOMノードがドキュメントツリーから除去されても、JavaScript側の参照が残っていると GC されない。

```
Detached DOM ツリーのリークメカニズム:

  === DOMツリーに接続中 ===
  document
  └── body
      └── #container
          ├── .card-1 ←─── JS: this.cards[0]
          ├── .card-2 ←─── JS: this.cards[1]
          └── .card-3 ←─── JS: this.cards[2]

  === #container を DOM から除去 ===
  document
  └── body
      (空)

  切り離された DOM ツリー (Detached):
  #container          ← GC したいが...
  ├── .card-1 ←────── JS: this.cards[0] がまだ参照!
  ├── .card-2 ←────── JS: this.cards[1] がまだ参照!
  └── .card-3 ←────── JS: this.cards[2] がまだ参照!

  → this.cards 配列が参照を保持しているため
    #container 以下のツリー全体が GC されない
  → DevTools の Heap Snapshot で "Detached" と表示される
```

```javascript
// コード例 5: Detached DOM の検出と修正

// 問題のあるコード
class CardList {
  constructor(container) {
    this.container = container;
    this.cards = [];
    this.listeners = new Map();
  }

  addCard(data) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <h3>${data.title}</h3>
      <p>${data.description}</p>
      <button class="delete-btn">削除</button>
    `;

    const deleteBtn = card.querySelector(".delete-btn");
    const handler = () => this.removeCard(card);
    deleteBtn.addEventListener("click", handler);

    // 問題: cards 配列とリスナーMap に参照を保持
    this.cards.push(card);
    this.listeners.set(card, { element: deleteBtn, handler });

    this.container.appendChild(card);
  }

  removeCard(card) {
    this.container.removeChild(card);
    // ここで cards 配列から除去しないと Detached DOM リーク!
  }

  // 修正版: 完全なクリーンアップ
  removeCardFixed(card) {
    // 1. イベントリスナーの解除
    const listenerInfo = this.listeners.get(card);
    if (listenerInfo) {
      listenerInfo.element.removeEventListener("click", listenerInfo.handler);
      this.listeners.delete(card);
    }

    // 2. 配列からの除去
    const index = this.cards.indexOf(card);
    if (index !== -1) {
      this.cards.splice(index, 1);
    }

    // 3. DOM からの除去
    card.remove();
  }

  // 全体のクリーンアップ
  destroy() {
    // 全カードのクリーンアップ
    for (const card of [...this.cards]) {
      this.removeCardFixed(card);
    }
    this.cards = [];
    this.listeners.clear();
    this.container = null;
  }
}
```

### 4.7 パターン 6: console.log によるオブジェクト保持

開発時に見落としがちなリークパターン。`console.log` に渡されたオブジェクトは DevTools がそのオブジェクトを表示するために参照を保持する。

```javascript
// console.log によるリーク
function processLargeData() {
  const data = generateHugeDataset(); // 数十MBのデータ

  console.log("Processing data:", data); // DevTools がdata を保持!

  const result = transform(data);

  // data は本来ここで GC 対象になるべきだが、
  // console.log が参照を保持し続ける

  return result;
}

// 対策 1: 本番環境ではログを除去
// webpack / vite の設定で console.log を strip

// 対策 2: 必要な情報のみをログ出力
function processLargeDataFixed() {
  const data = generateHugeDataset();

  console.log("Processing data, size:", data.length); // サイズのみ

  const result = transform(data);
  return result;
}

// 対策 3: 条件付きロギング
const IS_DEV = process.env.NODE_ENV === "development";

function debugLog(label, data) {
  if (IS_DEV) {
    console.log(label, typeof data === "object" ? JSON.stringify(data).slice(0, 200) : data);
  }
}
```

---

## 5. WeakRef / WeakMap / WeakSet / FinalizationRegistry

### 5.1 弱参照の概念

通常の参照 (強参照) はオブジェクトの GC を妨げるが、弱参照 (Weak Reference) はGCを妨げない。オブジェクトへの弱参照だけが残っている場合、そのオブジェクトはGC対象となる。

| 型 | キーの参照 | 値の参照 | GC への影響 | イテレーション |
|-----|-----------|---------|-----------|-------------|
| Map | 強参照 | 強参照 | キーも値もGCを妨げる | 可能 |
| WeakMap | 弱参照 | 強参照 | キーがGCされるとエントリ削除 | 不可 |
| Set | - | 強参照 | 値のGCを妨げる | 可能 |
| WeakSet | - | 弱参照 | 値のGCを妨げない | 不可 |
| WeakRef | - | 弱参照 | 対象のGCを妨げない | - |

### 5.2 WeakMap を使ったキャッシュの実装

```javascript
// コード例 6: WeakMap によるメモリ安全なキャッシュ

// 問題: Map を使うと、キーオブジェクトが他で不要になっても解放されない
class UnsafeCache {
  constructor() {
    this.cache = new Map();
  }

  compute(obj) {
    if (this.cache.has(obj)) {
      return this.cache.get(obj);
    }
    const result = expensiveComputation(obj);
    this.cache.set(obj, result);
    return result;
  }
  // obj がどこでも不要になっても、this.cache が参照を保持 → リーク
}

// 修正: WeakMap を使うとキーが GC されたとき自動的にエントリが削除される
class SafeCache {
  constructor() {
    this.cache = new WeakMap();
  }

  compute(obj) {
    if (this.cache.has(obj)) {
      return this.cache.get(obj);
    }
    const result = expensiveComputation(obj);
    this.cache.set(obj, result);
    return result;
  }
  // obj が他のどこからも参照されなくなれば、エントリも自動削除
}

// 実践例: DOM要素に関連データを紐付ける
const elementData = new WeakMap();

function attachData(element, data) {
  elementData.set(element, data);
  // element が DOM から除去され、他の参照もなくなれば
  // data も自動的に GC される
}

function getData(element) {
  return elementData.get(element);
}
```

### 5.3 WeakRef と FinalizationRegistry

```javascript
// コード例 7: WeakRef を使ったサイズ制限なしキャッシュ

class WeakCache {
  constructor() {
    this.cache = new Map(); // key: string, value: WeakRef<Object>
    this.registry = new FinalizationRegistry((key) => {
      // オブジェクトがGCされたら、Mapからエントリを削除
      const ref = this.cache.get(key);
      if (ref && ref.deref() === undefined) {
        this.cache.delete(key);
        console.log(`Cache entry "${key}" was cleaned up by GC`);
      }
    });
  }

  set(key, value) {
    // 既存のエントリがあれば FinalizationRegistry から登録解除
    const existingRef = this.cache.get(key);
    if (existingRef) {
      this.registry.unregister(existingRef);
    }

    const ref = new WeakRef(value);
    this.cache.set(key, ref);

    // GC時の通知を登録 (unregister用のトークンとして ref を使用)
    this.registry.register(value, key, ref);
  }

  get(key) {
    const ref = this.cache.get(key);
    if (!ref) return undefined;

    const value = ref.deref();
    if (value === undefined) {
      // すでにGCされている → エントリを削除
      this.cache.delete(key);
      return undefined;
    }
    return value;
  }

  get size() {
    // 実際の生存エントリ数 (GC済みは含まない)
    let count = 0;
    for (const [key, ref] of this.cache) {
      if (ref.deref() !== undefined) {
        count++;
      } else {
        this.cache.delete(key);
      }
    }
    return count;
  }
}

// 使用例
const imageCache = new WeakCache();

async function loadImage(url) {
  let image = imageCache.get(url);
  if (image) {
    console.log("Cache hit:", url);
    return image;
  }

  console.log("Cache miss, loading:", url);
  image = await fetchAndDecodeImage(url);
  imageCache.set(url, image);
  return image;
}
```

> **注意**: FinalizationRegistry のコールバックは GC のタイミングに依存し、実行が保証されない。リソースの確実な解放には明示的な `dispose()` / `close()` メソッドを使用すること。FinalizationRegistry はセーフティネットとして位置づける。

### 5.4 Symbol.dispose と using 宣言 (TC39 Stage 3→4)

ECMAScript の Explicit Resource Management 提案では、`Symbol.dispose` と `using` 宣言によりリソースの確実な解放を言語レベルでサポートする。

```javascript
// コード例 8: Explicit Resource Management (using 宣言)

class DatabaseConnection {
  #connection;

  constructor(url) {
    this.#connection = connect(url);
  }

  query(sql) {
    return this.#connection.execute(sql);
  }

  // Symbol.dispose を実装 → using 宣言で自動解放
  [Symbol.dispose]() {
    this.#connection.close();
    this.#connection = null;
    console.log("Connection disposed");
  }
}

// using 宣言: スコープ終了時に自動的に dispose が呼ばれる
async function fetchUserData(userId) {
  using db = new DatabaseConnection("postgres://localhost/mydb");

  const user = await db.query(`SELECT * FROM users WHERE id = ${userId}`);
  const orders = await db.query(`SELECT * FROM orders WHERE user_id = ${userId}`);

  return { user, orders };
  // ← ここで自動的に db[Symbol.dispose]() が呼ばれる
  // 例外が発生しても確実に呼ばれる (try-finally と同等)
}

// 非同期リソース用: Symbol.asyncDispose + await using
class StreamProcessor {
  #stream;

  constructor(stream) {
    this.#stream = stream;
  }

  async [Symbol.asyncDispose]() {
    await this.#stream.close();
    console.log("Stream closed");
  }
}

async function processFile(path) {
  await using processor = new StreamProcessor(openFile(path));
  // ... 処理 ...
  // スコープ終了時に await processor[Symbol.asyncDispose]() が呼ばれる
}
```

---

## 6. Chrome DevTools によるメモリプロファイリング

### 6.1 Memory パネルの概要

Chrome DevTools の Memory パネルは3つの主要なプロファイリング手法を提供する。

| 手法 | 目的 | オーバーヘッド | 適したシナリオ |
|------|------|-------------|-------------|
| Heap Snapshot | ある時点の全オブジェクトを記録 | 高 (一時停止あり) | リーク箇所の特定、オブジェクト保持チェーンの分析 |
| Allocation Timeline | 時間軸でのメモリ割り当て記録 | 中 | いつメモリが割り当てられたかの特定 |
| Allocation Sampling | サンプリングベースの割り当て記録 | 低 | 長時間のプロファイリング、本番に近い環境 |

### 6.2 Heap Snapshot の操作手順

```
Heap Snapshot によるリーク検出 (ステップバイステップ):

  ┌────────────────────────────────────────────────────────┐
  │ Step 1: 準備                                          │
  │  - Chrome DevTools を開く (F12 / Cmd+Opt+I)           │
  │  - Memory タブを選択                                   │
  │  - "Heap snapshot" ラジオボタンを選択                   │
  └───────────────────────────┬────────────────────────────┘
                              ↓
  ┌────────────────────────────────────────────────────────┐
  │ Step 2: ベースラインスナップショット                    │
  │  - GCを強制実行 (ゴミ箱アイコンをクリック)              │
  │  - "Take snapshot" ボタンをクリック                     │
  │  - → Snapshot 1 が記録される                           │
  └───────────────────────────┬────────────────────────────┘
                              ↓
  ┌────────────────────────────────────────────────────────┐
  │ Step 3: リークが疑われる操作を実行                     │
  │  - ページ遷移、ダイアログ開閉、データ読み込み等         │
  │  - 操作前の状態に戻す (ダイアログを閉じる等)           │
  └───────────────────────────┬────────────────────────────┘
                              ↓
  ┌────────────────────────────────────────────────────────┐
  │ Step 4: 比較用スナップショット                         │
  │  - GCを強制実行                                        │
  │  - "Take snapshot" → Snapshot 2                        │
  └───────────────────────────┬────────────────────────────┘
                              ↓
  ┌────────────────────────────────────────────────────────┐
  │ Step 5: 比較分析                                       │
  │  - Snapshot 2 を選択                                    │
  │  - ビューを "Comparison" に切り替え                     │
  │  - "Compared to Snapshot 1" を選択                      │
  │  - "#Delta" 列でソート → 正の値 = リーク候補           │
  │  - "#New" 列 = 新規割り当てされたオブジェクト数        │
  │  - "#Deleted" 列 = GCされたオブジェクト数              │
  │  - "#Delta" = #New - #Deleted                          │
  └───────────────────────────┬────────────────────────────┘
                              ↓
  ┌────────────────────────────────────────────────────────┐
  │ Step 6: 保持チェーンの調査                             │
  │  - リーク候補のオブジェクトをクリック                    │
  │  - 下部の "Retainers" パネルで保持者を確認              │
  │  - ルートまでのチェーンを辿り、リーク原因を特定         │
  └────────────────────────────────────────────────────────┘
```

### 6.3 Heap Snapshot のビューモード

| ビュー | 表示内容 | 用途 |
|--------|---------|------|
| Summary | コンストラクタ名でグループ化 | オブジェクト種別ごとのメモリ消費把握 |
| Comparison | 2つのスナップショットの差分 | リーク特定に最適 |
| Containment | オブジェクトの包含関係 | ヒープ構造の理解 |
| Statistics | メモリ種別の円グラフ | 全体像の把握 |

#### Summary ビューの重要カラム

| カラム | 意味 |
|--------|------|
| Constructor | オブジェクトのコンストラクタ名 |
| Distance | GCルートからの最短距離 |
| Shallow Size | オブジェクト自体のサイズ (バイト) |
| Retained Size | オブジェクトとその排他的依存先の合計サイズ |

**Shallow Size vs Retained Size の違い:**

```
例: オブジェクト A が B, C を排他的に保持し、D は A と E の両方が保持

  [A] ──→ [B] ──→ [C]
   │
   └──→ [D] ←── [E]

  A の Shallow Size  = A 自体のサイズ (例: 64 bytes)
  A の Retained Size = A + B + C のサイズ合計 (例: 64 + 128 + 256 = 448 bytes)
                       D は E からも参照されているため含まない
```

### 6.4 Allocation Timeline の活用

Allocation Timeline は時間軸でメモリ割り当てを追跡し、どの操作がメモリ消費の原因かを特定するのに有効。

```
Allocation Timeline の読み方:

  時間 →
  ┌──────────────────────────────────────────────────────┐
  │ ████                                                  │ ← 青: 生存中
  │ ░░░░██████                                            │ ← 灰: GC済み
  │         ░░░░████████████████████████████              │ ← 長い青 = リーク候補
  │              ░░░░░░░░                                 │
  │                    ████                               │
  │                         ░░░░██████████████████████    │ ← リーク候補
  │                              ░░░░░░░░                 │
  └──────────────────────────────────────────────────────┘
    ↑ ボタンクリック      ↑ ページ遷移     ↑ データ読み込み

  青いバーが記録終了時まで残っている = 生存オブジェクト
  → 本来解放されるべきなのに残っている = リーク
  → バーをクリックすると該当オブジェクトの詳細が表示される
```

### 6.5 Performance.memory API と measureUserAgentSpecificMemory

```javascript
// コード例 9: ブラウザ上でのメモリ使用量モニタリング

// Performance.memory API (Chrome 限定、非推奨傾向)
function getMemoryInfo() {
  if (performance.memory) {
    return {
      usedHeapMB: (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2),
      totalHeapMB: (performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(2),
      limitMB: (performance.memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2),
      usagePercent: (
        (performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit) * 100
      ).toFixed(1)
    };
  }
  return null;
}

// より新しい API: performance.measureUserAgentSpecificMemory()
// Cross-Origin Isolation が必要 (COOP + COEP ヘッダー)
async function measureMemory() {
  if (performance.measureUserAgentSpecificMemory) {
    try {
      const result = await performance.measureUserAgentSpecificMemory();
      console.log("Total bytes:", result.bytes);
      for (const breakdown of result.breakdown) {
        console.log(
          `  ${breakdown.types.join(", ")}: ${(breakdown.bytes / 1024).toFixed(0)} KB`
        );
        if (breakdown.attribution.length > 0) {
          for (const attr of breakdown.attribution) {
            console.log(`    scope: ${attr.scope}`);
            if (attr.container) {
              console.log(`    container: ${attr.container.src}`);
            }
          }
        }
      }
      return result;
    } catch (e) {
      console.error("measureUserAgentSpecificMemory failed:", e);
    }
  }
  return null;
}

// 定期的なメモリ監視
class MemoryMonitor {
  constructor(options = {}) {
    this.intervalMs = options.intervalMs || 10000;
    this.warningThresholdMB = options.warningThresholdMB || 100;
    this.criticalThresholdMB = options.criticalThresholdMB || 200;
    this.history = [];
    this.timerId = null;
  }

  start() {
    this.timerId = setInterval(() => {
      const info = getMemoryInfo();
      if (!info) return;

      const entry = {
        timestamp: Date.now(),
        ...info
      };
      this.history.push(entry);

      // 警告判定
      const usedMB = parseFloat(info.usedHeapMB);
      if (usedMB > this.criticalThresholdMB) {
        console.warn(`[Memory CRITICAL] ${usedMB} MB used`);
        this.onCritical?.(entry);
      } else if (usedMB > this.warningThresholdMB) {
        console.warn(`[Memory WARNING] ${usedMB} MB used`);
        this.onWarning?.(entry);
      }

      // 上昇傾向の検出
      if (this.history.length >= 10) {
        const recent = this.history.slice(-10);
        const trend = this.calculateTrend(recent);
        if (trend > 0.5) { // 1サンプルあたり0.5MB以上の増加
          console.warn(`[Memory TREND] Increasing at ${trend.toFixed(2)} MB/sample`);
        }
      }
    }, this.intervalMs);
  }

  stop() {
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  calculateTrend(entries) {
    if (entries.length < 2) return 0;
    const first = parseFloat(entries[0].usedHeapMB);
    const last = parseFloat(entries[entries.length - 1].usedHeapMB);
    return (last - first) / (entries.length - 1);
  }
}
```

---

## 7. Node.js 環境でのメモリ管理

### 7.1 Node.js のヒープサイズ制御

Node.js はデフォルトでヒープサイズに上限が設けられている。大量のデータを処理するアプリケーションでは、この上限を意識する必要がある。

| Node.js バージョン | デフォルト Old Space 上限 | 備考 |
|-------------------|-------------------------|------|
| v12 以前 | ~1.5 GB (64bit) | 32bit では ~512 MB |
| v12〜v16 | ~2 GB | 段階的に増加 |
| v17 以降 | ~4 GB | 物理メモリの50%まで自動調整 |

```bash
# ヒープサイズの明示的な設定
node --max-old-space-size=8192 server.js  # Old Space を 8GB に設定
node --max-semi-space-size=64 server.js   # Semi-Space を 64MB に設定

# V8 の GC フラグ
node --expose-gc server.js                # global.gc() を有効化
node --trace-gc server.js                 # GC イベントをログ出力
node --trace-gc-verbose server.js         # 詳細な GC ログ
```

### 7.2 process.memoryUsage() による監視

```javascript
// コード例 10: Node.js でのメモリ使用量監視

function printMemoryUsage(label = "") {
  const usage = process.memoryUsage();
  const formatMB = (bytes) => (bytes / 1024 / 1024).toFixed(2) + " MB";

  console.log(`=== Memory Usage ${label} ===`);
  console.log(`  rss:          ${formatMB(usage.rss)}`);        // OS から割り当てられた総メモリ
  console.log(`  heapTotal:    ${formatMB(usage.heapTotal)}`);   // V8 ヒープ合計
  console.log(`  heapUsed:     ${formatMB(usage.heapUsed)}`);    // V8 ヒープ使用量
  console.log(`  external:     ${formatMB(usage.external)}`);    // C++ オブジェクト (Buffer等)
  console.log(`  arrayBuffers: ${formatMB(usage.arrayBuffers)}`);// ArrayBuffer の合計
}

// rss vs heapUsed の違い
// rss (Resident Set Size): プロセスが使用している物理メモリ全体
//   → V8ヒープ + C++オブジェクト + ネイティブアドオン + スタック
// heapUsed: V8 が管理する JavaScript オブジェクトのメモリ
//   → rss の一部

// 使用例: 処理前後のメモリ差分を計測
async function measureMemoryImpact(fn) {
  // GCを強制実行してベースラインを安定化
  if (global.gc) global.gc();

  const before = process.memoryUsage();
  await fn();

  if (global.gc) global.gc();

  const after = process.memoryUsage();

  const delta = {
    rss: after.rss - before.rss,
    heapTotal: after.heapTotal - before.heapTotal,
    heapUsed: after.heapUsed - before.heapUsed,
    external: after.external - before.external,
  };

  const formatMB = (bytes) => {
    const mb = bytes / 1024 / 1024;
    return (mb >= 0 ? "+" : "") + mb.toFixed(2) + " MB";
  };

  console.log("Memory impact:");
  console.log(`  rss:       ${formatMB(delta.rss)}`);
  console.log(`  heapUsed:  ${formatMB(delta.heapUsed)}`);
  console.log(`  external:  ${formatMB(delta.external)}`);

  return delta;
}
```

### 7.3 Buffer と ArrayBuffer のメモリ特性

Node.js の Buffer は V8 ヒープの外側 (external memory) に割り当てられることがある。これにより `heapUsed` には反映されないメモリ消費が発生する。

```javascript
// Buffer のメモリ特性の確認
function demonstrateBufferMemory() {
  printMemoryUsage("Before");

  // Buffer.alloc: external memory に割り当て
  const buf1 = Buffer.alloc(50 * 1024 * 1024); // 50MB
  printMemoryUsage("After Buffer.alloc(50MB)");
  // → external が増加、heapUsed はほぼ変わらない

  // 通常の配列: V8 ヒープに割り当て
  const arr = new Array(5 * 1024 * 1024).fill(0);
  printMemoryUsage("After Array(5M)");
  // → heapUsed が増加
}
```

---

## 8. フレームワーク別のメモリ管理ベストプラクティス

### 8.1 React でのメモリ管理

```javascript
// コード例 11: React コンポーネントでのメモリリーク対策

import { useState, useEffect, useRef, useCallback } from "react";

// アンチパターン: アンマウント後の state 更新
function LeakyComponent({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // 問題: アンマウント後に setUser が呼ばれるとメモリリーク + 警告
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(data => setUser(data));
  }, [userId]);

  return <div>{user?.name}</div>;
}

// 修正版: AbortController + クリーンアップ
function SafeComponent({ userId }) {
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();

    async function fetchUser() {
      try {
        const res = await fetch(`/api/users/${userId}`, {
          signal: controller.signal
        });
        const data = await res.json();
        setUser(data);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err);
        }
      }
    }

    fetchUser();

    return () => {
      controller.abort(); // アンマウント時にリクエストをキャンセル
    };
  }, [userId]);

  if (error) return <div>Error: {error.message}</div>;
  return <div>{user?.name}</div>;
}

// WebSocket のクリーンアップ
function useWebSocket(url) {
  const [messages, setMessages] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      setMessages(prev => {
        // メモリ制限: 最新1000件のみ保持
        const updated = [...prev, JSON.parse(event.data)];
        return updated.slice(-1000);
      });
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      ws.close(); // アンマウント時に接続を閉じる
      wsRef.current = null;
    };
  }, [url]);

  const send = useCallback((data) => {
    wsRef.current?.send(JSON.stringify(data));
  }, []);

  return { messages, send };
}

// IntersectionObserver のクリーンアップ
function useLazyLoad(ref) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(element); // 一度表示されたら監視を停止
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(element);

    return () => {
      observer.disconnect(); // クリーンアップ
    };
  }, [ref]);

  return isVisible;
}
```

### 8.2 Vue.js でのメモリ管理

```javascript
// Vue 3 Composition API でのメモリ管理
import { ref, onMounted, onBeforeUnmount, watch } from "vue";

export function usePolling(fetchFn, intervalMs = 5000) {
  const data = ref(null);
  const error = ref(null);
  let timerId = null;
  let abortController = null;

  async function poll() {
    abortController = new AbortController();
    try {
      data.value = await fetchFn({ signal: abortController.signal });
    } catch (err) {
      if (err.name !== "AbortError") {
        error.value = err;
      }
    }
  }

  onMounted(() => {
    poll();
    timerId = setInterval(poll, intervalMs);
  });

  onBeforeUnmount(() => {
    // 確実にクリーンアップ
    if (timerId) clearInterval(timerId);
    if (abortController) abortController.abort();
  });

  return { data, error };
}
```

---

## 9. 本番環境でのメモリ監視戦略

### 9.1 メモリ予算 (Memory Budget) の設定

パフォーマンスバジェットと同様に、メモリ消費にも予算を設けて継続的に監視することが重要。

| メトリクス | 推奨上限 (モバイル) | 推奨上限 (デスクトップ) | 測定方法 |
|-----------|-------------------|---------------------|---------|
| JS Heap (初期ロード後) | 30 MB | 80 MB | `performance.memory.usedJSHeapSize` |
| JS Heap (ピーク時) | 80 MB | 200 MB | Heap Snapshot |
| DOM ノード数 | 800 | 1500 | `document.querySelectorAll("*").length` |
| JS イベントリスナー数 | 200 | 500 | DevTools > Elements > Event Listeners |
| Detached DOM ノード | 0 | 0 | Heap Snapshot で "Detached" 検索 |

### 9.2 自動リーク検出テスト

```javascript
// コード例 12: Puppeteer を使った自動メモリリーク検出

const puppeteer = require("puppeteer");

async function detectMemoryLeak(url, action, iterations = 10) {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto(url, { waitUntil: "networkidle0" });

  // ウォームアップ: 最初の数回は計測対象外
  for (let i = 0; i < 3; i++) {
    await action(page);
  }

  // GC を実行してベースラインを取得
  await page.evaluate(() => {
    if (window.gc) window.gc();
  });

  const memorySnapshots = [];

  for (let i = 0; i < iterations; i++) {
    await action(page);

    // GC を強制実行
    await page.evaluate(() => {
      if (window.gc) window.gc();
    });

    // メモリ使用量を記録
    const metrics = await page.metrics();
    memorySnapshots.push({
      iteration: i,
      jsHeapUsedSize: metrics.JSHeapUsedSize,
      jsHeapTotalSize: metrics.JSHeapTotalSize,
      documents: metrics.Documents,
      nodes: metrics.Nodes,
      jsEventListeners: metrics.JSEventListeners,
    });
  }

  await browser.close();

  // リーク判定: メモリが単調増加しているか
  const heapValues = memorySnapshots.map(s => s.jsHeapUsedSize);
  const firstHalf = heapValues.slice(0, Math.floor(heapValues.length / 2));
  const secondHalf = heapValues.slice(Math.floor(heapValues.length / 2));

  const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
  const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

  const leakDetected = avgSecond > avgFirst * 1.1; // 10%以上の増加

  return {
    leakDetected,
    snapshots: memorySnapshots,
    averageFirstHalf: (avgFirst / 1024 / 1024).toFixed(2) + " MB",
    averageSecondHalf: (avgSecond / 1024 / 1024).toFixed(2) + " MB",
    growth: ((avgSecond - avgFirst) / avgFirst * 100).toFixed(1) + "%",
  };
}

// 使用例:
// const result = await detectMemoryLeak(
//   "http://localhost:3000",
//   async (page) => {
//     await page.click("#open-modal");
//     await page.waitForSelector(".modal");
//     await page.click("#close-modal");
//     await page.waitForSelector(".modal", { hidden: true });
//   },
//   20
// );
// console.log("Leak detected:", result.leakDetected);
```

---

## 10. 演習問題

### 演習 1 (初級): メモリリークの識別

以下のコードに含まれるメモリリークを全て指摘し、修正せよ。

```javascript
// 演習 1: 以下のコードのメモリリークを修正せよ

class ChatRoom {
  constructor() {
    this.messages = [];
    this.subscribers = [];

    // (A) リサイズハンドラ
    window.addEventListener("resize", () => {
      this.adjustLayout();
    });

    // (B) メッセージポーリング
    setInterval(async () => {
      const newMessages = await fetch("/api/messages").then(r => r.json());
      this.messages.push(...newMessages);
      this.render();
    }, 3000);
  }

  subscribe(callback) {
    this.subscribers.push(callback);
  }

  adjustLayout() {
    // レイアウト調整処理
  }

  render() {
    this.subscribers.forEach(cb => cb(this.messages));
  }

  destroy() {
    // 何もしていない!
  }
}
```

<details>
<summary>解答例 (クリックで展開)</summary>

```javascript
class ChatRoomFixed {
  constructor() {
    this.messages = [];
    this.subscribers = [];
    this.abortController = new AbortController();

    // (A) 修正: AbortController で管理
    window.addEventListener("resize", () => {
      this.adjustLayout();
    }, { signal: this.abortController.signal });

    // (B) 修正: intervalId を保持 + メッセージ上限
    this.pollingId = setInterval(async () => {
      try {
        const res = await fetch("/api/messages", {
          signal: this.abortController.signal
        });
        const newMessages = await res.json();
        this.messages.push(...newMessages);

        // メッセージ数の上限を設ける (メモリ無制限増加を防止)
        if (this.messages.length > 1000) {
          this.messages = this.messages.slice(-500);
        }

        this.render();
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error("Polling failed:", err);
        }
      }
    }, 3000);
  }

  subscribe(callback) {
    this.subscribers.push(callback);
    // unsubscribe 関数を返す
    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index !== -1) this.subscribers.splice(index, 1);
    };
  }

  adjustLayout() { /* ... */ }

  render() {
    this.subscribers.forEach(cb => cb(this.messages));
  }

  destroy() {
    // 全リスナーを一括解除
    this.abortController.abort();
    // タイマー停止
    clearInterval(this.pollingId);
    // 参照をクリア
    this.messages = [];
    this.subscribers = [];
  }
}
```

**指摘ポイント:**
1. `window.addEventListener` に匿名関数を使用しており、`removeEventListener` できない → AbortController で管理
2. `setInterval` の戻り値を保持しておらず、`clearInterval` できない → `this.pollingId` で保持
3. `this.messages` が無制限に増加する → 上限を設けて古いメッセージを削除
4. `subscribe` で登録した `callback` を解除する手段がない → unsubscribe 関数を返す
5. `destroy()` が空 → 全リソースを確実に解放

</details>

### 演習 2 (中級): Heap Snapshot の分析

以下のシナリオで Heap Snapshot を取得し、リーク原因を特定せよ。

```
シナリオ:
1. SPAアプリケーションでユーザー一覧ページを表示
2. ユーザー詳細モーダルを10回開閉する
3. メモリが開閉のたびに増加し、解放されない

手順:
(a) DevTools Memory タブを開き、初期スナップショットを取得
(b) モーダルを10回開閉する
(c) ゴミ箱アイコンでGCを強制実行
(d) 2つ目のスナップショットを取得
(e) Comparison ビューで分析

注目すべきポイント:
- "Detached" で検索 → 切り離されたDOMノード
- #Delta が正の大きい値 → リーク候補
- Retainers パネルで保持チェーンを確認
- EventListener や closure が保持者になっていないか
```

**確認項目チェックリスト:**

| 確認項目 | 期待値 | リーク時の傾向 |
|---------|--------|-------------|
| Detached HTMLDivElement | 0 | モーダルの開閉回数に比例して増加 |
| (closure) | 安定 | 開閉ごとに新しいクロージャが蓄積 |
| EventListener count | 安定 | 開閉ごとに増加 |
| Array entries | 安定 | 内部配列にDOM参照が蓄積 |

### 演習 3 (上級): メモリ安全なキャッシュシステムの設計

以下の要件を満たすキャッシュシステムを設計・実装せよ。

**要件:**
1. LRU (Least Recently Used) 方式で最大エントリ数を制限
2. 個々のエントリに TTL (Time To Live) を設定可能
3. メモリプレッシャー時に自動的にエントリを削減
4. キャッシュヒット率の統計情報を提供

<details>
<summary>解答例 (クリックで展開)</summary>

```javascript
class MemorySafeCache {
  constructor(options = {}) {
    this.maxEntries = options.maxEntries || 1000;
    this.defaultTTL = options.defaultTTL || 60000; // 60秒
    this.pressureThreshold = options.pressureThreshold || 0.8; // 80%

    // LRU 用の Map (挿入順序を保持)
    this.cache = new Map();

    // 統計情報
    this.stats = { hits: 0, misses: 0, evictions: 0, expired: 0 };

    // メモリプレッシャー監視
    this.startMemoryMonitoring();
  }

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) {
      this.stats.misses++;
      return undefined;
    }

    // TTL チェック
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      this.stats.expired++;
      this.stats.misses++;
      return undefined;
    }

    // LRU: アクセスされたエントリを末尾に移動
    this.cache.delete(key);
    this.cache.set(key, entry);

    this.stats.hits++;
    return entry.value;
  }

  set(key, value, ttl = this.defaultTTL) {
    // 既存エントリがあれば削除 (更新)
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    // サイズ制限チェック
    while (this.cache.size >= this.maxEntries) {
      this.evictOldest();
    }

    this.cache.set(key, {
      value,
      expiresAt: Date.now() + ttl,
      size: this.estimateSize(value),
    });
  }

  delete(key) {
    return this.cache.delete(key);
  }

  evictOldest() {
    // Map の最初のエントリ (最も古い) を削除
    const firstKey = this.cache.keys().next().value;
    if (firstKey !== undefined) {
      this.cache.delete(firstKey);
      this.stats.evictions++;
    }
  }

  evictPercent(percent) {
    const count = Math.ceil(this.cache.size * percent);
    for (let i = 0; i < count; i++) {
      this.evictOldest();
    }
  }

  estimateSize(value) {
    // 大まかなサイズ推定
    if (typeof value === "string") return value.length * 2;
    if (typeof value === "number") return 8;
    if (value === null || value === undefined) return 0;
    try {
      return JSON.stringify(value).length * 2;
    } catch {
      return 1024; // 推定不能な場合のデフォルト
    }
  }

  startMemoryMonitoring() {
    if (typeof performance !== "undefined" && performance.memory) {
      this.monitorId = setInterval(() => {
        const ratio =
          performance.memory.usedJSHeapSize /
          performance.memory.jsHeapSizeLimit;
        if (ratio > this.pressureThreshold) {
          console.warn(
            `[Cache] Memory pressure detected (${(ratio * 100).toFixed(1)}%), evicting 25% of entries`
          );
          this.evictPercent(0.25);
        }
      }, 5000);
    }
  }

  getStats() {
    const total = this.stats.hits + this.stats.misses;
    return {
      ...this.stats,
      size: this.cache.size,
      hitRate: total > 0 ? ((this.stats.hits / total) * 100).toFixed(1) + "%" : "N/A",
    };
  }

  destroy() {
    if (this.monitorId) clearInterval(this.monitorId);
    this.cache.clear();
  }
}
```

</details>

---

## 11. よくある質問 (FAQ)

### Q1: ガベージコレクションを手動で実行できるか?

ブラウザ環境では `gc()` 関数は通常利用できない。Chrome DevTools の Memory パネルにあるゴミ箱アイコンをクリックすることで手動GCを実行できるが、これはデバッグ目的に限られる。

Node.js では `--expose-gc` フラグを付けて起動することで `global.gc()` が利用可能になる。ただし、本番環境で手動GCを多用するのは推奨されない。V8のGCスケジューラはヒューリスティックに基づいて最適なタイミングでGCを実行しており、手動介入はパフォーマンスを悪化させることがある。

```javascript
// Node.js: --expose-gc フラグが必要
if (global.gc) {
  global.gc(); // Minor GC + Major GC を実行
} else {
  console.warn("GC is not exposed. Run with --expose-gc flag.");
}
```

### Q2: メモリリークとメモリ膨張 (Memory Bloat) の違いは?

**メモリリーク**: 不要になったオブジェクトが意図せず保持され続け、使用メモリが単調増加する現象。GCルートからの参照が残っているため、GCが回収できない。

**メモリ膨張 (Memory Bloat)**: アプリケーションが正当に必要とするメモリが設計上多すぎる現象。リークではないが、パフォーマンスに悪影響を与える。

| 特性 | メモリリーク | メモリ膨張 |
|------|------------|----------|
| メモリ推移 | 単調増加 (時間とともに悪化) | 高いが安定 |
| GC | 回収できないオブジェクトが蓄積 | GCは正常に動作 |
| 原因 | バグ (参照の解放忘れ) | 設計上の問題 (非効率なデータ構造) |
| 対策 | 参照の解放、リスナーの解除 | データ構造の最適化、仮想化、遅延読み込み |
| 検出方法 | Heap Snapshot の Comparison | Performance Monitor の JS Heap Size |

### Q3: WeakMap と通常の Map はどちらを使うべきか?

**Map を使うべきケース:**
- キーがプリミティブ値 (文字列、数値) の場合 → WeakMap はオブジェクトキーのみ
- キーの列挙が必要な場合 → WeakMap は `keys()`, `values()`, `entries()` を持たない
- キャッシュのサイズを明示的に管理したい場合

**WeakMap を使うべきケース:**
- DOM要素にメタデータを関連付ける場合
- オブジェクトへの追加データを、そのオブジェクトのライフサイクルに連動させたい場合
- プライベートデータの格納 (外部からアクセス不可)
- メモリリークを避けたいキャッシュ

```javascript
// WeakMap が最適: DOM要素へのメタデータ付与
const tooltipData = new WeakMap();

function setTooltip(element, text) {
  tooltipData.set(element, { text, visible: false });
  // element がDOMから除去されGCされると、tooltipDataのエントリも自動消滅
}

// Map が最適: 文字列キーのキャッシュ
const apiCache = new Map();

function cacheResponse(url, data) {
  apiCache.set(url, { data, timestamp: Date.now() });
  // 明示的なサイズ管理が可能
  if (apiCache.size > 100) {
    const oldestKey = apiCache.keys().next().value;
    apiCache.delete(oldestKey);
  }
}
```

### Q4: ArrayBuffer や TypedArray のメモリはどこに割り当てられるか?

ArrayBuffer のバッキングストアは V8 ヒープの外側 (external memory) に割り当てられる。ただし、ArrayBuffer オブジェクト自体は V8 ヒープ上に存在する。

```javascript
// ArrayBuffer: バッキングストアは external memory
const buffer = new ArrayBuffer(1024 * 1024); // 1MB
// → V8 ヒープ上には ArrayBuffer オブジェクト (~100 bytes)
// → external memory に 1MB の連続メモリブロック

// SharedArrayBuffer: 複数の Worker 間で共有可能
const shared = new SharedArrayBuffer(1024);
// → Web Worker 間でメモリを共有
// → Cross-Origin Isolation (COOP + COEP) が必要

// TypedArray: ArrayBuffer のビューであり、追加のメモリは消費しない
const view1 = new Uint8Array(buffer);       // buffer の全体を参照
const view2 = new Float64Array(buffer, 0, 128); // buffer の一部を参照
// → view1, view2 は同じ buffer のメモリを共有
```

### Q5: Web Worker を使うとメモリ管理はどう変わるか?

各 Web Worker は独立した V8 インスタンスとヒープを持つ。Worker 間でのデータ転送は、構造化複製 (Structured Clone) によるコピーか、Transferable オブジェクトによる所有権の移転で行われる。

```javascript
// メインスレッド
const worker = new Worker("worker.js");

// コピー転送: データが複製される (元データは保持される)
const data = new Uint8Array(1024 * 1024);
worker.postMessage({ type: "process", data: data });
// → data のコピーが Worker に送られる (メモリが一時的に2倍)

// 所有権移転 (Transfer): ゼロコピーでデータを移動
const buffer = new ArrayBuffer(1024 * 1024);
worker.postMessage({ type: "process", buffer: buffer }, [buffer]);
// → buffer の所有権が Worker に移転
// → メインスレッド側の buffer.byteLength は 0 になる (使用不可)
// → メモリの複製が発生しない
```

### Q6: メモリリークを検出する方法は？（Chrome DevTools）

Chrome DevTools の Memory パネルを使った体系的なメモリリーク検出手順:

**ステップ1: ベースラインの取得**
1. ページをリロードして初期状態にする
2. Memory パネル → Heap snapshot を撮影 (Snapshot 1)

**ステップ2: 操作の実行**
3. リークが疑われる操作を実行（例: モーダルを開いて閉じる、ページ遷移、データ読み込み）
4. 操作を繰り返す（5〜10回程度）

**ステップ3: 比較分析**
5. もう一度 Heap snapshot を撮影 (Snapshot 2)
6. Snapshot 2 を選択し、表示モードを "Comparison" に変更
7. "Objects allocated between Snapshot 1 and Snapshot 2" を確認

**ステップ4: リークの特定**
```
Comparison ビューの見方:
- New: 新しく作成されたオブジェクト数
- Deleted: 削除されたオブジェクト数
- Delta: New - Deleted (正の値が大きいとリークの可能性)
- Size Delta: メモリ増加量
```

リークの典型的な兆候:
- Delta が大きく正の値のまま推移する
- Detached HTMLDivElement などの DOM ノードが残存
- Array や Object が際限なく増加している

**ステップ5: Retainers パスの分析**
8. 増加しているオブジェクトを選択
9. "Retainers" パネルで、GC ルートからの参照チェーンを確認
10. 意図しない参照を特定して修正

```javascript
// 検出例: イベントリスナーの解除忘れ
class ComponentWithLeak {
  constructor() {
    this.data = new Array(10000).fill(0);
    // 問題: removeEventListener していない
    window.addEventListener('resize', this.handleResize.bind(this));
  }
  handleResize() { /* ... */ }
}

// Heap Snapshot の Retainers:
// GC root → Window → listeners → resize → Function → ComponentWithLeak → data
//                                                       ^^^^^^^^^^^^^^^^
//                                                       ここでリークが発生
```

### Q7: クロージャによるメモリリークを防ぐには？

クロージャは便利だが、不要な外部変数をキャプチャし続けるとメモリリークの原因になる。

**問題パターン1: 大きなオブジェクトを不要にキャプチャ**
```javascript
function createProcessor() {
  const hugeData = new Array(1000000).fill(Math.random()); // 8MB
  const metadata = { size: hugeData.length, created: Date.now() };

  // 問題: metadata だけ使いたいのに、hugeData も一緒にキャプチャされる
  return function() {
    console.log(`Processed ${metadata.size} items`);
  };
}

const process = createProcessor();
// → hugeData は関数スコープ内で定義されているため、
//   返された関数がクロージャとしてキャプチャし続ける
//   (process が生きている限り 8MB が解放されない)
```

**解決策1: 必要な値だけを抽出**
```javascript
function createProcessor() {
  const hugeData = new Array(1000000).fill(Math.random());
  const size = hugeData.length; // プリミティブ値を抽出
  const created = Date.now();

  // hugeData はここでスコープを抜けるので GC 対象になる

  return function() {
    console.log(`Processed ${size} items at ${created}`);
  };
  // → クロージャは size と created だけをキャプチャ (16バイト程度)
}
```

**問題パターン2: イベントハンドラでの this キャプチャ**
```javascript
class DataGrid {
  constructor(data) {
    this.data = data; // 大量のデータ
    this.renderCache = new Map();

    // 問題: アロー関数で this を暗黙的にキャプチャ
    document.getElementById('refresh-btn').addEventListener('click', () => {
      this.refresh(); // this.data も一緒にキャプチャされる
    });
  }

  refresh() {
    this.renderCache.clear();
    // ... 再描画処理
  }

  destroy() {
    // 問題: イベントリスナーが解除されていない
    // → this (と this.data) が解放されない
  }
}
```

**解決策2: 明示的なクリーンアップ**
```javascript
class DataGrid {
  constructor(data) {
    this.data = data;
    this.renderCache = new Map();

    // 解決策: ハンドラへの参照を保持
    this.handleRefresh = () => this.refresh();
    this.refreshBtn = document.getElementById('refresh-btn');
    this.refreshBtn.addEventListener('click', this.handleRefresh);
  }

  refresh() {
    this.renderCache.clear();
  }

  destroy() {
    // 正しくリスナーを解除
    this.refreshBtn.removeEventListener('click', this.handleRefresh);
    this.handleRefresh = null; // 参照も切断
    this.data = null; // 大きなデータも明示的に解放
    this.renderCache.clear();
  }
}
```

**問題パターン3: タイマーコールバックでのキャプチャ**
```javascript
function startPolling(url) {
  const history = []; // 結果の履歴

  const intervalId = setInterval(async () => {
    const result = await fetch(url).then(r => r.json());
    history.push(result); // 無制限に蓄積
    processResult(result);
  }, 5000);

  return () => clearInterval(intervalId);
  // 問題: history は clearInterval しても残り続ける
}
```

**解決策3: リングバッファで上限を設ける**
```javascript
function startPolling(url, maxHistory = 10) {
  const history = [];

  const intervalId = setInterval(async () => {
    const result = await fetch(url).then(r => r.json());

    // 上限を超えたら古いものを削除
    if (history.length >= maxHistory) {
      history.shift();
    }
    history.push(result);

    processResult(result);
  }, 5000);

  return () => {
    clearInterval(intervalId);
    history.length = 0; // 配列もクリア
  };
}
```

**ベストプラクティス:**
1. クロージャがキャプチャする変数を意識する（DevTools の Scope パネルで確認可能）
2. 大きなデータは必要な値だけ抽出してから関数を返す
3. イベントリスナーやタイマーのクリーンアップを必ず実装
4. 無制限に成長する配列/Map には上限を設ける

### Q8: 大規模SPAでのメモリ管理戦略は？

シングルページアプリケーション (SPA) は長時間稼働するため、メモリ管理が特に重要。

**戦略1: ページ遷移時のクリーンアップ**
```javascript
// React の例
function UserProfile({ userId }) {
  useEffect(() => {
    // データ購読の開始
    const subscription = userService.subscribe(userId, handleUpdate);
    const timerId = setInterval(refreshData, 30000);

    // クリーンアップ関数: コンポーネントのアンマウント時に実行
    return () => {
      subscription.unsubscribe(); // 購読解除
      clearInterval(timerId);     // タイマー解除
      userService.clearCache(userId); // キャッシュクリア
    };
  }, [userId]);

  // ... コンポーネント本体
}
```

**戦略2: 仮想化 (Virtualization) で大量データを扱う**
```javascript
// 問題: 10万件のリストを全てDOMレンダリング
function HugeList({ items }) {
  return (
    <div>
      {items.map(item => <ListItem key={item.id} {...item} />)}
      {/* 10万個のDOMノード → メモリ膨張 */}
    </div>
  );
}

// 解決策: react-window による仮想化
import { FixedSizeList } from 'react-window';

function VirtualizedList({ items }) {
  return (
    <FixedSizeList
      height={600}
      itemCount={items.length}
      itemSize={50}
      width="100%"
    >
      {({ index, style }) => (
        <div style={style}>
          <ListItem {...items[index]} />
        </div>
      )}
    </FixedSizeList>
    // 画面に表示される分だけレンダリング (例: 15個)
    // → メモリ使用量が数千分の一に削減
  );
}
```

**戦略3: メモリ予算の設定と監視**
```javascript
class MemoryBudgetMonitor {
  constructor(budgetMB = 150) {
    this.budgetBytes = budgetMB * 1024 * 1024;
    this.warningThreshold = this.budgetBytes * 0.8;
    this.startMonitoring();
  }

  async checkMemory() {
    if ('memory' in performance) {
      const { usedJSHeapSize, jsHeapSizeLimit } = performance.memory;

      if (usedJSHeapSize > this.budgetBytes) {
        console.error(`Memory budget exceeded: ${(usedJSHeapSize / 1024 / 1024).toFixed(1)} MB`);
        this.triggerEmergencyCleanup();
      } else if (usedJSHeapSize > this.warningThreshold) {
        console.warn(`Memory warning: ${(usedJSHeapSize / 1024 / 1024).toFixed(1)} MB`);
        this.triggerSoftCleanup();
      }
    }
  }

  triggerSoftCleanup() {
    // 優先度の低いキャッシュをクリア
    imageCache.evictOldest(50);
    dataCache.trim(100);
  }

  triggerEmergencyCleanup() {
    // 全てのキャッシュをクリア
    imageCache.clear();
    dataCache.clear();

    // ユーザーに通知
    showNotification("メモリ不足のため、一部のデータをクリアしました");
  }

  startMonitoring() {
    setInterval(() => this.checkMemory(), 30000); // 30秒ごと
  }
}

const monitor = new MemoryBudgetMonitor(150); // 予算 150MB
```

**戦略4: キャッシュの有効期限と上限**
```javascript
class BoundedCache {
  constructor(maxSize = 100, maxAge = 5 * 60 * 1000) { // 5分
    this.cache = new Map();
    this.maxSize = maxSize;
    this.maxAge = maxAge;
  }

  set(key, value) {
    // 上限チェック: LRU削除
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      value,
      timestamp: Date.now()
    });
  }

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    // 期限チェック
    if (Date.now() - entry.timestamp > this.maxAge) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.value;
  }

  prune() {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > this.maxAge) {
        this.cache.delete(key);
      }
    }
  }
}
```

**戦略5: Service Worker によるオフロード**
```javascript
// メインスレッド
const worker = new Worker('heavy-processor.js');

worker.postMessage({
  type: 'processLargeDataset',
  data: hugeDataset
}, [hugeDataset.buffer]); // Transferable で所有権移転

worker.onmessage = (e) => {
  const result = e.data;
  updateUI(result); // 結果だけ受け取る
};

// heavy-processor.js (Worker)
self.onmessage = (e) => {
  const { type, data } = e.data;

  if (type === 'processLargeDataset') {
    const result = processData(data); // 重い処理
    self.postMessage(result);
    // Worker のヒープで処理されるため、メインスレッドに影響しない
  }
};
```

**戦略6: 定期的なページリロード（最終手段）**
```javascript
// 長時間稼働するダッシュボードなど
class AutoReloadManager {
  constructor(maxUptimeHours = 8) {
    this.maxUptime = maxUptimeHours * 60 * 60 * 1000;
    this.startTime = Date.now();
    this.checkInterval = setInterval(() => this.checkUptime(), 60000); // 1分ごと
  }

  checkUptime() {
    const uptime = Date.now() - this.startTime;

    if (uptime > this.maxUptime) {
      // ユーザーに通知してリロード
      if (confirm("アプリケーションを最新の状態に更新します。よろしいですか？")) {
        location.reload();
      }
    }
  }
}

// 8時間で自動リロード提案
const reloadManager = new AutoReloadManager(8);
```

**チェックリスト:**
- [ ] コンポーネントのクリーンアップ関数を実装
- [ ] 大量データは仮想化 (react-window, virtual-scroller など)
- [ ] メモリ予算を設定し、超過時にアラートを発火
- [ ] キャッシュに上限と有効期限を設ける
- [ ] 重い処理は Web Worker でオフロード
- [ ] E2Eテストでメモリリークテストを自動化
- [ ] 長時間稼働アプリでは定期リロードを検討

---

## 12. アンチパターン集

### アンチパターン 1: 無制限に成長する配列/Map

```javascript
// 問題: イベントログが際限なく蓄積される
class EventLogger {
  constructor() {
    this.events = []; // 上限がない!
  }

  log(event) {
    this.events.push({
      ...event,
      timestamp: Date.now(),
      stack: new Error().stack // スタックトレースも保持 → メモリ消費大
    });
  }
}

// 対策: リングバッファを使用
class BoundedEventLogger {
  constructor(maxSize = 1000) {
    this.events = new Array(maxSize);
    this.maxSize = maxSize;
    this.index = 0;
    this.count = 0;
  }

  log(event) {
    this.events[this.index] = {
      ...event,
      timestamp: Date.now()
      // stack は本番では省略
    };
    this.index = (this.index + 1) % this.maxSize;
    this.count = Math.min(this.count + 1, this.maxSize);
  }

  getRecent(n = 10) {
    const result = [];
    let idx = (this.index - 1 + this.maxSize) % this.maxSize;
    for (let i = 0; i < Math.min(n, this.count); i++) {
      result.push(this.events[idx]);
      idx = (idx - 1 + this.maxSize) % this.maxSize;
    }
    return result;
  }
}
```

### アンチパターン 2: MutationObserver の disconnect 忘れ

```javascript
// 問題: observer が disconnect されない
function watchDOMChanges(target) {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      processMutation(mutation); // mutation が大量のDOM参照を含む
    }
  });

  observer.observe(target, {
    childList: true,
    subtree: true,
    attributes: true,
    characterData: true,
  });

  // disconnect が呼ばれないと、target が DOMから除去されても
  // observer が内部的に参照を保持し続ける
}

// 対策: 必ず disconnect を呼ぶ
function watchDOMChangesSafe(target) {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      processMutation(mutation);
    }
  });

  observer.observe(target, {
    childList: true,
    subtree: true,
  });

  // クリーンアップ関数を返す
  return () => {
    observer.disconnect();
  };
}

// React での使用例
function useObserveDOMChanges(ref, callback) {
  React.useEffect(() => {
    if (!ref.current) return;

    const observer = new MutationObserver(callback);
    observer.observe(ref.current, { childList: true, subtree: true });

    return () => observer.disconnect();
  }, [ref, callback]);
}
```

---

## 13. メモリ管理チェックリスト

### 開発フェーズ

- [ ] `"use strict"` または ESLint の `no-implicit-globals` ルールを有効化
- [ ] `setInterval` / `setTimeout` の戻り値を保持し、クリーンアップで `clear*` を呼ぶ
- [ ] `addEventListener` には対応する `removeEventListener` または AbortController を使用
- [ ] クロージャのキャプチャ変数を最小限にする (関数を別スコープで定義)
- [ ] DOM参照を JS 変数に保持する場合、DOM除去時に null を代入
- [ ] 配列や Map に上限サイズを設ける
- [ ] `console.log` に大きなオブジェクトを渡さない (本番では除去)

### テストフェーズ

- [ ] Chrome DevTools の Heap Snapshot Comparison でリーク検出テストを実施
- [ ] Puppeteer / Playwright による自動メモリリーク検出をCIに組み込む
- [ ] Performance Monitor で長時間稼働時のメモリ推移を確認
- [ ] Mobile デバイスでのメモリ消費を確認 (メモリ制約が厳しい)

### 本番運用フェーズ

- [ ] `performance.measureUserAgentSpecificMemory()` または RUM ツールでメモリ監視
- [ ] メモリ予算を設定し、超過時にアラートを発火
- [ ] 長時間稼働するSPAでは、定期的なページリロードを検討

---

## FAQ

### Q1: メモリリークが疑われるとき、最初に確認すべきことは何ですか?
Chrome DevTools の Performance Monitor パネルで「JS Heap Size」の推移を監視することから始めます。特定の操作（ページ遷移、モーダルの開閉、リスト操作など）を繰り返した際にヒープサイズが単調増加していればメモリリークの可能性が高いです。次に Memory パネルで操作前後の Heap Snapshot を取得し、Comparison ビューで増加したオブジェクトを特定します。Detached DOM ノード、未解除のイベントリスナー、クロージャによる意図しない参照保持が主要な原因です。

### Q2: WeakMapとMapの使い分けの判断基準は何ですか?
キーとなるオブジェクトのライフサイクルに依存するメタデータを格納する場合は WeakMap を使います。例えば、DOM要素に関連するキャッシュデータや、オブジェクトごとのプライベートデータの格納に適しています。WeakMap のキーはGCに回収されうるため、キーの列挙や `.size` プロパティは利用できません。一方、キーを列挙する必要がある場合や、プリミティブ値をキーにしたい場合、キーの生存をMap側で保証したい場合は通常の Map を使用します。

### Q3: SPAでメモリ使用量が増え続ける場合の一般的な対処法は?
SPA（Single Page Application）では画面遷移してもページがリロードされないため、メモリが蓄積しやすい構造です。対処法として、(1) コンポーネントのアンマウント時にタイマー（setInterval）、WebSocket接続、イベントリスナーを確実にクリーンアップする、(2) AbortController を使ってfetchリクエストやイベントリスナーを一括解除する、(3) 大量のデータを保持するキャッシュには LRU（Least Recently Used）方式の上限を設ける、(4) `performance.measureUserAgentSpecificMemory()` を使って本番環境でもメモリ使用量を定期的に監視する、といった施策が有効です。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| メモリモデル | スタック (プリミティブ + 参照) とヒープ (オブジェクト) の二層構造 |
| V8ヒープ | New Space (Scavenge) と Old Space (Mark-Sweep/Compact) の世代別構成 |
| GCアルゴリズム | Minor GC (Scavenge, ms単位) と Major GC (Mark-Sweep, インクリメンタル) |
| リークパターン | タイマー、リスナー、クロージャ、Detached DOM、console.log |
| 弱参照 | WeakMap/WeakRef で GC を妨げない参照を実現 |
| DevTools | Heap Snapshot の Comparison ビューがリーク検出の決定打 |
| 本番監視 | メモリ予算の設定と自動テストの組み込み |

---

## 次に読むべきガイド

- [DOM API](../03-web-apis/00-dom-api.md)
- [イベントモデル](../03-web-apis/01-events.md)
- [イベントループ](./02-event-loop.md)

---

## 参考文献

1. V8 Team. "Trash talk: the Orinoco garbage collector." V8 Blog, 2019. https://v8.dev/blog/trash-talk
2. Google. "Fix memory problems." Chrome DevTools Documentation, 2024. https://developer.chrome.com/docs/devtools/memory-problems
3. Nicol Ribaudo. "TC39 Proposal: Explicit Resource Management." TC39, 2024. https://github.com/tc39/proposal-explicit-resource-management
4. Addy Osmani. "Memory Management Reference." 2012.
5. MDN Web Docs. "Memory management." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Memory_management
6. Lin Clark. "A Cartoon Intro to ArrayBuffers and SharedArrayBuffers." Mozilla Hacks, 2017.

