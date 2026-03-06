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

