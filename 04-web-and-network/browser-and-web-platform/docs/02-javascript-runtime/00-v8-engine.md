# V8エンジン

> V8はGoogleが開発したJavaScriptエンジン。Chrome、Node.js、Denoで使用される。JITコンパイル、Hidden Class、インラインキャッシュ、ガベージコレクションの仕組みを理解する。

## この章で学ぶこと

- [ ] V8のコンパイルパイプラインを理解する
- [ ] Hidden Class とインラインキャッシュの最適化を把握する
- [ ] ガベージコレクションの仕組みを学ぶ

---

## 1. V8のコンパイルパイプライン

```
JavaScript ソースコード
       ↓
  パーサー（AST生成）
       ↓
  Ignition（インタプリタ）
       ↓ バイトコード実行
       ↓ プロファイリング（ホットスポット検出）
       ↓
  TurboFan（最適化コンパイラ）
       ↓ 最適化されたマシンコード
  実行

  Ignition（インタプリタ）:
  → ソースコードをバイトコードに変換
  → 高速な起動（コンパイル時間が短い）
  → 実行しながらプロファイル情報を収集

  TurboFan（最適化コンパイラ）:
  → ホットスポット（頻繁に実行されるコード）を最適化
  → 型情報に基づいた最適化（型推論）
  → インライン展開、定数畳み込み等
  → 実行速度は速いが、コンパイルに時間がかかる

  脱最適化（Deoptimization）:
  → 最適化の前提が崩れた場合（型が変わった等）
  → 最適化コードを破棄してIgnitionに戻る
  → パフォーマンスの急降下を引き起こす
```

---

## 2. Hidden Class

```
Hidden Class = V8内部のオブジェクト構造の表現

  JavaScriptのオブジェクトは動的（プロパティを自由に追加/削除）
  → C++のクラスのように固定構造がない
  → V8は Hidden Class で内部的に構造を管理

  例:
  const obj = {};          // Hidden Class C0 (空)
  obj.x = 1;              // Hidden Class C1 (x: offset 0)
  obj.y = 2;              // Hidden Class C2 (x: offset 0, y: offset 1)

  同じ順序でプロパティを追加 → 同じ Hidden Class を共有:
  const a = { x: 1, y: 2 };  // Hidden Class C2
  const b = { x: 3, y: 4 };  // Hidden Class C2（同じ！）

  異なる順序 → 別の Hidden Class:
  const c = { y: 2, x: 1 };  // Hidden Class C3（別！）

最適化のための指針:
  ✓ オブジェクトのプロパティは同じ順序で初期化
  ✓ コンストラクタで全プロパティを初期化
  ✗ 後からプロパティを追加しない（delete も避ける）

  // 良い: コンストラクタで全て初期化
  class Point {
    constructor(x, y) {
      this.x = x;
      this.y = y;
    }
  }

  // 悪い: 条件付きでプロパティ追加
  function createPoint(x, y, z) {
    const p = { x, y };
    if (z !== undefined) p.z = z;  // Hidden Class が分岐
    return p;
  }
```

---

## 3. インラインキャッシュ

```
インラインキャッシュ（IC）:
  → プロパティアクセスの結果をキャッシュ
  → 同じHidden Classのオブジェクトなら高速アクセス

  function getX(obj) {
    return obj.x;  // ← ここにICが生成される
  }

  const p1 = { x: 1, y: 2 };
  const p2 = { x: 3, y: 4 };  // p1と同じ Hidden Class

  getX(p1);  // IC miss → Hidden Class を記録
  getX(p2);  // IC hit → 高速アクセス（同じ Hidden Class）

ICの状態:
  monomorphic:   1つの Hidden Class のみ → 最速
  polymorphic:   2-4つの Hidden Class → やや遅い
  megamorphic:   5つ以上 → 最適化断念、遅い

  // 良い: monomorphic
  points.forEach(p => p.x);  // 全て同じ Hidden Class

  // 悪い: megamorphic
  mixedObjects.forEach(obj => obj.x);  // 様々な構造のオブジェクト
```

---

## 4. ガベージコレクション

```
V8のGC（世代別GC）:

  メモリ空間:
  ┌──────────────────────────────────────┐
  │ Young Generation（新世代）            │
  │ ┌───────────┬──────────────────┐     │
  │ │ Semi-space │ Semi-space      │     │
  │ │ (From)    │ (To)            │     │
  │ └───────────┴──────────────────┘     │
  │ サイズ: 1-8MB                        │
  │ 頻繁にGC（Minor GC / Scavenge）      │
  ├──────────────────────────────────────┤
  │ Old Generation（旧世代）              │
  │ 2回のMinor GCを生き延びたオブジェクト │
  │ サイズ: 数百MB〜数GB                 │
  │ 低頻度GC（Major GC / Mark-Sweep）    │
  └──────────────────────────────────────┘

  Minor GC（Scavenge）:
  → 新世代のみ対象（高速: 1-5ms）
  → From空間の生存オブジェクトをTo空間にコピー
  → 2回生き延びたオブジェクトは旧世代に昇格

  Major GC（Mark-Sweep-Compact）:
  → 旧世代が対象（低速: 数十ms）
  → Mark: ルートから到達可能なオブジェクトをマーク
  → Sweep: マークされていないオブジェクトを解放
  → Compact: メモリの断片化を解消

  Incremental Marking:
  → Major GCを小さなステップに分割
  → JS実行との交互実行で停止時間を削減

  Concurrent GC:
  → GC作業の一部をバックグラウンドスレッドで実行
  → メインスレッドの停止時間をさらに削減
```

---

## 5. パフォーマンス最適化のまとめ

```
V8に優しいコードの書き方:

  ① 型を安定させる:
     → 同じ変数に異なる型を入れない
     → 配列は同じ型で統一（SMI配列、Double配列等）

  ② オブジェクト構造を統一:
     → 同じ順序でプロパティを初期化
     → 後からプロパティを追加/削除しない
     → delete 演算子を避ける

  ③ 関数をmonomorphicに保つ:
     → 同じ型の引数を渡す
     → 多様なオブジェクトを同じ関数に渡さない

  ④ メモリリークを防ぐ:
     → イベントリスナーの解除
     → setInterval のクリア
     → クロージャでの大きなオブジェクト参照に注意
     → WeakMap / WeakRef の活用
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Ignition | インタプリタ、バイトコード実行 |
| TurboFan | JIT最適化コンパイラ、ホットスポット |
| Hidden Class | オブジェクト構造の内部表現 |
| インラインキャッシュ | プロパティアクセスのキャッシュ |
| GC | 世代別GC、Minor/Major |

---

## 次に読むべきガイド
→ [[01-event-loop-browser.md]] — ブラウザのイベントループ

---

## 参考文献
1. Mathias Bynens. "JavaScript engine fundamentals." web.dev, 2018.
2. V8 Blog. "Launching Ignition and TurboFan." v8.dev, 2017.
