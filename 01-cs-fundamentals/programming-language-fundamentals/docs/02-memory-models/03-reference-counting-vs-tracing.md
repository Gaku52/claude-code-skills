# 参照カウント vs トレーシングGC

> メモリ回収の2大戦略。それぞれの仕組み・トレードオフ・適用場面を理解し、言語の特性を深く把握する。

## この章で学ぶこと

- [ ] 参照カウントとトレーシングGCの動作原理を理解する
- [ ] 各方式の長所・短所を判断できる
- [ ] ハイブリッド方式の意義を理解する

---

## 1. 参照カウント（Reference Counting）

```
仕組み: 各オブジェクトが「自分を参照している数」を保持

  操作:
    参照の作成 → カウント +1
    参照の消滅 → カウント -1
    カウント = 0 → 即座に解放

  例:
    a = Object()     # count: 1
    b = a             # count: 2
    del a             # count: 1
    del b             # count: 0 → 解放
```

### 参照カウントの実装

```python
# Python の参照カウント
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))  # 2（a + getrefcount の引数）

b = a
print(sys.getrefcount(a))  # 3（a + b + getrefcount の引数）

del b
print(sys.getrefcount(a))  # 2
```

```swift
// Swift: ARC（Automatic Reference Counting）
class Node {
    var value: Int
    var next: Node?

    init(_ value: Int) {
        self.value = value
        print("Node \(value) created")
    }
    deinit {
        print("Node \(value) deallocated")
    }
}

var n1: Node? = Node(1)    // refcount: 1
var n2 = n1                 // refcount: 2
n1 = nil                    // refcount: 1
n2 = nil                    // refcount: 0 → deinit（即座に解放）
```

### 循環参照問題

```
参照カウントの致命的な弱点:

  a ──→ [Obj A] ──→ [Obj B]
         count:1      count:1
            ↑            │
            └────────────┘

  del a, del b の後:
  [Obj A] ──→ [Obj B]
  count:1      count:1    ← 互いに参照しているのでカウントが0にならない
     ↑            │       ← メモリリーク
     └────────────┘

解決策:
  1. 弱参照（weak reference）を使う
  2. トレーシングGCで補完する（Python方式）
  3. プログラマが循環を避ける設計にする
```

```swift
// Swift: weak/unowned で循環参照を回避
class Parent {
    var child: Child?
    deinit { print("Parent deallocated") }
}

class Child {
    weak var parent: Parent?  // weak: 参照カウントを増やさない
    deinit { print("Child deallocated") }
}

var parent: Parent? = Parent()
var child: Child? = Child()
parent!.child = child
child!.parent = parent

parent = nil  // Parent と Child の両方が解放される
child = nil
```

### 参照カウントの特性

```
利点:
  ✓ 即座に回収（予測可能なタイミング）
  ✓ 停止時間がない（インクリメンタル）
  ✓ デストラクタが確実に呼ばれる（RAII互換）
  ✓ メモリ使用量が安定
  ✓ 実装がシンプル

欠点:
  ✗ 循環参照を処理できない
  ✗ カウント操作のオーバーヘッド（毎回の加減算）
  ✗ マルチスレッドではアトミック操作が必要（高コスト）
  ✗ キャッシュミスが多い（オブジェクトが散在）
```

---

## 2. トレーシングGC（Tracing GC）

```
仕組み: ルートから到達可能なオブジェクトを辿り、
        到達不能なオブジェクトを回収する

  Phase 1: ルートから到達可能なオブジェクトを全て辿る（マーク）
  Phase 2: 到達不能なオブジェクトを回収（スイープ）

  ルート（スタック、グローバル）
    │
    ├──→ [A] ──→ [B] ──→ [C]     到達可能 → 生存
    │
    └──→ [D]                       到達可能 → 生存

         [E] ──→ [F]               到達不能 → 回収
         [G] ←─→ [H]               循環でも到達不能 → 回収 ✓
```

### トレーシングGCの特性

```
利点:
  ✓ 循環参照を正しく処理できる
  ✓ 参照の作成・消滅時のオーバーヘッドがない
  ✓ スループットが高い（バッチ処理の方が効率的）
  ✓ コンパクション（メモリの断片化解消）が可能

欠点:
  ✗ GCパーズ（Stop-The-World）が発生
  ✗ GC タイミングが予測困難
  ✗ メモリ使用量のピークが高い（回収が遅れる）
  ✗ 実装が複雑
```

---

## 3. 比較

```
┌──────────────────┬──────────────────┬──────────────────┐
│                  │ 参照カウント      │ トレーシングGC    │
├──────────────────┼──────────────────┼──────────────────┤
│ 回収タイミング    │ 即座（決定的）   │ 不定（非決定的）  │
│ 循環参照         │ 処理不可         │ 処理可能          │
│ 停止時間         │ なし             │ あり（STW）       │
│ オーバーヘッド    │ 参照操作ごと     │ GC時にまとめて    │
│ スループット      │ やや低い         │ 高い              │
│ メモリ使用量      │ 安定             │ ピークが高い      │
│ マルチスレッド    │ アトミック操作必要│ 効率的            │
│ 代表言語         │ Swift, Rust(Rc)  │ Java, Go, JS     │
│                  │ Python(メイン)   │ C#, Ruby          │
└──────────────────┴──────────────────┴──────────────────┘
```

---

## 4. ハイブリッド方式

```
多くの実用的な言語は両方の要素を組み合わせている:

Python:
  メイン: 参照カウント（即座回収）
  補完: 世代別GC（循環参照の検出・回収）
  → 大半のオブジェクトは参照カウントで即回収
  → 循環参照だけGCで回収

Objective-C → Swift:
  Objective-C: 手動参照カウント → ARC（自動参照カウント）
  Swift: ARC + weak/unowned で循環参照回避
  → GCパーズがない（リアルタイム性が必要なiOSに適合）

Rust:
  デフォルト: 所有権システム（GCもRCもなし）
  必要時: Rc<T>（単一スレッド参照カウント）
         Arc<T>（マルチスレッド参照カウント）
  → 必要な場所でだけ参照カウントを使う

.NET (C#):
  メイン: 世代別トレーシングGC
  補完: IDisposable + using で即座解放（ファイル、DB接続等）
```

---

## 5. 用途別の選択指針

```
リアルタイムシステム（ゲーム、音声処理）:
  → 参照カウント or 所有権（予測可能な停止時間）
  → Swift, Rust

サーバーサイド（高スループット）:
  → トレーシングGC（スループット重視）
  → Java(ZGC), Go

スクリプト・プロトタイプ:
  → ハイブリッド（開発者の負担軽減）
  → Python, Ruby

組み込みシステム:
  → 所有権（GCオーバーヘッドなし）
  → Rust, C

大規模エンタープライズ:
  → 成熟したGC実装
  → Java(G1/ZGC), C#
```

---

## まとめ

| 方式 | 回収タイミング | 循環参照 | 停止時間 | 代表言語 |
|------|-------------|---------|---------|---------|
| 参照カウント | 即座 | 不可 | なし | Swift, Python |
| トレーシング | バッチ | 可能 | あり | Java, Go, JS |
| ハイブリッド | 混合 | 可能 | 最小 | Python, C# |
| 所有権 | スコープ終了 | N/A | なし | Rust |

---

## 次に読むべきガイド
→ [[../03-control-flow/00-branching-and-loops.md]] — 制御構造

---

## 参考文献
1. Jones, R., Hosking, A. & Moss, E. "The Garbage Collection Handbook." 2nd Ed, CRC Press, 2023.
2. Bacon, D. et al. "A Unified Theory of Garbage Collection." ACM OOPSLA, 2004.
