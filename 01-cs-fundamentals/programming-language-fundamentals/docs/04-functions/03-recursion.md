# 再帰（Recursion）

> 再帰は「問題を同じ構造の小さな問題に分解する」技法。数学的に美しく、ツリー・グラフ・分割統治に不可欠。

## この章で学ぶこと

- [ ] 再帰の仕組みと基本パターンを理解する
- [ ] 末尾再帰と最適化を理解する
- [ ] 再帰とループの使い分けを判断できる

---

## 1. 再帰の基本

```
再帰関数 = 自分自身を呼び出す関数

構成要素:
  1. ベースケース（終了条件）: 再帰を止める条件
  2. 再帰ステップ: 問題を小さくして自分自身を呼び出す
```

```python
# 階乗: n! = n × (n-1)!
def factorial(n):
    if n <= 1:          # ベースケース
        return 1
    return n * factorial(n - 1)  # 再帰ステップ

# コールスタックの展開:
# factorial(5)
#   → 5 * factorial(4)
#     → 4 * factorial(3)
#       → 3 * factorial(2)
#         → 2 * factorial(1)
#           → 1（ベースケース）
#         → 2 * 1 = 2
#       → 3 * 2 = 6
#     → 4 * 6 = 24
#   → 5 * 24 = 120
```

---

## 2. 再帰のパターン

```python
# パターン1: 線形再帰（リスト処理）
def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

# パターン2: 二分再帰（分割統治）
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

# パターン3: ツリー再帰
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
# 注意: 指数的な計算量。メモ化が必要

# パターン4: 相互再帰
def is_even(n):
    if n == 0: return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0: return False
    return is_even(n - 1)
```

### ツリー構造の再帰処理

```typescript
// ファイルシステムの走査
interface FSNode {
    name: string;
    type: "file" | "directory";
    children?: FSNode[];
    size?: number;
}

function totalSize(node: FSNode): number {
    if (node.type === "file") {
        return node.size ?? 0;  // ベースケース
    }
    return (node.children ?? [])
        .reduce((sum, child) => sum + totalSize(child), 0);
}

// JSON の深い値を取得
function deepGet(obj: any, path: string[]): any {
    if (path.length === 0) return obj;
    if (obj == null) return undefined;
    const [head, ...tail] = path;
    return deepGet(obj[head], tail);
}

deepGet({ a: { b: { c: 42 } } }, ["a", "b", "c"]);  // → 42
```

---

## 3. 末尾再帰（Tail Recursion）

```
末尾再帰 = 再帰呼び出しが関数の最後の操作

  通常の再帰: return n * factorial(n - 1)
              ↑ 再帰の結果に n を掛ける → スタックに n を保持

  末尾再帰:   return factorial_tail(n - 1, acc * n)
              ↑ 再帰呼び出しが最後 → スタックフレームを再利用可能
```

```python
# 通常の再帰（スタックが O(n) 成長）
def factorial(n):
    if n <= 1: return 1
    return n * factorial(n - 1)

# 末尾再帰（アキュムレータ使用）
def factorial_tail(n, acc=1):
    if n <= 1: return acc
    return factorial_tail(n - 1, acc * n)

# Python は末尾再帰最適化を行わない
# → 大きな n でスタックオーバーフロー
# → ループで書き直すのが推奨
```

```scheme
;; Scheme: 末尾再帰最適化（TCO）あり
(define (factorial n)
  (define (iter n acc)
    (if (<= n 1)
        acc
        (iter (- n 1) (* acc n))))  ; 末尾位置 → TCOで最適化
  (iter n 1))
;; スタックは成長しない（ループと同等の効率）
```

### TCO をサポートする言語

```
TCO あり:  Scheme, Haskell, Elixir/Erlang, Scala(@tailrec)
TCO 限定:  JavaScript(strict mode、実装依存)
TCO なし:  Python, Java, Go, Rust(明示的に使用しない)

TCO がない言語での対策:
  → ループに書き直す
  → トランポリン（後述）
```

---

## 4. メモ化（Memoization）

```python
# フィボナッチの問題: 同じ計算を何度も繰り返す
# fib(5)
#   fib(4) + fib(3)
#     fib(3)+fib(2)   fib(2)+fib(1)
#     ↑ fib(3) を2回計算している

# メモ化で解決
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(100)  # 一瞬で計算（O(n)）
```

```rust
// Rust: HashMap でメモ化
use std::collections::HashMap;

fn fibonacci(n: u64, memo: &mut HashMap<u64, u64>) -> u64 {
    if let Some(&result) = memo.get(&n) {
        return result;
    }
    let result = if n <= 1 { n } else {
        fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    };
    memo.insert(n, result);
    result
}
```

---

## 5. 再帰 vs ループの使い分け

```
再帰が適する場面:
  ✓ ツリー・グラフの走査
  ✓ 分割統治アルゴリズム（マージソート、クイックソート）
  ✓ パーサー・コンパイラ（構文木の処理）
  ✓ 数学的な定義に直接対応する場合
  ✓ バックトラッキング

ループが適する場面:
  ✓ 単純な繰り返し
  ✓ パフォーマンスが重要（TCOがない言語）
  ✓ 状態の更新が中心の処理
  ✓ 深い再帰でスタックオーバーフローの危険がある場合
```

```rust
// ループへの変換例: 木の走査
// 再帰版
fn sum_tree(node: &Node) -> i32 {
    let mut total = node.value;
    for child in &node.children {
        total += sum_tree(child);
    }
    total
}

// スタック明示版（再帰なし）
fn sum_tree_iterative(root: &Node) -> i32 {
    let mut stack = vec![root];
    let mut total = 0;
    while let Some(node) = stack.pop() {
        total += node.value;
        for child in &node.children {
            stack.push(child);
        }
    }
    total
}
```

---

## まとめ

| 概念 | 説明 |
|------|------|
| ベースケース | 再帰の終了条件（必須） |
| 再帰ステップ | 問題を小さくして再帰呼び出し |
| 末尾再帰 | 再帰が最後の操作。TCOで最適化可能 |
| メモ化 | 計算結果をキャッシュして重複排除 |
| 分割統治 | 問題を半分に分割して再帰（O(n log n)） |

---

## 次に読むべきガイド
→ [[../05-concurrency/00-threads-and-processes.md]] — 並行処理

---

## 参考文献
1. Abelson, H. & Sussman, G. "SICP." Ch.1.2, MIT Press, 1996.
2. Cormen, T. et al. "Introduction to Algorithms." Ch.4, MIT Press, 2022.
