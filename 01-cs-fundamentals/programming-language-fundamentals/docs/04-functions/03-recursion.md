# 再帰（Recursion）

> 再帰は「問題を同じ構造の小さな問題に分解する」技法。数学的に美しく、ツリー・グラフ・分割統治に不可欠。

## この章で学ぶこと

- [ ] 再帰の仕組みと基本パターンを理解する
- [ ] 末尾再帰と最適化を理解する
- [ ] 再帰とループの使い分けを判断できる
- [ ] メモ化による再帰の最適化を実践できる
- [ ] 分割統治法の各種アルゴリズムを理解する
- [ ] バックトラッキングと探索アルゴリズムを実装できる
- [ ] トランポリンと CPS による末尾呼び出し最適化の代替手法を理解する

---

## 1. 再帰の基本

### 1.1 再帰関数とは

```
再帰関数 = 自分自身を呼び出す関数

構成要素:
  1. ベースケース（終了条件）: 再帰を止める条件
  2. 再帰ステップ: 問題を小さくして自分自身を呼び出す

必須ルール:
  - 必ずベースケースに到達すること（無限再帰を防ぐ）
  - 各再帰呼び出しで問題が小さくなること（収束すること）
```

```
再帰呼び出しのイメージ:

  factorial(5)
    ├── 5 * factorial(4)
    │       ├── 4 * factorial(3)
    │       │       ├── 3 * factorial(2)
    │       │       │       ├── 2 * factorial(1)
    │       │       │       │       └── return 1  ← ベースケース
    │       │       │       └── return 2 * 1 = 2
    │       │       └── return 3 * 2 = 6
    │       └── return 4 * 6 = 24
    └── return 5 * 24 = 120
```

### 1.2 Python での基本的な再帰

```python
# 階乗: n! = n * (n-1)!
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

# 自然数の合計: sum(n) = n + sum(n-1)
def sum_natural(n):
    if n <= 0:
        return 0
    return n + sum_natural(n - 1)

# 累乗: power(base, exp) = base * power(base, exp-1)
def power(base, exp):
    if exp == 0:
        return 1
    if exp < 0:
        return 1 / power(base, -exp)
    return base * power(base, exp - 1)

# 最大公約数（ユークリッドの互除法）
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

# 文字列の反転
def reverse_string(s):
    if len(s) <= 1:
        return s
    return reverse_string(s[1:]) + s[0]

# 回文判定
def is_palindrome(s):
    if len(s) <= 1:
        return True
    if s[0] != s[-1]:
        return False
    return is_palindrome(s[1:-1])
```

### 1.3 TypeScript での基本的な再帰

```typescript
// 階乗
function factorial(n: number): number {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// フィボナッチ数列（素朴な再帰 - 非効率）
function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// 二分探索（再帰版）
function binarySearch(
    arr: number[],
    target: number,
    lo: number = 0,
    hi: number = arr.length - 1
): number {
    if (lo > hi) return -1;  // ベースケース: 見つからない
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] === target) return mid;  // ベースケース: 発見
    if (arr[mid] < target) {
        return binarySearch(arr, target, mid + 1, hi);
    }
    return binarySearch(arr, target, lo, mid - 1);
}

// 文字列のすべての部分集合（パワーセット）
function powerSet(s: string): string[] {
    if (s.length === 0) return [""];
    const first = s[0];
    const rest = powerSet(s.slice(1));
    return [...rest, ...rest.map(sub => first + sub)];
}

powerSet("abc");
// → ["", "c", "b", "bc", "a", "ac", "ab", "abc"]
```

### 1.4 コールスタックの視覚化

```
コールスタックの成長と縮小:

factorial(4) を呼び出した時のスタックの変化:

  Step 1:  [factorial(4)]
  Step 2:  [factorial(4), factorial(3)]
  Step 3:  [factorial(4), factorial(3), factorial(2)]
  Step 4:  [factorial(4), factorial(3), factorial(2), factorial(1)]
  Step 5:  [factorial(4), factorial(3), factorial(2)]  ← 1 を返す
  Step 6:  [factorial(4), factorial(3)]                ← 2 を返す
  Step 7:  [factorial(4)]                              ← 6 を返す
  Step 8:  []                                          ← 24 を返す

スタックの深さ = n（線形再帰の場合）
スタックオーバーフロー: 深さが言語のスタック制限を超えた場合に発生
  - Python: デフォルト 1000（sys.setrecursionlimit() で変更可能）
  - JavaScript: エンジン依存（通常 10,000～25,000）
  - Java: スレッドスタックサイズに依存（デフォルト 512KB～1MB）
```

---

## 2. 再帰のパターン

### 2.1 線形再帰

```python
# パターン1: 線形再帰（リスト処理）
# 各呼び出しで再帰を1回だけ行う → O(n)

def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

# リストの長さ
def length(lst):
    if not lst:
        return 0
    return 1 + length(lst[1:])

# リストの最大値
def maximum(lst):
    if len(lst) == 1:
        return lst[0]
    rest_max = maximum(lst[1:])
    return lst[0] if lst[0] > rest_max else rest_max

# リストのフラット化
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

flatten([1, [2, [3, 4], 5], [6, 7]])
# → [1, 2, 3, 4, 5, 6, 7]

# zip（再帰版）
def zip_lists(lst1, lst2):
    if not lst1 or not lst2:
        return []
    return [(lst1[0], lst2[0])] + zip_lists(lst1[1:], lst2[1:])
```

### 2.2 二分再帰（分割統治）

```python
# パターン2: 二分再帰（分割統治）
# 各呼び出しで再帰を2回行う

# マージソート
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# クイックソート
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# 高速累乗（分割統治）: O(log n)
def fast_power(base, exp):
    if exp == 0:
        return 1
    if exp % 2 == 0:
        half = fast_power(base, exp // 2)
        return half * half
    else:
        return base * fast_power(base, exp - 1)

# カラツバ乗算（大きな整数の高速乗算）
def karatsuba(x, y):
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    half = n // 2
    power = 10 ** half

    a, b = divmod(x, power)  # x = a * 10^half + b
    c, d = divmod(y, power)  # y = c * 10^half + d

    # 3回の乗算で済む（通常は4回必要）
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    ad_bc = karatsuba(a + b, c + d) - ac - bd

    return ac * (10 ** (2 * half)) + ad_bc * (10 ** half) + bd
```

### 2.3 ツリー再帰

```python
# パターン3: ツリー再帰
# 各呼び出しで2回以上の再帰呼び出しを行う

# フィボナッチ（ナイーブ版 - O(2^n)）
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
# 注意: 指数的な計算量。メモ化が必要

# フィボナッチの呼び出しツリー:
# fib(5)
# ├── fib(4)
# │   ├── fib(3)
# │   │   ├── fib(2)
# │   │   │   ├── fib(1) → 1
# │   │   │   └── fib(0) → 0
# │   │   └── fib(1) → 1
# │   └── fib(2)
# │       ├── fib(1) → 1
# │       └── fib(0) → 0
# └── fib(3)
#     ├── fib(2)
#     │   ├── fib(1) → 1
#     │   └── fib(0) → 0
#     └── fib(1) → 1
# → 同じ計算が何度も繰り返される

# パスカルの三角形
def pascal(row, col):
    if col == 0 or col == row:
        return 1
    return pascal(row - 1, col - 1) + pascal(row - 1, col)

# パスカルの三角形を表示
def print_pascal(n):
    for row in range(n):
        values = [pascal(row, col) for col in range(row + 1)]
        print(" " * (n - row), " ".join(f"{v:3}" for v in values))

print_pascal(6)
#       1
#      1   1
#     1   2   1
#    1   3   3   1
#   1   4   6   4   1
#  1   5  10  10   5   1

# 組み合わせの数 C(n, k) = C(n-1, k-1) + C(n-1, k)
def combinations_count(n, k):
    if k == 0 or k == n:
        return 1
    if k < 0 or k > n:
        return 0
    return combinations_count(n - 1, k - 1) + combinations_count(n - 1, k)
```

### 2.4 相互再帰

```python
# パターン4: 相互再帰
# 2つ以上の関数が互いを呼び出す

def is_even(n):
    if n == 0: return True
    return is_odd(n - 1)

def is_odd(n):
    if n == 0: return False
    return is_even(n - 1)

# 数式パーサーの相互再帰
# expr   = term (('+' | '-') term)*
# term   = factor (('*' | '/') factor)*
# factor = number | '(' expr ')'

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse_expr(self):
        """expr = term (('+' | '-') term)*"""
        result = self.parse_term()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result

    def parse_term(self):
        """term = factor (('*' | '/') factor)*"""
        result = self.parse_factor()
        while self.peek() in ('*', '/'):
            op = self.consume()
            right = self.parse_factor()
            if op == '*':
                result *= right
            else:
                result /= right
        return result

    def parse_factor(self):
        """factor = number | '(' expr ')'"""
        if self.peek() == '(':
            self.consume()  # '(' を消費
            result = self.parse_expr()  # 再帰: factor → expr → term → factor
            self.consume()  # ')' を消費
            return result
        return float(self.consume())

# 使用例
tokens = ['(', '2', '+', '3', ')', '*', '4']
parser = Parser(tokens)
print(parser.parse_expr())  # → 20.0
```

---

## 3. ツリー構造の再帰処理

### 3.1 ファイルシステムの走査

```typescript
// ファイルシステムの走査
interface FSNode {
    name: string;
    type: "file" | "directory";
    children?: FSNode[];
    size?: number;
}

// 総サイズの計算
function totalSize(node: FSNode): number {
    if (node.type === "file") {
        return node.size ?? 0;  // ベースケース
    }
    return (node.children ?? [])
        .reduce((sum, child) => sum + totalSize(child), 0);
}

// ファイルの検索（深さ優先）
function findFiles(
    node: FSNode,
    predicate: (node: FSNode) => boolean
): FSNode[] {
    const results: FSNode[] = [];
    if (node.type === "file" && predicate(node)) {
        results.push(node);
    }
    if (node.children) {
        for (const child of node.children) {
            results.push(...findFiles(child, predicate));
        }
    }
    return results;
}

// ディレクトリツリーの文字列表現
function renderTree(node: FSNode, indent: string = "", isLast: boolean = true): string {
    const prefix = indent + (isLast ? "└── " : "├── ");
    const childIndent = indent + (isLast ? "    " : "│   ");
    let result = prefix + node.name + "\n";

    if (node.children) {
        node.children.forEach((child, i) => {
            result += renderTree(child, childIndent, i === node.children!.length - 1);
        });
    }
    return result;
}

// 使用例
const root: FSNode = {
    name: "project",
    type: "directory",
    children: [
        {
            name: "src",
            type: "directory",
            children: [
                { name: "index.ts", type: "file", size: 1024 },
                { name: "utils.ts", type: "file", size: 512 },
                {
                    name: "components",
                    type: "directory",
                    children: [
                        { name: "Header.tsx", type: "file", size: 2048 },
                        { name: "Footer.tsx", type: "file", size: 1536 },
                    ],
                },
            ],
        },
        { name: "README.md", type: "file", size: 256 },
    ],
};

console.log(totalSize(root));
// → 5376

console.log(findFiles(root, n => n.name.endsWith(".tsx")).map(n => n.name));
// → ["Header.tsx", "Footer.tsx"]

console.log(renderTree(root));
// └── project
//     ├── src
//     │   ├── index.ts
//     │   ├── utils.ts
//     │   └── components
//     │       ├── Header.tsx
//     │       └── Footer.tsx
//     └── README.md
```

### 3.2 JSON / ネストオブジェクトの再帰処理

```typescript
// JSON の深い値を取得
function deepGet(obj: any, path: string[]): any {
    if (path.length === 0) return obj;
    if (obj == null) return undefined;
    const [head, ...tail] = path;
    return deepGet(obj[head], tail);
}

deepGet({ a: { b: { c: 42 } } }, ["a", "b", "c"]);  // → 42

// 深いマージ
function deepMerge(target: any, source: any): any {
    if (typeof target !== "object" || typeof source !== "object") {
        return source;
    }
    if (Array.isArray(target) && Array.isArray(source)) {
        return [...target, ...source];
    }
    const result = { ...target };
    for (const key of Object.keys(source)) {
        if (key in result && typeof result[key] === "object" && typeof source[key] === "object") {
            result[key] = deepMerge(result[key], source[key]);
        } else {
            result[key] = source[key];
        }
    }
    return result;
}

// 深い比較（deep equality）
function deepEqual(a: any, b: any): boolean {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (typeof a !== typeof b) return false;

    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return false;
        return a.every((item, i) => deepEqual(item, b[i]));
    }

    if (typeof a === "object") {
        const keysA = Object.keys(a);
        const keysB = Object.keys(b);
        if (keysA.length !== keysB.length) return false;
        return keysA.every(key => deepEqual(a[key], b[key]));
    }

    return false;
}

// 深いクローン
function deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== "object") return obj;
    if (obj instanceof Date) return new Date(obj.getTime()) as any;
    if (obj instanceof RegExp) return new RegExp(obj.source, obj.flags) as any;
    if (Array.isArray(obj)) return obj.map(item => deepClone(item)) as any;

    const cloned = {} as T;
    for (const key in obj) {
        if (Object.prototype.hasOwnProperty.call(obj, key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}

// オブジェクトの全パスを列挙
function allPaths(obj: any, prefix: string = ""): string[] {
    if (typeof obj !== "object" || obj === null) {
        return [prefix];
    }
    return Object.entries(obj).flatMap(([key, value]) => {
        const path = prefix ? `${prefix}.${key}` : key;
        return allPaths(value, path);
    });
}

allPaths({ a: { b: 1, c: { d: 2 } }, e: 3 });
// → ["a.b", "a.c.d", "e"]

// オブジェクトの平坦化
function flattenObject(obj: any, prefix: string = ""): Record<string, any> {
    const result: Record<string, any> = {};
    for (const [key, value] of Object.entries(obj)) {
        const path = prefix ? `${prefix}.${key}` : key;
        if (typeof value === "object" && value !== null && !Array.isArray(value)) {
            Object.assign(result, flattenObject(value, path));
        } else {
            result[path] = value;
        }
    }
    return result;
}

flattenObject({ a: { b: 1, c: { d: 2 } }, e: 3 });
// → { "a.b": 1, "a.c.d": 2, "e": 3 }
```

### 3.3 DOM ツリーの再帰処理

```typescript
// DOM ツリーの走査
function walkDOM(node: Node, callback: (node: Node) => void): void {
    callback(node);
    let child = node.firstChild;
    while (child) {
        walkDOM(child, callback);
        child = child.nextSibling;
    }
}

// 特定の条件に一致する要素を再帰的に検索
function findElement(
    node: Element,
    predicate: (el: Element) => boolean
): Element | null {
    if (predicate(node)) return node;
    for (const child of Array.from(node.children)) {
        const found = findElement(child, predicate);
        if (found) return found;
    }
    return null;
}

// React コンポーネントツリーの再帰レンダリング
interface TreeItem {
    id: string;
    label: string;
    children?: TreeItem[];
}

// 再帰的なツリーコンポーネント
function TreeView({ items, depth = 0 }: { items: TreeItem[]; depth?: number }) {
    return (
        <ul style={{ paddingLeft: depth > 0 ? 20 : 0 }}>
            {items.map(item => (
                <li key={item.id}>
                    {item.label}
                    {item.children && item.children.length > 0 && (
                        <TreeView items={item.children} depth={depth + 1} />
                    )}
                </li>
            ))}
        </ul>
    );
}
```

---

## 4. 末尾再帰（Tail Recursion）

### 4.1 末尾再帰とは

```
末尾再帰 = 再帰呼び出しが関数の最後の操作

  通常の再帰: return n * factorial(n - 1)
              ↑ 再帰の結果に n を掛ける → スタックに n を保持

  末尾再帰:   return factorial_tail(n - 1, acc * n)
              ↑ 再帰呼び出しが最後 → スタックフレームを再利用可能

通常の再帰のスタック:
  factorial(4)           ← スタックに 4 を保持
    factorial(3)         ← スタックに 3 を保持
      factorial(2)       ← スタックに 2 を保持
        factorial(1)     ← ベースケース
      return 2 * 1       ← 巻き戻し
    return 3 * 2
  return 4 * 6
→ スタック深さ: O(n)

末尾再帰のスタック（TCO あり）:
  factorial_tail(4, 1)   → factorial_tail(3, 4)
                         → factorial_tail(2, 12)
                         → factorial_tail(1, 24)
                         → return 24
→ スタック深さ: O(1)（スタックフレームを再利用）
```

### 4.2 末尾再帰への変換

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

# 末尾再帰への変換パターン
# 通常の再帰を末尾再帰に変換する一般的な手法:
# 1. アキュムレータ（accumulator）を導入
# 2. 計算結果をアキュムレータに蓄積
# 3. ベースケースでアキュムレータを返す

# 例: リストの合計
# 通常の再帰
def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

# 末尾再帰
def sum_list_tail(lst, acc=0):
    if not lst:
        return acc
    return sum_list_tail(lst[1:], acc + lst[0])

# 例: リストの反転
# 通常の再帰
def reverse_list(lst):
    if not lst:
        return []
    return reverse_list(lst[1:]) + [lst[0]]

# 末尾再帰
def reverse_list_tail(lst, acc=None):
    if acc is None:
        acc = []
    if not lst:
        return acc
    return reverse_list_tail(lst[1:], [lst[0]] + acc)

# 例: フィボナッチ
# 通常の再帰（ツリー再帰 - O(2^n)）
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# 末尾再帰（線形 - O(n)）
def fib_tail(n, a=0, b=1):
    if n == 0:
        return a
    return fib_tail(n - 1, b, a + b)
```

### 4.3 TCO をサポートする言語

```
TCO あり:  Scheme, Haskell, Elixir/Erlang, Scala(@tailrec)
TCO 限定:  JavaScript(strict mode、実装依存)
TCO なし:  Python, Java, Go, Rust(明示的に使用しない)

TCO がない言語での対策:
  → ループに書き直す
  → トランポリン（後述）
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

```scala
// Scala: @tailrec アノテーションで末尾再帰を保証
import scala.annotation.tailrec

def factorial(n: Long): Long = {
  @tailrec
  def loop(n: Long, acc: Long): Long = {
    if (n <= 1) acc
    else loop(n - 1, acc * n)  // コンパイラが末尾再帰を検証
  }
  loop(n, 1)
}

// 末尾再帰でない場合はコンパイルエラー
// @tailrec
// def badFactorial(n: Long): Long = {
//   if (n <= 1) 1
//   else n * badFactorial(n - 1)  // エラー: 末尾位置ではない
// }
```

```elixir
# Elixir: 末尾再帰が推奨される関数型言語

defmodule Math do
  # 末尾再帰版
  def factorial(n), do: factorial(n, 1)

  defp factorial(0, acc), do: acc
  defp factorial(n, acc) when n > 0 do
    factorial(n - 1, acc * n)
  end

  # リストの長さ（末尾再帰）
  def length(list), do: length(list, 0)

  defp length([], acc), do: acc
  defp length([_head | tail], acc) do
    length(tail, acc + 1)
  end

  # map（末尾再帰 + アキュムレータ + reverse）
  def map(list, func), do: map(list, func, [])

  defp map([], _func, acc), do: Enum.reverse(acc)
  defp map([head | tail], func, acc) do
    map(tail, func, [func.(head) | acc])
  end
end
```

---

## 5. メモ化（Memoization）

### 5.1 メモ化の基本

```python
# フィボナッチの問題: 同じ計算を何度も繰り返す
# fib(5)
#   fib(4) + fib(3)
#     fib(3)+fib(2)   fib(2)+fib(1)
#     ↑ fib(3) を2回計算している

# 計算量の比較:
# メモ化なし: O(2^n) - 指数的に爆発
# メモ化あり: O(n)   - 各値を1回だけ計算

# 手動メモ化
def fibonacci_memo(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# デコレータによるメモ化
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(100)  # 一瞬で計算（O(n)）

# キャッシュ情報の確認
print(fibonacci.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=None, currsize=101)
```

### 5.2 各言語でのメモ化

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

fn main() {
    let mut memo = HashMap::new();
    println!("{}", fibonacci(50, &mut memo));  // → 12586269025
}
```

```typescript
// TypeScript: 汎用メモ化デコレータ
function memoize<Args extends any[], R>(
    fn: (...args: Args) => R,
    keyFn: (...args: Args) => string = (...args) => JSON.stringify(args)
): (...args: Args) => R {
    const cache = new Map<string, R>();
    return (...args: Args): R => {
        const key = keyFn(...args);
        if (cache.has(key)) return cache.get(key)!;
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
}

// 使用例: 経路の数（格子上の右下への移動）
const gridPaths = memoize((rows: number, cols: number): number => {
    if (rows === 1 || cols === 1) return 1;
    return gridPaths(rows - 1, cols) + gridPaths(rows, cols - 1);
});

console.log(gridPaths(18, 18));  // → 2333606220（メモ化なしでは非常に遅い）

// LRU キャッシュ（サイズ制限付きメモ化）
class LRUCache<K, V> {
    private cache = new Map<K, V>();

    constructor(private maxSize: number) {}

    get(key: K): V | undefined {
        if (!this.cache.has(key)) return undefined;
        // アクセスされたエントリを末尾に移動（最近使用された）
        const value = this.cache.get(key)!;
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    set(key: K, value: V): void {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            // 最も古いエントリを削除
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey!);
        }
        this.cache.set(key, value);
    }
}
```

### 5.3 動的計画法との関係

```python
# メモ化再帰 = トップダウン動的計画法（Top-down DP）
# テーブル法 = ボトムアップ動的計画法（Bottom-up DP）

# 例: 最長共通部分列（LCS）

# メモ化再帰（トップダウン）
@lru_cache(maxsize=None)
def lcs_topdown(s1, s2, i=None, j=None):
    if i is None: i = len(s1) - 1
    if j is None: j = len(s2) - 1
    if i < 0 or j < 0:
        return 0
    if s1[i] == s2[j]:
        return 1 + lcs_topdown(s1, s2, i - 1, j - 1)
    return max(lcs_topdown(s1, s2, i - 1, j), lcs_topdown(s1, s2, i, j - 1))

# テーブル法（ボトムアップ）
def lcs_bottomup(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# 例: ナップサック問題
@lru_cache(maxsize=None)
def knapsack(weights, values, capacity, i=None):
    if i is None:
        i = len(weights) - 1
    if i < 0 or capacity <= 0:
        return 0
    # アイテム i を入れない場合
    without = knapsack(weights, values, capacity, i - 1)
    # アイテム i を入れる場合
    if weights[i] <= capacity:
        with_item = values[i] + knapsack(weights, values, capacity - weights[i], i - 1)
        return max(without, with_item)
    return without

# 例: コイン交換問題
@lru_cache(maxsize=None)
def coin_change(coins, amount):
    """金額 amount を作るのに必要な最小コイン枚数"""
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')
    min_coins = float('inf')
    for coin in coins:
        result = coin_change(coins, amount - coin)
        min_coins = min(min_coins, result + 1)
    return min_coins

print(coin_change((1, 5, 10, 25), 63))  # → 6 (25+25+10+1+1+1)

# 例: 階段の登り方（1段 or 2段）
@lru_cache(maxsize=None)
def climb_stairs(n):
    if n <= 1:
        return 1
    return climb_stairs(n - 1) + climb_stairs(n - 2)

print(climb_stairs(10))  # → 89
```

---

## 6. バックトラッキング

### 6.1 基本概念

```
バックトラッキング = 探索木を深さ優先で探索し、
                    行き詰まったら1つ前の状態に戻って別の選択肢を試す

アルゴリズム:
  1. 現在の状態が解か確認
  2. 解なら記録して終了（or 続行）
  3. 可能な次の選択肢を列挙
  4. 各選択肢について:
     a. 選択を適用
     b. 再帰的に探索
     c. 選択を取り消す（バックトラック）

┌─────────────┐
│   Start     │
└──────┬──────┘
       │
   ┌───┴───┐
   │ 選択1  │ 選択2  選択3
   └───┬───┘
       │
   ┌───┴───┐
   │ 選択A  │ 選択B
   └───┬───┘
       │
    行き詰まり → バックトラック → 選択B を試す
```

### 6.2 N-Queens 問題

```python
def solve_n_queens(n):
    """N-Queens問題: N×Nのチェス盤にN個のクイーンを互いに攻撃しない位置に配置"""
    solutions = []

    def is_safe(board, row, col):
        # 同じ列にクイーンがないか
        for r in range(row):
            if board[r] == col:
                return False
            # 対角線にクイーンがないか
            if abs(board[r] - col) == row - r:
                return False
        return True

    def backtrack(board, row):
        if row == n:
            solutions.append(board[:])  # 解を記録
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col         # 選択を適用
                backtrack(board, row + 1) # 再帰的に探索
                board[row] = -1          # バックトラック

    backtrack([-1] * n, 0)
    return solutions

# 8-Queens の解
solutions = solve_n_queens(8)
print(f"解の数: {len(solutions)}")  # → 92

# 解の可視化
def print_board(solution):
    n = len(solution)
    for row in range(n):
        line = ""
        for col in range(n):
            line += "Q " if solution[row] == col else ". "
        print(line)
    print()

print_board(solutions[0])
# Q . . . . . . .
# . . . . Q . . .
# . . . . . . . Q
# . . . . . Q . .
# . . Q . . . . .
# . . . . . . Q .
# . Q . . . . . .
# . . . Q . . . .
```

### 6.3 数独ソルバー

```python
def solve_sudoku(board):
    """数独を解く（バックトラッキング）"""
    def find_empty():
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return (r, c)
        return None

    def is_valid(num, row, col):
        # 行チェック
        if num in board[row]:
            return False
        # 列チェック
        if num in [board[r][col] for r in range(9)]:
            return False
        # 3x3ブロックチェック
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if board[r][c] == num:
                    return False
        return True

    empty = find_empty()
    if not empty:
        return True  # 全マス埋まった → 解決

    row, col = empty
    for num in range(1, 10):
        if is_valid(num, row, col):
            board[row][col] = num       # 選択
            if solve_sudoku(board):     # 再帰
                return True
            board[row][col] = 0         # バックトラック

    return False  # この分岐には解がない
```

### 6.4 迷路の解法

```typescript
// 迷路の解法（バックトラッキング）
type Maze = number[][];  // 0 = 通路, 1 = 壁
type Position = [number, number];

function solveMaze(
    maze: Maze,
    start: Position,
    end: Position
): Position[] | null {
    const rows = maze.length;
    const cols = maze[0].length;
    const visited = Array.from({ length: rows }, () =>
        Array(cols).fill(false)
    );

    function backtrack(r: number, c: number, path: Position[]): Position[] | null {
        // 境界外 or 壁 or 訪問済み
        if (r < 0 || r >= rows || c < 0 || c >= cols) return null;
        if (maze[r][c] === 1 || visited[r][c]) return null;

        path.push([r, c]);
        visited[r][c] = true;

        // ゴールに到達
        if (r === end[0] && c === end[1]) {
            return [...path];
        }

        // 4方向を探索
        const directions: Position[] = [[0, 1], [1, 0], [0, -1], [-1, 0]];
        for (const [dr, dc] of directions) {
            const result = backtrack(r + dr, c + dc, path);
            if (result) return result;
        }

        // バックトラック
        path.pop();
        visited[r][c] = false;
        return null;
    }

    return backtrack(start[0], start[1], []);
}

// 使用例
const maze: Maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
];

const path = solveMaze(maze, [0, 0], [4, 4]);
// → [[0,0], [1,0], [2,0], [2,1], [2,2], [3,2], [3,3], [3,4], [4,4]]
// (...が正しいルートの一例)
```

### 6.5 順列と組み合わせの生成

```typescript
// 順列の生成（バックトラッキング）
function permutations<T>(arr: T[]): T[][] {
    const results: T[][] = [];

    function backtrack(current: T[], remaining: T[]) {
        if (remaining.length === 0) {
            results.push([...current]);
            return;
        }
        for (let i = 0; i < remaining.length; i++) {
            current.push(remaining[i]);
            const newRemaining = [...remaining.slice(0, i), ...remaining.slice(i + 1)];
            backtrack(current, newRemaining);
            current.pop();  // バックトラック
        }
    }

    backtrack([], arr);
    return results;
}

// 組み合わせの生成
function combinations<T>(arr: T[], k: number): T[][] {
    const results: T[][] = [];

    function backtrack(start: number, current: T[]) {
        if (current.length === k) {
            results.push([...current]);
            return;
        }
        for (let i = start; i < arr.length; i++) {
            current.push(arr[i]);
            backtrack(i + 1, current);
            current.pop();
        }
    }

    backtrack(0, []);
    return results;
}

// 部分集合の合計が目標値になる組み合わせ
function subsetSum(nums: number[], target: number): number[][] {
    const results: number[][] = [];

    function backtrack(start: number, current: number[], remaining: number) {
        if (remaining === 0) {
            results.push([...current]);
            return;
        }
        if (remaining < 0) return;

        for (let i = start; i < nums.length; i++) {
            // 重複をスキップ
            if (i > start && nums[i] === nums[i - 1]) continue;
            current.push(nums[i]);
            backtrack(i + 1, current, remaining - nums[i]);
            current.pop();
        }
    }

    nums.sort((a, b) => a - b);
    backtrack(0, [], target);
    return results;
}
```

---

## 7. 再帰 vs ループの使い分け

### 7.1 判断基準

```
再帰が適する場面:
  - ツリー・グラフの走査
  - 分割統治アルゴリズム（マージソート、クイックソート）
  - パーサー・コンパイラ（構文木の処理）
  - 数学的な定義に直接対応する場合
  - バックトラッキング
  - ネストしたデータ構造の処理

ループが適する場面:
  - 単純な繰り返し
  - パフォーマンスが重要（TCOがない言語）
  - 状態の更新が中心の処理
  - 深い再帰でスタックオーバーフローの危険がある場合
  - データが平坦な場合

判断フローチャート:
  データ構造が再帰的（ツリー、グラフ）？
    → YES → 再帰が自然
    → NO → ループが自然

  スタックオーバーフローの危険がある？
    → YES → ループ or トランポリン
    → NO → 再帰OK

  TCOが使える言語？
    → YES → 末尾再帰
    → NO → 深い再帰はループに変換
```

### 7.2 ループへの変換例

```rust
// ループへの変換例: 木の走査

// 再帰版
struct Node {
    value: i32,
    children: Vec<Node>,
}

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

// キュー版（幅優先探索 BFS）
use std::collections::VecDeque;

fn sum_tree_bfs(root: &Node) -> i32 {
    let mut queue = VecDeque::new();
    queue.push_back(root);
    let mut total = 0;
    while let Some(node) = queue.pop_front() {
        total += node.value;
        for child in &node.children {
            queue.push_back(child);
        }
    }
    total
}
```

```typescript
// TypeScript: 再帰からループへの変換パターン

// パターン1: 単純な末尾再帰 → while ループ
// 再帰版
function gcd(a: number, b: number): number {
    if (b === 0) return a;
    return gcd(b, a % b);
}

// ループ版
function gcdLoop(a: number, b: number): number {
    while (b !== 0) {
        [a, b] = [b, a % b];
    }
    return a;
}

// パターン2: 深さ優先探索 → 明示的スタック
// 再帰版
function flattenTree<T>(node: TreeNode<T>): T[] {
    const result = [node.value];
    for (const child of node.children) {
        result.push(...flattenTree(child));
    }
    return result;
}

// 明示的スタック版
function flattenTreeIterative<T>(root: TreeNode<T>): T[] {
    const result: T[] = [];
    const stack: TreeNode<T>[] = [root];
    while (stack.length > 0) {
        const node = stack.pop()!;
        result.push(node.value);
        // children を逆順で push（最初の子が先に pop されるように）
        for (let i = node.children.length - 1; i >= 0; i--) {
            stack.push(node.children[i]);
        }
    }
    return result;
}

// パターン3: バックトラッキング → 明示的状態管理
// 再帰版の順列
function permsRecursive(arr: number[]): number[][] {
    if (arr.length <= 1) return [arr];
    return arr.flatMap((item, i) => {
        const rest = [...arr.slice(0, i), ...arr.slice(i + 1)];
        return permsRecursive(rest).map(perm => [item, ...perm]);
    });
}

// 明示的状態管理版
function permsIterative(arr: number[]): number[][] {
    const results: number[][] = [];
    interface State {
        current: number[];
        remaining: number[];
    }
    const stack: State[] = [{ current: [], remaining: arr }];

    while (stack.length > 0) {
        const { current, remaining } = stack.pop()!;
        if (remaining.length === 0) {
            results.push(current);
            continue;
        }
        for (let i = remaining.length - 1; i >= 0; i--) {
            stack.push({
                current: [...current, remaining[i]],
                remaining: [
                    ...remaining.slice(0, i),
                    ...remaining.slice(i + 1),
                ],
            });
        }
    }
    return results;
}
```

---

## 8. トランポリンと CPS

### 8.1 トランポリン（Trampoline）

```typescript
// トランポリン: TCO のない言語で末尾再帰を安全に実行する技法
// 再帰呼び出しの代わりに「次に呼ぶべき関数」を返し、
// ループで繰り返し呼び出す

type Thunk<T> = () => T | Thunk<T>;

function trampoline<T>(fn: Thunk<T>): T {
    let result: any = fn;
    while (typeof result === "function") {
        result = result();
    }
    return result;
}

// 使用例: 安全な階乗（スタックオーバーフローなし）
function factorialTramp(n: number, acc: number = 1): number | (() => number | (() => any)) {
    if (n <= 1) return acc;
    return () => factorialTramp(n - 1, acc * n);  // 関数を返す（呼び出さない）
}

console.log(trampoline(() => factorialTramp(100000)));
// → Infinity（数値のオーバーフローだがスタックは溢れない）

// より型安全なトランポリン
type Bounce<T> =
    | { done: true; value: T }
    | { done: false; thunk: () => Bounce<T> };

function done<T>(value: T): Bounce<T> {
    return { done: true, value };
}

function bounce<T>(thunk: () => Bounce<T>): Bounce<T> {
    return { done: false, thunk };
}

function run<T>(b: Bounce<T>): T {
    let current = b;
    while (!current.done) {
        current = current.thunk();
    }
    return current.value;
}

// 使用例: 相互再帰のトランポリン化
function isEvenTramp(n: number): Bounce<boolean> {
    if (n === 0) return done(true);
    return bounce(() => isOddTramp(n - 1));
}

function isOddTramp(n: number): Bounce<boolean> {
    if (n === 0) return done(false);
    return bounce(() => isEvenTramp(n - 1));
}

console.log(run(isEvenTramp(1000000)));  // → true（スタックオーバーフローなし）
```

### 8.2 CPS（Continuation-Passing Style）

```typescript
// CPS: 再帰の結果をコールバックに渡すことで末尾再帰化する

// 直接スタイル
function sumList(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr[0] + sumList(arr.slice(1));
}

// CPS 変換
function sumListCPS(arr: number[], k: (result: number) => number): number {
    if (arr.length === 0) return k(0);
    return sumListCPS(arr.slice(1), (restSum) => k(arr[0] + restSum));
}

sumListCPS([1, 2, 3, 4, 5], x => x);  // → 15

// ツリーの CPS 走査
interface TreeNode {
    value: number;
    left?: TreeNode;
    right?: TreeNode;
}

// 直接スタイル（スタック深さ = ツリーの深さ）
function sumTree(node: TreeNode | undefined): number {
    if (!node) return 0;
    return node.value + sumTree(node.left) + sumTree(node.right);
}

// CPS（末尾再帰化可能）
function sumTreeCPS(
    node: TreeNode | undefined,
    k: (result: number) => number
): number {
    if (!node) return k(0);
    return sumTreeCPS(node.left, (leftSum) =>
        sumTreeCPS(node.right, (rightSum) =>
            k(node.value + leftSum + rightSum)
        )
    );
}
```

---

## 9. 再帰を使った実用的なアルゴリズム

### 9.1 分割統治アルゴリズム

```python
# 分割統治法の一般的な構造:
# 1. 分割（Divide）: 問題を小さな部分問題に分ける
# 2. 統治（Conquer）: 各部分問題を再帰的に解く
# 3. 結合（Combine）: 部分問題の解を統合する

# 最大部分配列和（Kadane's vs 分割統治）
def max_subarray_dc(arr, lo=0, hi=None):
    """分割統治法による最大部分配列和 O(n log n)"""
    if hi is None:
        hi = len(arr) - 1
    if lo == hi:
        return arr[lo]

    mid = (lo + hi) // 2

    # 左半分の最大部分配列和
    left_max = max_subarray_dc(arr, lo, mid)
    # 右半分の最大部分配列和
    right_max = max_subarray_dc(arr, mid + 1, hi)

    # 中央をまたぐ最大部分配列和
    left_sum = float('-inf')
    total = 0
    for i in range(mid, lo - 1, -1):
        total += arr[i]
        left_sum = max(left_sum, total)

    right_sum = float('-inf')
    total = 0
    for i in range(mid + 1, hi + 1):
        total += arr[i]
        right_sum = max(right_sum, total)

    cross_max = left_sum + right_sum

    return max(left_max, right_max, cross_max)

# 最近点対問題
import math

def closest_pair(points):
    """平面上の最近点対を求める O(n log n)"""
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_pair_rec(pts_x, pts_y):
        n = len(pts_x)
        if n <= 3:
            # ブルートフォース
            min_dist = float('inf')
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    d = distance(pts_x[i], pts_x[j])
                    if d < min_dist:
                        min_dist = d
                        best_pair = (pts_x[i], pts_x[j])
            return min_dist, best_pair

        mid = n // 2
        mid_point = pts_x[mid]

        # 分割
        left_x = pts_x[:mid]
        right_x = pts_x[mid:]
        left_y = [p for p in pts_y if p[0] <= mid_point[0]]
        right_y = [p for p in pts_y if p[0] > mid_point[0]]

        # 統治（再帰）
        dl, pair_l = closest_pair_rec(left_x, left_y)
        dr, pair_r = closest_pair_rec(right_x, right_y)

        d = min(dl, dr)
        best = pair_l if dl < dr else pair_r

        # 結合: ストリップ内の点をチェック
        strip = [p for p in pts_y if abs(p[0] - mid_point[0]) < d]
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 8, len(strip))):
                dd = distance(strip[i], strip[j])
                if dd < d:
                    d = dd
                    best = (strip[i], strip[j])

        return d, best

    pts_x = sorted(points, key=lambda p: p[0])
    pts_y = sorted(points, key=lambda p: p[1])
    return closest_pair_rec(pts_x, pts_y)

# ストラッセンの行列乗算（概念）
# 通常の行列乗算: O(n^3)
# ストラッセン: O(n^2.807) - 分割統治で7回の乗算に削減
```

### 9.2 グラフの再帰的探索

```typescript
// グラフの DFS（深さ優先探索）
type Graph = Map<string, string[]>;

// 再帰版 DFS
function dfs(
    graph: Graph,
    start: string,
    visited: Set<string> = new Set()
): string[] {
    visited.add(start);
    const result = [start];

    for (const neighbor of graph.get(start) || []) {
        if (!visited.has(neighbor)) {
            result.push(...dfs(graph, neighbor, visited));
        }
    }

    return result;
}

// トポロジカルソート（DAG の依存関係順序）
function topologicalSort(graph: Graph): string[] {
    const visited = new Set<string>();
    const result: string[] = [];

    function visit(node: string) {
        if (visited.has(node)) return;
        visited.add(node);
        for (const neighbor of graph.get(node) || []) {
            visit(neighbor);
        }
        result.unshift(node);  // 後処理で先頭に追加
    }

    for (const node of graph.keys()) {
        visit(node);
    }

    return result;
}

// 循環検出
function hasCycle(graph: Graph): boolean {
    const white = new Set<string>(graph.keys()); // 未訪問
    const gray = new Set<string>();               // 訪問中
    const black = new Set<string>();              // 完了

    function dfsVisit(node: string): boolean {
        white.delete(node);
        gray.add(node);

        for (const neighbor of graph.get(node) || []) {
            if (gray.has(neighbor)) return true;  // 循環検出
            if (white.has(neighbor) && dfsVisit(neighbor)) return true;
        }

        gray.delete(node);
        black.add(node);
        return false;
    }

    for (const node of [...white]) {
        if (dfsVisit(node)) return true;
    }
    return false;
}

// 使用例
const dependencyGraph: Graph = new Map([
    ["main", ["auth", "db", "logger"]],
    ["auth", ["db", "crypto"]],
    ["db", ["logger"]],
    ["crypto", []],
    ["logger", []],
]);

console.log(topologicalSort(dependencyGraph));
// → ["main", "auth", "db", "crypto", "logger"]
// （または別の有効な順序）
```

### 9.3 再帰下降パーサー

```typescript
// 再帰下降パーサー: 文法規則に対応する再帰関数群

// 簡単な数式言語:
// expr   = term (('+' | '-') term)*
// term   = factor (('*' | '/') factor)*
// factor = NUMBER | '(' expr ')'

type Token =
    | { type: "number"; value: number }
    | { type: "op"; value: string }
    | { type: "paren"; value: "(" | ")" };

class ExprParser {
    private pos = 0;

    constructor(private tokens: Token[]) {}

    private peek(): Token | null {
        return this.pos < this.tokens.length ? this.tokens[this.pos] : null;
    }

    private consume(): Token {
        return this.tokens[this.pos++];
    }

    private expect(type: string, value?: string): Token {
        const token = this.consume();
        if (token.type !== type || (value !== undefined && token.value !== value)) {
            throw new Error(`Expected ${type}(${value}), got ${token.type}(${token.value})`);
        }
        return token;
    }

    parse(): number {
        const result = this.parseExpr();
        if (this.pos < this.tokens.length) {
            throw new Error("Unexpected tokens after expression");
        }
        return result;
    }

    private parseExpr(): number {
        let result = this.parseTerm();
        while (this.peek()?.type === "op" &&
               (this.peek()!.value === "+" || this.peek()!.value === "-")) {
            const op = this.consume().value;
            const right = this.parseTerm();
            result = op === "+" ? result + right : result - right;
        }
        return result;
    }

    private parseTerm(): number {
        let result = this.parseFactor();
        while (this.peek()?.type === "op" &&
               (this.peek()!.value === "*" || this.peek()!.value === "/")) {
            const op = this.consume().value;
            const right = this.parseFactor();
            result = op === "*" ? result * right : result / right;
        }
        return result;
    }

    private parseFactor(): number {
        const token = this.peek();
        if (token?.type === "number") {
            this.consume();
            return token.value as number;
        }
        if (token?.type === "paren" && token.value === "(") {
            this.consume();                    // '(' を消費
            const result = this.parseExpr();   // 相互再帰
            this.expect("paren", ")");         // ')' を消費
            return result;
        }
        throw new Error(`Unexpected token: ${JSON.stringify(token)}`);
    }
}
```

---

## 10. 再帰の計算量分析

### 10.1 マスター定理

```
分割統治法の計算量を求めるマスター定理:

再帰の形: T(n) = a * T(n/b) + O(n^d)
  a = 部分問題の数
  b = 問題サイズの縮小率
  d = 結合ステップの計算量の指数

3つのケース:
  Case 1: d < log_b(a) → T(n) = O(n^(log_b(a)))
  Case 2: d = log_b(a) → T(n) = O(n^d * log(n))
  Case 3: d > log_b(a) → T(n) = O(n^d)

例:
  マージソート: T(n) = 2T(n/2) + O(n)
    a=2, b=2, d=1 → log_2(2) = 1 = d → Case 2 → O(n log n)

  二分探索: T(n) = T(n/2) + O(1)
    a=1, b=2, d=0 → log_2(1) = 0 = d → Case 2 → O(log n)

  ストラッセン: T(n) = 7T(n/2) + O(n^2)
    a=7, b=2, d=2 → log_2(7) ≈ 2.807 > 2 → Case 1 → O(n^2.807)

  通常の行列乗算: T(n) = 8T(n/2) + O(n^2)
    a=8, b=2, d=2 → log_2(8) = 3 > 2 → Case 1 → O(n^3)
```

### 10.2 各再帰パターンの計算量

```
┌──────────────────┬────────────┬──────────────┬──────────────┐
│ パターン          │ 時間計算量  │ 空間計算量    │ 例            │
├──────────────────┼────────────┼──────────────┼──────────────┤
│ 線形再帰          │ O(n)       │ O(n)         │ 階乗、合計    │
│ 末尾再帰(TCO)     │ O(n)       │ O(1)         │ 階乗(末尾版)  │
│ 二分探索再帰      │ O(log n)   │ O(log n)     │ 二分探索      │
│ 分割統治          │ O(n log n) │ O(n)         │ マージソート  │
│ ツリー再帰(naive) │ O(2^n)     │ O(n)         │ fib(naive)    │
│ ツリー再帰(memo)  │ O(n)       │ O(n)         │ fib(memo)     │
│ バックトラッキング │ O(b^d)     │ O(d)         │ N-Queens      │
│ 全順列生成        │ O(n!)      │ O(n)         │ permutations  │
└──────────────────┴────────────┴──────────────┴──────────────┘
  b = 分岐数, d = 深さ
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
| バックトラッキング | 探索木を DFS で辿り、行き詰まったら戻る |
| トランポリン | TCO のない言語で末尾再帰を安全に実行 |
| CPS | 結果をコールバックに渡して末尾再帰化 |

再帰を効果的に使うための原則:

1. **必ずベースケースを定義する**: 無限再帰を防ぐ最重要ルール
2. **問題が確実に小さくなることを保証する**: 各再帰呼び出しで問題サイズが減少
3. **重複計算にはメモ化を適用する**: 指数的な計算量を線形に改善
4. **スタックオーバーフローに注意する**: 深い再帰ではループ or トランポリンを検討
5. **適切な手法を選択する**: ツリー構造には再帰、平坦なデータにはループ
6. **可読性と効率のバランスを取る**: 再帰が自然なら再帰、そうでなければループ

---

## 次に読むべきガイド
-> [[../05-concurrency/00-threads-and-processes.md]] -- 並行処理

---

## 参考文献
1. Abelson, H. & Sussman, G. "SICP." Ch.1.2, MIT Press, 1996.
2. Cormen, T. et al. "Introduction to Algorithms." Ch.4, MIT Press, 2022.
3. Sedgewick, R. & Wayne, K. "Algorithms." 4th Edition, Addison-Wesley, 2011.
4. Skiena, S. "The Algorithm Design Manual." 3rd Edition, Springer, 2020.
5. Bird, R. "Thinking Functionally with Haskell." Cambridge, 2014.
6. Okasaki, C. "Purely Functional Data Structures." Cambridge, 1998.
7. Graham, R. et al. "Concrete Mathematics." Addison-Wesley, 1994.
