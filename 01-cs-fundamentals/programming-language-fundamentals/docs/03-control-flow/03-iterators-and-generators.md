# イテレータとジェネレータ

> イテレータは「コレクションの要素を1つずつ取り出す」抽象化であり、遅延評価により無限シーケンスやメモリ効率の良い処理を可能にする。ジェネレータは「実行を中断・再開できる関数」として、イテレータを手軽に作成するための構文糖衣である。本章では、これらの概念を基礎から応用まで体系的に解説する。

---

## この章で学ぶこと

- [ ] イテレータパターンの設計意図と仕組みを理解する
- [ ] 各言語におけるイテレータプロトコルの違いを把握する
- [ ] ジェネレータの動作原理（コルーチンとの関係）を理解する
- [ ] 遅延評価と正格評価のトレードオフを判断できる
- [ ] イテレータアダプタを組み合わせて宣言的なデータ処理パイプラインを構築できる
- [ ] 非同期イテレータの適用場面を理解する
- [ ] 無限シーケンスを安全に扱うテクニックを習得する

---

## 目次

1. [イテレータパターンの本質](#1-イテレータパターンの本質)
2. [各言語のイテレータプロトコル](#2-各言語のイテレータプロトコル)
3. [イテレータアダプタと遅延変換チェーン](#3-イテレータアダプタと遅延変換チェーン)
4. [ジェネレータの動作原理](#4-ジェネレータの動作原理)
5. [遅延評価 vs 正格評価](#5-遅延評価-vs-正格評価)
6. [非同期イテレータとストリーム](#6-非同期イテレータとストリーム)
7. [実践パターン集](#7-実践パターン集)
8. [アンチパターンと落とし穴](#8-アンチパターンと落とし穴)
9. [パフォーマンス特性と最適化](#9-パフォーマンス特性と最適化)
10. [演習問題](#10-演習問題)
11. [FAQ（よくある質問）](#11-faqよくある質問)
12. [まとめ](#12-まとめ)
13. [参考文献](#13-参考文献)

---

## 1. イテレータパターンの本質

### 1.1 デザインパターンとしてのイテレータ

GoF（Gang of Four）のデザインパターンにおいて、イテレータパターンは**振る舞いパターン（Behavioral Pattern）**に分類される。その目的は「コレクションの内部表現を公開することなく、要素に順次アクセスする手段を提供すること」である。

```
+-------------------------------------------------------------+
|  イテレータパターンの位置づけ                                   |
+-------------------------------------------------------------+
|                                                             |
|  クライアントコード                                           |
|       |                                                     |
|       | next() / hasNext()                                  |
|       v                                                     |
|  +------------------+                                       |
|  | Iterator         |  <--- 統一インターフェース               |
|  | - next()         |                                       |
|  | - hasNext()      |                                       |
|  +------------------+                                       |
|       ^        ^        ^                                   |
|       |        |        |                                   |
|  +--------+ +--------+ +--------+                           |
|  | Array  | | Tree   | | Graph  |   <--- 異なる内部構造      |
|  |Iterator| |Iterator| |Iterator|                           |
|  +--------+ +--------+ +--------+                           |
|                                                             |
|  要点: クライアントは内部構造を知らずに走査できる              |
+-------------------------------------------------------------+
```

この抽象化によって得られる恩恵は4つある。

| 恩恵 | 説明 | 具体例 |
|------|------|--------|
| **カプセル化** | コレクションの内部構造を隠蔽する | 配列・リンクリスト・ツリーを同じインターフェースで走査 |
| **統一アクセス** | 異なるデータ構造に対して同じループ構文を使用できる | `for x in collection` がどのコレクションにも適用可能 |
| **遅延評価** | 要素を必要な時に1つずつ生成する | 100万行のファイルを1行ずつ処理 |
| **合成可能性** | イテレータ同士を組み合わせてパイプラインを構築できる | `filter -> map -> take -> collect` |

### 1.2 内部イテレータと外部イテレータ

イテレータには、制御の主体がどちら側にあるかによって2種類の設計がある。

```
+-----------------------------------------------+
|  外部イテレータ（Pull 型）                       |
+-----------------------------------------------+
|                                               |
|  クライアント:  "次をくれ" --> イテレータ        |
|  イテレータ:    <-- 要素を返す                   |
|                                               |
|  制御: クライアント側                            |
|  例:   Python の next(), Rust の .next()       |
|  利点: クライアントが走査タイミングを制御         |
|        2つのイテレータを交互に読むことも可能      |
+-----------------------------------------------+

+-----------------------------------------------+
|  内部イテレータ（Push 型）                       |
+-----------------------------------------------+
|                                               |
|  クライアント:  クロージャを渡す --> コレクション |
|  コレクション:  各要素にクロージャを適用          |
|                                               |
|  制御: コレクション側                            |
|  例:   Ruby の each, JS の forEach             |
|  利点: 実装がシンプル、並列化しやすい            |
+-----------------------------------------------+
```

**比較表: 内部イテレータ vs 外部イテレータ**

| 特性 | 外部イテレータ（Pull） | 内部イテレータ（Push） |
|------|----------------------|----------------------|
| 制御の主体 | クライアント | コレクション |
| 走査の中断 | 自由に可能 | break/例外で中断 |
| 複数イテレータの交互使用 | 容易 | 困難 |
| 実装の複雑さ | 状態管理が必要 | 比較的シンプル |
| 並列化 | 手動で実装 | コレクション側で最適化可能 |
| 代表例 | Python `__next__`, Rust `Iterator` | Ruby `each`, JS `forEach` |
| 遅延評価との相性 | 自然に遅延 | 基本的に正格 |

```python
# 外部イテレータの例（Pull型）
it = iter([1, 2, 3, 4, 5])
a = next(it)  # 1 --- クライアントが主導
b = next(it)  # 2 --- 好きなタイミングで取得
# 残りの 3, 4, 5 はまだ「取り出されていない」

# 内部イテレータの例（Push型）
[1, 2, 3, 4, 5].forEach(x => console.log(x))
# コレクション側がすべての要素を走査する
# クライアントは「各要素に何をするか」だけを指定
```

### 1.3 イテレータプロトコルの共通構造

すべての言語のイテレータプロトコルは、根本的に同じ構造を持つ。

```
+-----------------------------------------------------------+
|  イテレータプロトコルの共通構造                               |
+-----------------------------------------------------------+
|                                                           |
|  状態: current_position, underlying_collection            |
|                                                           |
|  操作:                                                     |
|  +-----------------------+                                |
|  | next()                |                                |
|  | ┌──────────────────┐  |                                |
|  | │ 要素がある?       │  |                                |
|  | │   Yes → (値, 継続)│  |  ← "まだ続きがある" を示す     |
|  | │   No  → 終了通知  │  |  ← "もう要素がない" を示す     |
|  | └──────────────────┘  |                                |
|  +-----------------------+                                |
|                                                           |
|  終了通知の実現方法:                                        |
|    Python:  StopIteration 例外                             |
|    Rust:    Option<T> の None                              |
|    Java:    hasNext() + next() の2メソッド方式             |
|    JS:      { value, done: true/false } オブジェクト       |
|    C++:     end() イテレータとの比較                        |
+-----------------------------------------------------------+
```

---

## 2. 各言語のイテレータプロトコル

### 2.1 Python: `__iter__` / `__next__` プロトコル

Python のイテレータプロトコルは、2つのダンダーメソッドで構成される。

- `__iter__()`: イテレータオブジェクト自身を返す
- `__next__()`: 次の要素を返すか、要素がなければ `StopIteration` を送出する

```python
class Countdown:
    """カウントダウンイテレータの実装例"""

    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# 使用例
for n in Countdown(5):
    print(n)  # 5, 4, 3, 2, 1

# for ループの内部動作を展開すると:
it = iter(Countdown(5))   # __iter__() を呼ぶ
while True:
    try:
        n = next(it)      # __next__() を呼ぶ
        print(n)
    except StopIteration:
        break             # 終了
```

Python ではイテラブル（`__iter__` を持つ）とイテレータ（`__iter__` + `__next__` を持つ）を区別する。リストはイテラブルだがイテレータではない。`iter()` を呼ぶことでイテレータを取得する。

```python
# イテラブルとイテレータの違い
lst = [1, 2, 3]
print(type(lst))         # <class 'list'> --- イテラブル
it = iter(lst)
print(type(it))          # <class 'list_iterator'> --- イテレータ

# イテラブルは何度でもイテレータを生成できる
for x in lst: pass  # 1回目
for x in lst: pass  # 2回目（問題なく動作）

# イテレータは使い切ると空になる
it = iter(lst)
list(it)  # [1, 2, 3]
list(it)  # [] --- 使い切った後は空
```

### 2.2 Rust: `Iterator` トレイト

Rust のイテレータは `Iterator` トレイトを実装することで定義される。終了は `Option<Self::Item>` の `None` で表現される。

```rust
struct Countdown {
    current: u32,
}

impl Countdown {
    fn new(start: u32) -> Self {
        Countdown { current: start }
    }
}

impl Iterator for Countdown {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.current == 0 {
            None
        } else {
            self.current -= 1;
            Some(self.current + 1)
        }
    }
}

fn main() {
    // for ループで使用
    for n in Countdown::new(5) {
        println!("{}", n);  // 5, 4, 3, 2, 1
    }

    // IntoIterator トレイト: for ループの糖衣構文
    // for item in collection { ... }
    // は以下と等価:
    // let mut iter = collection.into_iter();
    // while let Some(item) = iter.next() { ... }
}
```

Rust のイテレータシステムは3種類の所有権モデルを持つ。

```rust
let v = vec![1, 2, 3];

// 1. iter(): 不変参照のイテレータ (&T)
for val in v.iter() {
    // val は &i32 型
    println!("{}", val);
}
// v はまだ使える

// 2. iter_mut(): 可変参照のイテレータ (&mut T)
let mut v2 = vec![1, 2, 3];
for val in v2.iter_mut() {
    // val は &mut i32 型
    *val *= 2;
}
// v2 は [2, 4, 6] に変更された

// 3. into_iter(): 所有権を移動するイテレータ (T)
for val in v.into_iter() {
    // val は i32 型（所有権が移動）
    println!("{}", val);
}
// v はもう使えない（所有権が移動した）
```

### 2.3 JavaScript: `Symbol.iterator` プロトコル

JavaScript のイテレータプロトコルは `Symbol.iterator` メソッドと `{ value, done }` オブジェクトで構成される。

```javascript
class Countdown {
    constructor(start) {
        this.start = start;
    }

    [Symbol.iterator]() {
        let current = this.start;
        return {
            next() {
                if (current <= 0) {
                    return { value: undefined, done: true };
                }
                return { value: current--, done: false };
            }
        };
    }
}

// for-of ループで使用
for (const n of new Countdown(5)) {
    console.log(n);  // 5, 4, 3, 2, 1
}

// スプレッド構文でも使用可能
const arr = [...new Countdown(5)];  // [5, 4, 3, 2, 1]

// 分割代入でも使用可能
const [a, b, c] = new Countdown(5);  // a=5, b=4, c=3
```

### 2.4 Java: `Iterator<T>` / `Iterable<T>` インターフェース

```java
import java.util.Iterator;

class Countdown implements Iterable<Integer> {
    private final int start;

    Countdown(int start) {
        this.start = start;
    }

    @Override
    public Iterator<Integer> iterator() {
        return new Iterator<Integer>() {
            int current = start;

            @Override
            public boolean hasNext() {
                return current > 0;
            }

            @Override
            public Integer next() {
                return current--;
            }
        };
    }
}

// 拡張 for ループで使用
for (int n : new Countdown(5)) {
    System.out.println(n);  // 5, 4, 3, 2, 1
}
```

### 2.5 各言語のプロトコル比較

| 言語 | イテラブル | イテレータ | 終了の表現 | for ループ |
|------|----------|----------|-----------|-----------|
| **Python** | `__iter__()` | `__next__()` | `StopIteration` 例外 | `for x in iterable:` |
| **Rust** | `IntoIterator` | `Iterator::next()` | `Option::None` | `for x in iterable {}` |
| **JavaScript** | `[Symbol.iterator]()` | `next()` | `{ done: true }` | `for (x of iterable)` |
| **Java** | `Iterable<T>` | `Iterator<T>` | `hasNext() == false` | `for (T x : iterable)` |
| **C++** | `begin()` / `end()` | `operator++` / `operator*` | `iter == end()` | `for (auto x : container)` |
| **C#** | `IEnumerable<T>` | `IEnumerator<T>` | `MoveNext() == false` | `foreach (var x in collection)` |
| **Go** | なし（channelで代用） | なし | channel close | `for x := range ch` |

---

## 3. イテレータアダプタと遅延変換チェーン

### 3.1 アダプタパターンの概念

イテレータアダプタは「イテレータを受け取り、変換されたイテレータを返す」関数である。重要な点は、アダプタは**遅延評価**であり、チェーンを組み立てた時点では何も実行されないことである。最終的に要素を消費する操作（`collect`, `sum`, `for` ループなど）が呼ばれた時に初めて、パイプライン全体が要素単位で実行される。

```
+-------------------------------------------------------------+
|  イテレータアダプタチェーンの実行モデル                          |
+-------------------------------------------------------------+
|                                                             |
|  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                           |
|       |                                                     |
|       v                                                     |
|  filter(n % 2 == 0)  --- 偶数のみ通す                       |
|       |                                                     |
|       v                                                     |
|  map(n * n)           --- 二乗する                           |
|       |                                                     |
|       v                                                     |
|  take(3)              --- 最初の3個で停止                     |
|       |                                                     |
|       v                                                     |
|  collect()            --- 結果を収集                          |
|                                                             |
|  要素ごとの実行順序:                                          |
|  1 -> filter(奇数=棄却)                                      |
|  2 -> filter(偶数=通過) -> map(4) -> take(1個目) -> [4]      |
|  3 -> filter(奇数=棄却)                                      |
|  4 -> filter(偶数=通過) -> map(16) -> take(2個目) -> [4,16]  |
|  5 -> filter(奇数=棄却)                                      |
|  6 -> filter(偶数=通過) -> map(36) -> take(3個目) -> [4,16,36]|
|  7以降は処理されない（take(3) が満たされた）                   |
|                                                             |
|  => 結果: [4, 16, 36]                                       |
|  => 10要素中 6要素しか見ていない（短絡評価）                   |
+-------------------------------------------------------------+
```

### 3.2 Rust のイテレータアダプタ

Rust はイテレータアダプタが最も充実している言語の一つである。すべてのアダプタはゼロコスト抽象化として、コンパイル時にインライン展開される。

```rust
// 基本的なアダプタチェーン
let result: Vec<i32> = (1..=100)
    .filter(|n| n % 3 == 0)      // 3の倍数
    .map(|n| n * n)               // 二乗
    .take(5)                      // 最初の5個
    .collect();
// => [9, 36, 81, 144, 225]

// 重要: collect() を呼ぶまで何も実行されない（遅延評価）
// 各要素は filter -> map -> take のパイプラインを1つずつ通過する
```

**主要な変換アダプタ（遅延）:**

```rust
let v = vec![1, 2, 3, 4, 5];

// map: 各要素を変換
v.iter().map(|x| x * 2);                    // [2, 4, 6, 8, 10]

// filter: 条件を満たす要素のみ残す
v.iter().filter(|x| **x > 2);               // [3, 4, 5]

// filter_map: filter と map を同時に（None を除去）
v.iter().filter_map(|x| {
    if *x > 2 { Some(x * 10) } else { None }
}); // [30, 40, 50]

// take / skip: 先頭N個を取得 / スキップ
v.iter().take(3);                            // [1, 2, 3]
v.iter().skip(2);                            // [3, 4, 5]

// take_while / skip_while: 条件による取得/スキップ
v.iter().take_while(|x| **x < 4);           // [1, 2, 3]
v.iter().skip_while(|x| **x < 4);           // [4, 5]

// enumerate: インデックスを付与
v.iter().enumerate();                        // [(0,1), (1,2), (2,3), ...]

// zip: 2つのイテレータを結合
v.iter().zip(vec![10, 20, 30].iter());       // [(1,10), (2,20), (3,30)]

// chain: イテレータを連結
v.iter().chain(vec![6, 7].iter());           // [1,2,3,4,5,6,7]

// flat_map: 要素をイテレータに展開して平坦化
v.iter().flat_map(|x| vec![*x, *x * 10]);   // [1,10,2,20,3,30,4,40,5,50]

// flatten: ネストしたイテレータを平坦化
vec![vec![1,2], vec![3,4]].into_iter().flatten(); // [1,2,3,4]

// peekable: 次の要素を消費せずに覗く
let mut it = v.iter().peekable();
assert_eq!(it.peek(), Some(&&1));            // 覗くだけ
assert_eq!(it.next(), Some(&1));             // 消費

// scan: 状態を持つ変換（累積和など）
v.iter().scan(0, |acc, x| {
    *acc += x;
    Some(*acc)
}); // [1, 3, 6, 10, 15]

// inspect: デバッグ用（要素を変更せずに副作用を実行）
v.iter()
    .inspect(|x| println!("before filter: {}", x))
    .filter(|x| **x > 2)
    .inspect(|x| println!("after filter: {}", x))
    .collect::<Vec<_>>();
```

**主要な消費アダプタ（正格/終端操作）:**

```rust
let v = vec![1, 2, 3, 4, 5];

// collect: コレクションに変換
let vec: Vec<_> = v.iter().collect();
let set: HashSet<_> = v.iter().collect();
let s: String = vec!['a', 'b', 'c'].into_iter().collect();

// sum / product: 合計 / 積
let total: i32 = v.iter().sum();             // 15
let product: i32 = v.iter().product();       // 120

// count: 要素数
v.iter().count();                             // 5

// any / all: 条件判定
v.iter().any(|x| *x > 3);                   // true
v.iter().all(|x| *x > 0);                   // true

// find: 条件を満たす最初の要素
v.iter().find(|x| **x > 3);                 // Some(&4)

// position: 条件を満たす最初のインデックス
v.iter().position(|x| *x > 3);              // Some(3)

// min / max: 最小/最大
v.iter().min();                               // Some(&1)
v.iter().max();                               // Some(&5)

// fold: 畳み込み（reduce の一般化）
v.iter().fold(0, |acc, x| acc + x);          // 15

// reduce: 初期値なしの畳み込み
v.iter().copied().reduce(|a, b| a + b);      // Some(15)

// for_each: 各要素に副作用を実行
v.iter().for_each(|x| println!("{}", x));
```

### 3.3 Python のイテレータツール

Python は組み込み関数と `itertools` モジュールでイテレータ操作を提供する。

```python
import itertools

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 組み込み関数によるイテレータ操作
list(map(lambda x: x ** 2, data))              # [1, 4, 9, 16, 25, ...]
list(filter(lambda x: x % 2 == 0, data))       # [2, 4, 6, 8, 10]
list(zip(data, range(10, 20)))                  # [(1,10), (2,11), ...]
list(enumerate(data))                           # [(0,1), (1,2), ...]
list(reversed(data))                            # [10, 9, 8, ..., 1]
sum(data)                                       # 55
min(data)                                       # 1
max(data)                                       # 10
any(x > 5 for x in data)                       # True
all(x > 0 for x in data)                       # True

# itertools による高度な操作
list(itertools.chain([1,2], [3,4], [5,6]))      # [1,2,3,4,5,6]
list(itertools.islice(range(100), 5, 10))       # [5,6,7,8,9]
list(itertools.takewhile(lambda x: x < 5, data))  # [1,2,3,4]
list(itertools.dropwhile(lambda x: x < 5, data))  # [5,6,...,10]
list(itertools.accumulate(data))                 # [1,3,6,10,15,...]

# groupby: 連続する同じキーの要素をグループ化
sorted_data = sorted(data, key=lambda x: x % 3)
for key, group in itertools.groupby(sorted_data, key=lambda x: x % 3):
    print(f"key={key}: {list(group)}")

# product / permutations / combinations: 組み合わせ生成
list(itertools.product('AB', '12'))         # [('A','1'),('A','2'),('B','1'),('B','2')]
list(itertools.permutations('ABC', 2))      # [('A','B'),('A','C'),('B','A'),...]
list(itertools.combinations('ABCD', 2))     # [('A','B'),('A','C'),...]

# tee: イテレータを複製
it1, it2 = itertools.tee(iter(data), 2)
# it1 と it2 は独立して走査できる
```

---

## 4. ジェネレータの動作原理

### 4.1 ジェネレータとは何か

ジェネレータは「実行を中断（yield）し、後で再開できる関数」である。通常の関数が「呼び出し -> 実行 -> 値を返す」の一方通行であるのに対し、ジェネレータは「呼び出し -> yield で値を返して中断 -> 再開 -> yield -> ... -> 終了」というサイクルを繰り返す。

```
+-------------------------------------------------------------+
|  通常の関数 vs ジェネレータ関数                                |
+-------------------------------------------------------------+
|                                                             |
|  通常の関数:                                                 |
|  call() ──> [実行開始] ──> [処理] ──> return 値 ──> 終了     |
|         呼び出し元に                                         |
|         制御が戻る                                           |
|                                                             |
|  ジェネレータ関数:                                            |
|  call() ──> ジェネレータオブジェクトを生成（まだ実行しない）    |
|                                                             |
|  next() ──> [実行開始] ──> yield 値1 ──> 中断（状態を保持）  |
|         呼び出し元に                  ^                      |
|         制御が戻る                    |                      |
|                                       |                      |
|  next() ──> [再開] ──> [処理] ──> yield 値2 ──> 中断        |
|         呼び出し元に                  ^                      |
|         制御が戻る                    |                      |
|                                       |                      |
|  next() ──> [再開] ──> [処理] ──> return ──> StopIteration  |
|                                                             |
|  保持される状態: ローカル変数、実行位置（プログラムカウンタ）   |
+-------------------------------------------------------------+
```

### 4.2 Python のジェネレータ

```python
# 基本的なジェネレータ
def fibonacci():
    """フィボナッチ数列を無限に生成するジェネレータ"""
    a, b = 0, 1
    while True:
        yield a           # 値を返して中断
        a, b = b, a + b   # 再開時にここから続行

# 使用例: 無限シーケンスだが、必要な分だけ生成
fib = fibonacci()
for _ in range(10):
    print(next(fib))  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# ジェネレータ式（内包表記のジェネレータ版）
# リスト内包表記との比較
squares_list = [x**2 for x in range(1_000_000)]  # メモリに全要素を保持
squares_gen  = (x**2 for x in range(1_000_000))  # 1要素ずつ生成（省メモリ）

# ジェネレータ式はメモリフットプリントが極めて小さい
import sys
print(sys.getsizeof(squares_list))  # 約 8.5 MB
print(sys.getsizeof(squares_gen))   # 約 200 バイト（固定）
```

### 4.3 yield from: サブジェネレータ委譲

```python
# yield from を使わない場合
def chain_v1(*iterables):
    for it in iterables:
        for item in it:
            yield item

# yield from を使う場合（等価だがより効率的）
def chain_v2(*iterables):
    for it in iterables:
        yield from it

list(chain_v2([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]

# yield from は再帰的なジェネレータに特に有用
def flatten(nested):
    """ネストしたリストを平坦化する"""
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)  # 再帰的に委譲
        else:
            yield item

list(flatten([1, [2, [3, 4], 5], [6, 7]]))  # [1, 2, 3, 4, 5, 6, 7]
```

### 4.4 send() による双方向通信

ジェネレータは `send()` メソッドを使って、呼び出し側からジェネレータに値を送り込むこともできる。

```python
def accumulator():
    """送り込まれた値を累積するジェネレータ"""
    total = 0
    while True:
        value = yield total    # total を返し、send された値を value に受け取る
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)            # 0 （ジェネレータを最初の yield まで進める）
acc.send(10)         # 10（total = 0 + 10）
acc.send(20)         # 30（total = 10 + 20）
acc.send(5)          # 35（total = 30 + 5）

# 実用例: コルーチンとしてのジェネレータ（移動平均の計算）
def moving_average(window_size):
    """移動平均を計算するコルーチン"""
    window = []
    average = None
    while True:
        value = yield average
        window.append(value)
        if len(window) > window_size:
            window.pop(0)
        average = sum(window) / len(window)

ma = moving_average(3)
next(ma)         # None（初期化）
ma.send(10)      # 10.0
ma.send(20)      # 15.0
ma.send(30)      # 20.0
ma.send(40)      # 30.0（窓: [20, 30, 40]）
```

### 4.5 JavaScript のジェネレータ

```javascript
// function* でジェネレータ関数を定義
function* fibonacci() {
    let a = 0, b = 1;
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

const fib = fibonacci();
console.log(fib.next());  // { value: 0, done: false }
console.log(fib.next());  // { value: 1, done: false }
console.log(fib.next());  // { value: 1, done: false }

// ヘルパージェネレータ: take
function* take(iter, n) {
    let count = 0;
    for (const item of iter) {
        if (count++ >= n) break;
        yield item;
    }
}

// for-of で消費
for (const n of take(fibonacci(), 10)) {
    console.log(n);  // 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
}

// yield* でサブジェネレータ委譲
function* flatten(arr) {
    for (const item of arr) {
        if (Array.isArray(item)) {
            yield* flatten(item);  // 再帰的に委譲
        } else {
            yield item;
        }
    }
}

console.log([...flatten([1, [2, [3, 4], 5], [6, 7]])]);
// [1, 2, 3, 4, 5, 6, 7]

// send に相当する機能: next(value)
function* accumulator() {
    let total = 0;
    while (true) {
        const value = yield total;
        total += value;
    }
}

const acc = accumulator();
acc.next();        // { value: 0, done: false } -- 初期化
acc.next(10);      // { value: 10, done: false }
acc.next(20);      // { value: 30, done: false }
acc.next(5);       // { value: 35, done: false }
```

### 4.6 ジェネレータとコルーチンの関係

ジェネレータは「セミコルーチン（semi-coroutine）」とも呼ばれる。完全なコルーチンが任意の相手に制御を譲れるのに対し、ジェネレータは呼び出し元にのみ制御を戻す。

```
+-------------------------------------------------------------+
|  コルーチンの分類                                              |
+-------------------------------------------------------------+
|                                                             |
|  対称コルーチン（Symmetric Coroutine）                       |
|    - 任意のコルーチンに制御を譲れる                           |
|    - A -> B -> C -> A のように自由に遷移                     |
|    - 例: Lua のコルーチン、Go の goroutine（近い）           |
|                                                             |
|  非対称コルーチン（Asymmetric Coroutine / Semi-Coroutine）   |
|    - 呼び出し元にのみ制御を戻す                              |
|    - 呼び出し元 <-> ジェネレータ の往復のみ                   |
|    - 例: Python / JavaScript のジェネレータ                  |
|                                                             |
|  +------------------+    +------------------+               |
|  | 呼び出し元       |    | ジェネレータ      |               |
|  |                  |--->|  実行...          |               |
|  |                  |    |  yield 値         |               |
|  |  値を受け取る    |<---|                   |               |
|  |  ...             |    |  (中断中)         |               |
|  |  next()          |--->|  再開...          |               |
|  |                  |    |  yield 値         |               |
|  |  値を受け取る    |<---|                   |               |
|  +------------------+    +------------------+               |
+-------------------------------------------------------------+
```

Python の `async/await` 構文は、実はジェネレータの上に構築されている。歴史的に、Python 3.4 以前は `@asyncio.coroutine` デコレータと `yield from` でコルーチンを実現していた。Python 3.5 で `async def` / `await` が導入され、構文的に分離されたが、内部的にはジェネレータと同じ中断・再開メカニズムを使用している。

---

## 5. 遅延評価 vs 正格評価

### 5.1 評価戦略の基本概念

プログラミング言語における式の評価タイミングには、大きく分けて2つの戦略がある。

```
+-------------------------------------------------------------+
|  正格評価（Eager / Strict Evaluation）                        |
+-------------------------------------------------------------+
|                                                             |
|  式が定義された時点で即座に評価される                          |
|                                                             |
|  例: result = [x*2 for x in range(1000000)]                 |
|                                                             |
|  1. range(1000000) で 0~999999 を生成                        |
|  2. 各要素に x*2 を適用                                      |
|  3. 結果の 100万要素のリストをメモリに保持                    |
|                                                             |
|  メモリ使用量: O(N)                                          |
|  計算コスト: 全要素を前もって処理                              |
|  利点: 予測可能、デバッグしやすい                              |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|  遅延評価（Lazy Evaluation）                                  |
+-------------------------------------------------------------+
|                                                             |
|  値が実際に必要になるまで評価を遅延させる                      |
|                                                             |
|  例: result = (x*2 for x in range(1000000))                 |
|                                                             |
|  1. ジェネレータオブジェクトを生成（まだ何も計算しない）       |
|  2. next() が呼ばれるたびに 1要素ずつ計算                     |
|  3. 使わなかった要素は計算もメモリ確保もされない               |
|                                                             |
|  メモリ使用量: O(1)                                          |
|  計算コスト: 必要な分だけ処理                                  |
|  利点: 無限データ構造、省メモリ、不要な計算の回避               |
+-------------------------------------------------------------+
```

### 5.2 各言語の評価戦略

| 言語 | デフォルト評価 | 遅延評価の手段 | 正格化の手段 |
|------|-------------|--------------|------------|
| **Haskell** | 遅延 | デフォルト | `seq`, `deepseq`, `BangPatterns` |
| **Rust** | 正格 | `Iterator` チェーン | `collect()`, `sum()`, `for` |
| **Python** | 正格 | ジェネレータ, `itertools` | `list()`, `tuple()` |
| **JavaScript** | 正格 | ジェネレータ | `Array.from()`, スプレッド `[...]` |
| **Scala** | 正格 | `LazyList`, `View` | `.toList`, `.toVector` |
| **C#** | 正格 | `IEnumerable<T>` (LINQ) | `.ToList()`, `.ToArray()` |
| **Kotlin** | 正格 | `Sequence<T>` | `.toList()` |
| **Java** | 正格 | `Stream<T>` | `.collect()`, `.toList()` |

### 5.3 遅延評価の利点と注意点

**利点1: 無限シーケンスの表現**

```python
def natural_numbers():
    """自然数を無限に生成"""
    n = 1
    while True:
        yield n
        n += 1

def primes():
    """素数を無限に生成（エラトステネスの篩の変形）"""
    yield 2
    composites = {}
    n = 3
    while True:
        if n not in composites:
            yield n
            composites[n * n] = [n]
        else:
            for prime in composites[n]:
                composites.setdefault(prime + n, []).append(prime)
            del composites[n]
        n += 2

# 最初の20個の素数を取得
from itertools import islice
print(list(islice(primes(), 20)))
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
```

**利点2: パイプライン処理の効率化**

```rust
// 遅延評価により、take(5) の条件が満たされた時点で処理が停止する
// 10万要素すべてを処理する必要がない
let result: Vec<i32> = (1..=100_000)
    .filter(|n| is_prime(*n))    // 素数のみ
    .map(|n| n * n)              // 二乗
    .take(5)                     // 最初の5個で停止！
    .collect();
// is_prime は最初の5つの素数が見つかるまでしか呼ばれない
// => [4, 9, 25, 49, 121]
```

**注意点1: 副作用との相性**

```python
# 危険: 遅延評価と副作用の組み合わせ
def log_and_double(x):
    print(f"Processing: {x}")  # 副作用
    return x * 2

gen = (log_and_double(x) for x in range(5))
# この時点では何も出力されない！

# ジェネレータを消費して初めて副作用が発生する
result = list(gen)
# Processing: 0
# Processing: 1
# Processing: 2
# Processing: 3
# Processing: 4
```

**注意点2: 複数回の走査**

```python
gen = (x ** 2 for x in range(5))

# 1回目の走査
print(sum(gen))   # 30（0 + 1 + 4 + 9 + 16）

# 2回目の走査（空！ジェネレータは使い切りである）
print(sum(gen))   # 0 --- よくあるバグ
```

### 5.4 Haskell の遅延評価

Haskell はデフォルトで遅延評価を採用する唯一の主要言語である。

```haskell
-- 無限リストが自然に書ける
naturals = [1..]                    -- [1, 2, 3, 4, ...]
evens    = [2, 4..]                 -- [2, 4, 6, 8, ...]
fibs     = 0 : 1 : zipWith (+) fibs (tail fibs)  -- フィボナッチ数列

-- 必要な分だけ取得
take 10 fibs    -- [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
take 5 naturals -- [1, 2, 3, 4, 5]

-- 無限リストを使ったエレガントな表現
-- 「最初の10個の素数」
take 10 [n | n <- [2..], isPrime n]

-- 遅延評価があるからこそ可能な定義
-- ones = 1 : ones  -- [1, 1, 1, 1, ...] 無限に1が続くリスト
```

---

## 6. 非同期イテレータとストリーム

### 6.1 非同期イテレータの動機

通常のイテレータは同期的に値を生成する。しかし、以下のようなケースでは非同期に値が到着する。

- ネットワークからのデータストリーム
- ファイルの行ごとの読み取り
- データベースのカーソルからの行取得
- WebSocket のメッセージ受信
- ページネーションされた API のレスポンス

これらを扱うために、非同期イテレータ（Async Iterator）が必要になる。

```
+-------------------------------------------------------------+
|  同期イテレータ vs 非同期イテレータ                            |
+-------------------------------------------------------------+
|                                                             |
|  同期イテレータ:                                              |
|  next() --> 値がすぐに返る（ブロッキング）                    |
|  next() --> 値がすぐに返る                                    |
|  next() --> 終了                                              |
|                                                             |
|  非同期イテレータ:                                            |
|  next() --> Future/Promise を返す                            |
|         --> await で値の到着を待つ（ノンブロッキング）          |
|  next() --> Future/Promise を返す                            |
|         --> await で値の到着を待つ                            |
|  next() --> 終了                                              |
|                                                             |
|  待っている間、他のタスクを実行できる                          |
+-------------------------------------------------------------+
```

### 6.2 JavaScript の非同期イテレータ

```javascript
// async function* で非同期ジェネレータ
async function* fetchPages(baseUrl) {
    let page = 1;
    while (true) {
        const response = await fetch(`${baseUrl}?page=${page}`);
        const data = await response.json();
        if (data.items.length === 0) break;
        yield data.items;       // 各ページのデータを yield
        page++;
    }
}

// for await ... of で消費
async function processAllUsers(url) {
    for await (const users of fetchPages(url)) {
        for (const user of users) {
            console.log(user.name);
        }
    }
}

// Symbol.asyncIterator プロトコル
class EventStream {
    constructor(eventSource) {
        this.source = eventSource;
    }

    [Symbol.asyncIterator]() {
        const source = this.source;
        return {
            next() {
                return new Promise((resolve) => {
                    source.once('data', (data) => {
                        resolve({ value: data, done: false });
                    });
                    source.once('end', () => {
                        resolve({ value: undefined, done: true });
                    });
                });
            }
        };
    }
}

// 非同期ジェネレータのユーティリティ
async function* asyncMap(asyncIter, fn) {
    for await (const item of asyncIter) {
        yield fn(item);
    }
}

async function* asyncFilter(asyncIter, predicate) {
    for await (const item of asyncIter) {
        if (predicate(item)) {
            yield item;
        }
    }
}

async function* asyncTake(asyncIter, n) {
    let count = 0;
    for await (const item of asyncIter) {
        if (count++ >= n) break;
        yield item;
    }
}

// 組み合わせて使用
const activeUsers = asyncFilter(
    asyncMap(
        fetchPages("/api/users"),
        page => page.filter(u => u.active)
    ),
    page => page.length > 0
);
```

### 6.3 Python の非同期ジェネレータ

```python
import aiohttp
import asyncio

# async def + yield = 非同期ジェネレータ
async def fetch_pages(base_url):
    """ページネーションされたAPIからデータを非同期に取得"""
    page = 1
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(f"{base_url}?page={page}") as resp:
                data = await resp.json()
                if not data["items"]:
                    break
                yield data["items"]
                page += 1

# async for で消費
async def process_all_users():
    async for users in fetch_pages("https://api.example.com/users"):
        for user in users:
            print(user["name"])

asyncio.run(process_all_users())

# 非同期内包表記
async def get_active_users():
    return [
        user
        async for page in fetch_pages("/api/users")
        for user in page
        if user.get("active")
    ]

# 非同期ジェネレータの aclose() による明示的クリーンアップ
async def resource_stream():
    resource = await acquire_resource()
    try:
        while True:
            data = await resource.read()
            if data is None:
                break
            yield data
    finally:
        await resource.release()  # クリーンアップが保証される

# aclose() で途中終了してもクリーンアップが実行される
stream = resource_stream()
first_item = await stream.__anext__()
await stream.aclose()  # finally ブロックが実行される
```

### 6.4 Rust の Stream（非同期イテレータ）

```rust
use tokio_stream::{self, StreamExt};
use tokio::time::{self, Duration};

// Stream は非同期版 Iterator
// poll_next() が Future を返す

// tokio_stream を使った例
#[tokio::main]
async fn main() {
    // 基本的な Stream の作成と消費
    let mut stream = tokio_stream::iter(vec![1, 2, 3, 4, 5])
        .filter(|n| *n % 2 == 0)
        .map(|n| n * 10);

    while let Some(value) = stream.next().await {
        println!("{}", value);  // 20, 40
    }

    // インターバルストリーム
    let mut interval = tokio_stream::wrappers::IntervalStream::new(
        time::interval(Duration::from_secs(1))
    );

    // 最初の5回だけ処理
    let mut count = 0;
    while let Some(_tick) = interval.next().await {
        count += 1;
        println!("Tick {}", count);
        if count >= 5 { break; }
    }
}

// async-stream クレートを使ったストリーム生成
// （Rust にはまだネイティブの async ジェネレータ構文がない）
use async_stream::stream;

fn countdown(from: u32) -> impl tokio_stream::Stream<Item = u32> {
    stream! {
        for i in (1..=from).rev() {
            tokio::time::sleep(Duration::from_millis(100)).await;
            yield i;
        }
    }
}
```

### 6.5 非同期イテレータの比較

| 特性 | JavaScript | Python | Rust |
|------|-----------|--------|------|
| 構文 | `async function*` | `async def` + `yield` | `async-stream` クレート |
| 消費 | `for await...of` | `async for` | `while let Some(x) = stream.next().await` |
| プロトコル | `Symbol.asyncIterator` | `__aiter__` / `__anext__` | `Stream` トレイト |
| 委譲 | `yield*` | `async for` + `yield` | なし（手動） |
| ネイティブ対応 | ES2018 | Python 3.6 | 未安定（nightly） |
| エラー処理 | `try-catch` | `try-except` | `Result<Option<T>>` |

---

## 7. 実践パターン集

### 7.1 パイプラインパターン

データ処理のパイプラインをイテレータで構築する。Unix パイプの概念をプログラム内で実現する。

```python
import csv
from typing import Iterator, Dict, Any

def read_csv(filename: str) -> Iterator[Dict[str, str]]:
    """CSVファイルを1行ずつ読み取るジェネレータ"""
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
    # ファイルは自動的にクローズされる

def parse_numbers(rows: Iterator[Dict]) -> Iterator[Dict]:
    """数値フィールドを変換"""
    for row in rows:
        row["age"] = int(row["age"])
        row["salary"] = float(row["salary"])
        yield row

def filter_by_age(rows: Iterator[Dict], min_age: int) -> Iterator[Dict]:
    """年齢でフィルタ"""
    for row in rows:
        if row["age"] >= min_age:
            yield row

def top_n(rows: Iterator[Dict], n: int, key: str) -> list:
    """上位N件を取得（ヒープを使用してメモリ効率的に）"""
    import heapq
    return heapq.nlargest(n, rows, key=lambda r: r[key])

# パイプラインの構築と実行
# 100万行のCSVでもメモリ使用量は一定
pipeline = read_csv("employees.csv")
pipeline = parse_numbers(pipeline)
pipeline = filter_by_age(pipeline, 30)
top_earners = top_n(pipeline, 10, "salary")

for emp in top_earners:
    print(f"{emp['name']}: ${emp['salary']:,.0f}")
```

### 7.2 ウィンドウ処理パターン

時系列データや連続データに対して、固定幅のウィンドウをスライドさせながら処理する。

```python
from collections import deque
from typing import Iterator, TypeVar, Tuple
import itertools

T = TypeVar('T')

def sliding_window(iterable, size: int) -> Iterator[Tuple]:
    """スライディングウィンドウ"""
    it = iter(iterable)
    window = deque(itertools.islice(it, size), maxlen=size)
    if len(window) == size:
        yield tuple(window)
    for item in it:
        window.append(item)
        yield tuple(window)

# 使用例: 移動平均
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for window in sliding_window(data, 3):
    avg = sum(window) / len(window)
    print(f"window={window}, avg={avg:.1f}")
# window=(10, 20, 30), avg=20.0
# window=(20, 30, 40), avg=30.0
# window=(30, 40, 50), avg=40.0
# ...

def chunked(iterable, size: int) -> Iterator[list]:
    """イテラブルを固定サイズのチャンクに分割"""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

# 使用例: バッチ処理
for batch in chunked(range(100), 10):
    process_batch(batch)  # 10件ずつ処理
```

### 7.3 ツリー走査パターン

再帰的なデータ構造をイテレータ/ジェネレータで平坦に走査する。

```python
class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

def dfs_preorder(node: TreeNode) -> Iterator:
    """深さ優先探索（前順）をジェネレータで実装"""
    yield node.value
    for child in node.children:
        yield from dfs_preorder(child)

def dfs_postorder(node: TreeNode) -> Iterator:
    """深さ優先探索（後順）"""
    for child in node.children:
        yield from dfs_postorder(child)
    yield node.value

def bfs(node: TreeNode) -> Iterator:
    """幅優先探索をジェネレータで実装"""
    from collections import deque
    queue = deque([node])
    while queue:
        current = queue.popleft()
        yield current.value
        queue.extend(current.children)

# ツリー構築
tree = TreeNode(1, [
    TreeNode(2, [TreeNode(4), TreeNode(5)]),
    TreeNode(3, [TreeNode(6), TreeNode(7)])
])

print(list(dfs_preorder(tree)))   # [1, 2, 4, 5, 3, 6, 7]
print(list(dfs_postorder(tree)))  # [4, 5, 2, 6, 7, 3, 1]
print(list(bfs(tree)))            # [1, 2, 3, 4, 5, 6, 7]
```

### 7.4 状態機械パターン

ジェネレータを使って状態機械（State Machine）をシンプルに表現する。

```python
def lexer(text: str):
    """簡易的なトークナイザを状態機械で実装"""
    i = 0
    while i < len(text):
        # 空白をスキップ
        if text[i].isspace():
            i += 1
            continue

        # 数値トークン
        if text[i].isdigit():
            start = i
            while i < len(text) and text[i].isdigit():
                i += 1
            yield ("NUMBER", text[start:i])
            continue

        # 識別子トークン
        if text[i].isalpha():
            start = i
            while i < len(text) and text[i].isalnum():
                i += 1
            yield ("IDENT", text[start:i])
            continue

        # 演算子トークン
        if text[i] in "+-*/=<>":
            yield ("OP", text[i])
            i += 1
            continue

        raise ValueError(f"Unexpected character: {text[i]}")

# 使用例
for token_type, value in lexer("x = 42 + y * 3"):
    print(f"{token_type}: {value}")
# IDENT: x
# OP: =
# NUMBER: 42
# OP: +
# IDENT: y
# OP: *
# NUMBER: 3
```

### 7.5 リソース管理パターン

ジェネレータとコンテキストマネージャを組み合わせて、リソースの確実な解放を保証する。

```python
from contextlib import contextmanager

@contextmanager
def managed_cursor(connection):
    """データベースカーソルのライフサイクルを管理"""
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()

def query_rows(connection, sql, params=None):
    """大量の行を1行ずつ取得するジェネレータ"""
    with managed_cursor(connection) as cursor:
        cursor.execute(sql, params or ())
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row
    # with ブロックを抜けると cursor.close() が呼ばれる

# 100万行のクエリ結果をメモリ効率的に処理
for row in query_rows(conn, "SELECT * FROM large_table"):
    process_row(row)
# イテレーション途中で break しても、カーソルは確実にクローズされる
```

---

## 8. アンチパターンと落とし穴

### 8.1 アンチパターン1: ジェネレータの二重消費

```python
# =========================================
# ANTI-PATTERN: ジェネレータの二重消費
# =========================================

def get_numbers():
    yield from range(10)

numbers = get_numbers()

# 1回目: 正しく動作
total = sum(numbers)          # 45

# 2回目: 空のジェネレータ！ 値は 0 になる
average = total / sum(numbers) if sum(numbers) > 0 else 0
# ZeroDivisionError or 想定外の結果

# -----------------------------------------
# 修正方法1: リストに変換して再利用
# -----------------------------------------
numbers = list(get_numbers())  # メモリに全要素を保持
total = sum(numbers)           # 45
average = total / len(numbers) # 4.5

# -----------------------------------------
# 修正方法2: itertools.tee で複製
# -----------------------------------------
import itertools
nums1, nums2 = itertools.tee(get_numbers(), 2)
total = sum(nums1)
count = sum(1 for _ in nums2)
average = total / count  # 4.5
# 注意: tee は内部にバッファを持つため、大きなイテレータには不向き

# -----------------------------------------
# 修正方法3: イテラブルクラスを定義（推奨）
# -----------------------------------------
class NumberRange:
    """何度でもイテレートできるイテラブル"""
    def __iter__(self):
        yield from range(10)

numbers = NumberRange()
total = sum(numbers)           # 45（1回目）
average = total / sum(1 for _ in numbers)  # 4.5（2回目も動作）
```

### 8.2 アンチパターン2: 遅延評価とクロージャ変数の罠

```python
# =========================================
# ANTI-PATTERN: ループ変数のクロージャキャプチャ
# =========================================

# 期待: [0, 1, 4, 9, 16] を生成するジェネレータのリスト
generators = [lambda: i ** 2 for i in range(5)]

# 全てが 16 を返す！（i は最終値 4 を参照している）
results = [g() for g in generators]
print(results)  # [16, 16, 16, 16, 16]

# -----------------------------------------
# 修正方法1: デフォルト引数でキャプチャ
# -----------------------------------------
generators = [lambda i=i: i ** 2 for i in range(5)]
results = [g() for g in generators]
print(results)  # [0, 1, 4, 9, 16]

# -----------------------------------------
# 修正方法2: ジェネレータ式を使う
# -----------------------------------------
gen = (i ** 2 for i in range(5))
print(list(gen))  # [0, 1, 4, 9, 16]
```

### 8.3 アンチパターン3: 不要な中間リストの生成

```python
# =========================================
# ANTI-PATTERN: 不要な中間リストの生成
# =========================================

# 悪い例: 各ステップでリストを全生成
data = list(range(1_000_000))
filtered = [x for x in data if x % 2 == 0]        # 50万要素のリスト
mapped   = [x ** 2 for x in filtered]              # 50万要素のリスト
result   = [x for x in mapped if x > 1000]         # さらにリスト
final    = sum(result)
# メモリ: 元データ + filtered + mapped + result = 約4倍

# -----------------------------------------
# 修正: ジェネレータ式でパイプライン化
# -----------------------------------------
data = range(1_000_000)                             # range は遅延
filtered = (x for x in data if x % 2 == 0)         # ジェネレータ
mapped   = (x ** 2 for x in filtered)               # ジェネレータ
result   = (x for x in mapped if x > 1000)          # ジェネレータ
final    = sum(result)                               # ここで初めて計算
# メモリ: O(1) --- 中間リストなし
```

### 8.4 アンチパターン4: ジェネレータ内での例外処理の見落とし

```python
# =========================================
# ANTI-PATTERN: ジェネレータの finally が実行されない場合
# =========================================

def file_lines(path):
    """ファイルを1行ずつ読むジェネレータ"""
    f = open(path)
    try:
        for line in f:
            yield line.strip()
    finally:
        f.close()
        print("File closed")

# 問題: ジェネレータが途中で放棄されると...
gen = file_lines("data.txt")
first_line = next(gen)
# gen が GC されるまで finally は実行されない
# CPython では参照カウントにより即座に GC されるが、
# PyPy や他の実装では遅延する可能性がある

# -----------------------------------------
# 修正: close() を明示的に呼ぶ、またはコンテキストマネージャを使用
# -----------------------------------------

# 方法1: close() を明示的に呼ぶ
gen = file_lines("data.txt")
first_line = next(gen)
gen.close()  # finally ブロックが実行される

# 方法2: contextlib.closing を使用
from contextlib import closing
with closing(file_lines("data.txt")) as lines:
    first_line = next(lines)
# with ブロックを抜けると close() が自動的に呼ばれる
```

