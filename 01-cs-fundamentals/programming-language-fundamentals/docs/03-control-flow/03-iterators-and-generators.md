# イテレータとジェネレータ

> イテレータは「コレクションの要素を1つずつ取り出す」抽象化。遅延評価により、無限シーケンスやメモリ効率の良い処理を可能にする。

## この章で学ぶこと

- [ ] イテレータパターンの仕組みを理解する
- [ ] ジェネレータの動作原理を理解する
- [ ] 遅延評価の利点を活用できる

---

## 1. イテレータパターン

```
イテレータ = 「次の要素を取得する」インターフェース

  共通のプロトコル:
    next() → 次の要素 or 終了

  利点:
    - コレクションの内部構造を隠蔽
    - 統一的な方法で走査
    - 遅延評価（必要な時にだけ計算）
    - メモリ効率（全要素をメモリに保持しない）
```

### 各言語のイテレータ

```python
# Python: __iter__ / __next__ プロトコル
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for n in Countdown(5):
    print(n)  # 5, 4, 3, 2, 1
```

```rust
// Rust: Iterator トレイト
struct Countdown {
    current: u32,
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

for n in Countdown { current: 5 } {
    println!("{}", n);  // 5, 4, 3, 2, 1
}
```

```javascript
// JavaScript: Symbol.iterator プロトコル
class Countdown {
    constructor(start) { this.start = start; }

    [Symbol.iterator]() {
        let current = this.start;
        return {
            next() {
                if (current <= 0) return { done: true };
                return { value: current--, done: false };
            }
        };
    }
}

for (const n of new Countdown(5)) {
    console.log(n);  // 5, 4, 3, 2, 1
}
```

---

## 2. イテレータアダプタ（遅延変換）

```rust
// Rust: イテレータチェーン（遅延評価）
let result: Vec<i32> = (1..=100)
    .filter(|n| n % 3 == 0)      // 3の倍数
    .map(|n| n * n)               // 二乗
    .take(5)                      // 最初の5個
    .collect();
// → [9, 36, 81, 144, 225]

// 重要: collect() を呼ぶまで何も実行されない（遅延評価）
// 各要素は filter → map → take のパイプラインを1つずつ通過

// 主要なアダプタ
let iter = vec![1, 2, 3, 4, 5].into_iter();

iter.map(|x| x * 2)           // 変換
iter.filter(|x| x > &2)       // フィルタ
iter.take(3)                   // 最初のN個
iter.skip(2)                   // 最初のN個をスキップ
iter.enumerate()               // (index, value) のペア
iter.zip(other)                // 2つのイテレータを結合
iter.chain(other)              // 連結
iter.flat_map(|x| vec![x, x]) // 平坦化マップ
iter.peekable()                // 次の要素を消費せずに覗く
iter.scan(0, |acc, x| { *acc += x; Some(*acc) })  // 累積

// 消費アダプタ（結果を生成）
iter.collect::<Vec<_>>()       // コレクションに変換
iter.sum::<i32>()              // 合計
iter.count()                   // 要素数
iter.any(|x| x > 3)           // いずれかが条件を満たす
iter.all(|x| x > 0)           // 全てが条件を満たす
iter.find(|x| x > &3)         // 条件を満たす最初の要素
iter.min() / iter.max()        // 最小/最大
iter.fold(0, |acc, x| acc + x) // 畳み込み
```

---

## 3. ジェネレータ（Generator）

```
ジェネレータ = 「実行を中断・再開できる関数」

  通常の関数:  呼び出し → 最後まで実行 → 返す
  ジェネレータ: 呼び出し → yield で中断 → 再開 → yield → ...

  利点:
    - 無限シーケンスの表現
    - メモリ効率（全要素を一度に生成しない）
    - 複雑な状態機械をシンプルに記述
```

```python
# Python: yield でジェネレータ
def fibonacci():
    a, b = 0, 1
    while True:
        yield a           # 値を返して中断
        a, b = b, a + b   # 再開時にここから続行

# 無限シーケンスだが、必要な分だけ生成
fib = fibonacci()
for _ in range(10):
    print(next(fib))  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# ジェネレータ式（内包表記のジェネレータ版）
squares = (x**2 for x in range(1000000))  # メモリを消費しない
first_ten = [next(squares) for _ in range(10)]

# yield from（サブジェネレータ委譲）
def chain(*iterables):
    for it in iterables:
        yield from it

list(chain([1,2], [3,4], [5,6]))  # [1, 2, 3, 4, 5, 6]
```

```javascript
// JavaScript: function* でジェネレータ
function* fibonacci() {
    let a = 0, b = 1;
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

const fib = fibonacci();
console.log(fib.next().value);  // 0
console.log(fib.next().value);  // 1
console.log(fib.next().value);  // 1

// for-of で消費
for (const n of take(fibonacci(), 10)) {
    console.log(n);
}

function* take(iter, n) {
    let count = 0;
    for (const item of iter) {
        if (count++ >= n) break;
        yield item;
    }
}

// yield* でサブジェネレータ委譲
function* flatten(arr) {
    for (const item of arr) {
        if (Array.isArray(item)) {
            yield* flatten(item);
        } else {
            yield item;
        }
    }
}
```

---

## 4. 非同期イテレータ

```javascript
// JavaScript: async イテレータ
async function* fetchPages(url) {
    let page = 1;
    while (true) {
        const response = await fetch(`${url}?page=${page}`);
        const data = await response.json();
        if (data.length === 0) break;
        yield data;
        page++;
    }
}

// for await ... of で消費
for await (const page of fetchPages("/api/users")) {
    for (const user of page) {
        console.log(user.name);
    }
}
```

```python
# Python: async ジェネレータ
async def fetch_pages(url):
    page = 1
    while True:
        data = await fetch(f"{url}?page={page}")
        if not data:
            break
        yield data
        page += 1

async for page in fetch_pages("/api/users"):
    for user in page:
        print(user["name"])
```

```rust
// Rust: Stream（非同期イテレータ）
use tokio_stream::StreamExt;

let mut stream = tokio_stream::iter(vec![1, 2, 3, 4, 5])
    .filter(|n| n % 2 == 0)
    .map(|n| n * 10);

while let Some(value) = stream.next().await {
    println!("{}", value);  // 20, 40
}
```

---

## 5. 遅延評価 vs 正格評価

```
正格評価（Eager）:
  [1,2,3,4,5].map(x => x*2).filter(x => x>4)
  → 中間配列 [2,4,6,8,10] を生成 → [6,8,10] をフィルタ
  メモリ: 元の配列 + 中間配列 + 結果配列

遅延評価（Lazy）:
  iter.map(x => x*2).filter(x => x>4)
  → 要素を1つずつ処理（中間配列なし）
  → 1→2(×) → 2→4(×) → 3→6(✓) → 4→8(✓) → 5→10(✓)
  メモリ: 定数

Haskell: デフォルトで遅延評価
  take 5 [1..]  → [1,2,3,4,5]（無限リストの先頭5個）

Rust: イテレータは遅延、collect()で正格化
Python: ジェネレータは遅延、list()で正格化
JS: Array メソッドは正格、ジェネレータは遅延
```

---

## まとめ

| 概念 | 説明 | 代表言語 |
|------|------|---------|
| イテレータ | 要素を1つずつ取得 | 全言語 |
| アダプタ | 遅延変換チェーン | Rust, Python, JS |
| ジェネレータ | yield で中断・再開 | Python, JS |
| 非同期イテレータ | await + yield | Python, JS, Rust |
| 遅延評価 | 必要な時だけ計算 | Haskell, Rust iter |

---

## 次に読むべきガイド
→ [[../04-functions/00-first-class-functions.md]] — 第一級関数

---

## 参考文献
1. "Rust by Example: Iterators." doc.rust-lang.org.
2. "PEP 255: Simple Generators." python.org, 2001.
