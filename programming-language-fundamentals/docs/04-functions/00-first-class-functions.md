# 第一級関数（First-Class Functions）

> 関数が「値」として扱える言語では、関数を変数に代入し、引数として渡し、戻り値として返すことができる。これがモダンプログラミングの基盤。

## この章で学ぶこと

- [ ] 第一級関数の概念と意義を理解する
- [ ] 関数を値として操作する方法を習得する
- [ ] コールバックパターンを理解する

---

## 1. 第一級関数とは

```
第一級（First-Class）= 他の値と同等に扱える

  第一級関数が可能にすること:
    1. 変数に代入できる
    2. 関数の引数として渡せる
    3. 関数の戻り値として返せる
    4. データ構造に格納できる
```

```javascript
// JavaScript: 関数は第一級オブジェクト

// 1. 変数に代入
const greet = function(name) { return `Hello, ${name}!`; };
const greet2 = (name) => `Hello, ${name}!`;  // アロー関数

// 2. 引数として渡す
function applyTwice(fn, value) {
    return fn(fn(value));
}
applyTwice(x => x * 2, 3);  // → 12

// 3. 戻り値として返す
function multiplier(factor) {
    return (x) => x * factor;
}
const double = multiplier(2);
const triple = multiplier(3);
double(5);  // → 10
triple(5);  // → 15

// 4. データ構造に格納
const operations = {
    add: (a, b) => a + b,
    sub: (a, b) => a - b,
    mul: (a, b) => a * b,
};
operations.add(3, 4);  // → 7
```

```python
# Python: 関数は第一級オブジェクト
def square(x):
    return x ** 2

# 関数を変数に代入
f = square
f(5)  # → 25

# 関数のリスト
transforms = [str.upper, str.lower, str.title]
for t in transforms:
    print(t("hello world"))
# → HELLO WORLD / hello world / Hello World

# 関数をディクショナリに格納（ディスパッチテーブル）
handlers = {
    "greet": lambda name: f"Hello, {name}!",
    "farewell": lambda name: f"Goodbye, {name}!",
}
handlers["greet"]("Gaku")  # → "Hello, Gaku!"
```

```rust
// Rust: 関数ポインタとクロージャ
fn square(x: i32) -> i32 { x * x }

// 関数ポインタ
let f: fn(i32) -> i32 = square;
f(5);  // → 25

// クロージャ
let double = |x: i32| x * 2;
let result = double(5);  // → 10

// 高階関数の引数として
let numbers = vec![1, 2, 3, 4, 5];
let squared: Vec<i32> = numbers.iter().map(|x| x * x).collect();
```

---

## 2. コールバックパターン

```javascript
// JavaScript: コールバック（非同期処理の基本）

// イベントハンドラ
button.addEventListener("click", (event) => {
    console.log("Clicked!", event.target);
});

// Array メソッド
const users = [
    { name: "Alice", age: 30 },
    { name: "Bob", age: 25 },
];

// sort にコールバック
users.sort((a, b) => a.age - b.age);

// カスタムフィルタ
function filterBy(arr, predicate) {
    const result = [];
    for (const item of arr) {
        if (predicate(item)) result.push(item);
    }
    return result;
}

const adults = filterBy(users, user => user.age >= 18);
```

### ストラテジーパターン

```typescript
// 関数を戦略として切り替え
type SortStrategy<T> = (a: T, b: T) => number;

const byName: SortStrategy<User> = (a, b) => a.name.localeCompare(b.name);
const byAge: SortStrategy<User> = (a, b) => a.age - b.age;
const byNameDesc: SortStrategy<User> = (a, b) => b.name.localeCompare(a.name);

function sortUsers(users: User[], strategy: SortStrategy<User>): User[] {
    return [...users].sort(strategy);
}

sortUsers(users, byName);     // 名前順
sortUsers(users, byAge);      // 年齢順
sortUsers(users, byNameDesc); // 名前逆順
```

---

## 3. 関数合成

```typescript
// 関数を組み合わせて新しい関数を作る
const compose = <A, B, C>(f: (b: B) => C, g: (a: A) => B) =>
    (a: A): C => f(g(a));

const double = (x: number) => x * 2;
const addOne = (x: number) => x + 1;

const doubleAndAddOne = compose(addOne, double);
doubleAndAddOne(5);  // → 11

// pipe（左から右へ適用）
const pipe = <T>(...fns: ((x: T) => T)[]) =>
    (x: T): T => fns.reduce((acc, fn) => fn(acc), x);

const transform = pipe(
    (x: number) => x * 2,
    (x: number) => x + 1,
    (x: number) => x.toString(),
);
// ↑ 型の問題があるため、実用では fp-ts 等のライブラリを使用
```

```python
# Python: functools を活用
from functools import reduce, partial

# partial: 引数の部分適用
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

square(5)  # → 25
cube(3)    # → 27

# reduce: 畳み込み
numbers = [1, 2, 3, 4, 5]
total = reduce(lambda acc, x: acc + x, numbers, 0)  # → 15
```

---

## 4. 言語ごとの関数の扱い

```
┌──────────────┬──────────────────────────────────────┐
│ 言語          │ 関数の扱い                            │
├──────────────┼──────────────────────────────────────┤
│ JavaScript   │ 完全な第一級。function, arrow, method │
│ Python       │ 完全な第一級。def, lambda             │
│ Rust         │ fn ポインタ + クロージャ（3種類）     │
│ Go           │ 第一級。func リテラル                  │
│ Java         │ 限定的。ラムダ式（SAMインターフェース）│
│ C            │ 関数ポインタのみ（制限的）              │
│ Haskell      │ 全てが関数。最も自然に第一級           │
└──────────────┴──────────────────────────────────────┘
```

---

## まとめ

| 概念 | 説明 |
|------|------|
| 第一級関数 | 関数を値として扱える |
| コールバック | 関数を引数として渡す |
| ストラテジー | 振る舞いを関数で差し替える |
| 部分適用 | 引数の一部を固定して新関数を作る |
| 関数合成 | 関数を組み合わせて新関数を作る |

---

## 次に読むべきガイド
→ [[01-closures.md]] — クロージャ

---

## 参考文献
1. Abelson, H. & Sussman, G. "SICP." Ch.1, MIT Press, 1996.
