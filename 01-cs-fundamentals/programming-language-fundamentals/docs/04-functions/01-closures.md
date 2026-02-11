# クロージャ（Closures）

> クロージャは「定義時の環境を捕捉する関数」。状態を持つ関数を作り出す、関数型プログラミングの核心技術。

## この章で学ぶこと

- [ ] クロージャの仕組み（レキシカルスコープ捕捉）を理解する
- [ ] 各言語のクロージャの実装の違いを把握する
- [ ] クロージャの実践的な活用パターンを習得する

---

## 1. クロージャとは

```
クロージャ = 関数 + その関数が定義された環境（自由変数）

  通常の関数: 引数のみを入力として受け取る
  クロージャ: 引数 + 定義時のスコープの変数を使える
```

```javascript
// JavaScript: クロージャの基本
function makeCounter() {
    let count = 0;  // 自由変数（クロージャが捕捉する）
    return {
        increment: () => ++count,
        decrement: () => --count,
        getCount: () => count,
    };
}

const counter = makeCounter();
counter.increment();  // 1
counter.increment();  // 2
counter.decrement();  // 1
counter.getCount();   // 1

// count は makeCounter の実行が終わっても生存する
// → クロージャが count への参照を保持しているため
```

---

## 2. 各言語のクロージャ

### Python

```python
# Python: クロージャ
def make_multiplier(factor):
    def multiply(x):
        return x * factor  # factor を捕捉
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
double(5)   # → 10
triple(5)   # → 15

# 注意: Python のクロージャは参照を捕捉する
def make_functions():
    functions = []
    for i in range(5):
        functions.append(lambda: i)  # i の「参照」を捕捉
    return functions

# 全て 4 を返す（最後の i の値）
[f() for f in make_functions()]  # → [4, 4, 4, 4, 4]

# 修正: デフォルト引数で値を捕捉
def make_functions_fixed():
    functions = []
    for i in range(5):
        functions.append(lambda i=i: i)  # 値を捕捉
    return functions
[f() for f in make_functions_fixed()]  # → [0, 1, 2, 3, 4]
```

### Rust

```rust
// Rust: 3種類のクロージャトレイト

// Fn: 環境を不変に借用
let name = String::from("Gaku");
let greet = || println!("Hello, {}!", name);  // &name を捕捉
greet();
greet();  // 何度でも呼べる
println!("{}", name);  // name はまだ使える

// FnMut: 環境を可変に借用
let mut count = 0;
let mut increment = || { count += 1; count };  // &mut count を捕捉
increment();  // 1
increment();  // 2

// FnOnce: 環境を所有権ごと移動
let name = String::from("Gaku");
let consume = move || {  // move: 所有権を移動
    println!("Consumed: {}", name);
    drop(name);
};
consume();
// consume();  // ❌ FnOnce は1回しか呼べない
// println!("{}", name);  // ❌ name は移動済み

// クロージャを引数に取る関数
fn apply<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(x)
}
let result = apply(|x| x * 2 + 1, 5);  // → 11
```

### Go

```go
// Go: クロージャ
func makeAdder(base int) func(int) int {
    return func(x int) int {
        return base + x  // base を捕捉（参照）
    }
}

add5 := makeAdder(5)
add10 := makeAdder(10)
fmt.Println(add5(3))   // → 8
fmt.Println(add10(3))  // → 13
```

---

## 3. クロージャの実践パターン

```javascript
// パターン1: プライベート状態（モジュールパターン）
const createLogger = (prefix) => {
    let logCount = 0;
    return {
        log: (msg) => console.log(`[${prefix}] #${++logCount}: ${msg}`),
        getCount: () => logCount,
    };
};

const appLog = createLogger("APP");
appLog.log("Started");   // [APP] #1: Started
appLog.log("Running");   // [APP] #2: Running

// パターン2: メモ化（キャッシュ）
function memoize(fn) {
    const cache = new Map();
    return function(...args) {
        const key = JSON.stringify(args);
        if (cache.has(key)) return cache.get(key);
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
}

const expensiveFn = memoize((n) => {
    console.log("Computing...");
    return n * n;
});
expensiveFn(5);  // Computing... → 25
expensiveFn(5);  // → 25（キャッシュから）

// パターン3: カリー化
const curry = (fn) => {
    const arity = fn.length;
    return function curried(...args) {
        if (args.length >= arity) return fn(...args);
        return (...moreArgs) => curried(...args, ...moreArgs);
    };
};

const add = curry((a, b, c) => a + b + c);
add(1)(2)(3);     // → 6
add(1, 2)(3);     // → 6
add(1)(2, 3);     // → 6

// パターン4: イベントハンドラの状態管理
function createDebounce(fn, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn(...args), delay);
    };
}

const debouncedSearch = createDebounce(query => {
    fetch(`/api/search?q=${query}`);
}, 300);
```

---

## 4. メモリとクロージャ

```
クロージャのメモリへの影響:

  1. 捕捉した変数はクロージャが生存する限り解放されない
     → メモリリークの原因になりうる

  2. 不要になったクロージャは参照を切る
     → イベントリスナーの解除を忘れない

  3. 捕捉する変数を最小限にする
     → 必要な値だけを捕捉、大きなオブジェクトの参照を避ける
```

```javascript
// ❌ メモリリークの例
function setupHandler() {
    const hugeData = loadHugeData();  // 大量のデータ
    element.addEventListener("click", () => {
        // hugeData の一部しか使わないが、全体が保持される
        console.log(hugeData.name);
    });
}

// ✅ 必要な値だけ捕捉
function setupHandler() {
    const hugeData = loadHugeData();
    const name = hugeData.name;  // 必要な値だけ抽出
    element.addEventListener("click", () => {
        console.log(name);
    });
}
```

---

## まとめ

| 言語 | クロージャの捕捉 | 特徴 |
|------|---------------|------|
| JavaScript | 参照（自動） | 最も自然に使える |
| Python | 参照（注意が必要） | ループ変数の罠 |
| Rust | Fn/FnMut/FnOnce | 所有権と統合。最も安全 |
| Go | 参照（自動） | シンプル |
| Java | 実質 final のみ | 制限的 |

---

## 次に読むべきガイド
→ [[02-higher-order-functions.md]] — 高階関数

---

## 参考文献
1. Abelson, H. & Sussman, G. "SICP." Ch.3, MIT Press, 1996.
2. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.13, 2023.
