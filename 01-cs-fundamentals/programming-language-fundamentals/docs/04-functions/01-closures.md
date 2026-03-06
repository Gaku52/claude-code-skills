# クロージャ（Closures）

> クロージャは「定義時の環境を捕捉する関数」。状態を持つ関数を作り出す、関数型プログラミングの核心技術。

## この章で学ぶこと

- [ ] クロージャの仕組み（レキシカルスコープ捕捉）を理解する
- [ ] 自由変数・束縛変数の区別と、環境がどのように保持されるかを把握する
- [ ] 各言語のクロージャの実装の違い（JavaScript, Python, Rust, Go, Java, C++）を比較する
- [ ] クロージャの実践的な活用パターン（メモ化、カリー化、デバウンスなど）を習得する
- [ ] メモリリークやループ変数の罠などのアンチパターンを回避できるようになる
- [ ] クロージャとオブジェクト指向の関係を理解し、適材適所で使い分けられるようになる

---

## 1. クロージャとは何か

### 1.1 定義と直感的理解

クロージャ（closure）とは、**関数とその関数が定義された時点のレキシカル環境（lexical environment）をひとまとめにしたもの**である。1964年に Peter J. Landin が SECD マシンの文脈で初めて提案し、その後 Scheme（1975年）で実用化された概念である。

通常の関数は、呼び出し時に渡される引数のみを入力として受け取る。一方クロージャは、引数に加えて「定義時のスコープに存在した変数」にもアクセスできる。この「定義時のスコープの変数」を**自由変数（free variable）**と呼ぶ。

```
クロージャ = 関数本体（コード）+ 環境（自由変数への束縛のセット）

  通常の関数:
    入力: 引数のみ
    出力: 戻り値

  クロージャ:
    入力: 引数 + 捕捉された自由変数
    出力: 戻り値
    副作用: 捕捉された可変変数の更新（言語による）
```

### 1.2 自由変数と束縛変数

クロージャを正確に理解するには、**自由変数**と**束縛変数（bound variable）**の区別が不可欠である。

```
+----------------------------------------------------------+
|  function multiply(x) {    // x は束縛変数（仮引数）     |
|      return x * factor;    // factor は自由変数           |
|  }                                                        |
|                                                           |
|  束縛変数: 関数自身が定義する変数（引数、ローカル変数）   |
|  自由変数: 関数内で使われるが、関数自身が定義しない変数   |
+----------------------------------------------------------+
```

関数内に自由変数が存在し、かつその自由変数が外側のスコープから捕捉される場合に、その関数はクロージャとなる。逆に言えば、自由変数を持たない関数（すべての変数が束縛変数である関数）は、単なる「閉じた関数（closed function）」であり、クロージャではない。

```
  用語の整理:

  開いた式（open expression）: 自由変数を含む式
    例: x * factor   （factor が自由変数）

  閉じた式（closed expression）: 自由変数を含まない式
    例: (x) => x * 2  （すべて束縛変数）

  クロージャ（closure）: 開いた式を「閉じる」操作
    = 自由変数に具体的な束縛を与えて閉じた式にする
    例: factor=3 の環境 + (x) => x * factor
        → 実質 (x) => x * 3 として動作
```

### 1.3 レキシカルスコープとダイナミックスコープ

クロージャの動作はスコープ規則と密接に関係する。現代の主要言語はほぼすべて**レキシカルスコープ（lexical scope / static scope）**を採用しており、クロージャはこの仕組みの上に成り立つ。

```
+-------------------------------------------------------------------+
|  レキシカルスコープ vs ダイナミックスコープ                         |
+-------------------------------------------------------------------+
|                                                                     |
|  レキシカルスコープ（静的スコープ）:                                |
|    変数の解決は「ソースコード上の構造」で決まる                     |
|    → 関数がどこで "定義" されたかが重要                            |
|    → コンパイル時に変数の参照先が確定                               |
|    → JavaScript, Python, Rust, Go, Java, C++ ...                   |
|                                                                     |
|  ダイナミックスコープ（動的スコープ）:                              |
|    変数の解決は「実行時のコールスタック」で決まる                   |
|    → 関数がどこで "呼び出" されたかが重要                          |
|    → 実行時まで変数の参照先が不確定                                 |
|    → Emacs Lisp, Bash, 一部の古い Lisp ...                        |
|                                                                     |
+-------------------------------------------------------------------+
```

レキシカルスコープの例を見てみよう。

```javascript
// レキシカルスコープの実証
const x = "global";

function outer() {
    const x = "outer";

    function inner() {
        // inner は outer の中で「定義」されている
        // → レキシカルスコープにより x は "outer"
        console.log(x);
    }

    return inner;
}

const fn = outer();
fn();  // "outer"（"global" ではない）

// inner が outer の外で「呼び出」されても、
// 定義時の環境（x = "outer"）を覚えている
// → これがクロージャの本質
```

もしダイナミックスコープであれば、`fn()` を呼び出した時点のスコープチェーンが使われるため、`x` は `"global"` になる。レキシカルスコープでは定義時のスコープが使われるため `"outer"` が出力される。

### 1.4 クロージャの生成過程（メモリモデル）

クロージャがどのようにメモリ上で実現されるかを、ステップバイステップで追跡しよう。

```
ステップ1: makeCounter() が呼び出される
+------------------------------------------+
|  Call Frame: makeCounter()               |
|  ┌────────────────────────────┐          |
|  │  count: 0                  │          |
|  │  (ローカル変数)            │          |
|  └────────────────────────────┘          |
|  return { increment, decrement, get }    |
+------------------------------------------+

ステップ2: 内部関数がクロージャとして返される
+------------------------------------------+
|  返されたオブジェクト:                    |
|  {                                        |
|    increment: [Function + Env{count}],   |
|    decrement: [Function + Env{count}],   |
|    getCount:  [Function + Env{count}]    |
|  }                                        |
|                                           |
|  ※ 3つの関数が同一の count を共有       |
+------------------------------------------+

ステップ3: makeCounter のフレームは通常なら破棄されるが...
+------------------------------------------+
|  makeCounter のスタックフレーム → 破棄   |
|                                           |
|  しかし count はヒープに移動済み         |
|  （クロージャが参照を保持しているため）  |
|                                           |
|  ┌─────────────┐                         |
|  │ Heap        │                         |
|  │  count: 0   │ ← increment が参照     |
|  │             │ ← decrement が参照     |
|  │             │ ← getCount が参照      |
|  └─────────────┘                         |
+------------------------------------------+

ステップ4: increment() を呼ぶと
+------------------------------------------+
|  increment() 実行:                        |
|    1. 捕捉した環境から count を取得       |
|    2. count を 0 → 1 に更新              |
|    3. 1 を返す                            |
|                                           |
|  ┌─────────────┐                         |
|  │ Heap        │                         |
|  │  count: 1   │ ← 更新された！         |
|  └─────────────┘                         |
+------------------------------------------+
```

```javascript
// JavaScript: クロージャの基本（完全版）
function makeCounter() {
    let count = 0;  // 自由変数（クロージャが捕捉する）
    return {
        increment: () => ++count,
        decrement: () => --count,
        getCount: () => count,
    };
}

const counter = makeCounter();
console.log(counter.increment());  // 1
console.log(counter.increment());  // 2
console.log(counter.decrement());  // 1
console.log(counter.getCount());   // 1

// count は makeCounter の実行が終わっても生存する
// → クロージャが count への参照を保持しているため

// 独立したカウンタを複数生成できる
const counter2 = makeCounter();
console.log(counter2.increment());  // 1（counter とは独立）
console.log(counter.getCount());    // 1（counter は影響を受けない）
```

### 1.5 クロージャとオブジェクトの双対性

コンピュータサイエンスにおいて、クロージャとオブジェクトは**双対（dual）**の関係にあるとされる。これは Norman Adams と Jonathan Rees による有名な格言に由来する。

> "Objects are a poor man's closures. Closures are a poor man's objects."
> （オブジェクトは貧者のクロージャ。クロージャは貧者のオブジェクト。）

```
+------------------------------------------------------+
|  クロージャとオブジェクトの双対性                      |
+------------------------------------------------------+
|                                                        |
|  クロージャ版:                                         |
|  function makeCounter() {                              |
|      let count = 0;            // ← 隠された状態      |
|      return {                                          |
|          increment: () => ++count,                     |
|          getCount: () => count,                        |
|      };                                                |
|  }                                                     |
|                                                        |
|  オブジェクト版:                                       |
|  class Counter {                                       |
|      #count = 0;               // ← 隠された状態      |
|      increment() { return ++this.#count; }             |
|      getCount() { return this.#count; }                |
|  }                                                     |
|                                                        |
|  → どちらも「隠された状態 + それを操作するメソッド」  |
|  → 表現形式が違うだけで、本質的に同等                  |
+------------------------------------------------------+
```

| 観点 | クロージャ | オブジェクト |
|------|----------|------------|
| 状態の保持 | 捕捉された自由変数 | インスタンスフィールド |
| 操作の定義 | 返された関数群 | メソッド |
| カプセル化 | スコープによる自然な隠蔽 | アクセス修飾子（private 等） |
| 継承 | 直接的にはサポートしない | クラス継承、インターフェース |
| 多態性 | 高階関数で実現 | メソッドオーバーライドで実現 |
| メモリ効率 | 各クロージャが独立した環境を持つ | プロトタイプ/vtable で共有可能 |
| 適したケース | 単一操作、コールバック、部分適用 | 複数操作、状態が複雑な場合 |

---

## 2. 各言語のクロージャ詳解

### 2.1 JavaScript / TypeScript

JavaScript はクロージャを最も自然に扱える言語の一つである。関数がファーストクラスオブジェクトであり、レキシカルスコープを持つため、クロージャは言語の基盤となっている。

```javascript
// ─── 捕捉の仕組み ───
// JavaScript のクロージャは常に「参照」を捕捉する

function createAccumulator(initial) {
    let total = initial;

    return {
        add(value) {
            total += value;
            return total;
        },
        subtract(value) {
            total -= value;
            return total;
        },
        reset() {
            total = initial;  // initial も捕捉されている
            return total;
        },
        getTotal() {
            return total;
        }
    };
}

const acc = createAccumulator(100);
console.log(acc.add(50));       // 150
console.log(acc.add(30));       // 180
console.log(acc.subtract(20));  // 160
console.log(acc.reset());       // 100（initial に戻る）

// ─── IIFE（即時実行関数式）とクロージャ ───
// ES5 時代のモジュールパターン
const Module = (function() {
    let privateState = 0;
    const privateHelper = (x) => x * 2;

    return {
        publicMethod(value) {
            privateState += privateHelper(value);
            return privateState;
        },
        getState() {
            return privateState;
        }
    };
})();

Module.publicMethod(5);  // 10
Module.publicMethod(3);  // 16
// privateState, privateHelper は外部からアクセス不可
```

```typescript
// TypeScript: 型付きクロージャ
type Validator<T> = (value: T) => boolean;
type ValidationRule<T> = {
    validate: Validator<T>;
    message: string;
};

function createValidator<T>(
    rules: ValidationRule<T>[]
): (value: T) => string[] {
    // rules をクロージャが捕捉
    return (value: T): string[] => {
        return rules
            .filter(rule => !rule.validate(value))
            .map(rule => rule.message);
    };
}

const validateAge = createValidator<number>([
    {
        validate: (n) => n >= 0,
        message: "年齢は0以上でなければなりません"
    },
    {
        validate: (n) => n <= 150,
        message: "年齢は150以下でなければなりません"
    },
    {
        validate: (n) => Number.isInteger(n),
        message: "年齢は整数でなければなりません"
    }
]);

console.log(validateAge(25));   // []
console.log(validateAge(-5));   // ["年齢は0以上でなければなりません"]
console.log(validateAge(200));  // ["年齢は150以下でなければなりません"]
```

### 2.2 Python

Python のクロージャには独特の注意点がある。特に**変数の再束縛に対する制限**と**ループ変数の罠**は、多くの開発者がつまずくポイントである。

```python
# ─── 基本的なクロージャ ───
def make_multiplier(factor):
    def multiply(x):
        return x * factor  # factor を捕捉（参照）
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))   # 10
print(triple(5))   # 15

# ─── nonlocal キーワード ───
# Python 3 で導入。捕捉した変数を再束縛するために必要
def make_counter():
    count = 0
    def increment():
        nonlocal count     # これがないと UnboundLocalError
        count += 1
        return count
    def get_count():
        return count       # 読み取りのみなら nonlocal 不要
    return increment, get_count

inc, get = make_counter()
print(inc())    # 1
print(inc())    # 2
print(get())    # 2

# ─── ループ変数の罠（重要！）───
def make_functions():
    functions = []
    for i in range(5):
        functions.append(lambda: i)  # i の「参照」を捕捉
    return functions

# 全て 4 を返す（最後の i の値）
results = [f() for f in make_functions()]
print(results)  # [4, 4, 4, 4, 4]

# 修正法1: デフォルト引数で値を捕捉
def make_functions_fixed_v1():
    functions = []
    for i in range(5):
        functions.append(lambda i=i: i)  # デフォルト引数で値をコピー
    return functions

print([f() for f in make_functions_fixed_v1()])  # [0, 1, 2, 3, 4]

# 修正法2: functools.partial を使う
from functools import partial

def make_functions_fixed_v2():
    def identity(x):
        return x
    return [partial(identity, i) for i in range(5)]

print([f() for f in make_functions_fixed_v2()])  # [0, 1, 2, 3, 4]

# 修正法3: クロージャファクトリを使う
def make_functions_fixed_v3():
    def make_f(val):
        return lambda: val   # val は make_f のローカル変数
    return [make_f(i) for i in range(5)]

print([f() for f in make_functions_fixed_v3()])  # [0, 1, 2, 3, 4]
```

### 2.3 Rust

Rust のクロージャは所有権システムと統合されており、3つのトレイト `Fn`, `FnMut`, `FnOnce` によって分類される。これにより、コンパイル時にクロージャの安全性が保証される。

```rust
// ─── 3種類のクロージャトレイト ───

// 1. Fn: 環境を不変参照（&T）で借用
//    → 何度でも呼べる、環境を変更しない
fn demonstrate_fn() {
    let name = String::from("Alice");
    let greet = || println!("Hello, {}!", name);  // &name を捕捉

    greet();  // 何度でも呼べる
    greet();
    println!("name is still valid: {}", name);  // name はまだ使える
}

// 2. FnMut: 環境を可変参照（&mut T）で借用
//    → 何度でも呼べるが、環境を変更する
fn demonstrate_fn_mut() {
    let mut count = 0;
    let mut increment = || {
        count += 1;  // &mut count を捕捉
        count
    };

    println!("{}", increment());  // 1
    println!("{}", increment());  // 2
    // 注意: increment を使っている間は count を直接触れない
}

// 3. FnOnce: 環境の所有権を奪取
//    → 1回しか呼べない（所有権が消費されるため）
fn demonstrate_fn_once() {
    let data = vec![1, 2, 3, 4, 5];
    let consume = move || {
        // move キーワードで data の所有権を移動
        let sum: i32 = data.iter().sum();
        println!("Sum: {}, dropping data", sum);
        drop(data);  // data を消費
    };

    consume();
    // consume();  // コンパイルエラー: FnOnce は1回のみ
    // println!("{:?}", data);  // コンパイルエラー: data は移動済み
}

// ─── トレイトの包含関係 ───
//
//  FnOnce ⊃ FnMut ⊃ Fn
//
//  Fn を実装するクロージャは FnMut も FnOnce も実装する
//  FnMut を実装するクロージャは FnOnce も実装する
//  FnOnce のみを実装するクロージャは Fn, FnMut を実装しない

// ─── クロージャを引数に取る関数 ───
fn apply_twice<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(f(x))
}

fn apply_and_collect<F: FnMut(i32) -> i32>(mut f: F, items: &[i32]) -> Vec<i32> {
    items.iter().map(|&x| f(x)).collect()
}

fn consume_and_run<F: FnOnce() -> String>(f: F) -> String {
    f()  // F は FnOnce なので1回だけ呼べる
}

fn main() {
    // Fn の例
    let double = |x: i32| x * 2;
    println!("{}", apply_twice(double, 3));  // 12 (3*2=6, 6*2=12)

    // FnMut の例
    let mut offset = 0;
    let add_increasing = |x: i32| {
        offset += 1;
        x + offset
    };
    let result = apply_and_collect(add_increasing, &[10, 20, 30]);
    println!("{:?}", result);  // [11, 22, 33]

    // FnOnce の例
    let name = String::from("World");
    let greeting = move || format!("Hello, {}!", name);
    println!("{}", consume_and_run(greeting));  // "Hello, World!"
}
```

```
Rust クロージャトレイトの判定フロー:

  クロージャが環境の値を...
    │
    ├─ 消費する（move + drop）→ FnOnce のみ
    │    例: move || { drop(data); }
    │
    ├─ 変更する（&mut）→ FnMut（+ FnOnce）
    │    例: || { count += 1; }
    │
    └─ 読み取るのみ（&）→ Fn（+ FnMut + FnOnce）
         例: || { println!("{}", name); }

  ※ move キーワードは「捕捉方法」を変えるだけで、
    トレイトを直接決めるわけではない
    move || println!("{}", name)  は Fn を実装できる
    （所有権を移動しても、読み取りしかしないなら Fn）
```

### 2.4 Go

Go のクロージャはシンプルで、関数リテラル（無名関数）が外側のスコープの変数を参照で捕捉する。

```go
package main

import (
    "fmt"
    "sync"
)

// 基本的なクロージャ
func makeAdder(base int) func(int) int {
    return func(x int) int {
        return base + x  // base を参照で捕捉
    }
}

// ジェネレータパターン
func fibonacci() func() int {
    a, b := 0, 1
    return func() int {
        result := a
        a, b = b, a+b  // a, b を可変に捕捉
        return result
    }
}

// ミドルウェアパターン（Go で頻出）
type Middleware func(http.HandlerFunc) http.HandlerFunc

func withLogging(logger *log.Logger) Middleware {
    return func(next http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            logger.Printf("%s %s", r.Method, r.URL.Path)
            next(w, r)
        }
    }
}

// ─── Go でのループ変数の罠と修正 ───
func main() {
    add5 := makeAdder(5)
    add10 := makeAdder(10)
    fmt.Println(add5(3))   // 8
    fmt.Println(add10(3))  // 13

    fib := fibonacci()
    for i := 0; i < 8; i++ {
        fmt.Printf("%d ", fib())  // 0 1 1 2 3 5 8 13
    }
    fmt.Println()

    // ループ変数の罠（Go 1.21 以前）
    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        // Go 1.22+ では各イテレーションで i が新しくなるため問題なし
        // Go 1.21 以前では i をシャドウイングする必要がある
        i := i  // シャドウイング（Go 1.21 以前で必要）
        go func() {
            defer wg.Done()
            fmt.Println(i)
        }()
    }
    wg.Wait()
}
```

### 2.5 Java

Java はラムダ式（Java 8+）でクロージャに近い機能を提供するが、**実質的に final（effectively final）な変数のみ**を捕捉できるという制限がある。

```java
import java.util.function.*;
import java.util.*;
import java.util.stream.*;

public class ClosureExamples {

    // 基本: ラムダ式による捕捉
    public static Function<Integer, Integer> makeAdder(int base) {
        // base は effectively final（再代入されない）
        return x -> x + base;
    }

    // 可変状態が必要な場合は配列やAtomicで包む
    public static Supplier<Integer> makeCounter() {
        // int count = 0; だと再代入できない
        final int[] count = {0};  // 配列は要素を変更できる
        return () -> ++count[0];
    }

    // Stream API でのクロージャ活用
    public static List<String> filterAndTransform(
            List<String> items, String prefix, int minLength) {
        // prefix, minLength は effectively final
        return items.stream()
            .filter(s -> s.length() >= minLength)
            .map(s -> prefix + s.toUpperCase())
            .collect(Collectors.toList());
    }

    public static void main(String[] args) {
        Function<Integer, Integer> add5 = makeAdder(5);
        System.out.println(add5.apply(3));  // 8

        Supplier<Integer> counter = makeCounter();
        System.out.println(counter.get());  // 1
        System.out.println(counter.get());  // 2

        List<String> result = filterAndTransform(
            Arrays.asList("hi", "hello", "hey", "greetings"),
            ">> ", 4
        );
        System.out.println(result);
        // [>> HELLO, >> GREETINGS]

        // コンパイルエラーの例
        // int x = 10;
        // Runnable r = () -> { x = 20; };  // エラー: x は effectively final でない
    }
}
```

### 2.6 C++

C++ のクロージャ（ラムダ式、C++11 以降）は、キャプチャリストで捕捉方法を明示的に指定する。

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

int main() {
    // ─── キャプチャの種類 ───

    // [=] : すべての外部変数を値でキャプチャ（コピー）
    int x = 10, y = 20;
    auto by_value = [=]() { return x + y; };
    x = 100;  // 変更しても...
    std::cout << by_value() << std::endl;  // 30（コピー時の値）

    // [&] : すべての外部変数を参照でキャプチャ
    int count = 0;
    auto by_ref = [&]() { return ++count; };
    std::cout << by_ref() << std::endl;  // 1
    std::cout << by_ref() << std::endl;  // 2
    std::cout << count << std::endl;     // 2（参照なので変更される）

    // 個別指定: [x, &y] → x は値、y は参照
    int a = 1, b = 2;
    auto mixed = [a, &b]() {
        // a は読み取り専用（値コピー）
        b += a;  // b は参照なので変更可能
        return b;
    };
    std::cout << mixed() << std::endl;  // 3
    std::cout << b << std::endl;        // 3

    // mutable: 値キャプチャした変数をラムダ内で変更可能にする
    int counter = 0;
    auto mut_lambda = [counter]() mutable {
        return ++counter;  // ラムダ内のコピーを変更
    };
    std::cout << mut_lambda() << std::endl;  // 1
    std::cout << mut_lambda() << std::endl;  // 2
    std::cout << counter << std::endl;       // 0（元の変数は不変）

    // ─── STL アルゴリズムとの組み合わせ ───
    std::vector<int> nums = {5, 2, 8, 1, 9, 3};
    int threshold = 4;

    // threshold より大きい要素だけをフィルタ
    std::vector<int> filtered;
    std::copy_if(nums.begin(), nums.end(),
                 std::back_inserter(filtered),
                 [threshold](int n) { return n > threshold; });
    // filtered: {5, 8, 9}

    return 0;
}
```

---

## 3. 言語間比較表

### 3.1 捕捉方式の比較

| 言語 | 捕捉方式 | 可変捕捉 | 明示的指定 | 特記事項 |
|------|---------|---------|-----------|---------|
| JavaScript | 参照（自動） | 可能 | 不要 | 最も自然、var のホイスティングに注意 |
| Python | 参照（自動） | nonlocal 必要 | 不要 | ループ変数の罠、nonlocal/global |
| Rust | 借用 or 移動 | FnMut で可能 | move で制御 | 所有権と統合、コンパイル時安全性保証 |
| Go | 参照（自動） | 可能 | 不要 | goroutine との組み合わせに注意 |
| Java | 値コピー | 不可（effectively final） | 不要 | 匿名クラスの延長線上 |
| C++ | 値 or 参照 | mutable で可能 | キャプチャリスト | 最も細かい制御が可能 |
| Swift | 参照（自動） | 可能 | [weak/unowned] | ARC との統合、循環参照に注意 |

### 3.2 構文の比較

| 言語 | クロージャ構文 | 型推論 |
|------|-------------|-------|
| JavaScript | `(x) => x * 2` / `function(x) { return x * 2; }` | 完全 |
| Python | `lambda x: x * 2` / `def f(x): return x * 2` | 型ヒント任意 |
| Rust | `\|x\| x * 2` / `\|x: i32\| -> i32 { x * 2 }` | 多くの場合推論可能 |
| Go | `func(x int) int { return x * 2 }` | 戻り値型は明示 |
| Java | `(x) -> x * 2` / `(Integer x) -> x * 2` | 関数型インターフェースから推論 |
| C++ | `[](int x) { return x * 2; }` | auto で受け取り可能 |
| Swift | `{ x in x * 2 }` / `{ $0 * 2 }` | 完全 |

---

## 4. クロージャの実践パターン

### 4.1 パターン1: プライベート状態（モジュールパターン）

```javascript
// データのカプセル化 - 外部から直接アクセスできない状態を持つ
const createLogger = (prefix) => {
    let logCount = 0;
    const history = [];

    return {
        log(msg) {
            logCount++;
            const entry = `[${prefix}] #${logCount}: ${msg}`;
            history.push(entry);
            console.log(entry);
        },
        warn(msg) {
            logCount++;
            const entry = `[${prefix}] #${logCount} ⚠: ${msg}`;
            history.push(entry);
            console.warn(entry);
        },
        getCount() {
            return logCount;
        },
        getHistory() {
            return [...history];  // コピーを返す（防御的コピー）
        }
    };
};

const appLog = createLogger("APP");
appLog.log("Started");    // [APP] #1: Started
appLog.warn("Low memory"); // [APP] #2 ⚠: Low memory
console.log(appLog.getCount());    // 2
console.log(appLog.getHistory());  // ["[APP] #1: Started", ...]
// logCount, history は外部からアクセス不可
```

### 4.2 パターン2: メモ化（Memoization）

```javascript
// 汎用メモ化関数
function memoize(fn) {
    const cache = new Map();
    const memoized = function(...args) {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };

    // キャッシュ管理メソッドも提供
    memoized.clearCache = () => cache.clear();
    memoized.cacheSize = () => cache.size;
    memoized.hasCache = (...args) => cache.has(JSON.stringify(args));

    return memoized;
}

// 使用例: フィボナッチ数列
const fibonacci = memoize(function fib(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
});

console.log(fibonacci(50));  // 12586269025（即座に計算完了）
console.log(fibonacci.cacheSize());  // 51

// 使用例: API呼び出しのキャッシュ
const fetchUser = memoize(async (userId) => {
    const response = await fetch(`/api/users/${userId}`);
    return response.json();
});
```

### 4.3 パターン3: カリー化（Currying）

```javascript
// 汎用カリー化関数
const curry = (fn) => {
    const arity = fn.length;
    return function curried(...args) {
        if (args.length >= arity) {
            return fn(...args);
        }
        return (...moreArgs) => curried(...args, ...moreArgs);
    };
};

// 使用例
const add = curry((a, b, c) => a + b + c);
console.log(add(1)(2)(3));     // 6
console.log(add(1, 2)(3));     // 6
console.log(add(1)(2, 3));     // 6
console.log(add(1, 2, 3));     // 6

// 実用的な例: ログフォーマッタ
const formatLog = curry((level, module, message) =>
    `[${new Date().toISOString()}] [${level}] [${module}] ${message}`
);

const errorLog = formatLog("ERROR");
const errorLogAuth = errorLog("AUTH");

console.log(errorLogAuth("Login failed"));
// [2026-03-06T...] [ERROR] [AUTH] Login failed
```

### 4.4 パターン4: デバウンスとスロットル

```javascript
// デバウンス: 最後の呼び出しから一定時間後に実行
function createDebounce(fn, delay) {
    let timeoutId = null;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

// スロットル: 一定間隔で最大1回だけ実行
function createThrottle(fn, interval) {
    let lastTime = 0;
    let timeoutId = null;
    return function(...args) {
        const now = Date.now();
        const remaining = interval - (now - lastTime);
        clearTimeout(timeoutId);
        if (remaining <= 0) {
            lastTime = now;
            fn.apply(this, args);
        } else {
            timeoutId = setTimeout(() => {
                lastTime = Date.now();
                fn.apply(this, args);
            }, remaining);
        }
    };
}

// 使用例
const debouncedSearch = createDebounce((query) => {
    console.log(`Searching for: ${query}`);
}, 300);

const throttledScroll = createThrottle(() => {
    console.log("Scroll position:", window.scrollY);
}, 100);
```

### 4.5 パターン5: 関数合成（Function Composition）

```javascript
// 関数合成: 複数の関数を連結して新しい関数を作る
const compose = (...fns) =>
    fns.reduce((f, g) => (...args) => f(g(...args)));

const pipe = (...fns) =>
    fns.reduce((f, g) => (...args) => g(f(...args)));

// 使用例: データ変換パイプライン
const trim = (s) => s.trim();
const toLowerCase = (s) => s.toLowerCase();
const replaceSpaces = (s) => s.replace(/\s+/g, "-");
const addPrefix = (prefix) => (s) => `${prefix}${s}`;  // クロージャ！

const slugify = pipe(
    trim,
    toLowerCase,
    replaceSpaces,
    addPrefix("/blog/")
);

console.log(slugify("  Hello World  "));  // "/blog/hello-world"
console.log(slugify(" Closures Are Fun "));  // "/blog/closures-are-fun"
```

### 4.6 パターン6: イテレータ / ジェネレータ

クロージャを使って遅延評価のイテレータを構築できる。これは内部状態をクロージャが保持することで実現する。

```javascript
// 無限数列ジェネレータ
function range(start = 0, step = 1) {
    let current = start;
    return {
        next() {
            const value = current;
            current += step;
            return { value, done: false };
        },
        take(n) {
            const result = [];
            for (let i = 0; i < n; i++) {
                result.push(this.next().value);
            }
            return result;
        },
        [Symbol.iterator]() {
            return this;
        }
    };
}

const odds = range(1, 2);
console.log(odds.take(5));  // [1, 3, 5, 7, 9]

// フィルタ付きイテレータ（クロージャの連鎖）
function filterIterator(iterator, predicate) {
    return {
        next() {
            while (true) {
                const item = iterator.next();
                if (item.done) return item;
                if (predicate(item.value)) return item;
            }
        }
    };
}

function mapIterator(iterator, transform) {
    return {
        next() {
            const item = iterator.next();
            if (item.done) return item;
            return { value: transform(item.value), done: false };
        }
    };
}

// 使用例: 1から始まる偶数を2倍にして5個取得
const nums = range(1, 1);
const evens = filterIterator(nums, n => n % 2 === 0);
const doubled = mapIterator(evens, n => n * 2);

const results = [];
for (let i = 0; i < 5; i++) {
    results.push(doubled.next().value);
}
console.log(results);  // [4, 8, 12, 16, 20]
```

```python
# Python: クロージャベースのイテレータ
def infinite_counter(start=0):
    count = start
    def next_val():
        nonlocal count
        result = count
        count += 1
        return result
    return next_val

counter = infinite_counter(10)
print([counter() for _ in range(5)])  # [10, 11, 12, 13, 14]

# ジェネレータとクロージャの組み合わせ
def sliding_window(size):
    """スライディングウィンドウを管理するクロージャ"""
    window = []
    def add(value):
        window.append(value)
        if len(window) > size:
            window.pop(0)
        return list(window)  # 防御的コピー
    return add

slider = sliding_window(3)
print(slider(1))   # [1]
print(slider(2))   # [1, 2]
print(slider(3))   # [1, 2, 3]
print(slider(4))   # [2, 3, 4]
print(slider(5))   # [3, 4, 5]
```

### 4.7 パターン7: ミドルウェア / デコレータ

```python
# Python: デコレータはクロージャの典型的な活用例
import time
import functools

def retry(max_attempts=3, delay=1.0):
    """リトライデコレータ（クロージャで設定を捕捉）"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def rate_limit(calls_per_second):
    """レートリミットデコレータ"""
    min_interval = 1.0 / calls_per_second
    last_call_time = [0.0]  # リストで包んで nonlocal の代わりに

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call_time[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call_time[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
@rate_limit(calls_per_second=2)
def fetch_data(url):
    """リトライとレートリミット付きのデータ取得"""
    import urllib.request
    return urllib.request.urlopen(url).read()
```

```
デコレータのクロージャ構造（ネスト図解）:

  @retry(max_attempts=3, delay=0.5)
  def fetch_data(url): ...

  展開すると:

  retry(max_attempts=3, delay=0.5)     ← 設定を捕捉
    └─ decorator(func)                 ← 元の関数を捕捉
        └─ wrapper(*args, **kwargs)    ← 引数を受け取り実行
            │
            │  wrapper が捕捉している変数:
            │    - func (元の関数)
            │    - max_attempts (3)
            │    - delay (0.5)
            │
            │  呼び出しチェーン:
            │    fetch_data("...") → wrapper("...") → func("...")
            │                        ↑ リトライロジック
            │
            └─ 3つのスコープが入れ子になったクロージャ
```

---

## 5. メモリとクロージャ

### 5.1 クロージャのメモリモデル

クロージャは定義時の環境への参照を保持するため、ガベージコレクション（GC）の対象にならない変数が増える可能性がある。

```
クロージャのメモリライフサイクル:

  [関数定義時]
  ┌──────────────────────────────────────────┐
  │  外側のスコープ                           │
  │  ┌─────────────────────────┐             │
  │  │ largeData = [...]       │─────┐       │
  │  │ config = {...}          │──┐  │       │
  │  │ name = "test"           │  │  │       │
  │  └─────────────────────────┘  │  │       │
  │                                │  │       │
  │  クロージャ = function() {     │  │       │
  │      use(config);         ─────┘  │       │
  │      // name は使っていない       │       │
  │      // largeData も使っていない──┘       │
  │  }                                        │
  └──────────────────────────────────────────┘

  [外側のスコープ終了後]
  ┌──────────────────────────────────────────┐
  │  GC 対象:                                 │
  │    name = "test"        ← 参照なし → 解放 │
  │                                           │
  │  GC 対象外（言語による）:                  │
  │    config = {...}       ← クロージャが参照 │
  │    largeData = [...]    ← 言語により異なる │
  │                                           │
  │  ※ V8 エンジンは使われない変数を          │
  │    最適化で捕捉しないことがある            │
  │  ※ ただしデバッガ使用時は全変数を保持     │
  └──────────────────────────────────────────┘
```

### 5.2 各言語のメモリ管理方式

| 言語 | GC方式 | クロージャの影響 | 対策 |
|------|-------|---------------|------|
| JavaScript | Mark-and-Sweep | 参照保持で寿命延長 | 不要な参照を null に |
| Python | 参照カウント + 循環検出 | 循環参照のリスク | weakref の活用 |
| Rust | 所有権（GC なし） | コンパイル時に安全性保証 | 明示的な lifetime |
| Go | 並行 Mark-and-Sweep | エスケープ解析で最適化 | 大きなデータはポインタで |
| Java | 世代別 GC | 匿名クラスと同様 | WeakReference の活用 |
| C++ | 手動（GC なし） | ダングリング参照の危険 | スマートポインタ、値キャプチャ |
| Swift | ARC（参照カウント） | 循環参照のリスク | [weak], [unowned] |

### 5.3 メモリリークの具体例と対策

```javascript
// ❌ アンチパターン1: 巨大なデータを不必要に捕捉
function setupHandler() {
    const hugeData = loadHugeData();  // 100MB のデータ
    const element = document.getElementById("btn");

    element.addEventListener("click", () => {
        // hugeData の name プロパティしか使わないが、
        // hugeData 全体（100MB）がクロージャに捕捉される
        console.log(hugeData.name);
    });
}

// ✅ 修正: 必要な値だけを事前に抽出
function setupHandlerFixed() {
    const hugeData = loadHugeData();
    const name = hugeData.name;  // 必要な値だけ抽出
    // hugeData はこの後 GC 対象になれる

    const element = document.getElementById("btn");
    element.addEventListener("click", () => {
        console.log(name);  // name だけを捕捉（数バイト）
    });
}

// ❌ アンチパターン2: イベントリスナーの解除忘れ
function createWidget() {
    const state = { count: 0, data: new Array(10000) };

    const handler = () => {
        state.count++;
        updateUI(state);
    };

    window.addEventListener("resize", handler);

    // Widget が破棄されても handler が残り続ける
    // → state も解放されない → メモリリーク
}

// ✅ 修正: クリーンアップ関数を返す
function createWidgetFixed() {
    const state = { count: 0, data: new Array(10000) };

    const handler = () => {
        state.count++;
        updateUI(state);
    };

    window.addEventListener("resize", handler);

    // クリーンアップ関数を返す
    return {
        destroy() {
            window.removeEventListener("resize", handler);
            // handler への参照がなくなり、state も GC 対象に
        }
    };
}

// React Hooks でのクリーンアップパターン
function useWindowResize(callback) {
    useEffect(() => {
        window.addEventListener("resize", callback);
        return () => {
            // クリーンアップ: コンポーネントアンマウント時に実行
            window.removeEventListener("resize", callback);
        };
    }, [callback]);
}
```

```python
# Python: 循環参照によるメモリリーク

# ❌ クロージャの循環参照
class Node:
    def __init__(self, value):
        self.value = value
        self.get_value = lambda: self.value
        # self.get_value → lambda → self → self.get_value
        # 循環参照が発生！

# ✅ weakref で解決
import weakref

class NodeFixed:
    def __init__(self, value):
        self.value = value
        weak_self = weakref.ref(self)
        self.get_value = lambda: weak_self().value if weak_self() else None
```

### 5.4 C++ のダングリング参照問題

```cpp
#include <iostream>
#include <functional>

// ❌ 危険: ローカル変数への参照キャプチャ後、スコープを抜ける
std::function<int()> createDangling() {
    int x = 42;
    return [&x]() { return x; };
    // x はスコープを抜けると破棄される
    // → 返されたラムダは破棄された x を参照
    // → 未定義動作！
}

// ✅ 安全: 値キャプチャを使う
std::function<int()> createSafe() {
    int x = 42;
    return [x]() { return x; };  // x のコピーを保持
    // ラムダ内のコピーはラムダと共に生存
}

// ✅ 安全: shared_ptr でヒープに保持
std::function<int()> createWithSharedPtr() {
    auto x = std::make_shared<int>(42);
    return [x]() { return *x; };
    // shared_ptr のコピーにより参照カウントが増加
    // ラムダが生存する限り x も生存
}
```

---

## 6. 高度なトピック

### 6.1 クロージャとコルーチン

クロージャは内部状態を保持できるため、コルーチン（中断と再開が可能な関数）の基盤となる。

```python
# Python: ジェネレータはクロージャの発展形
def coroutine_accumulator():
    """send() で値を受け取り、累積合計を返すコルーチン"""
    total = 0
    while True:
        value = yield total
        if value is not None:
            total += value

# 使用例
acc = coroutine_accumulator()
next(acc)          # ジェネレータを初期化（最初の yield まで進む）
print(acc.send(10))  # 10
print(acc.send(20))  # 30
print(acc.send(5))   # 35

# クロージャで同等の機能を実装
def closure_accumulator():
    total = [0]
    def send(value):
        total[0] += value
        return total[0]
    return send

acc2 = closure_accumulator()
print(acc2(10))  # 10
print(acc2(20))  # 30
print(acc2(5))   # 35
```

### 6.2 クロージャによる遅延評価

```javascript
// 遅延評価（Lazy Evaluation）をクロージャで実装
function lazy(computation) {
    let result;
    let computed = false;

    return () => {
        if (!computed) {
            result = computation();
            computed = true;
        }
        return result;
    };
}

// 使用例: 重い計算を必要になるまで遅延
const expensiveResult = lazy(() => {
    console.log("Computing...");
    let sum = 0;
    for (let i = 0; i < 1000000; i++) sum += i;
    return sum;
});

// この時点では計算されていない
console.log("Before access");
console.log(expensiveResult());  // "Computing..." → 499999500000
console.log(expensiveResult());  // 499999500000（キャッシュから、再計算なし）
```

```python
# Python: 遅延プロパティ（Lazy Property）
class LazyProperty:
    """クロージャを使った遅延プロパティデスクリプタ"""
    def __init__(self, func):
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.attr_name):
            setattr(obj, self.attr_name, self.func(obj))
        return getattr(obj, self.attr_name)

class DataProcessor:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    @LazyProperty
    def processed(self):
        """初回アクセス時にのみ計算される"""
        print("Processing data...")
        return [x * 2 for x in self.raw_data]

    @LazyProperty
    def statistics(self):
        """processed に依存する遅延プロパティ"""
        print("Computing statistics...")
        data = self.processed
        return {
            "mean": sum(data) / len(data),
            "max": max(data),
            "min": min(data)
        }

dp = DataProcessor([1, 2, 3, 4, 5])
# この時点では何も計算されていない
print(dp.statistics)  # "Processing data..." → "Computing statistics..." → {...}
print(dp.statistics)  # キャッシュから即座に返される
```

### 6.3 クロージャと並行処理

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// スレッドセーフなクロージャカウンタ
func makeAtomicCounter() func() int64 {
    var count int64 = 0
    return func() int64 {
        return atomic.AddInt64(&count, 1)
    }
}

// ワーカープールパターン（クロージャでタスクを定義）
func workerPool(numWorkers int, tasks []func()) {
    var wg sync.WaitGroup
    taskCh := make(chan func(), len(tasks))

    // ワーカーを起動
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for task := range taskCh {
                task()  // クロージャを実行
            }
        }(i)
    }

    // タスクを送信
    for _, task := range tasks {
        taskCh <- task
    }
    close(taskCh)

    wg.Wait()
}

func main() {
    counter := makeAtomicCounter()

    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter()
        }()
    }
    wg.Wait()
    fmt.Println(counter() - 1)  // 100

    // ワーカープールの使用例
    var mu sync.Mutex
    results := make([]int, 0)

    tasks := make([]func(), 10)
    for i := 0; i < 10; i++ {
        val := i  // ループ変数をコピー
        tasks[i] = func() {
            result := val * val
            mu.Lock()
            results = append(results, result)
            mu.Unlock()
        }
    }

    workerPool(4, tasks)
    fmt.Println(results)  // [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]（順序は不定）
}
```

### 6.4 型システムとクロージャ

```rust
// Rust: クロージャの型はそれぞれ一意（匿名型）
// → ジェネリクスか trait object で受け取る

// ジェネリクス版（ゼロコスト抽象化、静的ディスパッチ）
fn apply_generic<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(x)
}

// trait object 版（動的ディスパッチ、実行時コスト）
fn apply_dynamic(f: &dyn Fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

// Box<dyn Fn> でヒープに格納（所有権あり）
fn make_adder(n: i32) -> Box<dyn Fn(i32) -> i32> {
    Box::new(move |x| x + n)
}

// impl Fn で返す（ゼロコスト、ただし型は不透明）
fn make_multiplier(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x * n
}

fn main() {
    let double = |x| x * 2;

    // ジェネリクス: コンパイル時に型が決定、インライン化可能
    println!("{}", apply_generic(double, 5));   // 10
    println!("{}", apply_generic(|x| x + 1, 5)); // 6

    // trait object: 実行時にディスパッチ
    println!("{}", apply_dynamic(&double, 5));  // 10

    // Box<dyn Fn>: ヒープ上に格納
    let add5 = make_adder(5);
    println!("{}", add5(10));  // 15

    // impl Fn: スタック上に格納（効率的）
    let triple = make_multiplier(3);
    println!("{}", triple(10));  // 30
}
```

```
Rust クロージャの格納方式の比較:

  ┌─────────────────────────────────────────────────────┐
  │  方式              │ ディスパッチ │ メモリ   │ コスト │
  ├─────────────────────────────────────────────────────┤
  │  ジェネリクス <F>   │ 静的         │ スタック │ ゼロ   │
  │  impl Fn           │ 静的         │ スタック │ ゼロ   │
  │  &dyn Fn           │ 動的         │ 参照     │ 小     │
  │  Box<dyn Fn>       │ 動的         │ ヒープ   │ 中     │
  └─────────────────────────────────────────────────────┘

  使い分け:
    - 関数の引数 → ジェネリクス（最も効率的）
    - 関数の戻り値（単一型）→ impl Fn
    - 関数の戻り値（複数型の可能性）→ Box<dyn Fn>
    - コレクションに格納 → Box<dyn Fn> or Vec<Box<dyn Fn>>
```

---

## 7. アンチパターンと落とし穴

### 7.1 アンチパターン1: ループ変数の非意図的捕捉

これは JavaScript, Python, Go など多くの言語で発生する古典的な罠である。ループ変数は各イテレーションで新しい変数が作られるのではなく、同一の変数が更新される（言語・バージョンによる）。

```javascript
// ❌ アンチパターン: var を使ったループでのクロージャ
function createButtons() {
    const buttons = [];
    for (var i = 0; i < 5; i++) {
        // var i はループ全体で1つの変数（関数スコープ）
        buttons.push({
            label: `Button ${i}`,
            onClick: function() {
                console.log(`Clicked button ${i}`);
                // 全てのクロージャが同じ i を参照
            }
        });
    }
    return buttons;
}

const btns = createButtons();
btns[0].onClick();  // "Clicked button 5" ← 期待は 0
btns[1].onClick();  // "Clicked button 5" ← 期待は 1
btns[4].onClick();  // "Clicked button 5" ← 期待は 4

// ✅ 修正法1: let を使う（ES6+、最も推奨）
function createButtonsFixed1() {
    const buttons = [];
    for (let i = 0; i < 5; i++) {
        // let i は各イテレーションで新しい変数（ブロックスコープ）
        buttons.push({
            label: `Button ${i}`,
            onClick: function() {
                console.log(`Clicked button ${i}`);
            }
        });
    }
    return buttons;
}

// ✅ 修正法2: IIFE で新しいスコープを作る（ES5 時代の手法）
function createButtonsFixed2() {
    const buttons = [];
    for (var i = 0; i < 5; i++) {
        (function(index) {
            buttons.push({
                label: `Button ${index}`,
                onClick: function() {
                    console.log(`Clicked button ${index}`);
                }
            });
        })(i);  // i の値をコピーして渡す
    }
    return buttons;
}

// ✅ 修正法3: forEach を使う
function createButtonsFixed3() {
    return Array.from({ length: 5 }, (_, i) => ({
        label: `Button ${i}`,
        onClick() {
            console.log(`Clicked button ${i}`);
        }
    }));
}
```

```
ループ変数の罠のメカニズム:

  var を使った場合:
  ┌──────────────────────────────────────────┐
  │  スコープ: createButtons 全体            │
  │                                          │
  │  var i;  ← 1つの変数                    │
  │                                          │
  │  i=0: クロージャ0 → i を参照 ─────┐     │
  │  i=1: クロージャ1 → i を参照 ─────┤     │
  │  i=2: クロージャ2 → i を参照 ─────┤     │
  │  i=3: クロージャ3 → i を参照 ─────┤     │
  │  i=4: クロージャ4 → i を参照 ─────┤     │
  │  ループ終了: i=5                   │     │
  │                                    ▼     │
  │  全クロージャが参照する i の値 → 5       │
  └──────────────────────────────────────────┘

  let を使った場合:
  ┌──────────────────────────────────────────┐
  │  スコープ: createButtons 全体            │
  │                                          │
  │  { let i=0; クロージャ0 → i を参照 }    │
  │  { let i=1; クロージャ1 → i を参照 }    │
  │  { let i=2; クロージャ2 → i を参照 }    │
  │  { let i=3; クロージャ3 → i を参照 }    │
  │  { let i=4; クロージャ4 → i を参照 }    │
  │                                          │
  │  各クロージャが独自の i を参照            │
  │  → 期待通りの動作                        │
  └──────────────────────────────────────────┘
```

### 7.2 アンチパターン2: 過度なクロージャのネスト（コールバック地獄）

```javascript
// ❌ アンチパターン: クロージャの深いネスト
function processOrder(orderId) {
    fetchOrder(orderId, function(order) {
        fetchUser(order.userId, function(user) {
            fetchProducts(order.productIds, function(products) {
                calculateShipping(user.address, products, function(shipping) {
                    applyDiscount(user.membership, order.total, function(finalPrice) {
                        createInvoice(order, user, products, shipping, finalPrice,
                            function(invoice) {
                                sendEmail(user.email, invoice, function(result) {
                                    console.log("Done!", result);
                                });
                            }
                        );
                    });
                });
            });
        });
    });
}

// ✅ 修正: Promise チェーン
function processOrderFixed(orderId) {
    return fetchOrder(orderId)
        .then(order => Promise.all([
            order,
            fetchUser(order.userId),
            fetchProducts(order.productIds)
        ]))
        .then(([order, user, products]) => Promise.all([
            order, user, products,
            calculateShipping(user.address, products)
        ]))
        .then(([order, user, products, shipping]) => Promise.all([
            order, user, products, shipping,
            applyDiscount(user.membership, order.total)
        ]))
        .then(([order, user, products, shipping, finalPrice]) =>
            createInvoice(order, user, products, shipping, finalPrice)
        )
        .then(invoice => sendEmail(invoice.user.email, invoice));
}

// ✅ さらに改善: async/await
async function processOrderAsync(orderId) {
    const order = await fetchOrder(orderId);
    const [user, products] = await Promise.all([
        fetchUser(order.userId),
        fetchProducts(order.productIds)
    ]);
    const shipping = await calculateShipping(user.address, products);
    const finalPrice = await applyDiscount(user.membership, order.total);
    const invoice = await createInvoice(order, user, products, shipping, finalPrice);
    return sendEmail(user.email, invoice);
}
```

### 7.3 アンチパターン3: this の束縛ミス

```javascript
// ❌ アンチパターン: メソッド内のクロージャで this を見失う
class Timer {
    constructor() {
        this.seconds = 0;
    }

    start() {
        // function キーワードは独自の this を持つ
        setInterval(function() {
            this.seconds++;  // this は Timer ではなく global/undefined
            console.log(this.seconds);  // NaN or エラー
        }, 1000);
    }
}

// ✅ 修正法1: アロー関数を使う（最も推奨）
class TimerFixed1 {
    constructor() {
        this.seconds = 0;
    }

    start() {
        // アロー関数は this を外側のスコープから継承
        setInterval(() => {
            this.seconds++;  // this は TimerFixed1 インスタンス
            console.log(this.seconds);
        }, 1000);
    }
}

// ✅ 修正法2: self/that パターン（ES5 時代の手法）
class TimerFixed2 {
    constructor() {
        this.seconds = 0;
    }

    start() {
        const self = this;  // this をクロージャで捕捉
        setInterval(function() {
            self.seconds++;
            console.log(self.seconds);
        }, 1000);
    }
}

// ✅ 修正法3: bind を使う
class TimerFixed3 {
    constructor() {
        this.seconds = 0;
    }

    start() {
        setInterval(function() {
            this.seconds++;
            console.log(this.seconds);
        }.bind(this), 1000);  // this を明示的に束縛
    }
}
```

---

## 8. パフォーマンス考察

### 8.1 クロージャのオーバーヘッド

```
クロージャのコスト分析:

  ┌─────────────────────────────────────────────────────────┐
  │  コスト要因          │ 影響度 │ 説明                     │
  ├─────────────────────────────────────────────────────────┤
  │  環境オブジェクト生成 │ 小     │ ヒープ割り当て1回       │
  │  変数アクセス         │ 微小   │ 間接参照1回追加         │
  │  GC への負荷          │ 中     │ 参照追跡が増える        │
  │  インライン化阻害     │ 中     │ 最適化が困難になる場合  │
  │  メモリ使用量         │ 変動   │ 捕捉する変数量に依存   │
  └─────────────────────────────────────────────────────────┘

  一般的な指針:
  - 通常のアプリケーションコード: パフォーマンス影響は無視できる
  - ホットループ内（1秒間に数百万回実行）: クロージャ生成を避ける
  - 大量のクロージャを配列に格納: メモリ使用量に注意
```

```javascript
// パフォーマンスを意識したクロージャの使い方

// ❌ ホットループ内でクロージャを毎回生成
function processItemsBad(items, threshold) {
    return items.map(item => {
        // このクロージャは毎回新しい環境を生成する（が、実際は最適化される場合が多い）
        const check = (val) => val > threshold;  // 不要なクロージャ
        return check(item.value) ? item : null;
    }).filter(Boolean);
}

// ✅ クロージャを一度だけ生成
function processItemsGood(items, threshold) {
    const check = (val) => val > threshold;  // 1回だけ生成
    return items.map(item =>
        check(item.value) ? item : null
    ).filter(Boolean);
}

// ✅ さらに良い: そもそもクロージャが不要なら使わない
function processItemsBest(items, threshold) {
    return items.filter(item => item.value > threshold);
}
```

### 8.2 V8 エンジンの最適化

現代の JavaScript エンジン（V8、SpiderMonkey 等）は、クロージャに対して多くの最適化を行う。

```
V8 のクロージャ最適化:

  1. コンテキスト最適化（Context Optimization）
     - 使われない変数は捕捉しない
     - 例: 10個の変数があるスコープで1個だけ使う場合、
       その1個だけの環境が作られる

  2. インライン化（Inlining）
     - 小さなクロージャは呼び出し元に展開される
     - 環境オブジェクトの生成自体が省略される

  3. エスケープ解析（Escape Analysis）
     - クロージャが関数の外に出ない場合、
       ヒープ割り当てをスタック割り当てに変換

  4. Hidden Class による最適化
     - 同じ形状のクロージャオブジェクトは
       同じ Hidden Class を共有

  ※ ただし eval() や debugger 使用時はこれらの最適化が
    無効化されることがある
```

---

## 9. 演習問題

### 演習1: 基礎レベル

**問題1-1: カウンタの実装**

以下の仕様を満たす `createCounter` 関数を JavaScript で実装せよ。

```javascript
// 仕様:
// - createCounter(initial) は初期値 initial のカウンタを返す
// - increment() は値を1増やして返す
// - decrement() は値を1減らして返す
// - reset() は初期値に戻して返す
// - getCount() は現在の値を返す

// 使用例:
const c = createCounter(10);
c.increment();  // 11
c.increment();  // 12
c.decrement();  // 11
c.reset();      // 10
c.getCount();   // 10
```

<details>
<summary>解答例</summary>

```javascript
function createCounter(initial) {
    let count = initial;
    return {
        increment() { return ++count; },
        decrement() { return --count; },
        reset() { count = initial; return count; },
        getCount() { return count; }
    };
}
```
</details>

**問題1-2: once 関数**

一度だけ実行可能な関数を作る `once` 関数を実装せよ。

```javascript
// 仕様:
// - once(fn) は fn を一度だけ実行する関数を返す
// - 2回目以降の呼び出しは最初の結果を返す

const expensiveCalc = once(() => {
    console.log("Computing...");
    return 42;
});

expensiveCalc();  // "Computing..." → 42
expensiveCalc();  // 42（再計算なし、ログも出ない）
```

<details>
<summary>解答例</summary>

```javascript
function once(fn) {
    let called = false;
    let result;
    return function(...args) {
        if (!called) {
            result = fn.apply(this, args);
            called = true;
        }
        return result;
    };
}
```
</details>

### 演習2: 中級レベル

**問題2-1: パイプライン関数**

関数を左から右に合成する `pipe` と、右から左に合成する `compose` を実装せよ。

```javascript
// 仕様:
// - pipe(f, g, h)(x) は h(g(f(x))) と同等
// - compose(f, g, h)(x) は f(g(h(x))) と同等

const double = x => x * 2;
const addOne = x => x + 1;
const square = x => x * x;

pipe(double, addOne, square)(3);     // square(addOne(double(3))) = square(7) = 49
compose(double, addOne, square)(3);  // double(addOne(square(3))) = double(10) = 20
```

<details>
<summary>解答例</summary>

```javascript
function pipe(...fns) {
    return function(x) {
        return fns.reduce((acc, fn) => fn(acc), x);
    };
}

function compose(...fns) {
    return function(x) {
        return fns.reduceRight((acc, fn) => fn(acc), x);
    };
}
```
</details>

**問題2-2: LRU キャッシュ付きメモ化**

最大 `capacity` 個のキャッシュエントリを保持するメモ化関数を実装せよ。キャッシュが満杯の場合、最も古い（Least Recently Used）エントリを削除する。

```javascript
// 仕様:
// - memoizeLRU(fn, capacity) を実装
// - キャッシュ容量を超えたら最も古いエントリを削除
// - 既存エントリがアクセスされたら最新に更新

const cachedFn = memoizeLRU((x) => x * x, 3);
cachedFn(1);  // 計算: 1  → キャッシュ: [1]
cachedFn(2);  // 計算: 4  → キャッシュ: [1, 2]
cachedFn(3);  // 計算: 9  → キャッシュ: [1, 2, 3]
cachedFn(1);  // キャッシュヒット → キャッシュ: [2, 3, 1]（1が最新に）
cachedFn(4);  // 計算: 16 → キャッシュ: [3, 1, 4]（2が削除）
cachedFn(2);  // 計算: 4  → キャッシュ: [1, 4, 2]（3が削除、2は再計算）
```

<details>
<summary>解答例</summary>

```javascript
function memoizeLRU(fn, capacity) {
    // Map は挿入順序を保持するため LRU に適している
    const cache = new Map();

    return function(...args) {
        const key = JSON.stringify(args);

        if (cache.has(key)) {
            // アクセスされたエントリを最新に移動
            const value = cache.get(key);
            cache.delete(key);
            cache.set(key, value);
            return value;
        }

        const result = fn.apply(this, args);

        if (cache.size >= capacity) {
            // 最も古い（最初の）エントリを削除
            const oldestKey = cache.keys().next().value;
            cache.delete(oldestKey);
        }

        cache.set(key, result);
        return result;
    };
}
```
</details>

### 演習3: 上級レベル

**問題3-1: Observable パターン**

クロージャを使ってリアクティブなデータバインディングを実装せよ。

```javascript
// 仕様:
// - observable(initialValue) は { get, set, subscribe } を返す
// - subscribe(callback) は値が変更されるたびに callback を呼ぶ
// - subscribe は unsubscribe 関数を返す
// - computed(observables, computeFn) は派生値を作る

const firstName = observable("John");
const lastName = observable("Doe");
const fullName = computed(
    [firstName, lastName],
    (first, last) => `${first} ${last}`
);

fullName.subscribe(name => console.log(`Name: ${name}`));

firstName.set("Jane");  // "Name: Jane Doe"
lastName.set("Smith");  // "Name: Jane Smith"
```

<details>
<summary>解答例</summary>

```javascript
function observable(initialValue) {
    let value = initialValue;
    const subscribers = new Set();

    return {
        get() {
            return value;
        },
        set(newValue) {
            if (value !== newValue) {
                value = newValue;
                subscribers.forEach(cb => cb(value));
            }
        },
        subscribe(callback) {
            subscribers.add(callback);
            // 登録時に現在の値を通知（オプション）
            callback(value);
            // unsubscribe 関数を返す
            return () => subscribers.delete(callback);
        }
    };
}

function computed(observables, computeFn) {
    const result = observable(
        computeFn(...observables.map(o => o.get()))
    );

    // 依存する observable のいずれかが変更されたら再計算
    observables.forEach(obs => {
        obs.subscribe(() => {
            const newValue = computeFn(...observables.map(o => o.get()));
            result.set(newValue);
        });
    });

    return result;
}
```
</details>

**問題3-2: Rust でのクロージャ型パズル**

以下の Rust コードのコンパイルエラーを修正し、各クロージャが実装するトレイトを答えよ。

```rust
// このコードには複数のコンパイルエラーがある。修正せよ。
fn main() {
    let mut items = vec![1, 2, 3, 4, 5];

    // (a) フィルタ
    let threshold = 3;
    let big_items: Vec<&i32> = items.iter().filter(|x| **x > threshold).collect();

    // (b) 変換と蓄積
    let mut sum = 0;
    let doubled: Vec<i32> = items.iter().map(|x| {
        sum += x;   // エラー箇所
        x * 2
    }).collect();

    // (c) 所有権の移動
    let data = String::from("hello");
    let printer = || println!("{}", data);
    printer();
    println!("{}", data);  // エラー箇所の可能性

    // (d) 関数から返す
    fn make_greeter(name: String) -> impl Fn() {
        || println!("Hello, {}!", name)  // エラー箇所
    }
}
```

<details>
<summary>解答例</summary>

```rust
fn main() {
    let items = vec![1, 2, 3, 4, 5];

    // (a) Fn トレイト: threshold を不変借用
    let threshold = 3;
    let big_items: Vec<&i32> = items.iter().filter(|x| **x > threshold).collect();
    println!("{:?}", big_items);  // [4, 5]

    // (b) FnMut トレイト: sum を可変借用
    let mut sum = 0;
    let doubled: Vec<i32> = items.iter().map(|x| {
        sum += x;  // sum を &mut で借用 → FnMut
        x * 2
    }).collect();
    println!("{:?}, sum={}", doubled, sum);

    // (c) Fn トレイト: data を不変借用（move なし）
    let data = String::from("hello");
    let printer = || println!("{}", data);  // &data を借用
    printer();
    println!("{}", data);  // data はまだ有効

    // (d) move が必要: name の所有権をクロージャに移動
    fn make_greeter(name: String) -> impl Fn() {
        move || println!("Hello, {}!", name)
        // move で name の所有権を移動
        // println! は &name しか使わないので Fn を実装
    }

    let greet = make_greeter(String::from("World"));
    greet();
    greet();  // Fn なので何度でも呼べる
}
```
</details>

---

## 10. FAQ（よくある質問）

### Q1: クロージャとラムダ式は同じものですか？

**A:** 厳密には異なる概念だが、実用上はほぼ同義で使われることが多い。

- **ラムダ式（lambda expression）**: 無名関数を定義するための構文。`(x) => x * 2` や `lambda x: x * 2` など。
- **クロージャ（closure）**: 自由変数を捕捉した関数。ラムダ式であるかどうかに関係ない。

名前付き関数もクロージャになりうるし、ラムダ式でも自由変数を捕捉しなければクロージャではない。ただし、多くの言語でラムダ式はクロージャとして動作するため、混同されがちである。

```javascript
// ラムダ式だがクロージャではない（自由変数がない）
const double = (x) => x * 2;

// ラムダ式でありクロージャでもある（factor を捕捉）
const factor = 3;
const multiply = (x) => x * factor;

// 名前付き関数だがクロージャである（count を捕捉）
function makeCounter() {
    let count = 0;
    function increment() {  // 名前付き関数
        return ++count;     // count を捕捉 → クロージャ
    }
    return increment;
}
```

### Q2: クロージャを使うべき場面とクラスを使うべき場面の判断基準は？

**A:** 以下の基準で判断するとよい。

| 判断基準 | クロージャが適切 | クラスが適切 |
|---------|----------------|-------------|
| 操作の数 | 1-3個の操作 | 4個以上の操作 |
| 状態の複雑さ | 単純（数個の変数） | 複雑（多数のフィールド） |
| 継承の必要性 | 不要 | 必要 |
| テストのしやすさ | 純粋関数的なら容易 | モック/スタブが必要 |
| コールバック | 最適 | 過剰 |
| 設定の注入 | 部分適用で十分 | DI コンテナが望ましい |
| チームの慣習 | 関数型スタイル | OOP スタイル |

```javascript
// クロージャが適切: 単一のコールバック生成
const createHandler = (eventType) => (event) => {
    console.log(`${eventType}: ${event.target.id}`);
};

// クラスが適切: 複雑な状態と多数の操作
class ShoppingCart {
    #items = [];
    #discountRate = 0;

    addItem(item) { /* ... */ }
    removeItem(id) { /* ... */ }
    applyDiscount(rate) { /* ... */ }
    calculateTotal() { /* ... */ }
    checkout() { /* ... */ }
    getItems() { /* ... */ }
    toJSON() { /* ... */ }
}
```

### Q3: クロージャはガベージコレクションにどう影響しますか？

**A:** クロージャが生存している限り、捕捉した変数は GC の対象にならない。これは意図的な動作だが、注意しないとメモリリークの原因になる。

主な対策:
1. **必要な値だけを捕捉する**: 巨大なオブジェクトの一部だけが必要なら、事前に抽出する
2. **不要になったクロージャへの参照を切る**: イベントリスナーの解除、タイマーのクリアなど
3. **WeakRef / WeakMap を活用する**: 強い参照を避けたい場合に使う（JavaScript、Python の weakref）
4. **スコープを最小化する**: クロージャの定義位置をできるだけ内側に移動し、捕捉される変数を減らす

### Q4: async/await とクロージャの関係は？

**A:** `async` 関数は内部的にクロージャとジェネレータ（またはステートマシン）の組み合わせで実現されている。`await` で中断・再開する際、ローカル変数の状態がクロージャのように保存される。

```javascript
// async/await は内部的にクロージャ的な仕組みで変数を保持
async function fetchAndProcess(url) {
    const response = await fetch(url);
    // ↑ ここで中断される
    // response はクロージャ的に保存される

    const data = await response.json();
    // ↑ ここでも中断される
    // response と data の両方が保存されている

    return processData(data);
}
```

### Q5: Rust の move クロージャと通常のクロージャの使い分けは？

**A:** `move` キーワードは、クロージャが捕捉する変数の所有権を強制的にクロージャ内に移動させる。以下の場合に `move` を使う。

1. **クロージャが関数の戻り値になる場合**: ローカル変数への参照が無効になるのを防ぐ
2. **クロージャを別スレッドに送る場合**: `'static` ライフタイムが要求されるため
3. **所有権の明示的な移動が必要な場合**: 元のスコープで変数を使わないことを明示

```rust
use std::thread;

// move が必要: 別スレッドにクロージャを送る
fn spawn_worker(data: Vec<i32>) -> thread::JoinHandle<i32> {
    thread::spawn(move || {
        // move がないと data への参照が無効になる可能性
        data.iter().sum()
    })
}
```

---

## 11. クロージャの歴史と理論的背景

### 11.1 歴史年表

```
クロージャの歴史:

  1936  Alonzo Church がラムダ計算を発表
        → 関数を値として扱う数学的基盤

  1958  Lisp が誕生（John McCarthy）
        → 最初の関数型プログラミング言語
        → ただし初期 Lisp はダイナミックスコープ

  1964  Peter Landin が SECD マシンを提案
        → 「closure」という用語の初出
        → 関数 + 環境の組み合わせを形式化

  1970  Scheme が誕生（Guy Steele, Gerald Sussman）
        → Lisp にレキシカルスコープを導入
        → クロージャの実用化

  1973  ML が誕生（Robin Milner）
        → 型付きの関数型言語でクロージャを採用

  1987  Haskell の設計開始
        → 遅延評価とクロージャの組み合わせ

  1995  JavaScript が誕生（Brendan Eich）
        → Scheme の影響でクロージャを採用
        → Web 開発でクロージャが広く普及

  2004  Groovy（JVM 上のクロージャ）

  2007  C# 3.0 にラムダ式追加

  2010  Rust 開発開始（クロージャ + 所有権）

  2011  C++11 にラムダ式追加

  2014  Java 8 にラムダ式追加
        Swift が誕生（クロージャを中核機能に）

  2015  ES6（JavaScript）でアロー関数追加
```

### 11.2 ラムダ計算との関係

クロージャの理論的基盤はラムダ計算（lambda calculus）にある。ラムダ計算では、すべての計算を「関数の定義」と「関数の適用」だけで表現する。

```
ラムダ計算の基本:

  ラムダ抽象:  λx.M     （x を引数とする関数、本体は M）
  関数適用:    (M N)     （M に N を適用）
  変数:        x         （変数参照）

  例: 加算関数
    λx.λy.(x + y)        二引数関数
    (λx.λy.(x + y)) 3    部分適用 → λy.(3 + y)
    ((λx.λy.(x + y)) 3) 5  完全適用 → 8

  クロージャとの対応:
    λx.λy.(x + y)  →  const add = (x) => (y) => x + y;
    部分適用       →  const add3 = add(3);  // y => 3 + y
    完全適用       →  add3(5);              // 8
```

---

## 12. まとめ

### 12.1 総合比較表

| 言語 | クロージャの捕捉 | 可変捕捉 | 特徴 | 主な用途 |
|------|---------------|---------|------|---------|
| JavaScript | 参照（自動） | 可能 | 最も自然に使える | コールバック、モジュール |
| Python | 参照（自動） | nonlocal 必要 | ループ変数の罠 | デコレータ、高階関数 |
| Rust | 借用/移動 | FnMut で可能 | 所有権と統合、最も安全 | イテレータ、並行処理 |
| Go | 参照（自動） | 可能 | シンプル | goroutine、ミドルウェア |
| Java | 値コピー | 不可（effectively final） | 制限的 | Stream API、コールバック |
| C++ | 値 or 参照（指定） | mutable で可能 | 最も細かい制御 | STL アルゴリズム |
| Swift | 参照（自動） | 可能 | ARC との統合 | 非同期処理、UIイベント |

### 12.2 設計判断のフローチャート

```
クロージャの採用判断:

  状態を持つ振る舞いが必要？
    │
    ├─ No → 通常の関数で十分
    │
    └─ Yes → 操作は何個？
              │
              ├─ 1-2個 → クロージャが最適
              │           例: コールバック、フィルタ条件
              │
              ├─ 3-5個 → クロージャ or 小さなクラス
              │           チームの慣習に合わせる
              │
              └─ 6個以上 → クラスが適切
                            状態管理が複雑すぎる

  追加の考慮事項:
    - テスタビリティ: 純粋関数的に書けるならクロージャ
    - 再利用性: 継承やインターフェースが必要ならクラス
    - パフォーマンス: ホットパスでは不要なクロージャ生成を避ける
    - チームのスキル: 関数型に慣れたチームならクロージャ寄り
```

### 12.3 重要ポイントの復習

1. **クロージャ = 関数 + 環境**: 自由変数への束縛を関数と一緒に保持する仕組み
2. **レキシカルスコープ**: クロージャは定義時のスコープを参照する（呼び出し時ではない）
3. **言語ごとの違い**: 捕捉方式（参照 vs 値 vs 所有権）が言語によって大きく異なる
4. **実践パターン**: メモ化、カリー化、モジュールパターン、デバウンス/スロットル
5. **メモリへの影響**: 不要なクロージャへの参照を切ること、必要な値だけを捕捉すること
6. **双対性**: クロージャとオブジェクトは本質的に同等の表現力を持つ

---

## 次に読むべきガイド

- [[02-higher-order-functions.md]] -- 高階関数とクロージャの組み合わせ
- [[03-recursion.md]] -- 再帰とクロージャによるメモ化の応用
- [[04-lambda-calculus.md]] -- ラムダ計算の理論的基盤

---

## 参考文献

1. Abelson, H. & Sussman, G. J. *Structure and Interpretation of Computer Programs (SICP)*. Ch.3 "Modularity, Objects, and State," MIT Press, 1996. -- クロージャによる状態管理の古典的解説。環境モデルの詳細な説明が含まれる。
2. Klabnik, S. & Nichols, C. *The Rust Programming Language*. Ch.13 "Functional Language Features: Iterators and Closures," No Starch Press, 2023. -- Rust の Fn/FnMut/FnOnce トレイトと所有権システムの統合について。
3. Crockford, D. *JavaScript: The Good Parts*. O'Reilly Media, 2008. -- JavaScript におけるクロージャとモジュールパターンの実践的解説。
4. Landin, P. J. "The Mechanical Evaluation of Expressions." *The Computer Journal*, Vol.6, No.4, pp.308-320, 1964. -- 「closure」という用語が最初に使われた論文。SECD マシンの文脈で環境と関数の組み合わせを形式化。
5. Sussman, G. J. & Steele, G. L. "Scheme: An Interpreter for Extended Lambda Calculus." *MIT AI Memo 349*, 1975. -- Lisp にレキシカルスコープを導入し、クロージャを実用化した画期的な論文。
6. Friedman, D. P. & Wand, M. *Essentials of Programming Languages*. 3rd Edition, MIT Press, 2008. -- プログラミング言語の意味論におけるクロージャの位置づけ。環境渡しインタプリタの構築を通じた理解。

