# クロージャとFnトレイト -- 環境をキャプチャする関数オブジェクト

> クロージャは環境から変数をキャプチャできる匿名関数であり、Fn/FnMut/FnOnceの3つのトレイトによってキャプチャ方式とコンパイラ最適化が決定される。

---

## この章で学ぶこと

1. **クロージャの基本** -- 構文、型推論、環境キャプチャの仕組みを理解する
2. **Fn / FnMut / FnOnce** -- 3つのクロージャトレイトの違いと選択基準を習得する
3. **クロージャの活用** -- イテレータ、コールバック、高階関数での実践パターンを学ぶ

---

## 1. クロージャの基本

### 例1: クロージャの構文

```rust
fn main() {
    // 基本的なクロージャ
    let add = |a, b| a + b;
    println!("{}", add(3, 4)); // 7

    // 型注釈付き
    let add_typed = |a: i32, b: i32| -> i32 { a + b };
    println!("{}", add_typed(3, 4)); // 7

    // 環境変数のキャプチャ
    let multiplier = 3;
    let multiply = |x| x * multiplier; // multiplier をキャプチャ
    println!("{}", multiply(5)); // 15

    // 複数行のクロージャ
    let process = |x: i32| {
        let doubled = x * 2;
        let formatted = format!("結果: {}", doubled);
        formatted
    };
    println!("{}", process(21));
}
```

### クロージャの構文比較

```
  関数:       fn add(x: i32, y: i32) -> i32 { x + y }
  クロージャ: |x: i32, y: i32| -> i32 { x + y }
  省略形:     |x, y| x + y

  ┌──────────────────────────────────────────────────┐
  │ クロージャ vs 関数                                │
  ├──────────────────────────────────────────────────┤
  │ - クロージャは環境から変数をキャプチャできる       │
  │ - クロージャの型は各定義ごとに固有(匿名型)       │
  │ - 関数ポインタ fn(T) -> U への変換は限定的        │
  │ - 型推論が効くので型注釈は省略可能                │
  └──────────────────────────────────────────────────┘
```

---

## 2. キャプチャ方式

### 例2: 3つのキャプチャ方式

```rust
fn main() {
    let name = String::from("Rust");
    let mut count = 0;
    let data = vec![1, 2, 3];

    // 不変参照でキャプチャ (Fn)
    let greet = || println!("Hello, {}!", name);
    greet();
    greet();
    println!("name はまだ使える: {}", name);

    // 可変参照でキャプチャ (FnMut)
    let mut increment = || {
        count += 1;
        println!("count: {}", count);
    };
    increment();
    increment();
    println!("count: {}", count); // 2

    // 所有権をムーブ (FnOnce)
    let consume = move || {
        println!("data を消費: {:?}", data);
        drop(data); // 所有権を持っている
    };
    consume();
    // consume(); // エラー: FnOnce は1回のみ
    // println!("{:?}", data); // エラー: data はムーブ済み
}
```

### キャプチャ階層図

```
  コンパイラは最小限のキャプチャ方式を自動選択する

  ┌─────────────────────────────────────────┐
  │   FnOnce  (全てのクロージャが実装)       │
  │   ┌───────────────────────────────────┐ │
  │   │  FnMut (FnOnce のサブセット)      │ │
  │   │  ┌─────────────────────────────┐  │ │
  │   │  │  Fn (FnMut のサブセット)    │  │ │
  │   │  │                             │  │ │
  │   │  │ 不変参照のみ                │  │ │
  │   │  └─────────────────────────────┘  │ │
  │   │ 可変参照を使用                     │ │
  │   └───────────────────────────────────┘ │
  │ 所有権をムーブ / 値を消費                │
  └─────────────────────────────────────────┘

  Fn ⊂ FnMut ⊂ FnOnce
  Fn を実装する型は FnMut と FnOnce も自動実装
```

---

## 3. Fn / FnMut / FnOnce トレイト

### 例3: 関数引数としてのクロージャ

```rust
// Fn: 何回でも呼べる。環境を変更しない
fn apply_twice<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(x) + f(x)
}

// FnMut: 何回でも呼べる。環境を変更できる
fn apply_mutably<F: FnMut()>(mut f: F) {
    f();
    f();
    f();
}

// FnOnce: 1回だけ呼べる。所有権を消費できる
fn apply_once<F: FnOnce() -> String>(f: F) -> String {
    f()
}

fn main() {
    // Fn
    let double = |x| x * 2;
    println!("{}", apply_twice(double, 5)); // 20

    // FnMut
    let mut total = 0;
    apply_mutably(|| {
        total += 1;
    });
    println!("total: {}", total); // 3

    // FnOnce
    let name = String::from("World");
    let greeting = apply_once(move || format!("Hello, {}!", name));
    println!("{}", greeting);
}
```

### 例4: 戻り値としてのクロージャ

```rust
// impl Fn で返す
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    move |y| x + y
}

// Box<dyn Fn> で返す(動的ディスパッチ)
fn make_operation(op: &str) -> Box<dyn Fn(i32, i32) -> i32> {
    match op {
        "add" => Box::new(|a, b| a + b),
        "mul" => Box::new(|a, b| a * b),
        _     => Box::new(|a, b| a - b),
    }
}

fn main() {
    let add5 = make_adder(5);
    println!("{}", add5(3)); // 8
    println!("{}", add5(7)); // 12

    let op = make_operation("mul");
    println!("{}", op(4, 5)); // 20
}
```

### 例5: イテレータでのクロージャ活用

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // 複合パイプライン
    let result: Vec<String> = numbers
        .iter()
        .filter(|&&x| x % 2 == 0)       // Fn
        .map(|&x| x * x)                 // Fn
        .filter(|&x| x > 10)             // Fn
        .map(|x| format!("値: {}", x))   // FnMut (Stringの生成)
        .collect();

    for s in &result {
        println!("{}", s);
    }

    // fold でステートフルな集約
    let stats = numbers.iter().fold(
        (0, 0, i32::MAX, i32::MIN),
        |(sum, count, min, max), &x| {
            (sum + x, count + 1, min.min(x), max.max(x))
        },
    );
    println!("合計={}, 個数={}, 最小={}, 最大={}", stats.0, stats.1, stats.2, stats.3);
}
```

---

## 4. move クロージャ

### 例6: move キーワード

```rust
use std::thread;

fn main() {
    let message = String::from("Hello from thread!");

    // move で所有権をクロージャに移転
    let handle = thread::spawn(move || {
        println!("{}", message);
    });

    // println!("{}", message); // エラー: message はムーブ済み
    handle.join().unwrap();

    // Copy型は move してもコピーされる
    let x = 42;
    let closure = move || println!("{}", x);
    closure();
    println!("x はまだ使える: {}", x); // OK: i32 は Copy
}
```

---

## 5. 関数ポインタ

```
┌──────────────────────────────────────────────────────┐
│  関数ポインタ fn(T) -> U                              │
├──────────────────────────────────────────────────────┤
│ - 通常の関数を指す固定サイズのポインタ                │
│ - 環境をキャプチャしないクロージャと互換               │
│ - Fn, FnMut, FnOnce を全て実装する                   │
│ - C の関数ポインタとFFIで互換                        │
└──────────────────────────────────────────────────────┘
```

### 例7: 関数ポインタとクロージャの使い分け

```rust
fn double(x: i32) -> i32 {
    x * 2
}

fn apply_fn_ptr(f: fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

fn apply_closure(f: impl Fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

fn main() {
    // 関数ポインタとして渡す
    println!("{}", apply_fn_ptr(double, 5));  // 10
    println!("{}", apply_fn_ptr(|x| x + 1, 5)); // 6 (キャプチャなしクロージャ)

    // クロージャトレイトとして渡す
    let offset = 10;
    println!("{}", apply_closure(|x| x + offset, 5)); // 15
    println!("{}", apply_closure(double, 5)); // 10 (関数もFn実装)
}
```

---

## 6. 比較表

### 6.1 Fn / FnMut / FnOnce の比較

| 特性 | `Fn` | `FnMut` | `FnOnce` |
|------|------|---------|----------|
| 呼び出し回数 | 何回でも | 何回でも | 1回のみ |
| 環境の借用 | &self (不変) | &mut self (可変) | self (ムーブ) |
| キャプチャ | 不変参照 | 可変参照 | 所有権の取得 |
| 上位実装 | FnMut + FnOnce | FnOnce | - |
| 例 | `\|x\| x + captured` | `\|\| count += 1` | `\|\| drop(owned)` |

### 6.2 クロージャ vs 関数ポインタ

| 特性 | クロージャ | 関数ポインタ `fn()` |
|------|----------|------------------|
| 環境キャプチャ | 可能 | 不可能 |
| 型 | 匿名型(各定義ごとに固有) | fn(T) -> U (固定型) |
| サイズ | キャプチャする変数に依存 | ポインタ1つ分 |
| FFI互換 | なし | C関数ポインタと互換 |
| impl Fn(T) | 使用可能 | 使用可能 |
| Box<dyn Fn(T)> | 使用可能 | 使用可能 |

---

## 7. アンチパターン

### アンチパターン1: 過度に制限的なトレイト境界

```rust
// BAD: FnOnce で十分なのに Fn を要求
fn execute<F: Fn()>(f: F) {
    f();
    // f は1回しか呼ばないのに Fn を要求している
}

// GOOD: 最小限のトレイト境界
fn execute_good<F: FnOnce()>(f: F) {
    f();
}
// FnOnce を受け入れれば、Fn と FnMut のクロージャも渡せる
```

### アンチパターン2: 不要な move

```rust
// BAD: Copy型なのに move を意識しすぎ
let x = 42;
let closure = move || println!("{}", x);
// x は Copy なので move しなくても同じ結果

// GOOD: 必要な場合のみ move を使う
// - スレッドに渡す場合
// - 'static 境界が必要な場合
// - 明示的にライフタイムを切り離したい場合
let closure = || println!("{}", x); // i32 は Copy なのでこれで十分
```

---

## 8. FAQ

### Q1: クロージャはなぜ変数に代入できるのに、型名を書けないのですか？

**A:** クロージャの型はコンパイラが生成する匿名型であり、プログラマが名前を指定できません。`impl Fn(i32) -> i32` や `Box<dyn Fn(i32) -> i32>` のようにトレイト経由で参照します。

### Q2: move クロージャで Copy 型をキャプチャした場合はどうなりますか？

**A:** Copy型(i32, bool等)はムーブではなくコピーされます。そのため元の変数は move 後も使用可能です。String や Vec のような非Copy型のみが実際にムーブされます。

### Q3: クロージャを構造体のフィールドにするにはどうしますか？

**A:** ジェネリクスか `Box<dyn Fn>` を使います:
```rust
// ジェネリクス (静的ディスパッチ・高速)
struct Callback<F: Fn(i32)> {
    func: F,
}

// Box<dyn Fn> (動的ディスパッチ・柔軟)
struct CallbackDyn {
    func: Box<dyn Fn(i32)>,
}
```

---

## 9. まとめ

| 概念 | 要点 |
|------|------|
| クロージャ | 環境をキャプチャする匿名関数 |
| Fn | 不変参照でキャプチャ。何回でも呼べる |
| FnMut | 可変参照でキャプチャ。何回でも呼べる |
| FnOnce | 所有権を取得。1回のみ呼べる |
| move | 所有権を強制的にクロージャに移転 |
| 匿名型 | 各クロージャは固有の型を持つ |
| 関数ポインタ | fn(T)->U。キャプチャなしクロージャと互換 |

---

## 次に読むべきガイド

- [../00-basics/04-collections-iterators.md](../00-basics/04-collections-iterators.md) -- イテレータでのクロージャ活用
- [../02-async/00-async-basics.md](../02-async/00-async-basics.md) -- async クロージャとFuture
- [00-lifetimes.md](00-lifetimes.md) -- クロージャとライフタイムの関係

---

## 参考文献

1. **The Rust Programming Language - Ch.13 Closures** -- https://doc.rust-lang.org/book/ch13-01-closures.html
2. **Rust Reference - Closure Types** -- https://doc.rust-lang.org/reference/types/closure.html
3. **Rust by Example - Closures** -- https://doc.rust-lang.org/rust-by-example/fn/closures.html
