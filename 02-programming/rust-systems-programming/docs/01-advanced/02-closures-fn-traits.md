# クロージャとFnトレイト -- 環境をキャプチャする関数オブジェクト

> クロージャは環境から変数をキャプチャできる匿名関数であり、Fn/FnMut/FnOnceの3つのトレイトによってキャプチャ方式とコンパイラ最適化が決定される。

---

## この章で学ぶこと

1. **クロージャの基本** -- 構文、型推論、環境キャプチャの仕組みを理解する
2. **Fn / FnMut / FnOnce** -- 3つのクロージャトレイトの違いと選択基準を習得する
3. **クロージャの活用** -- イテレータ、コールバック、高階関数での実践パターンを学ぶ
4. **move クロージャ** -- 所有権のムーブとスレッド間でのクロージャ利用を理解する
5. **高度なパターン** -- クロージャのサイズ、動的ディスパッチ、パフォーマンス最適化を学ぶ

---

## 1. クロージャの基本

### 1.1 クロージャとは何か

クロージャとは、定義されたスコープの環境から変数をキャプチャ(捕捉)できる匿名関数である。通常の関数 `fn` とは異なり、外側のスコープの変数に直接アクセスできる。

クロージャの特徴:
- 環境から変数をキャプチャできる
- 各クロージャはコンパイラが生成する固有の匿名型を持つ
- 型推論が効くため、引数と戻り値の型注釈は省略可能
- Fn / FnMut / FnOnce のいずれかのトレイトを実装する

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

    // 引数なしのクロージャ
    let greeting = || println!("こんにちは!");
    greeting();

    // キャプチャした変数を使う引数なしクロージャ
    let name = String::from("Rust");
    let greet = || println!("Hello, {}!", name);
    greet();
    greet();  // 何回でも呼べる
    println!("name はまだ使える: {}", name);
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
  │ - クロージャは構造体に変換される(ゼロコスト)      │
  └──────────────────────────────────────────────────┘
```

### 1.2 クロージャの内部構造

コンパイラはクロージャを匿名の構造体に変換する。キャプチャした変数がフィールドとなり、Fn/FnMut/FnOnce のメソッドが実装される。

```rust
// このクロージャ:
let x = 42;
let y = String::from("hello");
let closure = |a: i32| a + x;

// コンパイラが内部的に生成するイメージ:
// struct __Closure_1<'a> {
//     x: &'a i32,  // 不変参照でキャプチャ
// }
// impl<'a> Fn(i32) -> i32 for __Closure_1<'a> {
//     fn call(&self, a: i32) -> i32 {
//         a + *self.x
//     }
// }

fn main() {
    let x = 42;
    let closure = |a: i32| a + x;

    // クロージャのサイズを確認
    println!("クロージャのサイズ: {} bytes",
        std::mem::size_of_val(&closure));
    // x を不変参照でキャプチャ → ポインタ1つ分 = 8 bytes

    // キャプチャなしクロージャのサイズ
    let no_capture = |a: i32, b: i32| a + b;
    println!("キャプチャなしクロージャ: {} bytes",
        std::mem::size_of_val(&no_capture));
    // 0 bytes (キャプチャなし → ゼロサイズ型)

    println!("closure(8) = {}", closure(8)); // 50
}
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

### 例3: キャプチャ方式の自動選択

```rust
fn main() {
    let s = String::from("hello");

    // ケース1: 読むだけ → Fn (不変参照でキャプチャ)
    let read_only = || println!("{}", s);
    read_only();
    read_only();
    println!("s はまだ使える: {}", s);

    // ケース2: 変更する → FnMut (可変参照でキャプチャ)
    let mut numbers = vec![1, 2, 3];
    let mut add_number = || numbers.push(4);
    add_number();
    // println!("{:?}", numbers); // エラー: numbers は可変借用中
    drop(add_number); // クロージャをドロップして借用を解放
    println!("{:?}", numbers); // [1, 2, 3, 4]

    // ケース3: 所有権を消費 → FnOnce
    let s2 = String::from("world");
    let consume = || {
        let taken = s2; // 所有権をムーブ
        println!("消費: {}", taken);
    };
    consume();
    // consume(); // エラー: 所有権はすでに消費された

    // ケース4: 部分的なキャプチャ (Rust 2021+)
    struct Point {
        x: f64,
        y: f64,
    }

    let p = Point { x: 1.0, y: 2.0 };
    let get_x = || p.x; // x フィールドだけキャプチャ
    println!("x = {}", get_x());
    println!("y = {}", p.y); // y は借用されていないので使える
}
```

---

## 3. Fn / FnMut / FnOnce トレイト

### 3.1 トレイトの定義

```rust
// 標準ライブラリでの定義(簡略化)

// FnOnce: 1回だけ呼べる (selfを消費)
// trait FnOnce<Args> {
//     type Output;
//     fn call_once(self, args: Args) -> Self::Output;
// }

// FnMut: 何回でも呼べるが環境を変更できる (&mut selfを取る)
// trait FnMut<Args>: FnOnce<Args> {
//     fn call_mut(&mut self, args: Args) -> Self::Output;
// }

// Fn: 何回でも呼べて環境を変更しない (&selfを取る)
// trait Fn<Args>: FnMut<Args> {
//     fn call(&self, args: Args) -> Self::Output;
// }
```

### 例4: 関数引数としてのクロージャ

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

### 例5: トレイト境界の選択指針

```rust
// 最も制限が緩い境界を選ぶべき
// FnOnce > FnMut > Fn (制限の緩い順)

// 1回しか呼ばないなら FnOnce で十分
fn run_once<F: FnOnce()>(f: F) {
    f();
}

// 複数回呼ぶが変更を許容するなら FnMut
fn run_multiple<F: FnMut()>(mut f: F) {
    for _ in 0..5 {
        f();
    }
}

// 並行実行するなど不変性が必要なら Fn
fn run_parallel<F: Fn() + Send + Sync + 'static>(f: F) {
    let f = std::sync::Arc::new(f);
    let mut handles = vec![];
    for _ in 0..3 {
        let f = std::sync::Arc::clone(&f);
        handles.push(std::thread::spawn(move || {
            f();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

fn main() {
    // FnOnce: あらゆるクロージャを受け入れられる
    let s = String::from("hello");
    run_once(|| {
        println!("{}", s);
        drop(s); // 所有権を消費
    });

    // FnMut: Fn と FnMut のクロージャを受け入れられる
    let mut count = 0;
    run_multiple(|| {
        count += 1;
        println!("count: {}", count);
    });

    // Fn: Fn のクロージャのみ
    let message = String::from("Hello from thread!");
    run_parallel(move || {
        println!("{}", message);
    });
}
```

### 例6: 戻り値としてのクロージャ

```rust
// impl Fn で返す (静的ディスパッチ)
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    move |y| x + y
}

// Box<dyn Fn> で返す (動的ディスパッチ)
fn make_operation(op: &str) -> Box<dyn Fn(i32, i32) -> i32> {
    match op {
        "add" => Box::new(|a, b| a + b),
        "mul" => Box::new(|a, b| a * b),
        "sub" => Box::new(|a, b| a - b),
        _     => Box::new(|a, _b| a),
    }
}

// impl FnMut で返す
fn make_counter(start: i32) -> impl FnMut() -> i32 {
    let mut count = start;
    move || {
        count += 1;
        count
    }
}

// impl FnOnce で返す
fn make_greeting(name: String) -> impl FnOnce() -> String {
    move || format!("Hello, {}! (消費済み)", name)
}

fn main() {
    // impl Fn
    let add5 = make_adder(5);
    println!("{}", add5(3)); // 8
    println!("{}", add5(7)); // 12

    // Box<dyn Fn>
    let ops: Vec<(&str, Box<dyn Fn(i32, i32) -> i32>)> = vec![
        ("add", make_operation("add")),
        ("mul", make_operation("mul")),
        ("sub", make_operation("sub")),
    ];
    for (name, op) in &ops {
        println!("{}: {} {} = {}", name, 10, 3, op(10, 3));
    }

    // impl FnMut
    let mut counter = make_counter(0);
    println!("{}", counter()); // 1
    println!("{}", counter()); // 2
    println!("{}", counter()); // 3

    // impl FnOnce
    let greet = make_greeting("Rust".to_string());
    println!("{}", greet());
    // println!("{}", greet()); // エラー: FnOnce は1回のみ
}
```

---

## 4. クロージャとイテレータ

### 例7: イテレータでのクロージャ活用

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // 複合パイプライン
    let result: Vec<String> = numbers
        .iter()
        .filter(|&&x| x % 2 == 0)       // Fn: 偶数のみ
        .map(|&x| x * x)                 // Fn: 二乗
        .filter(|&x| x > 10)             // Fn: 10より大きい
        .map(|x| format!("値: {}", x))   // Fn: 文字列化
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

### 例8: イテレータアダプタの実践パターン

```rust
fn main() {
    // チェーンされたイテレータ処理
    let words = vec!["hello", "world", "rust", "programming"];

    // map + filter + collect
    let long_upper: Vec<String> = words
        .iter()
        .filter(|w| w.len() > 4)
        .map(|w| w.to_uppercase())
        .collect();
    println!("長い単語(大文字): {:?}", long_upper);

    // enumerate + for_each
    words.iter().enumerate().for_each(|(i, w)| {
        println!("  [{}] {}", i, w);
    });

    // flat_map: ネストしたイテレータを平坦化
    let sentences = vec!["hello world", "rust is great"];
    let all_words: Vec<&str> = sentences
        .iter()
        .flat_map(|s| s.split_whitespace())
        .collect();
    println!("全単語: {:?}", all_words);

    // scan: 状態を持つ変換
    let running_sum: Vec<i32> = vec![1, 2, 3, 4, 5]
        .iter()
        .scan(0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    println!("累積和: {:?}", running_sum); // [1, 3, 6, 10, 15]

    // take_while / skip_while
    let data = vec![1, 2, 3, 10, 20, 30, 1, 2];
    let before_ten: Vec<&i32> = data.iter().take_while(|&&x| x < 10).collect();
    println!("10未満のプレフィクス: {:?}", before_ten); // [1, 2, 3]

    // zip で2つのイテレータを合体
    let keys = vec!["name", "age", "city"];
    let values = vec!["Alice", "30", "Tokyo"];
    let pairs: Vec<(&&str, &&str)> = keys.iter().zip(values.iter()).collect();
    println!("ペア: {:?}", pairs);

    // partition: 条件で2つに分割
    let numbers: Vec<i32> = (1..=10).collect();
    let (evens, odds): (Vec<i32>, Vec<i32>) = numbers
        .into_iter()
        .partition(|x| x % 2 == 0);
    println!("偶数: {:?}", evens);
    println!("奇数: {:?}", odds);

    // window/chunks 的な処理
    let data2 = vec![1, 2, 3, 4, 5, 6];
    let chunks: Vec<&[i32]> = data2.chunks(2).collect();
    println!("チャンク: {:?}", chunks); // [[1, 2], [3, 4], [5, 6]]

    let windows: Vec<&[i32]> = data2.windows(3).collect();
    println!("ウィンドウ: {:?}", windows); // [[1,2,3], [2,3,4], [3,4,5], [4,5,6]]
}
```

### 例9: カスタムイテレータとクロージャ

```rust
struct FibIterator {
    a: u64,
    b: u64,
}

impl FibIterator {
    fn new() -> Self {
        FibIterator { a: 0, b: 1 }
    }
}

impl Iterator for FibIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.a;
        let new_b = self.a + self.b;
        self.a = self.b;
        self.b = new_b;
        Some(result)
    }
}

// クロージャベースのイテレータ生成
fn generate<T, F: FnMut() -> Option<T>>(f: F) -> impl Iterator<Item = T> {
    std::iter::from_fn(f)
}

fn main() {
    // Fibonacci のフィルタリング
    let even_fibs: Vec<u64> = FibIterator::new()
        .filter(|&x| x % 2 == 0)
        .take(10)
        .collect();
    println!("偶数フィボナッチ: {:?}", even_fibs);

    // クロージャベースのイテレータ
    let mut n = 0;
    let squares = generate(move || {
        if n >= 10 {
            None
        } else {
            n += 1;
            Some(n * n)
        }
    });
    let result: Vec<i32> = squares.collect();
    println!("二乗数: {:?}", result);

    // std::iter::successors で再帰的な列を生成
    let powers_of_two: Vec<u64> = std::iter::successors(Some(1u64), |&prev| {
        prev.checked_mul(2) // オーバーフローしたら None を返す
    })
    .take(20)
    .collect();
    println!("2のべき乗: {:?}", powers_of_two);
}
```

---

## 5. move クロージャ

### 例10: move キーワード

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

### 例11: move とライフタイムの関係

```rust
use std::thread;

// move が必要なケース: スレッドに渡す場合
fn spawn_with_data() {
    let data = vec![1, 2, 3, 4, 5];

    // move なしだと data のライフタイムが不足
    // thread::spawn(|| { println!("{:?}", data); }); // エラー

    // move で所有権を転送
    let handle = thread::spawn(move || {
        let sum: i32 = data.iter().sum();
        println!("スレッド内の合計: {}", sum);
    });
    handle.join().unwrap();
}

// move が必要なケース: 関数から返す場合
fn create_closure() -> impl Fn() -> String {
    let prefix = String::from("Hello");
    // move なしだと prefix のライフタイムが不足
    move || format!("{}, World!", prefix)
}

// 部分的なムーブ
fn partial_move() {
    let name = String::from("Rust");
    let version = 2021;

    // name はムーブ、version はコピー
    let closure = move || {
        println!("{} {}", name, version);
    };
    closure();

    // name は使えない (String はムーブ)
    // println!("{}", name); // エラー

    // version は使える (i32 は Copy)
    println!("version: {}", version);
}

fn main() {
    spawn_with_data();

    let greet = create_closure();
    println!("{}", greet());
    println!("{}", greet()); // 何回でも呼べる

    partial_move();
}
```

---

## 6. 関数ポインタ

```
┌──────────────────────────────────────────────────────┐
│  関数ポインタ fn(T) -> U                              │
├──────────────────────────────────────────────────────┤
│ - 通常の関数を指す固定サイズのポインタ                │
│ - 環境をキャプチャしないクロージャと互換               │
│ - Fn, FnMut, FnOnce を全て実装する                   │
│ - C の関数ポインタとFFIで互換                        │
│ - サイズは常にポインタ1つ分 (8 bytes on 64-bit)      │
└──────────────────────────────────────────────────────┘
```

### 例12: 関数ポインタとクロージャの使い分け

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

    // 関数ポインタの配列
    let operations: Vec<fn(i32) -> i32> = vec![
        |x| x + 1,
        |x| x * 2,
        |x| x * x,
    ];

    for (i, op) in operations.iter().enumerate() {
        println!("op[{}](5) = {}", i, op(5));
    }
}
```

### 例13: 関数ポインタとタプル構造体

```rust
// タプル構造体のコンストラクタは関数ポインタとして使える
#[derive(Debug)]
struct Wrapper(i32);

#[derive(Debug)]
enum Color {
    Red(u8),
    Green(u8),
    Blue(u8),
}

fn main() {
    // 構造体コンストラクタを map に渡す
    let numbers = vec![1, 2, 3];
    let wrapped: Vec<Wrapper> = numbers.into_iter().map(Wrapper).collect();
    println!("{:?}", wrapped); // [Wrapper(1), Wrapper(2), Wrapper(3)]

    // 列挙体バリアントのコンストラクタ
    let values: Vec<u8> = vec![255, 128, 64];
    let reds: Vec<Color> = values.into_iter().map(Color::Red).collect();
    println!("{:?}", reds);

    // Option::Some も関数ポインタ
    let nums = vec![1, 2, 3];
    let options: Vec<Option<i32>> = nums.into_iter().map(Some).collect();
    println!("{:?}", options); // [Some(1), Some(2), Some(3)]
}
```

---

## 7. 高度なクロージャパターン

### 例14: コールバックパターン

```rust
type Callback = Box<dyn Fn(&str)>;

struct EventEmitter {
    listeners: std::collections::HashMap<String, Vec<Callback>>,
}

impl EventEmitter {
    fn new() -> Self {
        EventEmitter {
            listeners: std::collections::HashMap::new(),
        }
    }

    fn on(&mut self, event: &str, callback: impl Fn(&str) + 'static) {
        self.listeners
            .entry(event.to_string())
            .or_insert_with(Vec::new)
            .push(Box::new(callback));
    }

    fn emit(&self, event: &str, data: &str) {
        if let Some(callbacks) = self.listeners.get(event) {
            for callback in callbacks {
                callback(data);
            }
        }
    }
}

fn main() {
    let mut emitter = EventEmitter::new();

    // コールバックの登録
    emitter.on("click", |data| {
        println!("クリックイベント: {}", data);
    });

    emitter.on("click", |data| {
        println!("別のクリックハンドラ: {}", data);
    });

    emitter.on("submit", |data| {
        println!("送信イベント: {}", data);
    });

    // イベントの発火
    emitter.emit("click", "ボタンA");
    emitter.emit("submit", "フォーム1");
    emitter.emit("click", "ボタンB");
}
```

### 例15: ミドルウェアパターン

```rust
type Middleware = Box<dyn Fn(&str) -> String>;

struct Pipeline {
    middlewares: Vec<Middleware>,
}

impl Pipeline {
    fn new() -> Self {
        Pipeline {
            middlewares: Vec::new(),
        }
    }

    fn add(&mut self, middleware: impl Fn(&str) -> String + 'static) {
        self.middlewares.push(Box::new(middleware));
    }

    fn execute(&self, input: &str) -> String {
        let mut result = input.to_string();
        for middleware in &self.middlewares {
            result = middleware(&result);
        }
        result
    }
}

fn main() {
    let mut pipeline = Pipeline::new();

    // ミドルウェアの追加
    pipeline.add(|s| {
        println!("  [1] トリミング");
        s.trim().to_string()
    });

    pipeline.add(|s| {
        println!("  [2] 大文字変換");
        s.to_uppercase()
    });

    pipeline.add(|s| {
        println!("  [3] プレフィクス追加");
        format!("[PROCESSED] {}", s)
    });

    let input = "  hello world  ";
    println!("入力: '{}'", input);
    let output = pipeline.execute(input);
    println!("出力: '{}'", output);
}
```

### 例16: ビルダーパターンとクロージャ

```rust
struct QueryBuilder {
    table: String,
    conditions: Vec<Box<dyn Fn(&str) -> bool>>,
    transforms: Vec<Box<dyn Fn(String) -> String>>,
    limit: Option<usize>,
}

impl QueryBuilder {
    fn new(table: &str) -> Self {
        QueryBuilder {
            table: table.to_string(),
            conditions: Vec::new(),
            transforms: Vec::new(),
            limit: None,
        }
    }

    fn where_clause(mut self, condition: impl Fn(&str) -> bool + 'static) -> Self {
        self.conditions.push(Box::new(condition));
        self
    }

    fn transform(mut self, transform: impl Fn(String) -> String + 'static) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    fn execute(&self, data: &[&str]) -> Vec<String> {
        let mut results: Vec<String> = data
            .iter()
            .filter(|item| {
                self.conditions.iter().all(|cond| cond(item))
            })
            .map(|item| {
                let mut result = item.to_string();
                for transform in &self.transforms {
                    result = transform(result);
                }
                result
            })
            .collect();

        if let Some(limit) = self.limit {
            results.truncate(limit);
        }

        results
    }
}

fn main() {
    let data = vec![
        "Alice", "Bob", "Charlie", "David", "Eve",
        "Frank", "Grace", "Heidi", "Ivan", "Judy",
    ];

    let results = QueryBuilder::new("users")
        .where_clause(|name| name.len() > 3)
        .where_clause(|name| !name.starts_with('D'))
        .transform(|name| name.to_uppercase())
        .transform(|name| format!("USER:{}", name))
        .limit(5)
        .execute(&data);

    println!("クエリ結果:");
    for r in &results {
        println!("  {}", r);
    }
}
```

### 例17: メモ化パターン

```rust
use std::collections::HashMap;

struct Memoize<F, K, V>
where
    F: Fn(K) -> V,
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    func: F,
    cache: HashMap<K, V>,
}

impl<F, K, V> Memoize<F, K, V>
where
    F: Fn(K) -> V,
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    fn new(func: F) -> Self {
        Memoize {
            func,
            cache: HashMap::new(),
        }
    }

    fn call(&mut self, arg: K) -> V {
        if let Some(cached) = self.cache.get(&arg) {
            return cached.clone();
        }
        let result = (self.func)(arg.clone());
        self.cache.insert(arg, result.clone());
        result
    }

    fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

fn main() {
    // フィボナッチ数の計算をメモ化
    let mut fib = Memoize::new(|n: u64| -> u64 {
        // 注意: この素朴な再帰はメモ化されない(外側のcacheを使えない)
        // 実用ではHashMapを直接使う方がよい
        if n <= 1 {
            n
        } else {
            // 簡易版: 反復的に計算
            let mut a = 0u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let c = a + b;
                a = b;
                b = c;
            }
            b
        }
    });

    println!("fib(10) = {}", fib.call(10));
    println!("fib(20) = {}", fib.call(20));
    println!("fib(10) = {} (キャッシュヒット)", fib.call(10));
    println!("キャッシュサイズ: {}", fib.cache_size());

    // 文字列処理のメモ化
    let mut expensive_transform = Memoize::new(|s: String| -> String {
        println!("  [計算中] '{}'", s);
        std::thread::sleep(std::time::Duration::from_millis(10));
        s.chars().rev().collect::<String>().to_uppercase()
    });

    println!("{}", expensive_transform.call("hello".to_string()));
    println!("{}", expensive_transform.call("world".to_string()));
    println!("{}", expensive_transform.call("hello".to_string())); // キャッシュヒット
}
```

---

## 8. クロージャのサイズとパフォーマンス

### 8.1 静的ディスパッチ vs 動的ディスパッチ

```
┌──────────────────────────────────────────────────────┐
│  静的ディスパッチ (impl Fn / ジェネリクス)            │
│  - コンパイル時にインライン展開可能                   │
│  - ゼロコスト抽象化                                  │
│  - バイナリサイズが増加(単相化のため)                │
│                                                      │
│  動的ディスパッチ (Box<dyn Fn> / &dyn Fn)            │
│  - 実行時にvtableを経由して呼び出し                  │
│  - わずかなオーバーヘッド(vtable参照 + 間接呼出し)   │
│  - バイナリサイズは小さい                             │
│  - 異なる型のクロージャを1つのコレクションに格納可能  │
└──────────────────────────────────────────────────────┘
```

```rust
// 静的ディスパッチ: 型ごとにコードが生成される
fn apply_static<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(x) // インライン展開される可能性が高い
}

// 動的ディスパッチ: vtable を経由
fn apply_dynamic(f: &dyn Fn(i32) -> i32, x: i32) -> i32 {
    f(x) // 間接呼出し
}

fn main() {
    let double = |x: i32| x * 2;
    let triple = |x: i32| x * 3;

    // 静的ディスパッチ
    println!("{}", apply_static(double, 5));
    println!("{}", apply_static(triple, 5));

    // 動的ディスパッチ: 異なるクロージャを統一的に扱える
    let operations: Vec<Box<dyn Fn(i32) -> i32>> = vec![
        Box::new(|x| x + 1),
        Box::new(|x| x * 2),
        Box::new(|x| x * x),
    ];

    for (i, op) in operations.iter().enumerate() {
        println!("op[{}](5) = {}", i, op(5));
    }

    // クロージャのサイズ確認
    let x = 42i32;
    let y = String::from("hello");

    let c1 = || {};                            // キャプチャなし
    let c2 = || println!("{}", x);            // i32 参照
    let c3 = || println!("{}", y);            // String 参照
    let c4 = move || println!("{}", x);       // i32 コピー
    let c5 = move || println!("{}", y.len()); // String ムーブ

    println!("c1 (キャプチャなし): {} bytes", std::mem::size_of_val(&c1));
    println!("c2 (i32参照): {} bytes", std::mem::size_of_val(&c2));
    println!("c3 (String参照): {} bytes", std::mem::size_of_val(&c3));
    println!("c4 (i32コピー): {} bytes", std::mem::size_of_val(&c4));
    // c5 は y をムーブしているのでここでは使えない
}
```

---

## 9. クロージャと構造体

### 例18: クロージャをフィールドに持つ構造体

```rust
// ジェネリクス版 (静的ディスパッチ・高速)
struct Processor<F: Fn(i32) -> i32> {
    transform: F,
    name: String,
}

impl<F: Fn(i32) -> i32> Processor<F> {
    fn new(name: &str, transform: F) -> Self {
        Processor {
            transform,
            name: name.to_string(),
        }
    }

    fn process(&self, value: i32) -> i32 {
        (self.transform)(value)
    }
}

// Box<dyn Fn> 版 (動的ディスパッチ・柔軟)
struct DynProcessor {
    transform: Box<dyn Fn(i32) -> i32>,
    name: String,
}

impl DynProcessor {
    fn new(name: &str, transform: impl Fn(i32) -> i32 + 'static) -> Self {
        DynProcessor {
            transform: Box::new(transform),
            name: name.to_string(),
        }
    }

    fn process(&self, value: i32) -> i32 {
        (self.transform)(value)
    }
}

fn main() {
    // ジェネリクス版
    let doubler = Processor::new("doubler", |x| x * 2);
    println!("{}: {} → {}", doubler.name, 5, doubler.process(5));

    // Box<dyn Fn> 版: 異なるクロージャを同じ型として扱える
    let processors: Vec<DynProcessor> = vec![
        DynProcessor::new("double", |x| x * 2),
        DynProcessor::new("square", |x| x * x),
        DynProcessor::new("negate", |x| -x),
    ];

    for p in &processors {
        println!("{}: 5 → {}", p.name, p.process(5));
    }
}
```

### 例19: Option/Result メソッドとクロージャ

```rust
fn main() {
    // Option のクロージャメソッド
    let some_value: Option<i32> = Some(42);
    let none_value: Option<i32> = None;

    // map: Some の中身を変換
    let doubled = some_value.map(|x| x * 2);
    println!("map: {:?}", doubled); // Some(84)

    // and_then (flat_map): ネストした Option を平坦化
    let result = some_value.and_then(|x| {
        if x > 0 { Some(x.to_string()) } else { None }
    });
    println!("and_then: {:?}", result);

    // unwrap_or_else: None の場合にクロージャで代替値を生成
    let value = none_value.unwrap_or_else(|| {
        println!("  デフォルト値を計算中...");
        99
    });
    println!("unwrap_or_else: {}", value);

    // filter: 条件を満たさなければ None
    let filtered = some_value.filter(|&x| x > 50);
    println!("filter(>50): {:?}", filtered); // None

    // Result のクロージャメソッド
    let ok_val: Result<i32, String> = Ok(42);
    let err_val: Result<i32, String> = Err("エラー".to_string());

    // map / map_err
    let mapped = ok_val.map(|x| x * 2);
    println!("Ok.map: {:?}", mapped);

    let mapped_err = err_val.map_err(|e| format!("ラップ: {}", e));
    println!("Err.map_err: {:?}", mapped_err);

    // and_then でエラーチェーンを構築
    fn parse_and_double(s: &str) -> Result<i32, String> {
        s.parse::<i32>()
            .map_err(|e| format!("パースエラー: {}", e))
            .and_then(|n| {
                if n >= 0 {
                    Ok(n * 2)
                } else {
                    Err("負の数は不可".to_string())
                }
            })
    }

    println!("parse_and_double(\"21\"): {:?}", parse_and_double("21"));
    println!("parse_and_double(\"-5\"): {:?}", parse_and_double("-5"));
    println!("parse_and_double(\"abc\"): {:?}", parse_and_double("abc"));
}
```

---

## 10. 比較表

### 10.1 Fn / FnMut / FnOnce の比較

| 特性 | `Fn` | `FnMut` | `FnOnce` |
|------|------|---------|----------|
| 呼び出し回数 | 何回でも | 何回でも | 1回のみ |
| 環境の借用 | &self (不変) | &mut self (可変) | self (ムーブ) |
| キャプチャ | 不変参照 | 可変参照 | 所有権の取得 |
| 上位実装 | FnMut + FnOnce | FnOnce | - |
| 例 | `\|x\| x + captured` | `\|\| count += 1` | `\|\| drop(owned)` |
| 引数で受ける時 | `F: Fn(T) -> U` | `mut f: F where F: FnMut(T) -> U` | `F: FnOnce(T) -> U` |

### 10.2 クロージャ vs 関数ポインタ

| 特性 | クロージャ | 関数ポインタ `fn()` |
|------|----------|------------------|
| 環境キャプチャ | 可能 | 不可能 |
| 型 | 匿名型(各定義ごとに固有) | fn(T) -> U (固定型) |
| サイズ | キャプチャする変数に依存 | ポインタ1つ分 (8B) |
| FFI互換 | なし | C関数ポインタと互換 |
| impl Fn(T) | 使用可能 | 使用可能 |
| Box<dyn Fn(T)> | 使用可能 | 使用可能 |
| コレクション格納 | Box<dyn Fn> が必要 | Vec<fn(T) -> U> で直接格納 |

### 10.3 静的 vs 動的ディスパッチ

| 特性 | `impl Fn(T)` / ジェネリクス | `Box<dyn Fn(T)>` / `&dyn Fn(T)` |
|------|---------------------------|--------------------------------|
| ディスパッチ | 静的 (コンパイル時) | 動的 (実行時 vtable) |
| インライン化 | 可能 | 不可能 |
| パフォーマンス | 最高 | わずかなオーバーヘッド |
| バイナリサイズ | 大きい (単相化) | 小さい |
| 異なる型の混在 | 不可能 | 可能 |
| ヒープ確保 | 不要 | Box の場合は必要 |

---

## 11. アンチパターン

### アンチパターン1: 過度に制限的なトレイト境界

```rust
// BAD: FnOnce で十分なのに Fn を要求
fn execute_bad<F: Fn()>(f: F) {
    f();
    // f は1回しか呼ばないのに Fn を要求している
}

// GOOD: 最小限のトレイト境界
fn execute_good<F: FnOnce()>(f: F) {
    f();
}
// FnOnce を受け入れれば、Fn と FnMut のクロージャも渡せる

fn main() {
    let s = String::from("hello");

    // execute_bad だと所有権を消費するクロージャは渡せない
    // execute_bad(|| drop(s)); // エラー

    // execute_good なら全てのクロージャを受け入れられる
    execute_good(|| drop(s)); // OK
}
```

### アンチパターン2: 不要な move

```rust
fn main() {
    // BAD: Copy型なのに move を意識しすぎ
    let x = 42;
    let closure = move || println!("{}", x);
    // x は Copy なので move しなくても同じ結果

    // GOOD: 必要な場合のみ move を使う
    // - スレッドに渡す場合
    // - 'static 境界が必要な場合
    // - 明示的にライフタイムを切り離したい場合
    let closure = || println!("{}", x); // i32 は Copy なのでこれで十分
    closure();
    println!("x = {}", x);
}
```

### アンチパターン3: クロージャの型を具体的に書こうとする

```rust
// BAD: クロージャの型を変数の型注釈で指定しようとする
// let f: |i32| -> i32 = |x| x + 1;  // コンパイルエラー

// GOOD: 型推論に任せる
fn main() {
    let f = |x: i32| -> i32 { x + 1 };

    // 型を明示する必要がある場合は関数ポインタかトレイトオブジェクト
    let fp: fn(i32) -> i32 = |x| x + 1; // キャプチャなし限定
    let boxed: Box<dyn Fn(i32) -> i32> = Box::new(|x| x + 1);

    println!("{}, {}, {}", f(5), fp(5), boxed(5));
}
```

### アンチパターン4: 過度な Box<dyn Fn> の使用

```rust
// BAD: ジェネリクスで済む場面で Box<dyn Fn> を使う
fn apply_bad(f: Box<dyn Fn(i32) -> i32>, x: i32) -> i32 {
    f(x) // 不要なヒープ確保 + 動的ディスパッチ
}

// GOOD: ジェネリクスで静的ディスパッチ
fn apply_good<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
    f(x) // ゼロコスト
}

// Box<dyn Fn> が必要なケース
// - 異なるクロージャをコレクションに格納する場合
// - 構造体のフィールドにジェネリクスを使いたくない場合
// - trait object が必要なAPIの場合

fn main() {
    println!("{}", apply_good(|x| x * 2, 5));
}
```

---

## 12. FAQ

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

### Q4: async クロージャはどのように書きますか？

**A:** Rust 2021時点では `async` クロージャの直接的な構文はまだ安定化されていませんが、async ブロックを返すクロージャで代用できます:
```rust
// async ブロックを返すクロージャ
let fetch = |url: String| async move {
    // 非同期処理
    format!("結果: {}", url)
};

// 型としては impl Fn(String) -> impl Future<Output = String>
// Box<dyn Fn> にする場合は Pin<Box<dyn Future>> を使う
```

### Q5: なぜ `Fn` を実装する型は `FnMut` と `FnOnce` も実装するのですか？

**A:** `Fn` は `&self` で呼ぶので環境を変更しません。`FnMut` は `&mut self` で呼ぶので環境を変更できます。`FnOnce` は `self` で呼ぶので所有権を消費します。不変参照を持つクロージャは当然可変参照として呼ぶこともでき、1回だけ呼ぶこともできます。つまり `Fn ⊂ FnMut ⊂ FnOnce` という包含関係です。

### Q6: クロージャのキャプチャは部分的に行われますか？

**A:** Rust 2021 Edition 以降、クロージャは構造体の個々のフィールドを部分的にキャプチャします。これにより、同じ構造体の異なるフィールドを別々のクロージャでキャプチャできるようになりました:
```rust
struct Data {
    name: String,
    value: i32,
}

let d = Data { name: "test".to_string(), value: 42 };
let c1 = || println!("{}", d.name);   // name だけキャプチャ
let c2 = || println!("{}", d.value);  // value だけキャプチャ
c1();
c2(); // 両方同時に使える (Rust 2021)
```

---

## 13. まとめ

| 概念 | 要点 |
|------|------|
| クロージャ | 環境をキャプチャする匿名関数 |
| Fn | 不変参照でキャプチャ。何回でも呼べる |
| FnMut | 可変参照でキャプチャ。何回でも呼べる |
| FnOnce | 所有権を取得。1回のみ呼べる |
| move | 所有権を強制的にクロージャに移転 |
| 匿名型 | 各クロージャは固有の型を持つ |
| 関数ポインタ | fn(T)->U。キャプチャなしクロージャと互換 |
| 静的ディスパッチ | impl Fn / ジェネリクス。ゼロコスト |
| 動的ディスパッチ | Box<dyn Fn>。異なる型を統一的に扱う |
| 部分キャプチャ | Rust 2021 でフィールド単位のキャプチャ |

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
4. **Rust Edition Guide - Disjoint Capture in Closures** -- https://doc.rust-lang.org/edition-guide/rust-2021/disjoint-capture-in-closures.html
5. **Rust std::ops Module (Fn traits)** -- https://doc.rust-lang.org/std/ops/index.html
