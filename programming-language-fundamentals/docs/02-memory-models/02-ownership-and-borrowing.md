# 所有権と借用（Rust のメモリ管理）

> Rust の所有権システムは「コンパイル時にメモリ安全性を保証する」革命的な仕組み。GC なしでメモリリーク・ダングリングポインタ・データ競合を防ぐ。

## この章で学ぶこと

- [ ] 所有権の3つのルールを理解する
- [ ] 借用（参照）と借用チェッカーの動作を理解する
- [ ] ライフタイムの概念を理解する

---

## 1. 所有権の3つのルール

```
Rust の所有権ルール:

  1. 全ての値には「所有者」（変数）が1つだけ存在する
  2. 所有者がスコープを抜けると、値は自動的に破棄される
  3. 所有権は「移動」（ムーブ）できるが、コピーはデフォルトでは行われない
```

```rust
// ルール1 & 2: 所有者とスコープ
{
    let s = String::from("hello");  // s が "hello" の所有者
    println!("{}", s);               // s を使用
}  // s がスコープを抜ける → "hello" のメモリが自動解放（drop）

// ルール3: ムーブ（所有権の移動）
let s1 = String::from("hello");
let s2 = s1;  // 所有権が s1 → s2 に移動
// println!("{}", s1);  // ❌ コンパイルエラー: s1 はもう使えない
println!("{}", s2);     // ✅ s2 が所有者

// なぜムーブが必要か？
// → 二重解放（double free）を防ぐため
// s1 と s2 が同じメモリを指すと、両方がスコープを抜ける時に
// 同じメモリを2回解放してしまう → ムーブで防止
```

### Copy トレイト

```rust
// プリミティブ型はコピーされる（ムーブではない）
let x = 42;
let y = x;     // コピー（x はまだ使える）
println!("{}", x);  // ✅ OK

// Copy トレイトを実装する型:
// i32, f64, bool, char, タプル（全要素がCopy）
// &T（不変参照）

// String, Vec, Box などはコピーされない → ムーブ
// 明示的にクローンする場合:
let s1 = String::from("hello");
let s2 = s1.clone();  // ヒープデータのディープコピー
println!("{} {}", s1, s2);  // ✅ 両方使える
```

---

## 2. 借用（Borrowing）— 参照

```
所有権を移さずに値を使いたい → 「借用」（参照を渡す）

借用のルール:
  1. 不変参照（&T）は何個でも同時に存在できる
  2. 可変参照（&mut T）は同時に1つだけ
  3. 不変参照と可変参照は同時に存在できない
```

```rust
// 不変借用（&T）
fn calculate_length(s: &String) -> usize {
    s.len()  // 読み取りのみ
}  // s はここで消えるが、元のデータは解放されない（借りているだけ）

let s = String::from("hello");
let len = calculate_length(&s);  // &s で参照を渡す
println!("{}: {}", s, len);      // ✅ s はまだ使える

// 可変借用（&mut T）
fn append_world(s: &mut String) {
    s.push_str(", world!");
}

let mut s = String::from("hello");
append_world(&mut s);
println!("{}", s);  // → "hello, world!"

// 借用ルールの違反例
let mut s = String::from("hello");
let r1 = &s;      // ✅ 不変借用1
let r2 = &s;      // ✅ 不変借用2（何個でもOK）
// let r3 = &mut s; // ❌ 不変借用がある間は可変借用不可
println!("{} {}", r1, r2);

let r3 = &mut s;  // ✅ r1, r2 の使用が終わった後なのでOK
r3.push_str("!");
```

### なぜこのルールが必要か

```
データ競合の防止:

  データ競合が発生する3条件（全て同時に成立する場合）:
    1. 2つ以上のポインタが同じデータにアクセス
    2. 少なくとも1つが書き込み
    3. アクセスの同期がない

  Rust の借用ルールは条件2を排除:
    「複数の読み取り」OR「1つの書き込み」のどちらか
    → データ競合がコンパイル時に不可能になる
```

---

## 3. ライフタイム（Lifetime）

```
ライフタイム = 「参照が有効な期間」

コンパイラは参照が「ダングリング」しないことを保証する
```

```rust
// ダングリング参照の防止
fn dangling() -> &String {
    let s = String::from("hello");
    &s  // ❌ コンパイルエラー: s はこの関数で破棄されるのに参照を返そうとしている
}

// 正しい方法: 所有権を返す
fn not_dangling() -> String {
    let s = String::from("hello");
    s  // ✅ 所有権をムーブして返す
}

// ライフタイム注釈
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
// 'a は「x と y の参照の短い方の寿命」を表す
// 戻り値の参照は 'a の期間だけ有効

// ライフタイムの省略規則（Elision Rules）
// 多くの場合、コンパイラがライフタイムを自動推論
fn first_word(s: &str) -> &str {
    // ↑ コンパイラが自動的に fn first_word<'a>(s: &'a str) -> &'a str と推論
    &s[..s.find(' ').unwrap_or(s.len())]
}
```

---

## 4. スマートポインタ

```rust
// Box<T>: ヒープ上にデータを配置
let b = Box::new(5);  // ヒープに i32 を配置
// 用途: 再帰的データ型、大きなデータのムーブ

// Rc<T>: 参照カウント（単一スレッド）
use std::rc::Rc;
let a = Rc::new(String::from("hello"));
let b = Rc::clone(&a);  // 参照カウント +1
let c = Rc::clone(&a);  // 参照カウント +1
println!("count: {}", Rc::strong_count(&a));  // → 3

// Arc<T>: アトミック参照カウント（マルチスレッド）
use std::sync::Arc;
let data = Arc::new(vec![1, 2, 3]);
let data_clone = Arc::clone(&data);
std::thread::spawn(move || {
    println!("{:?}", data_clone);  // 別スレッドで安全にアクセス
});

// RefCell<T>: 実行時借用チェック（内部可変性）
use std::cell::RefCell;
let data = RefCell::new(42);
*data.borrow_mut() += 1;  // 実行時に借用ルールをチェック
println!("{}", data.borrow());  // → 43
```

---

## 5. 他の言語のメモリ安全性アプローチ

```
Rust:     所有権 + 借用チェッカー（コンパイル時）
Swift:    ARC（自動参照カウント）+ 値型の活用
C++:      RAII + unique_ptr/shared_ptr（プログラマの規律）
Go:       GC（並行マーク&スイープ）
Java:     GC（世代別 + ZGC等）
Python:   参照カウント + GC
```

```swift
// Swift: ARC（Automatic Reference Counting）
class User {
    var name: String
    init(name: String) { self.name = name }
    deinit { print("User \(name) deallocated") }
}

var user1: User? = User(name: "Gaku")   // 参照カウント: 1
var user2 = user1                         // 参照カウント: 2
user1 = nil                               // 参照カウント: 1
user2 = nil                               // 参照カウント: 0 → deinit

// 循環参照の回避: weak / unowned
class A {
    var b: B?
}
class B {
    weak var a: A?  // weak で循環参照を防止
}
```

---

## 実践演習

### 演習1: [基礎] — ムーブと借用の体験
Rust で文字列のムーブ、不変借用、可変借用を使い分けるプログラムを書く。

### 演習2: [応用] — リンクリストの実装
Rust で Box を使った単方向リンクリストを実装する。

### 演習3: [発展] — 所有権パズル
借用チェッカーのエラーが出るコードを修正するパズル（5問）に挑戦する。

---

## まとめ

| 概念 | 説明 |
|------|------|
| 所有権 | 値に対して所有者は1つだけ |
| ムーブ | 所有権の移動（元の変数は無効） |
| 借用（&T） | 不変参照（複数可） |
| 借用（&mut T） | 可変参照（1つだけ） |
| ライフタイム | 参照の有効期間 |
| スマートポインタ | Box, Rc, Arc, RefCell |

---

## 次に読むべきガイド
→ [[03-reference-counting-vs-tracing.md]] — 参照カウントとトレーシング

---

## 参考文献
1. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.4, 2023.
2. "Rust Nomicon." doc.rust-lang.org/nomicon.
