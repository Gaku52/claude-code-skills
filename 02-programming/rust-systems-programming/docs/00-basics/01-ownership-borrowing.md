# 所有権と借用 -- Rustの最も革新的なメモリ管理パラダイム

> 所有権(Ownership)と借用(Borrowing)はRust独自のメモリ管理モデルであり、ガベージコレクタなしでメモリ安全とデータ競合防止をコンパイル時に保証する。

---

## この章で学ぶこと

1. **所有権の3つの規則** -- 各値は唯一の所有者を持ち、スコープを抜けると解放される仕組みを理解する
2. **ムーブとコピー** -- 値の移動と複製の違い、Copy/Clone トレイトの使い分けを習得する
3. **借用とライフタイム基礎** -- 不変参照・可変参照の規則とライフタイムの入門を学ぶ

---

## 1. 所有権の基本規則

### 1.1 三つの規則

```
┌──────────────────────────────────────────────────┐
│            所有権の3つの規則                       │
├──────────────────────────────────────────────────┤
│ 1. 各値は「所有者」と呼ばれる変数を持つ           │
│ 2. 所有者は同時に1つだけ存在する                  │
│ 3. 所有者がスコープを抜けると値は破棄される       │
└──────────────────────────────────────────────────┘
```

### 例1: 所有権とスコープ

```rust
fn main() {
    {
        let s = String::from("hello"); // s がスコープに入る
        println!("{}", s);             // s は有効
    }                                  // s がスコープを抜ける → drop() 呼び出し
    // println!("{}", s);              // コンパイルエラー: s は存在しない
}
```

### 例2: ムーブセマンティクス

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;                    // ムーブ: s1 → s2
    // println!("{}", s1);          // エラー: s1 は無効化済み
    println!("{}", s2);             // OK
}
```

### 1.2 ムーブの図解

```
  ムーブ前:                     ムーブ後:
  s1                            s1 (無効)
  ┌──────────┐                  ┌──────────┐
  │ ptr ─────────┐              │ (無効)   │
  │ len: 5   │   │              └──────────┘
  │ cap: 5   │   │
  └──────────┘   │              s2
                 │              ┌──────────┐
                 │              │ ptr ─────────┐
                 │              │ len: 5   │   │
                 │              │ cap: 5   │   │
                 ▼              └──────────┘   │
  ┌──────────────┐                             │
  │ h e l l o    │<────────────────────────────┘
  └──────────────┘
  ヒープ上のデータは1つだけ（コピーされない）
```

---

## 2. コピーとクローン

### 例3: Copy トレイト(スタック上の値)

```rust
fn main() {
    let x: i32 = 42;
    let y = x;          // コピー（i32 は Copy トレイト実装済み）
    println!("x={}, y={}", x, y); // 両方有効！
}
```

### 例4: Clone トレイト(明示的な深いコピー)

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();           // 明示的にヒープデータもコピー
    println!("s1={}, s2={}", s1, s2); // 両方有効
}
```

### Copy が実装される型と実装されない型

```
┌─────────────────────────────┬────────────────────────────┐
│   Copy される型             │   Copy されない型           │
├─────────────────────────────┼────────────────────────────┤
│ i32, u64, f64, bool, char  │ String                     │
│ (i32, bool) -- タプル全要素 │ Vec<T>                     │
│   がCopyの場合              │ Box<T>                     │
│ [i32; 5] -- 固定長配列      │ HashMap<K, V>              │
│ &T -- 不変参照              │ File, TcpStream            │
└─────────────────────────────┴────────────────────────────┘
```

---

## 3. 借用(参照)

### 3.1 借用の規則

```
┌──────────────────────────────────────────────────┐
│            借用の規則                              │
├──────────────────────────────────────────────────┤
│ 1. 不変参照(&T)は同時に複数持てる                │
│ 2. 可変参照(&mut T)は同時に1つだけ               │
│ 3. 不変参照と可変参照は同時に存在できない         │
│ 4. 参照は常に有効でなければならない               │
└──────────────────────────────────────────────────┘
```

### 例5: 不変参照(共有参照)

```rust
fn calculate_length(s: &String) -> usize {
    s.len()
    // s はここで破棄されるが、所有権は持っていないのでデータは解放されない
}

fn main() {
    let s = String::from("hello");
    let len = calculate_length(&s);  // 借用(参照を渡す)
    println!("'{}' の長さは {}", s, len); // s はまだ有効
}
```

### 例6: 可変参照

```rust
fn append_world(s: &mut String) {
    s.push_str(", world!");
}

fn main() {
    let mut s = String::from("hello");
    append_world(&mut s);
    println!("{}", s); // "hello, world!"
}
```

### 例7: 借用規則の違反

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;        // OK: 不変参照1
    let r2 = &s;        // OK: 不変参照2
    println!("{}, {}", r1, r2);
    // r1, r2 はここ以降使われない (NLL)

    let r3 = &mut s;    // OK: r1, r2 のライフタイムは終了済み
    println!("{}", r3);
}
```

### 3.2 参照のライフサイクル図

```
    時間軸 →
    ├───────────┤
    │ r1 = &s   │   (不変参照: 生存)
    ├───────────┤
    │ r2 = &s   │   (不変参照: 生存)
    ├───────────┤
    │ println!  │   (r1, r2 最後の使用 = NLLにより終了)
    │           │
    │ r3 = &mut │   (可変参照: ここから生存 → OK)
    ├───────────┤
    │ println!  │   (r3 最後の使用)
    └───────────┘

    NLL (Non-Lexical Lifetimes):
    参照のライフタイムは「最後に使用された地点」で終了する
```

---

## 4. 関数と所有権

### 例8: 所有権の移動と返却

```rust
fn takes_ownership(s: String) -> String {
    println!("受け取った: {}", s);
    s  // 所有権を返す
}

fn main() {
    let s1 = String::from("hello");
    let s2 = takes_ownership(s1); // s1 → 関数 → s2
    // println!("{}", s1);        // エラー: s1 は無効
    println!("{}", s2);           // OK
}
```

### 例9: スライスによる効率的な借用

```rust
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}

fn main() {
    let sentence = String::from("hello world");
    let word = first_word(&sentence);
    println!("最初の単語: {}", word); // "hello"
}
```

---

## 5. 比較表

### 5.1 ムーブ vs コピー vs クローン

| 操作 | ヒープコピー | 元の値 | 自動/明示 | コスト |
|------|------------|--------|-----------|--------|
| ムーブ | なし | 無効化 | 自動 | O(1) |
| コピー (Copy) | N/A (スタックのみ) | 有効 | 自動 | O(1) |
| クローン (Clone) | あり | 有効 | 明示 (.clone()) | O(n) |

### 5.2 不変参照 vs 可変参照

| 特性 | `&T` (不変参照) | `&mut T` (可変参照) |
|------|-----------------|---------------------|
| 同時に持てる数 | 複数 | 1つだけ |
| データの変更 | 不可 | 可能 |
| 別名 | 共有参照 (shared ref) | 排他参照 (exclusive ref) |
| Send/Sync | T: Sync なら安全 | T: Send なら安全 |
| 他の参照と共存 | &mut T と共存不可 | &T と共存不可 |

---

## 6. アンチパターン

### アンチパターン1: 必要以上のクローン

```rust
// BAD: 参照で十分なのにクローンする
fn print_length(s: String) {
    println!("長さ: {}", s.len());
}
fn main() {
    let s = String::from("hello");
    print_length(s.clone()); // 不要なクローン
    print_length(s.clone()); // また不要なクローン
    println!("{}", s);
}

// GOOD: 参照を使う
fn print_length_good(s: &str) {
    println!("長さ: {}", s.len());
}
fn main() {
    let s = String::from("hello");
    print_length_good(&s);
    print_length_good(&s);
    println!("{}", s);
}
```

### アンチパターン2: ダングリング参照の試み

```rust
// BAD: ローカル変数への参照を返そうとする
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // s はこの関数終了時にドロップされる → ダングリング参照！
// }

// GOOD: 所有権ごと返す
fn no_dangle() -> String {
    let s = String::from("hello");
    s  // 所有権をムーブして返す
}
```

---

## 7. FAQ

### Q1: いつムーブが起こりますか？

**A:** 以下のケースでムーブが発生します:
- `let y = x;` (Copy未実装の型)
- 関数に値を渡す: `func(x)`
- 関数から値を返す: `return x`
- コレクションに値を入れる: `vec.push(x)`

Copy トレイトを実装している型(i32, bool, f64 など)はムーブではなくコピーされます。

### Q2: `&str` と `&String` の違いは何ですか？

**A:** `&str` は文字列スライスで、文字列データへの参照+長さの情報を持つ「ファットポインタ」です。`&String` は String 型への参照です。関数の引数には `&str` を使うのが慣例です。`&String` は自動的に `&str` にデリファレンスされるため(Deref coercion)、`&str` の方がより汎用的です。

### Q3: なぜ可変参照は同時に1つだけなのですか？

**A:** データ競合を防止するためです。データ競合は以下の3条件が揃うと発生します:
1. 2つ以上のポインタが同じデータにアクセス
2. 少なくとも1つが書き込み
3. アクセスの同期がない

可変参照を1つに制限することで、条件1,2の組み合わせをコンパイル時に排除できます。

---

## 8. まとめ

| 概念 | 要点 |
|------|------|
| 所有権 | 各値は唯一の所有者を持つ。スコープ終了で自動 drop |
| ムーブ | 代入/関数呼び出しで所有権が移転。元の変数は無効化 |
| Copy | スタック上の小さな値は暗黙にコピー(i32, bool 等) |
| Clone | ヒープデータの明示的な深いコピー |
| 不変参照 (&T) | 同時に複数可能。データ変更不可 |
| 可変参照 (&mut T) | 同時に1つだけ。データ変更可能 |
| NLL | 参照の寿命は最後の使用地点で終了 |
| スライス | データの一部への参照。所有しない |

---

## 次に読むべきガイド

- [02-types-and-traits.md](02-types-and-traits.md) -- 型とトレイトで抽象化を学ぶ
- [../01-advanced/00-lifetimes.md](../01-advanced/00-lifetimes.md) -- ライフタイムを詳しく理解する
- [../01-advanced/01-smart-pointers.md](../01-advanced/01-smart-pointers.md) -- Box/Rc/Arc で所有権を拡張する

---

## 参考文献

1. **The Rust Programming Language - Ch.4 Understanding Ownership** -- https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
2. **Rust by Example - Ownership** -- https://doc.rust-lang.org/rust-by-example/scope/move.html
3. **The Rustonomicon - Ownership** -- https://doc.rust-lang.org/nomicon/ownership.html
4. **Non-Lexical Lifetimes (NLL) RFC** -- https://rust-lang.github.io/rfcs/2094-nll.html
