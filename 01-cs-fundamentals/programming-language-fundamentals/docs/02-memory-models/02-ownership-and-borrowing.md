# 所有権と借用 --- Rust のメモリ管理モデル完全ガイド

> **到達目標**: Rust の所有権システム（Ownership）・借用（Borrowing）・ライフタイム（Lifetime）の 3 本柱を理解し、GC なしで安全かつ高速なプログラムを設計できるようになること。コンパイル時にメモリ安全性を保証するという革命的な仕組みの原理を、豊富なコード例と図解で体系的に学ぶ。

---

## 目次

1. [導入 --- なぜ所有権が必要か](#1-導入----なぜ所有権が必要か)
2. [所有権の 3 つのルール](#2-所有権の-3-つのルール)
3. [ムーブセマンティクスとコピーセマンティクス](#3-ムーブセマンティクスとコピーセマンティクス)
4. [借用（Borrowing）--- 参照の規律](#4-借用borrowing---参照の規律)
5. [ライフタイム（Lifetime）](#5-ライフタイムlifetime)
6. [スマートポインタと所有権の拡張](#6-スマートポインタと所有権の拡張)
7. [所有権パターン集 --- 設計への応用](#7-所有権パターン集----設計への応用)
8. [他言語との比較 --- メモリ安全性アプローチ](#8-他言語との比較----メモリ安全性アプローチ)
9. [アンチパターンと落とし穴](#9-アンチパターンと落とし穴)
10. [実践演習（3 段階）](#10-実践演習3-段階)
11. [FAQ --- よくある質問](#11-faq----よくある質問)
12. [まとめ](#12-まとめ)
13. [参考文献](#13-参考文献)

---

## この章で学ぶこと

- [ ] 所有権の 3 つのルールを理解し、ムーブとコピーの違いを説明できる
- [ ] 不変参照・可変参照の借用ルールを理解し、データ競合が起きない理由を説明できる
- [ ] ライフタイムの概念を理解し、ダングリング参照が防がれる仕組みを説明できる
- [ ] スマートポインタ（Box, Rc, Arc, RefCell）の使い分けを判断できる
- [ ] 所有権パターンを用いて安全な API を設計できる
- [ ] C++/Swift/Go/Java など他言語のメモリ管理手法との違いを比較できる

---

## 1. 導入 --- なぜ所有権が必要か

### 1.1 メモリ管理の歴史的課題

プログラミング言語の歴史は「メモリをどう安全に管理するか」という問いとの闘いだった。C 言語の時代から、プログラマは以下のバグと格闘してきた。

```
メモリ安全性に関する主要なバグの分類:

+-------------------------+----------------------------------------+
| バグの種類              | 説明                                   |
+-------------------------+----------------------------------------+
| ダングリングポインタ    | 解放済みメモリへの参照                 |
| 二重解放 (double free)  | 同じメモリを 2 回 free する            |
| メモリリーク            | 不要なメモリを解放し忘れる             |
| バッファオーバーフロー  | 確保した領域を超えてアクセス           |
| データ競合              | 複数スレッドが同時に読み書き           |
| Use-After-Free          | 解放後にメモリを使用する               |
+-------------------------+----------------------------------------+
```

Microsoft のセキュリティチームによる調査では、同社製品のセキュリティ脆弱性の約 70% がメモリ安全性に起因する問題であると報告されている。Google の Chrome チームも同様の数値を報告している。

### 1.2 従来のアプローチとその限界

```
メモリ管理の 3 つのアプローチ:

  [手動管理]          [GC]               [所有権]
   C / C++         Java / Go / Python      Rust
      |                  |                   |
      v                  v                   v
  malloc/free      ランタイムが自動回収   コンパイル時に検証
      |                  |                   |
  +--------+       +----------+        +----------+
  | 高速   |       | 安全     |        | 高速     |
  | 危険   |       | GC停止   |        | 安全     |
  +--------+       +----------+        +----------+
```

| アプローチ | 代表的な言語 | 安全性 | 性能 | 予測可能性 |
|-----------|-------------|--------|------|-----------|
| 手動管理 | C, C++ | 低（プログラマ依存） | 高 | 高 |
| GC（トレーシング） | Java, Go, C# | 高 | 中（GC 停止あり） | 低 |
| 参照カウント | Swift, Python, Obj-C | 中（循環参照問題） | 中 | 中 |
| **所有権システム** | **Rust** | **高** | **高** | **高** |

Rust はこれらの従来アプローチの弱点を克服するために、所有権（Ownership）という新しいパラダイムを導入した。コンパイル時に安全性を検証するため、実行時のオーバーヘッドがゼロでありながら、GC と同等以上の安全性を達成する。

### 1.3 所有権システムの核心的な洞察

Rust の所有権システムは、次の洞察に基づいている。

> **「全ての値にただ一人の所有者を持たせ、所有者がスコープを抜けた時点で値を破棄すれば、メモリリークも二重解放も起きない」**

この単純な原則から、ムーブセマンティクス、借用、ライフタイムという精巧な仕組みが導き出される。

```
所有権システムの全体像:

  ┌─────────────────────────────────────────────────────┐
  │                  所有権システム                       │
  │                                                       │
  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
  │  │ 所有権ルール │  │   借用規則   │  │ ライフタイム│  │
  │  │             │  │              │  │            │  │
  │  │ - 唯一の    │  │ - &T: 複数可 │  │ - 参照の   │  │
  │  │   所有者    │  │ - &mut T:    │  │   有効期間 │  │
  │  │ - スコープ  │  │   1つだけ    │  │ - 'a 注釈  │  │
  │  │   でdrop    │  │ - 排他制御   │  │ - 省略規則 │  │
  │  │ - ムーブ    │  │              │  │            │  │
  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │
  │         │               │                │          │
  │         └───────────────┼────────────────┘          │
  │                         │                            │
  │                   借用チェッカー                      │
  │                  (Borrow Checker)                     │
  │                   コンパイル時に全検証                 │
  └─────────────────────────────────────────────────────┘
```

---

## 2. 所有権の 3 つのルール

### 2.1 ルールの定義

Rust の所有権は、次の 3 つのルールで定義される。

```
┌──────────────────────────────────────────────────────────────┐
│                    所有権の 3 つのルール                       │
│                                                                │
│  Rule 1: 全ての値には「所有者」（変数）が 1 つだけ存在する    │
│                                                                │
│  Rule 2: 所有者がスコープを抜けると、値は自動的に破棄される   │
│          （drop が呼ばれる）                                   │
│                                                                │
│  Rule 3: 所有権は「移動」（ムーブ）できるが、                 │
│          コピーはデフォルトでは行われない                       │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 ルール 1 --- 唯一の所有者

全ての値は、ある瞬間において必ず 1 つの変数によって所有される。この変数を「所有者（Owner）」と呼ぶ。

```rust
// ===== コード例1: 所有者の基本 =====

fn main() {
    // s が "hello" という String 値の所有者
    let s = String::from("hello");

    // String のメモリレイアウト:
    //
    //  スタック (s)         ヒープ
    //  +---------+         +---+---+---+---+---+
    //  | ptr   --|-------->| h | e | l | l | o |
    //  | len: 5  |         +---+---+---+---+---+
    //  | cap: 5  |
    //  +---------+
    //
    // s はスタック上のメタデータ (ポインタ, 長さ, 容量) を持ち、
    // 実際の文字列データはヒープ上に格納される。

    println!("{}", s); // s を通じて値にアクセス
}
// ← s がスコープを抜ける → drop(s) が呼ばれ、ヒープメモリが解放される
```

### 2.3 ルール 2 --- スコープと自動破棄

所有者がスコープを抜けると、Rust は自動的に `drop` 関数を呼び出して値を破棄する。この仕組みは C++ の RAII (Resource Acquisition Is Initialization) と同じ原理だが、Rust ではコンパイラが厳密に強制する。

```rust
// ===== コード例2: スコープとdropの動作 =====

struct DatabaseConnection {
    url: String,
}

impl DatabaseConnection {
    fn new(url: &str) -> Self {
        println!("[OPEN] DB接続を確立: {}", url);
        DatabaseConnection { url: url.to_string() }
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        println!("[CLOSE] DB接続を切断: {}", self.url);
    }
}

fn process_data() {
    let conn = DatabaseConnection::new("postgres://localhost/mydb");
    // conn を使った処理...
    println!("[QUERY] データを取得中...");

    {
        let temp_conn = DatabaseConnection::new("postgres://localhost/tempdb");
        println!("[QUERY] 一時データを取得中...");
    } // ← temp_conn がスコープを抜ける → drop 呼び出し
      //   出力: [CLOSE] DB接続を切断: postgres://localhost/tempdb

    println!("[QUERY] 追加処理中...");
} // ← conn がスコープを抜ける → drop 呼び出し
  //   出力: [CLOSE] DB接続を切断: postgres://localhost/mydb

// 実行結果:
// [OPEN]  DB接続を確立: postgres://localhost/mydb
// [QUERY] データを取得中...
// [OPEN]  DB接続を確立: postgres://localhost/tempdb
// [QUERY] 一時データを取得中...
// [CLOSE] DB接続を切断: postgres://localhost/tempdb
// [QUERY] 追加処理中...
// [CLOSE] DB接続を切断: postgres://localhost/mydb
```

この例から分かるように、所有権とスコープの仕組みにより、リソースの解放漏れが構造的に防止される。ファイルハンドル、ネットワーク接続、ロックなど、あらゆるリソースに適用できる。

### 2.4 ルール 3 --- ムーブ（所有権の移動）

値の所有権は別の変数に移動できる。移動後、元の変数は無効になる。

```rust
let s1 = String::from("hello");
let s2 = s1;  // 所有権が s1 → s2 に移動（ムーブ）
// println!("{}", s1);  // コンパイルエラー! s1 はもう使えない
println!("{}", s2);     // OK: s2 が新しい所有者
```

ムーブが発生した際のメモリ上の変化を図示する。

```
ムーブ前:
  s1                     ヒープ
  +---------+           +---+---+---+---+---+
  | ptr   --|---------->| h | e | l | l | o |
  | len: 5  |           +---+---+---+---+---+
  | cap: 5  |
  +---------+

ムーブ後 (let s2 = s1):
  s1 (無効)              ヒープ
  +---------+           +---+---+---+---+---+
  | (無効)  |     .---->| h | e | l | l | o |
  +---------+     |     +---+---+---+---+---+
                  |
  s2              |
  +---------+     |
  | ptr   --|-----'
  | len: 5  |
  | cap: 5  |
  +---------+

  ポイント:
  - ヒープデータはコピーされない（ポインタだけが移動）
  - s1 は無効化され、以後アクセス不可
  - s2 がスコープを抜けた時のみ、ヒープメモリが解放される
  - 二重解放（double free）が構造的に不可能
```

---

## 3. ムーブセマンティクスとコピーセマンティクス

### 3.1 ムーブが発生する場面

ムーブは代入だけでなく、様々な場面で発生する。

```rust
// ===== コード例3: ムーブが発生する各場面 =====

fn main() {
    // (1) 変数への代入
    let s1 = String::from("hello");
    let s2 = s1;  // ムーブ

    // (2) 関数の引数に渡す
    let s3 = String::from("world");
    take_ownership(s3);  // s3 の所有権が関数に移動
    // println!("{}", s3);  // コンパイルエラー!

    // (3) 関数の戻り値
    let s4 = give_ownership();  // 関数から所有権を受け取る
    println!("{}", s4);  // OK

    // (4) ベクタへの push
    let s5 = String::from("item");
    let mut vec = Vec::new();
    vec.push(s5);  // s5 の所有権がベクタに移動
    // println!("{}", s5);  // コンパイルエラー!

    // (5) パターンマッチ
    let opt = Some(String::from("data"));
    match opt {
        Some(s) => println!("Got: {}", s),  // s に所有権がムーブ
        None => println!("None"),
    }
    // println!("{:?}", opt);  // コンパイルエラー!
}

fn take_ownership(s: String) {
    println!("Took ownership of: {}", s);
} // ← s がスコープを抜ける → メモリ解放

fn give_ownership() -> String {
    String::from("gifted")  // 所有権を呼び出し元に移動
}
```

### 3.2 Copy トレイトと Clone トレイト

一部の型はムーブではなくコピーされる。これは `Copy` トレイトを実装している型に限られる。

```rust
// Copy トレイトを実装する型（コピーされる型）
let x: i32 = 42;
let y = x;     // コピー（x はまだ有効）
println!("{}", x);  // OK

let a: f64 = 3.14;
let b = a;     // コピー
println!("{}", a);  // OK

let c: bool = true;
let d = c;     // コピー

let e: char = 'A';
let f = e;     // コピー

let g: (i32, f64) = (1, 2.0);
let h = g;     // タプルの全要素が Copy なのでコピー

// Copy トレイトを実装しない型（ムーブされる型）
let s1 = String::from("hello");
let s2 = s1;   // ムーブ (s1 は無効)

let v1 = vec![1, 2, 3];
let v2 = v1;   // ムーブ (v1 は無効)

// 明示的なクローン（ディープコピー）
let s3 = String::from("world");
let s4 = s3.clone();  // ヒープデータも含めた完全なコピー
println!("{} {}", s3, s4);  // 両方有効
```

### 3.3 Copy と Clone の違い

| 特性 | Copy | Clone |
|------|------|-------|
| 動作 | ビット単位の浅いコピー | 任意のカスタムロジック（通常はディープコピー） |
| 暗黙性 | 暗黙的（代入時に自動コピー） | 明示的（`.clone()` の呼び出しが必要） |
| 性能 | 常に軽量（スタック上のコピー） | 型による（ヒープ割り当てを含む場合がある） |
| 要件 | 型の全フィールドが Copy である必要 | Drop を実装していても可 |
| 適用例 | i32, f64, bool, char, &T | String, Vec, HashMap |
| Drop との共存 | 不可（Copy と Drop は排他的） | 可能 |

```
Copy と Clone の判断フローチャート:

  型 T を代入するとき
       │
       ▼
  T は Copy を実装しているか？
       │
  ┌────┴────┐
  Yes       No
  │         │
  ▼         ▼
 暗黙コピー  ムーブ（所有権移動）
 (T はまだ   (元の変数は無効)
  有効)          │
                ▼
           明示的に clone() を
           呼べばコピー可能
```

### 3.4 自作型の Copy/Clone 実装

```rust
// Copy + Clone を derive する（全フィールドが Copy の場合のみ）
#[derive(Debug, Copy, Clone)]
struct Point {
    x: f64,
    y: f64,
}

let p1 = Point { x: 1.0, y: 2.0 };
let p2 = p1;  // コピー（p1 は有効）
println!("{:?} {:?}", p1, p2);

// Clone のみ derive する（ヒープデータを含む場合）
#[derive(Debug, Clone)]
struct Person {
    name: String,    // String は Copy でない
    age: u32,
}

let alice = Person { name: String::from("Alice"), age: 30 };
let alice2 = alice.clone();  // 明示的クローン
// let alice3 = alice;       // ムーブ（clone しないとムーブされる）
println!("{:?}", alice2);

// カスタム Clone の実装
#[derive(Debug)]
struct Config {
    name: String,
    values: Vec<i32>,
    read_count: std::cell::Cell<u32>,
}

impl Clone for Config {
    fn clone(&self) -> Self {
        Config {
            name: self.name.clone(),
            values: self.values.clone(),
            read_count: std::cell::Cell::new(0),  // クローン時はカウントをリセット
        }
    }
}
```

---

## 4. 借用（Borrowing）--- 参照の規律

### 4.1 借用の基本概念

所有権の移動なしに値を使う仕組みが「借用（Borrowing）」である。借用は参照（`&T` または `&mut T`）を通じて行われる。

```
借用の概念図:

  所有者 s              借用者 r
  +---------+          +---------+
  | ptr   --|------.   |         |
  | len: 5  |      |   | ptr   --|---.
  | cap: 5  |      |   +---------+   |
  +---------+      |                  |
                   v                  |
                +---+---+---+---+---+ |
                | h | e | l | l | o |<'
                +---+---+---+---+---+

  r = &s  →  r は s が所有するデータを「借りている」
  ポイント:
  - r はデータの所有権を持たない
  - r がスコープを抜けてもデータは解放されない
  - s が先にスコープを抜けると r はダングリングになる
    → コンパイラがこれを防ぐ（ライフタイムチェック）
```

### 4.2 借用の 3 つのルール

```
┌──────────────────────────────────────────────────────────────┐
│                    借用の 3 つのルール                        │
│                                                                │
│  Rule 1: 不変参照 (&T) は同時にいくつでも存在できる          │
│          → 「複数の読者」は問題ない                           │
│                                                                │
│  Rule 2: 可変参照 (&mut T) は同時に 1 つだけ存在できる       │
│          → 「書き手は 1 人だけ」                              │
│                                                                │
│  Rule 3: 不変参照と可変参照は同時に存在できない              │
│          → 「読んでいる最中に書き換えられたら困る」           │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 不変借用（&T）

```rust
// ===== コード例4: 不変借用の詳細 =====

fn main() {
    let s = String::from("hello, world");

    // 不変参照を複数同時に作成できる
    let r1 = &s;
    let r2 = &s;
    let r3 = &s;

    println!("{}, {}, {}", r1, r2, r3); // 全て有効

    // 関数に不変参照を渡す
    let length = calculate_length(&s);
    let first = first_word(&s);

    println!("'{}' の長さ: {}, 最初の単語: '{}'", s, length, first);
    // s はまだ有効（所有権は移動していない）
}

// 不変参照を受け取る関数
fn calculate_length(s: &String) -> usize {
    s.len()
    // s はここでスコープを抜けるが、参照なので何も起きない
    // 元のデータは解放されない
}

fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}
```

### 4.4 可変借用（&mut T）

```rust
fn main() {
    let mut s = String::from("hello");

    // 可変参照を作成
    let r = &mut s;
    r.push_str(", world!");
    println!("{}", r);  // "hello, world!"

    // 可変参照は同時に 1 つだけ
    let mut data = vec![1, 2, 3];

    let r1 = &mut data;
    // let r2 = &mut data;  // コンパイルエラー! 2 つ目の可変参照
    r1.push(4);
    println!("{:?}", r1);

    // r1 の使用が終わった後なら新しい可変参照を作れる
    let r2 = &mut data;
    r2.push(5);
    println!("{:?}", r2);
}

// 可変参照を受け取る関数
fn append_greeting(s: &mut String) {
    s.push_str(", world!");
}
```

### 4.5 借用ルールの違反例と対処法

```rust
fn main() {
    // --- 違反例 1: 不変参照中に可変参照を取る ---
    let mut s = String::from("hello");
    let r1 = &s;         // 不変参照
    let r2 = &s;         // 不変参照（OK）
    // let r3 = &mut s;  // コンパイルエラー!
    //   不変参照 r1, r2 が生きている間は可変参照を取れない
    println!("{} {}", r1, r2);
    // ここで r1, r2 の最後の使用が終わる（NLL: Non-Lexical Lifetimes）

    let r3 = &mut s;  // OK: r1, r2 はもう使われない
    r3.push_str("!");
    println!("{}", r3);

    // --- 違反例 2: 同時に 2 つの可変参照 ---
    let mut v = vec![1, 2, 3, 4, 5];
    // split_at_mut は安全に 2 つの可変スライスを得る方法
    let (left, right) = v.split_at_mut(3);
    left[0] = 10;
    right[0] = 40;
    println!("{:?} {:?}", left, right); // [10, 2, 3] [40, 5]
}
```

### 4.6 NLL（Non-Lexical Lifetimes）

Rust 2018 以降、借用のスコープはレキシカル（波括弧）ではなく、**最後に使用された地点**で終了する。これを NLL（Non-Lexical Lifetimes）と呼ぶ。

```rust
fn main() {
    let mut s = String::from("hello");

    // Rust 2015 では、r1 はブロックの終わりまで有効だった
    // Rust 2018+ (NLL) では、r1 は最後に使用された行で終了

    let r1 = &s;
    println!("{}", r1);  // r1 の最後の使用 → ここで r1 の借用が終了

    let r2 = &mut s;     // OK: r1 はもう使われない
    r2.push_str(", world");
    println!("{}", r2);
}
```

```
NLL の動作イメージ:

  行番号   コード                   r1 の有効範囲   r2 の有効範囲
  ------   ----                     --------------   --------------
  1        let r1 = &s;            |-- 開始
  2        println!("{}", r1);      |-- 終了 (最後の使用)
  3        let r2 = &mut s;                          |-- 開始
  4        r2.push_str(", world");                   |
  5        println!("{}", r2);                       |-- 終了

  r1 と r2 の有効範囲が重ならないため、コンパイル成功
```

### 4.7 なぜこの借用ルールが必要か --- データ競合の防止

```
データ競合が発生する 3 つの条件（全て同時に成立する場合）:

  条件 1: 2 つ以上のポインタが同じデータにアクセスする
  条件 2: 少なくとも 1 つが書き込みを行う
  条件 3: アクセスの同期が取られていない

Rust の借用ルールはこれを構造的に排除する:

  パターン A: 複数の不変参照 (&T, &T, &T)
    → 条件 2 が不成立（全て読み取りのみ）→ 安全

  パターン B: 1 つの可変参照 (&mut T)
    → 条件 1 が不成立（アクセスするのは 1 つだけ）→ 安全

  パターン C: 不変参照 + 可変参照 → コンパイルエラー!
    → 条件 1, 2, 3 全て成立しうる → コンパイラが拒否

  結論: Rust ではデータ競合がコンパイル時に不可能
```

この仕組みが特に威力を発揮するのは並行プログラミングの場面である。他の言語では実行時にしか検出できないデータ競合が、Rust ではコンパイル時に全て検出される。

### 4.8 借用とイテレーション

コレクションの借用で特に注意が必要なのが、イテレーション中のコレクション変更である。

```rust
fn main() {
    let mut scores = vec![100, 85, 92, 78, 95];

    // NG: イテレーション中にベクタを変更しようとする
    // for &score in &scores {
    //     if score < 80 {
    //         scores.push(0);  // コンパイルエラー!
    //         // &scores (不変借用) と scores.push (可変借用) が衝突
    //     }
    // }

    // OK: まず条件を収集し、後で変更する
    let low_scores: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, &s)| s < 80)
        .map(|(i, _)| i)
        .collect();

    for &idx in &low_scores {
        scores[idx] = 0;  // 不変借用はもう存在しない
    }
    println!("{:?}", scores); // [100, 85, 92, 0, 95]

    // OK: retain を使う（内部的に安全に処理される）
    let mut names = vec!["Alice", "Bob", "Charlie", "Dave"];
    names.retain(|name| name.len() > 3);
    println!("{:?}", names); // ["Alice", "Charlie", "Dave"]
}
```

---

## 5. ライフタイム（Lifetime）

### 5.1 ライフタイムとは何か

ライフタイムとは、参照が有効である期間を表す概念である。全ての参照にはライフタイムが存在し、コンパイラはライフタイムを追跡することで、ダングリング参照（解放済みメモリへの参照）を防ぐ。

```
ライフタイムの概念図:

  fn main() {
      let r;                  // -----+-- 'a (r のライフタイム)
      {                       //      |
          let x = 5;          // -+-- 'b (x のライフタイム)
          r = &x;             //  |   |   r が x を参照
      }                       // -+   |   x がスコープを抜ける → 解放
      // println!("{}", r);   //      |   ダングリング! コンパイルエラー
  }                           // -----+

  'b は 'a より短い → x の参照を r に代入するのは不安全
  コンパイラがこれを検出してエラーにする
```

### 5.2 ダングリング参照の防止

```rust
// ===== コード例5: ダングリング参照とその防止 =====

// NG: ダングリング参照を返そうとする
// fn dangling() -> &String {
//     let s = String::from("hello");
//     &s  // コンパイルエラー: s はこの関数で破棄されるのに参照を返そうとしている
// }
// エラーメッセージ:
//   this function's return type contains a borrowed value,
//   but there is no value for it to be borrowed from

// 解決策 1: 所有権を返す（ムーブ）
fn not_dangling_v1() -> String {
    let s = String::from("hello");
    s  // 所有権をムーブして返す
}

// 解決策 2: 'static ライフタイム（プログラム全体で有効）
fn not_dangling_v2() -> &'static str {
    "hello"  // 文字列リテラルは 'static ライフタイムを持つ
}

// 解決策 3: 引数の参照を返す（ライフタイムを明示）
fn not_dangling_v3<'a>(s: &'a str) -> &'a str {
    &s[..3]  // 引数と同じライフタイムの参照を返す
}
```

### 5.3 ライフタイム注釈

関数が参照を受け取って参照を返す場合、コンパイラは戻り値のライフタイムと引数のライフタイムの関係を知る必要がある。これをライフタイム注釈で明示する。

```rust
// ライフタイム注釈の構文: 'a, 'b, 'c ...（慣例的に小文字のアルファベット）

// 2 つの文字列スライスのうち長い方を返す関数
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 'a の意味:
// 「戻り値の参照は、x と y の両方のライフタイムの
//   短い方と同じかそれより短い期間だけ有効である」

fn main() {
    let string1 = String::from("long string");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        println!("Longest: {}", result);  // OK: string2 はまだ有効
    }
    // println!("{}", result);  // コンパイルエラー!
    // string2 が解放されたため、result は無効
}

// 異なるライフタイムを持つ引数
fn first_or_default<'a, 'b>(first: &'a str, _default: &'b str) -> &'a str {
    first  // 戻り値は first と同じライフタイム
}
```

### 5.4 構造体のライフタイム

構造体が参照を保持する場合、ライフタイム注釈が必要になる。

```rust
// 参照を保持する構造体にはライフタイム注釈が必須
#[derive(Debug)]
struct Excerpt<'a> {
    part: &'a str,  // 'a の期間だけ有効な参照
}

impl<'a> Excerpt<'a> {
    fn level(&self) -> i32 {
        3  // 参照を返さないのでライフタイム注釈不要
    }

    fn announce_and_return(&self, announcement: &str) -> &str {
        println!("Attention: {}", announcement);
        self.part  // self.part と同じライフタイムの参照を返す
    }
}

fn main() {
    let novel = String::from("Call me Ishmael. Some years ago...");
    let first_sentence;
    {
        let sentences: Vec<&str> = novel.split('.').collect();
        first_sentence = Excerpt { part: sentences[0] };
    }
    // first_sentence はまだ有効:
    // part が参照する novel はまだスコープ内にあるため
    println!("{:?}", first_sentence);
}
```

### 5.5 ライフタイム省略規則（Elision Rules）

多くの場合、コンパイラはライフタイムを自動的に推論する。これをライフタイム省略規則と呼ぶ。

```
ライフタイム省略規則（3 つのルール）:

  ルール 1 (入力ライフタイム):
    各参照パラメータにそれぞれ別のライフタイムを割り当てる
    fn foo(x: &str, y: &str) → fn foo<'a, 'b>(x: &'a str, y: &'b str)

  ルール 2 (出力ライフタイム - 単一入力):
    入力ライフタイムが 1 つだけなら、出力に同じライフタイムを割り当てる
    fn foo(x: &str) -> &str → fn foo<'a>(x: &'a str) -> &'a str

  ルール 3 (出力ライフタイム - メソッド):
    メソッドで &self または &mut self がある場合、
    self のライフタイムを出力に割り当てる
    fn foo(&self, x: &str) -> &str → self のライフタイム

  3 つのルールを適用しても出力ライフタイムが決まらない場合:
  → コンパイルエラー → プログラマが明示的に注釈する必要がある
```

```rust
// 省略規則が適用される例

// 書いたコード                  コンパイラが推論する完全形
fn first_word(s: &str) -> &str  // fn first_word<'a>(s: &'a str) -> &'a str
{
    &s[..s.find(' ').unwrap_or(s.len())]
}

// 省略できない例（2 つの入力参照）
// fn longest(x: &str, y: &str) -> &str  // コンパイルエラー!
// → どちらのライフタイムを返すか不明
// → 明示的な注釈が必要:
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 5.6 'static ライフタイム

`'static` はプログラムの全実行期間を通じて有効なライフタイムである。

```rust
// 文字列リテラルは 'static
let s: &'static str = "I live forever";

// 'static な値はプログラム終了まで有効
// バイナリに直接埋め込まれるため、解放の必要がない

// 注意: 'static を安易に使うのはアンチパターン
// 本当にプログラム全体で有効にする必要がある場合のみ使用すること
// エラーメッセージで 'static を提案されても、多くの場合は
// 設計を見直すべきサインである
```

---

## 6. スマートポインタと所有権の拡張

### 6.1 スマートポインタの概要

所有権の「値に対して所有者は 1 つだけ」というルールでは対応しきれないケースがある。スマートポインタはこれらのケースに対応するための型である。

```
スマートポインタの分類:

  ┌─────────────────────────────────────────────────────────┐
  │                スマートポインタ体系                       │
  │                                                           │
  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │
  │  │  Box<T>   │  │   Rc<T>   │  │     Arc<T>        │    │
  │  │           │  │           │  │                   │    │
  │  │ ヒープ    │  │ 参照      │  │ アトミック        │    │
  │  │ 配置      │  │ カウント  │  │ 参照カウント      │    │
  │  │ 単一所有  │  │ 共有所有  │  │ スレッド安全      │    │
  │  │           │  │ 単一      │  │ 共有所有          │    │
  │  │           │  │ スレッド  │  │                   │    │
  │  └───────────┘  └───────────┘  └───────────────────┘    │
  │                                                           │
  │  ┌───────────────────┐  ┌─────────────────────────┐     │
  │  │   RefCell<T>      │  │   Cow<'a, T>            │     │
  │  │                   │  │                         │     │
  │  │ 内部可変性        │  │ Clone-on-Write          │     │
  │  │ 実行時借用チェック│  │ 遅延クローン            │     │
  │  └───────────────────┘  └─────────────────────────┘     │
  └─────────────────────────────────────────────────────────┘
```

### 6.2 Box -- ヒープ配置と再帰型

```rust
// Box<T>: データをヒープに配置し、スタックにはポインタのみ置く

// 用途 1: コンパイル時にサイズ不明な型
// 再帰的データ型（Box なしではコンパイルエラー）
#[derive(Debug)]
enum List {
    Cons(i32, Box<List>),  // Box で間接参照にすることでサイズが確定
    Nil,
}

use List::{Cons, Nil};

fn main() {
    let list = Cons(1, Box::new(Cons(2, Box::new(Cons(3, Box::new(Nil))))));
    println!("{:?}", list);
    // Cons(1, Cons(2, Cons(3, Nil)))
}

// 用途 2: 大きなデータをムーブする際のコスト削減
struct LargeData {
    buffer: [u8; 1_000_000],  // 1MB のデータ
}

fn process(data: Box<LargeData>) {
    // Box のムーブはポインタのコピーのみ（8 バイト）
    // LargeData を直接ムーブすると 1MB のコピーが発生する
    println!("Processing {} bytes", data.buffer.len());
}

// 用途 3: トレイトオブジェクト（動的ディスパッチ）
trait Animal {
    fn speak(&self) -> &str;
}

struct Dog;
struct Cat;

impl Animal for Dog {
    fn speak(&self) -> &str { "Woof!" }
}

impl Animal for Cat {
    fn speak(&self) -> &str { "Meow!" }
}

fn get_animal(is_dog: bool) -> Box<dyn Animal> {
    if is_dog {
        Box::new(Dog)
    } else {
        Box::new(Cat)
    }
}
```

### 6.3 Rc -- 参照カウント（単一スレッド）

```rust
use std::rc::Rc;

// Rc<T>: 複数の所有者を持つための参照カウント
// 単一スレッドでのみ使用可能

fn main() {
    // グラフ構造: ノード C を A と B の両方が参照する
    //
    //     A ---+
    //          |
    //          v
    //          C
    //          ^
    //          |
    //     B ---+

    let c = Rc::new(String::from("shared data"));
    println!("参照カウント (初期): {}", Rc::strong_count(&c));  // 1

    let a = Rc::clone(&c);  // カウント +1 (データのクローンではない!)
    println!("参照カウント (a 作成後): {}", Rc::strong_count(&c));  // 2

    {
        let b = Rc::clone(&c);  // カウント +1
        println!("参照カウント (b 作成後): {}", Rc::strong_count(&c));  // 3
    }  // b がスコープを抜ける → カウント -1

    println!("参照カウント (b 破棄後): {}", Rc::strong_count(&c));  // 2

    // 全ての Rc が破棄された時点で、データが解放される
}
// a, c が破棄 → カウント 0 → データ解放

// Weak<T>: 循環参照を防ぐ弱い参照
use std::rc::Weak;

#[derive(Debug)]
struct Node {
    value: i32,
    parent: RefCell<Weak<Node>>,      // 弱い参照（カウントに含まれない）
    children: RefCell<Vec<Rc<Node>>>,  // 強い参照
}

// 親 → 子: Rc（強い参照）
// 子 → 親: Weak（弱い参照）
// → 循環参照にならないため、メモリリークしない
```

### 6.4 Arc -- アトミック参照カウント（マルチスレッド）

```rust
use std::sync::Arc;
use std::thread;

// Arc<T>: Rc のスレッド安全版
// アトミック操作で参照カウントを管理するため、若干のオーバーヘッドがある

fn main() {
    let data = Arc::new(vec![1, 2, 3, 4, 5]);
    let mut handles = vec![];

    for i in 0..3 {
        let data_clone = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let sum: i32 = data_clone.iter().sum();
            println!("Thread {}: sum = {}", i, sum);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final ref count: {}", Arc::strong_count(&data));  // 1
}

// Arc + Mutex: スレッド間で可変データを共有
use std::sync::Mutex;

fn concurrent_counter() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter_clone.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());  // 10
}
```

### 6.5 RefCell -- 内部可変性

```rust
use std::cell::RefCell;

// RefCell<T>: コンパイル時ではなく実行時に借用ルールをチェック
// 「不変参照しか持てない状況で、内部を変更したい」場合に使う

fn main() {
    let data = RefCell::new(vec![1, 2, 3]);

    // 不変な data から可変借用を取得
    data.borrow_mut().push(4);
    println!("{:?}", data.borrow());  // [1, 2, 3, 4]

    // 実行時の借用ルール違反はパニックになる
    // let r1 = data.borrow();      // 不変借用
    // let r2 = data.borrow_mut();  // パニック! 不変借用中に可変借用
}

// よくある使用パターン: Rc<RefCell<T>>
// 複数の所有者 + 内部可変性
use std::rc::Rc;

#[derive(Debug)]
struct SharedState {
    value: Rc<RefCell<i32>>,
}

impl SharedState {
    fn new(val: i32) -> Self {
        SharedState { value: Rc::new(RefCell::new(val)) }
    }

    fn increment(&self) {
        *self.value.borrow_mut() += 1;
    }

    fn get(&self) -> i32 {
        *self.value.borrow()
    }
}
```

### 6.6 スマートポインタの選択ガイド

| 要件 | 推奨型 | 備考 |
|------|--------|------|
| ヒープ配置、単一所有 | `Box<T>` | 最もシンプル |
| 複数所有（単一スレッド） | `Rc<T>` | 参照カウント |
| 複数所有（マルチスレッド） | `Arc<T>` | アトミック参照カウント |
| 不変参照から内部を変更 | `RefCell<T>` | 実行時チェック |
| 共有 + 変更（単一スレッド） | `Rc<RefCell<T>>` | よくある組み合わせ |
| 共有 + 変更（マルチスレッド） | `Arc<Mutex<T>>` | ロックによる排他制御 |
| 循環参照の親 → 子 | `Rc<T>` / `Arc<T>` | 強い参照 |
| 循環参照の子 → 親 | `Weak<T>` | 弱い参照 |
| 遅延コピー | `Cow<'a, T>` | 必要時のみクローン |

```
スマートポインタ選択フローチャート:

  データの所有パターンは？
       │
  ┌────┼──────────────┐
  単一  共有            共有+変更
  所有  所有(読取のみ)   が必要
  │     │               │
  ▼     ▼               ▼
Box<T>  スレッド跨ぐ？   スレッド跨ぐ？
        │               │
     ┌──┴──┐         ┌──┴──┐
     No    Yes       No    Yes
     │      │        │      │
     ▼      ▼        ▼      ▼
   Rc<T>  Arc<T>  Rc<       Arc<
                  RefCell   Mutex
                  <T>>      <T>>
```

---

## 7. 所有権パターン集 --- 設計への応用

### 7.1 Builder パターンと所有権

Builder パターンは所有権の移動を活用して、メソッドチェーンによるオブジェクト構築を実現する。

```rust
// ===== コード例6: Builder パターン =====

#[derive(Debug)]
struct HttpRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

struct HttpRequestBuilder {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

impl HttpRequestBuilder {
    fn new(url: &str) -> Self {
        HttpRequestBuilder {
            method: "GET".to_string(),
            url: url.to_string(),
            headers: Vec::new(),
            body: None,
        }
    }

    // self を消費して返す → メソッドチェーンが可能
    fn method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.push((key.to_string(), value.to_string()));
        self
    }

    fn body(mut self, body: &str) -> Self {
        self.body = Some(body.to_string());
        self
    }

    fn build(self) -> HttpRequest {
        HttpRequest {
            method: self.method,
            url: self.url,
            headers: self.headers,
            body: self.body,
        }
    }
}

fn main() {
    let request = HttpRequestBuilder::new("https://api.example.com/data")
        .method("POST")
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer token123")
        .body(r#"{"key": "value"}"#)
        .build();

    println!("{:#?}", request);
}
```

### 7.2 所有権による状態マシン

型システムと所有権を使って、不正な状態遷移をコンパイル時に防ぐパターン。

```rust
// ===== コード例7: 型駆動状態マシン =====

// 各状態を型で表現
struct Draft;
struct PendingReview;
struct Published;

struct BlogPost<State> {
    title: String,
    content: String,
    state: std::marker::PhantomData<State>,
}

// Draft 状態でのみ利用可能なメソッド
impl BlogPost<Draft> {
    fn new(title: &str) -> Self {
        BlogPost {
            title: title.to_string(),
            content: String::new(),
            state: std::marker::PhantomData,
        }
    }

    fn add_content(&mut self, text: &str) {
        self.content.push_str(text);
    }

    // Draft → PendingReview への遷移（元の値を消費して新しい型を返す）
    fn request_review(self) -> BlogPost<PendingReview> {
        BlogPost {
            title: self.title,
            content: self.content,
            state: std::marker::PhantomData,
        }
    }
}

// PendingReview 状態でのみ利用可能なメソッド
impl BlogPost<PendingReview> {
    fn approve(self) -> BlogPost<Published> {
        BlogPost {
            title: self.title,
            content: self.content,
            state: std::marker::PhantomData,
        }
    }

    fn reject(self) -> BlogPost<Draft> {
        BlogPost {
            title: self.title,
            content: self.content,
            state: std::marker::PhantomData,
        }
    }
}

// Published 状態でのみ content を公開
impl BlogPost<Published> {
    fn content(&self) -> &str {
        &self.content
    }
}

fn main() {
    let mut post = BlogPost::<Draft>::new("Ownership Guide");
    post.add_content("Rust's ownership system is...");

    // post.content();  // コンパイルエラー! Draft 状態では content() がない
    let post = post.request_review();
    // post.add_content("more");  // コンパイルエラー! PendingReview では変更不可
    let post = post.approve();
    println!("{}", post.content());  // OK: Published 状態でのみ公開
}
```

### 7.3 借用による効率的な API 設計

```rust
// 引数に &str を受け取ると、String と &str の両方を受け付けられる
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    let owned = String::from("Alice");
    let borrowed = "Bob";

    greet(&owned);    // String → &str への自動変換 (Deref)
    greet(borrowed);  // &str はそのまま

    // AsRef トレイトを使うとさらに柔軟に
    fn process_path<P: AsRef<std::path::Path>>(path: P) {
        let path = path.as_ref();
        println!("Processing: {:?}", path);
    }

    process_path("/home/user/file.txt");         // &str
    process_path(String::from("/tmp/data.csv")); // String
    process_path(std::path::PathBuf::from("/var/log"));  // PathBuf
}
```

### 7.4 Cow --- Clone on Write パターン

```rust
use std::borrow::Cow;

// Cow<'a, T>: 必要な時だけクローンする遅延コピー
// - 読み取りのみ → 参照のまま（コスト 0）
// - 変更が必要 → その時点でクローン

fn normalize_name(name: &str) -> Cow<'_, str> {
    if name.contains(char::is_uppercase) {
        // 大文字が含まれる → 変換が必要 → 新しい String を作成
        Cow::Owned(name.to_lowercase())
    } else {
        // 変換不要 → 元の参照をそのまま返す（コスト 0）
        Cow::Borrowed(name)
    }
}

fn main() {
    let name1 = normalize_name("alice");    // Borrowed: クローンなし
    let name2 = normalize_name("BOB");      // Owned: to_lowercase() のコスト
    let name3 = normalize_name("charlie");  // Borrowed: クローンなし

    println!("{}, {}, {}", name1, name2, name3);
}
```

---

## 8. 他言語との比較 --- メモリ安全性アプローチ

### 8.1 各言語のメモリ管理方式の全体像

| 言語 | 方式 | GC 停止 | メモリ安全性 | データ競合防止 | 性能オーバーヘッド |
|------|------|---------|-------------|---------------|------------------|
| **Rust** | 所有権 + 借用 | なし | コンパイル時保証 | コンパイル時保証 | ゼロ |
| C | 手動 (malloc/free) | なし | なし | なし | ゼロ |
| C++ | RAII + スマートポインタ | なし | 部分的（規律依存） | なし | ほぼゼロ |
| Swift | ARC | なし | 高い | 部分的 (Actor) | 低い |
| Go | トレーシング GC | あり（短い） | 高い | 実行時検出 (race detector) | 中程度 |
| Java | 世代別 GC | あり | 高い | 実行時検出 | 中〜高 |
| Python | 参照カウント + GC | あり | 高い (GIL) | GIL で緩和 | 高い |
| Kotlin | JVM の GC | あり | 高い | 実行時検出 | 中〜高 |

### 8.2 C++ との比較 --- RAII とスマートポインタ

```cpp
// C++: RAII + unique_ptr（Rust の Box に相当）
#include <memory>
#include <string>
#include <iostream>

class Resource {
    std::string name_;
public:
    Resource(const std::string& name) : name_(name) {
        std::cout << "Acquired: " << name_ << std::endl;
    }
    ~Resource() {
        std::cout << "Released: " << name_ << std::endl;
    }
};

void cpp_example() {
    // unique_ptr: 単一所有（Rust の Box に近い）
    auto r1 = std::make_unique<Resource>("DB Connection");
    auto r2 = std::move(r1);  // ムーブ（r1 は nullptr になる）
    // r1->...  // 未定義動作! (Rust ではコンパイルエラー)

    // shared_ptr: 共有所有（Rust の Rc/Arc に近い）
    auto r3 = std::make_shared<Resource>("Cache");
    auto r4 = r3;  // 参照カウント +1
}
// r2, r3, r4 がスコープを抜ける → 自動解放
```

```
C++ vs Rust の安全性比較:

  問題                    C++                          Rust
  ─────────────────────────────────────────────────────────────
  ダングリングポインタ    unique_ptr で nullptr化       コンパイルエラー
                          → 実行時の未定義動作の可能性   → ゼロコスト
  データ競合              検出ツール (TSan) で発見       コンパイルエラー
                          → 実行時のみ                  → コンパイル時
  二重解放                スマートポインタで防止         構造的に不可能
                          → 生ポインタでは可能           → ムーブセマンティクス
  Use-After-Free          未定義動作                     コンパイルエラー
                          → バグとして残る可能性         → 完全に防止
```

### 8.3 Swift との比較 --- ARC

```swift
// Swift: ARC（自動参照カウント）
class User {
    var name: String
    var friend: User?  // 強い参照 → 循環参照のリスク

    init(name: String) {
        self.name = name
        print("User \(name) created")
    }

    deinit {
        print("User \(name) deallocated")
    }
}

// 循環参照の例
var alice: User? = User(name: "Alice")
var bob: User? = User(name: "Bob")
alice?.friend = bob    // Alice → Bob (強い参照)
bob?.friend = alice    // Bob → Alice (強い参照) → 循環参照!
alice = nil
bob = nil
// 両方 nil にしても deinit は呼ばれない → メモリリーク!

// 解決: weak または unowned を使う
class SafeUser {
    var name: String
    weak var friend: SafeUser?  // 弱い参照

    init(name: String) { self.name = name }
    deinit { print("SafeUser \(name) deallocated") }
}
```

```
Rust vs Swift の所有権モデル比較:

  特性                    Rust                        Swift
  ─────────────────────────────────────────────────────────────
  所有権モデル            静的所有権 + 借用            ARC（参照カウント）
  メモリ安全性検証時期    コンパイル時                 実行時
  実行時オーバーヘッド    ゼロ                        カウント操作のコスト
  循環参照                Weak<T> で防止              weak/unowned で防止
                          コンパイル時に構造を強制     プログラマの判断に依存
  値型 vs 参照型          全てが値型（ムーブ）        struct = 値型
                                                      class = 参照型
  並行安全性              Send/Sync トレイト          Actor モデル
                          コンパイル時保証             実行時検証
```

### 8.4 Go/Java との比較 --- GC ベースのアプローチ

```go
// Go: ガベージコレクション
package main

import "fmt"

func main() {
    // Go では所有権の概念がない
    // GC が不要なメモリを自動回収
    s1 := "hello"
    s2 := s1  // コピー（文字列は不変）
    fmt.Println(s1, s2)

    // スライスは参照型（暗黙的な共有）
    a := []int{1, 2, 3}
    b := a  // 浅いコピー（同じ配列を指す）
    b[0] = 100
    fmt.Println(a)  // [100 2 3]  ← a も変わっている!
    // Rust ではこのような暗黙的な共有はムーブで防がれる
}
```

```
GC vs 所有権の性能特性:

  ┌───────────────────────────────────────────────────────┐
  │    レイテンシの比較（概念図）                          │
  │                                                       │
  │  Rust (所有権):                                       │
  │  ──────────────────────────────────── 一定のレイテンシ │
  │                                                       │
  │  Go (GC):                                             │
  │  ─────────┬──────────────┬─────────── GC停止が散発    │
  │           |              |                            │
  │           GC pause       GC pause                     │
  │           (~1ms)         (~1ms)                       │
  │                                                       │
  │  Java (GC, ZGC以前):                                  │
  │  ──────────────┬─────────────────── 長いGC停止        │
  │                |                                      │
  │                GC pause                               │
  │                (~10-100ms)                            │
  │                                                       │
  │  適用領域:                                            │
  │  - Rust: リアルタイムシステム、OS、ゲームエンジン     │
  │  - Go: Web サーバー、マイクロサービス                 │
  │  - Java: エンタープライズ、大規模 Web アプリ          │
  └───────────────────────────────────────────────────────┘
```
```

