# 所有権と借用 -- Rustの最も革新的なメモリ管理パラダイム

> 所有権(Ownership)と借用(Borrowing)はRust独自のメモリ管理モデルであり、ガベージコレクタなしでメモリ安全とデータ競合防止をコンパイル時に保証する。

---

## この章で学ぶこと

1. **所有権の3つの規則** -- 各値は唯一の所有者を持ち、スコープを抜けると解放される仕組みを理解する
2. **ムーブとコピー** -- 値の移動と複製の違い、Copy/Clone トレイトの使い分けを習得する
3. **借用とライフタイム基礎** -- 不変参照・可変参照の規則とライフタイムの入門を学ぶ
4. **実践的なパターン** -- 所有権を活かした関数設計、構造体の設計パターンを身につける

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

これら3つの規則はRustのメモリ管理の根幹を成す。C/C++ ではプログラマが手動でメモリを管理し、Java/Python ではガベージコレクタが自動管理する。Rustは第三の道として、コンパイル時に所有権を追跡することで、実行時コストゼロのメモリ管理を実現する。

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

変数 `s` が中括弧のスコープを抜けると、Rustは自動的に `drop` 関数を呼び出してメモリを解放する。これは C++ の RAII（Resource Acquisition Is Initialization）パターンに類似しているが、Rustでは所有権の概念により、ダブルフリーやダングリングポインタが構造的に排除される。

### 例2: ムーブセマンティクス

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;                    // ムーブ: s1 → s2
    // println!("{}", s1);          // エラー: s1 は無効化済み
    println!("{}", s2);             // OK
}
```

`String` 型はヒープ上にデータを持つため、代入時に「ムーブ」が発生する。ムーブとは所有権の移転であり、元の変数は無効化される。これにより、同じヒープ領域を2つの変数が所有する状態（ダブルフリーの原因）が防止される。

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

### 1.3 ムーブが発生する場面

ムーブはさまざまな場面で暗黙的に発生する。どのような操作でムーブが起こるかを理解することは、Rustプログラミングにおいて極めて重要である。

```rust
fn main() {
    let s = String::from("hello");

    // (1) 変数束縛でムーブ
    let s2 = s;
    // s は無効

    // (2) 関数への引数渡しでムーブ
    let s3 = String::from("world");
    takes_string(s3);
    // s3 は無効

    // (3) 関数からの戻り値でムーブ
    let s4 = gives_string();
    // s4 が所有権を受け取る

    // (4) コレクションへの挿入でムーブ
    let s5 = String::from("item");
    let mut v = Vec::new();
    v.push(s5);
    // s5 は無効（Vec が所有権を持つ）

    // (5) パターンマッチでのムーブ
    let opt = Some(String::from("data"));
    if let Some(inner) = opt {
        println!("{}", inner);
    }
    // opt は無効（inner にムーブ済み）

    // (6) 構造体の構築でムーブ
    let name = String::from("太郎");
    let user = User { name };   // name は無効
    println!("{}", user.name);   // OK: user.name としてアクセス
}

fn takes_string(s: String) {
    println!("受け取った: {}", s);
    // s はこの関数の終了時に drop される
}

fn gives_string() -> String {
    let s = String::from("新しい文字列");
    s // 所有権を呼び出し元に返す
}

struct User {
    name: String,
}
```

### 1.4 Drop トレイトとRAII

Rustでは所有者がスコープを抜けると、自動的に `Drop` トレイトの `drop` メソッドが呼ばれる。これを利用して、ファイルハンドル、ネットワーク接続、ロックなどのリソースを自動的に解放できる。

```rust
struct DatabaseConnection {
    url: String,
    connected: bool,
}

impl DatabaseConnection {
    fn new(url: &str) -> Self {
        println!("接続を開きます: {}", url);
        DatabaseConnection {
            url: url.to_string(),
            connected: true,
        }
    }

    fn query(&self, sql: &str) -> Vec<String> {
        println!("クエリ実行: {}", sql);
        vec!["結果1".to_string(), "結果2".to_string()]
    }
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        println!("接続を閉じます: {}", self.url);
        self.connected = false;
    }
}

fn main() {
    {
        let conn = DatabaseConnection::new("postgres://localhost/mydb");
        let results = conn.query("SELECT * FROM users");
        println!("結果: {:?}", results);
    } // conn がスコープを抜ける → drop() が自動呼び出し → 接続クローズ

    println!("接続は自動的に閉じられました");
}
```

### 1.5 スタック上のデータとヒープ上のデータ

```
┌────────────────────────────────────────────────────────────┐
│              メモリ配置とムーブの関係                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  スタックのみ（Copy型）        ヒープ使用（非Copy型）       │
│  ┌─────┐   コピー  ┌─────┐    ┌─────┐  ムーブ ┌─────┐    │
│  │  42 │ ───────> │  42 │    │ ptr │ ─────> │ ptr │    │
│  └─────┘           └─────┘    │ len │        │ len │    │
│  i32: 両方有効                │ cap │        │ cap │    │
│                               └──┬──┘        └──┬──┘    │
│                                  │ (無効)       │        │
│                                  └──────┐   ┌───┘        │
│                                         ▼   ▼            │
│                                    ┌──────────┐          │
│                                    │ヒープデータ│          │
│                                    └──────────┘          │
│                                    1つのポインタのみ有効   │
└────────────────────────────────────────────────────────────┘
```

---

## 2. コピーとクローン

### 例3: Copy トレイト(スタック上の値)

```rust
fn main() {
    let x: i32 = 42;
    let y = x;          // コピー（i32 は Copy トレイト実装済み）
    println!("x={}, y={}", x, y); // 両方有効！

    // タプルも全要素がCopyなら Copy
    let point = (3, 4);
    let point2 = point;
    println!("point={:?}, point2={:?}", point, point2); // 両方有効

    // 配列も要素が Copy なら Copy
    let arr = [1, 2, 3, 4, 5];
    let arr2 = arr;
    println!("arr={:?}, arr2={:?}", arr, arr2); // 両方有効

    // 参照も Copy
    let s = String::from("hello");
    let r1 = &s;
    let r2 = r1;  // 参照のコピー（String 自体はコピーされない）
    println!("r1={}, r2={}", r1, r2); // 両方有効
}
```

### 例4: Clone トレイト(明示的な深いコピー)

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1.clone();           // 明示的にヒープデータもコピー
    println!("s1={}, s2={}", s1, s2); // 両方有効

    // Vec の clone
    let v1 = vec![1, 2, 3, 4, 5];
    let v2 = v1.clone();
    println!("v1={:?}, v2={:?}", v1, v2);

    // ネストしたデータ構造の clone
    let nested = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
    ];
    let nested_clone = nested.clone(); // 全てのデータが深くコピーされる
    println!("nested={:?}", nested);
    println!("nested_clone={:?}", nested_clone);
}
```

### Clone の図解

```
  clone() 前:
  s1                     ヒープ
  ┌──────────┐          ┌───────────┐
  │ ptr ────────────────>│ h e l l o │
  │ len: 5   │          └───────────┘
  │ cap: 5   │
  └──────────┘

  clone() 後:
  s1                     ヒープ
  ┌──────────┐          ┌───────────┐
  │ ptr ────────────────>│ h e l l o │  ← 元データ
  │ len: 5   │          └───────────┘
  │ cap: 5   │
  └──────────┘
                         ┌───────────┐
  s2                     │ h e l l o │  ← 新しいコピー
  ┌──────────┐          └───────────┘
  │ ptr ────────────────>│
  │ len: 5   │
  │ cap: 5   │
  └──────────┘
  独立した2つのヒープ領域が存在する
```

### Copy が実装される型と実装されない型

```
┌─────────────────────────────┬────────────────────────────┐
│   Copy される型             │   Copy されない型           │
├─────────────────────────────┼────────────────────────────┤
│ i8, i16, i32, i64, i128    │ String                     │
│ u8, u16, u32, u64, u128    │ Vec<T>                     │
│ f32, f64                    │ Box<T>                     │
│ bool                        │ HashMap<K, V>              │
│ char                        │ HashSet<T>                 │
│ isize, usize               │ File, TcpStream            │
│ (i32, bool) -- 全要素Copy  │ Rc<T>, Arc<T>              │
│ [i32; 5] -- 固定長配列      │ MutexGuard<T>              │
│ &T -- 不変参照              │ &mut T -- 可変参照         │
│ fn ポインタ                 │ クロージャ（キャプチャ次第）│
│ *const T, *mut T -- 生ポ   │ dyn Trait                  │
└─────────────────────────────┴────────────────────────────┘
```

### 2.1 Copy トレイトの自作実装

```rust
// Copy を derive するには、全フィールドが Copy でなければならない
#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

// Copy できない構造体
// #[derive(Clone, Copy)]  // コンパイルエラー！ String は Copy ではない
#[derive(Debug, Clone)]
struct NamedPoint {
    name: String,
    x: f64,
    y: f64,
}

fn main() {
    let p1 = Point { x: 1.0, y: 2.0 };
    let p2 = p1;           // Copy
    println!("p1={:?}", p1); // OK: p1 はまだ有効

    let np1 = NamedPoint {
        name: "原点".to_string(),
        x: 0.0,
        y: 0.0,
    };
    let np2 = np1.clone();  // Clone 必須
    // let np3 = np1;       // ムーブ! np1 は無効になる
    println!("np2={:?}", np2);
}
```

### 2.2 Copy と Clone の関係

```
┌─────────────────────────────────────────────────────┐
│             Copy と Clone の関係                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Copy は Clone のサブトレイト                         │
│  (Copy を実装するには Clone も必要)                   │
│                                                      │
│  pub trait Copy: Clone { }                           │
│                                                      │
│  Copy の意味:                                        │
│  - ビット単位のコピーで安全な型                       │
│  - 暗黙的にコピーされる（代入・関数引数渡し時）       │
│  - ヒープアロケーションを持たない型のみ              │
│                                                      │
│  Clone の意味:                                       │
│  - 明示的な深いコピーを提供する型                     │
│  - .clone() の呼び出しが必要                         │
│  - 任意の型に実装可能（ヒープアロケーション含む）     │
│                                                      │
│  ┌────────────────┐                                  │
│  │    Clone        │                                 │
│  │  ┌──────────┐  │                                  │
│  │  │   Copy   │  │                                  │
│  │  │ i32,bool │  │                                  │
│  │  │ f64,char │  │                                  │
│  │  └──────────┘  │                                  │
│  │  String, Vec   │                                  │
│  │  HashMap       │                                  │
│  └────────────────┘                                  │
└─────────────────────────────────────────────────────┘
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

これらの規則は、Rustがコンパイル時にデータ競合を防止するための中核メカニズムである。データ競合は以下の3つの条件が同時に満たされたときに発生する:

1. 2つ以上のポインタが同じデータに同時にアクセスする
2. 少なくとも1つのポインタがデータに書き込みを行う
3. データへのアクセスを同期するメカニズムがない

Rustの借用規則は、条件1と2の組み合わせをコンパイル時に排除することで、データ競合を構造的に不可能にする。

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

`&` 記号を使って参照を作成する。参照は値の所有権を持たないため、参照がスコープを抜けても元の値は解放されない。

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

可変参照 `&mut` を使えば、借用先で値を変更できる。ただし、可変参照は同時に1つしか存在できない。

### 例7: 借用規則の違反と NLL

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

NLL（Non-Lexical Lifetimes）は Rust 2018 Edition で導入された機能で、参照のライフタイムがレキシカルスコープ（中括弧の範囲）ではなく、「最後に使用された地点」で終了するようになった。これにより、上記のコードは正しくコンパイルされる。

### 例8: 借用規則の違反（コンパイルエラー）

```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;          // 不変参照
    let r2 = &mut s;      // エラー！不変参照が生きている間に可変参照は作れない
    println!("{}", r1);    // r1 がまだ使われている
}
```

```
error[E0502]: cannot borrow `s` as mutable because it is also borrowed as immutable
 --> src/main.rs:4:14
  |
3 |     let r1 = &s;
  |              -- immutable borrow occurs here
4 |     let r2 = &mut s;
  |              ^^^^^^ mutable borrow occurs here
5 |     println!("{}", r1);
  |                    -- immutable borrow later used here
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

### 3.3 可変参照の排他性が重要な理由

```rust
// もし同時に2つの可変参照が許されたら...
// （以下は架空の危険な例：実際のRustではコンパイルエラー）
fn hypothetical_danger() {
    let mut data = vec![1, 2, 3];

    // 仮に2つの可変参照が同時に存在できたとすると:
    // let r1 = &mut data;  // 可変参照1
    // let r2 = &mut data;  // 可変参照2（実際はエラー）

    // r1.push(4);          // Vec がリアロケーションを起こす可能性
    // println!("{}", r2[0]); // r2 は無効なメモリを参照！
    //                        // → use-after-free の脆弱性
}

// Rustはこれをコンパイル時に防止する
fn safe_version() {
    let mut data = vec![1, 2, 3];

    // 可変参照は1つだけ
    let r1 = &mut data;
    r1.push(4);
    // r1 のライフタイム終了

    // 新しい可変参照を取得
    let r2 = &mut data;
    println!("{}", r2[0]); // 安全
}
```

### 3.4 再借用（Reborrowing）

```rust
fn main() {
    let mut s = String::from("hello");
    let r = &mut s;

    // 再借用: 可変参照から不変参照を作る
    let r2 = &*r;  // 再借用（暗黙的にも起こる）
    println!("{}", r2);

    // 再借用: 可変参照から一時的な可変参照を作る
    modify(r);  // &mut String が &mut String として再借用される
    println!("{}", r);
}

fn modify(s: &mut String) {
    s.push_str(", world!");
}
```

再借用は、既存の参照から一時的に別の参照を作成する仕組みである。関数に `&mut` 引数を渡すとき、元の可変参照は一時的に「凍結」され、関数の実行が終わると再び使用可能になる。

---

## 4. 関数と所有権

### 例9: 所有権の移動と返却

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

### 例10: 参照を使った関数設計のベストプラクティス

```rust
// パターン1: 読み取りのみ → 不変参照 &T
fn print_info(s: &str) {
    println!("文字列: {}, 長さ: {}", s, s.len());
}

// パターン2: 変更が必要 → 可変参照 &mut T
fn make_uppercase(s: &mut String) {
    *s = s.to_uppercase();
}

// パターン3: 所有権が必要 → T（値渡し）
fn into_bytes(s: String) -> Vec<u8> {
    s.into_bytes()  // String を消費して Vec<u8> を返す
}

// パターン4: 条件付きで所有権を取る → Cow (Clone on Write)
use std::borrow::Cow;

fn ensure_uppercase(s: &str) -> Cow<'_, str> {
    if s.chars().all(|c| c.is_uppercase()) {
        Cow::Borrowed(s)        // 変更不要: 借用をそのまま返す
    } else {
        Cow::Owned(s.to_uppercase()) // 変更必要: 新しい String を返す
    }
}

fn main() {
    let s = String::from("hello");

    // パターン1: 不変参照
    print_info(&s);
    println!("s はまだ使える: {}", s);

    // パターン2: 可変参照
    let mut s2 = String::from("hello");
    make_uppercase(&mut s2);
    println!("大文字: {}", s2);

    // パターン3: 所有権の移動
    let s3 = String::from("hello");
    let bytes = into_bytes(s3);
    // s3 は使えない
    println!("バイト: {:?}", bytes);

    // パターン4: Cow
    let result = ensure_uppercase("HELLO");
    println!("Cow: {}", result);
}
```

### 例11: スライスによる効率的な借用

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

fn longest_word(s: &str) -> &str {
    s.split_whitespace()
        .max_by_key(|word| word.len())
        .unwrap_or("")
}

fn main() {
    let sentence = String::from("hello world foo bar");
    let word = first_word(&sentence);
    println!("最初の単語: {}", word); // "hello"

    let longest = longest_word(&sentence);
    println!("最長の単語: {}", longest); // "hello" or "world"

    // 文字列リテラルもスライスとして渡せる
    let word2 = first_word("good morning");
    println!("最初の単語: {}", word2); // "good"
}
```

### 例12: 構造体での借用と所有権

```rust
// 所有型を使う構造体（最も一般的）
struct OwnedUser {
    name: String,
    email: String,
}

// 借用型を使う構造体（ライフタイム注釈が必要）
struct BorrowedUser<'a> {
    name: &'a str,
    email: &'a str,
}

// 使い分けの例
fn create_owned_user(name: &str, email: &str) -> OwnedUser {
    OwnedUser {
        name: name.to_string(),
        email: email.to_string(),
    }
}

fn create_borrowed_user<'a>(name: &'a str, email: &'a str) -> BorrowedUser<'a> {
    BorrowedUser { name, email }
}

fn main() {
    // 所有型: 構造体がデータを所有するため、ライフタイムの制約がない
    let owned = create_owned_user("田中", "tanaka@example.com");
    println!("{}: {}", owned.name, owned.email);

    // 借用型: 元データより長く生きることはできない
    let name = String::from("鈴木");
    let email = String::from("suzuki@example.com");
    let borrowed = create_borrowed_user(&name, &email);
    println!("{}: {}", borrowed.name, borrowed.email);
    // name, email はここで有効 → borrowed も有効
}
```

---

## 5. ライフタイムの基礎

### 5.1 ライフタイム注釈の基本

ライフタイム注釈は参照がどのくらいの期間有効であるかをコンパイラに伝える仕組みである。ライフタイム注釈自体は参照の寿命を変更するものではなく、複数の参照間の関係をコンパイラに説明するものである。

```rust
// ライフタイム注釈なし（コンパイルエラー）
// fn longest(x: &str, y: &str) -> &str {
//     if x.len() > y.len() { x } else { y }
// }

// ライフタイム注釈あり
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

fn main() {
    let string1 = String::from("長い文字列");
    let result;
    {
        let string2 = String::from("短い");
        result = longest(string1.as_str(), string2.as_str());
        println!("長い方: {}", result); // OK: string2 はまだ有効
    }
    // println!("{}", result); // エラー: string2 のライフタイムが切れている
}
```

### 5.2 ライフタイム省略規則

Rustコンパイラには3つのライフタイム省略規則があり、多くの場合は明示的なライフタイム注釈を書く必要がない。

```rust
// 規則1: 各入力参照に固有のライフタイムが割り当てられる
fn first(s: &str) -> &str { &s[..1] }
// 展開: fn first<'a>(s: &'a str) -> &'a str

// 規則2: 入力参照が1つだけなら、その参照のライフタイムが全出力に適用
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}
// 展開: fn first_word<'a>(s: &'a str) -> &'a str

// 規則3: メソッドの場合、&self のライフタイムが出力に適用
struct Parser {
    input: String,
}

impl Parser {
    fn first_token(&self) -> &str {
        self.input.split_whitespace().next().unwrap_or("")
    }
    // 展開: fn first_token<'a>(&'a self) -> &'a str
}

// 規則が適用できない場合は明示的なライフタイム注釈が必要
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 5.3 'static ライフタイム

```rust
// 'static ライフタイム: プログラム全体の実行期間中有効
fn get_greeting() -> &'static str {
    "こんにちは！"  // 文字列リテラルは 'static
}

// 定数も 'static
static GLOBAL_CONFIG: &str = "デフォルト設定";

fn main() {
    let greeting = get_greeting();
    println!("{}", greeting);
    println!("{}", GLOBAL_CONFIG);

    // T: 'static は「所有型」であることを意味する場合もある
    // String は 'static を満たす（参照を含まないため）
    fn takes_owned<T: 'static>(value: T) {
        // T は参照を含まない型
        std::mem::drop(value);
    }

    takes_owned(String::from("hello")); // OK
    takes_owned(42i32);                  // OK
    // takes_owned(&String::from("hello")); // エラー: 一時的な参照は 'static ではない
}
```

---

## 6. 高度な所有権パターン

### 6.1 内部可変性（Interior Mutability）

```rust
use std::cell::{Cell, RefCell};

// Cell<T>: Copy な型に対する内部可変性
struct Counter {
    count: Cell<u32>,
}

impl Counter {
    fn new() -> Self {
        Counter { count: Cell::new(0) }
    }

    fn increment(&self) {
        // &self（不変参照）なのに内部状態を変更できる
        self.count.set(self.count.get() + 1);
    }

    fn get(&self) -> u32 {
        self.count.get()
    }
}

// RefCell<T>: 任意の型に対する内部可変性（実行時チェック）
struct CachedValue {
    value: String,
    cache: RefCell<Option<String>>,
}

impl CachedValue {
    fn new(value: String) -> Self {
        CachedValue {
            value,
            cache: RefCell::new(None),
        }
    }

    fn get_computed(&self) -> String {
        // &self なのに cache を変更できる
        let mut cache = self.cache.borrow_mut();
        if cache.is_none() {
            println!("キャッシュ計算中...");
            *cache = Some(format!("computed_{}", self.value));
        }
        cache.clone().unwrap()
    }
}

fn main() {
    let counter = Counter::new();
    counter.increment();
    counter.increment();
    counter.increment();
    println!("カウント: {}", counter.get()); // 3

    let cached = CachedValue::new("hello".to_string());
    println!("{}", cached.get_computed()); // "キャッシュ計算中..." → "computed_hello"
    println!("{}", cached.get_computed()); // キャッシュヒット → "computed_hello"
}
```

### 6.2 スマートポインタと所有権

```rust
use std::rc::Rc;

// Box<T>: ヒープ上への単一所有権
fn box_example() {
    let b = Box::new(5);
    println!("Box: {}", b);

    // 再帰的なデータ構造
    enum List {
        Cons(i32, Box<List>),
        Nil,
    }

    let list = List::Cons(1,
        Box::new(List::Cons(2,
            Box::new(List::Cons(3,
                Box::new(List::Nil))))));
}

// Rc<T>: 参照カウントによる共有所有権（シングルスレッド）
fn rc_example() {
    let a = Rc::new(String::from("共有データ"));
    println!("参照カウント: {}", Rc::strong_count(&a)); // 1

    let b = Rc::clone(&a);  // 参照カウントを増やす（データのコピーではない）
    println!("参照カウント: {}", Rc::strong_count(&a)); // 2

    {
        let c = Rc::clone(&a);
        println!("参照カウント: {}", Rc::strong_count(&a)); // 3
    }
    // c がドロップ
    println!("参照カウント: {}", Rc::strong_count(&a)); // 2

    println!("a={}, b={}", a, b);
}

fn main() {
    box_example();
    rc_example();
}
```

### 6.3 所有権とパターンマッチング

```rust
enum Message {
    Text(String),
    Number(i32),
    Pair(String, String),
}

fn process_message(msg: Message) {
    match msg {
        // msg の所有権はムーブされる
        Message::Text(text) => {
            println!("テキスト: {}", text);
            // text を所有している
        }
        Message::Number(n) => {
            println!("数値: {}", n);
        }
        Message::Pair(a, b) => {
            println!("ペア: {} / {}", a, b);
        }
    }
    // msg は完全にムーブされたため使えない
}

fn process_message_ref(msg: &Message) {
    match msg {
        // 参照のパターンマッチでは借用のみ
        Message::Text(text) => {
            println!("テキスト: {}", text);
            // text は &String
        }
        Message::Number(n) => {
            println!("数値: {}", n);
        }
        Message::Pair(a, b) => {
            println!("ペア: {} / {}", a, b);
        }
    }
    // msg はまだ使える
}

fn main() {
    let msg = Message::Text("hello".to_string());
    process_message_ref(&msg);
    process_message_ref(&msg);  // OK: 参照なので再利用可能

    let msg2 = Message::Pair("左".to_string(), "右".to_string());
    process_message(msg2);
    // process_message(msg2);  // エラー: ムーブ済み
}
```

---

## 7. 比較表

### 7.1 ムーブ vs コピー vs クローン

| 操作 | ヒープコピー | 元の値 | 自動/明示 | コスト | 用途 |
|------|------------|--------|-----------|--------|------|
| ムーブ | なし | 無効化 | 自動 | O(1) | 所有権の移転 |
| コピー (Copy) | N/A (スタックのみ) | 有効 | 自動 | O(1) | 小さな値の複製 |
| クローン (Clone) | あり | 有効 | 明示 (.clone()) | O(n) | 深いコピーが必要な場合 |
| 参照 (&T) | なし | 有効 | 明示 (&) | O(1) | 読み取り専用アクセス |

### 7.2 不変参照 vs 可変参照

| 特性 | `&T` (不変参照) | `&mut T` (可変参照) |
|------|-----------------|---------------------|
| 同時に持てる数 | 複数 | 1つだけ |
| データの変更 | 不可 | 可能 |
| 別名 | 共有参照 (shared ref) | 排他参照 (exclusive ref) |
| Send/Sync | T: Sync なら安全 | T: Send なら安全 |
| 他の参照と共存 | &mut T と共存不可 | &T と共存不可 |
| コンパイラの最適化 | エイリアシング最適化可 | 排他性により強力な最適化 |

### 7.3 所有型 vs 借用型の使い分け

| 状況 | 推奨 | 理由 |
|------|------|------|
| 構造体のフィールド | String（所有型） | ライフタイムの複雑さを回避 |
| 関数の引数（読み取り） | &str（借用） | 柔軟性が高い |
| 関数の引数（変更） | &mut String | 呼び出し元がまだ使える |
| 関数の引数（消費） | String（所有型） | 値を消費する場合 |
| 戻り値（新しい値） | String（所有型） | ローカル変数への参照は返せない |
| 戻り値（引数の一部） | &str + ライフタイム | 効率的 |
| 短命な一時構造体 | &str + ライフタイム | パフォーマンス重視 |

---

## 8. アンチパターン

### アンチパターン1: 必要以上のクローン

```rust
// BAD: 参照で十分なのにクローンする
fn print_length(s: String) {
    println!("長さ: {}", s.len());
}
fn bad_example() {
    let s = String::from("hello");
    print_length(s.clone()); // 不要なクローン
    print_length(s.clone()); // また不要なクローン
    println!("{}", s);
}

// GOOD: 参照を使う
fn print_length_good(s: &str) {
    println!("長さ: {}", s.len());
}
fn good_example() {
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

### アンチパターン3: 不必要な可変参照

```rust
// BAD: 変更しないのに &mut を使う
fn just_read(data: &mut Vec<i32>) -> i32 {
    data.iter().sum()
}

// GOOD: 読み取りだけなら不変参照で十分
fn just_read_good(data: &[i32]) -> i32 {
    data.iter().sum()
}
```

### アンチパターン4: 構造体フィールドに参照を使いすぎる

```rust
// BAD: 不必要にライフタイムが伝播する
struct Config<'a> {
    host: &'a str,
    port: u16,
    database: &'a str,
}

// このような構造体を返す関数は非常に複雑になる
// fn load_config<'a>() -> Config<'a> { ... }  // ← ライフタイムの管理が困難

// GOOD: 所有型を使う
struct ConfigGood {
    host: String,
    port: u16,
    database: String,
}

fn load_config() -> ConfigGood {
    ConfigGood {
        host: "localhost".to_string(),
        port: 5432,
        database: "myapp".to_string(),
    }
}
```

### アンチパターン5: イテレータ使用中のコレクション変更

```rust
fn main() {
    let mut v = vec![1, 2, 3, 4, 5];

    // BAD: イテレーション中にコレクションを変更しようとする
    // for item in &v {
    //     if *item > 3 {
    //         v.push(*item * 2);  // エラー！不変借用中に可変借用できない
    //     }
    // }

    // GOOD: 結果を別のコレクションに集めてから追加
    let additions: Vec<i32> = v.iter()
        .filter(|&&x| x > 3)
        .map(|&x| x * 2)
        .collect();
    v.extend(additions);
    println!("{:?}", v); // [1, 2, 3, 4, 5, 8, 10]

    // GOOD: retain で条件に合わない要素を除去
    let mut v2 = vec![1, 2, 3, 4, 5];
    v2.retain(|&x| x % 2 == 0);
    println!("{:?}", v2); // [2, 4]
}
```

---

## 9. 実践例: 所有権を活かした設計

### 9.1 状態機械パターン

```rust
// 所有権を使って状態遷移を型レベルで表現
struct Idle;
struct Running {
    start_time: std::time::Instant,
}
struct Finished {
    duration: std::time::Duration,
}

struct Task<State> {
    name: String,
    state: State,
}

impl Task<Idle> {
    fn new(name: &str) -> Self {
        Task {
            name: name.to_string(),
            state: Idle,
        }
    }

    // self を消費して新しい状態の Task を返す
    fn start(self) -> Task<Running> {
        println!("タスク '{}' を開始", self.name);
        Task {
            name: self.name,
            state: Running {
                start_time: std::time::Instant::now(),
            },
        }
    }
}

impl Task<Running> {
    fn finish(self) -> Task<Finished> {
        let duration = self.state.start_time.elapsed();
        println!("タスク '{}' が完了 ({:?})", self.name, duration);
        Task {
            name: self.name,
            state: Finished { duration },
        }
    }
}

impl Task<Finished> {
    fn report(&self) {
        println!("レポート: '{}' は {:?} で完了", self.name, self.state.duration);
    }
}

fn main() {
    let task = Task::new("データ処理");
    // task.finish();  // コンパイルエラー！Idle から直接 Finished にはなれない
    let running = task.start();
    // task.start();   // コンパイルエラー！task は既にムーブ済み
    let finished = running.finish();
    finished.report();
}
```

### 9.2 所有権を活かしたリソース管理

```rust
use std::fs::File;
use std::io::{self, Write, BufWriter};

// ファイルの所有権を持つ構造体
struct Logger {
    writer: BufWriter<File>,
    count: u64,
}

impl Logger {
    fn new(path: &str) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Logger {
            writer: BufWriter::new(file),
            count: 0,
        })
    }

    fn log(&mut self, message: &str) -> io::Result<()> {
        self.count += 1;
        writeln!(self.writer, "[{}] {}", self.count, message)?;
        Ok(())
    }

    // 所有権を消費してファイルを確実にフラッシュ
    fn close(mut self) -> io::Result<()> {
        self.writer.flush()?;
        println!("ログファイルを閉じました（{} 件のログ）", self.count);
        Ok(())
        // self がドロップされ、File が自動的にクローズされる
    }
}

fn main() -> io::Result<()> {
    let mut logger = Logger::new("/tmp/app.log")?;
    logger.log("アプリケーション開始")?;
    logger.log("処理実行中")?;
    logger.log("アプリケーション終了")?;
    logger.close()?;
    // logger は使えない（close で消費済み）
    // logger.log("もう一つ");  // コンパイルエラー！
    Ok(())
}
```

---

## 10. FAQ

### Q1: いつムーブが起こりますか？

**A:** 以下のケースでムーブが発生します:
- `let y = x;` (Copy未実装の型)
- 関数に値を渡す: `func(x)`
- 関数から値を返す: `return x`
- コレクションに値を入れる: `vec.push(x)`
- パターンマッチで値を取り出す: `if let Some(v) = opt`
- 構造体のフィールド初期化: `Struct { field: x }`

Copy トレイトを実装している型（i32, bool, f64 など）はムーブではなくコピーされます。

### Q2: `&str` と `&String` の違いは何ですか？

**A:** `&str` は文字列スライスで、文字列データへの参照+長さの情報を持つ「ファットポインタ」です。`&String` は String 型への参照です。関数の引数には `&str` を使うのが慣例です。`&String` は自動的に `&str` にデリファレンスされるため（Deref coercion）、`&str` の方がより汎用的です。

```rust
fn accepts_str(s: &str) {
    println!("{}", s);
}

fn main() {
    let owned = String::from("hello");
    let literal = "world";

    accepts_str(&owned);    // &String → &str (Deref coercion)
    accepts_str(literal);   // &str そのまま
    accepts_str(&owned[1..]); // スライスも渡せる
}
```

### Q3: なぜ可変参照は同時に1つだけなのですか？

**A:** データ競合を防止するためです。データ競合は以下の3条件が揃うと発生します:
1. 2つ以上のポインタが同じデータにアクセス
2. 少なくとも1つが書き込み
3. アクセスの同期がない

可変参照を1つに制限することで、条件1,2の組み合わせをコンパイル時に排除できます。

### Q4: Clone と Copy の使い分けは？

**A:**
- **Copy**: スタック上の小さな値（i32, f64, bool など）。暗黙的に複製される
- **Clone**: ヒープデータを含む型（String, Vec など）。`.clone()` の明示的な呼び出しが必要

自作の型に Copy を実装するには、全フィールドが Copy でなければなりません。Copy は「安いコピー」を意味し、Clone は「任意のコスト」を意味します。

### Q5: ライフタイムはいつ明示的に書く必要がありますか？

**A:** コンパイラの省略規則（elision rules）で推論できない場合に書く必要があります。主に:
- 複数の入力参照がある関数で、戻り値に参照を含む場合
- 構造体に参照フィールドがある場合
- トレイト実装で参照のライフタイム関係が複雑な場合

```rust
// 省略規則で推論できる → 注釈不要
fn first(s: &str) -> &str { &s[..1] }

// 複数の入力参照 → 注釈必要
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// 構造体の参照フィールド → 注釈必要
struct Excerpt<'a> {
    text: &'a str,
}
```

### Q6: RefCell<T> はいつ使うべきですか？

**A:** 借用規則をコンパイル時ではなく実行時にチェックしたい場合に使います。典型的な用途:
- 不変参照を通じて内部状態を変更したい場合（内部可変性パターン）
- トレイトオブジェクトの内部状態を変更する場合
- コンパイラが安全性を証明できないが、プログラマが安全だと確信している場合

ただし、RefCell は実行時にパニックする可能性があるため、可能な限り通常の借用を使うべきです。

---

## 11. まとめ

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
| ライフタイム | 参照の有効期間をコンパイラに伝える注釈 |
| Drop | スコープ終了時に自動呼び出されるデストラクタ |
| 内部可変性 | Cell/RefCell で不変参照を通じた変更を実現 |
| スマートポインタ | Box/Rc/Arc で所有権のパターンを拡張 |

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
5. **Rust API Guidelines - Ownership** -- https://rust-lang.github.io/api-guidelines/ownership.html
6. **Learning Rust With Entirely Too Many Linked Lists** -- https://rust-unofficial.github.io/too-many-lists/
