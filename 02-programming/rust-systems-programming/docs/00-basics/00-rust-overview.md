# Rust概要 -- 安全性・パフォーマンス・所有権が融合したシステムプログラミング言語

> Rustは「安全性」「速度」「並行性」の3つを同時に達成する唯一のシステムプログラミング言語であり、ガベージコレクタなしでメモリ安全を保証する。

---

## この章で学ぶこと

1. **Rustの設計哲学** -- ゼロコスト抽象化・所有権・型安全の3本柱を理解する
2. **他言語との違い** -- C/C++/Go/Pythonとの比較で立ち位置を把握する
3. **エコシステムの全体像** -- Cargo、crates.io、ツールチェーンの構成を掴む
4. **言語の基本構文** -- 変数、関数、制御フロー、パターンマッチングの基礎を学ぶ
5. **実践的な開発フロー** -- プロジェクト作成からテスト・ドキュメント生成までの一連の流れを体験する

---

## 1. Rustの歴史と設計哲学

### 1.1 誕生の経緯

```
+-----------------------------------------------------------+
|  2006  Graydon Hoare が個人プロジェクトとして開始          |
|  2009  Mozilla が公式にスポンサー                          |
|  2010  初の公開発表                                        |
|  2012  Rust 0.1 リリース（初の公式プレリリース）           |
|  2015  Rust 1.0 安定版リリース                             |
|  2018  Rust 2018 Edition (NLL, async 準備)                 |
|  2020  Mozilla リストラ後もコミュニティが継続               |
|  2021  Rust 2021 Edition / Rust Foundation 設立            |
|  2022  Linux カーネルが Rust を公式サポート                |
|  2024  Rust 2024 Edition                                   |
+-----------------------------------------------------------+
```

Rustは元々、Mozilla のエンジニアである Graydon Hoare が2006年に個人プロジェクトとして開発を始めた言語である。Hoare は C++ でブラウザエンジンの開発に携わる中で、メモリ安全性とパフォーマンスを両立する言語の必要性を感じ、Rust の設計を開始した。

2009年に Mozilla が公式スポンサーとなり、Servo ブラウザエンジンの開発言語として採用された。Servo プロジェクトは Rust の実用性を証明する場となり、並行処理やメモリ安全性に関する多くの知見がフィードバックされた。

2015年5月15日に Rust 1.0 が安定版としてリリースされ、後方互換性の保証が始まった。以降、6週間ごとのリリースサイクルで継続的に改善が続けられている。

2021年には Rust Foundation が設立され、AWS、Google、Huawei、Microsoft、Mozilla の5社が創設メンバーとなった。これにより、言語の長期的な持続可能性が確保された。

### 1.2 設計原則

```
+---------------------+---------------------+---------------------+
|  安全性 (Safety)    |  速度 (Speed)       |  並行性 (Concur.)   |
+---------------------+---------------------+---------------------+
| - 所有権システム    | - ゼロコスト抽象化  | - Send / Sync       |
| - 借用チェッカー    | - LLVMバックエンド  | - データ競合防止     |
| - ライフタイム      | - インライン展開    | - fearless concur.  |
| - Optionでnull排除  | - 単態化           | - async/await       |
| - Result型でエラー  | - スタック優先      | - チャネル通信       |
| - unsafe境界の明示  | - SIMD対応         | - Arc/Mutex          |
+---------------------+---------------------+---------------------+
```

#### ゼロコスト抽象化 (Zero-Cost Abstractions)

Rustの最も重要な設計原則の一つがゼロコスト抽象化である。Bjarne Stroustrup が C++ で提唱した原則を Rust は徹底している:

> 「使わないものにはコストを払わない。使うものについては、手書きでこれ以上効率的に書くことはできない」

Rustではイテレータ、ジェネリクス、トレイトなどの高レベルな抽象化を使っても、コンパイラが最適化した結果は手書きの低レベルコードと同等の性能を発揮する。

```rust
// 高レベルなイテレータチェーン
let sum: i32 = (0..1000)
    .filter(|x| x % 2 == 0)
    .map(|x| x * x)
    .sum();

// コンパイラはこれを手書きのループと同等の機械語に最適化する
// LLVM の最適化パスにより、中間的なイテレータオブジェクトは完全に消去される
```

#### 型安全性 (Type Safety)

Rustの型システムは、C/C++ では実行時にしか発見できない多くのバグをコンパイル時に検出する。null ポインタの代わりに `Option<T>`、例外の代わりに `Result<T, E>` を使うことで、エラー処理のパスが型レベルで強制される。

```rust
// null の代わりに Option を使う
fn find_user(id: u64) -> Option<User> {
    // None を返すことで「見つからなかった」を安全に表現
    if id == 0 { return None; }
    Some(User { id, name: "example".to_string() })
}

// 呼び出し側は None のケースを必ず処理する必要がある
match find_user(42) {
    Some(user) => println!("見つかった: {}", user.name),
    None => println!("ユーザーが見つかりません"),
}
```

#### unsafe の境界

Rustは完全な安全性を保証しつつも、`unsafe` キーワードによって低レベルな操作を行う「脱出ハッチ」を提供する。これにより、OSのシステムコール、FFI（外部関数インターフェース）、パフォーマンスに特化した最適化が可能になる。

```rust
// unsafe ブロックでは以下の操作が許可される:
// 1. 生ポインタのデリファレンス
// 2. unsafe な関数・メソッドの呼び出し
// 3. 可変なスタティック変数へのアクセス
// 4. unsafe トレイトの実装
// 5. union のフィールドアクセス

fn raw_pointer_example() {
    let mut num = 5;
    let r1 = &num as *const i32;     // 生ポインタ（不変）
    let r2 = &mut num as *mut i32;   // 生ポインタ（可変）

    unsafe {
        println!("r1 = {}", *r1);
        *r2 = 10;
        println!("r2 = {}", *r2);
    }
}
```

重要なのは、`unsafe` はコンパイラの安全性チェックを「一部」無効にするだけであり、所有権やライフタイムのルール自体は依然として適用されるという点である。

### 1.3 Rustが解決する問題

Rustは主に以下の問題を解決するために設計された:

```
┌──────────────────────────────────────────────────────────────┐
│ C/C++ の問題              │ Rust の解決策                     │
├───────────────────────────┼──────────────────────────────────┤
│ ダングリングポインタ       │ 所有権 + ライフタイムで防止      │
│ ダブルフリー              │ ムーブセマンティクスで防止        │
│ バッファオーバーフロー     │ 境界チェック + スライスで防止    │
│ データ競合                │ 借用規則 + Send/Sync で防止      │
│ null ポインタ参照         │ Option<T> で型レベルで防止       │
│ メモリリーク（一般的）    │ RAII + Drop で自動管理           │
│ 未初期化変数の使用        │ コンパイラが初期化を強制         │
│ 整数オーバーフロー        │ デバッグビルドでパニック         │
└───────────────────────────┴──────────────────────────────────┘
```

---

## 2. コード例

### 例1: Hello, World!

```rust
fn main() {
    println!("Hello, World!");
}
```

`println!` はマクロであり（末尾の `!` が目印）、コンパイル時にフォーマット文字列を検証する。C の `printf` のようなフォーマット文字列の不整合はコンパイルエラーになる。

### 例2: 変数とイミュータビリティ

```rust
fn main() {
    let x = 5;          // デフォルトで不変
    // x = 6;           // コンパイルエラー！
    let mut y = 10;     // mut で可変宣言
    y += 1;
    println!("x={}, y={}", x, y);

    // シャドウイング: 同じ名前で新しい変数を束縛
    let x = x + 1;      // 新しい x（型を変えることも可能）
    let x = x * 2;
    println!("シャドウイング後の x = {}", x); // 12

    // シャドウイングでは型を変更できる
    let spaces = "   ";           // &str 型
    let spaces = spaces.len();    // usize 型に変更
    println!("スペース数: {}", spaces);
}
```

Rustではデフォルトで変数は不変（immutable）である。これは関数型プログラミングの影響を受けた設計であり、不変性をデフォルトにすることでコードの安全性と予測可能性が向上する。可変にする必要がある場合は `mut` キーワードを明示的に付ける。

### 例3: 所有権の基本

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;        // s1 の所有権が s2 にムーブ
    // println!("{}", s1); // コンパイルエラー: s1 は無効
    println!("{}", s2);
}
```

### 例4: 関数と戻り値

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b   // セミコロンなし = 式として返却
}

fn main() {
    let result = add(3, 4);
    println!("3 + 4 = {}", result);
}
```

Rustでは関数の最後の式がセミコロンなしで書かれると、それが戻り値として扱われる。`return` キーワードは早期リターンの場合にのみ使用するのが慣例である。

### 例5: パターンマッチング

```rust
enum Direction {
    North,
    South,
    East,
    West,
}

fn describe(dir: Direction) -> &'static str {
    match dir {
        Direction::North => "北",
        Direction::South => "南",
        Direction::East  => "東",
        Direction::West  => "西",
    }
}

fn main() {
    println!("{}", describe(Direction::North));
}
```

`match` 式は網羅的（exhaustive）であることが要求される。全てのパターンをカバーしていないとコンパイルエラーになるため、パターンの漏れを防止できる。

### 例6: 構造体とメソッド

```rust
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    // 関連関数（コンストラクタの慣例）
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    // メソッド（&self を受け取る）
    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    // 原点からの距離
    fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    // 可変メソッド（&mut self を受け取る）
    fn translate(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
}

fn main() {
    let a = Point::new(0.0, 0.0);
    let b = Point::new(3.0, 4.0);
    println!("距離: {}", a.distance(&b)); // 5.0
    println!("bの大きさ: {}", b.magnitude()); // 5.0

    let mut c = Point::new(1.0, 1.0);
    c.translate(2.0, 3.0);
    println!("移動後: ({}, {})", c.x, c.y); // (3.0, 4.0)
}
```

### 例7: 制御フロー

```rust
fn main() {
    // if 式（Rustでは if は式として値を返す）
    let number = 7;
    let kind = if number % 2 == 0 { "偶数" } else { "奇数" };
    println!("{} は {}", number, kind);

    // loop（無限ループ、break で値を返せる）
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2; // 20 が result に束縛される
        }
    };
    println!("loop の結果: {}", result);

    // while ループ
    let mut n = 5;
    while n > 0 {
        println!("{}!", n);
        n -= 1;
    }
    println!("発射！");

    // for ループ（Range）
    for i in 1..=5 {
        print!("{} ", i); // 1 2 3 4 5
    }
    println!();

    // for ループ（コレクション）
    let fruits = vec!["りんご", "みかん", "バナナ"];
    for (i, fruit) in fruits.iter().enumerate() {
        println!("{}: {}", i, fruit);
    }

    // while let（パターンマッチングと組み合わせ）
    let mut stack = vec![1, 2, 3];
    while let Some(top) = stack.pop() {
        println!("ポップ: {}", top);
    }
}
```

### 例8: クロージャの基本

```rust
fn main() {
    // クロージャ（無名関数）
    let add = |a: i32, b: i32| -> i32 { a + b };
    println!("3 + 4 = {}", add(3, 4));

    // 型推論でより簡潔に
    let multiply = |a, b| a * b;
    println!("3 * 4 = {}", multiply(3, 4));

    // 環境のキャプチャ
    let offset = 10;
    let add_offset = |x| x + offset;
    println!("5 + 10 = {}", add_offset(5));

    // イテレータとの組み合わせ
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter()
        .filter(|&&x| x > 2)
        .map(|&x| x * x)
        .sum();
    println!("2より大きい数の二乗和: {}", sum); // 9 + 16 + 25 = 50
}
```

### 例9: トレイトの基本

```rust
trait Printable {
    fn format_string(&self) -> String;

    // デフォルト実装
    fn print(&self) {
        println!("{}", self.format_string());
    }
}

struct Article {
    title: String,
    content: String,
}

impl Printable for Article {
    fn format_string(&self) -> String {
        format!("【記事】{}\n{}", self.title, self.content)
    }
}

struct Tweet {
    user: String,
    message: String,
}

impl Printable for Tweet {
    fn format_string(&self) -> String {
        format!("@{}: {}", self.user, self.message)
    }
}

// トレイト境界を使った関数
fn display_item(item: &impl Printable) {
    item.print();
}

fn main() {
    let article = Article {
        title: "Rust入門".to_string(),
        content: "Rustは素晴らしい言語です".to_string(),
    };
    let tweet = Tweet {
        user: "rustlang".to_string(),
        message: "Rust 1.0 released!".to_string(),
    };

    display_item(&article);
    display_item(&tweet);
}
```

### 例10: エラーハンドリングの基本

```rust
use std::fs;
use std::io;

fn read_username_from_file() -> Result<String, io::Error> {
    let content = fs::read_to_string("username.txt")?;
    Ok(content.trim().to_string())
}

fn main() {
    match read_username_from_file() {
        Ok(name) => println!("ユーザー名: {}", name),
        Err(e) => eprintln!("エラー: {}", e),
    }

    // Option の活用例
    let numbers = vec![10, 20, 30];
    let first = numbers.first();
    match first {
        Some(n) => println!("最初の要素: {}", n),
        None => println!("空のベクタ"),
    }

    // if let による簡潔なパターンマッチ
    if let Some(n) = numbers.get(1) {
        println!("2番目の要素: {}", n);
    }
}
```

---

## 3. 図解

### 3.1 ビルドパイプライン

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  .rs     │───>│  rustc   │───>│   LLVM   │───>│ バイナリ  │
│ ソース   │    │ フロント │    │  IR/最適化│    │  実行可能 │
│          │    │ エンド   │    │          │    │  ファイル │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │                │
                     ▼                ▼
              ┌──────────────┐  ┌──────────────┐
              │ 借用チェック │  │ 最適化パス   │
              │ 型チェック   │  │ インライン化  │
              │ ライフタイム │  │ デッドコード  │
              │ パターン検査 │  │ 除去、SIMD   │
              └──────────────┘  └──────────────┘
```

ビルドパイプラインの詳細:

1. **字句解析・構文解析**: `.rs` ファイルを AST（抽象構文木）に変換
2. **名前解決**: 変数名・型名・モジュール名を解決
3. **型チェック**: 型の整合性を検証（型推論含む）
4. **借用チェック**: 所有権・借用・ライフタイムの規則を検証
5. **MIR生成**: 中間表現（Mid-level IR）を生成
6. **MIR最適化**: 定数伝播、デッドコード除去など
7. **LLVM IR生成**: LLVM が理解する中間表現に変換
8. **LLVM最適化**: インライン展開、ループ最適化、SIMD 変換など
9. **コード生成**: ターゲットアーキテクチャ向けの機械語を生成
10. **リンク**: オブジェクトファイルをリンクして実行可能バイナリを生成

### 3.2 メモリモデル(スタック vs ヒープ)

```
       スタック                      ヒープ
  ┌─────────────────┐         ┌───────────────┐
  │ ptr ──────────────────────>│ h e l l o     │
  │ len = 5         │         └───────────────┘
  │ capacity = 5    │
  ├─────────────────┤         ┌───────────────────┐
  │ ptr ──────────────────────>│ [1, 2, 3, 4, 5]  │
  │ len = 5         │         └───────────────────┘
  │ capacity = 8    │
  ├─────────────────┤
  │ x: i32 = 42     │
  ├─────────────────┤
  │ y: bool = true   │
  ├─────────────────┤
  │ z: f64 = 3.14   │
  └─────────────────┘
    固定サイズ・高速            可変サイズ・動的
    LIFO（後入れ先出し）       OS が管理（malloc/free）
    自動的に確保・解放          Rust が RAII で自動管理
```

スタックとヒープの使い分け:

- **スタック**: コンパイル時にサイズがわかる型（`i32`, `f64`, `bool`, 固定長配列, タプル）
- **ヒープ**: 実行時にサイズが変わる型（`String`, `Vec<T>`, `HashMap<K,V>`, `Box<T>`）

Rustの `String` 型はスタック上に3つのフィールド（ポインタ、長さ、容量）を持ち、実際の文字列データはヒープ上に格納される。所有者がスコープを抜けると、`Drop` トレイトによってヒープメモリが自動的に解放される。

### 3.3 ツールチェーン構成

```
┌─────────────────────────────────────────────────────────┐
│                      rustup                             │
│  (ツールチェーン管理: stable/beta/nightly)              │
├──────────┬──────────┬───────────────────────────────────┤
│  rustc   │  cargo   │  その他ツール                     │
│ コンパイラ│ ビルド   │  rustfmt   -- コードフォーマッタ  │
│          │ パッケージ│  clippy    -- 静的解析リンター    │
│          │ テスト   │  rust-analyzer -- LSP サーバー    │
│          │ ドキュメント│ miri    -- 未定義動作検出        │
│          │ ベンチマーク│ cargo-audit -- 脆弱性検査       │
│          │          │  cargo-expand -- マクロ展開確認   │
└──────────┴──────────┴───────────────────────────────────┘
                │
                ▼
         ┌────────────┐
         │ crates.io  │
         │ レジストリ │  15万以上のクレートが公開
         └────────────┘
```

### 3.4 Rustのモジュールシステム

```
my_project/
├── Cargo.toml          # プロジェクト設定・依存関係
├── Cargo.lock          # 依存関係のロックファイル
├── src/
│   ├── main.rs         # バイナリクレートのエントリポイント
│   ├── lib.rs          # ライブラリクレートのルート
│   ├── config.rs       # モジュールファイル
│   └── utils/          # サブモジュールディレクトリ
│       ├── mod.rs      # utils モジュールのルート
│       ├── parser.rs   # utils::parser サブモジュール
│       └── formatter.rs# utils::formatter サブモジュール
├── tests/
│   └── integration_test.rs  # 統合テスト
├── benches/
│   └── benchmark.rs    # ベンチマーク
└── examples/
    └── demo.rs         # サンプルコード
```

```rust
// src/lib.rs
pub mod config;       // config.rs を読み込む
pub mod utils;        // utils/mod.rs を読み込む

// src/utils/mod.rs
pub mod parser;       // utils/parser.rs を読み込む
pub mod formatter;    // utils/formatter.rs を読み込む

// src/main.rs
use my_project::config::Config;
use my_project::utils::parser::parse;

fn main() {
    let config = Config::load("settings.toml").unwrap();
    let data = parse(&config.input_file).unwrap();
    println!("パース完了: {} 件", data.len());
}
```

### 3.5 Rustの所有権モデル概要図

```
┌──────────────────────────────────────────────────────────┐
│                    所有権システム                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────┐  ムーブ  ┌────────┐                         │
│  │ 変数 A │ ──────> │ 変数 B │  A は無効化される        │
│  └────────┘          └────────┘                         │
│                                                          │
│  ┌────────┐  借用    ┌────────┐                         │
│  │ 変数 A │ <────── │ 参照&A │  A は有効なまま          │
│  └────────┘          └────────┘                         │
│                                                          │
│  ┌────────┐  クローン ┌────────┐                        │
│  │ 変数 A │ ──────── │ 変数 B │  独立したコピー         │
│  └────────┘           └────────┘                        │
│                                                          │
│  ルール:                                                 │
│  1. 各値は唯一の所有者を持つ                             │
│  2. 所有者がスコープを抜けると Drop が呼ばれる           │
│  3. 不変参照は同時に複数可能                             │
│  4. 可変参照は同時に1つだけ                              │
│  5. 参照は常に有効でなければならない                     │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 比較表

### 4.1 Rust vs 他のシステム言語

| 特性 | Rust | C | C++ | Go | Python |
|------|------|---|-----|----|--------|
| メモリ安全 | コンパイル時保証 | 手動管理 | 手動+RAII | GC | GC |
| データ競合防止 | コンパイル時保証 | なし | なし | 実行時検出 | GIL |
| ゼロコスト抽象化 | あり | N/A | あり | なし(GC) | なし |
| パッケージマネージャ | Cargo (標準) | なし (CMakeなど) | なし (vcpkgなど) | go mod | pip |
| 学習曲線 | 急峻 | 中程度 | 急峻 | 緩やか | 緩やか |
| コンパイル速度 | 遅め | 速い | 遅い | 非常に速い | N/A |
| null | Option型 | NULLポインタ | nullptr | nil | None |
| エラー処理 | Result型 | 戻り値/errno | 例外 | error戻り値 | 例外 |
| 並行処理 | async/await, スレッド | pthread | std::thread | goroutine | asyncio |
| バイナリサイズ | 小さい | 最小 | 大きめ | やや大きい | N/A |
| クロスコンパイル | 容易 | 複雑 | 複雑 | 非常に容易 | N/A |
| Wasm対応 | 優秀 | Emscripten | Emscripten | TinyGo | Pyodide |

### 4.2 Rust Edition 比較

| 機能 | 2015 | 2018 | 2021 | 2024 |
|------|------|------|------|------|
| NLL (Non-Lexical Lifetimes) | - | 導入 | 安定 | 安定 |
| async/await | - | 導入 | 安定 | 改善 |
| モジュールパス | 旧方式 | 新方式 | 新方式 | 新方式 |
| クロージャキャプチャ | 全体 | 全体 | 部分 | 部分 |
| dyn Trait | 暗黙 | 明示必須 | 明示必須 | 明示必須 |
| try ブロック | - | - | - | 安定化へ |
| let-else | - | - | 導入 | 安定 |
| impl Trait in type aliases | - | - | - | 安定化へ |

### 4.3 Rustの適用領域

| 領域 | 代表的なプロジェクト | Rustの強み |
|------|---------------------|-----------|
| Webバックエンド | Actix Web, Axum, Rocket | 高スループット、低メモリ |
| CLI ツール | ripgrep, bat, fd, exa | 高速起動、シングルバイナリ |
| WebAssembly | wasm-bindgen, Yew, Leptos | 小さなバイナリ、高速実行 |
| 組み込み | embedded-hal, RTIC | ゼロコスト、no_std対応 |
| OS/カーネル | Redox OS, Linux ドライバ | メモリ安全、低レベル制御 |
| ゲーム | Bevy, Amethyst | 高パフォーマンス、ECS |
| ブロックチェーン | Solana, Polkadot, Near | 安全性、パフォーマンス |
| データベース | TiKV, SurrealDB | 並行処理、信頼性 |
| ネットワーク | Tokio, Hyper, Cloudflare | 非同期I/O、低レイテンシ |

---

## 5. Cargoの使い方

### 5.1 基本コマンド

```bash
# プロジェクト作成
cargo new my_project          # バイナリプロジェクト
cargo new my_lib --lib        # ライブラリプロジェクト
cargo init                    # 既存ディレクトリで初期化

# ビルドと実行
cargo build                   # デバッグビルド
cargo build --release         # リリースビルド（最適化あり）
cargo run                     # ビルドして実行
cargo run --release           # リリースモードで実行
cargo run -- arg1 arg2        # 引数付き実行

# テスト
cargo test                    # 全テスト実行
cargo test test_name          # 特定テスト実行
cargo test -- --nocapture     # println! を表示
cargo test --doc              # ドキュメントテストのみ

# 品質ツール
cargo clippy                  # 静的解析
cargo fmt                     # コードフォーマット
cargo doc --open              # ドキュメント生成・表示
cargo audit                   # セキュリティ監査（要インストール）

# 依存関係
cargo add serde               # 依存関係追加（cargo-edit）
cargo update                  # 依存関係更新
cargo tree                    # 依存関係ツリー表示

# その他
cargo bench                   # ベンチマーク実行
cargo clean                   # ビルド成果物を削除
cargo check                   # コンパイルチェック（バイナリ生成なし・高速）
```

### 5.2 Cargo.toml の構成

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A useful project"
license = "MIT"
repository = "https://github.com/user/my_project"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
clap = { version = "4", features = ["derive"] }

[dev-dependencies]
criterion = "0.5"
mockall = "0.12"

[profile.release]
opt-level = 3      # 最大最適化
lto = true          # リンク時最適化
codegen-units = 1   # コード生成ユニット数（遅いがより最適化）
strip = true        # デバッグシンボル除去

[profile.dev]
opt-level = 0       # 最適化なし（ビルド高速）
debug = true        # デバッグ情報あり
```

---

## 6. アンチパターン

### アンチパターン1: 何でも `clone()` で解決する

```rust
// BAD: 所有権エラーを全て clone で回避
fn process(data: Vec<String>) {
    let copy = data.clone(); // 不必要なヒープアロケーション
    for item in data.clone() {
        println!("{}", item);
    }
}

// GOOD: 参照を活用する
fn process_good(data: &[String]) {
    for item in data {
        println!("{}", item);
    }
}
```

### アンチパターン2: `unwrap()` の乱用

```rust
// BAD: 本番コードで unwrap は危険
fn read_config() -> String {
    std::fs::read_to_string("config.toml").unwrap() // パニック！
}

// GOOD: Result を適切に伝播
fn read_config_good() -> Result<String, std::io::Error> {
    std::fs::read_to_string("config.toml")
}
```

### アンチパターン3: 不必要な `String` の所有

```rust
// BAD: 引数に String を要求する
fn greet(name: String) {
    println!("こんにちは、{}さん", name);
}

// GOOD: &str を受け取れば String も &str も渡せる
fn greet_good(name: &str) {
    println!("こんにちは、{}さん", name);
}

fn main() {
    let owned = String::from("太郎");
    let borrowed = "花子";

    greet_good(&owned);    // String → &str (自動 Deref)
    greet_good(borrowed);  // &str そのまま
}
```

### アンチパターン4: 過度に深いネスト

```rust
// BAD: match のネストが深い
fn process_data(input: Option<Result<String, io::Error>>) {
    match input {
        Some(result) => {
            match result {
                Ok(data) => {
                    match data.parse::<i32>() {
                        Ok(num) => println!("{}", num),
                        Err(e) => eprintln!("パースエラー: {}", e),
                    }
                }
                Err(e) => eprintln!("IOエラー: {}", e),
            }
        }
        None => eprintln!("入力なし"),
    }
}

// GOOD: 早期リターンやコンビネータで平坦化
fn process_data_good(input: Option<Result<String, io::Error>>) -> Result<(), Box<dyn std::error::Error>> {
    let data = input.ok_or("入力なし")?;
    let text = data?;
    let num: i32 = text.parse()?;
    println!("{}", num);
    Ok(())
}
```

### アンチパターン5: C スタイルの for ループ

```rust
// BAD: インデックスベースのループ
fn sum_vec(v: &[i32]) -> i32 {
    let mut sum = 0;
    for i in 0..v.len() {
        sum += v[i]; // 毎回境界チェックが走る
    }
    sum
}

// GOOD: イテレータを使う
fn sum_vec_good(v: &[i32]) -> i32 {
    v.iter().sum()
}
```

---

## 7. Rustのテスト

### 7.1 ユニットテスト

```rust
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("ゼロ除算エラー".to_string())
    } else {
        Ok(a / b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
        assert_eq!(add(0, 0), 0);
    }

    #[test]
    fn test_divide_success() {
        let result = divide(10.0, 2.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5.0);
    }

    #[test]
    fn test_divide_by_zero() {
        let result = divide(10.0, 0.0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "ゼロ除算エラー");
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_out_of_bounds() {
        let v = vec![1, 2, 3];
        let _ = v[10]; // パニック！
    }

    #[test]
    fn test_with_result() -> Result<(), String> {
        let result = divide(10.0, 2.0)?;
        assert_eq!(result, 5.0);
        Ok(())
    }
}
```

### 7.2 ドキュメントテスト

```rust
/// 2つの数値を加算する。
///
/// # 引数
///
/// * `a` - 最初の数値
/// * `b` - 2番目の数値
///
/// # 戻り値
///
/// 2つの数値の合計
///
/// # 例
///
/// ```
/// use my_crate::add;
/// assert_eq!(add(2, 3), 5);
/// assert_eq!(add(-1, 1), 0);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

`cargo test --doc` でドキュメント内のコード例がテストとして実行される。これにより、ドキュメントのコード例が常に正しいことが保証される。

---

## 8. FAQ

### Q1: Rustの学習曲線は本当に急峻ですか？

**A:** はい、特に所有権・ライフタイムの概念は他言語にない独自のものです。しかし一度習得すれば、コンパイラが多くのバグを事前に防いでくれるため、デバッグ時間が大幅に減少します。多くの開発者が「3-6ヶ月で生産性が上がり始める」と報告しています。

学習のコツ:
- まずは The Rust Programming Language（公式Book）を通読する
- Rustlings の演習で手を動かす
- コンパイラのエラーメッセージを丁寧に読む（Rustのエラーメッセージは非常に親切）
- 最初は `clone()` を多用しても構わない。動くコードを書いてから最適化する

### Q2: RustはどんなプロジェクトにGo向きですか？

**A:** 以下のようなケースでRustが特に適しています:
- パフォーマンスが最重要（ゲームエンジン、ブラウザエンジン）
- メモリ安全が必須（OS、組み込み、セキュリティツール）
- WebAssembly（Wasmとの親和性が高い）
- CLIツール（シングルバイナリ配布が容易）
- 高スループットなネットワークサービス（プロキシ、ロードバランサ）

逆に、以下のケースではGoの方が適している場合もあります:
- 素早いプロトタイピングが必要
- チームの学習コストを最小化したい
- マイクロサービスの量産

### Q3: Rustにガベージコレクタがないのに、なぜ安全なのですか？

**A:** Rustは「所有権システム」でメモリを管理します。各値には唯一の「所有者」が存在し、所有者がスコープを抜けると自動的にメモリが解放されます（Drop トレイト）。コンパイラが借用規則を静的に検証するため、ダングリングポインタやダブルフリーが発生しません。

### Q4: Rustのコンパイルが遅い理由は何ですか？

**A:** Rustのコンパイルが遅い主な理由は:
1. **借用チェック**: 所有権とライフタイムの検証は計算コストが高い
2. **単態化**: ジェネリクスを使用すると、具体的な型ごとにコードが生成される
3. **LLVM最適化**: 高品質な最適化には時間がかかる
4. **リンク時最適化 (LTO)**: リリースビルドで有効な場合、特に遅くなる

改善策として `cargo check`（バイナリ生成をスキップ）の使用、増分コンパイルの活用、`sccache` の導入などがある。

### Q5: RustとC++の違いを一言で言うと？

**A:** 「C++は正しいコードも危険なコードも書ける。Rustは危険なコードをコンパイラが拒否する。」

C++はプログラマに全ての自由を与えるが、その結果としてメモリ安全性のバグが発生しうる。Rustはコンパイラが安全性を保証するため、C++で長年問題となってきたメモリ安全性のバグの約70%（Microsoft Researchの調査）をコンパイル時に防止できる。

### Q6: Rustで Web 開発はできますか？

**A:** はい。バックエンドには Actix Web、Axum、Rocket などの成熟したフレームワークがあります。フロントエンドには WebAssembly を活用した Yew、Leptos、Dioxus などのフレームワークがあります。また、Tauri を使えばデスクトップアプリケーションも開発できます。

```rust
// Axum を使ったシンプルな Web サーバー
use axum::{routing::get, Router};

async fn hello() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(hello));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## 9. 実践的な開発のヒント

### 9.1 コンパイラとの付き合い方

Rustのコンパイラ（rustc）は非常に厳格だが、そのエラーメッセージは極めて親切である。エラーメッセージに含まれる「help:」や「note:」の行を丁寧に読むと、ほとんどの場合で解決策が提示される。

```
error[E0382]: borrow of moved value: `s1`
 --> src/main.rs:4:20
  |
2 |     let s1 = String::from("hello");
  |         -- move occurs because `s1` has type `String`
3 |     let s2 = s1;
  |              -- value moved here
4 |     println!("{}", s1);
  |                    ^^ value borrowed here after move
  |
  = note: this error originates in the macro `$crate::format_args_nl`
help: consider cloning the value if the performance cost is acceptable
  |
3 |     let s2 = s1.clone();
  |                ++++++++
```

### 9.2 推奨する開発ツール

| ツール | 用途 | インストール |
|--------|------|-------------|
| rust-analyzer | IDE サポート (LSP) | VS Code 拡張等 |
| clippy | リンター | `rustup component add clippy` |
| rustfmt | フォーマッタ | `rustup component add rustfmt` |
| cargo-watch | ファイル変更時に自動ビルド | `cargo install cargo-watch` |
| cargo-expand | マクロ展開の確認 | `cargo install cargo-expand` |
| cargo-audit | セキュリティ脆弱性検査 | `cargo install cargo-audit` |
| cargo-flamegraph | パフォーマンスプロファイリング | `cargo install flamegraph` |
| bacon | 軽量なビルドウォッチャー | `cargo install bacon` |

### 9.3 よく使うデザインパターン

```rust
// ビルダーパターン
struct ServerConfig {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_ms: u64,
}

struct ServerConfigBuilder {
    host: String,
    port: u16,
    max_connections: u32,
    timeout_ms: u64,
}

impl ServerConfigBuilder {
    fn new() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
            max_connections: 100,
            timeout_ms: 30000,
        }
    }

    fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    fn timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    fn build(self) -> ServerConfig {
        ServerConfig {
            host: self.host,
            port: self.port,
            max_connections: self.max_connections,
            timeout_ms: self.timeout_ms,
        }
    }
}

fn main() {
    let config = ServerConfigBuilder::new()
        .host("0.0.0.0")
        .port(3000)
        .max_connections(1000)
        .build();

    println!("サーバー: {}:{}", config.host, config.port);
}
```

```rust
// ニュータイプパターン
struct Email(String);
struct UserId(u64);

impl Email {
    fn new(email: &str) -> Result<Self, String> {
        if email.contains('@') {
            Ok(Email(email.to_string()))
        } else {
            Err("無効なメールアドレス".to_string())
        }
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

fn send_email(to: &Email, subject: &str) {
    println!("送信先: {}, 件名: {}", to.as_str(), subject);
}

fn main() {
    let email = Email::new("user@example.com").unwrap();
    send_email(&email, "テスト");
    // send_email(&"invalid", "テスト"); // コンパイルエラー！型が違う
}
```

---

## 10. まとめ

| 項目 | 要点 |
|------|------|
| 設計哲学 | 安全性・速度・並行性を同時達成 |
| メモリ管理 | 所有権システム（GCなし） |
| 型システム | 強い静的型付け + ジェネリクス + トレイト |
| ツールチェーン | rustup / cargo / clippy / rustfmt |
| エコシステム | crates.io に 15万+ のクレート |
| 用途 | システム、Web、CLI、Wasm、組み込み |
| Edition | 後方互換を保ちつつ段階的に進化 |
| テスト | ユニットテスト、統合テスト、ドキュメントテストが標準装備 |
| エラー処理 | Result/Option による型安全なエラーハンドリング |
| 並行処理 | Send/Sync トレイトでコンパイル時にデータ競合を防止 |

---

## 次に読むべきガイド

- [01-ownership-borrowing.md](01-ownership-borrowing.md) -- 所有権と借用を深く理解する
- [02-types-and-traits.md](02-types-and-traits.md) -- 型とトレイトで抽象化を学ぶ
- [03-error-handling.md](03-error-handling.md) -- Result/Optionを活用したエラー処理
- [04-collections-iterators.md](04-collections-iterators.md) -- コレクションとイテレータ
- [../04-ecosystem/00-cargo-workspace.md](../04-ecosystem/00-cargo-workspace.md) -- Cargoの使い方を習得する

---

## 参考文献

1. **The Rust Programming Language (公式Book)** -- https://doc.rust-lang.org/book/
2. **Rust by Example** -- https://doc.rust-lang.org/rust-by-example/
3. **Rust Reference** -- https://doc.rust-lang.org/reference/
4. **Rustlings (演習)** -- https://github.com/rust-lang/rustlings
5. **Rust Foundation** -- https://foundation.rust-lang.org/
6. **Rust API Guidelines** -- https://rust-lang.github.io/api-guidelines/
7. **The Rustonomicon (unsafe Rust)** -- https://doc.rust-lang.org/nomicon/
8. **Rust Playground** -- https://play.rust-lang.org/
9. **This Week in Rust** -- https://this-week-in-rust.org/
10. **Rust Design Patterns** -- https://rust-unofficial.github.io/patterns/
