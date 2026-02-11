# Rust概要 -- 安全性・パフォーマンス・所有権が融合したシステムプログラミング言語

> Rustは「安全性」「速度」「並行性」の3つを同時に達成する唯一のシステムプログラミング言語であり、ガベージコレクタなしでメモリ安全を保証する。

---

## この章で学ぶこと

1. **Rustの設計哲学** -- ゼロコスト抽象化・所有権・型安全の3本柱を理解する
2. **他言語との違い** -- C/C++/Go/Pythonとの比較で立ち位置を把握する
3. **エコシステムの全体像** -- Cargo、crates.io、ツールチェーンの構成を掴む

---

## 1. Rustの歴史と設計哲学

### 1.1 誕生の経緯

```
+-----------------------------------------------------------+
|  2006  Graydon Hoare が個人プロジェクトとして開始          |
|  2009  Mozilla が公式にスポンサー                          |
|  2010  初の公開発表                                        |
|  2015  Rust 1.0 安定版リリース                             |
|  2018  Rust 2018 Edition (NLL, async 準備)                 |
|  2021  Rust 2021 Edition / Rust Foundation 設立            |
|  2024  Rust 2024 Edition                                   |
+-----------------------------------------------------------+
```

### 1.2 設計原則

```
+---------------------+---------------------+---------------------+
|  安全性 (Safety)    |  速度 (Speed)       |  並行性 (Concur.)   |
+---------------------+---------------------+---------------------+
| - 所有権システム    | - ゼロコスト抽象化  | - Send / Sync       |
| - 借用チェッカー    | - LLVMバックエンド  | - データ競合防止     |
| - ライフタイム      | - インライン展開    | - fearless concur.  |
| - Optionでnull排除  | - 単態化           | - async/await       |
+---------------------+---------------------+---------------------+
```

---

## 2. コード例

### 例1: Hello, World!

```rust
fn main() {
    println!("Hello, World!");
}
```

### 例2: 変数とイミュータビリティ

```rust
fn main() {
    let x = 5;          // デフォルトで不変
    // x = 6;           // コンパイルエラー！
    let mut y = 10;     // mut で可変宣言
    y += 1;
    println!("x={}, y={}", x, y);
}
```

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

### 例6: 構造体とメソッド

```rust
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

fn main() {
    let a = Point { x: 0.0, y: 0.0 };
    let b = Point { x: 3.0, y: 4.0 };
    println!("距離: {}", a.distance(&b)); // 5.0
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
                     │
                     ▼
              ┌──────────────┐
              │ 借用チェック │
              │ 型チェック   │
              │ ライフタイム │
              └──────────────┘
```

### 3.2 メモリモデル(スタック vs ヒープ)

```
       スタック                      ヒープ
  ┌─────────────────┐         ┌───────────────┐
  │ ptr ──────────────────────>│ h e l l o     │
  │ len = 5         │         └───────────────┘
  │ capacity = 5    │
  ├─────────────────┤
  │ x: i32 = 42     │
  ├─────────────────┤
  │ y: bool = true   │
  └─────────────────┘
    固定サイズ・高速            可変サイズ・動的
```

### 3.3 ツールチェーン構成

```
┌─────────────────────────────────────────────┐
│                 rustup                      │
│  (ツールチェーン管理: stable/beta/nightly)  │
├──────────┬──────────┬───────────────────────┤
│  rustc   │  cargo   │  その他ツール         │
│ コンパイラ│ ビルド   │  rustfmt, clippy,     │
│          │ パッケージ│  rust-analyzer, miri  │
└──────────┴──────────┴───────────────────────┘
                │
                ▼
         ┌────────────┐
         │ crates.io  │
         │ レジストリ │
         └────────────┘
```

---

## 4. 比較表

### 4.1 Rust vs 他のシステム言語

| 特性 | Rust | C | C++ | Go |
|------|------|---|-----|----|
| メモリ安全 | コンパイル時保証 | 手動管理 | 手動+RAII | GC |
| データ競合防止 | コンパイル時保証 | なし | なし | 実行時検出 |
| ゼロコスト抽象化 | あり | N/A | あり | なし(GC) |
| パッケージマネージャ | Cargo (標準) | なし (CMakeなど) | なし (vcpkgなど) | go mod |
| 学習曲線 | 急峻 | 中程度 | 急峻 | 緩やか |
| コンパイル速度 | 遅め | 速い | 遅い | 非常に速い |
| null | Option型 | NULLポインタ | nullptr | nil |
| エラー処理 | Result型 | 戻り値/errno | 例外 | error戻り値 |

### 4.2 Rust Edition 比較

| 機能 | 2015 | 2018 | 2021 | 2024 |
|------|------|------|------|------|
| NLL (Non-Lexical Lifetimes) | - | 導入 | 安定 | 安定 |
| async/await | - | 導入 | 安定 | 改善 |
| モジュールパス | 旧方式 | 新方式 | 新方式 | 新方式 |
| クロージャキャプチャ | 全体 | 全体 | 部分 | 部分 |
| dyn Trait | 暗黙 | 明示必須 | 明示必須 | 明示必須 |

---

## 5. アンチパターン

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

---

## 6. FAQ

### Q1: Rustの学習曲線は本当に急峻ですか？

**A:** はい、特に所有権・ライフタイムの概念は他言語にない独自のものです。しかし一度習得すれば、コンパイラが多くのバグを事前に防いでくれるため、デバッグ時間が大幅に減少します。多くの開発者が「3-6ヶ月で生産性が上がり始める」と報告しています。

### Q2: RustはどんなプロジェクトにGo向きですか？

**A:** 以下のようなケースでRustが特に適しています:
- パフォーマンスが最重要(ゲームエンジン、ブラウザエンジン)
- メモリ安全が必須(OS、組み込み、セキュリティツール)
- WebAssembly(Wasmとの親和性が高い)
- CLIツール(シングルバイナリ配布が容易)

### Q3: Rustにガベージコレクタがないのに、なぜ安全なのですか？

**A:** Rustは「所有権システム」でメモリを管理します。各値には唯一の「所有者」が存在し、所有者がスコープを抜けると自動的にメモリが解放されます(Drop トレイト)。コンパイラが借用規則を静的に検証するため、ダングリングポインタやダブルフリーが発生しません。

---

## 7. まとめ

| 項目 | 要点 |
|------|------|
| 設計哲学 | 安全性・速度・並行性を同時達成 |
| メモリ管理 | 所有権システム(GCなし) |
| 型システム | 強い静的型付け + ジェネリクス + トレイト |
| ツールチェーン | rustup / cargo / clippy / rustfmt |
| エコシステム | crates.io に 15万+ のクレート |
| 用途 | システム、Web、CLI、Wasm、組み込み |
| Edition | 後方互換を保ちつつ段階的に進化 |

---

## 次に読むべきガイド

- [01-ownership-borrowing.md](01-ownership-borrowing.md) -- 所有権と借用を深く理解する
- [02-types-and-traits.md](02-types-and-traits.md) -- 型とトレイトで抽象化を学ぶ
- [../04-ecosystem/00-cargo-workspace.md](../04-ecosystem/00-cargo-workspace.md) -- Cargoの使い方を習得する

---

## 参考文献

1. **The Rust Programming Language (公式Book)** -- https://doc.rust-lang.org/book/
2. **Rust by Example** -- https://doc.rust-lang.org/rust-by-example/
3. **Rust Reference** -- https://doc.rust-lang.org/reference/
4. **Rustlings (演習)** -- https://github.com/rust-lang/rustlings
5. **Rust Foundation** -- https://foundation.rust-lang.org/
