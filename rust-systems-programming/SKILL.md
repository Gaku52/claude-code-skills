# Rust システムプログラミング

> Rust は安全性・速度・並行性を兼ね備えたシステムプログラミング言語。所有権システム、ライフタイム、トレイト、非同期プログラミング、unsafe まで、Rust の全てを体系的に解説する。

## このSkillの対象者

- Rust を基礎から学びたいエンジニア
- システムプログラミング（CLI/サーバー/WebAssembly）を行いたい方
- メモリ安全性と高パフォーマンスを両立したい方

## 前提知識

- 何らかのプログラミング言語の経験
- メモリ管理の基礎概念

## 学習ガイド

### 00-basics — Rust の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-basics/00-rust-overview.md]] | Rust の歴史、cargo、プロジェクト構造、基本構文 |
| 01 | [[docs/00-basics/01-types-and-control.md]] | 型システム、変数束縛、制御フロー、パターンマッチ |
| 02 | [[docs/00-basics/02-structs-and-enums.md]] | 構造体、列挙型、Option/Result、impl、メソッド |
| 03 | [[docs/00-basics/03-error-handling.md]] | Result/Option、?演算子、anyhow/thiserror、カスタムエラー |
| 04 | [[docs/00-basics/04-collections-and-iterators.md]] | Vec/HashMap/HashSet、イテレータ、クロージャ、関数型パターン |

### 01-ownership — 所有権システム

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-ownership/00-ownership-basics.md]] | 所有権ルール、ムーブセマンティクス、Copy/Clone |
| 01 | [[docs/01-ownership/01-borrowing-and-references.md]] | 借用、参照、可変参照、借用チェッカー |
| 02 | [[docs/01-ownership/02-lifetimes.md]] | ライフタイム注釈、省略規則、'static、HRTB |
| 03 | [[docs/01-ownership/03-smart-pointers.md]] | Box/Rc/Arc/RefCell/Mutex、内部可変性、Deref |

### 02-advanced — 高度な機能

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-advanced/00-traits-deep-dive.md]] | トレイト、トレイト境界、関連型、動的ディスパッチ |
| 01 | [[docs/02-advanced/01-generics-and-macros.md]] | ジェネリクス、宣言的マクロ、手続き的マクロ、derive |
| 02 | [[docs/02-advanced/02-unsafe-rust.md]] | unsafe ブロック、生ポインタ、FFI、安全な抽象化 |
| 03 | [[docs/02-advanced/03-type-system-advanced.md]] | PhantomData、型状態パターン、ゼロコスト抽象化 |

### 03-systems — システム応用

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-systems/00-cli-development.md]] | clap、CLI ツール開発、引数パース、TUI（ratatui） |
| 01 | [[docs/03-systems/01-web-development.md]] | Axum/Actix-web、REST API、ミドルウェア、SQLx |
| 02 | [[docs/03-systems/02-wasm.md]] | WebAssembly、wasm-bindgen、wasm-pack、ブラウザ統合 |
| 03 | [[docs/03-systems/03-embedded-and-os.md]] | 組み込み Rust、no_std、OS 開発入門 |

### 04-async — 非同期プログラミング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-async/00-async-basics.md]] | async/await、Future、tokio ランタイム、タスク |
| 01 | [[docs/04-async/01-async-patterns.md]] | Stream、select!、チャネル、並行パターン |
| 02 | [[docs/04-async/02-performance.md]] | ベンチマーク、プロファイリング、最適化テクニック |

## クイックリファレンス

```
Rust 早見表:
  cargo new myapp          — プロジェクト作成
  cargo build --release    — リリースビルド
  cargo test               — テスト実行
  cargo clippy             — リント
  cargo fmt                — フォーマット

  所有権ルール:
    1. 各値は1つのオーナーを持つ
    2. オーナーがスコープを抜けると値は破棄
    3. 参照は値より長生きできない
```

## 参考文献

1. Klabnik, S. & Nichols, C. "The Rust Programming Language." doc.rust-lang.org/book, 2024.
2. Rust. "Rust by Example." doc.rust-lang.org/rust-by-example, 2024.
3. Blandy, J. et al. "Programming Rust." O'Reilly, 2021.
