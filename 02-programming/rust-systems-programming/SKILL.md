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

### 01-ownership — 所有権システム

| # | ファイル | 内容 |
|---|---------|------|

### 02-advanced — 高度な機能

| # | ファイル | 内容 |
|---|---------|------|

### 03-systems — システム応用

| # | ファイル | 内容 |
|---|---------|------|

### 04-async — 非同期プログラミング

| # | ファイル | 内容 |
|---|---------|------|

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
