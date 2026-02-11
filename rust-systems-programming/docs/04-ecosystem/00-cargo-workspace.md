# Cargo/ワークスペース — features、publish

> Cargo のパッケージ管理、ワークスペースによるモノレポ構成、feature フラグ、crates.io への公開手順を体系的に習得する

## この章で学ぶこと

1. **Cargo 基本操作** — 依存関係管理、プロファイル設定、ビルドスクリプト
2. **ワークスペース** — マルチクレート構成、依存の共有、選択的ビルド
3. **Feature フラグと公開** — 条件付きコンパイル、セマンティックバージョニング、crates.io 公開

---

## 1. Cargo プロジェクト構造

```
┌──────────────── Cargo プロジェクト構成 ──────────────┐
│                                                       │
│  単一クレート:                                        │
│  my-app/                                              │
│  ├── Cargo.toml        (パッケージ定義)               │
│  ├── Cargo.lock        (依存バージョン固定)            │
│  ├── src/                                             │
│  │   ├── main.rs       (バイナリクレート)              │
│  │   ├── lib.rs        (ライブラリクレート)            │
│  │   └── bin/          (追加バイナリ)                  │
│  │       └── tool.rs                                  │
│  ├── tests/            (統合テスト)                    │
│  ├── benches/          (ベンチマーク)                  │
│  ├── examples/         (使用例)                       │
│  └── build.rs          (ビルドスクリプト)              │
│                                                       │
│  ワークスペース:                                      │
│  my-workspace/                                        │
│  ├── Cargo.toml        (workspace 定義)               │
│  ├── crates/                                          │
│  │   ├── core/         (共通ライブラリ)               │
│  │   ├── cli/          (CLI アプリ)                   │
│  │   └── server/       (Web サーバー)                 │
│  └── Cargo.lock        (ワークスペース全体で共有)      │
└───────────────────────────────────────────────────────┘
```

---

## 2. Cargo.toml の詳細設定

### コード例1: 充実した Cargo.toml

```toml
[package]
name = "my-awesome-lib"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <you@example.com>"]
description = "A short description of the library"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/my-awesome-lib"
documentation = "https://docs.rs/my-awesome-lib"
readme = "README.md"
keywords = ["async", "web", "http"]
categories = ["web-programming"]
exclude = ["tests/fixtures/**", ".github/**"]

[lib]
# クレートタイプ (複数指定可)
# crate-type = ["lib", "cdylib"]

[dependencies]
# バージョン指定の方法
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"], optional = true }
log = "0.4"

# Git リポジトリから
# my-dep = { git = "https://github.com/user/repo", branch = "main" }

# ローカルパスから (開発時)
# my-local = { path = "../my-local" }

[dev-dependencies]
# テスト・ベンチマークのみ
tokio = { version = "1", features = ["full", "test-util"] }
criterion = { version = "0.5", features = ["html_reports"] }
tempfile = "3"

[build-dependencies]
# build.rs でのみ使用
cc = "1"

[features]
default = ["json"]
json = ["serde/derive"]
async = ["dep:tokio"]
full = ["json", "async"]

# プロファイル設定
[profile.dev]
opt-level = 0
debug = true

[profile.release]
opt-level = 3
lto = "thin"          # リンク時最適化
strip = true           # デバッグ情報除去
codegen-units = 1      # 最大最適化 (ビルド遅い)
panic = "abort"        # パニック時即座に終了

[profile.bench]
inherits = "release"
debug = true           # ベンチマークのプロファイリング用
```

### コード例2: ビルドスクリプト (build.rs)

```rust
// build.rs
use std::process::Command;

fn main() {
    // Git コミットハッシュを環境変数に設定
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .expect("git コマンド実行失敗");
    let git_hash = String::from_utf8(output.stdout).unwrap();
    println!("cargo:rustc-env=GIT_HASH={}", git_hash.trim());

    // ビルド日時
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    println!("cargo:rustc-env=BUILD_TIME={}", now);

    // 条件付きコンパイルフラグ
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-cfg=has_epoll");
    }

    // 再実行条件
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/schema.sql");
}

// src/main.rs で使用:
// const GIT_HASH: &str = env!("GIT_HASH");
// const BUILD_TIME: &str = env!("BUILD_TIME");
```

---

## 3. ワークスペース

### コード例3: ワークスペース構成

```toml
# Cargo.toml (ルート)
[workspace]
members = [
    "crates/core",
    "crates/cli",
    "crates/server",
    "crates/macros",
]
resolver = "2"

# ワークスペース全体の依存バージョンを統一
[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1"
tracing = "0.1"

# ワークスペース共通のパッケージメタデータ
[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/user/project"
```

```toml
# crates/core/Cargo.toml
[package]
name = "project-core"
version.workspace = true
edition.workspace = true

[dependencies]
serde.workspace = true
anyhow.workspace = true
```

```toml
# crates/cli/Cargo.toml
[package]
name = "project-cli"
version.workspace = true
edition.workspace = true

[dependencies]
project-core = { path = "../core" }
tokio.workspace = true
clap = { version = "4", features = ["derive"] }
```

### ワークスペースの依存関係

```
┌──────────── ワークスペース依存グラフ ──────────────┐
│                                                     │
│  project-cli                project-server          │
│    │                          │                     │
│    ├── project-core          ├── project-core      │
│    ├── clap                  ├── axum              │
│    └── tokio                 └── tokio              │
│                                                     │
│  project-core                                       │
│    ├── serde                                        │
│    └── anyhow                                       │
│                                                     │
│  project-macros (proc-macro)                        │
│    ├── syn                                          │
│    ├── quote                                        │
│    └── proc-macro2                                  │
│                                                     │
│  コマンド:                                          │
│  $ cargo build -p project-cli    # 特定クレートのみ │
│  $ cargo test --workspace        # 全テスト実行     │
│  $ cargo doc --workspace         # 全ドキュメント   │
└─────────────────────────────────────────────────────┘
```

---

## 4. Feature フラグ

### コード例4: Feature フラグの実装

```rust
// Cargo.toml
// [features]
// default = ["json"]
// json = ["dep:serde_json"]
// yaml = ["dep:serde_yaml"]
// toml-support = ["dep:toml"]
// async = ["dep:tokio"]
// full = ["json", "yaml", "toml-support", "async"]

// src/lib.rs

/// JSON サポート (feature = "json" 時のみ有効)
#[cfg(feature = "json")]
pub mod json {
    use serde::{Serialize, de::DeserializeOwned};

    pub fn to_string<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(value)
    }

    pub fn from_str<T: DeserializeOwned>(s: &str) -> Result<T, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// YAML サポート
#[cfg(feature = "yaml")]
pub mod yaml {
    use serde::{Serialize, de::DeserializeOwned};

    pub fn to_string<T: Serialize>(value: &T) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(value)
    }
}

/// 非同期クライアント (feature = "async" 時のみ)
#[cfg(feature = "async")]
pub mod async_client {
    pub struct Client {
        inner: tokio::sync::RwLock<Vec<String>>,
    }

    impl Client {
        pub fn new() -> Self {
            Client {
                inner: tokio::sync::RwLock::new(Vec::new()),
            }
        }
    }
}

// feature の有無でコンパイル時にコードを切り替え
pub fn available_formats() -> Vec<&'static str> {
    let mut formats = Vec::new();

    #[cfg(feature = "json")]
    formats.push("json");

    #[cfg(feature = "yaml")]
    formats.push("yaml");

    #[cfg(feature = "toml-support")]
    formats.push("toml");

    formats
}
```

### コード例5: crates.io への公開

```bash
# 1. 公開前チェック
cargo publish --dry-run

# 2. ドキュメントの確認
cargo doc --open

# 3. パッケージ内容の確認
cargo package --list

# 4. 公開 (crates.io アカウントとトークンが必要)
cargo login  # 初回のみ
cargo publish

# ワークスペースの場合は個別に公開 (依存順序に注意)
cargo publish -p project-core
cargo publish -p project-macros
cargo publish -p project-cli
```

```rust
// バージョニング規則 (SemVer)
// MAJOR.MINOR.PATCH
//
// PATCH: バグ修正 (後方互換)
//   0.1.0 → 0.1.1
//
// MINOR: 機能追加 (後方互換)
//   0.1.0 → 0.2.0
//
// MAJOR: 破壊的変更
//   0.x.y → 1.0.0  (0.x は全て破壊的変更扱い)
//   1.0.0 → 2.0.0

// 依存バージョン指定の意味:
// "1.0"     → >=1.0.0 && <2.0.0
// "1.2.3"   → >=1.2.3 && <2.0.0
// "~1.2.3"  → >=1.2.3 && <1.3.0
// "=1.2.3"  → 1.2.3 のみ
// ">=1, <2" → 範囲指定
```

---

## 5. 比較表

### クレートタイプ比較

| crate-type | 出力 | 用途 |
|---|---|---|
| `lib` | rlib | 通常のRustライブラリ (デフォルト) |
| `bin` | 実行ファイル | CLIアプリ・サーバー |
| `cdylib` | .so / .dylib / .dll | FFI 用共有ライブラリ |
| `staticlib` | .a / .lib | C/C++ への静的リンク |
| `dylib` | .so / .dylib | Rust間の動的リンク (稀) |
| `proc-macro` | コンパイラプラグイン | derive / attribute マクロ |

### プロファイル比較

| 設定 | dev | release | 効果 |
|---|---|---|---|
| opt-level | 0 | 3 | 最適化レベル (0=なし, 3=最大) |
| debug | true | false | デバッグ情報の有無 |
| lto | false | "thin" | リンク時最適化 |
| codegen-units | 256 | 1 | 並列コンパイル単位 |
| strip | false | true | シンボル除去 |
| panic | "unwind" | "abort" | パニック時の動作 |

---

## 6. アンチパターン

### アンチパターン1: Cargo.lock をライブラリで Git 管理

```
# NG: ライブラリクレートで Cargo.lock を commit
# → 依存者が自分の Cargo.lock と競合する

# .gitignore
# ライブラリの場合:
Cargo.lock    ← ライブラリは .gitignore に含める

# OK: バイナリ(アプリ)の場合は commit する
# Cargo.lock  ← アプリは再現可能ビルドのため commit
```

### アンチパターン2: Feature フラグの過剰使用

```toml
# NG: 細かすぎる feature 分割
[features]
default = ["std"]
std = []
alloc = []
parse-int = []
parse-float = []
parse-string = []
format-int = []
format-float = []
# → ユーザーが何を有効にすべきか分からない

# OK: 意味のある粒度で分割
[features]
default = ["std"]
std = ["alloc"]
alloc = []
serde = ["dep:serde"]
async = ["dep:tokio"]
full = ["std", "serde", "async"]
# → default で基本動作、full で全機能
```

---

## FAQ

### Q1: `cargo update` と `cargo upgrade` の違いは?

**A:** `cargo update` は Cargo.lock を Cargo.toml の範囲内で最新に更新します(Cargo.toml は変更しない)。`cargo upgrade` は `cargo-edit` が提供するコマンドで、Cargo.toml 自体のバージョン指定を最新に書き換えます。

### Q2: ワークスペースの一部だけテストするには?

**A:** `-p` フラグで特定のクレートを指定します。

```bash
cargo test -p project-core          # core のテストのみ
cargo test -p project-cli -- --nocapture  # cli のテスト (出力表示)
cargo build -p project-server       # server のビルドのみ
```

### Q3: `optional = true` と `dep:` の違いは?

**A:** `optional = true` はクレート名がそのまま feature 名になります。`dep:` 構文(Rust 2021+)は feature 名とクレート名を分離でき、名前の衝突を防ぎます。

```toml
# 旧方式: tokio という feature が暗黙的に作成される
[dependencies]
tokio = { version = "1", optional = true }

# 新方式: feature 名を自由に設定
[features]
async-runtime = ["dep:tokio"]

[dependencies]
tokio = { version = "1", optional = true }
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Cargo.toml | パッケージ定義、依存、features、プロファイル |
| Cargo.lock | 再現可能ビルド。アプリは commit、ライブラリは除外 |
| ワークスペース | members で複数クレート管理。依存バージョン統一 |
| workspace.dependencies | 全クレートで依存バージョンを一元管理 |
| features | 条件付きコンパイル。default + full パターン |
| プロファイル | dev (高速ビルド) vs release (高速実行) |
| crates.io | `cargo publish` で公開。SemVer 遵守 |
| build.rs | コード生成、環境変数設定、ネイティブライブラリリンク |

## 次に読むべきガイド

- [テスト](./01-testing.md) — ワークスペース全体のテスト戦略
- [Serde](./02-serde.md) — Cargo.toml でよく使う serde 設定
- [ベストプラクティス](./04-best-practices.md) — API 設計とバージョニング

## 参考文献

1. **The Cargo Book**: https://doc.rust-lang.org/cargo/
2. **Cargo Reference — Features**: https://doc.rust-lang.org/cargo/reference/features.html
3. **API Guidelines — Crate naming**: https://rust-lang.github.io/api-guidelines/naming.html
