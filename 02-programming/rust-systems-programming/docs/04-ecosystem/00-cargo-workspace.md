# Cargo/ワークスペース — features、publish

> Cargo のパッケージ管理、ワークスペースによるモノレポ構成、feature フラグ、crates.io への公開手順を体系的に習得する

## この章で学ぶこと

1. **Cargo 基本操作** — 依存関係管理、プロファイル設定、ビルドスクリプト
2. **ワークスペース** — マルチクレート構成、依存の共有、選択的ビルド
3. **Feature フラグと公開** — 条件付きコンパイル、セマンティックバージョニング、crates.io 公開
4. **Cargo ツールチェーン** — clippy、rustfmt、cargo-deny、cargo-audit などの品質管理ツール
5. **CI/CD 統合** — GitHub Actions 等での自動ビルド・テスト・公開パイプライン

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

### 1.1 ディレクトリ規約の詳細

Cargo は規約ベースのディレクトリ構造を採用しており、各ディレクトリには明確な役割があります。

```
my-project/
├── Cargo.toml          # パッケージマニフェスト
├── Cargo.lock          # 依存の正確なバージョンを記録
├── rust-toolchain.toml # ツールチェーンバージョン指定
├── .cargo/
│   └── config.toml     # Cargo ローカル設定
├── src/
│   ├── main.rs         # デフォルトのバイナリエントリポイント
│   ├── lib.rs          # ライブラリのルートモジュール
│   └── bin/
│       ├── admin.rs    # 追加バイナリ: cargo run --bin admin
│       └── migrate/
│           └── main.rs # 追加バイナリ: cargo run --bin migrate
├── tests/
│   ├── integration_test.rs    # 各ファイルが独立したテストクレート
│   └── common/
│       └── mod.rs             # テスト共通ヘルパー
├── benches/
│   └── my_bench.rs            # criterion ベンチマーク
├── examples/
│   ├── basic_usage.rs         # cargo run --example basic_usage
│   └── advanced/
│       └── main.rs            # cargo run --example advanced
└── build.rs                   # ビルドスクリプト
```

### 1.2 .cargo/config.toml の設定

`.cargo/config.toml` はプロジェクトレベルの Cargo 設定を行うファイルです。

```toml
# .cargo/config.toml

# ビルドターゲットの指定
[build]
# クロスコンパイルのデフォルトターゲット
# target = "x86_64-unknown-linux-musl"

# ジョブ数の制限（CI 環境で有用）
jobs = 4

# リンカーの指定
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

# macOS での設定
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

# Windows MSVC での設定
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static"]

# エイリアス（カスタムコマンド）
[alias]
xtask = "run --package xtask --"
ci = "test --workspace --all-features"
lint = "clippy --workspace --all-targets --all-features -- -D warnings"
fmt-check = "fmt --all -- --check"

# レジストリ設定
[registries.my-private]
index = "sparse+https://cargo.my-company.com/index/"

# 環境変数
[env]
RUST_BACKTRACE = "1"
RUST_LOG = { value = "info", force = false }

# ネットワーク設定
[net]
retry = 3
git-fetch-with-cli = true
```

### 1.3 rust-toolchain.toml

プロジェクト全体で使用する Rust バージョンを固定するファイルです。

```toml
# rust-toolchain.toml
[toolchain]
channel = "1.78.0"
components = ["rustfmt", "clippy", "rust-analyzer", "rust-src"]
targets = ["x86_64-unknown-linux-musl", "wasm32-unknown-unknown"]
profile = "default"
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

### 2.1 依存関係の詳細な指定方法

```toml
[dependencies]
# 基本的なバージョン指定
serde = "1.0"                           # ^1.0.0 と同等
serde_json = "1.0.100"                  # ^1.0.100

# 厳密なバージョン指定
pin-project = "=1.1.3"                  # 完全一致

# バージョン範囲
rand = ">=0.8, <0.9"                    # 範囲指定

# チルダ要件（マイナーバージョンを固定）
semver = "~1.0.4"                       # >=1.0.4, <1.1.0

# ワイルドカード
uuid = "1.*"                            # >=1.0.0, <2.0.0

# Git リポジトリからの依存
my-lib = { git = "https://github.com/user/repo" }
my-lib-branch = { git = "https://github.com/user/repo", branch = "develop" }
my-lib-tag = { git = "https://github.com/user/repo", tag = "v1.0.0" }
my-lib-rev = { git = "https://github.com/user/repo", rev = "abc1234" }

# ローカルパスからの依存（開発時）
my-local = { path = "../my-local-crate" }

# パスと バージョンの両方指定（公開時に path は無視される）
my-lib = { path = "../my-lib", version = "0.1.0" }

# プラットフォーム固有の依存
[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.52", features = ["Win32_Foundation"] }

[target.'cfg(unix)'.dependencies]
nix = { version = "0.28", features = ["signal"] }

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = ["Window", "Document"] }
```

### 2.2 パッケージメタデータの詳細

```toml
[package]
name = "my-crate"
version = "1.2.3"
edition = "2021"                    # 2015, 2018, 2021, 2024
rust-version = "1.75"              # MSRV（最低サポートバージョン）

# 著者情報
authors = ["Alice <alice@example.com>", "Bob <bob@example.com>"]
description = "A brief description for crates.io"
license = "MIT OR Apache-2.0"      # SPDX 表現
license-file = "LICENSE"            # カスタムライセンスファイル

# リンク
homepage = "https://my-crate.example.com"
repository = "https://github.com/user/my-crate"
documentation = "https://docs.rs/my-crate"
readme = "README.md"

# 分類
keywords = ["parser", "json", "serialization"]   # 最大 5 個
categories = ["encoding", "parser-implementations"]

# 公開制御
publish = true                      # false で公開禁止
# publish = ["my-private-registry"]  # 特定レジストリのみに公開

# ファイル制御
include = ["src/**/*", "Cargo.toml", "LICENSE*", "README.md"]
# exclude = ["tests/fixtures/**", ".github/**"]

# バイナリ定義
[[bin]]
name = "my-tool"
path = "src/bin/tool.rs"
required-features = ["cli"]         # この feature が有効な時のみビルド

[[bin]]
name = "my-server"
path = "src/bin/server.rs"
required-features = ["server"]

# ライブラリ定義
[lib]
name = "my_crate"                   # ハイフンはアンダースコアに変換
crate-type = ["lib"]
path = "src/lib.rs"
doc = true                          # ドキュメント生成対象
doctest = true                      # ドキュメントテスト有効
test = true                         # テスト対象
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

### 2.3 高度なビルドスクリプト

```rust
// build.rs — C ライブラリとのリンク、コード生成を含む高度な例
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    let profile = env::var("PROFILE").unwrap();

    // --- 1. C/C++ ライブラリのコンパイル ---
    cc::Build::new()
        .file("native/crypto.c")
        .file("native/hash.c")
        .include("native/include")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .opt_level(if profile == "release" { 3 } else { 0 })
        .compile("native_crypto");

    // --- 2. ネイティブライブラリのリンク ---
    println!("cargo:rustc-link-lib=static=native_crypto");
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    // システムライブラリのリンク（プラットフォーム依存）
    if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=ssl");
        println!("cargo:rustc-link-lib=dylib=crypto");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Security");
    }

    // --- 3. protobuf コード生成 ---
    let proto_files = &["proto/api.proto", "proto/models.proto"];
    let proto_include = &["proto/"];

    prost_build::Config::new()
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .out_dir(&out_dir)
        .compile_protos(proto_files, proto_include)
        .expect("protobuf コンパイル失敗");

    // --- 4. バージョン情報ファイルの生成 ---
    let version_info = format!(
        r#"
        pub const VERSION: &str = "{}";
        pub const TARGET: &str = "{}";
        pub const PROFILE: &str = "{}";
        pub const GIT_HASH: &str = "{}";
        "#,
        env::var("CARGO_PKG_VERSION").unwrap(),
        target,
        profile,
        get_git_hash(),
    );
    fs::write(out_dir.join("version_info.rs"), version_info).unwrap();

    // --- 5. 再実行トリガー ---
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/");
    println!("cargo:rerun-if-changed=proto/");
    // 環境変数の変更も監視
    println!("cargo:rerun-if-env-changed=DATABASE_URL");
}

fn get_git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
```

```rust
// src/lib.rs での利用
// build.rs で生成したコードのインクルード
include!(concat!(env!("OUT_DIR"), "/version_info.rs"));

// protobuf 生成コードのインクルード
pub mod api {
    include!(concat!(env!("OUT_DIR"), "/api.rs"));
}

pub mod models {
    include!(concat!(env!("OUT_DIR"), "/models.rs"));
}
```

### 2.4 Cargo のビルドスクリプト出力命令一覧

```rust
// build.rs で使用可能な cargo: 命令の一覧

fn main() {
    // --- リンク関連 ---
    // ネイティブライブラリをリンク
    println!("cargo:rustc-link-lib=static=mylib");      // 静的リンク
    println!("cargo:rustc-link-lib=dylib=ssl");          // 動的リンク
    println!("cargo:rustc-link-lib=framework=Security"); // macOS フレームワーク

    // ライブラリ検索パス
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-search=all=/opt/lib");

    // --- コンパイラフラグ ---
    // 環境変数として設定（env!() で取得可能）
    println!("cargo:rustc-env=MY_VAR=my_value");

    // 条件付きコンパイル cfg
    println!("cargo:rustc-cfg=my_feature");
    println!("cargo:rustc-cfg=my_key=\"my_value\"");

    // コンパイラフラグ
    println!("cargo:rustc-flags=-l dylib=foo");
    println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,/usr/local/lib");

    // --- 再実行制御 ---
    // ファイル変更で再実行
    println!("cargo:rerun-if-changed=src/schema.sql");
    println!("cargo:rerun-if-changed=native/");

    // 環境変数の変更で再実行
    println!("cargo:rerun-if-env-changed=DATABASE_URL");

    // --- メタデータ ---
    // 他のクレートの build.rs から DEP_<name>_<key> として参照可能
    println!("cargo:metadata=key=value");

    // --- 警告 ---
    println!("cargo:warning=This is a build warning");
}
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

### 3.1 実践的なワークスペース設計パターン

#### パターン1: Webアプリケーション構成

```
web-platform/
├── Cargo.toml                 # ワークスペースルート
├── crates/
│   ├── domain/                # ドメインモデル（外部依存最小）
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── repository/            # データアクセス層
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── usecase/               # ビジネスロジック
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── api/                   # HTTP API サーバー
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── routes/
│   │       └── middleware/
│   ├── worker/                # バックグラウンドジョブ
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   ├── migration/             # DB マイグレーション
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   └── shared/                # 共通ユーティリティ
│       ├── Cargo.toml
│       └── src/lib.rs
└── xtask/                     # ビルドタスクランナー
    ├── Cargo.toml
    └── src/main.rs
```

```toml
# Cargo.toml (ルート)
[workspace]
members = [
    "crates/domain",
    "crates/repository",
    "crates/usecase",
    "crates/api",
    "crates/worker",
    "crates/migration",
    "crates/shared",
    "xtask",
]
resolver = "2"

[workspace.dependencies]
# ドメイン層
serde = { version = "1", features = ["derive"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4", "serde"] }

# インフラ層
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres", "chrono", "uuid"] }
redis = { version = "0.25", features = ["tokio-comp"] }

# Web 層
axum = { version = "0.7", features = ["macros"] }
tower = { version = "0.4", features = ["full"] }
tower-http = { version = "0.5", features = ["cors", "trace", "compression-gzip"] }
tokio = { version = "1", features = ["full"] }

# 横断的関心事
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
anyhow = "1"
thiserror = "1"
config = "0.14"

# テスト
wiremock = "0.6"
testcontainers = "0.15"

# ワークスペース内部
platform-domain = { path = "crates/domain" }
platform-repository = { path = "crates/repository" }
platform-usecase = { path = "crates/usecase" }
platform-shared = { path = "crates/shared" }
```

```toml
# crates/domain/Cargo.toml — 外部依存を最小限に
[package]
name = "platform-domain"
version.workspace = true
edition.workspace = true

[dependencies]
serde.workspace = true
chrono.workspace = true
uuid.workspace = true
thiserror.workspace = true
```

```toml
# crates/repository/Cargo.toml — DB アクセス
[package]
name = "platform-repository"
version.workspace = true
edition.workspace = true

[dependencies]
platform-domain.workspace = true
platform-shared.workspace = true
sqlx.workspace = true
redis.workspace = true
tokio.workspace = true
anyhow.workspace = true
tracing.workspace = true

[dev-dependencies]
testcontainers.workspace = true
tokio = { workspace = true, features = ["test-util"] }
```

```toml
# crates/api/Cargo.toml — HTTP サーバー
[package]
name = "platform-api"
version.workspace = true
edition.workspace = true

[dependencies]
platform-domain.workspace = true
platform-repository.workspace = true
platform-usecase.workspace = true
platform-shared.workspace = true
axum.workspace = true
tower.workspace = true
tower-http.workspace = true
tokio.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
config.workspace = true
anyhow.workspace = true

[dev-dependencies]
wiremock.workspace = true
reqwest = { version = "0.12", features = ["json"] }
```

#### パターン2: xtask パターン（ビルドタスクランナー）

```toml
# xtask/Cargo.toml
[package]
name = "xtask"
version = "0.1.0"
edition = "2021"
publish = false    # 公開しない

[dependencies]
clap = { version = "4", features = ["derive"] }
xshell = "0.2"
anyhow = "1"
```

```rust
// xtask/src/main.rs
use clap::{Parser, Subcommand};
use xshell::{cmd, Shell};
use anyhow::Result;

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// すべてのチェックを実行（CI 用）
    Ci,
    /// データベースマイグレーションを実行
    Migrate,
    /// Docker イメージをビルド
    Docker {
        #[arg(long, default_value = "latest")]
        tag: String,
    },
    /// リリースの準備
    Release {
        #[arg(long)]
        version: String,
    },
    /// コードカバレッジを計測
    Coverage,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;

    match cli.command {
        Commands::Ci => {
            // フォーマットチェック
            cmd!(sh, "cargo fmt --all -- --check").run()?;
            // Clippy
            cmd!(sh, "cargo clippy --workspace --all-targets --all-features -- -D warnings").run()?;
            // テスト
            cmd!(sh, "cargo test --workspace --all-features").run()?;
            // ドキュメントテスト
            cmd!(sh, "cargo doc --workspace --no-deps").run()?;
            println!("CI checks passed!");
        }
        Commands::Migrate => {
            cmd!(sh, "cargo run -p platform-migration").run()?;
        }
        Commands::Docker { tag } => {
            cmd!(sh, "docker build -t platform-api:{tag} -f docker/Dockerfile .").run()?;
        }
        Commands::Release { version } => {
            // バージョン更新、タグ作成、公開
            println!("Releasing version {version}...");
            cmd!(sh, "cargo set-version --workspace {version}").run()?;
            cmd!(sh, "cargo check --workspace").run()?;
            cmd!(sh, "git add -A").run()?;
            cmd!(sh, "git commit -m 'Release v{version}'").run()?;
            cmd!(sh, "git tag v{version}").run()?;
        }
        Commands::Coverage => {
            cmd!(sh, "cargo llvm-cov --workspace --html").run()?;
            println!("Coverage report generated in target/llvm-cov/html/");
        }
    }

    Ok(())
}
```

```toml
# .cargo/config.toml でエイリアス設定
[alias]
xtask = "run --package xtask --"
```

```bash
# 使用例
cargo xtask ci          # CI チェック実行
cargo xtask migrate     # マイグレーション
cargo xtask docker --tag v1.0.0  # Docker ビルド
cargo xtask release --version 1.0.0  # リリース
cargo xtask coverage    # カバレッジ計測
```

### 3.2 ワークスペースの exclude と default-members

```toml
[workspace]
members = [
    "crates/*",
    "xtask",
]
# ワークスペースのメンバーから除外
exclude = [
    "crates/experimental",
    "tools/standalone",
]
# cargo build 時のデフォルト対象
default-members = [
    "crates/api",
    "crates/worker",
]
resolver = "2"
```

### 3.3 ワークスペース内のクレート間依存ルール

```
┌─────────────────────────────────────────────────────────┐
│            依存の方向（許可される矢印のみ）               │
│                                                          │
│   api ──────┐                                            │
│   worker ───┤                                            │
│             ▼                                            │
│          usecase ──► domain                              │
│             │           ▲                                │
│             ▼           │                                │
│         repository ─────┘                                │
│                                                          │
│   ルール:                                                │
│   ✓ 上位層 → 下位層 への依存                             │
│   ✗ 下位層 → 上位層 への依存（循環依存禁止）             │
│   ✓ domain は外部クレートへの依存を最小限にする          │
│   ✓ shared はどの層からも参照可能                        │
└─────────────────────────────────────────────────────────┘
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

### 4.1 Feature フラグの高度なパターン

```toml
# Cargo.toml — 実践的な feature 設計
[features]
default = ["std", "json"]

# 基盤 features
std = ["alloc", "serde/std", "chrono/std"]
alloc = []

# フォーマット features
json = ["dep:serde_json"]
yaml = ["dep:serde_yaml"]
toml-support = ["dep:toml"]
msgpack = ["dep:rmp-serde"]

# ランタイム features
async-tokio = ["dep:tokio", "dep:tokio-stream"]
async-async-std = ["dep:async-std"]

# データベース features
postgres = ["dep:sqlx", "sqlx/postgres"]
mysql = ["dep:sqlx", "sqlx/mysql"]
sqlite = ["dep:sqlx", "sqlx/sqlite"]

# TLS features
native-tls = ["dep:native-tls"]
rustls = ["dep:rustls", "dep:webpki-roots"]

# メタ features
full = [
    "std", "json", "yaml", "toml-support", "msgpack",
    "async-tokio", "postgres", "mysql", "sqlite",
    "rustls",
]

# 内部 features（ユーザーが直接使用しない）
__internal_bench = []

[dependencies]
serde = { version = "1", default-features = false, features = ["derive"] }
chrono = { version = "0.4", default-features = false, optional = false }

serde_json = { version = "1", optional = true }
serde_yaml = { version = "0.9", optional = true }
toml = { version = "0.8", optional = true }
rmp-serde = { version = "1", optional = true }

tokio = { version = "1", features = ["full"], optional = true }
tokio-stream = { version = "0.1", optional = true }
async-std = { version = "1", optional = true }

sqlx = { version = "0.7", optional = true, default-features = false, features = ["runtime-tokio"] }

native-tls = { version = "0.2", optional = true }
rustls = { version = "0.23", optional = true }
webpki-roots = { version = "0.26", optional = true }
```

```rust
// src/lib.rs — feature フラグを活用した条件付きコンパイル

// std/alloc の段階的対応
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

// 複数 feature の組み合わせ
#[cfg(all(feature = "json", feature = "async-tokio"))]
pub mod async_json {
    use serde::{Serialize, de::DeserializeOwned};
    use tokio::io::{AsyncRead, AsyncReadExt};

    pub async fn from_reader<T, R>(reader: &mut R) -> Result<T, Box<dyn std::error::Error>>
    where
        T: DeserializeOwned,
        R: AsyncRead + Unpin,
    {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        let value = serde_json::from_slice(&buf)?;
        Ok(value)
    }
}

// いずれかの feature が有効な場合
#[cfg(any(feature = "postgres", feature = "mysql", feature = "sqlite"))]
pub mod database {
    pub fn is_database_enabled() -> bool {
        true
    }

    #[cfg(feature = "postgres")]
    pub mod postgres {
        pub fn connection_string_prefix() -> &'static str {
            "postgres://"
        }
    }
}

// feature が無効な場合のフォールバック
#[cfg(not(any(feature = "native-tls", feature = "rustls")))]
compile_error!("Either 'native-tls' or 'rustls' feature must be enabled");

// テストでの feature 確認
#[cfg(test)]
mod tests {
    #[test]
    fn test_available_features() {
        let features = crate::available_formats();

        #[cfg(feature = "json")]
        assert!(features.contains(&"json"));

        #[cfg(not(feature = "json"))]
        assert!(!features.contains(&"json"));
    }
}
```

### 4.2 Feature の注意事項と cfg 属性の詳細

```rust
// cfg 属性の様々な使い方

// --- プラットフォーム検出 ---
#[cfg(target_os = "linux")]
fn platform_specific() -> &'static str { "Linux" }

#[cfg(target_os = "macos")]
fn platform_specific() -> &'static str { "macOS" }

#[cfg(target_os = "windows")]
fn platform_specific() -> &'static str { "Windows" }

// --- アーキテクチャ検出 ---
#[cfg(target_arch = "x86_64")]
fn simd_operation(data: &[f32]) -> f32 {
    // SSE/AVX 命令を使用した最適化
    data.iter().sum()
}

#[cfg(target_arch = "aarch64")]
fn simd_operation(data: &[f32]) -> f32 {
    // NEON 命令を使用した最適化
    data.iter().sum()
}

// --- cfg_attr: 条件付き属性 ---
// serde feature が有効な場合のみ derive
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Config {
    pub name: String,
    pub value: i32,
}

// release ビルドではインライン化
#[cfg_attr(not(debug_assertions), inline(always))]
fn hot_path(x: i32) -> i32 {
    x * 2 + 1
}

// --- cfg! マクロ（ランタイム判定ではなくコンパイル時判定） ---
fn log_platform() {
    if cfg!(target_os = "linux") {
        println!("Running on Linux");
    } else if cfg!(target_os = "macos") {
        println!("Running on macOS");
    }
    // 注意: 上記は if 文に見えるがコンパイル時に確定する
    // 到達不能なブランチはコンパイラが除去する
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

### 4.3 公開前チェックリストと自動化

```bash
# --- 公開前の包括的チェック ---

# 1. コンパイルチェック（全 feature 組み合わせ）
cargo check --all-features
cargo check --no-default-features
cargo check --features "json"
cargo check --features "yaml"

# 2. テスト
cargo test --all-features
cargo test --no-default-features

# 3. Clippy
cargo clippy --all-features -- -D warnings

# 4. フォーマット
cargo fmt --check

# 5. ドキュメント（壊れたリンクの検出）
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# 6. 最低サポートバージョン (MSRV) チェック
cargo +1.75.0 check --all-features

# 7. セキュリティ監査
cargo audit
cargo deny check

# 8. ライセンスチェック
cargo deny check licenses

# 9. パッケージサイズの確認
cargo package --list | wc -l
du -sh target/package/

# 10. ドライラン
cargo publish --dry-run
```

```rust
// 自動公開スクリプト（xtask）
// xtask/src/publish.rs
use xshell::{cmd, Shell};
use anyhow::Result;

pub fn publish_workspace(sh: &Shell, dry_run: bool) -> Result<()> {
    // 公開順序（依存関係順）
    let publish_order = vec![
        "project-core",
        "project-macros",
        "project-shared",
        "project-repository",
        "project-usecase",
        "project-cli",
        "project-api",
    ];

    for crate_name in &publish_order {
        println!("Publishing {}...", crate_name);

        if dry_run {
            cmd!(sh, "cargo publish -p {crate_name} --dry-run").run()?;
        } else {
            cmd!(sh, "cargo publish -p {crate_name}").run()?;
            // crates.io のインデックス更新を待つ
            std::thread::sleep(std::time::Duration::from_secs(30));
        }
    }

    Ok(())
}
```

---

## 5. プロファイル設定の詳細

### 5.1 各プロファイルの詳細比較

```toml
# --- 開発用プロファイル ---
[profile.dev]
opt-level = 0          # 最適化なし（ビルド高速）
debug = true           # 完全なデバッグ情報
debug-assertions = true # debug_assert!() 有効
overflow-checks = true  # 整数オーバーフロー検査
lto = false            # LTO 無効
codegen-units = 256    # 並列コンパイル（ビルド高速）
incremental = true     # インクリメンタルコンパイル
panic = "unwind"       # パニック時スタック巻き戻し
strip = "none"         # シンボル除去なし
split-debuginfo = "packed"  # デバッグ情報の格納方式

# --- リリース用プロファイル ---
[profile.release]
opt-level = 3          # 最大最適化
debug = false          # デバッグ情報なし
debug-assertions = false
overflow-checks = false
lto = "thin"           # Thin LTO（バランス良い）
codegen-units = 1      # 単一コンパイル単位（最大最適化）
incremental = false    # インクリメンタル無効
panic = "abort"        # パニック時即座に終了（バイナリサイズ削減）
strip = true           # シンボル除去

# --- ベンチマーク用（release を継承） ---
[profile.bench]
inherits = "release"
debug = true           # プロファイリング用にデバッグ情報を残す

# --- カスタムプロファイル: 本番デプロイ用 ---
[profile.production]
inherits = "release"
lto = "fat"            # Fat LTO（最大最適化、ビルド非常に遅い）
codegen-units = 1
strip = true
panic = "abort"

# --- カスタムプロファイル: 開発時だが少し最適化 ---
[profile.dev-fast]
inherits = "dev"
opt-level = 1          # 最小限の最適化（主にジェネリクスの肥大化防止）
debug = true

# --- 依存クレートのみ最適化（自分のコードは高速ビルド） ---
[profile.dev.package."*"]
opt-level = 2          # 依存は最適化（実行速度改善）

# 特定の依存を最適化
[profile.dev.package.sqlx]
opt-level = 3

[profile.dev.package.image]
opt-level = 3
```

### 5.2 バイナリサイズ最適化

```toml
# 最小バイナリサイズを目指すプロファイル
[profile.min-size]
inherits = "release"
opt-level = "z"        # サイズ最適化 ("s" でもOK)
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true

# Cargo.toml の [dependencies] で default-features = false を使う
# 不要な機能を除外してバイナリサイズを削減
```

```bash
# バイナリサイズの確認と分析
cargo build --release
ls -la target/release/my-app

# cargo-bloat でサイズ分析
cargo install cargo-bloat
cargo bloat --release -n 20          # 大きい関数 TOP20
cargo bloat --release --crates       # クレート別サイズ

# twiggy で詳細分析
cargo install twiggy
twiggy top target/release/my-app -n 20
twiggy dominators target/release/my-app
```

---

## 6. Cargo ツールチェーン

### 6.1 必須ツール一覧

```bash
# --- フォーマッター ---
rustup component add rustfmt
cargo fmt                          # フォーマット実行
cargo fmt -- --check               # フォーマットチェック（CI 用）
cargo fmt -- --emit files          # 変更ファイルのみ表示

# --- リンター ---
rustup component add clippy
cargo clippy                                           # 基本チェック
cargo clippy --all-targets --all-features              # 全対象チェック
cargo clippy -- -D warnings                            # 警告をエラーに
cargo clippy -- -W clippy::pedantic                    # より厳格なチェック
cargo clippy --fix                                     # 自動修正

# --- セキュリティ監査 ---
cargo install cargo-audit
cargo audit                        # 脆弱性チェック
cargo audit fix                    # 自動修正

# --- 依存ライセンス・ポリシーチェック ---
cargo install cargo-deny
cargo deny init                    # deny.toml 生成
cargo deny check                   # 全チェック
cargo deny check licenses          # ライセンスのみ
cargo deny check bans              # 禁止クレートチェック
cargo deny check advisories        # セキュリティアドバイザリ

# --- 依存の更新 ---
cargo install cargo-edit
cargo upgrade                      # 依存を最新に更新
cargo upgrade --incompatible       # 破壊的変更含む更新
cargo add serde --features derive  # 依存の追加

# --- 未使用依存の検出 ---
cargo install cargo-machete
cargo machete                      # 未使用クレートの検出

# --- ビルド時間分析 ---
cargo install cargo-timings
cargo build --timings              # ビルド時間の詳細レポート

# --- コードカバレッジ ---
cargo install cargo-llvm-cov
cargo llvm-cov                     # テストカバレッジ計測
cargo llvm-cov --html              # HTML レポート
cargo llvm-cov --lcov              # LCOV 形式

# --- クロスコンパイル ---
cargo install cross
cross build --target x86_64-unknown-linux-musl
cross build --target aarch64-unknown-linux-gnu

# --- ドキュメント ---
cargo doc --all-features --no-deps --open
```

### 6.2 rustfmt.toml 設定

```toml
# rustfmt.toml
edition = "2021"
max_width = 100
tab_spaces = 4
use_small_heuristics = "Default"
newline_style = "Unix"

# インポート
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
reorder_imports = true

# 関数
fn_params_layout = "Tall"
fn_single_line = false

# 構造体
struct_lit_single_line = true
struct_variant_force_align = false

# コメント
normalize_comments = false
normalize_doc_attributes = false
wrap_comments = false

# その他
format_code_in_doc_comments = true
format_macro_matchers = true
use_field_init_shorthand = true
use_try_shorthand = true
```

### 6.3 clippy.toml 設定

```toml
# clippy.toml
# Clippy のプロジェクト固有設定

# 関数の認知的複雑度の上限
cognitive-complexity-threshold = 25

# 列挙型のバリアント数の上限
enum-variant-size-threshold = 200

# 型の最大サイズ（バイト）
too-large-for-stack = 200

# 許可されるワイルドカードインポートのモジュール
allowed-wildcard-imports = ["prelude"]

# MSRVの指定
msrv = "1.75.0"

# 型名の最大長
type-complexity-threshold = 250
```

```rust
// Clippy 属性の使い方

// ファイル全体で特定の警告を許可
#![allow(clippy::module_inception)]

// 関数レベルで許可
#[allow(clippy::too_many_arguments)]
fn complex_function(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32) {
    // ...
}

// 構造体レベルで許可
#[allow(clippy::large_enum_variant)]
enum Message {
    Small(u8),
    Large([u8; 1024]),
}

// 厳格モード（pedantic lint を有効化）
#![warn(clippy::pedantic)]
// 特定の pedantic lint を除外
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]

// restriction lint（明示的に有効化が必要）
#![warn(clippy::dbg_macro)]        // dbg! マクロの使用警告
#![warn(clippy::print_stdout)]     // println! の使用警告
#![warn(clippy::unwrap_used)]      // unwrap() の使用警告
```

### 6.4 cargo-deny の設定

```toml
# deny.toml

[graph]
targets = [
    { triple = "x86_64-unknown-linux-gnu" },
    { triple = "aarch64-apple-darwin" },
    { triple = "x86_64-pc-windows-msvc" },
]

[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "deny"
notice = "warn"
ignore = []

[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
    "Zlib",
]
deny = [
    "GPL-2.0",
    "GPL-3.0",
    "AGPL-3.0",
]
copyleft = "warn"

[[licenses.clarify]]
name = "ring"
expression = "MIT AND ISC AND OpenSSL"
license-files = [{ path = "LICENSE", hash = 0xbd0eed23 }]

[bans]
multiple-versions = "warn"
wildcards = "deny"
highlight = "all"

# 特定クレートの禁止
deny = [
    { name = "openssl" },  # rustls を使用するポリシー
]

# 特定クレートの重複を許可
skip = [
    { name = "syn", version = "1" },  # proc-macro で syn v1 と v2 が共存
]

[sources]
unknown-registry = "deny"
unknown-git = "warn"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

---

## 7. CI/CD 統合

### 7.1 GitHub Actions の設定

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    name: Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2

      - name: Format check
        run: cargo fmt --all -- --check

      - name: Clippy
        run: cargo clippy --workspace --all-targets --all-features

      - name: Build
        run: cargo build --workspace --all-features

  test:
    name: Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, "1.75.0"]  # stable + MSRV
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
      - uses: Swatinem/rust-cache@v2

      - name: Test (default features)
        run: cargo test --workspace

      - name: Test (all features)
        run: cargo test --workspace --all-features

      - name: Test (no default features)
        run: cargo test --workspace --no-default-features

  coverage:
    name: Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: llvm-tools-preview
      - uses: Swatinem/rust-cache@v2
      - uses: taiki-e/install-action@cargo-llvm-cov

      - name: Generate coverage
        run: cargo llvm-cov --workspace --all-features --lcov --output-path lcov.info

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: lcov.info

  security:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: rustsec/audit-check@v2
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

  deny:
    name: Dependency Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: EmbarkStudios/cargo-deny-action@v1
        with:
          command: check
          arguments: --all-features

  publish:
    name: Publish
    needs: [check, test, security, deny]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Publish to crates.io
        run: |
          cargo publish -p project-core
          sleep 30
          cargo publish -p project-macros
          sleep 30
          cargo publish -p project-cli
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}
```

### 7.2 Docker マルチステージビルド

```dockerfile
# Dockerfile — Rust マルチステージビルド
# ステージ1: ビルド
FROM rust:1.78-slim-bookworm AS builder

WORKDIR /app

# 依存のキャッシュ層（ソースコード変更時に再ビルド不要）
COPY Cargo.toml Cargo.lock ./
COPY crates/core/Cargo.toml crates/core/Cargo.toml
COPY crates/api/Cargo.toml crates/api/Cargo.toml
# ダミーソースでキャッシュ層を作成
RUN mkdir -p crates/core/src && echo "pub fn dummy() {}" > crates/core/src/lib.rs && \
    mkdir -p crates/api/src && echo "fn main() {}" > crates/api/src/main.rs && \
    cargo build --release -p platform-api && \
    rm -rf crates/

# 実際のソースをコピーしてビルド
COPY . .
RUN touch crates/core/src/lib.rs crates/api/src/main.rs && \
    cargo build --release -p platform-api

# ステージ2: 実行環境
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/platform-api /usr/local/bin/
EXPOSE 8080
CMD ["platform-api"]
```

---

## 8. 比較表

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

### バージョン指定方法の比較

| 記法 | 例 | 意味 | ユースケース |
|---|---|---|---|
| キャレット（デフォルト） | `"1.2.3"` | `>=1.2.3, <2.0.0` | 通常使用 |
| チルダ | `"~1.2.3"` | `>=1.2.3, <1.3.0` | パッチ更新のみ許可 |
| 完全一致 | `"=1.2.3"` | `1.2.3` のみ | 厳密なバージョン固定 |
| ワイルドカード | `"1.*"` | `>=1.0.0, <2.0.0` | メジャーバージョンのみ固定 |
| 範囲 | `">=1.2, <1.5"` | `>=1.2.0, <1.5.0` | 細かい範囲指定 |
| 複合 | `">=1.2, <2"` | `>=1.2.0, <2.0.0` | 最低バージョン指定 |

### Cargo コマンド一覧

| コマンド | 説明 | よく使うフラグ |
|---|---|---|
| `cargo new` | プロジェクト作成 | `--lib`, `--name` |
| `cargo init` | 既存ディレクトリで初期化 | `--lib` |
| `cargo build` | ビルド | `--release`, `-p <crate>` |
| `cargo run` | ビルド＆実行 | `--release`, `--bin <name>` |
| `cargo test` | テスト実行 | `--workspace`, `--lib`, `--doc` |
| `cargo bench` | ベンチマーク | `-p <crate>` |
| `cargo check` | コンパイルチェック（バイナリ生成なし） | `--all-features` |
| `cargo clippy` | リント | `-- -D warnings` |
| `cargo fmt` | フォーマット | `--check` |
| `cargo doc` | ドキュメント生成 | `--open`, `--no-deps` |
| `cargo publish` | crates.io 公開 | `--dry-run`, `-p <crate>` |
| `cargo update` | Cargo.lock の更新 | `-p <crate>` |
| `cargo tree` | 依存ツリー表示 | `--duplicates`, `-i <crate>` |
| `cargo clean` | ビルド成果物削除 | `-p <crate>` |
| `cargo vendor` | 依存をローカルにコピー | |
| `cargo fix` | 自動修正 | `--edition` |

---

## 9. アンチパターン

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

### アンチパターン3: 循環依存

```
# NG: クレート間の循環依存
crate-a → crate-b → crate-a  ← コンパイルエラー

# OK: 共通部分を分離
crate-a → crate-common
crate-b → crate-common
```

### アンチパターン4: ワークスペースでバージョンが不統一

```toml
# NG: 各クレートでバージョンがバラバラ
# crates/a/Cargo.toml
[dependencies]
serde = "1.0.190"

# crates/b/Cargo.toml
[dependencies]
serde = "1.0.195"

# OK: workspace.dependencies で統一
[workspace.dependencies]
serde = { version = "1.0.195", features = ["derive"] }

# 各クレートで workspace = true を使う
[dependencies]
serde.workspace = true
```

### アンチパターン5: `*` ワイルドカードバージョン

```toml
# NG: ワイルドカード使用
[dependencies]
serde = "*"                # どのバージョンでも OK（危険）

# OK: 適切なバージョン範囲
[dependencies]
serde = "1"                # ^1.0.0（SemVer 互換範囲）
```

### アンチパターン6: dev-dependencies にリリース依存を混在

```toml
# NG: テスト用の依存が本番にも含まれる
[dependencies]
tokio = { version = "1", features = ["full"] }
pretty_assertions = "1"    # テスト用なのに通常 dep に

# OK: テスト用は dev-dependencies に
[dependencies]
tokio = { version = "1", features = ["full"] }

[dev-dependencies]
pretty_assertions = "1"    # テスト・例のみで使用
tokio = { version = "1", features = ["test-util"] }  # テスト用追加 feature
```

---

## 10. パフォーマンス最適化

### 10.1 ビルド時間の短縮

```toml
# .cargo/config.toml — ビルド高速化設定

# mold リンカー（Linux）
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]

# lld リンカー（macOS）
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

# sccache を使用（分散キャッシュ）
[build]
rustc-wrapper = "sccache"
```

```bash
# ビルド時間の計測と分析
cargo build --timings             # HTML レポート生成
# target/cargo-timings/cargo-timing.html を確認

# 特定クレートの依存チェーン分析
cargo tree -p heavy-crate --depth 3

# コンパイル時間が長いクレートの特定
cargo build 2>&1 | grep "Compiling" | tail -20
```

### 10.2 依存関係の管理戦略

```bash
# 依存ツリーの分析
cargo tree                         # 完全な依存ツリー
cargo tree --duplicates            # 重複する依存の表示
cargo tree -i serde                # serde を使っているクレートを逆引き
cargo tree --depth 1               # 直接依存のみ
cargo tree -e features             # feature の伝播を表示
cargo tree -f "{p} {f}"            # パッケージと feature を表示

# 未使用依存の検出
cargo machete

# 依存の最新バージョン確認
cargo outdated                     # cargo-outdated が必要
```

---

## FAQ

### Q1: `cargo update` と `cargo upgrade` の違いは?

**A:** `cargo update` は Cargo.lock を Cargo.toml の範囲内で最新に更新します(Cargo.toml は変更しない)。`cargo upgrade` は `cargo-edit` が提供するコマンドで、Cargo.toml 自体のバージョン指定を最新に書き換えます。

```bash
# Cargo.toml: serde = "1.0.190" の場合
cargo update -p serde    # Cargo.lock を 1.0.x の最新に
                          # (Cargo.toml は変更しない)

cargo upgrade -p serde   # Cargo.toml の serde を最新バージョンに書き換え
                          # serde = "1.0.190" → serde = "1.0.210"
```

### Q2: ワークスペースの一部だけテストするには?

**A:** `-p` フラグで特定のクレートを指定します。

```bash
cargo test -p project-core          # core のテストのみ
cargo test -p project-cli -- --nocapture  # cli のテスト (出力表示)
cargo build -p project-server       # server のビルドのみ
cargo clippy -p project-core        # core の lint のみ
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

### Q4: プライベートレジストリの設定方法は?

**A:** `.cargo/config.toml` でレジストリを設定し、`Cargo.toml` で指定します。

```toml
# .cargo/config.toml
[registries.my-company]
index = "sparse+https://cargo.my-company.com/index/"

# Cargo.toml
[dependencies]
internal-lib = { version = "1.0", registry = "my-company" }

[package]
publish = ["my-company"]  # このレジストリにのみ公開可能
```

### Q5: MSRV（最低サポートバージョン）の管理方法は?

**A:** `rust-version` フィールドと CI で管理します。

```toml
# Cargo.toml
[package]
rust-version = "1.75"  # この Rust バージョン以上で動作保証

# CI で MSRV チェック
# cargo +1.75.0 check --all-features
```

```bash
# cargo-msrv で最低バージョンを自動検出
cargo install cargo-msrv
cargo msrv find              # 自動的に MSRV を検出
cargo msrv verify            # 指定した MSRV で動作確認
```

### Q6: ワークスペース全体の依存を一括更新するには?

**A:** `cargo update` はワークスペース全体の `Cargo.lock` を更新します。`cargo upgrade` で `workspace.dependencies` も更新可能です。

```bash
# Cargo.lock の更新（Cargo.toml の範囲内）
cargo update

# Cargo.toml の workspace.dependencies を最新に
cargo upgrade --workspace

# 特定のクレートのみ更新
cargo update -p serde
cargo upgrade -p serde --workspace
```

### Q7: ビルドキャッシュの問題をデバッグするには?

**A:** `CARGO_LOG` 環境変数や `--verbose` フラグを使用します。

```bash
# 詳細なビルドログ
cargo build -v               # verbose モード
cargo build -vv              # さらに詳細

# Cargo の内部ログ
CARGO_LOG=cargo::core::compiler=trace cargo build

# キャッシュの無効化（クリーンビルド）
cargo clean
cargo build

# 特定クレートのみクリーン
cargo clean -p my-crate
```

### Q8: feature フラグの組み合わせテストを効率化するには?

**A:** `cargo hack` を使用して feature の組み合わせを自動テストできます。

```bash
# cargo-hack インストール
cargo install cargo-hack

# 全 feature の組み合わせをテスト
cargo hack test --feature-powerset

# 各 feature を個別にテスト
cargo hack check --each-feature

# default features なしで各 feature を個別テスト
cargo hack check --each-feature --no-default-features

# --group-features で関連 feature をまとめてテスト
cargo hack check --feature-powerset --group-features json,yaml,toml-support
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
| .cargo/config.toml | リンカー設定、エイリアス、ネットワーク設定 |
| rust-toolchain.toml | ツールチェーンバージョン固定 |
| cargo-deny | ライセンス・セキュリティ・依存ポリシーチェック |
| xtask パターン | Rust 製のタスクランナーでビルド自動化 |
| CI/CD | GitHub Actions で自動テスト・公開パイプライン |
| パフォーマンス | mold/lld リンカー、sccache、profile 最適化 |

## 次に読むべきガイド

- [テスト](./01-testing.md) — ワークスペース全体のテスト戦略
- [Serde](./02-serde.md) — Cargo.toml でよく使う serde 設定
- [ベストプラクティス](./04-best-practices.md) — API 設計とバージョニング

## 参考文献

1. **The Cargo Book**: https://doc.rust-lang.org/cargo/
2. **Cargo Reference — Features**: https://doc.rust-lang.org/cargo/reference/features.html
3. **API Guidelines — Crate naming**: https://rust-lang.github.io/api-guidelines/naming.html
4. **Cargo Reference — Build Scripts**: https://doc.rust-lang.org/cargo/reference/build-scripts.html
5. **Cargo Reference — Profiles**: https://doc.rust-lang.org/cargo/reference/profiles.html
6. **Cargo Reference — Workspaces**: https://doc.rust-lang.org/cargo/reference/workspaces.html
7. **cargo-deny documentation**: https://embarkstudios.github.io/cargo-deny/
8. **Rust API Guidelines**: https://rust-lang.github.io/api-guidelines/
9. **cargo-hack documentation**: https://github.com/taiki-e/cargo-hack
10. **Swatinem/rust-cache GitHub Action**: https://github.com/Swatinem/rust-cache
