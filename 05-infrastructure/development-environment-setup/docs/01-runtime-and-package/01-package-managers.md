# パッケージマネージャー

> npm / pnpm / yarn、pip / poetry / uv、cargo、Homebrew など主要パッケージマネージャーの設定・運用・使い分けを解説する実践ガイド。

## この章で学ぶこと

1. Node.js エコシステムのパッケージマネージャー（npm / pnpm / yarn）の特性と選定基準
2. Python（pip / poetry / uv）と Rust（cargo）のパッケージ管理手法
3. Homebrew によるシステムツール管理とチーム統一の方法

---

## 1. Node.js パッケージマネージャー

### 1.1 比較表

| 特徴 | npm | pnpm | yarn (berry/v4) |
|------|-----|------|-----------------|
| 付属 | Node.js 同梱 | 別途インストール | 別途インストール |
| ディスク効率 | 低い (各プロジェクトに複製) | 高い (コンテンツアドレス) | 中 (PnP で最小) |
| インストール速度 | 普通 | 高速 | 高速 |
| ロックファイル | package-lock.json | pnpm-lock.yaml | yarn.lock |
| ワークスペース | あり | あり (優秀) | あり |
| Phantom Dependencies | あり | なし (厳密) | なし (PnP) |
| 学習コスト | 低 | 低 | 中 (PnP) |

### 1.2 npm の設定

```bash
# ─── 初期設定 ───
npm config set init-author-name "Your Name"
npm config set init-author-email "your@email.com"
npm config set init-license "MIT"
npm config set save-exact true        # バージョン固定

# ─── .npmrc (プロジェクトルート) ───
cat << 'EOF' > .npmrc
engine-strict=true
save-exact=true
package-lock=true
fund=false
audit-level=moderate
EOF

# ─── 基本操作 ───
npm init -y                           # 初期化
npm install express                   # 依存追加
npm install -D typescript             # 開発依存追加
npm ci                                # ロックファイルから厳密インストール (CI用)
npm audit                             # 脆弱性チェック
npm outdated                          # 更新可能パッケージ表示
npm update                            # パッチ/マイナー更新
```

### 1.3 pnpm の設定

```bash
# ─── インストール ───
# corepack (Node.js 16.13+ 同梱)
corepack enable
corepack prepare pnpm@latest --activate

# または直接インストール
npm install -g pnpm

# ─── .npmrc (pnpm対応) ───
cat << 'EOF' > .npmrc
shamefully-hoist=false
strict-peer-dependencies=true
auto-install-peers=true
EOF

# ─── 基本操作 ───
pnpm install                          # 依存インストール
pnpm add express                      # 依存追加
pnpm add -D typescript                # 開発依存追加
pnpm remove lodash                    # 依存削除
pnpm up --latest                      # 全パッケージ最新化
pnpm store prune                      # 不要キャッシュ削除
```

### 1.4 pnpm の仕組み

```
pnpm のコンテンツアドレス型ストレージ:

  従来 (npm):
  ┌──────────────┐  ┌──────────────┐
  │ Project A    │  │ Project B    │
  │ node_modules │  │ node_modules │
  │ ├── lodash/  │  │ ├── lodash/  │  同じパッケージが
  │ │  (4.17.21) │  │ │  (4.17.21) │  プロジェクトごとに複製
  │ └── express/ │  │ └── axios/   │  → ディスク浪費
  └──────────────┘  └──────────────┘

  pnpm:
  ~/.local/share/pnpm/store/
  ├── lodash@4.17.21/        ← 1箇所に保存
  └── express@4.18.2/

  Project A/node_modules/     Project B/node_modules/
  └── .pnpm/                  └── .pnpm/
      └── lodash@4.17.21         └── lodash@4.17.21
          → ハードリンク             → ハードリンク
            (ディスクコピーなし)       (ディスクコピーなし)

  効果: ディスク使用量 50-70% 削減、インストール 2-3倍高速
```

### 1.5 yarn (v4/berry) の設定

```bash
# ─── インストール ───
corepack enable
corepack prepare yarn@stable --activate

# ─── プロジェクト初期化 ───
yarn init -2                          # yarn berry プロジェクト作成
yarn set version stable               # 最新安定版に設定

# ─── .yarnrc.yml ───
cat << 'EOF' > .yarnrc.yml
nodeLinker: node-modules               # PnP を使わない場合
enableGlobalCache: true
nmHoistingLimits: workspaces
EOF

# ─── 基本操作 ───
yarn install                          # 依存インストール
yarn add express                      # 依存追加
yarn add -D typescript                # 開発依存追加
yarn up --interactive                 # 対話的アップデート
```

---

## 2. Python パッケージマネージャー

### 2.1 比較表

| 特徴 | pip + venv | Poetry | uv |
|------|-----------|--------|-----|
| 速度 | 遅い | 普通 | 超高速 (10-100倍) |
| 依存解決 | 基本的 | 高度 | 高度 |
| ロックファイル | なし (手動生成) | poetry.lock | uv.lock |
| 仮想環境 | 手動作成 | 自動 | 自動 |
| ビルド | setuptools | 組み込み | 組み込み |
| pyproject.toml | 部分対応 | 完全対応 | 完全対応 |

### 2.2 uv (推奨 — 次世代パッケージマネージャー)

```bash
# ─── インストール ───
curl -LsSf https://astral.sh/uv/install.sh | sh
# または
brew install uv

# ─── プロジェクト初期化 ───
uv init my-project
cd my-project

# ─── 依存管理 ───
uv add requests                       # 依存追加
uv add --dev pytest ruff              # 開発依存追加
uv remove requests                    # 依存削除
uv lock                               # ロックファイル生成
uv sync                               # ロックファイルから同期

# ─── スクリプト実行 ───
uv run python main.py                 # 仮想環境内で実行
uv run pytest                         # テスト実行
uv run ruff check .                   # リント実行

# ─── Python バージョン管理 (pyenv 不要) ───
uv python install 3.12                # Python 自体もインストール可能
uv python pin 3.12                    # プロジェクトに固定
```

### 2.3 Poetry

```bash
# ─── インストール ───
curl -sSL https://install.python-poetry.org | python3 -

# ─── 設定 ───
poetry config virtualenvs.in-project true   # .venv をプロジェクト内に作成

# ─── プロジェクト初期化 ───
poetry init                           # 対話的初期化
poetry install                        # 依存インストール

# ─── 依存管理 ───
poetry add requests                   # 依存追加
poetry add --group dev pytest         # 開発依存追加
poetry lock                           # ロックファイル更新
poetry show --outdated                # 更新可能パッケージ

# ─── 実行 ───
poetry run python main.py
poetry shell                          # 仮想環境を有効化
```

---

## 3. Rust パッケージマネージャー (Cargo)

### 3.1 基本操作

```bash
# ─── プロジェクト作成 ───
cargo new my-project                  # バイナリプロジェクト
cargo new --lib my-lib                # ライブラリプロジェクト

# ─── 依存管理 (Cargo.toml) ───
cargo add serde --features derive     # 依存追加
cargo add tokio -F full               # feature flag 付き
cargo add --dev mockall               # 開発依存

# ─── ビルド & 実行 ───
cargo build                           # デバッグビルド
cargo build --release                 # リリースビルド
cargo run                             # ビルド + 実行
cargo test                            # テスト実行
cargo clippy                          # リント
cargo fmt                             # フォーマット
```

### 3.2 Cargo.toml の設定

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1"
tracing = "0.1"

[dev-dependencies]
mockall = "0.12"
tokio-test = "0.4"

[profile.release]
lto = true
codegen-units = 1
strip = true
```

---

## 4. Homebrew (macOS / Linux)

### 4.1 セットアップと運用

```bash
# ─── インストール ───
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# ─── 基本操作 ───
brew install ripgrep fd bat           # CLI ツール
brew install --cask firefox           # GUI アプリ
brew update                           # Homebrew 自体を更新
brew upgrade                          # 全パッケージ更新
brew cleanup                          # 古いバージョン削除
brew list                             # インストール済み一覧
brew doctor                           # 環境診断

# ─── Brewfile でチーム統一 ───
brew bundle dump                      # 現在の環境を Brewfile に出力
brew bundle install                   # Brewfile からインストール
```

### 4.2 Brewfile

```ruby
# Brewfile
# brew bundle install で一括インストール

# ─── タップ ───
tap "homebrew/bundle"

# ─── CLI ツール ───
brew "git"
brew "gh"                             # GitHub CLI
brew "fnm"                            # Node.js バージョン管理
brew "mise"                           # 多言語バージョン管理
brew "ripgrep"                        # 高速 grep
brew "fd"                             # 高速 find
brew "bat"                            # cat 代替
brew "eza"                            # ls 代替
brew "fzf"                            # ファジーファインダー
brew "jq"                             # JSON パーサー
brew "delta"                          # git diff 表示
brew "starship"                       # プロンプト
brew "tmux"                           # ターミナルマルチプレクサ

# ─── GUI アプリ ───
cask "visual-studio-code"
cask "iterm2"
cask "docker"
cask "firefox"
cask "raycast"
cask "1password"

# ─── フォント ───
cask "font-jetbrains-mono-nerd-font"
```

---

## 5. パッケージマネージャー選定フロー

```
Node.js プロジェクトのパッケージマネージャー選定:

                    START
                      │
                      ▼
              モノレポ？ ─── Yes ──→ pnpm
                   │                  (ワークスペース最強)
                  No
                   │
                   ▼
          ディスク容量が
          気になる？ ─── Yes ──→ pnpm
                   │              (コンテンツアドレス)
                  No
                   │
                   ▼
          チームに npm 以外の
          経験がない？ ─── Yes ──→ npm
                   │               (追加学習不要)
                  No
                   │
                   ▼
              pnpm (総合的に推奨)
```

---

## 6. Corepack によるバージョン統一

```bash
# Corepack は Node.js 16.13+ に同梱
corepack enable

# package.json で指定
{
  "packageManager": "pnpm@9.1.0"
}

# チームメンバーが npm install を実行すると:
# → "This project is configured to use pnpm" とエラーで教えてくれる
# → 正しいバージョンの pnpm が自動で使われる
```

```
Corepack の動作フロー:

  package.json
  "packageManager": "pnpm@9.1.0"
         │
         ▼
  ┌──────────────────────────────┐
  │  corepack                     │
  │  (Node.js 同梱のプロキシ)      │
  │                                │
  │  pnpm コマンド実行時:          │
  │  1. package.json を確認        │
  │  2. 指定バージョンを検証       │
  │  3. 未インストールなら自動DL   │
  │  4. 正しいバージョンで実行     │
  └──────────────────────────────┘
```

---

## 7. アンチパターン

### 7.1 ロックファイルをコミットしない

```
❌ アンチパターン: .gitignore にロックファイルを追加

  .gitignore:
    package-lock.json    # ← NG
    pnpm-lock.yaml       # ← NG

問題:
  - チーム内でインストールされるバージョンがバラバラ
  - CI と開発環境で異なる依存バージョン
  - 再現不能なバグの原因

✅ 正しいアプローチ:
  - ロックファイルは必ずコミット
  - CI では npm ci / pnpm install --frozen-lockfile を使用
  - ロックファイルの差分レビューでセキュリティチェック
```

### 7.2 グローバルインストールの乱用

```
❌ アンチパターン: npm install -g でプロジェクトツールを入れる

  npm install -g eslint typescript ts-node

問題:
  - プロジェクト間でバージョン競合
  - チームメンバーと異なるバージョン
  - CI で再現できない

✅ 正しいアプローチ:
  - devDependencies に入れて npx で実行
  - npm install -D eslint typescript
  - npx eslint .  /  pnpm exec eslint .
  - package.json の scripts に定義
```

---

## 8. FAQ

### Q1: npm と pnpm、チーム導入するならどちら？

**A:** 新規プロジェクトなら pnpm を推奨する。理由は以下の通り。
- ディスク使用量 50-70% 削減
- インストール速度 2-3倍
- 厳密な依存解決（phantom dependencies 防止）
- Corepack で npm との共存も容易

既存プロジェクトで npm を使っている場合、`pnpm import` で `package-lock.json` から `pnpm-lock.yaml` に変換可能。

### Q2: uv は Poetry を置き換えられる？

**A:** 多くのケースでは置き換え可能。uv は Poetry と比較して 10-100倍高速で、`pyproject.toml` を共通フォーマットとして使える。ただし Poetry のプラグインエコシステムに依存している場合は段階的な移行を推奨。2025年時点で uv は急速に成熟しており、新規プロジェクトでは uv がファーストチョイス。

### Q3: Brewfile はどこに置くべき？

**A:** 2つのパターンがある。
1. **dotfiles リポジトリ** — 個人の開発環境再構築用。全マシンで共通のツールセット。
2. **プロジェクトリポジトリ** — チーム全員が必要なツールのみ記述。`scripts/setup.sh` から `brew bundle install` を呼ぶ。

---

## 9. まとめ

| エコシステム | 推奨ツール | ロックファイル | 備考 |
|------------|-----------|---------------|------|
| Node.js | pnpm | pnpm-lock.yaml | ディスク効率最良 |
| Node.js (シンプル) | npm | package-lock.json | 追加インストール不要 |
| Python | uv | uv.lock | 超高速・次世代 |
| Python (既存) | Poetry | poetry.lock | エコシステム成熟 |
| Rust | Cargo | Cargo.lock | 公式唯一 |
| macOS ツール | Homebrew | Brewfile.lock.json | Brewfile でチーム統一 |

---

## 次に読むべきガイド

- [02-monorepo-setup.md](./02-monorepo-setup.md) — モノレポでのワークスペース活用
- [03-linter-formatter.md](./03-linter-formatter.md) — Linter/Formatter の設定
- [00-version-managers.md](./00-version-managers.md) — ランタイムのバージョン管理

---

## 参考文献

1. **pnpm Documentation** — https://pnpm.io/ja/ — pnpm 公式ドキュメント（日本語）。
2. **uv Documentation** — https://docs.astral.sh/uv/ — uv 公式ドキュメント。pip 比較ベンチマークあり。
3. **Corepack Documentation** — https://nodejs.org/api/corepack.html — Node.js 公式の Corepack 解説。
4. **Homebrew Bundle** — https://github.com/Homebrew/homebrew-bundle — Brewfile の仕様と使い方。
