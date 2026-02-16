# パッケージマネージャー

> npm / pnpm / yarn、pip / poetry / uv、cargo、Homebrew など主要パッケージマネージャーの設定・運用・使い分けを解説する実践ガイド。

## この章で学ぶこと

1. Node.js エコシステムのパッケージマネージャー（npm / pnpm / yarn）の特性と選定基準
2. Python（pip / poetry / uv）と Rust（cargo）のパッケージ管理手法
3. Homebrew によるシステムツール管理とチーム統一の方法
4. Corepack によるパッケージマネージャーのバージョン統一
5. セキュリティ対策とサプライチェーン攻撃への防御
6. モノレポ環境でのパッケージ管理

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
| node_modules 構造 | フラット | シンボリックリンク | PnP / node_modules |
| オフラインインストール | 部分的 | あり | あり (Zero-Installs) |
| パッチ機能 | なし | あり (pnpm patch) | あり (yarn patch) |

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

### 1.2.1 npm の高度な設定

```bash
# ─── .npmrc の全オプション ───
cat << 'EOF' > .npmrc
# バージョン管理
save-exact=true                       # ^ や ~ なしの正確なバージョン
save-prefix=""                        # バージョンプレフィックスを空に

# セキュリティ
engine-strict=true                    # engines フィールドを厳格に検証
audit-level=moderate                  # npm audit の最小レベル
ignore-scripts=false                  # postinstall スクリプトの制御

# パフォーマンス
package-lock=true                     # ロックファイルを必ず生成
prefer-offline=true                   # オフラインキャッシュを優先
fund=false                            # funding メッセージを非表示
update-notifier=false                 # npm 更新通知を無効化

# レジストリ設定
registry=https://registry.npmjs.org/
# プライベートレジストリ（スコープ付き）
@mycompany:registry=https://npm.pkg.github.com/
//npm.pkg.github.com/:_authToken=${NPM_TOKEN}

# プロキシ設定（企業環境）
# proxy=http://proxy.example.com:8080
# https-proxy=http://proxy.example.com:8080
# no-proxy=localhost,127.0.0.1

# ログ設定
loglevel=warn
EOF

# ─── package.json の engines フィールド ───
# バージョンマネージャーと合わせて設定
{
  "engines": {
    "node": ">=20.0.0 <21",
    "npm": ">=10.0.0"
  },
  "engineStrict": true
}
```

### 1.2.2 npm のスクリプト活用

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint . --max-warnings 0",
    "lint:fix": "eslint . --fix",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "validate": "npm run lint && npm run typecheck && npm run test",
    "prepare": "husky",
    "precommit": "lint-staged",
    "clean": "rm -rf node_modules .next out",
    "clean:install": "npm run clean && npm ci"
  }
}
```

```bash
# ─── npm スクリプトの実行 ───
npm run dev                           # scripts のコマンド実行
npm run build -- --debug              # 追加引数を渡す (-- の後)
npm run validate                      # 複合スクリプト実行

# ─── npm ライフサイクルスクリプト ───
# preinstall  → install → postinstall
# prepublish  → prepare → postpublish
# preversion  → version → postversion

# ─── npx でローカルパッケージを実行 ───
npx eslint .                          # devDependencies の eslint を実行
npx -y create-next-app@latest         # 未インストールパッケージの一時実行
npx --package=typescript tsc --init   # 特定パッケージのコマンドを実行
```

### 1.2.3 npm のセキュリティ機能

```bash
# ─── 脆弱性監査 ───
npm audit                             # 脆弱性レポート表示
npm audit --json                      # JSON 形式で出力
npm audit fix                         # 自動修正（セーフなアップデート）
npm audit fix --force                 # 破壊的変更を含む修正（注意）
npm audit signatures                  # パッケージ署名の検証

# ─── provenance（出所証明） ───
# npm v9.5+ で利用可能
# CI/CD でパッケージ公開時に SLSA provenance を付与
npm publish --provenance

# ─── overrides で脆弱なバージョンを強制更新 ───
# package.json に追加
{
  "overrides": {
    "lodash": "4.17.21",
    "minimist": ">=1.2.6",
    "json5": ">=2.2.2"
  }
}

# ─── パッケージロックの差分確認（コードレビュー時） ───
# package-lock.json の変更を確認するポイント:
# 1. 新しい依存の追加（意図したものか）
# 2. integrity ハッシュの変更（改竄検知）
# 3. resolved URL の変更（レジストリ改竄検知）
```

### 1.3 pnpm の設定

```bash
# ─── インストール ───
# corepack (Node.js 16.13+ 同梱)
corepack enable
corepack prepare pnpm@latest --activate

# または直接インストール
npm install -g pnpm

# Homebrew
brew install pnpm

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
pnpm why lodash                       # なぜこの依存があるか表示
pnpm list --depth=0                   # 直接依存の一覧
```

### 1.3.1 pnpm の詳細設定

```yaml
# .pnpmrc または .npmrc で設定可能
# 推奨設定:

# ─── 厳密モード ───
# shamefully-hoist=false               # phantom dependencies を防止（デフォルト）
# strict-peer-dependencies=true        # peer dependency の不整合をエラーに
# auto-install-peers=true              # peer dependency を自動インストール

# ─── パフォーマンス ───
# store-dir=~/.local/share/pnpm/store  # ストアの場所（デフォルト）
# network-concurrency=16               # 同時ダウンロード数
# prefer-offline=true                  # オフラインキャッシュ優先

# ─── セキュリティ ───
# ignore-scripts=true                  # postinstall スクリプトを無効化
# side-effects-cache=true              # ビルドキャッシュを有効化
```

```bash
# ─── pnpm の便利機能 ───

# パッチ機能（依存パッケージにパッチを当てる）
pnpm patch express@4.18.2             # パッチ用の一時ディレクトリを作成
# → 表示されたディレクトリでファイルを編集
pnpm patch-commit <patch-dir>         # パッチを確定

# パッチの結果:
# patches/express@4.18.2.patch が生成される
# package.json に以下が追加:
# {
#   "pnpm": {
#     "patchedDependencies": {
#       "express@4.18.2": "patches/express@4.18.2.patch"
#     }
#   }
# }

# ─── overrides（依存バージョンの上書き） ───
# package.json に追加
{
  "pnpm": {
    "overrides": {
      "lodash": "4.17.21",
      "glob": ">=10.0.0"
    }
  }
}

# ─── カタログ機能（モノレポでのバージョン統一） ───
# pnpm-workspace.yaml に定義
# catalog:
#   react: "^18.2.0"
#   typescript: "^5.4.0"
# → ワークスペース内で catalog: プレフィックスで参照可能

# ─── pnpm deploy（依存のみのデプロイ） ───
pnpm deploy --filter=my-app /deploy/my-app
# → 本番に必要な依存のみを別ディレクトリに展開
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

  node_modules の内部構造:
  ┌──────────────────────────────────────────────┐
  │ Project/node_modules/                         │
  │ ├── .pnpm/                                   │
  │ │   ├── express@4.18.2/                      │
  │ │   │   └── node_modules/                    │
  │ │   │       ├── express/ → ハードリンク      │
  │ │   │       ├── accepts/ → ハードリンク      │
  │ │   │       └── body-parser/ → ハードリンク  │
  │ │   └── lodash@4.17.21/                      │
  │ │       └── node_modules/                    │
  │ │           └── lodash/ → ハードリンク       │
  │ ├── express → .pnpm/express@4.18.2/...       │
  │ └── lodash → .pnpm/lodash@4.17.21/...       │
  └──────────────────────────────────────────────┘

  これにより:
  - express の内部依存に直接アクセスできない（phantom dependencies 防止）
  - 各パッケージは自分の依存のみ参照可能（厳密な依存解決）
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
yarn remove lodash                    # 依存削除
yarn up --interactive                 # 対話的アップデート
yarn why lodash                       # 依存関係の確認
yarn info express                     # パッケージ情報表示
yarn dlx create-next-app              # npx 相当（一時実行）
```

### 1.5.1 yarn の Plug'n'Play (PnP)

```yaml
# .yarnrc.yml - PnP モードの設定
nodeLinker: pnp                        # PnP を有効化
pnpMode: loose                         # loose モード（互換性重視）
# pnpMode: strict                      # strict モード（厳密）

# PnP の動作原理:
# node_modules ディレクトリを作成しない
# 代わりに .pnp.cjs (または .pnp.loader.mjs) を生成
# Node.js のモジュール解決をオーバーライドして直接パッケージを参照

# PnP のメリット:
# - インストール時間の大幅短縮（ファイルコピー不要）
# - ディスク使用量の削減
# - phantom dependencies の完全防止
# - Zero-Installs（.yarn/cache をコミットして CI 高速化）

# PnP のデメリット:
# - 一部ツールとの互換性問題（対応が必要）
# - エディタ統合に追加設定が必要
```

```bash
# ─── PnP で Zero-Installs を実現 ───
# .gitattributes に追加（バイナリとして扱う）
cat << 'EOF' >> .gitattributes
.yarn/cache/** binary
.yarn/releases/** binary
.yarn/plugins/** binary
.pnp.* binary
EOF

# .gitignore
cat << 'EOF' >> .gitignore
.yarn/*
!.yarn/cache
!.yarn/patches
!.yarn/plugins
!.yarn/releases
!.yarn/sdks
!.yarn/versions
EOF

# Zero-Installs の効果:
# clone 後に yarn install 不要
# CI のインストールステップが不要（数分の短縮）

# ─── PnP + VSCode の連携 ───
yarn dlx @yarnpkg/sdks vscode         # VSCode SDK をインストール
# → .vscode/settings.json に TypeScript SDK パスが設定される
```

### 1.5.2 yarn のワークスペース機能

```json
// package.json (ルート)
{
  "workspaces": [
    "packages/*",
    "apps/*"
  ]
}
```

```bash
# ─── ワークスペース操作 ───
yarn workspaces list                   # ワークスペース一覧
yarn workspace my-app add express      # 特定ワークスペースに依存追加
yarn workspace my-app run build        # 特定ワークスペースでスクリプト実行

# ─── yarn constraints（依存の制約ルール） ───
# yarn.config.cjs で制約を定義
module.exports = {
  async constraints({ Yarn }) {
    // 全ワークスペースで同じ TypeScript バージョンを強制
    for (const dep of Yarn.dependencies({ ident: 'typescript' })) {
      dep.update('^5.4.0');
    }
    // react のバージョンを統一
    for (const dep of Yarn.dependencies({ ident: 'react' })) {
      dep.update('^18.2.0');
    }
  }
};

# yarn constraints            # 制約チェック
# yarn constraints --fix      # 自動修正
```

### 1.6 Node.js パッケージマネージャーのベンチマーク

```
npm vs pnpm vs yarn のベンチマーク比較:

  テスト環境: 中規模プロジェクト (約200依存)

  クリーンインストール (キャッシュあり):
  ┌──────────┬──────────┬──────────┬──────────┐
  │          │   npm    │   pnpm   │   yarn   │
  ├──────────┼──────────┼──────────┼──────────┤
  │ 時間     │  18.5s   │   6.2s   │   7.8s   │
  │ ディスク │  245MB   │   98MB   │  112MB   │
  └──────────┴──────────┴──────────┴──────────┘

  ロックファイルからのインストール (npm ci 相当):
  ┌──────────┬──────────┬──────────┬──────────┐
  │          │  npm ci  │ pnpm i   │ yarn i   │
  │          │          │ --frozen │          │
  ├──────────┼──────────┼──────────┼──────────┤
  │ 時間     │  12.3s   │   4.8s   │   5.1s   │
  └──────────┴──────────┴──────────┴──────────┘

  10プロジェクトでの累計ディスク使用量:
  ┌──────────┬──────────┬──────────┬──────────┐
  │          │   npm    │   pnpm   │   yarn   │
  ├──────────┼──────────┼──────────┼──────────┤
  │ 合計     │  2.45GB  │  0.35GB  │  1.12GB  │
  │ 削減率   │   -      │  -86%    │  -54%    │
  └──────────┴──────────┴──────────┴──────────┘

  pnpm のディスク効率はコンテンツアドレスストレージによるもの
  同じパッケージバージョンは1つだけ保存され、ハードリンクで共有
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
| Python バージョン管理 | なし | なし | あり |
| スクリプト実行 | なし | poetry run | uv run |

### 2.2 uv (推奨 -- 次世代パッケージマネージャー)

```bash
# ─── インストール ───
curl -LsSf https://astral.sh/uv/install.sh | sh
# または
brew install uv
# pip 経由
pip install uv

# ─── プロジェクト初期化 ───
uv init my-project
cd my-project

# ─── 依存管理 ───
uv add requests                       # 依存追加
uv add --dev pytest ruff              # 開発依存追加
uv add --optional docs sphinx         # オプション依存追加
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

### 2.2.1 uv の高度な使い方

```bash
# ─── pip 互換モード ───
# 既存プロジェクトで uv を pip の高速代替として使う
uv pip install -r requirements.txt    # pip の 10-100倍高速
uv pip install flask                  # 個別パッケージ
uv pip compile requirements.in -o requirements.txt  # ロックファイル生成
uv pip sync requirements.txt          # ロックファイルから同期

# ─── uv の仮想環境管理 ───
uv venv                               # .venv を作成
uv venv --python 3.12                  # 特定バージョンで作成
uv venv my-env                         # 名前付き仮想環境
source .venv/bin/activate              # 有効化（uv run を使えば不要）

# ─── ツール実行（pipx 代替） ───
uv tool run ruff check .              # ツールを一時的にインストールして実行
uv tool install ruff                   # ツールを永続的にインストール
uv tool list                           # インストール済みツール一覧
uvx ruff check .                       # uv tool run のショートカット
```

```toml
# pyproject.toml - uv プロジェクト設定

[project]
name = "my-project"
version = "0.1.0"
description = "My project description"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "sqlalchemy>=2.0",
    "pydantic>=2.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.3.0",
    "mypy>=1.8",
    "coverage>=7.4",
]
docs = [
    "sphinx>=7.2",
    "sphinx-rtd-theme>=2.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "ruff>=0.3.0",
    "mypy>=1.8",
]

[tool.uv.sources]
# プライベートレジストリ
# my-internal-package = { index = "internal" }

# Git リポジトリから直接
# my-lib = { git = "https://github.com/org/my-lib.git", tag = "v1.0.0" }

# ローカルパス（モノレポ内の別パッケージ）
# shared-utils = { path = "../shared-utils" }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 2.2.2 uv のベンチマーク

```
uv vs pip vs poetry のインストール速度比較:

  テスト: Django プロジェクト（約80依存）

  コールドインストール:
  ┌──────────────┬──────────┬──────────┬──────────┐
  │              │   pip    │  poetry  │    uv    │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ 時間         │  32.5s   │  28.1s   │   1.2s   │
  │ 速度比       │   1x     │   1.2x   │   27x    │
  └──────────────┴──────────┴──────────┴──────────┘

  ウォームインストール（キャッシュあり）:
  ┌──────────────┬──────────┬──────────┬──────────┐
  │              │   pip    │  poetry  │    uv    │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ 時間         │  15.8s   │  12.3s   │   0.3s   │
  │ 速度比       │   1x     │   1.3x   │   53x    │
  └──────────────┴──────────┴──────────┴──────────┘

  依存解決:
  ┌──────────────┬──────────┬──────────┬──────────┐
  │              │   pip    │  poetry  │    uv    │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ 時間         │  8.2s    │  5.1s    │   0.4s   │
  └──────────────┴──────────┴──────────┴──────────┘

  uv が高速な理由:
  - Rust で実装（pip は Python、Poetry も Python）
  - 並列ダウンロード・解凍
  - 効率的な依存解決アルゴリズム（PubGrub）
  - インテリジェントなキャッシュ
```

### 2.3 Poetry

```bash
# ─── インストール ───
curl -sSL https://install.python-poetry.org | python3 -
# または
pipx install poetry

# ─── 設定 ───
poetry config virtualenvs.in-project true   # .venv をプロジェクト内に作成
poetry config virtualenvs.prefer-active-python true  # アクティブな Python を優先

# ─── プロジェクト初期化 ───
poetry init                           # 対話的初期化
poetry new my-project                 # プロジェクトテンプレート作成
poetry install                        # 依存インストール

# ─── 依存管理 ───
poetry add requests                   # 依存追加
poetry add --group dev pytest         # 開発依存追加
poetry add --group docs sphinx        # ドキュメント依存追加
poetry remove requests                # 依存削除
poetry lock                           # ロックファイル更新
poetry show --outdated                # 更新可能パッケージ
poetry show --tree                    # 依存ツリー表示

# ─── 実行 ───
poetry run python main.py
poetry shell                          # 仮想環境を有効化

# ─── ビルド & 公開 ───
poetry build                          # sdist + wheel ビルド
poetry publish                        # PyPI に公開
poetry publish --build                # ビルド + 公開を同時に
```

### 2.3.1 Poetry の pyproject.toml

```toml
# pyproject.toml - Poetry 形式

[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "My project description"
authors = ["Your Name <your@email.com>"]
readme = "README.md"
packages = [{include = "my_project"}]

[tool.poetry.dependencies]
python = "^3.12"
fastapi = "^0.110.0"
uvicorn = {version = "^0.27.0", extras = ["standard"]}
sqlalchemy = "^2.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
ruff = "^0.3.0"
mypy = "^1.8"

[tool.poetry.group.docs.dependencies]
sphinx = "^7.2"

[tool.poetry.scripts]
my-cli = "my_project.cli:main"

[tool.poetry.plugins."my_framework.plugins"]
my-plugin = "my_project.plugins:MyPlugin"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 2.4 pip + venv（標準ライブラリのみ）

```bash
# ─── 仮想環境の作成と有効化 ───
python -m venv .venv
source .venv/bin/activate             # macOS/Linux
# .venv\Scripts\activate              # Windows

# ─── 依存管理 ───
pip install flask                     # パッケージインストール
pip install -r requirements.txt       # 一括インストール
pip freeze > requirements.txt         # 現在の依存を出力

# ─── pip-compile で再現性を確保 ───
pip install pip-tools
# requirements.in に直接依存を記述:
# flask
# sqlalchemy>=2.0
pip-compile requirements.in           # ロックファイル生成
pip-sync requirements.txt             # ロックファイルから同期

# ─── 開発依存と本番依存の分離 ───
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
```

### 2.5 Python パッケージマネージャー選定フロー

```
Python プロジェクトのパッケージマネージャー選定:

                    START
                      │
                      ▼
            新規プロジェクト？ ─── No ──→ 既存ツールを継続
                   │                     (移行コスト考慮)
                  Yes
                   │
                   ▼
            速度が最優先？ ─── Yes ──→ uv
                   │                  (Rust 製・超高速)
                  No
                   │
                   ▼
            プラグインエコシステム
            が必要？ ──── Yes ──→ Poetry
                   │               (成熟したエコシステム)
                  No
                   │
                   ▼
            Python バージョンも
            管理したい？ ──── Yes ──→ uv
                   │                  (pyenv 不要)
                  No
                   │
                   ▼
            最小限の依存で
            済ませたい？ ──── Yes ──→ pip + venv + pip-tools
                   │                  (標準ライブラリのみ)
                  No
                   │
                   ▼
              uv (総合的に推奨)
```

---

## 3. Rust パッケージマネージャー (Cargo)

### 3.1 基本操作

```bash
# ─── プロジェクト作成 ───
cargo new my-project                  # バイナリプロジェクト
cargo new --lib my-lib                # ライブラリプロジェクト
cargo init                            # 既存ディレクトリで初期化

# ─── 依存管理 (Cargo.toml) ───
cargo add serde --features derive     # 依存追加
cargo add tokio -F full               # feature flag 付き
cargo add --dev mockall               # 開発依存
cargo add --build bindgen             # ビルド依存
cargo remove serde                    # 依存削除

# ─── ビルド & 実行 ───
cargo build                           # デバッグビルド
cargo build --release                 # リリースビルド
cargo run                             # ビルド + 実行
cargo run --release                   # リリースモードで実行
cargo test                            # テスト実行
cargo test -- --nocapture             # テスト出力を表示
cargo clippy                          # リント
cargo clippy -- -D warnings           # 警告をエラーとして扱う
cargo fmt                             # フォーマット
cargo fmt -- --check                  # フォーマットチェック（CI用）
cargo doc --open                      # ドキュメント生成 + ブラウザで開く
cargo bench                           # ベンチマーク実行
```

### 3.2 Cargo.toml の設定

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <your@email.com>"]
description = "My project description"
license = "MIT"
repository = "https://github.com/user/my-project"

[dependencies]
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
axum = "0.7"

[dev-dependencies]
mockall = "0.12"
tokio-test = "0.4"
criterion = { version = "0.5", features = ["html_reports"] }
insta = "1"                           # スナップショットテスト

[build-dependencies]
# ビルドスクリプトで使う依存

[profile.release]
lto = true                            # Link-Time Optimization
codegen-units = 1                     # 1つのコード生成ユニット
strip = true                          # デバッグ情報を除去
panic = "abort"                       # パニック時にアボート
opt-level = 3                         # 最大最適化

[profile.dev]
opt-level = 1                         # デバッグビルドでも若干の最適化
# debug = true                        # デバッグ情報（デフォルトで有効）

[profile.dev.package."*"]
opt-level = 2                         # 依存のデバッグビルドを最適化

# ─── Feature フラグ ───
[features]
default = ["json"]
json = ["serde_json"]
full = ["json", "yaml", "toml-support"]
yaml = ["serde_yaml"]
toml-support = ["toml"]

# ─── ワークスペース ───
[workspace]
members = [
    "crates/core",
    "crates/cli",
    "crates/server",
]
resolver = "2"

[workspace.dependencies]
# ワークスペース全体で共有する依存バージョン
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
```

### 3.3 Cargo の便利なサブコマンド

```bash
# ─── cargo-install で追加ツールをインストール ───
cargo install cargo-watch              # ファイル変更を監視して自動ビルド
cargo install cargo-expand             # マクロ展開を表示
cargo install cargo-audit              # 脆弱性チェック
cargo install cargo-tarpaulin          # コードカバレッジ
cargo install cargo-nextest            # 高速テストランナー
cargo install cargo-deny               # 依存ポリシーチェック
cargo install cargo-udeps              # 未使用依存の検出
cargo install cargo-bloat              # バイナリサイズ分析

# ─── 使い方 ───
cargo watch -x test                    # テストを自動実行
cargo watch -x "run -- --port 8080"    # サーバーを自動再起動
cargo expand                           # マクロ展開結果を表示
cargo audit                            # 脆弱性レポート
cargo nextest run                      # 並列テスト実行
cargo tarpaulin --out html             # カバレッジレポート生成
cargo udeps                            # 未使用依存の検出
cargo bloat --release                  # バイナリサイズの分析

# ─── cargo-deny で依存ポリシーを強制 ───
cargo deny init                        # deny.toml を生成
cargo deny check                       # ポリシーチェック
```

```toml
# deny.toml - 依存ポリシー設定
[licenses]
allow = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"]
deny = ["GPL-3.0"]

[bans]
multiple-versions = "warn"
deny = [
    { name = "openssl" },              # 代わりに rustls を使用
]

[advisories]
vulnerability = "deny"
unmaintained = "warn"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
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
brew upgrade ripgrep                  # 特定パッケージを更新
brew cleanup                          # 古いバージョン削除
brew cleanup --prune=7                # 7日以上前のキャッシュを削除
brew list                             # インストール済み一覧
brew list --cask                      # cask 一覧
brew doctor                           # 環境診断
brew info ripgrep                     # パッケージ情報
brew deps --tree ripgrep              # 依存ツリー
brew leaves                           # 他の依存にならないパッケージ一覧
brew autoremove                       # 不要な依存を自動削除

# ─── Brewfile でチーム統一 ───
brew bundle dump                      # 現在の環境を Brewfile に出力
brew bundle dump --force              # 既存の Brewfile を上書き
brew bundle install                   # Brewfile からインストール
brew bundle check                     # Brewfile との差分確認
brew bundle cleanup                   # Brewfile にないパッケージを削除
```

### 4.2 Brewfile

```ruby
# Brewfile
# brew bundle install で一括インストール

# ─── タップ ───
tap "homebrew/bundle"
tap "homebrew/services"               # サービス管理

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
brew "zoxide"                         # cd 代替（学習型）
brew "jq"                             # JSON パーサー
brew "yq"                             # YAML パーサー
brew "delta"                          # git diff 表示
brew "starship"                       # プロンプト
brew "tmux"                           # ターミナルマルチプレクサ
brew "direnv"                         # ディレクトリごとの環境変数
brew "hyperfine"                      # コマンドベンチマーク
brew "tokei"                          # コード行数カウント
brew "dust"                           # du 代替
brew "bottom"                         # top 代替
brew "procs"                          # ps 代替
brew "httpie"                         # HTTP クライアント
brew "wget"                           # ダウンロード
brew "tree"                           # ディレクトリツリー
brew "watch"                          # コマンド定期実行

# ─── 開発ツール ───
brew "docker"                         # コンテナ
brew "docker-compose"                 # コンテナオーケストレーション
brew "kubectl"                        # Kubernetes CLI
brew "helm"                           # Kubernetes パッケージマネージャー
brew "terraform"                      # IaC
brew "awscli"                         # AWS CLI
brew "uv"                             # Python パッケージマネージャー

# ─── データベース ───
brew "postgresql@16"                  # PostgreSQL
brew "redis"                          # Redis
brew "sqlite"                         # SQLite

# ─── GUI アプリ ───
cask "visual-studio-code"
cask "cursor"                         # AI エディタ
cask "iterm2"
cask "warp"                           # AI ターミナル
cask "docker"                         # Docker Desktop
cask "firefox"
cask "google-chrome"
cask "raycast"                        # Spotlight 代替
cask "1password"                      # パスワード管理
cask "figma"                          # デザインツール
cask "notion"                         # ドキュメント
cask "slack"                          # チャット
cask "zoom"                           # ビデオ会議
cask "obsidian"                       # ノートアプリ

# ─── フォント ───
cask "font-jetbrains-mono-nerd-font"
cask "font-fira-code-nerd-font"
cask "font-hack-nerd-font"

# ─── Mac App Store (mas) ───
# mas "Xcode", id: 497799835
# mas "Keynote", id: 409183694
```

### 4.3 Homebrew のサービス管理

```bash
# ─── サービスの起動・停止 ───
brew services start postgresql@16     # PostgreSQL を起動
brew services stop postgresql@16      # 停止
brew services restart postgresql@16   # 再起動
brew services list                    # サービス一覧
brew services info postgresql@16      # サービス情報

# ─── ログイン時自動起動 ───
# brew services start で自動的に LaunchAgent が設定される
# 手動で無効化する場合:
brew services stop postgresql@16
```

### 4.4 Homebrew のトラブルシューティング

```bash
# ─── よくある問題と解決策 ───

# 問題: brew update が失敗する
brew update-reset                     # Homebrew リポジトリをリセット

# 問題: Permission denied
sudo chown -R $(whoami) /opt/homebrew  # Apple Silicon Mac
sudo chown -R $(whoami) /usr/local     # Intel Mac

# 問題: cask のインストールがブロックされる（Gatekeeper）
# システム環境設定 > プライバシーとセキュリティ で許可
# または:
xattr -cr /Applications/SomeApp.app

# 問題: 古いバージョンを使いたい
brew tap homebrew/cask-versions
brew install --cask firefox@esr       # ESR 版

# 問題: ディスク使用量が多い
brew cleanup --prune=0                # 全てのキャッシュを削除
du -sh $(brew --cache)                # キャッシュサイズ確認
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
          Zero-Installs が
          必要？ ─── Yes ──→ yarn (PnP)
                   │            (CI 高速化)
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

### 6.1 Corepack の詳細設定

```bash
# ─── Corepack の有効化と設定 ───
corepack enable                        # Corepack を有効化
corepack enable pnpm                   # pnpm のみ有効化
corepack enable yarn                   # yarn のみ有効化

# ─── パッケージマネージャーの準備 ───
corepack prepare pnpm@9.1.0 --activate # 特定バージョンを準備
corepack prepare yarn@4.1.0 --activate

# ─── package.json への記述 ───
# npm init で自動設定されないため手動で追加
{
  "packageManager": "pnpm@9.1.0+sha512.xxxxx"
}

# ハッシュ付きで指定すると整合性チェックが行われる
# corepack use pnpm@9.1.0 で自動生成される

# ─── CI での Corepack ───
# GitHub Actions
- uses: actions/setup-node@v4
  with:
    node-version-file: '.node-version'
- run: corepack enable
- run: pnpm install --frozen-lockfile

# ─── Corepack のオフラインモード ───
# CI でネットワークアクセスを制限する場合
corepack prepare pnpm@9.1.0 --activate
corepack pack                          # バンドルを作成
# → corepack.tgz を CI にキャッシュ

# ─── 間違ったパッケージマネージャーの使用を防ぐ ───
# package.json に追加:
{
  "scripts": {
    "preinstall": "npx only-allow pnpm"
  }
}
# → npm install や yarn install を実行するとエラーになる
```

---

## 7. セキュリティとサプライチェーン対策

### 7.1 npm / pnpm のセキュリティ設定

```bash
# ─── 脆弱性の定期チェック ───
npm audit                              # 脆弱性レポート
pnpm audit                             # pnpm 版
yarn npm audit                         # yarn 版

# ─── CI でのセキュリティチェック ───
npm audit --audit-level=high           # high 以上で失敗
npm audit --omit=dev                   # 本番依存のみチェック

# ─── Socket.dev との連携 ───
# サプライチェーン攻撃の検知に特化したツール
npx socket-security audit              # Socket.dev による分析

# ─── npm provenance（出所証明） ───
# GitHub Actions からの公開時に SLSA provenance を付与
# npm publish --provenance

# ─── lockfile-lint でロックファイルを検証 ───
npx lockfile-lint \
  --path pnpm-lock.yaml \
  --type pnpm \
  --allowed-hosts npm \
  --allowed-schemes "https:"
```

### 7.2 サプライチェーン攻撃の防御

```
サプライチェーン攻撃のベクトルと対策:

  1. Typosquatting（タイポスクワッティング）
  ┌──────────────────────────────────────────────┐
  │ 攻撃: 類似名パッケージの公開                    │
  │   例: lodash → lodahs, lod-ash                │
  │                                                │
  │ 対策:                                          │
  │   - パッケージ名をコピー&ペースト               │
  │   - npm info <package> で確認してからインストール│
  │   - Socket.dev などの検知ツール導入              │
  └──────────────────────────────────────────────┘

  2. Dependency Confusion（依存関係混乱攻撃）
  ┌──────────────────────────────────────────────┐
  │ 攻撃: 社内パッケージ名で公開レジストリに同名公開│
  │   → npm が公開レジストリのバージョンを優先        │
  │                                                │
  │ 対策:                                          │
  │   - .npmrc でスコープとレジストリを明示設定     │
  │   @mycompany:registry=https://internal-npm/    │
  │   - パッケージ名にスコープを必ず付ける          │
  │   @mycompany/my-package                        │
  └──────────────────────────────────────────────┘

  3. Malicious postinstall scripts
  ┌──────────────────────────────────────────────┐
  │ 攻撃: postinstall スクリプトで悪意のあるコード │
  │   → npm install 時に自動実行                   │
  │                                                │
  │ 対策:                                          │
  │   - .npmrc: ignore-scripts=true                │
  │   - 信頼できるパッケージのスクリプトのみ許可    │
  │   - pnpm の onlyBuiltDependencies で制御       │
  └──────────────────────────────────────────────┘

  4. Compromised Maintainer（メンテナアカウント侵害）
  ┌──────────────────────────────────────────────┐
  │ 攻撃: メンテナのアカウントが乗っ取られ         │
  │   → 正規パッケージに悪意のあるバージョン公開    │
  │                                                │
  │ 対策:                                          │
  │   - ロックファイルの差分レビュー               │
  │   - npm audit signatures で署名検証            │
  │   - 依存の自動更新に注意（Dependabot PR は要確認）│
  │   - save-exact=true でバージョンを固定          │
  └──────────────────────────────────────────────┘
```

### 7.3 Python のセキュリティ対策

```bash
# ─── pip-audit（脆弱性チェック） ───
pip install pip-audit
pip-audit                              # 現在の環境の脆弱性チェック
pip-audit -r requirements.txt          # requirements.txt のチェック

# ─── uv での脆弱性チェック ───
# uv は依存解決時に脆弱性を自動チェックする設定が可能

# ─── safety（Python 脆弱性データベース） ───
pip install safety
safety check                           # 脆弱性チェック
safety check -r requirements.txt       # ファイル指定

# ─── ハッシュ検証 ───
# requirements.txt にハッシュを含めることで改竄を検知
pip install --require-hashes -r requirements.txt

# uv pip compile で自動ハッシュ付与
uv pip compile requirements.in --generate-hashes -o requirements.txt
```

---

## 8. モノレポでのパッケージ管理

### 8.1 pnpm ワークスペース

```yaml
# pnpm-workspace.yaml
packages:
  - "packages/*"
  - "apps/*"
  - "tools/*"
```

```bash
# ─── ワークスペース操作 ───
pnpm install                           # 全ワークスペースの依存をインストール
pnpm add -D typescript -w              # ルートに依存追加
pnpm add express --filter my-app       # 特定ワークスペースに依存追加
pnpm run build --filter my-app         # 特定ワークスペースでビルド
pnpm run build --filter "./packages/*" # パターンマッチでビルド
pnpm run test -r                       # 全ワークスペースでテスト（再帰）
pnpm run build --filter my-app...      # my-app とその依存を全てビルド
pnpm run build --filter ...my-app      # my-app に依存するものを全てビルド

# ─── ワークスペース間の依存 ───
# packages/my-lib/package.json
{
  "name": "@myproject/my-lib",
  "version": "1.0.0"
}

# apps/my-app/package.json
{
  "dependencies": {
    "@myproject/my-lib": "workspace:*"  # ワークスペース内の最新版を参照
  }
}

# ─── カタログ機能（バージョン統一） ───
# pnpm-workspace.yaml
packages:
  - "packages/*"

catalog:
  react: "^18.2.0"
  react-dom: "^18.2.0"
  typescript: "^5.4.0"
  vitest: "^1.3.0"

# 各パッケージで catalog: プレフィックスを使用
# packages/my-app/package.json
{
  "dependencies": {
    "react": "catalog:",
    "react-dom": "catalog:"
  }
}
```

### 8.2 Turborepo との組み合わせ

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": [".env"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"]
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": []
    },
    "lint": {
      "outputs": []
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

```bash
# ─── Turborepo の基本操作 ───
npx turbo run build                    # 全パッケージをビルド（依存順・キャッシュ付き）
npx turbo run test --filter=my-app     # 特定パッケージのテスト
npx turbo run lint test build          # 複数タスクを依存順に実行
npx turbo run build --dry-run          # 実行計画を表示（実行はしない）
npx turbo run build --graph            # 依存グラフを表示
```

### 8.3 Nx との組み合わせ

```bash
# ─── Nx の初期化 ───
npx nx init                            # 既存プロジェクトに Nx を追加

# ─── 基本操作 ───
npx nx build my-app                    # 特定プロジェクトのビルド
npx nx run-many -t build               # 全プロジェクトのビルド
npx nx affected -t test                # 変更の影響を受けるプロジェクトのテスト
npx nx graph                           # 依存グラフの可視化

# ─── キャッシュ ───
# Nx はビルド結果をキャッシュし、変更がない場合はキャッシュを返す
# リモートキャッシュ（Nx Cloud）を使うとチーム全体で共有可能
```

---

## 9. プライベートレジストリ

### 9.1 npm プライベートレジストリ

```bash
# ─── GitHub Packages ───
# .npmrc
@mycompany:registry=https://npm.pkg.github.com/
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}

# ─── npm Enterprise / Artifactory ───
# .npmrc
@mycompany:registry=https://npm.mycompany.com/
//npm.mycompany.com/:_authToken=${NPM_TOKEN}
always-auth=true

# ─── Verdaccio（セルフホスト） ───
# Docker で起動
docker run -d --name verdaccio -p 4873:4873 verdaccio/verdaccio

# .npmrc
registry=http://localhost:4873/
# プロキシ設定（不在のパッケージは npmjs.org にフォールバック）
```

### 9.2 Python プライベートレジストリ

```bash
# ─── pip の設定 ───
pip install my-package --index-url https://pypi.mycompany.com/simple/
pip install my-package --extra-index-url https://pypi.mycompany.com/simple/

# ─── uv の設定 ───
# pyproject.toml
[tool.uv]
index-url = "https://pypi.mycompany.com/simple/"
extra-index-url = ["https://pypi.org/simple/"]

# ─── Poetry の設定 ───
poetry config repositories.mycompany https://pypi.mycompany.com/simple/
poetry config http-basic.mycompany username password
```

---

## 10. アンチパターン

### 10.1 ロックファイルをコミットしない

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

### 10.2 グローバルインストールの乱用

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

例外（グローバルインストールが適切なケース）:
  - パッケージマネージャー自体: pnpm, yarn
  - プロジェクト横断ツール: vercel, netlify-cli
  - シェル統合ツール: nvm, fnm
```

### 10.3 node_modules をコミットする

```
❌ アンチパターン: node_modules をリポジトリにコミット

問題:
  - リポジトリサイズが巨大化（数百MB～数GB）
  - clone / pull が極端に遅くなる
  - OS / アーキテクチャ依存のバイナリが含まれる
  - ロックファイルの意味がなくなる

✅ 正しいアプローチ:
  .gitignore に追加:
    node_modules/
    .venv/
    __pycache__/
    target/          # Rust

  CI では毎回クリーンインストール:
    npm ci / pnpm install --frozen-lockfile
```

### 10.4 バージョン範囲を広く取りすぎる

```
❌ アンチパターン: dependencies でワイルドカードや広すぎる範囲を指定

  {
    "dependencies": {
      "express": "*",              # 任意のバージョン
      "lodash": ">=4.0.0",        # 4.x 以上なら何でも
      "react": "^17 || ^18"       # 複数メジャーバージョン
    }
  }

問題:
  - ロックファイルなしだとインストールのたびに異なるバージョン
  - 破壊的変更を含むバージョンがインストールされる可能性
  - セキュリティ脆弱性のあるバージョンが入る可能性

✅ 正しいアプローチ:
  {
    "dependencies": {
      "express": "4.18.2",         # 正確なバージョン (save-exact)
      "lodash": "^4.17.21",       # パッチ/マイナー更新のみ許可
      "react": "^18.2.0"          # 1つのメジャーバージョン
    }
  }
```

### 10.5 複数のロックファイルを混在させる

```
❌ アンチパターン: 同一プロジェクトに複数のロックファイル

  my-project/
  ├── package-lock.json   # npm
  ├── pnpm-lock.yaml      # pnpm
  └── yarn.lock           # yarn

問題:
  - どのパッケージマネージャーを使うべきか不明
  - ロックファイル間で依存バージョンが異なる
  - CI での挙動が予測不能

✅ 正しいアプローチ:
  - 1つのパッケージマネージャーに統一
  - 不要なロックファイルを削除して .gitignore に追加
  - package.json の packageManager フィールドで明示
  - "preinstall": "npx only-allow pnpm" で強制
```

---

## 11. FAQ

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
1. **dotfiles リポジトリ** -- 個人の開発環境再構築用。全マシンで共通のツールセット。
2. **プロジェクトリポジトリ** -- チーム全員が必要なツールのみ記述。`scripts/setup.sh` から `brew bundle install` を呼ぶ。

### Q4: pnpm の shamefully-hoist はいつ使うべき？

**A:** 基本的には使わないべき。`shamefully-hoist=true` は npm と同じフラットな node_modules を作成し、phantom dependencies を許してしまう。ただし、以下のケースではやむを得ず使用する。
- 古いパッケージが正しく依存を宣言していない場合
- React Native など特定のフレームワークが要求する場合
- 段階的な npm → pnpm 移行の初期段階

代替策として `public-hoist-pattern` で特定パッケージのみ巻き上げる方がよい。

```
# .npmrc
public-hoist-pattern[]=*eslint*
public-hoist-pattern[]=*prettier*
```

### Q5: Cargo.lock はライブラリプロジェクトでもコミットすべき？

**A:** はい、Cargo.lock はライブラリプロジェクトでもコミットすることが推奨されている（Rust 公式ガイドラインの変更により）。ただし、ライブラリの利用者は自身の Cargo.lock を使用するため、ライブラリの Cargo.lock は開発時の再現性のためにのみ使われる。

### Q6: npm ci と npm install の違いは？

**A:** 主な違いは以下の通り。

| | npm install | npm ci |
|---|---|---|
| ロックファイル | 更新される場合がある | 厳密に従う（不一致ならエラー） |
| node_modules | 差分更新 | 削除して再作成 |
| 速度 | 速い（差分のみ） | 遅い（毎回クリーン） |
| 用途 | 開発中の追加・更新 | CI / クリーンインストール |

CI では必ず `npm ci`（pnpm なら `pnpm install --frozen-lockfile`）を使うべき。開発中は `npm install` で十分。

### Q7: Python で requirements.txt と pyproject.toml、どちらを使うべき？

**A:** 新規プロジェクトでは pyproject.toml を推奨。PEP 621 で標準化されたフォーマットであり、uv / Poetry / Flit / Hatch など主要ツールが全て対応している。requirements.txt は以下のケースで引き続き使用する。
- レガシープロジェクトとの互換性
- Docker の `pip install -r requirements.txt` との統合
- ハッシュ検証（`--require-hashes`）が必要な場合

uv や pip-compile で `pyproject.toml` から `requirements.txt` を生成するのが最も再現性が高い。

### Q8: パッケージマネージャーの移行はどう進めるべき？

**A:** 段階的なアプローチを推奨。
1. **調査**: 現在の依存で互換性問題がないか確認
2. **ブランチで検証**: 移行ブランチで CI が通ることを確認
3. **ロックファイル変換**: `pnpm import` 等で既存ロックファイルを変換
4. **チーム通知**: 移行日を決めて全員に周知
5. **一斉切替**: マージ後に古いロックファイルを削除
6. **Corepack 設定**: 間違ったパッケージマネージャーの使用を防止

---

## 12. まとめ

| エコシステム | 推奨ツール | ロックファイル | 備考 |
|------------|-----------|---------------|------|
| Node.js | pnpm | pnpm-lock.yaml | ディスク効率最良 |
| Node.js (シンプル) | npm | package-lock.json | 追加インストール不要 |
| Node.js (Zero-Installs) | yarn (PnP) | yarn.lock | CI 高速化 |
| Python | uv | uv.lock | 超高速・次世代 |
| Python (既存) | Poetry | poetry.lock | エコシステム成熟 |
| Python (最小) | pip + pip-tools | requirements.txt | 標準ライブラリのみ |
| Rust | Cargo | Cargo.lock | 公式唯一 |
| macOS ツール | Homebrew | Brewfile.lock.json | Brewfile でチーム統一 |

### パッケージ管理の5原則

```
1. ロックファイルは必ずコミットする
   → 再現性のないビルドは信頼できない

2. CI ではクリーンインストールを行う
   → npm ci / pnpm install --frozen-lockfile

3. グローバルインストールを避ける
   → devDependencies + npx / pnpm exec

4. チーム全体で1つのパッケージマネージャーに統一する
   → Corepack + packageManager フィールドで強制

5. 依存の更新は計画的に行う
   → Renovate / Dependabot で自動 PR + 人間がレビュー
```

---

## 次に読むべきガイド

- [02-monorepo-setup.md](./02-monorepo-setup.md) -- モノレポでのワークスペース活用
- [03-linter-formatter.md](./03-linter-formatter.md) -- Linter/Formatter の設定
- [00-version-managers.md](./00-version-managers.md) -- ランタイムのバージョン管理

---

## 参考文献

1. **pnpm Documentation** -- https://pnpm.io/ja/ -- pnpm 公式ドキュメント（日本語）。
2. **uv Documentation** -- https://docs.astral.sh/uv/ -- uv 公式ドキュメント。pip 比較ベンチマークあり。
3. **Corepack Documentation** -- https://nodejs.org/api/corepack.html -- Node.js 公式の Corepack 解説。
4. **Homebrew Bundle** -- https://github.com/Homebrew/homebrew-bundle -- Brewfile の仕様と使い方。
5. **Yarn Berry** -- https://yarnpkg.com/ -- Yarn v4 の公式ドキュメント。PnP の詳細解説。
6. **npm Documentation** -- https://docs.npmjs.com/ -- npm 公式ドキュメント。セキュリティ機能の解説。
7. **Cargo Book** -- https://doc.rust-lang.org/cargo/ -- Cargo 公式ドキュメント。ワークスペースの詳細。
8. **Poetry Documentation** -- https://python-poetry.org/docs/ -- Poetry 公式ドキュメント。
9. **Turborepo** -- https://turbo.build/ -- Turborepo 公式。モノレポのビルドシステム。
10. **Socket.dev** -- https://socket.dev/ -- サプライチェーンセキュリティ。npm パッケージの安全性分析。
