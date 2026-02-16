# プロジェクト標準 (Project Standards)

> EditorConfig、.npmrc、.nvmrc などの共通設定ファイルを活用し、チーム全体で一貫したコーディング規約と開発環境を維持するための標準化手法を学ぶ。

## この章で学ぶこと

1. **EditorConfig によるエディタ横断のフォーマット統一** -- タブ/スペース、改行コード、文字コードをエディタに依存せず統一する設定を理解する
2. **.npmrc / .nvmrc / .node-version によるランタイム統一** -- Node.js のバージョンとパッケージマネージャの動作をチーム内で揃える手法を習得する
3. **Linter / Formatter / Git Hooks の統合設定** -- ESLint、Prettier、husky、lint-staged を組み合わせた品質ゲートを構築する
4. **多言語プロジェクトへの標準化適用** -- Python、Go、Rust など複数言語が混在するプロジェクトでの統一設定手法を習得する
5. **CI/CD との統合による品質の二重防御** -- ローカルフックと CI パイプラインを連携させ、品質基準を自動的に強制する仕組みを構築する

---

## 1. プロジェクト標準化の全体像

```
+------------------------------------------------------------------+
|             プロジェクト標準化レイヤー                               |
+------------------------------------------------------------------+
|                                                                  |
|  [レイヤー 1] エディタ設定                                        |
|    .editorconfig        -- タブ幅、改行コード、文字コード           |
|    .vscode/settings.json -- VS Code 固有設定                     |
|    .idea/               -- JetBrains 固有設定                    |
|                                                                  |
|  [レイヤー 2] ランタイム設定                                      |
|    .nvmrc / .node-version -- Node.js バージョン固定               |
|    .npmrc               -- パッケージマネージャ設定                |
|    .tool-versions       -- asdf 全般 (Ruby, Python 等)           |
|    .python-version      -- Python バージョン固定                  |
|    rust-toolchain.toml  -- Rust ツールチェーン固定                 |
|                                                                  |
|  [レイヤー 3] コード品質                                          |
|    eslint.config.js     -- Lint ルール                            |
|    .prettierrc          -- フォーマットルール                      |
|    biome.json           -- Biome 統合設定                         |
|    tsconfig.json        -- TypeScript 設定                       |
|    pyproject.toml       -- Python Lint/Format (ruff, black)      |
|    .golangci.yml        -- Go Lint 設定                           |
|                                                                  |
|  [レイヤー 4] Git ワークフロー                                    |
|    .husky/              -- Git フック                             |
|    .lintstagedrc        -- ステージファイルの自動修正               |
|    .commitlintrc        -- コミットメッセージ規約                  |
|    .gitattributes       -- 改行コード・バイナリ判定                |
|    .gitignore           -- 追跡対象外ファイルの定義                 |
|                                                                  |
|  [レイヤー 5] CI/CD パイプライン                                  |
|    .github/workflows/   -- GitHub Actions ワークフロー             |
|    .gitlab-ci.yml       -- GitLab CI 設定                         |
|    Dockerfile           -- コンテナビルド設定                      |
|    docker-compose.yml   -- ローカルサービス構成                    |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.1 標準化のメリットと導入コスト

| 観点 | 標準化なし | 標準化あり |
|------|----------|----------|
| 新メンバーの環境構築 | 1-2日 | 5-15分 |
| コードレビューでのスタイル指摘 | 毎回発生 | 自動修正で不要 |
| CI での想定外エラー | 頻発 | 事前に防止 |
| バージョン不一致バグ | 再現困難 | engines で検知 |
| コミットメッセージの品質 | バラバラ | Conventional Commits 準拠 |
| 導入コスト | -- | 初回 2-4 時間 |
| メンテナンスコスト | -- | 月 30 分程度 |

### 1.2 段階的な導入戦略

既存プロジェクトへの標準化導入は、一度にすべてを適用するのではなく、段階的に進めるのが安全である。

```
+------------------------------------------------------------------+
|              標準化の段階的導入ロードマップ                          |
+------------------------------------------------------------------+
|                                                                  |
|  Phase 1 (初日): 基盤設定                                        |
|    ✓ .editorconfig                                               |
|    ✓ .gitattributes                                              |
|    ✓ .gitignore                                                  |
|    ✓ .nvmrc / .node-version                                     |
|                                                                  |
|  Phase 2 (1週目): コード品質                                     |
|    ✓ ESLint / Biome 設定                                         |
|    ✓ Prettier / フォーマッター設定                                |
|    ✓ .npmrc 設定                                                 |
|    ✓ VS Code 共有設定                                             |
|                                                                  |
|  Phase 3 (2週目): Git ワークフロー                                |
|    ✓ husky + lint-staged                                         |
|    ✓ commitlint                                                  |
|    ✓ PR テンプレート                                              |
|                                                                  |
|  Phase 4 (3週目): CI/CD 統合                                     |
|    ✓ GitHub Actions / GitLab CI での品質チェック                   |
|    ✓ 自動テスト                                                   |
|    ✓ 自動デプロイ                                                 |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. EditorConfig

### 2.1 基本設定

```ini
# .editorconfig
# https://editorconfig.org

root = true

# 全ファイル共通
[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2

# Python
[*.py]
indent_size = 4

# Go
[*.go]
indent_style = tab
indent_size = 4

# Rust
[*.rs]
indent_size = 4

# Java / Kotlin
[*.{java,kt,kts}]
indent_size = 4

# C# / .NET
[*.{cs,csx}]
indent_size = 4

# Makefile (タブ必須)
[Makefile]
indent_style = tab

# マークダウン (末尾スペースは意味がある)
[*.md]
trim_trailing_whitespace = false

# YAML
[*.{yml,yaml}]
indent_size = 2

# JSON
[*.json]
indent_size = 2

# TOML
[*.toml]
indent_size = 2

# Shell scripts
[*.sh]
end_of_line = lf
indent_size = 2

# Docker
[Dockerfile*]
indent_size = 4

# Terraform / HCL
[*.{tf,tfvars,hcl}]
indent_size = 2

# XML / SVG
[*.{xml,svg}]
indent_size = 2
```

### 2.2 EditorConfig の対応状況

| エディタ | ネイティブ対応 | プラグイン |
|---------|-------------|-----------|
| VS Code | プラグイン必要 | EditorConfig for VS Code |
| JetBrains (IntelliJ等) | 標準対応 | 不要 |
| Vim / Neovim | プラグイン必要 | editorconfig-vim |
| Sublime Text | プラグイン必要 | EditorConfig |
| Emacs | プラグイン必要 | editorconfig-emacs |
| GitHub Web | 標準対応 | 不要 |
| Zed | 標準対応 | 不要 |
| Cursor | プラグイン必要 | EditorConfig for VS Code |

### 2.3 EditorConfig の高度な設定パターン

```ini
# .editorconfig (大規模プロジェクト向け拡張)

root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2
max_line_length = 120

# プロトコルバッファ
[*.proto]
indent_size = 2

# GraphQL
[*.{graphql,gql}]
indent_size = 2

# 環境変数ファイル
[.env*]
insert_final_newline = true

# バッチファイル / PowerShell
[*.{bat,cmd}]
end_of_line = crlf
[*.ps1]
end_of_line = crlf
charset = utf-8-bom

# ライセンスファイル
[LICENSE*]
insert_final_newline = true
trim_trailing_whitespace = true

# Gemfile / Rakefile (Ruby)
[{Gemfile,Rakefile,*.rb}]
indent_size = 2

# PHP
[*.php]
indent_size = 4

# ソリューションファイル
[*.sln]
end_of_line = crlf

# csproj (Microsoft XML形式)
[*.{csproj,vbproj,vcxproj,proj}]
indent_size = 2
end_of_line = crlf
```

### 2.4 EditorConfig の検証

EditorConfig の設定が正しく適用されているかを検証するスクリプトを用意しておくと、トラブルシューティングが容易になる。

```bash
#!/bin/bash
# scripts/check-editorconfig.sh
# EditorConfig の適用状態を検証する

set -euo pipefail

ERRORS=0

# UTF-8 BOM チェック
echo "=== UTF-8 BOM チェック ==="
BOM_FILES=$(find . -type f \( -name "*.ts" -o -name "*.js" -o -name "*.json" \) \
  -exec grep -Pl '\xEF\xBB\xBF' {} \; 2>/dev/null || true)
if [ -n "$BOM_FILES" ]; then
  echo "FAIL: BOM 付きファイルが見つかりました:"
  echo "$BOM_FILES"
  ((ERRORS++))
else
  echo "PASS: BOM 付きファイルなし"
fi

# 末尾改行チェック
echo ""
echo "=== 末尾改行チェック ==="
MISSING_NEWLINE=$(find . -type f \( -name "*.ts" -o -name "*.js" \) \
  -not -path "*/node_modules/*" -not -path "*/.git/*" \
  -exec sh -c '[ -s "$1" ] && [ "$(tail -c1 "$1" | xxd -p)" != "0a" ] && echo "$1"' _ {} \; 2>/dev/null || true)
if [ -n "$MISSING_NEWLINE" ]; then
  echo "WARN: 末尾改行がないファイル:"
  echo "$MISSING_NEWLINE"
else
  echo "PASS: 全ファイルに末尾改行あり"
fi

# CRLF チェック (LF が正しい環境)
echo ""
echo "=== 改行コードチェック ==="
CRLF_FILES=$(find . -type f \( -name "*.ts" -o -name "*.js" -o -name "*.json" \) \
  -not -path "*/node_modules/*" -not -path "*/.git/*" \
  -exec grep -Prl '\r\n' {} \; 2>/dev/null || true)
if [ -n "$CRLF_FILES" ]; then
  echo "WARN: CRLF が検出されたファイル:"
  echo "$CRLF_FILES"
else
  echo "PASS: CRLF ファイルなし"
fi

echo ""
echo "=== 結果: ${ERRORS} エラー ==="
exit $ERRORS
```

---

## 3. Node.js バージョン管理

### 3.1 .nvmrc

```
# .nvmrc
20.11.0
```

### 3.2 .node-version (fnm / nodenv / volta 対応)

```
# .node-version
20.11.0
```

### 3.3 package.json の engines フィールド

```jsonc
// package.json
{
  "engines": {
    "node": ">=20.0.0",
    "npm": ">=10.0.0"
  },
  "packageManager": "pnpm@9.0.0",
  "volta": {
    "node": "20.11.0",
    "pnpm": "9.0.0"
  }
}
```

### 3.4 バージョン管理ツール比較

```
+------------------------------------------------------------------+
|          Node.js バージョン管理ツール比較                           |
+------------------------------------------------------------------+
| ツール  | 設定ファイル        | 自動切替 | 速度   | 対応言語       |
|---------|-------------------|---------|--------|---------------|
| nvm     | .nvmrc            | フック   | 遅い   | Node.js のみ   |
| fnm     | .node-version     | 自動    | 高速   | Node.js のみ   |
| volta   | package.json      | 自動    | 高速   | Node.js のみ   |
| asdf    | .tool-versions    | 自動    | 中     | 多言語対応     |
| mise    | .tool-versions    | 自動    | 高速   | 多言語対応     |
| nodenv  | .node-version     | 自動    | 高速   | Node.js のみ   |
+------------------------------------------------------------------+
```

### 3.5 fnm の詳細設定

fnm (Fast Node Manager) は Rust 製の高速な Node.js バージョン管理ツールで、nvm の代替として推奨される。

```bash
# fnm のインストール
# macOS
brew install fnm

# Linux / macOS (curl)
curl -fsSL https://fnm.vercel.app/install | bash

# Windows (winget)
winget install Schniz.fnm

# シェル設定 (~/.zshrc or ~/.bashrc)
eval "$(fnm env --use-on-cd)"
# --use-on-cd: ディレクトリ移動時に .nvmrc / .node-version を自動読み込み

# 基本操作
fnm install 20.11.0        # インストール
fnm use 20.11.0             # 切り替え
fnm default 20.11.0         # デフォルト設定
fnm list                    # インストール済み一覧
fnm list-remote             # 利用可能なバージョン一覧
fnm current                 # 現在のバージョン

# .nvmrc を読み込んでインストール & 使用
fnm install
fnm use
```

### 3.6 Volta の詳細設定

Volta は package.json 内でバージョンを管理する独自のアプローチを採る。

```bash
# Volta のインストール
curl https://get.volta.sh | bash

# Node.js のインストール & ピン留め
volta install node@20.11.0
volta pin node@20.11.0      # package.json に記録

# パッケージマネージャのピン留め
volta install pnpm@9.0.0
volta pin pnpm@9.0.0

# グローバルツールのインストール
volta install typescript
volta install @biomejs/biome

# package.json に自動追記される:
# {
#   "volta": {
#     "node": "20.11.0",
#     "pnpm": "9.0.0"
#   }
# }
```

### 3.7 mise (旧 rtx) による多言語バージョン管理

```toml
# .mise.toml (旧 .tool-versions 形式も対応)
[tools]
node = "20.11.0"
python = "3.12.1"
ruby = "3.3.0"
go = "1.22.0"
rust = "1.76.0"
java = "temurin-21.0.2+13.0.LTS"
terraform = "1.7.0"

[env]
NODE_ENV = "development"

[tasks.dev]
run = "npm run dev"
description = "開発サーバー起動"

[tasks.test]
run = "npm test"
description = "テスト実行"
```

```bash
# mise のインストール
brew install mise

# シェル設定
eval "$(mise activate zsh)"

# バージョンのインストール
mise install           # .mise.toml に記載の全ツールをインストール
mise use node@20.11.0  # .mise.toml にバージョンを記録

# タスク実行
mise run dev
mise run test
```

---

## 4. .npmrc の設定

### 4.1 プロジェクト用 .npmrc

```ini
# .npmrc

# エンジンバージョンを厳密にチェック
engine-strict=true

# package-lock.json を必ず生成
package-lock=true

# 正確なバージョンでインストール (^ や ~ を付けない)
save-exact=true

# npm audit のレベル設定
audit-level=moderate

# プライベートレジストリ (社内パッケージがある場合)
# @mycompany:registry=https://npm.mycompany.com/
# //npm.mycompany.com/:_authToken=${NPM_TOKEN}

# ピアデプとの自動解決
legacy-peer-deps=false
auto-install-peers=true

# ログレベル
loglevel=warn

# Node.js のバージョンが合わない場合に npm install を阻止
# （engines + engine-strict=true と併用）
```

### 4.2 pnpm の場合 (.npmrc + pnpm-workspace.yaml)

```ini
# .npmrc (pnpm 用)
shamefully-hoist=false
strict-peer-dependencies=true
auto-install-peers=true
```

```yaml
# pnpm-workspace.yaml
packages:
  - 'apps/*'
  - 'packages/*'
  - 'tools/*'
```

### 4.3 yarn の設定 (.yarnrc.yml)

```yaml
# .yarnrc.yml (Yarn Berry / Yarn 4)
nodeLinker: node-modules  # PnP を使わない場合
enableGlobalCache: false
checksumBehavior: update

# プライベートレジストリ
npmScopes:
  mycompany:
    npmRegistryServer: "https://npm.mycompany.com/"
    npmAuthToken: "${NPM_TOKEN}"

# プラグイン
plugins:
  - path: .yarn/plugins/@yarnpkg/plugin-interactive-tools.cjs
    spec: "@yarnpkg/plugin-interactive-tools"
  - path: .yarn/plugins/@yarnpkg/plugin-workspace-tools.cjs
    spec: "@yarnpkg/plugin-workspace-tools"
```

### 4.4 .npmrc のセキュリティ設定

```ini
# .npmrc (セキュリティ強化)

# postinstall スクリプトの実行を制限
ignore-scripts=false

# パッケージの provenance (出所) を検証
# npm v9.5+ で利用可能
audit=true
audit-level=moderate

# パッケージの署名を検証 (npm v9.8+)
# sign-git-tag=true

# 2FA を強制 (npm publish 時)
# auth-type=web
```

### 4.5 パッケージマネージャ比較

```
+------------------------------------------------------------------+
|            パッケージマネージャ比較                                  |
+------------------------------------------------------------------+
| 項目          | npm     | pnpm     | yarn     | bun       |
|---------------|---------|----------|----------|-----------|
| ディスク使用量 | 多い    | 少ない   | 中       | 少ない     |
| インストール速度| 中     | 高速     | 高速     | 最速       |
| ロックファイル | package-lock.json | pnpm-lock.yaml | yarn.lock | bun.lockb |
| モノレポ対応   | workspaces | workspaces | workspaces | workspaces |
| 厳密な依存解決 | 普通   | 厳密     | 普通     | 普通       |
| Phantom Deps  | あり   | なし     | あり     | あり       |
| Node.js 不要  | いいえ | いいえ   | いいえ   | はい(Bun)  |
+------------------------------------------------------------------+
```

---

## 5. .gitattributes

### 5.1 基本設定

```gitattributes
# .gitattributes

# 改行コードの統一
* text=auto eol=lf

# 明示的なテキストファイル
*.js    text eol=lf
*.ts    text eol=lf
*.jsx   text eol=lf
*.tsx   text eol=lf
*.json  text eol=lf
*.yml   text eol=lf
*.yaml  text eol=lf
*.md    text eol=lf
*.css   text eol=lf
*.html  text eol=lf
*.sh    text eol=lf
*.py    text eol=lf
*.go    text eol=lf
*.rs    text eol=lf
*.java  text eol=lf
*.kt    text eol=lf
*.rb    text eol=lf
*.php   text eol=lf
*.sql   text eol=lf
*.graphql text eol=lf
*.proto text eol=lf
*.toml  text eol=lf
*.ini   text eol=lf
*.cfg   text eol=lf
*.env   text eol=lf

# Windows バッチファイル
*.bat   text eol=crlf
*.cmd   text eol=crlf
*.ps1   text eol=crlf

# バイナリファイル
*.png   binary
*.jpg   binary
*.jpeg  binary
*.gif   binary
*.ico   binary
*.webp  binary
*.avif  binary
*.woff  binary
*.woff2 binary
*.ttf   binary
*.eot   binary
*.pdf   binary
*.zip   binary
*.tar.gz binary
*.mp4   binary
*.mp3   binary
*.wav   binary

# ロックファイル (マージ時にコンフリクトを防ぐ)
package-lock.json merge=ours linguist-generated
pnpm-lock.yaml   merge=ours linguist-generated
yarn.lock        merge=ours linguist-generated

# 自動生成ファイル (diff に表示しない)
*.min.js linguist-generated
*.min.css linguist-generated
dist/** linguist-generated
```

### 5.2 .gitattributes の高度な設定

```gitattributes
# Git LFS (Large File Storage) 設定
# 大きなバイナリファイルを LFS で管理
*.psd filter=lfs diff=lfs merge=lfs -text
*.sketch filter=lfs diff=lfs merge=lfs -text
*.fig filter=lfs diff=lfs merge=lfs -text
*.ai filter=lfs diff=lfs merge=lfs -text
*.mov filter=lfs diff=lfs merge=lfs -text

# Diff のカスタマイズ
# JSON の差分を見やすくする
*.json diff=json

# CSS / SCSS の差分を関数単位で表示
*.css diff=css
*.scss diff=css

# Markdown の差分を見出し単位で表示
*.md diff=markdown

# Go の差分を関数単位で表示
*.go diff=golang

# Ruby の差分をメソッド単位で表示
*.rb diff=ruby
```

### 5.3 .gitignore の標準テンプレート

```gitignore
# .gitignore

# === 依存関係 ===
node_modules/
.pnpm-store/
vendor/
__pycache__/
*.pyc
.venv/
venv/

# === ビルド成果物 ===
dist/
build/
.next/
.nuxt/
.output/
out/
*.tsbuildinfo

# === 環境変数 ===
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# === エディタ ===
.idea/
*.swp
*.swo
*~
.project
.classpath
.settings/

# === OS ===
.DS_Store
Thumbs.db
Desktop.ini

# === テスト / カバレッジ ===
coverage/
.nyc_output/
*.lcov
test-results/
playwright-report/

# === ログ ===
*.log
logs/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*

# === Docker ===
docker-compose.override.yml

# === Terraform ===
*.tfstate
*.tfstate.backup
.terraform/

# === その他 ===
.cache/
.temp/
.tmp/
*.bak
*.orig
```

---

## 6. VS Code 共有設定

### 6.1 .vscode/settings.json

```jsonc
// .vscode/settings.json
{
  // エディタ基本設定
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
    "source.organizeImports": "explicit"
  },
  "editor.defaultFormatter": "esbenp.prettier-vscode",

  // TypeScript
  "typescript.tsdk": "node_modules/typescript/lib",
  "typescript.enablePromptUseWorkspaceTsdk": true,

  // ファイル除外
  "files.exclude": {
    "**/.git": true,
    "**/node_modules": true,
    "**/dist": true,
    "**/.next": true
  },

  // 検索除外
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/*.min.js": true,
    "**/pnpm-lock.yaml": true
  },

  // ファイルの自動保存
  "files.autoSave": "onFocusChange",

  // 末尾の空白を自動トリム
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.trimFinalNewlines": true,

  // 言語固有設定
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[jsonc]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[markdown]": {
    "editor.wordWrap": "on",
    "files.trimTrailingWhitespace": false
  },
  "[yaml]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[css]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[html]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },

  // Tailwind CSS IntelliSense
  "tailwindCSS.experimental.classRegex": [
    ["cva\\(([^)]*)\\)", "[\"'`]([^\"'`]*).*?[\"'`]"],
    ["cx\\(([^)]*)\\)", "(?:'|\"|`)([^']*)(?:'|\"|`)"]
  ],

  // ESLint
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ],

  // テスト
  "testing.automaticallyOpenPeekView": "never"
}
```

### 6.2 .vscode/extensions.json

```jsonc
// .vscode/extensions.json
{
  "recommendations": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "editorconfig.editorconfig",
    "bradlc.vscode-tailwindcss",
    "prisma.prisma",
    "ms-azuretools.vscode-docker",
    "github.copilot",
    "github.copilot-chat",
    "ms-vscode.vscode-typescript-next",
    "streetsidesoftware.code-spell-checker",
    "usernamehw.errorlens",
    "eamodio.gitlens",
    "biomejs.biome"
  ],
  "unwantedRecommendations": [
    "hookyqr.beautify",
    "ms-vscode.vscode-typescript-tslint-plugin"
  ]
}
```

### 6.3 .vscode/launch.json (デバッグ設定)

```jsonc
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Next.js: Debug Server",
      "type": "node",
      "request": "launch",
      "runtimeExecutable": "${workspaceFolder}/node_modules/.bin/next",
      "args": ["dev"],
      "skipFiles": ["<node_internals>/**"],
      "console": "integratedTerminal"
    },
    {
      "name": "Vitest: Current File",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/node_modules/.bin/vitest",
      "args": ["run", "${relativeFile}"],
      "console": "integratedTerminal",
      "skipFiles": ["<node_internals>/**"]
    },
    {
      "name": "Vitest: Watch Mode",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/node_modules/.bin/vitest",
      "args": ["--watch"],
      "console": "integratedTerminal",
      "skipFiles": ["<node_internals>/**"]
    }
  ]
}
```

### 6.4 .vscode/tasks.json (タスク設定)

```jsonc
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "dev",
      "type": "shell",
      "command": "make dev",
      "group": "build",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "test",
      "type": "shell",
      "command": "make test",
      "group": "test"
    },
    {
      "label": "lint",
      "type": "shell",
      "command": "make lint",
      "group": "test",
      "problemMatcher": ["$eslint-stylish"]
    },
    {
      "label": "typecheck",
      "type": "shell",
      "command": "make typecheck",
      "group": "build",
      "problemMatcher": ["$tsc"]
    }
  ]
}
```

---

## 7. Git Hooks (husky + lint-staged)

### 7.1 セットアップ

```bash
# husky と lint-staged のインストール
pnpm add -D husky lint-staged

# husky の初期化
pnpm exec husky init

# pre-commit フックの作成
echo "npx lint-staged" > .husky/pre-commit
```

```jsonc
// package.json
{
  "scripts": {
    "prepare": "husky"
  },
  "lint-staged": {
    "*.{ts,tsx,js,jsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,yml,yaml,md}": [
      "prettier --write"
    ],
    "*.css": [
      "prettier --write"
    ],
    "*.py": [
      "ruff check --fix",
      "ruff format"
    ],
    "*.go": [
      "gofmt -w",
      "go vet"
    ]
  }
}
```

### 7.2 husky フック

```bash
#!/bin/sh
# .husky/pre-commit
npx lint-staged
```

```bash
#!/bin/sh
# .husky/commit-msg
npx --no -- commitlint --edit "$1"
```

```bash
#!/bin/sh
# .husky/pre-push
# プッシュ前にテストと型チェックを実行
npm run typecheck
npm run test -- --run
```

### 7.3 Commitlint 設定

```javascript
// commitlint.config.js
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',     // 新機能
        'fix',      // バグ修正
        'docs',     // ドキュメント
        'style',    // フォーマット変更
        'refactor', // リファクタリング
        'perf',     // パフォーマンス改善
        'test',     // テスト
        'chore',    // ビルド・ツール
        'ci',       // CI 設定
        'revert',   // 取り消し
        'build',    // ビルドシステム変更
        'deps',     // 依存関係更新
      ],
    ],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [1, 'always', 100],
    'header-max-length': [2, 'always', 100],
    'scope-case': [2, 'always', 'lower-case'],
  },
};
```

### 7.4 Conventional Commits の運用ガイド

```
+------------------------------------------------------------------+
|            Conventional Commits フォーマット                        |
+------------------------------------------------------------------+
|                                                                  |
|  <type>(<scope>): <description>                                  |
|                                                                  |
|  [body]                                                          |
|                                                                  |
|  [footer]                                                        |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  例:                                                             |
|  feat(auth): ソーシャルログイン機能を追加                          |
|  fix(api): ユーザー検索のN+1問題を修正                             |
|  docs(readme): セットアップ手順を更新                              |
|  refactor(db): クエリビルダーをPrismaに移行                        |
|  perf(search): 全文検索のインデックスを最適化                      |
|  test(user): ユーザー登録のE2Eテストを追加                         |
|  ci(deploy): ステージング環境の自動デプロイを設定                   |
|  chore(deps): TypeScript を 5.4 にアップデート                    |
|                                                                  |
|  BREAKING CHANGE:                                                |
|  feat(api)!: レスポンス形式をJSONAPIに変更                         |
|  → "!" または footer の BREAKING CHANGE で破壊的変更を示す         |
|                                                                  |
+------------------------------------------------------------------+
```

### 7.5 lefthook (husky の代替)

lefthook は Go 製の高速な Git フック管理ツールで、husky の代替として注目されている。

```yaml
# lefthook.yml
pre-commit:
  parallel: true
  commands:
    lint:
      glob: "*.{ts,tsx,js,jsx}"
      run: npx eslint --fix {staged_files} && git add {staged_files}
    format:
      glob: "*.{ts,tsx,js,jsx,json,yml,yaml,md,css}"
      run: npx prettier --write {staged_files} && git add {staged_files}
    typecheck:
      run: npx tsc --noEmit

commit-msg:
  commands:
    commitlint:
      run: npx commitlint --edit {1}

pre-push:
  commands:
    test:
      run: npx vitest run
```

---

## 8. ESLint (Flat Config) の設定

### 8.1 eslint.config.js (ESLint v9+)

```javascript
// eslint.config.js
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactPlugin from 'eslint-plugin-react';
import reactHooksPlugin from 'eslint-plugin-react-hooks';
import importPlugin from 'eslint-plugin-import';

export default tseslint.config(
  // グローバル無視
  {
    ignores: [
      'dist/**',
      'build/**',
      '.next/**',
      'node_modules/**',
      'coverage/**',
      '*.config.js',
      '*.config.ts',
    ],
  },

  // 基本ルール
  js.configs.recommended,

  // TypeScript ルール
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        project: './tsconfig.json',
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },

  // React ルール
  {
    plugins: {
      react: reactPlugin,
      'react-hooks': reactHooksPlugin,
    },
    rules: {
      ...reactPlugin.configs.recommended.rules,
      ...reactHooksPlugin.configs.recommended.rules,
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',
    },
    settings: {
      react: { version: 'detect' },
    },
  },

  // Import ルール
  {
    plugins: {
      import: importPlugin,
    },
    rules: {
      'import/order': [
        'error',
        {
          groups: [
            'builtin',
            'external',
            'internal',
            ['parent', 'sibling'],
            'index',
            'type',
          ],
          'newlines-between': 'always',
          alphabetize: { order: 'asc', caseInsensitive: true },
        },
      ],
      'import/no-duplicates': 'error',
    },
  },

  // カスタムルール
  {
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/consistent-type-imports': [
        'error',
        { prefer: 'type-imports' },
      ],
      'no-console': ['warn', { allow: ['warn', 'error'] }],
    },
  },
);
```

### 8.2 Prettier 設定

```jsonc
// .prettierrc
{
  "semi": true,
  "trailingComma": "all",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true,
  "arrowParens": "always",
  "endOfLine": "lf",
  "plugins": ["prettier-plugin-tailwindcss"],
  "overrides": [
    {
      "files": "*.md",
      "options": {
        "printWidth": 80,
        "proseWrap": "always"
      }
    }
  ]
}
```

```
# .prettierignore
dist/
build/
.next/
node_modules/
coverage/
pnpm-lock.yaml
package-lock.json
yarn.lock
*.min.js
*.min.css
```

### 8.3 Biome 設定 (ESLint + Prettier の代替)

```jsonc
// biome.json
{
  "$schema": "https://biomejs.dev/schemas/1.9.0/schema.json",
  "organizeImports": {
    "enabled": true
  },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "complexity": {
        "noBannedTypes": "error",
        "noExtraBooleanCast": "error"
      },
      "correctness": {
        "noUnusedVariables": "error",
        "noUnusedImports": "error",
        "useExhaustiveDependencies": "warn"
      },
      "suspicious": {
        "noExplicitAny": "error",
        "noConsoleLog": "warn"
      },
      "style": {
        "useConst": "error",
        "useTemplate": "error"
      }
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100,
    "lineEnding": "lf"
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "single",
      "trailingCommas": "all",
      "semicolons": "always",
      "arrowParentheses": "always"
    }
  },
  "json": {
    "formatter": {
      "trailingCommas": "none"
    }
  },
  "files": {
    "ignore": [
      "dist/**",
      "build/**",
      ".next/**",
      "node_modules/**",
      "coverage/**"
    ]
  }
}
```

---

## 9. TypeScript 設定

### 9.1 tsconfig.json (共通ベース)

```jsonc
// tsconfig.json
{
  "compilerOptions": {
    // 言語設定
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",

    // 厳密モード
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,

    // 出力設定
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",

    // パスエイリアス
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"],
      "@/components/*": ["./src/components/*"],
      "@/lib/*": ["./src/lib/*"],
      "@/types/*": ["./src/types/*"]
    },

    // インポート
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,
    "isolatedModules": true,

    // 型チェックの追加設定
    "skipLibCheck": true,
    "incremental": true
  },
  "include": ["src/**/*.ts", "src/**/*.tsx"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## 10. プロジェクト標準ファイル一覧

```
+------------------------------------------------------------------+
|           プロジェクトルートに配置すべきファイル一覧                  |
+------------------------------------------------------------------+
| ファイル                  | 用途                    | 必須度     |
|--------------------------|------------------------|-----------|
| .editorconfig            | エディタ横断フォーマット  | 必須       |
| .gitattributes           | Git の改行/バイナリ設定  | 必須       |
| .gitignore               | Git 除外ルール          | 必須       |
| .nvmrc / .node-version   | Node.js バージョン固定  | 推奨       |
| .npmrc                   | npm/pnpm 動作設定       | 推奨       |
| .prettierrc              | Prettier ルール         | 推奨       |
| .prettierignore          | Prettier 除外           | 推奨       |
| eslint.config.js         | ESLint ルール           | 推奨       |
| biome.json               | Biome 統合設定          | 代替推奨   |
| tsconfig.json            | TypeScript 設定         | TS利用時必須 |
| .vscode/settings.json    | VS Code 共有設定        | 推奨       |
| .vscode/extensions.json  | 推奨拡張機能            | 推奨       |
| .vscode/launch.json      | デバッグ設定            | 任意       |
| .husky/                  | Git フック              | 推奨       |
| commitlint.config.js     | コミットメッセージ規約   | 推奨       |
| .mise.toml               | 多言語バージョン管理     | 任意       |
| Makefile                 | タスクランナー          | 推奨       |
| docker-compose.yml       | ローカルサービス構成     | 推奨       |
| renovate.json            | 依存関係自動更新        | 推奨       |
+------------------------------------------------------------------+
```

---

## 11. 依存関係の自動更新 (Renovate / Dependabot)

### 11.1 Renovate 設定

```jsonc
// renovate.json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": [
    "config:recommended",
    ":timezone(Asia/Tokyo)",
    ":semanticCommitTypeAll(chore)",
    "group:allNonMajor"
  ],
  "schedule": ["before 9am on Monday"],
  "labels": ["dependencies"],
  "automerge": true,
  "automergeType": "pr",
  "platformAutomerge": true,
  "packageRules": [
    {
      "matchUpdateTypes": ["major"],
      "automerge": false,
      "labels": ["dependencies", "major"]
    },
    {
      "matchPackagePatterns": ["eslint", "prettier", "biome"],
      "groupName": "linting tools"
    },
    {
      "matchPackagePatterns": ["vitest", "playwright", "@testing-library"],
      "groupName": "testing tools"
    },
    {
      "matchUpdateTypes": ["patch", "minor"],
      "matchPackagePatterns": ["*"],
      "automerge": true
    }
  ],
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security"]
  }
}
```

### 11.2 Dependabot 設定

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "Asia/Tokyo"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
    groups:
      development-dependencies:
        dependency-type: "development"
        update-types:
          - "minor"
          - "patch"
      production-dependencies:
        dependency-type: "production"
        update-types:
          - "minor"
          - "patch"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "ci"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "docker"
```

---

## 12. CI/CD との統合

### 12.1 GitHub Actions による品質チェック

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm run lint
      - run: pnpm run typecheck
      - run: pnpm run format:check

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm run test -- --coverage
      - uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage/

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'
      - run: pnpm install --frozen-lockfile
      - run: pnpm run build
```

### 12.2 ブランチ保護ルール

GitHub のブランチ保護ルールを設定し、CI が通らないマージを防止する。

```
+------------------------------------------------------------------+
|           main ブランチ保護ルール推奨設定                            |
+------------------------------------------------------------------+
|                                                                  |
|  ✓ Require a pull request before merging                         |
|    ✓ Require approvals: 1                                        |
|    ✓ Dismiss stale pull request approvals when new commits       |
|      are pushed                                                  |
|    ✓ Require review from Code Owners                             |
|                                                                  |
|  ✓ Require status checks to pass before merging                  |
|    ✓ Require branches to be up to date before merging            |
|    Required checks:                                              |
|      - lint                                                      |
|      - test                                                      |
|      - build                                                     |
|                                                                  |
|  ✓ Require signed commits                                        |
|  ✓ Require linear history                                        |
|  ✗ Allow force pushes (無効にすること)                              |
|  ✗ Allow deletions (無効にすること)                                 |
|                                                                  |
+------------------------------------------------------------------+
```

---

## アンチパターン

### アンチパターン 1: 個人設定をリポジトリにコミット

```jsonc
// NG: .vscode/settings.json に個人の好みを入れる
{
  "editor.fontSize": 18,
  "editor.fontFamily": "JetBrains Mono",
  "workbench.colorTheme": "One Dark Pro",
  "terminal.integrated.shell.osx": "/bin/zsh"
}

// OK: チームに関係する設定のみ
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "typescript.tsdk": "node_modules/typescript/lib"
}
```

**問題点**: フォントサイズやテーマなどの個人設定はチームメンバー間で異なるのが当然であり、コミットすると不要なコンフリクトが発生する。個人設定は VS Code のユーザー設定で管理し、リポジトリには Linter/Formatter 関連の設定のみコミットする。

### アンチパターン 2: engines フィールドなしでの運用

```jsonc
// NG: engines 未指定 → 各自のバージョンで動かす
{
  "name": "myapp",
  "version": "1.0.0"
}

// OK: engines + .nvmrc + engine-strict で強制
// package.json
{
  "name": "myapp",
  "version": "1.0.0",
  "engines": {
    "node": ">=20.0.0 <21.0.0",
    "npm": ">=10.0.0"
  }
}
// .npmrc
// engine-strict=true
```

**問題点**: Node.js のバージョン不一致はしばしば再現困難なバグを引き起こす。特に `Optional Chaining` や `import.meta` などの構文サポートはバージョンに依存する。`engines` + `engine-strict` で明示的にエラーにすることで、環境不一致を早期に検出できる。

### アンチパターン 3: Linter ルールの一斉適用

```bash
# NG: 既存プロジェクトに厳密なルールを一度に適用
# → 数百〜数千のエラーが出てチームが混乱
eslint . --max-warnings 0

# OK: 段階的に導入
# Step 1: warn レベルで導入（既存コードは触らない）
# Step 2: 新規コードのみ strict 適用
# Step 3: 既存コードを段階的に修正
# Step 4: error レベルに昇格
```

**問題点**: 既存の大規模コードベースに厳密な Linter ルールを一度に適用すると、大量のエラーが発生し、開発が停止する。ESLint の `--max-warnings` を段階的に減らしていく、または `eslint-disable` コメントを一括挿入してから段階的に除去する戦略が有効。

### アンチパターン 4: 設定ファイルのコピペ伝播

```
# NG: 別プロジェクトの設定をそのままコピー
# → プロジェクト固有の要件に合っていない

# OK: 共有設定パッケージを作成
# @mycompany/eslint-config
# @mycompany/prettier-config
# @mycompany/tsconfig
```

**問題点**: 設定ファイルをプロジェクト間でコピーすると、元の設定が更新されても伝播されない。npm パッケージとして共有設定を公開し、各プロジェクトから `extends` で参照する方式にすれば、一箇所の更新が全プロジェクトに反映される。

---

## FAQ

### Q1: EditorConfig と Prettier の両方が必要ですか？

**A**: はい。役割が異なる。EditorConfig はエディタの入力時の動作（タブ幅、改行コード）を制御し、Prettier は保存時のコード整形（括弧の位置、セミコロン等）を行う。EditorConfig は Prettier 非対応のファイル（Makefile、INI ファイル等）にも適用でき、エディタの種類にも依存しない。両方設定しておくことで、入力時と保存時の両方で一貫性が保たれる。

### Q2: husky の Git フックはチーム全員に自動適用されますか？

**A**: `package.json` の `"prepare": "husky"` スクリプトにより、`npm install` 実行時に自動でフックがインストールされる。ただし、`--no-verify` フラグで個人がフックをスキップすることは可能なため、CI/CD でも同じチェックを実行する二重防御が推奨される。また、pnpm の場合は `"prepare": "husky"` が自動実行されないため、明示的に `pnpm exec husky` を実行する手順をドキュメント化する必要がある。

### Q3: Biome を使えば ESLint + Prettier は不要になりますか？

**A**: ほぼ不要になるケースが多い。Biome は Rust 製の高速ツールで、Lint とフォーマットの両方を 1 つのツールで処理する。ESLint + Prettier の組み合わせに比べて 10-100 倍速い。ただし、ESLint の一部プラグイン（eslint-plugin-react-hooks, @typescript-eslint の高度なルール等）に相当する機能がまだ不足している場合がある。新規プロジェクトでは Biome を第一候補として検討し、足りないルールのみ ESLint で補完する戦略が有効。

### Q4: モノレポでの標準化はどうすればいいですか？

**A**: モノレポでは、ルートに共通設定を置き、各パッケージでオーバーライドする構成が基本。

```
monorepo/
  .editorconfig          # 全パッケージ共通
  .prettierrc            # 全パッケージ共通
  tsconfig.base.json     # 共通 TypeScript 設定
  eslint.config.js       # 共通 ESLint 設定
  packages/
    app/
      tsconfig.json      # extends: "../../tsconfig.base.json"
    api/
      tsconfig.json      # extends: "../../tsconfig.base.json"
    shared/
      tsconfig.json      # extends: "../../tsconfig.base.json"
```

Turborepo や Nx を使う場合は、タスクのキャッシュと並列実行により、CI 時間を大幅に短縮できる。

### Q5: 設定の共有パッケージはどう作りますか？

**A**: 組織内で統一した設定を配布するには、npm パッケージとして公開するのが最も効果的。

```bash
# @mycompany/eslint-config パッケージ
mkdir eslint-config && cd eslint-config
npm init --scope=@mycompany

# package.json
{
  "name": "@mycompany/eslint-config",
  "version": "1.0.0",
  "main": "index.js",
  "peerDependencies": {
    "eslint": ">=9.0.0",
    "typescript-eslint": ">=8.0.0"
  }
}

# 利用側の package.json
{
  "devDependencies": {
    "@mycompany/eslint-config": "^1.0.0"
  }
}

# 利用側の eslint.config.js
import mycompanyConfig from '@mycompany/eslint-config';
export default [...mycompanyConfig, /* プロジェクト固有のルール */];
```

### Q6: Python や Go のプロジェクトでも同じ戦略は使えますか？

**A**: 基本的な考え方は同じだが、ツールが異なる。

- **Python**: `pyproject.toml` に ruff (Linter + Formatter)、mypy (型チェック) の設定を統合。`.python-version` でバージョン固定。pre-commit フレームワークでフック管理。
- **Go**: `go.mod` でバージョン管理。`.golangci.yml` で golangci-lint の設定。`gofmt` / `goimports` でフォーマット。
- **Rust**: `rust-toolchain.toml` でバージョン固定。`clippy.toml` で Lint 設定。`rustfmt.toml` でフォーマット設定。

---

## まとめ

| 項目 | 要点 |
|------|------|
| EditorConfig | エディタ横断でタブ幅・改行コード・文字コードを統一 |
| .nvmrc | Node.js バージョンをチームで固定。volta / fnm でも対応 |
| .npmrc | `engine-strict=true` と `save-exact=true` を推奨 |
| .gitattributes | 改行コードの自動変換とバイナリファイルの判定 |
| .gitignore | 追跡対象外ファイルの明示的な定義 |
| VS Code 設定 | チーム共通設定のみコミット。個人設定は除外 |
| Git Hooks | husky + lint-staged でコミット前に自動 Lint/Format |
| Commitlint | Conventional Commits でコミットメッセージの品質を担保 |
| ESLint / Biome | Flat Config で統一。Biome は高速な代替 |
| Prettier | コードフォーマットの自動統一 |
| TypeScript | strict モード + noUncheckedIndexedAccess が推奨 |
| Renovate / Dependabot | 依存関係の自動更新で脆弱性を早期対処 |
| CI/CD 統合 | ローカルフック + CI/CD の両方で品質チェックを実行 |
| 二重防御 | ローカルフックは `--no-verify` で回避できるため CI が最後の砦 |

## 次に読むべきガイド

- [オンボーディング自動化](./01-onboarding-automation.md) -- セットアップスクリプトと Makefile
- [ドキュメント環境](./02-documentation-setup.md) -- VitePress / Docusaurus / ADR
- [Dev Container](../02-docker-dev/01-devcontainer.md) -- コンテナベースの統一開発環境

## 参考文献

1. **EditorConfig 公式** -- https://editorconfig.org/ -- EditorConfig の仕様とエディタ対応状況
2. **Conventional Commits** -- https://www.conventionalcommits.org/ja/ -- コミットメッセージ規約の仕様
3. **husky 公式ドキュメント** -- https://typicode.github.io/husky/ -- Git フックの管理ツール
4. **Biome 公式** -- https://biomejs.dev/ -- Rust 製の高速 Linter/Formatter ツール
5. **ESLint v9 Flat Config** -- https://eslint.org/docs/latest/use/configure/configuration-files -- ESLint の新しい設定形式
6. **Renovate 公式ドキュメント** -- https://docs.renovatebot.com/ -- 依存関係自動更新ツール
7. **typescript-eslint** -- https://typescript-eslint.io/ -- TypeScript 向け ESLint プラグイン
8. **fnm (Fast Node Manager)** -- https://github.com/Schniz/fnm -- Rust 製の高速な Node.js バージョン管理ツール
9. **mise** -- https://mise.jdx.dev/ -- 多言語対応のバージョン管理ツール (旧 rtx)
10. **lefthook** -- https://github.com/evilmartians/lefthook -- Go 製の高速 Git フック管理ツール
