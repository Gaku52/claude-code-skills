# VS Code セットアップ

> Visual Studio Code のインストールから実践的なカスタマイズまで、開発生産性を最大化するための完全ガイド。

## この章で学ぶこと

1. VS Code のインストール・初期設定・設定同期を正しく構成する方法
2. 開発効率を飛躍的に向上させる拡張機能の選定と管理手法
3. マルチカーソル・スニペット・キーバインドを使いこなす実践テクニック
4. ワークスペース設定を活用したチーム統一環境の構築方法
5. デバッグ・タスク・リモート開発の実践的な構成手法
6. パフォーマンス最適化とトラブルシューティングの体系的アプローチ

---

## 1. インストールと初期設定

### 1.1 プラットフォーム別インストール

```bash
# macOS (Homebrew)
brew install --cask visual-studio-code

# macOS (手動ダウンロード後、CLI でバージョン確認)
# https://code.visualstudio.com/download からダウンロード
# /Applications/Visual Studio Code.app にドラッグ&ドロップ

# Ubuntu/Debian
sudo apt install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update && sudo apt install code

# Fedora/RHEL
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
sudo dnf install code

# Windows (winget)
winget install Microsoft.VisualStudioCode

# Windows (Chocolatey)
choco install vscode

# Windows (Scoop)
scoop bucket add extras
scoop install vscode
```

### 1.2 CLI ツール `code` の有効化

macOS では手動でパスを通す必要がある場合がある。

```
コマンドパレット (Cmd+Shift+P)
  → "Shell Command: Install 'code' command in PATH"
    → 完了
```

確認:

```bash
code --version
# 1.96.2
# e54c774e0add60467559eb0d1e229c6452cf8447
# arm64
```

CLI の便利な使い方:

```bash
# カレントディレクトリを VS Code で開く
code .

# 特定ファイルを開く
code src/index.ts

# 差分表示
code --diff file1.ts file2.ts

# 特定の行にジャンプして開く
code --goto src/app.ts:42:10

# 拡張機能のインストール
code --install-extension dbaeumer.vscode-eslint

# 拡張機能の一覧表示
code --list-extensions

# 拡張機能のアンインストール
code --uninstall-extension <extension-id>

# 新しいウィンドウで開く
code --new-window .

# 既存ウィンドウに追加
code --add /path/to/another/folder

# ユーザーデータディレクトリを指定して起動（ポータブル運用）
code --user-data-dir /path/to/portable-data .

# 拡張機能を無効にして起動（トラブルシュート）
code --disable-extensions .

# 拡張機能のバージョン指定インストール
code --install-extension dbaeumer.vscode-eslint@2.4.0

# VS Code のログレベル指定
code --log trace
```

### 1.3 アーキテクチャ概要

```
+--------------------------------------------------+
|                VS Code アーキテクチャ               |
+--------------------------------------------------+
|  UI Layer (Electron)                              |
|  +--------------------------------------------+  |
|  |  Editor        | Activity Bar | Side Bar   |  |
|  |  (Monaco)      | (Icons)     | (Explorer)  |  |
|  +--------------------------------------------+  |
|  |  Status Bar    | Panel (Terminal/Output)    |  |
|  +--------------------------------------------+  |
|                                                    |
|  Extension Host (Node.js プロセス)                 |
|  +--------------------------------------------+  |
|  |  Language Server Protocol (LSP)             |  |
|  |  Debug Adapter Protocol (DAP)               |  |
|  |  拡張機能 API                                |  |
|  +--------------------------------------------+  |
|                                                    |
|  Workspace Storage / Settings                     |
|  +--------------------------------------------+  |
|  |  User Settings    | Workspace Settings     |  |
|  |  (~/.config/Code) | (.vscode/settings.json)|  |
|  +--------------------------------------------+  |
+--------------------------------------------------+

プロセスモデル:
┌──────────────────────────────────────────────────┐
│  Main Process (Electron)                          │
│  ├── Renderer Process (UI / Monaco Editor)        │
│  ├── Extension Host Process (Node.js)             │
│  │   ├── Language Extension (TypeScript, Python)  │
│  │   ├── Linter Extension (ESLint)                │
│  │   └── Theme Extension                          │
│  ├── Shared Process (拡張機能管理, Settings Sync) │
│  ├── File Watcher Process (chokidar)              │
│  ├── Search Process (ripgrep)                     │
│  └── Terminal Process (pty)                       │
└──────────────────────────────────────────────────┘
```

### 1.4 初回起動時にやるべきこと

```
VS Code 初回セットアップチェックリスト:

□ 1. Settings Sync を有効化（GitHub アカウント連携）
□ 2. カラーテーマの選択
     コマンドパレット → "Preferences: Color Theme"
□ 3. フォントのインストール・設定
     - JetBrains Mono: https://www.jetbrains.com/lp/mono/
     - Fira Code: https://github.com/tonsky/FiraCode
□ 4. 日本語 UI（必要な場合）
     拡張機能: MS-CEINTL.vscode-language-pack-ja
□ 5. デフォルトターミナルシェルの設定
□ 6. code CLI コマンドのパス登録（macOS）
□ 7. プロジェクト固有拡張機能のインストール
□ 8. .vscode フォルダの確認・設定
```

---

## 2. 設定ファイル（settings.json）

### 2.1 推奨初期設定（詳細版）

```jsonc
// .vscode/settings.json (ワークスペース設定)
{
  // ===================================
  // エディタ基本設定
  // ===================================
  "editor.fontSize": 14,
  "editor.fontFamily": "'JetBrains Mono', 'Fira Code', Menlo, monospace",
  "editor.fontLigatures": true,
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "editor.renderWhitespace": "boundary",
  "editor.wordWrap": "on",
  "editor.minimap.enabled": false,
  "editor.bracketPairColorization.enabled": true,
  "editor.guides.bracketPairs": "active",
  "editor.stickyScroll.enabled": true,
  "editor.linkedEditing": true,
  "editor.cursorBlinking": "smooth",
  "editor.cursorSmoothCaretAnimation": "on",
  "editor.smoothScrolling": true,
  "editor.suggest.preview": true,
  "editor.suggest.showMethods": true,
  "editor.suggest.showFunctions": true,
  "editor.inlineSuggest.enabled": true,
  "editor.parameterHints.enabled": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.rulers": [80, 120],
  "editor.renderLineHighlight": "all",
  "editor.occurrencesHighlight": "singleFile",
  "editor.unicodeHighlight.ambiguousCharacters": true,

  // ===================================
  // 保存時の自動処理
  // ===================================
  "editor.formatOnSave": true,
  "editor.formatOnPaste": false,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
    "source.organizeImports": "explicit",
    "source.removeUnusedImports": "explicit"
  },

  // ===================================
  // ファイル設定
  // ===================================
  "files.autoSave": "onFocusChange",
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.trimFinalNewlines": true,
  "files.encoding": "utf8",
  "files.eol": "\n",
  "files.exclude": {
    "**/.git": true,
    "**/node_modules": true,
    "**/.DS_Store": true,
    "**/*.pyc": true,
    "**/__pycache__": true
  },
  "files.associations": {
    "*.env.*": "dotenv",
    "*.css": "css",
    "*.mdx": "mdx",
    "Dockerfile.*": "dockerfile"
  },
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/dist/**": true,
    "**/.next/**": true
  },

  // ===================================
  // ターミナル設定
  // ===================================
  "terminal.integrated.fontSize": 13,
  "terminal.integrated.fontFamily": "'JetBrains Mono', 'MesloLGS NF', monospace",
  "terminal.integrated.defaultProfile.osx": "zsh",
  "terminal.integrated.scrollback": 10000,
  "terminal.integrated.copyOnSelection": true,
  "terminal.integrated.cursorBlinking": true,
  "terminal.integrated.env.osx": {
    "EDITOR": "code --wait"
  },

  // ===================================
  // 検索設定
  // ===================================
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/.next": true,
    "**/coverage": true,
    "**/build": true,
    "**/.turbo": true,
    "**/package-lock.json": true,
    "**/yarn.lock": true,
    "**/pnpm-lock.yaml": true
  },
  "search.useIgnoreFiles": true,
  "search.smartCase": true,

  // ===================================
  // Explorer 設定
  // ===================================
  "explorer.confirmDelete": false,
  "explorer.confirmDragAndDrop": false,
  "explorer.compactFolders": true,
  "explorer.fileNesting.enabled": true,
  "explorer.fileNesting.expand": false,
  "explorer.fileNesting.patterns": {
    "*.ts": "${capture}.test.ts, ${capture}.spec.ts, ${capture}.d.ts",
    "*.tsx": "${capture}.test.tsx, ${capture}.spec.tsx, ${capture}.stories.tsx",
    "package.json": "package-lock.json, yarn.lock, pnpm-lock.yaml, .npmrc, .yarnrc.yml",
    "tsconfig.json": "tsconfig.*.json",
    ".eslintrc.*": ".eslintignore, .prettierrc*, .prettierignore",
    "tailwind.config.*": "postcss.config.*"
  },

  // ===================================
  // 言語固有設定
  // ===================================
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.tabSize": 4,
    "editor.formatOnSave": true
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.tabSize": 2
  },
  "[jsonc]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[markdown]": {
    "editor.wordWrap": "on",
    "editor.quickSuggestions": {
      "other": false,
      "comments": false,
      "strings": false
    },
    "files.trimTrailingWhitespace": false
  },
  "[yaml]": {
    "editor.tabSize": 2,
    "editor.autoIndent": "advanced"
  },
  "[go]": {
    "editor.defaultFormatter": "golang.go",
    "editor.tabSize": 4,
    "editor.insertSpaces": false
  },
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.tabSize": 4
  },
  "[css]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[html]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[dockerfile]": {
    "editor.defaultFormatter": "ms-azuretools.vscode-docker"
  },
  "[shellscript]": {
    "editor.tabSize": 2,
    "editor.defaultFormatter": "foxundermoon.shell-format"
  },

  // ===================================
  // Git 設定
  // ===================================
  "git.autofetch": true,
  "git.confirmSync": false,
  "git.enableSmartCommit": true,
  "git.openRepositoryInParentFolders": "always",
  "diffEditor.ignoreTrimWhitespace": false,

  // ===================================
  // ワークベンチ設定
  // ===================================
  "workbench.startupEditor": "none",
  "workbench.editor.enablePreview": true,
  "workbench.editor.closeOnFileDelete": true,
  "workbench.tree.indent": 16,
  "workbench.iconTheme": "material-icon-theme",
  "workbench.colorTheme": "One Dark Pro",
  "workbench.editor.labelFormat": "short",
  "workbench.editor.tabSizing": "shrink",

  // ===================================
  // Breadcrumbs（パンくずリスト）
  // ===================================
  "breadcrumbs.enabled": true,
  "breadcrumbs.filePath": "on",
  "breadcrumbs.symbolPath": "on"
}
```

### 2.2 設定の優先順位

```
高 ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ → 低

  Workspace      >    User       >   Default
  Folder              Settings       Settings
  (.vscode/           (~/.config/
   settings.json)      Code/User/
                       settings.json)

※ Workspace 設定が User 設定を上書きする
※ マルチルートワークスペースでは Folder 設定が最優先
※ 言語固有設定（[typescript] など）は同レベルの汎用設定より優先

詳細な優先順位チェーン:
  1. 言語固有ワークスペースフォルダ設定
  2. ワークスペースフォルダ設定
  3. 言語固有ワークスペース設定
  4. ワークスペース設定
  5. 言語固有ユーザー設定
  6. ユーザー設定
  7. デフォルト設定
```

### 2.3 マルチルートワークスペース設定

```jsonc
// workspace.code-workspace（マルチルート構成ファイル）
{
  "folders": [
    {
      "name": "Frontend",
      "path": "./packages/frontend"
    },
    {
      "name": "Backend API",
      "path": "./packages/api"
    },
    {
      "name": "Shared Library",
      "path": "./packages/shared"
    },
    {
      "name": "Infrastructure",
      "path": "./infra"
    }
  ],
  "settings": {
    // ワークスペース全体に適用される設定
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "search.exclude": {
      "**/node_modules": true,
      "**/dist": true,
      "**/coverage": true
    },
    // フォルダ固有の設定は各フォルダの .vscode/settings.json に記述
    "files.exclude": {
      "**/.git": true,
      "**/node_modules": true
    }
  },
  "extensions": {
    "recommendations": [
      "dbaeumer.vscode-eslint",
      "esbenp.prettier-vscode"
    ]
  }
}
```

### 2.4 設定のエクスポート・インポート

```bash
# 現在の設定を確認
cat ~/Library/Application\ Support/Code/User/settings.json

# 拡張機能一覧をエクスポート
code --list-extensions > vscode-extensions.txt

# 拡張機能を一括インストール
cat vscode-extensions.txt | xargs -L 1 code --install-extension

# プロファイルのエクスポート（CLI）
# コマンドパレット → "Profiles: Export Profile" でも可能
```

---

## 3. 推奨拡張機能リスト

### 3.1 カテゴリ別一覧（詳細版）

| カテゴリ | 拡張機能 | ID | 用途 |
|---------|---------|-----|------|
| **言語サポート** | ESLint | `dbaeumer.vscode-eslint` | JavaScript/TypeScript リント |
| | Prettier | `esbenp.prettier-vscode` | コードフォーマッタ |
| | Python | `ms-python.python` | Python 言語サポート |
| | Pylance | `ms-python.vscode-pylance` | Python 高速型チェック |
| | Black Formatter | `ms-python.black-formatter` | Python フォーマッタ |
| | Rust Analyzer | `rust-lang.rust-analyzer` | Rust 言語サポート |
| | Go | `golang.go` | Go 言語サポート |
| | C/C++ | `ms-vscode.cpptools` | C/C++ 言語サポート |
| **TypeScript** | Pretty TypeScript Errors | `yoavbls.pretty-ts-errors` | TS エラーを読みやすく |
| | TypeScript Importer | `pmneo.tsimporter` | 自動 import |
| | Total TypeScript | `mattpocock.ts-error-translator` | TSエラー日本語化 |
| **Git** | GitLens | `eamodio.gitlens` | Git 履歴・blame 表示 |
| | Git Graph | `mhutchie.git-graph` | ブランチグラフ可視化 |
| | Conventional Commits | `vivaxy.vscode-conventional-commits` | コミットメッセージ補助 |
| **AI** | GitHub Copilot | `github.copilot` | AI コード補完 |
| | GitHub Copilot Chat | `github.copilot-chat` | AI チャット |
| **開発効率** | Error Lens | `usernamehw.errorlens` | インラインエラー表示 |
| | TODO Highlight | `wayou.vscode-todo-highlight` | TODO/FIXME ハイライト |
| | Path Intellisense | `christian-kohler.path-intellisense` | パス自動補完 |
| | Auto Rename Tag | `formulahendry.auto-rename-tag` | HTML/JSX タグ連動変更 |
| | Code Spell Checker | `streetsidesoftware.code-spell-checker` | スペルチェック |
| | Better Comments | `aaron-bond.better-comments` | コメント色分け |
| | Bookmarks | `alefragnani.bookmarks` | コードにブックマーク |
| | Import Cost | `wix.vscode-import-cost` | import サイズ表示 |
| **テスト** | Jest Runner | `firsttris.vscode-jest-runner` | Jest テスト実行 |
| | Vitest | `vitest.explorer` | Vitest テスト実行 |
| | Test Explorer UI | `hbenl.vscode-test-explorer` | テスト統合 UI |
| **外観** | Material Icon Theme | `pkief.material-icon-theme` | ファイルアイコン |
| | One Dark Pro | `zhuangtongfa.material-theme` | カラーテーマ |
| | GitHub Theme | `github.github-vscode-theme` | GitHub 風テーマ |
| | Catppuccin | `catppuccin.catppuccin-vsc` | パステルテーマ |
| | Indent Rainbow | `oderwat.indent-rainbow` | インデント可視化 |
| **コンテナ** | Dev Containers | `ms-vscode-remote.remote-containers` | Docker 開発環境 |
| | Docker | `ms-azuretools.vscode-docker` | Docker 管理 |
| **リモート** | Remote - SSH | `ms-vscode-remote.remote-ssh` | SSH リモート開発 |
| | Remote - WSL | `ms-vscode-remote.remote-wsl` | WSL 連携 |
| | Remote - Tunnels | `ms-vscode.remote-server` | トンネル接続 |
| **データ** | Thunder Client | `rangav.vscode-thunder-client` | REST API クライアント |
| | Database Client | `cweijan.vscode-database-client2` | DB クライアント |
| | YAML | `redhat.vscode-yaml` | YAML バリデーション |
| | DotENV | `mikestead.dotenv` | .env ハイライト |
| **Markdown** | Markdown All in One | `yzhang.markdown-all-in-one` | Markdown 拡張 |
| | Markdown Preview Enhanced | `shd101wyy.markdown-preview-enhanced` | Markdown プレビュー |
| | Mermaid Preview | `bierner.markdown-mermaid` | Mermaid 図表プレビュー |
| **CSS/HTML** | Tailwind CSS IntelliSense | `bradlc.vscode-tailwindcss` | Tailwind 補完 |
| | CSS Peek | `pranaygp.vscode-css-peek` | CSS 定義へジャンプ |
| | HTML CSS Support | `ecmel.vscode-html-css` | HTML/CSS 補完 |

### 3.2 拡張機能の一括インストール

```bash
# プロジェクトで推奨する拡張機能を一括インストール
cat extensions.txt | xargs -L 1 code --install-extension

# extensions.txt の例:
# dbaeumer.vscode-eslint
# esbenp.prettier-vscode
# eamodio.gitlens
# github.copilot
# usernamehw.errorlens
# pkief.material-icon-theme

# プロジェクト種別ごとのセットアップスクリプト
# setup-vscode-extensions.sh
#!/bin/bash

COMMON_EXTENSIONS=(
  "dbaeumer.vscode-eslint"
  "esbenp.prettier-vscode"
  "eamodio.gitlens"
  "mhutchie.git-graph"
  "usernamehw.errorlens"
  "pkief.material-icon-theme"
  "streetsidesoftware.code-spell-checker"
  "github.copilot"
  "github.copilot-chat"
  "aaron-bond.better-comments"
)

FRONTEND_EXTENSIONS=(
  "bradlc.vscode-tailwindcss"
  "formulahendry.auto-rename-tag"
  "yoavbls.pretty-ts-errors"
  "wix.vscode-import-cost"
  "vitest.explorer"
  "styled-components.vscode-styled-components"
)

BACKEND_EXTENSIONS=(
  "ms-python.python"
  "ms-python.vscode-pylance"
  "ms-python.black-formatter"
  "golang.go"
  "rangav.vscode-thunder-client"
  "cweijan.vscode-database-client2"
)

DEVOPS_EXTENSIONS=(
  "ms-vscode-remote.remote-containers"
  "ms-azuretools.vscode-docker"
  "hashicorp.terraform"
  "ms-kubernetes-tools.vscode-kubernetes-tools"
  "redhat.vscode-yaml"
)

install_extensions() {
  local extensions=("$@")
  for ext in "${extensions[@]}"; do
    echo "Installing: $ext"
    code --install-extension "$ext" --force
  done
}

echo "=== Installing Common Extensions ==="
install_extensions "${COMMON_EXTENSIONS[@]}"

case "$1" in
  "frontend")
    echo "=== Installing Frontend Extensions ==="
    install_extensions "${FRONTEND_EXTENSIONS[@]}"
    ;;
  "backend")
    echo "=== Installing Backend Extensions ==="
    install_extensions "${BACKEND_EXTENSIONS[@]}"
    ;;
  "devops")
    echo "=== Installing DevOps Extensions ==="
    install_extensions "${DEVOPS_EXTENSIONS[@]}"
    ;;
  "all")
    echo "=== Installing All Extensions ==="
    install_extensions "${FRONTEND_EXTENSIONS[@]}"
    install_extensions "${BACKEND_EXTENSIONS[@]}"
    install_extensions "${DEVOPS_EXTENSIONS[@]}"
    ;;
  *)
    echo "Usage: $0 {frontend|backend|devops|all}"
    ;;
esac
```

### 3.3 ワークスペース推奨設定

```jsonc
// .vscode/extensions.json
{
  "recommendations": [
    // 必須
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "eamodio.gitlens",
    "github.copilot",
    "usernamehw.errorlens",
    // フロントエンド
    "bradlc.vscode-tailwindcss",
    "formulahendry.auto-rename-tag",
    "yoavbls.pretty-ts-errors",
    // テスト
    "vitest.explorer"
  ],
  "unwantedRecommendations": [
    "hookyqr.beautify",   // Prettier と競合
    "esbenp.prettier-vscode" // Python プロジェクトの場合
  ]
}
```

### 3.4 拡張機能の競合と解決策

```
よくある拡張機能の競合パターン:

1. フォーマッタの競合
   ❌ Prettier + Beautify が同時に有効
   ✅ defaultFormatter で明示的に指定
   "[typescript]": {
     "editor.defaultFormatter": "esbenp.prettier-vscode"
   }

2. Linter の競合
   ❌ ESLint + TSLint が同時に動作
   ✅ TSLint は非推奨。ESLint に統一。

3. IntelliSense の競合
   ❌ Tabnine + Copilot + IntelliCode が同時有効
   ✅ AI 補完は1つに絞る（推奨: Copilot）

4. Git 拡張の競合
   ❌ GitLens + Git History + Git Blame が同時に動作
   ✅ GitLens 1つで網羅可能。他は無効化。

5. 括弧ペアの色分け
   ❌ Bracket Pair Colorizer 拡張（非推奨）
   ✅ VS Code ネイティブ機能を使用
   "editor.bracketPairColorization.enabled": true
```

---

## 4. キーバインド

### 4.1 必須ショートカット一覧

| 操作 | macOS | Windows/Linux |
|------|-------|---------------|
| コマンドパレット | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| ファイル検索 | `Cmd+P` | `Ctrl+P` |
| シンボル検索（ワークスペース） | `Cmd+T` | `Ctrl+T` |
| シンボル検索（ファイル内） | `Cmd+Shift+O` | `Ctrl+Shift+O` |
| 全文検索 | `Cmd+Shift+F` | `Ctrl+Shift+F` |
| 全文置換 | `Cmd+Shift+H` | `Ctrl+Shift+H` |
| ターミナル切替 | `` Ctrl+` `` | `` Ctrl+` `` |
| 新しいターミナル | `` Ctrl+Shift+` `` | `` Ctrl+Shift+` `` |
| サイドバー切替 | `Cmd+B` | `Ctrl+B` |
| パネル切替 | `Cmd+J` | `Ctrl+J` |
| 行の移動 | `Alt+Up/Down` | `Alt+Up/Down` |
| 行の複製 | `Shift+Alt+Up/Down` | `Shift+Alt+Up/Down` |
| 行の削除 | `Cmd+Shift+K` | `Ctrl+Shift+K` |
| 定義へ移動 | `F12` | `F12` |
| 定義をピーク | `Alt+F12` | `Alt+F12` |
| 型定義へ移動 | `Cmd+Click` | `Ctrl+Click` |
| 参照を検索 | `Shift+F12` | `Shift+F12` |
| リネーム | `F2` | `F2` |
| クイックフィックス | `Cmd+.` | `Ctrl+.` |
| ファイルを閉じる | `Cmd+W` | `Ctrl+W` |
| すべてのファイルを閉じる | `Cmd+K Cmd+W` | `Ctrl+K Ctrl+W` |
| 分割エディタ | `Cmd+\` | `Ctrl+\` |
| エディタグループ切替 | `Cmd+1/2/3` | `Ctrl+1/2/3` |
| 折りたたみ | `Cmd+Shift+[` | `Ctrl+Shift+[` |
| 展開 | `Cmd+Shift+]` | `Ctrl+Shift+]` |
| コメントトグル | `Cmd+/` | `Ctrl+/` |
| ブロックコメント | `Shift+Alt+A` | `Shift+Alt+A` |
| インデント追加 | `Cmd+]` | `Ctrl+]` |
| インデント削除 | `Cmd+[` | `Ctrl+[` |
| 前のエディタに戻る | `Ctrl+-` | `Alt+Left` |
| 次のエディタに進む | `Ctrl+Shift+-` | `Alt+Right` |
| Zen Mode | `Cmd+K Z` | `Ctrl+K Z` |

### 4.2 カスタムキーバインド

```jsonc
// keybindings.json
[
  // --- 基本編集 ---
  {
    "key": "cmd+shift+d",
    "command": "editor.action.copyLinesDownAction",
    "when": "editorTextFocus"
  },
  {
    "key": "cmd+shift+k",
    "command": "editor.action.deleteLines",
    "when": "editorTextFocus"
  },
  {
    "key": "cmd+shift+l",
    "command": "editor.action.selectHighlights",
    "when": "editorTextFocus"
  },

  // --- ファイル操作 ---
  {
    "key": "cmd+k cmd+o",
    "command": "workbench.action.openRecent"
  },

  // --- ターミナル ---
  {
    "key": "cmd+shift+t",
    "command": "workbench.action.terminal.new"
  },
  {
    "key": "cmd+shift+[",
    "command": "workbench.action.terminal.focusPrevious",
    "when": "terminalFocus"
  },
  {
    "key": "cmd+shift+]",
    "command": "workbench.action.terminal.focusNext",
    "when": "terminalFocus"
  },

  // --- パネル操作 ---
  {
    "key": "cmd+shift+m",
    "command": "workbench.actions.view.problems"
  },
  {
    "key": "cmd+shift+u",
    "command": "workbench.action.output.toggleOutput"
  },

  // --- Git 操作 ---
  {
    "key": "cmd+shift+g cmd+shift+g",
    "command": "workbench.view.scm"
  },

  // --- テスト実行（Jest/Vitest） ---
  {
    "key": "cmd+shift+r",
    "command": "testing.runAtCursor",
    "when": "editorTextFocus"
  },
  {
    "key": "cmd+shift+e",
    "command": "testing.debugAtCursor",
    "when": "editorTextFocus"
  },

  // --- エクスプローラーでファイルを表示 ---
  {
    "key": "cmd+shift+1",
    "command": "revealFileInOS"
  },

  // --- マルチカーソル追加操作 ---
  {
    "key": "cmd+alt+up",
    "command": "editor.action.insertCursorAbove",
    "when": "editorTextFocus"
  },
  {
    "key": "cmd+alt+down",
    "command": "editor.action.insertCursorBelow",
    "when": "editorTextFocus"
  }
]
```

### 4.3 Vim キーバインドの導入

```jsonc
// Vim 拡張機能（vscodevim.vim）を使用する場合の設定
// settings.json に追加
{
  "vim.enable": true,
  "vim.leader": "<space>",
  "vim.useSystemClipboard": true,
  "vim.useCtrlKeys": true,
  "vim.handleKeys": {
    "<C-a>": false,   // VS Code の「全選択」を維持
    "<C-f>": false,   // VS Code の「検索」を維持
    "<C-c>": false,   // VS Code の「コピー」を維持
    "<C-v>": false,   // VS Code の「ペースト」を維持
    "<C-x>": false,   // VS Code の「切り取り」を維持
    "<C-z>": false,   // VS Code の「元に戻す」を維持
    "<C-shift-p>": false  // コマンドパレットを維持
  },
  "vim.normalModeKeyBindingsNonRecursive": [
    // <space>f でファイル検索
    { "before": ["<leader>", "f"], "commands": ["workbench.action.quickOpen"] },
    // <space>g で全文検索
    { "before": ["<leader>", "g"], "commands": ["workbench.action.findInFiles"] },
    // <space>e でエクスプローラー切替
    { "before": ["<leader>", "e"], "commands": ["workbench.view.explorer"] },
    // <space>w で保存
    { "before": ["<leader>", "w"], "commands": [":w"] },
    // <space>q で閉じる
    { "before": ["<leader>", "q"], "commands": [":q"] },
    // gh で定義をホバー表示
    { "before": ["g", "h"], "commands": ["editor.action.showHover"] },
    // gd で定義へジャンプ
    { "before": ["g", "d"], "commands": ["editor.action.revealDefinition"] },
    // gr で参照検索
    { "before": ["g", "r"], "commands": ["editor.action.goToReferences"] }
  ],
  "vim.visualModeKeyBindingsNonRecursive": [
    // > でインデント（ビジュアルモード維持）
    { "before": [">"], "commands": ["editor.action.indentLines"] },
    { "before": ["<"], "commands": ["editor.action.outdentLines"] }
  ]
}
```

---

## 5. マルチカーソル

### 5.1 基本操作

```
テキスト例:
  const firstName = "Alice";
  const lastName = "Bob";
  const nickName = "Charlie";

操作: Cmd+D で "Name" を連続選択 → 3箇所同時編集

  1. カーソルを "firstName" の "Name" に置く
  2. Cmd+D → "lastName" の "Name" も選択
  3. Cmd+D → "nickName" の "Name" も選択
  4. 一括で編集: "Name" → "Label"

結果:
  const firstLabel = "Alice";
  const lastLabel = "Bob";
  const nickLabel = "Charlie";
```

### 5.2 矩形選択

```
操作フロー:

  Step 1: Alt+Shift+ドラッグ で矩形選択
  ┌─────────────────────────┐
  │ item1: "apple"          │
  │ item2: "banana"    ←──── 縦にカーソル追加
  │ item3: "cherry"         │
  └─────────────────────────┘

  Step 2: 同時に入力
  ┌─────────────────────────┐
  │ item1: "fresh_apple"    │
  │ item2: "fresh_banana"   │
  │ item3: "fresh_cherry"   │
  └─────────────────────────┘
```

### 5.3 実践的なマルチカーソル活用例

```
例1: JSON のキーを一括変更

変更前:
{
  "user_name": "Alice",
  "user_age": 30,
  "user_email": "alice@example.com",
  "user_phone": "090-1234-5678"
}

操作:
  1. "user_" を選択
  2. Cmd+Shift+L → すべての "user_" を一括選択
  3. 全て削除して空にする

変更後:
{
  "name": "Alice",
  "age": 30,
  "email": "alice@example.com",
  "phone": "090-1234-5678"
}

─────────────────────────────────

例2: 配列要素をオブジェクトに変換

変更前:
const colors = [
  "red",
  "green",
  "blue",
  "yellow",
];

操作:
  1. 各行の先頭 " にカーソルを置く（Alt+Click で追加）
  2. "{ name: " と入力
  3. End キーで行末に移動
  4. ", value: '#000' }" と入力

変更後:
const colors = [
  { name: "red", value: '#000' },
  { name: "green", value: '#000' },
  { name: "blue", value: '#000' },
  { name: "yellow", value: '#000' },
];

─────────────────────────────────

例3: CSS プロパティの一括追加

変更前:
.card {
  border-radius: 8px;
}
.button {
  border-radius: 4px;
}
.input {
  border-radius: 6px;
}

操作:
  1. Cmd+Shift+F で "border-radius" を検索
  2. 各行末にカーソルを追加
  3. Enter して新しい行で "overflow: hidden;" を入力
```

---

## 6. スニペット

### 6.1 ユーザースニペット定義（拡充版）

```jsonc
// .vscode/project.code-snippets
{
  // ===================================
  // React コンポーネント
  // ===================================
  "React Functional Component": {
    "prefix": "rfc",
    "scope": "typescriptreact",
    "body": [
      "type ${1:Component}Props = {",
      "  $2",
      "};",
      "",
      "export function ${1:Component}({ $3 }: ${1:Component}Props) {",
      "  return (",
      "    <div>",
      "      $0",
      "    </div>",
      "  );",
      "}"
    ],
    "description": "React Functional Component with TypeScript"
  },
  "React Component with useState": {
    "prefix": "rfcs",
    "scope": "typescriptreact",
    "body": [
      "import { useState } from 'react';",
      "",
      "type ${1:Component}Props = {",
      "  $2",
      "};",
      "",
      "export function ${1:Component}({ $3 }: ${1:Component}Props) {",
      "  const [${4:state}, set${4/(.*)/${1:/capitalize}/}] = useState<${5:string}>($6);",
      "",
      "  return (",
      "    <div>",
      "      $0",
      "    </div>",
      "  );",
      "}"
    ],
    "description": "React Component with useState hook"
  },
  "React Custom Hook": {
    "prefix": "rhook",
    "scope": "typescript,typescriptreact",
    "body": [
      "import { useState, useEffect } from 'react';",
      "",
      "export function use${1:Hook}(${2:params}: ${3:ParamType}) {",
      "  const [${4:data}, set${4/(.*)/${1:/capitalize}/}] = useState<${5:DataType}>();",
      "  const [isLoading, setIsLoading] = useState(false);",
      "  const [error, setError] = useState<Error | null>(null);",
      "",
      "  useEffect(() => {",
      "    $0",
      "  }, [${2:params}]);",
      "",
      "  return { ${4:data}, isLoading, error };",
      "}"
    ],
    "description": "React Custom Hook"
  },

  // ===================================
  // テスト
  // ===================================
  "Describe Block": {
    "prefix": "desc",
    "scope": "typescript,typescriptreact,javascript",
    "body": [
      "describe('${1:subject}', () => {",
      "  $0",
      "});"
    ],
    "description": "Jest/Vitest describe block"
  },
  "Test Case": {
    "prefix": "it",
    "scope": "typescript,typescriptreact,javascript",
    "body": [
      "it('should ${1:description}', () => {",
      "  // Arrange",
      "  $2",
      "",
      "  // Act",
      "  $3",
      "",
      "  // Assert",
      "  expect($4).${5:toBe}($6);",
      "});"
    ],
    "description": "Test case with AAA pattern"
  },
  "Async Test Case": {
    "prefix": "ita",
    "scope": "typescript,typescriptreact,javascript",
    "body": [
      "it('should ${1:description}', async () => {",
      "  // Arrange",
      "  $2",
      "",
      "  // Act",
      "  const result = await ${3:asyncFunction}();",
      "",
      "  // Assert",
      "  expect(result).${4:toBe}($5);",
      "});"
    ],
    "description": "Async test case"
  },

  // ===================================
  // 一般的なパターン
  // ===================================
  "Console Log Variable": {
    "prefix": "clv",
    "scope": "javascript,typescript,typescriptreact,javascriptreact",
    "body": [
      "console.log('${1:variable}:', ${1:variable});"
    ],
    "description": "Console log with variable name"
  },
  "Console Log JSON": {
    "prefix": "clj",
    "scope": "javascript,typescript,typescriptreact,javascriptreact",
    "body": [
      "console.log('${1:variable}:', JSON.stringify(${1:variable}, null, 2));"
    ],
    "description": "Console log with JSON stringify"
  },
  "Try-Catch Block": {
    "prefix": "trycatch",
    "scope": "typescript,javascript",
    "body": [
      "try {",
      "  $1",
      "} catch (error) {",
      "  if (error instanceof Error) {",
      "    console.error(error.message);",
      "  }",
      "  throw error;",
      "}"
    ],
    "description": "Try-catch with type guard"
  },
  "Async Arrow Function": {
    "prefix": "aaf",
    "scope": "typescript,javascript,typescriptreact",
    "body": [
      "const ${1:functionName} = async (${2:params}: ${3:ParamType}): Promise<${4:ReturnType}> => {",
      "  $0",
      "};"
    ],
    "description": "Async arrow function with types"
  },
  "Zod Schema": {
    "prefix": "zod",
    "scope": "typescript,typescriptreact",
    "body": [
      "import { z } from 'zod';",
      "",
      "export const ${1:name}Schema = z.object({",
      "  ${2:field}: z.${3:string}(),",
      "  $0",
      "});",
      "",
      "export type ${1:name} = z.infer<typeof ${1:name}Schema>;"
    ],
    "description": "Zod schema definition"
  },

  // ===================================
  // Next.js
  // ===================================
  "Next.js Page Component": {
    "prefix": "npage",
    "scope": "typescriptreact",
    "body": [
      "export default function ${1:Page}() {",
      "  return (",
      "    <main>",
      "      <h1>${2:Title}</h1>",
      "      $0",
      "    </main>",
      "  );",
      "}"
    ],
    "description": "Next.js App Router page component"
  },
  "Next.js Server Action": {
    "prefix": "naction",
    "scope": "typescript,typescriptreact",
    "body": [
      "'use server';",
      "",
      "export async function ${1:actionName}(formData: FormData) {",
      "  const ${2:field} = formData.get('${2:field}') as string;",
      "",
      "  $0",
      "}"
    ],
    "description": "Next.js Server Action"
  },

  // ===================================
  // ドキュメントコメント
  // ===================================
  "JSDoc Function": {
    "prefix": "jsdoc",
    "scope": "typescript,javascript,typescriptreact",
    "body": [
      "/**",
      " * ${1:Description}",
      " *",
      " * @param ${2:param} - ${3:description}",
      " * @returns ${4:description}",
      " * @throws {${5:Error}} ${6:description}",
      " *",
      " * @example",
      " * ```typescript",
      " * ${7:example}",
      " * ```",
      " */"
    ],
    "description": "JSDoc function documentation"
  }
}
```

### 6.2 スニペット変数リファレンス

```
VS Code スニペットで使える組み込み変数:

ファイル関連:
  $TM_FILENAME          → "index.ts"
  $TM_FILENAME_BASE     → "index"
  $TM_DIRECTORY         → "/Users/gaku/project/src"
  $TM_FILEPATH          → "/Users/gaku/project/src/index.ts"
  $RELATIVE_FILEPATH    → "src/index.ts"
  $WORKSPACE_NAME       → "my-project"
  $WORKSPACE_FOLDER     → "/Users/gaku/project"

日付・時刻:
  $CURRENT_YEAR         → "2026"
  $CURRENT_MONTH        → "02"
  $CURRENT_DATE         → "15"
  $CURRENT_HOUR         → "14"
  $CURRENT_MINUTE       → "30"
  $CURRENT_SECOND       → "00"

その他:
  $CLIPBOARD            → クリップボードの内容
  $LINE_COMMENT         → 言語のラインコメント（// や #）
  $BLOCK_COMMENT_START  → 言語のブロックコメント開始（/* や <!--）
  $BLOCK_COMMENT_END    → 言語のブロックコメント終了（*/ や -->）
  $UUID                 → UUID v4 生成
  $RANDOM               → 6桁のランダム整数
  $RANDOM_HEX           → 6桁のランダム hex

変換（Transform）:
  ${1/(.*)/${1:/upcase}/}      → 大文字変換
  ${1/(.*)/${1:/downcase}/}    → 小文字変換
  ${1/(.*)/${1:/capitalize}/}  → 先頭大文字
  ${1/(.*)/${1:/pascalcase}/}  → PascalCase
  ${1/(.*)/${1:/camelcase}/}   → camelCase

プレースホルダ:
  $1, $2, ...           → タブストップ（入力順）
  $0                    → 最終カーソル位置
  ${1:default}          → デフォルト値付きタブストップ
  ${1|one,two,three|}   → 選択肢付きタブストップ
```

---

## 7. デバッグ設定

### 7.1 launch.json の基本構成

```jsonc
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    // ===================================
    // Node.js アプリケーション
    // ===================================
    {
      "name": "Node.js: Current File",
      "type": "node",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "skipFiles": ["<node_internals>/**"]
    },
    {
      "name": "Node.js: ts-node",
      "type": "node",
      "request": "launch",
      "runtimeArgs": ["-r", "ts-node/register"],
      "args": ["${file}"],
      "console": "integratedTerminal",
      "skipFiles": ["<node_internals>/**"],
      "env": {
        "TS_NODE_PROJECT": "${workspaceFolder}/tsconfig.json"
      }
    },

    // ===================================
    // Next.js
    // ===================================
    {
      "name": "Next.js: Dev Server",
      "type": "node",
      "request": "launch",
      "cwd": "${workspaceFolder}",
      "runtimeExecutable": "npx",
      "runtimeArgs": ["next", "dev"],
      "console": "integratedTerminal",
      "serverReadyAction": {
        "pattern": "- Local:.+(https?://.+)",
        "uriFormat": "%s",
        "action": "debugWithChrome"
      }
    },
    {
      "name": "Next.js: Server-side",
      "type": "node",
      "request": "launch",
      "cwd": "${workspaceFolder}",
      "runtimeExecutable": "npx",
      "runtimeArgs": ["next", "dev"],
      "skipFiles": ["<node_internals>/**"],
      "console": "integratedTerminal",
      "env": {
        "NODE_OPTIONS": "--inspect"
      }
    },

    // ===================================
    // テスト
    // ===================================
    {
      "name": "Jest: Current File",
      "type": "node",
      "request": "launch",
      "runtimeExecutable": "npx",
      "runtimeArgs": [
        "jest",
        "--runInBand",
        "--no-coverage",
        "${relativeFile}"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "Vitest: Current File",
      "type": "node",
      "request": "launch",
      "runtimeExecutable": "npx",
      "runtimeArgs": [
        "vitest",
        "run",
        "${relativeFile}"
      ],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },

    // ===================================
    // Python
    // ===================================
    {
      "name": "Python: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Python: FastAPI",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Django",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/manage.py",
      "args": ["runserver", "0.0.0.0:8000"],
      "django": true,
      "console": "integratedTerminal"
    },
    {
      "name": "Python: pytest",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["-xvs", "${file}"],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },

    // ===================================
    // Go
    // ===================================
    {
      "name": "Go: Launch Package",
      "type": "go",
      "request": "launch",
      "mode": "auto",
      "program": "${workspaceFolder}/cmd/server",
      "env": {
        "GO_ENV": "development"
      }
    },
    {
      "name": "Go: Test Current File",
      "type": "go",
      "request": "launch",
      "mode": "test",
      "program": "${file}"
    },

    // ===================================
    // ブラウザ
    // ===================================
    {
      "name": "Chrome: Open URL",
      "type": "chrome",
      "request": "launch",
      "url": "http://localhost:3000",
      "webRoot": "${workspaceFolder}/src",
      "sourceMapPathOverrides": {
        "webpack:///./src/*": "${webRoot}/*"
      }
    },

    // ===================================
    // Docker（リモートアタッチ）
    // ===================================
    {
      "name": "Docker: Attach to Node",
      "type": "node",
      "request": "attach",
      "port": 9229,
      "address": "localhost",
      "localRoot": "${workspaceFolder}",
      "remoteRoot": "/app",
      "restart": true,
      "skipFiles": ["<node_internals>/**"]
    }
  ],

  // ===================================
  // 複合実行（Compound）
  // ===================================
  "compounds": [
    {
      "name": "Full Stack: Frontend + Backend",
      "configurations": [
        "Next.js: Dev Server",
        "Python: FastAPI"
      ],
      "stopAll": true
    }
  ]
}
```

### 7.2 デバッグのテクニック

```
ブレークポイントの種類:

1. 通常ブレークポイント
   - 行番号の左をクリック（赤い丸）
   - F9 でトグル

2. 条件付きブレークポイント
   - ブレークポイントを右クリック → "Edit Breakpoint"
   - 条件式: i > 100 && user.role === "admin"

3. ログポイント（Logpoint）
   - 実行を止めずにログを出力
   - ブレークポイントを右クリック → "Add Logpoint"
   - メッセージ: "User: {user.name}, Count: {count}"
   - ダイヤ型のアイコンで表示

4. ヒットカウントブレークポイント
   - N回目のヒットで停止
   - 条件: "Hit Count" → 100

5. 例外ブレークポイント
   - Debug パネル → Breakpoints → "Caught Exceptions" をチェック
   - 全ての例外で停止

デバッグコンソールの活用:
  - 停止中に変数を評価: user.name
  - メソッド呼び出し: JSON.stringify(data, null, 2)
  - 値の変更: user.name = "NewValue"（注意して使用）
```

---

## 8. タスク設定

### 8.1 tasks.json の構成

```jsonc
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    // ===================================
    // ビルドタスク
    // ===================================
    {
      "label": "Build: TypeScript",
      "type": "shell",
      "command": "npx tsc --build",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "problemMatcher": ["$tsc"],
      "presentation": {
        "reveal": "silent",
        "panel": "shared"
      }
    },
    {
      "label": "Build: Next.js",
      "type": "shell",
      "command": "npx next build",
      "group": "build",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "dedicated"
      }
    },

    // ===================================
    // 開発サーバー
    // ===================================
    {
      "label": "Dev: Frontend",
      "type": "shell",
      "command": "npm run dev",
      "isBackground": true,
      "problemMatcher": {
        "pattern": {
          "regexp": ".",
          "file": 1,
          "location": 2,
          "message": 3
        },
        "background": {
          "activeOnStart": true,
          "beginsPattern": ".",
          "endsPattern": "ready"
        }
      },
      "presentation": {
        "reveal": "always",
        "panel": "dedicated",
        "group": "dev"
      }
    },
    {
      "label": "Dev: Backend",
      "type": "shell",
      "command": "python -m uvicorn main:app --reload",
      "isBackground": true,
      "presentation": {
        "reveal": "always",
        "panel": "dedicated",
        "group": "dev"
      }
    },
    {
      "label": "Dev: Full Stack",
      "dependsOn": ["Dev: Frontend", "Dev: Backend"],
      "dependsOrder": "parallel",
      "problemMatcher": []
    },

    // ===================================
    // テスト
    // ===================================
    {
      "label": "Test: Unit",
      "type": "shell",
      "command": "npx vitest run",
      "group": {
        "kind": "test",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "shared"
      }
    },
    {
      "label": "Test: Watch",
      "type": "shell",
      "command": "npx vitest watch",
      "isBackground": true,
      "presentation": {
        "reveal": "always",
        "panel": "dedicated"
      }
    },
    {
      "label": "Test: E2E",
      "type": "shell",
      "command": "npx playwright test",
      "presentation": {
        "reveal": "always",
        "panel": "dedicated"
      }
    },

    // ===================================
    // Lint & Format
    // ===================================
    {
      "label": "Lint: ESLint Fix",
      "type": "shell",
      "command": "npx eslint . --fix",
      "problemMatcher": ["$eslint-stylish"]
    },
    {
      "label": "Format: Prettier",
      "type": "shell",
      "command": "npx prettier --write .",
      "presentation": {
        "reveal": "silent"
      }
    },
    {
      "label": "Lint: Type Check",
      "type": "shell",
      "command": "npx tsc --noEmit",
      "problemMatcher": ["$tsc"]
    },

    // ===================================
    // Docker
    // ===================================
    {
      "label": "Docker: Up",
      "type": "shell",
      "command": "docker compose up -d",
      "presentation": {
        "reveal": "always"
      }
    },
    {
      "label": "Docker: Down",
      "type": "shell",
      "command": "docker compose down",
      "presentation": {
        "reveal": "always"
      }
    },
    {
      "label": "Docker: Rebuild",
      "type": "shell",
      "command": "docker compose up -d --build",
      "presentation": {
        "reveal": "always"
      }
    },

    // ===================================
    // データベース
    // ===================================
    {
      "label": "DB: Migrate",
      "type": "shell",
      "command": "npx prisma migrate dev",
      "presentation": {
        "reveal": "always"
      }
    },
    {
      "label": "DB: Seed",
      "type": "shell",
      "command": "npx prisma db seed",
      "presentation": {
        "reveal": "always"
      }
    },
    {
      "label": "DB: Studio",
      "type": "shell",
      "command": "npx prisma studio",
      "isBackground": true,
      "presentation": {
        "reveal": "always",
        "panel": "dedicated"
      }
    }
  ]
}
```

### 8.2 タスクの実行方法

```
タスク実行のショートカット:

1. コマンドパレットから:
   Cmd+Shift+P → "Tasks: Run Task" → タスク選択

2. ショートカットキー:
   Cmd+Shift+B → デフォルトビルドタスク実行
   (テストタスクもキーバインド設定可能)

3. ターミナルのタスク表示:
   Terminal パネル → ドロップダウンでタスクごとのターミナル切替

4. 複合タスクの活用:
   dependsOn で複数タスクを並列・逐次実行
   dependsOrder: "parallel" | "sequence"
```

---

## 9. 設定同期

### 9.1 Settings Sync の構成

```
Settings Sync で同期される項目:
┌─────────────────────────────────────┐
│  ✅ Settings (settings.json)         │
│  ✅ Keyboard Shortcuts               │
│  ✅ Extensions                        │
│  ✅ User Snippets                     │
│  ✅ UI State                          │
│                                       │
│  同期方法:                            │
│  ├── GitHub アカウント               │
│  └── Microsoft アカウント            │
│                                       │
│  有効化:                              │
│  Cmd+Shift+P → "Settings Sync: Turn On" │
└─────────────────────────────────────┘

同期の競合解決:
  - 初回同期時に「マージ」or「置換」を選択
  - ローカルとリモートの差分がプレビュー表示される
  - 設定項目ごとに同期の有効/無効を制御可能

同期から除外する設定:
  settings.json に以下を追加:
  "settingsSync.ignoredSettings": [
    "editor.fontSize",     // 個人の好みで異なる
    "terminal.integrated.fontSize",
    "workbench.colorTheme"
  ]
```

### 9.2 プロファイル機能

```bash
# プロファイルの切替で用途別の環境を管理
#
# 例: "Frontend" プロファイル
#   - ESLint, Prettier, Tailwind CSS IntelliSense
#   - React/Vue 拡張
#   - フロントエンド向けテーマ
#
# 例: "Backend" プロファイル
#   - Python, Go, Rust 言語サポート
#   - REST Client, Database Client
#   - シンプルなテーマ
#
# 例: "Python Data Science" プロファイル
#   - Python, Jupyter, Pylance
#   - データ可視化拡張
#   - 分析向け設定
#
# 例: "DevOps" プロファイル
#   - Docker, Kubernetes, Terraform
#   - YAML, Helm
#   - SSH Remote
#
# 例: "Writing" プロファイル
#   - Markdown All in One
#   - Spell Checker
#   - Zen Mode 設定

# CLI でプロファイル指定して起動
code --profile "Frontend" .

# プロファイルの管理
# コマンドパレット → "Profiles: Create Profile"
# コマンドパレット → "Profiles: Switch Profile"
# コマンドパレット → "Profiles: Delete Profile"
# コマンドパレット → "Profiles: Export Profile"
# コマンドパレット → "Profiles: Import Profile"

# プロファイルの共有
# エクスポートしたプロファイルは URL として共有可能
# チームメンバーが URL をクリックするだけで同一環境を構築
```

---

## 10. リモート開発

### 10.1 Remote - SSH

```jsonc
// SSH 設定（~/.ssh/config）
// Host dev-server
//   HostName 192.168.1.100
//   User developer
//   Port 22
//   IdentityFile ~/.ssh/id_ed25519
//   ForwardAgent yes

// VS Code 設定（settings.json）
{
  "remote.SSH.remotePlatform": {
    "dev-server": "linux"
  },
  "remote.SSH.useLocalServer": true,
  "remote.SSH.connectTimeout": 30,
  "remote.SSH.defaultExtensions": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode"
  ]
}
```

```
Remote SSH の接続フロー:

ローカルPC                    リモートサーバー
┌──────────┐                ┌──────────────────┐
│ VS Code  │ ── SSH ───── →│ VS Code Server   │
│ (UI)     │                │ (Extension Host) │
│          │← File Ops ─── │                  │
│          │← LSP Data ─── │ ソースコード      │
│          │← Terminal ─── │ ランタイム         │
└──────────┘                └──────────────────┘

パフォーマンス最適化:
  1. files.watcherExclude に大きなディレクトリを追加
  2. search.exclude で不要なパスを除外
  3. リモート側に必要な拡張機能のみインストール
  4. SSH ControlMaster で接続を再利用

  ~/.ssh/config に追加:
  Host *
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

### 10.2 Remote - WSL

```jsonc
// WSL 環境での設定
{
  "remote.WSL.useShellEnvironment": true,
  "terminal.integrated.defaultProfile.linux": "zsh",
  // Windows 側のパスを WSL パスに変換
  "remote.WSL.fileWatcher.polling": false
}
```

```bash
# WSL から VS Code を開く
wsl
cd /home/user/project
code .

# 特定のディストリビューションを指定
code --remote wsl+Ubuntu-22.04 /home/user/project
```

### 10.3 Remote Tunnels（VS Code Server）

```bash
# リモートマシンで VS Code Server を起動
# ブラウザや別の VS Code からアクセス可能

# サーバー側でトンネル作成
code tunnel

# 名前付きトンネル
code tunnel --name my-dev-server

# サービスとして登録（Linux）
code tunnel service install

# クライアント側から接続
# vscode.dev にアクセス → リモートマシンを選択
# または VS Code デスクトップ版からリモート接続
```

---

## 11. パフォーマンス最適化

### 11.1 起動速度の改善

```
VS Code が遅い場合のチェックリスト:

□ 1. 拡張機能の数を確認
     コマンドパレット → "Extensions: Show Running Extensions"
     起動時間が表示される。100ms 以上の拡張機能に注目。

□ 2. 不要な拡張機能を無効化
     ワークスペースごとに必要な拡張機能のみ有効化
     プロファイル機能で用途別に管理

□ 3. files.exclude / files.watcherExclude を設定
     node_modules, .git, dist, build 等を除外

□ 4. search.exclude を設定
     大きなバイナリやロック系ファイルを除外

□ 5. メモリ使用量の確認
     コマンドパレット → "Developer: Open Process Explorer"
     Extension Host のメモリが 500MB 超なら拡張機能の見直し

□ 6. 設定の見直し
     editor.minimap.enabled: false（CPU 負荷軽減）
     editor.renderWhitespace: "boundary"（"all" は避ける）
     editor.occurrencesHighlight: false（大きなファイルで軽快に）

□ 7. TypeScript の設定
     tsconfig.json の include/exclude を適切に設定
     不要なファイルを型チェック対象から除外
```

### 11.2 大規模プロジェクトでの対策

```jsonc
// 大規模モノレポでの設定例
{
  // ファイル監視の最適化
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/dist/**": true,
    "**/.next/**": true,
    "**/coverage/**": true,
    "**/.turbo/**": true,
    "**/build/**": true,
    "**/.cache/**": true,
    "**/tmp/**": true
  },

  // 検索除外
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/.next": true,
    "**/coverage": true,
    "**/package-lock.json": true,
    "**/yarn.lock": true,
    "**/pnpm-lock.yaml": true,
    "**/*.min.js": true,
    "**/*.map": true,
    "**/.turbo": true
  },

  // TypeScript の最適化
  "typescript.tsserver.maxTsServerMemory": 4096,
  "typescript.tsserver.watchOptions": {
    "watchFile": "useFsEventsOnParentDirectory",
    "watchDirectory": "useFsEvents"
  },

  // ESLint の最適化
  "eslint.workingDirectories": [
    { "mode": "auto" }
  ],

  // Git の最適化
  "git.repositoryScanMaxDepth": 1
}
```

### 11.3 Startup Performance の診断

```bash
# VS Code の起動パフォーマンスレポート
code --status

# 拡張機能の起動時間を確認
# コマンドパレット → "Developer: Startup Performance"
# 各フェーズの所要時間が表示される:
#   - Electron Main → Window Load
#   - Window Load → Require Main
#   - Require Main → Workbench Ready
#   - Extensions Activated

# Extension Bisect（問題拡張機能の特定）
# コマンドパレット → "Help: Start Extension Bisect"
# 二分探索で問題のある拡張機能を自動特定

# verbose ログで起動
code --verbose --log trace
```

---

## 12. アンチパターン

### 12.1 拡張機能の入れすぎ

```
❌ アンチパターン: 拡張機能を100個以上インストール

問題:
  - 起動時間が 10秒以上 に増大
  - メモリ使用量が 2GB 超
  - 拡張機能同士の競合（フォーマッタの二重適用など）
  - IntelliSense の応答が遅延

✅ 正しいアプローチ:
  - プロファイル機能でプロジェクト種別ごとに分離
  - 使っていない拡張機能は無効化/アンインストール
  - ワークスペース単位で拡張機能を推奨・制限
  - Extension Bisect で問題拡張機能を特定
```

### 12.2 User Settings にプロジェクト固有設定を書く

```
❌ アンチパターン: 全プロジェクト共通の User Settings に
   特定フレームワーク用の設定を書く

{
  // User settings.json に書いてしまう
  "eslint.workingDirectories": [{ "mode": "auto" }],
  "tailwindCSS.experimental.classRegex": [...]
}

✅ 正しいアプローチ:
  - プロジェクト固有設定は .vscode/settings.json に記述
  - チームで共有すべき設定はリポジトリにコミット
  - 個人的な見た目設定のみ User Settings に置く
```

### 12.3 その他のアンチパターン

```
❌ .vscode フォルダを .gitignore に入れる
  → チーム統一設定が共有できない

✅ コミットすべきファイル:
  - .vscode/settings.json（プロジェクト設定）
  - .vscode/extensions.json（推奨拡張機能）
  - .vscode/launch.json（デバッグ設定）
  - .vscode/*.code-snippets（スニペット）

✅ .gitignore に入れるべきファイル:
  - .vscode/tasks.json（個人のタスク設定の場合）
  - .vscode/*.code-workspace（個人のワークスペース設定の場合）

─────────────────────────────────

❌ formatOnSave を無効にして手動フォーマット
  → フォーマットの適用漏れ、チーム間で差異が発生

✅ 正しいアプローチ:
  - formatOnSave: true を必ず有効化
  - .prettierrc でルールを統一
  - CI でもフォーマットチェックを実行

─────────────────────────────────

❌ settings.json に秘密情報を書く
  → API キーや認証情報がリポジトリに漏洩

✅ 正しいアプローチ:
  - .env ファイルに分離
  - .env を .gitignore に追加
  - .env.example をコミットしてテンプレート共有

─────────────────────────────────

❌ すべての警告・エラーを無視する設定
  → コードの品質低下

✅ 正しいアプローチ:
  - 必要に応じて特定のルールのみ無効化
  - eslint-disable は理由をコメントで記載
  - チームで共通の除外ルールを設定
```

---

## 13. 実践的なワークフロー

### 13.1 日常開発のワークフロー

```
典型的な VS Code 開発ワークフロー:

1. プロジェクトを開く
   $ code ~/projects/my-app

2. ターミナルで開発サーバー起動
   Ctrl+` → npm run dev

3. コーディング
   - Cmd+P でファイル検索
   - F12 で定義ジャンプ
   - Shift+F12 で参照検索
   - Cmd+Shift+O でシンボル検索

4. Git 操作
   - Cmd+Shift+G で Source Control パネル
   - 変更をステージング（+ ボタン）
   - コミットメッセージ入力 → Cmd+Enter でコミット
   - GitLens で blame 確認

5. デバッグ
   - ブレークポイント設定
   - F5 でデバッグ開始
   - F10 ステップオーバー
   - F11 ステップイン
   - Shift+F11 ステップアウト

6. テスト
   - Testing パネルでテスト一覧表示
   - テストの実行・デバッグ
   - カバレッジ確認

7. PR 作成前チェック
   - Cmd+Shift+B でビルド
   - 型チェック（tsc --noEmit）
   - Lint チェック
   - テスト全実行
```

### 13.2 コードレビューワークフロー

```
PR レビューを VS Code で行う:

1. GitHub Pull Requests 拡張機能をインストール
   code --install-extension github.vscode-pull-request-github

2. PR の一覧表示
   Activity Bar → GitHub アイコン → Pull Requests

3. レビュー操作
   - PR を選択 → Checkout
   - 変更されたファイルの差分表示
   - インラインコメント追加
   - Approve / Request Changes

4. Suggested Changes
   - コード変更を提案として追加
   - レビュー相手が1クリックで適用可能

設定:
{
  "githubPullRequests.pullBranch": "prompt",
  "githubPullRequests.defaultMergeMethod": "squash",
  "githubPullRequests.showPullRequestNumberInTree": true
}
```

### 13.3 ペアプログラミング（Live Share）

```
VS Code Live Share の活用:

セットアップ:
  1. 拡張機能インストール: ms-vsliveshare.vsliveshare
  2. GitHub / Microsoft アカウントでサインイン
  3. ステータスバーの "Live Share" → "Share" をクリック
  4. 共有リンクをチームメンバーに送付

機能:
  - リアルタイムのコード共同編集
  - カーソル位置の共有
  - ターミナルの共有
  - ローカルサーバーの共有（ポートフォワーディング）
  - デバッグセッションの共有
  - 音声通話（Live Share Audio 拡張）

設定:
{
  "liveshare.presence": true,
  "liveshare.guestApprovalRequired": true,
  "liveshare.focusBehavior": "prompt",
  "liveshare.allowGuestTaskControl": false,
  "liveshare.allowGuestDebugControl": false
}
```

---

## 14. FAQ

### Q1: VS Code と VS Code Insiders の違いは？

**A:** Insiders は毎日更新されるプレビュー版。最新機能をいち早く試せるが安定性は劣る。本番開発には安定版を推奨。Insiders は別アプリとして共存可能なので、両方インストールして使い分けるのが理想的。

### Q2: `.vscode` フォルダはリポジトリにコミットすべき？

**A:** 以下のルールで判断する。

| ファイル | コミット | 理由 |
|---------|---------|------|
| `settings.json` | する | チーム共通設定（フォーマッタ等） |
| `extensions.json` | する | 推奨拡張機能の共有 |
| `launch.json` | する | デバッグ設定の共有 |
| `*.code-snippets` | する | プロジェクト固有スニペット |
| `tasks.json` | 場合による | CI と重複しないか確認 |

### Q3: Remote Development (SSH/WSL) が遅い場合の対処法は？

**A:** 以下を確認する。
1. `remote.SSH.useLocalServer` を `true` に設定
2. リモート側で不要な拡張機能を無効化
3. `files.watcherExclude` に `node_modules` 等を追加
4. SSH 接続に `ControlMaster` を設定して接続を再利用
5. `remote.SSH.remotePlatform` を明示的に設定（OS 自動検出をスキップ）

### Q4: VS Code のメモリ使用量が多すぎる場合は？

**A:** 以下の手順で調査・対策する。

```bash
# Process Explorer で確認
# コマンドパレット → "Developer: Open Process Explorer"

# 対策:
# 1. 拡張機能の見直し（Running Extensions で起動時間確認）
# 2. TypeScript サーバーのメモリ制限
"typescript.tsserver.maxTsServerMemory": 2048

# 3. ファイル監視の最適化
"files.watcherExclude": { ... }

# 4. 大きなファイルの自動折りたたみ
"editor.foldingMaximumRegions": 5000

# 5. ミニマップの無効化
"editor.minimap.enabled": false
```

### Q5: 複数の VS Code ウィンドウ間で設定を分けたい場合は？

**A:** プロファイル機能を使用する。各プロファイルには独立した設定・拡張機能・スニペットが保持される。`code --profile "プロファイル名" .` でプロファイル指定起動が可能。

### Q6: VS Code をポータブルモードで使うには？

**A:** `--user-data-dir` と `--extensions-dir` を指定して起動する。

```bash
# ポータブルモード
code \
  --user-data-dir /path/to/portable/data \
  --extensions-dir /path/to/portable/extensions \
  /path/to/project
```

USB ドライブに VS Code と設定を入れて持ち運びが可能。

### Q7: 特定のファイルタイプで異なるフォーマッタを使いたい場合は？

**A:** 言語固有設定で `editor.defaultFormatter` を指定する。

```jsonc
{
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "[go]": {
    "editor.defaultFormatter": "golang.go"
  },
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

### Q8: 大きなファイル（1MB 以上）を VS Code で開くと重い場合は？

**A:** 以下の設定を調整する。

```jsonc
{
  // 大きなファイルの処理
  "editor.largeFileOptimizations": true,
  // トークン化の上限（行数）
  "editor.maxTokenizationLineLength": 20000,
  // ミニマップを無効化
  "editor.minimap.enabled": false,
  // 折りたたみを無効化
  "editor.folding": false,
  // ワードラップを無効化
  "editor.wordWrap": "off"
}
```

---

## 15. まとめ

| 項目 | 推奨設定・ツール | 備考 |
|------|-----------------|------|
| インストール | Homebrew / winget | パッケージマネージャー経由 |
| フォント | JetBrains Mono | リガチャ対応 |
| テーマ | One Dark Pro / GitHub Theme | 好みで選択 |
| フォーマッタ | Prettier | `formatOnSave` 有効化 |
| リンター | ESLint | `codeActionsOnSave` 連携 |
| Git | GitLens + Git Graph | 必須レベル |
| AI | GitHub Copilot | 有料だが投資対効果大 |
| 設定同期 | Settings Sync | GitHub アカウント推奨 |
| プロファイル | 用途別に分離 | Frontend / Backend / Data |
| デバッグ | launch.json | 言語・フレームワーク別に設定 |
| タスク | tasks.json | ビルド・テスト・Dev Server |
| リモート | Remote SSH / WSL / Tunnels | 用途に応じて選択 |
| パフォーマンス | watcherExclude / search.exclude | 大規模プロジェクト対応 |

---

## 次に読むべきガイド

- [01-terminal-setup.md](./01-terminal-setup.md) -- ターミナル環境の構築
- [02-git-config.md](./02-git-config.md) -- Git の詳細設定
- [03-ai-tools.md](./03-ai-tools.md) -- AI 開発ツールの導入

---

## 参考文献

1. **Visual Studio Code Documentation** -- https://code.visualstudio.com/docs -- 公式ドキュメント。設定リファレンスが最も正確。
2. **VS Code Can Do That?!** (Burke Holland, Sarah Drasner) -- https://vscodecandothat.com/ -- 知られざる便利機能のコレクション。
3. **Awesome VS Code** -- https://github.com/viatsko/awesome-vscode -- コミュニティが管理する拡張機能・リソース集。
4. **VS Code Tips and Tricks** -- https://code.visualstudio.com/docs/getstarted/tips-and-tricks -- 公式の Tips 集。初心者から上級者まで有用。
5. **VS Code API Reference** -- https://code.visualstudio.com/api -- 拡張機能開発者向けの API リファレンス。
6. **Language Server Protocol** -- https://microsoft.github.io/language-server-protocol/ -- LSP の仕様。VS Code の言語サポートの基盤。
7. **Debug Adapter Protocol** -- https://microsoft.github.io/debug-adapter-protocol/ -- DAP の仕様。デバッグ機能の基盤。
