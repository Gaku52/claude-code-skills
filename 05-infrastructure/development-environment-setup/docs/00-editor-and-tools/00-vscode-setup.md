# VS Code セットアップ

> Visual Studio Code のインストールから実践的なカスタマイズまで、開発生産性を最大化するための完全ガイド。

## この章で学ぶこと

1. VS Code のインストール・初期設定・設定同期を正しく構成する方法
2. 開発効率を飛躍的に向上させる拡張機能の選定と管理手法
3. マルチカーソル・スニペット・キーバインドを使いこなす実践テクニック

---

## 1. インストールと初期設定

### 1.1 プラットフォーム別インストール

```bash
# macOS (Homebrew)
brew install --cask visual-studio-code

# Ubuntu/Debian
sudo apt install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update && sudo apt install code

# Windows (winget)
winget install Microsoft.VisualStudioCode
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
```

---

## 2. 設定ファイル（settings.json）

### 2.1 推奨初期設定

```jsonc
// .vscode/settings.json (ワークスペース設定)
{
  // エディタ基本設定
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

  // 保存時の自動処理
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
    "source.organizeImports": "explicit"
  },

  // ファイル設定
  "files.autoSave": "onFocusChange",
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.exclude": {
    "**/.git": true,
    "**/node_modules": true,
    "**/.DS_Store": true
  },

  // ターミナル設定
  "terminal.integrated.fontSize": 13,
  "terminal.integrated.defaultProfile.osx": "zsh",

  // 検索設定
  "search.exclude": {
    "**/node_modules": true,
    "**/dist": true,
    "**/.next": true,
    "**/coverage": true
  }
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
```

---

## 3. 推奨拡張機能リスト

### 3.1 カテゴリ別一覧

| カテゴリ | 拡張機能 | 用途 |
|---------|---------|------|
| **言語サポート** | ESLint | JavaScript/TypeScript リント |
| | Prettier | コードフォーマッタ |
| | Python (ms-python) | Python 言語サポート |
| | Rust Analyzer | Rust 言語サポート |
| **Git** | GitLens | Git 履歴・blame 表示 |
| | Git Graph | ブランチグラフ可視化 |
| **AI** | GitHub Copilot | AI コード補完 |
| | Claude Code (CLI) | AI エージェント |
| **開発効率** | Error Lens | インラインエラー表示 |
| | TODO Highlight | TODO/FIXME ハイライト |
| | Path Intellisense | パス自動補完 |
| **外観** | Material Icon Theme | ファイルアイコン |
| | One Dark Pro | カラーテーマ |
| **コンテナ** | Dev Containers | Docker 開発環境 |
| | Docker | Docker 管理 |

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
```

### 3.3 ワークスペース推奨設定

```jsonc
// .vscode/extensions.json
{
  "recommendations": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "eamodio.gitlens",
    "github.copilot",
    "usernamehw.errorlens",
    "bradlc.vscode-tailwindcss"
  ],
  "unwantedRecommendations": [
    "hookyqr.beautify"  // Prettier と競合
  ]
}
```

---

## 4. キーバインド

### 4.1 必須ショートカット一覧

| 操作 | macOS | Windows/Linux |
|------|-------|---------------|
| コマンドパレット | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| ファイル検索 | `Cmd+P` | `Ctrl+P` |
| シンボル検索 | `Cmd+T` | `Ctrl+T` |
| 全文検索 | `Cmd+Shift+F` | `Ctrl+Shift+F` |
| ターミナル切替 | `` Ctrl+` `` | `` Ctrl+` `` |
| サイドバー切替 | `Cmd+B` | `Ctrl+B` |
| 行の移動 | `Alt+↑/↓` | `Alt+↑/↓` |
| 行の複製 | `Shift+Alt+↑/↓` | `Shift+Alt+↑/↓` |
| 定義へ移動 | `F12` | `F12` |
| 参照を検索 | `Shift+F12` | `Shift+F12` |
| リネーム | `F2` | `F2` |
| クイックフィックス | `Cmd+.` | `Ctrl+.` |

### 4.2 カスタムキーバインド

```jsonc
// keybindings.json
[
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
  }
]
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

  ① カーソルを "firstName" の "Name" に置く
  ② Cmd+D → "lastName" の "Name" も選択
  ③ Cmd+D → "nickName" の "Name" も選択
  ④ 一括で編集: "Name" → "Label"

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

---

## 6. スニペット

### 6.1 ユーザースニペット定義

```jsonc
// .vscode/project.code-snippets
{
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
  "Console Log Variable": {
    "prefix": "clv",
    "scope": "javascript,typescript,typescriptreact,javascriptreact",
    "body": [
      "console.log('${1:variable}:', ${1:variable});"
    ],
    "description": "Console log with variable name"
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
  }
}
```

---

## 7. 設定同期

### 7.1 Settings Sync の構成

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
```

### 7.2 プロファイル機能

```bash
# プロファイルの切替で用途別の環境を管理
#
# 例: "Frontend" プロファイル
#   - ESLint, Prettier, Tailwind CSS IntelliSense
#   - React/Vue 拡張
#   - フロントエンド向けテーマ
#
# 例: "Python Data Science" プロファイル
#   - Python, Jupyter, Pylance
#   - データ可視化拡張
#   - 分析向け設定

# CLI でプロファイル指定して起動
code --profile "Frontend" .
```

---

## 8. アンチパターン

### 8.1 拡張機能の入れすぎ

```
❌ アンチパターン: 拡張機能を100個以上インストール

問題:
  - 起動時間が 10秒以上 に増大
  - メモリ使用量が 2GB 超
  - 拡張機能同士の競合（フォーマッタの二重適用など）

✅ 正しいアプローチ:
  - プロファイル機能でプロジェクト種別ごとに分離
  - 使っていない拡張機能は無効化/アンインストール
  - ワークスペース単位で拡張機能を推奨・制限
```

### 8.2 User Settings にプロジェクト固有設定を書く

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

---

## 9. FAQ

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

---

## 10. まとめ

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

---

## 次に読むべきガイド

- [01-terminal-setup.md](./01-terminal-setup.md) — ターミナル環境の構築
- [02-git-config.md](./02-git-config.md) — Git の詳細設定
- [03-ai-tools.md](./03-ai-tools.md) — AI 開発ツールの導入

---

## 参考文献

1. **Visual Studio Code Documentation** — https://code.visualstudio.com/docs — 公式ドキュメント。設定リファレンスが最も正確。
2. **VS Code Can Do That?!** (Burke Holland, Sarah Drasner) — https://vscodecandothat.com/ — 知られざる便利機能のコレクション。
3. **Awesome VS Code** — https://github.com/viatsko/awesome-vscode — コミュニティが管理する拡張機能・リソース集。
4. **VS Code Tips and Tricks** — https://code.visualstudio.com/docs/getstarted/tips-and-tricks — 公式の Tips 集。初心者から上級者まで有用。
