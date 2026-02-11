# プロジェクト標準 (Project Standards)

> EditorConfig、.npmrc、.nvmrc などの共通設定ファイルを活用し、チーム全体で一貫したコーディング規約と開発環境を維持するための標準化手法を学ぶ。

## この章で学ぶこと

1. **EditorConfig によるエディタ横断のフォーマット統一** -- タブ/スペース、改行コード、文字コードをエディタに依存せず統一する設定を理解する
2. **.npmrc / .nvmrc / .node-version によるランタイム統一** -- Node.js のバージョンとパッケージマネージャの動作をチーム内で揃える手法を習得する
3. **Linter / Formatter / Git Hooks の統合設定** -- ESLint、Prettier、husky、lint-staged を組み合わせた品質ゲートを構築する

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
|                                                                  |
|  [レイヤー 2] ランタイム設定                                      |
|    .nvmrc / .node-version -- Node.js バージョン固定               |
|    .npmrc               -- パッケージマネージャ設定                |
|    .tool-versions       -- asdf 全般 (Ruby, Python 等)           |
|                                                                  |
|  [レイヤー 3] コード品質                                          |
|    eslint.config.js     -- Lint ルール                            |
|    .prettierrc          -- フォーマットルール                      |
|    biome.json           -- Biome 統合設定                         |
|    tsconfig.json        -- TypeScript 設定                       |
|                                                                  |
|  [レイヤー 4] Git ワークフロー                                    |
|    .husky/              -- Git フック                             |
|    .lintstagedrc        -- ステージファイルの自動修正               |
|    .commitlintrc        -- コミットメッセージ規約                  |
|    .gitattributes       -- 改行コード・バイナリ判定                |
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

# Shell scripts
[*.sh]
end_of_line = lf
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
+------------------------------------------------------------------+
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
*.woff  binary
*.woff2 binary
*.ttf   binary
*.eot   binary
*.pdf   binary

# ロックファイル (マージ時にコンフリクトを防ぐ)
package-lock.json merge=ours linguist-generated
pnpm-lock.yaml   merge=ours linguist-generated
yarn.lock        merge=ours linguist-generated

# 自動生成ファイル (diff に表示しない)
*.min.js linguist-generated
*.min.css linguist-generated
dist/** linguist-generated
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

  // 言語固有設定
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[markdown]": {
    "editor.wordWrap": "on"
  }
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
    "github.copilot"
  ],
  "unwantedRecommendations": [
    "hookyqr.beautify"
  ]
}
```

---

## 7. Git Hooks (husky + lint-staged)

### 7.1 セットアップ

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
      ],
    ],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [1, 'always', 100],
  },
};
```

---

## 8. プロジェクト標準ファイル一覧

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
| tsconfig.json            | TypeScript 設定         | TS利用時必須 |
| .vscode/settings.json    | VS Code 共有設定        | 推奨       |
| .vscode/extensions.json  | 推奨拡張機能            | 推奨       |
| .husky/                  | Git フック              | 推奨       |
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

---

## FAQ

### Q1: EditorConfig と Prettier の両方が必要ですか？

**A**: はい。役割が異なる。EditorConfig はエディタの入力時の動作（タブ幅、改行コード）を制御し、Prettier は保存時のコード整形（括弧の位置、セミコロン等）を行う。EditorConfig は Prettier 非対応のファイル（Makefile、INI ファイル等）にも適用でき、エディタの種類にも依存しない。両方設定しておくことで、入力時と保存時の両方で一貫性が保たれる。

### Q2: husky の Git フックはチーム全員に自動適用されますか？

**A**: `package.json` の `"prepare": "husky"` スクリプトにより、`npm install` 実行時に自動でフックがインストールされる。ただし、`--no-verify` フラグで個人がフックをスキップすることは可能なため、CI/CD でも同じチェックを実行する二重防御が推奨される。また、pnpm の場合は `"prepare": "husky"` が自動実行されないため、明示的に `pnpm exec husky` を実行する手順をドキュメント化する必要がある。

### Q3: Biome を使えば ESLint + Prettier は不要になりますか？

**A**: ほぼ不要になるケースが多い。Biome は Rust 製の高速ツールで、Lint とフォーマットの両方を 1 つのツールで処理する。ESLint + Prettier の組み合わせに比べて 10-100 倍速い。ただし、ESLint の一部プラグイン（eslint-plugin-react-hooks, @typescript-eslint の高度なルール等）に相当する機能がまだ不足している場合がある。新規プロジェクトでは Biome を第一候補として検討し、足りないルールのみ ESLint で補完する戦略が有効。

---

## まとめ

| 項目 | 要点 |
|------|------|
| EditorConfig | エディタ横断でタブ幅・改行コード・文字コードを統一 |
| .nvmrc | Node.js バージョンをチームで固定。volta / fnm でも対応 |
| .npmrc | `engine-strict=true` と `save-exact=true` を推奨 |
| .gitattributes | 改行コードの自動変換とバイナリファイルの判定 |
| VS Code 設定 | チーム共通設定のみコミット。個人設定は除外 |
| Git Hooks | husky + lint-staged でコミット前に自動 Lint/Format |
| Commitlint | Conventional Commits でコミットメッセージの品質を担保 |
| 二重防御 | ローカルフック + CI/CD の両方で品質チェックを実行 |

## 次に読むべきガイド

- [オンボーディング自動化](./01-onboarding-automation.md) -- セットアップスクリプトと Makefile
- [ドキュメント環境](./02-documentation-setup.md) -- VitePress / Docusaurus / ADR
- [Dev Container](../02-docker-dev/01-devcontainer.md) -- コンテナベースの統一開発環境

## 参考文献

1. **EditorConfig 公式** -- https://editorconfig.org/ -- EditorConfig の仕様とエディタ対応状況
2. **Conventional Commits** -- https://www.conventionalcommits.org/ja/ -- コミットメッセージ規約の仕様
3. **husky 公式ドキュメント** -- https://typicode.github.io/husky/ -- Git フックの管理ツール
4. **Biome 公式** -- https://biomejs.dev/ -- Rust 製の高速 Linter/Formatter ツール
