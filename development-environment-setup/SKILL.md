# 開発環境セットアップ

> 生産性の高い開発は適切な環境から始まる。エディタ設定、ランタイム管理、パッケージマネージャー、Docker 開発環境、チーム統一設定まで、モダンな開発環境構築の全てを解説する。

## このSkillの対象者

- 新しいプロジェクトの開発環境を構築するエンジニア
- チーム全体の開発環境を統一したいリード
- Docker を使った開発環境の構築を学びたい方

## 前提知識

- ターミナルの基本操作
- Git の基礎知識

## 学習ガイド

### 00-editor-and-tools — エディタと開発ツール

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-editor-and-tools/00-vscode-setup.md]] | VS Code 設定、拡張機能、キーバインド、プロファイル |
| 01 | [[docs/00-editor-and-tools/01-terminal-setup.md]] | ターミナル設定（Warp、iTerm2、Windows Terminal）、シェル（zsh/fish） |
| 02 | [[docs/00-editor-and-tools/02-dev-tools.md]] | HTTPクライアント、DB GUI、デバッガー、プロファイラー |
| 03 | [[docs/00-editor-and-tools/03-ai-tools.md]] | GitHub Copilot、Claude Code、Cursor、AI 支援開発 |

### 01-runtime-and-package — ランタイムとパッケージ管理

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-runtime-and-package/00-version-managers.md]] | fnm/nvm（Node.js）、pyenv（Python）、rustup（Rust）、mise |
| 01 | [[docs/01-runtime-and-package/01-package-managers.md]] | npm/pnpm/yarn/bun、pip/uv、cargo の比較と使い分け |
| 02 | [[docs/01-runtime-and-package/02-monorepo-setup.md]] | Turborepo、pnpm workspace、Nx の設定と運用 |
| 03 | [[docs/01-runtime-and-package/03-linter-formatter.md]] | ESLint、Prettier、Biome、lint-staged、husky |

### 02-docker-dev — Docker 開発環境

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-docker-dev/00-docker-for-dev.md]] | 開発用 Docker Compose、ホットリロード、ボリュームマウント |
| 01 | [[docs/02-docker-dev/01-devcontainer.md]] | VS Code Dev Containers、devcontainer.json、GitHub Codespaces |
| 02 | [[docs/02-docker-dev/02-local-services.md]] | PostgreSQL、Redis、MinIO、MailHog のローカル環境 |

### 03-team-setup — チーム統一設定

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-team-setup/00-project-standards.md]] | EditorConfig、.nvmrc、engine制約、共有設定ファイル |
| 01 | [[docs/03-team-setup/01-git-hooks-and-ci.md]] | pre-commit、commitlint、GitHub Actions、PR テンプレート |
| 02 | [[docs/03-team-setup/02-documentation-setup.md]] | README テンプレート、ADR、開発者オンボーディング |

## クイックリファレンス

```
推奨開発環境スタック:
  エディタ:     VS Code + 拡張機能 or Cursor
  ターミナル:   Warp (macOS) / Windows Terminal
  シェル:      zsh + starship prompt
  Node.js:    fnm（推奨）or mise
  パッケージ:   pnpm（推奨）
  リンター:    Biome（推奨）or ESLint + Prettier
  Git Hooks:  husky + lint-staged
  Docker:     Docker Desktop or OrbStack (macOS)
  DB GUI:     TablePlus or DBeaver
  API:        Bruno or Hoppscotch
  AI:         GitHub Copilot + Claude Code
```

## 参考文献

1. VS Code. "Documentation." code.visualstudio.com, 2024.
2. Docker. "Docker Desktop." docs.docker.com, 2024.
3. pnpm. "Documentation." pnpm.io, 2024.
