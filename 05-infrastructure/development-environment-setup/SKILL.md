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

### 01-runtime-and-package — ランタイムとパッケージ管理

| # | ファイル | 内容 |
|---|---------|------|

### 02-docker-dev — Docker 開発環境

| # | ファイル | 内容 |
|---|---------|------|

### 03-team-setup — チーム統一設定

| # | ファイル | 内容 |
|---|---------|------|

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
