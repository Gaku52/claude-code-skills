# Windows アプリケーション開発

> Web 技術を活用したデスクトップアプリ開発が主流の時代。Electron、Tauri、WPF/WinUI の特徴と選定基準、クロスプラットフォーム対応、ネイティブ機能の活用、配布とアップデートまで、Windows デスクトップアプリ開発の全体像を解説する。

## このSkillの対象者

- Web 技術（React/TypeScript）でデスクトップアプリを開発したいエンジニア
- Electron / Tauri の選定・実装を行う方
- Windows ネイティブ機能（通知、トレイ、ファイルシステム等）を活用したい方

## 前提知識

- HTML/CSS/JavaScript の基礎
- React/TypeScript の基本的な開発経験
- Node.js の基礎知識

## 学習ガイド

### 00-fundamentals — デスクトップアプリの基礎

| # | ファイル | 内容 |
|---|---------|------|

### 01-wpf-and-winui — Windows ネイティブ

| # | ファイル | 内容 |
|---|---------|------|

### 02-electron-and-tauri — Web技術ベース

| # | ファイル | 内容 |
|---|---------|------|

### 03-distribution — 配布とアップデート

| # | ファイル | 内容 |
|---|---------|------|

## クイックリファレンス

```
技術選定ガイド:

  軽量 + セキュリティ重視 → Tauri（推奨）
  エコシステム + 実績 → Electron
  Windows 専用 + ネイティブ感 → WinUI 3
  クロスプラットフォーム + .NET → MAUI

  バンドルサイズ比較:
    Electron: ~150MB（Chromium同梱）
    Tauri:    ~5MB（OS の WebView 使用）
    WinUI 3:  ~20MB（.NET ランタイム）

  メモリ使用量:
    Electron: ~200MB+
    Tauri:    ~50MB
    WinUI 3:  ~100MB
```

## 参考文献

1. Electron. "Documentation." electronjs.org, 2024.
2. Tauri. "Documentation." tauri.app, 2024.
3. Microsoft. "WinUI 3 Documentation." learn.microsoft.com, 2024.
