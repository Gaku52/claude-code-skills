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
| 00 | [[docs/00-fundamentals/00-desktop-app-overview.md]] | デスクトップアプリの種類、技術選定、Web技術 vs ネイティブ |
| 01 | [[docs/00-fundamentals/01-architecture-patterns.md]] | メインプロセス/レンダラー、IPC通信、セキュリティモデル |
| 02 | [[docs/00-fundamentals/02-native-features.md]] | ファイルシステム、通知、トレイ、自動起動、ショートカット |
| 03 | [[docs/00-fundamentals/03-cross-platform.md]] | Windows/macOS/Linux 対応、OS固有APIの抽象化 |

### 01-wpf-and-winui — Windows ネイティブ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-wpf-and-winui/00-windows-ui-frameworks.md]] | WPF、WinUI 3、MAUI の比較と選定 |
| 01 | [[docs/01-wpf-and-winui/01-winui3-basics.md]] | WinUI 3 の基本、XAML、データバインディング |
| 02 | [[docs/01-wpf-and-winui/02-webview2.md]] | WebView2 でWeb技術をネイティブアプリに統合 |

### 02-electron-and-tauri — Web技術ベース

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-electron-and-tauri/00-electron-setup.md]] | Electron セットアップ、React統合、electron-forge |
| 01 | [[docs/02-electron-and-tauri/01-electron-advanced.md]] | IPC通信、preload、セキュリティ、パフォーマンス |
| 02 | [[docs/02-electron-and-tauri/02-tauri-setup.md]] | Tauri v2 セットアップ、Rust バックエンド、Commands |
| 03 | [[docs/02-electron-and-tauri/03-tauri-advanced.md]] | プラグイン、マルチウィンドウ、アップデート、セキュリティ |

### 03-distribution — 配布とアップデート

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-distribution/00-packaging-and-signing.md]] | パッケージング、コード署名、インストーラー作成 |
| 01 | [[docs/03-distribution/01-auto-update.md]] | 自動アップデート（electron-updater、Tauri updater） |
| 02 | [[docs/03-distribution/02-store-distribution.md]] | Microsoft Store、GitHub Releases、自前配布 |

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
